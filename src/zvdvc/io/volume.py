"""Dense image volumes (reference and deformed), read one brick at a time.

A *brick* is the box a tile needs: the tile's points plus the halo
(:meth:`zvdvc.config.RunConfig.halo`). The deformed brick is shifted by the
tile's median seed displacement, so a large rigid offset does not inflate the halo.
Bricks land on the device in their **native dtype** (u8/u16/f32). Kernels
convert in registers, which keeps a u16 brick at half the size of a float32 one
and halves the host-to-device copy.

Sources
-------
``ZarrVolume``
    OME-Zarr v3 array. Recommended layout: 128^3 chunks inside shards aligned
    to ``cluster.tile_shape``, so a brick read touches a handful of shards.
    Device path, in order of preference:

    1. kvikio / GPUDirect Storage for local NVMe (``[gpu-io]`` extra);
    2. zarr-python's GPU buffer mode (``zarr.config.enable_gpu()``) where the
       codec pipeline supports it;
    3. host decode into pinned memory, then one host-to-device copy per brick
       on the worker's prefetch stream.

    Implemented today: path 3. The brick is decoded on the host (zarr-python
    reads the shards' chunks concurrently) into a pinned buffer and copied up
    in one transfer on the caller's stream. Paths 1 and 2 are optimisations to
    measure against it (docs/MVP_PLAN.md, risk "GPU zstd decode").
``RawVolume``
    CCPi/iDVC flat inputs (``.raw`` with header/endianness, ``.mhd``, ``.npy``)
    through ``numpy.memmap``. Used for development and the parity case. Large
    inputs should be converted once with :func:`convert_to_ome_zarr`.

Boxes that extend past the volume are edge-padded. :class:`Brick` records the
valid part, so the solver flags points whose samples leave it as
``RANGE_FAIL`` instead of correlating them against padding.

Every source reads bricks in the host's byte order, so a big-endian file reaches
the engines as native data, and ``.dtype`` is that native type. Float bricks
are checked for NaN/inf as they are read (a brick with any is an error): at open
time the check would read the whole volume, which for 100 GB+ inputs costs as
much as the run's own reads, while per brick it is one pass over data already in
memory. A NaN would otherwise spread through the interpolation to every point
whose subvolume touches it.
"""

from __future__ import annotations

import math
import os
import shutil
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from zvdvc.config import VolumeSpec
from zvdvc.geometry.box import Box
from zvdvc.kernels.xp import Device, get_xp


@dataclass
class Brick:
    data: Any       # (z, y, x) array, numpy or cupy, native dtype
    box: Box        # the box that was asked for
    valid: Box      # the part of ``box`` inside the volume
    io: str = "host"   # how it reached the device: "host" (host decode, one copy up) or "kvikio" (zvdvc.io.gds)

    @property
    def origin_xyz(self) -> tuple[float, float, float]:
        """Point-space coordinate of ``data[0, 0, 0]``."""
        return (float(self.box.lo[2]), float(self.box.lo[1]), float(self.box.lo[0]))

    @property
    def valid_lo_hi_xyz(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """``valid`` as a half-open voxel range in point order, for :func:`zvdvc.kernels.interpolate.sample`."""
        lo, hi = self.valid.lo, self.valid.hi
        return (lo[2], lo[1], lo[0]), (hi[2], hi[1], hi[0])


class VolumeSource(Protocol):
    shape: tuple[int, int, int]   # (z, y, x)
    dtype: np.dtype

    def read_brick(self, box: Box, *, device: Device = "cuda", stream: Any = None) -> Brick: ...


def native(dtype: Any) -> np.dtype:
    """``dtype`` in the host's byte order."""
    dtype = np.dtype(dtype)
    return dtype if dtype.isnative else dtype.newbyteorder("=")


def _check_finite(data: np.ndarray, name: str, box: Box, slab: int = 64) -> None:
    """Raise if a float brick holds NaN or inf (checked in z-slabs to bound the temporary mask)."""
    if data.dtype.kind != "f":
        return
    bad = sum(int(np.count_nonzero(~np.isfinite(data[z:z + slab]))) for z in range(0, data.shape[0], slab))
    if bad:
        raise ValueError(f"{name}: {bad:,} non-finite voxels (NaN/inf) in box {box}; replace them "
                         f"(e.g. with the background level) before correlating")


def _read_brick(array: Any, shape: tuple[int, int, int], box: Box, device: Device, stream: Any = None,
                name: str = "volume", reader: Any = None) -> Brick:
    """Read ``box`` from a (z, y, x) array-like in native byte order, edge-padding whatever lies outside the volume.

    With a device ``reader`` (:mod:`zvdvc.io.gds`) a ``device="cuda"`` read goes straight to GPU memory.
    """
    valid = box.intersect(Box((0, 0, 0), shape))
    if valid is None:
        raise ValueError(f"{box} does not overlap a volume of shape {shape}")
    if device == "cuda" and reader is not None:
        return _read_brick_on_device(reader, native(array.dtype), box, valid, stream, name)
    part = np.asarray(array[valid.slices()])
    if not part.dtype.isnative:
        part = part.astype(native(part.dtype))
    _check_finite(part, name, valid)
    pad = [(v - b, bh - vh) for b, v, vh, bh in zip(box.lo, valid.lo, valid.hi, box.hi)]
    data = np.pad(part, pad, mode="edge") if any(p != (0, 0) for p in pad) else np.ascontiguousarray(part)
    if device == "cuda":
        data = to_device(data, stream=stream)
    return Brick(data=data, box=box, valid=valid)


def _read_brick_on_device(reader: Any, dtype: np.dtype, box: Box, valid: Box, stream: Any, name: str) -> Brick:
    """The device path of :func:`_read_brick`: read into the padded brick, then edge-pad and check it on the GPU."""
    cp = get_xp("cuda")
    out = padded_empty(box.shape, dtype, cp)
    at = tuple(v - b for v, b in zip(valid.lo, box.lo))
    reader.read_into(out, valid, at, stream)
    ctx = stream if stream is not None else cp.cuda.Stream.null
    with ctx:
        if dtype.kind == "f":
            inner = out[tuple(slice(a, a + n) for a, n in zip(at, valid.shape))]
            bad = int(cp.count_nonzero(~cp.isfinite(inner)))
            if bad:
                raise ValueError(f"{name}: {bad:,} non-finite voxels (NaN/inf) in box {valid}; replace them "
                                 f"(e.g. with the background level) before correlating")
        _edge_pad_in_place(out, at, valid.shape)
    ctx.synchronize()
    return Brick(data=out, box=box, valid=valid, io="kvikio")


def _edge_pad_in_place(a: Any, at: tuple[int, ...], n: tuple[int, ...]) -> None:
    """Fill ``a`` outside ``a[at : at + n]`` by repeating the edge voxels, axis by axis (``np.pad`` mode "edge")."""
    for axis in range(a.ndim):
        lo, hi = at[axis], at[axis] + n[axis]
        idx = [slice(None)] * a.ndim
        if lo > 0:
            idx[axis] = slice(0, lo)
            src = [slice(None)] * a.ndim
            src[axis] = slice(lo, lo + 1)
            a[tuple(idx)] = a[tuple(src)]
        if hi < a.shape[axis]:
            idx[axis] = slice(hi, None)
            src = [slice(None)] * a.ndim
            src[axis] = slice(hi - 1, hi)
            a[tuple(idx)] = a[tuple(src)]


# Bytes kept after a device brick's last voxel. The u8 tricubic kernel reads each stencil row as two
# aligned 32-bit words, which can reach up to 4 bytes past the row's last tap (kernels/cuda/fused_gn.cu).
PAD_BYTES = 16


def padded_empty(shape: tuple[int, ...], dtype: Any, xp: Any = np, pad_bytes: int = PAD_BYTES) -> Any:
    """An uninitialised C-contiguous array at the start of a fresh allocation, with ``pad_bytes`` after it.

    Fresh allocations are at least 16-byte aligned (cupy: 256), as the u8 kernel's word loads need.
    """
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    return xp.empty(nbytes + pad_bytes, dtype=np.uint8)[:nbytes].view(dtype).reshape(shape)


def is_padded(a: Any, pad_bytes: int = PAD_BYTES) -> bool:
    """True for a C-contiguous cupy array, 16-byte aligned, with ``pad_bytes`` of its allocation after it."""
    mem = getattr(getattr(a, "data", None), "mem", None)
    if mem is None or not a.flags.c_contiguous:
        return False
    offset = a.data.ptr - mem.ptr
    return a.data.ptr % 16 == 0 and offset + a.nbytes + pad_bytes <= mem.size


def to_device(data: np.ndarray, *, stream: Any = None) -> Any:
    """Host array -> device (padded, see :data:`PAD_BYTES`), staged through pinned memory as one DMA on ``stream``."""
    cp = get_xp("cuda")
    import cupyx

    pinned = cupyx.empty_pinned(data.shape, dtype=data.dtype)
    pinned[...] = data
    out = padded_empty(data.shape, data.dtype, cp)
    if stream is None:
        out.set(pinned)
    else:
        out.set(pinned, stream=stream)
        stream.synchronize()
    return out


class ZarrVolume:
    """An OME-Zarr level (or bare Zarr array). Opening checks that every chunk is stored (``check_chunks``)."""

    def __init__(self, uri: str, array_path: str = "0", *, check_chunks: bool = True, gpu_io: str = "auto") -> None:
        import zarr

        node = zarr.open(uri, mode="r")
        if isinstance(node, zarr.Group):
            if array_path not in node or not isinstance(node[array_path], zarr.Array):
                levels = sorted(node.array_keys())
                raise ValueError(f"{uri}: no array (level) {array_path!r} here; available: {levels} (volumes.array_path)")
            node = node[array_path]
        self.array = node
        if self.array.ndim != 3:
            raise ValueError(f"{uri}: expected a 3-D (z, y, x) array, got shape {self.array.shape}")
        self.uri = uri
        self.shape: tuple[int, int, int] = tuple(int(n) for n in self.array.shape)
        self.dtype = native(self.array.dtype)
        stored, total = (self.array.nchunks_initialized, self.array.nchunks) if check_chunks else (0, 0)
        if stored < total:
            # zarr leaves chunks that hold only the fill value unwritten by default, so an array from another
            # tool (or an older zvDVC) can be complete with chunks missing; a truncated copy looks the same.
            warnings.warn(
                f"{uri}: only {stored} of {total} chunks are stored; missing chunks read as {self.array.fill_value}. "
                f"Fine if those regions are empty; otherwise the copy or conversion is incomplete "
                f"(`zvdvc convert` stores every chunk)", stacklevel=2)
        self.io, self.io_note, self._reader = _device_reader(gpu_io, lambda: _gds().ShardedZarrReader(self.array))

    def read_brick(self, box: Box, *, device: Device = "cuda", stream: Any = None) -> Brick:
        return _read_brick(self.array, self.shape, box, device, stream, self.uri, self._reader)


_MHD_TYPES = {
    "MET_UCHAR": "u1", "MET_CHAR": "i1", "MET_USHORT": "u2", "MET_SHORT": "i2",
    "MET_UINT": "u4", "MET_INT": "i4", "MET_FLOAT": "f4", "MET_DOUBLE": "f8",
}


def _read_mhd(path: Path) -> tuple[Path, tuple[int, int, int], np.dtype, int]:
    """(data file, shape_xyz, dtype, header bytes) from a MetaImage header: one uncompressed, single-channel 3-D file."""
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    for key in ("DimSize", "ElementType", "ElementDataFile"):
        if key not in fields:
            raise ValueError(f"{path}: missing {key} in the MetaImage header")
    try:
        shape_xyz = tuple(int(v) for v in fields["DimSize"].split())
    except ValueError:
        raise ValueError(f"{path}: DimSize = {fields['DimSize']!r} is not a list of integers") from None
    if len(shape_xyz) != 3 or fields.get("NDims", "3") != "3":
        raise ValueError(f"{path}: expected a 3-D image, got NDims = {fields.get('NDims')}, DimSize = {fields['DimSize']}")
    if fields["ElementType"] not in _MHD_TYPES:
        raise ValueError(f"{path}: ElementType {fields['ElementType']!r} is not supported; use one of {sorted(_MHD_TYPES)}")
    if fields.get("CompressedData", "False").lower() == "true":
        raise ValueError(f"{path}: CompressedData = True is not supported; write the image uncompressed")
    if fields.get("ElementNumberOfChannels", "1") != "1":
        raise ValueError(f"{path}: ElementNumberOfChannels = {fields['ElementNumberOfChannels']}; only single-channel "
                         f"(grey-level) images are supported")
    big = fields.get("BinaryDataByteOrderMSB", fields.get("ElementByteOrderMSB", "False")).lower() == "true"
    dtype = np.dtype(_MHD_TYPES[fields["ElementType"]]).newbyteorder(">" if big else "<")
    data_file = fields["ElementDataFile"]
    if data_file in ("LOCAL", "LIST") or "%" in data_file:
        raise ValueError(f"{path}: ElementDataFile = {data_file}: only one detached data file is supported "
                         f"(not LOCAL / .mha, LIST or a file pattern)")
    try:
        header = int(fields.get("HeaderSize", "0"))
    except ValueError:
        raise ValueError(f"{path}: HeaderSize = {fields['HeaderSize']!r} is not an integer") from None
    if header < 0:
        raise ValueError(f"{path}: HeaderSize = -1 (auto) is not supported; give the header length in bytes")
    return path.parent / data_file, shape_xyz, dtype, header


class RawVolume:
    def __init__(
        self,
        path: str | Path,
        *,
        shape_xyz: tuple[int, int, int] | None = None,
        dtype: str | None = None,
        header_bytes: int = 0,
        gpu_io: str = "auto",
    ) -> None:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            array = np.load(path, mmap_mode="r")
            if array.ndim != 3:
                raise ValueError(f"{path}: expected a 3-D (z, y, x) array, got shape {array.shape}")
            self.header_bytes = int(array.offset)
        else:
            if suffix == ".mhd":
                mhd = path
                path, shape_xyz, dt, header_bytes = _read_mhd(path)
                hint = f"check DimSize, ElementType and HeaderSize in {mhd}"
            else:
                if shape_xyz is None or dtype is None:
                    raise ValueError(f"{path}: raw volumes need raw_shape_xyz and raw_dtype")
                dt = np.dtype(dtype)
                hint = ("check the bit depth (vol_bit_depth / raw_dtype), the dimensions (vol_wide, vol_high, vol_tall / "
                        "raw_shape_xyz) and the header length (vol_hdr_lngth / raw_header_bytes)")
            x, y, z = shape_xyz
            _check_raw_size(path, (x, y, z), dt, int(header_bytes), hint)
            array = np.memmap(path, dtype=dt, mode="r", offset=header_bytes, shape=(z, y, x))
            self.header_bytes = int(header_bytes)
        self.path = path
        self.array = array                   # in the file's byte order
        self.shape: tuple[int, int, int] = tuple(int(n) for n in array.shape)
        self.dtype = native(array.dtype)

        def reader() -> Any:
            gds = _gds()
            if not array.flags.c_contiguous:
                r = gds.DeviceReader()
                r.why_not = "not C-ordered (a Fortran-ordered .npy)"
                return r
            return gds.RawDeviceReader(path, header_bytes=self.header_bytes, dtype=array.dtype, shape_zyx=self.shape)

        self.io, self.io_note, self._reader = _device_reader(gpu_io, reader)

    def read_brick(self, box: Box, *, device: Device = "cuda", stream: Any = None) -> Brick:
        return _read_brick(self.array, self.shape, box, device, stream, str(self.path), self._reader)


def _gds() -> Any:
    from zvdvc.io import gds

    return gds


def _device_reader(gpu_io: str, make: Any) -> tuple[str, str | None, Any]:
    """(read path, why the device path is not used, device reader or None) for a volume source."""
    gds = _gds()
    if gds.resolve(gpu_io) == "host":
        return "host", (None if gpu_io == "host" else gds.status().get("why")), None
    reader = make()
    if reader.why_not:
        return "host", reader.why_not, None
    return "kvikio", None, reader


def _check_raw_size(path: Path, shape_xyz: tuple[int, int, int], dtype: np.dtype, header: int, hint: str) -> None:
    """The file must hold exactly ``header`` bytes plus the voxels: smaller is truncated, larger is misdescribed."""
    if not path.is_file():
        raise FileNotFoundError(f"{path}: volume file not found")
    size = path.stat().st_size
    data = int(np.prod(shape_xyz, dtype=np.int64)) * dtype.itemsize
    expected = header + data
    if size == expected:
        return
    dims = "x".join(str(n) for n in shape_xyz)
    if size < expected:
        raise ValueError(f"{path}: file is {size:,} bytes, smaller than the {expected:,} that {dims} {dtype} voxels "
                         f"after a {header}-byte header need (truncated?); {hint}")
    raise ValueError(f"{path}: file is {size:,} bytes, not the {expected:,} that {dims} {dtype} voxels after a "
                     f"{header}-byte header take (its data is {(size - header) / max(data, 1):.3g} times as large); {hint}")


def is_zarr_uri(uri: str | Path) -> bool:
    p = Path(uri)
    return p.suffix == ".zarr" or p.name.endswith(".ome.zarr") or (p / "zarr.json").exists()


def open_volume(spec: VolumeSpec, which: Literal["reference", "deformed"]) -> VolumeSource:
    """Pick ``ZarrVolume`` or ``RawVolume`` from the URI; wrapped in :class:`FilteredVolume` if ``prefilter_sigma``."""
    uri = getattr(spec, which)
    if is_zarr_uri(uri):
        vol: Any = ZarrVolume(uri, spec.array_path, gpu_io=spec.gpu_io)
    else:
        vol = RawVolume(uri, shape_xyz=spec.raw_shape_xyz, dtype=spec.raw_dtype, header_bytes=spec.raw_header_bytes,
                        gpu_io=spec.gpu_io)
    return FilteredVolume(vol, spec.prefilter_sigma) if spec.prefilter_sigma > 0 else vol


TRUNCATE = 4.0          # Gaussian kernel radius, in sigmas


class FilteredVolume:
    """A volume seen through a Gaussian low-pass of ``sigma`` voxels.

    Each brick is read with a margin of the kernel's radius, filtered and cropped,
    so a brick equals the same box of the filtered whole volume: tiles meet
    without seams, and the in-memory and tiled runs see the same data. Outside
    the volume, voxels are edge-padded before filtering, as ``read_brick`` pads.
    Filtering runs on the GPU when cupy has a device (in z-slabs, so a host
    brick of any size fits), else on the host. The result keeps the volume's
    dtype (rounded to the nearest level), so bricks cost the same memory and
    u8 volumes keep the kernels' packed loads.
    """

    SLAB = 64            # z-slices filtered at a time on the GPU

    def __init__(self, inner: Any, sigma: float) -> None:
        self.inner, self.sigma = inner, float(sigma)
        self.shape, self.dtype = inner.shape, np.dtype(inner.dtype)
        self.array = inner.array             # unfiltered; for tools that inspect the raw data
        self.io, self.io_note = getattr(inner, "io", "host"), getattr(inner, "io_note", None)
        self.margin = int(math.ceil(TRUNCATE * self.sigma))

    def read_brick(self, box: Box, *, device: Device = "cuda", stream: Any = None) -> Brick:
        m = self.margin
        grown = self.inner.read_brick(box.grow(m), device=device, stream=stream)
        data = gaussian_filtered(grown.data, self.sigma, self.dtype)
        inner = tuple(slice(m, n - m) for n in data.shape)
        return Brick(data=data[inner], box=box, valid=box.intersect(Box((0, 0, 0), self.shape)) or box, io=grown.io)


def gaussian_filtered(data: Any, sigma: float, dtype: Any, slab: int = FilteredVolume.SLAB) -> Any:
    """``data`` filtered with a Gaussian of ``sigma`` voxels (radius 4 sigma), as ``dtype``, on data's device.

    Host data is filtered slab by slab on the GPU when one is available.
    """
    dtype = np.dtype(dtype)
    on_device = hasattr(type(data), "__cuda_array_interface__")
    try:
        cp = get_xp("cuda")
        from cupyx.scipy.ndimage import gaussian_filter as gf_gpu
    except Exception:
        cp = None
    if cp is None:
        from scipy.ndimage import gaussian_filter

        return _to_levels(gaussian_filter(np.asarray(data, dtype=np.float32), sigma, mode="nearest", truncate=TRUNCATE), dtype, np)
    xp = cp if on_device else np
    out = xp.empty(data.shape, dtype=dtype)
    r = int(math.ceil(TRUNCATE * sigma))
    nz = data.shape[0]
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        lo, hi = max(z0 - r, 0), min(z1 + r, nz)
        part = cp.asarray(data[lo:hi], dtype=cp.float32)
        if lo > z0 - r or hi < z1 + r:          # edge-pad in z like the whole-array filter would
            part = cp.pad(part, ((lo - (z0 - r), (z1 + r) - hi), (0, 0), (0, 0)), mode="edge")
        f = gf_gpu(part, sigma, mode="nearest", truncate=TRUNCATE)[r:r + (z1 - z0)]
        f = _to_levels(f, dtype, cp)
        out[z0:z1] = f if on_device else cp.asnumpy(f)
    return out


def _to_levels(f: Any, dtype: np.dtype, xp: Any) -> Any:
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        return xp.clip(xp.rint(f), info.min, info.max).astype(dtype)
    return f.astype(dtype)


def iter_blocks(shape_zyx: tuple[int, int, int], block: tuple[int, int, int]) -> Iterator[Box]:
    """Boxes of at most ``block`` tiling ``shape_zyx`` in C order."""
    for z in range(0, shape_zyx[0], block[0]):
        for y in range(0, shape_zyx[1], block[1]):
            for x in range(0, shape_zyx[2], block[2]):
                lo = (z, y, x)
                yield Box(lo, tuple(min(l + b, n) for l, b, n in zip(lo, block, shape_zyx)))


def create_ome_zarr(
    path: str | Path,
    shape_zyx: tuple[int, int, int],
    dtype: str | np.dtype,
    *,
    chunk: int = 128,
    shard: int = 1024,
    codec: Literal["zstd", "none"] = "zstd",
    level: int = 3,
    name: str | None = None,
    overwrite: bool = False,
) -> Any:
    """Create an OME-Zarr v0.5 (Zarr v3) image with one sharded level ``"0"``; returns that array.

    Shards are rounded up to a multiple of the chunk and never exceed what the
    volume needs, so small test volumes get one shard. Every chunk is stored,
    even an empty one, so a missing chunk means an incomplete array
    (:class:`ZarrVolume` warns about those). An existing ``path`` is an error unless
    ``overwrite``, and even then only an existing Zarr store is replaced.
    """
    import zarr
    from zarr.codecs import ZstdCodec

    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; remove it or pass overwrite=True (--overwrite)")
        if not _is_zarr_store(path):
            raise FileExistsError(f"{path} exists and is not a Zarr store; refusing to overwrite it")
    chunks = tuple(min(chunk, n) for n in shape_zyx)
    shards = tuple(c * math.ceil(min(shard, n) / c) for c, n in zip(chunks, shape_zyx))
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    array = group.create_array(
        "0",
        shape=shape_zyx,
        dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=ZstdCodec(level=level) if codec == "zstd" else None,
        dimension_names=("z", "y", "x"),
        fill_value=0,
        config={"write_empty_chunks": True},
    )
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "name": name or Path(path).name,
                "axes": [{"name": a, "type": "space"} for a in ("z", "y", "x")],
                "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}],
            }
        ],
    }
    return array


def _is_zarr_store(path: Path) -> bool:
    return path.is_dir() and any((path / f).exists() for f in ("zarr.json", ".zgroup", ".zarray"))


def write_raw(source: VolumeSource, path: str | Path, *, slab: int = 64) -> None:
    """Stream a volume to a headerless little-endian ``.raw`` file (x fastest), as CCPi reads it."""
    nz = source.shape[0]
    dtype = source.dtype.newbyteorder("<") if source.dtype.itemsize > 1 else source.dtype
    with open(path, "wb") as fh:
        for z in range(0, nz, slab):
            box = Box((z, 0, 0), (min(z + slab, nz), source.shape[1], source.shape[2]))
            fh.write(np.ascontiguousarray(source.read_brick(box, device="cpu").data, dtype=dtype).tobytes())


def convert_to_ome_zarr(
    src: str | Path,
    dst: str | Path,
    *,
    chunk: int = 128,
    shard: int = 1024,
    codec: Literal["zstd", "none"] = "zstd",
    level: int = 3,
    shape_xyz: tuple[int, int, int] | None = None,
    dtype: str | None = None,
    header_bytes: int = 0,
    overwrite: bool = False,
) -> None:
    """One-off conversion of raw/mhd/npy/tiff input to sharded OME-Zarr v3.

    Streams slabs of ``shard`` slices so memory stays bounded for 100 GB+ inputs
    (each slab is written as whole shards). ``.raw`` needs ``shape_xyz`` and
    ``dtype``; ``.tif``/``.tiff`` stacks need ``tifffile`` (``zvdvc[tiff]``).
    The output is written next to ``dst`` and renamed into place at the end, so
    an interrupted conversion leaves no partial ``dst``. An existing ``dst`` is an
    error unless ``overwrite``; ``dst`` may not be ``src`` or contain it, or lie inside it.
    """
    src, dst = Path(src), Path(dst)
    s, d = src.resolve(), dst.resolve()
    if s == d or s in d.parents or d in s.parents:
        raise ValueError(f"convert: the output {dst} would overwrite the input {src}; choose another output path")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists; remove it or pass overwrite=True (--overwrite)")
    if dst.exists() and not _is_zarr_store(dst):
        raise FileExistsError(f"{dst} exists and is not a Zarr store; refusing to overwrite it")
    if src.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("TIFF input needs tifffile: pip install 'zvdvc[tiff]'") from exc
        array = tifffile.memmap(src) if tifffile.TiffFile(src).is_uniform else tifffile.imread(src)
        if array.ndim != 3:
            raise ValueError(f"{src}: expected a 3-D stack, got shape {array.shape}")
        source_shape, source_dtype = tuple(array.shape), np.dtype(array.dtype)
    elif is_zarr_uri(src):
        vol = ZarrVolume(str(src), check_chunks=False, gpu_io="host")   # missing chunks convert to the fill value
        array, source_shape, source_dtype = vol.array, vol.shape, vol.dtype
    else:
        vol = RawVolume(src, shape_xyz=shape_xyz, dtype=dtype, header_bytes=header_bytes, gpu_io="host")
        array, source_shape, source_dtype = vol.array, vol.shape, vol.dtype
    out_dtype = native(source_dtype)
    tmp = d.parent / f".{d.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        out = create_ome_zarr(tmp, source_shape, out_dtype, chunk=chunk, shard=shard, codec=codec, level=level, name=d.name)
        slab = int(out.shards[0])
        for z in range(0, source_shape[0], slab):
            z1 = min(z + slab, source_shape[0])
            out[z:z1] = np.asarray(array[z:z1]).astype(out_dtype, copy=False)
        if dst.exists():                                 # overwrite: move the old store aside only once the new one is complete
            old = d.parent / f".{d.name}.old-{os.getpid()}"
            os.replace(d, old)
            os.replace(tmp, d)
            shutil.rmtree(old)
        else:
            os.replace(tmp, d)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
