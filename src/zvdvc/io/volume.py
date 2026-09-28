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
"""

from __future__ import annotations

import math
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


def _read_brick(array: Any, shape: tuple[int, int, int], box: Box, device: Device, stream: Any = None) -> Brick:
    """Read ``box`` from a (z, y, x) array-like, edge-padding whatever lies outside the volume."""
    valid = box.intersect(Box((0, 0, 0), shape))
    if valid is None:
        raise ValueError(f"{box} does not overlap a volume of shape {shape}")
    part = np.asarray(array[valid.slices()])
    pad = [(v - b, bh - vh) for b, v, vh, bh in zip(box.lo, valid.lo, valid.hi, box.hi)]
    data = np.pad(part, pad, mode="edge") if any(p != (0, 0) for p in pad) else np.ascontiguousarray(part)
    if device == "cuda":
        data = to_device(data, stream=stream)
    return Brick(data=data, box=box, valid=valid)


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
    def __init__(self, uri: str, array_path: str = "0") -> None:
        import zarr

        node = zarr.open(uri, mode="r")
        self.array = node[array_path] if isinstance(node, zarr.Group) else node
        if self.array.ndim != 3:
            raise ValueError(f"{uri}: expected a 3-D (z, y, x) array, got shape {self.array.shape}")
        self.uri = uri
        self.shape: tuple[int, int, int] = tuple(int(n) for n in self.array.shape)
        self.dtype = np.dtype(self.array.dtype)

    def read_brick(self, box: Box, *, device: Device = "cuda", stream: Any = None) -> Brick:
        return _read_brick(self.array, self.shape, box, device, stream)


_MHD_TYPES = {
    "MET_UCHAR": "u1", "MET_CHAR": "i1", "MET_USHORT": "u2", "MET_SHORT": "i2",
    "MET_UINT": "u4", "MET_INT": "i4", "MET_FLOAT": "f4", "MET_DOUBLE": "f8",
}


def _read_mhd(path: Path) -> tuple[Path, tuple[int, int, int], np.dtype, int]:
    """(data file, shape_xyz, dtype, header bytes) from a MetaImage header."""
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    shape_xyz = tuple(int(v) for v in fields["DimSize"].split())
    if len(shape_xyz) != 3:
        raise ValueError(f"{path}: expected a 3-D image, DimSize = {fields['DimSize']}")
    big = fields.get("BinaryDataByteOrderMSB", fields.get("ElementByteOrderMSB", "False")).lower() == "true"
    dtype = np.dtype(_MHD_TYPES[fields["ElementType"]]).newbyteorder(">" if big else "<")
    data_file = fields["ElementDataFile"]
    if data_file == "LOCAL":
        raise ValueError(f"{path}: single-file .mha-style data is not supported; use a detached .raw")
    header = int(fields.get("HeaderSize", "0"))
    if header < 0:
        raise ValueError(f"{path}: HeaderSize = -1 (auto) is not supported")
    return path.parent / data_file, shape_xyz, dtype, header


class RawVolume:
    def __init__(
        self,
        path: str | Path,
        *,
        shape_xyz: tuple[int, int, int] | None = None,
        dtype: str | None = None,
        header_bytes: int = 0,
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
                path, shape_xyz, dt, header_bytes = _read_mhd(path)
            else:
                if shape_xyz is None or dtype is None:
                    raise ValueError(f"{path}: raw volumes need raw_shape_xyz and raw_dtype")
                dt = np.dtype(dtype)
            x, y, z = shape_xyz
            array = np.memmap(path, dtype=dt, mode="r", offset=header_bytes, shape=(z, y, x))
            self.header_bytes = int(header_bytes)
        self.path = path
        self.array = array
        self.shape: tuple[int, int, int] = tuple(int(n) for n in array.shape)
        self.dtype = np.dtype(array.dtype)

    def read_brick(self, box: Box, *, device: Device = "cuda", stream: Any = None) -> Brick:
        return _read_brick(self.array, self.shape, box, device, stream)


def is_zarr_uri(uri: str | Path) -> bool:
    p = Path(uri)
    return p.suffix == ".zarr" or p.name.endswith(".ome.zarr") or (p / "zarr.json").exists()


def open_volume(spec: VolumeSpec, which: Literal["reference", "deformed"]) -> VolumeSource:
    """Pick ``ZarrVolume`` or ``RawVolume`` from the URI; wrapped in :class:`FilteredVolume` if ``prefilter_sigma``."""
    uri = getattr(spec, which)
    if is_zarr_uri(uri):
        vol: Any = ZarrVolume(uri, spec.array_path)
    else:
        vol = RawVolume(uri, shape_xyz=spec.raw_shape_xyz, dtype=spec.raw_dtype, header_bytes=spec.raw_header_bytes)
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
        self.margin = int(math.ceil(TRUNCATE * self.sigma))

    def read_brick(self, box: Box, *, device: Device = "cuda", stream: Any = None) -> Brick:
        m = self.margin
        grown = self.inner.read_brick(box.grow(m), device=device, stream=stream)
        data = gaussian_filtered(grown.data, self.sigma, self.dtype)
        inner = tuple(slice(m, n - m) for n in data.shape)
        return Brick(data=data[inner], box=box, valid=box.intersect(Box((0, 0, 0), self.shape)) or box)


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
) -> Any:
    """Create an OME-Zarr v0.5 (Zarr v3) image with one sharded level ``"0"``; returns that array.

    Shards are rounded up to a multiple of the chunk and never exceed what the
    volume needs, so small test volumes get one shard.
    """
    import zarr
    from zarr.codecs import ZstdCodec

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
) -> None:
    """One-off conversion of raw/mhd/npy/tiff input to sharded OME-Zarr v3.

    Streams slabs of ``shard`` slices so memory stays bounded for 100 GB+ inputs
    (each slab is written as whole shards). ``.raw`` needs ``shape_xyz`` and
    ``dtype``; ``.tif``/``.tiff`` stacks need ``tifffile`` (``zvdvc[tiff]``).
    """
    src = Path(src)
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
        vol = ZarrVolume(str(src))
        array, source_shape, source_dtype = vol.array, vol.shape, vol.dtype
    else:
        vol = RawVolume(src, shape_xyz=shape_xyz, dtype=dtype, header_bytes=header_bytes)
        array, source_shape, source_dtype = vol.array, vol.shape, vol.dtype
    out_dtype = source_dtype.newbyteorder("=") if source_dtype.byteorder not in ("=", "|") else source_dtype
    out = create_ome_zarr(dst, source_shape, out_dtype, chunk=chunk, shard=shard, codec=codec, level=level)
    slab = int(out.shards[0])
    for z in range(0, source_shape[0], slab):
        z1 = min(z + slab, source_shape[0])
        out[z:z1] = np.asarray(array[z:z1]).astype(out_dtype, copy=False)
