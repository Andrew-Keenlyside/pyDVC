"""Dense image volumes (reference and deformed), read one brick at a time.

A *brick* is the box a tile needs: the tile's points plus the halo
(:meth:`pydvc.config.RunConfig.halo`). The deformed brick is shifted by the
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

from pydvc.config import VolumeSpec
from pydvc.geometry.box import Box
from pydvc.kernels.xp import Device, get_xp


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
        """``valid`` as a half-open voxel range in point order, for :func:`pydvc.kernels.interpolate.sample`."""
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


def to_device(data: np.ndarray, *, stream: Any = None) -> Any:
    """Host array -> device, staged through pinned memory so the copy is a single DMA on ``stream``."""
    cp = get_xp("cuda")
    import cupyx

    pinned = cupyx.empty_pinned(data.shape, dtype=data.dtype)
    pinned[...] = data
    out = cp.empty(data.shape, dtype=data.dtype)
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
    """Pick ``ZarrVolume`` or ``RawVolume`` from the URI."""
    uri = getattr(spec, which)
    if is_zarr_uri(uri):
        return ZarrVolume(uri, spec.array_path)
    return RawVolume(uri, shape_xyz=spec.raw_shape_xyz, dtype=spec.raw_dtype, header_bytes=spec.raw_header_bytes)


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
    ``dtype``; ``.tif``/``.tiff`` stacks need ``tifffile`` (``pydvc[tiff]``).
    """
    src = Path(src)
    if src.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("TIFF input needs tifffile: pip install 'pydvc[tiff]'") from exc
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
