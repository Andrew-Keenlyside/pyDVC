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
``RawVolume``
    CCPi/iDVC flat inputs (``.raw`` with header/endianness, ``.mhd``, ``.npy``)
    through ``numpy.memmap``. Used for development and the parity case. Large
    inputs should be converted once with :func:`convert_to_ome_zarr`.

Boxes that extend past the volume are edge-padded. :class:`Brick` records the
valid part, so the solver flags points whose samples leave it as
``RANGE_FAIL`` instead of correlating them against padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from pydvc._todo import todo
from pydvc.config import VolumeSpec
from pydvc.geometry.box import Box
from pydvc.kernels.xp import Device


@dataclass
class Brick:
    data: Any       # (z, y, x) array, numpy or cupy, native dtype
    box: Box        # the box that was asked for
    valid: Box      # the part of ``box`` inside the volume

    @property
    def origin_xyz(self) -> tuple[float, float, float]:
        """Point-space coordinate of ``data[0, 0, 0]``."""
        return (float(self.box.lo[2]), float(self.box.lo[1]), float(self.box.lo[0]))


class VolumeSource(Protocol):
    shape: tuple[int, int, int]   # (z, y, x)
    dtype: np.dtype

    def read_brick(self, box: Box, *, device: Device = "cuda", stream: Any = None) -> Brick: ...


class ZarrVolume:
    def __init__(self, uri: str, array_path: str = "0") -> None:
        raise todo("M3", "ZarrVolume")

    def read_brick(self, box: Box, *, device: Device = "cuda", stream: Any = None) -> Brick:
        raise todo("M3", "ZarrVolume.read_brick")


class RawVolume:
    def __init__(
        self,
        path: str | Path,
        *,
        shape_xyz: tuple[int, int, int] | None = None,
        dtype: str | None = None,
        header_bytes: int = 0,
    ) -> None:
        raise todo("M1", "RawVolume (memmap .raw/.mhd/.npy)")

    def read_brick(self, box: Box, *, device: Device = "cuda", stream: Any = None) -> Brick:
        raise todo("M1", "RawVolume.read_brick")


def open_volume(spec: VolumeSpec, which: Literal["reference", "deformed"]) -> VolumeSource:
    """Pick ``ZarrVolume`` or ``RawVolume`` from the URI."""
    raise todo("M1", "open_volume")


def convert_to_ome_zarr(
    src: str | Path,
    dst: str | Path,
    *,
    chunk: int = 128,
    shard: int = 1024,
    codec: Literal["zstd", "none"] = "zstd",
    level: int = 3,
) -> None:
    """One-off conversion of raw/mhd/npy/tiff input to sharded OME-Zarr v3.

    Streams slabs of ``shard`` slices so memory stays bounded for 100 GB+ inputs.
    """
    raise todo("M3", "convert_to_ome_zarr")
