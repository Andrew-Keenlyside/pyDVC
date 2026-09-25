"""Search points: import, zarr-vectors store, and per-tile reads.

Search points are stored as a zarr-vectors point cloud. **Its chunk grid is
pyDVC's work partition.** Tiles are unions of whole chunks, so the cells a
worker reads are exactly the cells it later writes in the results store.

Store layout (level 0)::

    vertices/                   (x, y, z) float32, voxel units
    vertex_fragments/           one fragment per bin (zarr-vectors guidance, keeps pyramids useful)
    vertex_attributes/point_id  int64: CCPi label (``n`` column of .roi/.disp)

* ``chunk_shape`` is chosen so ``cluster.tile_shape`` is an integer multiple of it.
* Tile reads use ``zarr_vectors.building.read_cells(level, cells,
  ["vertices", "vertex_attributes/point_id"], device="cuda")``. The result is a
  ``CellBatch`` whose ``offsets`` are CSR over cells, delivered to the GPU in one pooled read.
* Large clouds are written with zarr-vectors' three-phase HPC pattern:
  ``create_store`` + ``defer_presence``, then workers write disjoint cells with
  ``write_chunk_vertices(..., record_presence=False)``, then ``rebuild_presence``.
* Only ``zarr_vectors.api`` / ``zarr_vectors.building`` are used. Bin assignment
  (``zarr_vectors.spatial.chunking.assign_bins``) is internal upstream, so pyDVC
  assigns bins itself in :func:`write_pointcloud_store`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pydvc._todo import todo
from pydvc.kernels.xp import Device

CellCoord = tuple[int, int, int]


@dataclass
class TilePoints:
    xyz: Any                   # (N, 3) float32, device or host
    point_id: Any              # (N,) int64
    cells: list[CellCoord]     # the tile's cells, in read order
    cell_offsets: Any          # (len(cells) + 1,) CSR offsets into xyz / point_id
    bin_offsets: Any           # per-cell fragment (bin) offsets, so results reuse the input's fragments


def read_roi(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a CCPi ``.roi`` or iDVC ``.txt``/``.csv`` point cloud.

    Returns ``(point_id (N,) int64, xyz (N, 3) float64)``. The first point is
    CCPi's default start point.
    """
    raise todo("M1", "read_roi")


def write_pointcloud_store(
    path: str | Path,
    xyz: np.ndarray,
    point_id: np.ndarray,
    *,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    chunk_shape: tuple[float, float, float],
    bin_shape: tuple[float, float, float] | None = None,
) -> None:
    """Write a zarr-vectors point cloud, one fragment per bin."""
    raise todo("M3", "write_pointcloud_store")


class PointCloud:
    """Read side of a search-point store."""

    def __init__(self, path: str | Path) -> None:
        raise todo("M3", "PointCloud")

    @property
    def chunk_shape(self) -> tuple[float, float, float]:
        raise todo("M3", "PointCloud.chunk_shape")

    @property
    def n_points(self) -> int:
        raise todo("M3", "PointCloud.n_points")

    def cells(self) -> list[CellCoord]:
        """Occupied chunk coordinates, from metadata only."""
        raise todo("M3", "PointCloud.cells")

    def cell_counts(self) -> dict[CellCoord, int]:
        """Points per cell, used by the tile cost model."""
        raise todo("M3", "PointCloud.cell_counts")

    def read_tile(self, cells: list[CellCoord], *, device: Device = "cuda") -> TilePoints:
        raise todo("M3", "PointCloud.read_tile")

    def read_all(self, *, device: Device = "cpu") -> tuple[Any, Any]:
        """Every point, as ``(xyz, point_id)``. Used for kNN and seeding on the coordinator."""
        raise todo("M3", "PointCloud.read_all")
