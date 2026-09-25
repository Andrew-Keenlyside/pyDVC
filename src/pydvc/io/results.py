"""DVC results as a zarr-vectors point cloud on the input's chunk grid.

The results store mirrors the search-point store cell for cell and fragment
for fragment. Row ``r`` of cell ``c`` in the input is row ``r`` of cell ``c``
here, so no index table is needed and a tile's writes touch only that tile's cells.

Vertex attributes (level 0):

==============  =========  =====  ===========================================
name            dtype      cols   meaning
==============  =========  =====  ===========================================
point_id        int64      1      CCPi label
status          int8       1      :class:`pydvc.status.PointStatus`
objmin          float32    1      objective at the solution (CCPi ``objmin``)
displacement    float32    3      u, v, w (voxels)
params          float32    dof    full parameter vector (CCPi order)
n_iter          uint8      1      Gauss-Newton iterations used
seed            float32    3      starting displacement (for audit / repair)
strain          float32    6      exx, eyy, ezz, exy, eyz, exz (M5, post)
==============  =========  =====  ===========================================

Writes follow zarr-vectors' three-phase HPC pattern:

1. :meth:`ResultStore.allocate` (coordinator, once): ``create_store``,
   ``create_resolution_level``, ``create_vertices_array``, one
   ``create_attribute_array`` per row above, then ``defer_presence``.
   Workers never create arrays, because re-creating one inside a write
   session drops cells siblings have already written.
2. :meth:`ResultStore.write_tile` (workers, in parallel):
   ``write_chunk_vertices`` / ``write_chunk_attributes`` with
   ``record_presence=False`` on the tile's own cells only. Written cells are
   visible to readers straight away on a deferred level, which is what
   ``pydvc repair`` and resume rely on.
3. :meth:`ResultStore.finalize` (coordinator, once): ``rebuild_presence``,
   ``refresh_arrays_present``, ``update_level_metadata``,
   ``write_multiscale_metadata``, and optionally ``build_pyramid`` for
   Neuroglancer overviews.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydvc._todo import todo
from pydvc.io.pointcloud import CellCoord, PointCloud, TilePoints
from pydvc.kernels.xp import Device

RESULT_ATTRIBUTES: dict[str, tuple[str, int | None]] = {
    "point_id": ("int64", 1),
    "status": ("int8", 1),
    "objmin": ("float32", 1),
    "displacement": ("float32", 3),
    "params": ("float32", None),     # ncols = dof, fixed at allocate()
    "n_iter": ("uint8", 1),
    "seed": ("float32", 3),
}


class ResultStore:
    def __init__(self, path: str | Path, mode: str = "r") -> None:
        raise todo("M3", "ResultStore.open")

    @classmethod
    def allocate(cls, path: str | Path, *, points: PointCloud, dof: int) -> ResultStore:
        """Phase 1 (coordinator): create every array, declare presence deferred."""
        raise todo("M3", "ResultStore.allocate")

    def write_tile(self, tile: TilePoints, results: dict[str, Any]) -> None:
        """Phase 2 (worker): write one tile's cells. ``results`` maps attribute name to (N, cols) arrays in tile row order."""
        raise todo("M3", "ResultStore.write_tile")

    def written_cells(self) -> set[CellCoord]:
        """Cells already on disk, used to resume after a crash or pre-emption."""
        raise todo("M4", "ResultStore.written_cells")

    def read_neighbourhood(self, cell: CellCoord, *, halo: int = 1, device: Device = "cuda") -> dict[str, Any]:
        """A cell and its neighbours, in one ``zarr_vectors.building.read_neighbourhood`` prefetch."""
        raise todo("M5", "ResultStore.read_neighbourhood")

    def finalize(self, *, n_points: int, pyramid: bool = False) -> None:
        """Phase 3 (coordinator): rebuild presence and write metadata. Never run while workers are writing."""
        raise todo("M4", "ResultStore.finalize")
