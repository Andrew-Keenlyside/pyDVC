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

import numpy as np

from pydvc._todo import todo
from pydvc.io.pointcloud import CellCoord, PointCloud, TilePoints, allocate_store, finalize_store, write_cell
from pydvc.kernels.xp import Device

RESULT_ATTRIBUTES: dict[str, tuple[str, int | None]] = {
    "point_id": ("int64", 1),
    "objmin": ("float32", 1),
    "displacement": ("float32", 3),
    "params": ("float32", None),     # ncols = dof, fixed at allocate()
    "n_iter": ("uint8", 1),
    "seed": ("float32", 3),
    "status": ("int8", 1),           # written last: a cell with a status is complete (see written_cells)
}
_ATTR = "pydvc_results"          # root attribute holding the run's layout (dof, grid, source points)


def _root_attrs(path: str, mode: str = "r") -> Any:
    import zarr

    return zarr.open_group(path, mode=mode, zarr_format=3).attrs


class ResultStore:
    def __init__(self, path: str | Path, mode: str = "r") -> None:
        from zarr_vectors import building as zb

        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        meta = _root_attrs(self.path).get(_ATTR)
        if meta is None:
            raise ValueError(f"{self.path} is not a pyDVC results store (no '{_ATTR}' attribute)")
        self.meta = dict(meta)
        self.dof = int(self.meta["dof"])
        self.chunk_shape = tuple(self.meta["chunk_shape"])
        self.bin_shape = tuple(self.meta["bin_shape"])
        self._root = zb.open_store(self.path, mode="r+" if mode != "r" else "r")
        self._level = zb.get_resolution_level(self._root, 0)
        self.mode = mode

    @classmethod
    def allocate(cls, path: str | Path, *, points: PointCloud, dof: int) -> ResultStore:
        """Phase 1 (coordinator): create every array, declare presence deferred."""
        attrs = {name: (dtype, dof if ncols is None else ncols) for name, (dtype, ncols) in RESULT_ATTRIBUTES.items()}
        allocate_store(path, bounds=points.bounds, chunk_shape=points.chunk_shape, bin_shape=points.bin_shape, attributes=attrs)
        meta = {
            "dof": dof,
            "chunk_shape": list(points.chunk_shape),
            "bin_shape": list(points.bin_shape),
            "points": str(Path(points.path).resolve()),
            "n_points": points.n_points,
        }
        _root_attrs(str(path), "r+")[_ATTR] = meta
        return cls(path, mode="r+")

    def write_tile(self, tile: TilePoints, results: dict[str, Any]) -> None:
        """Phase 2 (worker): write one tile's cells. ``results`` maps attribute name to (N, cols) arrays in tile row order."""
        if self.mode == "r":
            raise PermissionError("results store opened read-only")
        host = {k: np.asarray(v.get() if hasattr(v, "get") else v) for k, v in results.items()}
        missing = set(RESULT_ATTRIBUTES) - set(host) - {"point_id"}
        if missing:
            raise ValueError(f"missing result attributes {sorted(missing)}")
        host["point_id"] = np.asarray(tile.point_id.get() if hasattr(tile.point_id, "get") else tile.point_id)
        xyz = np.asarray(tile.xyz.get() if hasattr(tile.xyz, "get") else tile.xyz)
        for i, cell in enumerate(tile.cells):
            lo, hi = int(tile.cell_offsets[i]), int(tile.cell_offsets[i + 1])
            attrs = {name: host[name][lo:hi].astype(dtype) for name, (dtype, _) in RESULT_ATTRIBUTES.items()}
            write_cell(self._level, cell, xyz[lo:hi], attrs, np.asarray(tile.bin_offsets[i]))

    def written_cells(self) -> set[CellCoord]:
        """Cells already on disk, used to resume after a crash or pre-emption.

        A cell counts only if every result array holds it. ``write_tile``
        writes ``status`` last, so a worker killed half-way through a cell
        leaves a cell that does not count, and the resubmitted job rewrites it.
        """
        from zarr_vectors import building as zb

        arrays = ["vertices"] + [f"vertex_attributes/{n}" for n in RESULT_ATTRIBUTES]
        cells = None
        for name in arrays:
            keys = {tuple(int(v) for v in c) for c in zb.list_chunk_keys(self._level, name)}
            cells = keys if cells is None else cells & keys
        return cells or set()

    def read_neighbourhood(self, cell: CellCoord, *, halo: int = 1, device: Device = "cuda") -> dict[str, Any]:
        """A cell and its neighbours, in one ``zarr_vectors.building.read_neighbourhood`` prefetch."""
        raise todo("M5", "ResultStore.read_neighbourhood")

    def read_all(self) -> dict[str, np.ndarray]:
        """Every written point: ``xyz`` and each result attribute, in cell order."""
        from zarr_vectors import building as zb

        cells = sorted(self.written_cells())
        names = ["vertices"] + [f"vertex_attributes/{n}" for n in RESULT_ATTRIBUTES]
        batch = zb.read_cells(self._level, np.asarray(cells, dtype=np.int64).reshape(-1, 3), names, on_error="raise")
        out = {"xyz": np.asarray(batch["vertices"].data, dtype=np.float32).reshape(-1, 3)}
        for name, (dtype, ncols) in RESULT_ATTRIBUTES.items():
            cols = self.dof if ncols is None else ncols
            data = np.asarray(batch[f"vertex_attributes/{name}"].data, dtype=dtype)
            out[name] = data.reshape(-1, cols) if cols > 1 else data.reshape(-1)
        return out

    def finalize(self, *, n_points: int, pyramid: bool = False) -> None:
        """Phase 3 (coordinator): rebuild presence and write metadata. Never run while workers are writing."""
        finalize_store(self.path, n_points=n_points, pyramid=pyramid)
