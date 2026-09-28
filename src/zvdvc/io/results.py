"""DVC results as a zarr-vectors point cloud on the input's chunk grid.

The results store mirrors the search-point store cell for cell and fragment
for fragment. Row ``r`` of cell ``c`` in the input is row ``r`` of cell ``c``
here, so no index table is needed and a tile's writes touch only that tile's cells.

Vertex attributes (level 0):

==============  =========  =====  ===========================================
name            dtype      cols   meaning
==============  =========  =====  ===========================================
point_id        int64      1      CCPi label
status          int8       1      :class:`zvdvc.status.PointStatus`
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
   ``zvdvc repair`` and resume rely on.
3. :meth:`ResultStore.finalize` (coordinator, once): ``rebuild_presence``,
   ``refresh_arrays_present``, ``update_level_metadata``,
   ``write_multiscale_metadata``, and optionally ``build_pyramid`` for
   Neuroglancer overviews.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zvdvc._todo import todo
from zvdvc.io.pointcloud import CellCoord, PointCloud, TilePoints, allocate_store, finalize_store, write_cell
from zvdvc.kernels.xp import Device

RESULT_ATTRIBUTES: dict[str, tuple[str, int | None]] = {
    "point_id": ("int64", 1),
    "objmin": ("float32", 1),
    "displacement": ("float32", 3),
    "params": ("float32", None),     # ncols = dof, fixed at allocate()
    "n_iter": ("uint8", 1),
    "seed": ("float32", 3),
    "status": ("int8", 1),           # written last: a cell with a status is complete (see written_cells)
}
# Written when present in a tile's results; stores created before an attribute existed simply lack it,
# and a cell counts as written without them (written_cells looks at RESULT_ATTRIBUTES only).
OPTIONAL_ATTRIBUTES: dict[str, tuple[str, int]] = {
    "displacement_sd": ("float32", 3),   # per-axis sd over repeat solves (uncertainty_seeds; NaN otherwise)
}
_ATTR = "zvdvc_results"          # root attribute holding the run's layout (dof, grid, source points)
_LEGACY_ATTR = "pydvc_results"   # the same, in stores written before the project was renamed from pyDVC


def _root_attrs(path: str, mode: str = "r") -> Any:
    import zarr

    return zarr.open_group(path, mode=mode, zarr_format=3).attrs


class ResultStore:
    def __init__(self, path: str | Path, mode: str = "r") -> None:
        from zarr_vectors import building as zb

        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        root_attrs = _root_attrs(self.path)
        meta = root_attrs.get(_ATTR, root_attrs.get(_LEGACY_ATTR))
        if meta is None:
            raise ValueError(f"{self.path} is not a zvDVC results store (no '{_ATTR}' attribute)")
        self.meta = dict(meta)
        self.dof = int(self.meta["dof"])
        self.chunk_shape = tuple(self.meta["chunk_shape"])
        self.bin_shape = tuple(self.meta["bin_shape"])
        self._root = zb.open_store(self.path, mode="r+" if mode != "r" else "r")
        self._level = zb.get_resolution_level(self._root, 0)
        self.mode = mode

    @classmethod
    def allocate(cls, path: str | Path, *, points: PointCloud, dof: int, template_digest: str | None = None,
                 prefilter_sigma: float = 0.0) -> ResultStore:
        """Phase 1 (coordinator): create every array, declare presence deferred.

        ``template_digest`` (:meth:`zvdvc.geometry.templates.Template.digest`) is recorded so
        that a resumed or repaired run refuses to mix subvolume templates (:meth:`check_template`).
        """
        attrs = {name: (dtype, dof if ncols is None else ncols) for name, (dtype, ncols) in RESULT_ATTRIBUTES.items()}
        attrs |= OPTIONAL_ATTRIBUTES
        allocate_store(path, bounds=points.bounds, chunk_shape=points.chunk_shape, bin_shape=points.bin_shape, attributes=attrs)
        meta = {
            "dof": dof,
            "chunk_shape": list(points.chunk_shape),
            "bin_shape": list(points.bin_shape),
            "points": str(Path(points.path).resolve()),
            "n_points": points.n_points,
            "template_digest": template_digest,
            "prefilter_sigma": float(prefilter_sigma),
            "optional_attributes": sorted(OPTIONAL_ATTRIBUTES),
        }
        _root_attrs(str(path), "r+")[_ATTR] = meta
        return cls(path, mode="r+")

    def check_template(self, digest: str) -> None:
        """Refuse to add results computed with another subvolume template than the ones already here."""
        stored = self.meta.get("template_digest")
        if stored is None:
            import warnings

            warnings.warn(f"{self.path} records no template digest (written before it was stored); "
                          "cannot check that resumed results use the same subvolume template", stacklevel=2)
        elif stored != digest:
            raise ValueError(f"{self.path} holds results for subvolume template {stored}, not {digest}: the "
                             "subvolume settings or zvDVC's template changed; remove it or choose another output")

    def check_prefilter(self, sigma: float) -> None:
        """Refuse to add results computed from differently filtered volumes (``volumes.prefilter_sigma``)."""
        stored = float(self.meta.get("prefilter_sigma", 0.0))
        if abs(stored - float(sigma)) > 1e-9:
            raise ValueError(f"{self.path} holds results for volumes prefiltered with sigma {stored:g}, not {float(sigma):g}; "
                             "remove it or choose another output")

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
        optional = [n for n in self.meta.get("optional_attributes", []) if n in OPTIONAL_ATTRIBUTES]
        n_rows = len(host["point_id"])
        for name in optional:                            # NaN when a run did not compute it
            if name not in host:
                host[name] = np.full((n_rows, OPTIONAL_ATTRIBUTES[name][1]), np.nan)
        for i, cell in enumerate(tile.cells):
            lo, hi = int(tile.cell_offsets[i]), int(tile.cell_offsets[i + 1])
            attrs = {name: host[name][lo:hi].astype(dtype) for name, (dtype, _) in RESULT_ATTRIBUTES.items() if name != "status"}
            attrs |= {name: host[name][lo:hi].astype(OPTIONAL_ATTRIBUTES[name][0]) for name in optional}
            attrs["status"] = host["status"][lo:hi].astype(RESULT_ATTRIBUTES["status"][0])   # last: marks the cell complete
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
        optional = [n for n in self.meta.get("optional_attributes", []) if n in OPTIONAL_ATTRIBUTES]
        if optional and cells:
            extra = zb.read_cells(self._level, np.asarray(cells, dtype=np.int64).reshape(-1, 3),
                                  [f"vertex_attributes/{n}" for n in optional], on_error="raise")
            for name in optional:
                dtype, cols = OPTIONAL_ATTRIBUTES[name]
                out[name] = np.asarray(extra[f"vertex_attributes/{name}"].data, dtype=dtype).reshape(-1, cols)
        return out

    def finalize(self, *, n_points: int, pyramid: bool = False) -> None:
        """Phase 3 (coordinator): rebuild presence and write metadata. Never run while workers are writing."""
        finalize_store(self.path, n_points=n_points, pyramid=pyramid)
