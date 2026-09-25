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
  assigns bins itself (:func:`bin_index`).

Cells and bins follow zarr-vectors: cell ``floor(x / chunk_shape)`` (store
bounds start at 0), and within a cell, fragment ``(bx * nby + by) * nbz + bz``
for the bin ``b = floor((x - cell_origin) / bin_shape)``. A cell's rows are its
fragments in order, each in input order. Bins are recomputed from the stored
float32 positions whenever they are needed, which reproduces them exactly, so
the results store can reuse the input's fragments without reading fragment
metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pydvc.kernels.xp import Device

CellCoord = tuple[int, int, int]


@dataclass
class TilePoints:
    xyz: Any                   # (N, 3) float32, device or host
    point_id: Any              # (N,) int64
    cells: list[CellCoord]     # the tile's cells, in read order
    cell_offsets: Any          # (len(cells) + 1,) CSR offsets into xyz / point_id (host)
    bin_offsets: Any           # per cell, (n_bins + 1,) fragment offsets within the cell (host), so results reuse the input's fragments


def _bins_per_chunk(chunk_shape: tuple[float, ...], bin_shape: tuple[float, ...]) -> tuple[int, int, int]:
    ratio = [c / b for c, b in zip(chunk_shape, bin_shape)]
    if any(abs(r - round(r)) > 1e-9 or round(r) < 1 for r in ratio):
        raise ValueError(f"bin_shape {bin_shape} must divide chunk_shape {chunk_shape}")
    return tuple(int(round(r)) for r in ratio)


def bin_index(xyz: np.ndarray, cell: CellCoord, chunk_shape: tuple[float, ...], bin_shape: tuple[float, ...]) -> np.ndarray:
    """Fragment index of each point of ``cell`` (C order over (x, y, z) bins, as zarr-vectors numbers them)."""
    nb = _bins_per_chunk(chunk_shape, bin_shape)
    origin = np.asarray(cell, dtype=np.float64) * np.asarray(chunk_shape, dtype=np.float64)
    local = np.floor((np.asarray(xyz, dtype=np.float64) - origin) / np.asarray(bin_shape, dtype=np.float64)).astype(np.int64)
    local = np.clip(local, 0, np.asarray(nb) - 1)
    return (local[:, 0] * nb[1] + local[:, 1]) * nb[2] + local[:, 2]


def group_cells(xyz: np.ndarray, chunk_shape: tuple[float, ...]) -> dict[CellCoord, np.ndarray]:
    """Row indices per cell, in input order."""
    xyz32 = np.asarray(xyz, dtype=np.float32).astype(np.float64)
    cells = np.floor(xyz32 / np.asarray(chunk_shape, dtype=np.float64)).astype(np.int64)
    if len(cells) == 0:
        return {}
    order = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]), axis=0)
    keys = cells[order]
    bounds = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
    out = {}
    for part in np.split(order, bounds):
        out[tuple(int(v) for v in cells[part[0]])] = np.sort(part)
    return out


def cell_fragments(xyz: np.ndarray, cell: CellCoord, chunk_shape: tuple[float, ...], bin_shape: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Row permutation putting a cell's points in fragment order, and the (n_bins + 1,) fragment offsets."""
    nbins = int(np.prod(_bins_per_chunk(chunk_shape, bin_shape)))
    b = bin_index(np.asarray(xyz, dtype=np.float32).astype(np.float64), cell, chunk_shape, bin_shape)
    order = np.argsort(b, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(np.bincount(b, minlength=nbins))])
    return order, offsets


def read_roi(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a CCPi ``.roi`` or iDVC ``.txt``/``.csv`` point cloud.

    Returns ``(point_id (N,) int64, xyz (N, 3) float64)``. The first point is
    CCPi's default start point. Lines are ``n x y z`` separated by tabs, spaces
    or commas; blank lines, ``#`` comments and non-numeric header lines are skipped.
    """
    ids: list[int] = []
    rows: list[tuple[float, float, float]] = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#", 1)[0].replace(",", " ").strip()
        if not line:
            continue
        fields = line.split()
        try:
            values = [float(v) for v in fields[:4]]
        except ValueError:
            if ids:
                raise ValueError(f"{path}:{lineno}: cannot parse {line!r}") from None
            continue                                  # header line
        if len(values) < 4:
            raise ValueError(f"{path}:{lineno}: expected 'n x y z', got {line!r}")
        ids.append(int(values[0]))
        rows.append((values[1], values[2], values[3]))
    return np.asarray(ids, dtype=np.int64), np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def default_bin_shape(chunk_shape: tuple[float, float, float]) -> tuple[float, float, float]:
    """Two bins per axis (8 fragments per cell), zarr-vectors' guidance for useful pyramids."""
    return tuple(c / 2.0 for c in chunk_shape)


def write_pointcloud_store(
    path: str | Path,
    xyz: np.ndarray,
    point_id: np.ndarray,
    *,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    chunk_shape: tuple[float, float, float],
    bin_shape: tuple[float, float, float] | None = None,
) -> None:
    """Write a zarr-vectors point cloud, one fragment per bin.

    Uses the three-phase pattern (allocate + ``defer_presence``, write disjoint
    cells, ``rebuild_presence``) in one process, so the same functions scale to
    a cluster by running phase 2 per partition.
    """
    from zarr_vectors import building as zb

    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    point_id = np.asarray(point_id, dtype=np.int64).reshape(-1)
    if len(xyz) != len(point_id):
        raise ValueError("xyz and point_id differ in length")
    if (xyz < np.asarray(bounds[0])).any() or (xyz > np.asarray(bounds[1])).any():
        raise ValueError("points lie outside the store bounds")
    bin_shape = bin_shape or default_bin_shape(chunk_shape)
    level = allocate_store(path, bounds=bounds, chunk_shape=chunk_shape, bin_shape=bin_shape,
                           attributes={"point_id": ("int64", 1)})
    with zb.open_write_session(level, bounds=[list(bounds[0]), list(bounds[1])], chunk_shape=tuple(chunk_shape)):
        for cell, rows in group_cells(xyz, chunk_shape).items():
            order, offsets = cell_fragments(xyz[rows], cell, chunk_shape, bin_shape)
            write_cell(level, cell, xyz[rows][order], {"point_id": point_id[rows][order]}, offsets)
    finalize_store(path, n_points=len(xyz))


def allocate_store(
    path: str | Path,
    *,
    bounds: tuple[Any, Any],
    chunk_shape: tuple[float, float, float],
    bin_shape: tuple[float, float, float],
    attributes: dict[str, tuple[str, int]],
) -> Any:
    """Phase 1: create the store, level 0, the vertex and attribute arrays; defer presence. Returns the level."""
    from zarr_vectors import building as zb

    b = [list(map(float, bounds[0])), list(map(float, bounds[1]))]
    root = zb.create_store(str(path), bounds=b, chunk_shape=tuple(map(float, chunk_shape)),
                           base_bin_shape=tuple(map(float, bin_shape)), geometry_types=["point_cloud"])
    level = zb.create_resolution_level(
        root, 0, zb.LevelMetadata(level=0, vertex_count=0, arrays_present=["vertices", "vertex_attributes"])
    )
    with zb.open_write_session(level, bounds=b, chunk_shape=tuple(map(float, chunk_shape))):
        zb.create_vertices_array(level, dtype="float32")
        for name, (dtype, ncols) in attributes.items():
            zb.create_attribute_array(level, name, dtype=dtype, ncols=ncols if ncols and ncols > 1 else None)
    zb.defer_presence(level)
    return zb.get_resolution_level(zb.open_store(str(path), mode="r+"), 0)


def write_cell(level: Any, cell: CellCoord, xyz: np.ndarray, attributes: dict[str, np.ndarray], offsets: np.ndarray) -> None:
    """Phase 2: one cell's vertices and attributes, split into fragments at ``offsets``; presence not recorded."""
    from zarr_vectors import building as zb

    def split(a: np.ndarray) -> list[np.ndarray]:
        return [a[offsets[k]:offsets[k + 1]] for k in range(len(offsets) - 1)]

    zb.write_chunk_vertices(level, tuple(cell), split(np.asarray(xyz, dtype=np.float32)), dtype="float32", record_presence=False)
    for name, values in attributes.items():
        v = np.asarray(values)
        zb.write_chunk_attributes(level, name, tuple(cell), split(v), dtype=v.dtype, record_presence=False)


def finalize_store(path: str | Path, *, n_points: int, pyramid: bool = False) -> None:
    """Phase 3: rebuild presence, record what is present, write level and multiscale metadata."""
    import zarr_vectors as zv
    from zarr_vectors import building as zb

    root = zb.open_store(str(path), mode="r+")
    level = zb.get_resolution_level(root, 0)
    zb.rebuild_presence(level)
    zb.refresh_arrays_present(level)
    zb.update_level_metadata(level, vertex_count=int(n_points))
    zb.write_multiscale_metadata(root)
    if pyramid:
        zv.open(str(path), mode="r+").build_pyramid(factors=[(2.0, 1.0), (2.0, 1.0)], chunk_scale_factors=[2, 2])


class PointCloud:
    """Read side of a search-point store."""

    def __init__(self, path: str | Path) -> None:
        from zarr_vectors import building as zb

        self.path = str(path)
        self._root = zb.open_store(self.path, mode="r")
        self._level = zb.get_resolution_level(self._root, 0)
        meta = zb.read_root_metadata(self._root)
        self._chunk = tuple(float(v) for v in meta.chunk_shape)
        self._bin = tuple(float(v) for v in (meta.base_bin_shape or self._chunk))
        self.bounds = (tuple(meta.bounds[0]), tuple(meta.bounds[1]))
        self._n = int(zb.read_level_metadata(self._root, 0).vertex_count)

    @property
    def chunk_shape(self) -> tuple[float, float, float]:
        return self._chunk

    @property
    def bin_shape(self) -> tuple[float, float, float]:
        return self._bin

    @property
    def n_points(self) -> int:
        return self._n

    def cells(self) -> list[CellCoord]:
        """Occupied chunk coordinates, from metadata only."""
        from zarr_vectors import building as zb

        return sorted(tuple(int(v) for v in c) for c in zb.list_chunk_keys(self._level))

    def cell_counts(self) -> dict[CellCoord, int]:
        """Points per cell, used by the tile cost model."""
        from zarr_vectors import building as zb

        return {c: int(zb.chunk_vertex_count(self._level, c)) for c in self.cells()}

    def read_tile(self, cells: list[CellCoord], *, device: Device = "cuda") -> TilePoints:
        from zarr_vectors import building as zb

        cells = [tuple(int(v) for v in c) for c in cells]
        batch = zb.read_cells(
            self._level, np.asarray(cells, dtype=np.int64).reshape(-1, 3), ["vertices", "vertex_attributes/point_id"],
            on_error="raise",
        )
        xyz = np.asarray(batch["vertices"].data, dtype=np.float32).reshape(-1, 3)
        point_id = np.asarray(batch["vertex_attributes/point_id"].data, dtype=np.int64).reshape(-1)
        cell_offsets = np.asarray(batch["vertices"].offsets, dtype=np.int64)
        bins = []
        for i, cell in enumerate(cells):
            rows = xyz[cell_offsets[i]:cell_offsets[i + 1]]
            _, offsets = cell_fragments(rows, cell, self._chunk, self._bin)
            bins.append(offsets)
        if device == "cuda":
            from pydvc.kernels.xp import get_xp

            cp = get_xp("cuda")
            xyz, point_id = cp.asarray(xyz), cp.asarray(point_id)
        return TilePoints(xyz=xyz, point_id=point_id, cells=cells, cell_offsets=cell_offsets, bin_offsets=bins)

    def read_all(self, *, device: Device = "cpu") -> tuple[Any, Any]:
        """Every point, as ``(xyz, point_id)``, in cell order. Used for kNN and seeding on the coordinator."""
        tile = self.read_tile(self.cells(), device=device)
        return tile.xyz, tile.point_id

    def cell_extents(self) -> dict[CellCoord, tuple[np.ndarray, np.ndarray]]:
        """Per cell, the (min, max) point coordinates: tight brick boxes for tile planning."""
        tile = self.read_tile(self.cells(), device="cpu")
        out = {}
        for i, cell in enumerate(tile.cells):
            rows = tile.xyz[tile.cell_offsets[i]:tile.cell_offsets[i + 1]].astype(np.float64)
            if len(rows):
                out[cell] = (rows.min(axis=0), rows.max(axis=0))
        return out


def import_points(
    roi_path: str | Path, store_path: str | Path, *, volume_shape_zyx: tuple[int, int, int], chunk_shape: tuple[float, float, float]
) -> PointCloud:
    """Convert a CCPi ``.roi`` into a point-cloud store bounded by the volume."""
    point_id, xyz = read_roi(roi_path)
    bounds = ((0.0, 0.0, 0.0), tuple(float(n) for n in volume_shape_zyx[::-1]))
    write_pointcloud_store(store_path, xyz, point_id, bounds=bounds, chunk_shape=chunk_shape)
    return PointCloud(store_path)


def is_store(path: str | Path) -> bool:
    p = Path(path)
    return p.suffix in (".zarrvectors", ".zv") or (p / "zarr.json").exists()


def chunk_shape_for(tile_shape_zyx: tuple[int, int, int], per_tile: int = 4) -> tuple[float, float, float]:
    """Default cell size: ``per_tile`` cells per tile edge (e.g. 256 for 1024^3 tiles)."""
    return tuple(float(max(1, math.ceil(t / per_tile))) for t in tile_shape_zyx[::-1])
