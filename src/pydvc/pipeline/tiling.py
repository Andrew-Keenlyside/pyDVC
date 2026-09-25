"""Tiles: the unit of work handed to a GPU.

A tile is a block of whole zarr-vectors point-cloud chunks
(``cluster.tile_shape`` / ``chunk_shape`` chunks per axis). Because tiles never
share a cell, workers write results without locks, and a tile can be retried
or resumed on its own.

Each tile carries two brick boxes:

* ``ref_box``: the tile's point bbox grown by the halo
  (:meth:`pydvc.config.RunConfig.halo`).
* ``def_box``: ``ref_box`` shifted by the tile's median seed and grown by the
  seed spread within the tile. A large rigid offset (the CCPi example has
  ``rigid_trans = 34, 4, 0``) therefore moves the brick instead of fattening it.

Read amplification per volume is ``((T + 2h) / T)^3`` for tile edge ``T`` and
halo ``h``:

=================================  ======  ========  ========
case (sphere, h = S/2 + dmax + 2)  h       T = 512   T = 1024
=================================  ======  ========  ========
S=48, disp_max=10                  36      1.49      1.23
S=80, disp_max=38 (CCPi example)   80      2.26      1.55
=================================  ======  ========  ========

With 80 GB per H100 and ``prefetch_depth = 2``, 1024^3 u16 tiles use about
2 bricks x 2 in flight x 1096^3 x 2 B = 10.5 GB. That is why ``T = 1024`` is
the H100 default (docs/ARCHITECTURE.md, memory budget).

Assignment: LPT (longest processing time first) over nodes, using the cost
model ``n_points x n_samples x iterations x dof_factor``. Within a node a
shared queue hands out tiles dynamically, which absorbs errors in the cost
model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pydvc.config import RunConfig
from pydvc.geometry.box import Box
from pydvc.io.pointcloud import CellCoord, PointCloud

# relative Gauss-Newton cost per sample by DOF (sums grow as dof^2), and a typical iteration count
_DOF_FACTOR = {3: 1.0, 6: 1.3, 12: 1.9}
_ITERATIONS = 5


@dataclass(frozen=True)
class Tile:
    id: int
    cells: tuple[CellCoord, ...]
    n_points: int
    points_box: Box
    ref_box: Box
    def_box: Box
    cost: float


@dataclass(frozen=True)
class MemoryReport:
    brick_bytes: int            # ref + def for the largest tile, native dtype
    in_flight_bytes: int        # x prefetch_depth
    batch_bytes: int            # solver scratch at the chosen batch size
    device_bytes: int           # usable = gpu_memory_fraction x total
    fits: bool


def cells_per_tile(cfg: RunConfig, chunk_shape: tuple[float, float, float]) -> tuple[int, int, int]:
    """Cells per tile along (x, y, z); the tile shape must be a whole number of cells."""
    tile_xyz = cfg.cluster.tile_shape[::-1]
    ratio = [t / c for t, c in zip(tile_xyz, chunk_shape)]
    if any(abs(r - round(r)) > 1e-9 or round(r) < 1 for r in ratio):
        raise ValueError(f"tile_shape {cfg.cluster.tile_shape} (z, y, x) is not a multiple of the point-cloud chunk {chunk_shape} (x, y, z)")
    return tuple(int(round(r)) for r in ratio)


def load_seed_field(path: str | Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    """``(point_id, seed)`` sorted by id, from ``seeds.npz`` written by the seed stage."""
    if path is None or not Path(path).exists():
        return None
    with np.load(path) as f:
        ids, seeds = f["point_id"].astype(np.int64), f["seed"].astype(np.float64)
    order = np.argsort(ids)
    return ids[order], seeds[order]


def lookup_seeds(point_id: np.ndarray, field: tuple[np.ndarray, np.ndarray] | None, rigid: tuple[float, float, float]) -> np.ndarray:
    """Seeds for ``point_id``: from the seed field where present, else ``rigid_trans``."""
    out = np.broadcast_to(np.asarray(rigid, dtype=np.float64), (len(point_id), 3)).copy()
    if field is not None and len(point_id):
        ids, seeds = field
        pos = np.clip(np.searchsorted(ids, point_id), 0, len(ids) - 1)
        hit = ids[pos] == point_id
        out[hit] = seeds[pos[hit]]
    return out


def plan_tiles(points: PointCloud, cfg: RunConfig, *, seed_field_path: str | Path | None = None) -> list[Tile]:
    """Group occupied chunks into tiles and compute brick boxes and costs.

    Reads the points once (positions for tight point boxes, ids for seeds);
    never touches the image volumes.
    """
    per_tile = cells_per_tile(cfg, points.chunk_shape)
    everything = points.read_tile(points.cells(), device="cpu")
    field = load_seed_field(seed_field_path)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, cell in enumerate(everything.cells):
        key = tuple(c // k for c, k in zip(cell, per_tile))
        groups.setdefault(key, []).append(i)
    halo = cfg.halo()
    n_samples = _n_samples(cfg)
    tiles = []
    for tid, key in enumerate(sorted(groups)):
        idx = groups[key]
        rows = np.concatenate([np.arange(everything.cell_offsets[i], everything.cell_offsets[i + 1]) for i in idx])
        cells = tuple(everything.cells[i] for i in idx)
        if len(rows) == 0:
            continue
        xyz = everything.xyz[rows].astype(np.float64)
        pbox = Box.around_points(tuple(xyz.min(axis=0)), tuple(xyz.max(axis=0)))
        ref_box = pbox.grow(halo)
        seeds = lookup_seeds(everything.point_id[rows], field, cfg.search.rigid_trans)
        centre = np.median(seeds, axis=0)
        spread = np.abs(seeds - centre).max(axis=0)
        def_box = ref_box.shift_xyz(tuple(centre)).grow(tuple(math.ceil(v) for v in spread[::-1]))
        cost = float(len(rows) * n_samples * _ITERATIONS * _DOF_FACTOR[cfg.search.dof])
        tiles.append(Tile(len(tiles), cells, len(rows), pbox, ref_box, def_box, cost))
    return tiles


def _n_samples(cfg: RunConfig) -> int:
    if cfg.subvolume.geometry == "cube":
        from pydvc.geometry.templates import cube_side

        return cube_side(cfg.subvolume.n_samples) ** 3
    return cfg.subvolume.n_samples


def check_memory(tiles: list[Tile], cfg: RunConfig, *, device_total_bytes: int, dtype_bytes: int) -> MemoryReport:
    from pydvc.pipeline.batching import batch_size

    brick = max((t.ref_box.voxels() + t.def_box.voxels()) * dtype_bytes for t in tiles) if tiles else 0
    in_flight = brick * max(cfg.cluster.prefetch_depth, 1)
    usable = int(cfg.cluster.gpu_memory_fraction * device_total_bytes)
    n_samples = _n_samples(cfg)
    largest = max((t.n_points for t in tiles), default=0)
    b = cfg.cluster.batch_points or batch_size(max(usable - in_flight, 1), n_samples, cfg.search.dof)
    b = min(b, max(largest, 1))
    per_point = n_samples * 9 + 4 * (2 * cfg.search.dof + 128)
    batch_bytes = b * per_point
    return MemoryReport(brick, in_flight, batch_bytes, usable, in_flight + batch_bytes <= usable)


def assign_lpt(tiles: list[Tile], n_bins: int) -> list[list[int]]:
    """Tile ids per bin (node), balancing summed cost: longest processing time first."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    loads = [0.0] * n_bins
    shares: list[list[int]] = [[] for _ in range(n_bins)]
    for t in sorted(tiles, key=lambda t: (-t.cost, t.id)):
        k = min(range(n_bins), key=lambda i: (loads[i], i))
        shares[k].append(t.id)
        loads[k] += t.cost
    return shares


def _box(b: Box) -> dict[str, list[int]]:
    return {"lo": list(b.lo), "hi": list(b.hi)}


def _unbox(d: dict[str, list[int]]) -> Box:
    return Box(tuple(d["lo"]), tuple(d["hi"]))


def save_plan(path: str | Path, tiles: list[Tile], assignment: list[list[int]], cfg: RunConfig, **extra: Any) -> None:
    """``plan.json``: tiles, node assignment, template hash, package versions (zarr-vectors commit included)."""
    import importlib.metadata as md

    from pydvc.geometry.templates import make_template

    def version(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    doc = {
        "format": 1,
        "tiles": [
            dict(id=t.id, cells=[list(c) for c in t.cells], n_points=t.n_points, points_box=_box(t.points_box),
                 ref_box=_box(t.ref_box), def_box=_box(t.def_box), cost=t.cost)
            for t in tiles
        ],
        "assignment": assignment,
        "config": cfg.to_dict(),
        "template_digest": make_template(cfg.subvolume).digest(),
        "halo": cfg.halo(),
        "versions": {pkg: version(pkg) for pkg in ("pydvc", "zarr-vectors", "zarr", "numpy")},
        **extra,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(path)


def load_plan(path: str | Path) -> tuple[list[Tile], list[list[int]]]:
    doc = json.loads(Path(path).read_text())
    tiles = [
        Tile(d["id"], tuple(tuple(c) for c in d["cells"]), d["n_points"], _unbox(d["points_box"]),
             _unbox(d["ref_box"]), _unbox(d["def_box"]), d["cost"])
        for d in doc["tiles"]
    ]
    return tiles, doc["assignment"]


def plan_document(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


