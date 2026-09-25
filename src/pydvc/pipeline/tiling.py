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

from dataclasses import dataclass
from pathlib import Path

from pydvc._todo import todo
from pydvc.config import RunConfig
from pydvc.geometry.box import Box
from pydvc.io.pointcloud import CellCoord, PointCloud


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


def plan_tiles(points: PointCloud, cfg: RunConfig, *, seed_field_path: str | Path | None = None) -> list[Tile]:
    """Metadata only: group occupied chunks into tiles and compute brick boxes and costs."""
    raise todo("M3", "plan_tiles")


def check_memory(tiles: list[Tile], cfg: RunConfig, *, device_total_bytes: int, dtype_bytes: int) -> MemoryReport:
    raise todo("M3", "check_memory")


def assign_lpt(tiles: list[Tile], n_bins: int) -> list[list[int]]:
    """Tile ids per bin (node), balancing summed cost."""
    raise todo("M4", "assign_lpt")


def save_plan(path: str | Path, tiles: list[Tile], assignment: list[list[int]], cfg: RunConfig) -> None:
    """``plan.json``: tiles, node assignment, template hash, package versions (zarr-vectors commit included)."""
    raise todo("M3", "save_plan")


def load_plan(path: str | Path) -> tuple[list[Tile], list[list[int]]]:
    raise todo("M3", "load_plan")
