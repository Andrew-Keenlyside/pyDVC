"""Per-GPU worker: one process, one device, a stream of tiles.

Loop (three CUDA streams: copy, compute, and a host thread for writes)::

    tile_k   = source.next()
    prefetch bricks for tile_k+1 .. tile_k+prefetch_depth  (copy stream, pinned buffers)
    points   = PointCloud.read_tile(tile_k.cells, device="cuda")  (zarr-vectors read_cells)
    seeds    = seed field lookup                     (rigid / interpolated coarse field)
    for batch in iter_batches(morton_order(points)):
        solve_batch(ref_brick, def_brick, ...)       (compute stream)
    ResultStore.write_tile(...)                      (host thread; overlaps tile_k+1 compute)
    source.done(tile_k)

Target: while there is work in the queue, the compute stream never waits on
I/O. Idempotent: a tile whose cells are all present in the results store is
skipped, which is how an interrupted or pre-empted job resumes. A tile that
raises is reported and requeued once, then left for the coordinator to report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pydvc._todo import todo
from pydvc.config import RunConfig
from pydvc.pipeline.tiling import Tile


class TileSource(Protocol):
    def next(self) -> Tile | None: ...
    def done(self, tile: Tile, ok: bool) -> None: ...


@dataclass
class WorkerStats:
    device: int
    tiles: int = 0
    points: int = 0
    seconds_compute: float = 0.0
    seconds_io_wait: float = 0.0       # compute stream idle waiting for bricks: should be ~0
    bytes_read: int = 0
    status_counts: dict[int, int] = field(default_factory=dict)


class TileWorker:
    def __init__(self, cfg: RunConfig, device: int) -> None:
        raise todo("M3", "TileWorker")

    def run(self, source: TileSource) -> WorkerStats:
        raise todo("M3", "TileWorker.run")
