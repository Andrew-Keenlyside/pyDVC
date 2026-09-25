"""Per-device worker: one process, one device, a stream of tiles.

Loop (a prefetch thread, the compute loop, and a writer thread)::

    prefetch:  bricks + points for the next ``prefetch_depth`` tiles  (host decode, pinned copy up)
    compute:   seeds = seed field lookup                              (rigid / interpolated coarse field)
               for batch in iter_batches(morton_order(points)):
                   solve_batch(ref_brick, def_brick, ...)
    writer:    ResultStore.write_tile(...)                            (overlaps the next tile's compute)
    source.done(tile)

Target: while there is work in the queue, the compute loop never waits on
I/O; ``WorkerStats.seconds_io_wait`` measures how far from that a run is.
Idempotent: a tile whose cells are all present in the results store is
skipped, which is how an interrupted or pre-empted job resumes. A tile that
raises is reported and requeued once, then left for the coordinator to report.

The same worker drives every backend: ``fused``/``cupy`` put bricks on the
GPU; ``cpu`` (numba) and ``numpy`` keep them on the host.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pydvc.config import RunConfig
from pydvc.pipeline.tiling import Tile


class TileSource(Protocol):
    def next(self) -> Tile | None: ...
    def done(self, tile: Tile, ok: bool) -> None: ...


class QueueSource:
    """A node-local tile queue shared by the threads (or, via a manager, processes) of one node.

    A failed tile is requeued once; after a second failure it is recorded in ``failed``.
    """

    def __init__(self, tiles: list[Tile], *, retries: int = 1) -> None:
        self._pending = list(tiles)
        self._attempts: dict[int, int] = {}
        self._retries = retries
        self._lock = threading.Lock()
        self.failed: list[int] = []

    def next(self) -> Tile | None:
        with self._lock:
            return self._pending.pop(0) if self._pending else None

    def done(self, tile: Tile, ok: bool) -> None:
        if ok:
            return
        with self._lock:
            n = self._attempts[tile.id] = self._attempts.get(tile.id, 0) + 1
            if n <= self._retries:
                self._pending.append(tile)
            else:
                self.failed.append(tile.id)


@dataclass
class WorkerStats:
    device: int
    tiles: int = 0
    points: int = 0
    seconds_compute: float = 0.0
    seconds_io_wait: float = 0.0       # compute loop idle waiting for bricks: should be ~0
    bytes_read: int = 0
    status_counts: dict[int, int] = field(default_factory=dict)
    tiles_skipped: int = 0
    seconds_write_wait: float = 0.0    # compute loop blocked on a full write queue
    errors: list[str] = field(default_factory=list)


@dataclass
class _Loaded:
    tile: Tile
    ref: Any
    deformed: Any
    points: Any
    bytes_read: int
    error: str | None = None


_END = object()


class TileWorker:
    def __init__(
        self,
        cfg: RunConfig,
        device: int = 0,
        *,
        backend: str | None = None,
        points_store: str | Path | None = None,
        seed_field_path: str | Path | None = None,
    ) -> None:
        from pydvc.geometry.templates import make_template
        from pydvc.io.pointcloud import PointCloud
        from pydvc.io.results import ResultStore
        from pydvc.io.volume import open_volume
        from pydvc.pipeline.tiling import load_seed_field
        from pydvc.solver.engines import default_backend, make_engine, on_device

        self.cfg = cfg
        self.device = device
        self.backend = backend or default_backend()
        self.on_device = on_device(self.backend)
        if self.on_device:
            import cupy

            cupy.cuda.Device(device).use()
        self.ref_vol = open_volume(cfg.volumes, "reference")
        self.def_vol = open_volume(cfg.volumes, "deformed")
        self.points = PointCloud(points_store or cfg.points)
        self.results = ResultStore(cfg.output, mode="r+")
        self.template = make_template(cfg.subvolume)
        self.engine = make_engine(self.backend)
        self.seed_field = load_seed_field(seed_field_path)

    # -------------------------------------------------------------- stages

    def _load(self, tile: Tile) -> _Loaded:
        dev = "cuda" if self.on_device else "cpu"
        try:
            ref = self.ref_vol.read_brick(tile.ref_box, device=dev)
            deformed = self.def_vol.read_brick(tile.def_box, device=dev)
            pts = self.points.read_tile(list(tile.cells), device="cpu")
            nbytes = int(ref.data.nbytes + deformed.data.nbytes)
            return _Loaded(tile, ref, deformed, pts, nbytes)
        except Exception:
            return _Loaded(tile, None, None, None, 0, traceback.format_exc())

    def _batch_size(self, n_points: int) -> int:
        from pydvc.pipeline.batching import batch_size

        if self.cfg.cluster.batch_points:
            return self.cfg.cluster.batch_points
        if self.on_device:
            free, _ = self.engine.xp.cuda.runtime.memGetInfo()
            return batch_size(free, self.template.n_samples, self.cfg.search.dof, fraction=0.5 * self.cfg.cluster.gpu_memory_fraction)
        return 8192

    def solve_tile(self, loaded: _Loaded) -> dict[str, np.ndarray]:
        """Solve every point of a loaded tile; results in the tile's row order, on the host."""
        from pydvc.pipeline.batching import iter_batches, morton_order
        from pydvc.pipeline.tiling import lookup_seeds
        from pydvc.solver.gauss_newton import solve_batch

        pts = loaded.points
        xyz = np.asarray(pts.xyz, dtype=np.float64)
        n, dof = len(xyz), self.cfg.search.dof
        seeds = lookup_seeds(np.asarray(pts.point_id), self.seed_field, self.cfg.search.rigid_trans)
        out = {
            "status": np.zeros(n, dtype=np.int8),
            "objmin": np.zeros(n, dtype=np.float32),
            "displacement": np.zeros((n, 3), dtype=np.float32),
            "params": np.zeros((n, dof), dtype=np.float32),
            "n_iter": np.zeros(n, dtype=np.uint8),
            "seed": seeds.astype(np.float32),
        }
        for batch in iter_batches(morton_order(xyz), self._batch_size(n)):
            res = solve_batch(loaded.ref, loaded.deformed, xyz[batch], seeds[batch], self.template, self.cfg.search,
                              engine=self.engine)
            params = _host(res.params)
            out["params"][batch] = params
            out["displacement"][batch] = params[:, :3]
            out["status"][batch] = _host(res.status)
            out["objmin"][batch] = _host(res.objmin)
            out["n_iter"][batch] = _host(res.n_iter)
        return out

    # -------------------------------------------------------------- loop

    def run(self, source: TileSource) -> WorkerStats:
        stats = WorkerStats(device=self.device)
        written = self.results.written_cells()
        loads: queue.Queue = queue.Queue(maxsize=max(self.cfg.cluster.prefetch_depth, 1))
        writes: queue.Queue = queue.Queue(maxsize=2)
        write_errors: list[str] = []

        def prefetch() -> None:
            while True:
                tile = source.next()
                if tile is None:
                    loads.put(_END)
                    return
                if written.issuperset(tile.cells):
                    stats.tiles_skipped += 1
                    source.done(tile, True)
                    continue
                loads.put(self._load(tile))

        def writer() -> None:
            while True:
                item = writes.get()
                if item is _END:
                    return
                tile, pts, res = item
                try:
                    self.results.write_tile(pts, res)
                    source.done(tile, True)
                except Exception:
                    write_errors.append(traceback.format_exc())
                    source.done(tile, False)

        def compute(loaded: _Loaded) -> tuple[Tile, Any, dict[str, np.ndarray]] | None:
            if loaded.error:
                stats.errors.append(f"tile {loaded.tile.id}: read failed\n{loaded.error}")
                source.done(loaded.tile, False)
                return None
            t1 = time.perf_counter()
            try:
                res = self.solve_tile(loaded)
            except Exception:
                stats.errors.append(f"tile {loaded.tile.id}: solve failed\n{traceback.format_exc()}")
                source.done(loaded.tile, False)
                return None
            stats.seconds_compute += time.perf_counter() - t1
            stats.tiles += 1
            stats.points += len(res["status"])
            stats.bytes_read += loaded.bytes_read
            values, counts = np.unique(res["status"], return_counts=True)
            for v, c in zip(values.tolist(), counts.tolist()):
                stats.status_counts[v] = stats.status_counts.get(v, 0) + c
            return loaded.tile, loaded.points, res

        t_pre = threading.Thread(target=prefetch, name="pydvc-prefetch", daemon=True)
        t_wri = threading.Thread(target=writer, name="pydvc-writer", daemon=True)
        t_pre.start()
        t_wri.start()
        try:
            while True:
                t0 = time.perf_counter()
                loaded = loads.get()
                if stats.tiles or stats.errors:          # the first read is start-up, not a stall
                    stats.seconds_io_wait += time.perf_counter() - t0
                if loaded is _END:
                    break
                item = compute(loaded)
                if item is not None:
                    t2 = time.perf_counter()
                    writes.put(item)
                    stats.seconds_write_wait += time.perf_counter() - t2
        finally:
            writes.put(_END)
            t_wri.join()
            t_pre.join(timeout=5)
        # tiles requeued after the prefetcher drained the source: one synchronous pass
        while (tile := source.next()) is not None:
            item = compute(self._load(tile))
            if item is None:
                continue
            try:
                self.results.write_tile(item[1], item[2])
                source.done(tile, True)
            except Exception:
                write_errors.append(traceback.format_exc())
                source.done(tile, False)
        stats.errors.extend(f"write failed\n{e}" for e in write_errors)
        return stats


def _host(a: Any) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)
