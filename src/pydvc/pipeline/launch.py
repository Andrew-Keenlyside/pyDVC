"""Launching workers: one process per GPU.

Single node (the MVP target, 8x H100)
    ``launch_local`` spawns one process per visible GPU (the ``spawn`` start
    method, because CUDA is not fork-safe; each child probes its own device,
    as zarr-vectors advises). The processes share a ``multiprocessing`` queue
    of tile ids, so scheduling is dynamic. GPUs need no inter-GPU
    communication (no NCCL): tiles are independent once seeds exist.

Multiple nodes (M5)
    Run one task per node (``srun --ntasks-per-node=1 --gpus-per-node=8``).
    Each node takes its LPT share from ``plan.json`` using ``SLURM_NODEID``
    (or the torchrun or Open MPI equivalents) and runs ``launch_local`` over
    that share. Nodes never talk to each other. The shared file system
    (results written to disjoint cells) is the only thing they have in common.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydvc.config import RunConfig
from pydvc.pipeline.worker import WorkerStats


@dataclass(frozen=True)
class NodeInfo:
    node_rank: int      # SLURM_NODEID | GROUP_RANK (torchrun) | derived from OMPI_COMM_WORLD_RANK
    n_nodes: int        # SLURM_JOB_NUM_NODES | ...
    local_devices: tuple[int, ...]


def _local_gpus() -> tuple[int, ...]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [v for v in visible.split(",") if v.strip() and v.strip() != "-1"]
        return tuple(range(len(ids)))                    # device ordinals are relative to the visible set
    try:
        import cupy

        return tuple(range(cupy.cuda.runtime.getDeviceCount()))
    except Exception:
        return ()


def node_info(env: dict[str, str] | None = None) -> NodeInfo:
    """Discover node rank and local GPUs from SLURM, torchrun or Open MPI variables; single node if none are set."""
    e = os.environ if env is None else env
    if "SLURM_NODEID" in e:
        rank, n = int(e["SLURM_NODEID"]), int(e.get("SLURM_JOB_NUM_NODES", e.get("SLURM_NNODES", "1")))
    elif "GROUP_RANK" in e:                                  # torchrun, one agent per node
        rank = int(e["GROUP_RANK"])
        n = int(e.get("GROUP_WORLD_SIZE", int(e.get("WORLD_SIZE", "1")) // max(int(e.get("LOCAL_WORLD_SIZE", "1")), 1)))
    elif "OMPI_COMM_WORLD_RANK" in e:
        local = max(int(e.get("OMPI_COMM_WORLD_LOCAL_SIZE", "1")), 1)
        rank = int(e["OMPI_COMM_WORLD_RANK"]) // local
        n = max(int(e.get("OMPI_COMM_WORLD_SIZE", "1")) // local, 1)
    else:
        rank, n = 0, 1
    if not 0 <= rank < n:
        raise ValueError(f"node rank {rank} outside 0..{n - 1}")
    return NodeInfo(node_rank=rank, n_nodes=n, local_devices=_local_gpus() if env is None else ())


class _SharedSource:
    """A worker's view of the node's shared tile queue (a manager queue of tile ids)."""

    def __init__(self, tiles: dict[int, Any], ids: Any, attempts: Any, failed: Any, lock: Any, retries: int = 1) -> None:
        self.tiles, self.ids, self.attempts, self.failed, self.lock, self.retries = tiles, ids, attempts, failed, lock, retries

    def next(self) -> Any:
        try:
            return self.tiles[self.ids.get_nowait()]
        except queue.Empty:
            return None

    def done(self, tile: Any, ok: bool) -> None:
        if ok:
            return
        with self.lock:
            n = self.attempts.get(tile.id, 0) + 1
            self.attempts[tile.id] = n
            if n <= self.retries:
                self.ids.put(tile.id)
            else:
                self.failed.append(tile.id)


def _child(cfg_dict: dict[str, Any], slot: int, backend: str, threads: int | None, plan_path: str,
           ids: Any, attempts: Any, failed: Any, lock: Any, results: Any) -> None:
    """Worker process body: one device (GPU ordinal, or a CPU slot with ``threads`` numba threads)."""
    if threads:
        os.environ["NUMBA_NUM_THREADS"] = str(threads)
    try:
        pid_dir = Path(cfg_dict["workdir"]) / "workers"
        pid_dir.mkdir(parents=True, exist_ok=True)
        (pid_dir / f"slot{slot}.pid").write_text(str(os.getpid()))
        from pydvc.pipeline.tiling import load_plan, plan_document
        from pydvc.pipeline.worker import TileWorker

        cfg = RunConfig.from_dict(cfg_dict)
        tiles, _ = load_plan(plan_path)
        doc = plan_document(plan_path)
        seeds = Path(cfg.workdir) / "seeds.npz"
        worker = TileWorker(cfg, slot, backend=backend, points_store=doc["points_store"],
                            seed_field_path=seeds if seeds.exists() else None)
        stats = worker.run(_SharedSource({t.id: t for t in tiles}, ids, attempts, failed, lock))
    except Exception:
        stats = WorkerStats(device=slot, errors=[traceback.format_exc()])
    results.put(stats)


def node_share(plan_path: str | Path, info: NodeInfo) -> list[int]:
    """This node's tile ids: its LPT share from ``plan.json``, re-balanced if the plan was made for another node count."""
    from pydvc.pipeline.tiling import assign_lpt, load_plan

    tiles, assignment = load_plan(plan_path)
    if len(assignment) != info.n_nodes:
        assignment = assign_lpt(tiles, info.n_nodes)
    return list(assignment[info.node_rank])


def launch_local(
    cfg: RunConfig,
    *,
    tile_ids: list[int] | None = None,
    devices: tuple[int, ...] | None = None,
    backend: str | None = None,
    cpu_workers: int = 1,
) -> list[WorkerStats]:
    """Run ``tile_ids`` (default: this node's share of the plan) on ``devices`` (default: all visible).

    GPU backends get one process per device. CPU backends (``cpu``, ``numpy``)
    get ``cpu_workers`` processes that split the cores. With one slot the
    worker runs in this process. A worker that dies (``kill -9``, OOM) takes
    only its in-flight tile with it: the others drain the queue, and the
    missing tile is solved when the job is resubmitted.
    """
    from pydvc.pipeline.tiling import load_plan
    from pydvc.pipeline.worker import QueueSource, TileWorker
    from pydvc.solver.engines import default_backend, on_device

    backend = backend or default_backend()
    plan_path = Path(cfg.workdir) / "plan.json"
    tiles, _ = load_plan(plan_path)
    info = node_info()
    ids = node_share(plan_path, info) if tile_ids is None else list(tile_ids)
    if on_device(backend):
        slots = tuple(devices if devices is not None else (cfg.cluster.devices or info.local_devices or (0,)))
        threads = None
    else:
        slots = tuple(range(max(cpu_workers, 1)))
        threads = max(1, (os.cpu_count() or 1) // len(slots)) if len(slots) > 1 else None
    by_id = {t.id: t for t in tiles}
    if len(slots) == 1:
        from pydvc.pipeline.coordinator import _seed_field, points_store

        worker = TileWorker(cfg, slots[0], backend=backend, points_store=points_store(cfg), seed_field_path=_seed_field(cfg))
        source = QueueSource([by_id[i] for i in ids])
        stats = worker.run(source)
        stats.errors.extend(f"tile {i} failed twice" for i in source.failed)
        return [stats]

    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        q, attempts, failed, lock, results = manager.Queue(), manager.dict(), manager.list(), manager.Lock(), manager.Queue()
        for i in ids:
            q.put(i)
        procs = [
            ctx.Process(target=_child, name=f"pydvc-worker-{slot}",
                        args=(cfg.to_dict(), slot, backend, threads, str(plan_path), q, attempts, failed, lock, results))
            for slot in slots
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        out: list[WorkerStats] = []
        while True:
            try:
                out.append(results.get_nowait())
            except queue.Empty:
                break
        for p, slot in zip(procs, slots):
            if p.exitcode != 0:
                out.append(WorkerStats(device=slot, errors=[f"worker process on slot {slot} exited with code {p.exitcode}"]))
        for tid in list(failed):
            out.append(WorkerStats(device=-1, errors=[f"tile {tid} failed twice"]))
    return out
