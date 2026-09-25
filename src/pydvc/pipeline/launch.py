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

from dataclasses import dataclass

from pydvc._todo import todo
from pydvc.config import RunConfig
from pydvc.pipeline.worker import WorkerStats


@dataclass(frozen=True)
class NodeInfo:
    node_rank: int      # SLURM_NODEID | GROUP_RANK (torchrun) | derived from OMPI_COMM_WORLD_RANK
    n_nodes: int        # SLURM_JOB_NUM_NODES | ...
    local_devices: tuple[int, ...]


def node_info() -> NodeInfo:
    """Discover node rank and local GPUs from SLURM, torchrun or Open MPI variables; single node if none are set."""
    raise todo("M4", "node_info")


def launch_local(cfg: RunConfig, *, tile_ids: list[int] | None = None, devices: tuple[int, ...] | None = None) -> list[WorkerStats]:
    """Run ``tile_ids`` (default: this node's share of the plan) on ``devices`` (default: all visible)."""
    raise todo("M4", "launch_local")
