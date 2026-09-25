"""Throughput and utilisation measurements.

Records points/s per stage (read, seed, solve, write), GPU time split by
NVTX range, achieved FLOP/s and L2 hit rate for the fused kernel (Nsight
Compute metrics, when available), and compute-stream idle time spent waiting
on I/O. These are the numbers that replace the model in docs/PERFORMANCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydvc._todo import todo
from pydvc.config import RunConfig


@dataclass
class Throughput:
    backend: str                      # "ccpi" | "numpy" | "cupy" | "fused"
    hardware: str
    points: int
    seconds_total: float
    seconds_by_stage: dict[str, float] = field(default_factory=dict)
    points_per_second: float = 0.0
    achieved_tflops: float | None = None
    io_wait_fraction: float | None = None


def kernel_microbench(*, n_samples: int, dof: int, interpolation: str, objective: str, batch: int) -> Throughput:
    """Fused GN step on a synthetic brick: points/s and FLOP/s with I/O excluded."""
    raise todo("M2", "kernel_microbench")


def end_to_end(cfg: RunConfig, *, backend: str) -> Throughput:
    raise todo("M3", "end_to_end")
