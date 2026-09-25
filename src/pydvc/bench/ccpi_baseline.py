"""Run the CCPi ``dvc`` executable (conda package ``ccpi-dvc``) on a pyDVC case.

Writes ``dvc_in`` + ``.roi`` equivalent to the pyDVC config
(:func:`pydvc.io.ccpi.write_dvc_input`), converts volumes to ``.raw`` if
needed, runs ``dvc`` with ``OMP_NUM_THREADS`` set, and parses points per
second from the ``.stat`` file. For clouds too large for CCPi's O(N^2)
neighbour sort, it runs a random sample of points and extrapolates, recording
that it did so.

It also sweeps ``OMP_NUM_THREADS`` (1, 2, 4, ..., cores) to record CCPi's
intra-point scaling, and runs ``P`` concurrent processes on disjoint point
subsets to get the best throughput achievable from one CPU node without
code changes. That is the fair "CCPi as shipped" baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydvc._todo import todo
from pydvc.config import RunConfig


@dataclass
class BaselineResult:
    points: int
    seconds: float
    points_per_second: float
    omp_threads: int
    processes: int
    sampled: bool                 # True if extrapolated from a subset
    disp_path: Path


def run_ccpi(cfg: RunConfig, *, workdir: str | Path, omp_threads: int, processes: int = 1, max_points: int | None = None) -> BaselineResult:
    raise todo("M0", "run_ccpi")


def thread_sweep(cfg: RunConfig, *, workdir: str | Path, max_points: int = 200) -> list[BaselineResult]:
    raise todo("M0", "thread_sweep")
