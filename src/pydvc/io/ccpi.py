"""CCPi DVC / iDVC file formats.

Used to (a) run pyDVC as a drop-in for the ``dvc`` executable that iDVC
launches (``pydvc ccpi dvc_config.txt``), (b) run the CCPi baseline on pyDVC's
test cases (M0), and (c) compare results point by point.

``dvc_in``
    ``key<ws>value  ### comment`` lines; ``#`` lines ignored. Keys as in iDVC's
    ``dvc_config_template.txt`` and CCPi ``InputRead``.
``.disp``
    Tab-separated, header ``n x y z status objmin u v w``. Current CCPi writes
    displacements only; older files also carry ``phi the psi`` and strain
    columns, and :func:`read_disp` accepts both. Rows are in point-cloud order.
``.stat``
    Input echo, run start/finish, points per second, and counts per status.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pydvc._todo import todo
from pydvc.config import RunConfig


@dataclass
class RunSummary:
    n_points: int
    seconds: float
    counts: dict[int, int]       # PointStatus (CCPi codes) -> count


def read_dvc_input(path: str | Path) -> dict[str, str]:
    raise todo("M4", "read_dvc_input")


def run_config_from_dvc_input(params: dict[str, str]) -> RunConfig:
    raise todo("M4", "run_config_from_dvc_input")


def write_dvc_input(cfg: RunConfig, path: str | Path, *, roi_path: str | Path, output_base: str | Path) -> None:
    """Write a CCPi input file equivalent to ``cfg``, to run the CPU baseline on the same case (M0)."""
    raise todo("M0", "write_dvc_input")


def write_roi(path: str | Path, point_id: np.ndarray, xyz: np.ndarray) -> None:
    raise todo("M0", "write_roi")


def read_disp(path: str | Path) -> np.ndarray:
    """Structured array with fields ``n, x, y, z, status, objmin, u, v, w`` (+ optional extra columns)."""
    raise todo("M0", "read_disp")


def write_disp(
    path: str | Path,
    point_id: np.ndarray,
    xyz: np.ndarray,
    status: np.ndarray,
    objmin: np.ndarray,
    displacement: np.ndarray,
) -> None:
    raise todo("M4", "write_disp")


def write_stat(path: str | Path, cfg: RunConfig, summary: RunSummary) -> None:
    raise todo("M4", "write_stat")


def read_stat_throughput(path: str | Path) -> float:
    """Points per second reported in a CCPi ``.stat`` file (the M0 baseline number)."""
    raise todo("M0", "read_stat_throughput")
