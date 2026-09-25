"""Accuracy metrics against ground truth and against CCPi output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydvc._todo import todo


@dataclass
class Accuracy:
    n_good: int
    frac_good: float
    bias: tuple[float, float, float]       # mean error (voxels) per axis
    rmse: tuple[float, float, float]       # per axis, GOOD points only
    p99_abs: float                         # 99th percentile |error|, voxels
    status_counts: dict[int, int]


def against_truth(results_path: str | Path, truth_path: str | Path) -> Accuracy:
    raise todo("M1", "against_truth")


def against_disp(results_path: str | Path, disp_path: str | Path) -> Accuracy:
    """Point-by-point agreement with a CCPi ``.disp`` (matched on ``point_id``), including status agreement."""
    raise todo("M1", "against_disp")


def compare_arrays(a: Any, b: Any) -> Accuracy:
    raise todo("M1", "compare_arrays")
