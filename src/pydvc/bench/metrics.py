"""Accuracy metrics against ground truth and against CCPi output.

Results can be read from a zarr-vectors results store, the in-memory runner's
``results.npz`` (:mod:`pydvc.pipeline.inmemory`) or a CCPi ``.disp``. Points are matched on ``point_id``. Errors are over
points that are GOOD in the results (and, against CCPi, GOOD in both).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pydvc.status import PointStatus


@dataclass
class Accuracy:
    n_good: int
    frac_good: float
    bias: tuple[float, float, float]       # mean error (voxels) per axis
    rmse: tuple[float, float, float]       # per axis, GOOD points only
    p99_abs: float                         # 99th percentile |error|, voxels
    status_counts: dict[int, int]
    n_points: int = 0
    median_abs: float = float("nan")       # median |error| (Euclidean), voxels
    p95_abs: float = float("nan")
    status_agreement: float | None = None  # against another code: fraction with the same (CCPi) status

    def summary(self) -> str:
        names = {int(s): s.name for s in PointStatus}
        counts = ", ".join(f"{names.get(k, k)} {v}" for k, v in sorted(self.status_counts.items(), reverse=True))
        lines = [
            f"points {self.n_points}, GOOD {self.n_good} ({100 * self.frac_good:.2f} %)  [{counts}]",
            "rmse (x, y, z)  " + "  ".join(f"{v:.4f}" for v in self.rmse),
            "bias (x, y, z)  " + "  ".join(f"{v:+.4f}" for v in self.bias),
            f"|error| median {self.median_abs:.4f}  p95 {self.p95_abs:.4f}  p99 {self.p99_abs:.4f}",
        ]
        if self.status_agreement is not None:
            lines.append(f"status agreement {100 * self.status_agreement:.2f} %")
        return "\n".join(lines)


def load_results(path: str | Path) -> dict[str, np.ndarray]:
    """``point_id``, ``xyz``, ``status``, ``objmin``, ``displacement`` from ``results.npz`` or a ``.disp``."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as r:
            return {k: r[k] for k in ("point_id", "xyz", "status", "objmin", "displacement")}
    if path.suffix == ".disp":
        from pydvc.io.ccpi import read_disp

        d = read_disp(path)
        return {
            "point_id": d["n"],
            "xyz": np.stack([d["x"], d["y"], d["z"]], axis=1),
            "status": d["status"],
            "objmin": d["objmin"],
            "displacement": np.stack([d["u"], d["v"], d["w"]], axis=1),
        }
    from pydvc.io.results import ResultStore

    r = ResultStore(path).read_all()
    return {"point_id": r["point_id"], "xyz": r["xyz"].astype(np.float64), "status": r["status"],
            "objmin": r["objmin"], "displacement": r["displacement"].astype(np.float64)}


def _match(a_ids: np.ndarray, b_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices into ``a`` and ``b`` of the ids present in both."""
    common, ia, ib = np.intersect1d(a_ids, b_ids, assume_unique=False, return_indices=True)
    if len(common) == 0:
        raise ValueError("no point_id in common")
    return ia, ib


def compare_arrays(
    a: Any,
    b: Any,
    status: Any = None,
    *,
    ref_status: Any = None,
) -> Accuracy:
    """Error of displacements ``a`` (N, 3) against ``b`` (N, 3).

    ``status`` is ``a``'s PointStatus (default: all GOOD). With ``ref_status``
    (another code's statuses, CCPi codes) errors are taken over points GOOD in
    both, and the status agreement is reported.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    n = len(a)
    status = np.zeros(n, dtype=np.int64) if status is None else np.asarray(status, dtype=np.int64)
    good = status == PointStatus.GOOD
    agreement = None
    if ref_status is not None:
        ref_status = np.asarray(ref_status, dtype=np.int64)
        ours = np.array([PointStatus(int(s)).to_ccpi() for s in status])
        agreement = float((ours == ref_status).mean()) if n else float("nan")
        good_both = good & (ref_status == PointStatus.GOOD)
    else:
        good_both = good
    err = a[good_both] - b[good_both]
    mag = np.linalg.norm(err, axis=1)
    nan3 = (float("nan"),) * 3

    def pct(q: float) -> float:
        return float(np.percentile(mag, q)) if len(mag) else float("nan")

    values, counts = np.unique(status, return_counts=True)
    return Accuracy(
        n_good=int(good.sum()),
        frac_good=float(good.mean()) if n else float("nan"),
        bias=tuple(float(v) for v in err.mean(axis=0)) if len(err) else nan3,
        rmse=tuple(float(v) for v in np.sqrt((err**2).mean(axis=0))) if len(err) else nan3,
        p99_abs=pct(99),
        status_counts={int(v): int(c) for v, c in zip(values, counts)},
        n_points=n,
        median_abs=pct(50),
        p95_abs=pct(95),
        status_agreement=agreement,
    )


def against_truth(results_path: str | Path, truth_path: str | Path) -> Accuracy:
    res = load_results(results_path)
    with np.load(truth_path) as t:
        ids, truth = t["point_id"], t["displacement"]
    ia, ib = _match(res["point_id"], ids)
    if len(ia) != len(ids):
        raise ValueError(f"results hold {len(ia)} of the {len(ids)} truth points")
    return compare_arrays(res["displacement"][ia], truth[ib], res["status"][ia])


def against_disp(results_path: str | Path, disp_path: str | Path) -> Accuracy:
    """Point-by-point agreement with a CCPi ``.disp`` (matched on ``point_id``), including status agreement."""
    res = load_results(results_path)
    ref = load_results(disp_path)
    ia, ib = _match(res["point_id"], ref["point_id"])
    return compare_arrays(res["displacement"][ia], ref["displacement"][ib], res["status"][ia], ref_status=ref["status"][ib])
