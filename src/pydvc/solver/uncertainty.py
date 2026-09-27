"""Per-point displacement uncertainty from repeat solves with other template seeds.

On real data the largest random error is which sample points each subvolume uses
(docs/benchmarks/2026-09-26-error-floor-case-A.md: ~0.03 voxel per axis at 8 000 samples,
falling as 1/sqrt(n)). With ``uncertainty_seeds = k`` in the run config, every GOOD point is
solved k more times, each with a template drawn with another seed, and the per-axis standard
deviation over the k + 1 estimates is stored as ``displacement_sd``. Each repeat starts from the
same seed displacement as the main solve, not from its answer: a solve that starts at an answer
stops within the convergence tolerance (0.01 voxel) of it and would understate the spread. With
k = 2 the cost is about twice the main solve's; the estimate of each sd has about 2 degrees of
freedom, so it is rough per point and good in aggregate.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from pydvc.config import SearchSpec, SubvolumeSpec


def seed_spread(ref: Any, deformed: Any, centres: np.ndarray, seeds: np.ndarray, displacement: np.ndarray,
                status: np.ndarray, subvolume: SubvolumeSpec, search: SearchSpec, engine: Any, k: int) -> np.ndarray:
    """(N, 3) per-axis sd of the displacement over the main solve and ``k`` repeats; NaN where < 2 GOOD estimates."""
    from pydvc.geometry.templates import make_template
    from pydvc.solver.gauss_newton import solve_batch

    n = len(centres)
    sd = np.full((n, 3), np.nan)
    good = np.flatnonzero(np.asarray(status) == 0)
    if k < 1 or good.size == 0:
        return sd
    est = [np.asarray(displacement, dtype=np.float64)[good]]
    ok = [np.ones(good.size, dtype=bool)]
    host = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)                   # noqa: E731
    for s in range(1, k + 1):
        t = make_template(dataclasses.replace(subvolume, seed=subvolume.seed + s))
        r = solve_batch(ref, deformed, np.asarray(centres)[good], np.asarray(seeds)[good], t, search, engine=engine)
        est.append(host(r.params)[:, :3].astype(np.float64))
        ok.append(host(r.status) == 0)
    e, m = np.stack(est), np.stack(ok)                                               # (k+1, G, 3), (k+1, G)
    cnt = m.sum(axis=0)
    mean = np.where(m[..., None], e, 0.0).sum(axis=0) / np.maximum(cnt, 1)[:, None]
    var = np.where(m[..., None], (e - mean) ** 2, 0.0).sum(axis=0) / np.maximum(cnt - 1, 1)[:, None]
    sd[good] = np.where((cnt >= 2)[:, None], np.sqrt(var), np.nan)
    return sd
