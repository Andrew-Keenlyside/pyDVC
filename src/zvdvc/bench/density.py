"""Wall time against point-cloud density on the iDVC case A pair, and a cost model fitted to it.

Grids of decreasing spacing fill a fixed block of the sample
(:data:`zvdvc.bench.case_a.GRID_BOX`) and are solved in memory with CCPi's
settings and wavefront seeding. For each backend the model

    t(N) = t0 + tau * S(N) + c * N,     S(N) ~ k N^(1/3) wavefront shells

is fitted in relative error with non-negative coefficients: ``t0`` is fixed
cost per run, ``tau`` per shell (launches and host syncs, which dominate
small grids on a GPU) and ``c`` per point. The GPU's speed-up over the CPU
tends to ``c_cpu / c_gpu`` for dense grids.

Command line (the CPU arm takes tens of minutes at the finest spacings)::

    python -m zvdvc.bench.density --backends fused cpu --json runs/density.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

SPACINGS = (128.0, 96.0, 64.0, 48.0, 40.0, 32.0, 24.0, 20.0, 16.0)


def sweep(cfg: Any, spacings: tuple[float, ...], backends: list[str], progress=print) -> list[dict[str, Any]]:
    from zvdvc.bench.case_a import grid_points
    from zvdvc.pipeline.inmemory import solve_in_memory
    from zvdvc.solver import seeding

    pid, xyz = grid_points(max(spacings))
    for b in backends:                                            # compile kernels outside the timings
        solve_in_memory(cfg, pid, xyz, backend=b)
    rows = []
    for s in spacings:
        pid, xyz = grid_points(s)
        shells = seeding.wavefront_shells(xyz, tuple(xyz[0]), seeding.median_spacing(xyz))
        row: dict[str, Any] = {"spacing": s, "n_points": len(xyz), "n_shells": len(shells)}
        for b in backends:
            t0 = time.perf_counter()
            r = solve_in_memory(cfg, pid, xyz, backend=b)
            row[b] = {"seconds": time.perf_counter() - t0, "iters": int(r.n_iter.sum()),
                      "good": float((r.status == 0).mean())}
        progress(json.dumps(row))
        rows.append(row)
    return rows


def fit(rows: list[dict[str, Any]], backend: str) -> dict[str, float]:
    """``t0``, ``tau``, ``c`` (seconds) and the largest relative misfit for one backend."""
    from scipy.optimize import nnls

    rows = [r for r in rows if backend in r]
    t = np.array([r[backend]["seconds"] for r in rows])
    A = np.stack([np.ones(len(rows)), [r["n_shells"] for r in rows], [r["n_points"] for r in rows]], 1).astype(float)
    coef, _ = nnls(A / t[:, None], np.ones_like(t))
    return {"t0": float(coef[0]), "tau": float(coef[1]), "c": float(coef[2]),
            "max_rel_err": float(np.max(np.abs(A @ coef - t) / t))}


def model(rows: list[dict[str, Any]], backends: list[str]) -> dict[str, Any]:
    n = np.array([r["n_points"] for r in rows], float)
    s = np.array([r["n_shells"] for r in rows], float)
    out: dict[str, Any] = {"k": float(np.exp(np.mean(np.log(s) - np.log(n) / 3)))}
    for b in backends:
        out[b] = fit(rows, b)
    if "fused" in out and "cpu" in out and out["fused"]["c"] > 0:
        out["asymptotic_speedup"] = out["cpu"]["c"] / out["fused"]["c"]
    return out


def main(argv: list[str] | None = None) -> None:
    from zvdvc.bench import case_a as ca
    from zvdvc.bench.smoke import environment

    p = argparse.ArgumentParser(prog="python -m zvdvc.bench.density", description=__doc__.split("\n\n")[0])
    p.add_argument("--backends", nargs="+", default=["fused"])
    p.add_argument("--spacings", type=float, nargs="+", default=list(SPACINGS))
    p.add_argument("--case-a", help="case A data directory (default $ZVDVC_CASE_A)")
    p.add_argument("--cache", help="directory for C-ordered .raw copies (default $ZVDVC_CASE_A_CACHE or runs/case_A)")
    p.add_argument("--json", help="write rows, the fitted model and the environment here")
    args = p.parse_args(argv)
    data = ca.data_dir(args.case_a)
    if data is None:
        raise SystemExit("case A data not found: pass --case-a DIR or set ZVDVC_CASE_A")
    cfg = ca.case_config(data, ca.cache_dir(args.cache))
    rows = sweep(cfg, tuple(sorted(args.spacings, reverse=True)), args.backends)
    fitted = model(rows, args.backends)
    print(json.dumps(fitted, indent=1))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"rows": rows, "model": fitted, "environment": environment()}, indent=1))


if __name__ == "__main__":
    main()
