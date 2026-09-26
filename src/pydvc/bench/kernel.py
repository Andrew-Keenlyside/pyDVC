"""Kernel benchmark: one Gauss-Newton iteration on real data, per engine.

Times the two kernels a solve spends its time in, on a fixed point set, with
volumes already prepared (no upload, no I/O):

* ``sums``: value, gradient and normal-equation sums at every template sample
  (``gn_sums`` on the GPU), reported in µs per point-iteration;
* ``sample``: values only (``sample_values``), reported in µs per point.

The data is the iDVC case A pair (``--case-a DIR`` or ``$PYDVC_CASE_A``) with
CCPi's settings and 2 048 points drawn from a 3D grid through the sample.
Without it, a synthetic u8 speckle volume stands in; results carry a ``data``
label so the two are never compared. The small synthetic volumes of
:mod:`pydvc.bench.throughput` fit in cache and overstate GPU speed.

Every run checks each engine's sums against the float64 numpy engine on the
first 64 points. ``rel_err`` is the largest |error| over all sums, each divided
by that sum's largest |reference| value, floored at 1e-6 of the largest sum:
some sums cancel to ~0 by construction (sum q under ZNSSD), and dividing by
them would measure rounding noise, not error.

Command line::

    python -m pydvc.bench.kernel --backends fused cpu --json runs/kernel.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from pydvc.config import SearchSpec, SubvolumeSpec

N_CHECK = 64                                      # points checked against the float64 engine


@dataclasses.dataclass
class Case:
    data: str                                     # "case_a" | "synthetic"
    ref: Any                                      # host Brick
    deformed: Any                                 # host Brick
    centres: np.ndarray                           # (N, 3) float64
    params: np.ndarray                            # (N, dof) float32: where the sums are evaluated
    subvolume: SubvolumeSpec
    search: SearchSpec


def case_a(data: Path, cache: Path, n_points: int, seed: int = 0) -> Case:
    from pydvc.bench.case_a import case_config, grid_points
    from pydvc.geometry.box import Box
    from pydvc.io.volume import open_volume

    cfg = case_config(data, cache)
    vols = [open_volume(cfg.volumes, w) for w in ("reference", "deformed")]
    ref, deformed = (v.read_brick(Box((0, 0, 0), v.shape), device="cpu") for v in vols)
    _, xyz = grid_points(24.0)
    xyz = xyz[np.sort(np.random.default_rng(seed).choice(len(xyz), n_points, replace=False))]
    params = np.zeros((n_points, cfg.search.dof), dtype=np.float32)
    params[:, :3] = cfg.search.rigid_trans
    return Case("case_a", ref, deformed, xyz, params, cfg.subvolume, cfg.search)


def synthetic(n_points: int, seed: int = 0, shape: tuple[int, int, int] = (256, 256, 256),
              subvolume: SubvolumeSpec | None = None) -> Case:
    """A u8 speckle volume large enough not to fit in cache; the deformed volume is the same data."""
    from pydvc.geometry.box import Box
    from pydvc.io.volume import Brick
    from pydvc.synth.phantoms import _to_dtype, speckle_field

    subvolume = subvolume or SubvolumeSpec(geometry="sphere", size=80.0, n_samples=8000)
    box = Box((0, 0, 0), shape)
    vol = _to_dtype(speckle_field(box, seed=seed), np.dtype(np.uint8))
    brick = Brick(vol, box, box)
    search = SearchSpec(dof=6, objective="znssd", interpolation="tricubic", disp_max=10.0)
    margin = subvolume.size / 2 + 4.0
    rng = np.random.default_rng(seed)
    centres = rng.uniform(margin, np.asarray(shape[::-1], dtype=np.float64) - 1.0 - margin, size=(n_points, 3))
    params = np.zeros((n_points, search.dof), dtype=np.float32)
    params[:, :3] = rng.uniform(-1.5, 1.5, size=(n_points, 3))
    return Case("synthetic", brick, brick, centres, params, subvolume, search)


def _timer(engine: Any):
    """``(fn, repeats) -> (median seconds, coefficient of variation)``: GPU event timing on cupy engines, wall clock otherwise."""
    xp = getattr(engine, "xp", np)

    def stats(times: Any) -> tuple[float, float]:
        t = np.asarray(times, dtype=np.float64).ravel()
        return float(np.median(t)), float(t.std() / t.mean()) if len(t) > 1 and t.mean() > 0 else 0.0

    if xp is not np:
        from cupyx.profiler import benchmark

        def gpu(fn, repeats):
            return stats(benchmark(fn, (), n_repeat=repeats, n_warmup=1).gpu_times)

        return gpu

    def cpu(fn, repeats):
        fn()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return stats(times)

    return cpu


def _host(a: Any) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)


def measure(case: Case, backend: str, repeats: int = 5) -> dict[str, Any]:
    """Time ``sample`` and ``sums`` on one engine; check its sums against the float64 engine."""
    from pydvc.geometry.templates import make_template
    from pydvc.solver.engines import make_engine
    from pydvc.solver.gauss_newton import reference_terms

    template = make_template(case.subvolume)

    def inputs(eng: Any, n: int) -> tuple:
        """Prepared bricks (once per engine) and the arrays for the first ``n`` points."""
        xp = getattr(eng, "xp", np)
        ref, dfm = eng.prepare(case.ref), eng.prepare(case.deformed)
        centres = xp.asarray(case.centres[:n])
        params = xp.asarray(case.params[:n], dtype=eng.dtype)
        offsets = xp.asarray(template.offsets, dtype=eng.dtype)
        f, _ = eng.sample(ref, centres, xp.zeros_like(params), offsets, case.search)
        q, shift = reference_terms(f, case.search.objective)
        return dfm, centres, params, xp.arange(n), q, shift, offsets

    def check_sums(eng: Any, dfm: Any, centres: Any, params: Any, q: Any, shift: Any, offsets: Any) -> np.ndarray:
        sl = slice(0, N_CHECK)
        xp = getattr(eng, "xp", np)
        return _host(eng.sums(dfm, centres[sl], params[sl], xp.arange(N_CHECK), q[sl], shift[sl], offsets,
                              case.search)[0]).astype(np.float64)

    ref64 = make_engine("numpy")
    d64, c64, p64, _, q64, s64, o64 = inputs(ref64, N_CHECK)
    want = check_sums(ref64, d64, c64, p64, q64, s64, o64)
    del ref64, d64

    eng = make_engine(backend)
    n = len(case.centres)
    dfm, centres, params, idx, q, shift, offsets = inputs(eng, n)
    got = check_sums(eng, dfm, centres, params, q, shift, offsets)
    timer = _timer(eng)
    t_sums, cv_sums = timer(lambda: eng.sums(dfm, centres, params, idx, q, shift, offsets, case.search), repeats)
    t_sample, cv_sample = timer(lambda: eng.sample(dfm, centres, params, offsets, case.search), repeats)
    return {
        "backend": backend,
        "sums_us_per_point_iter": t_sums / n * 1e6,
        "sample_us_per_point": t_sample / n * 1e6,
        "sums_cv": cv_sums,
        "sample_cv": cv_sample,
        "points": n,
        "repeats": repeats,
        "rel_err": rel_err(got, want),
    }


def rel_err(got: np.ndarray, want: np.ndarray) -> float:
    """Largest per-sum relative error; the scale of each sum is floored at 1e-6 of the largest (see module doc)."""
    scale = np.abs(want).max(axis=0)
    return float(np.max(np.abs(got - want) / np.maximum(scale, 1e-6 * scale.max())))


def run(backends: list[str], *, case_a_dir: str | None = None, cache: str | None = None, n_points: int = 2048,
        repeats: int = 5) -> dict[str, Any]:
    from pydvc.bench import case_a as ca
    from pydvc.bench.smoke import environment
    from pydvc.geometry.templates import make_template

    data = ca.data_dir(case_a_dir)
    case = case_a(data, ca.cache_dir(cache), n_points) if data else synthetic(n_points)
    return {
        "data": case.data,
        "settings": {"n_points": len(case.centres), "subvolume": dataclasses.asdict(case.subvolume),
                     "dof": case.search.dof, "objective": case.search.objective,
                     "interpolation": case.search.interpolation,
                     "template_digest": make_template(case.subvolume).digest()},
        "environment": environment(),
        "results": [measure(case, b, repeats) for b in backends],
    }


def main(argv: list[str] | None = None) -> None:
    from pydvc.solver.engines import gpu_available

    p = argparse.ArgumentParser(prog="python -m pydvc.bench.kernel", description=__doc__.split("\n\n")[0])
    p.add_argument("--backends", nargs="+", default=["fused", "cpu"] if gpu_available() else ["cpu"])
    p.add_argument("--case-a", help="case A data directory (default $PYDVC_CASE_A; synthetic if absent)")
    p.add_argument("--cache", help="directory for C-ordered .raw copies (default $PYDVC_CASE_A_CACHE or runs/case_A)")
    p.add_argument("--points", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--json", help="write the results here")
    args = p.parse_args(argv)
    res = run(args.backends, case_a_dir=args.case_a, cache=args.cache, n_points=args.points, repeats=args.repeats)
    for r in res["results"]:
        print(f"{res['data']:9s} {r['backend']:8s} sums {r['sums_us_per_point_iter']:8.2f} µs/point-iter   "
              f"sample {r['sample_us_per_point']:8.2f} µs/point   rel err vs float64 {r['rel_err']:.1e}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
