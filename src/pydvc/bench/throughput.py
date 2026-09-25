"""Throughput and utilisation measurements.

Records points/s per stage (read, seed, solve, write), GPU time split by
NVTX range, achieved FLOP/s and L2 hit rate for the fused kernel (Nsight
Compute metrics, when available), and compute-stream idle time spent waiting
on I/O. These are the numbers that replace the model in docs/PERFORMANCE.md.

``kernel_microbench`` times the solve alone on a synthetic speckle brick held
in memory (I/O excluded) on any backend. Achieved FLOP/s uses the operation
count of :func:`flops_per_sample_iteration`, a model of the fused kernel's
arithmetic, so it is an estimate; Nsight Compute gives the measured figure::

    ncu --set full --kernel-name regex:gn_sums python -m pydvc.bench.throughput --backend fused

Command line::

    python -m pydvc.bench.throughput --backend cpu numpy --samples 2000 --dof 6 12 --batch 2048
"""

from __future__ import annotations

import argparse
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from pydvc.config import RunConfig, SearchSpec, SubvolumeSpec


@dataclass
class Throughput:
    backend: str                      # "ccpi" | "numpy" | "cupy" | "fused" | "cpu"
    hardware: str
    points: int
    seconds_total: float
    seconds_by_stage: dict[str, float] = field(default_factory=dict)
    points_per_second: float = 0.0
    achieved_tflops: float | None = None
    io_wait_fraction: float | None = None
    mean_iterations: float | None = None
    settings: dict[str, Any] = field(default_factory=dict)


def flops_per_sample_iteration(dof: int, interpolation: str) -> int:
    """Floating-point operations per template sample per Gauss-Newton iteration in ``gn_sums``.

    Interpolation with gradient (tricubic: 12 weight/derivative-weight
    polynomials, 16 rows of 4 taps each for value and x-derivative, the y and z
    contractions), the warp (``F d``: 18), the Jacobian rows beyond translation
    (``(dof - 3)`` dot products of 15), and the sums (``dof (dof + 1)`` for
    ``J^T J``, ``6 dof`` for the three vector sums, 12 for the scalars).
    """
    interp = {"tricubic": 12 * 7 + 16 * 16 + 4 * 12 + 16, "trilinear": 60, "nearest": 2}[interpolation]
    return interp + 18 + 15 * (dof - 3) + dof * (dof + 1) + 6 * dof + 12


def hardware_name(backend: str) -> str:
    if backend in ("cupy", "fused"):
        try:
            import cupy

            return cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        except Exception:
            return "unknown GPU"
    import os

    return f"{platform.processor() or platform.machine()} x{os.cpu_count()}"


def _synthetic_pair(shape: tuple[int, int, int], shift: tuple[float, float, float], seed: int) -> tuple[Any, Any]:
    from pydvc.geometry.box import Box
    from pydvc.io.volume import Brick
    from pydvc.synth.phantoms import DisplacementField, _to_dtype, speckle_field, warp_volume

    f = speckle_field(Box((0, 0, 0), shape), seed=seed)
    g = warp_volume(f, DisplacementField("affine", {"translation": shift}))
    box = Box((0, 0, 0), shape)
    return (
        Brick(_to_dtype(f, np.dtype(np.uint16)), box, box),
        Brick(_to_dtype(g, np.dtype(np.uint16)), box, box),
    )


def kernel_microbench(
    *,
    n_samples: int,
    dof: int,
    interpolation: str,
    objective: str,
    batch: int,
    backend: str = "fused",
    size: float = 32.0,
    shape: tuple[int, int, int] = (160, 160, 160),
    repeats: int = 3,
    seed: int = 0,
) -> Throughput:
    """Fused GN step on a synthetic brick: points/s and FLOP/s with I/O excluded.

    ``batch`` points, uniformly spread where their subvolume fits, are solved
    from a seed 0.6 voxel off a pure translation; the best of ``repeats``
    timed runs (after one warm-up that also compiles) is reported.
    """
    from pydvc.geometry.templates import make_template
    from pydvc.solver.engines import make_engine
    from pydvc.solver.gauss_newton import solve_batch

    ref, deformed = _synthetic_pair(shape, (1.2, -0.7, 0.4), seed)
    template = make_template(SubvolumeSpec(geometry="sphere", size=size, n_samples=n_samples))
    search = SearchSpec(dof=dof, objective=objective, interpolation=interpolation, disp_max=4.0)
    margin = template.extent() + 6.0
    rng = np.random.default_rng(seed)
    centres = rng.uniform(margin, np.asarray(shape[::-1]) - 1.0 - margin, size=(batch, 3))
    seeds = np.broadcast_to([0.8, -0.4, 0.0], (batch, 3))
    engine = make_engine(backend)
    sync = _sync(engine)
    solve_batch(ref, deformed, centres[: min(batch, 64)], seeds[: min(batch, 64)], template, search, engine=engine)
    best, res = np.inf, None
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        res = solve_batch(ref, deformed, centres, seeds, template, search, engine=engine)
        sync()
        best = min(best, time.perf_counter() - t0)
    n_iter = np.asarray(res.n_iter.get() if hasattr(res.n_iter, "get") else res.n_iter, dtype=np.float64)
    evals = (n_iter.sum() + batch) * n_samples          # iterations, plus the final objective (value-only; counted fully)
    flops = evals * flops_per_sample_iteration(dof, interpolation)
    return Throughput(
        backend=backend,
        hardware=hardware_name(backend),
        points=batch,
        seconds_total=best,
        points_per_second=batch / best,
        achieved_tflops=flops / best / 1e12,
        mean_iterations=float(n_iter.mean()),
        settings=dict(n_samples=n_samples, dof=dof, interpolation=interpolation, objective=objective, size=size),
    )


def _sync(engine: Any):
    xp = engine.xp
    if xp is np:
        return lambda: None
    return lambda: xp.cuda.runtime.deviceSynchronize()


def end_to_end(cfg: RunConfig, *, backend: str) -> Throughput:
    """prepare -> seed -> run -> finalize on one process, timed per stage."""
    from pydvc.pipeline import coordinator

    stages: dict[str, float] = {}
    t_all = time.perf_counter()
    for name, fn in (("prepare", coordinator.prepare), ("seed", coordinator.seed)):
        t0 = time.perf_counter()
        fn(cfg, backend=backend) if name == "seed" else fn(cfg)
        stages[name] = time.perf_counter() - t0
    t0 = time.perf_counter()
    stats = coordinator.run(cfg, backend=backend)
    stages["run"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    summary = coordinator.finalize(cfg)
    stages["finalize"] = time.perf_counter() - t0
    total = time.perf_counter() - t_all
    busy = sum(s.seconds_compute for s in stats)
    wait = sum(s.seconds_io_wait for s in stats)
    return Throughput(
        backend=backend,
        hardware=hardware_name(backend),
        points=summary.n_points,
        seconds_total=total,
        seconds_by_stage=stages,
        points_per_second=summary.n_points / total,
        io_wait_fraction=wait / (busy + wait) if busy + wait > 0 else None,
        settings=dict(bytes_read=sum(s.bytes_read for s in stats), tiles=sum(s.tiles for s in stats)),
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pydvc.bench.throughput", description="fused-step micro-benchmark")
    p.add_argument("--backend", nargs="+", default=["fused"])
    p.add_argument("--samples", type=int, nargs="+", default=[2000])
    p.add_argument("--dof", type=int, nargs="+", default=[6])
    p.add_argument("--objective", default="znssd")
    p.add_argument("--interpolation", default="tricubic")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--size", type=float, default=32.0)
    args = p.parse_args(argv)
    for backend in args.backend:
        for m in args.samples:
            for dof in args.dof:
                r = kernel_microbench(
                    n_samples=m, dof=dof, interpolation=args.interpolation, objective=args.objective,
                    batch=args.batch, backend=backend, size=args.size,
                )
                print(
                    f"{backend:7s} M={m:5d} dof={dof:2d} {r.points:6d} pts {r.seconds_total:7.3f} s "
                    f"{r.points_per_second:10.1f} pt/s  ~{1e3 * r.achieved_tflops:8.2f} GFLOP/s  "
                    f"iters {r.mean_iterations:.2f}  [{r.hardware}]"
                )
                _ = asdict(r)


if __name__ == "__main__":
    main()
