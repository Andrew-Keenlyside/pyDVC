"""``pydvc selftest``: check an installation end to end on a small synthetic case.

1. Reports the environment: package versions, numba, cupy and GPUs, the CCPi ``dvc``.
2. Writes a 96^3 affine phantom (about 30 s).
3. For every backend available here (``cpu`` with numba, ``fused`` and
   ``cupy`` with a GPU, else ``numpy``), runs ``plan -> seed -> run -> finalize``
   through the pipeline and checks accuracy against ground truth (>= 99 % GOOD,
   RMSE <= 0.02 voxel).
4. With a GPU, checks that the GPU backends agree with the CPU engine
   (<= 1e-3 voxel, >= 99.9 % status agreement: the M2 parity criterion).
5. With CCPi ``dvc`` available (``$PYDVC_CCPI_DVC`` or on PATH), runs the
   head-to-head comparison on the same case.

Exit code 0 when every check passes. The summary ends with a PASS/FAIL table;
``--out`` keeps the case, stores and reports for inspection.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata as md
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def environment() -> dict[str, Any]:
    def version(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("pydvc", "numpy", "scipy", "zarr", "zarr-vectors", "numba", "cupy-cuda12x", "cupy")},
        "gpus": [],
        "ccpi_dvc": None,
    }
    try:
        import cupy

        for i in range(cupy.cuda.runtime.getDeviceCount()):
            p = cupy.cuda.runtime.getDeviceProperties(i)
            env["gpus"].append(f"{i}: {p['name'].decode()} ({p['totalGlobalMem'] / 2**30:.0f} GiB)")
        env["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
    except Exception as exc:                      # no cupy, no driver, or no device
        env["gpu_note"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        from pydvc.bench.ccpi_baseline import find_dvc

        env["ccpi_dvc"] = str(find_dvc())
    except FileNotFoundError:
        pass
    return env


def available_backends(env: dict[str, Any]) -> list[str]:
    backends = []
    if env["gpus"]:
        backends += ["fused", "cupy"]
    if env["packages"].get("numba"):
        backends.append("cpu")
    if not backends or (env["gpus"] and "cpu" not in backends):
        backends.append("numpy")                  # the reference, for GPU parity when numba is absent
    return backends


def _results(path: str) -> dict[str, np.ndarray]:
    from pydvc.io.results import ResultStore

    r = ResultStore(path).read_all()
    o = np.argsort(r["point_id"])
    return {k: v[o] for k, v in r.items()}


def run(out: Path, *, backends: list[str] | None = None, ccpi: bool = True) -> tuple[bool, dict[str, Any]]:
    from pydvc.bench.metrics import against_truth
    from pydvc.config import ClusterSpec, RunConfig, SearchSpec, SeedingSpec, SubvolumeSpec
    from pydvc.pipeline import coordinator
    from pydvc.synth.phantoms import default_field, make_case

    env = environment()
    backends = backends or available_backends(env)
    checks: list[tuple[str, bool, str]] = []
    report: dict[str, Any] = {"environment": env, "backends": backends, "runs": {}}
    shape = (96, 96, 96)
    case = out / "case"
    t0 = time.perf_counter()
    if not (case / "config.yaml").exists():
        make_case(case, shape_zyx=shape, field=default_field("affine", shape), spacing=12.0, chunk=48, shard=96,
                  subvolume=SubvolumeSpec(geometry="sphere", size=24, n_samples=1500),
                  search=SearchSpec(dof=12, disp_max=8.0))
    report["phantom_s"] = time.perf_counter() - t0
    base = RunConfig.from_yaml(case / "config.yaml")

    stores = {}
    for backend in backends:
        cfg = dataclasses.replace(base, output=str(out / f"{backend}.zarrvectors"), workdir=str(out / f"run_{backend}"),
                                  cluster=ClusterSpec(tile_shape=(48, 48, 48), prefetch_depth=2),
                                  seeding=SeedingSpec(strategy="rigid"))
        import shutil

        shutil.rmtree(cfg.output, ignore_errors=True)
        shutil.rmtree(cfg.workdir, ignore_errors=True)
        try:
            t0 = time.perf_counter()
            coordinator.prepare(cfg, backend=backend)
            coordinator.seed(cfg, backend=backend)
            stats = coordinator.run(cfg, backend=backend)
            coordinator.finalize(cfg)
            seconds = time.perf_counter() - t0
            errors = [e for s in stats for e in s.errors]
            acc = against_truth(cfg.output, case / "truth.npz")
            ok = not errors and acc.frac_good >= 0.99 and max(acc.rmse) <= 0.02
            detail = f"{acc.n_points} pts, {100 * acc.frac_good:.1f} % GOOD, RMSE {max(acc.rmse):.4f}, {seconds:.1f} s"
            if errors:
                detail += f"; {len(errors)} worker errors: {errors[0][:300]}"
            checks.append((f"pipeline on '{backend}' vs ground truth", ok, detail))
            report["runs"][backend] = {"seconds": seconds, "rmse": acc.rmse, "frac_good": acc.frac_good, "errors": errors}
            stores[backend] = cfg.output
        except Exception as exc:
            import traceback

            checks.append((f"pipeline on '{backend}'", False, f"{type(exc).__name__}: {exc}"))
            report["runs"][backend] = {"traceback": traceback.format_exc()}

    reference = "cpu" if "cpu" in stores else ("numpy" if "numpy" in stores else None)
    for backend in ("fused", "cupy"):
        if backend in stores and reference:
            a, b = _results(stores[reference]), _results(stores[backend])
            agree = float((a["status"] == b["status"]).mean())
            good = a["status"] == 0
            diff = float(np.abs(a["displacement"][good] - b["displacement"][good]).max()) if good.any() else float("nan")
            checks.append((f"'{backend}' agrees with '{reference}'", agree >= 0.999 and diff <= 1e-3,
                           f"status agreement {100 * agree:.2f} %, max |du| {diff:.2e}"))

    if ccpi and env["ccpi_dvc"] and stores:
        from pydvc.bench.compare_ccpi import compare

        try:
            cmp = compare(base, out / "ccpi_compare", ccpi_exes=[env["ccpi_dvc"]], ccpi_processes=2,
                          backends=[next(iter(stores))], cli=False, truth=case / "truth.npz", title="selftest: CCPi vs pyDVC")
            vs = [a for k, a in cmp["agreement"].items() if " vs CCPi " in k]
            ok = bool(vs) and all(a["median_abs"] <= 0.05 and a["p95_abs"] <= 0.2 for a in vs)
            runs = {r["name"]: r["seconds"] for r in cmp["runs"]}
            checks.append(("CCPi dvc runs and agrees with pyDVC", ok,
                           "; ".join(f"{n}: {s:.1f} s" for n, s in runs.items())))
        except Exception as exc:
            checks.append(("CCPi dvc comparison", False, f"{type(exc).__name__}: {exc}"[:400]))

    report["checks"] = [{"check": c, "pass": p, "detail": d} for c, p, d in checks]
    (out / "selftest.json").write_text(json.dumps(report, indent=1, default=str))
    return all(p for _, p, _ in checks) and bool(checks), report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pydvc selftest", description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="runs/selftest")
    p.add_argument("--backends", nargs="+", help="default: every backend available here")
    p.add_argument("--no-ccpi", action="store_true")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = environment()
    print("environment")
    for k, v in env.items():
        print(f"  {k}: {v}")
    ok, report = run(out, backends=args.backends, ccpi=not args.no_ccpi)
    print("\nchecks")
    for c in report["checks"]:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['detail']}")
    print(f"\n{'PASS' if ok else 'FAIL'}; details in {out / 'selftest.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
