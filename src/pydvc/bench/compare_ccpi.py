"""Head-to-head against CCPi ``dvc`` (the engine iDVC runs) on one case: speed and agreement.

Timed runs, each on the same machine, points and settings:

``ccpi <v> iDVC``      ``dvc`` as iDVC launches it: one process, OpenMP on every core.
``ccpi <v> xP``        ``P`` concurrent ``dvc`` processes on disjoint point subsets (1 thread
                       each): the best a node gets from CCPi without code changes.
``pydvc <b> parity``   whole volumes in memory, CCPi point order (wavefront), in this
                       process: read + solve, the like-for-like number.
``pydvc <b> cli``      ``pydvc plan / seed / run / finalize`` as separate processes: what a
                       user waits for, start-up (imports, kernel compilation) included.

Agreement: pyDVC against each CCPi result point by point (the Q1 criteria:
median |du| <= 0.05, p95 <= 0.2, status agreement >= 98 %), against a reference
``.disp`` if given, and against ground truth for synthetic cases.

Command line::

    python -m pydvc.bench.compare_ccpi CONFIG --out runs/compare [--ccpi-exe DVC [DVC ...]]
        [--ccpi-processes 4] [--backends cpu fused] [--reference-disp REF.disp] [--truth truth.npz]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydvc.config import RunConfig


@dataclass
class TimedRun:
    name: str
    points: int
    seconds: float
    detail: dict[str, Any] = field(default_factory=dict)
    results: str | None = None          # .disp or .npz or store with the run's displacements

    @property
    def points_per_second(self) -> float:
        return self.points / self.seconds if self.seconds > 0 else float("nan")


def dvc_version(exe: str | Path) -> str:
    """The version ``dvc`` reports in its manual (e.g. ``v22.0.0-19-gb0e2642``), else the conda package folder."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([str(exe), "manual"], cwd=tmp, capture_output=True, timeout=60)
        manual = Path(tmp) / "dvc_manual"
        if manual.exists():
            for line in manual.read_text(errors="replace").splitlines():
                if line.strip().startswith("version:"):
                    return line.split(":", 1)[1].strip()
    return Path(exe).resolve().parent.parent.name


def _previous_ccpi(workdir: Path, threads: int, procs: int) -> TimedRun | None:
    """A finished CCPi run of this shape in ``workdir``, timed from its ``.stat`` files (slowest process)."""
    import re

    merged = workdir / f"t{threads}_p{procs}.disp"
    stats = [workdir / f"t{threads}_p{procs}_{k}.stat" for k in range(procs)]
    if not merged.exists() or not all(s.exists() for s in stats):
        return None
    points, seconds = 0, 0.0
    for s in stats:
        m = re.search(r"(\d+) points processed in (\d+) seconds", s.read_text())
        if not m:
            return None
        points += int(m.group(1))
        seconds = max(seconds, float(m.group(2)))
    return TimedRun("", points, seconds, {"from_stat": True}, str(merged))


def run_ccpi_variants(
    cfg: RunConfig, out: Path, exe: str | Path, *, processes: int, max_points: int | None, reuse: bool = False
) -> list[TimedRun]:
    from pydvc.bench.ccpi_baseline import run_ccpi

    cores = os.cpu_count() or 1
    label = Path(exe).resolve().parent.parent.name or "dvc"          # e.g. the conda package folder, ccpi-dvc-22.0.0-0
    runs = []
    for threads, procs, tag in ((cores, 1, "iDVC (1 process, all cores)"), (1, processes, f"{processes} processes x 1 thread")):
        if procs > 1 and processes <= 1:
            continue
        name = f"CCPi {label}: {tag}"
        detail = {"omp_threads": threads, "processes": procs, "exe": str(exe)}
        prev = _previous_ccpi(out / f"ccpi_{label}", threads, procs) if reuse and not max_points else None
        if prev is not None:
            prev.name = name
            prev.detail.update(detail, sampled=False)
            runs.append(prev)
            continue
        r = run_ccpi(cfg, workdir=out / f"ccpi_{label}", omp_threads=threads, processes=procs, max_points=max_points, exe=exe)
        runs.append(TimedRun(name, r.points, r.seconds,
                             dict(detail, sampled=r.sampled, stat_pt_s=r.stat_points_per_second), str(r.disp_path)))
    return runs


def run_pydvc_parity(cfg: RunConfig, out: Path, backend: str) -> TimedRun:
    from pydvc.pipeline.inmemory import load_points, solve_in_memory

    point_id, xyz = load_points(cfg)
    solve_in_memory(cfg, point_id[:8], xyz[:8], strategy="rigid", backend=backend)       # compile/JIT outside the timing
    t0 = time.perf_counter()
    res = solve_in_memory(cfg, point_id, xyz, strategy="wavefront", backend=backend)
    seconds = time.perf_counter() - t0
    path = out / f"pydvc_{backend}_parity.npz"
    res.save(path)
    return TimedRun(f"pyDVC {backend}: parity (in memory, CCPi order)", len(point_id), seconds,
                    {"read_s": res.timings["read"], "solve_s": res.timings["solve"],
                     "mean_iterations": float(res.n_iter[res.status == 0].mean()) if (res.status == 0).any() else None},
                    str(path))


def run_pydvc_cli(cfg: RunConfig, out: Path, backend: str) -> TimedRun:
    """plan / seed / run / finalize as separate processes, each timed, start-up included."""
    import shutil

    work = out / f"pydvc_{backend}_cli"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    c = dataclasses.replace(cfg, output=str(work / "results.zarrvectors"), workdir=str(work / "work"))
    config = work / "config.yaml"
    c.to_yaml(config)
    stages = {}
    for stage in ("plan", "seed", "run", "finalize"):
        cmd = [sys.executable, "-m", "pydvc.cli", stage, str(config)] + (["--backend", backend] if stage != "finalize" else [])
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stages[stage] = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"pydvc {stage} failed:\n{proc.stderr[-3000:]}")
    from pydvc.io.results import ResultStore

    n = len(ResultStore(c.output).read_all()["point_id"])
    return TimedRun(f"pyDVC {backend}: CLI end to end", n, sum(stages.values()), {"stages_s": stages}, c.output)


def agreement(ours: str, theirs: str) -> dict[str, Any]:
    from pydvc.bench.metrics import against_disp

    a = against_disp(ours, theirs)
    return {"median_abs": a.median_abs, "p95_abs": a.p95_abs, "status_agreement": a.status_agreement,
            "n_good_both": a.n_good, "rmse": a.rmse,
            "q1_pass": bool(a.median_abs <= 0.05 and a.p95_abs <= 0.2 and (a.status_agreement or 0) >= 0.98)}


def compare(
    cfg: RunConfig,
    out: str | Path,
    *,
    ccpi_exes: list[str | Path],
    ccpi_processes: int = 4,
    backends: list[str],
    cli: bool = True,
    reference_disp: str | Path | None = None,
    truth: str | Path | None = None,
    max_ccpi_points: int | None = None,
    title: str = "CCPi vs pyDVC",
    reuse_ccpi: bool = False,
) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs: list[TimedRun] = []
    if ccpi_exes:
        from pydvc.bench.ccpi_baseline import ccpi_ready_config

        raw_dir = out / "raw"
        raw_dir.mkdir(exist_ok=True)
        cfg_ccpi = ccpi_ready_config(cfg, raw_dir)            # OME-Zarr -> raw once, shared by every CCPi run
        for exe in ccpi_exes:
            runs += run_ccpi_variants(cfg_ccpi, out, exe, processes=ccpi_processes, max_points=max_ccpi_points,
                                      reuse=reuse_ccpi)
    for b in backends:
        runs.append(run_pydvc_parity(cfg, out, b))
        if cli:
            runs.append(run_pydvc_cli(cfg, out, b))
    report: dict[str, Any] = {
        "title": title,
        "machine": {"host": platform.node(), "cpu": platform.processor() or platform.machine(), "cores": os.cpu_count()},
        "ccpi_versions": {str(e): dvc_version(e) for e in ccpi_exes},
        "runs": [dict(dataclasses.asdict(r), points_per_second=r.points_per_second) for r in runs],
        "agreement": {},
    }
    ours = [r for r in runs if r.name.startswith("pyDVC") and r.results]
    ccpi = [r for r in runs if r.name.startswith("CCPi") and r.results and not r.detail.get("sampled")]
    for o in ours:
        for c in ccpi:
            report["agreement"][f"{o.name} vs {c.name}"] = agreement(o.results, c.results)
        if reference_disp:
            report["agreement"][f"{o.name} vs reference .disp"] = agreement(o.results, str(reference_disp))
        if truth:
            from pydvc.bench.metrics import against_truth

            a = against_truth(o.results, truth)
            report["agreement"][f"{o.name} vs ground truth"] = {"rmse": a.rmse, "frac_good": a.frac_good,
                                                               "median_abs": a.median_abs, "p95_abs": a.p95_abs}
    if truth:
        from pydvc.bench.metrics import against_truth

        for c in ccpi:
            a = against_truth(c.results, truth)
            report["agreement"][f"{c.name} vs ground truth"] = {"rmse": a.rmse, "frac_good": a.frac_good,
                                                               "median_abs": a.median_abs, "p95_abs": a.p95_abs}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "report.md").write_text(markdown(report))
    return report


def markdown(report: dict[str, Any]) -> str:
    runs = report["runs"]
    idvc = next((r for r in runs if "iDVC" in r["name"]), None)
    best_ccpi = max((r for r in runs if r["name"].startswith("CCPi")), key=lambda r: r["points_per_second"], default=None)
    m = report["machine"]
    lines = [f"# {report['title']}", "", f"Machine: {m['host']}, {m['cpu']}, {m['cores']} cores. "
             f"CCPi: {', '.join(report['ccpi_versions'].values()) or 'not run'}.", "",
             "| run | points | wall time | pt/s | vs iDVC | vs best CCPi |", "|---|---|---|---|---|---|"]
    for r in runs:
        rel = f"{r['points_per_second'] / idvc['points_per_second']:.1f}x" if idvc else "-"
        relb = f"{r['points_per_second'] / best_ccpi['points_per_second']:.1f}x" if best_ccpi else "-"
        lines.append(f"| {r['name']} | {r['points']} | {r['seconds']:.1f} s | {r['points_per_second']:.1f} | {rel} | {relb} |")
    lines += ["", "| comparison | median \\|du\\| | p95 \\|du\\| | status agreement | RMSE (x, y, z) |", "|---|---|---|---|---|"]
    for name, a in report["agreement"].items():
        rmse = ", ".join(f"{v:.4f}" for v in a["rmse"])
        sa = f"{100 * a['status_agreement']:.1f} %" if a.get("status_agreement") is not None else "-"
        lines.append(f"| {name} | {a['median_abs']:.4f} | {a['p95_abs']:.4f} | {sa} | {rmse} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    from pydvc.solver.engines import default_backend

    p = argparse.ArgumentParser(prog="python -m pydvc.bench.compare_ccpi", description=__doc__.split("\n\n")[0])
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.add_argument("--ccpi-exe", nargs="*", default=None, help="dvc executables (default: $PYDVC_CCPI_DVC or dvc on PATH)")
    p.add_argument("--ccpi-processes", type=int, default=os.cpu_count() or 1)
    p.add_argument("--backends", nargs="+", default=None)
    p.add_argument("--no-cli", action="store_true")
    p.add_argument("--reference-disp")
    p.add_argument("--truth")
    p.add_argument("--max-ccpi-points", type=int)
    p.add_argument("--reuse-ccpi", action="store_true", help="time finished CCPi runs in --out from their .stat files")
    args = p.parse_args(argv)
    from pydvc.bench.ccpi_baseline import find_dvc

    exes = args.ccpi_exe if args.ccpi_exe is not None else [str(find_dvc())]
    report = compare(RunConfig.from_yaml(args.config), args.out, ccpi_exes=exes, ccpi_processes=args.ccpi_processes,
                     backends=args.backends or [default_backend()], cli=not args.no_cli,
                     reference_disp=args.reference_disp, truth=args.truth, max_ccpi_points=args.max_ccpi_points,
                     reuse_ccpi=args.reuse_ccpi)
    print(markdown(report))


if __name__ == "__main__":
    main()
