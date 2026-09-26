"""Local check suite: ``pydvc check {quick,gpu,full}``.

pyDVC's GPU code cannot be tested on hosted CI, so this runs on your own
machine. Each tier includes the one before:

=====  ==============================================================  =========
tier   runs                                                            time
=====  ==============================================================  =========
quick  the test suite without GPU and slow tests (the CUDA emulator    ~3 min
       included)
gpu    ``-m gpu`` tests and ``pydvc selftest``                          ~5 min
full   ``-m slow`` tests; the kernel benchmark and, with case A data,  ~5 min
       the central grid (4 680 points) on fused and cpu, again on fused
       with a 1-voxel prefilter, and an 86 k point 3D grid on fused
=====  ==============================================================  =========

Results go to ``runs/check/<UTC time>-<commit>[-dirty]/check.json`` (and
``runs/check/latest``) and are compared with this machine's baseline,
``docs/benchmarks/baselines/<machine>.json`` (``$PYDVC_MACHINE``, default the
host name), which is committed. Its reference result arrays live in
``runs/check/baseline/``. ``--update-baseline --reason "..."`` records a new one.

Rules:

* **Accuracy fails the check.** Tests and selftest must pass; kernel sums must
  be within 1e-4 of float64; case A results must agree with the baseline
  arrays (status on >= 99.9 % of points, |du| <= 1e-3 voxel) and with CCPi no
  worse than the baseline (median and p95 +0.005 voxel, status agreement
  -0.2 %). Bit-identical results are expected but only warned about, and only
  when the software fingerprint matches the baseline's.
* **Speed is compared on the same hardware only**, and not while other
  processes use the GPU. Kernel timings warn at 20 % slower and fail at 40 %,
  end-to-end timings at 25 % and 50 %, each widened to 3x the timing's
  coefficient of variation at baseline. Back-to-back runs differ by up to
  ~18 % on a workstation GPU (clocks, temperature), so the limits catch real
  regressions (the ones this suite exists for were 2-20x), not drift. More
  than 20 % faster is reported.

Case A data comes from ``--case-a DIR`` or ``$PYDVC_CASE_A``. Without it the
case A steps are skipped (result ``PASS (partial)``; ``--require-case-a``
makes that a failure) and the kernel benchmark uses a synthetic volume, which
has its own baseline entries.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable

import numpy as np

TIERS = ("quick", "gpu", "full")
# (warn, fail) fractional slow-down. Back-to-back runs on the RTX A2000 workstation differ by up to ~18 %
# (boost clocks, temperature), much more than the spread within one run, so tighter limits give false alarms.
PERF_LIMITS = {"kernel": (0.20, 0.40), "e2e": (0.25, 0.50)}
FASTER_NOTE = 0.20
CCPI_SLACK = {"median": 0.005, "p95": 0.005, "status_agreement": -0.002}
ARRAY_STATUS_MIN, ARRAY_DU_MAX = 0.999, 1e-3
KERNEL_REL_ERR_MAX = 1e-4
E2E_SPACING = 24.0                                             # 86 100 points on case A
CPU_BUSY_LOAD = 0.5                                            # load average per core above which CPU timings are skipped


@dataclasses.dataclass
class Step:
    name: str
    status: str = "pass"                  # pass | warn | fail | skip
    seconds: float = 0.0
    detail: str = ""
    metrics: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)   # name -> {value, cv}
    arrays: dict[str, str] = dataclasses.field(default_factory=dict)                 # name -> .npz in the run dir


@dataclasses.dataclass
class Finding:
    metric: str
    status: str                           # pass | warn | fail | skip | note
    detail: str


# --------------------------------------------------------------------------- environment


def repo_root() -> Path:
    """The source tree whose tests run: the git checkout, else ``$PYDVC_SRC``, the container's copy, or cwd."""
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
        return Path(out.stdout.strip())
    except Exception:
        for cand in (os.environ.get("PYDVC_SRC"), "/opt/pydvc-src"):
            if cand and (Path(cand) / "tests").is_dir():
                return Path(cand)
        return Path.cwd()


def out_root(src: Path) -> Path:
    """Where runs/check goes: ``$PYDVC_CHECK_OUT``, else the source tree if writable (a checkout), else cwd."""
    env = os.environ.get("PYDVC_CHECK_OUT")
    if env:
        return Path(env)
    return src if os.access(src, os.W_OK) else Path.cwd()


def baseline_dir(src: Path) -> Path:
    """``$PYDVC_BASELINE_DIR`` (e.g. shared storage on a cluster), else the checkout's docs/benchmarks/baselines."""
    return Path(os.environ.get("PYDVC_BASELINE_DIR") or src / "docs" / "benchmarks" / "baselines")


def git_info(root: Path) -> dict[str, Any]:
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()
        except OSError:                          # no git (a container)
            return ""

    sha = git("rev-parse", "HEAD")
    if not sha:                                  # the container records the commit it was built from
        commit = os.environ.get("PYDVC_COMMIT", "unknown")
        return {"sha": commit.removesuffix("-dirty"), "branch": "", "dirty": commit.endswith("-dirty")}
    return {"sha": sha, "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def fingerprint() -> dict[str, Any]:
    """Hardware (decides whether timings compare) and software (decides whether results should be bit-identical)."""
    from pydvc.bench.smoke import environment

    env = environment()
    hw = {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(), "gpus": env.get("gpus", [])}
    sw = {"python": env["python"], "packages": env["packages"], "cuda_runtime": env.get("cuda_runtime")}
    try:
        from cupy.cuda import nvrtc, runtime

        sw["nvrtc"] = list(nvrtc.getVersion())
        sw["driver"] = runtime.driverGetVersion()
    except Exception:
        pass
    return {"hardware": hw, "software": sw}


def gpu_state() -> dict[str, Any]:
    """Other processes on the GPU (they make timings meaningless), and clocks/temperature for the record."""
    state: dict[str, Any] = {"available": False, "other_processes": []}
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20).stdout
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return state
    state["available"] = True
    state["other_processes"] = [line.strip() for line in apps.splitlines()
                                if line.strip() and line.split(",")[0].strip() != str(os.getpid())]
    vals = [v.strip() for v in gpu.splitlines()[0].split(",")] if gpu.strip() else []
    if len(vals) == 4:
        state |= {"temperature_c": vals[0], "sm_clock_mhz": vals[1], "max_sm_clock_mhz": vals[2], "utilization_pct": vals[3]}
    return state


def preflight(tier: str) -> tuple[list[Finding], dict[str, Any]]:
    from pydvc.kernels.cuda.emulate import compiler
    from pydvc.solver.engines import gpu_available

    found: list[Finding] = []
    cxx = compiler()
    found.append(Finding("C++20 compiler (CUDA emulator)", "pass" if cxx else "warn",
                         cxx or "none: emulator tests will be skipped"))
    state = gpu_state()
    if tier != "quick":
        if not gpu_available():
            found.append(Finding("GPU", "fail", "no CUDA device visible to cupy"))
        else:
            import cupy

            prefix = os.environ.get("CONDA_PREFIX", "")
            cuda_path = str(cupy.cuda.get_cuda_path() or "")
            if prefix and cuda_path.startswith(prefix):
                found.append(Finding("CuPy CUDA headers", "pass", cuda_path))
            else:
                found.append(Finding("CuPy CUDA headers", "fail",
                                     f"CuPy uses {cuda_path!r}, outside CONDA_PREFIX={prefix!r}: activate the "
                                     "environment (conda run -n <env> ...), or NVRTC mixes CUDA versions"))
        if state["other_processes"]:
            found.append(Finding("GPU idle", "warn", "timings not compared; in use by " + "; ".join(state["other_processes"])))
    return found, state


# --------------------------------------------------------------------------- steps


def _timed(fn: Callable[[Step], None], step: Step) -> Step:
    t0 = time.perf_counter()
    try:
        fn(step)
    except Exception as exc:                      # a crashing step is a failed step, with its traceback kept
        import traceback

        step.status, step.detail = "fail", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}"
    step.seconds = time.perf_counter() - t0
    return step


def pytest_step(name: str, marker: str, run_dir: Path, root: Path) -> Step:
    def go(step: Step) -> None:
        xml = run_dir / f"{name}.xml"
        log = run_dir / f"{name}.log"
        cmd = [sys.executable, "-m", "pytest", "-q", "-m", marker, f"--junitxml={xml}", "-p", "no:cacheprovider"]
        with open(log, "w") as fh:
            proc = subprocess.run(cmd, cwd=root, stdout=fh, stderr=subprocess.STDOUT)
        counts = junit_counts(xml) if xml.exists() else {}
        ran = counts.get("tests", 0) - counts.get("skipped", 0)
        bad = counts.get("failures", 0) + counts.get("errors", 0)
        step.detail = (f"{ran - bad} passed, {bad} failed, {counts.get('skipped', 0)} skipped; log {log}"
                       if counts else f"pytest exit {proc.returncode}; log {log}")
        if bad or proc.returncode not in (0, 5):          # 5: no tests collected
            step.status = "fail"
        elif ran == 0:
            step.status = "skip"

    return _timed(go, Step(name))


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}


def selftest_step(run_dir: Path) -> Step:
    def go(step: Step) -> None:
        from pydvc.bench.smoke import run

        ok, report = run(run_dir / "selftest", ccpi=False)
        failed = [c["check"] for c in report.get("checks", []) if not c["pass"]]
        step.status = "pass" if ok else "fail"
        step.detail = f"{len(report.get('checks', []))} checks" + (f"; failed: {failed}" if failed else "")
        for backend, r in report.get("runs", {}).items():
            if "seconds" in r:
                step.metrics[f"selftest.{backend}.rmse"] = {"value": float(max(r["rmse"]))}

    return _timed(go, Step("selftest"))


def kernel_step(case_a: Path | None, cache: Path, run_dir: Path) -> Step:
    def go(step: Step) -> None:
        from pydvc.bench import kernel
        from pydvc.solver.engines import gpu_available

        backends = (["fused"] if gpu_available() else []) + ["cpu"]
        res = kernel.run(backends, case_a_dir=str(case_a) if case_a else None, cache=str(cache))
        (run_dir / "kernel.json").write_text(json.dumps(res, indent=1))
        tag = res["data"]
        for r in res["results"]:
            b = r["backend"]
            step.metrics[f"kernel.{tag}.{b}.sums_us"] = {"value": r["sums_us_per_point_iter"], "cv": r["sums_cv"]}
            step.metrics[f"kernel.{tag}.{b}.sample_us"] = {"value": r["sample_us_per_point"], "cv": r["sample_cv"]}
            if r["rel_err"] > KERNEL_REL_ERR_MAX:
                step.status = "fail"
                step.detail += f"{b}: sums differ from float64 by {r['rel_err']:.1e} (> {KERNEL_REL_ERR_MAX}). "
        step.detail += ", ".join(f"{r['backend']} {r['sums_us_per_point_iter']:.2f} µs/pt-iter" for r in res["results"])
        step.detail += f" [{tag}]"

    return _timed(go, Step("kernel benchmark"))


def casea_step(case_a: Path | None, cache: Path, run_dir: Path, baseline: dict[str, Any] | None,
               ccpi_disp: str | None = None, *, prefilter_sigma: float = 0.0) -> Step:
    """The central grid (CCPi's 4 680 points, wavefront, in memory) on fused and cpu; fused only when prefiltered."""

    def go(step: Step) -> None:
        from pydvc.bench.case_a import case_config
        from pydvc.bench.metrics import compare_arrays, load_results
        from pydvc.pipeline.inmemory import load_points, solve_in_memory
        from pydvc.solver.engines import gpu_available

        if case_a is None:
            step.status, step.detail = "skip", "no case A data (--case-a or $PYDVC_CASE_A)"
            return
        cfg = case_config(case_a, cache)
        tag = "casea" if not prefilter_sigma else f"casea_pf{prefilter_sigma:g}"
        if prefilter_sigma:
            cfg = dataclasses.replace(cfg, volumes=dataclasses.replace(cfg.volumes, prefilter_sigma=prefilter_sigma))
        pid, xyz = load_points(cfg)
        ccpi = _ccpi_reference(case_a, baseline, ccpi_disp)
        backends = (["fused"] if gpu_available() else []) + ([] if prefilter_sigma else ["cpu"])
        for b in backends:
            solve_in_memory(cfg, pid[:64], xyz[:64], backend=b, strategy="rigid")        # compile outside the timing
            t0 = time.perf_counter()
            r = solve_in_memory(cfg, pid, xyz, backend=b)
            step.metrics[f"{tag}.{b}.seconds"] = {"value": time.perf_counter() - t0}
            path = run_dir / f"{tag}_{b}.npz"
            np.savez(path, point_id=r.point_id, xyz=r.xyz, status=r.status, objmin=r.objmin,
                     displacement=r.params[:, :3].astype(np.float64), params=r.params, n_iter=r.n_iter)
            step.arrays[f"{tag}.{b}"] = str(path)
            ref = load_results(ccpi)
            common, ia, ib = np.intersect1d(r.point_id, ref["point_id"], return_indices=True)
            acc = compare_arrays(r.params[ia, :3], ref["displacement"][ib], r.status[ia], ref_status=ref["status"][ib])
            step.metrics[f"{tag}.{b}.ccpi.median"] = {"value": acc.median_abs}
            step.metrics[f"{tag}.{b}.ccpi.p95"] = {"value": acc.p95_abs}
            step.metrics[f"{tag}.{b}.ccpi.status_agreement"] = {"value": acc.status_agreement}
            step.detail += (f"{b}: {step.metrics[f'{tag}.{b}.seconds']['value']:.1f} s, vs CCPi ({len(common)} pts, "
                            f"{ccpi.name}) median {acc.median_abs:.4f}; ")
            if baseline and f"{tag}.{b}" in baseline.get("arrays", {}):
                verdict = compare_with_baseline_arrays(path, Path(baseline["arrays"][f"{tag}.{b}"]["path"]))
                step.detail += verdict.detail + "; "
                if verdict.status == "fail":
                    step.status = "fail"

    name = "case A central grid" + (f" (prefilter sigma {prefilter_sigma:g}, fused)" if prefilter_sigma else "")
    return _timed(go, Step(name))


def _ccpi_reference(case_a: Path, baseline: dict[str, Any] | None, explicit: str | None = None) -> Path:
    """CCPi's result for the central grid: ``--ccpi-disp``, else the one recorded in the baseline, else CCPi's 5-point reference."""
    if explicit:
        return Path(explicit)
    if baseline and baseline.get("ccpi_disp") and Path(baseline["ccpi_disp"]).exists():
        return Path(baseline["ccpi_disp"])
    return case_a / "completed_central_grid.disp"


def compare_with_baseline_arrays(path: Path, base_path: Path) -> Finding:
    if not base_path.exists():
        return Finding("arrays", "skip", f"baseline arrays {base_path} missing")
    a, b = np.load(path), np.load(base_path)
    agree = float((a["status"] == b["status"]).mean())
    good = (a["status"] == 0) & (b["status"] == 0)
    du = float(np.abs(a["displacement"][good] - b["displacement"][good]).max()) if good.any() else 0.0
    ok = agree >= ARRAY_STATUS_MIN and du <= ARRAY_DU_MAX
    return Finding("arrays", "pass" if ok else "fail", f"vs baseline: status {100 * agree:.2f} %, max |du| {du:.1e}")


def e2e_step(case_a: Path | None, cache: Path) -> Step:
    def go(step: Step) -> None:
        from pydvc.bench.case_a import case_config, grid_points
        from pydvc.pipeline.inmemory import solve_in_memory
        from pydvc.solver.engines import gpu_available

        if case_a is None or not gpu_available():
            step.status, step.detail = "skip", "needs case A data and a GPU"
            return
        cfg = case_config(case_a, cache)
        pid, xyz = grid_points(E2E_SPACING)
        t0 = time.perf_counter()
        r = solve_in_memory(cfg, pid, xyz, backend="fused")
        seconds = time.perf_counter() - t0
        step.metrics["e2e.casea_grid24.fused.seconds"] = {"value": seconds}
        step.detail = f"{len(pid)} points in {seconds:.1f} s, {100 * (r.status == 0).mean():.2f} % GOOD"

    return _timed(go, Step("case A 3D grid (fused)"))


# --------------------------------------------------------------------------- comparison


def perf_group(metric: str) -> str | None:
    if metric.endswith("_us"):
        return "kernel"
    if metric.endswith(".seconds"):
        return "e2e"
    return None


def compare(metrics: dict[str, dict[str, float]], baseline: dict[str, Any] | None, *, same_hardware: bool,
            gpu_busy: bool, cpu_busy: bool = False) -> list[Finding]:
    """Findings for every metric against the baseline (pure: unit-tested in tests/test_check.py)."""
    if not baseline:
        return [Finding("baseline", "skip", "no baseline for this machine; record one with --update-baseline")]
    found = []
    base = baseline.get("metrics", {})
    for name, cur in sorted(metrics.items()):
        if name not in base:
            found.append(Finding(name, "skip", "not in the baseline"))
            continue
        v, b = float(cur["value"]), float(base[name]["value"])
        group = perf_group(name)
        if group:
            if not same_hardware:
                found.append(Finding(name, "skip", "different hardware from the baseline"))
                continue
            if gpu_busy:
                found.append(Finding(name, "skip", "GPU in use by other processes"))
                continue
            if cpu_busy and ".cpu." in name:
                found.append(Finding(name, "skip", "CPU loaded by other processes"))
                continue
            warn, fail = PERF_LIMITS[group]
            noise = 3.0 * float(base[name].get("cv", 0.0))
            warn, fail = max(warn, noise), max(fail, noise * 2)
            change = v / b - 1.0 if b > 0 else 0.0
            status = "fail" if change > fail else "warn" if change > warn else "note" if change < -FASTER_NOTE else "pass"
            msg = f"{v:.4g} vs {b:.4g} ({100 * change:+.1f} %)"
            if status == "note":
                msg += "; faster: consider --update-baseline"
            found.append(Finding(name, status, msg))
            continue
        key = name.rsplit(".", 1)[-1]
        if ".ccpi." in name and key in CCPI_SLACK:
            slack = CCPI_SLACK[key]
            worse = v < b + slack if key == "status_agreement" else v > b + slack
            found.append(Finding(name, "fail" if worse else "pass", f"{v:.4g} vs baseline {b:.4g}"))
        else:
            found.append(Finding(name, "pass", f"{v:.4g} (baseline {b:.4g})"))
    return found


def overall(steps: list[Step], findings: list[Finding], *, require_case_a: bool) -> str:
    statuses = [s.status for s in steps] + [f.status for f in findings]
    if "fail" in statuses:
        return "FAIL"
    case_a_skipped = any(s.status == "skip" and s.name.startswith("case A") for s in steps)
    if case_a_skipped and require_case_a:
        return "FAIL"
    if "warn" in statuses:
        return "WARN"
    return "PASS (partial)" if case_a_skipped else "PASS"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bitwise_findings(steps: list[Step], baseline: dict[str, Any] | None, same_software: bool) -> list[Finding]:
    out = []
    for s in steps:
        for name, path in s.arrays.items():
            ref = (baseline or {}).get("arrays", {}).get(name)
            if not ref:
                continue
            same = sha256(Path(path)) == ref["sha256"]
            if same:
                out.append(Finding(f"{name} bitwise", "pass", "bit-identical to the baseline"))
            elif same_software:
                out.append(Finding(f"{name} bitwise", "warn", "same software as the baseline but results differ in the last bits"))
            else:
                out.append(Finding(f"{name} bitwise", "note", "software differs from the baseline; within tolerance"))
    return out


# --------------------------------------------------------------------------- baseline


def baseline_path(root: Path, machine: str) -> Path:
    return baseline_dir(root) / f"{machine}.json"


def write_baseline(root: Path, machine: str, report: dict[str, Any], steps: list[Step], reason: str,
                   ccpi_disp: str | None, out: Path | None = None) -> Path:
    """Baseline JSON in :func:`baseline_dir`; its arrays under ``out``/runs/check/baseline, paths relative to ``out``."""
    out = out or root
    store = out / "runs" / "check" / "baseline"
    store.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for s in steps:
        for name, path in s.arrays.items():
            dest = store / Path(path).name
            shutil.copyfile(path, dest)
            arrays[name] = {"path": str(dest.relative_to(out)), "sha256": sha256(dest)}
    metrics = {k: v for s in steps for k, v in s.metrics.items()}
    doc = {"machine": machine, "recorded": report["started"], "reason": reason, "git": report["git"],
           "tier": report["tier"], "fingerprint": report["fingerprint"], "metrics": metrics, "arrays": arrays,
           "ccpi_disp": ccpi_disp}
    path = baseline_path(root, machine)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


def load_baseline(root: Path, machine: str, out: Path | None = None) -> dict[str, Any] | None:
    out = out or root
    path = baseline_path(root, machine)
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    for a in doc.get("arrays", {}).values():             # stored relative to the output root
        a["path"] = str(out / a["path"])
    if doc.get("ccpi_disp") and not Path(doc["ccpi_disp"]).is_absolute():
        doc["ccpi_disp"] = str(out / doc["ccpi_disp"])
    return doc


# --------------------------------------------------------------------------- driver


def run(tier: str, *, case_a: str | None = None, cache: str | None = None, machine: str | None = None,
        update_baseline: bool = False, reason: str = "", require_case_a: bool = False,
        ccpi_disp: str | None = None) -> tuple[str, dict[str, Any]]:
    from pydvc.bench import case_a as ca

    root = repo_root()
    out = out_root(root)
    machine = machine or os.environ.get("PYDVC_MACHINE") or socket.gethostname()
    git = git_info(root)
    started = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out / "runs" / "check" / f"{started}-{git['sha'][:7]}{'-dirty' if git['dirty'] else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = out / "runs" / "check" / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    data = ca.data_dir(case_a)
    cache_dir = ca.cache_dir(cache)
    baseline = load_baseline(root, machine, out)
    pre, state = preflight(tier)
    fp = fingerprint()
    report: dict[str, Any] = {"tier": tier, "machine": machine, "started": started, "git": git, "fingerprint": fp,
                              "gpu": state, "case_a": str(data) if data else None}
    say = lambda msg: print(msg, flush=True)                                            # noqa: E731
    cores = os.cpu_count() or 1
    load_before = os.getloadavg()[0]                     # other work only: nothing of ours has started

    steps: list[Step] = []
    if not any(f.status == "fail" for f in pre):
        plan: list[Callable[[], Step]] = [lambda: pytest_step("tests (cpu, emulator)", "not slow and not gpu", run_dir, root)]
        if tier in ("gpu", "full"):
            plan += [lambda: pytest_step("tests (gpu)", "gpu", run_dir, root), lambda: selftest_step(run_dir)]
        if tier == "full":
            plan += [lambda: pytest_step("tests (slow)", "slow", run_dir, root),
                     lambda: kernel_step(data, cache_dir, run_dir),
                     lambda: casea_step(data, cache_dir, run_dir, baseline, ccpi_disp),
                     lambda: casea_step(data, cache_dir, run_dir, baseline, ccpi_disp, prefilter_sigma=1.0),
                     lambda: e2e_step(data, cache_dir)]
        for make in plan:
            step = make()
            say(f"[{step.status.upper():4s}] {step.name} ({step.seconds:.0f} s): {step.detail.splitlines()[0] if step.detail else ''}")
            steps.append(step)

    load_after = os.getloadavg()[0]                      # ours is mostly GPU work by now; a heavy load is someone else
    cpu_busy = max(load_before, load_after) > CPU_BUSY_LOAD * cores
    report["cpu_load"] = {"before": load_before, "after": load_after, "cores": cores, "busy": cpu_busy}
    if cpu_busy:
        pre.append(Finding("CPU idle", "warn", f"load average {max(load_before, load_after):.0f} on {cores} cores: "
                                               "CPU timings not compared"))
    metrics = {k: v for s in steps for k, v in s.metrics.items()}
    same_hw = bool(baseline) and baseline["fingerprint"]["hardware"] == fp["hardware"]
    same_sw = bool(baseline) and baseline["fingerprint"]["software"] == fp["software"]
    findings = pre + compare(metrics, baseline, same_hardware=same_hw, gpu_busy=bool(state["other_processes"]),
                             cpu_busy=cpu_busy)
    findings += bitwise_findings(steps, baseline, same_sw)
    result = overall(steps, findings, require_case_a=require_case_a)
    report |= {"steps": [dataclasses.asdict(s) for s in steps], "findings": [dataclasses.asdict(f) for f in findings],
               "result": result}
    for f in findings:
        if f.status in ("fail", "warn", "note"):
            say(f"  {f.status.upper():4s} {f.metric}: {f.detail}")
    if update_baseline:
        if result == "FAIL":
            say("not updating the baseline: the check failed")
        elif state["other_processes"] or cpu_busy:
            say("not updating the baseline: other processes were using the GPU or CPU, so timings are not representative")
        elif not reason:
            say("not updating the baseline: give --reason")
        else:
            path = write_baseline(root, machine, report, steps, reason, ccpi_disp, out)
            report["baseline_written"] = str(path)
            say(f"baseline written: {path}")
    (run_dir / "check.json").write_text(json.dumps(report, indent=1, default=str))
    say(f"{result}  ({run_dir / 'check.json'})")
    return result, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pydvc check", description=__doc__.split("\n\n")[0])
    add_arguments(p)
    args = p.parse_args(argv)
    return cmd(args)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("tier", choices=TIERS, nargs="?", default="quick")
    p.add_argument("--case-a", help="case A data directory (default $PYDVC_CASE_A)")
    p.add_argument("--cache", help="directory for C-ordered .raw copies (default $PYDVC_CASE_A_CACHE or runs/case_A)")
    p.add_argument("--machine", help="baseline name (default $PYDVC_MACHINE or the host name)")
    p.add_argument("--require-case-a", action="store_true", help="fail when the case A data is missing")
    p.add_argument("--update-baseline", action="store_true", help="record this run as the machine's baseline")
    p.add_argument("--reason", default="", help="why the baseline changes (required with --update-baseline)")
    p.add_argument("--ccpi-disp", help="CCPi's .disp for the central grid, recorded in the baseline")


def cmd(args: argparse.Namespace) -> int:
    result, _ = run(args.tier, case_a=args.case_a, cache=args.cache, machine=args.machine,
                    update_baseline=args.update_baseline, reason=args.reason, require_case_a=args.require_case_a,
                    ccpi_disp=args.ccpi_disp)
    return 1 if result == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
