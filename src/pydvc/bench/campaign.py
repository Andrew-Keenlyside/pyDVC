"""The cluster benchmark campaign: one command that measures how pyDVC scales on a GPU node.

Steps, in order (each writes ``<out>/<step>.json``; a finished step is skipped
when the job is resubmitted, so a job that hits its walltime just continues):

=========  =====================================================================
env        environment, GPUs (``nvidia-smi -q``), CPU and memory
check      ``pydvc check gpu``: the test suite and selftest on this node
kernel     per-GPU kernel speed (``bench.kernel``): case A if given, synthetic u8, u16
data       case L: a synthetic u16 pair with a known inclusion field (generated
           once, reused), tiled so every GPU count gets >= 4 tiles per GPU
storage    file-system bandwidth (fio, if present) and pyDVC's brick-read ceiling
strong     strong scaling over the device counts, with per-run timelines
weak       weak scaling (4 tiles per device, the fullest tiles)
profile    Nsight Systems on a short run over all devices (``pydvc run --max-tiles``)
ncu        Nsight Compute counters for gn_sums on one GPU (skipped if not permitted)
e2e        plan, run and finalize on all devices, and accuracy against the truth
report     ``campaign.md``: the tables above in one page
=========  =====================================================================

Command line (inside the container, or any environment with pyDVC)::

    python -m pydvc.bench.campaign --out RESULTS_DIR --data-dir DATA_DIR [--size 2048] [--devices 1 2 4 8]
    python -m pydvc.bench.campaign ... --steps strong weak     # a subset
    python -m pydvc.bench.campaign ... --redo strong           # run a finished step again
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

STEPS = ("env", "check", "kernel", "data", "storage", "strong", "weak", "profile", "ncu", "e2e", "report")


@dataclasses.dataclass
class Campaign:
    out: Path
    data_dir: Path
    size: int = 2048
    devices: tuple[int, ...] = (1, 2, 4, 8)
    case_a: Path | None = None
    tile: int | None = None               # tile edge; default: >= 4 tiles per GPU at the largest count
    profile_tiles: int = 16
    fio_gb: int = 4                       # per fio job (8 jobs)
    log: Callable[[str], None] = print

    @property
    def case_l(self) -> Path:
        return self.data_dir / f"caseL{self.size}"

    @property
    def config(self) -> Path:
        return self.case_l / "config.yaml"

    def tile_edge(self) -> int:
        """The largest power-of-two tile giving >= 4 tiles per GPU at the largest device count."""
        if self.tile:
            return self.tile
        need = 4 * max(self.devices)
        edge = self.size
        while edge > 128 and (self.size // edge) ** 3 < need:
            edge //= 2
        return edge


def _run(cmd: list[str], log_path: Path, env: dict[str, str] | None = None, check: bool = True) -> int:
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}); see {log_path}")
    return proc.returncode


def _py(*args: str) -> list[str]:
    return [sys.executable, "-m", *args]


# --------------------------------------------------------------------------- steps


def step_env(c: Campaign) -> dict[str, Any]:
    from pydvc.bench.smoke import environment

    info: dict[str, Any] = {"environment": environment(), "host": os.uname().nodename,
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "job_id": os.environ.get("JOB_ID"), "nslots": os.environ.get("NSLOTS")}
    for name, cmd in (("nvidia_smi", ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,pcie.link.gen.max,"
                                      "pcie.link.width.max,clocks.max.sm,power.limit", "--format=csv"]),
                      ("topology", ["nvidia-smi", "topo", "-m"]), ("lscpu", ["lscpu"]), ("free", ["free", "-g"])):
        try:
            info[name] = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.TimeoutExpired):
            info[name] = None
    return info


def step_check(c: Campaign) -> dict[str, Any]:
    env = {"PYDVC_CHECK_OUT": str(c.out / "check_runs")}
    code = _run([sys.executable, "-m", "pydvc.cli", "check", "gpu"], c.out / "check.log", env=env, check=False)
    latest = c.out / "check_runs" / "runs" / "check" / "latest" / "check.json"
    report = json.loads(latest.read_text()) if latest.exists() else {}
    if code != 0:
        raise RuntimeError(f"pydvc check gpu failed on this node; see {c.out / 'check.log'}")
    return {"result": report.get("result"), "steps": [(s["name"], s["status"]) for s in report.get("steps", [])]}


def step_kernel(c: Campaign) -> dict[str, Any]:
    from pydvc.bench import kernel

    out = {}
    if c.case_a:
        out["case_a"] = kernel.run(["fused"], case_a_dir=str(c.case_a), cache=str(c.data_dir / "case_A_cache"))
    for dt in ("uint8", "uint16"):
        out[f"synthetic_{dt}"] = kernel.run(["fused"], synthetic_dtype=dt)
    return {k: v["results"][0] for k, v in out.items()} | {"environment": next(iter(out.values()))["environment"]}


def step_data(c: Campaign) -> dict[str, Any]:
    from pydvc.config import ClusterSpec, RunConfig, SeedingSpec

    t0 = time.perf_counter()
    if not c.config.exists():
        c.log(f"generating case L {c.size}^3 in {c.case_l} (u16, inclusion field)")
        _run([sys.executable, "-m", "pydvc.cli", "synth", "--shape", *[str(c.size)] * 3, "--field", "inclusion",
              "--spacing", "16", "--out", str(c.case_l), "--subvol-size", "48", "--subvol-npts", "4096", "--dof", "6",
              "--disp-max", "10", "--chunk", "128", "--shard", str(min(512, c.size))], c.out / "data.log")
    edge = c.tile_edge()
    cfg = RunConfig.from_yaml(c.config)
    cfg = dataclasses.replace(cfg, cluster=ClusterSpec(tile_shape=(edge, edge, edge), prefetch_depth=2),
                              seeding=SeedingSpec(strategy="rigid"))
    cfg.to_yaml(c.config)
    return {"case_l": str(c.case_l), "size": c.size, "tile_edge": edge, "tiles_estimate": (c.size // edge) ** 3,
            "generate_s": time.perf_counter() - t0}


def step_storage(c: Campaign) -> dict[str, Any]:
    from pydvc.bench.scaling import read_bandwidth
    from pydvc.config import RunConfig
    from pydvc.pipeline import coordinator

    res: dict[str, Any] = {}
    fio = shutil.which("fio")
    if fio:
        d = c.case_l / ".fio"
        d.mkdir(exist_ok=True)
        try:
            out = subprocess.run([fio, "--name=seqread", f"--directory={d}", "--rw=read", "--bs=4M", f"--size={c.fio_gb}G",
                                  "--numjobs=8", "--ioengine=libaio", "--direct=1", "--group_reporting",
                                  "--output-format=json"], capture_output=True, text=True, timeout=3600)
            job = json.loads(out.stdout)["jobs"][0]["read"]
            res["fio_seq_read_gb_per_s"] = job["bw_bytes"] / 1e9
        except Exception as exc:                      # fio unusable here: record why, carry on
            res["fio_error"] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            shutil.rmtree(d, ignore_errors=True)
    cfg = RunConfig.from_yaml(c.config)
    work = dataclasses.replace(cfg, output=str(c.out / "storage" / "results.zarrvectors"), workdir=str(c.out / "storage" / "work"))
    shutil.rmtree(c.out / "storage", ignore_errors=True)
    coordinator.prepare(work, backend="fused")
    for readers in (8, 32):
        res[f"pydvc_read_{readers}"] = read_bandwidth(work, readers=readers)
    return res


def _scaling(c: Campaign, mode: str) -> dict[str, Any]:
    from pydvc.bench.scaling import scaling
    from pydvc.config import RunConfig

    return scaling(RunConfig.from_yaml(c.config), list(c.devices), backend="fused", out_dir=c.out / f"scaling_{mode}",
                   mode=mode, progress=c.log)


def step_strong(c: Campaign) -> dict[str, Any]:
    return _scaling(c, "strong")


def step_weak(c: Campaign) -> dict[str, Any]:
    return _scaling(c, "weak")


def _fresh_config(c: Campaign, name: str) -> Path:
    from pydvc.config import RunConfig

    d = c.out / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    cfg = dataclasses.replace(RunConfig.from_yaml(c.config), output=str(d / "results.zarrvectors"), workdir=str(d / "work"))
    path = d / "config.yaml"
    cfg.to_yaml(path)
    return path


def step_profile(c: Campaign) -> dict[str, Any]:
    nsys = shutil.which("nsys")
    if nsys is None:
        return {"skipped": "nsys not found"}
    cfg = _fresh_config(c, "profile")
    _run([sys.executable, "-m", "pydvc.cli", "plan", str(cfg), "--backend", "fused"], c.out / "profile" / "plan.log")
    rep = c.out / "profile" / "run"
    _run([nsys, "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none", "-o", str(rep), "-f", "true",
          sys.executable, "-m", "pydvc.cli", "run", str(cfg), "--backend", "fused", "--max-tiles", str(c.profile_tiles)],
         c.out / "profile" / "nsys.log")
    stats = subprocess.run([nsys, "stats", "-q", "--report", "cuda_gpu_kern_sum,nvtx_sum", "--format", "csv",
                            f"{rep}.nsys-rep"], capture_output=True, text=True).stdout
    (c.out / "profile" / "stats.csv").write_text(stats)
    work = json.loads((c.out / "profile" / "work" / "run_stats.json").read_text())
    from pydvc.bench.timeline import summarise

    return {"report": f"{rep}.nsys-rep", "stats_csv": str(c.out / "profile" / "stats.csv"), "tiles": c.profile_tiles,
            "timeline": summarise(work["events"])}


def step_ncu(c: Campaign) -> dict[str, Any]:
    ncu = shutil.which("ncu")
    if ncu is None:
        return {"skipped": "ncu not found"}
    rep = c.out / "ncu_gn_sums"
    code = _run([ncu, "--set", "full", "--kernel-name", "regex:gn_sums", "--launch-skip", "2", "--launch-count", "1",
                 "-o", str(rep), "-f", sys.executable, "-m", "pydvc.bench.kernel", "--backends", "fused",
                 "--synthetic", "uint16", "--points", "1024", "--repeats", "1"], c.out / "ncu.log", check=False)
    log = (c.out / "ncu.log").read_text()
    if "ERR_NVGPUCTRPERM" in log or code != 0:
        return {"skipped": "hardware counters not permitted (ERR_NVGPUCTRPERM) or ncu failed", "log": str(c.out / "ncu.log")}
    return {"report": f"{rep}.ncu-rep"}


def step_e2e(c: Campaign) -> dict[str, Any]:
    from pydvc.bench.metrics import against_truth
    from pydvc.bench.timeline import summarise
    from pydvc.config import RunConfig

    cfg = _fresh_config(c, "e2e")
    times = {}
    for stage in ("plan", "seed", "run", "finalize"):
        t0 = time.perf_counter()
        _run([sys.executable, "-m", "pydvc.cli", stage, str(cfg), *(["--backend", "fused"] if stage != "finalize" else [])],
             c.out / "e2e" / f"{stage}.log")
        times[stage] = time.perf_counter() - t0
    run_cfg = RunConfig.from_yaml(cfg)
    acc = against_truth(run_cfg.output, c.case_l / "truth.npz")
    work = json.loads((Path(run_cfg.workdir) / "run_stats.json").read_text())
    return {"stages_s": times, "total_s": sum(times.values()), "accuracy": dataclasses.asdict(acc),
            "timeline": summarise(work["events"])}


def step_report(c: Campaign) -> dict[str, Any]:
    text = report_markdown(c.out)
    (c.out / "campaign.md").write_text(text)
    return {"report": str(c.out / "campaign.md")}


def report_markdown(out: Path) -> str:
    def load(name: str) -> dict[str, Any] | None:
        p = out / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    lines = ["# pyDVC scaling campaign", ""]
    env = load("env")
    if env:
        gpus = env["environment"].get("gpus", [])
        lines += [f"Host {env['host']}, job {env.get('job_id')}, {len(gpus)} GPU(s): {', '.join(gpus[:1])}"
                  f"{' ...' if len(gpus) > 1 else ''}. pyDVC {env['environment']['packages'].get('pydvc')}.", ""]
    k = load("kernel")
    if k:
        lines += ["## Kernel speed (one GPU)", "", "| data | gn_sums µs / point-iteration | sample µs / point | error vs float64 |",
                  "|---|---|---|---|"]
        for name, r in k.items():
            if isinstance(r, dict) and "sums_us_per_point_iter" in r:
                lines.append(f"| {name} | {r['sums_us_per_point_iter']:.2f} | {r['sample_us_per_point']:.2f} | {r['rel_err']:.1e} |")
        lines.append("")
    st = load("storage")
    if st:
        fio = st.get("fio_seq_read_gb_per_s")
        lines += ["## Storage", "", f"fio sequential read (8 jobs): {f'{fio:.2f} GB/s' if fio else st.get('fio_error', 'not run')}. "
                  + ", ".join(f"pyDVC brick reads, {k.split('_')[-1]} threads: {v['gb_per_s']:.2f} GB/s"
                              for k, v in st.items() if k.startswith("pydvc_read")), ""]
    for mode in ("strong", "weak"):
        s = load(mode)
        if not s:
            continue
        lines += [f"## {mode.capitalize()} scaling", "", "| GPUs | tiles | time s | points/s | efficiency | utilisation | "
                  "imbalance | mean end idle s | I/O wait |", "|---|---|---|---|---|---|---|---|---|"]
        for r in s["rows"]:
            lines.append(f"| {r['devices']} | {r['tiles']} | {r['seconds']:.1f} | {r['points_per_second']:.0f} | "
                         f"{100 * r['efficiency']:.0f} % | {100 * (r['utilisation'] or 0):.0f} % | {r['load_imbalance'] or 0:.2f} | "
                         f"{r['tail_idle_s_mean'] or 0:.1f} | {100 * (r['io_wait_fraction'] or 0):.1f} % |")
        lines += [f"Warning: {w}" for w in s.get("warnings", [])] + [""]
    e = load("e2e")
    if e:
        a = e["accuracy"]
        lines += ["## End to end (all GPUs)", "", "Stages: " + ", ".join(f"{k} {v:.1f} s" for k, v in e["stages_s"].items())
                  + f"; total {e['total_s']:.1f} s. Accuracy against the truth: {100 * a['frac_good']:.2f} % GOOD, "
                  f"RMSE {max(a['rmse']):.4f} voxel.", ""]
    for name in ("profile", "ncu"):
        p = load(name)
        if p:
            lines.append(f"{name}: {p.get('report') or p.get('skipped')}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- driver


def run(c: Campaign, steps: list[str] | None = None, redo: list[str] | None = None) -> dict[str, str]:
    c.out.mkdir(parents=True, exist_ok=True)
    c.data_dir.mkdir(parents=True, exist_ok=True)
    fns = {name: globals()[f"step_{name}"] for name in STEPS}
    status = {}
    for name in steps or list(STEPS):
        dest = c.out / f"{name}.json"
        if dest.exists() and name not in (redo or []) and name != "report":
            c.log(f"[skip] {name}: done ({dest})")
            status[name] = "done"
            continue
        c.log(f"[run ] {name}  {time.strftime('%H:%M:%S')}")
        t0 = time.perf_counter()
        try:
            res = fns[name](c)
        except Exception as exc:
            import traceback

            (c.out / f"{name}.error.txt").write_text(traceback.format_exc())
            c.log(f"[FAIL] {name}: {exc}")
            status[name] = "failed"
            if name in ("env", "check", "data"):          # nothing after these can be trusted
                break
            continue
        res = {"step": name, "seconds": time.perf_counter() - t0, **res}
        dest.write_text(json.dumps(res, indent=1, default=str))
        c.log(f"[ ok ] {name} ({res['seconds']:.0f} s)")
        status[name] = "ok"
    return status


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m pydvc.bench.campaign", description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="results directory (shared storage)")
    p.add_argument("--data-dir", required=True, help="where case L is generated and kept (shared storage)")
    p.add_argument("--size", type=int, default=2048, help="case L edge (2048: ~16 GB per volume; 4096: ~137 GB)")
    p.add_argument("--devices", type=int, nargs="+", help="device counts to scale over (default 1 2 4 ... up to visible)")
    p.add_argument("--case-a", help="case A data directory, for the kernel step on real data")
    p.add_argument("--tile", type=int, help="tile edge (default: >= 4 tiles per GPU at the largest count)")
    p.add_argument("--profile-tiles", type=int, default=16)
    p.add_argument("--fio-gb", type=int, default=4, help="size of each of fio's 8 jobs")
    p.add_argument("--steps", nargs="+", choices=STEPS)
    p.add_argument("--redo", nargs="+", choices=STEPS, default=[])
    args = p.parse_args(argv)
    if args.devices:
        counts = tuple(args.devices)
    else:
        from pydvc.pipeline.launch import node_info

        n = len(node_info().local_devices) or 1
        counts = tuple(k for k in (1, 2, 4, 8, 16) if k <= n) + ((n,) if n not in (1, 2, 4, 8, 16) else ())
    c = Campaign(out=Path(args.out), data_dir=Path(args.data_dir), size=args.size, devices=counts,
                 case_a=Path(args.case_a) if args.case_a else None, tile=args.tile, profile_tiles=args.profile_tiles,
                 fio_gb=args.fio_gb, log=lambda m: print(m, flush=True))
    status = run(c, args.steps, args.redo)
    print(json.dumps(status))
    return 1 if "failed" in status.values() else 0


if __name__ == "__main__":
    sys.exit(main())
