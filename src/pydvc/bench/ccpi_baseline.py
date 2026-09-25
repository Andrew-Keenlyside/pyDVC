"""Run the CCPi ``dvc`` executable (conda package ``ccpi-dvc``) on a pyDVC case.

Writes ``dvc_in`` + ``.roi`` equivalent to the pyDVC config
(:func:`pydvc.io.ccpi.write_dvc_input`), converts volumes to ``.raw`` if
needed, runs ``dvc`` with ``OMP_NUM_THREADS`` set, and measures points per
second from wall-clock time (the ``.stat`` figure is kept alongside). For
clouds too large for CCPi's O(N^2) neighbour sort, it runs a random sample of
points and extrapolates, recording that it did so.

It also sweeps ``OMP_NUM_THREADS`` (1, 2, 4, ..., cores) to record CCPi's
intra-point scaling, and runs ``P`` concurrent processes on disjoint point
subsets to get the best throughput achievable from one CPU node without
code changes. That is the fair "CCPi as shipped" baseline.

The executable is found from ``$PYDVC_CCPI_DVC`` or ``dvc`` on ``PATH``.
**Use ccpi-dvc 22.0.0** (``conda install -c ccpi ccpi-dvc=22.0.0``). The
25.0.0 conda build (h2bc3f7f_0) has a broken tricubic path: seeded at the
exact fractional displacement of a pure translation it walks 1-2 voxels away,
while trilinear, and tricubic in 22.0.0 and 21.2.0, stay within 0.002 voxel
(docs/benchmarks/2026-09-25-M0-M1-case-S.md).
Command line::

    python -m pydvc.bench.ccpi_baseline CONFIG --workdir DIR [--threads 1 2 4] [--processes 1 2] [--max-points N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pydvc.config import RunConfig
from pydvc.io import ccpi
from pydvc.io.pointcloud import read_roi


@dataclass
class BaselineResult:
    points: int
    seconds: float
    points_per_second: float
    omp_threads: int
    processes: int
    sampled: bool                 # True if extrapolated from a subset
    disp_path: Path
    stat_points_per_second: float | None = None    # CCPi's own figure (process 0), for reference


def find_dvc(exe: str | Path | None = None) -> Path:
    candidate = exe or os.environ.get("PYDVC_CCPI_DVC") or shutil.which("dvc")
    if not candidate or not Path(candidate).exists():
        raise FileNotFoundError("CCPi dvc executable not found: install conda package ccpi-dvc or set PYDVC_CCPI_DVC")
    return Path(candidate)


def ccpi_ready_config(cfg: RunConfig, workdir: Path) -> RunConfig:
    """``cfg`` with volumes CCPi can read: OME-Zarr is exported once to ``workdir/{ref,def}.raw``."""
    from pydvc.io.volume import is_zarr_uri, open_volume, write_raw

    spec = cfg.volumes
    if not is_zarr_uri(spec.reference):
        return cfg
    paths = {}
    for which, name in (("reference", "ref.raw"), ("deformed", "def.raw")):
        src = open_volume(spec, which)
        dst = workdir / name
        if not dst.exists() or dst.stat().st_size != int(np.prod(src.shape)) * src.dtype.itemsize:
            write_raw(src, dst)
        paths[which] = str(dst)
        shape, dtype = src.shape, src.dtype
    volumes = dataclasses.replace(
        spec,
        reference=paths["reference"],
        deformed=paths["deformed"],
        raw_shape_xyz=(shape[2], shape[1], shape[0]),
        raw_dtype=dtype.newbyteorder("<").str,
        raw_header_bytes=0,
    )
    return dataclasses.replace(cfg, volumes=volumes)


def _subsets(point_id: np.ndarray, xyz: np.ndarray, processes: int) -> list[np.ndarray]:
    """Split into ``processes`` spatially contiguous z-slabs of equal size, each in input order."""
    if processes == 1:
        return [np.arange(len(xyz))]
    order = np.lexsort((xyz[:, 0], xyz[:, 1], xyz[:, 2]))
    return [np.sort(part) for part in np.array_split(order, processes)]


def run_ccpi(
    cfg: RunConfig,
    *,
    workdir: str | Path,
    omp_threads: int,
    processes: int = 1,
    max_points: int | None = None,
    seed: int = 0,
    exe: str | Path | None = None,
) -> BaselineResult:
    dvc = find_dvc(exe)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = ccpi_ready_config(cfg, workdir)
    point_id, xyz = read_roi(cfg.points)
    sampled = False
    if max_points is not None and len(xyz) > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(np.concatenate([[0], 1 + rng.choice(len(xyz) - 1, max_points - 1, replace=False)]))
        point_id, xyz, sampled = point_id[keep], xyz[keep], True

    tag = f"t{omp_threads}_p{processes}" + (f"_n{len(xyz)}" if sampled else "")
    runs = []
    for k, sel in enumerate(_subsets(point_id, xyz, processes)):
        base = workdir / f"{tag}_{k}"
        roi = base.with_suffix(".roi")
        ccpi.write_roi(roi, point_id[sel], xyz[sel])
        dvc_in = workdir / f"{tag}_{k}_dvc_in.txt"
        # each subset starts from its own first point, as CCPi does
        sub_cfg = dataclasses.replace(cfg, seeding=dataclasses.replace(cfg.seeding, start_point=None))
        ccpi.write_dvc_input(sub_cfg, dvc_in, roi_path=roi, output_base=base)
        runs.append((base, dvc_in))

    env = dict(os.environ, OMP_NUM_THREADS=str(omp_threads))
    t0 = time.perf_counter()
    procs = []
    for base, dvc_in in runs:
        log = open(base.with_suffix(".log"), "w")
        procs.append((subprocess.Popen([str(dvc), str(dvc_in)], cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    for proc, log in procs:
        proc.wait()
        log.close()
    seconds = time.perf_counter() - t0
    for (base, _), (proc, _) in zip(runs, procs):
        # dvc exits 0 after input errors, so a missing .disp is the failure signal
        if proc.returncode != 0 or not base.with_suffix(".disp").exists():
            log = base.with_suffix(".log")
            raise RuntimeError(f"dvc failed (exit {proc.returncode}): {log.read_text()[-2000:]}")

    merged = workdir / f"{tag}.disp"
    if processes == 1:
        shutil.copyfile(runs[0][0].with_suffix(".disp"), merged)
    else:
        parts = [ccpi.read_disp(base.with_suffix(".disp")) for base, _ in runs]
        rows = np.concatenate(parts)
        rows = rows[np.argsort(rows["n"], kind="stable")]
        rows = rows[np.searchsorted(rows["n"], point_id)]            # back to input order
        ccpi.write_disp(
            merged, rows["n"], np.stack([rows["x"], rows["y"], rows["z"]], 1), rows["status"], rows["objmin"],
            np.stack([rows["u"], rows["v"], rows["w"]], 1),
        )
    try:
        stat_rate = ccpi.read_stat_throughput(runs[0][0].with_suffix(".stat"))
    except (OSError, ValueError):
        stat_rate = None
    return BaselineResult(
        points=len(xyz),
        seconds=seconds,
        points_per_second=len(xyz) / seconds,
        omp_threads=omp_threads,
        processes=processes,
        sampled=sampled,
        disp_path=merged,
        stat_points_per_second=stat_rate,
    )


def thread_counts(cores: int | None = None) -> list[int]:
    cores = cores or os.cpu_count() or 1
    counts = [1 << k for k in range(cores.bit_length()) if (1 << k) <= cores]
    return counts + ([cores] if counts[-1] != cores else [])


def thread_sweep(cfg: RunConfig, *, workdir: str | Path, max_points: int = 200) -> list[BaselineResult]:
    return [run_ccpi(cfg, workdir=workdir, omp_threads=t, max_points=max_points) for t in thread_counts()]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pydvc.bench.ccpi_baseline", description=__doc__.split("\n\n")[0])
    p.add_argument("config")
    p.add_argument("--workdir", required=True)
    p.add_argument("--threads", type=int, nargs="+", default=None, help="OMP_NUM_THREADS values (default: 1, 2, 4, ..., cores)")
    p.add_argument("--processes", type=int, nargs="+", default=[1], help="concurrent processes (each with the thread count)")
    p.add_argument("--max-points", type=int, default=None)
    p.add_argument("--dof", type=int, choices=[3, 6, 12], default=None, help="override search.dof")
    p.add_argument("--rigid-trans", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="override search.rigid_trans (CCPi needs it within ~1 voxel at the start point)")
    p.add_argument("--exe", default=None)
    args = p.parse_args(argv)

    cfg = RunConfig.from_yaml(args.config)
    if args.dof:
        cfg = dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, dof=args.dof))
    if args.rigid_trans:
        cfg = dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, rigid_trans=tuple(args.rigid_trans)))
    results = []
    for procs in args.processes:
        for threads in args.threads or thread_counts():
            r = run_ccpi(cfg, workdir=args.workdir, omp_threads=threads, processes=procs, max_points=args.max_points, exe=args.exe)
            results.append(r)
            print(
                f"threads {threads:>2}  processes {procs:>2}  points {r.points:>6}  {r.seconds:8.1f} s  "
                f"{r.points_per_second:8.2f} pt/s" + ("  (sampled)" if r.sampled else "")
            )
    out = Path(args.workdir) / "baseline.json"
    out.write_text(json.dumps([dict(dataclasses.asdict(r), disp_path=str(r.disp_path)) for r in results], indent=2))


if __name__ == "__main__":
    main()
