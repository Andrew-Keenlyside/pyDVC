"""Q3: 1 -> N device scaling on one node, and the storage bandwidth that bounds it.

``scaling``
    Runs the tile loop on 1, 2, ..., N devices against fresh results stores
    (plan once, then ``run`` per device count) and reports pt/s and the
    efficiency ``t1 / (n tn)``. Q3 passes at >= 85 % on a compute-bound case.
``read_bandwidth``
    Reads every tile's bricks (both volumes) with ``readers`` threads and no
    compute, through the same path the workers use: pyDVC's own I/O ceiling on
    this node and file system. A run whose ``bytes_read / seconds`` reaches
    >= 70 % of it is I/O-bound in the Q3 sense. ``scripts/slurm/storage_baseline.sh``
    measures the raw file system with fio for comparison.

Command line::

    python -m pydvc.bench.scaling CONFIG --devices 1 2 4 8 [--mode strong|weak] [--backend fused] [--out runs/scaling]
    python -m pydvc.bench.scaling CONFIG --read-bandwidth --readers 8
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydvc.config import RunConfig


def read_bandwidth(cfg: RunConfig, *, readers: int = 8, max_tiles: int | None = None) -> dict[str, Any]:
    """GB/s reading the planned bricks (``pydvc plan`` first) with ``readers`` threads, host decode only."""
    from pydvc.io.volume import open_volume
    from pydvc.pipeline.tiling import load_plan

    tiles, _ = load_plan(Path(cfg.workdir) / "plan.json")
    tiles = tiles[:max_tiles] if max_tiles else tiles
    vols = {w: open_volume(cfg.volumes, w) for w in ("reference", "deformed")}
    jobs = [(w, t.ref_box if w == "reference" else t.def_box) for t in tiles for w in vols]

    def read(job: tuple[str, Any]) -> int:
        which, box = job
        return int(vols[which].read_brick(box, device="cpu").data.nbytes)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(readers) as pool:
        total = sum(pool.map(read, jobs))
    seconds = time.perf_counter() - t0
    return {"bytes": total, "seconds": seconds, "gb_per_s": total / seconds / 1e9, "readers": readers, "tiles": len(tiles)}


MIN_TILES_PER_DEVICE = 4          # below this, the end-of-queue tail dominates a strong-scaling run


def scaling(cfg: RunConfig, device_counts: list[int], *, backend: str, out_dir: str | Path,
            devices: tuple[int, ...] | None = None, mode: str = "strong", tiles_per_device: int = 4,
            progress=print) -> dict[str, Any]:
    """Fresh store per device count: pt/s and efficiency, with a timeline summary of each run.

    ``strong``: the whole plan on each device count; efficiency ``t1 / (n tn)``.
    ``weak``: ``tiles_per_device x n`` tiles on n devices, the fullest tiles first so
    every device gets similar work; efficiency ``t1 / tn`` (ideal: constant time).
    Seeding must be ``rigid`` (or seeds must already exist): no seed pass runs here.
    """
    from pydvc.bench.timeline import summarise
    from pydvc.pipeline import coordinator
    from pydvc.pipeline.launch import node_info
    from pydvc.pipeline.tiling import load_plan
    from pydvc.solver.engines import on_device

    if mode not in ("strong", "weak"):
        raise ValueError(f"mode must be 'strong' or 'weak', not {mode!r}")
    out_dir = Path(out_dir)
    pool = devices or node_info().local_devices or tuple(range(max(device_counts)))
    if on_device(backend) and max(device_counts) > len(pool):
        raise ValueError(f"asked for {max(device_counts)} devices but only {len(pool)} are visible: {pool}")
    rows, warnings = [], []
    for n in device_counts:
        run_dir = out_dir / f"{mode}_n{n}"
        shutil.rmtree(run_dir, ignore_errors=True)
        c = dataclasses.replace(cfg, output=str(run_dir / "results.zarrvectors"), workdir=str(run_dir / "work"))
        coordinator.prepare(c, backend=backend)
        tiles, _ = load_plan(Path(c.workdir) / "plan.json")
        if mode == "weak":
            want = tiles_per_device * n
            if want > len(tiles):
                raise ValueError(f"weak scaling needs {want} tiles for {n} devices; the plan has {len(tiles)}")
            ids = [t.id for t in sorted(tiles, key=lambda t: (-t.n_points, t.id))[:want]]
        else:
            ids = None
            if len(tiles) < MIN_TILES_PER_DEVICE * n:
                warnings.append(f"{n} devices, {len(tiles)} tiles: fewer than {MIN_TILES_PER_DEVICE} per device, "
                                "so the end of the queue will dominate; use a smaller cluster.tile_shape")
        kw = {"devices": tuple(pool[:n])} if on_device(backend) else {"cpu_workers": n}
        t0 = time.perf_counter()
        stats = coordinator.run(c, backend=backend, tile_ids=ids, **kw)
        seconds = time.perf_counter() - t0
        points = sum(s.points for s in stats)
        compute = sum(s.seconds_compute for s in stats)
        wait = sum(s.seconds_io_wait for s in stats)
        run_stats = json.loads((Path(c.workdir) / "run_stats.json").read_text())
        try:
            tl = summarise(run_stats["events"])
        except FileNotFoundError:
            tl = None
        row = {
            "devices": n, "mode": mode, "tiles": len(ids) if ids is not None else len(tiles), "seconds": seconds,
            "points": points, "points_per_second": points / seconds if seconds else float("nan"),
            "bytes_read": sum(s.bytes_read for s in stats),
            "io_wait_fraction": wait / (compute + wait) if compute + wait else None,
            "errors": sum(len(s.errors) for s in stats), "events": run_stats["events"],
            "utilisation": tl["mean_utilisation"] if tl else None,
            "load_imbalance": tl["load_imbalance"] if tl else None,
            "tail_idle_s_mean": (sum(w["tail_idle_s"] for w in tl["per_worker"]) / len(tl["per_worker"])) if tl else None,
            "timeline": tl,
        }
        rows.append(row)
        progress(f"{mode} {n:2d} devices: {seconds:8.1f} s  {row['points_per_second']:10.0f} pt/s  "
                 f"utilisation {100 * (row['utilisation'] or 0):5.1f} %  imbalance {row['load_imbalance'] or 0:.2f}")
    base = rows[0]
    for r in rows:
        if mode == "strong":
            r["efficiency"] = base["seconds"] * base["devices"] / (r["devices"] * r["seconds"])
        else:
            r["efficiency"] = base["seconds"] / r["seconds"]
    return {"mode": mode, "backend": backend, "rows": rows, "warnings": warnings}


def main(argv: list[str] | None = None) -> None:
    from pydvc.bench.smoke import environment
    from pydvc.solver.engines import default_backend

    p = argparse.ArgumentParser(prog="python -m pydvc.bench.scaling", description=__doc__.split("\n\n")[0])
    p.add_argument("config")
    p.add_argument("--devices", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--tiles-per-device", type=int, default=4, help="weak scaling: tiles per device")
    p.add_argument("--backend", default=None)
    p.add_argument("--out", default="runs/scaling")
    p.add_argument("--read-bandwidth", action="store_true")
    p.add_argument("--readers", type=int, default=8)
    args = p.parse_args(argv)
    cfg = RunConfig.from_yaml(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.read_bandwidth:
        res = read_bandwidth(cfg, readers=args.readers)
        print(json.dumps(res, indent=1))
        (out / "read_bandwidth.json").write_text(json.dumps(res, indent=1))
        return
    res = scaling(cfg, args.devices, backend=args.backend or default_backend(), out_dir=out, mode=args.mode,
                  tiles_per_device=args.tiles_per_device)
    for r in res["rows"]:
        print(f"{r['devices']:2d} devices  {r['seconds']:8.1f} s  {r['points_per_second']:10.0f} pt/s  "
              f"efficiency {100 * r['efficiency']:5.1f} %  I/O wait {100 * (r['io_wait_fraction'] or 0):5.1f} %")
    for w in res["warnings"]:
        print("WARNING:", w)
    res["environment"] = environment()
    (out / f"scaling_{args.mode}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
