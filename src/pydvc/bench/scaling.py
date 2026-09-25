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

    python -m pydvc.bench.scaling CONFIG --devices 1 2 4 8 [--backend fused] [--out runs/scaling]
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


def scaling(cfg: RunConfig, device_counts: list[int], *, backend: str, out_dir: str | Path,
            devices: tuple[int, ...] | None = None) -> list[dict[str, Any]]:
    """Fresh store per device count; returns pt/s, efficiency and I/O wait per count."""
    from pydvc.pipeline import coordinator
    from pydvc.pipeline.launch import node_info
    from pydvc.solver.engines import on_device

    out_dir = Path(out_dir)
    pool = devices or node_info().local_devices or tuple(range(max(device_counts)))
    rows = []
    for n in device_counts:
        run_dir = out_dir / f"n{n}"
        shutil.rmtree(run_dir, ignore_errors=True)
        c = dataclasses.replace(cfg, output=str(run_dir / "results.zarrvectors"), workdir=str(run_dir / "work"))
        coordinator.prepare(c, backend=backend)
        t0 = time.perf_counter()
        if on_device(backend):
            stats = coordinator.run(c, backend=backend, devices=tuple(pool[:n]))
        else:
            stats = coordinator.run(c, backend=backend, cpu_workers=n)
        seconds = time.perf_counter() - t0
        points = sum(s.points for s in stats)
        compute = sum(s.seconds_compute for s in stats)
        wait = sum(s.seconds_io_wait for s in stats)
        rows.append({
            "devices": n, "seconds": seconds, "points": points, "points_per_second": points / seconds,
            "bytes_read": sum(s.bytes_read for s in stats), "io_wait_fraction": wait / (compute + wait) if compute + wait else None,
            "errors": sum(len(s.errors) for s in stats),
        })
    t1 = rows[0]["seconds"] * rows[0]["devices"]
    for r in rows:
        r["efficiency"] = t1 / (r["devices"] * r["seconds"])
    return rows


def main(argv: list[str] | None = None) -> None:
    from pydvc.solver.engines import default_backend

    p = argparse.ArgumentParser(prog="python -m pydvc.bench.scaling", description=__doc__.split("\n\n")[0])
    p.add_argument("config")
    p.add_argument("--devices", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--backend", default=None)
    p.add_argument("--out", default="runs/scaling")
    p.add_argument("--read-bandwidth", action="store_true")
    p.add_argument("--readers", type=int, default=8)
    args = p.parse_args(argv)
    cfg = RunConfig.from_yaml(args.config)
    if args.read_bandwidth:
        print(json.dumps(read_bandwidth(cfg, readers=args.readers), indent=1))
        return
    rows = scaling(cfg, args.devices, backend=args.backend or default_backend(), out_dir=args.out)
    for r in rows:
        print(f"{r['devices']:2d} devices  {r['seconds']:8.1f} s  {r['points_per_second']:10.0f} pt/s  "
              f"efficiency {100 * r['efficiency']:5.1f} %  I/O wait {100 * (r['io_wait_fraction'] or 0):5.1f} %")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "scaling.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
