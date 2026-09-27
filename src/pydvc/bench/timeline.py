"""Where a run's time went, from its event logs (:mod:`pydvc.profiling`).

For each worker (one per GPU): wall time, the share spent solving (utilisation),
start-up before the first tile arrived, stalls waiting for reads and for the
writer, and idle time at the end while other workers finished (the tail).
Across workers: load imbalance (slowest worker's solve time over the mean),
the spread of tile solve times, and throughput. With ``gpu_telemetry.csv``:
mean utilisation, peak memory and mean power per GPU while the run was active.

Command line::

    python -m pydvc.bench.timeline WORKDIR/events/<run id> [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_events(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Events per worker file (stem -> list of records)."""
    out = {}
    for f in sorted(Path(path).glob("worker-*.jsonl")):
        out[f.stem] = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return out


def _sum(events: list[dict[str, Any]], ev: str) -> float:
    return float(sum(e["t1"] - e["t0"] for e in events if e["ev"] == ev))


def summarise(path: str | Path) -> dict[str, Any]:
    workers = load_events(path)
    if not workers:
        raise FileNotFoundError(f"{path}: no worker-*.jsonl event logs")
    starts = [e["t"] for evs in workers.values() for e in evs if e["ev"] == "worker_start"]
    ends = [e["t"] for evs in workers.values() for e in evs if e["ev"] == "worker_end"]
    t_start, t_end = min(starts), max(ends) if ends else max(e.get("t1", 0) for evs in workers.values() for e in evs)
    per_worker, tile_times, tile_rates = [], [], []
    for name, evs in workers.items():
        start = next(e for e in evs if e["ev"] == "worker_start")
        end = next((e for e in evs if e["ev"] == "worker_end"), None)
        solves = [e for e in evs if e["ev"] == "solve"]
        w_end = end["t"] if end else max((e["t1"] for e in evs if "t1" in e), default=start["t"])
        wall = w_end - start["t"]
        solve = _sum(evs, "solve")
        last_solve = max((e["t1"] for e in solves), default=start["t"])
        tile_times += [e["t1"] - e["t0"] for e in solves]
        tile_rates += [e["points"] / (e["t1"] - e["t0"]) for e in solves if e["t1"] > e["t0"]]
        per_worker.append({
            "worker": name, "device": start.get("device"), "tiles": len(solves),
            "points": int(sum(e["points"] for e in solves)), "iterations": int(sum(e.get("iters", 0) for e in solves)),
            "wall_s": wall, "solve_s": solve, "utilisation": solve / wall if wall > 0 else float("nan"),
            "startup_s": _sum(evs, "first_read"), "wait_read_s": _sum(evs, "wait_read"),
            "wait_write_s": _sum(evs, "wait_write"), "read_s": _sum(evs, "read"), "write_s": _sum(evs, "write"),
            "tail_idle_s": max(t_end - last_solve, 0.0),
            "bytes_read": int(sum(e.get("bytes", 0) for e in evs if e["ev"] == "read")),
        })
    solve_times = np.array([w["solve_s"] for w in per_worker])
    points = sum(w["points"] for w in per_worker)
    makespan = t_end - t_start
    out: dict[str, Any] = {
        "events": str(path), "workers": len(per_worker), "makespan_s": makespan, "points": points,
        "points_per_second": points / makespan if makespan > 0 else float("nan"),
        "mean_utilisation": float(np.mean([w["utilisation"] for w in per_worker])),
        "load_imbalance": float(solve_times.max() / solve_times.mean()) if solve_times.mean() > 0 else float("nan"),
        "tile_solve_s": _dist(tile_times), "tile_points_per_second": _dist(tile_rates),
        "read_gb_per_s": sum(w["bytes_read"] for w in per_worker) / makespan / 1e9 if makespan > 0 else float("nan"),
        "per_worker": per_worker,
    }
    tel = Path(path) / "gpu_telemetry.csv"
    if tel.exists():
        out["gpu"] = telemetry(tel)
    return out


def _dist(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    v = np.asarray(values, dtype=np.float64)
    return {"n": int(v.size), "min": float(v.min()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "max": float(v.max()), "mean": float(v.mean())}


def telemetry(path: str | Path) -> dict[str, dict[str, float]]:
    """Mean GPU and memory-controller utilisation, peak memory, mean SM clock and power, per GPU index."""
    rows: dict[str, list[list[float]]] = {}
    with open(path) as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            if len(r) < 8:
                continue
            try:
                vals = [float(x) for x in (r[2], r[3], r[4], r[5], r[6], r[7])]
            except ValueError:                   # "[N/A]" fields on some GPUs
                continue
            rows.setdefault(r[0].strip(), []).append(vals)
    out = {}
    for idx, vals in rows.items():
        a = np.asarray(vals)
        out[idx] = {"samples": int(len(a)), "util_gpu_mean": float(a[:, 0].mean()), "util_mem_mean": float(a[:, 1].mean()),
                    "memory_used_max_mib": float(a[:, 2].max()), "sm_clock_mean_mhz": float(a[:, 3].mean()),
                    "power_mean_w": float(a[:, 4].mean()), "temperature_max_c": float(a[:, 5].max())}
    return out


def markdown(s: dict[str, Any]) -> str:
    lines = [f"Run: {s['workers']} workers, {s['points']} points in {s['makespan_s']:.1f} s "
             f"({s['points_per_second']:.0f} pt/s); mean utilisation {100 * s['mean_utilisation']:.1f} %, "
             f"load imbalance {s['load_imbalance']:.2f}, reads {s['read_gb_per_s']:.2f} GB/s.", "",
             "| worker | tiles | points | wall s | solve s | util | start-up s | read wait s | write wait s | tail idle s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for w in s["per_worker"]:
        lines.append(f"| {w['device']} | {w['tiles']} | {w['points']} | {w['wall_s']:.1f} | {w['solve_s']:.1f} | "
                     f"{100 * w['utilisation']:.0f} % | {w['startup_s']:.1f} | {w['wait_read_s']:.1f} | "
                     f"{w['wait_write_s']:.1f} | {w['tail_idle_s']:.1f} |")
    t = s["tile_solve_s"]
    if t:
        lines += ["", f"Tile solve time: median {t['median']:.2f} s, p90 {t['p90']:.2f} s, max {t['max']:.2f} s ({t['n']} tiles)."]
    for idx, g in sorted(s.get("gpu", {}).items()):
        lines.append(f"GPU {idx}: util {g['util_gpu_mean']:.0f} %, memory controller {g['util_mem_mean']:.0f} %, "
                     f"peak {g['memory_used_max_mib'] / 1024:.1f} GiB, SM {g['sm_clock_mean_mhz']:.0f} MHz, "
                     f"{g['power_mean_w']:.0f} W.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pydvc.bench.timeline", description=__doc__.split("\n\n")[0])
    p.add_argument("events", help="WORKDIR/events/<run id>")
    p.add_argument("--json", help="write the summary here")
    args = p.parse_args(argv)
    s = summarise(args.events)
    print(markdown(s))
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
