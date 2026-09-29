"""Brick reads on the host path against the kvikio (GPUDirect Storage) path: speed and identity.

Reads the same random bricks of one volume through each read path of :mod:`zvdvc.io.gds` and
reports throughput, after checking that both paths deliver identical bricks. ``--cold`` drops the
files from the page cache before every read, so the bytes come from the drive rather than RAM:
the case GPUDirect Storage is for.

The report records what the machine offered (:func:`zvdvc.io.gds.status`): without GPUDirect
Storage the kvikio path still runs, through cuFile's or kvikio's POSIX fallback, and its numbers
measure that fallback, not GPUDirect Storage. The ``gds_available`` field says which.

Usage::

    python -m zvdvc.bench.gds --config runs/case_A/config.yaml --brick 256 --bricks 8 --cold
    python -m zvdvc.bench.gds --volume data/ref.ome.zarr --json runs/gds/ref.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from zvdvc.geometry.box import Box


def _files(vol: Any) -> list[str]:
    """The files a volume's bricks come from, for dropping them from the page cache."""
    if hasattr(vol, "path"):
        return [str(vol.path)]
    root = Path(str(vol.array.store.root)) / vol.array.path
    return [str(p) for p in root.rglob("*") if p.is_file()]


def drop_page_cache(paths: list[str]) -> None:
    import kvikio

    for p in paths:
        kvikio.drop_file_page_cache(p)


def random_boxes(shape: tuple[int, int, int], edge: int, n: int, seed: int = 0) -> list[Box]:
    rng = np.random.default_rng(seed)
    edge3 = [min(edge, s) for s in shape]
    boxes = []
    for _ in range(n):
        lo = tuple(int(rng.integers(0, s - e + 1)) for s, e in zip(shape, edge3))
        boxes.append(Box(lo, tuple(a + e for a, e in zip(lo, edge3))))
    return boxes


def bench_volume(make: Any, boxes: list[Box], modes: list[str], *, cold: bool, repeat: int = 1) -> dict[str, Any]:
    """``make(mode)`` opens the volume with that read path; every brick is read ``repeat`` times per mode."""
    import cupy as cp

    out: dict[str, Any] = {"modes": {}, "identical": None}
    digests: dict[str, list[str]] = {}
    for mode in modes:
        vol = make(mode)
        files = _files(vol)
        times, nbytes, digest = [], 0, []
        for box in boxes:
            for r in range(repeat):
                if cold:
                    drop_page_cache(files)
                t0 = time.perf_counter()
                brick = vol.read_brick(box, device="cuda")
                cp.cuda.Device().synchronize()
                times.append(time.perf_counter() - t0)
                if r == 0:
                    nbytes += int(brick.data.nbytes)
                    digest.append(f"{int(cp.asnumpy(brick.data.view(cp.uint8)).sum(dtype=np.uint64))}")
                del brick
        per_brick = nbytes / len(boxes)
        out["modes"][mode] = {
            "io": vol.io, "io_note": vol.io_note, "bricks": len(boxes), "bytes_per_brick": per_brick,
            "seconds_median": statistics.median(times), "seconds_min": min(times),
            "gb_per_s_median": per_brick / statistics.median(times) / 1e9, "gb_per_s_best": per_brick / min(times) / 1e9,
        }
        digests[mode] = digest
    if len(digests) > 1:
        first = next(iter(digests.values()))
        out["identical"] = all(d == first for d in digests.values())
    return out


def main(argv: list[str] | None = None) -> None:
    from zvdvc.config import RunConfig, VolumeSpec
    from zvdvc.io import gds
    from zvdvc.io.volume import open_volume

    p = argparse.ArgumentParser(prog="python -m zvdvc.bench.gds", description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="a run config: its reference volume (and raw layout) is read")
    src.add_argument("--volume", help="an OME-Zarr store, .npy or .mhd")
    p.add_argument("--which", choices=["reference", "deformed"], default="reference")
    p.add_argument("--brick", type=int, default=256, help="brick edge in voxels (default 256)")
    p.add_argument("--bricks", type=int, default=8)
    p.add_argument("--repeat", type=int, default=1, help="reads of each brick per mode")
    p.add_argument("--modes", nargs="+", default=["host", "kvikio"], choices=["host", "kvikio"])
    p.add_argument("--cold", action="store_true", help="drop the files from the page cache before every read")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="also write the report here")
    args = p.parse_args(argv)

    if args.config:
        spec = RunConfig.from_yaml(args.config).volumes
    else:
        spec = VolumeSpec(reference=args.volume, deformed=args.volume)

    def make(mode: str) -> Any:
        return open_volume(dataclasses.replace(spec, gpu_io=mode, prefilter_sigma=0.0), args.which)

    shape = make("host").shape
    report = {"volume": getattr(spec, args.which), "shape_zyx": list(shape), "brick": args.brick, "cold": args.cold,
              "machine": os.uname().nodename, "gpu_io_status": gds.status()}
    report |= bench_volume(make, random_boxes(shape, args.brick, args.bricks, args.seed), args.modes,
                           cold=args.cold, repeat=args.repeat)
    st = report["gpu_io_status"]
    print(f"GPUDirect Storage: {'available' if st['gds_available'] else 'NOT available'}"
          f" (kvikio {st['kvikio']}, cuFile {st['cufile']}, nvCOMP {st['nvcomp']})" + (f"; {st['why']}" if st.get("why") else ""))
    for mode, m in report["modes"].items():
        label = mode if m["io"] == mode else f"{mode} (fell back to {m['io']}: {m['io_note']})"
        print(f"{label:>10}: {m['gb_per_s_median']:.2f} GB/s median, {m['gb_per_s_best']:.2f} best, "
              f"{m['bytes_per_brick'] / 1e6:.0f} MB bricks, {'cold' if args.cold else 'warm'} cache")
    if report["identical"] is not None:
        print(f"bricks identical across paths: {report['identical']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
