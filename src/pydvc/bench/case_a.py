"""Case A: the iDVC example dataset against CCPi, as iDVC runs it (docs/MVP_PLAN.md, Q1 and Q5).

Data
    Zenodo record 7363345 ("Dynamic X-ray CT of Synthetic magma for Digital
    Volume Correlation analysis", P. Lee, Y. Lavallée, B. Bay), whose two
    ``.npy`` volumes (1520 x 1257 x 1260 u8) are presumed to be CCPi's
    ``frame_000`` / ``frame_010``. CCPi's own test files for them
    (``dvc_test/dvc_input.txt``, ``central_grid.roi``, and a reference
    ``completed_central_grid.disp``, which holds only 5 of the 4 680 points)
    come from TomographicImaging/DigitalVolumeCorrelation.

Steps::

    python -m pydvc.bench.case_a fetch --data data/magma
    python -m pydvc.bench.case_a run --data data/magma --out runs/case_A \\
        --ccpi-exe ~/miniforge/envs/ccpi/bin/dvc --backends fused cpu

``run`` builds the pyDVC config from CCPi's ``dvc_input.txt`` (same subvolume,
search and start point), checks the volumes' layout, and hands over to
:func:`pydvc.bench.compare_ccpi.compare`, which times CCPi as iDVC launches it
and pyDVC, and writes ``report.md`` / ``report.json``.

Use ``ccpi-dvc`` 22.0.0 as the reference (docs/benchmarks, M0: the 25.0.0 conda
build's tricubic path is broken). Pass the 25.0.0 binary as a second
``--ccpi-exe`` to time what a fresh iDVC install runs today.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

from pydvc.config import RunConfig

ZENODO_RECORD = "7363345"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
CCPI_REPO = "TomographicImaging/DigitalVolumeCorrelation"
CCPI_FILES = ("dvc_test/dvc_input.txt", "dvc_test/central_grid.roi", "dvc_test/central_grid_results/completed_central_grid.disp")
SHAPE_ZYX = (1260, 1257, 1520)


def _download(url: str, dest: Path, *, md5: str | None = None, chunk: int = 1 << 24) -> None:
    """Resumable download (HTTP Range) with an optional md5 check."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "ab" if have and resp.status == 206 else "wb") as fh:
        while block := resp.read(chunk):
            fh.write(block)
    if md5:
        h = hashlib.md5()
        with open(tmp, "rb") as fh:
            while block := fh.read(chunk):
                h.update(block)
        if h.hexdigest() != md5:
            raise IOError(f"{dest.name}: md5 {h.hexdigest()} != expected {md5}; delete {tmp} and retry")
    tmp.replace(dest)


def fetch(data: Path, *, all_files: bool = False) -> dict[str, str]:
    """Download the Zenodo volumes (``.npy`` only unless ``all_files``) and CCPi's test files into ``data``."""
    data.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_API, timeout=60) as resp:
        record = json.load(resp)
    got = {}
    for f in record["files"]:
        name = f["key"]
        if not all_files and not name.endswith(".npy"):
            continue
        dest = data / name
        md5 = f.get("checksum", "").removeprefix("md5:") or None
        if not dest.exists():
            print(f"downloading {name} ({f['size'] / 1e9:.2f} GB)")
            _download(f["links"]["self"], dest, md5=md5)
        got[name] = str(dest)
    for rel in CCPI_FILES:
        dest = data / Path(rel).name
        if dest.exists():
            continue
        try:
            _download(f"https://raw.githubusercontent.com/{CCPI_REPO}/master/{rel}", dest)
        except Exception:
            _git_fetch(rel, dest)
        got[dest.name] = str(dest)
    (data / "fetched.json").write_text(json.dumps({"zenodo": record.get("metadata", {}).get("title"), "files": got}, indent=1))
    return got


def _git_fetch(rel: str, dest: Path) -> None:
    """Fallback: blob-filtered clone of the CCPi repo, checking out one data file (no source)."""
    repo = dest.parent / ".ccpi_repo"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout",
                        f"https://github.com/{CCPI_REPO}.git", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "HEAD", "--", rel], check=True)
    (repo / rel).replace(dest)


def _volumes(data: Path) -> tuple[Path, Path]:
    npys = sorted(p for p in data.glob("*.npy"))
    if len(npys) < 2:
        raise FileNotFoundError(f"{data}: expected the two Zenodo .npy volumes; run `fetch` first")
    return npys[0], npys[1]


def _c_ordered(path: Path, out_dir: Path) -> tuple[str, tuple[int, int, int] | None, str | None]:
    """A path both codes can read as (z, y, x) C-order u8; transposed to ``.raw`` if the .npy is stored x-first."""
    arr = np.load(path, mmap_mode="r")
    if arr.shape == SHAPE_ZYX and arr.flags.c_contiguous:
        return str(path), None, None
    if arr.shape == SHAPE_ZYX[::-1]:
        raw = out_dir / (path.stem + "_zyx.raw")
        if not raw.exists():
            with open(raw, "wb") as fh:
                for z in range(SHAPE_ZYX[0]):
                    fh.write(np.ascontiguousarray(arr[:, :, z].T).tobytes())
        return str(raw), SHAPE_ZYX[::-1], "|u1"
    raise ValueError(f"{path}: shape {arr.shape}, expected {SHAPE_ZYX} (z, y, x) or its reverse")


def case_config(data: Path, out: Path) -> RunConfig:
    """CCPi's dvc_input.txt, with the Zenodo volumes and pyDVC's output locations."""
    cfg = RunConfig.from_ccpi(data / "dvc_input.txt")
    ref, deformed = _volumes(data)
    out.mkdir(parents=True, exist_ok=True)
    r, shape, dtype = _c_ordered(ref, out)
    d, _, _ = _c_ordered(deformed, out)
    volumes = dataclasses.replace(cfg.volumes, reference=r, deformed=d, raw_shape_xyz=shape or cfg.volumes.raw_shape_xyz,
                                  raw_dtype=dtype or cfg.volumes.raw_dtype, raw_header_bytes=0)
    from pydvc.config import ClusterSpec

    return dataclasses.replace(
        cfg, volumes=volumes, points=str(data / "central_grid.roi"),
        output=str(out / "results.zarrvectors"), workdir=str(out / "work"),
        cluster=ClusterSpec(tile_shape=SHAPE_ZYX, prefetch_depth=1),
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pydvc.bench.case_a", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--data", default="data/magma")
    f.add_argument("--all", action="store_true", help="every file of the record, not only the .npy volumes")
    r = sub.add_parser("run")
    r.add_argument("--data", default="data/magma")
    r.add_argument("--out", default="runs/case_A")
    r.add_argument("--ccpi-exe", nargs="*", default=None)
    r.add_argument("--ccpi-processes", type=int, default=os.cpu_count() or 1)
    r.add_argument("--backends", nargs="+", default=None)
    r.add_argument("--no-cli", action="store_true")
    r.add_argument("--max-ccpi-points", type=int)
    r.add_argument("--reuse-ccpi", action="store_true", help="time finished CCPi runs in --out from their .stat files")
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        print(json.dumps(fetch(Path(args.data), all_files=args.all), indent=1))
        return
    from pydvc.bench.ccpi_baseline import find_dvc
    from pydvc.bench.compare_ccpi import compare, markdown
    from pydvc.solver.engines import default_backend

    data, out = Path(args.data), Path(args.out)
    cfg = case_config(data, out)
    cfg.to_yaml(out / "config.yaml")
    exes = args.ccpi_exe if args.ccpi_exe is not None else [str(find_dvc())]
    report = compare(cfg, out, ccpi_exes=exes, ccpi_processes=args.ccpi_processes,
                     backends=args.backends or [default_backend()], cli=not args.no_cli,
                     reference_disp=data / "completed_central_grid.disp", max_ccpi_points=args.max_ccpi_points,
                     reuse_ccpi=args.reuse_ccpi,
                     title="Case A: iDVC example dataset (Zenodo 7363345), CCPi vs pyDVC")
    print(markdown(report))


if __name__ == "__main__":
    main()
