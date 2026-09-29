"""zvDVC and iDVC's engine on public validation datasets (docs/validation/datasets.md).

Datasets
    **DVC Challenge 1.0** (Croom et al., Exp. Mech. 61:395, 2021; UVA Dataverse
    doi:10.18130/V3/1UOVKO, CC0), system XCT1: a syntactic-foam cylinder, 350 x 300
    x 504 u16 at 17.86 um, scanned twice more in place (``Repeat1/2``: the noise
    floor), after a 1 mm stage move along the rotation axis (``Axial1/2``) and one
    across it (``Radial1/2``). No field is imposed, so there is no ground truth:
    the repeats should give zero displacement, the moves a near-uniform one.

    **DVC Challenge 2.0** (Tong et al., 2026; NIST doi:10.18434/mds2-4129), the
    synthetic bead volumes of J. Yang (512 x 512 x 192 u8): ``Yang_Beads_S2``, a
    uniform translation along y of 0.0, 0.1, ..., 1.0 voxel (files 1001-1011
    against 1000), and ``Yang_Beads_S3``, six uniaxial stretches along y (1002-1007
    against 1001). The archive ships the volumes but not the fields' values. They
    were recovered independently of either code: the translations are the nominal
    steps (both codes' mean error is below 3e-4 voxel), and the stretches are
    ``v = eps (y - 255)`` with eps = 0.05, 0.10, ..., 0.30, found by registering the
    stretched reference to each deformed volume (:func:`stretch_registration`;
    normalised correlation 0.95-0.97, eps within 1e-5 of the steps).

Settings are iDVC's defaults (cube subvolumes, 12-DOF, ZNSSD, tricubic) with
8 000 samples per subvolume (a 20^3 grid, as iDVC's example). A cube samples the
same grid in both codes, so the codes see identical data and any difference is
the solver's (docs/benchmarks/2026-09-29-idvc-comparison.md). Each code is run by
:func:`zvdvc.bench.compare_ccpi.compare`: CCPi ``dvc`` 22.0.0 as iDVC launches it
(one process, every core), zvDVC in its CCPi-parity mode.

Steps::

    python -m zvdvc.bench.public_datasets fetch   --data /path/to/downloads
    python -m zvdvc.bench.public_datasets prepare --data /path/to/downloads --out runs/public_validation
    python -m zvdvc.bench.public_datasets run     --out runs/public_validation [--cases dvc1_repeat1_32 ...]
    python -m zvdvc.bench.public_datasets prefilter --out runs/public_validation
    python -m zvdvc.bench.public_datasets basin   --out runs/public_validation
    python -m zvdvc.bench.public_datasets report  --out runs/public_validation
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zvdvc.config import RunConfig, SearchSpec, SeedingSpec, SubvolumeSpec, VolumeSpec

DATAVERSE = "https://dataverse.lib.virginia.edu/api/access/datafile/"
DVC1_FILES = {           # Dataverse file ids, XCT1 interlaboratory study
    "XCT1_Scan0_Reference": 24034, "XCT1_Scan1_Repeat1": 24035, "XCT1_Scan2_Repeat2": 24036,
    "XCT1_Scan3_Axial1": 24037, "XCT1_Scan4_Axial2": 24038, "XCT1_Scan5_Radial1": 24039,
    "XCT1_Scan6_Radial2": 24040,
}
NIST = "https://data.nist.gov/od/ds/ark:/88434/mds2-4129/synthetic_data/"
DVC2_FILES = ("Yang_Beads_S2", "Yang_Beads_S3")

DVC1_SHAPE_ZYX = (504, 300, 350)
DVC1_VOXEL_UM = 17.86
BEADS_SHAPE_ZYX = (192, 512, 512)

N_SAMPLES = 8000          # 20^3 grid in a cube, iDVC's example
DOF = 12                  # iDVC's default
DVC1_SPACING = 24.0       # point spacing, voxels
BEADS_SPACING = 32.0
BEADS_SIZE = 32.0
STENCIL = 2.0             # tricubic reach beyond the subvolume
STRETCH_CENTRE_Y = 255.0  # the stretches' fixed plane (fitted: -12.75 voxel offset at 5 %)
DISP_MAX = 10.0           # voxels from each point's seed; covers the stretches' largest step between neighbours


@dataclass(frozen=True)
class Case:
    name: str
    dataset: str                  # "dvc1" | "translation" | "stretch"
    reference: str                # .raw file name under <out>/<dataset dir>
    deformed: str
    size: float
    offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)   # rigid_trans (integer registration offset)
    truth_xyz: tuple[float, float, float] | None = None         # uniform true displacement, if known
    group: str = ""               # repeat | axial | radial | translation | stretch
    strain_y: float | None = None                               # stretch: v = strain_y (y - STRETCH_CENTRE_Y)

    def truth(self, xyz: np.ndarray) -> np.ndarray | None:
        """The true displacement at points ``xyz`` (N, 3), where it is known."""
        if self.truth_xyz is not None:
            return np.broadcast_to(np.asarray(self.truth_xyz, dtype=np.float64), xyz.shape).copy()
        if self.strain_y is not None:
            u = np.zeros_like(xyz, dtype=np.float64)
            u[:, 1] = self.strain_y * (xyz[:, 1] - STRETCH_CENTRE_Y)
            return u
        return None


def cases() -> list[Case]:
    out = []
    ref = "XCT1_Scan0_Reference"
    for size in (24.0, 32.0, 48.0):
        for k, scan in ((1, "XCT1_Scan1_Repeat1"), (2, "XCT1_Scan2_Repeat2")):
            out.append(Case(f"dvc1_repeat{k}_{int(size)}", "dvc1", ref, scan, size, group="repeat"))
    # integer offsets from phase correlation of the whole volumes (what iDVC's registration panel gives);
    # the nominal 1 mm is 56.0 voxels at 17.86 um
    for name, scan, off, group in (("axial1", "XCT1_Scan3_Axial1", (0, 0, 58), "axial"),
                                   ("axial2", "XCT1_Scan4_Axial2", (0, 0, 58), "axial"),
                                   ("radial1", "XCT1_Scan5_Radial1", (-57, 0, 0), "radial"),
                                   ("radial2", "XCT1_Scan6_Radial2", (-58, 0, 0), "radial")):
        out.append(Case(f"dvc1_{name}_32", "dvc1", ref, scan, 32.0, offset_xyz=tuple(map(float, off)), group=group))
    for k in range(1, 12):
        t = round(0.1 * (k - 1), 1)
        out.append(Case(f"translation_{t:.1f}", "translation", "vol_translation_1000", f"vol_translation_{1000 + k}",
                        BEADS_SIZE, truth_xyz=(0.0, t, 0.0), group="translation"))
    for k in range(2, 8):
        out.append(Case(f"stretch_{k - 1}", "stretch", "vol_stretch_1001", f"vol_stretch_{1000 + k}", BEADS_SIZE,
                        group="stretch", strain_y=round(0.05 * (k - 1), 2)))
    return out


# ---------------------------------------------------------------- data


def fetch(data: Path) -> None:
    """Download XCT1 (7 x 100 MB) and the two Yang bead sets (0.6 + 1.1 GB); existing files are kept."""
    (data / "dvc_challenge_1").mkdir(parents=True, exist_ok=True)
    (data / "dvc_challenge_2").mkdir(parents=True, exist_ok=True)
    jobs = [(f"{DATAVERSE}{fid}", data / "dvc_challenge_1" / f"{name}.zip") for name, fid in DVC1_FILES.items()]
    jobs += [(f"{NIST}{name}.zip", data / "dvc_challenge_2" / f"{name}.zip") for name in DVC2_FILES]
    for url, dest in jobs:
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"fetch {dest.name}", flush=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)


def prepare(data: Path, out: Path) -> None:
    """Unzip, stack each volume into a C-ordered little-endian ``.raw`` (what CCPi reads)."""
    import zipfile

    import tifffile

    (out / "dvc1").mkdir(parents=True, exist_ok=True)
    (out / "dvc2").mkdir(parents=True, exist_ok=True)
    for name in DVC1_FILES:
        dest = out / "dvc1" / f"{name}.raw"
        if dest.exists():
            continue
        with zipfile.ZipFile(data / "dvc_challenge_1" / f"{name}.zip") as z:
            members = sorted(m for m in z.namelist() if m.lower().endswith((".tif", ".tiff")))
            vol = np.stack([tifffile.imread(z.open(m)) for m in members])
        if vol.shape != DVC1_SHAPE_ZYX:
            raise ValueError(f"{name}: shape {vol.shape}, expected {DVC1_SHAPE_ZYX}")
        vol.astype("<u2").tofile(dest)
    for ds in DVC2_FILES:
        with zipfile.ZipFile(data / "dvc_challenge_2" / f"{ds}.zip") as z:
            for m in sorted(m for m in z.namelist() if "tiff_files/" in m and m.endswith(".tif")):
                dest = out / "dvc2" / f"{Path(m).stem}.raw"
                if dest.exists():
                    continue
                vol = tifffile.imread(z.open(m))
                if vol.shape != BEADS_SHAPE_ZYX or vol.dtype != np.uint8:
                    raise ValueError(f"{m}: {vol.shape} {vol.dtype}, expected {BEADS_SHAPE_ZYX} uint8")
                vol.tofile(dest)


def _volume(out: Path, case: Case, which: str) -> Path:
    return out / ("dvc1" if case.dataset == "dvc1" else "dvc2") / f"{getattr(case, which)}.raw"


def _shape(case: Case) -> tuple[int, int, int]:
    return DVC1_SHAPE_ZYX if case.dataset == "dvc1" else BEADS_SHAPE_ZYX


def sample_mask(path: Path) -> np.ndarray:
    """The foam cylinder of an XCT1 scan: smoothed image above the midpoint of air and foam, holes filled."""
    from scipy import ndimage

    vol = np.fromfile(path, "<u2").reshape(DVC1_SHAPE_ZYX).astype(np.float32)
    smooth = ndimage.gaussian_filter(vol, 3.0)
    air, foam = np.percentile(smooth, 5), np.percentile(smooth, 75)
    mask = smooth > 0.5 * (air + foam)
    mask = ndimage.binary_opening(mask, iterations=2)
    labels, n = ndimage.label(mask)
    if n > 1:
        mask = labels == (1 + int(np.argmax(ndimage.sum(mask, labels, range(1, n + 1)))))
    return ndimage.binary_fill_holes(mask)


def dvc1_points(out: Path, case: Case, *, erode_size: float) -> np.ndarray:
    """Grid points whose whole cube (``erode_size`` + stencil) lies in the foam in the reference and, moved by
    the case's offset, in the deformed scan."""
    from scipy import ndimage

    cache = out / "dvc1" / "masks"
    cache.mkdir(exist_ok=True)
    masks = []
    for which in ("reference", "deformed"):
        f = cache / f"{getattr(case, which)}.npy"
        if not f.exists():
            np.save(f, np.packbits(sample_mask(_volume(out, case, which))))
        masks.append(np.unpackbits(np.load(f))[: int(np.prod(DVC1_SHAPE_ZYX))].reshape(DVC1_SHAPE_ZYX).astype(bool))
    box = int(np.ceil(erode_size + 2 * STENCIL)) | 1
    inner = [ndimage.minimum_filter(m, size=box, mode="constant", cval=0) for m in masks]
    zyx_shape = np.asarray(DVC1_SHAPE_ZYX)
    axes = [np.arange(np.fmod(n - 1, DVC1_SPACING) / 2, n - 1e-9, DVC1_SPACING) for n in zyx_shape[::-1]]
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    ijk = np.rint(xyz[:, ::-1]).astype(int)
    moved = np.rint(xyz + np.asarray(case.offset_xyz))[:, ::-1].astype(int)
    inside = ((moved >= 0) & (moved < zyx_shape)).all(axis=1)
    keep = inner[0][tuple(ijk.T)] & inside
    keep[inside] &= inner[1][tuple(moved[inside].T)]
    return xyz[keep]


def beads_points() -> np.ndarray:
    """A regular grid over the bead volume, half a subvolume + ``disp_max`` + stencil in from each face.

    CCPi reads beyond the subvolume by up to ``disp_max``: with points 20 voxels from the x and y faces
    (half a subvolume + 4) ``dvc`` 22.0.0 crashed with a segmentation fault in 7 of 11 translation runs.
    """
    margin = BEADS_SIZE / 2 + DISP_MAX + STENCIL + 2
    axes = [np.arange(margin, n - 1 - margin + 1e-9, BEADS_SPACING) for n in BEADS_SHAPE_ZYX[::-1]]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def centre_first(xyz: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(point_id, xyz)`` with the point nearest ``centre`` first: CCPi and the wavefront start there."""
    xyz = xyz[np.lexsort((xyz[:, 0], xyz[:, 1], xyz[:, 2]))]
    c = int(np.argmin(np.linalg.norm(xyz - centre, axis=1)))
    xyz[[0, c]] = xyz[[c, 0]]
    return np.arange(1, len(xyz) + 1, dtype=np.int64), xyz


def build_case(out: Path, case: Case) -> RunConfig:
    """Write the case's points (.roi), config and (if known) truth; return the config."""
    from zvdvc.io.ccpi import write_roi

    work = out / "cases" / case.name
    work.mkdir(parents=True, exist_ok=True)
    shape = _shape(case)
    if case.dataset == "dvc1":
        # the repeats share one grid across subvolume sizes: eroded for the largest
        xyz = dvc1_points(out, case, erode_size=48.0 if case.group == "repeat" else case.size)
    else:
        xyz = beads_points()
    point_id, xyz = centre_first(xyz, 0.5 * (np.asarray(shape[::-1], dtype=float) - 1))
    roi = work / "points.roi"
    write_roi(roi, point_id, xyz)
    if (truth := case.truth(xyz)) is not None:
        np.savez(work / "truth.npz", point_id=point_id, xyz=xyz, displacement=truth)
    raw_dtype = "<u2" if case.dataset == "dvc1" else "u1"
    offset = np.asarray(case.offset_xyz)
    cfg = RunConfig(
        volumes=VolumeSpec(reference=str(_volume(out, case, "reference").resolve()),
                           deformed=str(_volume(out, case, "deformed").resolve()),
                           raw_shape_xyz=tuple(int(n) for n in shape[::-1]), raw_dtype=raw_dtype),
        points=str(roi.resolve()),
        output=str((work / "results.zarrvectors").resolve()),
        subvolume=SubvolumeSpec(geometry="cube", size=case.size, n_samples=N_SAMPLES),
        search=SearchSpec(dof=DOF, objective="znssd", interpolation="tricubic", disp_max=DISP_MAX,
                          rigid_trans=tuple(float(v) for v in offset)),
        seeding=SeedingSpec(strategy="wavefront", start_point=tuple(float(v) for v in xyz[0])),
        workdir=str((work / "run").resolve()),
    )
    cfg.to_yaml(work / "config.yaml")
    return cfg


def run(out: Path, names: list[str] | None, *, backend: str | None, ccpi_exe: str | None, reuse: bool,
        idvc_launch: bool = True) -> None:
    from zvdvc.bench.ccpi_baseline import find_dvc
    from zvdvc.bench.compare_ccpi import compare
    from zvdvc.solver.engines import default_backend

    exe = str(find_dvc(ccpi_exe))
    for case in cases():
        if names and case.name not in names:
            continue
        work = out / "cases" / case.name
        if reuse and (work / "compare" / "report.json").exists():
            continue
        cfg = build_case(out, case)
        print(f"== {case.name}: {len(open(cfg.points).readlines())} points", flush=True)
        compare(cfg, work / "compare", ccpi_exes=[exe], ccpi_processes=1, backends=[backend or default_backend()],
                cli=False, truth=str(work / "truth.npz") if case.truth(np.zeros((1, 3))) is not None else None,
                title=f"{case.name}: iDVC's engine vs zvDVC", reuse_ccpi=reuse, idvc_launch=idvc_launch)


BASIN_RADIUS = 4.0


def run_basin(out: Path, names: list[str] | None, *, backend: str | None, ccpi_exe: str | None) -> None:
    """Both codes on the stretches with CCPi's translation grid search on (``basin_radius`` = 4; iDVC fixes it at 0):
    at 15 % strain and more, neighbouring points 32 voxels apart differ by 5 voxels or more, beyond where a
    seed from the neighbour's translation converges."""
    from zvdvc.bench.ccpi_baseline import find_dvc
    from zvdvc.bench.compare_ccpi import compare
    from zvdvc.solver.engines import default_backend

    exe = str(find_dvc(ccpi_exe))
    for case in cases():
        work = out / "cases" / case.name
        if case.group != "stretch" or (names and case.name not in names) or not (work / "config.yaml").exists():
            continue
        cfg = RunConfig.from_yaml(work / "config.yaml")
        cfg = dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, basin_radius=BASIN_RADIUS))
        kw = dict(ccpi_processes=1, backends=[backend or default_backend()], cli=False, truth=str(work / "truth.npz"),
                  title=f"{case.name}, basin_radius {BASIN_RADIUS:g}", reuse_ccpi=True, idvc_launch=False)
        try:
            compare(cfg, work / "compare_basin", ccpi_exes=[exe], **kw)
        except RuntimeError as exc:           # dvc crashed (it reads outside the image): keep its log, run zvDVC alone
            print(f"{case.name}: CCPi failed: {str(exc)[:200]}", flush=True)
            compare(cfg, work / "compare_basin", ccpi_exes=[], **kw)


PREFILTER_SIGMA = 1.0


def run_prefilter(out: Path, *, backend: str | None) -> None:
    """zvDVC alone with ``volumes.prefilter_sigma`` = 1 (no iDVC equivalent) on the translations and repeats:
    how much of each code's error is tricubic interpolation bias."""
    from zvdvc.pipeline.inmemory import load_points, solve_in_memory
    from zvdvc.solver.engines import default_backend

    for case in cases():
        work = out / "cases" / case.name
        if case.group not in ("translation", "repeat") or not (work / "config.yaml").exists():
            continue
        cfg = RunConfig.from_yaml(work / "config.yaml")
        cfg = dataclasses.replace(cfg, volumes=dataclasses.replace(cfg.volumes, prefilter_sigma=PREFILTER_SIGMA))
        point_id, xyz = load_points(cfg)
        res = solve_in_memory(cfg, point_id, xyz, strategy="wavefront", backend=backend or default_backend())
        res.save(work / "compare" / "zvdvc_prefilter.npz")
        print(f"{case.name}: prefilter sigma {PREFILTER_SIGMA}, {res.status_counts()}", flush=True)


# ---------------------------------------------------------------- scoring


def _load_pair(work: Path, sub: str = "compare", case: Case | None = None) -> dict[str, Any]:
    """Both codes' results on the case's interior points (edge points, where CCPi reads outside the image, dropped).

    Edge points are judged at the true displacement where ``case`` knows it, else at either code's.
    """
    from zvdvc.bench.compare_ccpi import _edge_geometry
    from zvdvc.bench.metrics import _match, edge_mask, load_results

    rep = json.loads((work / sub / "report.json").read_text())
    # "ccpi": dvc as iDVC launches it (one process, every core); "ccpi_1t": the same on one thread, which gives
    # bit-identical results and stands in for it in cases run with --one-thread
    runs = {("zvdvc" if r["name"].startswith("zvDVC") else "ccpi" if "iDVC" in r["name"] else "ccpi_1t"): r
            for r in rep["runs"]}
    idvc_launch = "ccpi" in runs
    if not idvc_launch:
        runs["ccpi"] = runs.pop("ccpi_1t")
    a, b = load_results(runs["zvdvc"]["results"]), load_results(runs["ccpi"]["results"])
    ia, ib = _match(a["point_id"], b["point_id"])
    ccpi_self = None
    if "ccpi_1t" in runs:
        c = load_results(runs["ccpi_1t"]["results"])
        jb, jc = _match(b["point_id"], c["point_id"])
        same = (b["status"][jb] == c["status"][jc])
        ccpi_self = {"n": int(len(jb)), "status_same": int(same.sum()),
                     "max_abs": float(np.abs(b["displacement"][jb][same] - c["displacement"][jc][same]).max())}
    cfg = RunConfig.from_yaml(work / "config.yaml")
    shape_xyz, reach = _edge_geometry(cfg)
    truth = case.truth(a["xyz"][ia]) if case is not None else None
    if truth is not None:
        edge = edge_mask(a["xyz"][ia], truth, shape_xyz, reach)
    else:
        edge = edge_mask(b["xyz"][ib], b["displacement"][ib], shape_xyz, reach) | \
            edge_mask(a["xyz"][ia], a["displacement"][ia], shape_xyz, reach)
    good_z, good_c = a["status"][ia] == 0, b["status"][ib] == 0
    return {"xyz": a["xyz"][ia], "zvdvc": a["displacement"][ia], "ccpi": b["displacement"][ib],
            "good_zvdvc": good_z & ~edge, "good_ccpi": good_c & ~edge, "interior": ~edge,
            "seconds": {k: r["seconds"] for k, r in runs.items()}, "point_id": a["point_id"][ia],
            "status_zvdvc": a["status"][ia], "status_ccpi": b["status"][ib], "ccpi_self": ccpi_self,
            "idvc_launch": idvc_launch}


def affine_fit(xyz: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares ``u = t + G (x - centre)``; returns (coefficients (4, 3), prediction)."""
    x = np.column_stack([np.ones(len(xyz)), xyz])
    coef, *_ = np.linalg.lstsq(x, u, rcond=None)
    return coef, x @ coef


WRONG_ABS = 0.5           # voxels from the truth: a GOOD point further off is reported GOOD but wrong


def _stats(err: np.ndarray) -> dict[str, Any]:
    if len(err) == 0:
        nan3 = [float("nan")] * 3
        return {"n": 0, "mean": nan3, "sd": nan3, "rmse": nan3, "median_abs": float("nan"), "p95_abs": float("nan")}
    mag = np.linalg.norm(err, axis=1)
    return {"n": int(len(err)), "mean": err.mean(0).tolist(), "sd": err.std(0).tolist(),
            "rmse": np.sqrt((err ** 2).mean(0)).tolist(), "median_abs": float(np.median(mag)),
            "p95_abs": float(np.percentile(mag, 95))}


def _own(u: np.ndarray, good: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """One code on its own GOOD interior points: right (within WRONG_ABS of the truth) or reported GOOD but wrong."""
    err = u[good] - truth[good]
    right = np.linalg.norm(err, axis=1) <= WRONG_ABS
    return {"good": int(good.sum()), "right": int(right.sum()), "good_but_wrong": int((~right).sum()),
            "right_stats": _stats(err[right])}


def _basin(work: Path, case: Case, p: dict[str, Any]) -> dict[str, Any]:
    """Each code with ``basin_radius`` on, over the interior points of the main run; a CCPi crash is recorded."""
    from zvdvc.bench.metrics import load_results

    d = work / "compare_basin"
    out: dict[str, Any] = {}
    files = {"zvdvc": next(iter(sorted(d.glob("zvdvc_*_parity.npz"))), None),
             "ccpi": next(iter(sorted(d.glob("ccpi_*/t1_p1.disp"))), None)}
    order = np.argsort(p["point_id"])
    for k, f in files.items():
        if f is None:
            logs = sorted(d.glob("ccpi_*/t1_p1_0.log")) if k == "ccpi" else []
            if logs:
                # progress lines read "i/N label x y z status ..."; the crash point varies from run to run
                done = sum(1 for line in logs[0].read_text(errors="replace").splitlines()
                           if re.match(r"\d+/\d+\s", line))
                out[k] = {"crashed_after": done}
            continue
        r = load_results(f)
        rows = order[np.searchsorted(p["point_id"], r["point_id"], sorter=order)]
        u = np.full_like(p["zvdvc"], np.nan)
        good = np.zeros(len(u), dtype=bool)
        u[rows] = r["displacement"]
        good[rows] = r["status"] == 0
        out[k] = _own(u, good & p["interior"], case.truth(p["xyz"]))
    return out


def score(out: Path, case: Case) -> dict[str, Any] | None:
    work = out / "cases" / case.name
    if not (work / "compare" / "report.json").exists():
        return None
    p = _load_pair(work, case=case)
    both = p["good_zvdvc"] & p["good_ccpi"]
    res: dict[str, Any] = {"case": case.name, "group": case.group, "size": case.size,
                           "n_points": int(len(p["xyz"])), "n_interior": int(p["interior"].sum()),
                           "good": {k: int(p[f"good_{k}"].sum()) for k in ("zvdvc", "ccpi")},
                           "n_good_both": int(both.sum()), "seconds": p["seconds"], "ccpi_self": p["ccpi_self"],
                           "idvc_launch": p["idvc_launch"],
                           # interior points GOOD in both or in neither
                           "status_agreement": float(((p["status_zvdvc"] == 0) == (p["status_ccpi"] == 0))[
                               p["interior"]].mean())}
    d = p["zvdvc"][both] - p["ccpi"][both]
    res["agreement"] = _stats(d)
    xyz = p["xyz"][both]
    truth = case.truth(p["xyz"])
    if truth is not None:
        res["truth"] = {k: _stats(p[k][both] - truth[both]) for k in ("zvdvc", "ccpi")}
        # each code on its own GOOD points: right (within WRONG_ABS of the truth) or reported GOOD but wrong
        res["own"] = {k: _own(p[k], p[f"good_{k}"], truth) for k in ("zvdvc", "ccpi")}
    else:
        # DVC Challenge 1.0: the raw scatter (repeats: noise floor), and the scatter about each code's own affine
        # fit, which removes the rigid move and the scanner's smooth distortion (real, not error)
        res["raw"] = {k: _stats(p[k][both]) for k in ("zvdvc", "ccpi")}
        res["about_affine"] = {k: _stats(p[k][both] - affine_fit(xyz, p[k][both])[1]) for k in ("zvdvc", "ccpi")}
    if truth is not None and (work / "compare_basin").exists():
        res["own_basin"] = _basin(work, case, p)
    pf_path = work / "compare" / "zvdvc_prefilter.npz"
    if pf_path.exists():
        from zvdvc.bench.metrics import _match, load_results

        pf = load_results(pf_path)
        ia, ib = _match(p["point_id"][both], pf["point_id"])
        ok = pf["status"][ib] == 0
        u = pf["displacement"][ib][ok]
        if truth is not None:
            res["truth"]["zvdvc_prefilter"] = _stats(u - case.truth(pf["xyz"][ib][ok]))
        elif "raw" in res:
            res["raw"]["zvdvc_prefilter"] = _stats(u)
            res["about_affine"]["zvdvc_prefilter"] = _stats(u - affine_fit(xyz[ia][ok], u)[1])
    return res


LABELS = (("zvdvc", "zvDVC"), ("ccpi", "iDVC"), ("zvdvc_prefilter", f"zvDVC, prefilter σ = {PREFILTER_SIGMA:g}"))


def report(out: Path) -> dict[str, Any]:
    scores = [s for c in cases() if (s := score(out, c))]
    (out / "summary.json").write_text(json.dumps(scores, indent=1))
    lines = ["# Public datasets: zvDVC and iDVC's engine", ""]

    def f3(v: list[float], fmt: str = "{:.4f}") -> str:
        return ", ".join(fmt.format(x) for x in v)

    lines += ["| case | interior points | GOOD zvDVC / iDVC | zvDVC − iDVC median \\|Δu\\| | p95 | mean Δu (x, y, z) | "
              "time zvDVC / iDVC |", "|---|---:|---|---:|---:|---|---|"]
    for s in scores:
        a = s["agreement"]
        lines.append(f"| {s['case']} | {s['n_interior']} | {s['good']['zvdvc']} / {s['good']['ccpi']} | "
                     f"{a['median_abs']:.2e} | {a['p95_abs']:.2e} | {f3(a['mean'], '{:+.1e}')} | "
                     f"{s['seconds']['zvdvc']:.1f} s / {s['seconds']['ccpi']:.0f} s"
                     f"{'' if s['idvc_launch'] else ' (1 thread)'} |")
    lines += ["", "## Against the truth", "",
              "| case | code | mean error (x, y, z) | SD (x, y, z) | RMSE (x, y, z) |", "|---|---|---|---|---|"]
    for s in scores:
        for k, label in LABELS:
            if k in s.get("truth", {}):
                t = s["truth"][k]
                lines.append(f"| {s['case']} | {label} | {f3(t['mean'], '{:+.4f}')} | {f3(t['sd'])} | {f3(t['rmse'])} |")
    lines += ["", "## Each code on its own GOOD points", "",
              f"| case | code | interior | GOOD | right (≤ {WRONG_ABS} voxel) | GOOD but wrong | RMSE of the right (x, y, z) |",
              "|---|---|---:|---:|---:|---:|---|"]
    for s in scores:
        for k, label in LABELS:
            for key, suffix in (("own", ""), ("own_basin", f", basin_radius {BASIN_RADIUS:g}")):
                o = s.get(key, {}).get(k)
                if o and "crashed_after" in o:
                    lines.append(f"| {s['case']} | {label}{suffix} | {s['n_interior']} | crashed (segmentation fault) "
                                 f"after {o['crashed_after']} points | | | |")
                elif o:
                    lines.append(f"| {s['case']} | {label}{suffix} | {s['n_interior']} | {o['good']} | {o['right']} | "
                                 f"{o['good_but_wrong']} | {f3(o['right_stats']['rmse'])} |")
    lines += ["", "## DVC Challenge 1.0 scatter", "",
              "| case | code | mean (x, y, z) | SD (x, y, z) | SD about affine fit (x, y, z) |", "|---|---|---|---|---|"]
    for s in scores:
        for k, label in LABELS:
            if k in s.get("raw", {}):
                lines.append(f"| {s['case']} | {label} | {f3(s['raw'][k]['mean'], '{:+.3f}')} | {f3(s['raw'][k]['sd'])} | "
                             f"{f3(s['about_affine'][k]['sd'])} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return {"scores": scores}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m zvdvc.bench.public_datasets", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--data", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--data", required=True)
    pr.add_argument("--out", default="runs/public_validation")
    r = sub.add_parser("run")
    r.add_argument("--out", default="runs/public_validation")
    r.add_argument("--cases", nargs="*")
    r.add_argument("--backend")
    r.add_argument("--ccpi-exe")
    r.add_argument("--reuse", action="store_true", help="skip cases with a report; reuse finished CCPi runs")
    r.add_argument("--one-thread", action="store_true",
                   help="run CCPi on one thread only, not also as iDVC launches it (the results are identical)")
    pf = sub.add_parser("prefilter", help="zvDVC alone with prefilter_sigma = 1 on the translations and repeats")
    pf.add_argument("--out", default="runs/public_validation")
    pf.add_argument("--backend")
    b = sub.add_parser("basin", help="both codes on the stretches with basin_radius = 4")
    b.add_argument("--out", default="runs/public_validation")
    b.add_argument("--cases", nargs="*")
    b.add_argument("--backend")
    b.add_argument("--ccpi-exe")
    rp = sub.add_parser("report")
    rp.add_argument("--out", default="runs/public_validation")
    sub.add_parser("list")
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        fetch(Path(args.data))
    elif args.cmd == "prepare":
        prepare(Path(args.data), Path(args.out))
    elif args.cmd == "run":
        run(Path(args.out), args.cases, backend=args.backend, ccpi_exe=args.ccpi_exe or os.environ.get("ZVDVC_CCPI_DVC"),
            reuse=args.reuse, idvc_launch=not args.one_thread)
    elif args.cmd == "basin":
        run_basin(Path(args.out), args.cases, backend=args.backend,
                  ccpi_exe=args.ccpi_exe or os.environ.get("ZVDVC_CCPI_DVC"))
    elif args.cmd == "prefilter":
        run_prefilter(Path(args.out), backend=args.backend)
    elif args.cmd == "report":
        report(Path(args.out))
    else:
        for c in cases():
            print(c.name)


if __name__ == "__main__":
    main()
