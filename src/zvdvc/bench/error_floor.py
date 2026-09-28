"""Error floor on real data: known sub-voxel shifts of a crop of the iDVC case A reference.

There are no repeat scans of an unloaded sample for case A, so the floor is
measured by displacing a real image by a known amount and correlating it
against itself:

* A crop inside the sample (chosen by foreground fraction) is shifted by a
  fractional translation ``s`` along x and along the (1, 1, 1) diagonal,
  0 to 1 voxel in steps of 0.1, with ``scipy.ndimage.shift`` (quintic
  B-spline, mirror boundaries). zvDVC interpolates with Catmull-Rom cubics,
  a different interpolant, so the test does not grade the solver with its
  own model ("inverse crime").
* ``one-sided``: ref = I, def = shift(I, s), so only the deformed image is
  resampled; ``symmetric``: ref = shift(I, -s/2), def = shift(I, +s/2), so
  both are smoothed alike and the systematic (interpolation) error separates
  from the resampling's smoothing.
* Independent Gaussian noise is added to ref and def at 0, 0.5, 1 and 2 times
  the scan's own noise, estimated from the crop (Immerkær's Laplacian
  estimator, slice by slice). The scan's own noise is common to ref and def
  (it moves with the image), so the zero-added-noise rows measure
  interpolation and model error only.
* Subvolume size (40, 60, 80) and sample count (1000 to 8000) are swept, with
  CCPi's case A settings otherwise (6-DOF, ZNSSD, tricubic). Points lie on a
  grid whose spacing is the subvolume size, so their errors are independent.

For each case: bias (mean error per axis), random error (standard deviation
per axis), RMSE, fraction GOOD, and iterations. A sanity check correlates the
crop with itself (displacement must be 0).

Shifted copies of one image have no deformation inside a subvolume and share
the scan's noise, so they bound only part of the error. The **seed
repeatability** test solves the real pair (CCPi's central grid) several times,
changing only the template's random seed, and reports the spread: the
uncertainty that comes from which sample points each subvolume uses.

Command line (about 20-40 min on a GPU)::

    python -m zvdvc.bench.error_floor --case-a runs/case_A_data --out runs/error_floor
    python -m zvdvc.bench.error_floor --quick ...      # a few cases, for a smoke test
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from zvdvc.config import SearchSpec, SubvolumeSpec

SHIFTS = tuple(round(0.1 * i, 1) for i in range(11))
DIRECTIONS = {"x": (1.0, 0.0, 0.0), "diagonal": tuple(np.full(3, 1.0 / np.sqrt(3.0)))}
VARIANTS = ("one-sided", "symmetric")
NOISE_FACTORS = (0.0, 0.5, 1.0, 2.0)
SIZES = (40.0, 60.0, 80.0)
N_SAMPLES = (1000, 2000, 4000, 8000)


@dataclasses.dataclass(frozen=True)
class Case:
    shift: float
    direction: str
    variant: str
    noise_factor: float
    size: float
    n_samples: int


# --------------------------------------------------------------------------- data


def estimate_noise(vol: np.ndarray, n_slices: int = 16) -> float:
    """Gaussian noise sigma by Immerkær's method (fast noise variance estimation, 1996), averaged over z slices."""
    from scipy.ndimage import convolve

    kernel = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
    zs = np.linspace(0, vol.shape[0] - 1, min(n_slices, vol.shape[0])).astype(int)
    sig = []
    for z in zs:
        s = vol[z].astype(np.float64)
        h, w = s.shape
        lap = convolve(s, kernel)[1:-1, 1:-1]
        sig.append(np.sqrt(np.pi / 2.0) * np.abs(lap).sum() / (6.0 * (w - 2) * (h - 2)))
    return float(np.median(sig))


def choose_crop(volume: Any, shape_zyx: tuple[int, int, int], size: int, stride: int = 8) -> tuple[int, int, int]:
    """Origin (z, y, x) of the cubic crop of side ``size`` with the most foreground (Otsu threshold on a subsample)."""
    from scipy.ndimage import uniform_filter

    sub = np.asarray(volume[::stride, ::stride, ::stride], dtype=np.float64)
    hist, edges = np.histogram(sub, bins=256)
    w = np.cumsum(hist)
    m = np.cumsum(hist * (edges[:-1] + edges[1:]) / 2)
    between = (m[-1] * w - m * w[-1]) ** 2 / np.maximum(w * (w[-1] - w), 1)
    thr = edges[int(np.argmax(between))]
    k = max(size // stride, 1)
    frac = uniform_filter((sub > thr).astype(np.float64), size=k, mode="constant")
    half = k // 2
    valid = frac[half:sub.shape[0] - (k - half) + 1, half:sub.shape[1] - (k - half) + 1, half:sub.shape[2] - (k - half) + 1]
    iz, iy, ix = np.unravel_index(int(np.argmax(valid)), valid.shape)
    origin = (iz * stride, iy * stride, ix * stride)
    return tuple(int(min(o, n - size)) for o, n in zip(origin, shape_zyx))


def case_a_crop(case_a: Path, cache: Path, size: int) -> tuple[np.ndarray, dict[str, Any]]:
    from zvdvc.bench.case_a import case_config
    from zvdvc.geometry.box import Box
    from zvdvc.io.volume import open_volume

    cfg = case_config(case_a, cache)
    vol = open_volume(cfg.volumes, "reference")
    origin = choose_crop(vol.array, vol.shape, size)
    box = Box(origin, tuple(o + size for o in origin))
    data = vol.read_brick(box, device="cpu").data.astype(np.float32)
    return data, {"source": "case_a reference", "origin_zyx": list(origin), "size": size}


def shifted(coeffs: np.ndarray, shift_xyz: tuple[float, float, float]) -> np.ndarray:
    """The image moved by ``shift_xyz`` (content at c appears at c + shift), from quintic B-spline coefficients."""
    from scipy.ndimage import shift

    return shift(coeffs, tuple(shift_xyz[::-1]), order=5, mode="mirror", prefilter=False).astype(np.float32)


def _warp_job(args: tuple[str, tuple[float, float, float]]) -> np.ndarray:
    path, s = args
    return shifted(np.load(path, mmap_mode="r"), s)


# --------------------------------------------------------------------------- solve


def grid_points(shape_zyx: tuple[int, ...], size: float, margin: float) -> np.ndarray:
    """Points (x, y, z) spaced by the subvolume size, whose subvolumes stay ``margin`` inside the crop."""
    axes = [np.arange(margin, n - 1 - margin + 1e-9, size) for n in shape_zyx[::-1]]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def solve(ref: np.ndarray, deformed: np.ndarray, points: np.ndarray, subvolume: SubvolumeSpec, engine: Any) -> Any:
    from zvdvc.geometry.box import Box
    from zvdvc.geometry.templates import make_template
    from zvdvc.io.volume import Brick
    from zvdvc.solver.gauss_newton import solve_batch

    box = Box((0, 0, 0), ref.shape)
    search = SearchSpec(dof=6, objective="znssd", interpolation="tricubic", disp_max=5.0)
    r = solve_batch(Brick(ref, box, box), Brick(deformed, box, box), points, np.zeros((len(points), 3)),
                    make_template(subvolume), search, engine=engine)
    host = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)                    # noqa: E731
    return host(r.status), host(r.params)[:, :3].astype(np.float64), host(r.n_iter)


def stats(status: np.ndarray, disp: np.ndarray, truth: np.ndarray, n_iter: np.ndarray) -> dict[str, Any]:
    good = status == 0
    err = disp[good] - truth
    nan3 = [float("nan")] * 3
    return {"n_points": int(len(status)), "frac_good": float(good.mean()),
            "bias": err.mean(axis=0).tolist() if len(err) else nan3,
            "std": err.std(axis=0, ddof=1).tolist() if len(err) > 1 else nan3,
            "rmse": float(np.sqrt((err**2).sum(axis=1).mean())) if len(err) else float("nan"),
            "mean_iter": float(n_iter[good].mean()) if good.any() else float("nan")}


def run(image: np.ndarray, *, out: Path, backend: str, shifts=SHIFTS, directions=tuple(DIRECTIONS), variants=VARIANTS,
        noise_factors=NOISE_FACTORS, sizes=SIZES, n_samples=N_SAMPLES, workers: int = 8, seed: int = 0,
        progress=print) -> dict[str, Any]:
    from scipy.ndimage import spline_filter

    from zvdvc.solver.engines import make_engine

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sigma = estimate_noise(image)
    coeff_path = out / "coeffs.npy"
    np.save(coeff_path, spline_filter(image.astype(np.float64), order=5, mode="mirror"))
    progress(f"noise sigma {sigma:.3f}; spline coefficients in {time.perf_counter() - t0:.0f} s")

    # every image needed: the unshifted one, and the shifts of each variant
    needed: dict[tuple[float, float, float], None] = {}
    for s, d, v in itertools.product(shifts, directions, variants):
        vec = np.asarray(DIRECTIONS[d]) * s
        for part in ([vec] if v == "one-sided" else [-vec / 2, vec / 2]):
            needed[tuple(np.round(part, 6))] = None
    keys = list(needed)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        images = dict(zip(keys, pool.map(_warp_job, [(str(coeff_path), k) for k in keys])))
    images[(0.0, 0.0, 0.0)] = image.astype(np.float32)
    progress(f"{len(keys)} shifted images in {time.perf_counter() - t0:.0f} s")

    engine = make_engine(backend)
    rng = np.random.default_rng(seed)
    results = []

    # sanity: the crop against itself
    sub = SubvolumeSpec(geometry="sphere", size=max(sizes), n_samples=max(n_samples))
    pts = grid_points(image.shape, max(sizes), max(sizes) / 2 + 4.0)
    st, disp, it = solve(images[(0.0, 0.0, 0.0)], images[(0.0, 0.0, 0.0)], pts, sub, engine)
    self_corr = {"max_abs_displacement": float(np.abs(disp[st == 0]).max()) if (st == 0).any() else float("nan"),
                 "frac_good": float((st == 0).mean()), "n_points": int(len(pts))}
    progress(f"self-correlation: {self_corr}")

    for s, d, v in itertools.product(shifts, directions, variants):
        vec = np.asarray(DIRECTIONS[d]) * s
        if v == "one-sided":
            ref0, def0 = images[(0.0, 0.0, 0.0)], images[tuple(np.round(vec, 6))]
        else:
            ref0, def0 = images[tuple(np.round(-vec / 2, 6))], images[tuple(np.round(vec / 2, 6))]
        for k in noise_factors:
            if k > 0:
                ref = ref0 + rng.normal(0.0, k * sigma, ref0.shape).astype(np.float32)
                deformed = def0 + rng.normal(0.0, k * sigma, def0.shape).astype(np.float32)
            else:
                ref, deformed = ref0, def0
            for size, m in itertools.product(sizes, n_samples):
                pts = grid_points(image.shape, size, size / 2 + 4.0 + s)
                st, disp, it = solve(ref, deformed, pts, SubvolumeSpec(geometry="sphere", size=size, n_samples=m), engine)
                row = dataclasses.asdict(Case(s, d, v, k, size, m)) | stats(st, disp, vec, it)
                results.append(row)
        progress(f"shift {s} {d} {v}: done ({len(results)} cases)")
    return {"sigma": sigma, "self_correlation": self_corr, "backend": backend, "image_shape": list(image.shape),
            "cases": results}


# --------------------------------------------------------------------------- repeatability on the real pair


def seed_repeatability(cfg: Any, point_id: np.ndarray, xyz: np.ndarray, *, seeds=(0, 1, 2), backend: str = "fused",
                       progress=print) -> dict[str, Any]:
    """Spread between solves of the real pair that differ only in the template's random seed.

    Each seed draws a different set of sample points in every subvolume. On real
    data, with deformation that is not rigid within a subvolume and two
    independent acquisitions, the result depends on that set; this is the
    uncertainty the shifted-image cases above cannot see. It is also the
    disagreement to expect with CCPi, which draws its own sample set.
    """
    from zvdvc.pipeline.inmemory import solve_in_memory

    runs = {}
    for seed in seeds:
        c = dataclasses.replace(cfg, subvolume=dataclasses.replace(cfg.subvolume, seed=seed))
        runs[seed] = solve_in_memory(c, point_id, xyz, backend=backend)
    pairs = []
    for a, b in itertools.combinations(seeds, 2):
        ra, rb = runs[a], runs[b]
        good = (ra.status == 0) & (rb.status == 0)
        e = ra.params[good, :3].astype(np.float64) - rb.params[good, :3]
        mag = np.linalg.norm(e, axis=1)
        pairs.append({"seeds": [a, b], "n_good_both": int(good.sum()), "mean": e.mean(axis=0).tolist(),
                      "std": e.std(axis=0, ddof=1).tolist(), "median_abs": float(np.median(mag)),
                      "p95_abs": float(np.percentile(mag, 95))})
        progress(f"seeds {a} vs {b}: median |du| {pairs[-1]['median_abs']:.4f}, p95 {pairs[-1]['p95_abs']:.4f}")
    per_axis = float(np.mean([np.mean(p["std"]) for p in pairs]) / np.sqrt(2.0))
    return {"n_points": int(len(point_id)), "pairs": pairs, "per_estimate_std": per_axis,
            "settings": dataclasses.asdict(cfg.subvolume)}


# --------------------------------------------------------------------------- report


def summary(report: dict[str, Any]) -> dict[str, Any]:
    """The report's headline tables: bias against shift, random error against settings, and noise scaling."""
    cases = report["cases"]
    big = max(c["size"] for c in cases)
    most = max(c["n_samples"] for c in cases)

    def pick(**kw):
        return [c for c in cases if all(c[k] == v for k, v in kw.items())]

    s_curve = {v: [(c["shift"], c["bias"][0]) for c in sorted(pick(direction="x", variant=v, noise_factor=0.0, size=big,
                                                                      n_samples=most), key=lambda c: c["shift"])]
               for v in {c["variant"] for c in cases}}
    random_err = {}
    for size, m in sorted({(c["size"], c["n_samples"]) for c in cases}):
        for k in sorted({c["noise_factor"] for c in cases}):
            sel = pick(size=size, n_samples=m, noise_factor=k, variant="symmetric") or pick(size=size, n_samples=m, noise_factor=k)
            if sel:
                random_err[f"{size:g}/{m}/{k:g}"] = float(np.mean([np.mean(c["std"]) for c in sel]))
    # slope of log(random error) against log(noise) and log(n_samples), at the largest size, noisy cases only
    slope = {}
    noisy = [c for c in cases if c["noise_factor"] > 0 and c["size"] == big and np.isfinite(np.mean(c["std"]))]
    if len({c["noise_factor"] for c in noisy}) > 1:
        x = np.log([c["noise_factor"] for c in noisy if c["n_samples"] == most])
        y = np.log([np.mean(c["std"]) for c in noisy if c["n_samples"] == most])
        slope["vs_noise"] = float(np.polyfit(x, y, 1)[0])
    one = [c for c in noisy if c["noise_factor"] == 1.0]
    if len({c["n_samples"] for c in one}) > 1:
        slope["vs_n_samples"] = float(np.polyfit(np.log([c["n_samples"] for c in one]),
                                                 np.log([np.mean(c["std"]) for c in one]), 1)[0])
    worst_bias = max(abs(b) for c in pick(noise_factor=0.0) for b in c["bias"] if np.isfinite(b))
    return {"bias_vs_shift_x": s_curve, "random_error_by_size_samples_noise": random_err, "log_slopes": slope,
            "worst_abs_bias_noise_free": worst_bias}


def main(argv: list[str] | None = None) -> None:
    from zvdvc.bench import case_a as ca
    from zvdvc.bench.smoke import environment
    from zvdvc.solver.engines import gpu_available

    p = argparse.ArgumentParser(prog="python -m zvdvc.bench.error_floor", description=__doc__.split("\n\n")[0])
    p.add_argument("--case-a", help="case A data directory (default $ZVDVC_CASE_A)")
    p.add_argument("--cache", help="directory for C-ordered .raw copies (default $ZVDVC_CASE_A_CACHE or runs/case_A)")
    p.add_argument("--out", default="runs/error_floor")
    p.add_argument("--crop", type=int, default=448)
    p.add_argument("--backend", default="fused" if gpu_available() else "cpu")
    p.add_argument("--workers", type=int, default=16, help="processes for the shifted images")
    p.add_argument("--quick", action="store_true", help="a few cases only (smoke test)")
    p.add_argument("--prefilter-sigma", type=float, default=0.0,
                   help="Gaussian prefilter (voxels) on both images, as volumes.prefilter_sigma applies it")
    args = p.parse_args(argv)
    data = ca.data_dir(args.case_a)
    if data is None:
        raise SystemExit("case A data not found: pass --case-a DIR or set ZVDVC_CASE_A")
    image, crop = case_a_crop(data, ca.cache_dir(args.cache), args.crop)
    if args.prefilter_sigma:                     # a blur commutes with the shifts, so filtering the crop once is exact
        from scipy.ndimage import gaussian_filter

        from zvdvc.io.volume import TRUNCATE

        image = gaussian_filter(image, args.prefilter_sigma, mode="nearest", truncate=TRUNCATE)
        crop["prefilter_sigma"] = args.prefilter_sigma
    kw = dict(shifts=(0.0, 0.5, 1.0), variants=("symmetric",), noise_factors=(0.0, 1.0), sizes=(80.0,),
              n_samples=(8000,)) if args.quick else {}
    out = Path(args.out)
    report = run(image, out=out, backend=args.backend, workers=args.workers, **kw)
    report |= {"crop": crop, "environment": environment(), "summary": summary(report)}
    from zvdvc.pipeline.inmemory import load_points

    cfg = ca.case_config(data, ca.cache_dir(args.cache))
    if args.prefilter_sigma:
        cfg = dataclasses.replace(cfg, volumes=dataclasses.replace(cfg.volumes, prefilter_sigma=args.prefilter_sigma))
    pid, xyz = load_points(cfg)
    report["seed_repeatability"] = seed_repeatability(cfg, pid, xyz, seeds=(0, 1) if args.quick else (0, 1, 2),
                                                      backend=args.backend)
    (out / "coeffs.npy").unlink(missing_ok=True)
    (out / "error_floor.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
