"""Direct comparison with iDVC's engine (CCPi DVC) on case A: how close, and why not identical.

iDVC runs the CCPi ``dvc`` executable on a ``dvc_in`` file it writes. This study runs that
executable the same way (one process, every point in CCPi's order) and zvDVC in its CCPi-parity
mode (whole volumes in memory, wavefront seeding in CCPi's point order), then compares every
point.

Two findings from CCPi's source (v22.0.0) shape the design:

* **Sphere subvolumes are re-sampled at random for every point, and on every run.** ``dvc``
  calls ``Search::process_point`` with ``test = 0``, so ``FloatingCloud`` seeds its
  ``std::mt19937`` from ``system_clock::now()``. CCPi therefore cannot reproduce its own sphere
  results, and nothing can match them bit for bit. The spread this causes is measured here by
  running CCPi twice and zvDVC with two template seeds.
* **Cube subvolumes are a fixed grid**, the same ``k³`` points in both codes. With cubes, CCPi
  and zvDVC sample identical positions, so what differs is arithmetic alone: CCPi's float64
  finite-difference Gauss–Newton against zvDVC's analytic Jacobian (float32 on the GPU, float64
  in the numpy engine), and where each stops within CCPi's tolerances
  (``|d obj| < 1e-6`` or ``|d t| < 0.01``). A CCPi build with those tolerances tightened
  (``tight.patch``, research only) against zvDVC with the same tolerances measures the floor.

Usage (from the repository root, after the CCPi runs in ``<out>/ccpi`` have finished)::

    python -m zvdvc.bench.idvc_study run     --config runs/case_A/config.yaml --out runs/idvc_study
    python -m zvdvc.bench.idvc_study analyze --config runs/case_A/config.yaml --out runs/idvc_study \\
        --figures docs/benchmarks/figures/2026-09-29-idvc

``analyze`` needs matplotlib for the figures.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from zvdvc.config import RunConfig

# name -> (geometry, template seed, backend, tolerances); "ccpi" tolerances are CCPi's hard-coded ones
TIGHT = {"max_iterations": 200, "obj_tol": 1e-15, "disp_tol": 1e-9}
VARIANTS: dict[str, dict[str, Any]] = {
    # float32 fused kernels on the CPU (numba): the same arithmetic as the GPU engine, without the GPU
    "Z_sphere_s0_cpu":    {"geometry": "sphere", "seed": 0, "backend": "cpu"},
    "Z_sphere_s1_cpu":    {"geometry": "sphere", "seed": 1, "backend": "cpu"},
    "Z_cube_cpu":         {"geometry": "cube", "backend": "cpu"},
    "Z_cube_tight_cpu":   {"geometry": "cube", "backend": "cpu", "tolerances": TIGHT},
    # float64 reference engine
    "Z_cube_numpy":       {"geometry": "cube", "backend": "numpy"},
    "Z_cube_tight_numpy": {"geometry": "cube", "backend": "numpy", "tolerances": TIGHT},
    # the GPU (needs ~5 GB free for case A's two whole volumes)
    "Z_sphere_s0_fused":  {"geometry": "sphere", "seed": 0, "backend": "fused"},
    "Z_cube_fused":       {"geometry": "cube", "backend": "fused"},
}
DEFAULT_VARIANTS = [n for n, v in VARIANTS.items() if v["backend"] != "fused"]


def variant_config(cfg: RunConfig, spec: dict[str, Any]) -> RunConfig:
    sub = dataclasses.replace(cfg.subvolume, geometry=spec["geometry"], seed=spec.get("seed", cfg.subvolume.seed))
    search = dataclasses.replace(cfg.search, report_convg_fail=False, **spec.get("tolerances", {}))
    return dataclasses.replace(cfg, subvolume=sub, search=search)


def run_zvdvc(cfg: RunConfig, out: Path, names: list[str]) -> dict[str, Any]:
    """Solve case A once per variant, in CCPi's point order; write ``<name>.npz`` and ``<name>.disp``."""
    from zvdvc.pipeline.inmemory import load_points, solve_in_memory

    out.mkdir(parents=True, exist_ok=True)
    point_id, xyz = load_points(cfg)
    log_path = out / "zvdvc_runs.json"
    log: dict[str, Any] = json.loads(log_path.read_text()) if log_path.exists() else {}
    for name in names:
        spec = VARIANTS[name]
        vcfg = variant_config(cfg, spec)
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:     # e.g. a GPU out-of-memory fallback to the cpu engine
            warnings.simplefilter("always")
            res = solve_in_memory(vcfg, point_id, xyz, strategy="wavefront", backend=spec["backend"])
        seconds = time.perf_counter() - t0
        res.save(out / f"{name}.npz")
        res.write_disp(out / f"{name}.disp")
        notes = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        log[name] = {"seconds": seconds, "status_counts": res.status_counts(),
                     "mean_iterations": float(res.n_iter[res.status == 0].mean()), **spec, "warnings": notes}
        print(f"{name}: {seconds:.1f} s, {res.status_counts()}, mean iterations {log[name]['mean_iterations']:.2f}"
              + "".join(f"\n  warning: {n}" for n in notes), flush=True)
        log_path.write_text(json.dumps(log, indent=2))
    return log


# -- analysis --------------------------------------------------------------------------------------

# CCPi results: name -> (path relative to <out>, subvolume geometry)
CCPI_RUNS: dict[str, tuple[str, str]] = {
    "C_sphere_0926":    ("../case_A/ccpi_ccpi-dvc-22.0.0/t32_p1.disp", "sphere"),   # the 2026-09-26 case A run
    "C_sphere":         ("ccpi/R1_release_sphere.disp", "sphere"),
    "C_cube":           ("ccpi/R2_release_cube.disp", "cube"),
    "C_cube_500_rerun": ("ccpi/R3_release_cube_500.disp", "cube"),
    "C_cube_500_src":   ("ccpi/R4_src_cube_500.disp", "cube"),
    "C_cube_tight":     ("ccpi/R5_tight_cube.disp", "cube"),
}

# (group, label, a, b): every comparison the report shows, grouped by what can make them differ
PAIRS: list[tuple[str, str, str, str]] = [
    ("sampling", "CCPi vs CCPi: sphere, two runs", "C_sphere", "C_sphere_0926"),
    ("sampling", "zvDVC vs zvDVC: sphere, template seeds 0 and 1", "Z_sphere_s0_cpu", "Z_sphere_s1_cpu"),
    ("sampling", "zvDVC vs CCPi: sphere (iDVC's settings)", "Z_sphere_s0_cpu", "C_sphere"),
    ("sampling", "zvDVC GPU vs zvDVC CPU engine: sphere (same template)", "Z_sphere_s0_fused", "Z_sphere_s0_cpu"),
    ("same samples", "CCPi vs CCPi: cube, rerun on 1 thread (first 500 points)", "C_cube_500_rerun", "C_cube"),
    ("same samples", "CCPi release vs CCPi built from source: cube (first 500)", "C_cube_500_src", "C_cube"),
    ("same samples", "zvDVC float64 vs CCPi: cube", "Z_cube_numpy", "C_cube"),
    ("same samples", "zvDVC float32 vs CCPi: cube", "Z_cube_cpu", "C_cube"),
    ("same samples", "zvDVC float32 vs zvDVC float64: cube", "Z_cube_cpu", "Z_cube_numpy"),
    ("same samples", "zvDVC GPU vs zvDVC CPU engine (both float32): cube", "Z_cube_fused", "Z_cube_cpu"),
    ("same samples", "zvDVC GPU vs CCPi: cube", "Z_cube_fused", "C_cube"),
    ("converged", "CCPi vs CCPi run to convergence: cube", "C_cube", "C_cube_tight"),
    ("converged", "zvDVC float64 vs itself run to convergence: cube", "Z_cube_numpy", "Z_cube_tight_numpy"),
    ("converged", "zvDVC float64 vs CCPi, both run to convergence: cube", "Z_cube_tight_numpy", "C_cube_tight"),
    ("converged", "zvDVC float32 vs CCPi, both run to convergence: cube", "Z_cube_tight_cpu", "C_cube_tight"),
]


def _load(out: Path, name: str) -> dict[str, Any] | None:
    """One result as CCPi-coded arrays, or None if that run has not been made."""
    from zvdvc.bench.metrics import load_results
    from zvdvc.status import PointStatus

    if name in CCPI_RUNS:
        rel, geometry = CCPI_RUNS[name]
        path = (out / rel).resolve()
        if not path.exists():
            return None
        r = load_results(path)
        r["status_ccpi"] = r["status"]
    else:
        path = out / "zvdvc" / f"{name}.npz"
        if not path.exists():
            return None
        r = load_results(path)
        r["status_ccpi"] = np.array([PointStatus(int(s)).to_ccpi() for s in r["status"]])
        geometry = VARIANTS[name]["geometry"]
    r["geometry"], r["name"], r["path"] = geometry, name, str(path)
    return r


def compare_pair(a: dict[str, Any], b: dict[str, Any], cfg: RunConfig) -> dict[str, Any]:
    """Point-by-point agreement of two results on their common points.

    Displacements are compared on points GOOD in both and away from the image edge for both
    subvolume shapes (``edge_mask``: CCPi reads outside the image there). "Identical" means
    equal at the ``.disp`` precision CCPi prints (6 decimals), the only form iDVC ever sees.
    """
    from zvdvc.bench.compare_ccpi import _edge_geometry
    from zvdvc.bench.metrics import edge_mask

    common, ia, ib = np.intersect1d(a["point_id"], b["point_id"], return_indices=True)
    ua, ub = a["displacement"][ia].astype(np.float64), b["displacement"][ib].astype(np.float64)
    xyz = a["xyz"][ia]
    edge = np.zeros(len(common), dtype=bool)
    for r, u in ((a, ua), (b, ub)):
        shape_xyz, reach = _edge_geometry(variant_config(cfg, {"geometry": r["geometry"], "backend": "numpy"}))
        edge |= edge_mask(xyz, u, shape_xyz, reach)
    sa, sb = a["status_ccpi"][ia], b["status_ccpi"][ib]
    good = (sa == 0) & (sb == 0) & ~edge
    d = ua[good] - ub[good]
    mag = np.sqrt((d**2).sum(axis=1))
    same6 = (np.round(ua, 6) == np.round(ub, 6)).all(axis=1)
    q = (lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan"))
    return {
        "points": int(len(common)),
        "edge_points": int(edge.sum()),
        "status_agreement_all": float((sa == sb).mean()),
        "status_agreement_interior": float((sa[~edge] == sb[~edge]).mean()) if (~edge).any() else float("nan"),
        "compared": int(good.sum()),
        "identical_6dp": int(same6[good].sum()),
        "identical_6dp_all_points": int((same6 & (sa == sb)).sum()),
        "median": q(mag, 50), "p95": q(mag, 95), "p99": q(mag, 99),
        "max": float(mag.max()) if len(mag) else float("nan"),
        "mean_per_axis": d.mean(axis=0).tolist() if len(d) else [float("nan")] * 3,
        "max_abs_per_axis": np.abs(d).max(axis=0).tolist() if len(d) else [float("nan")] * 3,
        "objmin_rel_median": float(np.median(np.abs(a["objmin"][ia][good] - b["objmin"][ib][good])
                                             / np.maximum(np.abs(b["objmin"][ib][good]), 1e-12))) if good.any() else float("nan"),
        "_mag": mag, "_ids": common[good], "_good": good, "_common": common,
    }


def analyze(cfg: RunConfig, out: Path, figures: Path | None) -> dict[str, Any]:
    names = {n for _, _, a, b in PAIRS for n in (a, b)}
    runs = {n: r for n in sorted(names) if (r := _load(out, n)) is not None}
    missing = sorted(names - set(runs))
    rows = []
    for group, label, a, b in PAIRS:
        if a in runs and b in runs:
            rows.append({"group": group, "label": label, "a": a, "b": b, **compare_pair(runs[a], runs[b], cfg)})
    report = {"missing_runs": missing, "runs": {n: r["path"] for n, r in runs.items()},
              "pairs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]}
    (out / "study.json").write_text(json.dumps(report, indent=2))
    (out / "study.md").write_text(markdown_tables(report))
    if figures is not None:
        figures.mkdir(parents=True, exist_ok=True)
        plot_all(runs, rows, figures)
    return report


def _fmt(v: float) -> str:
    if v != v:
        return "–"
    if v == 0:
        return "0"
    return f"{v:.1e}" if v < 1e-3 else f"{v:.4f}"


def markdown_tables(report: dict[str, Any]) -> str:
    lines = ["| group | comparison | points compared | identical to 6 dp | median \\|Δu\\| | p95 | max | mean Δ (x, y, z) | status agreement (interior) |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for r in report["pairs"]:
        mean = ", ".join(f"{m:+.1e}" for m in r["mean_per_axis"])
        lines.append(f"| {r['group']} | {r['label']} | {r['compared']:,} | {r['identical_6dp']:,} ({r['identical_6dp'] / max(r['compared'], 1):.1%}) "
                     f"| {_fmt(r['median'])} | {_fmt(r['p95'])} | {_fmt(r['max'])} | {mean} | {r['status_agreement_interior']:.2%} |")
    if report["missing_runs"]:
        lines.append(f"\nNot yet run: {', '.join(report['missing_runs'])}")
    return "\n".join(lines) + "\n"


# -- figures ---------------------------------------------------------------------------------------
# Reference palette (dataviz skill): categorical slots 1–4 validated for light mode (adjacent CVD ΔE ≥ 9.1);
# slots 3–4 sit below 3:1 contrast on the surface, so every line carries a direct label.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a3a2", "#e34948", "#a8201f"]


def _style(plt: Any) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    })


def _grid_image(r: dict[str, Any], values: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Scatter ``values`` at points ``ids`` onto the (y, x) grid of case A's single-plane point cloud.

    ``r`` supplies the positions and the grid: a result holding every point (rows in any order).
    """
    order = np.argsort(r["point_id"])
    rows = order[np.searchsorted(r["point_id"], ids, sorter=order)]
    xy = r["xyz"][rows][:, :2]
    xs, ys = np.unique(r["xyz"][:, 0]), np.unique(r["xyz"][:, 1])
    img = np.full((len(ys), len(xs)), np.nan)
    img[np.searchsorted(ys, xy[:, 1]), np.searchsorted(xs, xy[:, 0])] = values
    step = (xs[1] - xs[0]) / 2 if len(xs) > 1 else 0.5
    return img, [xs[0] - step, xs[-1] + step, ys[-1] + step, ys[0] - step]


def _row(r: dict[str, Any], ids: np.ndarray) -> np.ndarray:
    order = np.argsort(r["point_id"])
    return order[np.searchsorted(r["point_id"], ids, sorter=order)]


def plot_all(runs: dict[str, dict[str, Any]], rows: list[dict[str, Any]], figures: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm

    _style(plt)
    seq = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    seq.set_bad("#f0efec")
    div = LinearSegmentedColormap.from_list("div", DIVERGING)
    div.set_bad("#f0efec")
    by_label = {r["label"]: r for r in rows}
    grid = max(runs.values(), key=lambda r: len(r["point_id"]))     # a result holding every point

    # 1. Error budget: median (dot) to p95 (bar end) of |Δu| for every comparison, log scale.
    groups = [r["group"] for r in rows]
    # one blank row above each group for its heading
    y, cur = [], 0.0
    for i, g in enumerate(groups):
        if i == 0 or g != groups[i - 1]:
            cur += 1.0
        y.append(-cur)
        cur += 1.0
    y = np.array(y)
    fig, ax = plt.subplots(figsize=(9.6, 0.36 * (cur + 1) + 1.2))
    floor = 1e-7
    for yi, r in zip(y, rows):
        if r["compared"] == 0:
            continue
        if r["identical_6dp"] == r["compared"]:
            ax.text(floor * 1.3, yi, f"identical at .disp precision: {r['compared']:,} of {r['compared']:,} points",
                    va="center", ha="left", color=INK2, fontsize=8.5)
            continue
        lo, hi = max(r["median"], floor), max(r["p95"], floor)
        ax.plot([lo, hi], [yi, yi], color=SERIES[0], lw=2, solid_capstyle="round", zorder=2)
        ax.plot(lo, yi, "o", ms=6.5, color=SERIES[0], mec=SURFACE, mew=1.5, zorder=3)
        ax.text(hi * 1.4, yi, f"{_fmt(r['median'])} / {_fmt(r['p95'])}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(floor, 20.0)
    ax.set_ylim(y.min() - 0.7, 0.3)
    ax.axvline(1e-6, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(1.1e-6, y.min() - 0.6, ".disp resolution", color=MUTED, fontsize=7.5, va="bottom")
    ax.grid(axis="x", which="major")
    ax.set_xlabel("|Δu| between the two results, voxels (log scale).  Dot: median.  Bar end: 95th percentile.")
    for i, g in enumerate(groups):
        if i == 0 or g != groups[i - 1]:
            ax.text(floor, y[i] + 1.0, {"sampling": "Different sample points (sphere subvolumes)",
                                        "same samples": "Same sample points (cube subvolumes), default stopping",
                                        "converged": "Same sample points, run to convergence"}[g].upper(),
                    fontsize=7.5, color=MUTED, va="center", fontweight="bold")
    fig.suptitle("zvDVC against CCPi DVC (iDVC's engine) on case A: how far apart, and why", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "error_budget.png", dpi=160)
    plt.close(fig)

    # 2. Distributions: CDF of |Δu| for the four headline comparisons. A legend carries identity (the curves
    # leave no room for direct labels); line style is the secondary encoding for the two sub-3:1 colours.
    picks = [("zvDVC vs CCPi: sphere (iDVC's settings)", "zvDVC vs CCPi, sphere (iDVC's settings)"),
             ("CCPi vs CCPi: sphere, two runs", "CCPi vs CCPi, sphere (two runs)"),
             ("zvDVC float64 vs CCPi: cube", "zvDVC vs CCPi, cube (same sample points)"),
             ("zvDVC float64 vs CCPi, both run to convergence: cube", "zvDVC vs CCPi, cube, both run to convergence")]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for i, (label, short) in enumerate(p for p in picks if p[0] in by_label):
        mag = np.sort(np.maximum(by_label[label]["_mag"], 1e-8))
        frac = np.arange(1, len(mag) + 1) / len(mag)
        ax.plot(mag, frac, color=SERIES[i], lw=2, ls=["-", "--", "-", "--"][i], label=short)
    if picks[0][0] in by_label and picks[1][0] in by_label:
        r = by_label[picks[0][0]]
        ax.annotate("the two sphere curves coincide", (r["median"], 0.5), xytext=(12, -30), textcoords="offset points",
                    fontsize=8, color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 3)
    ax.set_ylim(0, 1.02)
    ax.axvline(1e-6, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(1.15e-6, 0.03, ".disp resolution", color=MUTED, fontsize=7.5)
    ax.legend(loc="upper left", fontsize=8, handlelength=2.6, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1)
    ax.grid(True, which="major")
    ax.set_xlabel("|Δu| (voxels, log scale)")
    ax.set_ylabel("fraction of interior points")
    ax.set_title("Cumulative distribution of the point-by-point difference, case A", loc="left")
    fig.tight_layout()
    fig.savefig(figures / "difference_cdf.png", dpi=160)
    plt.close(fig)

    # 3. Fields: CCPi, zvDVC and zvDVC − CCPi for u, v, w (cube, default stopping, float64).
    pair = by_label.get("zvDVC float64 vs CCPi: cube")
    if pair is not None:
        a, b = runs[pair["a"]], runs[pair["b"]]
        ids = pair["_ids"]
        ua, ub = a["displacement"][_row(a, ids)], b["displacement"][_row(b, ids)]
        fig, axes = plt.subplots(3, 3, figsize=(10.5, 6.6), constrained_layout=True)
        for k, comp in enumerate(["u (x)", "v (y)", "w (z)"]):
            lo, hi = np.percentile(np.concatenate([ua[:, k], ub[:, k]]), [1, 99])
            for j, (vals, title) in enumerate([(ub[:, k], "CCPi DVC 22.0.0"), (ua[:, k], "zvDVC (float64)")]):
                img, ext = _grid_image(grid, vals, ids)
                im = axes[k, j].imshow(img, extent=ext, cmap=seq, vmin=lo, vmax=hi, interpolation="nearest")
                axes[k, j].set_title(f"{title}: {comp}", fontsize=9)
            fig.colorbar(im, ax=axes[k, :2], shrink=0.85, label="voxels")
            dd = ua[:, k] - ub[:, k]
            lim = float(np.percentile(np.abs(dd), 99)) or 1e-6
            img, ext = _grid_image(grid, dd, ids)
            im = axes[k, 2].imshow(img, extent=ext, cmap=div, norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
            axes[k, 2].set_title(f"zvDVC − CCPi: {comp}", fontsize=9)
            fig.colorbar(im, ax=axes[k, 2], shrink=0.85, label="voxels")
        for axx in axes.ravel():
            axx.set_xlabel("x (voxels)", fontsize=8)
            axx.set_ylabel("y", fontsize=8)
            axx.tick_params(labelsize=7)
        fig.suptitle("Case A displacement field, cube subvolumes (same sample points in both codes); z = 630 plane; "
                     "grey = edge or failed points", fontsize=10, x=0.01, ha="left")
        fig.savefig(figures / "fields_cube.png", dpi=150)
        plt.close(fig)

    # 4. Where the differences are: |Δu| maps on one log colour scale, 2 x 2.
    maps = [(label, short) for label, short in [
        ("zvDVC vs CCPi: sphere (iDVC's settings)", "zvDVC vs CCPi, sphere"),
        ("CCPi vs CCPi: sphere, two runs", "CCPi vs CCPi, sphere (two runs)"),
        ("zvDVC float64 vs CCPi: cube", "zvDVC vs CCPi, cube"),
        ("zvDVC float64 vs CCPi, both run to convergence: cube", "zvDVC vs CCPi, cube, both converged")]
        if label in by_label]
    if maps:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.4), constrained_layout=True)
        norm = LogNorm(1e-6, 1.0)
        for axx, (label, short) in zip(axes.ravel(), maps):
            r = by_label[label]
            img, ext = _grid_image(grid, np.maximum(r["_mag"], 1e-6), r["_ids"])
            im = axx.imshow(img, extent=ext, cmap=seq, norm=norm, interpolation="nearest")
            axx.set_title(f"{short}: median {_fmt(r['median'])}", fontsize=9, loc="left")
            axx.tick_params(labelsize=7)
        for axx in axes.ravel()[len(maps):]:
            axx.axis("off")
        fig.colorbar(im, ax=axes, shrink=0.85, label="|Δu| (voxels, log scale)")
        fig.suptitle("Where the two results differ (z = 630 plane; grey = edge or failed points)", fontsize=10, x=0.01, ha="left")
        fig.savefig(figures / "difference_maps.png", dpi=150)
        plt.close(fig)

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m zvdvc.bench.idvc_study", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="zvDVC variants on case A (CCPi's runs are separate: see the module docstring)")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    a = sub.add_parser("analyze", help="compare every available pair; write study.json, study.md and the figures")
    a.add_argument("--config", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--figures", help="directory for the PNG figures (needs matplotlib)")
    args = p.parse_args(argv)
    cfg = RunConfig.from_yaml(args.config)
    if args.command == "run":
        run_zvdvc(cfg, Path(args.out) / "zvdvc", args.variants)
    else:
        report = analyze(cfg, Path(args.out), Path(args.figures) if args.figures else None)
        print(markdown_tables(report))


if __name__ == "__main__":
    main()
