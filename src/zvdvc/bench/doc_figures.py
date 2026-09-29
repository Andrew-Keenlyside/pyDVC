"""The documentation's validation and benchmark summary figures, rebuilt from saved run data.

::

    python -m zvdvc.bench.doc_figures --runs runs --docs docs

writes ``docs/validation/figures/*.png`` and ``docs/benchmarks/figures/summary/*.png``. Each figure
is skipped, with a note, when its data has not been made. The data comes from:

``runs/idvc_study``
    :mod:`zvdvc.bench.idvc_study`: CCPi DVC 22.0.0 as iDVC runs it, and zvDVC, on case A.
``runs/case_A/report.json``
    :mod:`zvdvc.bench.case_a`: wall times of CCPi as iDVC runs it and of zvDVC on case A.
``runs/validation/<case>/``
    synthetic cases with a known field, each solved by CCPi and zvDVC
    (:mod:`zvdvc.bench.compare_ccpi` with ``--truth``).
``runs/error_floor``, ``runs/error_floor_prefilter``
    :mod:`zvdvc.bench.error_floor`: bias against sub-voxel shifts, and the sample-count spread.
``runs/gds``
    :mod:`zvdvc.bench.gds`: brick reads on the host and kvikio paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from zvdvc.bench.plotstyle import (BLUE_RAMP, GRID, INK, INK2, MUTED, SERIES, SURFACE, fields_figure, grid_image,
                                   row_of, style)

CCPI, ZVDVC, CCPI_RERUN = SERIES[1], SERIES[0], SERIES[2]      # one colour per code, the same in every figure


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    return plt


def _save(fig: Any, path: Path, made: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    import matplotlib.pyplot as plt

    plt.close(fig)
    made.append(str(path))


# ------------------------------------------------------------------------------------ agreement with iDVC


def idvc_figures(runs: Path, out: Path, made: list[str], skipped: list[str]) -> None:
    """Case A with iDVC's settings (sphere subvolumes): zvDVC against CCPi, and CCPi against itself."""
    from zvdvc.bench.idvc_study import _load, compare_pair
    from zvdvc.config import RunConfig

    study = runs / "idvc_study"
    cfg_path = runs / "case_A" / "config.yaml"
    loaded = {n: _load(study, n) for n in ("C_sphere", "C_sphere_0926", "Z_sphere_s0_cpu")}
    if any(v is None for v in loaded.values()) or not cfg_path.exists():
        skipped.append("iDVC agreement figures: run zvdvc.bench.idvc_study first")
        return
    cfg = RunConfig.from_yaml(cfg_path)
    ccpi, rerun, zv = loaded["C_sphere"], loaded["C_sphere_0926"], loaded["Z_sphere_s0_cpu"]
    pz, pc = compare_pair(zv, ccpi, cfg), compare_pair(rerun, ccpi, cfg)
    ids = np.intersect1d(pz["_ids"], pc["_ids"])
    u_c, u_z, u_r = (r["displacement"][row_of(r, ids)] for r in (ccpi, zv, rerun))
    plt = _plt()
    comps = ["u (x)", "v (y)", "w (z)"]

    # measured against measured, per component
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.1))
    for k, ax in enumerate(axes):
        x, y = u_c[:, k], u_z[:, k]
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.scatter(x, y, s=3, color=ZVDVC, alpha=0.5, lw=0, zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.grid(True)
        d = y - x
        ax.set_title(f"{comps[k]}\nmedian |Δ| {np.median(np.abs(d)):.3f}, mean Δ {d.mean():+.4f}", fontsize=8.5, loc="left")
        ax.set_xlabel("iDVC's engine (voxels)")
        if k == 0:
            ax.set_ylabel("zvDVC (voxels)")
    fig.suptitle(f"zvDVC against iDVC's engine (CCPi DVC 22.0.0) on iDVC's example data: {len(ids):,} interior "
                 "points, iDVC's settings; dashed: equality", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _save(fig, out / "idvc_scatter.png", made)

    # the differences, against CCPi's own run-to-run differences
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for k, ax in enumerate(axes):
        d_z, d_r = u_z[:, k] - u_c[:, k], u_r[:, k] - u_c[:, k]
        lim = float(np.percentile(np.abs(np.concatenate([d_z, d_r])), 99.5))
        bins = np.linspace(-lim, lim, 61)
        ax.hist(d_r, bins=bins, histtype="step", lw=2, color=CCPI_RERUN, label="iDVC's engine, run again")
        ax.hist(d_z, bins=bins, histtype="step", lw=2, color=ZVDVC, ls=(0, (4, 2)), label="zvDVC")
        ax.axvline(0, color=MUTED, lw=1)
        ax.grid(True, axis="y")
        ax.set_title(f"{comps[k]}: SD {d_z.std():.3f} (zvDVC), {d_r.std():.3f} (iDVC again)", fontsize=8.5, loc="left")
        ax.set_xlabel("difference from the iDVC run (voxels)")
        if k == 0:
            ax.set_ylabel("points")
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Differences from one iDVC run: zvDVC's, and a second iDVC run's (case A, iDVC's settings)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "idvc_differences.png", made)

    fields_figure(ccpi, ids, u_c, u_z, ("iDVC's engine (CCPi 22.0.0)", "zvDVC"), out / "idvc_fields_sphere.png",
                  "Case A displacement field with iDVC's settings (sphere subvolumes), z = 630 plane; "
                  "difference = zvDVC − iDVC; grey = edge points")
    made.append(str(out / "idvc_fields_sphere.png"))


# ------------------------------------------------------------------------------------ ground truth


def _synthetic(case: Path) -> dict[str, Any] | None:
    from zvdvc.bench.metrics import load_results

    zv = sorted((case / "compare").glob("zvdvc_*_parity.npz"))
    # the single-process run with the most threads: how iDVC runs CCPi
    cc = sorted((case / "compare").glob("ccpi_*/t*_p1.disp"), key=lambda p: -int(p.stem.split("_")[0][1:]))
    if not zv or not cc or not (case / "truth.npz").exists():
        return None
    with np.load(case / "truth.npz") as t:
        truth = {"point_id": t["point_id"], "displacement": t["displacement"]}
    return {"zvdvc": load_results(zv[0]), "ccpi": load_results(cc[0]), "truth": truth}


def _errors(res: dict[str, Any], truth: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    """Point ids and |error| of GOOD points, and the fraction GOOD."""
    ids, ia, ib = np.intersect1d(res["point_id"], truth["point_id"], return_indices=True)
    good = res["status"][ia] == 0
    err = res["displacement"][ia] - truth["displacement"][ib]
    return ids[good], np.sqrt((err[good] ** 2).sum(axis=1)), float(good.mean())


CASES = {"S_affine_noise2": "affine field, 2 % noise, 12-DOF", "S_sinusoid": "sinusoidal field, noise-free, 6-DOF"}


def truth_figures(runs: Path, out: Path, made: list[str], skipped: list[str]) -> None:
    cases = {name: d for name in CASES if (d := _synthetic(runs / "validation" / name)) is not None}
    if not cases:
        skipped.append("ground-truth figures: run runs/validation/run_synth.sh first")
        return
    plt = _plt()
    fig, axes = plt.subplots(1, len(cases), figsize=(5.4 * len(cases), 3.6), squeeze=False)
    for ax, (name, d) in zip(axes[0], cases.items()):
        for code, color, ls in (("ccpi", CCPI, "-"), ("zvdvc", ZVDVC, (0, (4, 2)))):
            _, mag, frac = _errors(d[code], d["truth"])
            rmse = float(np.sqrt(np.mean(mag**2)))
            label = f"{'iDVC' + chr(39) + 's engine' if code == 'ccpi' else 'zvDVC'}: RMS |error| {rmse:.4f}, {frac:.1%} GOOD"
            m = np.sort(mag)
            ax.plot(m, np.arange(1, len(m) + 1) / len(m), color=color, lw=2, ls=ls, label=label)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.grid(True)
        ax.set_xlabel("|measured − true displacement| (voxels, log scale)")
        ax.set_title(f"256³ synthetic, {CASES[name]}", fontsize=9, loc="left")
        ax.legend(loc="upper left", fontsize=8)
    axes[0][0].set_ylabel("fraction of points")
    fig.suptitle("Accuracy against a known displacement field: zvDVC and iDVC's engine (CCPi DVC 22.0.0)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "truth_accuracy.png", made)

    if "S_sinusoid" in cases:
        _truth_maps(plt, cases["S_sinusoid"], out, made)


def _truth_maps(plt: Any, d: dict[str, Any], out: Path, made: list[str]) -> None:
    """The sinusoid case's middle z plane: the true field's magnitude, and each code's error."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    seq = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    seq.set_bad("#f0efec")
    xyz = d["zvdvc"]["xyz"]
    z_mid = np.unique(xyz[:, 2])[len(np.unique(xyz[:, 2])) // 2]
    plane = {"point_id": d["zvdvc"]["point_id"][xyz[:, 2] == z_mid], "xyz": xyz[xyz[:, 2] == z_mid]}
    t_ids = np.intersect1d(plane["point_id"], d["truth"]["point_id"])
    t_rows = row_of(d["truth"], t_ids)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    mag = np.sqrt((d["truth"]["displacement"][t_rows] ** 2).sum(axis=1))
    img, ext = grid_image(plane, mag, t_ids)
    im = axes[0].imshow(img, extent=ext, cmap=seq, interpolation="nearest")
    axes[0].set_title("true |u| (voxels)", fontsize=9, loc="left")
    fig.colorbar(im, ax=axes[0], shrink=0.85)
    norm = LogNorm(1e-4, 1e-1)
    for ax, code, name in ((axes[1], "zvdvc", "zvDVC"), (axes[2], "ccpi", "iDVC's engine")):
        ids, err, _ = _errors(d[code], d["truth"])
        keep = np.isin(ids, plane["point_id"])
        img, ext = grid_image(plane, np.clip(err[keep], 1e-4, None), ids[keep])
        im = ax.imshow(img, extent=ext, cmap=seq, norm=norm, interpolation="nearest")
        ax.set_title(f"{name}: |error| (median {np.median(err):.4f})", fontsize=9, loc="left")
    fig.colorbar(im, ax=axes[1:], shrink=0.85, label="|error| (voxels, log scale)")
    for ax in axes:
        ax.set_xlabel("x (voxels)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Sinusoidal field, plane z = {z_mid:g}: the true field and each code's error", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    _save(fig, out / "truth_error_maps.png", made)


# ------------------------------------------------------------------------------------ error floor


def error_floor_figures(runs: Path, out: Path, made: list[str], skipped: list[str]) -> None:
    plt = _plt()
    bias_path = runs / "error_floor_prefilter" / "bias.json"
    if bias_path.exists():
        bias = json.loads(bias_path.read_text())
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        for i, (sigma, values) in enumerate(sorted(bias.items(), key=lambda kv: float(kv[0]))):
            shifts = np.linspace(0.0, 1.0, len(values))
            label = "no prefilter (CCPi's behaviour)" if float(sigma) == 0 else f"prefilter σ = {float(sigma):g}"
            ax.plot(shifts, values, color=SERIES[i], lw=2, marker="o", ms=4, label=label)
        ax.axhline(0, color=MUTED, lw=1)
        ax.grid(True)
        ax.set_xlabel("imposed sub-voxel shift (voxels)")
        ax.set_ylabel("mean measured − imposed (voxels)")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title("Interpolation bias against known shifts of the case A scan (tricubic)", loc="left")
        fig.tight_layout()
        _save(fig, out / "interpolation_bias.png", made)
    else:
        skipped.append("interpolation bias: run zvdvc.bench.error_floor with --prefilter-sigma")

    floor_path = runs / "error_floor" / "error_floor.json"
    spread = json.loads(floor_path.read_text()).get("seed_repeatability_vs_n_samples") if floor_path.exists() else None
    if spread:
        n = np.array(sorted(int(k) for k in spread))
        s = np.array([spread[str(k)] for k in n])
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ref = s[n == 8000][0] * np.sqrt(8000 / n) if (n == 8000).any() else s[0] * np.sqrt(n[0] / n)
        ax.plot(n, ref, color=GRID, lw=7, solid_capstyle="round", label="1/√n through the 8 000 point", zorder=1)
        ax.plot(n, s, color=ZVDVC, lw=2, marker="o", ms=6, label="measured", zorder=2)
        if (n == 8000).any():
            ax.annotate("iDVC's default (8 000)", (8000, s[n == 8000][0]), xytext=(10, 10), textcoords="offset points",
                        fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(n, [f"{v:,}" for v in n])
        ax.set_yticks(s, [f"{v:.3f}" for v in s])
        ax.minorticks_off()
        ax.grid(True)
        ax.set_xlabel("sample points per subvolume")
        ax.set_ylabel("per-axis σ of one estimate (voxels)")
        ax.legend(fontsize=8)
        ax.set_title("Uncertainty from the random choice of sample points, case A", loc="left")
        fig.tight_layout()
        _save(fig, out / "sampling_spread.png", made)
    else:
        skipped.append("sampling spread: runs/error_floor/error_floor.json has no seed_repeatability_vs_n_samples")


# ------------------------------------------------------------------------------------ benchmarks


def speed_figure(runs: Path, out: Path, made: list[str], skipped: list[str]) -> None:
    path = runs / "case_A" / "report.json"
    if not path.exists():
        skipped.append("case A speed: run zvdvc.bench.case_a first")
        return
    rep = json.loads(path.read_text())
    rows = []
    for r in rep["runs"]:
        name = r["name"].replace("pyDVC", "zvDVC")
        is_ccpi = name.startswith("CCPi")
        if is_ccpi:
            label = "iDVC's engine, as iDVC runs it (1 process, 32 threads)" if "iDVC" in name else \
                "iDVC's engine, 32 processes × 1 thread"
        else:
            engine = "GPU (RTX A2000)" if "fused" in name else "CPU, 32 cores"
            label = f"zvDVC {engine}, {'in memory' if 'parity' in name else 'CLI end to end'}"
        rows.append((label, float(r["seconds"]), is_ccpi))
    base = next(s for label, s, c in rows if c and "as iDVC" in label)
    rows.sort(key=lambda r: -r[1])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    y = np.arange(len(rows))[::-1]
    for yi, (label, s, is_ccpi) in zip(y, rows):
        ax.barh(yi, s, height=0.62, color=CCPI if is_ccpi else ZVDVC)
        text = f"{s:,.0f} s" if s >= 100 else f"{s:.1f} s"
        if s < base:
            text += f"  ({base / s:,.0f}× faster)"
        ax.text(s * 1.15, yi, text, va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.5, base * 12)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x")
    ax.set_xlabel("wall time for 4 680 points (seconds, log scale)")
    fig.suptitle("Case A, iDVC's example data and settings: iDVC's engine against zvDVC", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CCPI), plt.Rectangle((0, 0), 1, 1, color=ZVDVC)]
    ax.legend(handles, ["CCPi DVC 22.0.0 (iDVC's engine)", "zvDVC"], fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, out / "speed_case_a.png", made)


def gds_figure(runs: Path, out: Path, made: list[str], skipped: list[str]) -> None:
    files = {("raw", "warm"): "caseA_raw.json", ("raw", "cold"): "caseA_raw--cold.json",
             ("OME-Zarr, zstd", "warm"): "caseA_zarr.json", ("OME-Zarr, zstd", "cold"): "caseA_zarr--cold.json"}
    data = {k: json.loads((runs / "gds" / f).read_text()) for k, f in files.items() if (runs / "gds" / f).exists()}
    if not data:
        skipped.append("GDS read paths: run zvdvc.bench.gds first")
        return
    gds_on = any(d["gpu_io_status"]["gds_available"] for d in data.values())
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    keys = list(data)
    y = np.arange(len(keys))[::-1] * 1.0
    h = 0.36
    # orange is iDVC's engine in the other figures, so the kvikio path takes the next slot (aqua; values are labelled)
    for off, mode, color, name in ((h / 2, "host", SERIES[0], "host decode, one copy up"),
                                   (-h / 2, "kvikio", SERIES[2], "kvikio" + (" (GPUDirect Storage)" if gds_on else
                                                                              ", no GPUDirect Storage (compatibility mode)"))):
        vals = [data[k]["modes"][mode]["gb_per_s_median"] for k in keys]
        ax.barh(y + off, vals, height=h * 0.92, color=color, label=name)
        for yi, v in zip(y + off, vals):
            ax.text(v + 0.06, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [f"{k[0]}, {k[1]} cache" for k in keys], fontsize=8.5, color=INK)
    ax.set_axisbelow(True)
    ax.set_xlabel("brick read throughput to GPU memory (GB/s, median; 512³ u8 bricks)")
    ax.grid(True, axis="x")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Case A reference scan on NVMe: the two read paths" + ("" if gds_on else " (before GDS is enabled)"),
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "gds_read_paths.png", made)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m zvdvc.bench.doc_figures", description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", default="runs")
    p.add_argument("--docs", default="docs")
    args = p.parse_args(argv)
    runs, docs = Path(args.runs), Path(args.docs)
    made: list[str] = []
    skipped: list[str] = []
    idvc_figures(runs, docs / "validation" / "figures", made, skipped)
    truth_figures(runs, docs / "validation" / "figures", made, skipped)
    error_floor_figures(runs, docs / "validation" / "figures", made, skipped)
    speed_figure(runs, docs / "benchmarks" / "figures" / "summary", made, skipped)
    gds_figure(runs, docs / "benchmarks" / "figures" / "summary", made, skipped)
    for m in made:
        print("wrote", m)
    for s in skipped:
        print("skipped", s)


if __name__ == "__main__":
    main()
