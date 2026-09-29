"""Shared figure style for zvDVC's documentation figures (matplotlib).

A validated reference palette: categorical slots 1–4 pass the adjacent-pair colour-vision checks in
light mode (worst CVD ΔE 9.1). Slots 3 and 4 sit below 3:1 contrast on the surface, so charts that
use them label their series directly or carry a legend and a table. Sequential values use one blue
ramp; signed differences use a blue–red diverging ramp with a grey midpoint.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a3a2", "#e34948", "#a8201f"]


def style(plt: Any) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    })


def grid_image(r: dict[str, Any], values: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, list[float]]:
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


def row_of(r: dict[str, Any], ids: np.ndarray) -> np.ndarray:
    order = np.argsort(r["point_id"])
    return order[np.searchsorted(r["point_id"], ids, sorter=order)]


def fields_figure(grid: dict[str, Any], ids: np.ndarray, first: np.ndarray, second: np.ndarray, names: tuple[str, str],
                  path: Any, title: str) -> None:
    """u, v, w of two results on a point grid, side by side, and their difference (``second - first``).

    ``first`` and ``second`` are (N, 3) displacements of the points ``ids``; ``grid`` is a result
    holding every point (for positions, :func:`grid_image`).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    style(plt)
    seq = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    seq.set_bad("#f0efec")
    div = LinearSegmentedColormap.from_list("div", DIVERGING)
    div.set_bad("#f0efec")
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 6.6), constrained_layout=True)
    for k, comp in enumerate(["u (x)", "v (y)", "w (z)"]):
        lo, hi = np.percentile(np.concatenate([first[:, k], second[:, k]]), [1, 99])
        for j, (vals, name) in enumerate([(first[:, k], names[0]), (second[:, k], names[1])]):
            img, ext = grid_image(grid, vals, ids)
            im = axes[k, j].imshow(img, extent=ext, cmap=seq, vmin=lo, vmax=hi, interpolation="nearest")
            axes[k, j].set_title(f"{name}: {comp}", fontsize=9)
        fig.colorbar(im, ax=axes[k, :2], shrink=0.85, label="voxels")
        dd = second[:, k] - first[:, k]
        lim = float(np.percentile(np.abs(dd), 99)) or 1e-6
        img, ext = grid_image(grid, dd, ids)
        im = axes[k, 2].imshow(img, extent=ext, cmap=div, norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
        axes[k, 2].set_title(f"difference: {comp}", fontsize=9)
        fig.colorbar(im, ax=axes[k, 2], shrink=0.85, label="voxels")
    for axx in axes.ravel():
        axx.set_xlabel("x (voxels)", fontsize=8)
        axx.set_ylabel("y", fontsize=8)
        axx.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)
