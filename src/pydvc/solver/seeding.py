"""Starting displacements, without CCPi's serial point order.

CCPi (``DataCloud``, ``Search::starting_param``) sorts points by distance from
the start point, then processes them one at a time. Each point starts from the
mean translation of the already-solved ``point_good`` points among its 75
nearest neighbours, or from ``rigid_trans`` if there are none. Neighbours come
from an O(N^2) full sort.

pyDVC strategies (``SeedingSpec.strategy``):

``rigid``
    Every point starts at ``rigid_trans``. Fully parallel; enough for small,
    smooth deformations.
``wavefront`` (CCPi parity; single GPU, whole problem resident)
    Points are bucketed into shells of width ``shell_width`` by distance from
    the start point. Shells are solved in order, each shell as one batch
    seeded from GOOD neighbours in earlier shells. It differs from CCPi only
    in that points within the same shell cannot seed each other. With the
    shell width equal to the lattice spacing, a ``k^3``-point lattice has about
    ``1.7 k`` shells from a corner (half that from the centre). Shell sizes
    grow as ``r^2``, so all but the first few shells fill a GPU.
``coarse`` (production, multi-GPU)
    Solve every ``coarse_stride``-th lattice point first, on OME-Zarr pyramid
    level ``coarse_level``. Subvolume size, ``disp_max`` and coordinates are
    scaled by ``2**-level``, so the pass reads 1/8 of the bytes per level and
    fits one GPU. It uses ``wavefront`` or ``fft``. That field is then
    interpolated to seed every full-resolution point. Tiles become
    independent, which is what lets 8+ GPUs run without coordination.
``fft``
    :func:`pydvc.solver.coarse.fft_seed` for every point (M5).

After the main solve, :func:`repair_candidates` finds failed points with
enough GOOD neighbours and re-seeds them from the neighbour median for another
pass (``pydvc repair``).

Neighbours come from a KD-tree (scipy, host) in O(N log N), or a uniform-grid
hash on the GPU for clouds of more than about 10^7 points (M5).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pydvc._todo import todo
from pydvc.status import PointStatus


def knn(xyz: Any, k: int, *, device: str = "cpu") -> Any:
    """(N, 3) -> (N, k) neighbour indices (self excluded), nearest first. ``k`` is capped at ``N - 1``."""
    if device != "cpu":
        raise todo("M5", "GPU kNN (uniform-grid hash)")
    from scipy.spatial import cKDTree

    pts = np.asarray(xyz, dtype=np.float64)
    n = len(pts)
    k = min(k, n - 1)
    if k <= 0:
        return np.zeros((n, 0), dtype=np.int64)
    _, idx = cKDTree(pts).query(pts, k=k + 1)
    idx = idx.astype(np.int64)
    # drop self; with duplicate points it need not be column 0, and if absent drop the farthest
    is_self = idx == np.arange(n)[:, None]
    is_self[~is_self.any(axis=1), -1] = True
    order = np.argsort(is_self, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)[:, :k]


def processing_order(xyz: np.ndarray, start_xyz: tuple[float, float, float]) -> np.ndarray:
    """CCPi's order: indices sorted by distance from the start point."""
    dist = np.linalg.norm(np.asarray(xyz, dtype=np.float64) - np.asarray(start_xyz, dtype=np.float64), axis=1)
    return np.argsort(dist, kind="stable")


def median_spacing(xyz: np.ndarray) -> float:
    """Median nearest-neighbour distance: the default wavefront shell width."""
    from scipy.spatial import cKDTree

    pts = np.asarray(xyz, dtype=np.float64)
    if len(pts) < 2:
        return 1.0
    dist, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(dist[:, 1]))


def wavefront_shells(xyz: np.ndarray, start_xyz: tuple[float, float, float], width: float) -> list[np.ndarray]:
    """Index arrays, one per shell, in solve order.

    The point nearest the start is a shell of its own, like CCPi's first point:
    it is seeded from ``rigid_trans`` and everything after it from neighbours.
    The rest fall in shells ``floor(distance / width)``.
    """
    if width <= 0:
        raise ValueError("wavefront shell width must be positive")
    pts = np.asarray(xyz, dtype=np.float64)
    if len(pts) == 0:
        return []
    order = processing_order(pts, start_xyz)
    dist = np.linalg.norm(pts[order] - np.asarray(start_xyz, dtype=np.float64), axis=1)
    shells = [order[:1]]
    rest, rest_dist = order[1:], dist[1:]
    shell_of = np.floor(rest_dist / width).astype(np.int64)
    bounds = np.flatnonzero(np.diff(shell_of)) + 1
    shells.extend(part for part in np.split(rest, bounds) if part.size)
    return shells


def seed_from_neighbours(
    idx: Any,                     # (b,) points to seed
    neighbours: Any,              # (N, k)
    displacement: Any,            # (N, 3) solved so far
    status: Any,                  # (N,) PointStatus; unsolved points are NOT_SEARCHED
    rigid_trans: tuple[float, float, float],
) -> Any:
    """CCPi ``starting_param``: mean of GOOD neighbours' (u, v, w), else ``rigid_trans``."""
    nb = np.asarray(neighbours)[np.asarray(idx)]                    # (b, k)
    good = np.asarray(status)[nb] == PointStatus.GOOD
    count = good.sum(axis=1)
    total = (np.asarray(displacement)[nb] * good[..., None]).sum(axis=1)
    seeds = np.broadcast_to(np.asarray(rigid_trans, dtype=np.float64), total.shape).copy()
    has = count > 0
    seeds[has] = total[has] / count[has, None]
    return seeds


def coarse_subset(xyz: np.ndarray, stride: int) -> np.ndarray:
    """Indices of every ``stride``-th lattice point along each axis."""
    raise todo("M5", "coarse_subset")


def interpolate_seed_field(
    coarse_xyz: np.ndarray,
    coarse_disp: np.ndarray,
    coarse_status: np.ndarray,
    xyz: Any,
) -> Any:
    """Seeds for ``xyz`` from the GOOD coarse points (kNN inverse-distance weighting)."""
    raise todo("M5", "interpolate_seed_field")


def repair_candidates(
    status: np.ndarray,
    displacement: np.ndarray,
    neighbours: np.ndarray,
    *,
    min_good: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Failed points with at least ``min_good`` GOOD neighbours, and their median-of-neighbours seeds."""
    raise todo("M5", "repair_candidates")
