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


def knn(xyz: Any, k: int, *, device: str = "cpu") -> Any:
    """(N, 3) -> (N, k) neighbour indices (self excluded)."""
    raise todo("M1", "knn")


def processing_order(xyz: np.ndarray, start_xyz: tuple[float, float, float]) -> np.ndarray:
    """CCPi's order: indices sorted by distance from the start point."""
    raise todo("M1", "processing_order")


def wavefront_shells(xyz: np.ndarray, start_xyz: tuple[float, float, float], width: float) -> list[np.ndarray]:
    """Index arrays, one per shell, in solve order."""
    raise todo("M1", "wavefront_shells")


def seed_from_neighbours(
    idx: Any,                     # (b,) points to seed
    neighbours: Any,              # (N, k)
    displacement: Any,            # (N, 3) solved so far
    status: Any,                  # (N,) PointStatus; unsolved points are NOT_SEARCHED
    rigid_trans: tuple[float, float, float],
) -> Any:
    """CCPi ``starting_param``: mean of GOOD neighbours' (u, v, w), else ``rigid_trans``."""
    raise todo("M1", "seed_from_neighbours")


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
