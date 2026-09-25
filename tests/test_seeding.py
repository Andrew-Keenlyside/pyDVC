import numpy as np

from pydvc.solver import seeding
from pydvc.status import PointStatus

GRID = np.stack(np.meshgrid(*[np.arange(6.0) * 10] * 3, indexing="ij"), axis=-1).reshape(-1, 3)


def test_knn_excludes_self_and_orders_by_distance():
    nb = seeding.knn(GRID, 6)
    assert nb.shape == (len(GRID), 6)
    assert not (nb == np.arange(len(GRID))[:, None]).any()
    d = np.linalg.norm(GRID[nb] - GRID[:, None], axis=-1)
    assert (np.diff(d, axis=1) >= -1e-12).all()
    interior = 1 * 36 + 1 * 6 + 1      # (1, 1, 1): six face neighbours at distance 10
    np.testing.assert_allclose(d[interior], 10.0)


def test_knn_with_duplicates_and_small_clouds():
    pts = np.array([[0.0, 0, 0], [0, 0, 0], [1, 0, 0]])
    nb = seeding.knn(pts, 5)
    assert nb.shape == (3, 2)
    assert not (nb == np.arange(3)[:, None]).any()


def test_processing_order_and_shells_cover_every_point_once():
    start = (0.0, 0.0, 0.0)
    order = seeding.processing_order(GRID, start)
    assert order[0] == 0
    shells = seeding.wavefront_shells(GRID, start, width=10.0)
    assert len(shells[0]) == 1 and shells[0][0] == 0
    flat = np.concatenate(shells)
    assert sorted(flat) == list(range(len(GRID)))
    dist = [np.linalg.norm(GRID[s], axis=1) for s in shells]
    assert all(a.max() <= b.min() for a, b in zip(dist, dist[1:]))
    assert seeding.median_spacing(GRID) == 10.0


def test_seed_from_neighbours_averages_good_only():
    neighbours = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    disp = np.array([[0.0, 0, 0], [1.0, 2, 3], [3.0, 2, 1], [9.0, 9, 9]])
    status = np.array([PointStatus.NOT_SEARCHED, PointStatus.GOOD, PointStatus.GOOD, PointStatus.RANGE_FAIL])
    seeds = seeding.seed_from_neighbours(np.array([0, 3]), neighbours, disp, status, (5.0, 5.0, 5.0))
    np.testing.assert_allclose(seeds, [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    none_good = seeding.seed_from_neighbours(np.array([0]), neighbours, disp, np.full(4, -3), (5.0, 6.0, 7.0))
    np.testing.assert_allclose(none_good, [[5.0, 6.0, 7.0]])
