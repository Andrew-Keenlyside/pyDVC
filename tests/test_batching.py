import numpy as np
import pytest

from pydvc.pipeline.batching import batch_size, iter_batches, morton_order


def test_morton_order_visits_octants_in_turn():
    grid = np.stack(np.meshgrid(*[np.arange(4.0)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    order = morton_order(grid)
    assert sorted(order.tolist()) == list(range(64))
    first8 = grid[order[:8]]
    assert (first8 <= 1).all()                                   # the first 2^3 octant
    np.testing.assert_array_equal(grid[order[1]], [1, 0, 0])     # x varies fastest


def test_morton_keeps_consecutive_points_close():
    pts = np.random.default_rng(0).uniform(0, 1000, (4000, 3))
    ordered = pts[morton_order(pts)]
    step = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    random_step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.median(step) < 0.25 * np.median(random_step)


def test_batch_size_and_iteration():
    b = batch_size(8 << 30, 4096, 6)
    assert 50_000 < b < 200_000
    assert batch_size(8 << 30, 4096, 6, method="icgn") < b
    assert [len(x) for x in iter_batches(np.arange(10), 4)] == [4, 4, 2]
    with pytest.raises(ValueError):
        batch_size(0, 10, 6)
