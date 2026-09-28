"""The kernel benchmark runs on a small synthetic case and reports sane, checked numbers."""

import numpy as np
import pytest

from zvdvc.bench import kernel
from zvdvc.config import SubvolumeSpec


@pytest.mark.parametrize("backend", ["cpu", "numpy32"])
def test_kernel_benchmark_on_synthetic_data(backend):
    if backend == "cpu":
        pytest.importorskip("numba")
    case = kernel.synthetic(96, shape=(64, 64, 64), subvolume=SubvolumeSpec(geometry="sphere", size=16, n_samples=500))
    r = kernel.measure(case, backend, repeats=1)
    assert r["backend"] == backend and r["points"] == 96
    assert r["sums_us_per_point_iter"] > 0 and r["sample_us_per_point"] > 0
    assert r["rel_err"] < 1e-4


def test_rel_err_ignores_sums_that_cancel_to_zero():
    want = np.array([[1e6, 1e-14], [2e6, -3e-14]])
    got = want + np.array([[1.0, 1e-3], [0.0, 0.0]])
    # the cancelling column is judged against 1e-6 of the largest sum (2e6), not against ~1e-14
    assert kernel.rel_err(got, want) == pytest.approx(1e-3 / (1e-6 * 2e6))


def test_grid_points_start_at_the_centre():
    from zvdvc.bench.case_a import grid_points

    pid, xyz = grid_points(10.0, ((0.0, 0.0, 0.0), (40.0, 40.0, 20.0)))
    assert len(pid) == len(xyz) == 5 * 5 * 3 and pid[0] == 1
    np.testing.assert_array_equal(xyz[0], [20.0, 20.0, 10.0])
