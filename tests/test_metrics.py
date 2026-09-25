import numpy as np
import pytest

from pydvc.bench import metrics
from pydvc.io.ccpi import write_disp
from pydvc.status import PointStatus


def test_compare_arrays():
    truth = np.zeros((4, 3))
    est = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [5.0, 5.0, 5.0]])
    status = np.array([0, 0, 0, PointStatus.RANGE_FAIL])
    acc = metrics.compare_arrays(est, truth, status)
    assert acc.n_points == 4 and acc.n_good == 3 and acc.frac_good == 0.75
    assert acc.rmse[0] == pytest.approx(np.sqrt(0.02 / 3)) and acc.rmse[2] == 0.0
    assert acc.bias[0] == pytest.approx(0.0) and acc.bias[1] == pytest.approx(0.2 / 3)
    assert acc.median_abs == pytest.approx(0.1)
    assert acc.status_counts == {0: 3, -1: 1}


def test_against_disp_matches_ids_and_statuses(tmp_path):
    ids = np.array([3, 1, 2])
    xyz = np.zeros((3, 3))
    disp = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    np.savez(tmp_path / "r.npz", point_id=ids, xyz=xyz, status=np.array([0, 0, PointStatus.SINGULAR]),
             objmin=np.zeros(3), displacement=disp)
    order = np.argsort(ids)                             # CCPi file in id order: ids 1, 2, 3
    write_disp(tmp_path / "c.disp", ids[order], xyz, np.array([0, -3, 0]), np.zeros(3), disp[order] + 0.01)
    acc = metrics.against_disp(tmp_path / "r.npz", tmp_path / "c.disp")
    assert acc.status_agreement == 1.0                  # SINGULAR exports as NOT_SEARCHED
    assert acc.rmse[0] == pytest.approx(0.01) and acc.n_good == 2


def test_against_truth(tmp_path):
    ids = np.arange(1, 6)
    truth = np.random.default_rng(0).normal(size=(5, 3))
    np.savez(tmp_path / "t.npz", point_id=ids, displacement=truth)
    np.savez(tmp_path / "r.npz", point_id=ids[::-1], xyz=np.zeros((5, 3)), status=np.zeros(5, int),
             objmin=np.zeros(5), displacement=truth[::-1])
    acc = metrics.against_truth(tmp_path / "r.npz", tmp_path / "t.npz")
    assert acc.frac_good == 1.0 and max(acc.rmse) == 0.0
