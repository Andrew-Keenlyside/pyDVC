import numpy as np
import pytest

from zvdvc.bench import metrics
from zvdvc.io.ccpi import write_disp
from zvdvc.status import PointStatus


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


def test_agreement_with_another_code_counts_points_good_in_both():
    from zvdvc.bench.metrics import compare_arrays

    ours = np.array([0, 0, -1, 0])            # GOOD, GOOD, RANGE_FAIL, GOOD
    theirs = np.array([0, -1, 0, 0])          # CCPi codes
    d = np.zeros((4, 3))
    acc = compare_arrays(d, d, ours, ref_status=theirs)
    assert acc.n_good == 3 and acc.n_good_ref == 3 and acc.n_good_both == 2
    assert acc.status_agreement == 0.5
    assert acc.status_confusion == {"-1/0": 1, "0/-1": 1, "0/0": 2}


def test_edge_mask_flags_subvolumes_that_leave_the_volume():
    from zvdvc.bench.metrics import edge_mask

    xyz = np.array([[50.0, 50.0, 50.0], [95.0, 50.0, 50.0], [50.0, 50.0, 50.0], [3.0, 50.0, 50.0]])
    disp = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [44.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # volume 100^3, reach 6: the second point leaves at x = 101, the third only after moving, the fourth at x = -3
    np.testing.assert_array_equal(edge_mask(xyz, disp, (100, 100, 100), 6.0), [False, True, True, True])


def test_revised_q1_judges_bias_and_spread_against_zvdvcs_own():
    from zvdvc.bench.compare_ccpi import revised_q1

    rng = np.random.default_rng(0)
    n = 4000
    ids = np.arange(n)
    truth = rng.normal(size=(n, 3))
    ours = truth + rng.normal(scale=0.03, size=(n, 3))
    alt = truth + rng.normal(scale=0.03, size=(n, 3))

    def res(d):
        return {"point_id": ids, "status": np.zeros(n, dtype=np.int8), "displacement": d}

    idx, sel = np.arange(n), np.ones(n, dtype=bool)
    same = revised_q1(res(ours), res(truth + rng.normal(scale=0.03, size=(n, 3))), res(alt), idx, idx, sel)
    assert same["pass"] and max(same["spread_ratio"]) < 1.1
    biased = revised_q1(res(ours), res(truth + [0.02, 0, 0] + rng.normal(scale=0.03, size=(n, 3))), res(alt), idx, idx, sel)
    assert not biased["pass"] and abs(biased["mean_diff"][0]) > 0.015
    noisy = revised_q1(res(ours), res(truth + rng.normal(scale=0.06, size=(n, 3))), res(alt), idx, idx, sel)
    assert not noisy["pass"] and min(noisy["spread_ratio"]) > 1.25
