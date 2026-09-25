import numpy as np
import pytest

from pydvc.io import ccpi
from pydvc.status import PointStatus


def test_status_codes_match_ccpi():
    assert [PointStatus.GOOD, PointStatus.RANGE_FAIL, PointStatus.CONVG_FAIL, PointStatus.NOT_SEARCHED] == [0, -1, -2, -3]
    assert PointStatus.SINGULAR.to_ccpi() == PointStatus.NOT_SEARCHED
    assert PointStatus.RANGE_FAIL.to_ccpi() == PointStatus.RANGE_FAIL


def test_disp_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    n = 5
    point_id = np.arange(1, n + 1)
    xyz = rng.uniform(0, 100, (n, 3))
    status = np.array([0, 0, -1, -2, 0])
    objmin = np.linspace(0.01, 0.05, n)
    disp = rng.normal(size=(n, 3))

    path = tmp_path / "run.disp"
    ccpi.write_disp(path, point_id, xyz, status, objmin, disp)
    back = ccpi.read_disp(path)

    np.testing.assert_array_equal(back["n"], point_id)
    np.testing.assert_array_equal(back["status"], status)
    np.testing.assert_allclose(np.stack([back["u"], back["v"], back["w"]], axis=-1), disp, atol=1e-5)


def test_reads_legacy_disp_with_rotation_columns(tmp_path):
    path = tmp_path / "legacy.disp"
    path.write_text(
        "n\tx\ty\tz\tstatus\tobjmin\tu\tv\tw\tphi\tthe\tpsi\n"
        "1\t10\t20\t30\t0\t0.02\t1.5\t-0.25\t0.125\t0.001\t-0.002\t0.0005\n"
    )
    back = ccpi.read_disp(path)
    assert back["u"][0] == pytest.approx(1.5)
    assert back["w"][0] == pytest.approx(0.125)
