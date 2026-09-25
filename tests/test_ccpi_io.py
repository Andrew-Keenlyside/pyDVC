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


def test_pydvc_only_status_is_exported_as_not_searched(tmp_path):
    path = tmp_path / "run.disp"
    ccpi.write_disp(path, [1, 2], np.zeros((2, 3)), [PointStatus.SINGULAR, PointStatus.THRESH_FAIL], [0.0, 0.0], np.zeros((2, 3)))
    assert ccpi.read_disp(path)["status"].tolist() == [-3, -3]


def test_roi_round_trip(tmp_path):
    from pydvc.io.pointcloud import read_roi

    xyz = np.array([[1.5, 2.25, 3.0], [100.125, 0.0, 7.75]])
    ccpi.write_roi(tmp_path / "p.roi", np.array([7, 9]), xyz)
    ids, back = read_roi(tmp_path / "p.roi")
    assert ids.tolist() == [7, 9]
    np.testing.assert_array_equal(back, xyz)


def test_read_roi_accepts_headers_commas_and_comments(tmp_path):
    from pydvc.io.pointcloud import read_roi

    (tmp_path / "p.csv").write_text("n,x,y,z\n# comment\n1, 1.0, 2.0, 3.0\n\n2,4,5,6  # trailing\n")
    ids, xyz = read_roi(tmp_path / "p.csv")
    assert ids.tolist() == [1, 2] and xyz.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_dvc_input_matches_the_config(tmp_path):
    from pydvc.config import RunConfig, SearchSpec, SubvolumeSpec, ThresholdSpec, VolumeSpec

    np.save(tmp_path / "ref.npy", np.zeros((4, 5, 6), dtype=np.uint16))
    ccpi.write_roi(tmp_path / "p.roi", np.array([1]), np.array([[2.0, 2.0, 1.5]]))
    cfg = RunConfig(
        volumes=VolumeSpec(reference=str(tmp_path / "ref.npy"), deformed=str(tmp_path / "ref.npy")),
        points=str(tmp_path / "p.roi"),
        output="o",
        subvolume=SubvolumeSpec(geometry="cube", size=30, n_samples=1000),
        search=SearchSpec(dof=12, objective="zssd", disp_max=15, rigid_trans=(1.5, 0.0, -2.0), threshold=ThresholdSpec(10, 200)),
    )
    ccpi.write_dvc_input(cfg, tmp_path / "dvc_in", roi_path=tmp_path / "p.roi", output_base=tmp_path / "out")
    params = ccpi.read_dvc_input(tmp_path / "dvc_in")
    assert params["vol_wide"] == "6" and params["vol_high"] == "5" and params["vol_tall"] == "4"
    assert params["vol_bit_depth"] == "16" and params["vol_endian"] == "little"
    assert int(params["vol_hdr_lngth"]) == np.load(tmp_path / "ref.npy", mmap_mode="r").offset
    assert (params["subvol_geom"], params["subvol_size"], params["subvol_npts"]) == ("cube", "30", "1000")
    assert (params["num_srch_dof"], params["obj_function"], params["disp_max"]) == ("12", "zssd", "15")
    assert params["rigid_trans"] == "1.5 0 -2"
    assert (params["subvol_thresh"], params["gray_thresh_min"], params["gray_thresh_max"]) == ("on", "10", "200")
    assert params["starting_point"] == "2 2 1.5" and params["num_points_to_process"] == "0"
    # CCPi keeps the newline in a value unless a comment follows, so every value line carries one
    assert all("###" in line for line in (tmp_path / "dvc_in").read_text().splitlines())


def test_throughput_from_ccpi_stat(tmp_path):
    path = tmp_path / "run.stat"
    path.write_text("20 points processed in 0 seconds\n0.024 sec/pt\n41.911 pt/sec\n\nnumber successful = 20\n")
    assert ccpi.read_stat_throughput(path) == pytest.approx(41.911)


def test_stat_round_trip(tmp_path):
    from pydvc.config import RunConfig, VolumeSpec

    cfg = RunConfig(volumes=VolumeSpec(reference="a", deformed="b"), points="p", output="o")
    ccpi.write_stat(tmp_path / "run.stat", cfg, ccpi.RunSummary(n_points=100, seconds=4.0, counts={0: 98, -1: 2}))
    assert ccpi.read_stat_throughput(tmp_path / "run.stat") == pytest.approx(25.0)
