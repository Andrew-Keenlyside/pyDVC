"""M4 benchmark harness: case A volume handling, and the CCPi-vs-zvDVC comparison on a tiny case."""

import numpy as np
import pytest

from zvdvc.bench import case_a
from zvdvc.config import SearchSpec, SubvolumeSpec


def test_case_a_volume_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(case_a, "SHAPE_ZYX", (3, 4, 5))
    zyx = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
    np.save(tmp_path / "ok.npy", zyx)
    np.save(tmp_path / "xyz.npy", np.ascontiguousarray(zyx.transpose(2, 1, 0)))       # stored x-first
    np.save(tmp_path / "bad.npy", np.zeros((2, 2, 2), np.uint8))
    assert case_a._c_ordered(tmp_path / "ok.npy", tmp_path) == (str(tmp_path / "ok.npy"), None, None)
    path, shape_xyz, dtype = case_a._c_ordered(tmp_path / "xyz.npy", tmp_path)
    assert shape_xyz == (5, 4, 3) and dtype == "|u1"
    np.testing.assert_array_equal(np.fromfile(path, np.uint8).reshape(3, 4, 5), zyx)
    with pytest.raises(ValueError):
        case_a._c_ordered(tmp_path / "bad.npy", tmp_path)


def test_compare_runs_both_codes_and_reports(tmp_path):
    from zvdvc.bench.ccpi_baseline import find_dvc

    try:
        exe = find_dvc()
    except FileNotFoundError:
        pytest.skip("CCPi dvc not installed (conda ccpi-dvc, or set ZVDVC_CCPI_DVC)")
    pytest.importorskip("numba")
    from zvdvc.bench.compare_ccpi import compare
    from zvdvc.config import RunConfig
    from zvdvc.synth.phantoms import DisplacementField, make_case

    config = make_case(
        tmp_path / "case", shape_zyx=(64, 64, 64), field=DisplacementField("affine", {"translation": (1.3, -0.6, 0.4)}),
        spacing=12.0, chunk=32, shard=64, subvolume=SubvolumeSpec(geometry="sphere", size=20, n_samples=600),
        search=SearchSpec(dof=6, disp_max=4.0, rigid_trans=(1.0, -1.0, 0.0), report_convg_fail=False),
    )
    report = compare(RunConfig.from_yaml(config), tmp_path / "out", ccpi_exes=[exe], ccpi_processes=2,
                     backends=["cpu"], cli=True, truth=tmp_path / "case" / "truth.npz")
    names = [r["name"] for r in report["runs"]]
    assert len(names) == 4 and sum("zvDVC" in n for n in names) == 2
    assert all(a["q1_pass"] for k, a in report["agreement"].items() if "vs CCPi" in k)
    assert (tmp_path / "out" / "report.md").read_text().startswith("# CCPi vs zvDVC")
