"""zvdvc-dvc, the drop-in for CCPi's dvc: CCPi's file formats, iDVC's parsers, and a run against CCPi itself."""

import io
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from zvdvc import ccpi_dropin as dropin


def test_numbers_print_as_cpp_does():
    assert [dropin.g(v) for v in (50.0, 212, 0.0275154123, 33.1161201, -0.0, 1e-7)] == \
        ["50", "212", "0.0275154", "33.1161", "0", "1e-07"]


def test_disp_rows_match_a_ccpi_row():
    # CCPi 22.0.0 wrote this row for case A point 1
    assert dropin.disp_row(1, (50.0, 212.0, 630.0), 0, 0.0275154, (33.11612, 3.413023, -3.134589)) == \
        "1\t50\t212\t630\t0\t0.0275154\t33.116120\t3.413023\t-3.134589\n"


def _idvc_parse(text):
    """iDVC's RunResults .stat parser (utilities.py), line-position logic verbatim."""
    out, offset = {}, 0
    for count, line in enumerate(text.splitlines(keepends=True)):
        if count == 9 and line.split("\t")[0] == "vol_endian":
            offset = 1
        f = line.split("\t")
        if count == 14 + offset: out["subvol_geom"] = f[1].strip()
        if count == 15 + offset: out["subvol_size"] = round(int(f[1]))
        if count == 16 + offset: out["subvol_points"] = int(f[1])
        if count == 20 + offset: out["disp_max"] = int(f[1])
        if count == 21 + offset: out["num_srch_dof"] = int(f[1])
        if count == 22 + offset: out["obj_function"] = f[1].strip()
        if count == 23 + offset: out["interp_type"] = f[1].strip()
        if count == 25 + offset: out["rigid_trans"] = [int(f[1]), int(f[2]), int(f[3])]
    return out


@pytest.mark.parametrize("bits, thresh", [(8, "off"), (16, "off"), (8, "on")])
def test_idvc_reads_the_stat_echo(bits, thresh):
    params = {"reference_filename": "r.raw", "correlate_filename": "c.raw", "point_cloud_filename": "p.roi",
              "output_filename": "out", "vol_bit_depth": str(bits), "vol_endian": "little", "vol_hdr_lngth": "0",
              "vol_wide": "100", "vol_high": "90", "vol_tall": "80", "subvol_geom": "sphere", "subvol_size": "40",
              "subvol_npts": "5000", "subvol_thresh": thresh, "gray_thresh_min": "10", "gray_thresh_max": "200",
              "min_vol_fract": "0.5", "disp_max": "12", "num_srch_dof": "6", "obj_function": "znssd",
              "interp_type": "tricubic", "rigid_trans": "3.0 -2.0 1.0", "basin_radius": "0.0", "subvol_aspect": "1.0 1.0 1.0"}
    text = dropin.stat_echo(params, 10, (1, 2, 3), (4, 5, 6))
    if thresh == "on":
        # CCPi writes three threshold lines before disp_max, which shifts the layout iDVC parses by
        # position: iDVC then fails on such runs, with CCPi's .stat as with this one.
        lines = text.splitlines()
        assert lines[18:22] == ["subvol_thresh\ton", "gray_thresh_min\t10", "gray_thresh_max\t200", "min_vol_fract\t0.5"]
        return
    parsed = _idvc_parse(text)
    assert parsed == {"subvol_geom": "sphere", "subvol_size": 40, "subvol_points": 5000, "disp_max": 12,
                      "num_srch_dof": 6, "obj_function": "znssd", "interp_type": "tricubic", "rigid_trans": [3, -2, 1]}


def test_progress_lines_give_idvc_the_count():
    line = dropin.progress_line(12, 4680, 55, (66.0, 228.0, 630.0), 0, 0.0266, (33.1, 3.4, -3.1))
    assert int(line.split("/")[0]) == 12 and "Point_Good" in line
    assert int(dropin.progress_line(13, 4680, 56, (1, 2, 3), -1, 0.0, (0, 0, 0)).split("/")[0]) == 13


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    pytest.importorskip("numba")
    from zvdvc.config import RunConfig, SearchSpec, SubvolumeSpec
    from zvdvc.io.ccpi import write_dvc_input
    from zvdvc.synth.phantoms import DisplacementField, make_case

    root = tmp_path_factory.mktemp("dropin")
    config = make_case(root / "c", shape_zyx=(64, 64, 64), field=DisplacementField("affine", {"translation": (1.3, -0.6, 0.4)}),
                       spacing=12.0, chunk=32, shard=64, subvolume=SubvolumeSpec(geometry="sphere", size=20, n_samples=600),
                       search=SearchSpec(dof=6, disp_max=4.0, rigid_trans=(1.0, -1.0, 0.0), report_convg_fail=False))
    cfg = RunConfig.from_yaml(config)
    from zvdvc.bench.ccpi_baseline import ccpi_ready_config

    (root / "raw").mkdir()
    raw = ccpi_ready_config(cfg, root / "raw")
    write_dvc_input(raw, root / "dvc_in.txt", roi_path=root / "c" / "points.roi", output_base=root / "zvdvc_out")
    return root


def test_the_drop_in_runs_a_dvc_in_and_writes_ccpi_files(case, monkeypatch):
    monkeypatch.chdir(case)
    out = io.StringIO()
    assert dropin.run(case / "dvc_in.txt", backend="cpu", out=out) == 0
    lines = [l for l in out.getvalue().splitlines() if l[:1].isdigit() and "/" in l.split()[0]]
    counts = [int(l.split("/")[0]) for l in lines]
    assert counts == list(range(1, len(counts) + 1)) and len(counts) > 20
    disp = (case / "zvdvc_out.disp").read_text().splitlines()
    assert disp[0] == dropin.DISP_HEADER.strip() and len(disp) == len(counts) + 1
    stat = (case / "zvdvc_out.stat").read_text()
    assert _idvc_parse(stat)["subvol_size"] == 20 and "number successful" in stat


def test_the_drop_in_agrees_with_ccpi(case, monkeypatch):
    from zvdvc.bench.ccpi_baseline import find_dvc
    from zvdvc.bench.metrics import load_results, _match

    try:
        exe = find_dvc()
    except FileNotFoundError:
        pytest.skip("CCPi dvc not installed (set ZVDVC_CCPI_DVC)")
    import subprocess

    monkeypatch.chdir(case)
    ccpi_in = case / "ccpi_in.txt"
    ccpi_in.write_text((case / "dvc_in.txt").read_text().replace(str(case / "zvdvc_out"), str(case / "ccpi_out")))
    subprocess.run([str(exe), str(ccpi_in)], cwd=case, check=True, capture_output=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    assert dropin.run(case / "dvc_in.txt", backend="cpu", out=io.StringIO()) == 0
    ours, theirs = load_results(case / "zvdvc_out.disp"), load_results(case / "ccpi_out.disp")
    assert sorted(ours["point_id"]) == sorted(theirs["point_id"])
    ia, ib = _match(ours["point_id"], theirs["point_id"])
    good = (ours["status"][ia] == 0) & (theirs["status"][ib] == 0)
    assert good.mean() > 0.9
    assert np.abs(ours["displacement"][ia][good] - theirs["displacement"][ib][good]).max() < 0.05
    # the .stat echo is CCPi's, line for line, up to the output name and the version line
    a = (case / "zvdvc_out.stat").read_text().splitlines()[:30]
    b = (case / "ccpi_out.stat").read_text().splitlines()[:30]
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert all(x.startswith("output_filename") or x.startswith("running under") for x, _ in diff), diff
