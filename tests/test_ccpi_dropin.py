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


# --------------------------------------------------------------------------- failures


def _variant(case, name, output=None, **edits):
    """``dvc_in.txt`` with some values replaced (``None`` drops the key), writing to ``output`` (default ``name``)."""
    edits["output_filename"] = str(case / (output or name))
    lines = []
    for line in (case / "dvc_in.txt").read_text().splitlines():
        key = line.split("\t")[0]
        if key in edits:
            if edits[key] is None:
                continue
            line = f"{key}\t{edits[key]}\t### edited"
        lines.append(line)
    (case / f"{name}.dvc_in").write_text("\n".join(lines) + "\n")
    return case / f"{name}.dvc_in"


def _params(case):
    from zvdvc.io.ccpi import read_dvc_input

    return read_dvc_input(case / "dvc_in.txt")


def _bad_inputs(case):
    """name -> (dvc_in edits, text of the error). Files are written next to the case."""
    p = _params(case)
    raw = Path(p["correlate_filename"])
    (case / "short.raw").write_bytes(raw.read_bytes()[:-1000])
    ids_xyz = [line.split("\t") for line in Path(p["point_cloud_filename"]).read_text().splitlines()]
    roi = lambda rows: "".join("\t".join(r) + "\n" for r in rows)                  # noqa: E731
    (case / "nan.roi").write_text(roi(ids_xyz[:3] + [[ids_xyz[3][0], "nan", ids_xyz[3][2], ids_xyz[3][3]]]))
    (case / "empty.roi").write_text("")
    (case / "dup.roi").write_text(roi(ids_xyz[:3] + [[ids_xyz[0][0]] + ids_xyz[3][1:]]))
    return {
        "missing_def": (dict(correlate_filename=str(case / "nope.raw")), "volume file not found"),
        "short_def": (dict(correlate_filename=str(case / "short.raw")), "(truncated?)"),
        "bits8_on16": (dict(vol_bit_depth="8"), "2 times as large"),
        "hdr100": (dict(vol_hdr_lngth="100"), "vol_hdr_lngth"),
        "dof5": (dict(num_srch_dof="5"), "num_srch_dof: 5 is not one of"),
        "geom_cyl": (dict(subvol_geom="cylinder"), "subvol_geom"),
        "obj_foo": (dict(obj_function="foo"), "obj_function"),
        "interp_foo": (dict(interp_type="spline"), "interp_type"),
        "npts0": (dict(subvol_npts="0"), "subvol_npts: 0 is out of range"),
        "size_neg": (dict(subvol_size="-16"), "subvol_size"),
        "dispmax0": (dict(disp_max="0"), "disp_max"),
        "endian_typo": (dict(vol_endian="BIG_ENDIAN"), "vol_endian"),
        "missing_npts": (dict(subvol_npts=None), "missing required keys ['subvol_npts']"),
        "nan_roi": (dict(point_cloud_filename=str(case / "nan.roi")), "non-finite coordinate"),
        "empty_roi": (dict(point_cloud_filename=str(case / "empty.roi")), "no points"),
        "dup_roi": (dict(point_cloud_filename=str(case / "dup.roi")), "must be unique"),
        "missing_roi": (dict(point_cloud_filename=str(case / "nope.roi")), "FileNotFoundError"),
    }


def _snapshot(case, base):
    return {s: (case / f"{base}.{s}").read_bytes() for s in ("disp", "stat")}


def _no_temporaries(case):
    return not [p.name for p in case.iterdir() if p.name.endswith(".tmp")]


def test_failed_rerun_keeps_the_previous_outputs(case, monkeypatch):
    monkeypatch.chdir(case)
    assert dropin.run(_variant(case, "good", output="keep"), backend="cpu", out=io.StringIO()) == 0
    before = _snapshot(case, "keep")
    for name, (edits, message) in _bad_inputs(case).items():
        out = io.StringIO()
        assert dropin.run(_variant(case, name, output="keep", **edits), backend="cpu", out=out) == 1, name
        lines = [line for line in out.getvalue().splitlines() if line.strip()]
        assert len(lines) == 1 and lines[0].startswith("input file problem: ") and message in lines[0], (name, lines)
        assert _snapshot(case, "keep") == before, name
        assert _no_temporaries(case), name


def test_failed_first_run_writes_no_disp(case, monkeypatch):
    monkeypatch.chdir(case)
    for name, (edits, _) in _bad_inputs(case).items():
        assert dropin.run(_variant(case, name, **edits), backend="cpu", out=io.StringIO()) == 1
        assert not (case / f"{name}.disp").exists() and not (case / f"{name}.stat").exists(), name


def test_backend_from_the_environment_is_checked(case, monkeypatch):
    monkeypatch.chdir(case)
    monkeypatch.setenv("ZVDVC_BACKEND", "fusd")
    out = io.StringIO()
    assert dropin.run(_variant(case, "backend_typo"), out=out) == 1
    assert "input file problem: ValueError: ZVDVC_BACKEND: unknown backend 'fusd'" in out.getvalue()
    assert not (case / "backend_typo.disp").exists()


def test_missing_output_folder_is_an_input_problem(case, monkeypatch):
    monkeypatch.chdir(case)
    out = io.StringIO()
    assert dropin.run(_variant(case, "nodir", output="nodir/out"), backend="cpu", out=out) == 1
    assert "input file problem: FileNotFoundError: output_filename: folder" in out.getvalue()


def test_a_failing_run_is_reported_and_keeps_the_previous_outputs(case, monkeypatch):
    from zvdvc.pipeline import inmemory

    monkeypatch.chdir(case)
    assert dropin.run(_variant(case, "good", output="keep_run"), backend="cpu", out=io.StringIO()) == 0
    before = _snapshot(case, "keep_run")

    def boom(*args, **kwargs):
        raise RuntimeError("device lost\nsecond line")

    monkeypatch.setattr(inmemory, "solve_in_memory", boom)
    out = io.StringIO()
    assert dropin.run(_variant(case, "good", output="keep_run"), backend="cpu", out=out) == 1
    assert out.getvalue().rstrip().splitlines()[-1] == "run failed: RuntimeError: device lost second line"
    assert _snapshot(case, "keep_run") == before and _no_temporaries(case)
    assert dropin.run(_variant(case, "good", output="fresh_fail"), backend="cpu", out=io.StringIO()) == 1
    assert not (case / "fresh_fail.disp").exists()


def test_points_outside_the_volume_are_range_fail(case, monkeypatch):
    from zvdvc.io.ccpi import read_disp

    monkeypatch.chdir(case)
    rows = [line.split("\t") for line in Path(_params(case)["point_cloud_filename"]).read_text().splitlines()][:5]
    rows += [["9001", "500", "20", "20"], ["9002", "-40", "20", "20"], ["9003", "20", "20", "1e6"]]
    (case / "outside.roi").write_text("".join("\t".join(r) + "\n" for r in rows))
    path = _variant(case, "outside", point_cloud_filename=str(case / "outside.roi"), num_points_to_process="0",
                    starting_point=" ".join(rows[0][1:]))
    assert dropin.run(path, backend="cpu", out=io.StringIO()) == 0
    d = read_disp(case / "outside.disp")
    far = np.isin(d["n"], [9001, 9002, 9003])
    assert far.sum() == 3 and (d["status"][far] == -1).all()
    assert (d["u"][far] == 0).all() and np.isfinite(d["objmin"]).all()
    assert (d["status"][~far] == 0).all()


class _FakeOOM(MemoryError):
    """Stands in for cupy.cuda.memory.OutOfMemoryError (a MemoryError)."""


def _fake_gpu_engine(monkeypatch, cpu_ok=True):
    from zvdvc.pipeline import inmemory

    real = inmemory.make_engine

    class GpuEngine:
        def prepare(self, brick):
            raise _FakeOOM("Out of memory allocating 1,000,000 bytes")

    def make_engine(backend):
        if backend == "fused":
            return GpuEngine()
        if backend == "cpu" and not cpu_ok:
            raise ImportError("No module named 'numba'")
        return real(backend)

    monkeypatch.setattr(inmemory, "make_engine", make_engine)


def test_gpu_out_of_memory_falls_back_to_the_cpu_engine(case, monkeypatch):
    from zvdvc.config import RunConfig
    from zvdvc.io.pointcloud import read_roi
    from zvdvc.pipeline.inmemory import solve_in_memory

    monkeypatch.chdir(case)
    cfg = RunConfig.from_ccpi(case / "dvc_in.txt")
    pid, xyz = read_roi(cfg.points)
    pid, xyz = pid[:20], xyz[:20]
    want = solve_in_memory(cfg, pid, xyz, backend="cpu")
    _fake_gpu_engine(monkeypatch)
    with pytest.warns(RuntimeWarning, match="fused engine has not enough memory .* solving on the cpu engine instead"):
        got = solve_in_memory(cfg, pid, xyz, backend="fused")
    np.testing.assert_array_equal(got.status, want.status)
    np.testing.assert_array_equal(got.params, want.params)
    _fake_gpu_engine(monkeypatch, cpu_ok=False)
    with pytest.raises(MemoryError, match="the cpu engine needs numba"):
        solve_in_memory(cfg, pid, xyz, backend="fused")
