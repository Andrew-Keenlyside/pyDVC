"""The ``zvdvc`` command line: a whole run through ``cli.main``, exit codes, and one-line errors."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from zvdvc import cli
from zvdvc.config import RunConfig, SeedingSpec


def _backend():
    try:
        import numba  # noqa: F401

        return "cpu"
    except ImportError:
        return "numpy"


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    assert cli.main(["synth", "--shape", "48", "48", "48", "--field", "rigid", "--spacing", "8", "--subvol-size", "12",
                     "--subvol-npts", "200", "--dof", "6", "--disp-max", "5", "--chunk", "24", "--shard", "48",
                     "--out", str(root / "case")]) == 0
    return root, root / "case" / "config.yaml"


def _config(case, name, **changes):
    """A copy of the synthetic case's config with its own output and workdir."""
    root, base = case
    cfg = RunConfig.from_yaml(base)
    cfg = dataclasses.replace(cfg, output=str(root / f"{name}.zarrvectors"), workdir=str(root / f"run_{name}"), **changes)
    path = root / f"{name}.yaml"
    cfg.to_yaml(path)
    return cfg, str(path)


def _error(capsys, command):
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith(f"zvdvc {command}: error: "), err
    return err[0]


def test_synth_plan_seed_run_finalize(case, capsys):
    from zvdvc.io.ccpi import read_disp
    from zvdvc.io.results import ResultStore

    cfg, path = _config(case, "whole")
    for argv in (["plan", path], ["seed", path], ["run", path], ["finalize", path, "--disp"]):
        assert cli.main([*argv, *(["--backend", _backend()] if argv[0] != "finalize" else [])]) == 0, argv
    assert "0 tiles solved, 0 written, 1 already written" in capsys.readouterr().out   # the wavefront seed solved all
    disp = read_disp(Path(cfg.workdir) / "results.disp")
    got = ResultStore(cfg.output).read_all()
    order = np.argsort(got["point_id"])
    np.testing.assert_array_equal(disp["n"], got["point_id"][order])
    np.testing.assert_array_equal(disp["status"], got["status"][order])
    for col, axis in zip("uvw", range(3)):
        np.testing.assert_allclose(disp[col], got["displacement"][order, axis], rtol=1e-6)
    assert (disp["status"] == 0).mean() > 0.95


def test_run_with_unwritten_tiles_exits_3_and_finalize_refuses(case, capsys, monkeypatch):
    from zvdvc.pipeline import worker

    cfg, path = _config(case, "failing", seeding=SeedingSpec(strategy="rigid"))
    assert cli.main(["plan", path, "--backend", _backend()]) == 0

    def broken(self, loaded):
        raise RuntimeError("injected")

    monkeypatch.setattr(worker.TileWorker, "solve_tile", broken)
    assert cli.main(["run", path, "--backend", _backend()]) == 3
    captured = capsys.readouterr()
    assert "0 tiles solved, 0 written" in captured.out
    assert "zvdvc run: error: 1 tiles are not written" in captured.err
    assert cli.main(["finalize", path]) == 2
    assert "cells unwritten; run `zvdvc run` again" in _error(capsys, "finalize")
    assert cli.main(["finalize", path, "--disp", "--allow-partial"]) == 0
    from zvdvc.io.ccpi import read_disp

    assert (read_disp(Path(cfg.workdir) / "results.disp")["status"] == -3).all()
    monkeypatch.undo()
    assert cli.main(["run", path, "--backend", _backend()]) == 0
    assert cli.main(["finalize", path]) == 0


def test_user_errors_are_one_line_with_exit_code_2(case, capsys, tmp_path):
    cfg, path = _config(case, "errors", seeding=SeedingSpec(strategy="rigid"))
    assert cli.main(["run", path]) == 2                                          # before `plan`
    assert "run `zvdvc plan` first" in _error(capsys, "run")
    assert cli.main(["plan", str(tmp_path / "missing.yaml")]) == 2
    assert "missing.yaml" in _error(capsys, "plan")
    assert cli.main(["plan", path, "--backend", _backend()]) == 0
    capsys.readouterr()
    _, other = _config(case, "errors", seeding=SeedingSpec(strategy="rigid"), num_points_to_process=3)
    assert cli.main(["run", other]) == 2
    assert "num_points_to_process: run with the original config" in _error(capsys, "run")
    assert cli.main(["repair", path]) == 2
    assert "M5" in _error(capsys, "repair")
    with pytest.raises(ValueError):
        cli.main(["--traceback", "run", other])

    empty = tmp_path / "empty.roi"
    empty.write_text("n x y z\n")
    _, solve_cfg = _config(case, "empty", points=str(empty))
    assert cli.main(["solve", solve_cfg, "--quiet"]) == 2
    assert "empty.roi: no points" in _error(capsys, "solve")

    dup = tmp_path / "dup.roi"
    dup.write_text("1 20 20 20\n2 24 20 20\n2 28 20 20\n")
    _, dup_cfg = _config(case, "dup", points=str(dup), seeding=SeedingSpec(strategy="rigid"))
    assert cli.main(["plan", dup_cfg, "--backend", _backend()]) == 2
    assert "point ids occur more than once (2)" in _error(capsys, "plan")


def test_repair_help_says_it_is_not_implemented():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    assert "not implemented yet" in next(c.help for c in sub._choices_actions if c.dest == "repair")


def test_convert_passes_overwrite(monkeypatch, tmp_path):
    from zvdvc.io import volume

    calls = []
    monkeypatch.setattr(volume, "convert_to_ome_zarr", lambda src, dst, **kw: calls.append(kw))
    assert cli.main(["convert", "a.npy", str(tmp_path / "b.ome.zarr")]) == 0
    assert cli.main(["convert", "a.npy", str(tmp_path / "b.ome.zarr"), "--overwrite"]) == 0
    assert [c["overwrite"] for c in calls] == [False, True]
