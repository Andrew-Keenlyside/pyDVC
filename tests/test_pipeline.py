"""M3: plan -> seed -> run -> finalize on one process (CPU engines), resume, and tiled == whole-volume results."""

import dataclasses
import json

import numpy as np
import pytest

from zvdvc.bench.metrics import against_truth
from zvdvc.config import ClusterSpec, RunConfig, SearchSpec, SeedingSpec, SubvolumeSpec
from zvdvc.io.results import ResultStore
from zvdvc.pipeline import coordinator
from zvdvc.synth.phantoms import default_field, make_case

SHAPE = (80, 80, 80)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    config = make_case(
        root / "case", shape_zyx=SHAPE, field=default_field("affine", SHAPE), spacing=8.0, chunk=40, shard=80,
        subvolume=SubvolumeSpec(geometry="sphere", size=16, n_samples=500), search=SearchSpec(dof=12, disp_max=8.0),
    )
    return root, RunConfig.from_yaml(config)


def _variant(case, name, **kw):
    root, base = case
    return dataclasses.replace(
        base, output=str(root / f"{name}.zarrvectors"), workdir=str(root / f"run_{name}"),
        cluster=ClusterSpec(tile_shape=(40, 40, 40), prefetch_depth=2), **kw,
    )


def _backend():
    try:
        import numba  # noqa: F401

        return "cpu"
    except ImportError:
        return "numpy"


@pytest.mark.parametrize("strategy", ["rigid", "wavefront", "coarse"])
def test_pipeline_end_to_end(case, strategy):
    root, _ = case
    cfg = _variant(case, strategy, seeding=SeedingSpec(strategy=strategy, coarse_stride=2))
    info = coordinator.prepare(cfg, backend=_backend())
    assert info["tiles"] == 8
    coordinator.seed(cfg, backend=_backend())
    stats = coordinator.run(cfg, backend=_backend())
    summary = coordinator.finalize(cfg, export_disp=True)
    assert not stats[0].errors
    acc = against_truth(cfg.output, root / "case" / "truth.npz")
    assert summary.n_points == acc.n_points and acc.frac_good >= 0.99 and max(acc.rmse) <= 0.02
    if strategy == "wavefront":
        assert stats[0].tiles == 0 and stats[0].tiles_skipped == 8      # solved and written by the seed stage
    if strategy == "coarse":
        assert (root / f"run_{strategy}" / "seeds.npz").exists()
    assert (root / f"run_{strategy}" / "results.disp").exists()
    from zarr_vectors.validate import validate

    assert not validate(cfg.output).errors


def test_tiled_run_reproduces_the_whole_volume_solve_bit_for_bit(case):
    """M3 acceptance (on the CPU engine): bricks and tiles change nothing in the arithmetic."""
    from zvdvc.pipeline.inmemory import solve_in_memory

    cfg = _variant(case, "bitwise", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    truth = np.load(f"{case[0]}/case/truth.npz")          # a stale coarse seed field must not seed a rigid run
    np.savez(f"{cfg.workdir}/seeds.npz", point_id=truth["point_id"], seed=np.full((len(truth["point_id"]), 3), 5.0))
    stats = coordinator.run(cfg, backend=_backend())
    assert stats.missing == [] and stats[0].tiles_written == stats[0].tiles == 8
    got = ResultStore(cfg.output).read_all()
    order = np.argsort(got["point_id"])
    ref = solve_in_memory(cfg, got["point_id"][order], got["xyz"][order].astype(np.float64), strategy="rigid", backend=_backend())
    np.testing.assert_array_equal(got["status"][order], ref.status)
    np.testing.assert_array_equal(got["params"][order], ref.params.astype(np.float32))


def test_resume_skips_written_tiles_and_gives_the_same_store(case):
    from zvdvc.pipeline.tiling import load_plan
    from zvdvc.pipeline.worker import QueueSource, TileWorker

    full = _variant(case, "full", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(full, backend=_backend())
    coordinator.run(full, backend=_backend())

    part = _variant(case, "resumed", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(part, backend=_backend())
    tiles, _ = load_plan(f"{part.workdir}/plan.json")
    TileWorker(part, backend=_backend(), points_store=coordinator.points_store(part)).run(QueueSource(tiles[:3]))   # "crash" after 3 tiles
    stats = coordinator.run(part, backend=_backend())
    assert stats[0].tiles_skipped == 3 and stats[0].tiles == 5
    a, b = ResultStore(full.output).read_all(), ResultStore(part.output).read_all()
    oa, ob = np.argsort(a["point_id"]), np.argsort(b["point_id"])
    for name in ("status", "params", "objmin", "n_iter"):
        np.testing.assert_array_equal(a[name][oa], b[name][ob])
    with open(f"{part.workdir}/run_stats.json") as fh:
        assert json.load(fh)["workers"][0]["tiles_skipped"] == 3


def test_plan_records_versions_and_boxes(case):
    from zvdvc.pipeline.tiling import plan_document

    cfg = _variant(case, "plan", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    doc = plan_document(f"{cfg.workdir}/plan.json")
    assert doc["versions"]["zarr-vectors"] and doc["template_digest"] and doc["memory"]["fits"]
    truth = np.load(f"{case[0]}/case/truth.npz")
    assert len(doc["tiles"]) == 8 and sum(t["n_points"] for t in doc["tiles"]) == len(truth["point_id"])
    halo = cfg.halo()
    for t in doc["tiles"]:
        for p_lo, r_lo, p_hi, r_hi in zip(t["points_box"]["lo"], t["ref_box"]["lo"], t["points_box"]["hi"], t["ref_box"]["hi"]):
            assert r_lo <= p_lo - halo and r_hi >= p_hi + halo


def test_prepare_refuses_a_results_store_with_other_settings(case):
    cfg = _variant(case, "clash", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    other = dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, dof=3))
    with pytest.raises(ValueError, match="other settings"):
        coordinator.prepare(other, backend=_backend())


def test_a_changed_template_cannot_be_mixed_into_existing_results(case):
    cfg = _variant(case, "template", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    changed = dataclasses.replace(cfg, subvolume=dataclasses.replace(cfg.subvolume, n_samples=cfg.subvolume.n_samples + 1))
    with pytest.raises(ValueError, match="subvolume template"):
        coordinator.prepare(changed, backend=_backend())


def test_the_worker_refuses_a_store_written_with_another_template(case):
    from zvdvc.io.results import _ATTR, _root_attrs
    from zvdvc.pipeline.worker import TileWorker

    cfg = _variant(case, "doctored", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    attrs = _root_attrs(cfg.output, "r+")
    attrs[_ATTR] = {**attrs[_ATTR], "template_digest": "0" * 16}
    with pytest.raises(ValueError, match="subvolume template"):
        TileWorker(cfg, backend=_backend())


def test_a_changed_prefilter_cannot_be_mixed_into_existing_results(case):
    cfg = _variant(case, "prefilter", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    changed = dataclasses.replace(cfg, volumes=dataclasses.replace(cfg.volumes, prefilter_sigma=1.0))
    with pytest.raises(ValueError, match="prefiltered"):
        coordinator.prepare(changed, backend=_backend())


# ---------------------------------------------------------------- stale results (P0-1)


def test_prepare_refuses_a_store_from_other_settings_volumes_or_points(case):
    import os
    import shutil

    from zvdvc.io.ccpi import write_roi

    root, _ = case
    cfg = _variant(case, "stale", seeding=SeedingSpec(strategy="rigid"), points=str(root / "stale.roi"))
    truth = np.load(root / "case" / "truth.npz")
    write_roi(root / "stale.roi", truth["point_id"], truth["xyz"])
    coordinator.prepare(cfg, backend=_backend())
    coordinator.prepare(cfg, backend=_backend())                          # the same inputs: fine
    other_obj = dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, objective="ssd"))
    with pytest.raises(ValueError, match=r"differs in settings\.search\.objective.*remove it"):
        coordinator.prepare(other_obj, backend=_backend())
    shutil.copytree(root / "case" / "def.ome.zarr", root / "def_copy.ome.zarr", dirs_exist_ok=True)
    other_vol = dataclasses.replace(cfg, volumes=dataclasses.replace(cfg.volumes, deformed=str(root / "def_copy.ome.zarr")))
    with pytest.raises(ValueError, match=r"volumes\.deformed\.path"):
        coordinator.prepare(other_vol, backend=_backend())
    xyz = truth["xyz"].copy()
    xyz[0] += 0.5                                                          # an edited .roi
    write_roi(root / "stale.roi", truth["point_id"], xyz)
    later = os.stat(coordinator.points_store(cfg)).st_mtime + 10
    os.utime(root / "stale.roi", (later, later))
    with pytest.raises(ValueError, match=r"points\.digest"):
        coordinator.prepare(cfg, backend=_backend())


def test_a_store_without_a_fingerprint_is_accepted_with_a_warning(case):
    from zvdvc.io.results import _ATTR, _root_attrs

    cfg = _variant(case, "legacy", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    attrs = _root_attrs(cfg.output, "r+")
    attrs[_ATTR] = {k: v for k, v in attrs[_ATTR].items() if k != "fingerprint"}
    with pytest.warns(UserWarning, match="no run fingerprint"):
        coordinator.prepare(cfg, backend=_backend())


def test_run_refuses_a_config_that_differs_from_the_plan(case):
    root, _ = case
    cfg = _variant(case, "replan", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    changed = dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, objective="ssd", disp_max=7.0))
    with pytest.raises(ValueError, match=r"search\.disp_max, search\.objective: run with the original config"):
        coordinator.run(changed, backend=_backend())
    import shutil

    shutil.copytree(root / "case" / "def.ome.zarr", root / "def_replan.ome.zarr", dirs_exist_ok=True)
    moved = dataclasses.replace(cfg, volumes=dataclasses.replace(cfg.volumes, deformed=str(root / "def_replan.ome.zarr")))
    with pytest.raises(ValueError, match=r"volumes\.deformed\.path"):
        coordinator.run(moved, backend=_backend())
    assert not ResultStore(cfg.output).written_cells()


# ---------------------------------------------------------------- seeding, failed plans, partial runs


def test_run_enforces_the_seeding_strategy(case):
    coarse = _variant(case, "noseed", seeding=SeedingSpec(strategy="coarse", coarse_stride=2))
    coordinator.prepare(coarse, backend=_backend())
    with pytest.raises(FileNotFoundError, match="run `zvdvc seed` first"):
        coordinator.run(coarse, backend=_backend())
    seeds = f"{coarse.workdir}/seeds.npz"
    np.savez(seeds, point_id=np.arange(3), seed=np.zeros((3, 3)))          # stale, not made by `seed` for this plan
    coordinator.prepare(coarse, backend=_backend())
    with pytest.raises(FileNotFoundError, match="run `zvdvc seed` first"):
        coordinator.run(coarse, backend=_backend())
    coordinator.seed(coarse, backend=_backend())
    coordinator.prepare(coarse, backend=_backend())                        # a re-plan keeps the seed field
    with np.load(seeds) as f:
        np.savez(seeds, **{k: f[k] for k in f.files if k != "seed"}, seed=f["seed"] + 1.0)
    with pytest.raises(ValueError, match="changed after `zvdvc seed`"):
        coordinator.run(coarse, backend=_backend())

    wave = _variant(case, "nowave", seeding=SeedingSpec(strategy="wavefront"))
    coordinator.prepare(wave, backend=_backend())
    with pytest.raises(ValueError, match=r"cells are unwritten; run `zvdvc seed` first"):
        coordinator.run(wave, backend=_backend())


def test_a_failed_plan_leaves_no_results_store(case, monkeypatch):
    from pathlib import Path

    cfg = _variant(case, "failplan", seeding=SeedingSpec(strategy="rigid"))
    monkeypatch.setattr(coordinator, "_device_bytes", lambda backend: 1)
    with pytest.raises(MemoryError):
        coordinator.prepare(cfg, backend=_backend())
    assert not Path(cfg.output).exists()
    monkeypatch.undo()
    other = dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, dof=3))
    assert coordinator.prepare(other, backend=_backend())["tiles"] == 8


def test_finalize_refuses_unwritten_cells_unless_partial_is_allowed(case):
    from zvdvc.io.pointcloud import PointCloud
    from zvdvc.pipeline.tiling import load_plan

    cfg = _variant(case, "partial", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    coordinator.run(cfg, backend=_backend(), tile_ids=[0])
    tiles, _ = load_plan(f"{cfg.workdir}/plan.json")
    n_cells = len(PointCloud(coordinator.points_store(cfg)).cells())
    with pytest.raises(ValueError, match=rf"{n_cells - len(tiles[0].cells)} of {n_cells} cells unwritten; run `zvdvc run`"):
        coordinator.finalize(cfg, export_disp=True)
    summary = coordinator.finalize(cfg, export_disp=True, allow_partial=True)
    total = sum(t.n_points for t in tiles)
    assert summary.n_points == total and summary.counts[-3] == total - tiles[0].n_points
    from zvdvc.io.ccpi import read_disp

    disp = read_disp(f"{cfg.workdir}/results.disp")
    assert len(disp["n"]) == total and (disp["status"] == -3).sum() == total - tiles[0].n_points


def test_a_cell_written_without_its_status_is_rewritten_on_resume(case):
    from zvdvc.io.pointcloud import PointCloud, write_cell
    from zvdvc.pipeline.inmemory import solve_in_memory
    from zvdvc.pipeline.tiling import load_plan

    cfg = _variant(case, "halfcell", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    tiles, _ = load_plan(f"{cfg.workdir}/plan.json")
    cell = tiles[0].cells[0]
    pts = PointCloud(coordinator.points_store(cfg)).read_tile([cell], device="cpu")
    n = len(pts.point_id)
    junk = {"point_id": pts.point_id, "objmin": np.full(n, -1.0, np.float32), "displacement": np.full((n, 3), 9, np.float32),
            "params": np.full((n, 12), 9, np.float32), "n_iter": np.full(n, 1, np.uint8), "seed": np.zeros((n, 3), np.float32)}
    store = ResultStore(cfg.output, mode="r+")
    write_cell(store._level, cell, pts.xyz, junk, pts.bin_offsets[0])       # a worker killed before the status
    assert cell not in store.written_cells()
    stats = coordinator.run(cfg, backend=_backend())
    assert stats[0].tiles == 8 and stats.missing == []
    got = ResultStore(cfg.output).read_all()
    rows = np.isin(got["point_id"], pts.point_id)
    order = np.argsort(got["point_id"][rows])
    ref = solve_in_memory(cfg, got["point_id"][rows][order], got["xyz"][rows][order].astype(np.float64), strategy="rigid",
                          backend=_backend())
    np.testing.assert_array_equal(got["status"][rows][order], ref.status)
    np.testing.assert_array_equal(got["params"][rows][order], ref.params.astype(np.float32))


# ---------------------------------------------------------------- worker failures


def _fail(monkeypatch, stage, times, tile0):
    """Make ``stage`` fail ``times`` times for ``tile0`` (the plan's tile 0)."""
    from zvdvc.pipeline import worker as worker_mod

    left = {"n": times}

    def hit(tile_id):
        if tile_id == 0 and left["n"] > 0:
            left["n"] -= 1
            return True
        return False

    if stage == "read":
        real = worker_mod.TileWorker._load

        def load(self, tile):
            return worker_mod._Loaded(tile, None, None, None, 0, "injected read error") if hit(tile.id) else real(self, tile)

        monkeypatch.setattr(worker_mod.TileWorker, "_load", load)
    elif stage == "solve":
        real = worker_mod.TileWorker.solve_tile

        def solve(self, loaded):
            if hit(loaded.tile.id):
                raise RuntimeError("injected solve error")
            return real(self, loaded)

        monkeypatch.setattr(worker_mod.TileWorker, "solve_tile", solve)
    else:
        real = ResultStore.write_tile

        def write(self, tile, results):
            if hit(0 if tuple(tile.cells) == tuple(tile0.cells) else -1):
                raise OSError("injected write error")
            return real(self, tile, results)

        monkeypatch.setattr(ResultStore, "write_tile", write)


@pytest.mark.parametrize("stage", ["read", "solve", "write"])
def test_a_tile_that_fails_once_is_retried_and_written(case, monkeypatch, stage):
    from zvdvc.pipeline.tiling import load_plan

    cfg = _variant(case, f"once_{stage}", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    tiles, _ = load_plan(f"{cfg.workdir}/plan.json")
    _fail(monkeypatch, stage, 1, tiles[0])
    stats = coordinator.run(cfg, backend=_backend(), tile_ids=[0, 1])
    assert stats.missing == [] and stats[0].tiles_written == 2
    assert any(f"{stage} failed" in e for e in stats[0].errors)
    assert set(tiles[0].cells) | set(tiles[1].cells) <= ResultStore(cfg.output).written_cells()


@pytest.mark.parametrize("stage", ["read", "solve", "write"])
def test_a_tile_that_always_fails_is_reported_missing_until_a_resubmission_writes_it(case, monkeypatch, stage):
    from pathlib import Path

    from zvdvc.pipeline.tiling import load_plan

    cfg = _variant(case, f"always_{stage}", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    tiles, _ = load_plan(f"{cfg.workdir}/plan.json")
    _fail(monkeypatch, stage, 10**6, tiles[0])
    stats = coordinator.run(cfg, backend=_backend(), tile_ids=[0, 1])
    assert stats.missing == [0] and stats[0].tiles_written == 1
    assert any("tile 0 failed twice" in e for e in stats[0].errors)
    failed = Path(cfg.workdir) / "failed_tiles.json"
    assert json.loads(failed.read_text())["missing"] == [0]
    assert json.loads((Path(cfg.workdir) / "run_stats.json").read_text())["missing_tiles"] == [0]
    monkeypatch.undo()
    stats = coordinator.run(cfg, backend=_backend(), tile_ids=[0, 1])
    assert stats.missing == [] and stats[0].tiles_written == 1 and stats[0].tiles_skipped == 1
    assert not failed.exists()


def test_solve_tile_never_reports_an_unsolved_point_as_good(case, monkeypatch):
    from zvdvc.pipeline import batching
    from zvdvc.pipeline.tiling import load_plan
    from zvdvc.pipeline.worker import TileWorker
    from zvdvc.status import PointStatus

    cfg = _variant(case, "unsolved", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    tiles, _ = load_plan(f"{cfg.workdir}/plan.json")
    worker = TileWorker(cfg, backend=_backend(), points_store=coordinator.points_store(cfg))
    monkeypatch.setattr(batching, "iter_batches", lambda order, size: iter(()))
    out = worker.solve_tile(worker._load(tiles[0]))
    assert len(out["status"]) == tiles[0].n_points and (out["status"] == PointStatus.NOT_SEARCHED).all()
