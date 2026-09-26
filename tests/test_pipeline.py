"""M3: plan -> seed -> run -> finalize on one process (CPU engines), resume, and tiled == whole-volume results."""

import dataclasses
import json

import numpy as np
import pytest

from pydvc.bench.metrics import against_truth
from pydvc.config import ClusterSpec, RunConfig, SearchSpec, SeedingSpec, SubvolumeSpec
from pydvc.io.results import ResultStore
from pydvc.pipeline import coordinator
from pydvc.synth.phantoms import default_field, make_case

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
    from pydvc.pipeline.inmemory import solve_in_memory

    cfg = _variant(case, "bitwise", seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend=_backend())
    coordinator.run(cfg, backend=_backend())
    got = ResultStore(cfg.output).read_all()
    order = np.argsort(got["point_id"])
    ref = solve_in_memory(cfg, got["point_id"][order], got["xyz"][order].astype(np.float64), strategy="rigid", backend=_backend())
    np.testing.assert_array_equal(got["status"][order], ref.status)
    np.testing.assert_array_equal(got["params"][order], ref.params.astype(np.float32))


def test_resume_skips_written_tiles_and_gives_the_same_store(case):
    from pydvc.pipeline.tiling import load_plan
    from pydvc.pipeline.worker import QueueSource, TileWorker

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
    from pydvc.pipeline.tiling import plan_document

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
