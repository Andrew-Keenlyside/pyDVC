"""Per-point displacement uncertainty from repeat solves with other template seeds."""

import dataclasses

import numpy as np
import pytest

from zvdvc.config import ClusterSpec, RunConfig, SearchSpec, SeedingSpec, SubvolumeSpec


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    from zvdvc.synth.phantoms import default_field, make_case

    root = tmp_path_factory.mktemp("uncertainty")
    shape = (72, 72, 72)
    c = RunConfig.from_yaml(make_case(root / "c", shape_zyx=shape, field=default_field("affine", shape), spacing=10.0,
                                      chunk=36, shard=72, noise_sigma=0.03,
                                      subvolume=SubvolumeSpec(geometry="sphere", size=16, n_samples=300),
                                      search=SearchSpec(dof=6, disp_max=6.0)))
    return dataclasses.replace(c, seeding=SeedingSpec(strategy="rigid"), cluster=ClusterSpec(tile_shape=(36, 36, 36)))


def _backend():
    try:
        import numba  # noqa: F401

        return "cpu"
    except ImportError:
        return "numpy32"


def test_repeat_solve_sd_matches_the_spread_of_independent_runs(cfg):
    from zvdvc.pipeline.inmemory import load_points, solve_in_memory

    pid, xyz = load_points(cfg)
    r = solve_in_memory(dataclasses.replace(cfg, uncertainty_seeds=3), pid, xyz, backend=_backend())
    assert r.displacement_sd is not None and r.displacement_sd.shape == (len(pid), 3)
    good = r.status == 0
    assert np.isfinite(r.displacement_sd[good]).mean() > 0.95 and np.isnan(r.displacement_sd[~good]).all()
    # independent runs with two other seeds: sd of their difference / sqrt(2) is the per-estimate spread
    runs = [solve_in_memory(dataclasses.replace(cfg, subvolume=dataclasses.replace(cfg.subvolume, seed=s)), pid, xyz,
                            backend=_backend()) for s in (10, 11)]
    both = good & (runs[0].status == 0) & (runs[1].status == 0)
    empirical = (runs[0].params[both, :3] - runs[1].params[both, :3]).std(axis=0).mean() / np.sqrt(2)
    predicted = np.sqrt(np.nanmean(r.displacement_sd[both] ** 2))
    assert 0.6 < predicted / empirical < 1.6, (predicted, empirical)


def test_the_tiled_pipeline_writes_displacement_sd(cfg, tmp_path):
    from zvdvc.io.results import ResultStore
    from zvdvc.pipeline import coordinator

    c = dataclasses.replace(cfg, output=str(tmp_path / "r.zarrvectors"), workdir=str(tmp_path / "w"), uncertainty_seeds=2)
    coordinator.prepare(c, backend=_backend())
    coordinator.run(c, backend=_backend())
    got = ResultStore(c.output).read_all()
    assert got["displacement_sd"].shape == (len(got["point_id"]), 3)
    assert np.isfinite(got["displacement_sd"][got["status"] == 0]).mean() > 0.95


def test_wavefront_seeding_writes_the_in_memory_displacement_sd(cfg, tmp_path):
    from zvdvc.io.results import ResultStore
    from zvdvc.pipeline import coordinator
    from zvdvc.pipeline.inmemory import load_points, solve_in_memory

    c = dataclasses.replace(cfg, output=str(tmp_path / "w.zarrvectors"), workdir=str(tmp_path / "ww"), uncertainty_seeds=2,
                            seeding=SeedingSpec(strategy="wavefront"))
    coordinator.prepare(c, backend=_backend())
    coordinator.seed(c, backend=_backend())
    got = ResultStore(c.output).read_all()
    order = np.argsort(got["point_id"])
    pid, xyz = load_points(c)
    ref = solve_in_memory(c, pid, xyz, strategy="wavefront", backend=_backend())
    np.testing.assert_array_equal(got["point_id"][order], pid)
    assert np.isfinite(ref.displacement_sd[ref.status == 0]).mean() > 0.95
    np.testing.assert_array_equal(got["displacement_sd"][order], ref.displacement_sd.astype(np.float32))


def test_without_repeats_the_store_holds_nan_and_older_stores_still_work(cfg, tmp_path):
    from zvdvc.io import results as results_mod
    from zvdvc.io.results import ResultStore
    from zvdvc.pipeline import coordinator

    c = dataclasses.replace(cfg, output=str(tmp_path / "a.zarrvectors"), workdir=str(tmp_path / "wa"))
    coordinator.prepare(c, backend=_backend())
    coordinator.run(c, backend=_backend())
    assert np.isnan(ResultStore(c.output).read_all()["displacement_sd"]).all()

    old = dataclasses.replace(cfg, output=str(tmp_path / "old.zarrvectors"), workdir=str(tmp_path / "wo"))
    saved = dict(results_mod.OPTIONAL_ATTRIBUTES)
    results_mod.OPTIONAL_ATTRIBUTES.clear()                          # a store from before optional attributes
    try:
        coordinator.prepare(old, backend=_backend())
    finally:
        results_mod.OPTIONAL_ATTRIBUTES.update(saved)
    coordinator.run(dataclasses.replace(old, uncertainty_seeds=1), backend=_backend())   # computed, not stored
    got = ResultStore(old.output).read_all()
    assert "displacement_sd" not in got and len(got["point_id"]) > 0


def test_strain_uses_the_per_point_uncertainty(tmp_path):
    from zvdvc.pipeline.inmemory import Results
    from zvdvc.post.strain import compute_strain

    ax = np.arange(9) * 8.0
    xyz = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    n = len(xyz)
    params = np.zeros((n, 6))
    params[:, :3] = xyz * 0.01
    base = dict(point_id=np.arange(n), xyz=xyz, status=np.zeros(n, np.int8), objmin=np.full(n, 0.01), params=params,
                n_iter=np.ones(n, np.uint8), seed=np.zeros((n, 3)))
    for sd, name in ((0.01, "small"), (0.04, "large")):
        Results(**base, displacement_sd=np.full((n, 3), sd)).save(tmp_path / f"{name}.npz")
        compute_strain(tmp_path / f"{name}.npz", outputs=())
    small = np.load(tmp_path / "small-sw25.strain.npz")["strain_sd"]
    large = np.load(tmp_path / "large-sw25.strain.npz")["strain_sd"]
    np.testing.assert_allclose(np.nanmedian(large / small), 4.0, rtol=1e-6)
