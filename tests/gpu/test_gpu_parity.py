"""M2: the cupy and fused paths agree with the float64 numpy reference."""

import numpy as np
import pytest

from _fields import wave_field, whole_brick
from zvdvc.config import SearchSpec, SubvolumeSpec
from zvdvc.geometry.templates import make_template
from zvdvc.solver.gauss_newton import solve_batch

pytestmark = pytest.mark.gpu

SHAPE = (64, 64, 64)


@pytest.mark.parametrize("backend", ["cupy", "fused"])
@pytest.mark.parametrize("dof", [3, 6, 12])
@pytest.mark.parametrize("kind", ["ssd", "znssd"])
def test_gpu_matches_numpy(backend, dof, kind):
    import cupy as cp

    u = (0.6, -0.3, 0.2)
    ref_np = wave_field(SHAPE)
    def_np = wave_field(SHAPE, shift_xyz=u)
    rng = np.random.default_rng(7)
    centres = rng.uniform(20.0, 44.0, size=(256, 3))
    seeds = np.zeros((256, 3))
    template = make_template(SubvolumeSpec(geometry="sphere", size=16, n_samples=1000))
    search = SearchSpec(dof=dof, objective=kind, disp_max=3.0)

    expected = solve_batch(whole_brick(ref_np), whole_brick(def_np), centres, seeds, template, search, backend="numpy")
    got = solve_batch(
        whole_brick(cp.asarray(ref_np, dtype=cp.float32)),
        whole_brick(cp.asarray(def_np, dtype=cp.float32)),
        cp.asarray(centres, dtype=cp.float32),
        cp.asarray(seeds, dtype=cp.float32),
        template,
        search,
        backend=backend,
    )

    status = cp.asnumpy(got.status)
    assert (status == expected.status).mean() >= 0.999
    good = expected.status == 0
    np.testing.assert_allclose(cp.asnumpy(got.displacement)[good], expected.displacement[good], atol=1e-3)


def test_gpu_pipeline_matches_the_cpu_engine(tmp_path):
    """M3 on a GPU: bricks read to the device, fused kernels, results store; parity with the CPU engine."""
    import dataclasses

    from zvdvc.config import ClusterSpec, RunConfig, SeedingSpec
    from zvdvc.io.results import ResultStore
    from zvdvc.pipeline import coordinator
    from zvdvc.synth.phantoms import default_field, make_case

    shape = (80, 80, 80)
    config = make_case(tmp_path / "case", shape_zyx=shape, field=default_field("affine", shape), spacing=8.0,
                       chunk=40, shard=80, subvolume=SubvolumeSpec(geometry="sphere", size=16, n_samples=500),
                       search=SearchSpec(dof=12, disp_max=8.0))
    base = RunConfig.from_yaml(config)
    out = {}
    for backend in ("cpu", "fused"):
        cfg = dataclasses.replace(base, output=str(tmp_path / f"{backend}.zarrvectors"), workdir=str(tmp_path / backend),
                                  cluster=ClusterSpec(tile_shape=(40, 40, 40)), seeding=SeedingSpec(strategy="rigid"))
        coordinator.prepare(cfg, backend=backend)
        stats = coordinator.run(cfg, backend=backend)
        assert not stats[0].errors
        r = ResultStore(cfg.output).read_all()
        o = np.argsort(r["point_id"])
        out[backend] = {k: v[o] for k, v in r.items()}
    assert (out["cpu"]["status"] == out["fused"]["status"]).mean() >= 0.999
    good = out["cpu"]["status"] == 0
    np.testing.assert_allclose(out["fused"]["displacement"][good], out["cpu"]["displacement"][good], atol=1e-3)


def _wave_bricks(u=(0.6, -0.3, 0.2)):
    import cupy as cp

    ref_np, def_np = wave_field(SHAPE), wave_field(SHAPE, shift_xyz=u)
    gpu = (whole_brick(cp.asarray(ref_np, dtype=cp.float32)), whole_brick(cp.asarray(def_np, dtype=cp.float32)))
    return (whole_brick(ref_np), whole_brick(def_np)), gpu


def test_fused_threshold_and_basin_search_match_numpy():
    """The status bookkeeping around the solve (threshold test, basin grid search) on real cupy arrays."""
    import cupy as cp

    from zvdvc.config import ThresholdSpec

    (ref, dfm), (ref_g, dfm_g) = _wave_bricks()
    rng = np.random.default_rng(3)
    centres = rng.uniform(20.0, 44.0, size=(128, 3))
    template = make_template(SubvolumeSpec(geometry="sphere", size=16, n_samples=1000))
    search = SearchSpec(dof=6, disp_max=3.0, basin_radius=1.0, threshold=ThresholdSpec(gray_min=100.0, gray_max=1e4, min_fraction=0.5))

    expected = solve_batch(ref, dfm, centres, np.zeros((128, 3)), template, search, backend="numpy")
    got = solve_batch(ref_g, dfm_g, cp.asarray(centres), cp.zeros((128, 3), dtype=cp.float32), template, search, backend="fused")

    status = cp.asnumpy(got.status)
    assert (expected.status == -4).any() and (expected.status == 0).any()      # both branches exercised
    assert (status == expected.status).mean() >= 0.999
    good = expected.status == 0
    np.testing.assert_allclose(cp.asnumpy(got.displacement)[good], expected.displacement[good], atol=1e-3)


def test_fused_solve_is_deterministic():
    """No atomics in the fused kernels, so the same batch gives bit-identical results (the check suite relies on it)."""
    import cupy as cp

    _, (ref_g, dfm_g) = _wave_bricks()
    centres = cp.asarray(np.random.default_rng(5).uniform(20.0, 44.0, size=(256, 3)))
    template = make_template(SubvolumeSpec(geometry="sphere", size=16, n_samples=1000))
    search = SearchSpec(dof=12, disp_max=3.0)
    runs = [solve_batch(ref_g, dfm_g, centres, cp.zeros((256, 3), dtype=cp.float32), template, search, backend="fused")
            for _ in range(2)]
    for name in ("params", "status", "n_iter", "objmin"):
        np.testing.assert_array_equal(cp.asnumpy(getattr(runs[0], name)), cp.asnumpy(getattr(runs[1], name)), err_msg=name)


def test_inmemory_wavefront_on_fused_matches_the_cpu_engine(tmp_path):
    """The in-memory wavefront runner (parity mode, `zvdvc seed`) on a GPU engine: host results, parity with cpu."""
    from zvdvc.config import RunConfig
    from zvdvc.pipeline.inmemory import load_points, solve_in_memory
    from zvdvc.synth.phantoms import default_field, make_case

    shape = (80, 80, 80)
    config = make_case(tmp_path / "case", shape_zyx=shape, field=default_field("affine", shape), spacing=8.0,
                       chunk=40, shard=80, subvolume=SubvolumeSpec(geometry="sphere", size=16, n_samples=500),
                       search=SearchSpec(dof=6, disp_max=8.0))
    cfg = RunConfig.from_yaml(config)
    point_id, xyz = load_points(cfg)
    out = {b: solve_in_memory(cfg, point_id, xyz, strategy="wavefront", backend=b) for b in ("cpu", "fused")}
    for name in ("status", "params", "objmin", "n_iter", "seed"):
        assert isinstance(getattr(out["fused"], name), np.ndarray), name
    assert (out["cpu"].status == out["fused"].status).mean() >= 0.999
    good = out["cpu"].status == 0
    assert good.mean() > 0.9
    np.testing.assert_allclose(out["fused"].params[good, :3], out["cpu"].params[good, :3], atol=1e-3)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_fused_packed_loads_on_an_odd_sized_misaligned_brick(dtype):
    """The u8/u16 word loads on the GPU, where stencils touch the last bytes of the buffer and the brick starts mid-word."""
    import cupy as cp

    from test_fused import ODD, _u8_odd_case
    from zvdvc.io.volume import is_padded

    ref, deformed, centres = _u8_odd_case(dtype)
    template = make_template(SubvolumeSpec(geometry="sphere", size=12, n_samples=400))
    search = SearchSpec(dof=6, objective="znssd", interpolation="tricubic", disp_max=3.0)
    seeds = np.zeros((len(centres), 3))
    expected = solve_batch(whole_brick(ref), whole_brick(deformed), centres, seeds, template, search, backend="numpy")
    k = deformed.itemsize
    view = cp.empty(deformed.nbytes + k, dtype=cp.uint8)[k:].view(deformed.dtype).reshape(ODD)   # misaligned brick
    view[...] = cp.asarray(deformed)
    assert not is_padded(view)
    got = solve_batch(whole_brick(cp.asarray(ref)), whole_brick(view), cp.asarray(centres), cp.asarray(seeds),
                      template, search, backend="fused")
    status = cp.asnumpy(got.status)
    good = expected.status == 0
    assert good.sum() > 100 and (~good).any()
    assert (status == expected.status).mean() >= 0.999
    np.testing.assert_allclose(cp.asnumpy(got.displacement)[good], expected.displacement[good], atol=1e-3)


def test_device_bricks_are_padded_and_used_without_a_copy():
    import cupy as cp

    from zvdvc.io.volume import is_padded, to_device
    from zvdvc.solver.engines import make_engine

    host = np.arange(np.prod((5, 7, 9)), dtype=np.uint8).reshape(5, 7, 9)
    dev = to_device(host)
    assert is_padded(dev) and np.array_equal(cp.asnumpy(dev), host)
    prepared = make_engine("fused").prepare(whole_brick(dev))
    assert prepared.data.data.ptr == dev.data.ptr                            # no second device copy
