import numpy as np
import pytest

from _fields import wave_field, whole_brick
from zvdvc.config import SearchSpec, SubvolumeSpec
from zvdvc.geometry.templates import make_template
from zvdvc.solver.gauss_newton import solve_batch
from zvdvc.status import PointStatus

SHAPE = (64, 64, 64)
CENTRES = np.array([[32.0, 32.0, 32.0], [28.0, 36.0, 30.0]])


@pytest.mark.parametrize("dof", [3, 6, 12])
def test_recovers_subvoxel_translation(dof):
    u = np.array([1.3, -0.7, 0.45])
    ref = whole_brick(wave_field(SHAPE))
    deformed = whole_brick(wave_field(SHAPE, shift_xyz=tuple(u)))
    template = make_template(SubvolumeSpec(geometry="sphere", size=20, n_samples=1500))
    seeds = np.broadcast_to(np.round(u), (2, 3)).copy()      # as a coarse search would give

    res = solve_batch(ref, deformed, CENTRES, seeds, template, SearchSpec(dof=dof, disp_max=4.0), backend="numpy")

    assert (res.status == PointStatus.GOOD).all()
    np.testing.assert_allclose(res.displacement, np.broadcast_to(u, (2, 3)), atol=0.02)
    if dof > 3:
        np.testing.assert_allclose(res.params[:, 3:], 0.0, atol=1e-3)


def test_samples_leaving_the_brick_are_range_fail():
    ref = whole_brick(wave_field(SHAPE))
    template = make_template(SubvolumeSpec(geometry="sphere", size=20, n_samples=500))
    centres = np.array([[5.0, 32.0, 32.0]])                   # radius 10 reaches x < 0

    res = solve_batch(ref, ref, centres, np.zeros((1, 3)), template, SearchSpec(dof=3, disp_max=2.0), backend="numpy")

    assert res.status[0] == PointStatus.RANGE_FAIL


@pytest.mark.parametrize("kind", ["sad", "ssd", "zssd", "nssd", "znssd"])
@pytest.mark.parametrize("interpolation", ["trilinear", "tricubic"])
def test_every_objective_and_interpolation_converges(kind, interpolation):
    u = np.array([0.6, -0.35, 0.2])
    ref = whole_brick(wave_field(SHAPE))
    deformed = whole_brick(1.4 * wave_field(SHAPE, shift_xyz=tuple(u)) - 20.0 if kind in ("znssd",) else wave_field(SHAPE, shift_xyz=tuple(u)))
    template = make_template(SubvolumeSpec(geometry="sphere", size=20, n_samples=1000))
    search = SearchSpec(dof=6, objective=kind, interpolation=interpolation, disp_max=3.0)
    res = solve_batch(ref, deformed, CENTRES, np.zeros((2, 3)), template, search, backend="numpy")
    assert (res.status == PointStatus.GOOD).all()
    tol = 0.02 if interpolation == "tricubic" else 0.05          # trilinear is biased, as documented
    np.testing.assert_allclose(res.displacement, np.broadcast_to(u, (2, 3)), atol=tol)


def test_featureless_subvolume_is_singular():
    flat = whole_brick(np.full(SHAPE, 100.0))
    template = make_template(SubvolumeSpec(geometry="sphere", size=20, n_samples=500))
    res = solve_batch(flat, flat, CENTRES, np.zeros((2, 3)), template, SearchSpec(dof=6, objective="ssd"), backend="numpy")
    assert (res.status == PointStatus.SINGULAR).all()


def test_threshold_and_convergence_reporting():
    from zvdvc.config import ThresholdSpec

    ref = whole_brick(wave_field(SHAPE))
    deformed = whole_brick(wave_field(SHAPE, shift_xyz=(0.7, 0.0, 0.0)))
    template = make_template(SubvolumeSpec(geometry="sphere", size=20, n_samples=500))
    dark = SearchSpec(dof=3, threshold=ThresholdSpec(gray_min=0.0, gray_max=50.0))     # field is ~100 +- 60
    res = solve_batch(ref, deformed, CENTRES, np.zeros((2, 3)), template, dark, backend="numpy")
    assert (res.status == PointStatus.THRESH_FAIL).all() and (res.n_iter == 0).all()
    one_step = SearchSpec(dof=3, max_iterations=1, disp_tol=1e-9, obj_tol=0.0)
    res = solve_batch(ref, deformed, CENTRES, np.zeros((2, 3)), template, one_step, backend="numpy")
    assert (res.status == PointStatus.CONVG_FAIL).all()
    ccpi_like = SearchSpec(dof=3, max_iterations=1, disp_tol=1e-9, obj_tol=0.0, report_convg_fail=False)
    res = solve_batch(ref, deformed, CENTRES, np.zeros((2, 3)), template, ccpi_like, backend="numpy")
    assert (res.status == PointStatus.GOOD).all()


def test_displacement_beyond_disp_max_is_range_fail():
    u = (2.4, 0.0, 0.0)
    ref = whole_brick(wave_field(SHAPE))
    deformed = whole_brick(wave_field(SHAPE, shift_xyz=u))
    template = make_template(SubvolumeSpec(geometry="sphere", size=20, n_samples=800))
    seeds = np.broadcast_to([2.0, 0.0, 0.0], (2, 3))
    res = solve_batch(ref, deformed, CENTRES, seeds, template, SearchSpec(dof=3, disp_max=4.0), backend="numpy")
    assert (res.status == PointStatus.GOOD).all()
    res = solve_batch(ref, deformed, CENTRES, np.zeros((2, 3)) + [0.9, 0, 0], template, SearchSpec(dof=3, disp_max=1.0), backend="numpy")
    assert (res.status == PointStatus.RANGE_FAIL).all()


@pytest.mark.parametrize("dof", [3, 6, 12])
@pytest.mark.parametrize("kind", ["ssd", "zssd", "nssd", "znssd"])
def test_float32_path_matches_the_float64_reference(dof, kind):
    """``numpy32`` runs the cupy path's float32 arithmetic on the host (M2 parity criterion: 1e-3 voxel)."""
    u = (0.6, -0.3, 0.2)
    ref = whole_brick(wave_field(SHAPE).astype(np.float32))
    deformed = whole_brick(wave_field(SHAPE, shift_xyz=u).astype(np.float32))
    centres = np.random.default_rng(7).uniform(20.0, 44.0, size=(24, 3)) + 1000.0   # large coordinates on purpose
    ref.box, deformed.box = (_shifted(b.box) for b in (ref, deformed))
    ref.valid, deformed.valid = ref.box, deformed.box
    template = make_template(SubvolumeSpec(geometry="sphere", size=16, n_samples=800))
    search = SearchSpec(dof=dof, objective=kind, disp_max=3.0)
    expected = solve_batch(ref, deformed, centres, np.zeros((24, 3)), template, search, backend="numpy")
    got = solve_batch(ref, deformed, centres, np.zeros((24, 3)), template, search, backend="numpy32")
    assert got.params.dtype == np.float32
    assert (got.status == expected.status).mean() >= 0.999
    good = expected.status == 0
    assert good.all()
    np.testing.assert_allclose(got.displacement[good], expected.displacement[good], atol=1e-3)


def _shifted(box):
    from zvdvc.geometry.box import Box

    return Box(tuple(v + 1000 for v in box.lo), tuple(v + 1000 for v in box.hi))


def test_basin_search_recovers_a_displacement_far_from_the_seed():
    from zvdvc.geometry.box import Box
    from zvdvc.synth.phantoms import DisplacementField, speckle_field, warp_volume

    u = np.array([6.4, -5.6, 4.8])                            # |u| = 9.7: beyond plain GN's capture range
    f = speckle_field(Box((0, 0, 0), SHAPE))
    ref = whole_brick(f)
    deformed = whole_brick(warp_volume(f, DisplacementField("affine", {"translation": tuple(u)})))
    template = make_template(SubvolumeSpec(geometry="sphere", size=20, n_samples=600))
    no_basin = SearchSpec(dof=3, disp_max=10.0)
    with_basin = SearchSpec(dof=3, disp_max=10.0, basin_radius=1.0)
    far = solve_batch(ref, deformed, CENTRES, np.zeros((2, 3)), template, no_basin, backend="numpy")
    res = solve_batch(ref, deformed, CENTRES, np.zeros((2, 3)), template, with_basin, backend="numpy")
    assert not np.allclose(far.displacement, u, atol=0.05)
    assert (res.status == PointStatus.GOOD).all()
    np.testing.assert_allclose(res.displacement, np.broadcast_to(u, (2, 3)), atol=0.02)


ENGINES = [
    "numpy",
    "numpy32",
    "emulated",
    "cpu",
    pytest.param("cupy", marks=pytest.mark.gpu),
    pytest.param("fused", marks=pytest.mark.gpu),
]


def _engine_or_skip(backend):
    from zvdvc.solver.engines import make_engine

    if backend == "emulated":
        from zvdvc.kernels.cuda.emulate import compiler

        if compiler() is None:
            pytest.skip("no C++20 compiler for the CUDA emulator")
    if backend == "cpu":
        pytest.importorskip("numba")
    return make_engine(backend)


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _solve_on(backend, ref, deformed, search, seeds=None):
    eng = _engine_or_skip(backend)
    n = 60 if backend == "emulated" else 400
    template = make_template(SubvolumeSpec(geometry="sphere", size=12, n_samples=n))
    seeds = np.zeros((2, 3)) if seeds is None else seeds
    res = solve_batch(whole_brick(ref), whole_brick(deformed), CENTRES, seeds, template, search, engine=eng)
    return {k: _host(getattr(res, k)) for k in ("params", "status", "objmin", "n_iter")}


@pytest.mark.parametrize("backend", ENGINES)
@pytest.mark.parametrize("report_convg_fail", [True, False])
@pytest.mark.parametrize("kind", ["sad", "ssd", "zssd", "nssd", "znssd"])
@pytest.mark.parametrize("level", [0.0, 3000.0])
def test_constant_reference_is_singular_on_every_engine(backend, report_convg_fail, kind, level):
    """Air or zero padding in the reference: nothing to correlate, whatever the target holds (P0-6)."""
    ref = np.full(SHAPE, level, dtype=np.float32)
    deformed = wave_field(SHAPE, shift_xyz=(0.6, -0.3, 0.2)).astype(np.float32) * 30.0
    search = SearchSpec(dof=6, objective=kind, disp_max=6.0, report_convg_fail=report_convg_fail)
    for seeds in (np.zeros((2, 3)), np.full((2, 3), 2.0)):
        res = _solve_on(backend, ref, deformed, search, seeds)
        assert (res["status"] == PointStatus.SINGULAR).all()
        assert (res["n_iter"] == 0).all()


@pytest.mark.parametrize("backend", ENGINES)
@pytest.mark.parametrize("report_convg_fail", [True, False])
@pytest.mark.parametrize("kind", ["ssd", "znssd"])
def test_constant_target_is_not_good_on_every_engine(backend, report_convg_fail, kind):
    ref = wave_field(SHAPE).astype(np.float32)
    deformed = np.full(SHAPE, 100.0, dtype=np.float32)
    search = SearchSpec(dof=6, objective=kind, disp_max=6.0, report_convg_fail=report_convg_fail)
    assert (_solve_on(backend, ref, deformed, search)["status"] == PointStatus.SINGULAR).all()
    no_steps = SearchSpec(dof=6, objective=kind, max_iterations=0, report_convg_fail=report_convg_fail)
    assert (_solve_on(backend, ref, deformed, no_steps)["status"] == PointStatus.SINGULAR).all()


@pytest.mark.parametrize("backend", ENGINES)
@pytest.mark.parametrize("report_convg_fail", [True, False])
def test_textured_subvolume_is_still_solved_on_every_engine(backend, report_convg_fail):
    u = (0.6, -0.3, 0.2)
    ref = wave_field(SHAPE).astype(np.float32)
    deformed = wave_field(SHAPE, shift_xyz=u).astype(np.float32)
    search = SearchSpec(dof=6, objective="znssd", disp_max=3.0, report_convg_fail=report_convg_fail)
    res = _solve_on(backend, ref, deformed, search)
    assert (res["status"] == PointStatus.GOOD).all()
    np.testing.assert_allclose(res["params"][:, :3], np.broadcast_to(u, (2, 3)), atol=0.05)


@pytest.mark.parametrize("report_convg_fail", [True, False])
def test_non_finite_solution_is_never_good(report_convg_fail):
    """Whatever an engine's update leaves behind, NaN parameters are not reported GOOD."""
    from zvdvc.solver.engines import make_engine

    class NaNStep:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def update(self, st, idx, sums, outside, shift, search):
            st.params[idx] = np.nan
            return np.ones(len(idx), dtype=bool)

    template = make_template(SubvolumeSpec(geometry="sphere", size=12, n_samples=300))
    search = SearchSpec(dof=3, report_convg_fail=report_convg_fail)
    res = solve_batch(whole_brick(wave_field(SHAPE)), whole_brick(wave_field(SHAPE, shift_xyz=(0.6, 0.0, 0.0))),
                      CENTRES, np.zeros((2, 3)), template, search, engine=NaNStep(make_engine("numpy")))
    assert (res.status == PointStatus.SINGULAR).all()


@pytest.mark.parametrize("backend", ["numpy", "numpy32"])
@pytest.mark.parametrize("report_convg_fail", [True, False])
def test_non_finite_volumes_are_never_good(backend, report_convg_fail):
    for bad in (np.nan, np.inf):
        for which in ("ref", "deformed"):
            vols = {"ref": wave_field(SHAPE), "deformed": wave_field(SHAPE, shift_xyz=(0.6, 0.0, 0.0))}
            vols[which][26:38, 26:38, 26:38] = bad
            res = _solve_on(backend, vols["ref"], vols["deformed"], SearchSpec(dof=6, report_convg_fail=report_convg_fail))
            assert (res["status"] != PointStatus.GOOD).all()
