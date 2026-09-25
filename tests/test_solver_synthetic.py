import numpy as np
import pytest

from _fields import wave_field, whole_brick
from pydvc.config import SearchSpec, SubvolumeSpec
from pydvc.geometry.templates import make_template
from pydvc.solver.gauss_newton import solve_batch
from pydvc.status import PointStatus

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
    from pydvc.config import ThresholdSpec

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
    from pydvc.geometry.box import Box

    return Box(tuple(v + 1000 for v in box.lo), tuple(v + 1000 for v in box.hi))


def test_basin_search_recovers_a_displacement_far_from_the_seed():
    from pydvc.geometry.box import Box
    from pydvc.synth.phantoms import DisplacementField, speckle_field, warp_volume

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
