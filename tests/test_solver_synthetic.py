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
