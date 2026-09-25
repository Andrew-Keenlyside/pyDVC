import numpy as np
import pytest

from pydvc.geometry import warp

rng = np.random.default_rng(0)
OFFSETS = rng.uniform(-10.0, 10.0, size=(50, 3))


@pytest.mark.parametrize("dof", [3, 6, 12])
def test_zero_params_is_identity(dof):
    centres = np.array([[5.0, 6.0, 7.0]])
    out = warp.warp(centres, np.zeros((1, dof)), OFFSETS)
    np.testing.assert_allclose(out[0], centres[0] + OFFSETS, atol=1e-12)


def test_translation_only():
    p = np.array([[1.5, -2.0, 0.25]])
    out = warp.warp(np.zeros((1, 3)), p, OFFSETS)
    np.testing.assert_allclose(out[0], OFFSETS + p[0], atol=1e-12)


def test_rotation_is_proper_orthonormal():
    R = warp.rotation_matrix(rng.uniform(-0.3, 0.3, size=(8, 3)))
    np.testing.assert_allclose(R @ np.swapaxes(R, -1, -2), np.broadcast_to(np.eye(3), R.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_rotation_matches_ccpi_roll_pitch_yaw():
    phi, the, psi = 0.1, -0.2, 0.3
    R = warp.rotation_matrix(np.array([[phi, the, psi]]))[0]
    # SearchParams::set_param_vect
    assert R[0, 0] == pytest.approx(np.cos(the) * np.cos(phi))
    assert R[0, 2] == pytest.approx(-np.sin(the))
    assert R[1, 2] == pytest.approx(np.cos(the) * np.sin(psi))
    assert R[2, 2] == pytest.approx(np.cos(the) * np.cos(psi))


def test_strain_layout_and_order_after_rotation():
    p = np.zeros((1, 12))
    p[0, 3:6] = (0.1, 0.05, -0.02)
    p[0, 6:12] = (0.01, -0.02, 0.005, 0.003, 0.0015, -0.004)   # exx eyy ezz exy eyz exz
    E = warp.strain_tensor(p[:, 6:12])[0]
    assert (E[0, 1], E[1, 2], E[0, 2]) == pytest.approx((0.003, 0.0015, -0.004))
    np.testing.assert_allclose(E, E.T)
    R = warp.rotation_matrix(p[:, 3:6])[0]
    np.testing.assert_allclose(warp.deformation_matrix(p)[0], (np.eye(3) + E) @ R, atol=1e-12)


@pytest.mark.parametrize("dof", [3, 6, 12])
def test_analytic_jacobian_matches_finite_differences(dof):
    p = rng.normal(scale=0.01, size=(2, dof))
    p[:, :3] = rng.normal(size=(2, 3))
    centres = np.zeros((2, 3))
    J = warp.warp_jacobian(p, OFFSETS)                     # (B, M, 3, ndof)
    h = 1e-6
    for k in range(dof):
        dp = np.zeros_like(p)
        dp[:, k] = h
        fd = (warp.warp(centres, p + dp, OFFSETS) - warp.warp(centres, p - dp, OFFSETS)) / (2 * h)
        np.testing.assert_allclose(J[..., k], fd, rtol=1e-4, atol=1e-6)
