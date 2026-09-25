import numpy as np
import pytest

from pydvc.kernels import objective

rng = np.random.default_rng(2)
REF = rng.uniform(10.0, 200.0, size=(3, 500))


@pytest.mark.parametrize("kind", ["sad", "ssd", "zssd", "nssd", "znssd"])
def test_identical_subvolumes_score_zero(kind):
    np.testing.assert_allclose(objective.objective(REF, REF.copy(), kind), 0.0, atol=1e-12)


def test_ssd_value():
    np.testing.assert_allclose(objective.objective(REF, REF + 2.0, "ssd"), 4.0 * REF.shape[1])


def test_znssd_ignores_gain_and_offset():
    np.testing.assert_allclose(objective.objective(REF, 1.7 * REF + 30.0, "znssd"), 0.0, atol=1e-12)


def test_nssd_ignores_gain():
    np.testing.assert_allclose(objective.objective(REF, 2.5 * REF, "nssd"), 0.0, atol=1e-12)


def test_znssd_anticorrelated_scores_one():
    # CCPi scales ZNSSD by 1/4: fully anti-correlated subvolumes give exactly 1.
    np.testing.assert_allclose(objective.objective(REF, -REF, "znssd"), 1.0, rtol=1e-12)


@pytest.mark.parametrize("kind", ["nssd", "znssd"])
def test_normalised_objectives_in_unit_range(kind):
    tar = rng.uniform(0.0, 255.0, size=REF.shape)
    val = objective.objective(REF, tar, kind)
    assert np.all((val >= 0.0) & (val <= 1.0))


@pytest.mark.parametrize("kind", ["sad", "ssd", "zssd", "nssd", "znssd"])
@pytest.mark.parametrize("masked", [False, True])
def test_normal_equations_use_the_exact_residual_jacobian(kind, masked):
    """H = Jr'Jr and b = Jr'r for the finite-difference Jacobian of the residual vector."""
    B, M, n = 2, 300, 6
    tar = 1.3 * REF[:B, :M] + 5.0 + rng.normal(0.0, 8.0, (B, M))
    J = rng.normal(0.0, 5.0, (B, M, n))
    mask = rng.uniform(size=(B, M)) > 0.1 if masked else None
    H, b, obj = objective.normal_equations(REF[:B, :M], tar, J, kind, mask=mask)
    r0, _ = objective.residuals(REF[:B, :M], tar, kind, mask=mask)
    h = 1e-6
    Jr = np.stack(
        [
            (objective.residuals(REF[:B, :M], tar + h * J[..., k], kind, mask=mask)[0]
             - objective.residuals(REF[:B, :M], tar - h * J[..., k], kind, mask=mask)[0]) / (2 * h)
            for k in range(n)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(H, np.einsum("bmk,bml->bkl", Jr, Jr), rtol=1e-6, atol=1e-8 * np.abs(H).max())
    np.testing.assert_allclose(b, np.einsum("bmk,bm->bk", Jr, r0), rtol=1e-6, atol=1e-8 * np.abs(b).max())
    np.testing.assert_allclose(obj, objective.objective(REF[:B, :M], tar, kind, mask=mask))


def test_reference_stats():
    stats = objective.reference_stats(REF, "znssd")
    np.testing.assert_allclose(stats[:, 0], REF.mean(axis=1))
    np.testing.assert_allclose(stats[:, 1], np.linalg.norm(REF - REF.mean(axis=1, keepdims=True), axis=1))
    np.testing.assert_allclose(objective.reference_stats(REF, "ssd"), np.stack([np.zeros(3), np.ones(3)], axis=1))
