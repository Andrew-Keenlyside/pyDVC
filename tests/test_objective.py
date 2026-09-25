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
