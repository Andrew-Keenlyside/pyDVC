import numpy as np
import pytest

from zvdvc.config import SubvolumeSpec
from zvdvc.geometry.templates import cube_side, make_template


@pytest.mark.parametrize("n, k", [(1, 1), (8, 2), (27, 3), (28, 4), (64, 4), (8000, 20), (8001, 21)])
def test_cube_rounds_up_to_a_cube_number(n, k):
    assert cube_side(n) == k
    assert make_template(SubvolumeSpec(geometry="cube", size=10, n_samples=n)).n_samples == k**3


def test_cube_spans_the_side_with_aspect():
    t = make_template(SubvolumeSpec(geometry="cube", size=20, n_samples=125, aspect=(1.0, 0.5, 2.0)))
    np.testing.assert_allclose(t.offsets.min(axis=0), [-10, -5, -20])
    np.testing.assert_allclose(t.offsets.max(axis=0), [10, 5, 20])


def test_sphere_count_radius_and_uniformity():
    t = make_template(SubvolumeSpec(geometry="sphere", size=40, n_samples=20000))
    r = np.linalg.norm(t.offsets, axis=1)
    assert t.n_samples == 20000 and t.offsets.dtype == np.float32
    assert r.max() <= 20.0 + 1e-5
    # uniform in the ball: P(r < R/2) = 1/8
    assert abs((r < 10).mean() - 0.125) < 0.01
    assert np.abs(t.offsets.mean(axis=0)).max() < 0.3
    assert t.extent() == pytest.approx(r.max(), rel=1e-6)


def test_sphere_is_deterministic_per_seed():
    a = make_template(SubvolumeSpec(geometry="sphere", size=20, n_samples=500, seed=3))
    b = make_template(SubvolumeSpec(geometry="sphere", size=20, n_samples=500, seed=3))
    c = make_template(SubvolumeSpec(geometry="sphere", size=20, n_samples=500, seed=4))
    np.testing.assert_array_equal(a.offsets, b.offsets)
    assert a.digest() == b.digest() != c.digest()


_LEXSORT = np.lexsort         # the test below patches numpy's (templates.np is numpy)


def _zyx_sorted(o):
    return np.array_equal(o, o[_LEXSORT((o[:, 0], o[:, 1], o[:, 2]))])


def test_sphere_offsets_are_in_zyx_order_and_the_same_samples(monkeypatch):
    from zvdvc.geometry import templates

    spec = SubvolumeSpec(geometry="sphere", size=30, n_samples=3000)
    sorted_t = templates.make_template(spec)
    assert _zyx_sorted(sorted_t.offsets)
    monkeypatch.setattr(templates.np, "lexsort", lambda keys: np.arange(len(keys[0])))   # the pre-sorting template
    unsorted = templates.make_template(spec)
    assert not _zyx_sorted(unsorted.offsets)
    key = lambda o: o[_LEXSORT(o.T[::-1])]
    np.testing.assert_array_equal(key(sorted_t.offsets), key(unsorted.offsets))


def test_cube_offsets_were_already_in_zyx_order():
    t = make_template(SubvolumeSpec(geometry="cube", size=10, n_samples=1000))
    assert _zyx_sorted(t.offsets)
