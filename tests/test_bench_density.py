"""The density cost model recovers known coefficients."""

import numpy as np
import pytest

from pydvc.bench import density


def test_fit_recovers_t0_tau_c():
    n = np.array([500, 2000, 8000, 30000, 100000, 300000])
    shells = np.round(0.9 * n ** (1 / 3)).astype(int)
    t0, tau, c = 0.5, 0.08, 3e-4
    rows = [{"n_points": int(a), "n_shells": int(b), "fused": {"seconds": t0 + tau * b + c * a}} for a, b in zip(n, shells)]
    f = density.fit(rows, "fused")
    assert f["t0"] == pytest.approx(t0, rel=1e-6) and f["tau"] == pytest.approx(tau, rel=1e-6)
    assert f["c"] == pytest.approx(c, rel=1e-6) and f["max_rel_err"] < 1e-9


def test_model_reports_the_asymptotic_speedup():
    rows = [{"n_points": a, "n_shells": b, "fused": {"seconds": 1 + 1e-4 * a}, "cpu": {"seconds": 1 + 1e-3 * a}}
            for a, b in ((1000, 9), (10000, 20), (100000, 42))]
    m = density.model(rows, ["fused", "cpu"])
    assert m["asymptotic_speedup"] == pytest.approx(10.0, rel=1e-3)
