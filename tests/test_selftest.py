import pytest

from pydvc.bench import smoke


def test_backend_choice():
    env = {"gpus": [], "packages": {"numba": "0.60"}}
    assert smoke.available_backends(env) == ["cpu"]
    assert smoke.available_backends({"gpus": [], "packages": {"numba": None}}) == ["numpy"]
    assert smoke.available_backends({"gpus": ["0: H100"], "packages": {"numba": None}}) == ["fused", "cupy", "numpy"]
    assert smoke.available_backends({"gpus": ["0: H100"], "packages": {"numba": "0.60"}}) == ["fused", "cupy", "cpu"]


def test_selftest_passes_on_the_cpu_engine(tmp_path):
    pytest.importorskip("numba")
    ok, report = smoke.run(tmp_path, backends=["cpu"], ccpi=False)
    assert ok, report["checks"]
    assert (tmp_path / "selftest.json").exists()
