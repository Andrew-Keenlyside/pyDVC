"""Engines prepare a brick once: prepare() is idempotent, and the in-memory runner prepares each volume once."""

import numpy as np
import pytest

from _fields import wave_field, whole_brick
from zvdvc.solver.engines import gpu_available, make_engine

BACKENDS = ["numpy", "numpy32", "cpu", "emulated", pytest.param("cupy", marks=pytest.mark.gpu),
            pytest.param("fused", marks=pytest.mark.gpu)]


def _engine(backend):
    if backend == "cpu":
        pytest.importorskip("numba")
    if backend == "emulated":
        from zvdvc.kernels.cuda.emulate import compiler

        if compiler() is None:
            pytest.skip("no C++20 compiler for the CUDA emulator")
    if backend in ("cupy", "fused") and not gpu_available():
        pytest.skip("no GPU")
    return make_engine(backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_prepare_is_idempotent(backend):
    eng = _engine(backend)
    once = eng.prepare(whole_brick(wave_field((24, 24, 24)).astype(np.float32)))
    assert eng.prepare(once) is once


def test_a_host_engine_brick_is_not_taken_as_prepared_by_a_gpu_engine():
    if not gpu_available():
        pytest.skip("no GPU")
    host = make_engine("numpy32").prepare(whole_brick(wave_field((24, 24, 24)).astype(np.float32)))
    dev = make_engine("cupy").prepare(host)
    assert dev is not host and not isinstance(dev.data, np.ndarray)


def test_inmemory_wavefront_prepares_each_volume_once(tmp_path, monkeypatch):
    from zvdvc.config import RunConfig, SearchSpec, SubvolumeSpec
    from zvdvc.pipeline import inmemory
    from zvdvc.solver import gauss_newton
    from zvdvc.synth.phantoms import default_field, make_case

    shape = (64, 64, 64)
    config = make_case(tmp_path / "case", shape_zyx=shape, field=default_field("rigid", shape), spacing=8.0,
                       chunk=32, shard=64, subvolume=SubvolumeSpec(geometry="sphere", size=12, n_samples=200),
                       search=SearchSpec(dof=3, disp_max=6.0))
    cfg = RunConfig.from_yaml(config)
    calls = {"converted": 0, "solve_batch": 0}

    def counting_engine(backend):
        eng = make_engine(backend)
        inner = eng.prepare

        def prepare(brick):
            out = inner(brick)
            calls["converted"] += out is not brick        # a real conversion (an upload, for a GPU engine)
            return out

        eng.prepare = prepare
        return eng

    real_solve = gauss_newton.solve_batch

    def counting_solve(*a, **k):
        calls["solve_batch"] += 1
        return real_solve(*a, **k)

    monkeypatch.setattr(inmemory, "make_engine", counting_engine)
    monkeypatch.setattr(inmemory, "solve_batch", counting_solve)
    point_id, xyz = inmemory.load_points(cfg)
    inmemory.solve_in_memory(cfg, point_id, xyz, strategy="wavefront", backend="numpy")
    assert calls["solve_batch"] > 2                 # several wavefront shells ...
    assert calls["converted"] == 2                  # ... but each volume is converted once


def test_an_emulated_fused_brick_moves_to_the_gpu_engine():
    if not gpu_available():
        pytest.skip("no GPU")
    emu = _engine("emulated").prepare(whole_brick(wave_field((24, 24, 24)).astype(np.float32)))
    dev = make_engine("fused").prepare(emu)
    assert dev is not emu and not isinstance(dev.data, np.ndarray) and not isinstance(dev.geom, np.ndarray)
