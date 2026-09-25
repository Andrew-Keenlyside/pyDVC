"""M2 without a GPU: the CUDA source (emulated on the host, and compiled by NVRTC) and the numba CPU engine
agree with the float64 numpy reference (parity criterion: 1e-3 voxel on GOOD points, >= 99.9 % status agreement)."""

import numpy as np
import pytest

from _fields import wave_field, whole_brick
from pydvc.config import SearchSpec, SubvolumeSpec
from pydvc.geometry.templates import make_template
from pydvc.solver.gauss_newton import solve_batch
from pydvc.status import PointStatus

SHAPE = (48, 48, 48)
U = (0.6, -0.3, 0.2)


def _case(dtype, n_points, rng_seed=7):
    ref = wave_field(SHAPE)
    deformed = wave_field(SHAPE, shift_xyz=U)
    if np.dtype(dtype).kind == "u":
        scale = 250.0 / 220.0 if np.dtype(dtype) == np.uint8 else 200.0
        ref, deformed = (np.clip(np.rint((v - 100.0) * scale + (128 if dtype == np.uint8 else 30000)), 0, np.iinfo(dtype).max).astype(dtype)
                         for v in (ref, deformed))
    else:
        ref, deformed = ref.astype(dtype), deformed.astype(dtype)
    centres = np.random.default_rng(rng_seed).uniform(16.0, 32.0, size=(n_points, 3))
    return whole_brick(ref), whole_brick(deformed), centres


def _compare(backend, dtype, dof, kind, interpolation, n_points, n_samples, **engine_kw):
    ref, deformed, centres = _case(dtype, n_points)
    template = make_template(SubvolumeSpec(geometry="sphere", size=12, n_samples=n_samples))
    search = SearchSpec(dof=dof, objective=kind, interpolation=interpolation, disp_max=3.0)
    seeds = np.zeros((n_points, 3))
    expected = solve_batch(ref, deformed, centres, seeds, template, search, backend="numpy")
    got = solve_batch(ref, deformed, centres, seeds, template, search, backend=backend)
    assert got.params.dtype == np.float32
    assert (got.status == expected.status).mean() >= 0.999
    good = expected.status == PointStatus.GOOD
    assert good.mean() > 0.9
    np.testing.assert_allclose(got.displacement[good], expected.displacement[good], atol=1e-3)
    return expected, got


def _emulator_or_skip():
    from pydvc.kernels.cuda.emulate import compiler

    if compiler() is None:
        pytest.skip("no C++20 compiler for the CUDA emulator")


@pytest.mark.parametrize(
    "dof, kind, interpolation, dtype",
    [
        (6, "znssd", "tricubic", np.uint16),
        (12, "ssd", "tricubic", np.float32),
        (3, "nssd", "trilinear", np.uint8),
        (6, "zssd", "tricubic", np.uint8),
        (3, "sad", "nearest", np.float32),
    ],
)
def test_emulated_cuda_kernels_match_numpy(dof, kind, interpolation, dtype):
    _emulator_or_skip()
    if interpolation == "nearest":
        # nearest has no gradient: only the value path (reference and final samples) is meaningful
        from pydvc.solver.engines import make_engine

        ref, _, centres = _case(dtype, 4)
        template = make_template(SubvolumeSpec(geometry="sphere", size=12, n_samples=200))
        search = SearchSpec(dof=dof, objective=kind, interpolation="nearest")
        params = np.zeros((4, 3), dtype=np.float32) + np.float32(0.3)
        vals = {}
        for backend in ("numpy", "emulated"):
            eng = make_engine(backend)
            v, inside = eng.sample(eng.prepare(ref), centres, params.astype(eng.dtype), template.offsets.astype(eng.dtype), search)
            vals[backend] = np.asarray(v, dtype=np.float64)
            assert inside.all()
        np.testing.assert_allclose(vals["emulated"], vals["numpy"], rtol=1e-6)
        return
    _compare("emulated", dtype, dof, kind, interpolation, n_points=6, n_samples=300)


def test_emulated_kernels_flag_range_and_singular_points():
    _emulator_or_skip()
    ref, deformed, _ = _case(np.float32, 3)
    flat = whole_brick(np.full(SHAPE, 7.0, dtype=np.float32))
    template = make_template(SubvolumeSpec(geometry="sphere", size=12, n_samples=200))
    centres = np.array([[3.0, 24.0, 24.0], [24.0, 24.0, 24.0]])       # first: samples leave the brick
    res = solve_batch(ref, deformed, centres, np.zeros((2, 3)), template, SearchSpec(dof=6), backend="emulated")
    assert res.status[0] == PointStatus.RANGE_FAIL and res.status[1] == PointStatus.GOOD
    res = solve_batch(flat, flat, centres[1:], np.zeros((1, 3)), template, SearchSpec(dof=6, objective="ssd"), backend="emulated")
    assert res.status[0] == PointStatus.SINGULAR


@pytest.mark.parametrize("dof", [3, 6, 12])
@pytest.mark.parametrize("kind", ["sad", "ssd", "zssd", "nssd", "znssd"])
@pytest.mark.parametrize("interpolation", ["trilinear", "tricubic"])
def test_numba_cpu_engine_matches_numpy(dof, kind, interpolation):
    pytest.importorskip("numba")
    _compare("cpu", np.uint16, dof, kind, interpolation, n_points=40, n_samples=500)


def test_basin_search_on_the_fused_engines():
    pytest.importorskip("numba")
    u = np.array([2.6, -1.9, 1.4])
    ref, _, centres = _case(np.float32, 2)
    deformed = whole_brick(wave_field(SHAPE, shift_xyz=tuple(u)).astype(np.float32))
    template = make_template(SubvolumeSpec(geometry="sphere", size=12, n_samples=300))
    search = SearchSpec(dof=3, disp_max=4.0, basin_radius=1.0)
    res = solve_batch(ref, deformed, centres, np.zeros((2, 3)), template, search, backend="cpu")
    assert (res.status == PointStatus.GOOD).all()
    np.testing.assert_allclose(res.displacement, np.broadcast_to(u, (2, 3)), atol=2e-3)


def test_cuda_source_compiles_with_nvrtc():
    nvrtc = pytest.importorskip("cupy_backends.cuda.libs.nvrtc")
    from pydvc.kernels.cuda import SOURCE

    prog = nvrtc.createProgram(SOURCE.read_text(), "fused_gn.cu", [], [])
    exprs = [
        "pydvc::gn_sums<6, 4, 2, unsigned short>",
        "pydvc::gn_sums<12, 1, 2, float>",
        "pydvc::gn_solve<12, 4>",
        "pydvc::sample_values<3, 1, unsigned char>",
    ]
    for e in exprs:
        nvrtc.addNameExpression(prog, e)
    try:
        nvrtc.compileProgram(prog, ["-std=c++17", "-arch=compute_80"])
    except Exception as exc:  # the log names the error
        pytest.fail(f"NVRTC: {exc}\n{nvrtc.getProgramLog(prog)}")
    assert len(nvrtc.getPTX(prog)) > 10_000
