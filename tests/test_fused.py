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
        "pydvc::gn_sums<6, 4, 2, unsigned char>",          # u8 tricubic: the packed row loads
        "pydvc::sample_values<6, 2, unsigned char>",
        "pydvc::sample_values<6, 2, unsigned short>",
    ]
    for e in exprs:
        nvrtc.addNameExpression(prog, e)
    try:
        nvrtc.compileProgram(prog, ["-std=c++17", "-arch=compute_80"])
    except Exception as exc:  # the log names the error
        pytest.fail(f"NVRTC: {exc}\n{nvrtc.getProgramLog(prog)}")
    assert len(nvrtc.getPTX(prog)) > 10_000


ODD = (45, 47, 49)       # z, y, x: rows of 49 bytes, so words straddle rows and the buffer ends mid-word


def _u8_odd_case(dtype=np.uint8):
    ref = wave_field(ODD)
    deformed = wave_field(ODD, shift_xyz=U)
    scale, mid = (250.0 / 220.0, 128) if dtype == np.uint8 else (200.0, 30000)
    ref, deformed = (np.clip(np.rint((v - 100.0) * scale + mid), 0, np.iinfo(dtype).max).astype(dtype) for v in (ref, deformed))
    hi = np.asarray(ODD[::-1], dtype=np.float64) - 1.0
    rng = np.random.default_rng(11)
    # centres whose tricubic stencils reach (or just miss) the last voxels along x, and the last y/z rows
    corner = hi - rng.uniform(6.5, 10.5, size=(96, 3))      # from ~7.4 voxels in, stencils leave the brick
    faces = rng.uniform(12.0, 30.0, size=(96, 3))
    faces[:, 0] = hi[0] - rng.uniform(6.5, 10.5, size=96)
    return ref, deformed, np.concatenate([corner, faces])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_packed_row_loads_at_the_end_of_an_odd_sized_brick(dtype):
    """The u8/u16 word loads agree with the numpy reference where stencils touch the last bytes of the buffer."""
    _emulator_or_skip()
    ref, deformed, centres = _u8_odd_case(dtype)
    template = make_template(SubvolumeSpec(geometry="sphere", size=12, n_samples=400))
    search = SearchSpec(dof=6, objective="znssd", interpolation="tricubic", disp_max=3.0)
    seeds = np.zeros((len(centres), 3))
    expected = solve_batch(whole_brick(ref), whole_brick(deformed), centres, seeds, template, search, backend="numpy")
    # hand the engine a view that starts 1 byte into its buffer: prepare must re-align it
    raw = np.empty(deformed.nbytes + deformed.itemsize, dtype=np.uint8)[deformed.itemsize:]   # starts mid-word
    shifted = raw.view(deformed.dtype).reshape(ODD)
    shifted[...] = deformed
    got = solve_batch(whole_brick(ref), whole_brick(shifted), centres, seeds, template, search, backend="emulated")
    good = expected.status == PointStatus.GOOD
    assert good.sum() > 100 and (~good).any()        # both GOOD points and stencils that leave the brick
    assert (got.status == expected.status).mean() >= 0.999
    np.testing.assert_allclose(got.displacement[good], expected.displacement[good], atol=1e-3)


def test_padded_empty_is_aligned_with_a_readable_tail():
    from pydvc.io.volume import PAD_BYTES, padded_empty

    a = padded_empty(ODD, np.uint8)
    assert a.flags.c_contiguous and a.ctypes.data % 16 == 0 and a.shape == ODD
    assert a.base is not None and a.base.nbytes - a.nbytes >= PAD_BYTES
