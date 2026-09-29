"""The Gaussian prefilter: seamless bricks, GPU slabs equal to one pass, dtype kept, and runs that refuse to mix."""

import dataclasses

import numpy as np
import pytest

from zvdvc.config import VolumeSpec
from zvdvc.geometry.box import Box
from zvdvc.io import volume as vol
from zvdvc.solver.engines import gpu_available


def _spec(tmp_path, dtype=np.uint8, sigma=1.0, shape=(40, 36, 44)):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=shape).astype(dtype)
    path = tmp_path / "v.npy"
    np.save(path, data)
    return VolumeSpec(reference=str(path), deformed=str(path), prefilter_sigma=sigma), data


def test_zero_sigma_is_the_plain_volume(tmp_path):
    spec, _ = _spec(tmp_path, sigma=0.0)
    assert not isinstance(vol.open_volume(spec, "reference"), vol.FilteredVolume)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_a_brick_is_the_same_box_of_the_filtered_volume(tmp_path, dtype):
    spec, data = _spec(tmp_path, dtype=dtype)
    v = vol.open_volume(spec, "reference")
    whole = v.read_brick(Box((0, 0, 0), data.shape), device="cpu").data
    box = Box((5, 3, 30), (25, 30, 44))                       # touches the x edge, so padding is exercised
    part = v.read_brick(box, device="cpu").data
    assert whole.dtype == np.dtype(dtype) and part.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(part, whole[box.slices()])


def test_the_filter_matches_scipy_and_gpu_slabs_match_one_pass(tmp_path, monkeypatch):
    from scipy.ndimage import gaussian_filter

    _, data = _spec(tmp_path, dtype=np.float32)
    want = gaussian_filter(data, 1.0, mode="nearest", truncate=vol.TRUNCATE)
    got = vol.gaussian_filtered(data, 1.0, np.float32, slab=7)            # GPU in 7-slice slabs if there is one
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-3)
    monkeypatch.setattr(vol, "get_xp", lambda d: (_ for _ in ()).throw(ImportError("no cupy")))
    np.testing.assert_allclose(vol.gaussian_filtered(data, 1.0, np.float32), want, rtol=0, atol=1e-4)   # host path


def test_a_big_endian_volume_filters_as_the_little_endian_one(tmp_path, monkeypatch):
    monkeypatch.setattr(vol, "get_xp", lambda d: (_ for _ in ()).throw(ImportError("no cupy")))     # host path
    data = np.random.default_rng(1).integers(0, 4096, size=(20, 18, 22)).astype("<u2")
    for order in "<>":
        data.astype(f"{order}u2").tofile(tmp_path / f"{order == '>'}.raw")
    got = {}
    for order in "<>":
        spec = VolumeSpec(reference=str(tmp_path / f"{order == '>'}.raw"), deformed="-", raw_shape_xyz=(22, 18, 20),
                          raw_dtype=f"{order}u2", prefilter_sigma=1.0)
        v = vol.open_volume(spec, "reference")
        assert v.dtype == np.uint16 and v.dtype.isnative
        got[order] = v.read_brick(Box((2, -3, 4), (18, 15, 30)), device="cpu").data
    assert got[">"].dtype.isnative
    np.testing.assert_array_equal(got[">"], got["<"])


@pytest.mark.gpu
def test_device_bricks_are_filtered_on_the_device(tmp_path):
    if not gpu_available():
        pytest.skip("no GPU")
    spec, data = _spec(tmp_path)
    v = vol.open_volume(spec, "reference")
    box = Box((0, 0, 0), data.shape)
    dev = v.read_brick(box, device="cuda").data
    assert not isinstance(dev, np.ndarray)
    np.testing.assert_array_equal(dev.get(), v.read_brick(box, device="cpu").data)


def test_prefiltered_solve_recovers_a_known_field(tmp_path):
    from zvdvc.bench.metrics import compare_arrays
    from zvdvc.config import RunConfig, SearchSpec, SubvolumeSpec
    from zvdvc.pipeline.inmemory import load_points, solve_in_memory
    from zvdvc.synth.phantoms import default_field, make_case

    shape = (64, 64, 64)
    cfg = RunConfig.from_yaml(make_case(tmp_path / "c", shape_zyx=shape, field=default_field("affine", shape), spacing=12.0,
                                        chunk=32, shard=64, subvolume=SubvolumeSpec(geometry="sphere", size=16, n_samples=500),
                                        search=SearchSpec(dof=12, disp_max=6.0)))
    cfg = dataclasses.replace(cfg, volumes=dataclasses.replace(cfg.volumes, prefilter_sigma=1.0))
    pid, xyz = load_points(cfg)
    r = solve_in_memory(cfg, pid, xyz, strategy="rigid", backend="numpy")
    truth = np.load(tmp_path / "c" / "truth.npz")
    acc = compare_arrays(r.params[:, :3], truth["displacement"], r.status)
    assert acc.frac_good > 0.95 and max(acc.rmse) < 0.05
