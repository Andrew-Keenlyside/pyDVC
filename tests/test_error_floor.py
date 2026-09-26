"""The error-floor study runs end to end on a small speckle volume, and integer shifts are recovered exactly."""

import numpy as np

from pydvc.bench import error_floor as ef
from pydvc.geometry.box import Box
from pydvc.synth.phantoms import _to_dtype, speckle_field


def _image(shape=(64, 64, 64)):
    return _to_dtype(speckle_field(Box((0, 0, 0), shape), seed=1), np.dtype(np.uint8)).astype(np.float32)


def test_error_floor_on_speckle(tmp_path):
    rep = ef.run(_image(), out=tmp_path, backend="numpy32", shifts=(0.0, 0.5, 1.0), directions=("x",),
                 variants=("one-sided", "symmetric"), noise_factors=(0.0,), sizes=(16.0,), n_samples=(500,),
                 workers=1, progress=lambda m: None)
    assert rep["self_correlation"]["frac_good"] == 1.0 and rep["self_correlation"]["max_abs_displacement"] < 1e-5
    assert rep["sigma"] > 0
    for c in rep["cases"]:
        assert c["frac_good"] == 1.0 and c["n_points"] == 27
        if c["shift"] in (0.0, 1.0):           # B-spline shifts by whole voxels are exact; Catmull-Rom is shift-invariant
            assert max(abs(b) for b in c["bias"]) < 5e-3, c
        else:
            assert abs(c["bias"][0]) < 0.1, c
    s = ef.summary(rep)
    assert set(s["bias_vs_shift_x"]) == {"one-sided", "symmetric"} and np.isfinite(s["worst_abs_bias_noise_free"])


def test_noise_estimate_recovers_added_noise():
    rng = np.random.default_rng(0)
    smooth = np.zeros((32, 128, 128), dtype=np.float64)            # a flat image: all of the estimate is noise
    assert abs(ef.estimate_noise(smooth + rng.normal(0.0, 3.0, smooth.shape)) - 3.0) < 0.15


def test_crop_is_chosen_inside_the_material():
    vol = np.zeros((96, 96, 96), dtype=np.uint8)
    vol[40:90, 30:80, 10:60] = 200                                  # a bright block: the crop should land in it
    z, y, x = ef.choose_crop(vol, vol.shape, 32, stride=4)
    assert 40 <= z <= 58 and 30 <= y <= 48 and 10 <= x <= 28
