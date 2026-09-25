import numpy as np
import pytest

from pydvc.geometry.box import Box
from pydvc.io.volume import ZarrVolume
from pydvc.synth.phantoms import DisplacementField, default_field, load_field, make_case, speckle, speckle_field, warp_volume

KINDS = ["rigid", "affine", "sinusoid", "inclusion"]
rng = np.random.default_rng(4)


@pytest.mark.parametrize("kind", KINDS)
def test_field_gradient_matches_finite_differences(kind):
    field = default_field(kind, (128, 128, 128))
    X = rng.uniform(0.0, 127.0, (40, 3))
    h = 1e-5
    fd = np.stack([(field(X + h * e) - field(X - h * e)) / (2 * h) for e in np.eye(3)], axis=-1)
    np.testing.assert_allclose(field.gradient(X), fd, atol=1e-8)


@pytest.mark.parametrize("kind", KINDS)
def test_inverse_map(kind):
    field = default_field(kind, (128, 128, 128))
    x = rng.uniform(0.0, 127.0, (40, 3))
    X = field.inverse_map(x)
    np.testing.assert_allclose(X + field(X), x, atol=1e-9)


def test_rigid_field_is_a_rotation_about_the_centre():
    from pydvc.geometry.warp import rotation_matrix

    angles = (0.02, -0.01, 0.015)
    field = DisplacementField("rigid", {"translation": (1.0, 2.0, 3.0), "rotation": angles, "centre": (10.0, 20.0, 30.0)})
    R = rotation_matrix(np.array([angles]))[0]
    X = rng.uniform(0.0, 50.0, (5, 3))
    np.testing.assert_allclose(field(X), (1.0, 2.0, 3.0) + (X - (10.0, 20.0, 30.0)) @ (R - np.eye(3)).T)


def test_inclusion_is_continuous_at_the_interface():
    field = DisplacementField("inclusion", {"centre": (0.0, 0.0, 0.0), "radius": 10.0, "strain": (0.01, -0.005, 0.0, 0.002, 0.0, 0.0)})
    d = rng.normal(size=(10, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    np.testing.assert_allclose(field(d * (10.0 - 1e-9)), field(d * (10.0 + 1e-9)), atol=1e-9)


def test_speckle_statistics_and_reproducibility():
    z = speckle_field(Box((0, 0, 0), (64, 64, 64)))
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05
    part = speckle_field(Box((10, 20, 30), (40, 50, 60)))
    np.testing.assert_array_equal(part, z[10:40, 20:50, 30:60])
    u16 = speckle((32, 32, 32))
    assert u16.dtype == np.uint16 and 25000 < u16.mean() < 40000


def test_integer_translation_is_an_exact_shift():
    f = speckle_field(Box((0, 0, 0), (32, 32, 32)))
    g = warp_volume(f, DisplacementField("affine", {"translation": (2.0, -1.0, 3.0)}))
    # g(x) = f(x - u): g[z, y, x] = f[z - 3, y + 1, x - 2]
    np.testing.assert_allclose(g[8:24, 8:24, 8:24], f[5:21, 9:25, 6:22], atol=1e-9)


def test_make_case_writes_a_consistent_case(tmp_path):
    from pydvc.config import RunConfig

    field = DisplacementField("affine", {"translation": (0.5, -0.25, 1.0), "strain": (0.004, 0.0, -0.002, 0.001, 0.0, 0.0)})
    config = make_case(tmp_path / "c", shape_zyx=(80, 72, 64), field=field, spacing=12, chunk=32, shard=64)
    cfg = RunConfig.from_yaml(config)
    ref, deformed = ZarrVolume(cfg.volumes.reference), ZarrVolume(cfg.volumes.deformed)
    assert ref.shape == deformed.shape == (80, 72, 64)
    truth = np.load(tmp_path / "c" / "truth.npz")
    np.testing.assert_allclose(truth["displacement"], field(truth["xyz"]))
    assert load_field(tmp_path / "c" / "truth.npz") == DisplacementField("affine", {k: list(v) for k, v in field.params.items()})
    # every point's subvolume (+ displacement + stencil) stays inside the volume
    margin = 16.0 + np.abs(truth["displacement"]).max() + 2.0
    assert (truth["xyz"] >= margin).all() and (truth["xyz"] <= np.array([63, 71, 79]) - margin).all()


def test_make_case_is_independent_of_chunking(tmp_path):
    field = DisplacementField("sinusoid", {"amplitude": 1.0, "wavelength": 40.0})
    kw = dict(shape_zyx=(48, 48, 48), field=field, spacing=16, noise_sigma=0.02)
    make_case(tmp_path / "a", chunk=16, shard=32, **kw)
    make_case(tmp_path / "b", chunk=48, shard=48, **kw)
    for name in ("ref.ome.zarr", "def.ome.zarr"):
        np.testing.assert_array_equal(ZarrVolume(tmp_path / "a" / name).array[...], ZarrVolume(tmp_path / "b" / name).array[...])
