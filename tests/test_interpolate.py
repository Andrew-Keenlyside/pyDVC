import numpy as np

from pydvc.kernels import interpolate

SHAPE = (24, 20, 16)                              # (z, y, x)
rng = np.random.default_rng(1)
POS = rng.uniform(3.0, 12.0, size=(4, 64, 3))     # (x, y, z), every stencil inside


def _grid_field(fn):
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in SHAPE), indexing="ij")
    return fn(x, y, z)


def test_trilinear_exact_on_linear_field():
    def lin(x, y, z):
        return 2.0 * x - 3.0 * y + 0.5 * z + 7.0

    s = interpolate.sample(_grid_field(lin), (0.0, 0.0, 0.0), POS, method="trilinear", with_grad=True)
    np.testing.assert_allclose(s.values, lin(*np.moveaxis(POS, -1, 0)), rtol=1e-10)
    np.testing.assert_allclose(s.grad, np.broadcast_to([2.0, -3.0, 0.5], s.grad.shape), rtol=1e-10)


def test_tricubic_exact_on_quadratic_field():
    # Catmull-Rom (a = -0.5) reproduces polynomials up to degree 2 per axis.
    def quad(x, y, z):
        return 0.1 * x**2 - 0.05 * y * z + 0.2 * z**2 + x - 4.0

    s = interpolate.sample(_grid_field(quad), (0.0, 0.0, 0.0), POS, method="tricubic", with_grad=True)
    x, y, z = np.moveaxis(POS, -1, 0)
    np.testing.assert_allclose(s.values, quad(x, y, z), rtol=1e-10, atol=1e-10)
    grad = np.stack([0.2 * x + 1.0, -0.05 * z, -0.05 * y + 0.4 * z], axis=-1)
    np.testing.assert_allclose(s.grad, grad, rtol=1e-8, atol=1e-10)


def test_catmull_rom_equals_ccpi_lekien_marsden():
    """The equivalence the whole GPU interpolation design rests on (docs/ARCHITECTURE.md section 7)."""
    brick = rng.uniform(0.0, 255.0, size=SHAPE)
    reference = interpolate.lekien_marsden_reference(brick, POS)
    s = interpolate.sample(brick, (0.0, 0.0, 0.0), POS, method="tricubic")
    np.testing.assert_allclose(s.values, reference, rtol=1e-5)


def test_inside_mask_flags_stencils_leaving_the_brick():
    pos = np.array([[[0.5, 5.0, 5.0], [8.0, 8.0, 8.0], [14.9, 5.0, 5.0]]])
    s = interpolate.sample(np.zeros(SHAPE), (0.0, 0.0, 0.0), pos, method="tricubic")
    assert s.inside.tolist() == [[False, True, False]]
