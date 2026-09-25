"""Batched interpolation of a brick at warped sample positions, with gradients.

``tricubic`` is separable **Catmull-Rom** (cubic convolution, a = -0.5): 4
taps per axis, 64 voxel reads, with weights and derivative weights computed
once per axis. CCPi's tricubic is Lekien-Marsden, fed with central-difference
derivative kernels (f, fx, fy, fz, fxy, fxz, fyz, fxyz) precomputed over the
whole box. A tensor-product cubic Hermite interpolant with tensor-product
central-difference derivatives *is* tensor-product Catmull-Rom. So the same
interpolant needs no derivative kernels and no 64x64 coefficient solve per
cell, which were CCPi's two largest per-point costs.
**M1 checks this equivalence** against a direct port of the Lekien path on
random data (target: max relative difference below 1e-5). If it does not hold,
a ``tricubic_lekien`` variant evaluates the 64-voxel stencil with CCPi's exact
weights. It costs the same memory traffic and more FLOPs.

``trilinear`` has 8 taps and a piecewise-constant gradient; it is fine for
tuning but biased for strain. ``nearest`` is for the coarse grid search only,
since its gradient is zero.

Positions are point-space ``(x, y, z)``. The brick is ``(z, y, x)`` in native
dtype with its origin at ``Brick.origin_xyz``. Samples whose stencil leaves
the brick's valid region are reported in ``inside`` so the solver can flag
``RANGE_FAIL``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydvc._todo import todo

Method = Literal["nearest", "trilinear", "tricubic"]


@dataclass
class Samples:
    values: Any          # (B, M) float32
    grad: Any | None     # (B, M, 3) float32, d/dx, d/dy, d/dz; None unless requested
    inside: Any          # (B, M) bool: whole stencil inside the valid region


def catmull_rom_weights(t: Any) -> tuple[Any, Any]:
    """Fractional position (...,) -> (weights (..., 4), derivative weights (..., 4))."""
    raise todo("M1", "catmull_rom_weights")


def lekien_marsden_reference(brick: Any, positions: Any) -> Any:
    """float64 numpy port of CCPi's ``kernels_derivs`` + ``tri_cub_Lek`` (values only), for the M1 equivalence test.

    Test-only: slow, loops over cells.
    """
    raise todo("M1", "lekien_marsden_reference")


def sample(
    brick: Any,
    origin_xyz: tuple[float, float, float],
    positions: Any,
    *,
    method: Method = "tricubic",
    with_grad: bool = False,
    valid_lo_hi: tuple[Any, Any] | None = None,
) -> Samples:
    """Interpolate ``brick`` at ``positions`` (B, M, 3). numpy or cupy, following the inputs."""
    raise todo("M1", "interpolate.sample")
