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
from functools import lru_cache
from typing import Any, Literal

import numpy as np

from pydvc.kernels.xp import xp_of

Method = Literal["nearest", "trilinear", "tricubic"]

# taps each method reads either side of floor(x): [floor - before, floor + after]
_STENCIL = {"nearest": (0, 0), "trilinear": (0, 1), "tricubic": (1, 2)}
# samples per gather: bounds the (chunk, 64) index and value temporaries
_CHUNK_HOST = 1 << 16
_CHUNK_DEVICE = 1 << 21


@dataclass
class Samples:
    values: Any          # (B, M) float32
    grad: Any | None     # (B, M, 3) float32, d/dx, d/dy, d/dz; None unless requested
    inside: Any          # (B, M) bool: whole stencil inside the valid region


def catmull_rom_weights(t: Any) -> tuple[Any, Any]:
    """Fractional position (...,) -> (weights (..., 4), derivative weights (..., 4)).

    Taps at offsets -1, 0, 1, 2 from ``floor(x)``; ``t = x - floor(x)``.
    """
    xp = xp_of(t)
    t2 = t * t
    t3 = t2 * t
    w = xp.stack(
        [0.5 * (-t3 + 2.0 * t2 - t), 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0), 0.5 * (-3.0 * t3 + 4.0 * t2 + t), 0.5 * (t3 - t2)],
        axis=-1,
    )
    dw = xp.stack(
        [0.5 * (-3.0 * t2 + 4.0 * t - 1.0), 0.5 * (9.0 * t2 - 10.0 * t), 0.5 * (-9.0 * t2 + 8.0 * t + 1.0), 0.5 * (3.0 * t2 - 2.0 * t)],
        axis=-1,
    )
    return w, dw


def _linear_weights(t: Any) -> tuple[Any, Any]:
    xp = xp_of(t)
    return xp.stack([1.0 - t, t], axis=-1), xp.stack([-xp.ones_like(t), xp.ones_like(t)], axis=-1)


def sample(
    brick: Any,
    origin_xyz: tuple[float, float, float],
    positions: Any,
    *,
    method: Method = "tricubic",
    with_grad: bool = False,
    valid_lo_hi: tuple[Any, Any] | None = None,
) -> Samples:
    """Interpolate ``brick`` at ``positions`` (B, M, 3). numpy or cupy, following the inputs.

    ``valid_lo_hi = (lo_xyz, hi_xyz)`` is the half-open voxel range (point
    space, like :class:`pydvc.geometry.box.Box` but in ``(x, y, z)``) that holds
    real data. By default it is the whole brick. Samples whose stencil leaves it
    are marked ``inside = False``; their values are finite but meaningless.
    """
    if method not in _STENCIL:
        raise ValueError(f"unknown interpolation {method!r}")
    xp = xp_of(brick, positions)
    pos = xp.asarray(positions)
    dtype = xp.result_type(pos.dtype, xp.float32)
    data = xp.ascontiguousarray(xp.asarray(brick))
    nz, ny, nx = data.shape
    dims = (nx, ny, nz)
    origin = xp.asarray(origin_xyz, dtype=dtype)
    local = pos.astype(dtype, copy=False) - origin
    if valid_lo_hi is None:
        lo, hi = (0, 0, 0), dims
    else:
        lo = tuple(int(round(float(v) - float(o))) for v, o in zip(valid_lo_hi[0], origin_xyz))
        hi = tuple(int(round(float(v) - float(o))) for v, o in zip(valid_lo_hi[1], origin_xyz))
    before, after = _STENCIL[method]

    if method == "nearest":
        idx = xp.floor(local + 0.5).astype(xp.int64)
        frac = None
    else:
        base = xp.floor(local)
        frac = local - base
        idx = base.astype(xp.int64)

    inside = xp.ones(pos.shape[:-1], dtype=bool)
    safe = []
    for a in range(3):
        i = idx[..., a]
        inside &= (i - before >= max(lo[a], 0)) & (i + after <= min(hi[a], dims[a]) - 1)
        # clamp so every gather stays inside the array; masked samples are ignored downstream
        safe.append(xp.clip(i, before, max(dims[a] - 1 - after, before)))
    ix, iy, iz = safe
    flat = data.reshape(-1)

    if method == "nearest":
        values = flat[(iz * ny + iy) * nx + ix].astype(dtype)
        grad = xp.zeros(values.shape + (3,), dtype=dtype) if with_grad else None
        return Samples(values=values, grad=grad, inside=inside)

    weights = catmull_rom_weights if method == "tricubic" else _linear_weights
    n = before + after + 1
    taps = xp.arange(n, dtype=xp.int64)
    offsets = ((taps[:, None, None] * ny + taps[None, :, None]) * nx + taps[None, None, :]).reshape(-1)
    start = (((iz - before) * ny + (iy - before)) * nx + (ix - before)).reshape(-1)
    frac = frac.reshape(-1, 3)
    values = xp.empty(start.shape[0], dtype=dtype)
    grad = xp.empty((start.shape[0], 3), dtype=dtype) if with_grad else None
    chunk = _CHUNK_DEVICE if xp is not np else _CHUNK_HOST
    for s in range(0, start.shape[0], chunk):
        e = min(s + chunk, start.shape[0])
        # the whole (n, n, n) stencil of each sample, then separable contraction x -> y -> z
        blk = xp.take(flat, start[s:e, None] + offsets).astype(dtype).reshape(-1, n, n, n)
        (wx, dwx), (wy, dwy), (wz, dwz) = (weights(frac[s:e, a]) for a in range(3))
        tx = xp.einsum("pzyx,px->pzy", blk, wx)
        ty = xp.einsum("pzy,py->pz", tx, wy)
        values[s:e] = xp.einsum("pz,pz->p", ty, wz)
        if with_grad:
            dtx = xp.einsum("pzyx,px->pzy", blk, dwx)
            grad[s:e, 0] = xp.einsum("pz,pz->p", xp.einsum("pzy,py->pz", dtx, wy), wz)
            grad[s:e, 1] = xp.einsum("pz,pz->p", xp.einsum("pzy,py->pz", tx, dwy), wz)
            grad[s:e, 2] = xp.einsum("pz,pz->p", ty, dwz)
    values = values.reshape(pos.shape[:-1])
    if with_grad:
        grad = grad.reshape(pos.shape)
    return Samples(values=values, grad=grad, inside=inside)


@lru_cache(maxsize=1)
def _lekien_marsden_matrix() -> np.ndarray:
    """The 64 x 64 map from corner data to tricubic coefficients (Lekien & Marsden 2005).

    Built from the definition rather than transcribed: row ``r`` of the constraint
    matrix evaluates constraint ``r`` (f, fx, fy, fz, fxy, fxz, fyz, fxyz at the 8
    corners of the unit cell) on the monomial basis ``x^i y^j z^k``; the
    coefficients are its inverse applied to the corner data.
    """
    powers = [(i, j, k) for k in range(4) for j in range(4) for i in range(4)]
    derivs = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    corners = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]

    def dmono(p: int, d: int, x: int) -> float:
        if d > p:
            return 0.0
        coef = float(p) if d == 1 else 1.0
        e = p - d
        return coef * (1.0 if e == 0 else float(x) ** e)

    rows = []
    for dx, dy, dz in derivs:
        for cx, cy, cz in corners:
            rows.append([dmono(i, dx, cx) * dmono(j, dy, cy) * dmono(k, dz, cz) for i, j, k in powers])
    return np.linalg.inv(np.asarray(rows))


def lekien_marsden_reference(brick: Any, positions: Any) -> Any:
    """float64 numpy port of CCPi's ``kernels_derivs`` + ``tri_cub_Lek`` (values only), for the M1 equivalence test.

    Clean-room, from the published method: per-voxel derivative estimates by
    central differences (``fx = (f[i+1] - f[i-1]) / 2``, mixed terms as products
    of those operators), then the Lekien-Marsden tricubic on the enclosing cell.
    Test-only: slow, not vectorised over the stencil.
    """
    f = np.asarray(brick, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    shape = pos.shape[:-1]
    p = pos.reshape(-1, 3)
    base = np.floor(p).astype(np.int64)
    t = p - base
    A = _lekien_marsden_matrix()

    def at(ix: Any, iy: Any, iz: Any) -> np.ndarray:
        return f[iz, iy, ix]

    def central(ix: Any, iy: Any, iz: Any, dx: int, dy: int, dz: int) -> np.ndarray:
        total = np.zeros(len(p))
        for sx in ((-1, 1) if dx else (0,)):
            for sy in ((-1, 1) if dy else (0,)):
                for sz in ((-1, 1) if dz else (0,)):
                    sign = (sx if dx else 1) * (sy if dy else 1) * (sz if dz else 1)
                    total += sign * at(ix + sx, iy + sy, iz + sz)
        return total / (2.0 ** (dx + dy + dz))

    derivs = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    corners = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
    data = np.stack(
        [central(base[:, 0] + cx, base[:, 1] + cy, base[:, 2] + cz, *d) for d in derivs for cx, cy, cz in corners],
        axis=1,
    )
    coef = data @ A.T                                                     # (N, 64)
    powers = [(i, j, k) for k in range(4) for j in range(4) for i in range(4)]
    mono = np.stack([t[:, 0] ** i * t[:, 1] ** j * t[:, 2] ** k for i, j, k in powers], axis=1)
    return (coef * mono).sum(axis=1).reshape(shape)
