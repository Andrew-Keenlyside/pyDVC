"""Compute engines behind :func:`pydvc.solver.gauss_newton.solve_batch`.

The Gauss-Newton driver is written once. An engine supplies four operations:

``prepare(brick)``
    Put a brick where the engine computes (device memory for GPU engines).
``sample(brick, centres, params, offsets, search)``
    Interpolated values (B, M) at the warped template, and whether every
    stencil stayed in the valid region (B,).
``sums(brick, centres, params, idx, q, shift, offsets, search)``
    For the points ``idx``: the one-pass normal-equation sums (A, n_sums)
    (:func:`pydvc.kernels.objective.sum_layout`) and whether any sample left
    the valid region (A,).
``update(state, idx, sums, outside, shift, search)``
    Solve, step, and apply the stopping, range and singularity tests; returns
    which of ``idx`` are finished. :func:`gn_update` is the reference; the
    fused engine does the same per point in one kernel.

Engines

=============  =========================================================  =========
backend        engine                                                     precision
=============  =========================================================  =========
``numpy``      :class:`XpEngine` on numpy: the M1 reference                float64
``numpy32``    :class:`XpEngine` on numpy in float32: the cupy path's      float32
               arithmetic on the host, for tests without a GPU
``cupy``       :class:`XpEngine` on cupy: the unfused GPU path             float32
``fused``      :class:`pydvc.kernels.fused.FusedEngine` on CUDA            float32
``emulated``   the same fused kernels, run on the host by                  float32
               :mod:`pydvc.kernels.cuda.emulate` (tests only; slow)
``cpu``        :class:`pydvc.kernels.cpu_fused.NumbaEngine`: the fused     float32
               step on all CPU cores (restructured-CPU baseline, Q5)
=============  =========================================================  =========
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pydvc.config import SearchSpec
from pydvc.geometry.warp import deformation_matrix, warp_linear_jacobians
from pydvc.io.volume import Brick
from pydvc.kernels.interpolate import sample as interp_sample
from pydvc.kernels.objective import accumulate_sums, normal_equations_from_sums
from pydvc.status import PointStatus

FEATURELESS = 1e-12             # sum |grad g|^2 / sum g^2 below this: no texture to correlate (SINGULAR)


def singular_pivot(dtype: Any) -> float:
    """Smallest acceptable Cholesky pivot of the unit-diagonal (Jacobi-scaled) system."""
    return 1e-10 if np.dtype(dtype) == np.float64 else 1e-6


@dataclass
class BatchState:
    params: Any        # (B, ndof)
    seeds: Any         # (B, 3)
    status: Any        # (B,) int8
    objmin: Any        # (B,)
    prev_obj: Any      # (B,)
    n_iter: Any        # (B,) uint8

    @classmethod
    def start(cls, xp: Any, dtype: Any, seeds: Any, ndof: int) -> BatchState:
        B = seeds.shape[0]
        params = xp.zeros((B, ndof), dtype=dtype)
        params[:, :3] = seeds
        return cls(
            params=params,
            seeds=xp.asarray(seeds, dtype=dtype).copy(),
            status=xp.full(B, int(PointStatus.GOOD), dtype=xp.int8),
            objmin=xp.full(B, np.nan, dtype=dtype),
            prev_obj=xp.full(B, np.inf, dtype=dtype),
            n_iter=xp.zeros(B, dtype=xp.uint8),
        )


def cholesky_step(H: Any, b: Any, pivot_tol: float) -> tuple[Any, Any]:
    """Solve ``H dp = -b`` per point: Jacobi scaling, then an unrolled batched Cholesky.

    Returns ``(dp, singular)``. Written with array operations over the batch
    and loops over the (at most 12) unknowns, so numpy and cupy run the same
    arithmetic as the per-point solve in the CUDA kernel.
    """
    from pydvc.kernels.xp import xp_of

    xp = xp_of(H, b)
    n = H.shape[-1]
    diag = xp.stack([H[:, i, i] for i in range(n)], axis=1)
    singular = ~(diag > 0).all(axis=1)
    sc = 1.0 / xp.sqrt(xp.where(diag > 0, diag, 1.0))
    A = H * sc[:, :, None] * sc[:, None, :]
    L = xp.zeros_like(A)
    for j in range(n):
        s = A[:, j, j] - (L[:, j, :j] ** 2).sum(axis=1)
        singular |= ~(s > pivot_tol)
        d = xp.sqrt(xp.where(s > pivot_tol, s, 1.0))
        L[:, j, j] = d
        for i in range(j + 1, n):
            L[:, i, j] = (A[:, i, j] - (L[:, i, :j] * L[:, j, :j]).sum(axis=1)) / d
    y = xp.zeros_like(b)
    for i in range(n):
        y[:, i] = (-b[:, i] * sc[:, i] - (L[:, i, :i] * y[:, :i]).sum(axis=1)) / L[:, i, i]
    x = xp.zeros_like(b)
    for i in range(n - 1, -1, -1):
        x[:, i] = (y[:, i] - (L[:, i + 1:, i] * x[:, i + 1:]).sum(axis=1)) / L[:, i, i]
    dp = xp.where(singular[:, None], 0.0, x * sc)
    return dp, singular


def gn_update(st: BatchState, idx: Any, sums: Any, outside: Any, shift: Any, search: SearchSpec) -> Any:
    """One Gauss-Newton update of the points ``idx`` from their sums; returns which are finished."""
    from pydvc.kernels.xp import xp_of

    xp = xp_of(sums)
    H, b, obj, grad_energy, sum_g2 = normal_equations_from_sums(sums, shift[idx], search.objective, search.dof)
    left = outside
    st.objmin[idx[~left]] = obj[~left]
    dp, singular = cholesky_step(H, b, singular_pivot(sums.dtype))
    singular |= ~(grad_energy > FEATURELESS * sum_g2)
    failed = left | singular
    st.status[idx[left]] = int(PointStatus.RANGE_FAIL)
    st.status[idx[singular & ~left]] = int(PointStatus.SINGULAR)
    ok = idx[~failed]
    st.params[ok] += dp[~failed]
    st.n_iter[ok] += 1
    converged = (xp.abs(obj - st.prev_obj[idx]) < search.obj_tol) | (
        xp.sqrt((dp[:, :3] ** 2).sum(axis=1)) < search.disp_tol
    )
    st.prev_obj[idx] = obj
    out_of_range = (xp.abs(st.params[idx, :3] - st.seeds[idx]).max(axis=1) > search.disp_max) & ~failed
    st.status[idx[out_of_range]] = int(PointStatus.RANGE_FAIL)
    return failed | converged | out_of_range


@dataclass
class XpBrick:
    data: Any
    origin_xyz: tuple[float, float, float]
    valid_lo_hi: tuple[tuple[int, int, int], tuple[int, int, int]]


class XpEngine:
    """Unfused array path: numpy (reference) or cupy (M2's first GPU path)."""

    SAMPLES_PER_CALL = 1 << 20

    def __init__(self, xp: Any, dtype: Any) -> None:
        self.xp = xp
        self.dtype = np.dtype(dtype)

    def points_per_call(self, n_samples: int, ndof: int) -> int:
        return max(1, self.SAMPLES_PER_CALL // max(n_samples, 1))

    def prepare(self, brick: Brick) -> XpBrick:
        return XpBrick(self.xp.asarray(brick.data), brick.origin_xyz, brick.valid_lo_hi_xyz)

    def _moved(self, params: Any, offsets: Any) -> Any:
        F = deformation_matrix(params)
        return params[:, None, :3] + self.xp.einsum("bij,mj->bmi", F, offsets)

    def sample(self, brick: XpBrick, centres: Any, params: Any, offsets: Any, search: SearchSpec) -> tuple[Any, Any]:
        s = interp_sample(
            brick.data, brick.origin_xyz, self._moved(params, offsets), method=search.interpolation,
            valid_lo_hi=brick.valid_lo_hi, centres=centres,
        )
        return s.values.astype(self.dtype, copy=False), s.inside.all(axis=1)

    def sums(
        self, brick: XpBrick, centres: Any, params: Any, idx: Any, q: Any, shift: Any, offsets: Any, search: SearchSpec
    ) -> tuple[Any, Any]:
        xp = self.xp
        p = params[idx]
        s = interp_sample(
            brick.data, brick.origin_xyz, self._moved(p, offsets), method=search.interpolation, with_grad=True,
            valid_lo_hi=brick.valid_lo_hi, centres=centres[idx],
        )
        grad = s.grad.astype(self.dtype, copy=False)
        J = xp.empty(grad.shape[:2] + (p.shape[1],), dtype=self.dtype)
        J[..., :3] = grad
        if p.shape[1] > 3:
            J[..., 3:] = xp.einsum("bmi,bkij,mj->bmk", grad, warp_linear_jacobians(p), offsets)
        v = s.values.astype(self.dtype, copy=False) - shift[idx, None]
        return accumulate_sums(q[idx], v, J, search.objective), ~s.inside.all(axis=1)

    def update(self, st: BatchState, idx: Any, sums: Any, outside: Any, shift: Any, search: SearchSpec) -> Any:
        return gn_update(st, idx, sums, outside, shift, search)


def make_engine(backend: str) -> Any:
    if backend == "numpy":
        return XpEngine(np, np.float64)
    if backend == "numpy32":
        return XpEngine(np, np.float32)
    if backend == "cupy":
        from pydvc.kernels.xp import get_xp

        return XpEngine(get_xp("cuda"), np.float32)
    if backend == "fused":
        from pydvc.kernels.fused import CudaKernels, FusedEngine

        return FusedEngine(CudaKernels())
    if backend == "emulated":
        from pydvc.kernels.cuda.emulate import EmulatedKernels
        from pydvc.kernels.fused import FusedEngine

        return FusedEngine(EmulatedKernels())
    if backend == "cpu":
        from pydvc.kernels.cpu_fused import NumbaEngine

        return NumbaEngine()
    raise ValueError(f"unknown backend {backend!r}")


def gpu_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def default_backend() -> str:
    """``fused`` with a GPU, else ``cpu`` (numba) if installed, else the numpy reference."""
    if gpu_available():
        return "fused"
    try:
        import numba  # noqa: F401

        return "cpu"
    except ImportError:
        return "numpy"


def on_device(backend: str) -> bool:
    return backend in ("cupy", "fused")
