"""Fused Gauss-Newton step for a batch of points, as one CUDA launch.

The unfused path (``warp`` -> ``interpolate.sample`` -> ``objective.residuals``
-> einsum for ``J^T J``) materialises (B, M, 3, ndof) intermediates. At
B = 16k points and M = 8000 samples that is gigabytes per iteration. This
kernel keeps all of it in registers and shared memory.

Launch shape
    One thread block per active point, 256 threads striding over the M
    template samples. When M < 1024, pack several points per block with one
    warp per point. Template offsets live in constant memory (up to 64 KB, about
    5 400 samples as float32 xyz) or in shared memory above that. Brick reads go
    through the read-only data cache. Neighbouring points read overlapping
    footprints, so batches are ordered spatially (``pipeline.batching``) to
    keep the L2 (50 MB on H100) warm.

Per sample
    ``x' = c + t + F d``; Catmull-Rom value and gradient from the brick
    (converted from u8/u16 in registers); residual ``r``; Jacobian row
    ``j = s * grad(g)^T dx'/dp`` (``s`` = objective scale); accumulate the
    upper triangle of ``J^T J`` (21 floats for 6-DOF, 78 for 12-DOF), ``J^T r``
    (6 or 12) and the objective sums.

ZNSSD / NSSD
    These need the target mean and norm before residuals exist. Pass 1
    interpolates and stores the M target values in shared memory (8 000 x 4 B =
    32 KB, well under H100's 228 KB per SM) while accumulating sums. Pass 2
    forms residuals from shared memory and recomputes only the gradient.

Reduction and solve
    Warp-shuffle, then block reduction, gives one row per point:
    ``[JtJ_upper, Jtr, obj, n_inside]``. A second small kernel runs one thread
    per point: register Cholesky, step, convergence tests (CCPi: ``|d obj| <
    obj_tol`` or ``|d u| < disp_tol``), range test (``|u - seed| > disp_max``),
    and flags ``SINGULAR`` on a non-positive pivot.

Active set
    Converged or failed points leave the index list each iteration, so late
    iterations launch only for stragglers. There is no per-point control flow.

Source goes in ``kernels/cuda/fused_gn.cu``, templated on ``DOF``,
``OBJECTIVE``, ``INTERP`` and ``BRICK_T``. It is compiled through
``cupy.RawModule`` on first use, one specialisation per run configuration,
and cached by cupy.
"""

from __future__ import annotations

from typing import Any

from pydvc._todo import todo


class FusedGNStep:
    def __init__(self, *, dof: int, objective: str, interpolation: str, brick_dtype: str) -> None:
        raise todo("M2", "FusedGNStep (cupy.RawModule)")

    def __call__(
        self,
        brick: Any,              # (z, y, x) device array, native dtype
        origin_xyz: Any,         # (3,)
        valid_lo_hi: Any,        # (2, 3)
        centres: Any,            # (B, 3)
        params: Any,             # (B, ndof), updated in place
        active: Any,             # (A,) int32 indices into the batch
        ref_values: Any,         # (B, M) reference samples (computed once per point)
        ref_stats: Any,          # (B, 2) reference mean / norm
        out: Any,                # (B, ndof*(ndof+1)/2 + ndof + 2) normal equations + obj + n_inside
    ) -> None:
        raise todo("M2", "FusedGNStep.__call__")


def batched_cholesky_step(normal_eq: Any, params: Any, active: Any, state: Any) -> Any:
    """Solve, update, test convergence and range; returns the new active index list."""
    raise todo("M2", "batched_cholesky_step")
