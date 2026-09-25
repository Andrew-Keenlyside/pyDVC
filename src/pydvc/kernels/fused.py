"""Fused Gauss-Newton step for a batch of points, as CUDA kernels.

The unfused path (``warp`` -> ``interpolate.sample`` -> ``objective`` sums)
materialises (B, M, ndof) intermediates. At B = 16k points and M = 8000
samples that is gigabytes per iteration. These kernels keep all of it in
registers and shared memory. Source: ``kernels/cuda/fused_gn.cu``, templated on
``DOF``, ``OBJECTIVE``, ``INTERP`` and the brick type, compiled through
``cupy.RawModule`` on first use (one specialisation per run configuration,
cached by cupy).

Kernels, per Gauss-Newton iteration

``gn_sums``
    One thread block per active point, threads striding over the M template
    samples. Per sample: ``x' = c + t + F d`` (``F``, and the ``A_k`` with
    ``dx'/dp_k = A_k d``, computed once per point into shared memory);
    Catmull-Rom value and gradient from the native-dtype brick (converted in
    registers, read through the read-only cache); Jacobian row
    ``j = grad(g)^T dx'/dp``; accumulate the one-pass sums of
    :func:`pydvc.kernels.objective.sum_layout`: the upper triangle of
    ``J^T J`` (21 floats for 6-DOF, 78 for 12-DOF), ``sum j``, ``sum v j``,
    ``sum q j`` and seven scalars. Warp-shuffle, then block reduction.
``gn_solve``
    One thread per active point: exact normal equations from the sums (the
    ZSSD/NSSD/ZNSSD normalisation included, see
    :mod:`pydvc.kernels.objective`), Jacobi-scaled register Cholesky
    (``SINGULAR`` on a small pivot or a featureless target), the step, CCPi's
    stopping tests (``|d obj| < obj_tol`` or ``|dt| < disp_tol``) and the
    range test (``|u - seed|_inf > disp_max``).

One pass suffices even for ZNSSD. The sums are formed on the target shifted
by the reference mean, so the target mean and norm follow from them without
cancellation, and no second pass over the samples is needed (an earlier
design stored the M target values in shared memory for a second pass).

``sample_values`` (one thread per sample) serves the reference samples, the
final objective and the ``basin_radius`` grid search.

Active set
    Finished points leave the index list after each iteration, so late
    iterations launch only for stragglers. There is no per-point control flow
    inside a block.

Batches are ordered spatially (:mod:`pydvc.pipeline.batching`) so that
neighbouring blocks read overlapping footprints and hit in L2 (50 MB on H100).
Positions are formed from an integer voxel plus a float32 fraction, so
precision does not depend on the coordinate's magnitude.

Without a GPU, :class:`pydvc.kernels.cuda.emulate.EmulatedKernels` runs the
same source on the host (backend ``"emulated"``), which is how the kernels are
tested in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pydvc.config import SearchSpec
from pydvc.io.volume import Brick
from pydvc.kernels.cuda import SIGNATURES, SOURCE
from pydvc.kernels.objective import n_sums
from pydvc.solver.engines import FEATURELESS, BatchState, singular_pivot

OBJECTIVE_CODE = {"sad": "0", "ssd": "1", "zssd": "2", "nssd": "3", "znssd": "4"}
INTERP_CODE = {"nearest": "0", "trilinear": "1", "tricubic": "2"}
BRICK_TYPE = {np.dtype(np.uint8): "unsigned char", np.dtype(np.uint16): "unsigned short", np.dtype(np.float32): "float"}


class CudaKernels:
    """Launches the kernels of ``fused_gn.cu`` through ``cupy.RawModule``."""

    emulated = False
    block = 256

    def __init__(self) -> None:
        from pydvc.kernels.xp import get_xp

        self.xp = get_xp("cuda")
        self._functions: dict[str, Any] = {}

    def function(self, name: str, targs: tuple[str, ...]) -> Any:
        expr = f"pydvc::{name}<{', '.join(targs)}>"
        if expr not in self._functions:
            module = self.xp.RawModule(code=SOURCE.read_text(), options=("-std=c++17",), name_expressions=[expr])
            self._functions[expr] = module.get_function(expr)
        return self._functions[expr]

    def launch(self, name: str, targs: tuple[str, ...], grid: int, block: int, args: list[Any]) -> None:
        conv = []
        for (ctype, _), value in zip(SIGNATURES[name], args):
            if ctype == "int":
                conv.append(np.int32(value))
            elif ctype == "float":
                conv.append(np.float32(value))
            else:
                conv.append(value)
        self.function(name, targs)((int(grid),), (int(block),), tuple(conv))


@dataclass
class FusedBrick:
    data: Any                      # (z, y, x) native dtype, contiguous, on the kernels' device
    geom: Any                      # int32 [nx, ny, nz, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z], brick-local
    origin: tuple[float, float, float]
    ctype: str


def brick_geometry(brick: Brick) -> np.ndarray:
    nz, ny, nx = brick.data.shape
    lo, hi = brick.valid_lo_hi_xyz
    o = brick.box.lo[::-1]
    return np.array([nx, ny, nz, *(l - b for l, b in zip(lo, o)), *(h - b for h, b in zip(hi, o))], dtype=np.int32)


class FusedEngine:
    """The fused kernels behind the :mod:`pydvc.solver.engines` interface."""

    dtype = np.dtype(np.float32)
    # q and the sample buffers are (points x samples) float32: bound them per call
    SAMPLES_PER_CALL = 1 << 26

    def __init__(self, kernels: Any) -> None:
        self.kernels = kernels
        self.xp = kernels.xp
        self.block = kernels.block

    def points_per_call(self, n_samples: int, ndof: int) -> int:
        return max(1, self.SAMPLES_PER_CALL // max(n_samples, 1))

    def prepare(self, brick: Brick) -> FusedBrick:
        xp = self.xp
        data = np.asarray(brick.data) if not hasattr(type(brick.data), "__cuda_array_interface__") else brick.data
        dt = np.dtype(data.dtype)
        if dt not in BRICK_TYPE:
            data, dt = data.astype(np.float32), np.dtype(np.float32)
        data = xp.ascontiguousarray(xp.asarray(data))
        return FusedBrick(data, xp.asarray(brick_geometry(brick)), tuple(float(v) for v in brick.origin_xyz), BRICK_TYPE[dt])

    def _split(self, brick: FusedBrick, centres: Any) -> tuple[Any, Any]:
        xp = self.xp
        local = centres - xp.asarray(brick.origin, dtype=xp.float64)
        base = xp.floor(local)
        return xp.ascontiguousarray(base.astype(xp.int32)), xp.ascontiguousarray((local - base).astype(xp.float32))

    def sample(self, brick: FusedBrick, centres: Any, params: Any, offsets: Any, search: SearchSpec) -> tuple[Any, Any]:
        xp = self.xp
        B, dof = params.shape
        M = offsets.shape[0]
        values = xp.empty((B, M), dtype=xp.float32)
        inside = xp.empty((B, M), dtype=xp.uint8)
        if B == 0:
            return values, xp.zeros(0, dtype=bool)
        cint, cfrac = self._split(brick, centres)
        n = B * M
        self.kernels.launch(
            "sample_values", (str(dof), INTERP_CODE[search.interpolation], brick.ctype),
            -(-n // self.block), self.block,
            [brick.data, brick.geom, cint, cfrac, xp.ascontiguousarray(params, dtype=xp.float32),
             xp.ascontiguousarray(offsets, dtype=xp.float32), B, M, values, inside],
        )
        return values, inside.all(axis=1)

    def sums(
        self, brick: FusedBrick, centres: Any, params: Any, idx: Any, q: Any, shift: Any, offsets: Any, search: SearchSpec
    ) -> tuple[Any, Any]:
        xp = self.xp
        dof = params.shape[1]
        A = int(idx.shape[0])
        sums = xp.zeros((A, n_sums(dof)), dtype=xp.float32)
        outside = xp.zeros(A, dtype=xp.int32)
        cint, cfrac = self._split(brick, centres)
        self.kernels.launch(
            "gn_sums",
            (str(dof), OBJECTIVE_CODE[search.objective], INTERP_CODE[search.interpolation], brick.ctype),
            A, self.block,
            [brick.data, brick.geom, cint, cfrac, params, xp.ascontiguousarray(idx, dtype=xp.int32), A,
             xp.ascontiguousarray(q, dtype=xp.float32), xp.ascontiguousarray(shift, dtype=xp.float32),
             xp.ascontiguousarray(offsets, dtype=xp.float32), int(offsets.shape[0]), sums, outside],
        )
        return sums, outside > 0

    def update(self, st: BatchState, idx: Any, sums: Any, outside: Any, shift: Any, search: SearchSpec) -> Any:
        xp = self.xp
        A = int(idx.shape[0])
        done = xp.zeros(A, dtype=xp.int32)
        self.kernels.launch(
            "gn_solve", (str(search.dof), OBJECTIVE_CODE[search.objective]),
            -(-A // self.block), self.block,
            [sums, outside.astype(xp.int32), xp.ascontiguousarray(idx, dtype=xp.int32), A,
             xp.ascontiguousarray(shift, dtype=xp.float32), st.seeds, st.params, st.prev_obj, st.objmin, st.n_iter,
             st.status, done, search.obj_tol, search.disp_tol, search.disp_max, singular_pivot(np.float32), FEATURELESS],
        )
        return done > 0
