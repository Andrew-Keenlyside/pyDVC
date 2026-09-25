"""The fused Gauss-Newton step on multi-core CPUs (numba, ``prange`` over points).

This is not a production path. It exists so the benchmarks can separate the
gain from *restructuring* (bricks, analytic Jacobian, batching) from the gain
from *the GPU* (docs/PERFORMANCE.md section 5, MVP question Q5). It is a line
for line port of ``kernels/cuda/fused_gn.cu``: the same template, warp,
Catmull-Rom stencil, one-pass sums and stopping rules, with points spread
across cores. The per-point solve is the shared
:func:`pydvc.solver.engines.gn_update`. Accumulation is in float64 (free on a
CPU); results are returned as float32 like the GPU paths.

Needs ``numba`` (``pip install 'pydvc[cpu-fast]'``). ``NUMBA_NUM_THREADS``
sets the core count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pydvc.config import SearchSpec
from pydvc.io.volume import Brick
from pydvc.kernels.objective import n_sums
from pydvc.solver.engines import BatchState, gn_update

OBJECTIVE_CODE = {"sad": 0, "ssd": 1, "zssd": 2, "nssd": 3, "znssd": 4}
INTERP_CODE = {"nearest": 0, "trilinear": 1, "tricubic": 2}

_kernels: dict[str, Any] = {}


def _compile() -> dict[str, Any]:
    if _kernels:
        return _kernels
    try:
        import numba
    except ImportError as exc:
        raise ImportError("backend 'cpu' needs numba: pip install 'pydvc[cpu-fast]'") from exc
    njit, prange = numba.njit, numba.prange

    @njit(cache=True, inline="always")
    def mat3(a, b):
        c = np.empty((3, 3))
        for r in range(3):
            for k in range(3):
                c[r, k] = a[r, 0] * b[0, k] + a[r, 1] * b[1, k] + a[r, 2] * b[2, k]
        return c

    @njit(cache=True)
    def warp_matrices(p, dof):
        F = np.eye(3)
        A = np.zeros((max(dof - 3, 1), 3, 3))
        if dof == 3:
            return F, A
        cf, sf, ct, st, cs, ss = np.cos(p[3]), np.sin(p[3]), np.cos(p[4]), np.sin(p[4]), np.cos(p[5]), np.sin(p[5])
        rz = np.array([[cf, sf, 0.0], [-sf, cf, 0.0], [0.0, 0.0, 1.0]])
        drz = np.array([[-sf, cf, 0.0], [-cf, -sf, 0.0], [0.0, 0.0, 0.0]])
        ry = np.array([[ct, 0.0, -st], [0.0, 1.0, 0.0], [st, 0.0, ct]])
        dry = np.array([[-st, 0.0, -ct], [0.0, 0.0, 0.0], [ct, 0.0, -st]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cs, ss], [0.0, -ss, cs]])
        drx = np.array([[0.0, 0.0, 0.0], [0.0, -ss, cs], [0.0, -cs, -ss]])
        R = mat3(rx, mat3(ry, rz))
        dR0 = mat3(rx, mat3(ry, drz))
        dR1 = mat3(rx, mat3(dry, rz))
        dR2 = mat3(drx, mat3(ry, rz))
        if dof == 6:
            A[0], A[1], A[2] = dR0, dR1, dR2
            return R, A
        IE = np.array([[1.0 + p[6], p[9], p[11]], [p[9], 1.0 + p[7], p[10]], [p[11], p[10], 1.0 + p[8]]])
        F = mat3(IE, R)
        A[0], A[1], A[2] = mat3(IE, dR0), mat3(IE, dR1), mat3(IE, dR2)
        ra = (0, 1, 2, 0, 1, 0)
        rb = (0, 1, 2, 1, 2, 2)
        for k in range(6):
            a, b = ra[k], rb[k]
            for m in range(3):
                A[3 + k, a, m] += R[b, m]
                if a != b:
                    A[3 + k, b, m] += R[a, m]
        return F, A

    @njit(cache=True, inline="always")
    def cr(t, w, dw):
        t2 = t * t
        t3 = t2 * t
        w[0] = 0.5 * (-t3 + 2.0 * t2 - t)
        w[1] = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
        w[2] = 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
        w[3] = 0.5 * (t3 - t2)
        dw[0] = 0.5 * (-3.0 * t2 + 4.0 * t - 1.0)
        dw[1] = 0.5 * (9.0 * t2 - 10.0 * t)
        dw[2] = 0.5 * (-9.0 * t2 + 8.0 * t + 1.0)
        dw[3] = 0.5 * (3.0 * t2 - 2.0 * t)

    @njit(cache=True)
    def interpolate(flat, geom, i, f, interp, grad, out):
        """out = [val, gx, gy, gz]; returns False if the stencil leaves the valid region."""
        before = 1 if interp == 2 else 0
        after = 2 if interp == 2 else (1 if interp == 1 else 0)
        c = np.empty(3, np.int64)
        for a in range(3):
            c[a] = i[a] + 1 if (interp == 0 and f[a] >= 0.5) else i[a]
            lo = max(geom[3 + a], 0)
            hi = min(geom[6 + a], geom[a])
            if c[a] - before < lo or c[a] + after > hi - 1:
                return False
        nx, ny = geom[0], geom[1]
        if interp == 0:
            out[0] = flat[(c[2] * ny + c[1]) * nx + c[0]]
            out[1] = out[2] = out[3] = 0.0
            return True
        if interp == 1:
            v = gx = gy = gz = 0.0
            for zz in range(2):
                wz = f[2] if zz else 1.0 - f[2]
                dz = 1.0 if zz else -1.0
                for yy in range(2):
                    wy = f[1] if yy else 1.0 - f[1]
                    dy = 1.0 if yy else -1.0
                    row = ((c[2] + zz) * ny + (c[1] + yy)) * nx + c[0]
                    t0 = float(flat[row])
                    t1 = float(flat[row + 1])
                    sx = (1.0 - f[0]) * t0 + f[0] * t1
                    v += wz * wy * sx
                    gx += wz * wy * (t1 - t0)
                    gy += wz * dy * sx
                    gz += dz * wy * sx
            out[0], out[1], out[2], out[3] = v, gx, gy, gz
            return True
        wx = np.empty(4)
        dwx = np.empty(4)
        wy = np.empty(4)
        dwy = np.empty(4)
        wz = np.empty(4)
        dwz = np.empty(4)
        cr(f[0], wx, dwx)
        cr(f[1], wy, dwy)
        cr(f[2], wz, dwz)
        base = ((c[2] - 1) * ny + (c[1] - 1)) * nx + (c[0] - 1)
        v = gx = gy = gz = 0.0
        for zz in range(4):
            vz = vzdx = vzdy = 0.0
            for yy in range(4):
                row = base + (zz * ny + yy) * nx
                t0 = float(flat[row])
                t1 = float(flat[row + 1])
                t2 = float(flat[row + 2])
                t3 = float(flat[row + 3])
                sx = wx[0] * t0 + wx[1] * t1 + wx[2] * t2 + wx[3] * t3
                vz += wy[yy] * sx
                if grad:
                    vzdx += wy[yy] * (dwx[0] * t0 + dwx[1] * t1 + dwx[2] * t2 + dwx[3] * t3)
                    vzdy += dwy[yy] * sx
            v += wz[zz] * vz
            gx += wz[zz] * vzdx
            gy += wz[zz] * vzdy
            gz += dwz[zz] * vz
        out[0], out[1], out[2], out[3] = v, gx, gy, gz
        return True

    @njit(cache=True, inline="always")
    def locate(cint, cfrac, p, F, d, i, f):
        for a in range(3):
            s = cfrac[a] + p[a] + F[a, 0] * d[0] + F[a, 1] * d[1] + F[a, 2] * d[2]
            fl = np.floor(s)
            i[a] = cint[a] + np.int64(fl)
            f[a] = s - fl

    @njit(parallel=True, cache=True)
    def sample_values(flat, geom, cint, cfrac, params, offsets, interp, values, inside):
        B, M = values.shape
        dof = params.shape[1]
        for b in prange(B):
            p = params[b].astype(np.float64)
            F, _ = warp_matrices(p, dof)
            i = np.empty(3, np.int64)
            f = np.empty(3)
            out = np.empty(4)
            for m in range(M):
                locate(cint[b], cfrac[b], p, F, offsets[m], i, f)
                ok = interpolate(flat, geom, i, f, interp, False, out)
                values[b, m] = out[0] if ok else 0.0
                inside[b, m] = ok

    @njit(parallel=True, cache=True)
    def gn_sums(flat, geom, cint, cfrac, params, active, q, shift, offsets, interp, sums, outside):
        A = active.shape[0]
        M = offsets.shape[0]
        dof = params.shape[1]
        njj = dof * (dof + 1) // 2
        S = njj + 3 * dof
        for a in prange(A):
            pt = active[a]
            p = params[pt].astype(np.float64)
            F, Am = warp_matrices(p, dof)
            acc = np.zeros(S + 7)
            J = np.empty(dof)
            i = np.empty(3, np.int64)
            f = np.empty(3)
            out = np.empty(4)
            n_out = 0
            s0 = shift[pt]
            for m in range(M):
                d = offsets[m]
                locate(cint[pt], cfrac[pt], p, F, d, i, f)
                if not interpolate(flat, geom, i, f, interp, True, out):
                    n_out += 1
                    continue
                J[0], J[1], J[2] = out[1], out[2], out[3]
                for k in range(dof - 3):
                    J[3 + k] = (
                        out[1] * (Am[k, 0, 0] * d[0] + Am[k, 0, 1] * d[1] + Am[k, 0, 2] * d[2])
                        + out[2] * (Am[k, 1, 0] * d[0] + Am[k, 1, 1] * d[1] + Am[k, 1, 2] * d[2])
                        + out[3] * (Am[k, 2, 0] * d[0] + Am[k, 2, 1] * d[1] + Am[k, 2, 2] * d[2])
                    )
                v = out[0] - s0
                qq = q[pt, m]
                t = 0
                for r in range(dof):
                    for c in range(r, dof):
                        acc[t] += J[r] * J[c]
                        t += 1
                for r in range(dof):
                    acc[njj + r] += J[r]
                    acc[njj + dof + r] += v * J[r]
                    acc[njj + 2 * dof + r] += qq * J[r]
                acc[S] += v
                acc[S + 1] += v * v
                acc[S + 2] += v * qq
                acc[S + 3] += qq
                acc[S + 4] += qq * qq
                acc[S + 5] += 1.0
                acc[S + 6] += abs(v - qq)
            for k in range(S + 7):
                sums[a, k] = acc[k]
            outside[a] = n_out

    _kernels.update(sample_values=sample_values, gn_sums=gn_sums)
    return _kernels


@dataclass
class CpuBrick:
    flat: np.ndarray               # native dtype, raveled (z, y, x)
    geom: np.ndarray               # int64 [nx, ny, nz, lo_x, lo_y, lo_z, hi_x, hi_y, hi_z], brick-local
    origin: tuple[float, float, float]


class NumbaEngine:
    """The fused step on CPU cores, behind the :mod:`pydvc.solver.engines` interface."""

    xp = np
    dtype = np.dtype(np.float32)
    SAMPLES_PER_CALL = 1 << 26

    def __init__(self) -> None:
        self.k = _compile()

    def points_per_call(self, n_samples: int, ndof: int) -> int:
        return max(1, self.SAMPLES_PER_CALL // max(n_samples, 1))

    def prepare(self, brick: Brick) -> CpuBrick:
        from pydvc.kernels.fused import brick_geometry

        data = np.ascontiguousarray(np.asarray(brick.data))
        return CpuBrick(data.reshape(-1), brick_geometry(brick).astype(np.int64), tuple(float(v) for v in brick.origin_xyz))

    def _split(self, brick: CpuBrick, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local = np.asarray(centres, dtype=np.float64) - np.asarray(brick.origin)
        base = np.floor(local)
        return base.astype(np.int64), local - base

    def sample(self, brick: CpuBrick, centres: Any, params: Any, offsets: Any, search: SearchSpec) -> tuple[Any, Any]:
        B, M = params.shape[0], offsets.shape[0]
        values = np.empty((B, M), dtype=np.float32)
        inside = np.empty((B, M), dtype=np.bool_)
        if B:
            cint, cfrac = self._split(brick, centres)
            self.k["sample_values"](
                brick.flat, brick.geom, cint, cfrac, np.ascontiguousarray(params, dtype=np.float32),
                np.ascontiguousarray(offsets, dtype=np.float64), INTERP_CODE[search.interpolation], values, inside,
            )
        return values, inside.all(axis=1)

    def sums(
        self, brick: CpuBrick, centres: Any, params: Any, idx: Any, q: Any, shift: Any, offsets: Any, search: SearchSpec
    ) -> tuple[Any, Any]:
        A = len(idx)
        dof = params.shape[1]
        sums = np.zeros((A, n_sums(dof)), dtype=np.float64)
        outside = np.zeros(A, dtype=np.int64)
        cint, cfrac = self._split(brick, centres)
        self.k["gn_sums"](
            brick.flat, brick.geom, cint, cfrac, params, np.asarray(idx, dtype=np.int64), q,
            np.asarray(shift, dtype=np.float64), np.ascontiguousarray(offsets, dtype=np.float64),
            INTERP_CODE[search.interpolation], sums, outside,
        )
        return sums, outside > 0

    def update(self, st: BatchState, idx: Any, sums: Any, outside: Any, shift: Any, search: SearchSpec) -> Any:
        return gn_update(st, idx, sums.astype(np.float32), outside, shift, search)
