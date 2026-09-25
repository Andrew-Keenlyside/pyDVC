"""Batched Gauss-Newton: CCPi's per-point search, run for B points at once.

For each batch (following ``Search::process_point``):

1. Sample the reference brick at ``centre + template`` to get ``f`` (B, M)
   and its stats. This happens once per point; IC-GN also keeps ``grad f``.
2. Optional threshold test (``subvol_thresh``) -> ``THRESH_FAIL``.
3. ``params <- [seed, 0, ...]``. As in CCPi, only the translation is seeded;
   rotations and strains start at zero.
4. Optional translation grid search (``basin_radius > 0``), from
   :mod:`pydvc.solver.coarse`.
5. Up to ``max_iterations`` Gauss-Newton steps on the active set
   (:class:`pydvc.kernels.fused.FusedGNStep` + ``batched_cholesky_step``).
6. Exact final objective, and status.

Methods
    ``fagn``: forward-additive, the CCPi-parity default. CCPi forms the
    Jacobian by forward differences (``h = 1e-10``, ``ndof + 1`` objective
    evaluations per step) and solves ``J^T J dp = -J^T r`` by QR with no
    damping. pyDVC uses the analytic Jacobian ``grad(g)(x')^T dx'/dp`` (one
    value+gradient pass) and a batched Cholesky solve. It keeps the same
    undamped step and the same stopping rules.

    ``icgn``: inverse-compositional (M5). The Hessian comes from the
    reference gradient and is built once per point. Each iteration needs
    target values only, with no target gradient, which makes it roughly
    2-3x cheaper per iteration.

Differences from CCPi (also listed in docs/ARCHITECTURE.md)
    * Range test: pyDVC tests ``|u - seed|_inf > disp_max`` directly, plus
      brick validity. CCPi's test is implicit: samples leaving a box of margin
      ``disp_max + 2`` around the seeded subvolume.
    * Convergence: CCPi's LM path returns after ``maxit`` without flagging.
      pyDVC reports ``CONVG_FAIL`` unless ``search.report_convg_fail`` is False.
    * Arithmetic: float32 on device (CCPi uses float64). Positions are formed
      relative to the point centre, so float32 keeps about 1e-4 voxel
      resolution across 4096^3 volumes. The numpy reference is float64.
    * Linear solve: Cholesky of the Jacobi-scaled ``J^T J``. A point whose
      scaled system has a pivot below ``SINGULAR_PIVOT``, or whose target has
      no texture (``FEATURELESS``), is ``SINGULAR``.
    * Stencils: any reference or target sample whose interpolation stencil
      leaves the brick's valid region makes the point ``RANGE_FAIL`` (CCPi's
      implicit range test, made explicit).

Stopping (CCPi): after each step, a point stops if ``|obj_k - obj_{k-1}| <
obj_tol`` or ``|dt| < disp_tol`` (Euclidean norm of the translation step).
The reported objective is re-evaluated at the final parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pydvc._todo import todo
from pydvc.config import SearchSpec
from pydvc.geometry.templates import Template
from pydvc.geometry.warp import deformation_matrix, warp_linear_jacobians
from pydvc.io.volume import Brick
from pydvc.kernels.interpolate import sample
from pydvc.kernels.objective import normal_equations, objective
from pydvc.status import PointStatus

Backend = Literal["numpy", "cupy", "fused"]

SINGULAR_PIVOT = 1e-10          # smallest acceptable Cholesky pivot of the unit-diagonal system (float64)
FEATURELESS = 1e-12             # sum |grad g|^2 / sum g^2 below this: no texture to correlate (SINGULAR)
_SAMPLES_PER_CHUNK = 1 << 20    # numpy reference: points x template samples solved together


@dataclass
class BatchResult:
    params: Any        # (B, ndof) float32 on device, float64 from the numpy reference; CCPi order
    status: Any        # (B,) int8, PointStatus
    objmin: Any        # (B,) objective at the solution (NaN if never evaluated)
    n_iter: Any        # (B,) uint8
    seed: Any          # (B, 3) starting displacement

    @property
    def displacement(self) -> Any:
        return self.params[:, :3]


def solve_batch(
    ref_brick: Brick,
    def_brick: Brick,
    centres: Any,            # (B, 3) point-space (x, y, z)
    seeds: Any,              # (B, 3) starting displacement
    template: Template,
    search: SearchSpec,
    *,
    backend: Backend = "fused",
) -> BatchResult:
    """Correlate B points. ``numpy`` is the reference (M1), ``cupy`` the unfused GPU path (M2), ``fused`` production (M2)."""
    if backend != "numpy":
        raise todo("M2", f"solve_batch(backend={backend!r})")
    if search.method != "fagn":
        raise todo("M5", f"solve_batch(method={search.method!r})")
    if search.basin_radius > 0:
        raise todo("M2", "translation grid search (basin_radius > 0)")
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 3)
    B, ndof = centres.shape[0], search.dof
    out = BatchResult(
        params=np.zeros((B, ndof)),
        status=np.full(B, PointStatus.GOOD, dtype=np.int8),
        objmin=np.full(B, np.nan),
        n_iter=np.zeros(B, dtype=np.uint8),
        seed=seeds.copy(),
    )
    chunk = max(1, _SAMPLES_PER_CHUNK // template.n_samples)
    for lo in range(0, B, chunk):
        sl = slice(lo, min(lo + chunk, B))
        _solve_chunk(ref_brick, def_brick, centres[sl], seeds[sl], template, search, out, sl)
    return out


def _sample_points(brick: Brick, centres: np.ndarray, rel: np.ndarray, search: SearchSpec, *, grad: bool):
    """Sample ``brick`` at ``centres[:, None] + rel`` (positions formed relative to each centre)."""
    return sample(
        brick.data,
        brick.origin_xyz,
        centres[:, None, :] + rel,
        method=search.interpolation,
        with_grad=grad,
        valid_lo_hi=brick.valid_lo_hi_xyz,
    )


def _moved_offsets(params: np.ndarray, d: np.ndarray) -> np.ndarray:
    """``t + F d`` for each point: (b, M, 3)."""
    return params[:, None, :3] + np.einsum("bij,mj->bmi", deformation_matrix(params), d)


def _target_jacobian(grad: np.ndarray, params: np.ndarray, d: np.ndarray) -> np.ndarray:
    """``grad(g)^T dx'/dp``: (b, M, ndof), without forming (b, M, 3, ndof)."""
    ndof = params.shape[1]
    J = np.empty(grad.shape[:2] + (ndof,))
    J[..., :3] = grad
    if ndof > 3:
        A = warp_linear_jacobians(params)                           # (b, ndof-3, 3, 3)
        J[..., 3:] = np.einsum("bmi,bkij,mj->bmk", grad, A, d, optimize=True)
    return J


def cholesky_step(H: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``H dp = -b`` per point with Jacobi scaling; returns ``(dp, singular)``."""
    diag = np.einsum("bii->bi", H)
    singular = ~(diag > 0).all(axis=1)
    scale = 1.0 / np.sqrt(np.where(diag > 0, diag, 1.0))
    Hs = H * scale[:, :, None] * scale[:, None, :]
    L = np.zeros_like(Hs)
    try:
        L[~singular] = np.linalg.cholesky(Hs[~singular])
    except np.linalg.LinAlgError:
        for i in np.flatnonzero(~singular):
            try:
                L[i] = np.linalg.cholesky(Hs[i])
            except np.linalg.LinAlgError:
                singular[i] = True
    pivots = np.einsum("bii->bi", L) ** 2
    singular |= ~(pivots.min(axis=1) > SINGULAR_PIVOT)
    dp = np.zeros_like(b)
    ok = ~singular
    if ok.any():
        dp[ok] = -np.linalg.solve(Hs[ok], (b[ok] * scale[ok])[..., None])[..., 0] * scale[ok]
    return dp, singular


def _solve_chunk(
    ref_brick: Brick,
    def_brick: Brick,
    centres: np.ndarray,
    seeds: np.ndarray,
    template: Template,
    search: SearchSpec,
    out: BatchResult,
    sl: slice,
) -> None:
    B = centres.shape[0]
    d = template.offsets.astype(np.float64)
    params = np.zeros((B, search.dof))
    params[:, :3] = seeds
    status = np.full(B, PointStatus.GOOD, dtype=np.int8)
    objmin = np.full(B, np.nan)
    n_iter = np.zeros(B, dtype=np.uint8)

    ref = _sample_points(ref_brick, centres, np.broadcast_to(d, (B,) + d.shape), search, grad=False)
    f = ref.values
    status[~ref.inside.all(axis=1)] = PointStatus.RANGE_FAIL
    if search.threshold is not None:
        th = search.threshold
        frac = ((f >= th.gray_min) & (f <= th.gray_max)).mean(axis=1)
        status[(status == PointStatus.GOOD) & (frac < th.min_fraction)] = PointStatus.THRESH_FAIL

    active = status == PointStatus.GOOD
    prev_obj = np.full(B, np.inf)
    for _ in range(search.max_iterations):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        p = params[idx]
        tar = _sample_points(def_brick, centres[idx], _moved_offsets(p, d), search, grad=True)
        left = ~tar.inside.all(axis=1)
        J = _target_jacobian(tar.grad, p, d)
        H, b, obj = normal_equations(f[idx], tar.values, J, search.objective)
        objmin[idx] = obj
        dp, singular = cholesky_step(H, b)
        # Jacobi scaling makes round-off gradients of a flat image look well conditioned; test the texture itself
        grad_energy = (tar.grad**2).sum(axis=(1, 2))
        singular |= ~(grad_energy > FEATURELESS * (tar.values**2).sum(axis=1))
        failed = left | singular
        status[idx[left]] = PointStatus.RANGE_FAIL
        status[idx[singular & ~left]] = PointStatus.SINGULAR
        params[idx[~failed]] += dp[~failed]
        n_iter[idx[~failed]] += 1
        converged = (np.abs(obj - prev_obj[idx]) < search.obj_tol) | (
            np.linalg.norm(dp[:, :3], axis=1) < search.disp_tol
        )
        prev_obj[idx] = obj
        out_of_range = np.abs(params[idx, :3] - seeds[idx]).max(axis=1) > search.disp_max
        status[idx[out_of_range & ~failed]] = PointStatus.RANGE_FAIL
        active[idx[failed | converged | out_of_range]] = False

    if active.any():
        status[active] = PointStatus.CONVG_FAIL if search.report_convg_fail else PointStatus.GOOD

    final = np.flatnonzero((status == PointStatus.GOOD) | (status == PointStatus.CONVG_FAIL))
    if final.size:
        tar = _sample_points(def_brick, centres[final], _moved_offsets(params[final], d), search, grad=False)
        objmin[final] = objective(f[final], tar.values, search.objective)
        status[final[~tar.inside.all(axis=1)]] = PointStatus.RANGE_FAIL

    out.params[sl] = params
    out.status[sl] = status
    out.objmin[sl] = objmin
    out.n_iter[sl] = n_iter
