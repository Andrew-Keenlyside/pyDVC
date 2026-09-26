"""Batched Gauss-Newton: CCPi's per-point search, run for B points at once.

For each batch (following ``Search::process_point``):

1. Sample the reference brick at ``centre + template`` to get ``f`` (B, M)
   and its stats. This happens once per point; IC-GN also keeps ``grad f``.
2. Optional threshold test (``subvol_thresh``) -> ``THRESH_FAIL``.
3. ``params <- [seed, 0, ...]``. As in CCPi, only the translation is seeded;
   rotations and strains start at zero.
4. Optional translation grid search (``basin_radius > 0``), from
   :mod:`pydvc.solver.coarse`.
5. Up to ``max_iterations`` Gauss-Newton steps on the active set: the
   engine reduces each point's samples to the one-pass sums
   (:func:`pydvc.kernels.objective.sum_layout`), then solves and tests
   convergence (:func:`pydvc.solver.engines.gn_update`, or the ``gn_solve``
   kernel on the fused engine).
6. Exact final objective, and status.

The loop is written once; :mod:`pydvc.solver.engines` supplies the numerics
(numpy reference, cupy, fused CUDA, the CUDA source emulated on the host, and
numba on CPU cores).

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
      scaled system has a pivot below the precision's threshold
      (:func:`pydvc.solver.engines.singular_pivot`), or whose target has no
      texture (``FEATURELESS``), is ``SINGULAR``.
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
from pydvc.io.volume import Brick
from pydvc.kernels.objective import objective, reference_terms
from pydvc.solver.engines import BatchState, make_engine
from pydvc.status import PointStatus

Backend = Literal["numpy", "numpy32", "cupy", "fused", "emulated", "cpu"]


@dataclass
class BatchResult:
    params: Any        # (B, ndof) float32 (float64 from the numpy reference); CCPi order
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
    engine: Any = None,
) -> BatchResult:
    """Correlate B points on one engine (:mod:`pydvc.solver.engines`).

    ``numpy`` is the float64 reference (M1). ``cupy`` is the unfused GPU path,
    ``fused`` the production CUDA kernels and ``cpu`` the same fused step on
    CPU cores (M2). Results come back on the engine's device. Pass ``engine``
    to reuse one (and its compiled kernels) across calls.
    """
    if search.method != "fagn":
        raise todo("M5", f"solve_batch(method={search.method!r})")
    eng = engine if engine is not None else make_engine(backend)
    xp = eng.xp
    centres = xp.asarray(centres, dtype=xp.float64).reshape(-1, 3)
    seeds = xp.asarray(seeds, dtype=eng.dtype).reshape(-1, 3)
    B, ndof = centres.shape[0], search.dof
    out = BatchResult(
        params=xp.zeros((B, ndof), dtype=eng.dtype),
        status=xp.full(B, int(PointStatus.GOOD), dtype=xp.int8),
        objmin=xp.full(B, np.nan, dtype=eng.dtype),
        n_iter=xp.zeros(B, dtype=xp.uint8),
        seed=seeds.copy(),
    )
    ref = eng.prepare(ref_brick)
    deformed = eng.prepare(def_brick)
    offsets = xp.asarray(template.offsets, dtype=eng.dtype)
    chunk = eng.points_per_call(template.n_samples, ndof)
    for lo in range(0, B, chunk):
        sl = slice(lo, min(lo + chunk, B))
        st = _solve_chunk(eng, ref, deformed, centres[sl], seeds[sl], offsets, search)
        out.params[sl], out.status[sl], out.objmin[sl], out.n_iter[sl] = st.params, st.status, st.objmin, st.n_iter
    return out


def _solve_chunk(eng: Any, ref: Any, deformed: Any, centres: Any, seeds: Any, offsets: Any, search: SearchSpec) -> BatchState:
    xp = eng.xp
    B = centres.shape[0]
    st = BatchState.start(xp, eng.dtype, seeds, search.dof)
    f, ref_inside = eng.sample(ref, centres, xp.zeros((B, 3), dtype=eng.dtype), offsets, search)
    st.status[~ref_inside] = int(PointStatus.RANGE_FAIL)
    if search.threshold is not None:
        th = search.threshold
        frac = ((f >= th.gray_min) & (f <= th.gray_max)).mean(axis=1)
        st.status[(st.status == PointStatus.GOOD) & (frac < th.min_fraction)] = int(PointStatus.THRESH_FAIL)
    q, shift = reference_terms(f, search.objective)
    if search.basin_radius > 0:
        from pydvc.solver.coarse import translation_grid_search

        good = xp.flatnonzero(st.status == PointStatus.GOOD)
        if good.size:
            translation_grid_search(
                lambda c, p: eng.sample(deformed, c, p, offsets, search), f[good], centres[good], st.params, good, search
            )

    active = st.status == PointStatus.GOOD
    for _ in range(search.max_iterations):
        idx = xp.flatnonzero(active)
        if idx.size == 0:
            break
        sums, outside = eng.sums(deformed, centres, st.params, idx, q, shift, offsets, search)
        done = eng.update(st, idx, sums, outside, shift, search)
        active[idx[done]] = False

    if bool(active.any()):
        st.status[active] = int(PointStatus.CONVG_FAIL if search.report_convg_fail else PointStatus.GOOD)
    final = xp.flatnonzero((st.status == PointStatus.GOOD) | (st.status == PointStatus.CONVG_FAIL))
    if final.size:
        g, inside = eng.sample(deformed, centres[final], st.params[final], offsets, search)
        st.objmin[final] = objective(f[final], g, search.objective).astype(eng.dtype)
        st.status[final[~inside]] = int(PointStatus.RANGE_FAIL)
    return st
