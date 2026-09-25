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
      resolution across 4096^3 volumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydvc._todo import todo
from pydvc.config import SearchSpec
from pydvc.geometry.templates import Template
from pydvc.io.volume import Brick

Backend = Literal["numpy", "cupy", "fused"]


@dataclass
class BatchResult:
    params: Any        # (B, ndof) float32, CCPi order
    status: Any        # (B,) int8, PointStatus
    objmin: Any        # (B,) float32
    n_iter: Any        # (B,) uint8
    seed: Any          # (B, 3) float32

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
    raise todo("M1", "solve_batch")
