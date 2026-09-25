"""The fused Gauss-Newton step on multi-core CPUs (numba, ``prange`` over points).

This is not a production path. It exists so the benchmarks can separate the
gain from *restructuring* (bricks, analytic Jacobian, batching) from the gain
from *the GPU* (docs/PERFORMANCE.md section 5, MVP question Q5). It uses the
same algorithm, template and stopping rules as :mod:`pydvc.kernels.fused`,
with points spread across cores and samples vectorised within a point.
"""

from __future__ import annotations

from typing import Any

from pydvc._todo import todo


def fused_gn_step_cpu(
    brick: Any,
    origin_xyz: Any,
    valid_lo_hi: Any,
    centres: Any,
    params: Any,
    active: Any,
    ref_values: Any,
    ref_stats: Any,
    template_offsets: Any,
    out: Any,
    *,
    dof: int,
    objective: str,
    interpolation: str,
) -> None:
    raise todo("M2", "fused_gn_step_cpu (numba)")
