"""Batches inside a tile.

Batch size is limited by per-point solver scratch, not by the bricks. Scratch
per point is dominated by the reference samples: ``M x 4 B`` for FA-GN, plus
``M x 12 B`` of reference gradients for IC-GN. At M = 8 000 that is 32-128 KB,
so 16 k points need 0.5-2 GB. Batches are ordered along a Morton (Z-order)
curve so that consecutive thread blocks read overlapping brick regions and
hit in L2.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from pydvc.kernels.objective import n_sums

# bytes of solver scratch per point and template sample, by method:
# FA-GN keeps q (4 B); IC-GN adds the reference gradient (12 B).
_PER_SAMPLE = {"fagn": 4, "icgn": 16}


def batch_size(free_bytes: int, n_samples: int, dof: int, *, method: str = "fagn", fraction: float = 0.5) -> int:
    """Points per batch that fit ``fraction`` of ``free_bytes`` of device memory.

    Per point: the reference term ``q`` (``M`` floats, plus the reference
    gradient for IC-GN), the reference and final sample buffers, the
    parameters and state, and one sums row. The fused kernels never
    materialise per-sample Jacobians, so the batch is not limited by ``dof``
    beyond the sums row.
    """
    if n_samples < 1 or free_bytes <= 0:
        raise ValueError("need positive free memory and sample count")
    per_point = n_samples * (_PER_SAMPLE[method] + 4 + 1) + 4 * (2 * dof + n_sums(dof) + 16)
    return max(1, int(free_bytes * fraction) // per_point)


def _spread_bits(v: np.ndarray) -> np.ndarray:
    """Insert two zero bits between each of the low 21 bits (for 63-bit 3-D Morton codes)."""
    v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
    return v


def morton_order(xyz: Any, *, cell: float = 1.0) -> Any:
    """(N, 3) -> (N,) permutation along a Z-order curve (quantised to ``cell`` voxels).

    Host computation; the permutation is cheap next to a tile's solve.
    """
    pts = np.asarray(xyz.get() if hasattr(xyz, "get") else xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int64)
    q = np.floor((pts - pts.min(axis=0)) / cell).astype(np.int64)
    code = _spread_bits(q[:, 0]) | (_spread_bits(q[:, 1]) << np.uint64(1)) | (_spread_bits(q[:, 2]) << np.uint64(2))
    return np.argsort(code, kind="stable")


def iter_batches(order: Any, size: int) -> Iterator[Any]:
    """Consecutive slices of ``order`` of at most ``size`` indices."""
    if size < 1:
        raise ValueError("batch size must be >= 1")
    for lo in range(0, len(order), size):
        yield order[lo:lo + size]
