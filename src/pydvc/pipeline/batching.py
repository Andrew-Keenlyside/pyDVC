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

from pydvc._todo import todo


def batch_size(free_bytes: int, n_samples: int, dof: int, *, method: str = "fagn", fraction: float = 0.5) -> int:
    raise todo("M2", "batch_size")


def morton_order(xyz: Any) -> Any:
    """(N, 3) -> (N,) permutation along a Z-order curve."""
    raise todo("M2", "morton_order")


def iter_batches(order: Any, size: int) -> Iterator[Any]:
    raise todo("M2", "iter_batches")
