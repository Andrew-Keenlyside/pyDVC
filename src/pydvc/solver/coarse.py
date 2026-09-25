"""Global (coarse) searches that produce a starting translation.

``translation_grid_search``
    CCPi's ``basin_radius`` coarse search, batched. With ``n = floor(disp_max /
    basin_radius)`` and step ``disp_max / n``, it evaluates the objective at
    ``(2n + 1)^3`` translations around the seed and keeps the best. The batch
    is B points x G translations with value-only interpolation, chunked over G
    to fit memory. It is expensive: ``disp_max = 38, basin_radius = 4`` gives
    6 859 evaluations per point. That is cheap per point on the GPU, but it
    dominates the solve, so prefer good seeds.

``fft_seed`` (M5)
    Path-independent start, the FFT cross-correlation + Gauss-Newton scheme
    common in the DIC/DVC literature. It runs a batched zero-normalised cross
    correlation by cuFFT over a ``(S + 2 * disp_max)^3`` window per point and
    takes the integer peak with a parabolic sub-voxel fit. Every point is
    independent, so no serial seeding order is needed at all. The cost is about
    three FFTs of the window per point, so it suits sparse coarse grids, or
    pyramid levels where ``disp_max`` shrinks.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from pydvc._todo import todo
from pydvc.config import SearchSpec
from pydvc.geometry.templates import Template
from pydvc.io.volume import Brick
from pydvc.kernels.objective import objective
from pydvc.kernels.xp import xp_of

# samples evaluated per call: bounds the (points x translations x samples) values array
_SAMPLES_PER_CALL = 1 << 22


def grid_offsets(search: SearchSpec, xp: Any, dtype: Any) -> Any:
    """The ``(2n + 1)^3`` translations, ``n = floor(disp_max / basin_radius)``, step ``disp_max / n``."""
    n = max(1, int(math.floor(search.disp_max / search.basin_radius)))
    step = search.disp_max / n
    axis = xp.arange(-n, n + 1, dtype=dtype) * step
    gz, gy, gx = xp.meshgrid(axis, axis, axis, indexing="ij")
    return xp.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def translation_grid_search(
    sample: Callable[[Any, Any], tuple[Any, Any]],
    ref_values: Any,         # (b, M) reference samples of the points searched
    centres: Any,            # (b, 3)
    params: Any,             # (B, ndof); translations of rows ``idx`` are replaced by the best grid point
    idx: Any,                # (b,) rows of ``params`` being searched
    search: SearchSpec,
) -> None:
    """Keep, per point, the grid translation (around its seed) with the best objective.

    ``sample(centres, translations)`` returns values (n, M) and inside (n,) for
    3-DOF parameters; every engine provides it, so this runs unchanged on
    numpy, cupy, the fused kernels and numba. Translations whose samples leave
    the brick are never chosen.
    """
    xp = xp_of(params)
    b, M = ref_values.shape
    offsets = grid_offsets(search, xp, params.dtype)
    G = offsets.shape[0]
    seed = params[idx, :3].copy()
    best = xp.full(b, xp.inf, dtype=params.dtype)
    best_t = seed.copy()
    per_call = max(1, _SAMPLES_PER_CALL // max(M, 1))
    pts = max(1, min(b, per_call // G)) if per_call >= G else 1
    gstep = G if per_call >= G else per_call
    for p0 in range(0, b, pts):
        p1 = min(p0 + pts, b)
        for g0 in range(0, G, gstep):
            g1 = min(g0 + gstep, G)
            ng = g1 - g0
            trans = (seed[p0:p1, None, :] + offsets[None, g0:g1, :]).reshape(-1, 3)
            c = xp.repeat(centres[p0:p1], ng, axis=0)
            values, inside = sample(c, trans)
            f = xp.repeat(ref_values[p0:p1], ng, axis=0)
            obj = objective(f, values, search.objective).astype(params.dtype)
            obj = xp.where(inside, obj, xp.inf).reshape(p1 - p0, ng)
            k = obj.argmin(axis=1)
            val = obj[xp.arange(p1 - p0), k]
            better = val < best[p0:p1]
            best[p0:p1] = xp.where(better, val, best[p0:p1])
            cand = seed[p0:p1] + offsets[g0 + k]
            best_t[p0:p1] = xp.where(better[:, None], cand, best_t[p0:p1])
    params[idx, :3] = best_t


def fft_seed(
    ref_brick: Brick,
    def_brick: Brick,
    centres: Any,
    seeds: Any,
    template: Template,
    search: SearchSpec,
) -> Any:
    """(B, 3) integer-plus-subvoxel translations and (B,) peak NCC as a confidence."""
    raise todo("M5", "fft_seed")
