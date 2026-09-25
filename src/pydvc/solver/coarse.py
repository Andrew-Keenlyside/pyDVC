"""Global (coarse) searches that produce a starting translation.

``translation_grid_search``
    CCPi ``Search::trgrid_global``, batched. With ``n = floor(disp_max /
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

from typing import Any

from pydvc._todo import todo
from pydvc.config import SearchSpec
from pydvc.geometry.templates import Template
from pydvc.io.volume import Brick


def translation_grid_search(
    ref_values: Any,         # (B, M)
    ref_stats: Any,          # (B, 2)
    def_brick: Brick,
    centres: Any,            # (B, 3)
    params: Any,             # (B, ndof), translation updated in place
    template: Template,
    search: SearchSpec,
) -> None:
    raise todo("M2", "translation_grid_search")


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
