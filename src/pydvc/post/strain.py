"""Strain from displacement, in the spirit of CCPi ``StrainCalc`` and iDVC's strain panel.

At each point, fit an affine field ``u(x) = u0 + G (x - x0)`` by least squares
to the GOOD neighbours within a radius, or the k nearest. Take
``eps = (G + G^T) / 2`` and write it as the ``strain`` attribute. Neighbours
across tile borders come from ``ResultStore.read_neighbourhood(halo=1)``, so
the fit runs tile-parallel with no seams. Batched small least-squares solves
on the GPU.
"""

from __future__ import annotations

from typing import Any

from pydvc._todo import todo
from pydvc.config import RunConfig


def fit_strain(xyz: Any, displacement: Any, status: Any, neighbours: Any, *, min_good: int = 6) -> tuple[Any, Any]:
    """(N, 6) strain (exx, eyy, ezz, exy, eyz, exz) and (N,) fit residual."""
    raise todo("M5", "fit_strain")


def compute_strain(cfg: RunConfig, *, radius: float | None = None, k: int = 26) -> None:
    raise todo("M5", "compute_strain")
