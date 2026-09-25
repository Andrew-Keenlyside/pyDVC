"""Speckle phantoms and analytic displacement fields.

Ground truth is what makes the MVP a feasibility *test*: every accuracy claim
in docs/MVP_PLAN.md is measured against these fields, not against another code.

Volumes are generated and warped chunk by chunk (overlap = filter support +
max displacement), straight into sharded OME-Zarr, so 4096^3 cases can be
built on a workstation with bounded memory.

Fields (all analytic, with gradients for strain checks):

``rigid``      translation + small rotation. Tests seeding, range handling and 6-DOF.
``affine``     uniform strain up to ~1 %. Tests 12-DOF and strain post-processing.
``sinusoid``   ``u = A sin(2 pi x / lambda)`` along one axis. Tests spatial resolution versus subvolume size.
``inclusion``  Eshelby-like stiff sphere in a strained matrix. A realistic mix of gradients.

The deformed volume uses backward mapping: ``g(x) = f(X)`` where ``x = X +
u(X)``, with ``X`` found by fixed-point iteration. Interpolation is
quintic B-spline, so the phantom's interpolation error sits well below that of
the tricubic being tested. Optional Gaussian and Poisson noise are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydvc._todo import todo

FieldKind = Literal["rigid", "affine", "sinusoid", "inclusion"]


@dataclass(frozen=True)
class DisplacementField:
    kind: FieldKind
    params: dict[str, Any]

    def __call__(self, xyz: Any) -> Any:
        """(N, 3) -> (N, 3) displacement."""
        raise todo("M0", "DisplacementField.__call__")

    def gradient(self, xyz: Any) -> Any:
        """(N, 3) -> (N, 3, 3) displacement gradient, for strain checks."""
        raise todo("M0", "DisplacementField.gradient")


def speckle(
    shape_zyx: tuple[int, int, int],
    *,
    feature_size: float = 3.0,
    dtype: str = "uint16",
    seed: int = 0,
) -> Any:
    """Band-limited random speckle (filtered noise), scaled to the dtype's range."""
    raise todo("M0", "speckle")


def warp_volume(ref: Any, field: DisplacementField, *, noise_sigma: float = 0.0) -> Any:
    raise todo("M0", "warp_volume")


def make_case(
    out_dir: str | Path,
    *,
    shape_zyx: tuple[int, int, int],
    field: DisplacementField,
    spacing: float = 16.0,
    dtype: str = "uint16",
    noise_sigma: float = 0.0,
    chunk: int = 128,
    shard: int = 1024,
) -> Path:
    """Write ``ref.ome.zarr``, ``def.ome.zarr``, ``points.zarrvectors``, ``truth.npz`` and ``config.yaml``; return the config path."""
    raise todo("M0", "make_case")
