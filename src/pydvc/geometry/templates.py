"""Subvolume sample templates.

As in CCPi, samples are **not voxel-centred** and their number is independent
of subvolume size. Interpolation supplies values at arbitrary positions.

Unlike CCPi, which builds a ``FloatingCloud`` per point, pyDVC builds **one
template per run** and shares it across all points. On the GPU the template
offsets sit in constant/shared memory, so a batch is described by its centres
and parameters alone and per-point sample coordinates are never stored.

* ``cube``: a ``k x k x k`` grid with ``k = ceil(n_samples ** (1/3))``, spanning
  ``[-r*aspect, +r*aspect]`` per axis (CCPi ``Search`` rounds the count up the same way).
* ``sphere``: ``n_samples`` points uniform inside the ellipsoid of radius
  ``r*aspect``, by rejection from the bounding cube with a seeded Mersenne
  Twister. CCPi does the same with ``std::mt19937``. The streams differ, so
  parity with CCPi is statistical (docs/MVP_PLAN.md, M1).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from pydvc.config import SubvolumeSpec


@dataclass(frozen=True)
class Template:
    offsets: np.ndarray        # (M, 3) float32, (x, y, z) relative to the point centre, voxels
    spec: SubvolumeSpec

    @property
    def n_samples(self) -> int:
        return int(self.offsets.shape[0])

    def extent(self) -> float:
        """Largest ``|offset|``. Rotations can bring any sample this far along any axis, so it sizes the halo."""
        return float(np.sqrt((self.offsets.astype(np.float64) ** 2).sum(axis=1)).max())

    def digest(self) -> str:
        """Hash of the offsets, recorded with results so a resumed run cannot mix templates."""
        return hashlib.sha256(np.ascontiguousarray(self.offsets).tobytes()).hexdigest()[:16]


def cube_side(n_samples: int) -> int:
    """Smallest ``k`` with ``k**3 >= n_samples`` (exact in integers; ``ceil(n ** (1/3))`` is not)."""
    k = max(1, round(n_samples ** (1.0 / 3.0)))
    while k**3 < n_samples:
        k += 1
    while k > 1 and (k - 1) ** 3 >= n_samples:
        k -= 1
    return k


def make_template(spec: SubvolumeSpec) -> Template:
    if spec.n_samples < 1:
        raise ValueError("subvolume n_samples must be >= 1")
    radius = 0.5 * spec.size * np.asarray(spec.aspect, dtype=np.float64)       # (x, y, z)
    if spec.geometry == "cube":
        k = cube_side(spec.n_samples)
        axis = np.linspace(-1.0, 1.0, k) if k > 1 else np.zeros(1)
        z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
        offsets = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) * radius
    elif spec.geometry == "sphere":
        rng = np.random.Generator(np.random.MT19937(spec.seed))
        accepted: list[np.ndarray] = []
        n = 0
        while n < spec.n_samples:
            # the unit ball fills pi/6 of its bounding cube
            cand = rng.uniform(-1.0, 1.0, size=(2 * (spec.n_samples - n) + 64, 3))
            cand = cand[(cand**2).sum(axis=1) <= 1.0]
            accepted.append(cand)
            n += len(cand)
        offsets = np.concatenate(accepted)[: spec.n_samples] * radius
    else:
        raise ValueError(f"unknown subvolume geometry {spec.geometry!r}")
    return Template(offsets=offsets.astype(np.float32), spec=spec)
