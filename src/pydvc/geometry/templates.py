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

from dataclasses import dataclass

import numpy as np

from pydvc._todo import todo
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
        raise todo("M1", "Template.extent")


def make_template(spec: SubvolumeSpec) -> Template:
    raise todo("M1", "make_template")
