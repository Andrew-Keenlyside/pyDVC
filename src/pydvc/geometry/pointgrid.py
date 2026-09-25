"""Point-cloud generation, following iDVC's Point Cloud panel.

iDVC places points on a regular lattice inside a mask. The spacing comes from
subvolume size and overlap, with an optional lattice rotation. The mask can
optionally be eroded so whole subvolumes stay inside it. pyDVC does the same
in chunks, so a 4096^3 mask never has to be in memory at once, and writes the
result straight into a zarr-vectors store
(:func:`pydvc.io.pointcloud.write_pointcloud_store`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pydvc._todo import todo
from pydvc.config import SubvolumeSpec


def lattice_spacing(subvolume: SubvolumeSpec, overlap: tuple[float, float, float]) -> tuple[float, float, float]:
    """Point spacing (x, y, z) for a given fractional overlap, as iDVC computes it."""
    raise todo("M1", "lattice_spacing")


def grid_in_mask(
    shape_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    *,
    mask: Any = None,
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    erode_radius: float = 0.0,
    origin_xyz: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Lattice points (N, 3) inside ``mask`` (a (z, y, x) array or a zarr array read slab by slab)."""
    raise todo("M1", "grid_in_mask")
