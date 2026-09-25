"""Analytic test volumes: sums of plane waves, so a displaced copy is exact (no resampling error)."""

import numpy as np

from pydvc.geometry.box import Box
from pydvc.io.volume import Brick


def wave_field(shape_zyx, shift_xyz=(0.0, 0.0, 0.0), *, n_waves=40, seed=0):
    """``g(x) = f(x - shift)``: the reference point ``c`` appears at ``c + shift`` in the result."""
    rng = np.random.default_rng(seed)
    # |k| <= 0.87 rad/voxel (wavelength >= ~7 voxels): rich texture, small interpolation bias.
    k = rng.uniform(0.1, 0.5, (n_waves, 3)) * rng.choice([-1.0, 1.0], (n_waves, 3))
    phase = rng.uniform(0.0, 2.0 * np.pi, n_waves)
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape_zyx), indexing="ij")
    pos = np.stack([x - shift_xyz[0], y - shift_xyz[1], z - shift_xyz[2]], axis=-1)
    return 100.0 + 10.0 * np.cos(pos @ k.T + phase).sum(axis=-1)


def whole_brick(volume):
    box = Box((0, 0, 0), tuple(volume.shape))
    return Brick(data=volume, box=box, valid=box)
