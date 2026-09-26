"""Integer voxel boxes, half-open, in array order ``(z, y, x)``."""

from __future__ import annotations

import math
from dataclasses import dataclass

Int3 = tuple[int, int, int]


@dataclass(frozen=True)
class Box:
    lo: Int3   # inclusive, (z, y, x)
    hi: Int3   # exclusive, (z, y, x)

    @property
    def shape(self) -> Int3:
        return (self.hi[0] - self.lo[0], self.hi[1] - self.lo[1], self.hi[2] - self.lo[2])

    def voxels(self) -> int:
        z, y, x = self.shape
        return z * y * x

    def grow(self, margin: float | Int3) -> Box:
        """Expand by ``margin`` voxels on every side (rounded outward). A 3-tuple margin is ``(z, y, x)``."""
        m = (margin,) * 3 if isinstance(margin, (int, float)) else tuple(margin)
        m = tuple(math.ceil(v) for v in m)
        return Box(
            (self.lo[0] - m[0], self.lo[1] - m[1], self.lo[2] - m[2]),
            (self.hi[0] + m[0], self.hi[1] + m[1], self.hi[2] + m[2]),
        )

    def shift_xyz(self, offset_xyz: tuple[float, float, float]) -> Box:
        """Translate by a displacement given in point order ``(x, y, z)``, rounded outward."""
        off = (offset_xyz[2], offset_xyz[1], offset_xyz[0])
        return Box(
            tuple(math.floor(lo + o) for lo, o in zip(self.lo, off)),
            tuple(math.ceil(hi + o) for hi, o in zip(self.hi, off)),
        )

    def intersect(self, other: Box) -> Box | None:
        lo = tuple(max(a, b) for a, b in zip(self.lo, other.lo))
        hi = tuple(min(a, b) for a, b in zip(self.hi, other.hi))
        if any(h <= l for l, h in zip(lo, hi)):
            return None
        return Box(lo, hi)

    def slices(self, origin: Int3 = (0, 0, 0)) -> tuple[slice, slice, slice]:
        """Index expression selecting this box from an array whose ``[0, 0, 0]`` sits at ``origin``."""
        return tuple(slice(l - o, h - o) for l, h, o in zip(self.lo, self.hi, origin))

    @classmethod
    def around_points(cls, xyz_min: tuple[float, float, float], xyz_max: tuple[float, float, float]) -> Box:
        """Smallest box holding every voxel whose coordinate lies in ``[xyz_min, xyz_max]``."""
        lo = tuple(math.ceil(v) for v in reversed(xyz_min))
        hi = tuple(math.floor(v) + 1 for v in reversed(xyz_max))
        return cls(lo, hi)
