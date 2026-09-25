"""Integer voxel boxes, half-open, in array order ``(z, y, x)``."""

from __future__ import annotations

from dataclasses import dataclass

from pydvc._todo import todo

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
        """Expand by ``margin`` voxels on every side (rounded outward)."""
        raise todo("M3", "Box.grow")

    def shift_xyz(self, offset_xyz: tuple[float, float, float]) -> Box:
        """Translate by a displacement given in point order ``(x, y, z)``, rounded outward."""
        raise todo("M3", "Box.shift_xyz")

    def intersect(self, other: Box) -> Box | None:
        raise todo("M3", "Box.intersect")

    @classmethod
    def around_points(cls, xyz_min: tuple[float, float, float], xyz_max: tuple[float, float, float]) -> Box:
        """Smallest box holding every voxel whose coordinate lies in ``[xyz_min, xyz_max]``."""
        raise todo("M3", "Box.around_points")
