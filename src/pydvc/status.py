"""Per-point search status.

Codes 0 to -3 are CCPi DVC's (``Core/Utility.h``) and keep their meaning, so
exported ``.disp`` files read correctly in iDVC. Codes below -3 are pyDVC
additions; the CCPi exporter maps them to ``NOT_SEARCHED``.
"""

from __future__ import annotations

from enum import IntEnum


class PointStatus(IntEnum):
    GOOD = 0            # converged within disp_max of the seed
    RANGE_FAIL = -1     # left the disp_max range, or samples left the brick
    CONVG_FAIL = -2     # max_iterations reached without meeting a tolerance
    NOT_SEARCHED = -3   # never attempted (beyond num_points_to_process, or pending)
    # pyDVC extensions
    THRESH_FAIL = -4    # subvol_thresh: foreground fraction below min_fraction
    SINGULAR = -5       # normal equations ill-conditioned (e.g. featureless subvolume)

    def to_ccpi(self) -> int:
        return int(self) if self >= PointStatus.NOT_SEARCHED else int(PointStatus.NOT_SEARCHED)
