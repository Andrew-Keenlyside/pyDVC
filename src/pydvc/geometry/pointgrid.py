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

from pydvc.config import SubvolumeSpec

_SLAB = 128   # mask slices read at a time


def lattice_spacing(subvolume: SubvolumeSpec, overlap: tuple[float, float, float]) -> tuple[float, float, float]:
    """Point spacing (x, y, z) for a given fractional overlap, as iDVC computes it.

    Neighbouring subvolumes overlap by ``overlap`` of their extent along each
    axis, so the spacing is ``size * aspect * (1 - overlap)``.
    """
    if any(not 0.0 <= o < 1.0 for o in overlap):
        raise ValueError("overlap must be in [0, 1) per axis")
    return tuple(subvolume.size * a * (1.0 - o) for a, o in zip(subvolume.aspect, overlap))


def _rotation(rotation_deg: tuple[float, float, float]) -> np.ndarray:
    """Lattice rotation: about x, then y, then z (degrees)."""
    ax, ay, az = np.deg2rad(rotation_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def grid_in_mask(
    shape_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    *,
    mask: Any = None,
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    erode_radius: float = 0.0,
    origin_xyz: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Lattice points (N, 3) inside ``mask`` (a (z, y, x) array or a zarr array read slab by slab).

    The lattice passes through ``origin_xyz`` (default: the volume centre) and
    is rotated about it. Points are kept if they lie inside the volume and, with
    a mask, if the nearest mask voxel is set and (``erode_radius > 0``) at least
    ``erode_radius`` voxels from the nearest unset voxel. Points come out sorted
    by (z, y, x).
    """
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    if (spacing <= 0).any():
        raise ValueError("spacing must be positive")
    extent = np.asarray(shape_zyx[::-1], dtype=np.float64) - 1.0            # (x, y, z) upper bound
    origin = 0.5 * extent if origin_xyz is None else np.asarray(origin_xyz, dtype=np.float64)
    rot = _rotation(rotation_deg)
    # lattice indices whose rotated points can fall inside the volume box
    reach = np.linalg.norm(np.maximum(np.abs(origin), np.abs(extent - origin))) + spacing.max()
    n = np.ceil(reach / spacing).astype(int)
    axes = [np.arange(-k, k + 1) * s for k, s in zip(n, spacing)]
    gz, gy, gx = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    local = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    pts = local @ rot.T + origin
    keep = ((pts >= -1e-9) & (pts <= extent + 1e-9)).all(axis=1)
    pts = pts[keep]
    if mask is not None:
        pts = pts[_in_mask(pts, mask, erode_radius)]
    order = np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2]))
    return pts[order]


def _in_mask(pts: np.ndarray, mask: Any, erode_radius: float) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    nz = mask.shape[0]
    vox = np.clip(np.rint(pts[:, ::-1]).astype(np.int64), 0, np.asarray(mask.shape) - 1)   # (z, y, x)
    margin = int(np.ceil(erode_radius)) + 1 if erode_radius > 0 else 0
    keep = np.zeros(len(pts), dtype=bool)
    for z0 in range(0, nz, _SLAB):
        z1 = min(z0 + _SLAB, nz)
        sel = np.flatnonzero((vox[:, 0] >= z0) & (vox[:, 0] < z1))
        if sel.size == 0:
            continue
        lo, hi = max(z0 - margin, 0), min(z1 + margin, nz)
        slab = np.asarray(mask[lo:hi]).astype(bool)
        if erode_radius > 0:
            # the volume border counts as outside, so eroded points keep their subvolume in the image
            padded = np.pad(slab, 1, constant_values=False)
            if lo > 0:
                padded[0] = True
            if hi < nz:
                padded[-1] = True
            inner = distance_transform_edt(padded)[1:-1, 1:-1, 1:-1] >= erode_radius
        else:
            inner = slab
        v = vox[sel]
        keep[sel] = inner[v[:, 0] - lo, v[:, 1], v[:, 2]]
    return keep
