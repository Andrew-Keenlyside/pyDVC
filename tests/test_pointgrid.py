import numpy as np
import pytest

from pydvc.config import SubvolumeSpec
from pydvc.geometry.pointgrid import grid_in_mask, lattice_spacing


def test_lattice_spacing_from_overlap():
    assert lattice_spacing(SubvolumeSpec(size=40, aspect=(1.0, 1.0, 2.0)), (0.5, 0.25, 0.5)) == (20.0, 30.0, 40.0)
    with pytest.raises(ValueError):
        lattice_spacing(SubvolumeSpec(size=40), (1.0, 0.0, 0.0))


def test_lattice_is_centred_and_inside_the_volume():
    pts = grid_in_mask((33, 65, 65), (16.0, 16.0, 8.0))
    assert ((pts >= 0) & (pts <= [64, 64, 32])).all()
    assert [32.0, 32.0, 16.0] in pts.tolist()
    np.testing.assert_allclose(np.unique(pts[:, 0]), [0, 16, 32, 48, 64])


def test_mask_and_erosion():
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[:, :, :20] = True                                          # x < 20
    pts = grid_in_mask(mask.shape, (4.0, 4.0, 4.0), mask=mask)
    assert (pts[:, 0] < 20).all()
    eroded = grid_in_mask(mask.shape, (4.0, 4.0, 4.0), mask=mask, erode_radius=5.0)
    assert len(eroded) < len(pts)
    # nearest voxel at least 5 from unset voxels and from the volume border
    vox = np.rint(eroded)
    assert (vox[:, 0] <= 20 - 5).all() and (vox >= 4).all() and (vox <= 39 - 4).all()


def test_rotated_lattice_keeps_its_spacing():
    pts = grid_in_mask((64, 64, 64), (8.0, 8.0, 8.0), rotation_deg=(0.0, 0.0, 30.0))
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    np.testing.assert_allclose(d.min(axis=1), 8.0, atol=1e-9)
