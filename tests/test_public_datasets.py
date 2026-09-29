"""The public-dataset study's cases, point grids and field fit (the runs themselves need the downloaded data)."""

import numpy as np

from zvdvc.bench import public_datasets as pds


def test_cases_cover_the_series_once():
    cases = pds.cases()
    names = [c.name for c in cases]
    assert len(names) == len(set(names))
    tr = [c for c in cases if c.group == "translation"]
    assert [c.truth_xyz[1] for c in tr] == [round(0.1 * k, 1) for k in range(11)]
    assert {c.deformed for c in tr} == {f"vol_translation_{1000 + k}" for k in range(1, 12)}
    assert len([c for c in cases if c.group == "stretch"]) == 6
    assert {c.size for c in cases if c.group == "repeat"} == {24.0, 32.0, 48.0}
    # the moves are seeded within a few voxels of the nominal 1 mm (56 voxels at 17.86 um)
    for c in cases:
        if c.group in ("axial", "radial"):
            assert abs(np.linalg.norm(c.offset_xyz) - 1000 / pds.DVC1_VOXEL_UM) < 3


def test_bead_grid_keeps_subvolumes_inside_the_volume():
    xyz = pds.beads_points()
    reach = pds.BEADS_SIZE / 2 + pds.STENCIL
    hi = np.asarray(pds.BEADS_SHAPE_ZYX[::-1]) - 1
    assert (xyz - reach >= 0).all() and (xyz + reach <= hi).all()
    ids, first = pds.centre_first(xyz.copy(), hi / 2)
    assert list(ids) == list(range(1, len(xyz) + 1))
    assert np.linalg.norm(first[0] - hi / 2) == np.linalg.norm(xyz - hi / 2, axis=1).min()
    assert sorted(map(tuple, first)) == sorted(map(tuple, xyz))


def test_affine_fit_recovers_a_homogeneous_stretch():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 500, (400, 3))
    grad = np.diag([0.0, 0.05, 0.0])
    u = (xyz - 255.5) @ grad.T + rng.normal(0, 0.01, xyz.shape)
    coef, pred = pds.affine_fit(xyz, u)
    assert np.allclose(coef[1:].T, grad, atol=2e-4)
    assert np.allclose(coef[0], -255.5 * np.diag(grad), atol=0.01)
    assert abs((u - pred).std() - 0.01) < 0.002
