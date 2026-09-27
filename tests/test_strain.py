"""Strain (CCPi strain's method): exact fields, planar clouds, windows, calibrated uncertainty, and CCPi itself."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pydvc.post.strain import compute_strain, fit_strain, planar_axis

G_TRUE = np.array([[0.010, -0.004, 0.002], [0.003, -0.006, 0.001], [-0.002, 0.005, 0.008]])


def _grid(n=9, spacing=8.0, planar=False):
    ax = np.arange(n) * spacing
    z = [40.0] if planar else ax
    return np.stack(np.meshgrid(ax, ax, z, indexing="ij"), -1).reshape(-1, 3)


def test_an_affine_field_is_recovered_exactly():
    xyz = _grid()
    disp = np.array([1.0, -2.0, 0.5]) + xyz @ G_TRUE.T
    r = fit_strain(xyz, disp, np.zeros(len(xyz)))
    np.testing.assert_allclose(r.grad, np.broadcast_to(G_TRUE, r.grad.shape), atol=1e-10)
    eng = 0.5 * (G_TRUE + G_TRUE.T)
    np.testing.assert_allclose(r.engineering[0], [eng[0, 0], eng[1, 1], eng[2, 2], eng[0, 1], eng[1, 2], eng[0, 2]], atol=1e-10)
    lag = eng + 0.5 * G_TRUE.T @ G_TRUE
    np.testing.assert_allclose(r.lagrangian[0, :3], np.diag(lag), atol=1e-10)
    np.testing.assert_allclose(r.principal_engineering[0], np.sort(np.linalg.eigvalsh(eng))[::-1], atol=1e-10)
    np.testing.assert_allclose(r.u_fit, disp, atol=1e-9)


def test_a_quadratic_field_gives_the_gradient_at_the_point():
    xyz = _grid()
    x, y, z = xyz.T
    disp = np.stack([1e-4 * x * y, 2e-4 * z ** 2, 1e-4 * x * z], axis=1)
    r = fit_strain(xyz, disp, np.zeros(len(xyz)))
    i = np.argmin(np.linalg.norm(xyz - 32.0, axis=1))            # an interior point
    want = np.array([[1e-4 * y[i], 1e-4 * x[i], 0], [0, 0, 4e-4 * z[i]], [1e-4 * z[i], 0, 1e-4 * x[i]]])
    np.testing.assert_allclose(r.grad[i], want, atol=1e-10)


def test_a_planar_cloud_is_fitted_in_its_plane():
    xyz = _grid(planar=True)
    assert planar_axis(xyz) == 2
    disp = xyz @ G_TRUE.T
    r = fit_strain(xyz, disp, np.zeros(len(xyz)))
    want = G_TRUE.copy()
    want[:, 2] = 0.0
    want[2, :] = 0.0                                              # CCPi zeroes the out-of-plane component
    np.testing.assert_allclose(r.grad[40], want, atol=1e-10)
    assert (r.principal_engineering[:, 2] == 0).all()


def test_windows_skip_failed_points_and_refill_replaces_them():
    xyz = _grid()
    disp = xyz @ G_TRUE.T
    status = np.where(np.random.default_rng(3).random(len(xyz)) < 0.2, -1, 0)
    plain = fit_strain(xyz, disp, status, window=25)
    refilled = fit_strain(xyz, disp, status, window=25, refill=True)
    assert (plain.pts_in_sw < 25).any() and (refilled.pts_in_sw == 25).all()
    ok = np.isfinite(refilled.grad).all(axis=(1, 2))
    assert ok.mean() > 0.9                                        # corner windows that cannot fit a quadratic are NaN
    np.testing.assert_allclose(refilled.grad[ok], np.broadcast_to(G_TRUE, refilled.grad[ok].shape), atol=1e-9)


def test_windows_that_cannot_determine_the_model_are_nan_not_wrong():
    xyz = _grid(n=9)
    two_levels = xyz[xyz[:, 2] <= 8.0]                             # every window spans only z = 0 and z = 8
    r = fit_strain(two_levels, two_levels @ G_TRUE.T, np.zeros(len(two_levels)), window=25)
    assert np.isnan(r.grad).all()


def test_the_objmin_threshold_drops_window_points():
    xyz = _grid()
    obj = np.where(np.arange(len(xyz)) % 2 == 0, 0.5, 0.01)
    assert fit_strain(xyz, xyz @ G_TRUE.T, np.zeros(len(xyz)), obj, threshold=0.1).pts_in_sw.max() < 25


def test_too_few_good_points_give_nan():
    xyz = _grid(n=4)
    r = fit_strain(xyz, xyz @ G_TRUE.T, np.zeros(len(xyz)), window=6)        # 10 terms need >= 10 points
    assert np.isnan(r.engineering).all()


def test_strain_sd_is_calibrated():
    rng = np.random.default_rng(0)
    xyz = _grid(n=14, spacing=6.0)
    sigma = 0.03
    disp = xyz @ G_TRUE.T + rng.normal(scale=sigma, size=xyz.shape)
    r = fit_strain(xyz, disp, np.zeros(len(xyz)), sigma_u=sigma)
    interior = np.all((xyz > 18) & (xyz < 60), axis=1)
    err = r.engineering[interior, 0] - G_TRUE[0, 0]
    ratio = err.std() / np.median(r.strain_sd[interior, 0])
    assert 0.6 < ratio < 1.6                                      # predicted and actual spread agree
    r_resid = fit_strain(xyz, disp, np.zeros(len(xyz)))           # from the residual instead of a given sigma
    assert 0.6 < np.median(r_resid.strain_sd[interior, 0]) / np.median(r.strain_sd[interior, 0]) < 1.6


def test_strain_from_a_disp_file_writes_ccpi_csv(tmp_path):
    from pydvc.ccpi_dropin import write_disp

    xyz = _grid()
    write_disp(tmp_path / "run.disp", np.arange(1, len(xyz) + 1), xyz, np.zeros(len(xyz), dtype=int),
               np.full(len(xyz), 0.01), xyz @ G_TRUE.T)
    info = compute_strain(tmp_path / "run.disp", outputs=("Lstr", "Estr", "dgrd"))
    head = Path(info["files"]["Estr"]).read_text().splitlines()[0]
    assert head.startswith("n,x,y,z,u_fit,v_fit,w_fit,pts_in_sw,sw_radius,exx,eyy,ezz,exy,eyz,exz,ep1,ep2,ep3")
    assert Path(info["files"]["dgrd"]).read_text().splitlines()[0].endswith("ux,uy,uz,vx,vy,vz,wx,wy,wz")
    assert info["with_strain"] == len(xyz)


def test_matches_ccpi_strain_on_interior_points(tmp_path):
    exe = Path.home() / ".local/opt/ccpi-dvc-22.0.0/bin/strain"
    if not exe.exists() and shutil.which("strain") is None:
        pytest.skip("CCPi strain not installed")
    exe = exe if exe.exists() else Path(shutil.which("strain"))
    from pydvc.ccpi_dropin import write_disp

    rng = np.random.default_rng(1)
    # jittered, so no two neighbours are equidistant: on a regular 3D grid a 25-point window always ends
    # inside the ring of 8 corner neighbours, where CCPi's unstable sort and a KD-tree break ties differently
    xyz = _grid(n=12, spacing=8.0) + rng.uniform(-1.0, 1.0, size=(12 ** 3, 3))
    disp = xyz @ G_TRUE.T + 1e-5 * xyz ** 2 + rng.normal(scale=0.01, size=xyz.shape)
    write_disp(tmp_path / "c.disp", np.arange(1, len(xyz) + 1), xyz, np.zeros(len(xyz), dtype=int),
               np.full(len(xyz), 0.01), disp)
    subprocess.run([str(exe), "c.disp", "-E"], cwd=tmp_path, check=True, capture_output=True)
    compute_strain(tmp_path / "c.disp", out_base=tmp_path / "p", outputs=("Estr",))
    import csv

    load = lambda p: {k: np.array([float(r[k]) for r in rows]) for rows in [list(csv.DictReader(open(p)))] for k in rows[0]}  # noqa: E731
    c, p = load(tmp_path / "c-sw25.Estr.csv"), load(tmp_path / "p-sw25.Estr.csv")
    interior = np.all((xyz >= 15) & (xyz <= 73), axis=1)
    for k in ("exx", "eyy", "ezz", "exy", "eyz", "exz", "ep1", "ep2", "ep3"):
        np.testing.assert_allclose(p[k][interior], c[k][interior], atol=2e-6, err_msg=k)
