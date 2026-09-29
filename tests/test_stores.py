"""M3: zarr-vectors point-cloud and results stores (three-phase writes, reads, validation)."""

import numpy as np
import pytest

from zvdvc.io.ccpi import write_roi
from zvdvc.io.pointcloud import PointCloud, bin_index, import_points, write_pointcloud_store
from zvdvc.io.results import OPTIONAL_ATTRIBUTES, RESULT_ATTRIBUTES, ResultStore

BOUNDS = ((0.0, 0.0, 0.0), (256.0, 256.0, 256.0))
rng = np.random.default_rng(11)
XYZ = rng.uniform(0.0, 256.0, (3000, 3))
IDS = np.arange(1, 3001)


@pytest.fixture
def cloud(tmp_path):
    path = tmp_path / "points.zarrvectors"
    write_pointcloud_store(path, XYZ, IDS, bounds=BOUNDS, chunk_shape=(64.0, 64.0, 64.0))
    return PointCloud(path)


def _validate(path):
    from zarr_vectors.validate import validate

    return validate(str(path))


def test_point_store_round_trip_and_metadata(cloud):
    assert cloud.n_points == 3000 and cloud.chunk_shape == (64.0, 64.0, 64.0) and cloud.bin_shape == (32.0, 32.0, 32.0)
    assert len(cloud.cells()) == 64 and sum(cloud.cell_counts().values()) == 3000
    xyz, ids = cloud.read_all()
    np.testing.assert_array_equal(xyz, XYZ[ids - 1].astype(np.float32))
    r = _validate(cloud.path)
    assert not r.errors and not r.warnings            # fragments are the bins zarr-vectors expects


def test_tile_rows_are_grouped_by_cell_and_bin(cloud):
    cells = cloud.cells()[:5]
    tile = cloud.read_tile(cells, device="cpu")
    assert tile.cell_offsets[-1] == len(tile.point_id) == sum(cloud.cell_counts()[c] for c in cells)
    for i, cell in enumerate(cells):
        rows = tile.xyz[tile.cell_offsets[i]:tile.cell_offsets[i + 1]]
        assert (np.floor(rows / 64.0).astype(int) == cell).all()
        b = bin_index(rows, cell, (64.0,) * 3, (32.0,) * 3)
        assert (np.diff(b) >= 0).all()                   # fragment order
        np.testing.assert_array_equal(np.diff(tile.bin_offsets[i]), np.bincount(b, minlength=8))


def test_bbox_query_through_zarr_vectors(cloud):
    import zarr_vectors as zv

    q = zv.open(cloud.path).select(bbox=((0.0, 0.0, 0.0), (64.0, 128.0, 64.0)))
    assert q.count() == int(((XYZ[:, 0] < 64) & (XYZ[:, 1] < 128) & (XYZ[:, 2] < 64)).sum())


def test_import_from_roi(tmp_path):
    write_roi(tmp_path / "p.roi", IDS[:50], XYZ[:50] / 4.0)
    pc = import_points(tmp_path / "p.roi", tmp_path / "p.zarrvectors", volume_shape_zyx=(64, 64, 64), chunk_shape=(16.0,) * 3)
    xyz, ids = pc.read_all()
    assert sorted(ids.tolist()) == IDS[:50].tolist()


def test_points_outside_bounds_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        write_pointcloud_store(tmp_path / "p.zarrvectors", XYZ + 300.0, IDS, bounds=BOUNDS, chunk_shape=(64.0,) * 3)


def _fake_results(tile, dof):
    n = len(tile.point_id)
    return {
        "status": np.where(tile.point_id % 7 == 0, -1, 0).astype(np.int8),
        "objmin": tile.point_id.astype(np.float32) * 1e-3,
        "displacement": tile.xyz * 0.01,
        "params": np.tile(tile.point_id[:, None], (1, dof)).astype(np.float32),
        "n_iter": np.full(n, 4, np.uint8),
        "seed": np.zeros((n, 3), np.float32),
    }


def test_results_store_three_phase_write(cloud, tmp_path):
    out = tmp_path / "results.zarrvectors"
    store = ResultStore.allocate(out, points=cloud, dof=6)
    cells = cloud.cells()
    for part in (cells[::2], cells[1::2]):                   # two disjoint "workers"
        tile = cloud.read_tile(part, device="cpu")
        store.write_tile(tile, _fake_results(tile, 6))
    assert store.written_cells() == set(cells)              # visible before finalize (deferred presence)
    store.finalize(n_points=cloud.n_points)
    back = ResultStore(out).read_all()
    assert set(back) == {"xyz"} | set(RESULT_ATTRIBUTES) | set(OPTIONAL_ATTRIBUTES)
    assert np.isnan(back["displacement_sd"]).all()          # optional, not computed by these fake results
    assert sorted(back["point_id"].tolist()) == IDS.tolist()
    np.testing.assert_array_equal(back["params"][:, 3], back["point_id"].astype(np.float32))
    np.testing.assert_allclose(back["displacement"], back["xyz"] * 0.01, rtol=1e-6)
    assert (back["status"][back["point_id"] % 7 == 0] == -1).all()
    r = _validate(out)
    assert not r.errors and not r.warnings


def test_results_store_protects_itself(cloud, tmp_path):
    out = tmp_path / "results.zarrvectors"
    ResultStore.allocate(out, points=cloud, dof=3)
    ro = ResultStore(out)
    tile = cloud.read_tile(cloud.cells()[:1], device="cpu")
    with pytest.raises(PermissionError):
        ro.write_tile(tile, _fake_results(tile, 3))
    with pytest.raises(ValueError, match="missing"):
        ResultStore(out, mode="r+").write_tile(tile, {"status": np.zeros(len(tile.point_id))})
    with pytest.raises(ValueError, match="not a zvDVC results store"):
        ResultStore(cloud.path)


def test_results_store_written_by_pydvc_still_opens(cloud, tmp_path):
    import zarr

    out = tmp_path / "results.zarrvectors"
    ResultStore.allocate(out, points=cloud, dof=6)
    attrs = zarr.open_group(str(out), mode="r+", zarr_format=3).attrs
    attrs["pydvc_results"] = attrs.pop("zvdvc_results")    # the key before the rename
    assert ResultStore(out).dof == 6


def test_duplicate_point_ids_are_rejected(tmp_path):
    ids = IDS[:20].copy()
    ids[[3, 7, 9]] = ids[[2, 6, 6]]
    with pytest.raises(ValueError, match=r"2 point ids occur more than once \(3, 7\)"):
        write_pointcloud_store(tmp_path / "p.zarrvectors", XYZ[:20], ids, bounds=BOUNDS, chunk_shape=(64.0,) * 3)
    write_roi(tmp_path / "p.roi", ids, XYZ[:20] / 4.0)
    with pytest.raises(ValueError, match="more than once"):
        import_points(tmp_path / "p.roi", tmp_path / "q.zarrvectors", volume_shape_zyx=(64, 64, 64), chunk_shape=(16.0,) * 3)
    assert not (tmp_path / "p.zarrvectors").exists() and not (tmp_path / "q.zarrvectors").exists()


def test_non_finite_and_empty_point_clouds_are_rejected(tmp_path):
    from zvdvc.io.pointcloud import read_roi

    (tmp_path / "nan.roi").write_text("1\t15.5\t15.5\t15.5\n2\tnan\t15.5\t15.5\n")
    with pytest.raises(ValueError, match=r"nan.roi:2: point 2 has a non-finite coordinate"):
        read_roi(tmp_path / "nan.roi")
    xyz = XYZ[:5].copy()
    xyz[3, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite coordinates, e.g. point 4"):
        write_pointcloud_store(tmp_path / "p.zarrvectors", xyz, IDS[:5], bounds=BOUNDS, chunk_shape=(64.0,) * 3)
    (tmp_path / "empty.roi").write_text("n x y z\n")
    with pytest.raises(ValueError, match="empty.roi: no points"):
        read_roi(tmp_path / "empty.roi")


def test_a_cell_without_its_status_does_not_count_as_written(cloud, tmp_path):
    from zvdvc.io.pointcloud import write_cell

    store = ResultStore.allocate(tmp_path / "results.zarrvectors", points=cloud, dof=6)
    tile = cloud.read_tile(cloud.cells()[:2], device="cpu")
    store.write_tile(tile, _fake_results(tile, 6))
    other = cloud.read_tile(cloud.cells()[2:3], device="cpu")
    values = {k: v for k, v in _fake_results(other, 6).items() if k != "status"} | {"point_id": other.point_id}
    write_cell(store._level, other.cells[0], other.xyz, values, other.bin_offsets[0])   # killed before the status
    assert store.written_cells() == set(cloud.cells()[:2])


def test_cells_written_after_a_partial_finalize_are_seen(cloud, tmp_path):
    store = ResultStore.allocate(tmp_path / "results.zarrvectors", points=cloud, dof=3)
    cells = cloud.cells()
    first = cloud.read_tile(cells[:10], device="cpu")
    store.write_tile(first, _fake_results(first, 3))
    store.finalize(n_points=len(first.point_id))
    store = ResultStore(store.path, mode="r+")
    store.reopen_for_writes()
    rest = cloud.read_tile(cells[10:], device="cpu")
    ResultStore(store.path, mode="r+").write_tile(rest, _fake_results(rest, 3))
    assert store.written_cells() == set(cells) and len(store.read_all()["point_id"]) == cloud.n_points


def test_the_fingerprint_is_checked_and_older_stores_only_warn(cloud, tmp_path):
    from zvdvc.io.results import _ATTR, _root_attrs, fingerprint_diff

    fp = {"settings": {"search": {"objective": "znssd", "disp_max": 8.0}}, "points": {"digest": "ab"}}
    store = ResultStore.allocate(tmp_path / "results.zarrvectors", points=cloud, dof=6, fingerprint=fp)
    store.check_fingerprint(fp)
    other = {"settings": {"search": {"objective": "ssd", "disp_max": 8.0}}, "points": {"digest": "cd"}}
    assert fingerprint_diff(fp, other) == ["points.digest", "settings.search.objective"]
    with pytest.raises(ValueError, match=r"differs in points.digest, settings.search.objective.*original config"):
        ResultStore(store.path).check_fingerprint(other)
    attrs = _root_attrs(store.path, "r+")
    attrs[_ATTR] = {k: v for k, v in attrs[_ATTR].items() if k != "fingerprint"}   # a store from before fingerprints
    with pytest.warns(UserWarning, match="no run fingerprint"):
        ResultStore(store.path).check_fingerprint(other)
