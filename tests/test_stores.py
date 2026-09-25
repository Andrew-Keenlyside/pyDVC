"""M3: zarr-vectors point-cloud and results stores (three-phase writes, reads, validation)."""

import numpy as np
import pytest

from pydvc.io.ccpi import write_roi
from pydvc.io.pointcloud import PointCloud, bin_index, import_points, write_pointcloud_store
from pydvc.io.results import RESULT_ATTRIBUTES, ResultStore

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
    assert set(back) == {"xyz"} | set(RESULT_ATTRIBUTES)
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
    with pytest.raises(ValueError, match="not a pyDVC results store"):
        ResultStore(cloud.path)
