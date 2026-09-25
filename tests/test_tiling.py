import numpy as np

from pydvc.config import ClusterSpec, RunConfig, SearchSpec, SubvolumeSpec, VolumeSpec
from pydvc.geometry.box import Box
from pydvc.io.pointcloud import PointCloud, write_pointcloud_store
from pydvc.pipeline.tiling import Tile, assign_lpt, plan_tiles


def _contains(outer: Box, inner: Box) -> bool:
    return all(o <= i for o, i in zip(outer.lo, inner.lo)) and all(o >= i for o, i in zip(outer.hi, inner.hi))


def test_box_shape():
    assert Box((10, 20, 30), (20, 30, 45)).shape == (10, 10, 15)


def test_box_grow_and_shift_round_outward():
    b = Box((10, 20, 30), (20, 30, 40))
    assert b.grow(2) == Box((8, 18, 28), (22, 32, 42))
    # shift is (x, y, z); boxes are (z, y, x)
    assert b.shift_xyz((1.5, 0.0, -1.0)) == Box((9, 20, 31), (19, 30, 42))


def test_tiles_partition_cells_and_bricks_cover_the_halo(tmp_path):
    rng = np.random.default_rng(5)
    xyz = rng.uniform(0.0, 512.0, size=(5000, 3))
    store = tmp_path / "points.zarrvectors"
    write_pointcloud_store(store, xyz, np.arange(5000), bounds=((0, 0, 0), (512, 512, 512)), chunk_shape=(64, 64, 64))
    cfg = RunConfig(
        volumes=VolumeSpec(reference="ref", deformed="def"),
        points=str(store),
        output=str(tmp_path / "results.zarrvectors"),
        subvolume=SubvolumeSpec(size=20),
        search=SearchSpec(disp_max=5.0),
        cluster=ClusterSpec(tile_shape=(128, 128, 128)),
    )
    pc = PointCloud(store)

    tiles = plan_tiles(pc, cfg)

    cells = [c for t in tiles for c in t.cells]
    assert len(cells) == len(set(cells)) == len(pc.cells())
    assert sum(t.n_points for t in tiles) == 5000
    for t in tiles:
        assert _contains(t.ref_box, t.points_box.grow(cfg.halo()))


def test_lpt_balances_cost():
    box = Box((0, 0, 0), (1, 1, 1))
    costs = [90, 80, 70, 30, 20, 10, 5, 5]
    tiles = [Tile(i, ((i, 0, 0),), c, box, box, box, float(c)) for i, c in enumerate(costs)]

    shares = assign_lpt(tiles, 2)

    loads = [sum(tiles[i].cost for i in share) for share in shares]
    assert max(loads) - min(loads) <= 10
    assert sorted(i for share in shares for i in share) == list(range(len(costs)))
