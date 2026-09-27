"""Strong and weak scaling on CPU workers: structure, efficiency, tile choice and the too-few-tiles warning."""

import dataclasses

import pytest

from pydvc.bench.scaling import scaling
from pydvc.config import ClusterSpec, RunConfig, SearchSpec, SeedingSpec, SubvolumeSpec
from pydvc.synth.phantoms import default_field, make_case


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    pytest.importorskip("numba")
    root = tmp_path_factory.mktemp("scaling")
    shape = (64, 64, 64)
    c = RunConfig.from_yaml(make_case(root / "c", shape_zyx=shape, field=default_field("affine", shape), spacing=8.0,
                                      chunk=32, shard=64, subvolume=SubvolumeSpec(geometry="sphere", size=12, n_samples=300),
                                      search=SearchSpec(dof=6, disp_max=6.0)))
    return dataclasses.replace(c, cluster=ClusterSpec(tile_shape=(32, 32, 32)), seeding=SeedingSpec(strategy="rigid"))


def test_strong_scaling(cfg, tmp_path):
    res = scaling(cfg, [1, 2], backend="cpu", out_dir=tmp_path, mode="strong", progress=lambda m: None)
    r1, r2 = res["rows"]
    assert r1["points"] == r2["points"] > 0 and r1["efficiency"] == pytest.approx(1.0)
    assert r2["timeline"]["workers"] == 2 and 0 < r2["utilisation"] <= 1
    assert res["warnings"] == []                                   # 8 tiles: enough for 2 devices


def test_weak_scaling_uses_the_fullest_tiles(cfg, tmp_path):
    res = scaling(cfg, [1, 2], backend="cpu", out_dir=tmp_path, mode="weak", tiles_per_device=2, progress=lambda m: None)
    r1, r2 = res["rows"]
    assert (r1["tiles"], r2["tiles"]) == (2, 4) and r2["points"] > r1["points"]
    assert r1["efficiency"] == pytest.approx(1.0)


def test_too_few_tiles_are_flagged(cfg, tmp_path):
    big = dataclasses.replace(cfg, cluster=ClusterSpec(tile_shape=(64, 64, 64)))      # a single tile
    res = scaling(big, [1], backend="cpu", out_dir=tmp_path, progress=lambda m: None)
    assert res["warnings"] and "fewer than" in res["warnings"][0]
