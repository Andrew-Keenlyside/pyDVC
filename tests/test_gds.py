"""GPUDirect Storage read path (zvdvc.io.gds): read-path choice, and device reads equal to host reads.

The device reads run through kvikio whether or not the machine has GPUDirect Storage (without it,
cuFile falls back to POSIX reads), so they are tested everywhere kvikio and a GPU are present.
"""

import dataclasses

import numpy as np
import pytest

from zvdvc.geometry.box import Box
from zvdvc.io import gds
from zvdvc.io.volume import RawVolume, ZarrVolume, create_ome_zarr, is_padded

SHAPE = (37, 29, 43)                       # (z, y, x): not a multiple of any chunk
rng = np.random.default_rng(3)
U16 = rng.integers(0, 60000, SHAPE, dtype=np.uint16)
F32 = rng.normal(100.0, 20.0, SHAPE).astype(np.float32)

BOXES = [
    Box((5, 4, 6), (20, 18, 30)),          # inside
    Box((0, 0, 0), SHAPE),                 # the whole volume: whole planes, one read
    Box((-3, 10, -5), (12, 35, 20)),       # past three faces: edge padding
    Box((30, 0, 40), (45, 29, 50)),        # past the far faces, whole rows
]


@pytest.fixture(autouse=True)
def _fresh_status(monkeypatch):
    real = gds.status                     # tests may replace it; its cache is cleared either side
    real.cache_clear()
    monkeypatch.delenv(gds.ENV, raising=False)
    yield
    real.cache_clear()


def _fake_status(monkeypatch, **kw):
    base = {"kvikio": "x", "nvcomp": "x", "gds_available": False, "compat_mode_preferred": False, "cufile": "x"}
    monkeypatch.setattr(gds, "status", lambda: {**base, **kw})


# ------------------------------------------------------------------ choice (no GPU needed)


def test_auto_uses_kvikio_only_with_gpudirect_storage(monkeypatch):
    _fake_status(monkeypatch, gds_available=False, why="no nvidia-fs")
    assert gds.resolve("auto") == "host"
    _fake_status(monkeypatch, gds_available=True)
    assert gds.resolve("auto") == "kvikio"
    assert gds.resolve("host") == "host"


def test_the_environment_overrides_auto_only(monkeypatch):
    _fake_status(monkeypatch, gds_available=True)
    monkeypatch.setenv(gds.ENV, "host")
    assert gds.resolve("auto") == "host"
    assert gds.resolve("kvikio") == "kvikio"          # an explicit setting wins over the environment
    monkeypatch.setenv(gds.ENV, "gds")
    with pytest.raises(ValueError, match="ZVDVC_GPU_IO"):
        gds.resolve("auto")


def test_forcing_kvikio_without_it_is_a_clear_error(monkeypatch):
    _fake_status(monkeypatch, kvikio=None, why="kvikio is not installed")
    with pytest.raises(ImportError, match="kvikio is not installed"):
        gds.resolve("kvikio")


def test_the_read_path_is_not_part_of_the_run_fingerprint():
    from zvdvc.config import RunConfig, VolumeSpec
    from zvdvc.pipeline.coordinator import result_settings

    cfg = RunConfig(volumes=VolumeSpec("r.ome.zarr", "d.ome.zarr"), points="p.roi", output="o.zarrvectors")
    kv = dataclasses.replace(cfg, volumes=dataclasses.replace(cfg.volumes, gpu_io="kvikio"))
    assert result_settings(cfg) == result_settings(kv)


def test_gpu_io_is_validated_at_load():
    from zvdvc.config import RunConfig

    d = {"volumes": {"reference": "r.ome.zarr", "deformed": "d.ome.zarr", "gpu_io": "gds"}, "points": "p.roi",
         "output": "o.zarrvectors"}
    with pytest.raises(ValueError, match="gpu_io"):
        RunConfig.from_dict(d)


def test_host_reads_record_their_path(tmp_path):
    path = tmp_path / "v.raw"
    U16.tofile(path)
    vol = RawVolume(path, shape_xyz=SHAPE[::-1], dtype="<u2", gpu_io="host")
    assert vol.io == "host" and vol.read_brick(BOXES[0], device="cpu").io == "host"


# ------------------------------------------------------------------ device reads (GPU + kvikio)

needs_kvikio = pytest.mark.skipif(gds._kvikio() is None, reason="needs kvikio")


def _same_on_both_paths(make, boxes=BOXES, padded=True):
    import cupy as cp

    host, dev = make("host"), make("kvikio")
    assert dev.io == "kvikio", dev.io_note
    for box in boxes:
        a = host.read_brick(box, device="cuda")
        b = dev.read_brick(box, device="cuda")
        assert b.io == "kvikio" and a.io == "host"
        assert b.data.dtype == a.data.dtype and b.data.dtype.isnative
        assert is_padded(b.data) or not padded
        assert b.valid == a.valid
        np.testing.assert_array_equal(cp.asnumpy(b.data), cp.asnumpy(a.data), err_msg=str(box))


@pytest.mark.gpu
@needs_kvikio
@pytest.mark.parametrize("dtype, header", [("<u2", 0), (">u2", 100), ("|u1", 7), ("<f4", 0), (">f4", 32)])
def test_raw_device_reads_equal_host_reads(tmp_path, dtype, header):
    data = (F32 if dtype[1] == "f" else U16 if dtype[1:] == "u2" else (U16 >> 8).astype(np.uint8)).astype(dtype)
    path = tmp_path / "v.raw"
    with open(path, "wb") as fh:
        fh.write(b"\x00" * header)
        fh.write(data.tobytes())
    _same_on_both_paths(lambda io: RawVolume(path, shape_xyz=SHAPE[::-1], dtype=dtype, header_bytes=header, gpu_io=io))


@pytest.mark.gpu
@needs_kvikio
def test_npy_and_mhd_device_reads_equal_host_reads(tmp_path):
    np.save(tmp_path / "v.npy", U16)
    _same_on_both_paths(lambda io: RawVolume(tmp_path / "v.npy", gpu_io=io))
    U16.astype(">u2").tofile(tmp_path / "v.raw")
    (tmp_path / "v.mhd").write_text(
        "ObjectType = Image\nNDims = 3\nDimSize = {} {} {}\nElementType = MET_USHORT\n"
        "ElementByteOrderMSB = True\nElementDataFile = v.raw\n".format(*SHAPE[::-1]))
    _same_on_both_paths(lambda io: RawVolume(tmp_path / "v.mhd", gpu_io=io))


@pytest.mark.gpu
@needs_kvikio
def test_a_fortran_ordered_npy_falls_back_to_the_host(tmp_path):
    np.save(tmp_path / "f.npy", np.asfortranarray(U16))
    vol = RawVolume(tmp_path / "f.npy", gpu_io="kvikio")
    assert vol.io == "host" and "C-ordered" in vol.io_note


@pytest.mark.gpu
@needs_kvikio
@pytest.mark.parametrize("codec", ["zstd", "none"])
@pytest.mark.parametrize("dtype", ["uint16", "float32"])
def test_sharded_zarr_device_reads_equal_host_reads(tmp_path, codec, dtype):
    data = F32 if dtype == "float32" else U16
    arr = create_ome_zarr(tmp_path / "v.ome.zarr", SHAPE, dtype, chunk=8, shard=16, codec=codec)
    arr[...] = data
    _same_on_both_paths(lambda io: ZarrVolume(str(tmp_path / "v.ome.zarr"), gpu_io=io))


@pytest.mark.gpu
@needs_kvikio
def test_unwritten_zarr_chunks_read_as_the_fill_value_on_the_device(tmp_path):
    import zarr
    from zarr.codecs import ZstdCodec

    arr = zarr.create_array(str(tmp_path / "s.zarr"), shape=SHAPE, dtype="uint16", chunks=(8, 8, 8), shards=(16, 16, 16),
                            compressors=ZstdCodec(level=3), fill_value=7, config={"write_empty_chunks": False})
    arr[:10] = U16[:10]                                   # the rest of the volume is never written
    with pytest.warns(UserWarning, match="chunks are stored"):
        _same_on_both_paths(lambda io: ZarrVolume(str(tmp_path / "s.zarr"), gpu_io=io))


@pytest.mark.gpu
@needs_kvikio
def test_an_unsharded_zarr_array_falls_back_to_the_host(tmp_path):
    import zarr

    arr = zarr.create_array(str(tmp_path / "u.zarr"), shape=SHAPE, dtype="uint16", chunks=(8, 8, 8))
    arr[...] = U16
    vol = ZarrVolume(str(tmp_path / "u.zarr"), gpu_io="kvikio")
    assert vol.io == "host" and "not sharded" in vol.io_note


@pytest.mark.gpu
@needs_kvikio
def test_prefiltered_device_reads_equal_host_reads(tmp_path):
    from zvdvc.io.volume import FilteredVolume

    arr = create_ome_zarr(tmp_path / "v.ome.zarr", SHAPE, "uint16", chunk=8, shard=16)
    arr[...] = U16
    _same_on_both_paths(lambda io: FilteredVolume(ZarrVolume(str(tmp_path / "v.ome.zarr"), gpu_io=io), 1.0), BOXES[:2],
                        padded=False)                     # filtered bricks are cropped views, on either path


@pytest.mark.gpu
@needs_kvikio
def test_a_tiled_run_on_the_device_path_equals_the_host_path(tmp_path):
    """Same bricks, same results: the fused engine's output does not depend on how bricks arrive."""
    from zvdvc.config import ClusterSpec, RunConfig, SeedingSpec, SubvolumeSpec, SearchSpec
    from zvdvc.io.results import ResultStore
    from zvdvc.pipeline import coordinator
    from zvdvc.synth.phantoms import default_field, make_case

    shape = (64, 64, 64)
    cfg = RunConfig.from_yaml(make_case(tmp_path / "case", shape_zyx=shape, field=default_field("affine", shape),
                                        spacing=12.0, chunk=32, shard=64,
                                        subvolume=SubvolumeSpec(geometry="sphere", size=16, n_samples=500),
                                        search=SearchSpec(dof=6, disp_max=6.0)))
    out = {}
    for io in ("host", "kvikio"):
        c = dataclasses.replace(
            cfg, volumes=dataclasses.replace(cfg.volumes, gpu_io=io), output=str(tmp_path / f"{io}.zarrvectors"),
            workdir=str(tmp_path / io), seeding=SeedingSpec(strategy="rigid"),
            cluster=dataclasses.replace(cfg.cluster or ClusterSpec(), tile_shape=(32, 32, 32)))
        coordinator.prepare(c, backend="fused")
        stats = coordinator.run(c, backend="fused")
        assert {s.io for s in stats} == {io}
        coordinator.finalize(c)
        out[io] = ResultStore(c.output).read_all()
    order = [np.argsort(out[io]["point_id"]) for io in ("host", "kvikio")]
    for key in ("status", "displacement", "objmin", "params"):
        np.testing.assert_array_equal(out["host"][key][order[0]], out["kvikio"][key][order[1]], err_msg=key)


# ------------------------------------------------------------------ guards against damaged stores


def test_zstd_content_size_reads_the_frame_header():
    from numcodecs import Zstd

    for n in (1, 255, 256, 70_000, 5_000_000):
        frame = Zstd(level=3).encode(np.zeros(n, np.uint8).tobytes())
        assert gds.zstd_content_size(bytes(frame[:18])) == n
    with pytest.raises(ValueError, match="not a zstd frame"):
        gds.zstd_content_size(b"\x00" * 18)


def _one_shard(tmp_path):
    arr = create_ome_zarr(tmp_path / "v.ome.zarr", (16, 16, 16), "uint16", chunk=8, shard=16)
    arr[...] = U16[:16, :16, :16]
    return tmp_path / "v.ome.zarr", tmp_path / "v.ome.zarr" / "0" / "c" / "0" / "0" / "0"


@pytest.mark.gpu
@needs_kvikio
def test_a_damaged_shard_index_is_refused(tmp_path):
    root, shard = _one_shard(tmp_path)
    data = bytearray(shard.read_bytes())
    data[-10] ^= 0xFF                                      # inside the index, before its crc32c
    shard.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="crc32c"):
        ZarrVolume(str(root), gpu_io="kvikio").read_brick(Box((0, 0, 0), (16, 16, 16)), device="cuda")


@pytest.mark.gpu
@needs_kvikio
def test_a_corrupted_chunk_header_is_refused_not_decoded(tmp_path):
    root, shard = _one_shard(tmp_path)
    data = bytearray(shard.read_bytes())
    data[0] ^= 0xFF                                        # the first chunk's zstd magic number
    shard.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="zstd|decode"):
        ZarrVolume(str(root), gpu_io="kvikio").read_brick(Box((0, 0, 0), (16, 16, 16)), device="cuda")


def test_tile_bins_from_the_store_equal_rebinning(tmp_path):
    """With zarr-vectors' fragment index (``read-cells-fragments``) the bins come from the store;
    without it they are recomputed. Either way they are the bins the points were written with."""
    from zvdvc.io.pointcloud import PointCloud, cell_fragments, write_pointcloud_store

    rng = np.random.default_rng(5)
    xyz = rng.uniform(0.0, 64.0, (3000, 3))
    xyz[:40] = np.round(xyz[:40] / 8.0) * 8.0               # points exactly on bin edges
    write_pointcloud_store(tmp_path / "p.zarrvectors", xyz, np.arange(1, 3001), bounds=((0.0,) * 3, (64.0,) * 3),
                           chunk_shape=(16.0,) * 3)
    pc = PointCloud(tmp_path / "p.zarrvectors")
    tile = pc.read_tile(pc.cells(), device="cpu")
    for i, cell in enumerate(tile.cells):
        rows = tile.xyz[tile.cell_offsets[i]:tile.cell_offsets[i + 1]]
        np.testing.assert_array_equal(tile.bin_offsets[i], cell_fragments(rows, cell, pc._chunk, pc._bin)[1])
    import zarr_vectors as zv

    if "read-io-report" in getattr(zv, "FEATURES", ()):
        assert tile.io and all(p == "host" for p in tile.io)
