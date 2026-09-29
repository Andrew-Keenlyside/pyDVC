import numpy as np
import pytest

from zvdvc.config import VolumeSpec
from zvdvc.geometry.box import Box
from zvdvc.io.volume import RawVolume, ZarrVolume, create_ome_zarr, open_volume, write_raw

VOL = (np.arange(12 * 10 * 8) * 7 % 4096).astype("<u2").reshape(12, 10, 8)     # (z, y, x)


def test_npy_raw_and_mhd_read_the_same_data(tmp_path):
    np.save(tmp_path / "v.npy", VOL)
    (tmp_path / "v.raw").write_bytes(b"\0" * 16 + VOL.tobytes())
    (tmp_path / "v.mhd").write_text(
        "ObjectType = Image\nNDims = 3\nDimSize = 8 10 12\nElementType = MET_USHORT\n"
        "HeaderSize = 16\nBinaryDataByteOrderMSB = False\nElementDataFile = v.raw\n"
    )
    for vol in (
        RawVolume(tmp_path / "v.npy"),
        RawVolume(tmp_path / "v.raw", shape_xyz=(8, 10, 12), dtype="<u2", header_bytes=16),
        RawVolume(tmp_path / "v.mhd"),
    ):
        assert vol.shape == (12, 10, 8)
        np.testing.assert_array_equal(vol.read_brick(Box((0, 0, 0), (12, 10, 8)), device="cpu").data, VOL)


def test_raw_needs_shape_and_dtype(tmp_path):
    (tmp_path / "v.raw").write_bytes(VOL.tobytes())
    with pytest.raises(ValueError, match="raw_shape_xyz"):
        RawVolume(tmp_path / "v.raw")


def test_brick_outside_the_volume_is_edge_padded(tmp_path):
    np.save(tmp_path / "v.npy", VOL)
    brick = RawVolume(tmp_path / "v.npy").read_brick(Box((-2, 3, 5), (4, 7, 11)), device="cpu")
    assert brick.data.shape == (6, 4, 6)
    assert brick.valid == Box((0, 3, 5), (4, 7, 8))
    np.testing.assert_array_equal(brick.data[2:, :, :3], VOL[0:4, 3:7, 5:8])
    np.testing.assert_array_equal(brick.data[0], brick.data[2])              # z padding repeats the edge
    np.testing.assert_array_equal(brick.data[:, :, 5], brick.data[:, :, 2])
    assert brick.origin_xyz == (5.0, 3.0, -2.0)
    assert brick.valid_lo_hi_xyz == ((5, 3, 0), (8, 7, 4))


def test_ome_zarr_round_trip_and_open_volume(tmp_path):
    arr = create_ome_zarr(tmp_path / "v.ome.zarr", VOL.shape, VOL.dtype, chunk=4, shard=8)
    arr[...] = VOL
    spec = VolumeSpec(reference=str(tmp_path / "v.ome.zarr"), deformed=str(tmp_path / "v.ome.zarr"))
    vol = open_volume(spec, "reference")
    assert isinstance(vol, ZarrVolume) and vol.shape == VOL.shape
    np.testing.assert_array_equal(vol.read_brick(Box((1, 2, 3), (9, 8, 7)), device="cpu").data, VOL[1:9, 2:8, 3:7])
    write_raw(vol, tmp_path / "v.raw", slab=5)
    np.testing.assert_array_equal(np.fromfile(tmp_path / "v.raw", dtype="<u2").reshape(VOL.shape), VOL)


def test_convert_to_ome_zarr_from_big_endian_raw_and_npy(tmp_path):
    from zvdvc.io.volume import convert_to_ome_zarr

    VOL.astype(">u2").tofile(tmp_path / "v.raw")
    np.save(tmp_path / "v.npy", VOL)
    convert_to_ome_zarr(tmp_path / "v.raw", tmp_path / "a.ome.zarr", chunk=4, shard=8, shape_xyz=(8, 10, 12), dtype=">u2")
    convert_to_ome_zarr(tmp_path / "v.npy", tmp_path / "b.ome.zarr", chunk=4, shard=8)
    a, b = ZarrVolume(str(tmp_path / "a.ome.zarr")), ZarrVolume(str(tmp_path / "b.ome.zarr"))
    np.testing.assert_array_equal(a.array[...], VOL)
    np.testing.assert_array_equal(b.array[...], VOL)
    assert a.dtype == np.uint16 and a.dtype.byteorder in ("=", "<", "|")


# --------------------------------------------------------------------------- byte order


def test_big_endian_bricks_are_native(tmp_path):
    VOL.astype(">u2").tofile(tmp_path / "v.raw")
    (tmp_path / "v.mhd").write_text("NDims = 3\nDimSize = 8 10 12\nElementType = MET_USHORT\n"
                                    "BinaryDataByteOrderMSB = True\nElementDataFile = v.raw\n")
    for vol in (RawVolume(tmp_path / "v.raw", shape_xyz=(8, 10, 12), dtype=">u2"), RawVolume(tmp_path / "v.mhd")):
        assert vol.dtype == np.uint16 and vol.dtype.isnative
        brick = vol.read_brick(Box((-1, 0, 0), (12, 10, 8)), device="cpu")
        assert brick.data.dtype.isnative and brick.data.flags.c_contiguous
        np.testing.assert_array_equal(brick.data[1:], VOL)          # (and one edge-padded slice)


@pytest.fixture(scope="module")
def endian_case(tmp_path_factory):
    """The same pair as little- and big-endian raw files, with configs that differ only in byte order."""
    import dataclasses

    from zvdvc.config import ClusterSpec, RunConfig, SearchSpec, SeedingSpec, SubvolumeSpec
    from zvdvc.synth.phantoms import default_field, make_case

    root = tmp_path_factory.mktemp("endian")
    shape = (48, 48, 48)
    cfg = RunConfig.from_yaml(make_case(root / "c", shape_zyx=shape, field=default_field("affine", shape), spacing=12.0,
                                        chunk=24, shard=48, subvolume=SubvolumeSpec(geometry="sphere", size=16, n_samples=400),
                                        search=SearchSpec(dof=6, disp_max=6.0), workers=1))
    configs = {}
    for order in ("<", ">"):
        paths = {}
        for which in ("reference", "deformed"):
            p = root / f"{which}{'le' if order == '<' else 'be'}.raw"
            write_raw(open_volume(cfg.volumes, which), p)
            if order == ">":
                np.fromfile(p, dtype="<u2").astype(">u2").tofile(p)
            paths[which] = str(p)
        tag = "le" if order == "<" else "be"
        configs[tag] = dataclasses.replace(
            cfg, volumes=VolumeSpec(**paths, raw_shape_xyz=(48, 48, 48), raw_dtype=f"{order}u2"),
            output=str(root / f"{tag}.zarrvectors"), workdir=str(root / f"run_{tag}"),
            seeding=SeedingSpec(strategy="rigid"), cluster=ClusterSpec(tile_shape=(24, 24, 24), prefetch_depth=1))
    return configs


def _engines():
    try:
        import numba  # noqa: F401

        return ["numpy", "cpu"]
    except ImportError:
        return ["numpy"]


@pytest.mark.parametrize("backend", _engines())
def test_big_endian_volumes_solve_as_little_endian_in_memory(endian_case, backend):
    from zvdvc.pipeline.inmemory import load_points, solve_in_memory

    pid, xyz = load_points(endian_case["le"])
    le = solve_in_memory(endian_case["le"], pid, xyz, backend=backend)
    be = solve_in_memory(endian_case["be"], pid, xyz, backend=backend)
    assert (le.status == 0).mean() > 0.9
    np.testing.assert_array_equal(be.status, le.status)
    np.testing.assert_array_equal(be.params, le.params)


@pytest.mark.parametrize("backend", _engines())
def test_big_endian_volumes_solve_as_little_endian_tiled(endian_case, backend):
    import dataclasses

    from zvdvc.io.results import ResultStore
    from zvdvc.pipeline import coordinator

    got = {}
    for tag in ("le", "be"):
        cfg = dataclasses.replace(endian_case[tag], output=endian_case[tag].output.replace(".zarr", f"_{backend}.zarr"),
                                  workdir=endian_case[tag].workdir + f"_{backend}")
        coordinator.prepare(cfg, backend=backend)
        coordinator.run(cfg, backend=backend)
        got[tag] = ResultStore(cfg.output).read_all()
    assert (got["le"]["status"] == 0).mean() > 0.9
    for key in ("point_id", "status", "params"):
        np.testing.assert_array_equal(got["be"][key], got["le"][key])


# --------------------------------------------------------------------------- input checks


def test_raw_file_size_must_match_the_description(tmp_path):
    VOL.tofile(tmp_path / "v.raw")
    with pytest.raises(ValueError, match=r"smaller than the .* \(truncated\?\)"):
        RawVolume(tmp_path / "v.raw", shape_xyz=(8, 10, 13), dtype="<u2")
    with pytest.raises(ValueError, match=r"2 times as large\); check the bit depth \(vol_bit_depth"):
        RawVolume(tmp_path / "v.raw", shape_xyz=(8, 10, 12), dtype="|u1")        # 16-bit data read as 8-bit
    with pytest.raises(ValueError, match=r"vol_hdr_lngth"):
        RawVolume(tmp_path / "v.raw", shape_xyz=(8, 10, 12), dtype="<u2", header_bytes=16)
    with pytest.raises(FileNotFoundError, match="volume file not found"):
        RawVolume(tmp_path / "nope.raw", shape_xyz=(8, 10, 12), dtype="<u2")
    (tmp_path / "v.mhd").write_text("NDims = 3\nDimSize = 8 10 6\nElementType = MET_USHORT\nElementDataFile = v.raw\n")
    with pytest.raises(ValueError, match=r"check DimSize, ElementType and HeaderSize in .*v.mhd"):
        RawVolume(tmp_path / "v.mhd")


@pytest.mark.parametrize("extra, match", [
    ("CompressedData = True\n", "CompressedData = True is not supported"),
    ("ElementNumberOfChannels = 3\n", "only single-channel"),
    ("HeaderSize = -1\n", "HeaderSize = -1"),
])
def test_unsupported_metaimage_headers_are_rejected(tmp_path, extra, match):
    VOL.tofile(tmp_path / "v.raw")
    (tmp_path / "v.mhd").write_text("NDims = 3\nDimSize = 8 10 12\nElementType = MET_USHORT\n" + extra + "ElementDataFile = v.raw\n")
    with pytest.raises(ValueError, match=match):
        RawVolume(tmp_path / "v.mhd")


@pytest.mark.parametrize("header, match", [
    ("NDims = 3\nElementType = MET_USHORT\nElementDataFile = v.raw\n", "missing DimSize"),
    ("NDims = 3\nDimSize = 8 10 12\nElementDataFile = v.raw\n", "missing ElementType"),
    ("NDims = 3\nDimSize = 8 10 12\nElementType = MET_USHORT\n", "missing ElementDataFile"),
    ("NDims = 3\nDimSize = 8 10 12\nElementType = MET_UINT16\nElementDataFile = v.raw\n", "ElementType 'MET_UINT16' is not supported"),
    ("NDims = 2\nDimSize = 8 10\nElementType = MET_USHORT\nElementDataFile = v.raw\n", "expected a 3-D image"),
    ("NDims = 3\nDimSize = 8 10 12\nElementType = MET_USHORT\nElementDataFile = LIST\n", "only one detached data file"),
    ("NDims = 3\nDimSize = 8 10 12\nElementType = MET_USHORT\nElementDataFile = LOCAL\n", "only one detached data file"),
    ("NDims = 3\nDimSize = 8 10 12\nElementType = MET_USHORT\nElementDataFile = s%03d.raw 1 12 1\n", "file pattern"),
])
def test_incomplete_metaimage_headers_are_clear_errors(tmp_path, header, match):
    VOL.tofile(tmp_path / "v.raw")
    (tmp_path / "v.mhd").write_text(header)
    with pytest.raises(ValueError, match=match):
        RawVolume(tmp_path / "v.mhd")


def test_missing_ome_zarr_level_names_the_available_ones(tmp_path):
    arr = create_ome_zarr(tmp_path / "v.ome.zarr", VOL.shape, VOL.dtype, chunk=4, shard=8)
    arr[...] = VOL
    with pytest.raises(ValueError, match=r"no array \(level\) '1' here; available: \['0'\]"):
        ZarrVolume(str(tmp_path / "v.ome.zarr"), "1")


def test_ome_zarr_with_missing_chunks_warns(tmp_path):
    import warnings

    import zarr

    from zvdvc.io.volume import convert_to_ome_zarr

    arr = create_ome_zarr(tmp_path / "part.ome.zarr", VOL.shape, VOL.dtype, chunk=4, shard=8)
    arr[:4] = VOL[:4]                                  # an interrupted write, or chunks left out as empty
    with pytest.warns(UserWarning, match=r"only \d+ of \d+ chunks are stored"):
        ZarrVolume(str(tmp_path / "part.ome.zarr"))
    zeros = create_ome_zarr(tmp_path / "zeros.ome.zarr", VOL.shape, VOL.dtype, chunk=4, shard=8)
    zeros[...] = 0                                     # empty chunks are stored too, so this one is complete
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ZarrVolume(str(tmp_path / "zeros.ome.zarr")).shape == VOL.shape
    # an array another tool wrote sparsely converts, missing chunks becoming the fill value
    sparse = zarr.open_array(str(tmp_path / "sparse.zarr"), mode="w", shape=VOL.shape, chunks=(4, 5, 4), dtype="<u2", fill_value=0)
    sparse[:4] = VOL[:4]
    convert_to_ome_zarr(tmp_path / "sparse.zarr", tmp_path / "full.ome.zarr", chunk=4, shard=8)
    full = ZarrVolume(str(tmp_path / "full.ome.zarr"))
    np.testing.assert_array_equal(full.array[:4], VOL[:4])
    assert not full.array[4:].any()


def test_non_finite_float_voxels_are_an_error(tmp_path):
    data = VOL.astype(np.float32)
    data[5, 5, 5] = np.nan
    data[6, 0, 0] = np.inf
    np.save(tmp_path / "f.npy", data)
    vol = RawVolume(tmp_path / "f.npy")
    with pytest.raises(ValueError, match=r"f.npy: 2 non-finite voxels \(NaN/inf\)"):
        vol.read_brick(Box((0, 0, 0), VOL.shape), device="cpu")
    ok = vol.read_brick(Box((0, 0, 0), (5, 10, 8)), device="cpu")     # a brick away from them is fine
    np.testing.assert_array_equal(ok.data, VOL[:5].astype(np.float32))


def test_convert_refuses_to_overwrite(tmp_path):
    from zvdvc.io.volume import convert_to_ome_zarr

    np.save(tmp_path / "v.npy", VOL)
    arr = create_ome_zarr(tmp_path / "src.ome.zarr", VOL.shape, VOL.dtype, chunk=4, shard=8)
    arr[...] = VOL
    with pytest.raises(ValueError, match="would overwrite the input"):
        convert_to_ome_zarr(tmp_path / "src.ome.zarr", tmp_path / "src.ome.zarr", overwrite=True)
    with pytest.raises(ValueError, match="would overwrite the input"):
        convert_to_ome_zarr(tmp_path / "src.ome.zarr", tmp_path / "src.ome.zarr" / "inner.ome.zarr")
    with pytest.raises(ValueError, match="would overwrite the input"):
        convert_to_ome_zarr(tmp_path / "v.npy", tmp_path, overwrite=True)
    np.testing.assert_array_equal(ZarrVolume(str(tmp_path / "src.ome.zarr")).array[...], VOL)      # untouched

    convert_to_ome_zarr(tmp_path / "v.npy", tmp_path / "out.ome.zarr", chunk=4, shard=8)
    with pytest.raises(FileExistsError, match="already exists"):
        convert_to_ome_zarr(tmp_path / "v.npy", tmp_path / "out.ome.zarr")
    np.save(tmp_path / "w.npy", VOL[::-1].copy())
    convert_to_ome_zarr(tmp_path / "w.npy", tmp_path / "out.ome.zarr", chunk=4, shard=8, overwrite=True)
    np.testing.assert_array_equal(ZarrVolume(str(tmp_path / "out.ome.zarr")).array[...], VOL[::-1])
    (tmp_path / "notzarr").mkdir()
    (tmp_path / "notzarr" / "keep.txt").write_text("data")
    with pytest.raises(FileExistsError, match="not a Zarr store"):
        convert_to_ome_zarr(tmp_path / "v.npy", tmp_path / "notzarr", overwrite=True)
    assert (tmp_path / "notzarr" / "keep.txt").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        create_ome_zarr(tmp_path / "out.ome.zarr", VOL.shape, VOL.dtype)
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


def test_interrupted_convert_leaves_no_output(tmp_path, monkeypatch):
    from zvdvc.io import volume
    from zvdvc.io.volume import convert_to_ome_zarr

    class Failing:
        def __init__(self, path, **kw):
            self.array, self.shape, self.dtype = self, VOL.shape, VOL.dtype

        def __getitem__(self, key):
            if key.start >= 8:
                raise OSError("read error")
            return VOL[key]

    monkeypatch.setattr(volume, "RawVolume", Failing)
    np.save(tmp_path / "v.npy", VOL)
    with pytest.raises(OSError, match="read error"):
        convert_to_ome_zarr(tmp_path / "v.npy", tmp_path / "out.ome.zarr", chunk=4, shard=8)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v.npy"]
