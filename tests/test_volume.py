import numpy as np
import pytest

from pydvc.config import VolumeSpec
from pydvc.geometry.box import Box
from pydvc.io.volume import RawVolume, ZarrVolume, create_ome_zarr, open_volume, write_raw

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
    from pydvc.io.volume import convert_to_ome_zarr

    VOL.astype(">u2").tofile(tmp_path / "v.raw")
    np.save(tmp_path / "v.npy", VOL)
    convert_to_ome_zarr(tmp_path / "v.raw", tmp_path / "a.ome.zarr", chunk=4, shard=8, shape_xyz=(8, 10, 12), dtype=">u2")
    convert_to_ome_zarr(tmp_path / "v.npy", tmp_path / "b.ome.zarr", chunk=4, shard=8)
    a, b = ZarrVolume(str(tmp_path / "a.ome.zarr")), ZarrVolume(str(tmp_path / "b.ome.zarr"))
    np.testing.assert_array_equal(a.array[...], VOL)
    np.testing.assert_array_equal(b.array[...], VOL)
    assert a.dtype == np.uint16 and a.dtype.byteorder in ("=", "<", "|")
