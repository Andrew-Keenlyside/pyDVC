# Image volumes

This page specifies how zvDVC reads the reference and deformed image volumes:
the coordinate conventions, the recommended OME-Zarr layout, the flat-file
inputs it accepts, `zvdvc convert`, and how a tile's *bricks* are cut from the
volumes, padded and checked. It is for users preparing data and for anyone
reasoning about memory, I/O or points near the edge of the image. Source:
[`io/volume.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/io/volume.py),
[`geometry/box.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/geometry/box.py),
[`pipeline/tiling.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/pipeline/tiling.py).

## Terms

Volume
: A 3-D greyscale image stored as an array indexed `[z, y, x]`. A run has two:
  the reference and the deformed volume, of the same shape.

Point space
: Coordinates $(x, y, z)$ in voxels, used for points, displacements and
  sample positions. Voxel `[k, j, i]` of an array sits at point $(i, j, k)$.

Box
: An integer, half-open voxel range `[lo, hi)` in array order $(z, y, x)$
  (`zvdvc.geometry.box.Box`).

Tile
: A block of whole point-cloud chunks: the unit of work handed to one GPU
  ({doc}`points_store`).

Brick
: The dense box of one volume that a tile needs: the tile's points plus a
  halo. Each tile has a reference brick and a deformed brick
  (`zvdvc.io.volume.Brick`: `data`, the requested `box`, and its `valid` part).

Halo
: The margin, in voxels, added around a tile's points so every sample a point
  can reach lies inside the brick.

Native dtype
: The volume's stored element type (u8, u16, f32, ...). Bricks keep it on the
  device.

---

## Introduction

zvDVC never loads a whole volume in the tiled pipeline. For each tile it reads
two bricks, one per volume, once, and solves every point of the tile against
them. The reference brick covers the tile's points plus the halo. The
deformed brick is the same box moved by the tile's median seed displacement
and grown by the spread of seeds, so a large rigid offset moves the brick
instead of fattening it.

Where a brick extends past the volume, the missing voxels are filled by edge
padding, and the brick records which part is real. A point whose samples reach
into the padding is reported as `RANGE_FAIL` rather than correlated against
invented data.

Large inputs should be OME-Zarr v3 with sharding. Flat CCPi/iDVC inputs
(`.raw`, `.mhd`, `.npy`) are read directly through `numpy.memmap`, which
suits development and the CCPi parity case; `zvdvc convert` turns them, or a
TIFF stack, into OME-Zarr once.

---

## Technical reference

### Coordinate conventions

| quantity | order | example |
|---|---|---|
| points, displacements, `rigid_trans`, `start_point`, template offsets | $(x, y, z)$ | `search.rigid_trans: [34.0, 4.0, 0.0]` |
| array shapes, boxes, `cluster.tile_shape` | $(z, y, x)$ | `tile_shape: [1260, 1257, 1520]` |
| `volumes.raw_shape_xyz` (CCPi `vol_wide`, `vol_high`, `vol_tall`) | $(x, y, z)$ | `[1520, 1257, 1260]` |

* **Voxel $i$ is at coordinate $i$.** A brick's `data[0, 0, 0]` sits at point
  $(\text{box.lo}_x, \text{box.lo}_y, \text{box.lo}_z)$ (`Brick.origin_xyz`).
  Interpolation at $x$ uses voxels around $\lfloor x\rfloor$ ({doc}`method`).
* **x is fastest in memory.** Arrays are C-ordered `[z, y, x]`, which is also
  how CCPi reads a flat file.
* **Around points.** `Box.around_points(min, max)` is the smallest box holding
  every voxel whose coordinate lies in `[min, max]`: $\text{lo} = \lceil\min\rceil$,
  $\text{hi} = \lfloor\max\rfloor + 1$ per axis.

### Supported inputs

`zvdvc.io.volume.open_volume` picks the reader from the path:

| input | reader | how it is recognised | needs |
|---|---|---|---|
| OME-Zarr group or Zarr array | `ZarrVolume` | suffix `.zarr` (including `.ome.zarr`), or a directory with `zarr.json` | a 3-D array; in a group, `volumes.array_path` (default `"0"`). A missing level is an error that lists the levels present. |
| NumPy `.npy` | `RawVolume` (`np.load(..., mmap_mode="r")`) | suffix `.npy` | a 3-D `[z, y, x]` array |
| MetaImage `.mhd` | `RawVolume` | suffix `.mhd` | one uncompressed, single-channel 3-D image in one detached data file: `ElementDataFile` not `LOCAL` (`.mha`), `LIST` or a file pattern; no `CompressedData = True`; `ElementNumberOfChannels` 1; `HeaderSize` $\ge 0$; element types `MET_UCHAR`, `MET_CHAR`, `MET_USHORT`, `MET_SHORT`, `MET_UINT`, `MET_INT`, `MET_FLOAT`, `MET_DOUBLE`; byte order from `BinaryDataByteOrderMSB` / `ElementByteOrderMSB` |
| flat raw | `RawVolume` (`np.memmap`) | any other suffix | `volumes.raw_shape_xyz`, `volumes.raw_dtype`, optional `volumes.raw_header_bytes` |
| TIFF stack | not read directly | | convert with `zvdvc convert` (needs `tifffile`, the `zvdvc[tiff]` extra) |

Opening a volume checks what can be checked cheaply:

* **Raw and `.mhd` size.** The data file must hold exactly the header plus
  the voxels. A smaller file is truncated; a larger one is misdescribed. The
  error gives both sizes and names the settings to check: the bit depth
  (`raw_dtype`, CCPi `vol_bit_depth`), the dimensions (`raw_shape_xyz`,
  `vol_wide`/`vol_high`/`vol_tall`) and the header length
  (`raw_header_bytes`, `vol_hdr_lngth`), or `DimSize`, `ElementType` and
  `HeaderSize` of a `.mhd`.
* **Unstored Zarr chunks.** An array with fewer stored chunks than it has
  **warns**; missing chunks read as the fill value. zarr leaves chunks that
  hold only the fill value unwritten by default, so an array from another tool
  can be complete with chunks missing, but a truncated copy looks the same.
  `zvdvc convert` stores every chunk, so a missing chunk in its output means
  an incomplete copy.

Two checks happen as bricks are read, where the data is already in memory:

* **Byte order.** Every reader returns bricks in the host's byte order, so
  big-endian data reaches every engine as native data, and the reader's
  `dtype` is the native type.
* **Non-finite voxels.** A float brick holding NaN or inf is an error. A NaN
  would otherwise spread through the interpolation to every point whose
  subvolume touches it. Checking at open time would read the whole volume.

When `volumes.prefilter_sigma` $> 0$ the reader is wrapped in
`FilteredVolume`, which Gaussian-filters each brick ({doc}`method`,
"Prefilter"). `zvdvc plan` refuses a reference and deformed volume of
different shapes.

### Recommended OME-Zarr layout

| setting | recommendation | why |
|---|---|---|
| format | OME-Zarr v0.5 on Zarr v3, one array per level, `dimension_names` `("z", "y", "x")` | what `zvdvc convert` writes |
| chunks | 128³ | the `zvdvc convert` default |
| shards | aligned to `cluster.tile_shape` (default 1024³) | a brick read touches only a handful of shards |
| compression | zstd | level 3 in `zvdvc convert` |
| dtype | the scanner's native type (u8 or u16) | bricks stay in it on the device |
| pyramid | a multiscale pyramid, level 1 downsampled by 2 | for the planned pyramid-level coarse seeding pass (M5); `zvdvc convert` writes level `"0"` only today |

The brick read path implemented today is host decode (zarr-python reads a
shard's chunks concurrently) into pinned memory, then one host-to-device copy
per brick. Reading through kvikio/GPUDirect Storage or zarr-python's GPU
buffers is planned and has **not** been implemented or measured; see
{doc}`/how_to/gpudirect_storage`.

### `zvdvc convert`

```bash
zvdvc convert SRC DST [--chunk 128] [--shard 1024] [--shape-xyz X Y Z] [--dtype DTYPE] [--header BYTES] [--overwrite]
```

`convert_to_ome_zarr` writes `DST` as an OME-Zarr v0.5 group with one sharded,
zstd-compressed level `"0"` and `multiscales` metadata (unit scale). It
streams slabs of one shard's depth in $z$, so memory stays bounded for inputs
of 100 GB or more. Every chunk is stored, even an empty one. The output is
written next to `DST` and renamed into place at the end, so an interrupted
conversion leaves no partial `DST`. An existing `DST` is an error unless
`--overwrite`, and even then only an existing Zarr store is replaced (the old
one is removed once the new one is complete). `DST` may not be `SRC`, contain
it or lie inside it.

| option | default | meaning |
|---|---|---|
| `--chunk` | 128 | chunk edge, voxels (capped at the volume's size per axis) |
| `--shard` | 1024 | shard edge, voxels; rounded up to a multiple of the chunk and never larger than the volume needs |
| `--shape-xyz X Y Z` | | `.raw` only: volume size |
| `--dtype` | | `.raw` only: numpy dtype, e.g. `'<u2'` or `'\|u1'` |
| `--header` | 0 | `.raw` only: header bytes to skip |
| `--overwrite` | off | replace an existing Zarr store at `DST` |

`SRC` may be `.raw`, `.mhd`, `.npy`, `.tif`/`.tiff` or an existing Zarr array
(its unstored chunks are written as the fill value). Input in non-native byte
order is written in native order. For example, a
headerless big-endian 16-bit volume of 1520 × 1257 × 1260 voxels:

```bash
zvdvc convert scan.raw scan.ome.zarr --shape-xyz 1520 1257 1260 --dtype '>u2'
```

The reverse, `zvdvc.io.volume.write_raw(source, path)`, streams a volume to a
headerless little-endian flat file, x fastest, as CCPi reads it.

### Bricks

A tile's bricks are planned once, from point positions only
(`plan_tiles`; the volumes are not read):

1. **Points box.** `Box.around_points` of the tile's point positions.
2. **Reference box.** The points box grown by the halo $h$ on every side
   (rounded outward):

   $$
   h = e + \texttt{disp\_max} + 2,
   $$

   where $e$ is the subvolume's largest possible sample distance: half the
   cube diagonal, $\tfrac12\,\texttt{size}\,\lVert\texttt{aspect}\rVert_2$, or
   the sphere's largest semi-axis, $\tfrac12\,\texttt{size}\,\max(\texttt{aspect})$.
   Rotations can turn any sample along any axis, hence the largest extent. The
   $+2$ covers the Catmull-Rom stencil (`RunConfig.halo()`).
3. **Deformed box.** The reference box shifted by the median seed of the
   tile's points (`Box.shift_xyz`, rounded outward), then grown per axis by
   $\lceil\max_i\lvert\text{seed}_i - \text{median}\rvert\rceil$, the spread of
   seeds in the tile. Seeds are `rigid_trans`, or, with
   `seeding.strategy: coarse`, the seed field in `<workdir>/seeds.npz`
   (another strategy ignores a `seeds.npz` left in the work directory).

These boxes are stored in `plan.json` (`points_box`, `ref_box`, `def_box`,
each `{"lo": [z, y, x], "hi": [z, y, x]}`). Read amplification per volume is
$((T + 2h)/T)^3$ for tile edge $T$; for example 1.23 for $T = 1024$,
$h = 36$ ({doc}`/ARCHITECTURE` §4).

**Reading.** `read_brick(box)` intersects the box with the volume. The inside
part is read; whatever lies outside is filled by edge padding (`np.pad`, mode
`edge`). The brick records `box` (what was asked for) and `valid` (the part
inside the volume). A brick that does not overlap the volume at all is an
error.

**Validity and `RANGE_FAIL`.** The solver is given `valid` as a half-open
range in point space. A sample whose interpolation stencil is not wholly
inside it counts as outside, and any outside sample makes the point
`RANGE_FAIL`, whether in the reference samples, a Gauss–Newton iteration, or
the final objective. CCPi has no such check and interpolates wrapped rows or
unset memory for subvolumes that leave the image
([case A report](../benchmarks/2026-09-26-case-A-real.md)).

**Memory check.** `zvdvc plan` estimates device memory as
(largest tile's reference + deformed brick bytes, native dtype) ×
`prefetch_depth`, plus batch scratch, and refuses a plan that exceeds
`gpu_memory_fraction` of the device ({doc}`/how_to/choose_tiles_and_batches`).

### Native dtypes

Bricks land on the device in the volume's dtype, in native byte order; the kernels convert to
float32 in registers. The fused CUDA kernels are compiled for `uint8`,
`uint16` and `float32` bricks; a brick of any other dtype is converted to
float32 when it is prepared. The CPU (numba) engine reads the native dtype
too. With a prefilter, the filtered brick keeps the volume's dtype (integer
data rounded to the nearest level and clipped), so filtering does not change
brick memory.

### `PAD_BYTES`

`zvdvc.io.volume.PAD_BYTES = 16`. Device bricks are allocated at the start of
a fresh allocation (at least 16-byte aligned) with 16 spare bytes after the
last voxel (`padded_empty`, `to_device`). The u8 tricubic kernel reads each
stencil row as two aligned 32-bit words, which can reach up to 4 bytes past
the row's last tap; the padding keeps those reads inside the allocation.
`is_padded` checks a device array for this layout, and the fused engine
copies a brick that lacks it.
