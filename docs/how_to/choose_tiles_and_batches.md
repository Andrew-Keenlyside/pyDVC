# Choose tiles, batches and chunk sizes

This guide explains how to set the sizes that decide how a zvDVC run uses
memory and I/O: the tile shape, the halo, the batch size, the point store's
chunk shape, and the OME-Zarr chunk and shard shape of the volumes. It is for
users moving from a small test to real data, or to a new GPU. Everything
here follows the code in
[`pipeline/tiling.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/pipeline/tiling.py),
[`pipeline/batching.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/pipeline/batching.py)
and [`config.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/config.py).

## Terms

Cell
: One chunk of the zarr-vectors point store, a box of `chunk_shape` voxels.
  The results store uses the same cells.

Tile
: A block of whole cells, `cluster.tile_shape` voxels. The unit of work a GPU
  takes from the queue, and the unit of resume.

Halo
: The margin, in voxels, added around a tile's points so that every
  subvolume, at every displacement it may reach, stays inside the brick.

Brick
: The region of one volume a tile needs: its points' bounding box plus the
  halo. Each tile reads one reference brick and one deformed brick.

Batch
: The points of a tile solved together in one kernel launch.

---

## The quick version

* Start from the defaults: `tile_shape: [1024, 1024, 1024]`,
  `prefetch_depth: 2`, `batch_points: null` (sized automatically), and
  OME-Zarr volumes with 128³ chunks in 1024³ shards. These are sized for an
  80 GB H100 with u16 volumes.
* Run `zvdvc plan`. It prints the memory check and stops with a
  `MemoryError` if the tiles do not fit. Shrink `tile_shape` or
  `prefetch_depth` until they do.
* Give every GPU several tiles: the scaling campaign aims for at least 4
  tiles per GPU ({doc}`/CLUSTER`).
* Keep `tile_shape` a whole multiple of the point store's chunk.

## How the sizes relate

```text
 point store cell (chunk_shape)        e.g. 256³
   └─ tile = k × k × k cells           e.g. 4³ cells = 1024³ (cluster.tile_shape)
        └─ brick = tile's points + halo on every side, per volume
             └─ batches of points solved against the bricks in GPU memory
```

### The halo

{py:meth}`zvdvc.config.RunConfig.halo` is

$$
h = r + d_\text{max} + 2
$$

where $r$ is the subvolume's farthest sample from its centre (for a sphere,
half the size times the largest aspect ratio; for a cube, the half-diagonal,
since 6- and 12-DOF warps can rotate it), $d_\text{max}$ is
`search.disp_max`, and 2 voxels cover the tricubic stencil.

| case | subvolume | `disp_max` | halo |
|---|---|---|---|
| synthetic case S | sphere 32 | 8 | 26 |
| scenario B ([`configs/large_8xh100.yaml`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/configs/large_8xh100.yaml)) | sphere 48 | 10 | 36 |
| case A (iDVC example) | sphere 80 | 38 | 80 |

The deformed brick is the reference brick shifted by the tile's median seed
and grown by the spread of the seeds within the tile. A large rigid offset
(case A's `rigid_trans` is 34, 4, 0) therefore moves the brick instead of
enlarging it. `disp_max` is measured from the seed, so after a rigid
pre-alignment it can be small, which keeps the halo small.

### Tile shape: read amplification against memory

Each tile reads its bricks once. For tile edge $T$ and halo $h$, the bytes
read per volume, relative to the volume, are

$$
\alpha = \left(\frac{T + 2h}{T}\right)^3
$$

| case (sphere, $h = S/2 + d_\text{max} + 2$) | $h$ | $T = 512$ | $T = 1024$ |
|---|---|---|---|
| S = 48, `disp_max` 10 | 36 | 1.49 | 1.23 |
| S = 80, `disp_max` 38 (CCPi example) | 80 | 2.26 | 1.55 |

(Table from `pipeline/tiling.py`.) This assumes points fill the tile. On case
M (1024³, 8 tiles of 512³), the bytes read were exactly the sum of the
per-tile brick boxes, below the formula's figure because the point lattice
stopped short of the faces ({doc}`/benchmarks/2026-09-25-M2-M3-cpu`).

Larger tiles read less, but:

* **memory** grows as $(T + 2h)^3$ per brick, times two volumes, times
  `prefetch_depth`;
* **load balance** needs several tiles per GPU, because tiles are handed out
  from a shared queue and the last tiles decide when the run ends;
* **resume** works per tile: a killed worker loses the tile it was solving and
  the ones it had prefetched.

`tile_shape` is given as `(z, y, x)` in voxels. It must be a whole number of
point-store cells per axis (within 1 %, for tiles that cover a volume edge
not divisible by the cell count). Otherwise `plan` refuses it:

```text
ValueError: tile_shape (48, 48, 48) (z, y, x) is not a multiple of the point-cloud chunk (32.0, 32.0, 32.0) (x, y, z)
```

## Memory: what `plan` checks

`zvdvc plan` computes, from the tiles it has just planned
({py:func}`zvdvc.pipeline.tiling.check_memory`):

| key | computed as |
|---|---|
| `brick_bytes` | the largest tile's reference plus deformed brick, in voxels × the volume's bytes per voxel (the native type stays on the device) |
| `in_flight_bytes` | `brick_bytes` × `prefetch_depth` |
| `batch_bytes` | batch size × per-point scratch, where per-point scratch is $9M + 4(2\,\text{dof} + 128)$ bytes for $M$ template samples |
| `device_bytes` | `gpu_memory_fraction` × the GPU's total memory (host RAM for the `cpu` and `numpy` backends) |
| `fits` | `in_flight_bytes + batch_bytes <= device_bytes` |

The batch size in this check is `cluster.batch_points` if set, otherwise the
automatic size (below), but never more than the largest tile's point count.

From the synthetic tutorial ({doc}`/tutorials/synthetic_first_run`), with
64³ tiles on a 12 GB GPU:

```text
{'tiles': 8, 'points': 125, 'memory': {'brick_bytes': 2370816, 'in_flight_bytes': 2370816, 'batch_bytes': 502416, 'device_bytes': 9983806668, 'fits': True}, 'plan': '/…/run/plan.json'}
```

When the tiles do not fit, `plan` stops with:

```text
MemoryError: tiles need … bytes of device memory with prefetch_depth 2; … usable. Reduce cluster.tile_shape or prefetch_depth.
```

### Estimating before you have the data

`check_memory` needs only tile boxes, so you can size a run from its
configuration alone. For one full interior tile, with rigid seeds:

```python
import dataclasses
from zvdvc import RunConfig
from zvdvc.geometry.box import Box
from zvdvc.pipeline.tiling import Tile, check_memory

cfg = RunConfig.from_yaml("large_8xh100.yaml")          # scenario B: sphere 48, disp_max 10, T = 1024
print("halo", cfg.halo())

def estimate(cfg, tile_edge, dtype_bytes=2, gpu_bytes=80e9, n_points=262_144):
    """Memory for one full interior tile: points fill the tile, seeds all equal (rigid)."""
    pts = Box((0, 0, 0), (tile_edge,) * 3)
    ref = pts.grow(cfg.halo())
    tile = Tile(0, (), n_points, pts, ref, ref, 0.0)
    return check_memory([tile], cfg, device_total_bytes=int(gpu_bytes), dtype_bytes=dtype_bytes)

for T in (512, 1024, 1536):
    c = dataclasses.replace(cfg, cluster=dataclasses.replace(cfg.cluster, tile_shape=(T, T, T)))
    m = estimate(c, T)
    alpha = ((T + 2 * cfg.halo()) / T) ** 3
    print(f"T={T}: alpha {alpha:.2f}, bricks {m.brick_bytes/1e9:.1f} GB, in flight {m.in_flight_bytes/1e9:.1f} GB, "
          f"batch {m.batch_bytes/1e9:.2f} GB, usable {m.device_bytes/1e9:.0f} GB, fits {m.fits}")
```

```text
halo 36.0
T=512: alpha 1.48, bricks 0.8 GB, in flight 1.6 GB, batch 9.81 GB, usable 64 GB, fits True
T=1024: alpha 1.23, bricks 5.3 GB, in flight 10.5 GB, batch 9.81 GB, usable 64 GB, fits True
T=1536: alpha 1.15, bricks 16.6 GB, in flight 33.3 GB, batch 9.81 GB, usable 64 GB, fits True
```

(`large_8xh100.yaml` is a copy of the repository's
`configs/large_8xh100.yaml`; `n_points` is a 1024³ tile at 16-voxel
spacing.) With `batch_points` unset, the batch here grows to the whole tile's
262 144 points, so `batch_bytes` is the same for every $T$. Setting
`batch_points: 32768` gives 1.23 GB instead. On a 12 GB workstation GPU the
same 1024³ u16 tile, with `batch_points: 32768`:

```python
for depth in (2, 1):
    c = dataclasses.replace(cfg, cluster=dataclasses.replace(cfg.cluster, prefetch_depth=depth, batch_points=32768))
    m = estimate(c, 1024, gpu_bytes=12e9)
    print(f"prefetch_depth {depth}: in flight {m.in_flight_bytes/1e9:.1f} GB + batch {m.batch_bytes/1e9:.2f} GB "
          f"vs usable {m.device_bytes/1e9:.1f} GB -> fits {m.fits}")
```

```text
prefetch_depth 2: in flight 10.5 GB + batch 1.23 GB vs usable 9.6 GB -> fits False
prefetch_depth 1: in flight 5.3 GB + batch 1.23 GB vs usable 9.6 GB -> fits True
```

The H100 figures match the budget in {doc}`/ARCHITECTURE` (section 8, memory
budget): about 5.3 GB of bricks per tile, 10.5 GB with one tile
prefetched, well inside 80 GB. That budget puts the batch scratch at 0.5 GB
for 32 k points, counting the reference samples only; `plan` counts more
per-point buffers and is the more conservative of the two.

## Batch size

Inside a tile, points are ordered along a Morton (Z-order) curve, so
consecutive batches read nearby parts of the bricks, and solved in batches
([`pipeline/batching.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/pipeline/batching.py)).
The batch size is:

* `cluster.batch_points`, if set;
* on a GPU otherwise, {py:func}`zvdvc.pipeline.batching.batch_size` of the
  device's free memory at the time, using half of `gpu_memory_fraction` of
  it;
* 8 192 points on the `cpu` and `numpy` backends.

`batch_size` divides the memory by the per-point scratch: the $M$ template
samples at 4 + 4 + 1 bytes each for FA-GN (the reference term and two
sample buffers), plus the parameters, state and one row of normal-equation sums.
The fused kernels never store per-sample Jacobians, so the batch is not
limited by the number of degrees of freedom beyond that sums row. For
example, `batch_size(40_000_000_000, 4096, 6)` is 538 213 points and
`batch_size(8_000_000_000, 8000, 6)` is 55 328.

Batch size is limited by this scratch, not by the bricks. Leave
`batch_points` unset unless you need a fixed size, for example to make
`plan`'s estimate match a small GPU as above.

## Other cluster settings

| field | default | effect |
|---|---|---|
| `prefetch_depth` | 2 | bricks of the next tiles read while the current one is solved; multiplies brick memory |
| `gpu_memory_fraction` | 0.8 | share of device memory `plan` treats as usable |
| `devices` | all visible | GPU ordinals; also `zvdvc run --devices` |
| `brick_dtype` | `native` | bricks keep the volume's type (u8 / u16 / f32) on the device |
| `scheduler` | `dynamic` | workers on a node share one tile queue |

`brick_dtype` and `scheduler` are accepted but not read by the pipeline yet:
bricks are always native and the queue always dynamic.

The synthetic cases and the case A configuration use `prefetch_depth: 1`,
with the whole volume as one tile.

## The point store's chunk shape

The search points live in a zarr-vectors point-cloud store with a chunk
(cell) shape in voxels, and a bin shape of half the chunk per axis (two bins
per axis inside each cell). The results store copies both. The chunk shape
matters because:

* a tile must be a whole number of chunks per axis;
* each cell is written whole by one worker, which is what lets workers write
  without locks and lets `run` resume per tile;
* whole-cloud readers (strain, export, viewers) read the results cell by
  cell.

When `plan` imports a `.roi` file, it chooses the chunk as a quarter of the
tile per axis ({py:func}`zvdvc.io.pointcloud.chunk_shape_for`): 1024³ tiles
give 256³ cells, 4³ cells per tile. `zvdvc synth` does the same with the
whole volume as the tile. If you build the point store yourself, choose a
chunk that divides the tile shape you intend to use.

```{note}
The `.roi` import is redone only when the `.roi` is newer than the imported
store (`<workdir>/points.zarrvectors`). If you change `tile_shape` to one
that the old chunk does not divide, delete that store, and the results
store, which `plan` refuses to reuse with another chunk shape.
```

## OME-Zarr chunks and shards for the volumes

Convert large inputs once:

```bash
zvdvc convert dataset_0.npy data/ref.ome.zarr --chunk 128 --shard 1024
```

* **Chunk** (`--chunk`, default 128): the unit zarr compresses and decodes.
  Compression is zstd, level 3.
* **Shard** (`--shard`, default 1024): the unit stored as one file, holding
  many chunks. Shards are rounded up to a whole number of chunks and never
  larger than the volume, so small volumes get one shard.
* **Alignment**: the recommended layout is 128³ chunks inside shards aligned
  to `cluster.tile_shape`, so that a brick read touches a handful of shards
  ([`io/volume.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/io/volume.py)).
* **Conversion memory**: `convert` streams slabs of one shard's depth of
  slices, so it holds about `shard` × Y × X voxels at a time. That keeps
  100 GB+ inputs within bounded memory.

`zvdvc synth` takes the same `--chunk` and `--shard` flags. `.raw` inputs also
need `--shape-xyz`, `--dtype` (for example `'<u2'`) and, if present,
`--header`; TIFF stacks need the `[tiff]` extra.

Bricks are decoded on the host into pinned memory and copied to the GPU in
one transfer. Reading bricks straight to the GPU (GPUDirect Storage, or
zarr's GPU buffers) is planned, not implemented; see
{doc}`/how_to/gpudirect_storage`.

## Checklist

1. Compute the halo (`RunConfig.halo()`), and keep `disp_max` small by
   pre-aligning and setting `rigid_trans`.
2. Pick the largest tile edge whose bricks, times two volumes, times
   `prefetch_depth`, fit in `gpu_memory_fraction` of the GPU with room for a
   batch, while still giving each GPU at least 4 tiles.
3. Make the point-store chunk divide the tile (a quarter of the tile edge is
   the default).
4. Store the volumes as OME-Zarr with 128³ chunks in shards aligned to the
   tile.
5. Run `zvdvc plan` and read the memory line before submitting the job.
