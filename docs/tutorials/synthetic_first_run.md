# Your first run, explained

This tutorial runs zvDVC on a small synthetic case with a known displacement
field and stops at every stage to look at what it wrote. It is the long
version of {doc}`/getting_started/quickstart`: same commands, with the
outputs opened and explained. It is for readers who will go on to run their
own data and want to know what each file and each number means.

Everything below was run on a workstation with an RTX A2000 12 GB GPU. The
outputs are real, trimmed where marked `…`; absolute paths are shortened to
`/…/`. Without a GPU, every command still works: zvDVC picks the `cpu`
backend (or `numpy` without numba), and only the times change.

---

## 1. Make a synthetic case

```bash
zvdvc synth --shape 128 128 128 --field affine --spacing 16 --out data/synth128
```

```text
/…/data/synth128/config.yaml
```

`zvdvc synth` builds a *speckle phantom*: a random texture, Gaussian-filtered
white noise, written as the reference volume. It then deforms that volume with
an analytic displacement field to make the deformed volume. Because the field
is known, every result can be checked against the truth. This is case S of
{doc}`/MVP_PLAN` at a smaller size (case S is 256³).

The flags ([`cli.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/cli.py)):

| flag | default | meaning |
|---|---|---|
| `--shape Z Y X` | required | volume size in voxels |
| `--field` | `affine` | displacement field, see below |
| `--spacing` | `16` | spacing of the search-point grid (voxels) |
| `--dtype` | `uint16` | voxel type of both volumes |
| `--noise` | `0` | Gaussian noise σ as a fraction of full scale (`0.02` = 2 %), independent in the two volumes |
| `--seed` | `0` | random seed of the speckle |
| `--chunk`, `--shard` | `128`, `1024` | OME-Zarr chunk and shard edge of the volumes |
| `--geometry` | `sphere` | subvolume shape, `sphere` or `cube` |
| `--subvol-size` | `32` | subvolume diameter (sphere) or side (cube), voxels |
| `--subvol-npts` | `2000` | sample points per subvolume |
| `--dof` | `12` | degrees of freedom of the shape function: 3, 6 or 12 |
| `--objective` | `znssd` | `sad`, `ssd`, `zssd`, `nssd` or `znssd` |
| `--disp-max` | `8` | search range from the seed (voxels) |
| `--out` | required | output folder |
| `--overwrite` | off | replace a case already in `--out`; without it `synth` refuses |

Interpolation is always tricubic for synthetic cases.

The four fields
([`synth/phantoms.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/synth/phantoms.py))
are scaled to the volume, so that `|u|` stays within about 3 voxels:

| field | what it is | what it tests |
|---|---|---|
| `rigid` | translation plus a small rotation | seeding, range handling, 6-DOF |
| `affine` | translation plus a uniform strain of up to about 1 % | 12-DOF and strain |
| `sinusoid` | `u = A sin(2πx/λ)` along one axis (one period across the volume) | spatial resolution against subvolume size |
| `inclusion` | an Eshelby-like stiff sphere in a strained matrix | a realistic mix of gradients |

`synth` refuses a field whose largest displacement plus one voxel exceeds
`--disp-max`.

The search points form a lattice through the volume centre. Points closer to a
face than the subvolume radius plus the largest displacement plus 3 voxels
are dropped, so every subvolume stays inside both volumes. At 128³ with
spacing 16 that leaves 5 × 5 × 5 = 125 points.

The output folder holds:

| path | contents |
|---|---|
| `ref.ome.zarr`, `def.ome.zarr` | reference and deformed volumes, sharded OME-Zarr v3 |
| `points.zarrvectors` | the search points as a zarr-vectors point-cloud store |
| `points.roi` | the same points as a CCPi `.roi` file (`n x y z` per line), for CCPi runs |
| `truth.npz` | per point: `point_id`, `xyz`, the true `displacement` and its `gradient`, and the field's definition |
| `config.yaml` | the run configuration, with absolute paths |

## 2. Read the generated configuration

`config.yaml` is a complete {py:class}`zvdvc.config.RunConfig`. Every field is
written out, defaults included (trimmed here):

```yaml
volumes:
  reference: /…/data/synth128/ref.ome.zarr
  deformed: /…/data/synth128/def.ome.zarr
  array_path: '0'
  prefilter_sigma: 0.0
points: /…/data/synth128/points.zarrvectors
output: /…/data/synth128/results.zarrvectors
subvolume:
  geometry: sphere
  size: 32.0
  n_samples: 2000
  seed: 0
search:
  dof: 12
  objective: znssd
  interpolation: tricubic
  disp_max: 8.0
  rigid_trans: [0.0, 0.0, 0.0]
  method: fagn
  max_iterations: 20
  obj_tol: 1.0e-06
  disp_tol: 0.01
seeding:
  strategy: wavefront
  n_neighbours: 75
cluster:
  tile_shape: [128, 128, 128]
  gpu_memory_fraction: 0.8
  batch_points: null
  prefetch_depth: 1
uncertainty_seeds: 0
workdir: /…/data/synth128/run
```

* `volumes`, `points`, `output`: the inputs and the results store. `workdir`
  is where the stages keep their own files (`plan.json`, logs, `.stat`).
* `subvolume` and `search`: the correlation settings. Each maps onto a CCPi
  `dvc_in` key (`subvol_size`, `num_srch_dof`, `obj_function`, …); see
  {doc}`/spec/run_config`.
* `seeding.strategy: wavefront` is CCPi's point order: solve outwards from a
  start point, seeding each point from solved neighbours.
* `cluster`: how the work is split. `synth` makes the whole volume one tile.

Every field, its default and its CCPi equivalent are listed in
{doc}`/spec/run_config`.

### Make more than one tile

A *tile* is the unit of work a GPU takes from the queue: a block of the
point store's chunks, solved against one *brick* (the tile's region of each
volume, plus a margin called the *halo*). `synth` chunks the point store at a
quarter of the volume per axis, 32³ voxels here, and makes the whole volume
one tile. To see the tile loop, and resume, at work, set a 64³ tile in
`data/synth128/config.yaml`:

```yaml
cluster:
  tile_shape: [64, 64, 64]
```

A tile must be a whole number of point chunks per axis. 64 is two chunks of
32; a 48³ tile is refused by `plan`:

```text
zvdvc plan: error: tile_shape (48, 48, 48) (z, y, x) is not a multiple of the point-cloud chunk (32.0, 32.0, 32.0) (x, y, z)
```

A user error like this prints one line and exits with status 2; add
`--traceback` before the command (`zvdvc --traceback plan ...`) to see where
it came from.

{doc}`/how_to/choose_tiles_and_batches` covers choosing tile sizes for real data.

## 3. Plan

```bash
zvdvc plan data/synth128/config.yaml
```

```text
{'tiles': 8, 'points': 125, 'memory': {'brick_bytes': 2370816, 'in_flight_bytes': 2370816, 'batch_bytes': 502416, 'device_bytes': 9983806668, 'fits': True}, 'plan': '/…/data/synth128/run/plan.json'}
```

`plan` reads the point store and the volumes' metadata (not their voxels),
groups the point chunks into tiles, works out each tile's bricks, checks
that they fit in device memory, and allocates the results store. It prints:

| key | meaning here |
|---|---|
| `tiles` | 8 tiles of 64³ |
| `points` | 125 search points |
| `brick_bytes` | reference plus deformed brick of the largest tile, in the volumes' native type: 2 × 84³ voxels × 2 bytes |
| `in_flight_bytes` | `brick_bytes` × `cluster.prefetch_depth` (1 here) |
| `batch_bytes` | solver scratch for one batch of points; the batch is never larger than the largest tile (27 points here) |
| `device_bytes` | usable device memory: `gpu_memory_fraction` (0.8) × the GPU's 12 GB |
| `fits` | `in_flight_bytes + batch_bytes <= device_bytes`; if not, `plan` stops with a `MemoryError` |

The brick is larger than the tile's points by the halo: the subvolume radius
(16), plus `disp_max` (8), plus 2 voxels for the tricubic stencil, 26 voxels
on each side ({py:meth}`zvdvc.config.RunConfig.halo`). The largest tile's
points span 32 voxels per axis, so its brick is 32 + 2 × 26 = 84 voxels.

`plan` also writes `run/plan.json`: every tile's cells, point count, point box,
reference and deformed brick boxes and cost estimate, the node assignment,
the full config, the halo, the subvolume template's hash, the run
fingerprint (the settings that change results, both volumes' identities and
a digest of the points), and the versions of zvDVC, zarr-vectors, zarr and
numpy. `seed` and `run` check the config against it and refuse one that
would change the results.

With a `.roi` file as `points`, `plan` first imports it into
`run/points.zarrvectors`.

## 4. Seed

```bash
zvdvc seed data/synth128/config.yaml
```

```text
{'strategy': 'wavefront', 'points': 125, 'seconds': 0.5325644640251994}
```

What `seed` does depends on `seeding.strategy`:

| strategy | `seed` does | then `run` |
|---|---|---|
| `rigid` | nothing; every point starts from `search.rigid_trans` | solves every tile independently |
| `wavefront` | the whole solve, in CCPi's order, with both volumes in memory; writes every result into the store | finds every tile written (and refuses to start if `seed` has not written them) |
| `coarse` | solves every `coarse_stride`-th grid point, interpolates a seed for every point, writes `seeds.npz`, re-plans the bricks | solves every tile from those seeds (and refuses to start without them) |

`wavefront` is CCPi-parity mode and the default for synthetic cases, so here
all 125 points were solved in about half a second. `wavefront` and `coarse`
both hold the two whole volumes in host memory today and refuse volumes that
do not fit. For large data use `rigid` after a rigid pre-alignment, as
[`configs/large_8xh100.yaml`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/configs/large_8xh100.yaml)
does. The pyramid-level coarse pass is planned (M5).

## 5. Run

```bash
zvdvc run data/synth128/config.yaml
```

```text
device 0: 0 tiles solved, 0 written, 8 already written, 0 points, 0.00 GB read, compute 0.0 s, I/O wait n/a, status counts {}, 0 errors
```

`run` is the tile loop: one worker process per GPU, each taking tiles from a
shared queue, reading their bricks ahead (`prefetch_depth`), solving the
points in batches, and writing the results. Before solving a tile, a worker
checks the results store, and skips a tile whose cells are all written.
Here `seed` has already written every tile, so there is nothing left to do.
Section 9 shows the loop doing real work, and resuming.

One line per worker: tiles solved, tiles written to the store, tiles skipped
as already written, points, bytes read, compute time, the share of time the
compute loop waited for bricks (`I/O wait`), status counts, and errors. A
tile that is solved but fails to write counts as solved, not written.

If any of this node's tiles is still unwritten at the end (a tile failed
twice), `run` prints `zvdvc run: error: N tiles are not written (see
failed_tiles.json in <workdir>); run `zvdvc run` again` and exits with status
3, so a job script can stop before `finalize`.

### Backends

`--backend` picks the compute engine. The default is `fused` on a GPU, else
`cpu` if numba is installed, else `numpy`.

| backend | runs on | what it is |
|---|---|---|
| `fused` | GPU | one CUDA kernel per Gauss–Newton step: warp, tricubic value and gradient, residual and normal-equation sums |
| `cupy` | GPU | the same algorithm as array operations in CuPy |
| `cpu` | CPU cores | the fused step compiled with numba; the restructured-CPU baseline |
| `numpy` | one CPU core | float64 reference implementation |

Other `run` flags: `--devices 0 1` (GPU ordinals; default every visible
GPU), `--cpu-workers N` (worker processes for the CPU backends), and
`--max-tiles N` (solve only the first N tiles, for short profiling runs).

## 6. Finalize

```bash
zvdvc finalize data/synth128/config.yaml --disp
```

```text
RunSummary(n_points=125, seconds=1.043102293042466, counts={0: 125})
```

`finalize` completes the results store (it rebuilds zarr-vectors' presence
index and writes the metadata), then writes a summary. It refuses while any
cell is unwritten; `--allow-partial` finalises anyway and exports the
missing points as `NOT_SEARCHED`. `--disp` adds a CCPi
`.disp` export and `--pyramid` a zarr-vectors pyramid for viewers. The
workdir now holds:

```text
events/  plan.json  results.disp  results.stat  run_stats.json
```

`results.disp` has CCPi's columns and can be opened by iDVC's results viewer
and CCPi's `strain` program:

```text
n	x	y	z	status	objmin	u	v	w
1	31.5	31.5	31.5	0	4.39302858e-05	0.672790647	-0.171503559	0.222977489
2	47.5	31.5	31.5	0	1.73230255e-05	0.973328233	-0.0970191434	0.110869206
…
```

Points that are not `GOOD` would be written with `u v w = 0`, as CCPi writes
them; the store keeps their raw values.

`results.stat` lists the point count, time, rate and status counts (zvDVC's,
then CCPi-style `number successful`, `range fail`, `convg fail`, `not
searched`), then the full configuration. Its time comes from the last `run` (`run_stats.json`), so
in `wavefront` mode, where `seed` does the solving, it is the time of a `run`
that had nothing to do.

## 7. Compare with the truth

```bash
zvdvc compare data/synth128/results.zarrvectors data/synth128/truth.npz
```

```text
points 125, GOOD 125 (100.00 %)  [GOOD 125]
rmse (x, y, z)  0.0010  0.0010  0.0010
bias (x, y, z)  +0.0002  +0.0000  -0.0001
|error| median 0.0014  p95 0.0030  p99 0.0035
```

Points are matched on `point_id`. Errors are measured over points whose
status is GOOD ([`bench/metrics.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/bench/metrics.py)):

| line | meaning |
|---|---|
| `GOOD` | points that converged, and their share; then the count per status |
| `rmse (x, y, z)` | root-mean-square error per axis, in voxels |
| `bias (x, y, z)` | mean error per axis: a systematic offset shows here, random error does not |
| `\|error\|` | median, 95th and 99th percentile of the error vector's length |

The MVP's synthetic accuracy criterion (Q1 in {doc}`/MVP_PLAN`) is an RMSE of
at most 0.02 voxel noise-free, with at least 99 % GOOD. This run is at 0.0010.
Such small errors are a property of the phantom: its texture is smooth at the
voxel scale and it has no noise. Real scans give larger errors; see
{doc}`/tutorials/idvc_example`.

`compare` also takes a CCPi `.disp` as the reference. It then adds the status
agreement between the two codes.

## 8. Inspect the results in Python

The results store is a zarr-vectors point cloud on the same chunk grid as
the points. Each result is a vertex attribute
({doc}`/spec/results_store`). {py:class}`zvdvc.io.results.ResultStore` reads it:

```python
import numpy as np
from zvdvc import PointStatus
from zvdvc.io.results import ResultStore

store = ResultStore("data/synth128/results.zarrvectors")
print(store.dof, store.chunk_shape, len(store.written_cells()))

r = store.read_all()
print(sorted(r))
print(r["displacement"].shape, r["params"].shape)

values, counts = np.unique(r["status"], return_counts=True)
print({PointStatus(int(v)).name: int(c) for v, c in zip(values, counts)})
```

```text
12 (32.0, 32.0, 32.0) 27
['displacement', 'displacement_sd', 'n_iter', 'objmin', 'params', 'point_id', 'seed', 'status', 'xyz']
(125, 3) (125, 12)
{'GOOD': 125}
```

* `written_cells()` lists the chunks (cells) that hold complete results: 27
  of the 4³ grid hold points.
* `read_all()` returns every written point in cell order, not input order;
  sort on `point_id` to line rows up with the input.
* `displacement` is `(u, v, w)` in voxels; `params` is the full 12-DOF vector
  in CCPi's order; `objmin` is the objective at the solution; `n_iter` the
  Gauss–Newton iterations; `seed` the starting displacement.
* `displacement_sd` is NaN unless the run set `uncertainty_seeds`
  ({doc}`/tutorials/strain_uncertainty`).

The status codes are CCPi's for 0 to −3 (GOOD, RANGE_FAIL, CONVG_FAIL,
NOT_SEARCHED) plus zvDVC's THRESH_FAIL (−4) and SINGULAR (−5);
{doc}`/spec/status_codes` defines them.

## 9. Resume: run again and only the missing tiles are solved

To watch the tile loop do the work, copy the configuration with
`seeding.strategy: rigid`, a new `output` and a new `workdir` (here
`config_rigid.yaml`, `results_rigid.zarrvectors`, `run_rigid`). With `rigid`,
`seed` does nothing and `run` solves every tile. The displacements here are
within about 3 voxels of zero, close enough for rigid seeds.

Solve three tiles, as if the job had been killed:

```bash
zvdvc plan data/synth128/config_rigid.yaml
zvdvc seed data/synth128/config_rigid.yaml
zvdvc run  data/synth128/config_rigid.yaml --max-tiles 3
```

```text
{'strategy': 'rigid'}
device 0: 3 tiles solved, 3 written, 0 already written, 63 points, 0.01 GB read, compute 0.1 s, I/O wait 0.0 %, status counts {0: 63}, 0 errors
```

Run it again, and only the other five are solved:

```bash
zvdvc run data/synth128/config_rigid.yaml
```

```text
device 0: 5 tiles solved, 5 written, 3 already written, 62 points, 0.01 GB read, compute 0.2 s, I/O wait 0.0 %, status counts {0: 62}, 0 errors
```

A third `run` finds nothing to do (`0 tiles solved, 0 written, 8 already
written`). After `finalize`, `zvdvc compare` gives the same figures as
section 7. Had you run `finalize` after the first, partial `run`, it would
have refused:

```text
zvdvc finalize: error: /…/data/synth128/results_rigid.zarrvectors: 11 of 27 cells unwritten; run `zvdvc run` again (or finalize --allow-partial to export their points as NOT_SEARCHED)
```

This is how an interrupted or pre-empted job resumes: resubmit `run`. Workers
write each cell's `status` last, and a cell counts as written only when every
result array holds it, so a worker killed half-way through a cell leaves
nothing that looks finished. Tiles still missing after a run are listed in
`run_stats.json` and `failed_tiles.json`. The store also records the
subvolume template's hash, `prefilter_sigma` and the run fingerprint, and
`plan`, `seed` and `run` refuse to add results made with other settings,
volumes or points to it. Resume with the same config. Changing the
objective in `config_rigid.yaml`, for example, stops the next `run`:

```text
zvdvc run: error: the config differs from the one `zvdvc plan` used (/…/data/synth128/run_rigid/plan.json) in search.objective: run with the original config, or plan again with a new output
```

The volumes' resolved paths are part of the fingerprint, so moving or
copying the case folder also refuses a resume into the old store; plan again
with a new `output` and `workdir`.

## 10. Other backends

The same case on the numba CPU engine, with two worker processes:

```bash
zvdvc plan data/synth128/config_cpu.yaml --backend cpu
zvdvc run  data/synth128/config_cpu.yaml --backend cpu --cpu-workers 2
```

```text
device 1: 5 tiles solved, 5 written, 0 already written, 68 points, 0.01 GB read, compute 12.9 s, I/O wait 0.0 %, status counts {0: 68}, 0 errors
device 0: 3 tiles solved, 3 written, 0 already written, 57 points, 0.01 GB read, compute 13.3 s, I/O wait 0.0 %, status counts {0: 57}, 0 errors
```

(`config_cpu.yaml` is `config_rigid.yaml` with its own `output` and
`workdir`.) The results match the GPU's to the four digits `compare` prints.
Most of the 13 s is numba compiling the kernels on first use, which is then
cached. For CPU backends `plan` checks the bricks against host memory
instead of GPU memory.

## Next steps

* {doc}`/tutorials/idvc_example`: the same stages on iDVC's example scan,
  against CCPi DVC.
* {doc}`/tutorials/strain_uncertainty`: strain, and an uncertainty for every
  displacement.
* {doc}`/tutorials/python_api`: the same pipeline from Python.
* {doc}`/how_to/choose_tiles_and_batches`: tile, batch and chunk sizes for
  real data.
