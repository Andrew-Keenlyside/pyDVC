# Using zvDVC from Python

This tutorial drives zvDVC from Python instead of the command line: building
a run configuration, running the pipeline stages, solving small problems in
memory, reading and exporting results, and making synthetic test cases. It is
for users who script parameter studies, or embed DVC in a larger analysis.

The page is one continuous Python session: later snippets reuse names and
files from earlier ones. It was run as shown on an RTX A2000 12 GB; outputs
are trimmed where marked `…` and absolute paths shortened to `/…/`. The full reference is
the {doc}`/api/index`.

---

## 1. Make a synthetic case

{py:func}`zvdvc.synth.phantoms.make_case` is what `zvdvc synth` calls. It
writes both volumes, the points (as a zarr-vectors store and a `.roi`),
`truth.npz` and `config.yaml`, and returns the config's path:

```python
from zvdvc.config import SearchSpec, SubvolumeSpec
from zvdvc.synth.phantoms import default_field, make_case

field = default_field("sinusoid", (96, 96, 96))
print(field.kind, field.params)
config_path = make_case(
    "data/sin96",
    shape_zyx=(96, 96, 96),
    field=field,
    spacing=12,
    noise_sigma=0.01,
    subvolume=SubvolumeSpec(geometry="sphere", size=32, n_samples=2000),
    search=SearchSpec(dof=12, objective="znssd", interpolation="tricubic", disp_max=8.0),
)
print(config_path)
```

```text
sinusoid {'amplitude': 1.0, 'wavelength': 96.0, 'axis': 0, 'component': 0}
/…/data/sin96/config.yaml
```

{py:func}`~zvdvc.synth.phantoms.default_field` gives the four standard fields
(`rigid`, `affine`, `sinusoid`, `inclusion`), scaled to the volume. You can
also pass your own {py:class}`~zvdvc.synth.phantoms.DisplacementField`, or
your own points with `points_xyz=`. `noise_sigma` is a fraction of full scale.
An existing case in the output folder is refused unless `overwrite=True`.

## 2. Build a run configuration

A run is described by a frozen dataclass tree,
{py:class}`zvdvc.config.RunConfig` ({doc}`/spec/run_config`). Load it from
YAML, or build it in code:

```python
from zvdvc import RunConfig
from zvdvc.config import ClusterSpec, SearchSpec, SeedingSpec, SubvolumeSpec, VolumeSpec

cfg = RunConfig.from_yaml("data/sin96/config.yaml")
print(cfg.subvolume, cfg.search.dof, cfg.halo())

cfg2 = RunConfig(
    volumes=VolumeSpec(reference="data/sin96/ref.ome.zarr", deformed="data/sin96/def.ome.zarr"),
    points="data/sin96/points.zarrvectors",
    output="data/sin96/results.zarrvectors",
    workdir="data/sin96/run",
    subvolume=SubvolumeSpec(geometry="sphere", size=32, n_samples=2000),
    search=SearchSpec(dof=12, objective="znssd", interpolation="tricubic", disp_max=8.0),
    seeding=SeedingSpec(strategy="wavefront"),
    cluster=ClusterSpec(tile_shape=(96, 96, 96), prefetch_depth=1),
)
print(cfg2.halo())
```

```text
SubvolumeSpec(geometry='sphere', size=32.0, n_samples=2000, aspect=(1.0, 1.0, 1.0), seed=0) 12 26.0
26.0
```

`halo()` is the brick margin around a tile's points: the subvolume radius,
plus `disp_max`, plus 2 voxels for the tricubic stencil.

Other ways in and out:

```python
import dataclasses

try:
    RunConfig.from_dict({**cfg2.to_dict(), "search": {"dof": 7}})
except ValueError as e:
    print("ValueError:", e)

cfg3 = dataclasses.replace(cfg2, uncertainty_seeds=2)     # configs are frozen: copy with changes
cfg3.to_yaml("data/sin96/config_sd.yaml")
print(RunConfig.from_yaml("data/sin96/config_sd.yaml").uncertainty_seeds)
```

```text
ValueError: config.search.dof: 7 is not one of [3, 6, 12]
2
```

* `from_dict` and `from_yaml` validate as they build: unknown or missing
  keys, invalid choices, out-of-range values and options not implemented yet
  (`method: icgn`, `strategy: fft`) raise `ValueError`, naming the field.
  A `RunConfig(...)` built directly, like `cfg2`, is checked only when you
  call `cfg2.validate()` (which returns it).
* `RunConfig.from_ccpi("dvc_input.txt")` reads a CCPi / iDVC `dvc_in` file
  (see {doc}`/tutorials/idvc_example`).
* To change a nested setting, replace the inner dataclass too:
  `dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, dof=6))`.

## 3. Run the pipeline stages

Each CLI stage is a function in {py:mod}`zvdvc.pipeline.coordinator` with the
same behaviour, safe to call again:

| CLI | Python | status |
|---|---|---|
| `zvdvc plan` | {py:func}`~zvdvc.pipeline.coordinator.prepare` | implemented |
| `zvdvc seed` | {py:func}`~zvdvc.pipeline.coordinator.seed` | implemented for `rigid`, `wavefront`, `coarse` |
| `zvdvc run` | {py:func}`~zvdvc.pipeline.coordinator.run` | implemented |
| `zvdvc repair` | {py:func}`~zvdvc.pipeline.coordinator.repair` | **not implemented** (M5): raises `NotImplementedError` |
| `zvdvc finalize` | {py:func}`~zvdvc.pipeline.coordinator.finalize` | implemented |

```python
from zvdvc import RunConfig
from zvdvc.pipeline import coordinator

cfg = RunConfig.from_yaml("data/sin96/config.yaml")
print(coordinator.prepare(cfg))            # = zvdvc plan
print(coordinator.seed(cfg))               # = zvdvc seed (wavefront: solves every point)
stats = coordinator.run(cfg)               # = zvdvc run
print(stats[0].tiles, stats[0].tiles_skipped)
summary = coordinator.finalize(cfg, export_disp=True)
print(summary)
try:
    coordinator.repair(cfg)
except NotImplementedError as e:
    print("repair:", e)
```

```text
{'tiles': 1, 'points': 125, 'memory': {'brick_bytes': 4000000, 'in_flight_bytes': 4000000, 'batch_bytes': 2326000, 'device_bytes': 9983806668, 'fits': True}, 'plan': '/…/data/sin96/run/plan.json'}
{'strategy': 'wavefront', 'points': 125, 'seconds': 0.3125790049089119}
0 1
RunSummary(n_points=125, seconds=0.022777106030844152, counts={0: 125})
repair: [M5] coordinator.repair - see docs/MVP_PLAN.md
```

* `prepare`, `seed` and `run` take `backend=` (`"fused"`, `"cupy"`, `"cpu"`,
  `"numpy"`; default `fused` on a GPU, else `cpu` with numba, else `numpy`).
* `run` also takes `devices=`, `cpu_workers=`, `max_tiles=`, and `tile_ids=`
  (solve these tiles only). It starts one worker process per GPU and returns
  a list of {py:class}`~zvdvc.pipeline.worker.WorkerStats`, one per worker,
  with tiles solved (`tiles`), written (`tiles_written`) and skipped
  (`tiles_skipped`), points, bytes read, compute and I/O-wait seconds, and
  status counts. The list's `missing` attribute holds this node's tiles still
  unwritten; the CLI exits with status 3 when it is not empty. Here `seed` had
  already written the only tile.
* `seed` and `run` raise `ValueError` if the config differs from the one
  `prepare` planned with in anything that changes results, or if the volumes
  or points changed ({doc}`/spec/results_store`, "Run fingerprint").
* `finalize` returns a `RunSummary` and writes `results.stat` (and, with
  `export_disp=True`, `results.disp`) in the workdir. It raises `ValueError`
  while cells are unwritten, unless `allow_partial=True`.

## 4. Solve small problems in memory

For data that fits in memory, {py:func}`zvdvc.pipeline.inmemory.run_in_memory`
skips tiles and stores: it reads both whole volumes, solves every point in
CCPi's order (or from rigid seeds), and returns the results as arrays. It is
what `zvdvc solve` and the iDVC drop-in (`zvdvc-dvc`) run.

```python
import numpy as np
from zvdvc import RunConfig
from zvdvc.io.ccpi import RunSummary, write_stat
from zvdvc.pipeline.inmemory import Results, run_in_memory

cfg = RunConfig.from_yaml("data/sin96/config_sd.yaml")      # uncertainty_seeds: 2
res = run_in_memory(cfg, backend="fused", progress=print)
print(res.status_counts(), f"{res.seconds:.2f} s", {k: round(v, 3) for k, v in res.timings.items()})
print(res.displacement.shape, res.displacement_sd.shape)
print("median sd per axis:", np.nanmedian(res.displacement_sd, axis=0))

res.save("data/sin96/inmem.npz")
res.write_disp("data/sin96/inmem.disp")
write_stat("data/sin96/inmem.stat", cfg, RunSummary(len(res.point_id), res.seconds, res.status_counts()))
again = Results.load("data/sin96/inmem.npz")
print(np.array_equal(again.params, res.params))
```

```text
shell 1/7: 1 points, 1/125 done
shell 2/7: 7 points, 8/125 done
…
shell 7/7: 7 points, 125/125 done
{0: 125} 0.05 s {'read': 0.014, 'solve': 0.023, 'uncertainty': 0.011}
(125, 3) (125, 3)
median sd per axis: [0.00457383 0.00402233 0.00384955]
True
```

{py:class}`~zvdvc.pipeline.inmemory.Results` holds per-point arrays in input
order: `point_id`, `xyz`, `status`, `objmin`, `params` (all `dof` shape
parameters), `n_iter`, `seed`, and `displacement_sd` when
`uncertainty_seeds` is set; `displacement` is `params[:, :3]`. The kernels
were already compiled by section 3, hence the short time. The progress
lines are wavefront *shells*: points at similar distance from the start
point, solved together as one batch and seeded from the shells before.

To solve chosen points, or with your own seeds, call
{py:func}`~zvdvc.pipeline.inmemory.solve_in_memory` with `point_id`, `xyz`,
and optionally `strategy="rigid"` and `seeds=`.

## 5. Read and compare results

A results store is read with {py:class}`zvdvc.io.results.ResultStore`
({doc}`/spec/results_store`):

```python
import numpy as np
from zvdvc import PointStatus
from zvdvc.io.results import ResultStore

store = ResultStore("data/sin96/results.zarrvectors")
print(store.dof, store.chunk_shape, len(store.written_cells()))
r = store.read_all()
print(sorted(r))

values, counts = np.unique(r["status"], return_counts=True)
print({PointStatus(int(v)).name: int(c) for v, c in zip(values, counts)})

order = np.argsort(r["point_id"])          # read_all returns cell order
print(r["point_id"][order][:3], r["displacement"][order][:3])
```

```text
12 (24.0, 24.0, 24.0) 27
['displacement', 'displacement_sd', 'n_iter', 'objmin', 'params', 'point_id', 'seed', 'status', 'xyz']
{'GOOD': 125}
[1 2 3] [[ 8.9980322e-01 -1.0210996e-02 -1.2898943e-02]
 [ 6.4346355e-01  1.0060617e-02 -1.6185013e-03]
 [ 2.7782403e-02 -1.2550915e-02 -4.1265332e-04]]
```

{py:mod}`zvdvc.bench.metrics` reads any of the three result formats (a
store, a zvDVC `.npz`, a CCPi `.disp`) and compares them, matching points on
`point_id`:

```python
from zvdvc.bench.metrics import against_disp, against_truth, load_results

acc = against_truth("data/sin96/results.zarrvectors", "data/sin96/truth.npz")
print(acc.summary())
print(against_disp("data/sin96/results.zarrvectors", "data/sin96/inmem.disp").summary())

r = load_results("data/sin96/inmem.npz")
print(sorted(r))
```

```text
points 125, GOOD 125 (100.00 %)  [GOOD 125]
rmse (x, y, z)  0.0840  0.0064  0.0059
bias (x, y, z)  -0.0013  +0.0001  +0.0003
|error| median 0.0789  p95 0.1170  p99 0.1200
points 125, GOOD 125 (100.00 %)  [GOOD 125]
rmse (x, y, z)  0.0000  0.0000  0.0000
…
status agreement 100.00 %
['displacement', 'displacement_sd', 'objmin', 'point_id', 'status', 'xyz']
```

The x error (RMSE 0.084 voxel) is not a solver fault. The sinusoid's
wavelength (96 voxels) is only three subvolume diameters, and a 12-DOF shape
function, linear across the subvolume, cannot follow the sine's curvature.
This is the spatial-resolution limit the `sinusoid` field exists to show; a
smaller subvolume or a longer wavelength reduces it. `against_disp` also
reports the status agreement, as used against CCPi. The tiled and in-memory
results agree exactly here because, with `wavefront` seeding, `seed` runs the
same in-memory solve and writes it into the store.

`Accuracy` objects carry the numbers as fields (`rmse`, `bias`,
`median_abs`, `p95_abs`, `p99_abs`, `frac_good`, `status_counts`, …).

## 6. Export for iDVC and CCPi

| file | from a `Results` | from a store |
|---|---|---|
| `.disp` (iDVC's results viewer, CCPi `strain`) | `res.write_disp(path)` | `coordinator.finalize(cfg, export_disp=True)`, or {py:func}`zvdvc.io.ccpi.write_disp` on the arrays of `read_all()` |
| `.stat` | {py:func}`zvdvc.io.ccpi.write_stat` with a `RunSummary` | written by `finalize` |
| `.npz` | `res.save(path)`, `Results.load(path)` | — |

```text
n	x	y	z	status	objmin	u	v	w
1	23.5	23.5	23.5	0	0.00227333419	0.899803221	-0.0102109956	-0.0128989434
2	35.5	23.5	23.5	0	0.00206826697	0.643463552	0.0100606168	-0.00161850126
…
```

(`data/sin96/inmem.disp`.) Statuses are written as CCPi codes: zvDVC's own
codes below −3 become `NOT_SEARCHED` (−3), so iDVC reads the file correctly.
Points that are not `GOOD` are written with `u v w = 0` and a finite `objmin`,
as CCPi writes them; the `Results`, `.npz` and store keep the raw values.

## 7. Strain from Python

`zvdvc strain` is {py:func}`zvdvc.post.strain.compute_strain`, which writes
the CSV files. {py:func}`~zvdvc.post.strain.fit_strain` returns the arrays
without writing anything:

```python
from zvdvc.bench.metrics import load_results
from zvdvc.post.strain import fit_strain

r = load_results("data/sin96/inmem.npz")
s = fit_strain(r["xyz"], r["displacement"], r["status"], r["objmin"], window=25, sigma_u=r["displacement_sd"])
print(s.engineering.shape, s.strain_sd.shape, s.planar_axis)
```

```text
(125, 6) (125, 6) None
```

`StrainResult` also has `lagrangian`, `grad`, the principal strains,
`residual_rms`, `pts_in_sw` and `sw_radius`. The method and the uncertainty
are covered in {doc}`/tutorials/strain_uncertainty`.

## 8. The analytic field

{py:func}`~zvdvc.synth.phantoms.load_field` rebuilds a case's field from its
`truth.npz`, so you can evaluate the true displacement, or its gradient,
anywhere:

```python
import numpy as np
from zvdvc.synth.phantoms import load_field

field = load_field("data/sin96/truth.npz")
r = load_results("data/sin96/inmem.npz")
print(np.abs(r["displacement"] - field(r["xyz"])).max())
```

```text
0.1206110875869919
```

`field.gradient(xyz)` gives the `(N, 3, 3)` displacement gradient.
