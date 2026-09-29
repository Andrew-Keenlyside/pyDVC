# Run configuration

This page lists every field of the run configuration: its section, type,
default, meaning and valid values, and how a CCPi `dvc_in` file maps onto it.
It is for anyone writing or reviewing a `config.yaml`. The configuration is a
frozen dataclass tree, {py:class}`zvdvc.config.RunConfig`, defined in
[`src/zvdvc/config.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/config.py).
Where a CCPi `dvc_in` key exists for a field, it is named in the tables.

## Terms

Run configuration
: One YAML file (or {py:class}`~zvdvc.config.RunConfig` object) that describes
  a whole run. Every `zvdvc` stage (`plan`, `seed`, `run`, `finalize`, `solve`)
  takes it as its only positional argument.

Section
: A nested mapping of the YAML file: `volumes`, `subvolume`, `search`
  (with an optional `threshold` sub-section), `seeding` and `cluster`. The
  fields `points`, `output`, `num_points_to_process`, `uncertainty_seeds` and
  `workdir` sit at the top level.

Axis order
: Points and vectors are $(x, y, z)$ in voxels. Shapes and boxes of arrays are
  $(z, y, x)$. `cluster.tile_shape` is therefore $(z, y, x)$ and
  `search.rigid_trans` $(x, y, z)$.

Work directory
: `workdir`, where the stages keep `plan.json`, the imported point store,
  `seeds.npz`, `run_stats.json`, event logs and the `.stat`/`.disp` summaries.

---

## Introduction

Load a configuration with `RunConfig.from_yaml(path)`, build one from nested
plain data with `RunConfig.from_dict(d)`, or import a CCPi input file with
`RunConfig.from_ccpi(path)`. `to_yaml(path)` and `to_dict()` write it back
out; `zvdvc synth` writes a complete `config.yaml` this way.

Loading is strict (`from_yaml` and `from_dict` call
{py:meth}`~zvdvc.config.RunConfig.validate`):

* a file that is not valid YAML raises `ValueError`;
* an unknown key raises `ValueError` naming it (`config.search: unknown keys [...]`);
* a missing required field (`volumes.reference`, `volumes.deformed`, `points`,
  `output`, and `gray_min`/`gray_max` when `threshold` is given) raises
  `ValueError` (`config: missing required key 'output'`);
* a value outside a fixed set of choices raises `ValueError`
  (`config.search.dof: 5 is not one of [3, 6, 12]`);
* a list of the wrong length, a non-number for a number field, a
  non-integral number for an integer field, or a non-boolean for a boolean
  field raises `ValueError`;
* a value out of range raises `ValueError` naming the key and the rule
  (`config.search.max_iterations: 300 is out of range; it must be between 1
  and 255`); the ranges are in the tables below;
* `search.method: icgn` and `seeding.strategy: fft` are refused as not
  implemented yet (M5);
* a field that is read but not used yet (`seeding.coarse_level`,
  `seeding.repair_passes`, `cluster.brick_dtype`, `cluster.scheduler`)
  warns when set to anything but its default.

Omitted optional fields take the defaults below. Integers are accepted for
float fields.

Once `zvdvc plan` has made a results store, `seed` and `run` refuse a config
that differs from the planned one in anything that changes results (every
setting but `cluster`, `workdir`, `uncertainty_seeds`,
`seeding.repair_passes` and the input and output paths), in the `points` or
`output` path, or in the volumes' identity. `plan` refuses to reuse a store
made with other settings, volumes or points ({doc}`results_store`, "Run
fingerprint").

---

## Technical reference

### Top level

| field | type | default | CCPi key | meaning |
|---|---|---|---|---|
| `volumes` | mapping | required | | Image volumes; see [volumes](#volumes). |
| `points` | string | required | `point_cloud_filename` | Search points: a zarr-vectors store (a path ending `.zarrvectors` or `.zv`, or a directory holding `zarr.json`), or a CCPi `.roi` / iDVC `.txt`/`.csv` point file. `zvdvc plan` imports a point file into `<workdir>/points.zarrvectors`, and imports it again when the file is newer than the store ({doc}`points_store`). `zvdvc solve` reads a point file directly. |
| `output` | string | required | `output_filename` | The zarr-vectors results store written by the tiled pipeline ({doc}`results_store`). `zvdvc solve` does not use it. |
| `subvolume` | mapping | defaults below | | Sample template; see [subvolume](#subvolume). |
| `search` | mapping | defaults below | | Solver settings; see [search](#search). |
| `seeding` | mapping | defaults below | | Starting displacements; see [seeding](#seeding). |
| `cluster` | mapping | defaults below | | Tiles, devices and memory; see [cluster](#cluster). |
| `num_points_to_process` | int or null | `null` (all) | `num_points_to_process` | Solve only the first N points in processing order (distance from the start point); the rest are `NOT_SEARCHED`. Honoured by the in-memory solves: `zvdvc solve`, the wavefront `zvdvc seed` stage (and the coarse stage's sub-grid solve) and the `zvdvc-dvc` drop-in. The tiled `zvdvc run` does not read it. `0` from a CCPi file means all. |
| `uncertainty_seeds` | int | `0` | | $k \ge 0$. $k > 0$: solve every `GOOD` point $k$ more times with other template seeds and store the per-axis standard deviation as `displacement_sd` ({doc}`method`, "Displacement uncertainty"). Costs about $k$ times the main solve. |
| `workdir` | string | `"runs/default"` | | Work directory (created if missing). Give each run its own: the default is shared by every config that does not set one. |

### `volumes`

{py:class}`zvdvc.config.VolumeSpec`. See {doc}`volumes` for formats.

| field | type | default | CCPi key | meaning |
|---|---|---|---|---|
| `reference` | string | required | `reference_filename` | Reference volume: an OME-Zarr group or array (a path ending `.zarr`/`.ome.zarr`, or a directory holding `zarr.json`), or a `.raw`, `.mhd` or `.npy` file. Any other suffix is read as a flat raw file. |
| `deformed` | string | required | `correlate_filename` | Deformed volume, same shape as the reference. `zvdvc plan` refuses a pair whose shapes differ. |
| `array_path` | string | `"0"` | | The array inside an OME-Zarr group (the multiscale level). Ignored for arrays and flat files. |
| `raw_shape_xyz` | 3 ints or null | `null` | `vol_wide`, `vol_high`, `vol_tall` | Size of a `.raw` volume, $(x, y, z)$. Required for `.raw`; ignored for `.npy`, `.mhd` and OME-Zarr. |
| `raw_dtype` | string or null | `null` | from `vol_bit_depth` and `vol_endian` | numpy dtype of a `.raw` volume, e.g. `"\|u1"`, `"<u2"`, `">u2"`. Required for `.raw`. |
| `raw_header_bytes` | int | `0` | `vol_hdr_lngth` | Bytes to skip at the start of a `.raw` file, $\ge 0$. The file must hold exactly this header plus the voxels ({doc}`volumes`). |
| `prefilter_sigma` | float | `0.0` | | Gaussian low-pass (voxels) applied to both volumes as they are read, $\ge 0$; 0 is off, as in CCPi. About 1.0 removes most interpolation bias. Recorded in the results store; a resumed run with another value is refused. |

### `subvolume`

{py:class}`zvdvc.config.SubvolumeSpec`. See {doc}`method`, "Sample templates".

| field | type | default | CCPi key | meaning and valid values |
|---|---|---|---|---|
| `geometry` | string | `"sphere"` | `subvol_geom` | `cube` or `sphere`. |
| `size` | float | `80.0` | `subvol_size` | Cube side or sphere diameter, voxels, $> 0$. |
| `n_samples` | int | `8000` | `subvol_npts` | Samples per subvolume, 1 to 1 000 000. A cube rounds up to $k^3$. |
| `aspect` | 3 floats | `[1.0, 1.0, 1.0]` | `subvol_aspect` | Per-axis $(x, y, z)$ scale of the subvolume, each $> 0$. |
| `seed` | int | `0` | | Random seed of the sphere template, $\ge 0$. One template is shared by all points. |

### `search`

{py:class}`zvdvc.config.SearchSpec`. See {doc}`method`.

| field | type | default | CCPi key | meaning and valid values |
|---|---|---|---|---|
| `dof` | int | `6` | `num_srch_dof` | `3` (translation), `6` (+ rotation) or `12` (+ strain). |
| `objective` | string | `"znssd"` | `obj_function` | `sad`, `ssd`, `zssd`, `nssd` or `znssd`. |
| `interpolation` | string | `"tricubic"` | `interp_type` | `nearest`, `trilinear` or `tricubic` (CCPi has only the last two). |
| `disp_max` | float | `38.0` | `disp_max` | Largest allowed $\lVert\mathbf{u} - \text{seed}\rVert_\infty$, voxels, $> 0$; also sizes the brick halo and the grid search. |
| `rigid_trans` | 3 floats | `[0.0, 0.0, 0.0]` | `rigid_trans` | Rigid offset $(x, y, z)$ of the deformed volume, voxels: the seed where no other seed exists. |
| `basin_radius` | float | `0.0` | `basin_radius` | Step of the translation grid search, voxels, $\ge 0$; 0 disables it. |
| `threshold` | mapping or null | `null` | `subvol_thresh` | Subvolume threshold test; see below. `null` is CCPi's `subvol_thresh off`. |
| `method` | string | `"fagn"` | | `fagn` (forward-additive Gauss–Newton, CCPi parity). `icgn` (inverse compositional) is **not implemented** and is refused when the config is loaded (M5). |
| `max_iterations` | int | `20` | | Gauss–Newton steps per point (CCPi's `maxit`), 1 to 255 (stored as uint8). |
| `obj_tol` | float | `1e-6` | | Stop when the objective changes by less than this, $\ge 0$. |
| `disp_tol` | float | `0.01` | | Stop when the translation step is shorter than this, voxels, $\ge 0$. |
| `report_convg_fail` | bool | `true` | | Mark points that reach `max_iterations` as `CONVG_FAIL`. `false` marks them `GOOD`, as CCPi's solver path does. |

#### `search.threshold`

{py:class}`zvdvc.config.ThresholdSpec`. When present, a point whose fraction of
reference samples within `[gray_min, gray_max]` is below `min_fraction` is
`THRESH_FAIL` and is not searched.

| field | type | default | CCPi key | meaning |
|---|---|---|---|---|
| `gray_min` | float | required | `gray_thresh_min` | Lower grey level. |
| `gray_max` | float | required | `gray_thresh_max` | Upper grey level, $\ge$ `gray_min`. |
| `min_fraction` | float | `0.2` | `min_vol_fract` | Minimum fraction of samples in range, 0 to 1. |

### `seeding`

{py:class}`zvdvc.config.SeedingSpec`. See {doc}`method`, "Seeding".

| field | type | default | CCPi key | meaning and valid values |
|---|---|---|---|---|
| `strategy` | string | `"wavefront"` | | `rigid`, `wavefront` or `coarse`. `fft` is not implemented and is refused when the config is loaded (M5). `zvdvc run` enforces the strategy: `coarse` needs `zvdvc seed` to have written the seed field, and `wavefront` needs `zvdvc seed` to have written every point. |
| `start_point` | 3 floats or null | `null` | `starting_point` | $(x, y, z)$ the wavefront starts from; `null` is the first point of the cloud. |
| `n_neighbours` | int | `75` | | Neighbours searched for `GOOD` seeds (CCPi's `nbr_num_save`), $\ge 1$. |
| `shell_width` | float or null | `null` | | Wavefront shell width, voxels, $> 0$; `null` is the median point spacing. |
| `coarse_stride` | int | `4` | | `coarse`: solve every n-th lattice point first, $\ge 1$. |
| `coarse_level` | int | `1` | | `coarse`: pyramid level for the sub-grid solve. **Not used yet**: the MVP solves the sub-grid at full resolution (M5); another value warns. |
| `repair_passes` | int | `1` | | **Not used yet**: the repair pass is not implemented (M5); another value warns. |

### `cluster`

{py:class}`zvdvc.config.ClusterSpec`. See {doc}`/how_to/choose_tiles_and_batches`.

| field | type | default | meaning and valid values |
|---|---|---|---|
| `tile_shape` | 3 ints | `[1024, 1024, 1024]` | Tile size in voxels, $(z, y, x)$, each $\ge 1$. Must be a whole number of point-cloud chunks per axis (to within 1 %). When `plan` imports a point file, the chunk is a quarter of the tile per axis ({doc}`points_store`). |
| `devices` | list of ints or null | `null` | GPU ordinals for `zvdvc run`; `null` is every visible GPU (`CUDA_VISIBLE_DEVICES`), else device 0. `zvdvc run --devices` overrides it. |
| `gpu_memory_fraction` | float | `0.8` | In $(0, 1]$. Share of device memory the plan's memory check may use (bricks in flight plus batch scratch). The worker sizes batches from half this share of free memory. |
| `batch_points` | int or null | `null` | Points per batch; `null` sizes it from free memory on a GPU (8 192 on CPU backends). |
| `prefetch_depth` | int | `2` | $\ge 1$. Tiles whose bricks are read ahead of the one being solved. The memory check multiplies the largest tile's brick bytes by it. |
| `brick_dtype` | string | `"native"` | `native` or `float32`. **Not used**; `float32` warns. Bricks always stay in their native dtype (u8, u16 and f32 are used as stored; other dtypes are converted to float32 by the fused engine). |
| `scheduler` | string | `"dynamic"` | `dynamic` or `static`. **Not used**; `static` warns. Tiles are always shared by LPT across nodes and handed out from a dynamic queue within a node. |

### Derived value: the halo

`RunConfig.halo()` is the brick margin around a tile's points, in voxels:

$$
h = e + \texttt{disp\_max} + 2,\qquad
e = \begin{cases}
\tfrac12\,\texttt{size}\,\lVert\texttt{aspect}\rVert_2 & \text{cube (half-diagonal)}\\
\tfrac12\,\texttt{size}\,\max(\texttt{aspect}) & \text{sphere}
\end{cases}
$$

The $+2$ covers the tricubic stencil. With the defaults (sphere 80,
`disp_max` 38) $h = 80$. The spread of seeds within a tile is added at plan
time ({doc}`volumes`).

---

### Complete annotated example

Every field, with its default unless the comment says otherwise. Only
`volumes.reference`, `volumes.deformed`, `points` and `output` are required.

```yaml
volumes:
  reference: data/scan/t00.ome.zarr   # OME-Zarr, or .raw/.mhd/.npy
  deformed: data/scan/t01.ome.zarr
  array_path: "0"                      # level inside an OME-Zarr group
  raw_shape_xyz: null                  # .raw only: [x, y, z]
  raw_dtype: null                      # .raw only: e.g. "<u2" or "|u1"
  raw_header_bytes: 0                  # .raw only
  prefilter_sigma: 0.0                 # Gaussian low-pass (voxels); 0 = off, as CCPi

points: data/scan/points.roi           # .roi/.txt (imported by plan) or a .zarrvectors store
output: runs/t00_t01/results.zarrvectors
workdir: runs/t00_t01                  # default: runs/default

subvolume:
  geometry: sphere                     # cube | sphere
  size: 80.0                           # side or diameter, voxels
  n_samples: 8000                      # a cube rounds up to k^3
  aspect: [1.0, 1.0, 1.0]              # (x, y, z)
  seed: 0                              # template random seed

search:
  dof: 6                               # 3 | 6 | 12
  objective: znssd                     # sad | ssd | zssd | nssd | znssd
  interpolation: tricubic              # nearest | trilinear | tricubic
  disp_max: 38.0                       # voxels, from the seed
  rigid_trans: [0.0, 0.0, 0.0]         # (x, y, z) voxels
  basin_radius: 0.0                    # translation grid-search step; 0 = off
  threshold:                           # default: null (no threshold test)
    gray_min: 27                       # required when threshold is given
    gray_max: 127                      # required when threshold is given
    min_fraction: 0.2
  method: fagn                         # fagn (icgn: M5, refused)
  max_iterations: 20
  obj_tol: 1.0e-06
  disp_tol: 0.01                       # voxels
  report_convg_fail: true              # false = CCPi's behaviour

seeding:
  strategy: wavefront                  # rigid | wavefront | coarse (fft: M5, refused)
  start_point: null                    # (x, y, z); null = first point
  n_neighbours: 75
  shell_width: null                    # null = median point spacing
  coarse_stride: 4
  coarse_level: 1                      # not used yet (M5); other values warn
  repair_passes: 1                     # not used yet (M5); other values warn

cluster:
  tile_shape: [1024, 1024, 1024]       # (z, y, x) voxels
  devices: null                        # null = every visible GPU
  gpu_memory_fraction: 0.8
  batch_points: null                   # null = sized from free memory
  prefetch_depth: 2
  brick_dtype: native                  # not used; other values warn
  scheduler: dynamic                   # not used; other values warn

num_points_to_process: null            # null = all (in-memory solves only)
uncertainty_seeds: 0                   # >0 adds displacement_sd
```

This example loads with `RunConfig.from_yaml`. The repository's
[`configs/`](https://github.com/Andrew-Keenlyside/zvDVC/tree/main/configs)
holds three working configurations: `synthetic_small.yaml` (a 256³ synthetic
case), `ccpi_central_grid.yaml` (the iDVC example with CCPi's own test
settings) and `large_8xh100.yaml` (a 4096³ production-style run).

---

### Importing a CCPi `dvc_in` file

`RunConfig.from_ccpi(path)` reads a CCPi input file with
`zvdvc.io.ccpi.read_dvc_input` and converts it with
`run_config_from_dvc_input`
([`io/ccpi.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/io/ccpi.py)).
Lines are `key<whitespace>value`. A line starting with `#` is a comment, and
a comment starts at `###` or at a `#` after whitespace, so a `#` inside a
path is kept. Relative paths are resolved against the input file's folder
(the `zvdvc-dvc` drop-in resolves them against the working directory
instead, as CCPi does).

| CCPi key | required | RunConfig field |
|---|---|---|
| `reference_filename` | yes | `volumes.reference` |
| `correlate_filename` | yes | `volumes.deformed` |
| `point_cloud_filename` | yes | `points` |
| `output_filename` | yes | `output` = `<output_filename>.zarrvectors`, `workdir` = `<output_filename>_zvdvc` |
| `vol_bit_depth` | yes | `volumes.raw_dtype`: 8 → `"\|u1"`, 16 → `"<u2"` or `">u2"`; any other value is refused |
| `vol_endian` | no (`little`) | byte order of 16-bit data: `little` or `big` (`">u2"`); anything else is refused. Ignored for 8-bit data. |
| `vol_hdr_lngth` | no (`0`) | `volumes.raw_header_bytes` |
| `vol_wide`, `vol_high`, `vol_tall` | yes | `volumes.raw_shape_xyz` |
| `subvol_geom` | yes | `subvolume.geometry` |
| `subvol_size` | yes | `subvolume.size` |
| `subvol_npts` | yes | `subvolume.n_samples` |
| `subvol_aspect` | no (`1 1 1`) | `subvolume.aspect` |
| `subvol_thresh` | no (`off`) | `on` creates `search.threshold`; only `on` or `off` |
| `gray_thresh_min`, `gray_thresh_max` | with `subvol_thresh on` | `search.threshold.gray_min`, `gray_max` |
| `min_vol_fract` | no (`0.2`) | `search.threshold.min_fraction` |
| `disp_max` | yes | `search.disp_max` |
| `num_srch_dof` | yes | `search.dof` |
| `obj_function` | yes | `search.objective` |
| `interp_type` | yes | `search.interpolation` |
| `rigid_trans` | no (`0 0 0`) | `search.rigid_trans` |
| `basin_radius` | no (`0`) | `search.basin_radius` |
| `num_points_to_process` | no | `num_points_to_process` (`0` → `null`, all points) |
| `starting_point` | no | `seeding.start_point` |

A missing required key raises `ValueError` listing every missing key. An
unknown key is ignored with a warning. The result is validated as a YAML
config is, and an error names the `dvc_in` key (`subvol_npts: 0 is out of
range; ...`). The import also fixes the settings that make zvDVC
behave like CCPi: `seeding.strategy: wavefront` and
`search.report_convg_fail: false`. Every other field keeps its default (in
particular `prefilter_sigma: 0`, `method: fagn`, `max_iterations: 20`).

The reverse, `zvdvc.io.ccpi.write_dvc_input(cfg, path, roi_path=..., output_base=...)`,
writes a CCPi input file equivalent to a configuration, which is how the CCPi
baseline runs on zvDVC's test cases. See {doc}`ccpi_compat` for the file
formats.
