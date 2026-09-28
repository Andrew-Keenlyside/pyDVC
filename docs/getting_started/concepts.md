# Concepts

This page gives the mental model behind a zvDVC run: what is solved at each
point, and how the work is cut up so that thousands of points run at once.
It is conceptual; the {doc}`specification </spec/index>` pages hold the exact
definitions, fields and formats.

The first half (points to status codes) is the correlation method, which
zvDVC shares with iDVC and the CCPi DVC engine. The second half (tiles to
resume) is how zvDVC organises the work, which is where it differs.

---

## Points and coordinates

A **point** (CCPi's "search point") is a location in the reference scan where
a displacement is wanted. Points usually lie on a grid inside a mask; iDVC
generates them and saves them as a `.roi` text file.

* Point positions are `(x, y, z)` in **voxel units**, as in CCPi and iDVC
  `.roi` files. Voxel `i` sits at coordinate `i`.
* Volumes are indexed **`[z, y, x]`**, as numpy arrays usually are, so shapes
  and boxes are written `(z, y, x)`.

A point's result is its status, its objective value, and its full parameter
vector, of which the first three entries are the displacement $(u, v, w)$.

## Subvolumes and templates

The **subvolume** is the neighbourhood of a point that is matched between the
scans. Its geometry is a **cube** or a **sphere** of a given size (cube side
or sphere diameter, in voxels).

As in CCPi, a subvolume is represented by a fixed number of **samples**:
positions inside it, not necessarily voxel centres, where image values are
interpolated. The set of sample offsets is the **template**.

* `cube`: a regular `k × k × k` lattice, with `k` rounded up from the requested
  sample count, as CCPi does.
* `sphere`: that many random points, uniform inside the sphere, from a seeded
  random generator.

CCPi builds a new random template for every point. zvDVC builds **one
template per run** and shares it across all points, which is what lets a
batch of points be described by their centres and parameters alone. Because
the two codes draw different random samples, their results differ by a small
sampling spread; see {doc}`/getting_started/faq`.

Fields: `subvolume.geometry`, `subvolume.size`, `subvolume.n_samples`,
`subvolume.seed` in {doc}`/spec/run_config`.

## Shape functions (3, 6 or 12 DOF)

The **shape function** says how the template may move and deform to match
the deformed scan. zvDVC uses CCPi's parameterisation. A sample at offset
$d$ from the point centre $c$ maps to

$$x' = c + t + (I + E)\,R\,d$$

with translation $t = (u, v, w)$, rotation $R$ from three angles, and a
symmetric strain tensor $E$ with six components.

| DOF | parameters | the subvolume may |
|---|---|---|
| 3 | $u, v, w$ | translate |
| 6 | + three rotation angles | translate and rotate |
| 12 | + six strain components | translate, rotate and stretch or shear |

More degrees of freedom fit real deformation better inside large subvolumes,
at more cost per point.

## Objective

The **objective** scores how well the warped template matches: a sum over
samples comparing reference values $f$ with deformed values $g$. zvDVC has
CCPi's five, with identical scaling: `sad` and `ssd` (sums of absolute and of
squared differences), `zssd` (zero-mean: insensitive to a brightness offset),
`nssd` (normalised: insensitive to a contrast scale) and `znssd` (both). A
point's `objmin` is the objective at its solution; lower is better, and
`nssd` and `znssd` lie between 0 and 1.

## Interpolation

Samples fall between voxels, so values are interpolated: `nearest`,
`trilinear` or `tricubic`. zvDVC's tricubic is separable Catmull-Rom, which is
the same interpolant as CCPi's Lekien–Marsden tricubic fed with
central-difference derivatives, but evaluated straight from the image instead
of from per-voxel precomputed coefficients.

Every interpolant has a small **interpolation bias** that depends on the
fractional part of the displacement. On iDVC's example scan it is up to
±0.03 voxel. A Gaussian prefilter of about 1 voxel on both scans
(`volumes.prefilter_sigma`) removes most of it
([error-floor study](../benchmarks/2026-09-26-error-floor-case-A.md)).

## Gauss–Newton: FA-GN and IC-GN

Each point's parameters are found by **Gauss–Newton** optimisation: starting
from a seed, repeatedly linearise the residual, solve a small `ndof × ndof`
system (the normal equations) for a parameter step, and stop when the
objective or the translation stops changing. CCPi's rules are kept: stop when
the objective changes by less than `obj_tol` (1e-6) or the translation step
is smaller than `disp_tol` (0.01 voxel), after at most `max_iterations` (20).

* **FA-GN** (forward-additive, `search.method: fagn`) is CCPi's scheme and
  zvDVC's default. zvDVC computes the Jacobian analytically in the same pass
  as the interpolation, where CCPi uses `ndof + 1` objective evaluations of
  finite differences, and solves all points' systems in one batched call.
* **IC-GN** (inverse-compositional, `icgn`) builds the Hessian once per point
  from the reference and is cheaper per iteration. It is planned (M5) and not
  implemented yet.

Optionally, a translation grid search (`search.basin_radius > 0`) runs before
Gauss–Newton. {doc}`/spec/method` has the full method.

## Status codes

Every point ends with a **status**. Codes 0 to −3 are CCPi's and keep their
meaning, so exported `.disp` files read correctly in iDVC:

| code | name | meaning |
|---|---|---|
| 0 | `GOOD` | converged within `disp_max` of the seed |
| −1 | `RANGE_FAIL` | moved more than `disp_max` from the seed, or samples left the image |
| −2 | `CONVG_FAIL` | reached `max_iterations` without meeting a tolerance |
| −3 | `NOT_SEARCHED` | never attempted |
| −4 | `THRESH_FAIL` | zvDVC: too little foreground in the subvolume (`search.threshold`, CCPi's `subvol_thresh`) |
| −5 | `SINGULAR` | zvDVC: normal equations ill-conditioned, e.g. a featureless subvolume |

CCPi never reports `CONVG_FAIL`; zvDVC does unless
`search.report_convg_fail` is false. In `.disp` exports, codes −4 and −5
become −3. See {doc}`/spec/status_codes`.

## Tiles

From here on, the concepts are zvDVC's own.

The search points are stored in a zarr-vectors store (see
[the results store](#the-results-store) below), which cuts space into a grid
of **cells** (chunks). A **tile** is a block of whole cells and is the unit
of work handed to one GPU (or one CPU worker). Because no two tiles share a
cell, workers write their results without locks, and a tile can be retried or
resumed on its own.

The tile size is `cluster.tile_shape`, in voxels `(z, y, x)`. The default,
1024³, is sized for an 80 GB H100. {doc}`/how_to/choose_tiles_and_batches`
covers the choice.

## Bricks and halo

For each tile, zvDVC reads one **brick** from each scan: a dense box of the
image covering every voxel the tile's points can touch. Every point in the
tile then correlates against these bricks, already in memory, so image data is
read once per tile instead of once per point.

The **halo** is the margin around the tile's points:

```text
h = extent(template) + disp_max + 2
```

The extent is the farthest template sample from the centre (half the
diagonal for a cube, since rotations can point any sample along any axis),
and the 2 voxels cover the tricubic stencil. The deformed brick is also
shifted by the tile's median seed and grown by the spread of seeds in the
tile, so a large rigid offset moves the brick rather than fattening it.
Bricks stay in the scan's native data type (u8, u16 or f32) on the device.

## Batches

Inside a tile, points are solved in **batches** of many points at once,
ordered along a Morton (Z-order) curve so that neighbouring points read
neighbouring parts of the brick. On the GPU, one fused kernel does the warp,
interpolation, residual and normal-equation sums for every sample of every
active point in a single pass; points drop out of the active set as they
converge. The batch size comes from free device memory unless
`cluster.batch_points` sets it.

## Seeding

Gauss–Newton needs a starting displacement, the **seed**, close enough to the
answer. CCPi seeds each point from the average of its already-solved
neighbours, which forces a serial order. zvDVC offers
(`seeding.strategy`):

| mode | how | use |
|---|---|---|
| `wavefront` | points are grouped into distance shells from the start point; each shell is solved as one batch, seeded from earlier shells | CCPi parity; the default. Solves the whole problem in `zvdvc seed`, on one device |
| `coarse` | every `coarse_stride`-th grid point is solved first (in wavefront order), and that sparse field is interpolated to seed every point | production and multi-GPU: tiles become independent |
| `rigid` | every point starts at `search.rigid_trans` | small or smooth deformations, tests |
| `fft` | FFT cross-correlation per point | planned (M5), not implemented |

Today `wavefront` and `coarse` read both whole volumes into host memory, and
`zvdvc seed` refuses volumes that do not fit; use `rigid` for very large
volumes until the pyramid-level coarse pass (M5).

## Repair

A **repair** pass is planned (M5): re-seed failed points that have enough
`GOOD` neighbours from the neighbours' median, and solve them again. The
`zvdvc repair` command exists but is not implemented yet.

## The results store

Results are written to a zarr-vectors point-cloud store, conventionally
`results.zarrvectors`, that mirrors the points store **cell for cell and row
for row**. Each point's results are per-vertex attributes: `point_id`,
`status`, `objmin`, `displacement` (u, v, w), `params` (the full parameter
vector), `n_iter`, `seed`, and `displacement_sd` when the run estimates
sampling uncertainty (`uncertainty_seeds`). `zvdvc finalize` completes the
store's metadata and can export CCPi `.disp` / `.stat` files for iDVC.

Writing follows zarr-vectors' three-phase pattern: `plan` allocates the
arrays, workers write their own cells in parallel, and `finalize` rebuilds
the store's index of which cells exist. See {doc}`/spec/results_store` and
{doc}`/spec/points_store`.

## Resume

Every stage can be re-run safely:

* `zvdvc run` skips tiles whose cells are all written. A cell counts as
  written only when every result array holds it (`status` is written last),
  so a worker killed mid-cell leaves nothing that looks finished.
* After a crash, a time-out or a killed worker, run the same `zvdvc run`
  command again: it solves only the missing tiles. `run_stats.json` and
  `failed_tiles.json` in the work directory list what was missing.
* A resumed run cannot silently mix settings: the results store records the
  template, the DOF, the chunk grid and the prefilter, and `zvdvc plan`
  refuses a store written with different ones.

## Where things live

| file | written by | contents |
|---|---|---|
| `config.yaml` | you, or `zvdvc synth` | the run configuration ({doc}`/spec/run_config`) |
| `<workdir>/plan.json` | `plan` | tiles, brick boxes, costs, memory check |
| `<workdir>/points.zarrvectors` | `plan` | the points, when `points` is a `.roi` file |
| `<workdir>/seeds.npz` | `seed` (`coarse`) | the seed for every point |
| `<output>` | `plan` (allocate), `seed`, `run` | the results store |
| `<workdir>/run_stats.json`, `failed_tiles.json` | `run` | timings, missing tiles |
| `<workdir>/results.stat`, `results.disp` | `finalize` | CCPi-format summary and displacements |
