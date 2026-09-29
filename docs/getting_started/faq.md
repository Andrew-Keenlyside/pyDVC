# Frequently asked questions

Short answers to the questions people ask first, with links to the pages
that hold the detail. Numbers come from the measured reports under
{doc}`/benchmarks/index`.

---

## Running zvDVC

### Do I need a GPU?

No. Every stage runs on the CPU as well. The `cpu` backend runs the same
fused Gauss–Newton step on all cores with numba (install the `[cpu-fast]`
extra); without numba, zvDVC falls back to the `numpy` reference, which is
much slower. The default backend is `fused` when a GPU is present, otherwise
`cpu`, otherwise `numpy`; `--backend` overrides it.

On iDVC's example dataset (4 680 points) on one workstation, the in-memory
solve took 1.6 s on an RTX A2000 GPU and 7.6 s on 32 CPU cores, against
37 min for CCPi DVC as iDVC runs it
([case A report](../benchmarks/2026-09-26-case-A-real.md)). Most of the gain
over CCPi comes from restructuring the work, not from the GPU; the GPU matters
more as point counts, sample counts and DOF grow.

### How do I start from an iDVC / CCPi setup?

Three ways, depending on how much of iDVC you want to keep:

* **Keep iDVC.** Run iDVC with zvDVC as its engine through the `zvdvc-dvc`
  drop-in; iDVC's interface is unchanged ({doc}`/IDVC`).
* **Keep the `dvc_in` file.** `zvdvc ccpi dvc_config.txt` reads CCPi's
  parameter file and writes CCPi's `.disp` and `.stat` files. From Python,
  `RunConfig.from_ccpi("dvc_config.txt")` gives a {py:class}`zvdvc.config.RunConfig`
  to run through the tiled pipeline.
* **Write a `config.yaml`.** CCPi's `dvc_in` keys map one to one onto the
  run configuration ({doc}`/spec/run_config`). The `points` field accepts a
  `.roi` file, which `zvdvc plan` imports.

### What input formats are supported?

Reference and deformed volumes can be OME-Zarr v3 arrays, or CCPi/iDVC flat
files (`.raw` with a shape, data type and header size; `.mhd`; `.npy`), read
directly. For large data, convert once to sharded OME-Zarr:

```bash
zvdvc convert dataset_0.npy data/ref.ome.zarr --chunk 128 --shard 1024
```

`zvdvc convert` also reads `.tif` / `.tiff` stacks with the `[tiff]` extra.
Points come as a CCPi `.roi` file or a zarr-vectors point-cloud store. See
{doc}`/spec/volumes` and {doc}`/spec/points_store`.

### How do I resume an interrupted run?

Run the same `zvdvc run CONFIG` command again. It skips every tile whose
results are already written and solves only the rest. A cell counts as
written only when all its result arrays are, so a worker killed mid-write
leaves nothing that looks finished. After a run, `run_stats.json` and
`failed_tiles.json` in the work directory list any tiles still missing, and
`zvdvc run` exits with status 3 when there are any; `zvdvc finalize` refuses
until they are written (or `--allow-partial` exports them as
`NOT_SEARCHED`). This
has been tested by killing one of two workers with `kill -9`: the resubmitted
job solved only the 4 missing tiles and wrote a bit-identical store.

Every other stage is also safe to re-run. The results store and `plan.json`
record a fingerprint of the run: the settings that change results, the
volumes (resolved path, shape, dtype, and file size and modification time, or
a digest of the Zarr metadata) and the points. `zvdvc plan`, `seed` and `run`
refuse to add results made with anything else, so a resumed run cannot mix
settings or inputs. Moving or copying the volumes, or running from another
directory with relative paths, therefore also refuses a resume: plan again
with a new `output` and `workdir`. A store written before fingerprints were
recorded gives a warning instead ({doc}`/spec/results_store`).

### How big can the data be?

The tiled `zvdvc run` reads one brick per tile, so the volume size is limited
by storage, not by memory. A 1024³ tile of u16 data reads about 5 GB of
bricks (reference and deformed, with the halo). The limits today are elsewhere:

* **Seeding.** The `wavefront` and `coarse` passes hold both whole volumes in
  host memory, and `zvdvc seed` refuses volumes that need more than 80 % of
  the available memory. For larger volumes use `rigid` seeding (as
  `configs/large_8xh100.yaml` does for a 4096³ pair) until the pyramid-level
  coarse pass lands (M5).
* **What has been measured.** The largest real case is iDVC's example
  (two 2.4 GB u8 volumes), including a 285 480-point grid on it that took
  18.7 s on the RTX A2000. The 2048³ and 4096³ synthetic cases on 8 × H100
  are prepared but not yet measured ({doc}`/CLUSTER`).

## Accuracy and compatibility

### Will my results match iDVC / CCPi exactly?

No, and they should not be expected to. Both codes draw each subvolume's
sample points at random, from different generators. On real data the best-fit
warp depends on which points are drawn, so any two correct runs differ by a
**sampling spread**. On iDVC's example at CCPi's settings (sphere 80,
8 000 samples, 6-DOF):

* zvDVC against CCPi: median |Δu| 0.051 voxel, 95th percentile 0.152, status
  agreement 100 % on interior points, mean difference at most 0.003 voxel per
  axis;
* zvDVC against itself with only the template seed changed: median 0.051,
  95th percentile 0.15.

The disagreement with CCPi is fully explained by the sampling spread: there
is no systematic difference. The spread is about 0.03 voxel per axis at 8 000
samples and falls as $1/\sqrt{n}$: 0.016 voxel at 32 000 samples
([error-floor study](../benchmarks/2026-09-26-error-floor-case-A.md)).

zvDVC can report this uncertainty per point: set `uncertainty_seeds: 2` in the
run configuration, and each `GOOD` point is repeated with two other template
seeds and the per-axis spread stored as `displacement_sd`.
`zvdvc strain` carries it into a strain uncertainty.

### Why are some points `RANGE_FAIL` in zvDVC but `GOOD` in CCPi?

They sit near the image edge, where the deformed subvolume reaches outside the
volume. CCPi reads a fixed box around each point without clipping it to the
image: past the last column a read continues into the next row, and past the
last slice the buffer holds uninitialised memory. It reports those points as
`GOOD`. zvDVC flags them `RANGE_FAIL` rather than correlate against data that
is not there. On iDVC's example this affects 52 of 4 680 points. The
comparison tools report such points separately.

### Can I view zvDVC's results in iDVC?

Yes. `zvdvc finalize CONFIG --disp` writes `results.disp` in CCPi's layout
into the work directory (points that are not `GOOD` get zero displacement,
as in CCPi), with a zvDVC `results.stat` summary; iDVC's results viewer reads
the same values from the `.disp` as from CCPi's. If you run iDVC with the `zvdvc-dvc` drop-in,
iDVC finds the files where it expects them ({doc}`/IDVC`). CCPi's `strain`
program reads zvDVC's `.disp` unchanged, and `zvdvc strain` computes the same
strain with an uncertainty per point.

The results store itself is a zarr-vectors point cloud, readable by
zarr-vectors tooling; `zvdvc finalize --pyramid` builds a multiscale pyramid
for viewers such as Neuroglancer.

### Which CCPi DVC version should I compare against?

**22.0.0.** The 25.0.0 conda build's tricubic interpolation is broken
([M0–M1 report](../benchmarks/2026-09-25-M0-M1-case-S.md)), so it is not a
valid reference. `envs/ccpi-dvc.yml` and `scripts/get_ccpi_dvc.sh` both
install 22.0.0 ({doc}`/getting_started/installation`).

One more caution from the case A measurements: splitting a CCPi run into
several processes on subsets of the grid is valid for timing but not as a
reference. Each process starts its own neighbour seeding, and 22 % of points
converged to wrong solutions that CCPi still reported as `GOOD`.

## Design choices

### Why do u8 and u16 volumes stay in their native type?

Bricks are copied to the device and held there in the scan's own data type
(u8, u16 or f32), and the kernels convert each value to float in registers
as they read it. A u16 brick is half the size of a float32 one, so it takes
half the device memory and half the host-to-device copy, and u8 rows can be
read as packed 32-bit words. Converting up front would buy no accuracy: in the
error-floor study, a prefiltered u8 volume rounded back to u8 was as good as
float32 (the same maximum bias and sampling uncertainty). `cluster.brick_dtype`
exists in the configuration, but only `native` is used today; another value
warns. Big-endian data is converted to the host's byte order as bricks are
read.

### Does zvDVC use GPUDirect Storage?

Not yet. GPUDirect Storage (GDS) is not wired in and has not been measured.
Every published zvDVC number uses host decode plus one pinned host-to-device
copy per brick. Adding device reads for points, results and image bricks, and
measuring host against GDS, is roadmap milestone M6. See
{doc}`/how_to/gpudirect_storage`.

### Why zarr-vectors for points and results?

Its spatial chunk grid doubles as zvDVC's work partition: a tile is a set of
whole cells, so a tile's input and output rows live in the same cells, and no
two workers ever write the same cell. It gives each tile its points in one
pooled read, lets workers write results in parallel without locks, and makes
the results viewable with zarr-vectors tooling. See {doc}`/ARCHITECTURE`,
section 5.
