# iDVC's example dataset (case A)

This tutorial runs zvDVC on the example scan that ships with iDVC's
documentation and compares it, point by point and in wall time, with the CCPi
DVC engine run the way iDVC runs it. It is for iDVC users who want to see
how zvDVC's results relate to the ones they know, and for anyone who wants to
reproduce the published case A numbers.

zvDVC reimplements the method of [iDVC](https://github.com/TomographicImaging/iDVC)
([documentation](https://tomographicimaging.github.io/iDVC/)), the graphical
DVC application of the Tomographic Imaging / CCPi team at UKRI-STFC (authors
Laura Murgatroyd and Edoardo Pasca; Apache-2.0). iDVC does its correlation by
running the CCPi DVC engine
([TomographicImaging/DigitalVolumeCorrelation](https://github.com/TomographicImaging/DigitalVolumeCorrelation),
GPL-3.0), code initially developed by Prof. Brian K. Bay and collaborators
(Bay, Smith, Fyhrie and Saad 1999, doi:[10.1007/BF02323555](https://doi.org/10.1007/BF02323555);
Bay 2008, doi:[10.1243/03093247JSA436](https://doi.org/10.1243/03093247JSA436)).
Case A is that engine's own test case on the example data.

To use zvDVC as the engine *inside* iDVC's interface instead, see {doc}`/IDVC`.

```{note}
The full comparison is heavy: 2 × 2.4 GB of data, and CCPi runs that took
37 min (one process) and 12 min (32 processes) on a 32-core workstation. The
fetch and the full `case_a run` were not re-run for this page. Their commands
come from the code, and every CCPi number quoted is from the dated reports.
The zvDVC-only commands in [section 6](#6-zvdvc-alone-on-the-same-data) were
run for this page, on the already downloaded data, on an RTX A2000 12 GB.
```

---

## The data

"Dynamic X-ray CT of Synthetic magma for Digital Volume Correlation analysis"
by P. Lee, Y. Lavallée and B. Bay (2022), Zenodo
doi:[10.5281/zenodo.7363345](https://doi.org/10.5281/zenodo.7363345). Please
cite it if you use it.

| | |
|---|---|
| volumes | `dataset_0.npy` (reference) and `dataset_1.npy` (deformed): 1520 × 1257 × 1260 voxels (x, y, z), u8, 2.4 GB each |
| points | CCPi's `dvc_test/central_grid.roi`: 4 680 points, a 90 × 52 grid in the slice z = 630 |
| settings | CCPi's `dvc_test/dvc_input.txt`: sphere of 80 voxels, 8 000 samples, 6-DOF, ZNSSD, tricubic, `disp_max` 38, `rigid_trans` (34, 4, 0) |
| reference result | CCPi's `completed_central_grid.disp`, which holds only 5 of the 4 680 points |

The `.roi`, `dvc_input.txt` and the reference `.disp` come from the CCPi DVC
repository. The two `.npy` volumes are presumed to be the `frame_000` and
`frame_010` that CCPi's `dvc_input.txt` names; they have the same
dimensions.

## 1. Install CCPi DVC 22.0.0

The comparison needs CCPi's `dvc` executable. Use version **22.0.0**. The
newer 25.0.0 conda build has a broken tricubic path: on case S it returns
errors of about 3 voxels while reporting every point GOOD
({doc}`/benchmarks/2026-09-25-M0-M1-case-S`).

```bash
# as a conda environment ...
micromamba create -f envs/ccpi-dvc.yml
export ZVDVC_CCPI_DVC=$(micromamba run -n ccpi-dvc which dvc)
# ... or unpacked without conda (Linux); sets ZVDVC_CCPI_DVC
eval "$(scripts/get_ccpi_dvc.sh)"
```

The benchmark tools look for `dvc` in this order: `--ccpi-exe`, then
`$ZVDVC_CCPI_DVC`, then `dvc` on `PATH`.

## 2. Fetch the data

```bash
python -m zvdvc.bench.case_a fetch --data data/magma
```

`fetch` downloads the two `.npy` volumes from Zenodo record 7363345 (`--all`
takes every file of the record), with resumable, md5-checked downloads. It
then fetches CCPi's `dvc_input.txt`, `central_grid.roi` and
`completed_central_grid.disp` from GitHub into the same folder, and writes a
list of what it fetched to `fetched.json`.

## 3. CCPi's settings as a zvDVC configuration

CCPi reads a `dvc_in` file of `key value` lines. The relevant part of
`dvc_input.txt`:

```text
vol_bit_depth		8			### 8 or 16
vol_wide		1520			### width in pixels of each slice
vol_high		1257			### height in pixels of each slice
vol_tall		1260			### number of slices in the stack
subvol_geom		sphere			### cube, sphere
subvol_size		80			### side length or diameter, in voxels
subvol_npts		8000			### number of points to distribute within the subvol
disp_max		38			### in voxels, used for range checking and global search limits
num_srch_dof		6			### 3, 6, or 12
obj_function		znssd			### sad, ssd, zssd, nssd, znssd
interp_type		tricubic		### trilinear, tricubic
rigid_trans		34.0 4.0 0.0		### rigid body offset of target volume, in voxels
```

{py:meth}`zvdvc.config.RunConfig.from_ccpi` maps it onto a run configuration
({doc}`/spec/ccpi_compat` lists the mapping):

```python
from zvdvc import RunConfig

cfg = RunConfig.from_ccpi("data/magma/dvc_input.txt")
print(cfg.subvolume)
print(cfg.search)
```

```text
SubvolumeSpec(geometry='sphere', size=80.0, n_samples=8000, aspect=(1.0, 1.0, 1.0), seed=0)
SearchSpec(dof=6, objective='znssd', interpolation='tricubic', disp_max=38.0, rigid_trans=(34.0, 4.0, 0.0), basin_radius=0.0, threshold=None, method='fagn', max_iterations=20, obj_tol=1e-06, disp_tol=0.01, report_convg_fail=False)
```

`report_convg_fail=False` reproduces CCPi, whose solver never reports
`Convg_Fail`.

The file names in `dvc_input.txt` (`f000_crop/frame_000.mdh`, …) are not in
the Zenodo record. {py:func}`zvdvc.bench.case_a.case_config` swaps in the
`.npy` volumes. They are stored x-first, so it writes C-ordered `.raw` copies
once (about 4.8 GB, into the output folder), which both codes then read.

## 4. Run the comparison

```bash
python -m zvdvc.bench.case_a run --data data/magma --out runs/case_A --backends fused cpu --ccpi-processes 32
```

This is the command behind the published numbers, on a 32-core machine with
a GPU. On a machine without one, use `--backends cpu`. The flags:

| flag | default | meaning |
|---|---|---|
| `--data` | `data/magma` | where `fetch` put the data |
| `--out` | `runs/case_A` | output folder: `config.yaml`, the `.raw` copies, every run's results, `report.md`, `report.json` |
| `--ccpi-exe DVC [DVC …]` | `$ZVDVC_CCPI_DVC`, else `dvc` on `PATH` | one or more CCPi executables; pass 22.0.0 and 25.0.0 to time both |
| `--ccpi-processes N` | number of cores | processes for the split CCPi run |
| `--backends` | `fused` with a GPU, else `cpu` | zvDVC backends to run |
| `--no-cli` | off | skip the end-to-end CLI runs |
| `--max-ccpi-points N` | all | run CCPi on a sample of the points only (not used as a reference then) |
| `--reuse-ccpi` | off | re-time finished CCPi runs in `--out` from their `.stat` files instead of repeating them |

`run` writes `config.yaml` into `--out`, then times these runs on the same
machine, points and settings
([`bench/compare_ccpi.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/bench/compare_ccpi.py)):

| run | what it is |
|---|---|
| CCPi, iDVC mode | one `dvc` process with OpenMP on every core, as iDVC launches it: the time an iDVC user waits |
| CCPi, N processes | N single-thread `dvc` processes on disjoint subsets of the points: the most a node gets from CCPi without code changes |
| zvDVC parity | both volumes in memory, CCPi's point order (wavefront), in one process; timed from reading the volumes to the last point, with kernel compilation done beforehand |
| zvDVC CLI | `zvdvc plan`, `seed`, `run`, `finalize` as four fresh processes, start-up and imports included, results in the zarr-vectors store |

For the first backend it also solves the grid a second time, untimed, with
the next template seed. This measures zvDVC's disagreement with itself, which
the revised accuracy criterion uses (below).

`--reuse-ccpi` is worth knowing: after one full run you can change zvDVC and
re-run the comparison without waiting for CCPi again.

## 5. Read the report

`report.md` has two tables and a verdict. The first times every run:

| run | points | wall time | pt/s | vs iDVC | vs best CCPi |
|---|---|---|---|---|---|
| CCPi ccpi-dvc-22.0.0: iDVC (1 process, all cores) | 4680 | 2230.6 s | 2.1 | 1.0x | 0.3x |
| CCPi ccpi-dvc-22.0.0: 32 processes x 1 thread | 4680 | 711.1 s | 6.6 | 3.1x | 1.0x |
| zvDVC fused: parity (in memory, CCPi order) | 4680 | 1.6 s | 2928.2 | 1395.7x | 444.9x |
| zvDVC fused: CLI end to end | 4680 | 4.3 s | 1098.4 | 523.5x | 166.9x |
| … | | | | | |

The second compares each zvDVC run with each CCPi run, point by point, on
points matched by `point_id`:

| comparison | points | median \|du\| | p95 \|du\| | status agreement | GOOD in both | RMSE (x, y, z) | Q1 |
|---|---|---|---|---|---|---|---|
| zvDVC fused: parity … vs CCPi …: iDVC (1 process, all cores) | 4680 | 0.0512 | 0.1517 | 98.9 % | 4628 | 0.0482, 0.0421, 0.0457 |  |
| … (interior) | 4628 | 0.0512 | 0.1517 | 100.0 % | 4628 | 0.0482, 0.0421, 0.0457 | FAIL |
| … (edge) | 52 | nan | nan | 0.0 % | 0 | nan, nan, nan |  |
| … vs reference .disp | 5 | 0.0428 | 0.0458 | 100.0 % | 5 | 0.0143, 0.0242, 0.0273 |  |

(Excerpts from the run of 2026-09-26, `runs/case_A/report.md`. That run
predates the rename, so its file labels the rows "pyDVC"; current code writes
"zvDVC", as shown.)

How to read it:

* **Interior and edge.** An *edge* point is one whose subvolume (template
  extent plus the 2-voxel tricubic stencil) leaves the image at CCPi's
  displacement. zvDVC marks such points `RANGE_FAIL`. CCPi does not test
  samples against the image. It reads a box around each point row by row
  without clipping it, so past the image edge it correlates wrapped rows or
  unset memory, and reports GOOD. The accuracy verdict is judged on interior
  points only.
* **Q1 column.** The MVP's original criterion: median |du| ≤ 0.05 voxel,
  p95 ≤ 0.2, status agreement ≥ 98 %.
* **Revised Q1.** Current code adds a line per comparison: every per-axis
  mean difference ≤ 0.01 voxel, and the spread of the difference no more than
  1.25 × zvDVC's own seed-to-seed spread. The reason is in section 7.
* **Against the 32-process CCPi run** the agreement is poor (p95 about
  16 voxels). That is CCPi's problem, not zvDVC's; see section 7.

`report.json` holds the same numbers, the environment, the git commit, every
run's detail (read and solve times, mean iterations) and a status confusion
table per comparison.

## 6. zvDVC alone on the same data

Without CCPi you can still run zvDVC on case A and compare it with a CCPi
`.disp` you already have. `case_config` builds the configuration from CCPi's
settings, as `case_a run` does:

```python
import dataclasses
from pathlib import Path
from zvdvc.bench.case_a import case_config

cfg = case_config(Path("data/magma"), Path("runs/case_A"))   # writes the .raw copies once
cfg = dataclasses.replace(cfg, output="caseA/results.zarrvectors", workdir="caseA/work")
cfg.to_yaml("caseA.yaml")
```

The in-memory solve, as `zvdvc solve` runs it (CCPi's order, both volumes in
memory):

```bash
zvdvc solve caseA.yaml --backend fused --disp --quiet
```

```text
caseA/work/results.npz: 4680 points in 2.1 s (2211.4 pt/s), status counts {-1: 52, 0: 4628}
```

The time runs from reading the volumes to the last point. This was a second
run, with the volumes in the page cache and the kernels already compiled; the
first run, reading from disk cold and compiling, took 8.7 s. The report's
1.6 s is the same solve timed with compilation done beforehand.

Against CCPi 22.0.0's result from the report run (iDVC mode, 32 threads;
`case_a run` names these files `ccpi_<package>/t<threads>_p<processes>.disp`):

```bash
zvdvc compare caseA/work/results.npz runs/case_A/ccpi_ccpi-dvc-22.0.0/t32_p1.disp
```

```text
points 4680, GOOD 4628 (98.89 %)  [GOOD 4628, RANGE_FAIL 52]
rmse (x, y, z)  0.0482  0.0421  0.0457
bias (x, y, z)  -0.0027  -0.0017  -0.0005
|error| median 0.0512  p95 0.1517  p99 0.2379
status agreement 98.89 %
```

These are the report's figures to the last digit. The 52 `RANGE_FAIL` points
are the edge points. `zvdvc compare` does not split interior from edge, so
its errors are over the 4 628 points GOOD in both codes, which are exactly
the interior ones here.

The tiled CLI stages give the same result:

```bash
zvdvc plan caseA.yaml && zvdvc seed caseA.yaml && zvdvc run caseA.yaml && zvdvc finalize caseA.yaml --disp
zvdvc compare caseA/results.zarrvectors runs/case_A/ccpi_ccpi-dvc-22.0.0/t32_p1.disp
```

```text
{'tiles': 1, 'points': 4680, 'memory': {'brick_bytes': 498631490, 'in_flight_bytes': 498631490, 'batch_bytes': 339580800, 'device_bytes': 9983806668, 'fits': True}, 'plan': 'caseA/work/plan.json'}
{'strategy': 'wavefront', 'points': 4680, 'seconds': 2.099516296060756}
device 0: 0 tiles solved, 0 written, 1 already written, 0 points, 0.00 GB read, compute 0.0 s, I/O wait n/a, status counts {}, 0 errors
RunSummary(n_points=4680, seconds=1.0903416230576113, counts={-1: 52, 0: 4628})
points 4680, GOOD 4628 (98.89 %)  [GOOD 4628, RANGE_FAIL 52]
…
|error| median 0.0512  p95 0.1517  p99 0.2379
```

The configuration makes the whole volume one tile with `wavefront` seeding,
so `seed` does the solving (see {doc}`/tutorials/synthetic_first_run`,
section 4). The brick is small because the grid is a single slice: the
reference brick covers the slice plus the halo, not the whole volume.
`caseA.yaml` holds relative paths, so run every stage from the same
directory: the results store records the volumes' resolved paths, and
`plan` or `run` from elsewhere refuses to resume into it.

## 7. Measured results

From {doc}`/benchmarks/2026-09-26-case-A-real` (2026-09-26, workstation
`msm12`: 32 cores, 251 GB RAM, RTX A2000 12 GB; CCPi `dvc` 22.0.0).

**Speed.**

| run | wall time | points/s | vs iDVC |
|---|---|---|---|
| CCPi, iDVC mode (1 process, 32 threads) | 2 231 s (37 min) | 2.1 | 1× |
| CCPi, 32 processes × 1 thread | 711 s | 6.6 | 3.1× |
| zvDVC fused (GPU), parity | 1.6 s | 2 928 | 1 396× |
| zvDVC fused (GPU), CLI end to end | 4.3 s | 1 098 | 524× |
| zvDVC cpu (32 cores), parity | 7.6 s | 615 | 293× |
| zvDVC cpu (32 cores), CLI end to end | 17.1 s | 274 | 131× |

Most of the gain does not need a GPU: zvDVC's CPU engine on the same 32
cores is 293× faster than CCPi in iDVC mode. The CLI rows include
interpreter start-up, imports and writing the store, which dominate at 4 680
points.

**Agreement** (interior points, GOOD in both):

| | points | median \|Δu\| | p95 \|Δu\| | status agreement | mean Δu (x, y, z) |
|---|---|---|---|---|---|
| zvDVC vs CCPi, interior | 4 628 | 0.0512 | 0.152 | 100.0 % | −0.0027, −0.0017, −0.0005 |
| zvDVC vs CCPi's 5-point reference `.disp` | 5 | 0.0428 | 0.0458 | 100 % | |

* **Original Q1: a marginal fail.** Status and p95 pass; the median misses
  0.05 by 0.0012 voxel.
* **Revised Q1: pass.** Mean differences are at most 0.003 voxel per axis,
  and the spread ratio against zvDVC's own seed-to-seed spread is
  (1.03, 0.98, 1.02).
* **Why the criterion was revised.** The two codes draw each subvolume's 8 000
  sample points at random, from different generators. zvDVC run twice with
  only the template seed changed disagrees with itself by median 0.051 and
  p95 0.15, the same as with CCPi. The original median criterion cannot be
  met even by zvDVC against itself at these settings.
* **Edge points.** 52 points reach past x = 1519 at the measured ~33-voxel
  displacement. zvDVC: `RANGE_FAIL`; CCPi: GOOD, computed from wrapped or
  undefined data.
* **Splitting CCPi.** The 32-process CCPi run is valid as a timing but not as
  a reference. Each process starts its own wavefront on a subset of the grid,
  so points lose the neighbours that would seed them. 1 029 of 4 680 points
  (22 %) differ from CCPi's single-process run by more than 1 voxel, all
  reported GOOD. Anyone splitting CCPi runs over cores should check the
  seeding.

**Error floor on this scan**, from {doc}`/benchmarks/2026-09-26-error-floor-case-A`
(same settings, per axis):

| error | size (voxel) | depends on |
|---|---|---|
| interpolation bias | up to ±0.03 | the displacement's fractional part |
| random error from image noise | 0.005 | noise, sample count |
| random error from the choice of sample points | 0.032 | sample count, as 1/√n (0.016 at 32 000 samples) |

The last one dominates, and it accounts for the whole difference from CCPi.
{doc}`/tutorials/strain_uncertainty` shows how to get it per point
(`uncertainty_seeds`) and how to remove most of the bias
(`prefilter_sigma`).

The local check suite repeats the central grid on every `zvdvc check full`
run, against the committed baseline for this machine; see
{doc}`/benchmarks/index`.

## What this does not show

* Only one dataset, and only a single slice of points.
* Only one workstation GPU. Tiled multi-GPU runs on 8 × H100 are implemented
  but not yet measured.
* GPUDirect Storage: not wired in. Every time here uses host reads.
