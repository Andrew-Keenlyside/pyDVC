# Public validation datasets

iDVC's example data (case A, {doc}`idvc_test_dataset`) has no ground truth: it
shows agreement with iDVC, not accuracy. This page runs zvDVC and iDVC's engine
on two public benchmark collections. Both codes get the same volumes, points
and settings, and wherever the answer is known, both are scored against it.

* **DVC Challenge 1.0**, system XCT1: real X-ray CT scans of a foam cylinder,
  repeated with no motion (the noise floor) and after 1 mm stage moves.
* **DVC Challenge 2.0**, J. Yang's synthetic bead volumes: uniform translations
  of 0 to 1 voxel, and uniaxial stretches of 5 to 30 %, all with a known field.

Everything was run on 2026-09-29, on the development workstation (32 cores,
RTX A2000 12 GB): 27 cases, 28 918 points.

## Summary

| Check | Result |
|---|---|
| Agreement with iDVC's engine, 23 cases where both codes succeed | median difference 0.00005–0.0006 voxel per case; 95th percentile at most 0.0023; mean difference at most 7 × 10⁻⁵ per axis |
| Noise floor, XCT1 repeat scans (cube 24–48) | SD 0.0095–0.0106 voxel, **the same in both codes** to 0.0001 |
| 1 mm stage moves, XCT1 | every point GOOD in both codes; both measure the same scanner distortion pattern, differing by 0.0002 voxel (median) |
| Known translations of 0–1 voxel, beads | mean error at most 0.0003 voxel, RMSE 0.004, **in both codes** |
| Known stretches of 5 and 10 %, beads | RMSE 0.004–0.006 voxel in both; strain from a fit to the field within 1.3 × 10⁻⁵ of the truth |
| Stretches of 15–30 %, beads | **both codes fail** at iDVC's settings. iDVC's engine reports 55–91 % of points GOOD with wrong answers; zvDVC reports 3–5 % |
| Robustness | iDVC's engine crashed (segmentation fault) in 10 runs: points near the image edge, and large stretches with its grid search on. zvDVC completed every run |

Where both codes succeed, zvDVC gives iDVC's answer. Where the truth is known,
both are equally accurate. They differ in what they report when the solve
fails: iDVC's engine never reports a failure to converge, and zvDVC does.

## Settings

| | |
|---|---|
| iDVC's engine | CCPi DVC 22.0.0 (`dvc`), from a `dvc_in` written for each case. DVC Challenge 1.0: launched as iDVC launches it (one process on every core) and again on one thread. The bead series: one thread only (see [Run time](#run-time)) |
| zvDVC | in-memory, CCPi-parity mode (wavefront seeding in CCPi's point order), GPU engine |
| subvolume, search | iDVC's defaults: cube subvolumes, 12-DOF, ZNSSD, tricubic. 8 000 samples per subvolume (a 20³ grid, as in iDVC's example). `disp_max` 10 voxels |
| points | regular grids. XCT1: 24 voxels apart, inside the foam in both scans. Beads: 32 voxels apart, at least 30 voxels from each face |
| seeding | the grid point nearest the volume centre first. For the moves, `rigid_trans` is the whole-volume integer offset (what iDVC's registration panel gives) |

A cube subvolume samples the same grid in both codes, so each code sees the same
image data. Any difference between the two is then the solver's, not the
random sample points' ({doc}`/benchmarks/2026-09-29-idvc-comparison`).

## DVC Challenge 1.0, XCT1

Croom et al., *Exp. Mech.* 61:395–410 (2021),
[doi:10.1007/s11340-020-00653-x](https://doi.org/10.1007/s11340-020-00653-x);
data [doi:10.18130/V3/1UOVKO](https://doi.org/10.18130/V3/1UOVKO) (CC0).
System XCT1 is an Xradia microCT-200 at 17.86 µm voxels. The volumes are
350 × 300 × 504 u16 slice stacks of a syntactic-foam cylinder. There are seven
scans: a reference, two repeats in place, two after a 1 mm move along the
rotation axis (axial), and two after a 1 mm move across it (radial).

### Repeat scans: the noise floor

Nothing moved between these scans, so any measured scatter is noise.

```{figure} figures/public_noise_floor.png
:alt: Two line charts of displacement standard deviation against cube subvolume size (24, 32, 48). zvDVC and iDVC's engine lie on top of each other: SD 0.0106, 0.0095, 0.0095 voxel; about an affine fit 0.0068, 0.0051, 0.0050. zvDVC with a Gaussian prefilter is higher: 0.0138, 0.0117, 0.0116 and 0.0106, 0.0072, 0.0058.
:figclass: zv-figure
:width: 100%

Scatter of the displacement over about 900 points, averaged over x, y, z and
both repeats. The two codes give the same noise floor at every subvolume size.
Right: the scatter left after removing each code's own affine fit, which takes
out small real motions of the sample between scans.
```

| cube | SD, zvDVC | SD, iDVC's engine | SD about affine fit, zvDVC | SD about affine fit, iDVC's engine |
|---:|---:|---:|---:|---:|
| 24 | 0.0106 | 0.0106 | 0.0068 | 0.0068 |
| 32 | 0.0095 | 0.0095 | 0.0051 | 0.0051 |
| 48 | 0.0095 | 0.0095 | 0.0050 | 0.0050 |

All 910 points are GOOD in both codes, in all six runs. The sample moved
slightly between scans: both codes measure a mean of (0.070, 0.011, 0.021)
voxel for repeat 1 and (0.107, 0.020, 0.019) for repeat 2 (up to 2 µm),
identical to the third decimal. For scale, the challenge's own noise analysis
used a third code, ALDVC local DVC with 17-voxel subsets, on repeat 2
(DVC Challenge 2.0 archive, `noise_analyses/xct_data/DVC1_S1`). It reports SD
0.013–0.014 voxel, consistent with 0.0106 at 24 voxels here.

zvDVC's Gaussian prefilter (`volumes.prefilter_sigma` = 1), which removes
interpolation bias ({doc}`ground_truth`), **raises** this noise floor on this
scan, by 16–56 % (averaged over both repeats). It smooths away some of the foam's fine texture. Use it where
bias matters more than scatter, and check on repeat scans of the material.

### 1 mm stage moves

The nominal move is 1 mm, 56.0 voxels. Whole-volume phase correlation gives
58 voxels (axial) and −57 or −58 (radial). These integer offsets seeded both
codes.

| move | interior points | GOOD, zvDVC / iDVC's engine | mean displacement (x, y, z), both codes | = mm at 17.86 µm | zvDVC − iDVC's engine, median \|Δu\| |
|---|---:|---|---|---:|---:|
| axial 1 | 999 | 999 / 999 | (0.265, 0.055, 57.859) | 1.033 | 0.00016 |
| axial 2 | 1 000 | 1 000 / 1 000 | (0.285, 0.041, 57.855) | 1.033 | 0.00016 |
| radial 1 | 1 087 | 1 087 / 1 087 | (−57.541, −0.104, 0.079) | 1.028 | 0.00023 |
| radial 2 | 1 088 | 1 088 / 1 088 | (−57.574, −0.111, 0.034) | 1.028 | 0.00019 |

Both codes measure the moves as about 3 % longer than nominal. The cause is
either the stage or the stated voxel size; the moves alone cannot tell which.
The displacement also varies across the sample, by up to ±0.15 voxel along the
move. Croom et al. report this as the scanners' spatial distortion, which is
real and not DVC error. Both codes measure the same pattern:

```{figure} figures/public_axial_move.png
:alt: Maps of the central x-z plane for the first axial move. The displacement along the move, minus its mean, is about +0.15 voxel at the top and bottom of the sample and -0.1 in the middle, and identical in iDVC's engine and zvDVC; the difference maps are unstructured noise of about 0.001 voxel.
:figclass: zv-figure
:width: 100%

The first axial move, displacement minus its mean, on the central plane. The
distortion pattern (±0.15 voxel along the move) is the same in both codes.
Their difference, on a scale a hundred times smaller, is unstructured.
```

About each code's affine fit, the scatter is 0.013–0.026 voxel per axis, and
0.073 along the axial moves. That is several times the noise floor: the
distortion is not affine. The two codes agree on these figures to 0.0001.

## DVC Challenge 2.0, synthetic beads

Tong et al. (2026, preprint) [doi:10.21203/rs.3.rs-9683321/v1](https://doi.org/10.21203/rs.3.rs-9683321/v1);
data NIST [doi:10.18434/mds2-4129](https://doi.org/10.18434/mds2-4129). These
are J. Yang's synthetic fluorescent-bead volumes, 512 × 512 × 192 u8. The beads
are dense and elongated along z, with a 1/e feature size of 2.4 voxels.

The archive ships the volumes but not the imposed fields' values. They were
recovered independently of either code:

* **Translations** (`Yang_Beads_S2`): files 1001–1011 are the reference, 1000,
  shifted along y by 0.0, 0.1, …, 1.0 voxel. Whole-volume phase correlation
  places the steps; both codes' mean error against the nominal values is below
  0.0003 voxel.
* **Stretches** (`Yang_Beads_S3`): files 1002–1007 are the reference, 1001,
  stretched uniaxially along y about y = 255, `v = ε (y − 255)`. ε was found by
  registering the stretched reference to each volume (normalised correlation
  0.95–0.97): 0.05, 0.10, …, 0.30, each within 10⁻⁵ of the step.

### Translations

```{figure} figures/public_translation.png
:alt: Two line charts against the imposed translation, 0 to 1 voxel. Left, mean error in y: within plus or minus 0.00025 voxel for all three series, with zvDVC and iDVC's engine on top of each other. Right, SD of the error: 0.0024 at zero shift, 0.0036 to 0.0043 otherwise for zvDVC and iDVC's engine (coincident); 0.0020 and 0.0028 to 0.0030 for zvDVC with a prefilter.
:figclass: zv-figure
:width: 100%

Error against the imposed translation, 1 125 points per step. The two codes
coincide. Neither has a measurable bias (the left panel's scale is 10⁻⁴ voxel).
```

Both codes: every point GOOD, mean error at most 0.0003 voxel per axis, RMSE
0.0035–0.0043 per axis (0.0024 at zero shift). With zvDVC's prefilter the RMSE
is 0.0028–0.0030.

There is no sign here of the ±0.03 voxel S-shaped bias that tricubic
interpolation shows on the case A scan ({doc}`ground_truth`). Either these bead
images are smooth enough for the cubic interpolant, or the shifted volumes were
made with a similar interpolant; the archive does not say how they were made.
This series therefore cannot show interpolation bias, in either code.

### Stretches

At 12-DOF a subvolume can follow a homogeneous stretch exactly, so the limit
here is seeding. Both codes seed each point from an already-solved neighbour's
translation. With points 32 voxels apart, a stretch of ε puts the true answer
32 ε voxels from that seed: 1.6 voxels at 5 %, 4.8 at 15 %, 9.6 at 30 %.

```{figure} figures/public_stretch.png
:alt: Two line charts against the imposed stretch (5 to 30 %). Left, share of interior points GOOD and within 0.5 voxel of the truth: 100 % at 5 % for both codes, then falling to 25 % (zvDVC) and 45 % (iDVC's engine) at 15 %, and to 9 % for both by 25 %. With basin_radius 4, zvDVC stays at 100 % to 15 % and 92 % at 20 %, then 32 and 24 %; iDVC's engine with basin_radius 4 is 100 % to 15 % and crashes at 20 to 30 %. Right, share reported GOOD but wrong: iDVC's engine 55, 78, 91 and 91 % from 15 to 30 %; zvDVC 3 to 5 %; with basin_radius 4, 0 to 2 %.
:figclass: zv-figure
:width: 100%

Left: the share of points each code gets right. Right: the share it reports
GOOD but gets wrong (more than 0.5 voxel from the truth). Dotted: CCPi's
translation grid search (`basin_radius` = 4), which iDVC fixes at 0.
```

| stretch | interior points | right: zvDVC / iDVC's engine | GOOD but wrong: zvDVC / iDVC's engine | with `basin_radius` 4, right: zvDVC / iDVC's engine |
|---:|---:|---|---|---|
| 5 % | 975 | 975 / 975 | 0 / 0 | 975 / 975 |
| 10 % | 975 | 960 / 975 | 0 / 0 | 975 / 975 |
| 15 % | 975 | 247 / 436 | 27 / 539 | 975 / 975 |
| 20 % | 825 | 75 / 184 | 35 / 641 | 760 / crashed |
| 25 % | 825 | 75 / 75 | 38 / 750 | 268 / crashed |
| 30 % | 825 | 75 / 75 | 36 / 750 | 197 / crashed |

* **Up to 10 %, both codes are right and equally accurate.** RMSE is 0.004–0.006
  voxel per axis. The strain from an affine fit to either code's field is
  within 1.3 × 10⁻⁵ of the truth. At 10 %, zvDVC reports 15 points
  `CONVG_FAIL`: the iteration limit was reached before the tolerance was met.
  iDVC's engine reports these points GOOD, and they are right. CCPi's solver
  never reports a failure to converge; `search.report_convg_fail: false` makes
  zvDVC do the same.
* **From 15 %, both codes lose the field.** Only the points near y = 255, where
  the displacement is small, stay right. iDVC's engine reports every point GOOD
  regardless, and 55–91 % of the interior points are wrong. zvDVC marks most
  failures `CONVG_FAIL`, but 27–38 points (3–5 %) are still reported GOOD
  with wrong answers.
* **CCPi's grid search** (`basin_radius`) rescues 15 % in both codes, and 20 %
  (92 % right) in zvDVC. iDVC always writes `basin_radius 0`, so iDVC users
  cannot turn it on. With it on, iDVC's engine crashed on the 20–30 % stretches
  after 94–289 points; the crash point changes from run to run.

(run-time)=
## Run time

| | points per case | iDVC's engine, as iDVC launches it | iDVC's engine, one thread | zvDVC |
|---|---:|---:|---:|---:|
| XCT1, 10 cases | 910–1 125 | 55–615 s | 34–63 s | 0.1–0.5 s |
| beads, 17 cases | 1 125 | not run | 73–286 s | 0.2–2.3 s |

zvDVC's times are the in-memory solve on the GPU, excluding start-up. The
all-core times varied widely because other jobs shared the machine, so read them
as indicative; {doc}`/benchmarks/index` has controlled timings.

On these small clouds, iDVC's engine is 1.6–11 times faster on one thread than on
all 32 cores. Its results do not depend on the thread count: all ten XCT1 cases
were bit-identical between the two launches (9 793 points). The bead cases
therefore ran on one thread, eight at a time.

## Two robustness findings

* **iDVC's engine reads outside the image and can crash.** With bead points 20
  voxels from the x and y faces (half a subvolume + 4), `dvc` 22.0.0 crashed
  with a segmentation fault in 7 of 11 translation runs. The same happened in
  3 of 6 stretch runs with `basin_radius` on. It reads a region around each
  point that grows with `disp_max` and the search, without checking it against
  the image. The grid used above keeps every point 30 voxels (half a subvolume
  + `disp_max` + 4) from each face. zvDVC checks every sample and reports
  `RANGE_FAIL` instead.
* **"GOOD" from iDVC's engine is not evidence of a right answer.** On the 15–30 %
  stretches it reported every point GOOD, and most were wrong by voxels. zvDVC's
  statuses caught most of these failures, but not all: do not rely on status
  alone for large deformations.

## Against the plan's pass criteria

The plan set these criteria before the runs.

| Criterion | Result |
|---|---|
| XCT1 repeats: zvDVC's SD within 1.5 × iDVC's on the same pair | **pass**: ratio 1.00 |
| XCT1 repeats: SD at most 0.05 voxel at a 32-voxel subvolume | **pass**: 0.0095 |
| XCT1 moves: every point converges | **pass**: all interior points GOOD in both codes |
| XCT1 moves: zvDVC's residual pattern matches iDVC's within 0.05 voxel RMS | **pass**: RMS difference ≤ 0.0006 |
| Beads: tricubic bias at most 0.02 voxel over 0–1 voxel translations | **pass** in both codes (≤ 0.0003), but see above: this series cannot show interpolation bias |
| Beads: strain error at most 10⁻³ up to 10 % stretch, 12-DOF | **pass** for strain fitted to the whole field (1.3 × 10⁻⁵). Pointwise strain was not assessed |

## Not yet run

| Dataset | What it adds |
|---|---|
| DVC Challenge 1.0, systems XCT2–XCT6 (same archive) | noise floor and distortion on five other scanners, 8- and 16-bit |
| DVC Challenge 2.0: `Yang_Beads_S1` (rotation), `Patel_Beads` (sinusoidal, sparse beads) | rotation; spatial resolution. Neither field ships with the data, so each must be recovered first, as was done for the stretches |
| **Vertebra zero-strain repeats**: Tozzi et al., *J. Mech. Behav. Biomed. Mater.* 67:117–126 (2017), [doi:10.1016/j.jmbbm.2016.12.006](https://doi.org/10.1016/j.jmbbm.2016.12.006); data [doi:10.6084/m9.figshare.4308926.v2](https://doi.org/10.6084/m9.figshare.4308926.v2) (CC BY 4.0) | strain noise floor on bone, against a published commercial local code |
| **Glenoid bone**: Boulanaache et al., *Med. Eng. Phys.* 85:48–54 (2020), [doi:10.1016/j.medengphy.2020.09.009](https://doi.org/10.1016/j.medengphy.2020.09.009); data [doi:10.5281/zenodo.3539508](https://doi.org/10.5281/zenodo.3539508) (CC BY 4.0) | noise floor on real bone; MetaImage input as shipped |
| **spam sandstone VEC4**: [doi:10.5281/zenodo.3888347](https://doi.org/10.5281/zenodo.3888347) (CC BY 4.0), with spam's local DVC (Stamati et al., *JOSS* 5:2286, 2020) | parity with a second, independent local DVC code |

`zvdvc convert` reads a single multi-page TIFF, so slice folders such as DVC
Challenge 1.0's need stacking first. The study below stacks them into raw files.

## Reproduce

```bash
export ZVDVC_CCPI_DVC=/path/to/ccpi-dvc-22.0.0/bin/dvc
python -m zvdvc.bench.public_datasets fetch   --data /data/validation       # 0.7 GB + 1.7 GB
python -m zvdvc.bench.public_datasets prepare --data /data/validation --out runs/public_validation
python -m zvdvc.bench.public_datasets run --out runs/public_validation \
    --cases $(python -m zvdvc.bench.public_datasets list | grep dvc1)
python -m zvdvc.bench.public_datasets run --out runs/public_validation --one-thread \
    --cases $(python -m zvdvc.bench.public_datasets list | grep -v dvc1)
python -m zvdvc.bench.public_datasets prefilter --out runs/public_validation
python -m zvdvc.bench.public_datasets basin     --out runs/public_validation   # iDVC's engine crashes on stretches 4-6
python -m zvdvc.bench.public_datasets report    --out runs/public_validation   # summary.md, summary.json
python -m zvdvc.bench.doc_figures                                             # the figures on this page
```
