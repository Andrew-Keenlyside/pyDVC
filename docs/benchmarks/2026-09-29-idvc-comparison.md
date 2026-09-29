# Direct comparison with iDVC's engine on case A: how close, and why not identical

Date: 2026-09-29. Workstation `msm12`: 32 cores, 251 GB RAM, NVIDIA RTX A2000
12 GB. CCPi `dvc` 22.0.0 (the conda release that iDVC installs), plus two builds
from its v22.0.0 source. zvDVC at `develop` (72dcfdd, plus the study module).

This study answers one question: can zvDVC reproduce what
[iDVC](https://github.com/TomographicImaging/iDVC) computes bit for bit, and if
not, how large is the difference exactly and where does it come from?

[iDVC](https://tomographicimaging.github.io/iDVC/) (Tomographic Imaging / CCPi,
UKRI-STFC) does not correlate by itself: it writes a `dvc_in` parameter file
and runs the CCPi DVC engine, `dvc`
([TomographicImaging/DigitalVolumeCorrelation](https://github.com/TomographicImaging/DigitalVolumeCorrelation),
the method of Bay et al., 1999). Every CCPi run here is that executable, run
the way iDVC runs it: one process on every point of iDVC's example grid, from a
`dvc_in` file in iDVC's format. The iDVC GUI itself was not used; it would run
exactly these commands.

## Summary

* **Bit-for-bit agreement with iDVC is not possible, and not a meaningful
  target.** iDVC's default subvolume is a sphere, and CCPi draws each sphere's
  sample points from a random generator **seeded with the clock, separately for
  every point and every run**. CCPi does not reproduce its own sphere results:
  two identical CCPi runs differ by a median 0.051 voxel. Even with a
  deterministic (cube) subvolume, CCPi rebuilt from its own source differs from
  the released binary by a median 1.7 × 10⁻⁵ voxel.
* **With iDVC's settings, zvDVC is exactly as close to CCPi as CCPi is to
  itself.** zvDVC against CCPi: median 0.0511 voxel, 95th percentile 0.149.
  CCPi against CCPi: 0.0513 and 0.155. Their distributions coincide
  ([figure 2](#distributions)).
* **With the same sample points (cube subvolumes), zvDVC and CCPi differ by a
  median 0.0010 voxel** (95th percentile 0.0044), 50 times less. That remaining
  difference comes from where each code *stops*, not from its arithmetic: CCPi
  stops when a step moves the subvolume by less than 0.01 voxel, and the two
  codes take different paths to that point. zvDVC in float32 and in float64
  agree to 5 × 10⁻⁷ voxel.
* **Run both to convergence and they agree to a median 2.1 × 10⁻⁵ voxel**
  (95th percentile 1.3 × 10⁻⁴, maximum 6.1 × 10⁻⁴), the same order as CCPi
  against a recompiled CCPi (1.7 × 10⁻⁵). At that level the two codes compute
  the same optimum; what remains is floating-point detail (CCPi's
  finite-difference Jacobian and double-precision sums against zvDVC's analytic
  Jacobian). zvDVC's GPU and CPU engines agree with each other to 2 × 10⁻⁷.
* **Status agreement is 100 %** on every interior point in every comparison.
  Points whose subvolume reaches the image edge are excluded (CCPi reads outside
  the image there; zvDVC reports `RANGE_FAIL`).
* **A finding for iDVC users:** CCPi's default stopping rule leaves each point
  a median 0.0066 voxel (95th percentile 0.025) short of the converged
  optimum, more than six times the zvDVC–CCPi difference at the same stopping
  rule; the sphere sampling spread (0.05 voxel) is eight times larger again.

## Why CCPi cannot be matched bit for bit

From CCPi's source at tag v22.0.0:

* `dvc.cpp` calls `Search::process_point(t, n, &data)` with the default
  `test = 0`. For a sphere, `process_point` then builds a `FloatingCloud`
  without a seed, and `FloatingCloud`'s default seed is
  `std::chrono::system_clock::now().time_since_epoch().count()`
  (`FloatingCloud.h`). Each point's 8 000 sample positions are therefore drawn
  afresh, from a clock-seeded `std::mt19937`, on every run. Only CCPi's own
  test harness (`tests.cpp`) passes `test = 1`, which fixes the seed to 1.
* For a cube, `FloatingCloud` lays a fixed `k × k × k` grid over the subvolume
  (`k = 20` for 8 000 samples), the same grid zvDVC builds. With cubes the two
  codes sample identical positions, which is what makes the cube comparisons
  below possible without changing either code.
* The stopping rule is hard-coded: `min_Lev_Mar(par_min, 0.000001, 0.01)`, at
  most 20 iterations, stopping when the objective changes by less than 10⁻⁶
  **or** the translation step is below 0.01 voxel.

## Setup

| | |
|---|---|
| data | iDVC's example: Zenodo [7363345](https://zenodo.org/records/7363345) (Lee, Lavallée and Bay, 2022), two 1520 × 1257 × 1260 u8 scans |
| points | CCPi `dvc_test/central_grid.roi`: 4 680 points, a 90 × 52 grid in slice z = 630 |
| settings | CCPi `dvc_input.txt`: size 80, 8 000 samples, 6-DOF, ZNSSD, tricubic, `disp_max` 38, `rigid_trans` (34, 4, 0); sphere (iDVC's default) or cube |
| zvDVC | parity mode: whole volumes in memory, wavefront seeding in CCPi's point order, `report_convg_fail: false` (CCPi never flags non-convergence) |

| run | code | subvolume | points | purpose |
|---|---|---|---|---|
| CCPi 2026-09-26 | release 22.0.0, 32 threads | sphere | all | the case A run of [2026-09-26](2026-09-26-case-A-real.md) |
| R1 | release 22.0.0, 10 threads | sphere | all | CCPi against itself |
| R2 | release 22.0.0, 10 threads | cube | all | CCPi with deterministic samples |
| R3 | release 22.0.0, **1 thread** | cube | first 500 | is CCPi deterministic, and independent of thread count? |
| R4 | built from v22.0.0 source (g++ 11.4 `-O3`, Eigen 3.4.0), 1 thread | cube | first 500 | does a rebuild change the result? |
| R5 | built from source with the stopping rule tightened ([patch](#reproduce)): objective change < 10⁻¹⁵ or step < 10⁻⁹, at most 200 iterations | cube | all | CCPi run to convergence (research build, not what iDVC runs) |
| zvDVC sphere | `cpu` engine (float32), template seeds 0 and 1 | sphere | all | zvDVC against itself, and against R1 |
| zvDVC cube | `cpu` (float32) and `numpy` (float64) engines | cube | all | same samples as R2 |
| zvDVC cube, converged | same, with the tolerances of R5 | cube | all | against R5 |

The zvDVC runs use the `cpu` engine, which runs the GPU engine's fused kernels
on the CPU in the same float32 arithmetic, and the float64 `numpy` reference
engine; the GPU (`fused`) engine was run too on the sphere and cube cases.
Case A's whole-volume parity mode needs about 5 GB of the GPU, which was shared
with other jobs during the study. CCPi's timings here are not
meaningful (runs shared the machine); for speed see
[2026-09-26-case-A-real.md](2026-09-26-case-A-real.md).

"Identical" below means equal to CCPi's printed precision, 6 decimals in the
`.disp` file, which is all iDVC ever reads.

## Results

Every comparison, on interior points (GOOD in both results, away from the image edge):

| cause | comparison | points compared | identical to 6 dp | median \|Δu\| | p95 | max | mean Δ (x, y, z) | status agreement (interior) |
|---|---|---:|---:|---:|---:|---:|---|---:|
| sampling | CCPi vs CCPi: sphere, two runs | 4,628 | 0 (0.0%) | 0.0513 | 0.1548 | 0.5838 | +4.7e-04, -2.4e-04, +4.1e-04 | 100.00% |
| sampling | zvDVC vs zvDVC: sphere, template seeds 0 and 1 | 4,628 | 0 (0.0%) | 0.0509 | 0.1505 | 0.4326 | -2.8e-03, -8.5e-04, -9.8e-04 | 100.00% |
| sampling | zvDVC vs CCPi: sphere (iDVC's settings) | 4,628 | 0 (0.0%) | 0.0511 | 0.1492 | 0.5530 | -3.2e-03, -1.5e-03, -9.1e-04 | 100.00% |
| sampling | zvDVC GPU vs zvDVC CPU engine: sphere (same template) | 4,628 | 3,681 (79.5%) | 1.2e-07 | 9.6e-07 | 3.9e-06 | -3.2e-09, +1.0e-10, -2.0e-09 | 100.00% |
| same samples | CCPi vs CCPi: cube, rerun on 1 thread (first 500 points) | 450 | 450 (100.0%) | 0 | 0 | 0 | +0.0e+00, +0.0e+00, +0.0e+00 | 100.00% |
| same samples | CCPi release vs CCPi built from source: cube (first 500) | 450 | 0 (0.0%) | 1.7e-05 | 3.6e-05 | 5.6e-05 | +2.7e-07, +7.1e-07, +4.6e-07 | 100.00% |
| same samples | zvDVC float64 vs CCPi: cube | 4,448 | 0 (0.0%) | 0.0010 | 0.0044 | 0.0566 | +2.4e-04, +2.0e-05, +1.1e-05 | 100.00% |
| same samples | zvDVC float32 vs CCPi: cube | 4,448 | 0 (0.0%) | 0.0010 | 0.0044 | 0.0566 | +2.4e-04, +2.0e-05, +1.1e-05 | 100.00% |
| same samples | zvDVC float32 vs zvDVC float64: cube | 4,448 | 1,885 (42.4%) | 5.0e-07 | 1.7e-06 | 2.8e-06 | -7.1e-09, +3.6e-10, -1.4e-10 | 100.00% |
| same samples | zvDVC GPU vs zvDVC CPU engine (both float32): cube | 4,448 | 3,242 (72.9%) | 2.4e-07 | 1.9e-06 | 3.9e-06 | -2.8e-08, -2.4e-09, -7.8e-10 | 100.00% |
| same samples | zvDVC GPU vs CCPi: cube | 4,448 | 0 (0.0%) | 0.0010 | 0.0044 | 0.0566 | +2.4e-04, +2.0e-05, +1.1e-05 | 100.00% |
| converged | CCPi vs CCPi run to convergence: cube | 4,448 | 0 (0.0%) | 0.0066 | 0.0248 | 0.6617 | +8.7e-03, -2.1e-04, -5.8e-04 | 100.00% |
| converged | zvDVC float64 vs itself run to convergence: cube | 4,448 | 0 (0.0%) | 0.0067 | 0.0252 | 0.7173 | +8.9e-03, -1.9e-04, -5.7e-04 | 100.00% |
| converged | zvDVC float64 vs CCPi, both run to convergence: cube | 4,448 | 0 (0.0%) | 2.1e-05 | 1.3e-04 | 6.1e-04 | +7.4e-07, -5.2e-07, +2.6e-07 | 100.00% |
| converged | zvDVC float32 vs CCPi, both run to convergence: cube | 4,448 | 0 (0.0%) | 1.1e-04 | 8.7e-04 | 0.0121 | +2.1e-04, -4.6e-06, -6.3e-06 | 100.00% |

(figure-budget)=
### Where the difference comes from

```{figure} figures/2026-09-29-idvc/error_budget.png
:alt: Horizontal range chart of the median and 95th-percentile difference between pairs of results on a log scale from 1e-7 to 10 voxels. Different sample points (sphere): CCPi against CCPi, zvDVC against zvDVC and zvDVC against CCPi all sit at about 0.05 voxel; zvDVC's GPU against its CPU engine at 1e-7. Same sample points (cube): CCPi rerun on one thread is identical; CCPi release against its source build 1.7e-5; zvDVC against CCPi 1e-3 whatever the engine. Run to convergence: each code against its own default result about 0.007; zvDVC float64 against CCPi 2.1e-5, float32 1.1e-4.
:figclass: zv-figure
:width: 100%

Median (dot) and 95th percentile (bar end) of the point-by-point displacement
difference |Δu| for every pair of results, on interior points GOOD in both.
The dashed line is the resolution of CCPi's `.disp` output.
```

(distributions)=
### Distributions

```{figure} figures/2026-09-29-idvc/difference_cdf.png
:alt: Cumulative distributions of |Δu|. The zvDVC-against-CCPi and CCPi-against-CCPi sphere curves lie on top of each other around 0.05 voxel; the cube curve sits near 1e-3 and the converged cube curve near 2e-5.
:figclass: zv-figure
:width: 90%

Cumulative distribution of |Δu| over interior points for the four headline
comparisons. The two sphere curves are indistinguishable.
```

### The displacement fields

```{figure} figures/2026-09-29-idvc/fields_cube.png
:alt: Nine maps of the z = 630 plane: CCPi's and zvDVC's u, v and w displacement fields, visually identical, and their difference, which is unstructured noise of a few thousandths of a voxel.
:figclass: zv-figure
:width: 100%

CCPi and zvDVC (float64) on cube subvolumes, each with its default stopping
rule, and their difference. The difference is unstructured and has no
systematic component (mean per axis ≤ 2.4 × 10⁻⁴ voxel).
```

### Where on the grid

```{figure} figures/2026-09-29-idvc/difference_maps.png
:alt: Four maps of |Δu| on a shared log colour scale from 1e-6 to 1 voxel. The two sphere comparisons look the same, uniformly around 0.05 voxel; the cube comparison is around 1e-3; the converged comparison is around 1e-5.
:figclass: zv-figure
:width: 85%

|Δu| on the point grid, one shared log colour scale. Grey: points at the image
edge, excluded from every comparison.
```

## What this means

1. **zvDVC reproduces iDVC to within iDVC's own reproducibility.** With
   iDVC's default settings the question "does zvDVC give the same answer as
   iDVC?" has the same answer as "does iDVC give the same answer as iDVC?":
   to about 0.05 voxel, set by the random sample points.
2. **The two codes compute the same optimum.** With the same sample points and
   both run to convergence, the difference (median 2.1 × 10⁻⁵ voxel) is of the
   size of the difference between two compilations of CCPi. It is largest
   where the displacement field changes fastest (the band near x = 1000 in the
   maps), as expected where the objective's minimum is least sharp.
3. **Arithmetic precision is not a factor at CCPi's stopping rule.** zvDVC's
   float32 engines (the GPU's and the CPU's) agree with its float64 engine to
   5 × 10⁻⁷ voxel there. Driven to full convergence, float32 becomes the limit
   (1.1 × 10⁻⁴ voxel against CCPi, where float64 reaches 2.1 × 10⁻⁵), still far
   below any stopping-rule or sampling effect.
4. **For reproducible iDVC results, use cube subvolumes or more samples.**
   Sphere results carry about 0.05 voxel of sampling noise per point, which
   falls as 1/√samples ([error-floor study](2026-09-26-error-floor-case-A.md)).
   With cubes, CCPi is deterministic: R3 on one thread matched R2 on ten
   threads in every printed digit on all 450 comparable points.

## Limitations

* One dataset, and one plane of points. The same comparison on the validation
  datasets planned in the {doc}`MVP review </MVP_REVIEW>` would widen it.
* R5 is a research build of CCPi with a patched stopping rule; iDVC never runs
  it. It exists only to show where both codes converge to.
* The GPU was shared with other jobs during the study, so most zvDVC runs use
  the `cpu` engine (the GPU's fused kernels, run on the CPU in the same float32
  arithmetic). The GPU engine was run on the sphere and cube cases when memory
  allowed: it agrees with the `cpu` engine to a median 2 × 10⁻⁷ voxel.

## Reproduce

```bash
# CCPi 22.0.0 release, and v22.0.0 built from source (needs g++, OpenMP and Eigen 3 headers)
eval "$(scripts/get_ccpi_dvc.sh)"
git clone https://github.com/TomographicImaging/DigitalVolumeCorrelation ccpi-src && git -C ccpi-src checkout v22.0.0
# tightened build (research only): in Core/Search.cpp,
#   min_Lev_Mar(par_min, 0.000001, 0.01)  ->  min_Lev_Mar(par_min, 1e-15, 1e-9)
#   int maxit = 20;                       ->  int maxit = 200;
# compile: g++ -std=c++11 -O3 -DNDEBUG -fopenmp -I<eigen3> Core/{BoundBox,Cloud,DataCloud,FloatingCloud,InputRead,
#          Interpolate,Matrix_4d,ObjectiveFunctions,Point,Search,SearchParams}.cpp Core/dvc.cpp -o dvc
#          (after generating Core/CCPiDefines.h from CCPiDefines.h.in)

# CCPi runs: dvc <dvc_in> for R1–R5, each dvc_in written as iDVC writes it (case A settings, subvol_geom sphere|cube)
# zvDVC runs and the analysis
python -m zvdvc.bench.idvc_study run     --config runs/case_A/config.yaml --out runs/idvc_study
python -m zvdvc.bench.idvc_study analyze --config runs/case_A/config.yaml --out runs/idvc_study \
    --figures docs/benchmarks/figures/2026-09-29-idvc
```

The study module is
[`zvdvc/bench/idvc_study.py`](../../src/zvdvc/bench/idvc_study.py); its
docstring describes the design. Point-by-point results are in
`runs/idvc_study/study.json`.
