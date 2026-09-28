# Error floor on real data: case A (iDVC example)

Date: 2026-09-26. Same workstation and commit as
[2026-09-26-case-A-real.md](2026-09-26-case-A-real.md): RTX A2000 12 GB,
backend `fused`. Module: [`bench/error_floor.py`](../../src/zvdvc/bench/error_floor.py).

**Summary.** At CCPi's case A settings (sphere 80, 8 000 samples, 6-DOF,
ZNSSD, tricubic), a zvDVC displacement on this scan carries three errors:

| error | size (voxel, per axis) | depends on |
|---|---|---|
| interpolation bias | up to ±0.03 (worst case over all settings: 0.041) | the displacement's fractional part (S-curve) |
| random error from image noise | 0.005 | noise level, sample count |
| **random error from the choice of sample points** | **0.032** | sample count, as 1/√n: 0.016 at 32 000 samples |

The last one dominates, and it is the one a synthetic test cannot see. It also
fully explains the disagreement with CCPi on the same data.

## Method

No repeat scans of an unloaded sample exist for case A. Two tests are used
instead.

**1. Known shifts of a real image.**
* **Image.** A 448³ crop of the case A reference volume, placed inside the
  sample by foreground fraction (origin z, y, x = 576, 416, 56).
* **Shifts.** Translations of 0 to 1 voxel in steps of 0.1, along x and along
  the (1, 1, 1) diagonal, applied with quintic B-splines
  (`scipy.ndimage.shift`, mirror boundaries). The solver uses a different
  interpolant (Catmull-Rom), so it is not graded against its own model.
* **Variants.** *One-sided*: ref = I, def = shift(I, s). *Symmetric*:
  ref = shift(I, −s/2), def = shift(I, +s/2).
* **Noise.** Independent Gaussian noise added to both images at 0, 0.5, 1 and
  2 × the scan's own noise. Immerkær's estimator gives σ = 2.70 grey levels
  on the u8 scale. The scan's own noise moves with the image, so the
  zero-added-noise rows measure interpolation and model error only.
* **Sweep.** Subvolume size 40, 60, 80; 1 000 to 8 000 samples; 6-DOF, ZNSSD,
  tricubic, seeded at zero. Points are on a grid spaced by the subvolume size,
  so their errors are independent (27 to 1 000 points per case).
* **Scale.** 2 112 cases; every case is ≥ 99.9 % GOOD.
* **Sanity check.** The crop correlated with itself gives |u| ≤ 2.9e-8 voxel
  at all 125 points.

**2. Sample-set repeatability on the real pair.** CCPi's central grid
(4 680 points) on the real reference and deformed volumes is solved several
times, changing only the template's random seed, so each run samples every
subvolume at different points. The spread between runs is the uncertainty
from the choice of sample points.

Reproduce:

```bash
python -m zvdvc.bench.error_floor --case-a runs/case_A_data --out runs/error_floor    # ~35 min on the A2000
```

## 1. Interpolation bias

Sphere 80, 8 000 samples, no added noise. Bias is the mean error over 125
points. For the diagonal, "shift" is the length of the shift vector and the
bias shown is along it; its per-axis shift is 0.577 × that, so 1.0 is not a
whole-voxel shift and its bias is not zero.

| fractional shift | one-sided, x | symmetric, x | one-sided, diagonal |
|---|---|---|---|
| 0.0 | −0.0000 | −0.0000 | +0.0000 |
| 0.1 | −0.0160 | −0.0163 | −0.0086 |
| 0.2 | −0.0273 | −0.0295 | −0.0162 |
| 0.3 | −0.0295 | −0.0364 | −0.0220 |
| 0.4 | −0.0192 | −0.0349 | −0.0250 |
| 0.5 | −0.0001 | −0.0254 | −0.0250 |
| 0.6 | +0.0189 | −0.0118 | −0.0218 |
| 0.7 | +0.0290 | −0.0008 | −0.0156 |
| 0.8 | +0.0271 | +0.0043 | −0.0072 |
| 0.9 | +0.0159 | +0.0039 | +0.0025 |
| 1.0 | +0.0000 | −0.0000 | +0.0121 |

The one-sided curve is the classic interpolation-bias S-curve: zero at whole
and half voxels and about ±0.03 in between. Resampling both images
symmetrically does not remove the bias; it changes its shape. This is a
systematic error. In a displacement field it appears as a small periodic error
tied to the fractional part of the displacement, which strain calculations
can amplify. Reducing it needs an interpolant with a flatter response
(quintic, or B-spline with prefiltering); this is not pursued in stage 1.

## 2. Random error from image noise

Mean per-axis standard deviation of the error (voxel), averaged over shifts,
directions and variants. σ = 2.70 grey levels.

| subvolume | samples | +0 σ | +0.5 σ | +1 σ | +2 σ |
|---|---|---|---|---|---|
| 40 | 1000 | 0.0066 | 0.0091 | 0.0141 | 0.0295 |
| 40 | 2000 | 0.0058 | 0.0074 | 0.0109 | 0.0217 |
| 40 | 4000 | 0.0054 | 0.0065 | 0.0088 | 0.0164 |
| 40 | 8000 | 0.0051 | 0.0058 | 0.0074 | 0.0129 |
| 60 | 1000 | 0.0053 | 0.0077 | 0.0124 | 0.0263 |
| 60 | 2000 | 0.0043 | 0.0060 | 0.0091 | 0.0188 |
| 60 | 4000 | 0.0037 | 0.0048 | 0.0070 | 0.0137 |
| 60 | 8000 | 0.0034 | 0.0041 | 0.0056 | 0.0102 |
| 80 | 1000 | 0.0047 | 0.0073 | 0.0122 | 0.0256 |
| 80 | 2000 | 0.0038 | 0.0055 | 0.0088 | 0.0180 |
| 80 | 4000 | 0.0032 | 0.0044 | 0.0065 | 0.0129 |
| 80 | 8000 | 0.0028 | 0.0036 | 0.0050 | 0.0095 |

As expected, the error grows with noise and falls with sample count, which
matters more than subvolume size here. The fitted log-slopes (sphere 80,
noisy cases) are 0.74 against noise and −0.44 against sample count. The ideal
values are 1 and −0.5; the zero-noise floor pulls the fits below them.

These numbers assume **white** noise added independently to each image. Real
CT noise is spatially correlated, and a second scan is a new acquisition, so
this table is a lower bound on noise-driven error.

## 3. Random error from the choice of sample points (real pair)

The central grid on the real volumes, three template seeds, fused backend:

| comparison | median \|Δu\| | p95 \|Δu\| | mean Δu (x, y, z) |
|---|---|---|---|
| seed 0 vs seed 1 | 0.0509 | 0.1505 | −0.0028, −0.0008, −0.0010 |
| seed 0 vs seed 2 | 0.0516 | 0.1558 | −0.0022, +0.0019, +0.0064 |
| seed 1 vs seed 2 | 0.0517 | 0.1533 | +0.0006, +0.0027, +0.0074 |
| *zvDVC vs CCPi (for comparison)* | *0.0512* | *0.1517* | *−0.0027, −0.0017, −0.0005* |

Uncertainty of one estimate (per axis), against sample count:

| samples | 2 000 | 4 000 | 8 000 | 16 000 | 32 000 |
|---|---|---|---|---|---|
| per-axis σ (voxel) | 0.064 | 0.045 | 0.032 | 0.022 | 0.016 |

It scales as 1/√n: each 4× more samples halves it. This is Monte-Carlo
sampling error. The subvolume is represented by n random points, and on real
data the best-fitting 6-DOF warp depends on which points they are, because
the true deformation is not rigid inside an 80-voxel sphere and the two scans
differ by more than a shift. It is about 6× the noise-driven error of
section 2 at the same settings, and it is invisible to shifted-copy tests,
where every sample set sees the same texture in both images.

It also **explains the disagreement with CCPi entirely**: zvDVC with a
different seed disagrees with itself exactly as much as it disagrees with CCPi
(median 0.051, p95 0.15), and the mean differences are all ≤ 0.007 voxel. The
case A comparison's median criterion (0.05) therefore cannot be met by zvDVC
against itself at 8 000 samples; the criterion should be revised (see below).

## 4. Reducing the bias

**CCPi's interpolation is not an alternative: it is the same interpolant.**
CCPi's Lekien–Marsden tricubic is fed central-difference derivatives
(`Matrix_4d.h`: `(f₊₁ − f₋₁)/2` and the tensor-product cross terms), and a
tricubic Hermite built from exactly those is separable Catmull-Rom, which
zvDVC evaluates directly. The real pair confirms it: binned by the fractional
part of the displacement, the mean zvDVC − CCPi difference is flat (all bins
within −0.004 to +0.0003 voxel on every axis), where a different interpolant
would trace an S-curve of ±0.03.

**A Gaussian prefilter removes most of it.** Low-pass filtering both images
(Pan 2013, *Optics and Lasers in Engineering*) removes the fine-scale
content that interpolation handles unevenly. Same shift test (sphere 80,
8 000 samples, one-sided, x), and the sample-set uncertainty of section 3
measured on the real pair:

| prefilter σ (voxel) | max bias | max slope (worst strain error) | sample-set uncertainty |
|---|---|---|---|
| none | 0.030 | 0.19 (19 %) | 0.031 |
| 0.7 | 0.0022 | 0.014 (1.4 %) | 0.026 |
| 1.0 | 0.0005 | 0.003 (0.3 %) | 0.025 |

On this scan the filter costs no precision: the random error falls too,
because it removes scan-to-scan noise that differs between the two
acquisitions. On data whose texture is at the voxel scale, filtering removes
information and raises random error, so σ should be chosen from this study
on each dataset.

**Storage.** A blurred u8 volume rounded back to u8 is as good as float32
(σ = 1: max bias 0.0004 for float32, 16-bit and 8-bit; slope 0.0023,
0.0023, 0.0034; sample-set uncertainty 0.0248 for all three). zvDVC
therefore keeps the volume's dtype: no extra memory, and u8 volumes keep the
packed loads.

**In zvDVC:** `volumes.prefilter_sigma` (default 0, CCPi parity). Bricks are
read with a 4σ margin, filtered and cropped, so tiles meet without seams; the
filter runs on the GPU in z-slabs. On case A, σ = 1 takes 5.9 s per 2.4 GB
volume, and the central grid's result shifts by ≤ 0.003 voxel on average,
with statuses unchanged and sample-set uncertainty 0.032 → 0.025 in the full
wavefront run. Results stores record σ; a resumed run with another σ is
refused. `zvdvc check full` includes a prefiltered central-grid step.

## Consequences

* **For users.** On data like this, precision is set by the sample count.
  CCPi's default of 8 000 gives ~0.03 voxel per axis; 32 000 gives ~0.016 at
  about 4× the solve time, still seconds for a few thousand points on a GPU.
  zvDVC reports this sampling uncertainty with results when asked:
  `uncertainty_seeds: 2` in the run config repeats each GOOD point with two
  other template seeds and stores the per-axis spread as `displacement_sd`
  (case A: RMS 0.033, 0.031, 0.032 voxel, matching section 3; 0.3 s extra on
  the central grid). `zvdvc strain` propagates it into a strain uncertainty.
* **For validation.** The Q1 criterion "median |Δu| ≤ 0.05 against CCPi" is
  set below the difference between two correct solvers at CCPi's own
  settings. A criterion that can fail for the right reasons: mean difference
  ≤ 0.01 voxel per axis, and a spread no larger than zvDVC's own seed-to-seed
  spread. zvDVC passes both against CCPi on case A.
* **Bias, and strain.** The ±0.03 S-curve is systematic and does not average
  out. Its slope reaches ~0.19 voxel of bias per voxel of fractional
  displacement (between 0.5 and 0.6 above). Strain is the gradient of
  displacement, so where the fractional displacement varies across a region
  the bias adds up to ~20 % of the local strain. With a 1-voxel prefilter
  (section 4) this falls to ~0.3 %; use one for strain on data like this.
