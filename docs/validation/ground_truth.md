# Accuracy against known answers

This page checks zvDVC against displacements that are known exactly: synthetic
volume pairs generated with an analytic field, and the real case A scan
shifted by known sub-voxel amounts. iDVC's engine (CCPi DVC 22.0.0, run as iDVC
runs it) is scored the same way on the synthetic cases, so the two codes'
accuracy can be compared directly.

## Synthetic volumes with a known field

`zvdvc synth` builds a speckle phantom, a random texture Gaussian-filtered to
features about 3 voxels across, and deforms it with an analytic displacement
field, so every point's true displacement is known. Two 256³ u16 cases:

| case | field | noise | points | subvolume | shape function |
|---|---|---|---|---|---|
| S (the MVP's accuracy case) | affine | 2 % of full scale, independent in each volume | 2 197 | sphere 32, 2 000 samples | 12-DOF |
| sinusoid | sinusoidal, varying across the volume | none | 4 913 | sphere 32, 2 000 samples | 6-DOF |

Both are solved by zvDVC (in-memory parity mode, CPU engine) and by iDVC's
engine, from the same volumes, points and settings.

```{figure} figures/truth_accuracy.png
:alt: Two cumulative-distribution panels of the error against the true displacement, one per synthetic case. In each, the zvDVC and iDVC curves lie on top of each other.
:figclass: zv-figure
:width: 100%

Error against the true displacement, per point, for both codes. The two codes
are equally accurate: their curves coincide.
```

| case | code | GOOD | RMSE per axis (x, y, z), voxels | median \|error\| | 95th percentile | mean error (x, y, z) |
|---|---|---:|---|---:|---:|---|
| S, affine, 2 % noise | zvDVC | 100 % | 0.0103, 0.0103, 0.0101 | 0.0159 | 0.0285 | +0.0003, +0.0003, −0.0001 |
| | iDVC's engine | 100 % | 0.0103, 0.0101, 0.0104 | 0.0158 | 0.0286 | −0.0004, +0.0003, +0.0000 |
| sinusoid | zvDVC | 100 % | 0.0142, 0.0048, 0.0050 | 0.0148 | 0.0246 | +0.0006, +0.0001, +0.0001 |
| | iDVC's engine | 100 % | 0.0139, 0.0047, 0.0048 | 0.0143 | 0.0247 | +0.0002, +0.0001, +0.0000 |

Both codes meet the MVP's accuracy criterion (RMSE ≤ 0.05 voxel with 2 %
noise, ≥ 99 % GOOD) with a wide margin, and neither has a systematic error.
In the sinusoid case the error is largest in x for both: there the x
displacement varies along x (u = A sin(2πx/λ)), so the subvolume is stretched,
which a 6-DOF shape function (translation and rotation) cannot represent;
12-DOF would. On these cases zvDVC's CPU engine took 0.8 s and 0.9 s, against
139 s and 262 s for iDVC's engine run as iDVC runs it.

```{figure} figures/truth_error_maps.png
:alt: Three maps of the middle plane of the sinusoid case: the true displacement magnitude, zvDVC's error and iDVC's error, on a log colour scale. The two error maps look alike.
:figclass: zv-figure
:width: 100%

The sinusoid case's middle plane: the true field, and each code's error. The
errors are small everywhere, of the same size in both codes, and show no
pattern tied to the field.
```

## Known shifts of the real scan

A synthetic phantom is not a real image. A crop of the case A reference scan,
inside the sample, was therefore shifted by known fractions of a voxel (0 to 1
in steps of 0.1, by quintic B-spline resampling, a higher order than the cubic
the correlation uses) and correlated against the unshifted crop: any
displacement measured beyond the imposed shift is error
({doc}`/benchmarks/2026-09-26-error-floor-case-A`).

```{figure} figures/interpolation_bias.png
:alt: Line chart of mean measured minus imposed displacement against the imposed sub-voxel shift. Without a prefilter the bias follows an S-curve of about ±0.03 voxel, zero at whole and half voxels; with a prefilter of sigma 0.7 it drops to ±0.002 and with sigma 1 to ±0.0005.
:figclass: zv-figure
:width: 90%

Interpolation bias against the imposed sub-voxel shift. Tricubic interpolation
of the raw scan, which is what iDVC does, pulls measurements towards whole
voxels by up to 0.03 voxel. A Gaussian prefilter (`volumes.prefilter_sigma`)
of σ = 1 cuts that to 0.0005.
```

The S-curve matters for strain: its slope, up to about 0.2 voxel per voxel,
distorts displacement gradients measured over a few voxels. It is the same in
iDVC, whose interpolant is the same Catmull-Rom cubic.

```{figure} figures/sampling_spread.png
:alt: Log-log chart of the per-axis uncertainty of one estimate against the number of sample points, from 0.064 voxel at 2 000 to 0.016 at 32 000, exactly on a 1/sqrt(n) line.
:figclass: zv-figure
:width: 80%

The uncertainty from the random choice of sample points, on the real scan: two
solves that differ only in the random sample points of each subvolume disagree
by this much per axis. It falls exactly as 1/√n: 0.032 voxel at iDVC's default
of 8 000 samples, 0.016 at 32 000.
```

On real data this is the largest error term, larger than interpolation bias or
image noise, and it is why iDVC's results change from run to run
({doc}`idvc_test_dataset`). More samples reduce it, at a cost in run time
roughly proportional to the sample count.

## Reproduce

```bash
zvdvc synth --shape 256 256 256 --field affine   --spacing 16 --noise 0.02 --dof 12 --out runs/validation/S_affine_noise2
zvdvc synth --shape 256 256 256 --field sinusoid --spacing 12 --dof 6 --out runs/validation/S_sinusoid
python -m zvdvc.bench.compare_ccpi runs/validation/S_affine_noise2/config.yaml --out runs/validation/S_affine_noise2/compare \
    --ccpi-processes 1 --backends cpu --no-cli --truth runs/validation/S_affine_noise2/truth.npz
# the same for S_sinusoid; the error-floor study:
python -m zvdvc.bench.error_floor --help
python -m zvdvc.bench.doc_figures        # the figures on this page
```
