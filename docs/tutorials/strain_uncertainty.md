# Strain and displacement uncertainty

This tutorial computes strain from a zvDVC result and attaches an uncertainty
to every displacement and every strain value. It uses a synthetic case whose
strain is known, so the numbers can be checked, and then quotes what the same
tools give on real data. It is for users who report strain and need to say
how far to trust it.

The strain method, flags and CSV layout follow CCPi's `strain` program, part
of the [CCPi DVC engine](https://github.com/TomographicImaging/DigitalVolumeCorrelation)
that [iDVC](https://github.com/TomographicImaging/iDVC) runs. The uncertainty
columns are zvDVC's addition.

All outputs below were produced for this page on an RTX A2000 12 GB, trimmed
where marked `…`.

---

## 1. A case with a known strain

The `affine` field applies a uniform strain of up to about 1 %. A finer grid
(spacing 8) gives each strain window close neighbours, and 2 % noise
makes the uncertainty worth measuring:

```bash
zvdvc synth --shape 128 128 128 --field affine --spacing 8 --noise 0.02 --out data/strain128
```

That gives 11³ = 1 331 points. `truth.npz` holds the true displacement
*gradient* `∂u_i/∂x_j` at every point as well as the displacement, so the
true strain is known exactly. Here it is the same everywhere:

| | exx | eyy | ezz | exy | eyz | exz |
|---|---|---|---|---|---|---|
| true engineering strain | 0.0189 | −0.01181 | 0.00945 | 0.00472 | 0 | −0.00709 |

## 2. Ask for an uncertainty per displacement

Set `uncertainty_seeds` in `data/strain128/config.yaml`:

```yaml
uncertainty_seeds: 2
```

A subvolume is represented by a fixed number of random sample points (the
*template*, drawn once per run from `subvolume.seed`). On real data the
answer depends on which points those are. With `uncertainty_seeds: k`, every
GOOD point is solved k more times, each with a template drawn from another
seed, starting from the same seed displacement as the main solve. The
per-axis standard deviation over the k + 1 estimates is stored as
`displacement_sd` ([`solver/uncertainty.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/solver/uncertainty.py)).

* With k = 2 the extra cost is about twice the main solve. Each point's
  estimate has about 2 degrees of freedom, so it is rough per point and good
  in aggregate.
* Points that are not GOOD, or have fewer than two GOOD estimates, get NaN.

Solve in memory:

```bash
zvdvc solve data/strain128/config.yaml --backend fused --quiet
```

```text
/…/data/strain128/run/results.npz: 1331 points in 0.6 s (2111.0 pt/s), status counts {0: 1331}
```

`zvdvc solve` writes `<workdir>/results.npz` (with `displacement_sd`) and
`results.stat`; `--disp` adds a `.disp`. Its `--backend` defaults to `numpy`,
the slow float64 reference, so pass `fused` or `cpu`.

In the tiled pipeline, `displacement_sd` reaches the results store with
every seeding strategy: `zvdvc run` computes it for `rigid` and `coarse`, and
`zvdvc seed` for `wavefront`, where it solves every point.

## 3. Compute strain

```bash
zvdvc strain data/strain128/run/results.npz -E -D
```

```text
{
 "files": {
  "Lstr": "data/strain128/run/results-sw25.Lstr.csv",
  "Estr": "data/strain128/run/results-sw25.Estr.csv",
  "dgrd": "data/strain128/run/results-sw25.dgrd.csv",
  "npz": "data/strain128/run/results-sw25.strain.npz"
 },
 "points": 1331,
 "with_strain": 1331,
 "planar_axis": null,
 "median_strain_sd": [
  0.00030005391740246904,
  …
 ]
}
```

`zvdvc strain` reads a results store, a zvDVC `.npz` or a CCPi `.disp`. The
flags keep CCPi's short names:

| flag | default | meaning |
|---|---|---|
| `--window`, `-sw` | `25` | points in each strain window |
| `--threshold`, `-t` | `1.0` | a window point is used only if GOOD and its `objmin` is at most this |
| `--refill`, `-r` | off | replace dropped points with further neighbours until the window is full |
| `--engineering`, `-E` | off | also write engineering strain (`.Estr.csv`) |
| `--engineering-only` | off | write engineering strain only |
| `--gradient`, `-D` | off | also write the displacement gradient (`.dgrd.csv`) |
| `--sigma-u` | from the results | displacement uncertainty (voxels) to propagate; see below |
| `--out` | next to the results | output base name |

Lagrangian strain (`.Lstr.csv`) is always written unless
`--engineering-only` is given. A `.strain.npz` with every array is always
written too.

### The method

For each point ([`post/strain.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/post/strain.py)):

1. The window is the `window` nearest points, the point itself included. Of
   these, the GOOD points with `objmin` ≤ `threshold` are kept. With
   `--refill`, further neighbours replace the dropped ones.
2. Each displacement component is fitted over the window by least squares
   with a full quadratic polynomial in x, y and z (10 terms).
3. The displacement gradient $G = \partial u / \partial x$ is the fit's
   derivative at the point.
4. From $G$: engineering strain $\tfrac12(G + G^T)$, Lagrangian strain
   $\tfrac12(G + G^T + G^T G)$, and their principal values in descending
   order. Shear components are tensor shears, as in CCPi.

A point cloud in a coordinate plane (every point sharing one coordinate, like
CCPi's central grid) is fitted in that plane, with the out-of-plane
components set to zero, as CCPi does; `planar_axis` reports which axis. A
window whose points cannot determine every term of the fit gets NaN, where
CCPi returns an ill-determined value. On CCPi's case A result this reproduces
CCPi's strain to every printed digit at all interior points (a few windows
at the cloud's edge can differ by one point, where equidistant neighbours
tie).

### The files

The CSVs use CCPi's layout, with zvDVC's uncertainty columns at the end
(`results-sw25.Lstr.csv`):

```text
n,x,y,z,u_fit,v_fit,w_fit,pts_in_sw,sw_radius,exx,eyy,ezz,exy,eyz,exz,ep1,ep2,ep3,sd_exx,sd_eyy,sd_ezz,sd_exy,sd_eyz,sd_exz
1,23.5,23.5,23.5,0.525475,-0.126854,0.193401,25,24,0.0203473,-0.0104832,0.0099013,0.00467931,0.00162023,-0.00575492,0.0232928,0.00794314,-0.0114706,0.00112923,0.0012378,0.00262808,0.000848464,0.00151574,0.0015889
…
```

| column | meaning |
|---|---|
| `n`, `x`, `y`, `z` | point id and position |
| `u_fit`, `v_fit`, `w_fit` | the fitted displacement at the point |
| `pts_in_sw`, `sw_radius` | points used, and the distance to the farthest one |
| `exx` … `exz` | strain components (Lagrangian in `.Lstr.csv`, engineering in `.Estr.csv`) |
| `ep1`, `ep2`, `ep3` | principal strains, descending |
| `sd_exx` … `sd_exz` | standard deviation of each **engineering** strain component (zvDVC only) |

The `.dgrd.csv` file has `ux, uy, uz, vx, vy, vz, wx, wy, wz` in place of
the strain columns. CCPi's own `strain` program reads `.disp` files, so it
also works on zvDVC's `.disp` output.

### Where `sd_*` comes from

The strain uncertainty propagates a displacement uncertainty through the
least-squares fit. That displacement uncertainty is, in order of preference:
`--sigma-u` if given; else the results' `displacement_sd` (from
`uncertainty_seeds`); else the fit's own residual per component. Point 1
above is a corner of the grid: its window is lopsided (`sw_radius` 24,
against 13.9 for interior points), and its `sd_*` values are several times
the median.

## 4. Check against the truth

```python
import numpy as np

truth = np.load("data/strain128/truth.npz")
s = np.load("data/strain128/run/results-sw25.strain.npz")
res = np.load("data/strain128/run/results.npz")

assert np.array_equal(truth["point_id"], s["point_id"])
G = truth["gradient"]                                     # (N, 3, 3), du_i/dx_j
E = 0.5 * (G + G.transpose(0, 2, 1))
voigt = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]  # exx, eyy, ezz, exy, eyz, exz
e_true = np.stack([E[:, i, j] for i, j in voigt], axis=1)
err = s["engineering"] - e_true
print("RMS error per component:  ", np.round(np.sqrt(np.nanmean(err**2, axis=0)), 5))
print("median strain_sd:          ", np.round(np.nanmedian(s["strain_sd"], axis=0), 5))
print("median displacement_sd:    ", np.round(np.nanmedian(res["displacement_sd"], axis=0), 4))
inner = s["sw_radius"] <= np.median(s["sw_radius"])       # the 729 interior points (compact windows)
print("RMS error, inner half:     ", np.round(np.sqrt(np.nanmean(err[inner]**2, axis=0)), 5))
```

```text
RMS error per component:   [0.00055 0.00059 0.00055 0.00045 0.00039 0.00041]
median strain_sd:           [0.0003  0.00029 0.00029 0.00024 0.00024 0.00024]
median displacement_sd:     [0.008  0.0074 0.0076]
RMS error, inner half:      [0.00037 0.00038 0.00035 0.00024 0.00024 0.00024]
```

And the displacements against the truth:

```bash
zvdvc compare data/strain128/run/results.npz data/strain128/truth.npz
```

```text
points 1331, GOOD 1331 (100.00 %)  [GOOD 1331]
rmse (x, y, z)  0.0101  0.0098  0.0101
bias (x, y, z)  -0.0012  -0.0001  -0.0008
|error| median 0.0151  p95 0.0285  p99 0.0349
```

What this shows:

* The strain error is about 0.0004 (0.04 % strain) at the 729 interior
  points, and `strain_sd` predicts it well there. Near the grid's edges and corners the
  windows are lopsided and the error is larger; `sd_*` rises there too.
* `displacement_sd` (median ~0.008 voxel per axis) is somewhat below the
  actual displacement error (RMSE ~0.010). Here most of the error comes from
  the added noise; on real data the choice of sample points dominates, which
  is what `uncertainty_seeds` measures (next section).
* An `affine` field has constant strain, the easiest case for a quadratic fit.
  Where strain varies over the window's size, the fit smooths it: the window
  sets the spatial resolution of the strain.

## 5. On real data: case A

The same two steps on iDVC's example scan ({doc}`/tutorials/idvc_example`),
using `caseA.yaml` from that tutorial with `workdir: caseA/work_sd` and
`uncertainty_seeds: 2` (saved as `caseA_sd.yaml`):

```bash
zvdvc solve caseA_sd.yaml --backend fused --quiet
zvdvc strain caseA/work_sd/results.npz
```

```text
caseA/work_sd/results.npz: 4680 points in 2.4 s (1961.9 pt/s), status counts {-1: 52, 0: 4628}
{
 …
 "points": 4680,
 "with_strain": 4680,
 "planar_axis": 2,
 …
}
```

The RMS of `displacement_sd` over the GOOD points is (0.0328, 0.0309, 0.0322)
voxel. That matches the error-floor study's measured uncertainty from the
choice of sample points, about 0.03 voxel per axis at 8 000 samples
({doc}`/benchmarks/2026-09-26-error-floor-case-A`, section 3). The grid is one
slice (z = 630), so the strain is fitted in the xy plane (`planar_axis` 2).

On this scan that sampling uncertainty is the largest random error. It falls
as $1/\sqrt{n}$ with the sample count:

| samples | 2 000 | 4 000 | 8 000 | 16 000 | 32 000 |
|---|---|---|---|---|---|
| per-axis σ (voxel) | 0.064 | 0.045 | 0.032 | 0.022 | 0.016 |

To halve it, use 4× the samples (`subvolume.n_samples`), at about 4× the
solve time.

## 6. Interpolation bias and `prefilter_sigma`

Random error averages out over a strain window. Interpolation bias does not.
Tricubic interpolation is least accurate between voxels, so a displacement's
error depends on its fractional part: on the case A scan it traces an S-curve
of about ±0.03 voxel, zero at whole and half voxels. Its steepest slope is
about 0.19 voxel of bias per voxel of displacement. Where the fractional
displacement changes across a region, the bias adds that much to the
measured gradient: up to ~20 % of the local strain.

A Gaussian low-pass filter on both images removes most of it
({doc}`/benchmarks/2026-09-26-error-floor-case-A`, section 4; sphere 80,
8 000 samples):

| prefilter σ (voxel) | max bias | worst strain error (max slope) | sample-set uncertainty |
|---|---|---|---|
| none | 0.030 | 0.19 (19 %) | 0.031 |
| 0.7 | 0.0022 | 0.014 (1.4 %) | 0.026 |
| 1.0 | 0.0005 | 0.003 (0.3 %) | 0.025 |

On this scan the filter also lowered the random error, because it removes
noise that differs between the two scans. On data whose texture is at the
voxel scale, filtering removes information and raises the random error, so
choose σ per dataset.

In zvDVC the filter is `volumes.prefilter_sigma`, in voxels:

```yaml
volumes:
  prefilter_sigma: 1.0      # default 0.0: no filter, as CCPi
```

Both volumes are filtered as they are read. Each brick is read with a margin
of 4σ, filtered and cropped, so neighbouring tiles meet without seams. The
filtered data keeps the volume's type (u8 stays u8). The results store
records σ, and adding results made with another σ to it is refused. On case
A, σ = 1 took 5.9 s per 2.4 GB volume and moved the central grid's result by
at most 0.003 voxel on average.

On the synthetic case above, σ = 1 halves the displacement error:

```bash
zvdvc solve data/strain128/config_pf1.yaml --backend fused --quiet   # prefilter_sigma: 1.0, workdir run_pf1
zvdvc compare data/strain128/run_pf1/results.npz data/strain128/truth.npz
```

```text
points 1331, GOOD 1331 (100.00 %)  [GOOD 1331]
rmse (x, y, z)  0.0045  0.0051  0.0048
bias (x, y, z)  -0.0003  -0.0003  -0.0002
|error| median 0.0073  p95 0.0136  p99 0.0169
```

On this phantom the gain comes mostly from filtering the added white noise.
The phantom's speckle is smooth at the voxel scale, so it has little
interpolation bias to remove. Real-data numbers should come from a study like
the error-floor one, not from a phantom.

## Summary

* Use `uncertainty_seeds: 2` whenever you report displacements or strain:
  it costs about one extra solve and gives `displacement_sd` and `sd_*`.
* For strain on data like the case A scan, use `prefilter_sigma` of about 1
  voxel, after checking on your own data that it does not raise the random
  error.
* More samples reduce the sampling uncertainty as $1/\sqrt{n}$; a larger
  strain window reduces strain noise at the cost of spatial resolution.
* The method is specified in {doc}`/spec/method`.
