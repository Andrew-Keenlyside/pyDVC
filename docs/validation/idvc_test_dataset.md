# Agreement with iDVC on its test dataset

This page compares zvDVC with iDVC point by point, on iDVC's own example data
and settings. It shows how closely the two agree, and how that compares with
iDVC's agreement with itself.

[iDVC](https://github.com/TomographicImaging/iDVC) (Tomographic Imaging /
CCPi, UKRI-STFC) runs the CCPi DVC engine, `dvc` 22.0.0, on a parameter file it
writes. Every "iDVC" result here is that engine, run the way iDVC runs it: one
process on every point, from an iDVC-format `dvc_in` file. zvDVC runs in its
CCPi-parity mode: wavefront seeding in CCPi's point order, with the same
settings.

## The test dataset

| | |
|---|---|
| scans | iDVC's example: synthetic magma before and after deformation, two 1520 × 1257 × 1260 u8 volumes (Lee, Lavallée and Bay, 2022, Zenodo [7363345](https://zenodo.org/records/7363345)) |
| points | CCPi's `dvc_test/central_grid.roi`: 4 680 points, a 90 × 52 grid in slice z = 630 |
| settings | CCPi's `dvc_input.txt`, as iDVC uses them: sphere subvolumes of diameter 80 with 8 000 samples, 6-DOF, ZNSSD, tricubic interpolation, `disp_max` 38, rigid offset (34, 4, 0) |

The 52 points whose subvolume reaches the image edge are left out. CCPi reads
outside the image there, and zvDVC reports them `RANGE_FAIL`. That leaves
4 628 interior points, every one `GOOD` in both codes.

## Point by point

```{figure} figures/idvc_scatter.png
:alt: Three scatter plots of zvDVC's displacement against iDVC's for u, v and w. The points lie on the line of equality across the whole range, for example u from 5 to 33 voxels; median differences 0.022 to 0.024 voxel.
:figclass: zv-figure
:width: 100%

zvDVC against iDVC's engine for each displacement component. Every point lies
on the line of equality; the median difference is 0.022–0.024 voxel per
component and the mean difference at most 0.0032 in magnitude.
```

## Compared with iDVC's own repeatability

iDVC draws each spherical subvolume's sample points at random, from a clock
seed, so two identical iDVC runs do not give identical results. The fair
question is therefore not "is zvDVC identical to iDVC?" but "does zvDVC differ
from an iDVC run more than a second iDVC run does?"

```{figure} figures/idvc_differences.png
:alt: Three histograms per component of the difference from one iDVC run: zvDVC's difference and a second iDVC run's difference. The two histograms coincide; standard deviations 0.048 and 0.048 for u, 0.043 and 0.044 for v, 0.045 and 0.046 for w.
:figclass: zv-figure
:width: 100%

Differences from one iDVC run, for zvDVC and for a second iDVC run with the
same settings. The distributions are the same: standard deviation 0.048,
0.043 and 0.045 voxel (u, v, w) for zvDVC against 0.048, 0.044 and 0.046 for
iDVC against itself.
```

It does not. zvDVC's results sit within iDVC's own run-to-run scatter, with no
systematic offset.

## The fields

```{figure} figures/idvc_fields_sphere.png
:alt: Nine maps of the z = 630 plane. iDVC's and zvDVC's u, v and w displacement fields are visually identical; the difference maps show unstructured noise of a few hundredths of a voxel, larger where the field changes fastest (x above about 800).
:figclass: zv-figure
:width: 100%

The displacement field from iDVC's engine and from zvDVC, and their difference.
The difference is unstructured noise, larger where the field changes fastest
(x above about 800), which is where the choice of sample points matters most.
```

## Where the remaining difference comes from

A separate study
({doc}`/benchmarks/2026-09-29-idvc-comparison`) takes the difference apart:

| Cause | Median \|Δu\| (voxel) |
|---|---|
| iDVC against itself, random sample points (as above) | 0.051 |
| zvDVC against iDVC, same settings | 0.051 |
| zvDVC against iDVC with the **same** sample points (cube subvolumes) | 0.0010 |
| iDVC's engine against a recompiled copy of itself | 1.7 × 10⁻⁵ |
| zvDVC against iDVC, same sample points, both run to convergence | 2.1 × 10⁻⁵ |

With identical sample points, the two codes differ by where each stops
iterating, and run to convergence they reach the same optimum to the precision
that separates two compilations of iDVC's own engine.

## Criteria

| Criterion (from the {doc}`MVP plan </MVP_PLAN>`) | Result |
|---|---|
| Status agreement ≥ 98 % | 100 % of interior points (98.9 % of all points, edge points included) |
| Mean difference ≤ 0.01 voxel per axis (revised Q1) | 0.0032 at most |
| Spread ≤ 1.25 × zvDVC's own seed-to-seed spread (revised Q1) | 0.98–1.03 × |
| Median \|Δu\| ≤ 0.05 voxel (original Q1) | 0.051: not met, but iDVC against itself gives 0.051 too, so no code can meet it |

Reproduce: {doc}`/tutorials/idvc_example` and
`python -m zvdvc.bench.idvc_study`; the figures are made by
`python -m zvdvc.bench.doc_figures`.
