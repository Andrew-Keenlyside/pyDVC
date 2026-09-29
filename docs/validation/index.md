# Validation

This section collects the evidence that zvDVC's results are right: against a
known answer, against its own error floor, and against iDVC, the established
code it reimplements. It is for anyone deciding whether to rely on zvDVC's
numbers, and for reviewers of work that uses them.

Three kinds of evidence, each answering a different question:

| Question | Evidence | Page |
|---|---|---|
| Does zvDVC give the same answer as iDVC? | iDVC's engine (CCPi DVC 22.0.0) and zvDVC on iDVC's own example data and settings, point by point | {doc}`idvc_test_dataset` |
| Is the answer correct? | Synthetic volume pairs with a known displacement field, solved by both codes and scored against the truth | {doc}`ground_truth` |
| How precise can any answer be on real data? | The case A scan shifted by known sub-voxel amounts, and repeat solves with different sample points | {doc}`ground_truth` |

What is still to come, with public datasets that have ground truth or a noise
floor, is in {doc}`datasets`.

## Summary

| Check | Criterion | Result |
|---|---|---|
| Agreement with iDVC, case A (4 628 interior points) | status agreement ≥ 98 % | **100 %** |
| | mean difference ≤ 0.01 voxel per axis | **0.0032** at most |
| | spread no larger than iDVC's own run-to-run spread | **the same**: SD 0.048 / 0.043 / 0.045 voxel against 0.048 / 0.044 / 0.046 |
| | with identical sample points, both run to convergence | median **2.1 × 10⁻⁵** voxel, the order of iDVC against a recompiled iDVC |
| Accuracy against a known field, case S (affine, 2 % noise, 12-DOF) | RMSE ≤ 0.05 voxel, ≥ 99 % GOOD | **0.0103** per axis, 100 % GOOD; iDVC's engine: 0.0103 |
| Interpolation bias on the real scan | none required; reported | ±0.03 voxel without prefilter (as iDVC); **0.0005** with prefilter σ = 1 |
| Sampling uncertainty on the real scan | none required; reported | 0.032 voxel per axis at iDVC's 8 000 samples, falling as 1/√n |

```{figure} figures/idvc_scatter.png
:alt: zvDVC's displacement against iDVC's for u, v and w on case A: every point on the line of equality.
:figclass: zv-figure
:width: 100%

zvDVC against iDVC's engine on iDVC's example data, every interior point
({doc}`idvc_test_dataset`).
```

## What has not been validated yet

* **Other materials and scanners.** Case A is one real dataset. The public
  datasets in {doc}`datasets` add repeat scans (noise floor), rigid moves and
  imposed fields.
* **Strain.** zvDVC's strain follows CCPi's method and agrees with CCPi's
  `strain` at interior points ({doc}`/tutorials/strain_uncertainty`), but has
  not been checked against a known strain field on real data.
* **Large runs.** The tiled pipeline matches the in-memory solve bit for bit
  on the CPU engines; multi-GPU runs on 8 × H100 have not been made
  ({doc}`/MVP_REVIEW`).

## Pages

```{toctree}
:maxdepth: 1

idvc_test_dataset
ground_truth
datasets
```
