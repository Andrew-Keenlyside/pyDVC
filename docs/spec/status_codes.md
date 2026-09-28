# Status codes

This page lists the per-point status codes zvDVC records, when each is
assigned, what it corresponds to in CCPi DVC, and how it is written to the
CCPi-format `.disp` and `.stat` files. It is for anyone filtering results or
comparing them with CCPi or [iDVC](https://github.com/TomographicImaging/iDVC).
Source: [`status.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/status.py),
[`solver/gauss_newton.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/solver/gauss_newton.py),
[`solver/engines.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/solver/engines.py),
[`io/ccpi.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/io/ccpi.py).

## Terms

Status
: A small signed integer per point, stored as int8 in the `status` attribute
  of the results store ({doc}`results_store`), in the `status` column of
  `.disp` files, and in `.npz` results. The enum is
  {py:class}`zvdvc.status.PointStatus`.

CCPi code
: The codes 0 to −3 defined by CCPi DVC (`Core/Utility.h`). zvDVC keeps their
  values and meaning, so exported `.disp` files read correctly in iDVC.

zvDVC-only code
: A code below −3, added by zvDVC for cases CCPi does not distinguish.

---

## Introduction

Only `GOOD` (0) points carry a trustworthy displacement. Every other status
says why a point has no result. Codes 0 to −3 are CCPi's; −4 and −5 are
zvDVC's. When results are exported for CCPi tools, each zvDVC-only code is
written as −3 (`NOT_SEARCHED`), the CCPi code for "no result", so iDVC and
CCPi's `strain` program never see a code they do not know.

---

## Technical reference

### Codes

| value | name | CCPi equivalent | assigned when |
|---|---|---|---|
| 0 | `GOOD` | 0, point good | Gauss–Newton met a stopping rule (`obj_tol` or `disp_tol`) within `disp_max` of the seed, and every sample stayed in the brick's valid region. Also points that reached `max_iterations` when `search.report_convg_fail` is false (CCPi's behaviour). |
| −1 | `RANGE_FAIL` | −1, range fail | The translation moved more than `disp_max` from the seed ($\lVert\mathbf{t} - \text{seed}\rVert_\infty > \texttt{disp\_max}$) after a step, or any sample's interpolation stencil left the brick's valid region: at the reference sampling, in a Gauss–Newton iteration, or at the final objective. The second case includes subvolumes that leave the image, which CCPi does not flag. |
| −2 | `CONVG_FAIL` | −2, convergence fail | `max_iterations` steps were taken without meeting a stopping rule, and `search.report_convg_fail` is true (the default). CCPi's solver path never reports this code. |
| −3 | `NOT_SEARCHED` | −3, not searched | Never attempted: beyond `num_points_to_process` in an in-memory solve, or pending. |
| −4 | `THRESH_FAIL` | none (CCPi skips the point) | `search.threshold` is set and the fraction of reference samples within `[gray_min, gray_max]` is below `min_fraction` (CCPi `subvol_thresh`). The point is not searched. |
| −5 | `SINGULAR` | none | The normal equations are ill-conditioned: a non-positive diagonal of $\mathbf{H}$, or a Cholesky pivot of the Jacobi-scaled system at or below $10^{-6}$ (float32 engines; $10^{-10}$ for the float64 reference); or the target is featureless, $\sum\lVert\nabla g\rVert^2 \le 10^{-12}\sum g^2$. CCPi's QR solve has no such test. |

A point's status is set by the first failure met, in the order of the
per-point pipeline ({doc}`method`): reference sampling, threshold test,
Gauss–Newton iterations (range, singularity), iteration limit, final
objective.

`PointStatus.to_ccpi()` maps a code to CCPi's range: codes $\ge -3$ are
unchanged, codes $< -3$ become −3.

### How statuses are written

| output | written by | codes |
|---|---|---|
| results store `status` | `zvdvc seed` (wavefront), `zvdvc run` | zvDVC codes, unchanged |
| `.npz` results | `zvdvc solve` | zvDVC codes, unchanged |
| `.disp` | `zvdvc finalize --disp`, `zvdvc solve --disp` (`io.ccpi.write_disp`) | mapped with `to_ccpi()`: −4 and −5 become −3. Displacement and `objmin` are written as stored. |
| `.stat` | `zvdvc finalize`, `zvdvc solve` (`io.ccpi.write_stat`) | one `status <NAME>\t<count>` line per status present, with zvDVC's names, **not** mapped (e.g. `status SINGULAR\t2`) |
| `<output_filename>.disp` | `zvdvc-dvc` / `zvdvc ccpi` drop-in | mapped with `to_ccpi()`. For every code other than 0 the displacement is written as `0.000000`, as CCPi writes failed points; a non-finite `objmin` is written as 0. |
| `<output_filename>.stat` | drop-in | CCPi's three counts, after mapping: `number successful` (0), `number range fail` (−1), `number convg fail` (−2). Points mapped to −3 are in none of them. |
| progress lines on stdout | drop-in | `Point_Good` (with objective and displacement), `Range_Fail`, `Convg_Fail`; any other mapped code prints `Not_Searched` |

The mapping is tested in
[`tests/test_ccpi_io.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/tests/test_ccpi_io.py):
CCPi's codes keep their values (0, −1, −2, −3), `SINGULAR.to_ccpi()` is
`NOT_SEARCHED`, and a `.disp` written with `SINGULAR` and `THRESH_FAIL` points
reads back as −3 for both.

```{note}
In a `.disp` exported by `zvdvc finalize --disp`, failed points keep the
parameters they had when they failed. Always select `status == 0` before using
displacements. Files written by the `zvdvc-dvc` drop-in zero them, as CCPi does.
```
