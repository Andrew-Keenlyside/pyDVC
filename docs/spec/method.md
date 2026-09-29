# Correlation method

This page specifies the correlation zvDVC performs at each search point: the
sample template, the shape functions, the objectives, the interpolation, the
Gauss–Newton solver and its stopping rules, the coarse grid search, seeding,
the displacement uncertainty and strain. It is for users who need to know
exactly what a result means, and for anyone comparing zvDVC with CCPi DVC.
The method is that of the [CCPi DVC engine](https://github.com/TomographicImaging/DigitalVolumeCorrelation)
run by [iDVC](https://github.com/TomographicImaging/iDVC), developed by
Bay and collaborators (Bay et al. 1999, doi:[10.1007/BF02323555](https://doi.org/10.1007/BF02323555);
Bay 2008, doi:[10.1243/03093247JSA436](https://doi.org/10.1243/03093247JSA436));
each section says where zvDVC follows CCPi and where it does not.

## Terms

Search point (point)
: A location $\mathbf{c} = (x, y, z)$ in the reference volume, in voxels,
  where a displacement is measured (CCPi "search point").

Subvolume
: The region around a point that is matched between the two volumes. Its
  shape is `subvolume.geometry` (cube or sphere) and its size
  `subvolume.size` (cube side or sphere diameter, voxels).

Template
: The set of $M$ sample offsets $\mathbf{d}_m$ (relative to the point) that
  represent the subvolume. zvDVC builds **one template per run** and shares it
  across all points.

Reference samples $f$, target samples $g$
: The reference volume interpolated at $\mathbf{c} + \mathbf{d}_m$, and the
  deformed volume interpolated at the warped positions $\mathbf{x}'_m$.

Parameter vector $\mathbf{p}$
: The 3, 6 or 12 shape-function parameters of one point, in CCPi's order.
  Its first three entries are the displacement $(u, v, w)$.

Seed
: The starting displacement of a point. Only the translation is seeded;
  rotations and strains start at zero, as in CCPi.

Objective
: The matching criterion minimised over $\mathbf{p}$: SAD, SSD, ZSSD, NSSD or
  ZNSSD. Its value at the solution is reported as `objmin`.

Brick
: The dense block of image data a tile of points is solved against
  ({doc}`volumes`). Its *valid* region is the part that lies inside the volume.

---

## Introduction

For each point, zvDVC samples the reference volume at the template offsets
once, then searches for the parameter vector that makes the deformed volume,
sampled at the warped template, match those reference samples best. The
search is Gauss–Newton on the chosen objective, started from a seed
displacement. It stops when the objective or the translation step becomes
small, or after `search.max_iterations` steps, and records a status
({doc}`status_codes`).

This is CCPi's per-point search (`Search::process_point`). What zvDVC changes
is how the work is organised: all points share one template, many points are
solved at once, derivatives are analytic rather than finite differences, and
the serial, neighbour-by-neighbour seeding order is replaced by strategies
that allow batching. The full list of differences is in
{doc}`/ARCHITECTURE` §10 and in [Divergences from CCPi](#divergences-from-ccpi) below.

The per-point pipeline, as in
[`solver/gauss_newton.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/solver/gauss_newton.py):

1. Sample the reference brick at $\mathbf{c} + \mathbf{d}_m$ to get $f$
   (once per point). A point whose reference stencil leaves the brick's valid
   region is `RANGE_FAIL`.
2. Optional threshold test (`search.threshold`) → `THRESH_FAIL`; then a
   reference with no texture → `SINGULAR`.
3. $\mathbf{p} \leftarrow (\text{seed}, 0, \dots, 0)$.
4. Optional translation grid search (`search.basin_radius > 0`).
5. Up to `search.max_iterations` Gauss–Newton steps on the points still active.
6. Exact final objective at the final parameters, and the status. Non-finite
   parameters or objective, or a target with no texture, → `SINGULAR`.

---

## Technical reference

### Sample templates

Source: [`geometry/templates.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/geometry/templates.py).

As in CCPi, samples are not voxel-centred, and their number is independent of
the subvolume size; interpolation supplies values at arbitrary positions. Let
$r = \tfrac{1}{2}\,\texttt{size}$ and $\mathbf{a} = \texttt{aspect}$ (per axis
$x, y, z$).

**Cube.** A $k \times k \times k$ lattice, where $k$ is the smallest integer
with $k^3 \ge \texttt{n\_samples}$ (computed exactly in integers). Along each
axis the lattice runs over `linspace(-1, 1, k)` scaled by $r\,a_i$, so the
outer samples lie on the cube faces. The template therefore has $k^3$ samples,
which can be more than `n_samples`; CCPi rounds the count up the same way.

**Sphere.** `n_samples` points uniform inside the ellipsoid of semi-axes
$r\,\mathbf{a}$: candidates are drawn uniformly in $[-1, 1]^3$ with a
Mersenne Twister (`numpy.random.Generator(MT19937(seed))`), those inside the
unit ball are kept, and the accepted points are scaled by $r\,\mathbf{a}$.
CCPi draws its sphere samples with `std::mt19937`; the streams differ, so
agreement with CCPi is statistical, not bitwise.

**Order.** Offsets are stored $(x, y, z)$ as float32 and ordered by $z$, then
$y$, then $x$ (the cube is built in that order; sphere samples are sorted).
Neighbouring GPU threads then read neighbouring voxels.

**Seed.** `subvolume.seed` selects the sphere's random stream. One template is
shared by every point of a run. Changing the seed changes which sample points
each subvolume uses; on real data this is the largest random error
([error-floor study](../benchmarks/2026-09-26-error-floor-case-A.md)), and
`uncertainty_seeds` measures it per point ([below](#displacement-uncertainty)).

**Extent.** `Template.extent()` is the largest $|\mathbf{d}_m|$. Rotations can
turn any sample along any axis, so this sizes the brick halo ({doc}`volumes`).

**Digest.** `Template.digest()` is the first 16 hexadecimal characters of the
SHA-256 hash of the float32 offset bytes. It is recorded in `plan.json` and in
the results store, and a resumed run whose template digest differs is refused
({doc}`results_store`).

### Shape functions (3, 6 and 12 DOF)

Source: [`geometry/warp.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/geometry/warp.py).

The parameter vector follows CCPi's `SearchParams`. Angles are in radians and
strains are dimensionless:

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| name | `u` | `v` | `w` | `phi` | `the` | `psi` | `exx` | `eyy` | `ezz` | `exy` | `eyz` | `exz` |
| 3-DOF | ✓ | ✓ | ✓ | | | | | | | | | |
| 6-DOF | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | | | | | |
| 12-DOF | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

A template offset $\mathbf{d}$ around centre $\mathbf{c}$ maps to

$$
\mathbf{x}' = \mathbf{c} + \mathbf{t} + (\mathbf{I} + \mathbf{E})\,\mathbf{R}\,\mathbf{d},
\qquad \mathbf{t} = (u, v, w).
$$

This matches CCPi's `FloatingCloud::affine_to`: translate, rotate
($\mathbf{d} \to \mathbf{R}\mathbf{d}$), then strain
($\mathbf{d}' \to \mathbf{d}' + \mathbf{E}\mathbf{d}'$), both relative to the
moved point. 3-DOF has $\mathbf{R} = \mathbf{I}$, $\mathbf{E} = 0$; 6-DOF has
$\mathbf{E} = 0$.

**Rotation.** CCPi's roll–pitch–yaw matrix (aerospace 3-2-1),
$\mathbf{R} = \mathbf{R}_x(\psi)\,\mathbf{R}_y(\theta)\,\mathbf{R}_z(\phi)$, with

$$
\mathbf{R}_z(\phi) = \begin{pmatrix} \cos\phi & \sin\phi & 0 \\ -\sin\phi & \cos\phi & 0 \\ 0 & 0 & 1 \end{pmatrix},\quad
\mathbf{R}_y(\theta) = \begin{pmatrix} \cos\theta & 0 & -\sin\theta \\ 0 & 1 & 0 \\ \sin\theta & 0 & \cos\theta \end{pmatrix},\quad
\mathbf{R}_x(\psi) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\psi & \sin\psi \\ 0 & -\sin\psi & \cos\psi \end{pmatrix}.
$$

$\phi$ (`phi`) is yaw about $z$, $\theta$ (`the`) pitch about $y$, $\psi$
(`psi`) roll about $x$.

**Strain.** The symmetric tensor with tensor (not engineering) shears:

$$
\mathbf{E} = \begin{pmatrix} e_{xx} & e_{xy} & e_{xz} \\ e_{xy} & e_{yy} & e_{yz} \\ e_{xz} & e_{yz} & e_{zz} \end{pmatrix}.
$$

**Jacobian of the warp.** The solver needs $\partial\mathbf{x}'/\partial\mathbf{p}$
at the current $\mathbf{p}$. For the translations it is $\mathbf{I}$. For the
other parameters it is $\mathbf{A}_k\,\mathbf{d}$, with

$$
\mathbf{A}_\phi = (\mathbf{I}+\mathbf{E})\,\mathbf{R}_x\mathbf{R}_y\mathbf{R}_z',\quad
\mathbf{A}_\theta = (\mathbf{I}+\mathbf{E})\,\mathbf{R}_x\mathbf{R}_y'\mathbf{R}_z,\quad
\mathbf{A}_\psi = (\mathbf{I}+\mathbf{E})\,\mathbf{R}_x'\mathbf{R}_y\mathbf{R}_z,\quad
\mathbf{A}_{e_k} = \frac{\partial\mathbf{E}}{\partial e_k}\,\mathbf{R},
$$

where a prime is the derivative of that factor with respect to its angle. It
is computed analytically per point and iteration (`warp_linear_jacobians`);
the $(B, M, 3, \text{ndof})$ array is never formed on the fused path.

### Objectives

Source: [`kernels/objective.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/kernels/objective.py).

CCPi's five criteria (`ObjectiveFunctions.cpp`), with identical scaling. For
the $M$ reference samples $f$ and target samples $g$ of one point, with
$\bar f$ the mean, $\tilde f = f - \bar f$ and $\lVert\cdot\rVert$ the
Euclidean norm over samples:

| `search.objective` | value | range |
|---|---|---|
| `sad` | $\sum_m \lvert g_m - f_m \rvert$ | $\ge 0$ |
| `ssd` | $\sum_m (g_m - f_m)^2$ | $\ge 0$ |
| `zssd` | $\sum_m (\tilde g_m - \tilde f_m)^2$ | $\ge 0$ |
| `nssd` | $\tfrac12 \sum_m \left(\dfrac{g_m}{\lVert g\rVert} - \dfrac{f_m}{\lVert f\rVert}\right)^2$ | $[0, 1]$ |
| `znssd` (default) | $\tfrac14 \sum_m \left(\dfrac{\tilde g_m}{\lVert\tilde g\rVert} - \dfrac{\tilde f_m}{\lVert\tilde f\rVert}\right)^2$ | $[0, 1]$ |

ZNSSD equals $(1 - \rho)/2$, where $\rho$ is the zero-normalised
cross-correlation coefficient, so it is insensitive to a linear change of grey
levels $g \to a g + b$ with $a > 0$.

**Residuals.** Gauss–Newton minimises the sum of squares of the residual
vector $\mathbf{r}$ (the bracketed terms above). For `sad` the residual is
SSD's, so the step is taken as if the objective were SSD, exactly as in CCPi.
The reported `objmin` is always the exact value in the table.

**Exact normalisation derivative.** Let $\mathbf{J}$ be the target Jacobian,
$J_{mk} = \nabla g(\mathbf{x}'_m)^{\mathsf T}\,\partial\mathbf{x}'_m/\partial p_k$,
$\mathbf{m} = \tfrac1M \sum_m \mathbf{J}_m$ its sample mean,
$\hat g = \tilde g / \lVert\tilde g\rVert$ (or $g/\lVert g\rVert$ for NSSD) and
$\mathbf{a} = \hat g^{\mathsf T}\mathbf{J}$. The normal equations use the exact
Jacobian of the normalised residual:

| objective | $\mathbf{H}$ | $\mathbf{b}$ |
|---|---|---|
| `sad`, `ssd` | $\mathbf{J}^{\mathsf T}\mathbf{J}$ | $\mathbf{J}^{\mathsf T}\mathbf{r}$ |
| `zssd` | $\mathbf{J}^{\mathsf T}\mathbf{J} - M\,\mathbf{m}\mathbf{m}^{\mathsf T}$ | $\mathbf{J}^{\mathsf T}\mathbf{r}$ |
| `nssd` | $(\mathbf{J}^{\mathsf T}\mathbf{J} - \mathbf{a}\mathbf{a}^{\mathsf T}) / \lVert g\rVert^2$ | $(\mathbf{J}^{\mathsf T}\mathbf{r} - \mathbf{a}\,(\hat g\cdot\mathbf{r})) / \lVert g\rVert$ |
| `znssd` | $(\mathbf{J}^{\mathsf T}\mathbf{J} - M\,\mathbf{m}\mathbf{m}^{\mathsf T} - \mathbf{a}\mathbf{a}^{\mathsf T}) / \lVert\tilde g\rVert^2$ | $(\mathbf{J}^{\mathsf T}\mathbf{r} - \mathbf{a}\,(\hat g\cdot\mathbf{r})) / \lVert\tilde g\rVert$ |

and the step solves $\mathbf{H}\,\Delta\mathbf{p} = -\mathbf{b}$. Holding the
target mean and norm fixed within an iteration, a common shortcut, would move
the fixed point by a term proportional to $(1-\rho)\,\partial\lVert\tilde g\rVert/\partial\mathbf{p}$.
CCPi differentiates its full objective numerically, so the exact form is also
the parity choice.

**One-pass sums.** Every engine reduces a point's samples to one fixed row of
sums, from which $\mathbf{H}$, $\mathbf{b}$ and the objective follow exactly
(`sum_layout`, `normal_equations_from_sums`): the upper triangle of
$\sum \mathbf{j}\mathbf{j}^{\mathsf T}$, $\sum\mathbf{j}$, $\sum v\,\mathbf{j}$,
$\sum q\,\mathbf{j}$ and the scalars $\sum v$, $\sum v^2$, $\sum vq$, $\sum q$,
$\sum q^2$, $n$ and $\sum\lvert v - q\rvert$, where $q$ is the reference term of
the residual and $v = g - s$ is the target shifted by the reference mean $s$
(zero-mean objectives) or by 0. The shift keeps $\sum v^2 - n\bar v^2$ free of
cancellation in float32. No objective needs the target mean or norm before the
sums exist, so one pass over the samples suffices.

### Interpolation

Source: [`kernels/interpolate.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/kernels/interpolate.py).

Positions are point-space $(x, y, z)$; voxel $i$ sits at coordinate $i$
({doc}`volumes`). With $t = x - \lfloor x\rfloor$ per axis:

| `search.interpolation` | taps per axis | stencil (per axis) | gradient |
|---|---|---|---|
| `nearest` | 1 | $\lfloor x + 0.5\rfloor$ | zero |
| `trilinear` | 2 | $\lfloor x\rfloor,\ \lfloor x\rfloor + 1$ | piecewise constant |
| `tricubic` (default) | 4 | $\lfloor x\rfloor - 1 \dots \lfloor x\rfloor + 2$ | continuous |

**Tricubic** is separable Catmull-Rom (cubic convolution with $a = -\tfrac12$),
64 voxel reads per sample, with weights for taps $-1, 0, 1, 2$:

$$
w_{-1} = \tfrac12(-t^3 + 2t^2 - t),\quad
w_0 = \tfrac12(3t^3 - 5t^2 + 2),\quad
w_1 = \tfrac12(-3t^3 + 4t^2 + t),\quad
w_2 = \tfrac12(t^3 - t^2),
$$

and their derivatives for the gradient. CCPi's tricubic is the Lekien–Marsden
interpolant fed with central-difference derivatives ($f$, $f_x$, $f_y$, $f_z$,
$f_{xy}$, $f_{xz}$, $f_{yz}$, $f_{xyz}$) precomputed over each point's box. A
tensor-product cubic Hermite interpolant with tensor-product central-difference
derivatives is tensor-product Catmull-Rom, so the two give the same values;
`lekien_marsden_reference` is a float64 port of CCPi's path, used by the tests
to check this.

**Trilinear** has 8 taps. It is adequate for tuning but biased for strain.
**Nearest** has a zero gradient, so Gauss–Newton steps computed with it do
not move the point; the source reserves it for grid searches. The translation
grid search uses the same `search.interpolation` as the solve.

**Validity.** A sample whose whole stencil is not inside the brick's valid
region is marked outside. Any outside sample makes the point `RANGE_FAIL`
(reference sampling, any Gauss–Newton iteration, or the final objective).
Values outside the volume are edge-padded in the brick but never used.

**Precision.** Device arithmetic is float32 (the `numpy` reference engine is
float64). Each point's brick-local centre is split into an integer voxel and a
fraction, and sample positions are formed from the fraction, so precision does
not depend on the coordinate.

### Prefilter

Source: [`io/volume.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/io/volume.py) (`FilteredVolume`).

`volumes.prefilter_sigma` $> 0$ applies a Gaussian low-pass of that standard
deviation (voxels) to both volumes as they are read. The kernel is truncated
at $4\sigma$; outside the volume, voxels are edge-padded before filtering.
Each brick is read with a margin of $\lceil 4\sigma\rceil$ voxels, filtered and
cropped, so a brick equals the same box of the filtered whole volume and tiles
meet without seams. The result keeps the volume's dtype (integer volumes are
rounded to the nearest level and clipped). It runs on the GPU when CuPy has a
device, else on the host.

The default, 0, is CCPi's behaviour (no filter). About 1 voxel removes most
interpolation bias ([error-floor study](../benchmarks/2026-09-26-error-floor-case-A.md)).
The value is recorded in the results store, and a resumed run with another
value is refused.

### Gauss–Newton solvers

Source: [`solver/gauss_newton.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/solver/gauss_newton.py),
[`solver/engines.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/solver/engines.py).

**FA-GN (`search.method: fagn`, the default).** Forward-additive Gauss–Newton,
CCPi's scheme: at each iteration form $\mathbf{H}$ and $\mathbf{b}$ at the
current $\mathbf{p}$, solve $\mathbf{H}\,\Delta\mathbf{p} = -\mathbf{b}$ and set
$\mathbf{p} \leftarrow \mathbf{p} + \Delta\mathbf{p}$, with no damping. CCPi
forms the Jacobian by forward differences ($h = 10^{-10}$, $\text{ndof}+1$
objective evaluations per step) and solves $\mathbf{J}^{\mathsf T}\mathbf{J}$
by column-pivoted QR. zvDVC uses the analytic Jacobian (one value-and-gradient
pass) and a batched Cholesky solve:

1. Jacobi scaling: $\mathbf{A} = \mathbf{D}^{-1/2}\mathbf{H}\mathbf{D}^{-1/2}$
   with $\mathbf{D} = \operatorname{diag}\mathbf{H}$.
2. Cholesky of $\mathbf{A}$. A non-positive diagonal of $\mathbf{H}$, or a
   pivot at or below $10^{-6}$ (float32 engines; $10^{-10}$ for the float64
   reference), makes the point `SINGULAR`.
3. A point whose target has no texture,
   $\sum\lVert\nabla g\rVert^2 \le 10^{-12}\sum g^2$, is also `SINGULAR`.
4. Before any iteration, for every objective, a point whose reference
   subvolume has no texture, $\max f - \min f \le \tau\max\lvert f\rvert$
   ($\tau = 10^{-5}$ on the float32 engines, $10^{-12}$ on the float64
   reference), is `SINGULAR` and is not searched (zero padding, air). The
   final target samples get the same test, and a point whose parameters or
   final objective are not finite is `SINGULAR` too, so a non-finite result
   is never reported `GOOD`.

**IC-GN (`search.method: icgn`).** Inverse-compositional Gauss–Newton is
**not implemented**. Loading a configuration with `icgn` raises `ValueError`
(roadmap milestone M5). Its warp helpers
(`warp_jacobian_at_identity`, `compose_inverse`) are stubs.

**Engines.** The loop is written once; an engine supplies sampling, the
one-pass sums and the update:

| backend | engine | precision |
|---|---|---|
| `numpy` | array path on numpy: the reference | float64 |
| `numpy32` | the same in float32, for tests without a GPU | float32 |
| `cupy` | array path on CuPy: unfused GPU | float32 |
| `fused` | CUDA kernels (`kernels/cuda/fused_gn.cu`) | float32 |
| `emulated` | the CUDA source run on the host (tests only) | float32 |
| `cpu` | the fused step in numba on all cores | float32 |

The default is `fused` with a GPU, else `cpu` if numba is installed, else
`numpy`.

### Stopping rules and iteration limits

After each step, a point stops when either

$$
\lvert O_k - O_{k-1}\rvert < \texttt{obj\_tol}
\quad\text{or}\quad
\lVert\Delta\mathbf{t}\rVert_2 < \texttt{disp\_tol},
$$

where $O_k$ is the objective at the parameters of iteration $k$ and
$\Delta\mathbf{t}$ the translation part of the step. These are CCPi's rules
and defaults: `obj_tol` $= 10^{-6}$, `disp_tol` $= 0.01$ voxel, at most
`max_iterations` $= 20$ steps. `n_iter` records the steps taken.

A point that is still active after `max_iterations` steps is `CONVG_FAIL` when
`search.report_convg_fail` is true (the default). CCPi's Gauss–Newton path
never reports a convergence failure; `report_convg_fail: false` reproduces
that by marking such points `GOOD`, and CCPi-parity configurations set it.

**Range test.** After each step, a point with
$\lVert\mathbf{t} - \text{seed}\rVert_\infty > \texttt{disp\_max}$ is
`RANGE_FAIL`. CCPi's test is implicit: samples leaving a box of margin
`disp_max + 2` around the seeded subvolume.

The final objective is re-evaluated at the final parameters for `GOOD` and
`CONVG_FAIL` points; if any sample then leaves the valid region the point
becomes `RANGE_FAIL`, and if the objective is not finite or the target
samples have no texture it becomes `SINGULAR`.

### Threshold test

With `search.threshold` set (CCPi `subvol_thresh on`), the fraction of
reference samples $f_m$ with `gray_min` $\le f_m \le$ `gray_max` is computed
once per point. A point whose fraction is below `min_fraction` is
`THRESH_FAIL` and is not searched. CCPi skips such points too; zvDVC gives
them their own status ({doc}`status_codes`).

### Coarse translation grid search

Source: [`solver/coarse.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/solver/coarse.py).

CCPi's `basin_radius` search, batched. With `search.basin_radius` $> 0$, let
$n = \max(1, \lfloor\texttt{disp\_max}/\texttt{basin\_radius}\rfloor)$ and
step $s = \texttt{disp\_max}/n$. The objective is evaluated, translation only,
at the $(2n+1)^3$ translations $\text{seed} + s\,(i, j, k)$,
$i, j, k \in \{-n, \dots, n\}$, and the best becomes the starting translation
for Gauss–Newton. Translations whose samples leave the brick are never
chosen. The seed used for the range test is unchanged.

This is expensive: `disp_max = 38, basin_radius = 4` gives 6 859 evaluations
per point, which dominates the solve. Prefer good seeds. The default,
`basin_radius: 0`, disables it.

### Seeding

Source: [`solver/seeding.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/solver/seeding.py),
[`pipeline/coordinator.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/pipeline/coordinator.py).

CCPi sorts all points by distance from the start point and solves them one at
a time. Each starts from the mean $(u, v, w)$ of the already-solved `GOOD`
points among its 75 nearest neighbours, or from `rigid_trans` if there are
none; neighbours come from an $O(N^2)$ sort. That order makes point $k$ depend
on points $0 \dots k-1$, which prevents batching. `seeding.strategy` selects a
replacement:

| strategy | status | what it does |
|---|---|---|
| `rigid` | implemented | Every point starts at `search.rigid_trans`. Fully parallel; enough for small, smooth deformation or after rigid pre-alignment. |
| `wavefront` (default) | implemented | CCPi parity. Points are bucketed into distance shells from the start point and solved shell by shell, each shell as one batch seeded from `GOOD` neighbours in earlier shells. Needs both whole volumes in host memory. |
| `coarse` | implemented (MVP form) | Solve a sub-grid first (wavefront order), interpolate its field to seed every point, then solve all tiles independently. The MVP solves the sub-grid at full resolution with both whole volumes in host memory. |
| `fft` | not implemented (M5) | FFT cross-correlation seeds, path-independent. Loading a configuration with `fft` raises `ValueError`. |

**Wavefront details.** The start point is `seeding.start_point`, or the first
point of the cloud when unset (CCPi's default; from a zarr-vectors store, the
point with the lowest `point_id`). The point nearest the start is a shell of
its own, seeded from `rigid_trans`. Every other point falls in shell
$\lfloor \text{distance}/w\rfloor$, where $w$ is `seeding.shell_width`, or the
median nearest-neighbour spacing when unset. Seeds are the mean displacement
of the `GOOD` points among each point's `seeding.n_neighbours` nearest
neighbours (KD-tree, self excluded), else `rigid_trans`. The only difference
from CCPi is that points within one shell cannot seed each other. For a
$k^3$ lattice with the shell width equal to the spacing there are about
$1.7k$ shells from a corner.

**Coarse details.** The sub-grid is every `seeding.coarse_stride`-th lattice
point: space is cut into cubes of `coarse_stride` × median spacing and the
point nearest each cube's lower corner is kept. The sub-grid is solved with
the wavefront strategy. Each point's seed is then the inverse-distance-squared
weighted mean of the displacements of its 8 nearest `GOOD` sub-grid points (a
point that coincides with one takes its value; with no `GOOD` sub-grid point
every seed is `rigid_trans`). Seeds are saved in `<workdir>/seeds.npz` and the
tiles are re-planned from them. `seeding.coarse_level` (solving the sub-grid
on a pyramid level) is not used yet; that pass is M5.

**Memory guard.** The wavefront and coarse seed stages hold both whole
volumes in host memory. `zvdvc seed` refuses with `MemoryError` when
$2 \times \text{voxels} \times \text{bytes per voxel}$ exceeds 80 % of the
available host memory, and suggests `rigid`.

**Tile seeds.** In the tiled run each point's seed comes from `seeds.npz`
with `coarse`, else `rigid_trans`; a `seeds.npz` left in the work directory
does not seed another strategy's run. `zvdvc run` enforces the strategy: with
`coarse` it refuses to start until `zvdvc seed` has written `seeds.npz`, or
when `seeds.npz` changed after `seed` planned the tiles from it. With
`wavefront`, the seed stage itself solves every point and writes the results,
so `zvdvc run` refuses to start unless every cell is written, and then finds
every tile written and skips it.

### Repair

**Not implemented.** `zvdvc repair` stops with a "not implemented (M5)"
error, and `seeding.repair_passes` is not used (a value other than 1 warns). The planned pass
(`repair_candidates`, a stub) finds failed points with at least 3 `GOOD`
neighbours, re-seeds them from the neighbours' median displacement and solves
them again. {doc}`/ARCHITECTURE` §6 describes the design.

### Displacement uncertainty

Source: [`solver/uncertainty.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/solver/uncertainty.py).

With `uncertainty_seeds` $= k > 0$, every `GOOD` point is solved $k$ more
times, repeat $s$ using a template with seed `subvolume.seed + s`
($s = 1 \dots k$). Each repeat starts from the same seed displacement as the
main solve, not from its answer: a solve started at an answer stops within
`disp_tol` of it and would understate the spread. `displacement_sd` is the
per-axis sample standard deviation (divisor $n - 1$) over the `GOOD` estimates
among the $k + 1$; it is NaN where fewer than 2 are `GOOD`, and for points
that were not `GOOD` in the main solve.

With $k = 2$ the extra cost is about twice the main solve's, and each
per-point estimate has about 2 degrees of freedom: rough per point, good in
aggregate. The effect it measures (~0.03 voxel per axis at 8 000 samples on
the iDVC example, falling as $1/\sqrt{n}$) is described in the
[error-floor study](../benchmarks/2026-09-26-error-floor-case-A.md).

The tiled worker (`zvdvc run`), the wavefront seed stage (`zvdvc seed`) and
the in-memory runner (`zvdvc solve`, `.npz` output) all compute and keep
`displacement_sd`.

### Strain

Source: [`post/strain.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/post/strain.py).

`zvdvc strain RESULTS` computes strain from displacements as CCPi's `strain`
program does. `RESULTS` is a results store, a zvDVC `.npz` or a CCPi `.disp`.

1. **Window.** For each point, the `--window` (`-sw`, default 25) nearest
   points, the point itself included. Points are kept if `GOOD` and
   `objmin` $\le$ `--threshold` (`-t`, default 1.0). With `--refill` (`-r`),
   further neighbours replace dropped ones until the window is full.
2. **Fit.** Each displacement component is fitted by least squares
   with a full quadratic in $(x, y, z)$ (10 terms), centred on the point and
   scaled by the window radius. The displacement gradient
   $\mathbf{G} = \partial\mathbf{u}/\partial\mathbf{x}$ is the fit's linear
   terms at the point.
3. **Strain.** Engineering $\tfrac12(\mathbf{G} + \mathbf{G}^{\mathsf T})$ and
   Lagrangian $\tfrac12(\mathbf{G} + \mathbf{G}^{\mathsf T} + \mathbf{G}^{\mathsf T}\mathbf{G})$,
   with tensor shears, and their principal values in descending order.
4. **Planar clouds.** When every point shares one coordinate (like CCPi's
   central grid), the fit is done in that plane with the out-of-plane
   component and derivatives zero, as CCPi does.
5. **Undetermined windows.** Fewer good points than terms, or a design matrix
   whose smallest eigenvalue is below $10^{-9}$ of its largest, gives NaN,
   where CCPi returns an ill-determined value.

zvDVC adds, per point, `residual_rms` (the fit residual per component) and
`strain_sd` (the standard deviation of each engineering component, from
propagating a displacement uncertainty through the fit). The uncertainty is
`--sigma-u` if given, else the results' `displacement_sd` if present, else the
fit residual.

Outputs use CCPi's CSV layout: `<base>-sw<N>.Lstr.csv` (default),
`.Estr.csv` (`-E`, or `--engineering-only`), `.dgrd.csv` (`-D`), plus
`<base>-sw<N>.strain.npz`; the strain CSVs carry extra `sd_*` columns at the
end. Strain is not written into the results store. On the iDVC example, the
values match CCPi's to every printed digit at all interior points; within a
grid step of the cloud's edge a few windows can differ by one equidistant
point.

### Divergences from CCPi

| aspect | CCPi DVC | zvDVC |
|---|---|---|
| sample template | a `FloatingCloud` per point (`std::mt19937` sphere) | one shared template per run (MT19937 via numpy); agreement is statistical |
| point order and seeding | serial by distance; mean of `GOOD` among the 75 nearest solved | wavefront shells (parity), coarse field or rigid |
| neighbour search | $O(N^2)$ full sort | KD-tree |
| tricubic | Lekien–Marsden with precomputed derivative kernels per box | separable Catmull-Rom from the raw brick (same interpolant) |
| Jacobian | forward differences, $h = 10^{-10}$ | analytic, normalisation included |
| linear solve | column-pivoted QR of $\mathbf{J}^{\mathsf T}\mathbf{J}$ | batched Jacobi-scaled Cholesky, with `SINGULAR` |
| precision | float64 | float32 on device with relative coordinates; float64 reference |
| range failure | implicit (samples leave the box) | explicit $\lVert\mathbf{t}-\text{seed}\rVert_\infty > $ `disp_max`, plus brick validity |
| subvolume leaving the image | interpolates wrapped rows or unset memory | `RANGE_FAIL` |
| convergence failure | never reported | `CONVG_FAIL` unless `report_convg_fail: false` |
| prefilter | none | optional Gaussian (`prefilter_sigma`, default 0) |

See also {doc}`/ARCHITECTURE` §10 and the
[case A report](../benchmarks/2026-09-26-case-A-real.md) for the measured
agreement on the iDVC example (Lee, Lavallée and Bay 2022,
doi:[10.5281/zenodo.7363345](https://doi.org/10.5281/zenodo.7363345)).
