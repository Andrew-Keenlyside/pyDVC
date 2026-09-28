# Benchmarks

This page lists every speed and accuracy result measured so far, on what
hardware and against which baseline, and what is still unmeasured. It is for
anyone judging whether zvDVC's numbers apply to their own data and hardware,
and for anyone who wants to reproduce them. Each row links to a dated report
with the full setup.

```{important}
Everything here was measured on a 4-vCPU cloud VM or on one workstation with
an RTX A2000 12 GB. **Nothing has been measured on an H100 or on more than one
GPU yet**, and **GPUDirect Storage is not wired in**: every time uses host
decode plus one pinned host-to-device copy. The H100 and 8 × H100 figures in
{doc}`/PERFORMANCE` are **modelled, not measured**.
```

---

## Hardware

| name | CPU | memory | GPU | used for |
|---|---|---|---|---|
| cloud VM | 4 vCPU Intel Xeon @ 2.10 GHz | 15 GB | none | 2026-09-25 reports |
| `msm12` workstation | 32 cores, x86-64 | 251 GB | NVIDIA RTX A2000 12 GB (CUDA 12.9, cupy 14.2) | 2026-09-26 reports, check-suite baseline |

CCPi DVC is always version 22.0.0 (see {ref}`below <ccpi-version>`).

## Summary of measured results

### Speed

| what | hardware | baseline | result | report |
|---|---|---|---|---|
| CCPi on case S (256³ u16, 2 197 points, sphere 32, 12-DOF) | cloud VM | CCPi, 1 process | 37.5 pt/s; OpenMP threads give no gain; 4 processes give 123 pt/s (3.3×) | {doc}`2026-09-25-M0-M1-case-S` |
| zvDVC numpy reference, case S | cloud VM | CCPi, 1 process | 109 pt/s on 1 process (2.9× a CCPi process) | {doc}`2026-09-25-M0-M1-case-S` |
| zvDVC CPU engine (numba), case S, in memory | cloud VM | CCPi, 4 processes on the same 4 cores | 839 pt/s at 12-DOF (6.8×), 1 154 pt/s at 6-DOF (7.4×) | {doc}`2026-09-25-M2-M3-cpu` |
| zvDVC CPU engine, fused step at scenario B settings (4 096 samples, 6-DOF) | cloud VM | model: 375–1 100 pt/s per core | 443 pt/s on 4 cores, ~110 pt/s per core: below the model | {doc}`2026-09-25-M2-M3-cpu` |
| zvDVC tiled pipeline, case M (1024³ u16, 226 981 points, 8 tiles, CPU engine) | cloud VM | — | `run` stage (read, solve, write) 516 s, 440 pt/s; plan 0.6 s, finalize 0.2 s | {doc}`2026-09-25-M2-M3-cpu` |
| Case A twin (synthetic, case A geometry and settings) | cloud VM | CCPi as iDVC runs it: 1 069 s | zvDVC CPU engine 28.6 s in memory (37×), 47.6 s through the CLI (22×) | {doc}`2026-09-25-M4-case-A-twin` |
| **Case A, real data** (4 680 points, sphere 80, 8 000 samples, 6-DOF) | `msm12` | CCPi as iDVC runs it: 2 231 s (37 min) | **zvDVC GPU 1.6 s in memory (1 396×)**, 4.3 s through the CLI (524×); zvDVC CPU engine on 32 cores 7.6 s (293×), 17.1 s (131×); CCPi as 32 processes 711 s (3.1×) | {doc}`2026-09-26-case-A-real` |
| Case A, 285 480-point 3D grid | `msm12` | zvDVC CPU engine, 32 cores: ~300 s | GPU solve 18.7 s | {doc}`2026-09-26-case-A-real` |
| Fused kernel `gn_sums` on case A data | `msm12` | first GPU run: 38 µs per point-iteration | 3.7 µs per point-iteration after sorting template samples and packing u8 loads; ~1.2 TFLOP/s by operation count | {doc}`2026-09-26-case-A-real`, {doc}`/PERFORMANCE` §4.4 |

### Accuracy

| what | hardware | reference | result | report |
|---|---|---|---|---|
| Synthetic accuracy, case S, affine field, 12-DOF | cloud VM | ground truth; criterion RMSE ≤ 0.02 voxel | RMSE 0.0011 (CCPi 0.0010), 100 % GOOD | {doc}`2026-09-25-M0-M1-case-S` |
| The same with 2 % noise | cloud VM | ground truth; criterion RMSE ≤ 0.05 | RMSE 0.0103 (CCPi 0.0102) | {doc}`2026-09-25-M0-M1-case-S` |
| zvDVC against CCPi, case S (4 variants) | cloud VM | CCPi 22.0.0 `.disp` | median \|Δu\| 0.0016–0.0202, 100 % status agreement | {doc}`2026-09-25-M0-M1-case-S` |
| CUDA kernels without a GPU (host emulator) | cloud VM | float64 numpy reference | ≤ 3.3e-7 voxel, identical statuses and iteration counts | {doc}`2026-09-25-M2-M3-cpu` |
| Tiled against in-memory, case M | cloud VM | in-memory solve | bit-identical parameters and statuses on 1 500 random points; bytes read equal the brick boxes (5.04 GB) | {doc}`2026-09-25-M2-M3-cpu` |
| Case A twin | cloud VM | ground truth; CCPi 22.0.0 | vs truth median 0.0072 (CCPi 0.0072); vs CCPi median 0.0097, p95 0.0175, status agreement 97.8 % (all disagreements at the grid's edge) | {doc}`2026-09-25-M4-case-A-twin` |
| **Case A, real data**, interior points | `msm12` | CCPi 22.0.0, iDVC mode | median \|Δu\| 0.0512, p95 0.152, status agreement 100 %, mean Δu ≤ 0.003 per axis. Original Q1: marginal fail on the median (by 0.0012). Revised Q1: pass | {doc}`2026-09-26-case-A-real` |
| Error floor on the case A scan (sphere 80, 8 000 samples) | `msm12` | known shifts of the real image; repeat solves | interpolation bias up to ±0.03 voxel; noise 0.005; choice of sample points 0.032 (falls as 1/√n); prefilter σ = 1 cuts the bias to 0.0005 | {doc}`2026-09-26-error-floor-case-A` |
| CCPi `ccpi-dvc` 25.0.0 | cloud VM | ground truth | tricubic path broken: ~3 voxel errors on case S, 3.1 voxels on the case A twin, every point reported GOOD | {doc}`2026-09-25-M0-M1-case-S`, {doc}`2026-09-25-M4-case-A-twin` |

### Check-suite baseline (`msm12`)

The local check suite (`zvdvc check full`) records a per-machine baseline,
[`baselines/msm12.json`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/docs/benchmarks/baselines/msm12.json),
recorded 2026-09-27 at commit `9bb8039` (Python 3.11, numpy 2.4.6, zarr
3.1.6, cupy 14.2.0, numba 0.67.0, CUDA runtime 12.9):

| metric | value |
|---|---|
| selftest RMSE against truth, `fused` / `cupy` / `cpu` | 0.00142 voxel each |
| kernel `gn_sums`, case A data, `fused` / `cpu` | 3.66 / 145 µs per point-iteration |
| kernel `sample_values`, case A data, `fused` / `cpu` | 7.21 / 119 µs per point |
| case A central grid, in memory, `fused` / `cpu` | 1.36 s / 8.14 s |
| the same vs CCPi (all 4 680 points, edge points included) | median 0.0512, p95 0.1517, status agreement 98.9 % |
| the same with `prefilter_sigma` 1, `fused` (filtering included) | 11.06 s; vs CCPi median 0.0485, p95 0.146 |
| case A 3D grid (86 100 points, spacing 24), `fused`, in memory | 6.17 s |

These are the reference values later runs on this machine are compared
against, not a separate benchmark.

## Not yet measured

| what | status |
|---|---|
| **8 × H100 multi-GPU scaling** (Q2 kernel throughput on H100, Q3 1→8 GPU efficiency) | The launcher (one process per GPU, shared tile queue) and the scaling campaign (`zvdvc.bench.campaign`, `sge/benchmark.qsub`) are implemented; not yet run. See {doc}`/CLUSTER`. |
| **GPUDirect Storage** (roadmap M6) | Not wired in. The planned benchmark compares host reads with GDS on node-local NVMe on an 8 × H100 node, per stage and end to end, for both the image bricks and the zarr-vectors point and result reads. See {doc}`/how_to/gpudirect_storage`. |
| **Scenario B** (4096³ u16 pair, ~275 GB, ~8.4 M points) | Configuration ([`configs/large_8xh100.yaml`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/configs/large_8xh100.yaml)) and case L generator exist; no run yet. The 1–3 min estimate in {doc}`/PERFORMANCE` is modelled. |
| Tiled GPU run with I/O wait (Q4) | Case M's I/O wait was measured only on the CPU engine, where compute is slow enough to hide I/O. |
| Nsight Compute counters for the fused kernel | Need profiling permissions not available on the workstation. |
| Multi-node runs | Implemented (nodes split tiles by rank), untested. |
| Other datasets | Only the case A scan has been used for real-data accuracy. |

## Methodology

**Three backends, so gains are attributed correctly.** zvDVC's speed-up over
CCPi has two sources. *Restructuring* (one brick read per tile instead of
per-point box reads, an analytic Jacobian, batched points) helps a CPU as much
as a GPU. The *GPU* adds its own factor on top. The benchmarks therefore time
three things on the same data and settings: CCPi as shipped, zvDVC's
restructured CPU engine (`cpu`, the fused step in numba) and zvDVC on the GPU
(`fused`). On case A, the CPU engine alone is 293× faster than CCPi as iDVC
runs it; the GPU is about 5× faster again on that case, and about 16× on the
285 480-point grid. {doc}`/PERFORMANCE` section 5 breaks the expected gain
into factors; those factors are modelled.

**What a time includes.**

* *CCPi, iDVC mode*: one `dvc` process with OpenMP on every core, as iDVC
  launches it. *CCPi, N processes*: N single-thread processes on disjoint
  subsets of the points, the best a node gets from CCPi without code changes
  (but not a valid reference, since splitting breaks its seeding).
* *zvDVC parity*: both volumes in memory, CCPi's point order, from reading
  the volumes to the last point, with kernel compilation done beforehand.
* *zvDVC CLI*: `plan`, `seed`, `run` and `finalize` as separate processes,
  interpreter start-up, imports and writing the results store included.

**Accuracy is judged where both codes are valid.** Points whose subvolume
leaves the image are reported separately: zvDVC marks them `RANGE_FAIL`, CCPi
correlates wrapped or undefined data there. The revised Q1 criterion (mean
difference ≤ 0.01 voxel per axis, spread ≤ 1.25 × zvDVC's own seed-to-seed
spread) replaced a median threshold that two correct solvers sampling
different points cannot meet ({doc}`/MVP_PLAN`).

**Timing drift.** Back-to-back runs on the RTX A2000 workstation differ by up
to ~18 % (clocks, temperature). Treat differences smaller than that on this
card as noise.

**Check-suite limits** (`zvdvc check`, {doc}`/TESTING`):

* Accuracy fails the check: tests and selftest must pass; kernel sums within
  1e-4 of float64; case A results against the baseline arrays (status on
  ≥ 99.9 % of points, |du| ≤ 1e-3 voxel) and against CCPi no worse than the
  baseline.
* Speed is compared only on the same hardware and only with the GPU idle.
  Kernel timings warn at 20 % slower and fail at 40 %; end-to-end timings at
  25 % and 50 %; each limit is widened to 3× the timing's coefficient of
  variation at baseline. CPU timings are not compared when the load average
  exceeds half the cores.
* The limits catch the regressions the suite exists for (2–20×), not drift.

(ccpi-version)=
**CCPi version.** Baselines use CCPi `dvc` 22.0.0. The 25.0.0 conda build,
which a fresh iDVC install pulls, has a broken tricubic path
({doc}`2026-09-25-M0-M1-case-S`).

## Reproduce

Local checks, with a PASS / WARN / FAIL verdict against this machine's
baseline:

```bash
zvdvc check quick                             # tests without GPU or slow ones
zvdvc check gpu                               # + GPU tests and zvdvc selftest
zvdvc check full --case-a runs/case_A_data    # + slow tests, kernel benchmark, case A accuracy and speed
zvdvc check full --case-a runs/case_A_data --update-baseline --reason "why the numbers moved"
```

The individual benchmarks:

| command | measures | report |
|---|---|---|
| `python -m zvdvc.bench.ccpi_baseline CONFIG --workdir DIR --threads 1 2 4` | CCPi pt/s with a thread and process sweep | case S |
| `python -m zvdvc.bench.throughput --backend fused cpu --samples 2000 4096 --dof 6 12` | fused-step pt/s on an in-cache synthetic volume (overstates GPU speed on real data) | M2/M3 |
| `python -m zvdvc.bench.case_a run --data DIR --out runs/case_A --backends fused cpu` | CCPi vs zvDVC on case A: speed and agreement ({doc}`/tutorials/idvc_example`) | case A |
| `python -m zvdvc.bench.compare_ccpi CONFIG --out DIR [--truth truth.npz]` | the same comparison for any configuration | twin |
| `python -m zvdvc.bench.kernel --backends fused cpu --json runs/kernel.json` | the two solve kernels in µs, on case A data if `ZVDVC_CASE_A` is set | case A |
| `python -m zvdvc.bench.density --backends fused --json runs/density.json` | wall time against point count on case A | {doc}`/PERFORMANCE` §4.4 |
| `python -m zvdvc.bench.error_floor --case-a DIR --out runs/error_floor` | the error-floor study (~35 min on the A2000) | error floor |
| `python -m zvdvc.bench.scaling CONFIG --devices 1 2 4 8` | pt/s and efficiency from 1 to 8 GPUs | not yet run |
| `python -m zvdvc.bench.campaign --out DIR --data-dir DIR` | the whole 8 × H100 campaign ({doc}`/CLUSTER`) | not yet run |

`ZVDVC_CCPI_DVC` points the CCPi tools at a `dvc` 22.0.0 executable;
`ZVDVC_CASE_A` at the case A data. {doc}`/TESTING` walks through all of this
on a fresh machine.

## Reports

```{toctree}
:maxdepth: 1

2026-09-26-case-A-real
2026-09-26-error-floor-case-A
2026-09-25-M4-case-A-twin
2026-09-25-M2-M3-cpu
2026-09-25-M0-M1-case-S
```
