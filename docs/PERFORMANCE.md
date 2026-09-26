# Performance model and expected improvement

> [!IMPORTANT]
> Every number here is **modelled, not measured**. The model is built from
> reading the CCPi DVC source and from published hardware specifications. The
> milestones that replace each assumption with a measurement are listed in
> [§7](#7-how-the-model-gets-replaced-by-measurements). Treat the ranges as
> order-of-magnitude guidance for a go/no-go decision, not as a benchmark.

## 1. Summary

| Scenario | CCPi DVC as shipped | pyDVC (GPU) | Speed-up |
|---|---|---|---|
| **A**: iDVC example / CCPi test case, 4 680 pts | 4–11 min, 1 workstation | 2–6 s, **1× H100**, bound by I/O | **~40–300×** |
| **B**: 4096³ u16 pair, ~8.4 M pts | 36–117 h per process (and the cloud must be split into ~10³ runs) | 1–3 min, **8× H100**, bound by I/O | **~700–7 000×** vs 1 process;<br>~250–3 500× vs a CPU node running 4 processes |

**Measured so far (CPU only, [case A twin](benchmarks/2026-09-25-M4-case-A-twin.md)).**
On 4 vCPU, with the iDVC example's geometry, points and settings, CCPi as iDVC
runs it takes 17.8 min. pyDVC's restructured CPU engine takes 29 s in parity
mode and 48 s through the CLI, **22–37× faster before any GPU**. That is
factors 1–4 of §5 alone. The GPU rows above are still modelled.

**Where the gain comes from.** Most of it is *restructuring*, not the GPU.
Compare against a CPU implementation restructured the same way (bricks,
analytic Jacobian, batched points, all cores busy). The 8×H100 node's
advantage is then:

* **~1.5–12×** when the run is I/O-bound. This is typical for scenario B,
  where the GPUs finish compute in seconds and wait on storage.
* **~30–300×** when it is compute-bound. That covers 12-DOF, full-voxel
  subvolumes, coarse grid searches, iDVC-style bulk parameter sweeps, time
  series that reuse the reference brick, and data already in HBM.

The MVP therefore benchmarks three backends: CCPi, a restructured CPU path,
and the GPU path. Section 5 breaks the gain down factor by factor.

## 2. Assumptions

| | Value used | Notes |
|---|---|---|
| CPU node | 16–32 cores, ~200 GB/s DRAM, AVX2 | Workstation for A, HPC node for B |
| H100 SXM5 | 80 GB HBM3 at 3.35 TB/s; 67 TFLOP/s FP32 (non-tensor); 50 MB L2 | 8 per node, no NVLink traffic needed |
| Achieved GPU FP32 for the fused kernel | **10–20 TFLOP/s** (15–30 % of peak) | Gather-heavy 64-tap stencil; checked against an LSU/cache-traffic estimate (§4.2) |
| Achieved CPU FP64 inside CCPi | 20–50 GFLOP/s | Scalar-ish loops, OpenMP over ~8 000-sample loops |
| Storage ingest per node | **10–25 GB/s** | Lustre/GPFS client or NVMe RAID; GDS optional |
| Gauss–Newton iterations | ~6 (A), ~5 (B) | CCPi caps at 20; typical with good seeds |

## 3. Where CCPi DVC spends its time, per point

From `Search::process_point`, `Interpolate::kernels`, `kernels_derivs` and
`tri_cub_Lek`: for every point, CCPi

1. reads a box of edge `L = S + 2·disp_max + 4` from **both** volumes with
   `2·L²` `seekg`/`read` calls. This is single-threaded and happens before
   the OpenMP region;
2. computes 8 float64 derivative terms (f, fx, fy, fz, fxy, fxz, fyz, fxyz)
   for **every voxel** of both boxes, about 100 B of memory traffic per voxel;
3. solves the 64-coefficient Lekien–Marsden system (a 64×64 matrix–vector
   product, 8 192 flop) for each cell the samples touch, about 2.5·M cells;
4. evaluates the objective `E = 1 + iters·(ndof + 1)` times, because the
   Jacobian comes from forward differences, at about 200 flop per sample each.

| Cost term | Formula | **A**: S=80, d=38, M=8000, 6-DOF | **B**: S=48, d=10, M=4096, 6-DOF |
|---|---|---|---|
| box edge `L`, voxels `L³` | | 160, 4.10 M | 72, 0.37 M |
| (1) box reads | `2L² × ~1 µs` | **~50 ms** (20–100) | ~10 ms from page cache; **10–40 ms** when the 275 GB input exceeds it |
| (2) derivative kernels | `2L³ × 100 B / 20–40 GB/s` | **20–40 ms** | 2–4 ms |
| (3) Lekien coefficients | `2.5M × 8192 / 20–50 GF` | 3–8 ms | 2–4 ms |
| (4) objective evaluations | `E × M × 200 / 20–50 GF` | 1.4–3.5 ms (E≈43) | 0.6–1.5 ms (E≈36) |
| **per point** | | **45–150 ms → 7–20 pt/s** | **15–50 ms → 20–65 pt/s** |
| **whole run** | | 4 680 pts: **4–11 min** | 8.4 M pts: **36–117 h** per process |

About 90 % of CCPi's per-point time in case A is fixed overhead, terms (1)
and (2), which are proportional to the box volume, not to the correlation
work. The reference volume is re-read and re-differentiated for every point,
even though neighbouring points share most of their box.

**Measured (M0, 2026-09-25, [benchmarks](benchmarks/2026-09-25-M0-M1-case-S.md)).**
On synthetic case S (256³ u16, 2 197 points, sphere 32 / 2 000 samples,
`disp_max` 8, so `L = 52`; ccpi-dvc 22.0.0 on a 4-vCPU 2.1 GHz Xeon VM):

| | model (terms 1–4) | measured |
|---|---|---|
| per point, 12-DOF, 1 process | 4–15 ms (67–250 pt/s) | **26.7 ms (37.5 pt/s)** |
| per point, 6-DOF, 1 process | slightly less | 22.9 ms (43.7 pt/s) |
| OMP threads 1 → 4 | "terms 2–4 only" | **no gain** (37.5 → 36.3 pt/s) |
| 4 processes on 4 cores | "2–3×" | 3.3× (123 pt/s, 12-DOF; 155 pt/s, 6-DOF) |
| case A twin (sphere 80, 8 000 samples, L = 160), 6-DOF, iDVC mode (1 process, 4 threads) | 45–150 ms (7–20 pt/s) | **228 ms (4.4 pt/s)**; 4 680 points in 17.8 min |
| case A twin, 4 processes × 1 thread | — | 807 s (5.8 pt/s); one thread alone takes 680 ms per point |

The case A rows come from a synthetic twin with the example's geometry, points
and settings ([benchmarks](benchmarks/2026-09-25-M4-case-A-twin.md)), because
the Zenodo data could not be downloaded in the development environment.
CCPi is 2–7× slower per point than modelled for case S, and 1.5–5× for case A.
For case S, threads do not help and 12-DOF costs only 14 % more than 6-DOF:
fixed per-point overhead dominates. For case A's larger boxes, 4 threads give
3× over one.

At scenario B's scale, two further limits apply:

* `DataCloud::sort_order_neighbors` sorts the whole cloud for every point:
  O(N² log N), about 10¹⁵ operations for 8.4 M points. That is infeasible, so
  the cloud has to be split into roughly 1 000 separate runs.
* Adding threads only speeds up terms (2) to (4). Term (1) and the per-point
  serial sections cap one process at a few cores' worth of useful work.
  Running about 4 processes per node on disjoint subsets gains perhaps 2–3×,
  if the file system keeps up with random 144-byte row reads.

## 4. pyDVC on H100

### 4.1 I/O: amortised by bricks

Each tile of edge `T` is read once per volume with a halo `h = S/2 + disp_max + 2`,
so the read amplification is `α = ((T + 2h)/T)³` (table in
[`pipeline/tiling.py`](../src/pydvc/pipeline/tiling.py)):

| | Bytes read | At 10–25 GB/s |
|---|---|---|
| A: whole volumes, once | 2 × 2.4 GB = 4.8 GB | 0.2–0.5 s (1–2 s from a single NVMe) |
| B: coarse pass on pyramid level 1 | 2 × 17 GB = 34 GB | 1.4–3.4 s |
| B: main pass, `T = 1024`, `h = 36`, `α = 1.23` | 2 × 137 GB × 1.23 = 337 GB | **13–34 s** |

No per-voxel preprocessing happens at all. Tricubic is evaluated as separable
Catmull-Rom directly from the native-dtype brick
([`kernels/interpolate.py`](../src/pydvc/kernels/interpolate.py) explains why
this matches CCPi's Lekien–Marsden interpolant).

### 4.2 Compute: batched Gauss–Newton with an analytic Jacobian

Cost per sample per iteration, in the fused kernel:

* Catmull-Rom value and gradient: 64 taps, weights and derivative weights,
  about 570 flop;
* Jacobian row and upper-triangle normal-equation accumulation (6-DOF): about 90 flop;
* ZNSSD bookkeeping and margin: about 140 flop.

That is **~800 flop per sample per iteration**, plus ~250 per sample once for the reference.

| | A | B |
|---|---|---|
| flop per point | 8000 × (6 × 800 + 250) ≈ **40 MFLOP** | 4096 × (5 × 800 + 250) ≈ **17 MFLOP** |
| ideal, 10–20 TFLOP/s | 2–4 µs → 250–500 k pt/s | 0.9–1.7 µs → 0.6–1.2 M pt/s |
| cross-check: cache traffic (64 × 2 B per sample-eval, ≥10 TB/s L1/L2) | ~0.6 µs | ~0.3 µs, so compute-bound, not cache-bound |
| cross-check: HBM (each point's footprint once per iteration, worst case) | ~1.1 µs | ~0.45 µs |
| **practical per GPU** (×0.25–0.5 for convergence divergence, batch tails, the solve kernel, orchestration) | **60–250 k pt/s** | **150–500 k pt/s** |
| run compute | 4 680 pts: < 0.1 s (plus ~100 wavefront shells × ~1 ms) | 8.4 M pts on 8 GPUs: **2–7 s** |

Compare CCPi's per-point cost of 45–150 ms. **Per point, the GPU path is
about 10⁴ times cheaper.** End-to-end, I/O and fixed costs dominate instead.

### 4.3 End-to-end

| Stage | A (1× H100) | B (8× H100) |
|---|---|---|
| start-up (CUDA context, kernel JIT, cached after the first run) | 1–3 s | 1–3 s |
| plan (metadata, kNN) | < 1 s | 5–20 s |
| seed | < 0.1 s (wavefront) | 2–5 s (coarse pass, pyramid level 1) |
| read + solve (solve overlaps the reads) | 1–2 s | 15–40 s |
| write results + finalize | < 1 s | 5–20 s (~0.5 GB of attributes) |
| **total** | **~2–6 s** | **~0.5–1.5 min**; budget 1–3 min for file-system contention |
| vs CCPi | 4–11 min → **~40–300×** | 36–117 h vs 1–3 min → **~700–7 000×** |

## 5. Attribution: which change buys what

Multiplicative factors relative to CCPi as shipped, for scenario B:

| # | Change | Applies to CPU too? | Modelled factor |
|---|---|---|---|
| 1 | Bricks: one read and no per-voxel derivative kernels per tile (removes terms 1–2 and the Lekien solve) | yes | **5–20×** |
| 2 | Analytic Jacobian (1 value+gradient pass instead of `ndof+1` evaluations) | yes | 2.5–4× on the solver |
| 3 | Batching points, so every core or SM is busy (CCPi parallelises inside one point only) | yes | 2–4× on a CPU node |
| 4 | O(N log N) neighbours; one run instead of ~10³ | yes | makes B possible at all |
| 5 | H100 vs a 32-core CPU node for this kernel (practical 150–500 k pt/s per GPU vs 12–35 k pt/s per node; the CPU reaches ~0.2–0.6 TFLOP/s on the gathers) | **GPU only** | **~4–40× per GPU** |
| 6 | 8 GPUs, independent tiles | **GPU only** | ~7–8× on compute; 1× on shared I/O |

**Measured (M2, [benchmarks](benchmarks/2026-09-25-M2-M3-cpu.md)).** The
restructured CPU backend as built, the fused step in numba on 4 vCPU,
reaches 443 pt/s at scenario B's settings (M = 4 096, 6-DOF, sphere 48), about
110 pt/s per core, and 6.8–7.4× CCPi's best on the same cores for case S.
Scaled linearly to 32 cores that is about 3.5 k pt/s, **3–10× below the
12–35 k assumed below**. Until a vectorised CPU kernel shows otherwise, the
GPU-vs-restructured-CPU ratios in §1 are conservative.

Factors 1–4 give the restructured CPU backend: about 12–35 k pt/s per node,
so scenario B takes 4–12 min per node, still limited mainly by compute. The
8×H100 node takes 1–3 min, limited by I/O. That ratio is the ~1.5–12× quoted
above. As work per byte rises, the ratio moves toward factors 5×6 (~30–300×):

| Workload shift | Effect on flop per point | GPU node vs restructured CPU node |
|---|---|---|
| 12-DOF instead of 6-DOF | ~1.6× (78 + 12 accumulators) | toward compute-bound |
| full-voxel 41³ subvolume (69 k samples) instead of 4 096 | ~17× | compute-bound: ~30–300× |
| `basin_radius` grid search (e.g. 7³–19³ evaluations per point) | 10–100× | compute-bound |
| iDVC bulk run (e.g. 30 subvolume/sample combinations, same data) | 30×, reading the bricks once | compute-bound |
| time series of `F` frames against one reference | reference brick reused; I/O per frame ~halved | toward compute-bound |

## 6. Sensitivities and what could make this wrong

| Assumption | If wrong | Mitigation |
|---|---|---|
| Achieved 10–20 TFLOP/s in the fused kernel | Gather patterns or register pressure (12-DOF: 78 accumulators) could halve it | Nsight Compute in M2; spread accumulators across a warp for 12-DOF |
| Storage delivers 10–25 GB/s per node | Shared Lustre often gives less; B becomes 2–5× slower end-to-end, but still ≫ CCPi | Stage to node-local NVMe; zstd at ~2–3:1 on speckle data (GPU decompression via nvCOMP); GDS |
| ~5–6 GN iterations | Poor seeds mean more iterations and failures | Coarse/FFT seeding, repair pass; report the iteration histogram |
| CCPi model: 1 µs per row read, 20–40 GB/s kernel build | CCPi could be faster than modelled, shrinking the headline | **M0 measures it.** The attribution in §5 does not depend on it |
| Catmull-Rom ≡ CCPi tricubic | If not, parity needs the 64-weight Lekien stencil | Same memory traffic; ~1.5–2× more flop; M1 decides |
| Small wavefront shells (parity mode) | GPU underused for the first shells | Parity mode is for validation; production uses coarse seeding |

## 7. How the model gets replaced by measurements

| Milestone | Measurement | Replaces |
|---|---|---|
| M0 | CCPi `dvc` pt/s on cases A and S (synthetic), with a thread sweep and N concurrent processes | §3 table. **Done for S** (§3, measured); A pending the data |
| M2 | Fused-kernel pt/s and achieved TFLOP/s on the dev GPU; restructured-CPU pt/s (numba port of the same step) | §4.2 and factor 5. **Restructured-CPU pt/s done** (§5); GPU pending |
| M3 | Single-GPU end-to-end, I/O wait fraction, bytes read vs `α` | §4.1, §4.3 A. **Bytes read done** (case M: exactly the planned bricks); GPU end-to-end and I/O wait pending |
| M4 | 1→8 GPU scaling on 2048³–4096³ synthetic cases on H100 | §4.3 B, factor 6 |

Dev-GPU numbers (for example from an RTX A2000 12 GB: 288 GB/s, ~8 TFLOP/s
FP32) are scaled to H100 using measured bandwidth- or compute-boundedness from
Nsight, not by a single ratio. Results are recorded in `docs/benchmarks/`.
