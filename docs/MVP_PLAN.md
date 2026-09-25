# MVP plan

## 1. What the MVP must answer

The MVP is a **feasibility test**. It is done when these five questions have
measured answers:

| # | Question | Pass criterion |
|---|---|---|
| Q1 | **Accuracy.** Does batched GPU correlation match ground truth, and match CCPi? | Synthetic: displacement RMSE ≤ 0.02 voxel noise-free and ≤ 0.05 with 2 % noise, ≥ 99 % GOOD. Case A vs CCPi `.disp`: median \|Δu\| ≤ 0.05 voxel, 95th percentile ≤ 0.2, status agreement ≥ 98 %. |
| Q2 | **Kernel throughput.** Is the fused step as fast as modelled? | On H100 with scenario-B settings: **≥ 100 k pt/s per GPU** (model: 150–500 k), or ≥ 10 % of FP32 peak achieved. |
| Q3 | **Scaling.** Do 8 GPUs deliver? | 1→8 GPU efficiency ≥ 85 % on a compute-bound configuration; an I/O-bound configuration sustains ≥ 70 % of the node's measured storage bandwidth. |
| Q4 | **Data path.** Do OME-Zarr bricks and zarr-vectors stores keep up at 10⁷ points? | Compute stream waits on I/O ≤ 20 % of the time; allocate + parallel write + finalize of a 10⁷-point results store takes ≤ 5 % of run time; the store passes zarr-vectors validation. |
| Q5 | **Attribution.** How much is GPU and how much is restructuring? | CCPi, restructured-CPU and GPU pt/s measured on the same cases, and [PERFORMANCE.md](PERFORMANCE.md) updated from them. |

**Go** if Q1, Q2 and Q3 pass. If Q2 or Q3 fails, the Nsight profiles and the
model in PERFORMANCE.md §6 show whether the cause is fixable (register
pressure, I/O path) or fundamental.

## 2. Scope

**In the MVP**

* FA-GN solver, 3/6/12-DOF; SSD, ZSSD, NSSD and ZNSSD objectives (SAD
  follows because it shares SSD's residuals); trilinear and tricubic;
  cube and sphere templates.
* Seeding: `rigid` and `wavefront`. Coarse-field seeding with a simple
  sub-grid, so multi-GPU tiles are independent.
* Optional translation grid search (`basin_radius`).
* OME-Zarr and raw/npy volumes; zarr-vectors point-cloud and results stores; `.roi` import; `.disp`/`.stat` export.
* One node, 1–8 GPUs, dynamic tile queue, resume.
* Synthetic phantoms, the CCPi baseline runner, accuracy and throughput tooling.

**After the MVP (M5)**

IC-GN, FFT seeding, the repair pass, strain, the `pydvc ccpi` drop-in for
iDVC, multi-node runs, GPU kNN, the time-series mode, subvolume
thresholding and Neuroglancer pyramids.

## 3. Test cases

| ID | Data | Points | Settings | Used for |
|---|---|---|---|---|
| **S** | synthetic 256³ u16, `affine` and `rigid` fields | ~3 k (spacing 16) | sphere 32 / 2 000 samples, 6- and 12-DOF | M1–M3 development; runs on the dev GPU |
| **A** | iDVC example: Zenodo 10.5281/zenodo.7363345, 2 × 1520×1257×1260 u8 | 4 680 (CCPi `central_grid.roi`) | CCPi test settings ([configs/ccpi_central_grid.yaml](../configs/ccpi_central_grid.yaml)) | parity against CCPi's reference `.disp`; scenario A |
| **M** | synthetic 1024³ u16, `sinusoid` field | ~262 k | sphere 48 / 4 096, 6-DOF | single-GPU end-to-end; I/O overlap |
| **L** | synthetic 2048³ and 4096³ u16, `inclusion` field | 2.1 M / ~8.4 M (masked) | scenario B ([configs/large_8xh100.yaml](../configs/large_8xh100.yaml)) | 8×H100 scaling and end-to-end time |

Ground truth for S, M and L comes from
[`synth/phantoms.py`](../src/pydvc/synth/phantoms.py). Case A's truth is
CCPi's own result, so its comparison is *agreement*, not accuracy.

## 4. Milestones

Effort assumes one developer who knows CUDA and Python. **MVP (M0–M4):
about 8–11 weeks.**

### M0: Baselines and fixtures (1 week)

| Task | Files |
|---|---|
| Speckle phantoms and analytic fields (rigid, affine, sinusoid, inclusion), chunked straight to OME-Zarr | `synth/phantoms.py`, `pydvc synth` |
| CCPi input writer, `.roi` writer, `.disp`/`.stat` readers | `io/ccpi.py` |
| Run CCPi `dvc` (conda `ccpi-dvc`) on S and A: OMP thread sweep, P concurrent processes | `bench/ccpi_baseline.py` |
| Confirm the Zenodo pair reproduces CCPi's `completed_central_grid.disp` | `docs/benchmarks/` |

**Done when:** CCPi pt/s is measured for S and A, the thread-scaling curve is
recorded, and PERFORMANCE.md §3 is updated with measured values.

### M1: numpy reference (2 weeks)

| Task | Files |
|---|---|
| `RunConfig` YAML round trip | `config.py` |
| Templates (cube lattice, sphere) | `geometry/templates.py` |
| Warp, Jacobians (CCPi roll–pitch–yaw, `(I+E)·R`) | `geometry/warp.py` |
| Catmull-Rom / trilinear / nearest with gradients; **Lekien equivalence test** | `kernels/interpolate.py` |
| All five objectives and residuals | `kernels/objective.py` |
| Batched FA-GN on numpy; status logic | `solver/gauss_newton.py` |
| kNN, CCPi order, wavefront shells, neighbour seeding | `solver/seeding.py` |
| RawVolume (memmap), `.roi` import, lattice generation | `io/volume.py`, `io/pointcloud.py`, `geometry/pointgrid.py` |
| Accuracy metrics | `bench/metrics.py` |

**Done when:**

* Catmull-Rom matches a direct Lekien–Marsden port within a relative 1e-5,
  or the fallback is chosen and documented.
* The analytic Jacobian matches finite differences within 1e-4 relative.
* Q1 synthetic accuracy passes on S.
* Q1 CCPi agreement passes on A. The numpy path runs A in minutes.

### M2: single-GPU kernels (2–3 weeks)

| Task | Files |
|---|---|
| CuPy path through the `xp` namespace (unfused), checked against M1 | `kernels/*`, `solver/gauss_newton.py` |
| Fused GN step (`cupy.RawModule`, templated on DOF, objective, interpolation, dtype) + batched Cholesky + active set | `kernels/fused.py`, `kernels/cuda/fused_gn.cu` |
| Batched translation grid search | `solver/coarse.py` |
| Batch sizing, Morton ordering | `pipeline/batching.py` |
| Kernel micro-benchmark with Nsight Compute metrics | `bench/throughput.py` |
| **Restructured-CPU baseline**: numba-parallel port of the same fused step (for Q5 attribution) | `kernels/cpu_fused.py` |

**Done when:**

* GPU results match numpy within 1e-3 voxel on GOOD points, with ≥ 99.9 % status agreement.
* Fused-kernel pt/s and achieved TFLOP/s are recorded on the dev GPU for S-
  and B-like settings, with a bottleneck analysis (compute, L2 or LSU)
  for the H100 extrapolation.
* The restructured-CPU pt/s is recorded.
* A single H100 measurement for Q2, if access allows. Otherwise it moves to M4.

### M3: data path and single-GPU end-to-end (2 weeks)

| Task | Files |
|---|---|
| `Box`, halo, OME-Zarr brick reads to device (host-decode path first, then GPU buffers or kvikio) | `geometry/box.py`, `io/volume.py` |
| `convert_to_ome_zarr` | `io/volume.py`, `pydvc convert` |
| zarr-vectors point-cloud store: write (three-phase), `read_tile` via `read_cells(device="cuda")` | `io/pointcloud.py` |
| Results store: allocate, `write_tile`, finalize | `io/results.py` |
| Tile planning, brick boxes, memory check, `plan.json` | `pipeline/tiling.py` |
| TileWorker with prefetch double-buffering and a writer thread | `pipeline/worker.py` |
| Coordinator: prepare / seed (rigid, wavefront, simple coarse) / finalize | `pipeline/coordinator.py` |

**Done when:**

* Case M end-to-end on one GPU reproduces the M2 in-memory results bit for bit.
* Bytes read are within 5 % of the `α` model.
* Compute-stream I/O wait is ≤ 20 % (Q4).
* The results store passes zarr-vectors validation and reads back through `zv.open(...).select(bbox=...)`.

### M4: 8×H100 (1–2 weeks). **MVP complete.**

| Task | Files |
|---|---|
| Node/rank discovery; process-per-GPU spawn; dynamic queue; LPT shares | `pipeline/launch.py`, `pipeline/tiling.py` |
| Resume (skip written cells), failed-tile requeue, `failed_tiles.json` | `pipeline/worker.py`, `io/results.py` |
| `.disp`/`.stat` export, run summary | `io/ccpi.py`, `pipeline/coordinator.py` |
| SLURM script, storage bandwidth baseline (`fio`/`ior`) | `scripts/slurm/` |

**Done when:**

* Q2 and Q3 are measured on case L (2048³ and 4096³).
* Killing one worker (`kill -9`) mid-run and resubmitting completes only
  the missing tiles and gives an identical final store.
* Case L at 4096³ finishes end-to-end in under 10 minutes. The model says
  1–3; 10 is the safe pass line.
* PERFORMANCE.md is rewritten from measurements (Q5).

### M5: after the MVP (6–8 weeks, prioritised by what M4 shows)

IC-GN · FFT-CC seeding · repair pass · strain · coarse pass on pyramid level
1 · `pydvc ccpi` drop-in with iDVC progress lines · GPU kNN · multi-node
(static shares, then a shared queue) · time-series mode · subvolume
thresholding · Neuroglancer pyramids.

## 5. Tests

Tests encode the acceptance criteria above. Until a milestone lands, its tests
hit `NotImplementedError` and are reported as **skipped**
([`tests/conftest.py`](../tests/conftest.py)), so the suite shows progress
rather than failures.

| File | Milestone | Checks |
|---|---|---|
| `tests/test_ccpi_io.py` | M0 | `.disp` round trip; CCPi status codes |
| `tests/test_warp.py` | M1 | identity, rotation orthonormality, `(I+E)·R` order, analytic vs finite-difference Jacobian |
| `tests/test_interpolate.py` | M1 | trilinear exact on linear fields, Catmull-Rom exact on quadratics, gradients, `inside` mask |
| `tests/test_objective.py` | M1 | known values, ranges, ZNSSD invariance to `a·g + b` |
| `tests/test_solver_synthetic.py` | M1 | recovers translation and affine fields; `RANGE_FAIL` beyond `disp_max` |
| `tests/gpu/test_gpu_parity.py` | M2 | numpy vs cupy vs fused agreement |
| `tests/test_tiling.py` | M3 | tiles disjoint and covering; bricks contain every sample for any seed within range |
| `tests/test_end_to_end.py` | M3 | synthetic 128³: plan → run → finalize → accuracy (`-m slow`) |

## 6. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| zarr-vectors gpu-backend API changes | high | medium | Pin the commit. Use only `api` and `building`. All access goes through `pydvc.io`. Report gaps upstream (bin assignment is internal). |
| Catmull-Rom ≠ CCPi tricubic | low | medium | Tested in M1; fallback is the 64-weight Lekien stencil (same traffic, ~1.5–2× flop). |
| The ZNSSD Jacobian approximation (fixed target stats per iteration) slows convergence | medium | low | Compare iteration histograms with CCPi; use the exact normalisation derivative if needed. |
| 12-DOF register pressure (78 + 12 accumulators) | medium | medium | Warp-cooperative accumulation; separate specialisation. |
| Storage bandwidth below 10 GB/s | medium | medium (end-to-end only) | Stage to node NVMe, zstd with GPU decode, GDS. The kernel result (Q2) is unaffected. |
| GPU zstd decode of image chunks not available through zarr-python | medium | low | Host decode into pinned memory with threads; measure before optimising. |
| Coarse seeds fail near discontinuities (cracks, slip bands) | medium | medium | Repair pass (M5); per-tile wavefront fallback; FFT seeding. |
| float32 precision | low | medium | Relative coordinates, compensated sums, float64 reference tests. |
| Licensing (CCPi is GPL-3.0) | — | high if ignored | Clean-room implementation from the published method; decide the licence before M1. |

## 7. Decisions needed from the project owner

1. **Licence**: this decides whether CCPi code may be consulted line by line or only the method (README).
2. **Target cluster**: file system and per-node bandwidth, container runtime, CUDA version, and whether GPUDirect Storage is enabled. This sets the M4 I/O path.
3. **Real datasets** beyond the iDVC example, to run after M4. Ideally one large time series.
4. **iDVC integration**: whether the `pydvc ccpi` drop-in moves into the MVP. It adds about 1 week.
