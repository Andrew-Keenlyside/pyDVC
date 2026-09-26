# Architecture

## 1. Terms

| Term | Meaning |
|---|---|
| **point** | A search location (CCPi "search point"). Its result is a displacement plus the full parameter vector. |
| **template** | The sample offsets that define a subvolume (cube lattice or random sphere). pyDVC uses one per run. |
| **sample** | One template offset, warped and interpolated. M samples per point (`subvol_npts`). |
| **seed** | The starting displacement for a point. |
| **cell** | One zarr-vectors chunk of the point-cloud grid. The atomic unit of storage. |
| **tile** | A block of whole cells. The unit of work handed to a GPU. |
| **brick** | The dense image box a tile needs, one for the reference and one for the deformed volume. |
| **batch** | The points of one tile solved together in one set of kernel launches. |
| **shell** | A distance band from the start point. The batch unit in CCPi-parity (wavefront) mode. |

## 2. The reference algorithm (CCPi DVC, driven by iDVC)

iDVC writes a `dvc_config.txt` and a `.roi` point cloud, then launches the
`dvc` executable once per (subvolume size × sample count) combination. Inside
`dvc` (`Core/dvc.cpp`, `DataCloud.cpp`, `Search.cpp`, `Interpolate.cpp`):

1. `organize_cloud`: sort all points by distance from the start point, and
   store the 75 nearest neighbours of each by a full O(N²) sort.
2. For each point, in that order (`Search::process_point`):
   1. read a `(S + 2·disp_max + 4)³` box around the point from the reference
      raw file, row by row, and build tricubic derivative kernels over it;
      interpolate the reference samples;
   2. seed with the mean `(u, v, w)` of `point_good` neighbours solved so far,
      or `rigid_trans` if there are none;
   3. read and prepare the same-sized box from the deformed file, centred on
      the seeded position;
   4. optionally run a translation grid search (`basin_radius`);
   5. run up to 20 undamped Gauss–Newton ("LM") steps with a forward-difference
      Jacobian and a QR solve of `JᵀJ`;
   6. record status, objective and parameters; append to `.disp`.
3. Rewrite `.disp` in point-cloud order and write the `.stat` summary.

OpenMP parallelises only *inside* steps 2.1–2.5. Section 3 of
[PERFORMANCE.md](PERFORMANCE.md) shows where that time goes.

## 3. Stages and data flow

```
            ┌──────────── inputs ────────────┐
            │ ref.ome.zarr  def.ome.zarr      │  dense, sharded, native dtype
            │ points.zarrvectors (or .roi)    │  zarr-vectors point cloud
            └──────────────┬─────────────────┘
 pydvc plan      (1 proc)  │  metadata only → plan.json (tiles, brick boxes, costs, node shares)
                           │  ResultStore.allocate (create arrays, defer presence)
 pydvc seed      (1 GPU)   │  rigid: nothing │ coarse: sub-grid on pyramid level 1 → seeds.zarr
                           │  wavefront: whole solve, shell by shell (parity mode)
 pydvc run       (N GPUs)  │  per GPU: tiles from a queue → bricks → batches → write cells
 pydvc repair    (N GPUs)  │  failed points with good neighbours → re-seed → re-solve their tiles
 pydvc finalize  (1 proc)  │  rebuild presence, metadata, .stat, optional .disp / pyramid
                           ▼
            results.zarrvectors: same cells, same rows as points.zarrvectors
```

Every stage is idempotent. `run` skips tiles whose cells already exist in the
results store, so a job resubmitted after a time-out or a node failure resumes.

## 4. Work decomposition

```
node ── shares tiles by LPT from plan.json (no communication between nodes)
 └─ GPU process ── takes tiles from a node-local dynamic queue
     └─ tile  (T³ voxels, e.g. 1024³; whole zarr-vectors cells)
         ├─ bricks: ref = points box + halo; def = ref box shifted by the median seed, + seed spread
         └─ batch  (B points, Morton order for L2 locality; B sized from free memory)
             └─ CUDA block per point  (several points per block when M < 1024)
                 └─ threads stride over the M template samples
```

GPUs never exchange data. Once seeds exist, tiles are independent, so there is
no NCCL, no halo exchange and no global synchronisation until `finalize`.

**Halo.** `h = extent(template) + disp_max + 2`. The extent is the largest
template offset, because rotations can point any sample along any axis. The
`+2` covers the Catmull-Rom stencil. The deformed brick also grows by the
spread of seeds within the tile. Read amplification is `((T + 2h)/T)³`: 1.23
for `T = 1024, h = 36`.

## 5. Data formats

### Images: OME-Zarr v3

Dense volumes are stored as OME-Zarr v3: 128³ chunks inside shards aligned to
the tile size, zstd, and a multiscale pyramid, since the coarse seeding pass
reads level 1. Bricks stay in their native dtype on the device. The read path
is kvikio/GPUDirect Storage, then zarr-python's GPU buffers, then host decode
into pinned memory with one copy up
([`io/volume.py`](../src/pydvc/io/volume.py)). CCPi raw, npy and mhd inputs
are converted once with `pydvc convert`. As of M3 the host-decode path is the
one implemented; the other two are optimisations to measure against it.

### Points and results: zarr-vectors

zarr-vectors is a format for **vector geometry**: points, lines, meshes and
skeletons. It does not store the dense images. In pyDVC it does four jobs:

1. **The work partition.** A zarr-vectors store is cut into a spatial chunk
   grid. Tiles are unions of chunks, so a tile's input rows and output rows
   live in the same cells, and no two workers ever write the same cell.
2. **Pooled reads.** `zarr_vectors.building.read_cells` delivers a tile's
   points in one pooled read, as CSR over cells (`device="cuda"` decodes on
   the GPU). `read_neighbourhood(halo=1)` serves repair and strain across tile
   borders. The worker reads point positions to the host today (they are a
   few MB per tile and seeds are looked up there); the device read is a
   drop-in when that becomes worth it.
3. **Parallel writes without locks.** Workers use zarr-vectors' three-phase
   HPC pattern. A coordinator allocates the arrays and calls
   `defer_presence`. Workers call `write_chunk_*` with
   `record_presence=False`. A coordinator calls `rebuild_presence` at the end.
   Cells written on a deferred level are visible straight away, which is what
   resume and repair use.
4. **Downstream use.** Results carry `displacement`, `params`, `status`,
   `objmin`, `n_iter`, `seed` and, later, `strain` as vertex attributes. The
   store can build a multiscale pyramid and be viewed in Neuroglancer through
   zarr-vectors' tooling, and it exports to CCPi `.disp`/`.stat` for iDVC.

Constraints taken from upstream, which is under active development:

* The dependency is pinned to a gpu-backend commit.
* pyDVC imports only `zarr_vectors.api` and `zarr_vectors.building`.
  Bin assignment (`spatial.chunking.assign_bins`) is internal, so pyDVC
  assigns bins itself (two bins per axis, eight fragments per cell, numbered
  C-order over (x, y, z) as zarr-vectors' validator expects). A cell's rows
  are its fragments in order, so the results store reproduces the input's
  fragments from positions alone.
* pyDVC's run layout (DOF, grid, source points) sits in the results store's
  root attributes under `pydvc_results`.
* `decode="device"` for zstd is used only on stores pyDVC wrote itself.
  nvCOMP trusts its input, and zarr-vectors' guide documents crashes on
  corrupt frames.
* All zarr-vectors access goes through `pydvc.io`, so API changes are
  absorbed in one place.

## 6. Seeding

CCPi's neighbour-average seeding makes point *k* depend on points *0..k−1*.
pyDVC replaces the serial order
([`solver/seeding.py`](../src/pydvc/solver/seeding.py)):

| Strategy | Parallelism | Use |
|---|---|---|
| `wavefront` | one batch per distance shell (~1.7·k shells for a k³ lattice from a corner) | CCPi parity; one GPU, whole problem resident |
| `coarse` | coarse sub-grid (every `coarse_stride`-th lattice point) in wavefront order, interpolated (kNN inverse distance) to every point; then every tile independent | production, multi-GPU. The MVP solves the sub-grid at full resolution with whole volumes in memory; the pyramid-level pass that scales to 4096³ is M5 |
| `rigid` | fully parallel | small or smooth deformation, tests |
| `fft` (M5) | fully parallel, path-independent | large or discontinuous displacement |

A **repair** pass follows in every mode. It re-seeds failed points that have
enough GOOD neighbours from the neighbours' median, and solves them again.

## 7. Solver

* **FA-GN (default, parity).** CCPi's undamped Gauss–Newton with the same
  stopping rules (`|Δobj| < 1e-6` or `|Δu| < 0.01`, at most 20 iterations),
  but with an analytic Jacobian `∇g(x′)ᵀ ∂x′/∂p` and a batched Cholesky
  solve. `∂x′/∂p` follows CCPi's warp `x′ = c + t + (I + E)·R·d`
  ([`geometry/warp.py`](../src/pydvc/geometry/warp.py)).
* **Exact normalisation derivative.** For ZSSD, NSSD and ZNSSD the normal
  equations use the exact Jacobian of the normalised residual. Holding the
  target mean and norm fixed within an iteration, the common shortcut, shifts
  the fixed point by a term proportional to `(1 − ρ)·∂|g̃|/∂p`, which is not
  zero under noise. CCPi differentiates the full objective numerically, so the
  exact form is also the parity choice. It costs two more per-point sums
  (`Σj`, `Σĝj`, ndof each) in the same pass
  ([`kernels/objective.py`](../src/pydvc/kernels/objective.py)).
* **IC-GN (M5).** Inverse compositional. The Hessian comes from the
  reference once per point, and each iteration needs target values only.
* **Interpolation.** Separable Catmull-Rom, which equals CCPi's
  Lekien–Marsden tricubic with central-difference derivatives (M1 verifies
  this), plus trilinear and nearest.
* **Objectives.** CCPi's five, with identical scaling
  ([`kernels/objective.py`](../src/pydvc/kernels/objective.py)).
* **Precision.** float32 on device. Each point's brick-local centre is split
  into an integer voxel and a float32 fraction, and sample positions are
  formed from the fraction, so precision does not depend on the coordinate
  (tests run at coordinates above 1000). The normal-equation sums are formed
  on the target shifted by the reference mean, which keeps ZNSSD's
  `Σv² − n·v̄²` free of cancellation; per-thread partial sums followed by a
  tree reduction give pairwise-summation error. Every result is checked
  against the float64 numpy reference in tests.
* **Engines.** The Gauss–Newton loop is written once
  ([`solver/gauss_newton.py`](../src/pydvc/solver/gauss_newton.py)); an
  engine supplies sampling, the one-pass sums, and the update
  ([`solver/engines.py`](../src/pydvc/solver/engines.py)): `numpy` (float64
  reference), `cupy` (unfused GPU), `fused` (CUDA kernels), `cpu` (the same
  fused step in numba, the restructured-CPU baseline) and `emulated` (the
  CUDA source run on the host, for tests).

## 8. GPU execution

`pydvc run` starts one process per local GPU (spawn start method; each
process picks its device before touching CUDA), all pulling tile ids from one
`multiprocessing` manager queue: dynamic scheduling within the node. Nodes take
their LPT share from `plan.json` by rank (`SLURM_NODEID`, torchrun's
`GROUP_RANK`, or Open MPI's rank ÷ local size) and never talk to each other
([`pipeline/launch.py`](../src/pydvc/pipeline/launch.py)). The CPU engines use
the same launcher, with `--cpu-workers` processes splitting the cores.

Each worker process runs three streams:

| Stream | Work |
|---|---|
| copy | brick reads for the next `prefetch_depth − 1` tiles into pinned buffers, then host-to-device copy |
| compute | reference sampling, then fused GN steps and batched solves over the active set, batch after batch |
| host thread | encodes and writes the previous tile's result cells through zarr-vectors (device encode where available) |

The fused kernels ([`kernels/fused.py`](../src/pydvc/kernels/fused.py),
[`kernels/cuda/fused_gn.cu`](../src/pydvc/kernels/cuda/fused_gn.cu)) never
write per-sample intermediates to global memory. `gn_sums` (one block per
active point) reduces every sample to one row of normal-equation sums in a
**single pass**, ZNSSD included, since the target mean and norm follow from
the sums. `gn_solve` (one thread per point) solves and tests each point. The
active set shrinks as points converge, so late iterations cost only the
stragglers.

Without a GPU the same `.cu` file is tested two ways: NVRTC compiles every
specialisation (177, for `compute_90`), and a host emulator
([`kernels/cuda/emulate.py`](../src/pydvc/kernels/cuda/emulate.py)) runs it
with every CUDA thread as an OS thread (`__syncthreads` and warp shuffles
included) against the numpy reference.

As implemented (M3), the worker uses one host prefetch thread (brick reads,
pinned staging, one copy up), the compute loop, and one writer thread. The
separate CUDA copy stream is an M4 refinement if the I/O-wait measurement asks
for it.

`ptxas -v` for sm_90 (no spills anywhere): `gn_sums` uses 128 registers for
3/6-DOF and 250 registers plus 520 B of stack for 12-DOF, which limits
occupancy to 16 and 8 warps per SM respectively. Whether that matters is an
Nsight question for the first GPU run (docs/MVP_PLAN.md risk "12-DOF register
pressure").

### Memory budget: H100 80 GB, scenario B (`T = 1024`, u16, M = 4 096, 6-DOF)

| Item | Size |
|---|---|
| CUDA context, cupy memory pool slack | ~2 GB |
| bricks for the current tile (ref + def, 1096³ × 2 B each) | 5.3 GB |
| bricks for the prefetched tile | 5.3 GB |
| batch scratch at B = 32 k points (reference samples 16 KB per point; ×4 with IC-GN gradients) | 0.5 GB (2 GB IC-GN) |
| tile points, seeds, results (~262 k points × ~64 B) | < 0.1 GB |
| **total** | **~13–15 GB of 80 GB** |

The headroom allows `T = 1536` (amplification 1.15 instead of 1.23), larger
halos for big `disp_max`, or holding the whole reference volume across the 8
GPUs for time series (§11).

## 9. Fault tolerance

* A tile is written only after it has been solved completely, and its cells
  are written whole, with `status` last. A cell counts as written only when
  every result array holds it (`ResultStore.written_cells`), so a worker
  killed in the middle of a cell leaves nothing that looks finished.
* `run` skips tiles whose cells are all written. A crash loses at most the
  tiles a worker had taken: the one being solved and up to `prefetch_depth`
  prefetched (measured: a `kill -9` of one of two workers left 4 of 8 tiles
  for the resubmitted job, which then wrote a bit-identical store).
* Worker processes share one node-local queue. When one dies (`kill -9`, OOM,
  a GPU fault), the others keep draining the queue, and `run` lists the tiles
  still missing in `run_stats.json` and `failed_tiles.json` for the resubmission.
* A tile that raises is requeued once, then recorded as failed.
* `plan.json` records the zarr-vectors commit, the template hash and the
  config, so a resumed run cannot silently mix settings.

## 10. Where pyDVC diverges from CCPi

| Aspect | CCPi DVC | pyDVC | Effect |
|---|---|---|---|
| Point order / seeding | serial by distance; mean of GOOD among the 75 nearest already solved | wavefront shells (parity) or coarse field (production) | enables batching; parity is statistical, not bitwise |
| Neighbour search | O(N²) full sort | KD-tree / GPU grid hash | millions of points in one run |
| Sample template | a `FloatingCloud` per point (`std::mt19937` sphere) | one shared template per run | fits GPU constant/shared memory; statistically equivalent |
| Tricubic | Lekien–Marsden + central-difference derivative kernels per box | separable Catmull-Rom from the raw brick | same interpolant (verified in M1), no per-voxel precompute |
| Jacobian | forward differences, `h = 1e-10`, `ndof + 1` evaluations | analytic, normalisation included | 2.5–4× less work, no step-size sensitivity |
| Linear solve | column-pivoted QR of `JᵀJ` | batched Cholesky with a `SINGULAR` status | batched; ill-conditioning reported |
| Precision | float64 | float32 with relative coordinates and compensated sums | GPU throughput; checked against a float64 reference |
| Range failure | implicit: samples leave the interpolation box | explicit `‖u − seed‖∞ > disp_max`, plus brick validity | clearer semantics |
| Convergence failure | never reported by the LM path | reported (`report_convg_fail`, off in parity configs) | honest status |
| I/O | per-point row reads from raw files | per-tile bricks from OME-Zarr | the main performance gain |
| Output | `.disp` / `.stat` text | zarr-vectors store, plus `.disp` / `.stat` export | scale, viewers, iDVC compatibility |

## 11. Later directions (not in the MVP)

* **Time series.** Keep the reference resident across the 8 GPUs, sharded by
  z-slab: 137 GB of 640 GB HBM. Only deformed frames stream in, halving I/O
  per frame.
* **iDVC integration.** `pydvc ccpi dvc_config.txt` prints CCPi-style progress
  lines, so iDVC can launch pyDVC in place of `dvc` without GUI changes.
* **Strain.** kNN least squares over results, tile-parallel with halo reads.
* **Multi-node.** Already shaped by per-node LPT shares. Add a shared dynamic
  queue (file lock or Redis) if static shares leave nodes idle.
