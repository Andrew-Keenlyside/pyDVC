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
are converted once with `pydvc convert`.

### Points and results: zarr-vectors

zarr-vectors is a format for **vector geometry**: points, lines, meshes and
skeletons. It does not store the dense images. In pyDVC it does four jobs:

1. **The work partition.** A zarr-vectors store is cut into a spatial chunk
   grid. Tiles are unions of chunks, so a tile's input rows and output rows
   live in the same cells, and no two workers ever write the same cell.
2. **Device reads.** `zarr_vectors.building.read_cells(..., device="cuda")`
   delivers a tile's points to the GPU in one pooled read, decoded on the
   device, as CSR over cells. `read_neighbourhood(halo=1)` serves repair and
   strain across tile borders.
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
  assigns bins itself.
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
| `coarse` | coarse sub-grid on pyramid level 1, then every tile independent | production, multi-GPU |
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
* **IC-GN (M5).** Inverse compositional. The Hessian comes from the
  reference once per point, and each iteration needs target values only.
* **Interpolation.** Separable Catmull-Rom, which equals CCPi's
  Lekien–Marsden tricubic with central-difference derivatives (M1 verifies
  this), plus trilinear and nearest.
* **Objectives.** CCPi's five, with identical scaling
  ([`kernels/objective.py`](../src/pydvc/kernels/objective.py)).
* **Precision.** float32 on device. Sample positions are formed relative to
  the point centre, so rounding stays near 1e-4 voxel even at coordinate
  4096. ZNSSD sums use compensated (Kahan) accumulation. Every GPU result is
  checked against the float64 numpy reference in tests.

## 8. GPU execution

Each worker process runs three streams:

| Stream | Work |
|---|---|
| copy | brick reads for the next `prefetch_depth − 1` tiles into pinned buffers, then host-to-device copy |
| compute | reference sampling, then fused GN steps and batched solves over the active set, batch after batch |
| host thread | encodes and writes the previous tile's result cells through zarr-vectors (device encode where available) |

The fused kernel ([`kernels/fused.py`](../src/pydvc/kernels/fused.py)) never
writes per-sample intermediates to global memory. The active set shrinks as
points converge, so late iterations cost only the stragglers.

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
  are written whole. A crash loses at most the tiles in flight.
* `run` checks which cells already exist (`ResultStore.written_cells`) and skips those tiles.
* A tile that raises is requeued once, then recorded in `workdir/failed_tiles.json` and reported by `finalize`.
* `plan.json` records the zarr-vectors commit, the template hash and the
  config, so a resumed run cannot silently mix settings.

## 10. Where pyDVC diverges from CCPi

| Aspect | CCPi DVC | pyDVC | Effect |
|---|---|---|---|
| Point order / seeding | serial by distance; mean of GOOD among the 75 nearest already solved | wavefront shells (parity) or coarse field (production) | enables batching; parity is statistical, not bitwise |
| Neighbour search | O(N²) full sort | KD-tree / GPU grid hash | millions of points in one run |
| Sample template | a `FloatingCloud` per point (`std::mt19937` sphere) | one shared template per run | fits GPU constant/shared memory; statistically equivalent |
| Tricubic | Lekien–Marsden + central-difference derivative kernels per box | separable Catmull-Rom from the raw brick | same interpolant (verified in M1), no per-voxel precompute |
| Jacobian | forward differences, `h = 1e-10`, `ndof + 1` evaluations | analytic | 2.5–4× less work, no step-size sensitivity |
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
