# Case A (iDVC example) on 4 CPU cores: pyDVC vs CCPi as iDVC runs it

Date: 2026-09-25. Same machine as the earlier pages: a cloud container with
4 vCPU (Intel Xeon @ 2.10 GHz) and 15 GB RAM, no GPU. pyDVC ran on its CPU
engine (backend `cpu`, numba); **no GPU number appears on this page**.

## What was compared, and on what data

**The real case A data could not be used here.** The iDVC example volumes
(Zenodo record [7363345](https://zenodo.org/records/7363345), 2 × 2.4 GB) sit
behind `zenodo.org`, which this environment's network policy blocks. Instead,
this page uses a **synthetic twin of case A** with the same geometry and
settings, so the timing ratio carries over:

| | case A (CCPi `dvc_test`) | twin used here |
|---|---|---|
| volumes | 1520 × 1257 × 1260 u8 (magma CT) | same size and dtype; speckle (feature size 3), 2 % noise |
| points | `central_grid.roi`, 4 680 points (90 × 52 grid in slice z = 630) | the same file |
| settings | `dvc_input.txt`: sphere 80, 8 000 samples, 6-DOF, ZNSSD, tricubic, `disp_max` 38, `rigid_trans` (34, 4, 0) | the same |
| displacement | ≈ (33.1, 3.4, −3.1) (CCPi's reference `.disp`, 5 points) | (33.14, 3.44, −3.11) + a 0.04 % strain; truth known |

Per-point cost is set by the geometry and settings above, not by the image
content. So the **speed** comparison should carry over to the real data,
within the effect of iteration counts. **Agreement** on the real data needs
the real data: see "To run on the real dataset".

Runs (`python -m pydvc.bench.compare_ccpi`):

* **iDVC**: CCPi `dvc` as iDVC launches it, one process with OpenMP on every
  core (4 threads). This is the number an iDVC user waits for.
* **CCPi best**: 4 `dvc` processes × 1 thread on disjoint quarters of the
  grid, the best this node gets from CCPi without code changes.
* **pyDVC parity**: whole volumes in memory, CCPi's point order (wavefront),
  timed from reading the volumes to the last point.
* **pyDVC CLI**: `pydvc plan / seed / run / finalize` as four fresh processes,
  so interpreter start, imports and loading the cached compiled kernels are
  counted. Results are written to the zarr-vectors store.

CCPi 22.0.0 is the reference, because its tricubic path works. 25.0.0 is what
a fresh iDVC install pulls today.

## Speed

| run | wall time | pt/s | vs iDVC |
|---|---|---|---|
| CCPi 22.0.0, iDVC mode (1 process, 4 threads) | 1 069 s (17.8 min) | 4.4 | 1.0× |
| CCPi 22.0.0, 4 processes × 1 thread | 807 s | 5.8 | 1.3× |
| CCPi 25.0.0, iDVC mode | 1 059 s | 4.4 | 1.0× |
| CCPi 25.0.0, 4 processes × 1 thread | 793 s | 5.9 | 1.3× |
| **pyDVC CPU engine, parity mode** | **28.6 s** (read 13.4 s + solve 15.2 s) | 164 | **37×** |
| **pyDVC CPU engine, CLI end to end** | **47.6 s** | 98 | **22×** |

* **On the same 4 cores, pyDVC is 22–37× faster than iDVC's engine.** The
  solve alone runs 309 pt/s, about 70× CCPi. Reading and decoding the two
  2.4 GB OME-Zarr volumes (13 s) is now half of pyDVC's time; CCPi re-reads a
  160³ box from both volumes for every point.
* CCPi takes 228 ms per point with 4 threads and 680 ms with 1. At this
  subvolume size OpenMP helps (3×), unlike case S. Four single-thread
  processes only reach 1.3× iDVC mode, probably because they compete for
  memory bandwidth.
* PERFORMANCE.md §3 modelled CCPi at 45–150 ms per point for case A. The
  measurement here, on 4 vCPU, is 228 ms: slower than modelled, as case S was.
* The **GPU** path is modelled at 2–6 s for case A on one H100 (bound by
  I/O). It has not run yet.

## Agreement and accuracy

| comparison | median \|du\| | p95 \|du\| | RMSE (x, y, z) | status agreement |
|---|---|---|---|---|
| pyDVC vs ground truth | 0.0072 | 0.0129 | 0.0048, 0.0047, 0.0045 | — |
| CCPi 22.0.0 vs ground truth | 0.0072 | 0.0137 | 0.0139, 0.0068, 0.0064 | — |
| pyDVC vs CCPi 22.0.0 | **0.0097** | **0.0175** | 0.0064, 0.0063, 0.0062 | 97.8 % |
| CCPi 25.0.0 vs ground truth | 3.1061 | 3.1496 | 0.87, 0.48, 2.95 | — |

* Against CCPi 22.0.0, the Q1 displacement criteria pass by a wide margin
  (median ≤ 0.05, p95 ≤ 0.2).
* **Status agreement is 97.8 %, just under Q1's 98 %.** All 104 disagreements
  are points at the right edge of the grid (x = 1458–1474). With the 33-voxel
  displacement, their deformed subvolume reaches x ≈ 1531, past the volume's
  last voxel (1519).
  * pyDVC reports these points `RANGE_FAIL`, because samples left the image
    (ARCHITECTURE.md §10, "Range failure").
  * CCPi reports them GOOD, with errors of 0.07 voxel median and 0.23 max:
    5–15× worse than its interior points (p99 0.016).
  * On the 4 576 interior points the two codes agree on every status.

  The criterion should count this divergence, or pyDVC could gain a
  CCPi-style "pad and continue" option for parity runs. That is a decision for
  the owner.
* **CCPi 25.0.0 is off by 3.1 voxels everywhere while reporting every point
  GOOD**: the broken tricubic path found in M0. iDVC users installing today get
  this build.
* The larger x RMSE of CCPi 22.0.0 against truth comes from the same edge
  points.

## To run on the real dataset

With network access to `zenodo.org` (and GitHub for CCPi's test files):

```bash
conda create -n ccpi -c ccpi -c conda-forge ccpi-dvc=22.0.0      # the reference build
python -m pydvc.bench.case_a fetch --data data/magma                # 2 x 2.4 GB + CCPi's dvc_test files
python -m pydvc.bench.case_a run --data data/magma --out runs/case_A \
    --ccpi-exe "$(conda run -n ccpi which dvc)" --backends cpu fused   # fused on a GPU machine
```

`run` builds the configuration from CCPi's own `dvc_input.txt` and checks the
volume layout (it transposes an x-first `.npy` if needed). It runs CCPi in
iDVC mode and as N processes, then pyDVC in parity and CLI modes, and writes
`report.md`. The report includes agreement with CCPi's reference `.disp`, which
holds only 5 of the 4 680 points because the 2018 reference run stopped early.
Budget about 20 min per CCPi run on a 4-core machine. `--reuse-ccpi` re-times
finished CCPi runs from their `.stat` files.
