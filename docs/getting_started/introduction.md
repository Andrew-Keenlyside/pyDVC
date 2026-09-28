# Introduction

This page explains what Digital Volume Correlation does, where zvDVC comes
from, and what it changes. It is for readers deciding whether zvDVC fits their
data, including those who already use iDVC.

zvDVC was previously called pyDVC.

---

## What DVC measures

Digital Volume Correlation (DVC) measures how a material deforms inside,
from two 3-D images of the same sample, usually X-ray computed tomography
(CT) scans:

**Reference scan**
: the sample before a change (unloaded, or at the first time step).

**Deformed scan**
: the same sample after the change (under load, or later in time).

DVC places many **search points** in the reference scan. Around each point it
takes a small block of voxels, the **subvolume**, and finds where that block
has moved to in the deformed scan. The pattern inside the block, such as
pores, grains or crystals, acts as a fingerprint. The answer at each point is a
**displacement** $(u, v, w)$ in voxels, and optionally a rotation and a local
strain of the subvolume.

Displacements at many points form a displacement field. **Strain**, the local
stretching and shearing of the material, is the spatial gradient of that
field, computed from neighbouring points after the correlation.

zvDVC implements the **local, subvolume-based** form of DVC: every point is
solved on its own, from its own subvolume. Global (finite-element) DVC, where
one displacement field is fitted to the whole volume at once, is out of scope.

## Where the method comes from: iDVC and CCPi DVC

zvDVC reimplements the method used by
**[iDVC](https://github.com/TomographicImaging/iDVC)**
([documentation](https://tomographicimaging.github.io/iDVC/)), the graphical
DVC application developed by the Tomographic Imaging / CCPi team at
UKRI-STFC (Apache-2.0; latest release v25.0.0, 2025). iDVC is where most users
set up a DVC run: it registers the two scans, draws masks, generates point
clouds, launches the correlation and shows the results.

iDVC does not correlate by itself. It writes a parameter file and runs the
**[CCPi DVC engine](https://github.com/TomographicImaging/DigitalVolumeCorrelation)**
(GPL-3.0), a C++/OpenMP program named `dvc`. That code was initially developed
by Prof. Brian K. Bay and collaborators, and its method is described in:

* B. K. Bay, T. S. Smith, D. P. Fyhrie, M. Saad, "Digital volume correlation:
  three-dimensional strain mapping using X-ray tomography", *Experimental
  Mechanics* 39, 217–226 (1999).
  [doi:10.1007/BF02323555](https://doi.org/10.1007/BF02323555)
* B. K. Bay, "Methods and applications of digital volume correlation",
  *J. Strain Analysis* 43, 745–760 (2008).
  [doi:10.1243/03093247JSA436](https://doi.org/10.1243/03093247JSA436)

zvDVC's method, status codes and file formats follow iDVC and CCPi DVC, and
its test case A is iDVC's own example dataset. If you use zvDVC, cite iDVC and
the CCPi DVC method papers as well as zvDVC: see {doc}`/how_to/cite`.

## Why a new implementation

CCPi DVC is accurate and well established, but its structure limits it to
modest point clouds:

| CCPi DVC today | Consequence at scale |
|---|---|
| Points are solved **one at a time**, in order of distance from a start point. Each is seeded from the average of already-solved neighbours. | OpenMP parallelism exists only *inside* a point (≈8 000 samples), so cores sit idle and GPUs cannot be used. |
| For **every point**, it re-reads a `(S + 2·disp_max + 4)³` box from **both** raw volumes row by row, then builds 8-term float64 derivative kernels over the whole box. | Most of each point's time is fixed overhead rather than correlation. The same voxels are read and processed hundreds of times. |
| The Jacobian comes from forward differences: `ndof + 1` objective evaluations per Gauss–Newton step. | ~2.5–4× more interpolation work than one analytic value+gradient pass. |
| Neighbour search is an O(N²) sort across the whole cloud. | Clouds of millions of points are impractical in a single run. |
| Inputs are flat raw/npy files and outputs are `.disp`/`.stat` text files. | Hard to use with 100 GB+ volumes, object stores or cluster file systems. |

(`S` is the subvolume size, `disp_max` the largest displacement searched, and
`ndof` the number of parameters of the shape function.) The measured effect on
iDVC's example: CCPi gains almost nothing from threads, 2.1 points/s with 32
threads against 6.6 points/s from 32 single-thread processes
([case A report](../benchmarks/2026-09-26-case-A-real.md)).

## What zvDVC keeps

* **The method.** Subvolume templates (cube or sphere), 3/6/12-DOF shape
  functions in CCPi's parameterisation, CCPi's five objectives
  (SAD, SSD, ZSSD, NSSD, ZNSSD), trilinear and tricubic interpolation, and
  CCPi's undamped Gauss–Newton with its stopping rules.
  zvDVC's tricubic interpolation is the same interpolant as CCPi's.
* **Status codes.** `GOOD`, `RANGE_FAIL`, `CONVG_FAIL` and `NOT_SEARCHED`
  keep CCPi's values (0 to −3). See {doc}`/spec/status_codes`.
* **Formats.** zvDVC reads CCPi's `dvc_in` parameter files and `.roi` point
  clouds, and writes CCPi's `.disp` and `.stat` files, so iDVC's results
  viewer opens its output. The `zvdvc-dvc` command is a drop-in for CCPi's
  `dvc`, so iDVC itself can run on zvDVC's engine ({doc}`/IDVC`).

## What zvDVC changes

zvDVC changes how the work is organised, not what is computed:

* **Tiles.** The point cloud is cut into tiles, blocks of whole storage cells.
  A tile is the unit of work handed to one GPU or CPU worker.
* **Bricks.** For each tile, zvDVC reads one box of the reference scan and
  one of the deformed scan, covering the tile plus a margin (the halo). Every
  point in the tile correlates against those bricks, so image data is read
  once per tile instead of once per point.
* **Batches.** Thousands of points are solved at once. All points share one
  sample template, and a fused GPU kernel does the warp, interpolation,
  residual and normal-equation sums in one pass, with an analytic Jacobian.
* **Seeding without serial order.** CCPi's neighbour-average seeding is
  sequential. zvDVC's `wavefront` mode solves whole distance shells at once
  (CCPi parity); its `coarse` mode solves a sparse sub-grid first and
  interpolates it to seed every point, which makes tiles independent.

{doc}`/getting_started/concepts` explains each of these;
{doc}`/ARCHITECTURE` has the design in full, including every place zvDVC
deliberately diverges from CCPi.

## Zarr Vectors and GPUDirect Storage

zvDVC is also a use case for **Zarr Vectors**, a chunked, cloud-native format
for vector geometry (points, lines, meshes), implemented in
[zarr-vectors-py](https://github.com/AllenInstitute/zarr-vectors-py) (BRIDGE
Neuroscience) from the [Zarr Vectors
specification](https://github.com/AllenInstitute/zarr_vectors) by Forest
Collman (Allen Institute). Search points and results live in zarr-vectors
stores, and their spatial chunk grid is zvDVC's unit of work:

* the chunk grid is the work partition (tiles are unions of chunks);
* one pooled `read_cells` call gives each tile its points;
* zarr-vectors' three-phase write pattern lets workers write results without
  locks;
* the results can be viewed with zarr-vectors / Neuroglancer tooling and
  exported to iDVC.

The images themselves are dense volumes, stored as sharded OME-Zarr v3.

The project also aims to show what zarr-vectors' GPU backend gains a real HPC
workload, including reads over **GPUDirect Storage (GDS)**, which moves bytes
from NVMe straight into GPU memory. **GDS is not wired into zvDVC and has not
been measured.** Every published zvDVC timing uses host decode plus one
pinned host-to-device copy. The GDS work, and the host-versus-GDS benchmark,
are roadmap milestone M6; see {doc}`/how_to/gpudirect_storage`.

## Current status

The single-GPU pipeline (milestones M0–M4 in {doc}`/MVP_PLAN`) is complete and
measured. On iDVC's example dataset (case A: two 1520 × 1257 × 1260 u8 scans,
CCPi's 4 680-point central grid, sphere 80 with 8 000 samples, 6-DOF, ZNSSD,
tricubic), on one workstation with an RTX A2000 12 GB and 32 CPU cores:

| run | wall time | vs iDVC |
|---|---|---|
| CCPi DVC as iDVC runs it (1 process, 32 threads) | 2 231 s (37 min) | 1× |
| CCPi DVC, 32 processes × 1 thread | 711 s | 3.1× |
| zvDVC, GPU (`fused`), in-memory parity mode | 1.6 s | 1 396× |
| zvDVC, GPU, CLI end to end (`plan` / `seed` / `run` / `finalize`) | 4.3 s | 524× |
| zvDVC, 32 CPU cores (`cpu`), in-memory parity mode | 7.6 s | 293× |

On the 4 628 interior points, zvDVC and CCPi agree on status for 100 % of
points, with a median displacement difference of 0.051 voxel, a 95th
percentile of 0.152 voxel, and no systematic difference (mean at most 0.003
voxel per axis). That spread is the same as zvDVC's disagreement with itself
when only the random choice of sample points changes
([error-floor study](../benchmarks/2026-09-26-error-floor-case-A.md)). The
other 52 points sit at the image edge, where CCPi reads outside the image and
zvDVC reports `RANGE_FAIL`. Full details:
[case A report](../benchmarks/2026-09-26-case-A-real.md) and
{doc}`/benchmarks/index`.

## What zvDVC is not (yet)

* **Not measured on more than one GPU.** The multi-GPU launcher (one process
  per GPU, a shared tile queue) is implemented and tested on CPU workers, but
  has not been measured on the 8 × H100 target. Multi-node runs are
  implemented and untested. The large-scale speed-ups in the README are
  modelled, not measured ({doc}`/PERFORMANCE`).
* **Not using GPUDirect Storage.** See above: M6.
* **Not a GUI.** zvDVC has no interface of its own. To work interactively, run
  iDVC with zvDVC as its engine through the `zvdvc-dvc` drop-in
  ({doc}`/IDVC`).
* **Not feature-complete against the plan.** IC-GN, FFT seeding, the repair
  pass (`zvdvc repair` exists as a command but is not implemented) and the
  pyramid-level coarse seeding pass are M5 work. The coarse and wavefront
  seeding passes hold both whole volumes in host memory.
* **Not a bit-for-bit copy of CCPi.** The two codes draw each subvolume's
  sample points at random from different generators, so results differ by the
  sampling spread (~0.03 voxel per axis at 8 000 samples on case A). Points
  whose subvolume leaves the image are `RANGE_FAIL` in zvDVC.
* **Validated on one real dataset.** Synthetic cases have known truth; case A
  is the only real scan pair measured so far.
