# Running pyDVC on the cluster (Grid Engine, 8 × H100)

This page takes pyDVC from a workstation to a scaling benchmark on a GPU node of
a Grid Engine cluster (written for UCL CS **pryor**; paths below use its
storage). Everything runs inside an Apptainer image, so the cluster needs no
Python environment and compute nodes need no internet access.

```
workstation                                   cluster (shared storage)
───────────                                   ────────────────────────
containers/build.sh ──► pydvc.sif ──rsync──►  …/containers/pydvc.sif
git push ─────────────────────────────────►   …/pydvc  (checkout: sge/, configs)
                                              qsub sge/benchmark.qsub
                                              ──► …/pydvc/results/campaign_size2048/campaign.md
```

## 1. Build the image (once per code change you want baked in)

On a Linux machine with internet access and Apptainer (the workstation has
1.4.2):

```bash
# pick CUDA from the driver: on a GPU node, `qrsh -l gpu=true,gpu_type=h100` then `nvidia-smi`
#   driver >= 570 → 12.8, >= 560 → 12.6, >= 550 → 12.4, >= 525 → 12.1
APPTAINER_TMPDIR=/hdd/andrew/apptainer-tmp bash containers/build.sh --cuda 12.4 --out pydvc.sif
rsync -avP pydvc.sif pryor:/home/akeenlys/storage_main/bridge_project_data/containers/
```

The build takes 10–20 minutes. The image holds pyDVC (with the commit it was
built from, `PYDVC_COMMIT`), CuPy with a matching NVRTC and CUDA headers,
numba, nvCOMP, a C++20 compiler for the CUDA emulator tests, Nsight Systems and
Compute, fio, and CCPi `dvc` 22.0.0. For code edits without a rebuild, point
`PYDVC_DEV_SRC` at a checkout (below); the checkout's `src/` is then used
instead of the image's copy.

## 2. Set up the cluster side (once)

```bash
ssh pryor
cd /home/akeenlys/storage_main/bridge_project_data
git clone -b cluster-groundwork https://github.com/Andrew-Keenlyside/pyDVC.git pydvc
mkdir -p logs containers
```

The job scripts read their settings from variables at the top (`ROOT`, `SIF`,
`BINDS`, …); the defaults point at `/home/akeenlys/storage_main/bridge_project_data`.
Every path a job touches must be under `BINDS`. If a directory is reached through a symlink or an
automount (for example under `/SAN/external/`), use its real path.

## 3. Run the scaling campaign

From the checkout:

```bash
cd /home/akeenlys/storage_main/bridge_project_data/pydvc
qsub sge/benchmark.qsub                                  # case L 2048³, 1/2/4/8 GPUs, every step
qsub -v SIZE=4096 sge/benchmark.qsub                     # the large case: ~275 GB of volumes
qsub -pe gpu 4 -v DEVICES="1 2 4" sge/benchmark.qsub     # four GPUs only
qsub -v STEPS="strong weak",REDO="strong" sge/benchmark.qsub   # re-run a subset
```

The job's first action is a preflight inside the container: every GPU it holds
must open and run a CUB reduction (which exercises NVRTC and the CUDA headers).
It stops within seconds, not after the queue wait, if the image, the GPUs or a path
is wrong. It also refuses to run if Grid Engine started it without allocating GPUs
(`CUDA_VISIBLE_DEVICES` unset), instead of using other people's cards.

Steps (`pydvc.bench.campaign`; each writes `RESULTS/<step>.json`, and a
finished step is skipped when the job is resubmitted, so a job that hits
`h_rt` simply continues):

| step | measures | time (2048³, 8 GPUs; estimates) |
|---|---|---|
| `env` | GPUs, driver, topology (`nvidia-smi topo -m`), CPU, memory | seconds |
| `check` | `pydvc check gpu`: the whole test suite and selftest on this node | ~5 min |
| `kernel` | per-GPU kernel speed: synthetic u8 and u16 (and case A with `CASE_A=`) | ~1 min |
| `data` | case L: synthetic u16 pair, inclusion field, known truth; generated once into `DATA` and reused; tiled for ≥ 4 tiles per GPU | ~5–15 min at 2048³ on the node's cores (once) |
| `storage` | fio sequential read (8 × 4 GB) and pyDVC's brick-read ceiling (8 and 32 threads) | minutes |
| `strong` | the whole case on 1, 2, 4, 8 GPUs; efficiency, and each run's timeline | tens of minutes |
| `weak` | 4 tiles per GPU on 1, 2, 4, 8 GPUs | tens of minutes |
| `profile` | Nsight Systems on a 16-tile run over all GPUs: kernels, NVTX stages, per-GPU streams | minutes |
| `ncu` | Nsight Compute counters for one `gn_sums` launch (skipped if counters need root) | minutes |
| `e2e` | plan, seed, run, finalize on all GPUs; accuracy against the truth | minutes |
| `report` | `campaign.md`: every table on one page | seconds |

## 4. Read the results

* **`campaign.md`**: kernel speed, storage, strong and weak scaling tables
  (points/s, efficiency, utilisation, load imbalance, end-of-queue idle, I/O
  wait), end to end and accuracy.
* **Why a run scaled as it did**: each scaling run has an events folder
  (`scaling_strong/strong_n8/work/events/<run id>/`). Summarise it with
  `python -m pydvc.bench.timeline <that folder>`: per GPU, the share of time spent
  solving, start-up, read and write stalls, and idle time at the end of the queue.
  `gpu_telemetry.csv` in the same folder has 1 Hz utilisation, memory, clock
  and power for every GPU.
* **Profiles**: `profile/run.nsys-rep` opens in Nsight Systems (`nsys-ui`) on
  the workstation. Stages are labelled (read / solve / write per tile, each
  Gauss–Newton iteration). `profile/stats.csv` has the kernel and NVTX summaries.

Send back the whole `RESULTS` folder, or at least `*.json`, `campaign.md`, and
`profile/`.

## 5. Your own data

```bash
qsub -v CONFIG=/home/akeenlys/storage_main/bridge_project_data/myrun/config.yaml sge/pydvc.qsub
qsub -v CONFIG=...,STAGES="run finalize" sge/pydvc.qsub     # after a time-out: only missing tiles are solved
```

Convert large inputs to OME-Zarr once (`pydvc convert`) and set
`cluster.tile_shape` so each GPU gets several tiles (the campaign's `data`
step shows the rule: ≥ 4 tiles per GPU). Results store, `.disp` export and
resume work as on the workstation.

## Troubleshooting

| symptom | cause |
|---|---|
| `FATAL: Grid Engine started this job without allocating it any GPUs` | the GPU prolog found none free; resubmit |
| `FATAL: … is not visible inside the container` | the path is outside `BINDS`, or an automount: use the real path |
| `incomplete type "__nv_fp8_e8m0"` | a CuPy using CUDA headers of another version; cannot happen in the image (outside it, see docs/TESTING.md) |
| `ncu` step `skipped: hardware counters not permitted` | profiling counters need root on this cluster; ask the admins or skip |
| scaling warning: fewer than 4 tiles per device | the tile is too large for the GPU count; pass `--tile` / `TILE` smaller |

The SLURM scripts in `scripts/slurm/` are for SLURM sites and have not been
run; on pryor use `sge/`.
