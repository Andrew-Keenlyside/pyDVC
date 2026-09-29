# Installation

This page installs zvDVC on a workstation or a cluster node and checks that it
works. It is the short version; {doc}`/TESTING` walks through installation,
the check suite, the first GPU run and the example dataset in full.

---

## Requirements

* **Python ≥ 3.11** (zarr v3, which zarr-vectors needs, requires it).
* **Linux x86-64.** macOS works for the CPU paths.
* **For the GPU backends:** an NVIDIA GPU with a CUDA 12 driver.
* **`git` and access to github.com** at install time: pip fetches
  zarr-vectors from a pinned commit on GitHub.

## Create an environment

The environment files in
[`envs/`](https://github.com/Andrew-Keenlyside/zvDVC/tree/main/envs) work
with conda, mamba and micromamba.

| file | environment name | for |
|---|---|---|
| `envs/zvdvc-cpu.yml` | `zvdvc` | laptops and CPU nodes: Python 3.11, numpy, scipy, numba, a C++ compiler for the CUDA-emulator tests |
| `envs/zvdvc-gpu.yml` | `zvdvc-gpu` | GPU workstations and nodes: the same, plus CuPy from conda-forge (with the CUDA runtime and NVRTC), `cuda-version=12` |
| `envs/ccpi-dvc.yml` | `ccpi-dvc` | optional: CCPi's `dvc` 22.0.0, kept apart from zvDVC |

CPU only:

```bash
git clone https://github.com/Andrew-Keenlyside/zvDVC.git && cd zvDVC
micromamba create -f envs/zvdvc-cpu.yml && micromamba activate zvdvc
pip install -e ".[test,cpu-fast]"
```

GPU:

```bash
git clone https://github.com/Andrew-Keenlyside/zvDVC.git && cd zvDVC
micromamba create -f envs/zvdvc-gpu.yml && micromamba activate zvdvc-gpu
pip install -e ".[test,cpu-fast]"
pip install nvidia-nvcomp-cu12      # zarr-vectors' GPU codecs, without a second CuPy
```

The editable install provides two commands: `zvdvc` (every stage of a run)
and `zvdvc-dvc` (the drop-in for CCPi's `dvc`, used from iDVC; see
{doc}`/IDVC`).

```{warning}
Three pitfalls, each met on the first GPU installation (details in
{doc}`/TESTING`):

* **Activate the environment** (or use `conda run -n zvdvc-gpu ...`). CuPy
  finds its CUDA headers through `CONDA_PREFIX`; calling the environment's
  `python` directly can pick up a system CUDA whose headers do not match, and
  CuPy's kernels then fail to compile. `zvdvc check` refuses to start in that
  state.
* **One CuPy only.** The `[gpu]` extra, like zarr-vectors' `[gpu-codecs]`
  extra, pulls in the pip wheel `cupy-cuda12x`, which overwrites conda's CuPy.
  In a conda environment, install `nvidia-nvcomp-cu12` as above instead. If it
  has happened: `pip uninstall -y cupy-cuda12x`, then
  `micromamba install --force-reinstall cupy-core cupy`.
* **Run tests as `python -m pytest`**, so a `pytest` elsewhere on `PATH`
  cannot shadow the environment's.
```

## Optional extras

From [`pyproject.toml`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/pyproject.toml):

| extra | installs | use |
|---|---|---|
| `[test]` | pytest ≥ 8 | the test suite and `zvdvc check` |
| `[cpu-fast]` | numba ≥ 0.59 | the `cpu` backend: the fused Gauss–Newton step on all CPU cores. Without it, CPU runs fall back to the slower `numpy` reference |
| `[gpu]` | `cupy-cuda12x` ≥ 13 (Linux), `zarr-vectors[gpu-codecs]` | the `fused` and `cupy` GPU backends, for pip-only environments (not with the conda GPU environment; see the warning above) |
| `[gpu-io]` | `zarr-vectors[gpu-io]` (kvikio), `nvidia-nvcomp-cu12` | the kvikio / GPUDirect Storage brick read path (`volumes.gpu_io`) with zstd decoded on the GPU. For pip-only environments: in the conda GPU environment install kvikio from the `rapidsai` channel instead (see {doc}`/how_to/gpudirect_storage`) |
| `[tiff]` | tifffile ≥ 2024.1 | `.tif` / `.tiff` stacks as input to `zvdvc convert` |
| `[mpi]` | mpi4py ≥ 3.1 | reserved for MPI launches; Open MPI ranks are currently read from environment variables, so it is not required |

## The zarr-vectors pin

zvDVC depends on zarr-vectors' `gpu-backend` branch, which is under active
development, so `pyproject.toml` pins one commit
(`06a3bc8a08afefc02cd1ffb265d67a03df798b87`) and it is bumped deliberately.
Every zarr-vectors call goes through `zvdvc.io`, so an API change is absorbed
in one place. `plan.json` records the zarr-vectors commit a run used.

## CCPi DVC (optional)

The baselines and the comparison tools can run CCPi's `dvc`, the engine iDVC
uses. zvDVC does not need it to run.

```bash
# either as a conda environment ...
micromamba create -f envs/ccpi-dvc.yml
export ZVDVC_CCPI_DVC=$(micromamba run -n ccpi-dvc which dvc)
# ... or unpacked without conda (Linux x86-64; into ~/.local/opt/ccpi-dvc-22.0.0)
eval "$(scripts/get_ccpi_dvc.sh)"
```

Both give **CCPi `dvc` 22.0.0** and set `ZVDVC_CCPI_DVC`, which
`zvdvc selftest` and the case A benchmark read. Do not use `ccpi-dvc` 25.0.0
as a reference: its tricubic interpolation is broken
([M0–M1 report](../benchmarks/2026-09-25-M0-M1-case-S.md)). Version 22.0.0
needs `_openmp_mutex >= 5.1`, which only Anaconda's `main` channel provides;
`scripts/get_ccpi_dvc.sh` exists for sites that cannot use that channel.

## Check the installation

```bash
zvdvc selftest          # PASS/FAIL per backend on a 96³ synthetic case
zvdvc check quick       # ~3 min: the test suite without GPU and slow tests
zvdvc check gpu         # adds the GPU tests and the selftest
```

`zvdvc selftest` reports the environment (packages, GPUs, CCPi), runs
`plan → seed → run → finalize` on every available backend, and checks the
result against ground truth (≥ 99 % GOOD, RMSE ≤ 0.02 voxel). With a GPU it
also checks that the GPU backends agree with the CPU engine (≤ 1e-3 voxel);
with `ZVDVC_CCPI_DVC` set it runs CCPi on the same case. It writes
`runs/selftest/selftest.json`. `zvdvc check` writes
`runs/check/<time>-<commit>/check.json`; the tiers and their rules are in
{doc}`/TESTING`.

## Container (clusters)

For clusters without a suitable Python environment, `containers/build.sh`
builds an Apptainer image, `zvdvc.sif`, with zvDVC, CuPy, the numba CPU
engine, Nsight profilers, fio and CCPi `dvc` 22.0.0:

```bash
bash containers/build.sh --cuda 12.4 --out zvdvc.sif      # --cuda 12.1, 12.4 (default), 12.6 or 12.8
apptainer exec --nv zvdvc.sif zvdvc check gpu
```

Pick `--cuda` from the node's driver version (`nvidia-smi`): 12.8 needs
driver ≥ 570, 12.6 ≥ 560, 12.4 ≥ 550, 12.1 ≥ 525. The build needs Linux
x86-64, Apptainer or Singularity, internet access, about 20 GB of temporary
space, and root, sudo or `--fakeroot`. It copies the working tree into the
image and records its commit in the image's labels. {doc}`/CLUSTER` covers
running the image on a Grid Engine cluster.

## Coming from pyDVC

The package is now `zvdvc`, the commands are `zvdvc` and `zvdvc-dvc`, and
environment variables are `ZVDVC_*` (for example `ZVDVC_CCPI_DVC`). Results
stores written by pyDVC still open.
