# Testing pyDVC on your own machines

This guide takes the MVP from a fresh checkout to numbers you can compare:
first on a local machine (CPU, then GPU if you have one), then on a cluster
node. Every step prints or writes a result you can check; the last section
lists what to send back.

Status: M0–M4 pass their tests on CPU and, since September 2026, on a GPU
(RTX A2000 12 GB, CUDA 12.9). Day to day, use the **local check suite**
(section 1): pyDVC has no hosted CI, because its GPU code needs a GPU.

## 0. Install

Python ≥ 3.11 on Linux x86-64 (macOS works for the CPU paths). The
environment files are in `envs/`; conda, mamba and micromamba all read them.

```bash
git clone <this repo> pyDVC && cd pyDVC

# CPU only (laptop, CPU cluster nodes)
micromamba create -f envs/pydvc-cpu.yml && micromamba activate pydvc
pip install -e ".[test,cpu-fast]"

# GPU (CUDA 12 driver): the same; cupy comes from conda, the GPU codecs from pip
micromamba create -f envs/pydvc-gpu.yml && micromamba activate pydvc-gpu
pip install -e ".[test,cpu-fast]"
pip install nvidia-nvcomp-cu12            # zarr-vectors' GPU codecs; NOT the [gpu-codecs] extra (see below)

# CCPi dvc 22.0.0 for the baselines, either as a conda environment ...
micromamba create -f envs/ccpi-dvc.yml
export PYDVC_CCPI_DVC=$(micromamba run -n ccpi-dvc which dvc)
# ... or unpacked without conda (Linux; uses the system libgomp)
eval "$(scripts/get_ccpi_dvc.sh)"
```

`pip install -e .` fetches zarr-vectors from GitHub at the pinned commit, so it
needs `git` and access to github.com.

Three GPU pitfalls, each met on the first GPU run:

* **Always activate the environment** (or use `conda run -n pydvc-gpu ...`).
  CuPy finds its CUDA headers through `CONDA_PREFIX`. Calling the env's
  `python` directly makes it fall back to a system CUDA (for example
  `/usr/local/cuda-12.4`), whose headers do not match the environment's NVRTC,
  and CuPy's reductions fail to compile (`incomplete type "__nv_fp8_e8m0"`).
  `pydvc check` refuses to start in that state.
* **One CuPy only.** zarr-vectors' `[gpu-codecs]` extra pulls in the pip wheel
  `cupy-cuda12x`, which overwrites conda's cupy in the same directory. If that
  happened: `pip uninstall -y cupy-cuda12x`, then
  `micromamba install --force-reinstall cupy-core cupy`.
* **Run tests as `python -m pytest`**, so a `pytest` elsewhere on `PATH`
  (for example `~/.local/bin`) cannot shadow the environment's. CCPi 22.0.0 needs `_openmp_mutex >= 5.1`,
which only Anaconda's `main` channel provides. That is why `envs/ccpi-dvc.yml`
lists it, and why `get_ccpi_dvc.sh` exists for sites that cannot use it. Do
not use `ccpi-dvc` 25.0.0 as a reference: its tricubic interpolation is broken
([benchmarks](benchmarks/2026-09-25-M0-M1-case-S.md)).

## 1. Local check suite

```bash
pydvc check quick                             # ~3 min: tests without GPU/slow ones (the CUDA emulator included)
pydvc check gpu                               # + GPU tests and `pydvc selftest`
pydvc check full --case-a runs/case_A_data    # + slow tests, kernel benchmark, case A accuracy and speed (~15 min)
```

Each run writes `runs/check/<time>-<commit>/check.json` (and `runs/check/latest`),
prints PASS / PASS (partial) / WARN / FAIL, and exits non-zero on FAIL. The
rules are in [`bench/check.py`](../src/pydvc/bench/check.py):

* **Accuracy fails the check**: tests, selftest, kernel sums within 1e-4 of
  float64, case A results against the baseline arrays (status ≥ 99.9 %,
  |du| ≤ 1e-3 voxel) and against CCPi (no worse than the baseline).
* **Speed is compared on the same hardware only**, and only on an idle GPU:
  kernel timings warn at +20 % and fail at +40 %, end-to-end at +25 % / +50 %
  (back-to-back runs on a workstation GPU differ by up to ~18 %).

Baselines are per machine: `docs/benchmarks/baselines/<machine>.json`
(`PYDVC_MACHINE`, default the host name) is committed, and the reference result
arrays it points to live in `runs/check/baseline/`. After a change that is meant
to move the numbers, record a new baseline with a reason, from a `full` run:

```bash
pydvc check full --case-a runs/case_A_data --update-baseline --reason "sorted templates"
```

| variable | meaning |
|---|---|
| `PYDVC_CASE_A` | case A data directory (instead of `--case-a`); without it the case A steps are skipped (`PASS (partial)`) |
| `PYDVC_CASE_A_CACHE` | where C-ordered `.raw` copies of the case A volumes go (default `runs/case_A`, ~4.8 GB) |
| `PYDVC_MACHINE` | baseline name |
| `PYDVC_CCPI_DVC` | CCPi `dvc` 22.0.0, for the selftest's CCPi check and `bench.case_a` |

To run the quick tier before every push (opt in, once per clone):
`git config core.hooksPath scripts/hooks`; skip once with `git push --no-verify`.

`pydvc selftest` on its own gives a PASS/FAIL table for a fresh install and
writes `runs/selftest/selftest.json`.

`selftest` reports the environment (packages, GPUs, CCPi). It writes a 96³
phantom and runs `plan → seed → run → finalize` on every backend available.
It then checks accuracy against ground truth (≥ 99 % GOOD, RMSE ≤ 0.02 voxel),
checks that the GPU backends agree with the CPU engine (≤ 1e-3 voxel), and,
if `PYDVC_CCPI_DVC` is set, runs CCPi on the same case. The reference output
on a 4-core VM:

```
[PASS] pipeline on 'cpu' vs ground truth: 125 pts, 100.0 % GOOD, RMSE 0.0014, 12.5 s
[PASS] CCPi dvc runs and agrees with pyDVC: ...
PASS
```

## 2. GPU runs by hand (workstation or one cluster GPU)

Do this on any new GPU or driver before a large run: it is where a kernel bug
would show up. `pydvc check gpu` runs the first two lines.

```bash
python -m pytest -m gpu -q           # cupy + fused vs the numpy reference; wavefront, threshold, u8 edge cases
pydvc selftest                       # now also runs "fused" and "cupy" and their parity checks
python -m pydvc.bench.kernel --backends fused cpu --json runs/kernel.json    # real data if PYDVC_CASE_A is set
python -m pydvc.bench.density --backends fused --json runs/density.json      # time vs points on case A
python -m pydvc.bench.throughput --backend fused cupy cpu --samples 2000 4096 --dof 6 12 --size 48 --batch 32768
```

`bench.kernel` times the two kernels a solve spends its time in (µs per
point-iteration) on 2 048 points of the case A volumes and checks them against
float64. `bench.throughput` gives pt/s and an estimated FLOP/s per backend on a
synthetic volume that fits in cache, so it overstates GPU speed on real data.
`bench.density` fits `t = t0 + τ·S + c·N` to wall time against point count.
For measured kernel counters, run under Nsight Compute (it needs root, or the
driver option `NVreg_RestrictProfilingToAdminUsers=0`, on most workstations):

```bash
ncu --set full --kernel-name regex:gn_sums -o gn_sums \
    python -m pydvc.bench.throughput --backend fused --samples 4096 --dof 6 --batch 8192
```

If a GPU test fails, the backend `emulated` runs the same CUDA source on the
host (`pytest tests/test_fused.py`). That tells a kernel logic error apart from
a launch or driver problem.

## 3. Test datasets

### 3a. Synthetic cases (any size, known truth)

```bash
pydvc synth --shape 256 256 256 --field affine --spacing 16 --out data/S          # case S, ~1 min
pydvc plan data/S/config.yaml && pydvc seed data/S/config.yaml
pydvc run data/S/config.yaml && pydvc finalize data/S/config.yaml --disp
pydvc compare data/S/results.zarrvectors data/S/truth.npz
```

Fields are `rigid`, `affine`, `sinusoid` and `inclusion`. `--noise 0.02` adds
2 % noise, and `--dof`, `--subvol-size`, `--subvol-npts` and `--disp-max` set
the search. Generation is parallel over all cores: 1024³ takes about 9 min on
4 cores, and 4096³ should take well under an hour on a 64-core node.

### 3b. The iDVC example dataset (case A): pyDVC vs iDVC

Zenodo [7363345](https://zenodo.org/records/7363345): two 2.4 GB volumes, with
CCPi's own settings and 4 680-point grid.

```bash
python -m pydvc.bench.case_a fetch --data data/magma
python -m pydvc.bench.case_a run --data data/magma --out runs/case_A --backends cpu    # add "fused" on a GPU
```

`run` builds the configuration from CCPi's `dvc_input.txt` and runs CCPi as
iDVC launches it (one process, all cores) and as N single-thread processes.
It then runs pyDVC in parity mode and through the CLI, and writes
`runs/case_A/report.md`: times, speed-ups, and point-by-point agreement with
CCPi and with CCPi's 5-point reference `.disp`. Allow about 20 min per CCPi
run on 4 cores. `--reuse-ccpi` re-times finished CCPi runs instead of
repeating them.

What to expect: on a synthetic twin of this case, on 4 cores, pyDVC's CPU
engine took 29–48 s against CCPi's 17.8 min
([benchmarks](benchmarks/2026-09-25-M4-case-A-twin.md)). On the real data the
iteration counts, and so the times, will differ somewhat. Points whose
deformed subvolume leaves the image are `RANGE_FAIL` in pyDVC but reported
by CCPi, so status agreement stops short of 100 % at the grid's edges.

`python -m pydvc.bench.compare_ccpi CONFIG --out DIR` does the same
comparison for any configuration (`--truth truth.npz` adds ground truth).

## 4. Cluster (SLURM, one 8×H100 node)

```bash
# the whole M4 sequence on case L: generate, storage baseline, Q2, Q3, end to end, accuracy
sbatch scripts/slurm/case_L.sbatch 2048 /scratch/$USER/pydvc/caseL2048
sbatch scripts/slurm/case_L.sbatch 4096 /scratch/$USER/pydvc/caseL4096      # ~275 GB of volumes

# your own data: plan -> seed -> run -> finalize, one process per GPU
sbatch scripts/slurm/pydvc_8xh100.sbatch configs/large_8xh100.yaml
```

Pieces you can run by hand on an allocated node:

| command | measures |
|---|---|
| `scripts/slurm/storage_baseline.sh DATA_DIR CONFIG` | file-system bandwidth (fio) and pyDVC's own brick-read bandwidth |
| `python -m pydvc.bench.scaling CONFIG --devices 1 2 4 8` | Q3: pt/s and efficiency from 1 to 8 GPUs, I/O wait per run |
| `pydvc run CONFIG --devices 0 1` | a run on chosen GPUs; `--cpu-workers N` for CPU nodes |

Resume and recovery (M4's `kill -9` criterion): during `pydvc run`, kill one
worker, whose pid is in `WORKDIR/workers/slot*.pid`. The other workers finish
the queue. `run_stats.json` and `failed_tiles.json` list the missing tiles, and
running `pydvc run` again solves only those.

Notes:

* Multi-node runs (`sbatch --nodes=N`) are implemented, with nodes splitting
  tiles by rank, but untested.
* `seed` with `strategy: coarse` or `wavefront` holds both whole volumes in
  memory and refuses volumes that do not fit. At 4096³, use `rigid` (as
  `configs/large_8xh100.yaml` and `case_L.sbatch` do) until the pyramid-level
  coarse pass (M5).

## 5. What to send back

* `runs/check/latest/check.json` (a `full` run) from each machine.
* `runs/selftest/selftest.json`, `runs/kernel.json` and, if you ran it, the Nsight report.
* `runs/case_A/report.md` and `report.json`.
* From the cluster: the case L job log, `scaling/scaling.json`, the
  `storage_baseline_*.txt` file, `WORKDIR/run_stats.json` and `plan.json`.

These replace the remaining modelled numbers in
[PERFORMANCE.md](PERFORMANCE.md) and close the MVP questions in
[MVP_PLAN.md](MVP_PLAN.md).
