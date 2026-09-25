#!/bin/bash
# pyDVC on one or more 8x H100 nodes: plan -> seed -> run -> repair -> finalize.
#
#   sbatch scripts/slurm/pydvc_8xh100.sbatch configs/large_8xh100.yaml
#   sbatch --nodes=4 scripts/slurm/pydvc_8xh100.sbatch configs/large_8xh100.yaml   # multi-node (M5)
#
# One task per node. `pydvc run` spawns one process per local GPU with a shared
# tile queue, and nodes split tiles by the LPT assignment in plan.json.
# Every stage is re-runnable; `run` skips tiles already written, so resubmitting
# after a time-out resumes the job.
#SBATCH --job-name=pydvc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch pydvc_8xh100.sbatch <config.yaml>}

# Optional: stage inputs to node-local NVMe when the parallel file system is the bottleneck
# (see docs/ARCHITECTURE.md, I/O). Point the config at $LOCAL_SCRATCH copies if you do.

# CUDA/zarr-vectors knobs
export ZARR_VECTORS_GPU_ENCODE=1
# export ZARR_VECTORS_GPU_IO=kvikio   # only with GPUDirect Storage configured (KVIKIO_COMPAT_MODE=OFF)

srun --nodes=1 --ntasks=1 pydvc plan     "$CONFIG"
srun --nodes=1 --ntasks=1 pydvc seed     "$CONFIG"
srun                      pydvc run      "$CONFIG"
srun                      pydvc repair   "$CONFIG"
srun --nodes=1 --ntasks=1 pydvc finalize "$CONFIG" --disp
