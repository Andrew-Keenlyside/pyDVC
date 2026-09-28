#!/bin/bash
# zvDVC on one or more 8x H100 nodes: plan -> seed -> run -> finalize.
#
#   sbatch scripts/slurm/zvdvc_8xh100.sbatch configs/large_8xh100.yaml
#   sbatch --nodes=4 scripts/slurm/zvdvc_8xh100.sbatch configs/large_8xh100.yaml   # multi-node: untested (M5)
#
# One task per node. `zvdvc run` spawns one process per local GPU with a shared
# tile queue; nodes split tiles by the LPT assignment in plan.json (SLURM_NODEID).
# Every stage is re-runnable: `run` skips tiles already written, so resubmitting
# after a time-out or a lost worker solves only the missing tiles.
#SBATCH --job-name=zvdvc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch zvdvc_8xh100.sbatch <config.yaml>}
BACKEND=${ZVDVC_BACKEND:-fused}

# Optional: stage inputs to node-local NVMe when the parallel file system is the bottleneck
# (see docs/ARCHITECTURE.md, I/O). Point the config at $LOCAL_SCRATCH copies if you do.

# CUDA/zarr-vectors knobs
export ZARR_VECTORS_GPU_ENCODE=1
# export ZARR_VECTORS_GPU_IO=kvikio   # only with GPUDirect Storage configured (KVIKIO_COMPAT_MODE=OFF)

srun --nodes=1 --ntasks=1 zvdvc plan     "$CONFIG" --backend "$BACKEND"
srun --nodes=1 --ntasks=1 zvdvc seed     "$CONFIG" --backend "$BACKEND"
srun                      zvdvc run      "$CONFIG" --backend "$BACKEND"
# `zvdvc repair` (re-seed failed points from good neighbours) arrives with M5.
srun --nodes=1 --ntasks=1 zvdvc finalize "$CONFIG" --disp
