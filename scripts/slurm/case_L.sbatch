#!/bin/bash
# Case L (docs/MVP_PLAN.md): synthesise a 2048^3 or 4096^3 u16 pair with an inclusion field,
# then run it on one 8x H100 node and record Q2/Q3.
#
#   sbatch scripts/slurm/case_L.sbatch 2048 /lustre/project/pydvc/caseL2048
#   sbatch scripts/slurm/case_L.sbatch 4096 /lustre/project/pydvc/caseL4096
#
# Generation runs on every core of the node (the phantom is block-parallel and
# reproducible whatever the chunking): ~16 GB per volume at 2048^3, ~137 GB at 4096^3.
#SBATCH --job-name=pydvc-caseL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
N=${1:?usage: case_L.sbatch <2048|4096> <output dir>}
OUT=${2:?usage: case_L.sbatch <2048|4096> <output dir>}

if [ ! -f "$OUT/config.yaml" ]; then
  pydvc synth --shape "$N" "$N" "$N" --field inclusion --spacing 16 --out "$OUT" \
      --subvol-size 48 --subvol-npts 4096 --dof 6 --disp-max 10 --chunk 128 --shard 1024
fi
CONFIG=$OUT/config.yaml
# 1024^3 tiles over 256^3 point cells, rigid seeds (the inclusion field stays within ~3 voxels)
python - "$CONFIG" <<'PY'
import sys, dataclasses
from pydvc.config import RunConfig, ClusterSpec, SeedingSpec
p = sys.argv[1]; c = RunConfig.from_yaml(p)
c = dataclasses.replace(c, cluster=ClusterSpec(tile_shape=(1024, 1024, 1024), prefetch_depth=2),
                        seeding=SeedingSpec(strategy="rigid"))
c.to_yaml(p)
PY

pydvc plan "$CONFIG" --backend fused
scripts/slurm/storage_baseline.sh "$OUT" "$CONFIG"
python -m pydvc.bench.throughput --backend fused --samples 4096 --dof 6 12 --size 48 --batch 32768   # Q2
python -m pydvc.bench.scaling "$CONFIG" --devices 1 2 4 8 --backend fused --out "$OUT/scaling"       # Q3
/usr/bin/time -v pydvc run "$CONFIG" --backend fused                                                 # end to end
pydvc finalize "$CONFIG"
pydvc compare "$OUT/results.zarrvectors" "$OUT/truth.npz"
