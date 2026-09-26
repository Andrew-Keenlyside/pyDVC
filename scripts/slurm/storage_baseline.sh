#!/bin/bash
# Storage bandwidth baseline for Q3 (docs/MVP_PLAN.md): what the node's file system delivers,
# measured two ways on the directory that holds the OME-Zarr volumes.
#
#   scripts/slurm/storage_baseline.sh /lustre/project/scan  configs/large_8xh100.yaml
#
# 1. fio: large sequential reads with 8 and 32 concurrent jobs (the file system's ceiling).
# 2. pyDVC's own brick reads (host decode, the path the workers use) with 8 and 32 threads,
#    over the tiles in plan.json (run `pydvc plan CONFIG` first).
# Drop the page cache between runs if you can (sudo sysctl vm.drop_caches=3), or use files
# larger than node memory; otherwise the numbers are cache bandwidth.
set -euo pipefail
DIR=${1:?usage: storage_baseline.sh <data dir> <config.yaml>}
CONFIG=${2:?usage: storage_baseline.sh <data dir> <config.yaml>}
OUT=${OUT:-storage_baseline_$(hostname)_$(date +%Y%m%d-%H%M).txt}

{
  echo "# $(date -Is) $(hostname) $DIR"
  if command -v fio >/dev/null; then
    for jobs in 8 32; do
      fio --name=seqread --directory="$DIR/.fio" --rw=read --bs=4M --size=8G --numjobs="$jobs" \
          --ioengine=libaio --direct=1 --group_reporting --output-format=terse --terse-version=3 \
          | awk -F';' -v j="$jobs" '{printf "fio seq read, %d jobs: %.2f GB/s\n", j, $7/1e6}'
    done
    rm -rf "$DIR/.fio"
  else
    echo "fio not installed: skipping the raw file-system measurement"
  fi
  for readers in 8 32; do
    python -m pydvc.bench.scaling "$CONFIG" --read-bandwidth --readers "$readers"
  done
} | tee "$OUT"
