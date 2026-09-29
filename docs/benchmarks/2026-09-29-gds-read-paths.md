# Brick read paths: host decode against kvikio, before GPUDirect Storage is enabled

Date: 2026-09-29. Workstation `msm12`: 32 cores, NVIDIA RTX A2000 12 GB (BAR1
256 MiB), Samsung 990 PRO 2 TB NVMe (ext4). kvikio 26.08, cuFile 1.14,
nvCOMP 5.3. zvDVC `develop` with the new read path ({doc}`/how_to/gpudirect_storage`).

**GPUDirect Storage was not available** for these measurements. The machine
has no `nvidia-fs` driver, and its IOMMU is on. cuFile reports
`is_gds_available = False` and falls back to POSIX reads through a bounce
buffer. So these numbers compare the host path with **kvikio in compatibility
mode**: the baseline that a GDS measurement will be compared against, not a
GDS result.

## What was measured

`python -m zvdvc.bench.gds` read six random 512³ bricks (134 MB of u8 each),
twice each, from case A's reference scan:

* **raw**: the flat `.raw` file (1520 × 1257 × 1260 u8, 2.4 GB), on the NVMe.
* **OME-Zarr**: the same scan converted with `zvdvc convert` (128³ chunks,
  512³ shards, zstd level 3; 1.6 GB), on the NVMe.

Two read paths:

* **host**: decode on the host (numpy memmap, or zarr-python with CPU zstd)
  into pinned memory, then one copy to the GPU.
* **kvikio**: stored bytes read into GPU memory by kvikio (16 I/O threads),
  zstd chunks decompressed on the GPU by nvCOMP.

"Cold" drops the files from the page cache before every read, so the bytes
come from the drive. Both paths delivered **identical bricks** in every case.

## Results

| Volume | Cache | host | kvikio (no GDS) | kvikio ÷ host |
|---|---|---|---|---|
| raw u8 | warm | 2.45 GB/s | 4.90 GB/s | 2.0× |
| raw u8 | cold | 2.52 GB/s | 1.64 GB/s | 0.65× |
| OME-Zarr, zstd | warm | 0.90 GB/s | 1.84 GB/s | 2.0× |
| OME-Zarr, zstd | cold | 0.68 GB/s | 1.48 GB/s | 2.2× |

Medians over 12 reads per row.

## What this shows

* **Compressed OME-Zarr, zvDVC's production format, is already twice as fast
  on the kvikio path, without GPUDirect Storage.** The gain is nvCOMP
  decompressing zstd on the GPU instead of the CPU.
* **Raw files from a warm cache are twice as fast** on the kvikio path: its
  parallel reads beat the host path's single-threaded copy into pinned memory.
* **Raw files from a cold cache are slower on the kvikio path** (1.64 against
  2.52 GB/s). Without `nvidia-fs`, every byte goes drive → bounce buffer →
  GPU, and the per-plane reads lose the kernel readahead the host path's
  memory map gets. This is the case GPUDirect Storage exists for: with it,
  the bytes go straight from the drive to the GPU.
* That is why `volumes.gpu_io: auto` uses the kvikio path only when cuFile
  reports GPUDirect Storage available. On this machine today, `auto` means
  `host`; set `gpu_io: kvikio` to use the kvikio path for OME-Zarr now.

## Not yet measured

* **GPUDirect Storage itself.** It needs the `nvidia-fs` driver and the IOMMU
  off (root, then a reboot). Then rerun the commands below: the report's
  `gpu_io_status.gds_available` field must be true.
* **Whole runs.** These are brick reads alone. A tiled run's I/O wait (`zvdvc
  run`'s per-worker `I/O wait`, `run_stats.json`) with each path is the next
  step, then the 8 × H100 nodes, whose BAR1 is not limited to 256 MiB.

## Reproduce

```bash
zvdvc convert runs/case_A/dataset_0_zyx.raw runs/gds/ref.ome.zarr --shape-xyz 1520 1257 1260 --dtype '|u1' \
    --chunk 128 --shard 512
python -m zvdvc.bench.gds --config runs/case_A/config.yaml --brick 512 --bricks 6 --repeat 2 [--cold]
python -m zvdvc.bench.gds --volume runs/gds/ref.ome.zarr   --brick 512 --bricks 6 --repeat 2 [--cold]
```

The JSON reports are in `runs/gds/`.
