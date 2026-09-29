# GPUDirect Storage

This page explains how zvDVC reads image bricks straight into GPU memory, how
to turn GPUDirect Storage on and check that it is really on, and how to measure
it. It is for HPC users and anyone evaluating GPU storage paths for DVC.

```{important}
**Status (2026-09-29).** The device read path is implemented and tested: bricks
read on it are bit-identical to host reads. It has been measured **without**
GPUDirect Storage (kvikio's compatibility mode), where it is already about twice
as fast for compressed OME-Zarr ({doc}`/benchmarks/2026-09-29-gds-read-paths`).
Measurements **with** GPUDirect Storage need a machine where it is enabled; none
has been made yet.
```

---

## What GDS is

**GPUDirect Storage (GDS)** moves file data between storage (typically local
NVMe) and GPU memory by direct memory access, without staging it in host
memory first. It is part of NVIDIA's cuFile library and needs the `nvidia-fs`
kernel driver and a supported filesystem (ext4 or XFS on local NVMe, or
specific parallel filesystems).

[kvikio](https://github.com/rapidsai/kvikio) is the Python library zvDVC uses
for these reads. It calls cuFile, which uses GDS when the system supports it
and otherwise reads through a host bounce buffer ("compatibility mode"). The
bytes delivered are the same either way.

## Read paths

`volumes.gpu_io` in the run configuration ({doc}`/spec/run_config`) chooses how
bricks reach the GPU:

| `gpu_io` | Bricks are |
|---|---|
| `host` | decoded on the host (a memory map for flat files; zarr-python with CPU zstd for OME-Zarr) into pinned memory, then copied to the GPU in one transfer |
| `kvikio` | read by kvikio from the file straight into GPU memory (through GDS when available); zstd chunks are decompressed on the GPU by nvCOMP; edge padding and the NaN check run on the GPU |
| `auto` (default) | `kvikio` when cuFile reports GDS available, otherwise `host`. `$ZVDVC_GPU_IO` (`host` or `kvikio`) overrides `auto` |

The read path never changes results: it is not part of the run fingerprint, so
a run can be resumed with a different `gpu_io`.

The kvikio path handles:

* flat files: `.raw`, `.mhd` and C-ordered `.npy`, in either byte order and
  with any header;
* OME-Zarr / Zarr v3 arrays on a local directory, sharded with the index at
  the end, stored as raw bytes or zstd. `zvdvc convert` and `zvdvc synth`
  write exactly this layout.

Anything else falls back to the host path with a reason: a Fortran-ordered
`.npy`, an unsharded Zarr array, a remote store, other codecs, or no nvCOMP for
zstd. Unwritten Zarr chunks read as the fill value, as on the host path.

nvCOMP trusts its input: a malformed zstd chunk can hang its kernel or fail the
CUDA context where the host decoder raises. The kvikio path therefore checks
each shard index's crc32c, requires every read to return all the bytes asked
for, and decodes through zarr-vectors' checked `decode_zstd` when the installed
zarr-vectors offers it (otherwise each frame header must declare the chunk's
size). Damage inside a compressed block passes all of these, so use the kvikio
path on stores you trust.

### Knowing which path ran

"Is GDS on?" has a misleading answer in kvikio: its
`defaults.is_compat_mode_preferred()` is False whenever libcufile loads, even
when cuFile then falls back to POSIX because `nvidia-fs` is missing. zvDVC
therefore asks cuFile itself (`kvikio.cufile_driver.properties.is_gds_available`)
and records the result:

* `zvdvc.io.gds.status()` returns kvikio, cuFile and nvCOMP versions,
  `gds_available`, and why not;
* each volume source has `.io` (`host` or `kvikio`) and `.io_note` (why it fell
  back);
* each worker's `worker_start` event and its `run_stats.json` entry record the
  read path its bricks took (`io`: `host`, `kvikio` or `mixed`);
* each tile's `read` event also records how zarr-vectors served the tile's
  points (`points_io`), when the installed zarr-vectors reports it.

### Points and results

Search points and results stay on the host read path. A tile's points are a
few MB against gigabytes of bricks, and each cell's bins are worked out on the
host. With a zarr-vectors that returns each cell's fragment index
(`read-cells-fragments`), those bins come from the store instead of being
recomputed from the coordinates. zarr-vectors has its own device read for cells
(kvikio and nvCOMP), which zvDVC does not use.

## Enabling GPUDirect Storage

On the machine (needs root):

1. Install NVIDIA's GDS packages matching the driver (`nvidia-gds`, which
   brings the `nvidia-fs` kernel module) from NVIDIA's CUDA repository, and
   load the module (`lsmod | grep nvidia_fs`).
2. Turn the IOMMU off, or to pass-through, as NVIDIA's GDS installation guide
   recommends for the platform, then reboot.
3. Put the data on a local NVMe with ext4 or XFS. FUSE mounts (for example
   NTFS through `ntfs-3g`) and most network filesystems cannot use GDS. For
   ext4, cuFile looks for `data=ordered` in the mount table: it is ext4's
   default mode but is not listed unless given, and cuFile then logs `mount
   option not found in mount table` (in `cufile.log`) and falls back. Add
   `data=ordered` to the filesystem's options in `/etc/fstab` (a root
   filesystem needs a reboot to pick it up).
4. Check with `gdscheck -p` (in the CUDA toolkit's `gds/tools`): the NVMe line
   must read `Supported`.

In the environment:

```bash
conda install -n zvdvc-gpu -c rapidsai -c conda-forge kvikio   # conda: not pip, which pins its own cupy
pip install nvidia-nvcomp-cu12                                 # GPU zstd decoding
python -c "from zvdvc.io import gds; print(gds.status())"       # gds_available must be True
```

kvikio uses a pool of 16 I/O threads unless `KVIKIO_NTHREADS` is set; its own
default, one thread, serialises the reads.

On a cluster, run `gdscheck -p` on a GPU node before planning GDS runs.
Network project storage rarely supports GDS; stage the volumes to node-local
NVMe instead. Inside Apptainer, the `nvidia-fs` device nodes must be available
in the container as well as `--nv`.

## Measuring it

```bash
python -m zvdvc.bench.gds --volume data/ref.ome.zarr --brick 512 --bricks 8 --cold --json runs/gds/ref.json
python -m zvdvc.bench.gds --config my_run.yaml --brick 512 --bricks 8 --cold
```

This reads the same random bricks on the host and kvikio paths, checks they are
identical, and reports GB/s. `--cold` drops the files from the page cache
before every read, so the bytes come from the drive. The first line of the
output, and `gpu_io_status` in the JSON, say whether GDS was available: without
it, the kvikio numbers measure the compatibility mode.

For whole runs, compare `zvdvc run` with `gpu_io: host` and `gpu_io: kvikio`:
each worker's `I/O wait` and `run_stats.json` show how long the compute loop
waited for bricks.

## Limits on the development workstation

The RTX A2000's BAR1 window is 256 MiB. GDS moves data into GPU memory through
that window, so large reads are split, which caps the gain on this card. The
8 × H100 nodes do not have this limit.
