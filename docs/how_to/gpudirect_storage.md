# GPUDirect Storage

This page explains where GPUDirect Storage could speed up zvDVC, how
zarr-vectors chooses its GPU read path, and what the planned benchmark will
measure. It is for HPC users and anyone evaluating Zarr Vectors' GPU backend.

```{important}
**Status: GPUDirect Storage is not wired into zvDVC and has not been
measured.** Every published zvDVC timing uses host reads: host decode into
pinned memory, then one host-to-device copy. The device read paths, and the
host-versus-GDS benchmark, are roadmap milestone **M6**. Nothing on this page
reports a GDS speed-up.
```

---

## What GDS is

**GPUDirect Storage (GDS)** is NVIDIA's path for moving file data between
storage (typically local NVMe) and GPU memory by direct memory access,
without staging the bytes in a host-memory buffer first. Without it, a read
lands in host memory and is then copied to the GPU.

[kvikio](https://github.com/rapidsai/kvikio) is the Python library
zarr-vectors uses for these reads. It uses GDS when the system supports it,
and otherwise can fall back to a *compatibility mode* that reads through a
host bounce buffer.

## Where it would apply in zvDVC

zvDVC reads three kinds of data:

| Data | Store | Read path today | GDS path (M6) |
|---|---|---|---|
| Search points, per tile | zarr-vectors | `read_cells`, decoded on the host | `read_cells(device="cuda")` over kvikio |
| Results, whole cloud (repair, strain, export) | zarr-vectors | host reads | the same calls with `device="cuda"` |
| Image bricks (tile + halo) | OME-Zarr | host decode → pinned buffer → one copy to the device | kvikio / GDS, or zarr-python GPU buffers |

**Per tile, image bricks dominate.** A 1024³ u16 tile reads about 5 GB of
bricks (reference and deformed, each grown by the halo) but only a few MB of
points. End-to-end gains from GDS therefore need the brick path, which is
zvDVC's own OME-Zarr reader, as well as zarr-vectors' device reads.

**Whole-cloud passes are where zarr-vectors alone decides the time.**
Repair, strain and export read every result, with its neighbours. At the
planned large scale (8.4 million points, scenario B in {doc}`/PERFORMANCE`)
that is several hundred MB of results.

### What the code does today

* **Image bricks.**
  [`io/volume.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/io/volume.py)
  lists three device paths in order of preference: kvikio / GDS,
  zarr-python's GPU buffers, and host decode. Only host decode is
  implemented: zarr-python decodes the brick's chunks on the host, the brick
  is placed in a pinned buffer, and it is copied to the GPU in one transfer.
* **Points.** The worker
  ([`pipeline/worker.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/pipeline/worker.py))
  reads each tile's points to the host, because seeds are looked up there
  and the points are a few MB per tile. `PointCloud.read_tile` calls
  zarr-vectors' `read_cells` as a host read.
* **Results.** Whole-store reads (finalize, export) are host reads. The
  neighbourhood read that repair and strain across tiles would use
  (`ResultStore.read_neighbourhood`) is M5 work.

So zarr-vectors' GPU read settings below **do not change any zvDVC read
today**.

## How zarr-vectors picks its GPU read path

This describes zarr-vectors-py at the commit zvDVC pins
(`06a3bc8a08afefc02cd1ffb265d67a03df798b87`; `zarr_vectors/gpu/_fetch.py`).

When a caller asks `read_cells(..., device="cuda")`, zarr-vectors fetches
the stored bytes of each cell into device memory and decodes them there, for
every array whose on-disk layout it recognises. Uncompressed cells are
decoded on the device by default; zstd cells are decompressed on the device
with nvCOMP only with `decode="device"`. nvCOMP trusts its input, so zvDVC's
design uses that option only on stores zvDVC wrote itself
({doc}`/ARCHITECTURE`, section 5). Anything else is read and
decoded on the host, then uploaded.

For local files, the fetch picks one of two ways to read:

| mode | what happens |
|---|---|
| `kvikio` | each cell is read with kvikio straight into one device buffer (GDS where the system has it) |
| `host` | plain reads on a thread pool into pinned host memory, then **one** host-to-device copy for all cells |

Stores that are not local files (fsspec, obstore, icechunk, memory) are read
through the store's own byte-range requests, then copied up once.

The choice comes from the environment variable **`ZARR_VECTORS_GPU_IO`**:

| `ZARR_VECTORS_GPU_IO` | result |
|---|---|
| `kvikio` | always kvikio |
| `host` | always the pinned host path |
| `auto`, or unset | `host` if kvikio is not installed. If it is installed: `kvikio` only when kvikio's compatibility mode is **OFF** (`KVIKIO_COMPAT_MODE=OFF`), otherwise `host` |

The reason for the last rule: in compatibility mode kvikio reads through a
bounce buffer, one cell at a time, which the pinned host path does in one
copy instead. So kvikio is chosen automatically only when GDS has been asked
for explicitly.

To see what an installation supports:

```python
import zarr_vectors as zv

caps = zv.runtime_capabilities(probe_device=True)
print(caps["device_arrays"], caps["gpu_codecs"], caps["gpu_io"])
```

`gpu_io` is true when kvikio imports alongside zarr-vectors' GPU extension;
`gpu_codecs` when nvCOMP does. `gpu_io` says that kvikio is present, not that
the system has GDS.

## Preparing for M6

```{note}
The steps below prepare a machine for the M6 measurements. They do not make
zvDVC faster today, because zvDVC does not yet request device reads.
```

1. **Install kvikio.** The `[gpu-io]` extra installs `zarr-vectors[gpu-io]`,
   which brings `kvikio-cu12`:

   ```bash
   pip install -e ".[gpu-io]"
   ```

   That extra also depends on the pip `cupy-cuda12x` wheel. In the conda GPU
   environment, which already has CuPy from conda, this replaces conda's CuPy;
   see the "one CuPy only" pitfall in {doc}`/getting_started/installation`.

2. **Check the platform.** GDS needs NVIDIA's GDS driver stack and a
   supported file system, typically local NVMe. NVIDIA's `gdscheck` tool,
   part of the GDS installation, reports whether a node supports it.

3. **Select the path explicitly** for a measurement:

   ```bash
   export ZARR_VECTORS_GPU_IO=kvikio      # or host, for the baseline
   export KVIKIO_COMPAT_MODE=OFF           # GDS, not kvikio's bounce buffer
   ```

4. **Record the host baseline first.** The tools that exist today measure
   the host path:

   * `scripts/slurm/storage_baseline.sh DATA_DIR CONFIG`: the file system's
     sequential-read ceiling with fio, and zvDVC's own brick-read bandwidth
     (`python -m zvdvc.bench.scaling CONFIG --read-bandwidth`);
   * `zvdvc run` prints each device's bytes read, compute time and I/O wait,
     and writes them to `run_stats.json`.

## What M6 will measure

M6 adds zarr-vectors device reads for points and results and kvikio reads for
image bricks in zvDVC, then measures both paths, host and GDS, on node-local
NVMe on an 8 × H100 node, **per stage and end to end**. The questions it
answers:

* **Bricks:** read bandwidth per GPU and the compute loop's I/O wait, host
  against GDS. The MVP's data-path target is that compute waits on I/O at
  most 20 % of the time ({doc}`/MVP_PLAN`, Q4).
* **Points and results:** the time of zarr-vectors' `read_cells` with
  `device="cpu"` against `device="cuda"` over kvikio, per tile and for the
  whole-cloud passes.
* **End to end:** whether GDS changes a full `plan → seed → run → finalize`
  run at scale, and by how much.

Results will be published in {doc}`/benchmarks/index` with the hardware,
file system and settings they were measured on.
