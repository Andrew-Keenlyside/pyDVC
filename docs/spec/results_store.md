# Results store

This page specifies the results store: the zarr-vectors point cloud in which
the tiled pipeline writes one row per search point. It lists every attribute,
the root metadata, the three-phase write protocol, how an interrupted run
resumes, what `zvdvc finalize` does, and the `.disp`/`.stat` exports, with an
example of reading results. It is for users analysing results and for anyone
writing tools that read the store. Source:
[`io/results.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/io/results.py),
[`pipeline/coordinator.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/pipeline/coordinator.py).

## Terms

Results store
: The zarr-vectors point cloud named by `output` in the run configuration,
  conventionally `results.zarrvectors`. Written by `zvdvc seed` (wavefront
  mode) and `zvdvc run`; completed by `zvdvc finalize`. The in-memory
  `zvdvc solve` writes a `.npz` instead.

Vertex attribute
: A per-point array stored alongside the vertices, one row per point, with a
  fixed dtype and number of columns.

Written cell
: A cell that every required result array holds. Only written cells count as
  done.

Presence
: zarr-vectors' record of which cells hold data. The pipeline defers it while
  workers write and rebuilds it at the end.

Template digest
: `Template.digest()`, a 16-character hash of the sample offsets
  ({doc}`method`).

---

## Introduction

The results store mirrors the search-point store cell for cell and fragment
for fragment: row $r$ of cell $c$ in the input is row $r$ of cell $c$ here.
No index table is needed, and a tile's writes touch only that tile's cells, so
workers write in parallel without locks.

A run creates the store in `zvdvc plan`, before any point is solved, with
every array allocated and presence deferred. Workers then write whole cells,
with the `status` array last. A cell counts as written only when every
required array holds it, so a worker killed in the middle of a cell leaves
nothing that looks finished, and a resubmitted `zvdvc run` solves exactly the
tiles still missing. `zvdvc finalize` rebuilds presence and the metadata, and
writes the summaries.

---

## Technical reference

### Layout

```text
results.zarrvectors/
  zarr.json                        root: zarr_vectors metadata, multiscales, ome,
                                   and zvDVC's zvdvc_results
  0/                               resolution level 0 (same grid as the points)
    vertices/                      (x, y, z) float32: the search point positions
    vertex_fragments/
    vertex_attributes/
      point_id/  status/  objmin/  displacement/  params/  n_iter/  seed/
      displacement_sd/             optional, see below
  1/ 2/                            only after `zvdvc finalize --pyramid`
```

Bounds, `chunk_shape` and `bin_shape` are copied from the search-point store
({doc}`points_store`).

### Vertex attributes

Required attributes (`RESULT_ATTRIBUTES`):

| name | dtype | columns | meaning |
|---|---|---|---|
| `point_id` | int64 | 1 | CCPi label, from the search-point store |
| `status` | int8 | 1 | {py:class}`zvdvc.status.PointStatus` code ({doc}`status_codes`). Written last in each cell. |
| `objmin` | float32 | 1 | Objective at the solution (CCPi `objmin`). The exact objective at the final parameters for `GOOD` and `CONVG_FAIL`; for other statuses the last value computed, not meaningful, or NaN if never evaluated. |
| `displacement` | float32 | 3 | $(u, v, w)$ in voxels: the first three entries of `params` |
| `params` | float32 | `dof` | Full parameter vector in CCPi order: `u v w`, then `phi the psi` (radians), then `exx eyy ezz exy eyz exz` ({doc}`method`). The column count is fixed when the store is allocated. |
| `n_iter` | uint8 | 1 | Gauss–Newton steps taken |
| `seed` | float32 | 3 | Starting displacement $(u, v, w)$, for audit |

Values are stored as solved, whatever the status: a `RANGE_FAIL` point keeps
the parameters it had when it failed. Filter on `status == 0` before using
displacements.

Optional attributes (`OPTIONAL_ATTRIBUTES`), written when present in a
tile's results and filled with NaN otherwise:

| name | dtype | columns | meaning |
|---|---|---|---|
| `displacement_sd` | float32 | 3 | Per-axis standard deviation of the displacement over repeat solves with other template seeds (`uncertainty_seeds` $> 0$); NaN when not computed or when fewer than two solves were `GOOD` ({doc}`method`, "Displacement uncertainty") |

Stores created before an optional attribute existed simply lack it; a cell
counts as written without optional attributes. Strain is not stored here;
`zvdvc strain` writes CSV and `.npz` files ({doc}`method`, "Strain").

### Root attribute `zvdvc_results`

The run's layout, written by `ResultStore.allocate` into the root group's
attributes:

| field | type | meaning |
|---|---|---|
| `dof` | int | 3, 6 or 12: the number of `params` columns |
| `chunk_shape` | 3 floats | cell size, $(x, y, z)$ voxels (from the point store) |
| `bin_shape` | 3 floats | bin size, $(x, y, z)$ voxels (from the point store) |
| `points` | string | absolute path of the search-point store the rows come from |
| `n_points` | int | points in that store |
| `template_digest` | string or null | digest of the sample template the results were computed with |
| `prefilter_sigma` | float | `volumes.prefilter_sigma` of the run |
| `optional_attributes` | list of strings | optional attributes this store has (e.g. `["displacement_sd"]`) |

A store written before the project was renamed from pyDVC carries the same
object under the key `pydvc_results`; `ResultStore` still reads it. A store
with neither key is refused (`ValueError`: not a zvDVC results store). The
root also holds zarr-vectors' own `zarr_vectors`, `multiscales` and `ome`
metadata, which zarr-vectors manages.

### Consistency checks on resume

`zvdvc plan` on an existing output store refuses to continue (`ValueError`)
when:

* the store's `dof` or `chunk_shape` differs from the run's;
* the store's `template_digest` differs from the run's template (the
  subvolume settings, or zvDVC's template construction, changed); a store with
  no digest only gives a warning, since the check cannot be made;
* the store's `prefilter_sigma` differs from `volumes.prefilter_sigma`.

Each worker repeats the template and prefilter checks when it opens the
store. `plan.json` also records the template digest, the config and the
package versions (including zarr-vectors).

### Three-phase write

Writes follow zarr-vectors' HPC pattern:

1. **Allocate** (`ResultStore.allocate`, coordinator, once, in `zvdvc plan`):
   `create_store`, `create_resolution_level`, `create_vertices_array`, one
   `create_attribute_array` per required and optional attribute, then
   `defer_presence`, then the `zvdvc_results` attribute. Workers never create
   arrays: re-creating one inside a write session would drop cells other
   workers have already written.
2. **Write** (`ResultStore.write_tile`, workers, in parallel): for each of the
   tile's cells, `write_chunk_vertices` and one `write_chunk_attributes` per
   attribute, with `record_presence=False` and the rows split into the same
   fragments as the input. Attributes are written in the order `point_id`,
   `objmin`, `displacement`, `params`, `n_iter`, `seed`, the optional
   attributes, and **`status` last**. `write_tile` refuses results that lack a
   required attribute. Cells written on a deferred level are visible to readers
   straight away.
3. **Finalise** (`ResultStore.finalize`, coordinator, once, in
   `zvdvc finalize`): `rebuild_presence`, `refresh_arrays_present`,
   `update_level_metadata(vertex_count=N)`, `write_multiscale_metadata`, and
   with `--pyramid`, `build_pyramid`. Never run it while workers are writing.

### Written cells and resume

`ResultStore.written_cells()` lists the chunk keys of `vertices` and of every
required attribute array and returns their intersection. Because `status` is
written last, a cell interrupted mid-write is missing from at least the
`status` array and does not count.

* A worker skips a tile whose cells are all written.
* A tile that raises is requeued once, then recorded as failed.
* After `zvdvc run`, tiles of this node still not written are listed in
  `<workdir>/failed_tiles.json` (with the errors) and in
  `<workdir>/run_stats.json` (`missing_tiles`); resubmitting `zvdvc run`
  solves only those. On a multi-node run the file names carry a
  `.node<rank>` suffix.
* A crash loses at most the tiles a worker had taken: the one being solved and
  up to `prefetch_depth` prefetched. In a test, a `kill -9` of one of two
  workers left 4 of 8 tiles for the resubmitted job, which then wrote a
  bit-identical store ({doc}`/ARCHITECTURE` §9).

### `zvdvc finalize`

```bash
zvdvc finalize CONFIG [--disp] [--pyramid]
```

1. Reads every written cell (`read_all`) and finalises the store (above).
2. Writes `<workdir>/results.stat`.
3. With `--disp`, writes `<workdir>/results.disp`, rows sorted by `point_id`.
4. With `--pyramid`, builds coarser levels `1` and `2` for viewers with
   zarr-vectors' `Dataset.build_pyramid(factors=[(2.0, 1.0), (2.0, 1.0)], chunk_scale_factors=[2, 2])`.

### Exports

**`.disp`** (`zvdvc.io.ccpi.write_disp`): tab-separated, header
`n x y z status objmin u v w`, one row per point, numbers in `%.9g`. Status
codes below −3 (zvDVC-only) are written as −3, `NOT_SEARCHED`
({doc}`status_codes`). Displacements and `objmin` are written as stored (a
NaN `objmin` prints as `nan`). iDVC and CCPi's `strain` program read this
format. The `zvdvc-dvc` drop-in writes a stricter, byte-for-byte CCPi layout
({doc}`ccpi_compat`).

**`.stat`** (`zvdvc.io.ccpi.write_stat`): a zvDVC run summary, not CCPi's
layout:

```text
zvdvc run summary
points	27
seconds	1.111
points_per_second	24.302
status GOOD	27

### config
volumes:
  reference: ...
```

One `status <NAME>\t<count>` line per status present, with zvDVC's names
(`GOOD`, `RANGE_FAIL`, `CONVG_FAIL`, `NOT_SEARCHED`, `THRESH_FAIL`,
`SINGULAR`), then the full configuration as YAML. `seconds` comes from
`<workdir>/run_stats.json`.

### Reading results

With zvDVC, {py:class}`zvdvc.io.results.ResultStore` returns every written
point as numpy arrays, in cell order:

```python
import numpy as np
from zvdvc.io.results import ResultStore

store = ResultStore("results.zarrvectors")          # read-only by default
print(store.dof, store.chunk_shape, store.meta["template_digest"])
r = store.read_all()        # xyz, point_id, status, objmin, displacement, params, n_iter, seed
                            # (+ displacement_sd when the store has it)
good = r["status"] == 0
u_good = r["displacement"][good]
order = np.argsort(r["point_id"])                  # CCPi order, if you need it
```

With zarr-vectors alone, read the cells directly:

```python
import numpy as np
import zarr
from zarr_vectors import building as zb

root = zb.open_store("results.zarrvectors", mode="r")
level = zb.get_resolution_level(root, 0)
cells = np.asarray(sorted(tuple(int(v) for v in c) for c in zb.list_chunk_keys(level)), dtype=np.int64)
batch = zb.read_cells(level, cells, ["vertices", "vertex_attributes/point_id",
                                     "vertex_attributes/status", "vertex_attributes/displacement"])
xyz = np.asarray(batch["vertices"].data).reshape(-1, 3)
status = np.asarray(batch["vertex_attributes/status"].data).reshape(-1)
disp = np.asarray(batch["vertex_attributes/displacement"].data).reshape(-1, 3)
offsets = np.asarray(batch["vertices"].offsets)    # rows of cell i: offsets[i]:offsets[i + 1]
dof = zarr.open_group("results.zarrvectors", mode="r").attrs["zvdvc_results"]["dof"]
```

Both snippets were run on a 96³ synthetic case (27 points, 8 cells) and
return the same values. Before `zvdvc finalize`, prefer
`ResultStore.read_all()`: it reads only cells that are completely written.
`zvdvc compare` and `zvdvc strain` accept a results store path directly.
