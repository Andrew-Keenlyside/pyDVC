# Search-point store

This page specifies how zvDVC stores its search points: a Zarr Vectors point
cloud whose spatial chunk grid is also zvDVC's work partition. It covers
importing CCPi/iDVC point files, the chunk and bin shapes, the fragment
numbering, the `point_id` attribute, how tiles map to cells, and how tiles are
read. It is for users preparing large point clouds and for anyone reading or
writing these stores with other tools. Source:
[`io/pointcloud.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/io/pointcloud.py),
[`pipeline/tiling.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/pipeline/tiling.py);
design notes in {doc}`/ARCHITECTURE` §5.

## Terms

Zarr Vectors
: A format for vector geometry (points, lines, meshes, skeletons) on a Zarr v3
  store, cut into a spatial chunk grid
  ([specification](https://github.com/AllenInstitute/zarr_vectors) by Forest
  Collman, Allen Institute; implementation
  [zarr-vectors-py](https://github.com/AllenInstitute/zarr-vectors-py)). It does
  not store dense images.

Cell (chunk)
: One chunk of the spatial grid, identified by integer coordinates
  $(c_x, c_y, c_z)$. The atomic unit of storage and of writing.

`chunk_shape`
: The size of a cell in voxels, $(x, y, z)$. May be fractional.

Bin, fragment
: A cell is divided into bins of `bin_shape`; the points of one bin are stored
  together as one fragment. zvDVC uses two bins per axis: eight fragments per
  cell.

Tile
: A block of whole cells, the unit of work handed to one GPU.

`point_id`
: The point's label: the `n` column of a CCPi `.roi` or `.disp`.

---

## Introduction

A search-point store holds the positions and labels of the points to
correlate. zvDVC uses its chunk grid to split the work: a tile is a block of
whole cells, the results store has exactly the same cells and fragments
({doc}`results_store`), and so the cells a worker reads are the cells it later
writes, with no other worker touching them.

You rarely create a store by hand. If `points` in the run configuration names
a CCPi `.roi` (or an iDVC `.txt`/`.csv`), `zvdvc plan` imports it into
`<workdir>/points.zarrvectors`, with a chunk shape derived from
`cluster.tile_shape`. `zvdvc synth` writes one directly. A store written by
another tool works if it follows the layout and fragment rule below.

zvDVC imports only `zarr_vectors.api` and `zarr_vectors.building`, and all
access goes through `zvdvc.io`. The dependency is pinned to a commit of
zarr-vectors' gpu-backend branch, which is under active development.

---

## Technical reference

### Point files

`zvdvc.io.pointcloud.read_roi` reads CCPi `.roi` and iDVC `.txt`/`.csv` point
clouds:

* one point per line, `n x y z`, separated by tabs, spaces or commas;
* blank lines and `#` comments are skipped, and so are non-numeric lines
  before the first point (headers);
* `n` becomes `point_id` (int64), `x y z` the position in voxels;
* the first point is CCPi's default start point.

A cloud with no points, a non-finite coordinate or a repeated `n` is refused
(`ValueError` naming the line or the repeated ids). Results are matched to
points by id, so ids must be unique. `write_pointcloud_store` applies the same
checks, and `zvdvc plan` applies them to a store written by another tool.

`zvdvc.io.ccpi.write_roi` writes one `n<TAB>x<TAB>y<TAB>z` line per point
(`%.9g`), with no header.

`zvdvc.geometry.pointgrid.grid_in_mask` generates points as iDVC's Point
Cloud panel does: a regular lattice (spacing
`size × aspect × (1 − overlap)` from `lattice_spacing`), optionally rotated,
kept inside a mask and optionally at least `erode_radius` voxels from its
edge, processed in slabs of 128 slices so a large mask is never wholly in
memory. It is a Python function; there is no CLI command for it.

### Import

`import_points(roi_path, store_path, volume_shape_zyx=..., chunk_shape=...)`
reads the point file and writes a store with

* **bounds** $((0, 0, 0), (n_x, n_y, n_z))$, the volume's size in voxels;
* **`chunk_shape`** from `chunk_shape_for(cluster.tile_shape)`: a quarter of
  the tile per axis, divided exactly, so a tile is always $4^3$ cells (for
  1024³ tiles, 256-voxel cells; cells may be fractional in voxels);
* **`bin_shape`** half the chunk shape (`default_bin_shape`).

Points outside the bounds are refused (`ValueError`). `zvdvc plan` re-imports
when the point file is newer than the store; a failed import leaves no store.
`zvdvc seed` and `zvdvc run` refuse a point file changed after `zvdvc plan`
imported it, and the results store refuses other points than it was made for
({doc}`results_store`, "Run fingerprint").

### Layout

A search-point store is a zarr-vectors point cloud with one resolution level:

```text
points.zarrvectors/
  zarr.json                        root: zarr_vectors metadata (bounds, chunk_shape,
                                   base_bin_shape, geometry_types: [point_cloud]),
                                   multiscales, ome
  0/                               resolution level 0
    zarr.json                      level metadata (vertex_count, arrays present)
    vertices/                      (x, y, z) float32, voxels
    vertex_fragments/              fragment offsets per cell
    vertex_attributes/
      point_id/                    int64, one value per vertex
```

| array | dtype | columns | meaning |
|---|---|---|---|
| `vertices` | float32 | 3 | position $(x, y, z)$ in voxels |
| `vertex_attributes/point_id` | int64 | 1 | CCPi label |

The number of points is the level's `vertex_count`
(`PointCloud.n_points`).

### Cells and fragments

For a point at $\mathbf{x}$ (its stored float32 value), with store bounds
starting at 0:

$$
\text{cell} = \left\lfloor \frac{\mathbf{x}}{\texttt{chunk\_shape}} \right\rfloor,
\qquad
\mathbf{b} = \operatorname{clip}\!\left(\left\lfloor \frac{\mathbf{x} - \text{cell}\cdot\texttt{chunk\_shape}}{\texttt{bin\_shape}} \right\rfloor,\ 0,\ \mathbf{n}_b - 1\right),
$$

where $\mathbf{n}_b = \texttt{chunk\_shape}/\texttt{bin\_shape}$ is the number
of bins per axis (`bin_shape` must divide `chunk_shape` exactly). The
fragment index is C-order over $(x, y, z)$ bins, as zarr-vectors' validator
expects:

$$
\text{fragment} = (b_x\, n_{b,y} + b_y)\, n_{b,z} + b_z .
$$

With two bins per axis there are eight fragments per cell, numbered 0 to 7.

A cell's rows are its fragments in order, and within a fragment the points
keep their input order (a stable sort). Bins are recomputed from the stored
float32 positions whenever they are needed, which reproduces them exactly.
The results store therefore reuses the input's fragments without reading
fragment metadata ({doc}`results_store`).

### Writing: three phases

`write_pointcloud_store` uses zarr-vectors' three-phase HPC pattern in one
process, so the same functions scale to a cluster by running phase 2 per
partition:

1. **Allocate** (`allocate_store`): `create_store` (bounds, `chunk_shape`,
   `base_bin_shape`, `geometry_types=["point_cloud"]`),
   `create_resolution_level(0)`, `create_vertices_array(dtype="float32")`,
   one `create_attribute_array` per attribute, then `defer_presence`.
2. **Write cells** (`write_cell`): per cell, `write_chunk_vertices` and
   `write_chunk_attributes` with the rows split into fragments and
   `record_presence=False`.
3. **Finalise** (`finalize_store`): `rebuild_presence`,
   `refresh_arrays_present`, `update_level_metadata(vertex_count=N)` and
   `write_multiscale_metadata`.

### Tiles and cells

`plan_tiles` groups the occupied cells (`PointCloud.cells()`, from metadata
only) into tiles:

* cells per tile along $(x, y, z)$ = `cluster.tile_shape` reversed to
  $(x, y, z)$, divided by `chunk_shape`. Each ratio must be a whole number to
  within 1 % (so a tile can cover a volume edge that does not divide evenly,
  such as 1257 voxels over 315-voxel cells); otherwise `ValueError`;
* a cell belongs to tile $\lfloor\text{cell}/\text{cells per tile}\rfloor$;
* tiles exist only where cells are occupied, and are numbered in sorted order
  of their tile coordinate.

Each tile records its cells, point count, point box, brick boxes
({doc}`volumes`) and a cost estimate
($n_\text{points} \times n_\text{samples} \times 5 \times f_\text{dof}$, with
$f_\text{dof}$ = 1.0, 1.3, 1.9 for 3, 6, 12 DOF) in `plan.json`. Tiles are
shared across nodes by longest-processing-time-first on that cost, and handed
out dynamically within a node.

### Reading tiles

`PointCloud.read_tile(cells, device=...)` reads a tile's points in one pooled
read:

```python
batch = zarr_vectors.building.read_cells(
    level, cells,                                   # (n_cells, 3) int64
    ["vertices", "vertex_attributes/point_id"],
    on_error="raise",
)
xyz = batch["vertices"].data                        # all rows, cell after cell
offsets = batch["vertices"].offsets                 # CSR: rows of cell i are offsets[i]:offsets[i+1]
```

It returns a `TilePoints` with `xyz`, `point_id`, `cells`, `cell_offsets`
and, per cell, the fragment offsets recomputed from positions
(`bin_offsets`), which the results writer reuses. The worker reads point
positions to the host (`device="cpu"`); they are a few MB per tile. With
`device="cuda"` the arrays are moved to the GPU after the read.
`PointCloud.read_all()` reads every cell (for seeding and kNN on the
coordinator).

### Pyramid

zvDVC does not build a pyramid for the search-point store. For the results
store, `zvdvc finalize --pyramid` calls zarr-vectors'
`Dataset.build_pyramid(factors=[(2.0, 1.0), (2.0, 1.0)], chunk_scale_factors=[2, 2])`,
which adds levels `1` and `2` for viewers ({doc}`results_store`).
