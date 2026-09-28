# CCPi and iDVC file compatibility

This page specifies which CCPi DVC and iDVC files zvDVC reads and writes: the
`dvc_in` parameter file, `.roi` point clouds, `.disp` displacement tables and
`.stat` run summaries, and the behaviour and limits of `zvdvc-dvc`, the
drop-in for CCPi's `dvc` executable. It is for iDVC users and for anyone
checking zvDVC against CCPi. For setting iDVC up to use zvDVC, see
{doc}`/IDVC`. Source:
[`io/ccpi.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/io/ccpi.py),
[`ccpi_dropin.py`](https://github.com/Andrew-Keenlyside/zvDVC/blob/main/src/zvdvc/ccpi_dropin.py).

[iDVC](https://github.com/TomographicImaging/iDVC)
([documentation](https://tomographicimaging.github.io/iDVC/)) is the graphical
DVC application of the Tomographic Imaging / CCPi team at UKRI-STFC (Laura
Murgatroyd, Edoardo Pasca; Apache-2.0). It does its correlation by running
the `dvc` executable of [CCPi DVC](https://github.com/TomographicImaging/DigitalVolumeCorrelation)
(GPL-3.0), code initially developed by Prof. Brian K. Bay and collaborators
(Bay et al. 1999, doi:[10.1007/BF02323555](https://doi.org/10.1007/BF02323555)).
The formats below are those files, as CCPi 22.0.0 writes them.

## Terms

`dvc_in`
: CCPi's parameter file: one `key value` pair per line. iDVC writes one per
  run (for example `dvc_config.txt`) and runs `dvc <dvc_in>`.

`.roi`
: CCPi's point-cloud file: one `n x y z` line per search point.

`.disp`
: CCPi's per-point result table, `<output_filename>.disp`.

`.stat`
: CCPi's run summary, `<output_filename>.stat`. iDVC parses it by line
  position.

Drop-in
: `zvdvc-dvc` (also `zvdvc ccpi`): a program with CCPi `dvc`'s command line
  and output files that runs zvDVC's solver.

---

## Introduction

zvDVC can meet CCPi and iDVC at three levels:

1. **Import settings.** `RunConfig.from_ccpi(dvc_in)` turns a CCPi input
   file into a zvDVC run configuration, and `.roi` files are imported into the
   zarr-vectors point store by `zvdvc plan` ({doc}`run_config`,
   {doc}`points_store`).
2. **Export results.** `zvdvc finalize --disp` and `zvdvc solve --disp` write
   a `.disp` that iDVC and CCPi's `strain` program read ({doc}`results_store`).
3. **Replace `dvc`.** `zvdvc-dvc dvc_in` reads the same input and writes the
   same `.disp`, `.stat` and progress lines as CCPi's `dvc`, so iDVC runs on
   zvDVC without changes ({doc}`/IDVC`).

The reverse also exists for testing: `write_dvc_input` and `write_roi` write
CCPi inputs for a zvDVC configuration, so the CCPi baseline runs on the same
case.

---

## Technical reference

### `dvc_in`

**Syntax** (`read_dvc_input`). Each non-empty line is `key<whitespace>value`;
`#` starts a comment anywhere on a line; values keep inner spaces (for
example `rigid_trans  34.0 4.0 0.0`). Lists may be separated by spaces or
commas.

**Keys read.** The keys below are mapped onto the run configuration; the full
mapping, with the fields each sets, is in {doc}`run_config` ("Importing a CCPi
`dvc_in` file").

| group | keys | required |
|---|---|---|
| files | `reference_filename`, `correlate_filename`, `point_cloud_filename`, `output_filename` | all |
| image layout | `vol_bit_depth` (8 or 16), `vol_wide`, `vol_high`, `vol_tall` | all |
| | `vol_endian` (`little`/`big`), `vol_hdr_lngth` | no |
| subvolume | `subvol_geom`, `subvol_size`, `subvol_npts` | all |
| | `subvol_aspect`, `subvol_thresh`, `gray_thresh_min`, `gray_thresh_max`, `min_vol_fract` | no |
| search | `disp_max`, `num_srch_dof`, `obj_function`, `interp_type` | all |
| | `rigid_trans`, `basin_radius` | no |
| run control | `num_points_to_process` (`0` = all), `starting_point` | no |

**Not supported or not mapped.**

* `vol_bit_depth` other than 8 or 16 is refused, as CCPi reads unsigned 8/16-bit
  voxels only.
* Any key not in the table is read and ignored, without a warning.
* Settings that exist only in zvDVC (`prefilter_sigma`, `method`,
  `max_iterations`, tolerances, seeding strategy, tiles) cannot be set from a
  `dvc_in`; the import uses their defaults, with `seeding.strategy: wavefront`
  and `search.report_convg_fail: false` to match CCPi.
* `interp_type nearest` is accepted by zvDVC but not by CCPi.

**Behaviour of CCPi `dvc` 25.0.0** (conda `ccpi-dvc`, reports v22.0.0-19),
found by running it and followed by `write_dvc_input`:

* a value keeps its line ending unless a `###` comment follows it on the line,
  so `reference_filename  ref.raw` fails with "cannot find file";
  `write_dvc_input` ends every line with a comment;
* `num_points_to_process` and `starting_point` are required, although the
  manual lists them as optional;
* `dvc` exits 0 after input errors; the missing `.disp` is the signal;
* the `.disp` holds `n x y z status objmin u v w` only, with no rotation or
  strain columns, even for 6/12-DOF runs.

zvDVC's own baselines use CCPi 22.0.0, because 25.0.0 has a broken tricubic
path ({doc}`/MVP_PLAN`, M0).

### `.roi`

| aspect | read (`read_roi`) | written (`write_roi`) |
|---|---|---|
| line | `n x y z` | `n<TAB>x<TAB>y<TAB>z` |
| separators | tabs, spaces or commas | tabs |
| numbers | any float; `n` is converted to an integer | `n` integer, coordinates `%.9g` |
| header | non-numeric lines before the first point are skipped (iDVC `.txt`/`.csv`) | none |
| comments | `#` to end of line, blank lines skipped | none |

Coordinates are voxels, $(x, y, z)$ with voxel $i$ at coordinate $i$
({doc}`volumes`). The first point is CCPi's default start point.

### `.disp`

Tab-separated text with one header line and one row per point:

```text
n	x	y	z	status	objmin	u	v	w
1	31.5	31.5	31.5	0	0.00343831	0.863728	-0.235165	0.231709
```

| column | meaning |
|---|---|
| `n` | point label (`point_id`) |
| `x y z` | point position, voxels |
| `status` | CCPi status code, 0 to −3 ({doc}`status_codes`) |
| `objmin` | objective at the solution |
| `u v w` | displacement, voxels |

zvDVC writes two variants:

| | `zvdvc finalize --disp`, `zvdvc solve --disp` (`io.ccpi.write_disp`) | `zvdvc-dvc` (`ccpi_dropin.write_disp`) |
|---|---|---|
| rows | every written point, sorted by `point_id` | the processed points (`num_points_to_process`), in processing order (distance from the start point) |
| number format | `%.9g` throughout | as CCPi: `x y z status objmin` in C++'s default format (6 significant digits, like `%g`), `u v w` fixed to 6 decimals |
| statuses | `to_ccpi()`: codes below −3 become −3 | the same |
| failed points | displacement and `objmin` as stored (`objmin` may be `nan`) | displacement written as 0, non-finite `objmin` as 0, as CCPi writes failed points |

`read_disp` reads both, and also older CCPi files that carry `phi the psi`
and strain columns after `w`.

### `.stat`

**Drop-in layout** (`stat_echo`, `stat_summary`): CCPi 22.0.0's `.stat`, line
for line, which iDVC parses by position:

```text

### echo of the input file for this run

reference_filename	ref.raw
...
subvol_aspect	1	1	1

### end of input file echo

Point Cloud contains 27 points

	bounding box min = [31.5 31.5 31.5]
	bounding box max = [63.5 63.5 63.5]

running under dvc code version: v22.0.0 (zvDVC 0.0.1.dev0 drop-in)

Run start:	2026-09-28 15:57:22
Run finish:	2026-09-28 15:57:22
27 points processed in 0 seconds
0.015 sec/pt
68.290 pt/sec

number successful = 27	(100.000%)
number range fail = 0	(0.000%)
number convg fail = 0	(0.000%)
```

The echo repeats the input settings in CCPi's order and grouping (the
`vol_endian` line only for 16-bit data; the three threshold lines only with
`subvol_thresh on`). While the run is in progress, the file holds the echo, a
`Run start:` line and one `<count> points of <N> at <rate> pt/sec` line per
batch; it is rewritten in the final layout at the end.

**zvDVC summary layout** (`io.ccpi.write_stat`, written by `zvdvc finalize`
and `zvdvc solve`) is different: point count, seconds, points per second,
one line per status with zvDVC's names, and the configuration as YAML
({doc}`results_store`). iDVC cannot parse it; it is for people and for
`read_stat_throughput`, which reads the rate from either layout.

### The `zvdvc-dvc` drop-in

```bash
zvdvc-dvc dvc_in            # or: zvdvc ccpi dvc_in [--backend fused|cupy|cpu|numpy]
zvdvc-dvc version           # dvc code version: v22.0.0 (zvDVC <version> drop-in)
zvdvc-dvc help              # the module's description
zvdvc-dvc                   # CCPi's usage text
```

**What it does** (`ccpi_dropin.run`):

1. Reads `dvc_in` and converts it with `run_config_from_dvc_input`, resolving
   relative paths against the **working directory**, as CCPi does. It reads
   the `.roi` directly.
2. Orders points by distance from `starting_point` (or the first point) and
   keeps the first `num_points_to_process`.
3. Writes the `.stat` echo and a `.disp` header, then prints a banner.
4. Runs the in-memory wavefront solve (CCPi's order and neighbour seeding,
   `report_convg_fail: false`) with both whole volumes in host memory. The
   backend is `--backend`, else `$ZVDVC_BACKEND`, else `fused` with a GPU,
   else `cpu` with numba, else `numpy`.
5. Prints one progress line per point in processing order, which iDVC's
   progress bar reads (the count before the `/`):

   ```text
   1/27 1	31.500 31.500 31.500 	Point_Good	obj= 0.003438        dx= 0.863728        dy= -0.235165        dz= 0.231709
   ```

   Failed points print `Range_Fail`, `Convg_Fail` or `Not_Searched` after the
   coordinates. If stdout is closed, it stops printing and still writes the
   results.
6. Writes `<output_filename>.disp` and the final `<output_filename>.stat`, and
   exits 0.

If `dvc_in` cannot be opened it prints `-> Can't open <file>` and exits 1; if
the input cannot be parsed or the `.roi` read, it prints `input file problem:
...` and exits 1. A volume file that is missing or cannot be read stops the run
with a Python traceback and a non-zero exit. Unlike CCPi, which exits 0 after input errors, a non-zero
exit tells the caller something went wrong. `dvc example` and `dvc manual`
print a note to run CCPi's own `dvc`.

**Limits and differences from CCPi.**

* **Whole volumes in memory.** The drop-in does not tile; both volumes must
  fit in host memory (and, for a GPU backend, in device memory). It writes no
  zarr-vectors store. Use the tiled pipeline (`zvdvc plan/seed/run/finalize`)
  for large data.
* **Points whose subvolume leaves the image** are `Range_Fail`; CCPi
  interpolates wrapped rows or unset memory there and reports them good
  ([case A report](../benchmarks/2026-09-26-case-A-real.md)).
* **Sample points** are drawn from a different random generator, so results
  differ from CCPi's by the sampling spread (about 0.03 voxel per axis at
  8 000 samples on the iDVC example); zvDVC with another template seed
  differs from itself by the same amount.
* **Ties in distance** from the start point may be processed, and listed, in
  a different order: CCPi uses the unstable `std::sort`, zvDVC keeps the
  point-cloud order. iDVC reads results by coordinates, so its views are
  unaffected.
* **`subvol_thresh on`**: CCPi's `.stat` has three extra lines that shift the
  layout iDVC parses, so iDVC's results view fails for such runs. zvDVC
  writes the same layout, so this happens with either engine.
* **No prefilter, no IC-GN, no uncertainty.** A `dvc_in` cannot set these.
* **Rotation and strain parameters** are solved (6/12-DOF) but, as in CCPi's
  `.disp`, not written.

On the iDVC example dataset (Lee, Lavallée and Bay 2022,
doi:[10.5281/zenodo.7363345](https://doi.org/10.5281/zenodo.7363345); 4 680
points), the drop-in's `.stat` echo matches CCPi's line for line apart from
the output name and the version line, and iDVC's own parser reads the same
values from both ({doc}`/IDVC`). Timings and agreement are in the
[case A report](../benchmarks/2026-09-26-case-A-real.md).
