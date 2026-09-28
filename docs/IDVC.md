# Using zvDVC from iDVC

iDVC does its correlation by running CCPi's `dvc` program. zvDVC ships a drop-in
for it, `zvdvc-dvc`, so iDVC's interface (registration, masks, point clouds, bulk
runs, results viewers) runs on zvDVC's engine: on the iDVC example, 1.6–2 s on a
GPU instead of 37 minutes.

## Set up (once)

In the environment where zvDVC is installed (`pip install -e .` provides `zvdvc-dvc`):

```bash
bash scripts/idvc/install_dvc_shim.sh          # writes ~/.local/zvdvc-dvc/bin/dvc (conda env zvdvc-gpu)
PATH=~/.local/zvdvc-dvc/bin:$PATH idvc          # start iDVC this way to use zvDVC
```

The shim is a two-line script named `dvc` that runs `zvdvc-dvc` in the zvDVC
environment. Start iDVC without the `PATH` prefix to use CCPi's `dvc` again.
Arguments: `bash scripts/idvc/install_dvc_shim.sh DIR ENV_NAME`.

## What it does

`zvdvc-dvc dvc_in` (also `zvdvc ccpi dvc_in`) reads the parameter file iDVC writes and
runs the in-memory wavefront solve: CCPi's point order, seeding and settings. It uses the GPU when
there is one (`ZVDVC_BACKEND=cpu` forces the CPU engine). It writes the files CCPi 22.0.0 writes, in
its layout:

* `<output_filename>.disp`: `n x y z status objmin u v w`, one row per processed point in
  processing order (`num_points_to_process` honoured), formatted as CCPi formats them;
* `<output_filename>.stat`: the echo of the input, the point-cloud bounding box, times, rate
  and status counts. iDVC parses this file by line position, and reads the same values from it
  as from CCPi's (checked with iDVC's own parser on case A);
* a progress line per point on stdout (`i/N label  x y z  Point_Good  obj= … dx= …`), which
  iDVC's progress bar and time estimate read.

Checked against CCPi on the same `dvc_in` (`tests/test_ccpi_dropin.py`, and case A): the same
points, displacements within the two codes' sampling spread, and the `.stat` echo identical
line for line apart from the output name and the version line.

## Differences from CCPi

* **Points whose subvolume leaves the image** are `Range_Fail`. CCPi interpolates wrapped
  rows or unset memory there and reports them GOOD
  ([case A report](benchmarks/2026-09-26-case-A-real.md)).
* **Points at equal distance from the start point** may be processed, and listed in `.disp`,
  in a different order. CCPi sorts with the unstable `std::sort`; zvDVC keeps the point-cloud
  order among ties. iDVC reads results by coordinates, so its views are unaffected.
* **Sample points**: both codes draw each subvolume's sample points at random, from different
  generators. Results therefore differ by the sampling spread (~0.03 voxel per axis at 8 000
  samples on case A); zvDVC run with another seed differs from itself by the same amount.
* `dvc example` and `dvc manual` (CCPi's printed documentation) are not reproduced; run CCPi's
  `dvc` for them.
* With `subvol_thresh on`, CCPi's `.stat` has three extra lines that shift the layout iDVC
  parses, so iDVC's results view fails for such runs. zvDVC writes the same layout, so this
  happens with either engine.

## Strain

CCPi's `strain` program reads `.disp` files, so it works unchanged on zvDVC's output. zvDVC's own
`zvdvc strain RESULTS` (a results store, `.npz` or `.disp`) uses the same method, flags and CSV layout
(`-sw`, `-t`, `-r`, `-E`, `-D`; `<base>-sw25.Lstr.csv` etc.), and matches CCPi's values at every interior
point of case A. It adds a strain uncertainty per point (`sd_*` columns) and returns NaN for windows whose
points cannot determine the fit, where CCPi returns an ill-determined value.
