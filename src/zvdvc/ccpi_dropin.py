"""A drop-in for CCPi's ``dvc`` executable, so iDVC (or any script that runs ``dvc``) uses zvDVC.

iDVC runs ``dvc <dvc_in>`` from ``PATH`` in its working directory, follows progress by
reading stdout lines that start with ``<count>/<total>``, treats a normal process exit as
success, then reads ``<output_filename>.disp`` (tab separated, one header line) and
``<output_filename>.stat``, which it parses **by line position** (iDVC ``utilities.RunResults``).
This module reproduces those files byte for byte in the layout of CCPi 22.0.0
(``InputRead::echo_input``, ``result_header``, ``append_result`` and ``dvc.cpp``), with
zvDVC's results in them:

* the ``.stat`` echo of the input, the point-cloud bounding box, run times, rate and counts;
* the ``.disp`` rows in processing order (distance from the starting point, as CCPi), for the
  ``num_points_to_process`` points processed; ``x y z status objmin`` in C++'s default
  (``%g``) format and ``u v w`` fixed to 6 decimals, as CCPi writes them;
* a progress line per point, ``i/N label<TAB>x y z <TAB>Point_Good<TAB>obj= ...``.

As in CCPi, relative paths in ``dvc_in`` are relative to the working directory. The run is
the in-memory wavefront solve (CCPi's order and seeding) on the default backend (the GPU
when there is one; ``ZVDVC_BACKEND`` overrides). One difference is deliberate: a point whose
subvolume leaves the image is ``Range_Fail``, where CCPi interpolates wrapped or unset data
(docs/benchmarks/2026-09-26-case-A-real.md).

To use it from iDVC, put an executable named ``dvc`` that runs ``zvdvc-dvc "$@"`` ahead of
CCPi's on ``PATH`` (docs/IDVC.md).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from zvdvc.status import PointStatus

CCPI_VERSION = "v22.0.0"
_WORDS = {int(PointStatus.GOOD): "Point_Good", int(PointStatus.RANGE_FAIL): "Range_Fail",
          int(PointStatus.CONVG_FAIL): "Convg_Fail"}


def g(v: Any) -> str:
    """A number as C++ ``operator<<`` prints it by default (6 significant digits, like ``%g``)."""
    v = float(v)
    if v != v:
        return "nan"
    s = f"{v:g}"
    return "0" if s == "-0" else s


# --------------------------------------------------------------------------- .disp


DISP_HEADER = "n\tx\ty\tz\tstatus\tobjmin\tu\tv\tw\n"


def disp_row(n: int, xyz: Any, status: int, objmin: float, uvw: Any) -> str:
    """One ``append_result`` line: default-format coordinates and objective, 6-decimal displacements."""
    x, y, z = xyz
    u, v, w = uvw
    return f"{int(n)}\t{g(x)}\t{g(y)}\t{g(z)}\t{int(status)}\t{g(objmin)}\t{u:.6f}\t{v:.6f}\t{w:.6f}\n"


def write_disp(path: str | Path, point_id: Any, xyz: Any, status: Any, objmin: Any, disp: Any) -> None:
    """CCPi's ``.disp``: failed points get zero displacement, as CCPi writes them (``blank_par_min``)."""
    with open(path, "w") as fh:
        fh.write(DISP_HEADER)
        for n, p, st, obj, u in zip(point_id, xyz, status, objmin, disp):
            code = PointStatus(int(st)).to_ccpi()
            fh.write(disp_row(n, p, code, obj if np.isfinite(obj) else 0.0, u if code == 0 else (0.0, 0.0, 0.0)))


# --------------------------------------------------------------------------- .stat


def stat_echo(params: dict[str, str], n_points: int, bbox_min: Any, bbox_max: Any) -> str:
    """``InputRead::echo_input``: the settings, point-cloud size and bounding box, and version."""
    p = params
    bits = int(float(p["vol_bit_depth"]))
    num = lambda key, default="0": g(p.get(key, default))                          # noqa: E731
    three = lambda key, default: "\t".join(g(v) for v in (p.get(key) or default).split()[:3])   # noqa: E731
    lines = ["", "### echo of the input file for this run", "",
             f"reference_filename\t{p['reference_filename']}", f"correlate_filename\t{p['correlate_filename']}",
             f"point_cloud_filename\t{p['point_cloud_filename']}", f"output_filename\t{p['output_filename']}", "",
             f"vol_bit_depth\t{bits}"]
    if bits != 8:
        lines.append(f"vol_endian\t{p.get('vol_endian', 'little')}")
    lines += [f"vol_hdr_lngth\t{num('vol_hdr_lngth')}", f"vol_wide\t{num('vol_wide')}", f"vol_high\t{num('vol_high')}",
              f"vol_tall\t{num('vol_tall')}", "",
              f"subvol_geom\t{p['subvol_geom']}", f"subvol_size\t{num('subvol_size')}", f"subvol_npts\t{num('subvol_npts')}", "",
              f"subvol_thresh\t{p.get('subvol_thresh', 'off')}"]
    if p.get("subvol_thresh", "off").lower() == "on":
        lines += [f"gray_thresh_min\t{num('gray_thresh_min')}", f"gray_thresh_max\t{num('gray_thresh_max')}",
                  f"min_vol_fract\t{num('min_vol_fract', '0.2')}"]
    lines += ["", f"disp_max\t{num('disp_max')}", f"num_srch_dof\t{num('num_srch_dof')}",
              f"obj_function\t{p['obj_function']}", f"interp_type\t{p['interp_type']}", "",
              f"rigid_trans\t{three('rigid_trans', '0 0 0')}", f"basin_radius\t{num('basin_radius')}",
              f"subvol_aspect\t{three('subvol_aspect', '1 1 1')}", "", "### end of input file echo", "",
              f"Point Cloud contains {n_points} points", "",
              f"\tbounding box min = [{g(bbox_min[0])} {g(bbox_min[1])} {g(bbox_min[2])}]",
              f"\tbounding box max = [{g(bbox_max[0])} {g(bbox_max[1])} {g(bbox_max[2])}]", "",
              f"running under dvc code version: {CCPI_VERSION} (zvDVC {_version()} drop-in)", ""]
    return "\n".join(lines) + "\n"


def stat_summary(start: float, finish: float, count: int, seconds: float, codes: Any) -> str:
    """The end of the ``.stat`` (``dvc.cpp``): times, rate and status counts."""
    codes = np.asarray(codes)
    n = max(count, 1)
    pct = lambda k: 100.0 * k / n                                                   # noqa: E731
    good, rng, cnv = int((codes == 0).sum()), int((codes == -1).sum()), int((codes == -2).sum())
    stamp = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))        # noqa: E731
    rate = count / seconds if seconds > 0 else 0.0
    return (f"Run start:\t{stamp(start)}\nRun finish:\t{stamp(finish)}\n"
            f"{count} points processed in {seconds:.0f} seconds\n"
            f"{(seconds / n):.3f} sec/pt\n{rate:.3f} pt/sec\n\n"
            f"number successful = {good}\t({pct(good):.3f}%)\n"
            f"number range fail = {rng}\t({pct(rng):.3f}%)\n"
            f"number convg fail = {cnv}\t({pct(cnv):.3f}%)\n\n")


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("zvdvc")
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- progress


def progress_line(i: int, total: int, label: int, xyz: Any, status: int, objmin: float, uvw: Any) -> str:
    """``dvc.cpp``'s per-point console line; iDVC reads the count before the ``/``."""
    x, y, z = xyz
    head = f"{i}/{total} {int(label)}\t{x:.3f} {y:.3f} {z:.3f} \t"
    if status == 0:
        return (head + f"Point_Good\tobj= {objmin:.6f}{'dx= ':>12}{uvw[0]:.6f}{'dy= ':>12}{uvw[1]:.6f}"
                f"{'dz= ':>12}{uvw[2]:.6f}")
    return head + _WORDS.get(status, "Not_Searched")


# --------------------------------------------------------------------------- run


def run(dvc_in: str | Path, *, backend: str | None = None, out: TextIO = sys.stdout) -> int:
    """Run ``dvc_in`` as CCPi would, with zvDVC; 0 on success."""
    from zvdvc.io.ccpi import read_dvc_input, run_config_from_dvc_input
    from zvdvc.io.pointcloud import read_roi
    from zvdvc.pipeline.inmemory import solve_in_memory
    from zvdvc.solver import seeding
    from zvdvc.solver.engines import default_backend

    path = Path(dvc_in)
    if not path.is_file():
        print(f"\n-> Can't open {dvc_in}\n", file=out)
        return 1
    try:
        params = read_dvc_input(path)
        cfg = run_config_from_dvc_input(params, base_dir=Path.cwd())       # CCPi: relative to the working dir
        point_id, xyz = read_roi(cfg.points)
    except Exception as exc:                                                 # CCPi prints the problem and stops
        print(f"\ninput file problem: {type(exc).__name__}: {exc}\n", file=out)
        return 1
    backend = backend or os.environ.get("ZVDVC_BACKEND") or default_backend()
    base = Path(params["output_filename"])
    base = base if base.is_absolute() else Path.cwd() / base
    disp_path, stat_path = base.with_name(base.name + ".disp"), base.with_name(base.name + ".stat")
    start = cfg.seeding.start_point or tuple(xyz[0])
    order = seeding.processing_order(np.asarray(xyz, dtype=np.float64), start)
    if cfg.num_points_to_process and cfg.num_points_to_process < len(order):
        order = order[: cfg.num_points_to_process]
    total = len(order)
    rank = np.full(len(xyz), -1)
    rank[order] = np.arange(total)

    echo = stat_echo(params, len(xyz), np.min(xyz, axis=0), np.max(xyz, axis=0))
    t_start = time.time()
    stat_path.write_text(echo + f"Run start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start))}\n")
    disp_path.write_text(DISP_HEADER)
    print(f"\n\n*************************\nzvDVC {_version()} as CCPi dvc ({backend} backend): {total} points\n", file=out)
    out.flush()

    done = {"count": 0, "stdout": True}
    t0 = time.perf_counter()

    def say(text: str) -> None:
        """Print progress while someone reads it; if the reader goes away, carry on and still write the results."""
        if not done["stdout"]:
            return
        try:
            print(text, file=out)
        except (BrokenPipeError, ValueError):
            done["stdout"] = False

    def on_solved(idx: np.ndarray, res: Any) -> None:
        for j in sorted((int(k) for k in idx if rank[k] >= 0), key=lambda k: rank[k]):
            done["count"] += 1
            code = PointStatus(int(res.status[j])).to_ccpi()
            say(progress_line(done["count"], total, point_id[j], xyz[j], code, float(res.objmin[j]), res.params[j, :3]))
        try:
            out.flush()
        except (BrokenPipeError, ValueError):
            done["stdout"] = False
        c = done["count"]
        with open(stat_path, "a") as fh:
            fh.write(f"{c} points of {len(xyz)} at {c / max(time.perf_counter() - t0, 1e-9):.6f} pt/sec\n")

    res = solve_in_memory(cfg, point_id, xyz, backend=backend, on_solved=on_solved)
    seconds = time.perf_counter() - t0
    t_finish = time.time()
    say(f"{total / seconds if seconds > 0 else 0.0:.6f} pt/sec average")
    sel = order
    write_disp(disp_path, point_id[sel], np.asarray(xyz)[sel], res.status[sel], res.objmin[sel], res.params[sel, :3])
    codes = [PointStatus(int(s)).to_ccpi() for s in res.status[sel]]
    stat_path.write_text(echo + stat_summary(t_start, t_finish, total, seconds, codes))
    if not done["stdout"]:                           # keep Python from reporting the closed pipe at exit
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass
    return 0


USAGE = """
Options:
dvc dvc_in\t\t// execute dvc code with dvc_in controlling the run
dvc help\t\t// provide additional detail about running the dvc code
dvc version\t\t// print the version (zvDVC's drop-in for CCPi dvc)
"""


def main(argv: list[str] | None = None) -> int:
    """``zvdvc-dvc``: CCPi ``dvc``'s command line (``dvc dvc_in``, ``help``, ``version``)."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(USAGE)
        return 0
    if args[0] == "help":
        print(__doc__)
        return 0
    if args[0] in ("version", "-v", "--version"):
        print(f"dvc code version: {CCPI_VERSION} (zvDVC {_version()} drop-in)")
        return 0
    if args[0] in ("example", "manual"):
        print(f"`dvc {args[0]}` is CCPi's own documentation; run CCPi's dvc for it. zvDVC's drop-in runs `dvc dvc_in`.")
        return 0
    return run(args[0])


if __name__ == "__main__":
    sys.exit(main())
