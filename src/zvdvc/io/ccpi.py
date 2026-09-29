"""CCPi DVC / iDVC file formats.

Used to (a) run zvDVC as a drop-in for the ``dvc`` executable that iDVC
launches (``zvdvc ccpi dvc_config.txt``), (b) run the CCPi baseline on zvDVC's
test cases (M0), and (c) compare results point by point.

``dvc_in``
    ``key<ws>value  ### comment`` lines; ``#`` lines ignored. Keys as in iDVC's
    ``dvc_config_template.txt`` and CCPi ``InputRead``.
``.disp``
    Tab-separated, header ``n x y z status objmin u v w``. Current CCPi writes
    displacements only; older files also carry ``phi the psi`` and strain
    columns, and :func:`read_disp` accepts both. Rows are in point-cloud order.
    One policy for every writer (:func:`write_disp`): a point that is not
    ``GOOD`` gets ``u v w = 0`` and a finite ``objmin`` (0 when it has none), as
    CCPi writes failed points; the raw values stay in the results store.
``.stat``
    Input echo, run start/finish, points per second, and counts per status.

Observed behaviour of ``dvc`` 25.0.0 (conda ``ccpi-dvc``, reports v22.0.0-19),
found by running it, not from its source:

* A value keeps its line ending unless a ``###`` comment follows it on the
  line (so ``reference_filename  ref.raw`` fails with "cannot find file");
  :func:`write_dvc_input` ends every line with a comment.
* ``num_points_to_process`` and ``starting_point`` are required although the
  manual lists them as optional.
* ``dvc`` exits 0 after input errors; the missing ``.disp`` is the signal.
  iDVC checks the ``QProcess`` exit *status* (normal exit or crash), not the
  exit code, so it too can only tell a failed run by the missing ``.disp``.
* The ``.disp`` holds ``n x y z status objmin u v w`` only, with no rotation
  or strain columns, even for 6/12-DOF runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zvdvc.config import RunConfig
from zvdvc.status import PointStatus

DISP_COLUMNS = ("n", "x", "y", "z", "status", "objmin", "u", "v", "w")


@dataclass
class RunSummary:
    n_points: int
    seconds: float
    counts: dict[int, int]       # PointStatus (zvDVC codes) -> count


DVC_KEYS = frozenset((
    "reference_filename", "correlate_filename", "point_cloud_filename", "output_filename", "vol_bit_depth", "vol_endian",
    "vol_hdr_lngth", "vol_wide", "vol_high", "vol_tall", "subvol_geom", "subvol_size", "subvol_npts", "subvol_thresh",
    "gray_thresh_min", "gray_thresh_max", "min_vol_fract", "disp_max", "num_srch_dof", "obj_function", "interp_type",
    "rigid_trans", "basin_radius", "subvol_aspect", "num_points_to_process", "starting_point",
    "fine_search",                          # in CCPi's manual, not read by dvc 22
))
# A comment starts at "###" or at a "#" after whitespace, so a "#" inside a path is kept (CCPi splits on tabs).
_COMMENT = re.compile(r"###|(?<=\s)#")


def read_dvc_input(path: str | Path) -> dict[str, str]:
    """``key value`` pairs of a CCPi input file. Values keep inner spaces.

    Lines starting with ``#`` are ignored, and a comment starts at ``###`` or at a
    ``#`` that follows whitespace. Unknown keys warn (CCPi ignores them silently).
    """
    params: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        line = _COMMENT.split(line, 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        params[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    unknown = sorted(set(params) - DVC_KEYS)
    if unknown:
        import warnings

        warnings.warn(f"{path}: ignoring unknown keys {unknown}", UserWarning, stacklevel=2)
    return params


def _floats(key: str, value: str, n: int) -> tuple[float, ...]:
    parts = value.replace(",", " ").split()
    if len(parts) != n:
        raise ValueError(f"{key}: expected {n} numbers, got {value!r}")
    return tuple(_number(key, v) for v in parts)


def _number(key: str, value: str, integer: bool = False) -> float:
    """``value`` as a float, or as an int when ``integer`` (``500.0`` is accepted, as CCPi reads it)."""
    try:
        x = float(value)
    except ValueError:
        raise ValueError(f"{key}: expected a number, got {value!r}") from None
    if integer:
        if not x.is_integer():
            raise ValueError(f"{key}: expected an integer, got {value!r}")
        return int(x)
    return x


# config key -> dvc_in key, to name the key the user wrote in validation errors
_DVC_NAMES = {
    "volumes.raw_shape_xyz": "vol_wide/vol_high/vol_tall", "volumes.raw_header_bytes": "vol_hdr_lngth",
    "subvolume.geometry": "subvol_geom", "subvolume.size": "subvol_size", "subvolume.n_samples": "subvol_npts",
    "subvolume.aspect": "subvol_aspect", "search.dof": "num_srch_dof", "search.objective": "obj_function",
    "search.interpolation": "interp_type", "search.disp_max": "disp_max", "search.rigid_trans": "rigid_trans",
    "search.basin_radius": "basin_radius", "search.threshold.gray_min": "gray_thresh_min",
    "search.threshold.gray_max": "gray_thresh_max", "search.threshold.min_fraction": "min_vol_fract",
    "seeding.start_point": "starting_point", "num_points_to_process": "num_points_to_process",
}


def run_config_from_dvc_input(params: dict[str, str], *, base_dir: str | Path = ".") -> RunConfig:
    """The :class:`RunConfig` equivalent to a CCPi ``dvc_in`` (settings as :func:`read_dvc_input` returns them).

    Relative paths are resolved against ``base_dir`` (the input file's
    folder, as ``dvc`` resolves them against its working directory). The
    output store and work directory are ``<output_filename>.zarrvectors`` and
    ``<output_filename>_zvdvc``. Search settings follow CCPi's behaviour:
    wavefront (CCPi-order) seeding and no ``CONVG_FAIL`` reporting. The result
    is validated as a YAML config is (:meth:`RunConfig.validate`); errors name
    the ``dvc_in`` key.
    """
    base = Path(base_dir)
    p = params

    def path(key: str) -> str:
        q = Path(p[key])
        return str(q if q.is_absolute() else base / q)

    def num(key: str, default: str | None = None, integer: bool = False) -> float:
        return _number(key, p.get(key, default), integer)            # type: ignore[arg-type]

    missing = [k for k in ("reference_filename", "correlate_filename", "point_cloud_filename", "output_filename",
                           "vol_bit_depth", "vol_wide", "vol_high", "vol_tall", "subvol_geom", "subvol_size",
                           "subvol_npts", "disp_max", "num_srch_dof", "obj_function", "interp_type") if k not in p]
    thresh = p.get("subvol_thresh", "off").lower()
    if thresh not in ("on", "off"):
        raise ValueError(f"subvol_thresh: expected on or off, got {thresh!r}")
    if thresh == "on":
        missing += [k for k in ("gray_thresh_min", "gray_thresh_max") if k not in p]
    if missing:
        raise ValueError(f"dvc_in is missing required keys {missing}")
    bits = num("vol_bit_depth", integer=True)
    if bits not in (8, 16):
        raise ValueError(f"vol_bit_depth: must be 8 or 16, got {p['vol_bit_depth']!r}")
    endian = "little"
    if bits != 8:                                    # CCPi ignores vol_endian for 8-bit data
        endian = p.get("vol_endian", "little").lower()
        if endian not in ("little", "big"):
            raise ValueError(f"vol_endian: expected little or big, got {p['vol_endian']!r}")
    output = path("output_filename")
    n_points = num("num_points_to_process", "0", integer=True)
    data = {
        "volumes": {
            "reference": path("reference_filename"),
            "deformed": path("correlate_filename"),
            "raw_shape_xyz": [num(k, integer=True) for k in ("vol_wide", "vol_high", "vol_tall")],
            "raw_dtype": "|u1" if bits == 8 else ("<u2" if endian == "little" else ">u2"),
            "raw_header_bytes": num("vol_hdr_lngth", "0", integer=True),
        },
        "points": path("point_cloud_filename"),
        "output": output + ".zarrvectors",
        "workdir": output + "_zvdvc",
        "subvolume": {
            "geometry": p["subvol_geom"], "size": num("subvol_size"), "n_samples": num("subvol_npts", integer=True),
            "aspect": list(_floats("subvol_aspect", p["subvol_aspect"], 3)) if "subvol_aspect" in p else [1.0, 1.0, 1.0],
        },
        "search": {
            "dof": num("num_srch_dof", integer=True),
            "objective": p["obj_function"],
            "interpolation": p["interp_type"],
            "disp_max": num("disp_max"),
            "rigid_trans": list(_floats("rigid_trans", p["rigid_trans"], 3)) if "rigid_trans" in p else [0.0, 0.0, 0.0],
            "basin_radius": num("basin_radius", "0"),
            "threshold": {"gray_min": num("gray_thresh_min"), "gray_max": num("gray_thresh_max"),
                          "min_fraction": num("min_vol_fract", "0.2")} if thresh == "on" else None,
            "report_convg_fail": False,
        },
        "seeding": {"strategy": "wavefront",
                    "start_point": list(_floats("starting_point", p["starting_point"], 3)) if p.get("starting_point") else None},
        "num_points_to_process": n_points or None,
    }
    try:
        return RunConfig.from_dict(data)
    except ValueError as exc:                        # name the dvc_in key rather than the config path
        m = re.match(r"config\.([\w.]+?)(?:\[\d+\])?: (.*)", str(exc))
        if m and m.group(1) in _DVC_NAMES:
            raise ValueError(f"{_DVC_NAMES[m.group(1)]}: {m.group(2)}") from None
        raise


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def ccpi_volume_layout(cfg: RunConfig) -> tuple[tuple[int, int, int], np.dtype, int]:
    """``(shape_xyz, dtype, header_bytes)`` of the reference volume as CCPi must read it.

    CCPi reads a flat file of unsigned 8/16-bit voxels with a fixed header.
    ``.npy`` qualifies (its header is the fixed header) when C-ordered; OME-Zarr
    must first be exported with :func:`zvdvc.io.volume.write_raw`.
    """
    from zvdvc.io.volume import RawVolume, is_zarr_uri

    spec = cfg.volumes
    if is_zarr_uri(spec.reference):
        raise ValueError("CCPi reads flat raw files; export OME-Zarr with zvdvc.io.volume.write_raw first")
    vol = RawVolume(spec.reference, shape_xyz=spec.raw_shape_xyz, dtype=spec.raw_dtype, header_bytes=spec.raw_header_bytes)
    dtype = vol.array.dtype                          # the file's byte order (bricks are read in native order)
    if dtype.kind != "u" or dtype.itemsize not in (1, 2):
        raise ValueError(f"CCPi reads unsigned 8/16-bit voxels only, not {dtype}")
    if not vol.array.flags.c_contiguous:
        raise ValueError(f"{spec.reference}: CCPi needs C-ordered data (x fastest)")
    z, y, x = vol.shape
    return (x, y, z), dtype, vol.header_bytes


def write_dvc_input(cfg: RunConfig, path: str | Path, *, roi_path: str | Path, output_base: str | Path) -> None:
    """Write a CCPi input file equivalent to ``cfg``, to run the CPU baseline on the same case (M0)."""
    (nx, ny, nz), dtype, header = ccpi_volume_layout(cfg)
    sub, srch = cfg.subvolume, cfg.search
    if srch.method != "fagn":
        raise ValueError("CCPi runs forward-additive Gauss-Newton only")
    lines = [
        "### written by zvdvc.io.ccpi.write_dvc_input",
        f"reference_filename\t{Path(cfg.volumes.reference).resolve()}",
        f"correlate_filename\t{Path(cfg.volumes.deformed).resolve()}",
        f"point_cloud_filename\t{Path(roi_path).resolve()}",
        f"output_filename\t{Path(output_base).resolve()}",
        f"vol_bit_depth\t{8 * dtype.itemsize}",
    ]
    if dtype.itemsize > 1:
        lines.append(f"vol_endian\t{'big' if dtype.byteorder == '>' else 'little'}")
    lines += [
        f"vol_hdr_lngth\t{header}",
        f"vol_wide\t{nx}",
        f"vol_high\t{ny}",
        f"vol_tall\t{nz}",
        f"subvol_geom\t{sub.geometry}",
        f"subvol_size\t{_num(sub.size)}",
        f"subvol_npts\t{sub.n_samples}",
        f"subvol_thresh\t{'on' if srch.threshold else 'off'}",
    ]
    if srch.threshold:
        lines += [
            f"gray_thresh_min\t{_num(srch.threshold.gray_min)}",
            f"gray_thresh_max\t{_num(srch.threshold.gray_max)}",
            f"min_vol_fract\t{_num(srch.threshold.min_fraction)}",
        ]
    lines += [
        f"disp_max\t{_num(srch.disp_max)}",
        f"num_srch_dof\t{srch.dof}",
        f"obj_function\t{srch.objective}",
        f"interp_type\t{srch.interpolation}",
        "rigid_trans\t" + " ".join(_num(v) for v in srch.rigid_trans),
        f"basin_radius\t{_num(srch.basin_radius)}",
        "subvol_aspect\t" + " ".join(_num(v) for v in sub.aspect),
    ]
    # dvc 25 requires both keys although its manual lists them as optional
    lines.append(f"num_points_to_process\t{cfg.num_points_to_process or 0}")
    start = cfg.seeding.start_point
    if start is None:
        from zvdvc.io.pointcloud import read_roi

        start = tuple(read_roi(roi_path)[1][0])
    lines.append("starting_point\t" + " ".join(_num(v) for v in start))
    # CCPi keeps the line ending in a value unless another tab-separated token follows it, as in its own
    # examples: InputRead appends "\n" to each line and takes the value up to the next tab, so
    # "reference_filename<TAB>ref.raw" at the end of a line fails with "cannot find file" (dvc 22.0.0).
    # Every value line therefore ends with a "### zvdvc" comment.
    body =[line if line.startswith("#") else f"{line}\t### zvdvc" for line in lines]
    Path(path).write_text("\n".join(body) + "\n")


def write_roi(path: str | Path, point_id: np.ndarray, xyz: np.ndarray) -> None:
    """CCPi point cloud: one ``n<TAB>x<TAB>y<TAB>z`` line per point, no header."""
    point_id = np.asarray(point_id).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    with open(path, "w") as fh:
        for n, (x, y, z) in zip(point_id, xyz):
            fh.write(f"{int(n)}\t{x:.9g}\t{y:.9g}\t{z:.9g}\n")


def read_disp(path: str | Path) -> np.ndarray:
    """Structured array with fields ``n, x, y, z, status, objmin, u, v, w`` (+ optional extra columns)."""
    text = Path(path).read_text().splitlines()
    rows = [line.split() for line in text if line.strip()]
    if not rows:
        raise ValueError(f"{path}: empty .disp file")
    header = rows[0]
    if header[: len(DISP_COLUMNS)] != list(DISP_COLUMNS):
        raise ValueError(f"{path}: unexpected .disp header {header}")
    data = rows[1:]
    dtype = [(name, np.int64 if name in ("n", "status") else np.float64) for name in header]
    out = np.empty(len(data), dtype=dtype)
    for i, name in enumerate(header):
        col = [r[i] for r in data]
        out[name] = np.asarray(col, dtype=np.float64).astype(out.dtype[name]) if col else []
    return out


DISP_HEADER = "\t".join(DISP_COLUMNS) + "\n"


def g(v: Any) -> str:
    """A number as C++ ``operator<<`` prints it by default (6 significant digits, like ``%g``)."""
    v = float(v)
    if v != v:
        return "nan"
    s = f"{v:g}"
    return "0" if s == "-0" else s


def disp_row(n: int, xyz: Any, status: int, objmin: float, uvw: Any) -> str:
    """One CCPi ``append_result`` line: default-format coordinates and objective, 6-decimal displacements."""
    x, y, z = xyz
    u, v, w = uvw
    return f"{int(n)}\t{g(x)}\t{g(y)}\t{g(z)}\t{int(status)}\t{g(objmin)}\t{u:.6f}\t{v:.6f}\t{w:.6f}\n"


def write_disp(
    path: str | Path,
    point_id: np.ndarray,
    xyz: np.ndarray,
    status: np.ndarray,
    objmin: np.ndarray,
    displacement: np.ndarray,
    *,
    ccpi_precision: bool = False,
) -> None:
    """CCPi ``.disp``, with the one policy for points that are not ``GOOD``: zero displacement and a finite objmin.

    zvDVC-only status codes are written as CCPi's ``NOT_SEARCHED``. Numbers are
    written to 9 significant digits, or with ``ccpi_precision`` exactly as CCPi
    prints them (:func:`disp_row`), as the drop-in does.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    disp = np.asarray(displacement, dtype=np.float64).reshape(-1, 3)
    codes = [PointStatus(int(s)).to_ccpi() for s in np.asarray(status).reshape(-1)]
    with open(path, "w") as fh:
        fh.write(DISP_HEADER)
        for n, p, st, obj, u in zip(np.asarray(point_id).reshape(-1), xyz, codes, np.asarray(objmin, dtype=np.float64).reshape(-1), disp):
            obj = obj if np.isfinite(obj) else 0.0
            u = u if st == PointStatus.GOOD else (0.0, 0.0, 0.0)
            if ccpi_precision:
                fh.write(disp_row(n, p, st, obj, u))
            else:
                fh.write(f"{int(n)}\t{p[0]:.9g}\t{p[1]:.9g}\t{p[2]:.9g}\t{st}\t{obj:.9g}\t{u[0]:.9g}\t{u[1]:.9g}\t{u[2]:.9g}\n")


def write_stat(path: str | Path, cfg: RunConfig, summary: RunSummary) -> None:
    """Run summary in the spirit of CCPi's ``.stat``: settings, time, rate and status counts.

    ``status <NAME>`` lines count zvDVC's statuses; the ``number ...`` lines below
    them count CCPi's codes, mapped as the ``.disp`` maps them (``THRESH_FAIL``
    and ``SINGULAR`` are ``not searched``). This is not the layout iDVC parses;
    the drop-in (:mod:`zvdvc.ccpi_dropin`) writes that one.
    """
    rate = summary.n_points / summary.seconds if summary.seconds > 0 else float("nan")
    names = {int(s): s.name for s in PointStatus}
    ccpi: dict[int, int] = {}
    for code, count in summary.counts.items():
        key = PointStatus(code).to_ccpi() if code in names else int(PointStatus.NOT_SEARCHED)
        ccpi[key] = ccpi.get(key, 0) + count
    n = max(summary.n_points, 1)
    lines = [
        "zvdvc run summary",
        f"points\t{summary.n_points}",
        f"seconds\t{summary.seconds:.3f}",
        f"points_per_second\t{rate:.3f}",
        *(f"status {names.get(code, code)}\t{count}" for code, count in sorted(summary.counts.items(), reverse=True)),
        "",
        *(f"number {label} = {ccpi.get(int(code), 0)}\t({100.0 * ccpi.get(int(code), 0) / n:.3f}%)"
          for label, code in (("successful", PointStatus.GOOD), ("range fail", PointStatus.RANGE_FAIL),
                              ("convg fail", PointStatus.CONVG_FAIL), ("not searched", PointStatus.NOT_SEARCHED))),
        "",
        "### config",
    ]
    import yaml

    lines.append(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    Path(path).write_text("\n".join(lines))


# zvDVC writes "points_per_second <v>"; CCPi writes "<v> pt/sec"
_RATE = re.compile(r"points_per_second\s+([0-9][0-9.eE+-]*)|([0-9][0-9.eE+-]*)\s*(?:pt|pts|points?)\s*(?:/|per)\s*sec", re.I)


def read_stat_throughput(path: str | Path) -> float:
    """Points per second reported in a CCPi ``.stat`` file (the M0 baseline number)."""
    text = Path(path).read_text()
    m = _RATE.search(text)
    if m:
        return float(m.group(1) or m.group(2))
    raise ValueError(f"{path}: no points-per-second figure found")
