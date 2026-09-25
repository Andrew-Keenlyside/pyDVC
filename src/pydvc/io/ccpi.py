"""CCPi DVC / iDVC file formats.

Used to (a) run pyDVC as a drop-in for the ``dvc`` executable that iDVC
launches (``pydvc ccpi dvc_config.txt``), (b) run the CCPi baseline on pyDVC's
test cases (M0), and (c) compare results point by point.

``dvc_in``
    ``key<ws>value  ### comment`` lines; ``#`` lines ignored. Keys as in iDVC's
    ``dvc_config_template.txt`` and CCPi ``InputRead``.
``.disp``
    Tab-separated, header ``n x y z status objmin u v w``. Current CCPi writes
    displacements only; older files also carry ``phi the psi`` and strain
    columns, and :func:`read_disp` accepts both. Rows are in point-cloud order.
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
* The ``.disp`` holds ``n x y z status objmin u v w`` only, with no rotation
  or strain columns, even for 6/12-DOF runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pydvc._todo import todo
from pydvc.config import RunConfig
from pydvc.status import PointStatus

DISP_COLUMNS = ("n", "x", "y", "z", "status", "objmin", "u", "v", "w")


@dataclass
class RunSummary:
    n_points: int
    seconds: float
    counts: dict[int, int]       # PointStatus (CCPi codes) -> count


def read_dvc_input(path: str | Path) -> dict[str, str]:
    """``key value`` pairs of a CCPi input file; ``#`` starts a comment. Values keep inner spaces."""
    params: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        params[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    return params


def run_config_from_dvc_input(params: dict[str, str]) -> RunConfig:
    raise todo("M4", "run_config_from_dvc_input")


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def ccpi_volume_layout(cfg: RunConfig) -> tuple[tuple[int, int, int], np.dtype, int]:
    """``(shape_xyz, dtype, header_bytes)`` of the reference volume as CCPi must read it.

    CCPi reads a flat file of unsigned 8/16-bit voxels with a fixed header.
    ``.npy`` qualifies (its header is the fixed header) when C-ordered; OME-Zarr
    must first be exported with :func:`pydvc.io.volume.write_raw`.
    """
    from pydvc.io.volume import RawVolume, is_zarr_uri

    spec = cfg.volumes
    if is_zarr_uri(spec.reference):
        raise ValueError("CCPi reads flat raw files; export OME-Zarr with pydvc.io.volume.write_raw first")
    vol = RawVolume(spec.reference, shape_xyz=spec.raw_shape_xyz, dtype=spec.raw_dtype, header_bytes=spec.raw_header_bytes)
    if vol.dtype.kind != "u" or vol.dtype.itemsize not in (1, 2):
        raise ValueError(f"CCPi reads unsigned 8/16-bit voxels only, not {vol.dtype}")
    if not vol.array.flags.c_contiguous:
        raise ValueError(f"{spec.reference}: CCPi needs C-ordered data (x fastest)")
    z, y, x = vol.shape
    return (x, y, z), vol.dtype, vol.header_bytes


def write_dvc_input(cfg: RunConfig, path: str | Path, *, roi_path: str | Path, output_base: str | Path) -> None:
    """Write a CCPi input file equivalent to ``cfg``, to run the CPU baseline on the same case (M0)."""
    (nx, ny, nz), dtype, header = ccpi_volume_layout(cfg)
    sub, srch = cfg.subvolume, cfg.search
    if srch.method != "fagn":
        raise ValueError("CCPi runs forward-additive Gauss-Newton only")
    lines = [
        "### written by pydvc.io.ccpi.write_dvc_input",
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
        from pydvc.io.pointcloud import read_roi

        start = tuple(read_roi(roi_path)[1][0])
    lines.append("starting_point\t" + " ".join(_num(v) for v in start))
    # CCPi keeps the line ending in a value unless a comment follows it, as in its own examples
    body = [line if line.startswith("#") else f"{line}\t### pydvc" for line in lines]
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


def write_disp(
    path: str | Path,
    point_id: np.ndarray,
    xyz: np.ndarray,
    status: np.ndarray,
    objmin: np.ndarray,
    displacement: np.ndarray,
) -> None:
    """CCPi ``.disp``. pyDVC-only status codes are written as CCPi's ``NOT_SEARCHED``."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    disp = np.asarray(displacement, dtype=np.float64).reshape(-1, 3)
    codes = [PointStatus(int(s)).to_ccpi() for s in np.asarray(status).reshape(-1)]
    with open(path, "w") as fh:
        fh.write("\t".join(DISP_COLUMNS) + "\n")
        for n, p, st, obj, u in zip(np.asarray(point_id).reshape(-1), xyz, codes, np.asarray(objmin).reshape(-1), disp):
            fh.write(
                f"{int(n)}\t{p[0]:.9g}\t{p[1]:.9g}\t{p[2]:.9g}\t{st}\t{obj:.9g}\t{u[0]:.9g}\t{u[1]:.9g}\t{u[2]:.9g}\n"
            )


def write_stat(path: str | Path, cfg: RunConfig, summary: RunSummary) -> None:
    """Run summary in the spirit of CCPi's ``.stat``: settings, time, rate and status counts."""
    rate = summary.n_points / summary.seconds if summary.seconds > 0 else float("nan")
    names = {int(s): s.name for s in PointStatus}
    lines = [
        "pydvc run summary",
        f"points\t{summary.n_points}",
        f"seconds\t{summary.seconds:.3f}",
        f"points_per_second\t{rate:.3f}",
        *(f"status {names.get(code, code)}\t{count}" for code, count in sorted(summary.counts.items(), reverse=True)),
        "",
        "### config",
    ]
    import yaml

    lines.append(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    Path(path).write_text("\n".join(lines))


# pyDVC writes "points_per_second <v>"; CCPi writes "<v> pt/sec"
_RATE = re.compile(r"points_per_second\s+([0-9][0-9.eE+-]*)|([0-9][0-9.eE+-]*)\s*(?:pt|pts|points?)\s*(?:/|per)\s*sec", re.I)


def read_stat_throughput(path: str | Path) -> float:
    """Points per second reported in a CCPi ``.stat`` file (the M0 baseline number)."""
    text = Path(path).read_text()
    m = _RATE.search(text)
    if m:
        return float(m.group(1) or m.group(2))
    raise ValueError(f"{path}: no points-per-second figure found")
