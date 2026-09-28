"""Strain from displacement, as CCPi's ``strain`` program computes it, with an uncertainty per point.

At each point, the displacements of its strain window are fitted by least squares with a 3D
quadratic polynomial per component, and the displacement gradient ``G = du/dx`` is taken at
the point. The window is the ``window`` nearest points (the point itself included), keeping
those that are GOOD and whose objective is at most ``threshold``. With ``refill``, further
neighbours replace the ones dropped until the window is full (CCPi ``-r``). From ``G``:
engineering strain ``(G + G^T)/2``, Lagrangian strain ``(G + G^T + G^T G)/2``, and their
principal values (descending). A point cloud in a coordinate plane (every point sharing one
coordinate, like CCPi's central grid) is fitted in that plane, with the out-of-plane
component and derivatives zero, as CCPi does.

Unlike CCPi, each point also gets:

* ``residual_rms`` (3,): the fit's residual per displacement component;
* ``strain_sd`` (6,): the standard deviation of each engineering strain component, from
  propagating a displacement uncertainty through the least-squares fit. The uncertainty is
  ``sigma_u`` if given (for example the sample-set spread from
  :func:`zvdvc.bench.error_floor.seed_repeatability`), else the fit residual.

The fit is centred on each point and scaled by its window radius, which is mathematically the
same model as CCPi's but better conditioned; points with fewer good neighbours than the
polynomial has terms, or whose points cannot determine every term (for example only two
levels along an axis), get NaN strain where CCPi would return an ill-determined value. On CCPi's case A result this reproduces CCPi's strain to
every printed digit at all interior points. Within a grid step of the cloud's edge the window
can end among equidistant points, which CCPi picks with an unstable sort and zvDVC in
neighbour-search order, so a few edge windows differ by one point.

``zvdvc strain RESULTS`` reads a results store, a zvDVC ``.npz`` or a CCPi ``.disp`` and writes
CCPi's CSV layout (``<base>-sw<N>.Lstr.csv``, ``.Estr.csv``, ``.dgrd.csv``) plus ``.strain.npz``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

# quadratic model terms: exponents of (x, y, z)
_TERMS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
_VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]          # exx, eyy, ezz, exy, eyz, exz


@dataclasses.dataclass
class StrainResult:
    pts_in_sw: np.ndarray          # (N,) good points in the strain window
    sw_radius: np.ndarray          # (N,) distance to the farthest point used
    u_fit: np.ndarray              # (N, 3) fitted displacement at the point
    grad: np.ndarray               # (N, 3, 3) du_i / dx_j
    engineering: np.ndarray        # (N, 6) exx, eyy, ezz, exy, eyz, exz (tensor shear, as CCPi)
    lagrangian: np.ndarray         # (N, 6)
    principal_engineering: np.ndarray   # (N, 3) descending
    principal_lagrangian: np.ndarray    # (N, 3)
    residual_rms: np.ndarray       # (N, 3)
    strain_sd: np.ndarray          # (N, 6) sd of the engineering components
    planar_axis: int | None        # 0/1/2 if the cloud lies in a coordinate plane
    window: int = 25


def planar_axis(xyz: np.ndarray) -> int | None:
    """The axis every point shares a coordinate on (CCPi's -xy/-yz/-zx auto-detection), if any."""
    flat = [a for a in range(3) if np.ptp(xyz[:, a]) == 0]
    return flat[0] if len(flat) == 1 else None


def fit_strain(xyz: Any, displacement: Any, status: Any, objmin: Any = None, *, window: int = 25,
               threshold: float = 1.0, refill: bool = False, sigma_u: Any = None, chunk: int = 100_000) -> StrainResult:
    """Strain at every point from its window's GOOD displacements (see module doc)."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    disp = np.asarray(displacement, dtype=np.float64).reshape(-1, 3)
    n = len(xyz)
    good = np.asarray(status).reshape(-1) == 0
    if objmin is not None:
        good &= np.nan_to_num(np.asarray(objmin, dtype=np.float64), nan=np.inf) <= threshold
    plane = planar_axis(xyz)
    axes = [a for a in range(3) if a != plane]
    terms = [t for t in _TERMS if plane is None or t[plane] == 0]
    lin = [terms.index(tuple(int(a == b) for b in range(3))) for a in axes]      # linear-term columns
    k = min(n, window if not refill else max(3 * window, window + 16))
    dist, nbr = cKDTree(xyz).query(xyz, k=k)
    nbr, dist = nbr.reshape(n, k), dist.reshape(n, k)
    use = good[nbr]
    if refill:                                           # the first `window` good neighbours
        use &= np.cumsum(use, axis=1) <= window
    else:                                                # the first `window` slots, good or not
        use[:, window:] = False
    sig = None if sigma_u is None else np.broadcast_to(np.asarray(sigma_u, dtype=np.float64), (n, 3))

    out = {name: np.full(shape, np.nan) for name, shape in (
        ("u_fit", (n, 3)), ("grad", (n, 3, 3)), ("residual_rms", (n, 3)), ("grad_var", (n, 3, 3)))}
    pts_in_sw = use.sum(axis=1)
    sw_radius = np.where(pts_in_sw > 0, np.max(np.where(use, dist, 0.0), axis=1), 0.0)
    for lo in range(0, n, chunk):
        sl = slice(lo, min(lo + chunk, n))
        w = use[sl].astype(np.float64)                                           # (B, k)
        scale = np.maximum(sw_radius[sl], 1e-12)[:, None, None]
        rel = (xyz[nbr[sl]] - xyz[sl, None, :]) / scale                          # (B, k, 3)
        X = np.stack([np.prod(rel ** np.asarray(t), axis=-1) for t in terms], axis=-1)   # (B, k, P)
        d = disp[nbr[sl]].copy()                                                 # (B, k, 3)
        if plane is not None:
            d[..., plane] = 0.0
        A = np.einsum("bk,bkp,bkq->bpq", w, X, X)
        Ainv = np.linalg.pinv(A, rcond=1e-10)
        par = Ainv @ np.einsum("bk,bkp,bkc->bpc", w, X, d)                      # (B, P, 3)
        resid = d - X @ par
        dof = np.maximum(pts_in_sw[sl] - len(terms), 1)[:, None]
        rms = np.sqrt(np.einsum("bk,bkc->bc", w, resid ** 2) / dof)
        # enough points, and a window whose geometry determines every term: the design is centred and
        # scaled, so a tiny smallest eigenvalue means e.g. only two levels along an axis for a quadratic
        ev = np.linalg.eigvalsh(A)
        ok = (pts_in_sw[sl] >= len(terms)) & (ev[:, 0] > 1e-9 * np.maximum(ev[:, -1], 1e-300))
        g = np.zeros((len(w), 3, 3))
        gv = np.zeros((len(w), 3, 3))
        s = scale[:, 0, 0]
        sc2 = rms ** 2 if sig is None else np.where(np.isfinite(sig[sl]), sig[sl], rms) ** 2   # (B, 3) displacement variance
        for j, a in enumerate(axes):
            g[:, :, a] = par[:, lin[j], :] / s[:, None]
            gv[:, :, a] = sc2 * Ainv[:, lin[j], lin[j]][:, None] / (s[:, None] ** 2)
        if plane is not None:
            g[:, plane, :] = 0.0
            gv[:, plane, :] = 0.0
        g[~ok], gv[~ok] = np.nan, np.nan
        out["grad"][sl], out["grad_var"][sl] = g, gv
        out["u_fit"][sl] = np.where(ok[:, None], par[:, 0, :], np.nan)
        out["residual_rms"][sl] = np.where(ok[:, None], rms, np.nan)

    G = out["grad"]
    eng_t = 0.5 * (G + np.swapaxes(G, 1, 2))
    lag_t = eng_t + 0.5 * np.einsum("nki,nkj->nij", G, G)
    voigt = lambda T: np.stack([T[:, i, j] for i, j in _VOIGT], axis=1)          # noqa: E731
    gv = out["grad_var"]
    sd = np.stack([np.sqrt(gv[:, i, i]) if i == j else 0.5 * np.sqrt(gv[:, i, j] + gv[:, j, i]) for i, j in _VOIGT], axis=1)
    return StrainResult(pts_in_sw=pts_in_sw, sw_radius=sw_radius, u_fit=out["u_fit"], grad=G,
                        engineering=voigt(eng_t), lagrangian=voigt(lag_t),
                        principal_engineering=_principal(eng_t, plane), principal_lagrangian=_principal(lag_t, plane),
                        residual_rms=out["residual_rms"], strain_sd=sd, planar_axis=plane, window=window)


def _principal(T: np.ndarray, plane: int | None = None) -> np.ndarray:
    """Principal values, descending; for a planar cloud the two in-plane ones, then 0 (CCPi's convention)."""
    ok = np.isfinite(T).all(axis=(1, 2))
    out = np.full((len(T), 3), np.nan)
    if not ok.any():
        return out
    if plane is None:
        out[ok] = np.linalg.eigvalsh(T[ok])[:, ::-1]
    else:
        keep = [a for a in range(3) if a != plane]
        out[ok, :2] = np.linalg.eigvalsh(T[ok][:, keep][:, :, keep])[:, ::-1]
        out[ok, 2] = 0.0
    return out


# --------------------------------------------------------------------------- files


def _g(v: float) -> str:
    return "nan" if not np.isfinite(v) else f"{v:g}"


def write_csv(path: str | Path, point_id: Any, xyz: Any, r: StrainResult, which: str) -> None:
    """CCPi's strain CSV (``Lstr``, ``Estr`` or ``dgrd``), plus zvDVC's uncertainty columns at the end."""
    head = "n,x,y,z,u_fit,v_fit,w_fit,pts_in_sw,sw_radius,"
    if which == "dgrd":
        head += "ux,uy,uz,vx,vy,vz,wx,wy,wz"
        vals = r.grad.reshape(len(r.grad), 9)
    else:
        head += "exx,eyy,ezz,exy,eyz,exz,ep1,ep2,ep3,sd_exx,sd_eyy,sd_ezz,sd_exy,sd_eyz,sd_exz"
        strain, principal = (r.lagrangian, r.principal_lagrangian) if which == "Lstr" else (r.engineering, r.principal_engineering)
        vals = np.concatenate([strain, principal, r.strain_sd], axis=1)
    with open(path, "w") as fh:
        fh.write(head + "\n")
        for i in range(len(point_id)):
            row = [str(int(point_id[i])), *map(_g, xyz[i]), *map(_g, r.u_fit[i]), str(int(r.pts_in_sw[i])),
                   _g(r.sw_radius[i]), *map(_g, vals[i])]
            fh.write(",".join(row) + "\n")


def compute_strain(results: str | Path, *, window: int = 25, threshold: float = 1.0, refill: bool = False,
                   sigma_u: float | None = None, out_base: str | Path | None = None,
                   outputs: tuple[str, ...] = ("Lstr",)) -> dict[str, Any]:
    """Strain for a results store, zvDVC ``.npz`` or CCPi ``.disp``; writes the CSVs and ``.strain.npz``."""
    from zvdvc.bench.metrics import load_results

    res = load_results(results)
    if sigma_u is None and "displacement_sd" in res:              # the run's repeat-solve uncertainty, per point
        sigma_u = res["displacement_sd"]
    r = fit_strain(res["xyz"], res["displacement"], res["status"], res.get("objmin"), window=window,
                   threshold=threshold, refill=refill, sigma_u=sigma_u)
    src = Path(results)
    base = Path(out_base) if out_base else src.with_suffix("") if src.suffix in (".disp", ".npz") else src.parent / src.stem
    files = {}
    for which in outputs:
        path = base.with_name(f"{base.name}-sw{window}.{which}.csv")
        write_csv(path, res["point_id"], res["xyz"], r, which)
        files[which] = str(path)
    npz = base.with_name(f"{base.name}-sw{window}.strain.npz")
    np.savez(npz, point_id=res["point_id"], xyz=res["xyz"], **{k: v for k, v in dataclasses.asdict(r).items()
                                                               if isinstance(v, np.ndarray)})
    files["npz"] = str(npz)
    finite = np.isfinite(r.engineering).all(axis=1)
    return {"files": files, "points": int(len(r.pts_in_sw)), "with_strain": int(finite.sum()),
            "planar_axis": r.planar_axis, "median_strain_sd": np.nanmedian(r.strain_sd, axis=0).tolist()}
