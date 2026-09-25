"""Speckle phantoms and analytic displacement fields.

Ground truth is what makes the MVP a feasibility *test*: every accuracy claim
in docs/MVP_PLAN.md is measured against these fields, not against another code.

Volumes are generated and warped chunk by chunk (overlap = filter support +
max displacement), straight into sharded OME-Zarr, so 4096^3 cases can be
built on a workstation with bounded memory.

Fields (all analytic, with gradients for strain checks):

``rigid``      translation + small rotation. Tests seeding, range handling and 6-DOF.
``affine``     uniform strain up to ~1 %. Tests 12-DOF and strain post-processing.
``sinusoid``   ``u = A sin(2 pi x / lambda)`` along one axis. Tests spatial resolution versus subvolume size.
``inclusion``  Eshelby-like stiff sphere in a strained matrix. A realistic mix of gradients.

The deformed volume uses backward mapping: ``g(x) = f(X)`` where ``x = X +
u(X)``, with ``X`` found by fixed-point iteration. Interpolation is
quintic B-spline, so the phantom's interpolation error sits well below that of
the tricubic being tested. Optional Gaussian noise, independent in the two
volumes, is added before quantisation (Poisson noise: later, if needed).

The speckle is Gaussian-filtered white noise. The noise is drawn per fixed
64^3 block from a seed derived from the block index, so any region of the
volume can be regenerated on its own and chunked writing gives bit-identical
volumes whatever the chunking. With the default ``feature_size = 3`` the
filter has sigma 1.5 voxels, which keeps the texture band-limited (spectrum
at Nyquist ~1e-5) so the B-spline resampling is effectively exact.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pydvc.config import ClusterSpec, RunConfig, SearchSpec, SeedingSpec, SubvolumeSpec, VolumeSpec
from pydvc.geometry.box import Box
from pydvc.geometry.warp import rotation_matrix, strain_tensor

FieldKind = Literal["rigid", "affine", "sinusoid", "inclusion"]

_NOISE_BLOCK = 64          # white noise is drawn per block of this edge
_WRITE_BLOCK = 128         # voxels generated at a time
_SPLINE_MARGIN = 16        # context around a resampled region for the quintic prefilter
_CONTRAST = 0.12           # speckle standard deviation as a fraction of full scale
_TRUNCATE = 4.0            # Gaussian filter support, in sigmas


def _vec(params: dict[str, Any], key: str, n: int, default: float = 0.0) -> np.ndarray:
    value = params.get(key)
    return np.full(n, default, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64).reshape(n)


@dataclass(frozen=True)
class DisplacementField:
    """Analytic displacement ``u(X)`` of reference position ``X``, in voxels.

    Parameters (all optional unless noted; vectors are ``(x, y, z)``):

    ``rigid``      ``translation``; ``rotation`` = CCPi angles ``(phi, the, psi)`` in
                   radians; ``centre``. ``u = t + (R - I)(X - c)``, so a 6-DOF
                   search recovers ``u(P)`` and the angles exactly.
    ``affine``     ``translation``; ``strain`` = ``(exx, eyy, ezz, exy, eyz, exz)``
                   (tensor shears) or a full 3x3 ``gradient``; ``centre``.
                   ``u = t + G (X - c)``.
    ``sinusoid``   ``amplitude`` (required), ``wavelength`` (required); ``axis``
                   the coordinate it varies along and ``component`` the displacement
                   component (0/1/2 = x/y/z, both default 0); ``phase``; ``translation``.
    ``inclusion``  ``centre``, ``radius`` (required); far-field ``strain``;
                   ``ratio`` = inclusion strain / far-field strain (default 0.25).
                   ``u = t + s(r) E (X - c)`` with ``s = ratio`` inside and
                   ``s = 1 - (1 - ratio) (a / r)^3`` outside: continuous, with the
                   ``r^-3`` decay of an Eshelby far field.
    """

    kind: FieldKind
    params: dict[str, Any] = field(default_factory=dict)

    def _gradient_matrix(self) -> np.ndarray:
        p = self.params
        if self.kind == "rigid":
            return rotation_matrix(_vec(p, "rotation", 3)[None])[0] - np.eye(3)
        if "gradient" in p:
            return np.asarray(p["gradient"], dtype=np.float64).reshape(3, 3)
        return strain_tensor(_vec(p, "strain", 6)[None])[0]

    def __call__(self, xyz: Any) -> Any:
        """(N, 3) -> (N, 3) displacement."""
        X = np.asarray(xyz, dtype=np.float64)
        p = self.params
        t = _vec(p, "translation", 3)
        if self.kind in ("rigid", "affine"):
            return t + (X - _vec(p, "centre", 3)) @ self._gradient_matrix().T
        if self.kind == "sinusoid":
            axis, comp = int(p.get("axis", 0)), int(p.get("component", 0))
            k = 2.0 * math.pi / float(p["wavelength"])
            u = np.broadcast_to(t, X.shape).copy()
            u[..., comp] += float(p["amplitude"]) * np.sin(k * X[..., axis] + float(p.get("phase", 0.0)))
            return u
        if self.kind == "inclusion":
            d = X - _vec(p, "centre", 3)
            return t + self._shape(d)[0][..., None] * (d @ self._gradient_matrix().T)
        raise ValueError(f"unknown field kind {self.kind!r}")

    def _shape(self, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inclusion profile ``s(r)`` and ``ds/dr``."""
        a, beta = float(self.params["radius"]), float(self.params.get("ratio", 0.25))
        r = np.linalg.norm(d, axis=-1)
        outside = r > a
        rs = np.where(outside, r, a)
        s = np.where(outside, 1.0 - (1.0 - beta) * (a / rs) ** 3, beta)
        ds = np.where(outside, 3.0 * (1.0 - beta) * a**3 / rs**4, 0.0)
        return s, ds

    def gradient(self, xyz: Any) -> Any:
        """(N, 3) -> (N, 3, 3) displacement gradient ``du_i / dX_j``, for strain checks."""
        X = np.asarray(xyz, dtype=np.float64)
        p = self.params
        if self.kind in ("rigid", "affine"):
            return np.broadcast_to(self._gradient_matrix(), X.shape[:-1] + (3, 3)).copy()
        if self.kind == "sinusoid":
            axis, comp = int(p.get("axis", 0)), int(p.get("component", 0))
            k = 2.0 * math.pi / float(p["wavelength"])
            G = np.zeros(X.shape[:-1] + (3, 3))
            G[..., comp, axis] = float(p["amplitude"]) * k * np.cos(k * X[..., axis] + float(p.get("phase", 0.0)))
            return G
        if self.kind == "inclusion":
            d = X - _vec(p, "centre", 3)
            E = self._gradient_matrix()
            s, ds = self._shape(d)
            r = np.maximum(np.linalg.norm(d, axis=-1), 1e-12)
            Ed = d @ E.T
            return s[..., None, None] * E + (ds / r)[..., None, None] * Ed[..., :, None] * d[..., None, :]
        raise ValueError(f"unknown field kind {self.kind!r}")

    def inverse_map(self, x: np.ndarray, *, iterations: int = 12) -> np.ndarray:
        """Reference positions ``X`` with ``X + u(X) = x``, by fixed-point iteration (needs ``|grad u| < 1``)."""
        X = x - self(x)
        for _ in range(iterations):
            X = x - self(X)
        return X


def default_field(kind: FieldKind, shape_zyx: tuple[int, int, int]) -> DisplacementField:
    """The synthetic cases' fields, scaled so ``|u|`` stays within about 3 voxels."""
    ext = np.asarray(shape_zyx[::-1], dtype=np.float64) - 1.0
    centre = tuple(0.5 * ext)
    half = 0.5 * float(ext.min())
    g = 1.5 / half                                       # gradient giving ~1.5 voxels at the edge
    if kind == "rigid":
        return DisplacementField("rigid", {"translation": (1.6, -0.9, 0.7), "rotation": (0.6 * g, -0.5 * g, 0.7 * g), "centre": centre})
    if kind == "affine":
        strain = (0.8 * g, -0.5 * g, 0.4 * g, 0.2 * g, 0.0, -0.3 * g)
        return DisplacementField("affine", {"translation": (1.2, -0.4, 0.3), "strain": strain, "centre": centre})
    if kind == "sinusoid":
        # one period across x; shorter wavelengths probe spatial resolution rather than accuracy
        return DisplacementField("sinusoid", {"amplitude": 1.0, "wavelength": float(ext[0] + 1), "axis": 0, "component": 0})
    if kind == "inclusion":
        strain = (1.5 * g, -0.75 * g, -0.75 * g, 0.0, 0.0, 0.0)
        return DisplacementField("inclusion", {"centre": centre, "radius": half / 3.0, "strain": strain, "ratio": 0.2})
    raise ValueError(f"unknown field kind {kind!r}")


# ---------------------------------------------------------------------------- speckle


def _kernel_norm(sigma: float) -> float:
    """Standard deviation of unit white noise after the 3-D Gaussian filter."""
    from scipy.ndimage import gaussian_filter1d

    radius = int(_TRUNCATE * sigma + 0.5)
    delta = np.zeros(2 * radius + 1)
    delta[radius] = 1.0
    k = gaussian_filter1d(delta, sigma, truncate=_TRUNCATE, mode="constant")
    return float(np.sqrt((k**2).sum()) ** 3)


def _block_noise(box: Box, seed: int, stream: int) -> np.ndarray:
    """Unit white noise over ``box`` (any integer box, negative allowed), reproducible per 64^3 block."""
    out = np.empty(box.shape)
    b = _NOISE_BLOCK
    lo_blk = [l // b for l in box.lo]
    hi_blk = [-(-h // b) for h in box.hi]
    for bz in range(lo_blk[0], hi_blk[0]):
        for by in range(lo_blk[1], hi_blk[1]):
            for bx in range(lo_blk[2], hi_blk[2]):
                blk = Box((bz * b, by * b, bx * b), ((bz + 1) * b, (by + 1) * b, (bx + 1) * b))
                part = blk.intersect(box)
                if part is None:
                    continue
                key = [seed, stream] + [v + (1 << 30) for v in (bz, by, bx)]
                noise = np.random.default_rng(key).standard_normal((b, b, b))
                out[part.slices(box.lo)] = noise[part.slices(blk.lo)]
    return out


def speckle_field(box: Box, *, feature_size: float = 3.0, seed: int = 0) -> np.ndarray:
    """Zero-mean, unit-variance speckle over ``box`` of the infinite, seed-defined pattern."""
    from scipy.ndimage import gaussian_filter

    sigma = 0.5 * feature_size
    margin = int(_TRUNCATE * sigma + 0.5)
    noise = _block_noise(box.grow(margin), seed, stream=0)
    filtered = gaussian_filter(noise, sigma, truncate=_TRUNCATE, mode="constant")
    return filtered[(slice(margin, -margin),) * 3] / _kernel_norm(sigma)


def _full_scale(dtype: np.dtype) -> float:
    return float(np.iinfo(dtype).max) if dtype.kind in "ui" else 1.0


def _to_dtype(z: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Map unit-variance speckle to the dtype: mid-grey plus ``_CONTRAST`` of full scale per sigma."""
    full = _full_scale(dtype)
    v = full * (0.5 + _CONTRAST * z)
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        return np.clip(np.rint(v), info.min, info.max).astype(dtype)
    return v.astype(dtype)


def speckle(
    shape_zyx: tuple[int, int, int],
    *,
    feature_size: float = 3.0,
    dtype: str = "uint16",
    seed: int = 0,
) -> Any:
    """Band-limited random speckle (filtered noise), scaled to the dtype's range."""
    z = speckle_field(Box((0, 0, 0), tuple(shape_zyx)), feature_size=feature_size, seed=seed)
    return _to_dtype(z, np.dtype(dtype))


def _resample(field: DisplacementField, box: Box, *, feature_size: float, seed: int) -> np.ndarray:
    """Unit-variance deformed speckle over ``box``: ``f(X(x))`` for every voxel ``x``."""
    from scipy.ndimage import map_coordinates, spline_filter

    zz, yy, xx = np.meshgrid(*(np.arange(l, h, dtype=np.float64) for l, h in zip(box.lo, box.hi)), indexing="ij")
    x = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    X = field.inverse_map(x)
    src = Box(
        tuple(int(math.floor(v)) - _SPLINE_MARGIN for v in X.min(axis=0)[::-1]),
        tuple(int(math.ceil(v)) + 1 + _SPLINE_MARGIN for v in X.max(axis=0)[::-1]),
    )
    coeffs = spline_filter(speckle_field(src, feature_size=feature_size, seed=seed), order=5, mode="mirror")
    coords = (X[:, ::-1] - np.asarray(src.lo, dtype=np.float64)).T          # (3, N) in (z, y, x)
    return map_coordinates(coeffs, coords, order=5, prefilter=False, mode="mirror").reshape(box.shape)


def warp_volume(ref: Any, field: DisplacementField, *, noise_sigma: float = 0.0, seed: int = 1) -> Any:
    """Deform an in-memory volume: ``g(x) = ref(X)`` with ``x = X + u(X)``, quintic B-spline, float64 out.

    ``noise_sigma`` is an absolute standard deviation in ``ref``'s units. For
    large cases use :func:`make_case`, which regenerates the speckle chunk by
    chunk instead of resampling one array.
    """
    from scipy.ndimage import map_coordinates

    ref = np.asarray(ref, dtype=np.float64)
    zz, yy, xx = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in ref.shape), indexing="ij")
    x = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    X = field.inverse_map(x)
    out = map_coordinates(ref, X[:, ::-1].T, order=5, mode="mirror").reshape(ref.shape)
    if noise_sigma > 0:
        out += np.random.default_rng(seed).normal(0.0, noise_sigma, out.shape)
    return out


# ---------------------------------------------------------------------------- cases


def _ref_block(box: Box, feature_size: float, seed: int, noise_sigma: float, dtype: np.dtype) -> np.ndarray:
    z = speckle_field(box, feature_size=feature_size, seed=seed)
    if noise_sigma > 0:
        z = z + _block_noise(box, seed, 1) * (noise_sigma / _CONTRAST)
    return _to_dtype(z, dtype)


def _def_block(
    box: Box, field: DisplacementField, feature_size: float, seed: int, noise_sigma: float, dtype: np.dtype
) -> np.ndarray:
    z = _resample(field, box, feature_size=feature_size, seed=seed)
    if noise_sigma > 0:
        z = z + _block_noise(box, seed, 2) * (noise_sigma / _CONTRAST)
    return _to_dtype(z, dtype)


def _write_volume(
    path: Path, shape: tuple[int, int, int], dtype: np.dtype, block_fn, args: tuple, *, chunk: int, shard: int, workers: int
) -> None:
    """Generate a volume one write block at a time (in parallel) and store it one whole shard at a time."""
    from concurrent.futures import ProcessPoolExecutor

    from pydvc.io.volume import create_ome_zarr, iter_blocks

    array = create_ome_zarr(path, shape, dtype, chunk=chunk, shard=shard)
    shard_shape = tuple(int(s) for s in array.shards)
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        for sbox in iter_blocks(shape, shard_shape):
            buf = np.empty(sbox.shape, dtype=dtype)
            boxes = [
                (bbox, Box(tuple(a + b for a, b in zip(sbox.lo, bbox.lo)), tuple(a + b for a, b in zip(sbox.lo, bbox.hi))))
                for bbox in iter_blocks(sbox.shape, (_WRITE_BLOCK,) * 3)
            ]
            if pool is None:
                for bbox, box in boxes:
                    buf[bbox.slices()] = block_fn(box, *args)
            else:
                futures = [(bbox, pool.submit(block_fn, box, *args)) for bbox, box in boxes]
                for bbox, fut in futures:
                    buf[bbox.slices()] = fut.result()
            array[sbox.slices()] = buf
    finally:
        if pool is not None:
            pool.shutdown()


def _max_displacement(field: DisplacementField, shape_zyx: tuple[int, int, int]) -> float:
    axes = [np.linspace(0.0, n - 1.0, 17) for n in shape_zyx[::-1]]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    probe = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return float(np.linalg.norm(field(probe), axis=1).max())


def case_points(
    shape_zyx: tuple[int, int, int], spacing: float, margin: float
) -> np.ndarray:
    """Lattice through the volume centre, keeping points at least ``margin`` from every face."""
    from pydvc.geometry.pointgrid import grid_in_mask

    pts = grid_in_mask(shape_zyx, (spacing,) * 3)
    hi = np.asarray(shape_zyx[::-1], dtype=np.float64) - 1.0 - margin
    return pts[((pts >= margin) & (pts <= hi)).all(axis=1)]


def make_case(
    out_dir: str | Path,
    *,
    shape_zyx: tuple[int, int, int],
    field: DisplacementField,
    spacing: float = 16.0,
    dtype: str = "uint16",
    noise_sigma: float = 0.0,
    chunk: int = 128,
    shard: int = 1024,
    subvolume: SubvolumeSpec | None = None,
    search: SearchSpec | None = None,
    feature_size: float = 3.0,
    seed: int = 0,
    workers: int | None = None,
) -> Path:
    """Write ``ref.ome.zarr``, ``def.ome.zarr``, ``points.zarrvectors`` (and the same points as
    ``points.roi`` for CCPi), ``truth.npz`` and ``config.yaml``; return the config path.

    ``noise_sigma`` is the Gaussian noise standard deviation as a fraction of the
    dtype's full scale (the speckle's own standard deviation is 0.12), drawn
    independently for the two volumes. Points form a ``spacing`` lattice kept
    far enough from the faces that every subvolume stays inside both volumes.
    The defaults for ``subvolume`` and ``search`` are case S's (docs/MVP_PLAN.md):
    sphere 32 / 2 000 samples, 12-DOF, ZNSSD, tricubic, ``disp_max`` 8.
    Blocks are generated by ``workers`` processes (default: every core); the
    output does not depend on it. The point store's cells are a quarter of the volume per axis, so the one
    tile (the whole volume) holds 4^3 cells.
    """
    from pydvc.geometry.templates import make_template
    from pydvc.io.ccpi import write_roi
    from pydvc.io.pointcloud import chunk_shape_for, write_pointcloud_store

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(n) for n in shape_zyx)
    dt = np.dtype(dtype)
    subvolume = subvolume or SubvolumeSpec(geometry="sphere", size=32, n_samples=2000)
    search = search or SearchSpec(dof=12, objective="znssd", interpolation="tricubic", disp_max=8.0)
    full = _full_scale(dt)

    import os

    workers = workers or os.cpu_count() or 1
    kw = dict(chunk=chunk, shard=shard, workers=workers)
    _write_volume(out / "ref.ome.zarr", shape, dt, _ref_block, (feature_size, seed, noise_sigma, dt), **kw)
    _write_volume(out / "def.ome.zarr", shape, dt, _def_block, (field, feature_size, seed, noise_sigma, dt), **kw)

    umax = _max_displacement(field, shape)
    if umax + 1.0 > search.disp_max:
        raise ValueError(f"field reaches |u| = {umax:.2f} voxels; raise search.disp_max above {umax + 1:.1f}")
    margin = make_template(subvolume).extent() + umax + 3.0
    xyz = case_points(shape, spacing, margin)
    if len(xyz) == 0:
        raise ValueError(f"no points fit: volume {shape} is too small for margin {margin:.1f}")
    point_id = np.arange(1, len(xyz) + 1, dtype=np.int64)
    write_roi(out / "points.roi", point_id, xyz)
    tile = tuple(int(n) for n in shape)
    bounds = ((0.0, 0.0, 0.0), tuple(float(n) for n in shape[::-1]))
    write_pointcloud_store(out / "points.zarrvectors", xyz, point_id, bounds=bounds, chunk_shape=chunk_shape_for(tile))
    np.savez(
        out / "truth.npz",
        point_id=point_id,
        xyz=xyz,
        displacement=field(xyz),
        gradient=field.gradient(xyz),
        field=json.dumps({"kind": field.kind, "params": _jsonable(field.params)}),
        noise_sigma=noise_sigma,
        full_scale=full,
    )
    cfg = RunConfig(
        volumes=VolumeSpec(reference=str(out / "ref.ome.zarr"), deformed=str(out / "def.ome.zarr")),
        points=str(out / "points.zarrvectors"),
        output=str(out / "results.zarrvectors"),
        subvolume=subvolume,
        search=search,
        seeding=SeedingSpec(strategy="wavefront"),
        cluster=ClusterSpec(tile_shape=shape, prefetch_depth=1),
        workdir=str(out / "run"),
    )
    cfg.to_yaml(out / "config.yaml")
    return out / "config.yaml"


def _jsonable(params: dict[str, Any]) -> dict[str, Any]:
    return {k: (np.asarray(v).tolist() if isinstance(v, (tuple, list, np.ndarray)) else v) for k, v in params.items()}


def load_field(truth_path: str | Path) -> DisplacementField:
    """The analytic field a case was generated with, from its ``truth.npz``."""
    with np.load(truth_path) as t:
        spec = json.loads(str(t["field"]))
    return DisplacementField(spec["kind"], spec["params"])
