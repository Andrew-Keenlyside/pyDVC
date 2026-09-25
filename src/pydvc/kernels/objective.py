"""Batched matching criteria, as defined in CCPi ``ObjectiveFunctions.cpp`` (after Pan 2010).

For reference samples ``f`` and target samples ``g`` of one subvolume (bars
are means, ``~`` is mean-subtracted)::

    sad    sum |g - f|
    ssd    sum (g - f)^2
    zssd   sum (g~ - f~)^2
    nssd   1/2 * sum (g/|g| - f/|f|)^2          in [0, 1]
    znssd  1/4 * sum (g~/|g~| - f~/|f~|)^2      in [0, 1]

The residual vectors are CCPi's (the bracketed terms). Gauss-Newton minimises
their sum of squares, which for ``sad`` means it steps as if the objective
were ``ssd``, exactly as CCPi does. The reported objective is always the exact
value above.

The normal equations (:func:`normal_equations`) use the **exact** Jacobian of
the residual vector, normalisation included. CCPi differentiates its full
objective numerically, so this is the parity choice. Holding the target mean
and norm fixed within an iteration (common FA-GN/IC-GN practice) moves the
fixed point for NSSD/ZNSSD by a term proportional to ``(1 - rho) * d|g~|/dp``,
which is not zero under noise. The exact form costs only two more per-point
sums, ``sum J`` and ``sum g J``, which the fused kernel accumulates in the
same pass. With ``J`` the target Jacobian (B, M, ndof) and ``g^ = g~/|g~|``:

    zssd   H = J'J - M m m',                 b = J'r
    nssd   H = (J'J - a a') / |g|^2,         b = (J'r - a (g^.r)) / |g|
    znssd  H = (J'J - M m m' - a a') / |g~|^2,  b = (J'r - a (g^.r)) / |g~|

where ``m`` is the sample mean of ``J`` and ``a = g^'J``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydvc.kernels.xp import xp_of

Kind = Literal["sad", "ssd", "zssd", "nssd", "znssd"]
KINDS = ("sad", "ssd", "zssd", "nssd", "znssd")
_SCALE = {"nssd": 0.5, "znssd": 0.25}
_CENTRED = ("zssd", "znssd")
_NORMALISED = ("nssd", "znssd")


def _weights(values: Any, mask: Any) -> Any:
    xp = xp_of(values)
    if mask is None:
        return None
    return xp.asarray(mask, dtype=values.dtype)


def _sum(x: Any, w: Any) -> Any:
    return x.sum(axis=-1) if w is None else (x * w).sum(axis=-1)


def _count(x: Any, w: Any) -> Any:
    xp = xp_of(x)
    return xp.full(x.shape[:-1], x.shape[-1], dtype=x.dtype) if w is None else w.sum(axis=-1)


def _normalise(v: Any, kind: Kind, w: Any) -> tuple[Any, Any]:
    """Mean-subtract and/or scale to unit norm as ``kind`` requires; returns (vector, norm)."""
    xp = xp_of(v)
    if kind in _CENTRED:
        v = v - (_sum(v, w) / xp.maximum(_count(v, w), 1))[..., None]
    if w is not None:
        v = v * w
    norm = xp.sqrt((v * v).sum(axis=-1))
    if kind in _NORMALISED:
        return v / xp.where(norm > 0, norm, 1.0)[..., None], norm
    return v, norm


def _check(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown objective {kind!r}")


def objective(ref: Any, tar: Any, kind: Kind, *, mask: Any = None) -> Any:
    """(B, M), (B, M) -> (B,). ``mask`` excludes samples (e.g. outside the brick) from every sum."""
    _check(kind)
    xp = xp_of(ref, tar)
    w = _weights(tar, mask)
    if kind == "sad":
        return _sum(xp.abs(tar - ref), w)
    r, _ = residuals(ref, tar, kind, mask=mask)
    return _SCALE.get(kind, 1.0) * (r * r).sum(axis=-1)


def residuals(ref: Any, tar: Any, kind: Kind, *, mask: Any = None) -> tuple[Any, Any]:
    """(B, M) residuals and (B,) per-point scale (``1/|g~|`` etc.) applied to the target gradient in the Jacobian.

    Masked samples have zero residual. For ``sad`` the residual is the ``ssd`` one.
    """
    _check(kind)
    xp = xp_of(ref, tar)
    w = _weights(tar, mask)
    f, _ = _normalise(ref, kind, w)
    g, gnorm = _normalise(tar, kind, w)
    if kind in _NORMALISED:
        scale = 1.0 / xp.where(gnorm > 0, gnorm, 1.0)
    else:
        scale = xp.ones(tar.shape[:-1], dtype=tar.dtype)
    return g - f, scale


def reference_stats(ref: Any, kind: Kind) -> Any:
    """Per-point reference mean and norm, computed once per point and reused every iteration.

    (B, 2): column 0 is the mean subtracted (0 unless zero-mean), column 1 the
    norm divided by (1 unless normalised), matching :func:`residuals`.
    """
    _check(kind)
    xp = xp_of(ref)
    mean = ref.mean(axis=-1) if kind in _CENTRED else xp.zeros(ref.shape[:-1], dtype=ref.dtype)
    if kind in _NORMALISED:
        c = ref - mean[..., None]
        norm = xp.sqrt((c * c).sum(axis=-1))
    else:
        norm = xp.ones(ref.shape[:-1], dtype=ref.dtype)
    return xp.stack([mean, norm], axis=-1)


def normal_equations(ref: Any, tar: Any, jac: Any, kind: Kind, *, mask: Any = None) -> tuple[Any, Any, Any]:
    """Gauss-Newton system for one iteration.

    ``jac`` (B, M, ndof) is ``d tar / d p``, i.e. ``grad(g)(x')^T dx'/dp``.
    Returns ``H`` (B, ndof, ndof), ``b`` (B, ndof) and the exact objective (B,),
    where the step is ``H dp = -b``. ``H = Jr'Jr`` and ``b = Jr'r`` for the exact
    Jacobian ``Jr`` of :func:`residuals`, computed from per-point sums only.
    """
    _check(kind)
    xp = xp_of(ref, tar, jac)
    w = _weights(tar, mask)
    r, _ = residuals(ref, tar, kind, mask=mask)
    Jw = jac if w is None else jac * w[..., None]
    H = xp.einsum("bmk,bml->bkl", Jw, jac)
    b = xp.einsum("bmk,bm->bk", Jw, r)
    if kind in _CENTRED:
        n = xp.maximum(_count(tar, w), 1)
        m = Jw.sum(axis=1) / n[..., None]
        H = H - n[..., None, None] * m[:, :, None] * m[:, None, :]
    if kind in _NORMALISED:
        ghat, gnorm = _normalise(tar, kind, w)
        inv = 1.0 / xp.where(gnorm > 0, gnorm, 1.0)
        a = xp.einsum("bm,bmk->bk", ghat, jac)
        H = (H - a[:, :, None] * a[:, None, :]) * (inv * inv)[:, None, None]
        b = (b - a * (ghat * r).sum(axis=-1)[..., None]) * inv[..., None]
    if kind == "sad":
        obj = _sum(xp.abs(tar - ref), w)
    else:
        obj = _SCALE.get(kind, 1.0) * (r * r).sum(axis=-1)
    return H, b, obj


# ----------------------------------------------------------------------------- one-pass sums
#
# Every backend reduces a point's samples to the same fixed set of sums, from which
# the normal equations follow exactly (:func:`normal_equations_from_sums`). The fused
# CUDA kernel, the numba CPU kernel and the unfused xp path all produce this row, so
# they share one formula and one test. Per sample, with the reference term ``q`` and
# the shifted target ``v = g - s`` (:func:`reference_terms`) and the Jacobian row ``j``:
#
#     JJ   upper triangle of sum j j'     (ndof (ndof + 1) / 2, row-major)
#     J    sum j                           (ndof)
#     vJ   sum v j                         (ndof)
#     qJ   sum q j                         (ndof)
#     v, vv, vq, q, qq, n, absdiff         sum v, v^2, v q, q, q^2, count, |v - q|
#
# The shift ``s`` (the reference mean, for zero-mean objectives) keeps ``sum v^2 - n
# vbar^2`` free of cancellation in float32. One pass suffices: no objective needs the
# target mean or norm before the sums exist.

_SCALARS = ("v", "vv", "vq", "q", "qq", "n", "absdiff")


def n_upper(ndof: int) -> int:
    return ndof * (ndof + 1) // 2


def sum_layout(ndof: int) -> dict[str, slice]:
    """Column ranges of the per-point sums row."""
    t = n_upper(ndof)
    out = {"JJ": slice(0, t), "J": slice(t, t + ndof), "vJ": slice(t + ndof, t + 2 * ndof), "qJ": slice(t + 2 * ndof, t + 3 * ndof)}
    base = t + 3 * ndof
    for k, name in enumerate(_SCALARS):
        out[name] = slice(base + k, base + k + 1)
    return out


def n_sums(ndof: int) -> int:
    return n_upper(ndof) + 3 * ndof + len(_SCALARS)


def reference_terms(ref: Any, kind: Kind) -> tuple[Any, Any]:
    """Per-sample reference term ``q`` (B, M) and per-point target shift ``s`` (B,).

    ``q`` is the reference side of the residual: ``f`` (sad, ssd), ``f - fbar``
    (zssd), ``f / |f|`` (nssd), ``(f - fbar) / |f~|`` (znssd). ``s`` is ``fbar`` for
    the zero-mean objectives and 0 otherwise.
    """
    _check(kind)
    xp = xp_of(ref)
    shift = ref.mean(axis=-1) if kind in _CENTRED else xp.zeros(ref.shape[:-1], dtype=ref.dtype)
    q, _ = _normalise(ref, kind, None)
    return q, shift


def accumulate_sums(q: Any, v: Any, jac: Any, kind: Kind) -> Any:
    """The sums row (B, n_sums) from dense per-sample arrays: the unfused path's reduction."""
    xp = xp_of(q, v, jac)
    B, M, ndof = jac.shape
    rows, cols = _triu(ndof)
    lay = sum_layout(ndof)
    out = xp.zeros((B, n_sums(ndof)), dtype=jac.dtype)
    JJ = xp.einsum("bmk,bml->bkl", jac, jac)
    out[:, lay["JJ"]] = JJ[:, rows, cols]
    out[:, lay["J"]] = jac.sum(axis=1)
    out[:, lay["vJ"]] = xp.einsum("bm,bmk->bk", v, jac)
    out[:, lay["qJ"]] = xp.einsum("bm,bmk->bk", q, jac)
    out[:, lay["v"]] = v.sum(axis=1, keepdims=True)
    out[:, lay["vv"]] = (v * v).sum(axis=1, keepdims=True)
    out[:, lay["vq"]] = (v * q).sum(axis=1, keepdims=True)
    out[:, lay["q"]] = q.sum(axis=1, keepdims=True)
    out[:, lay["qq"]] = (q * q).sum(axis=1, keepdims=True)
    out[:, lay["n"]] = M
    out[:, lay["absdiff"]] = xp.abs(v - q).sum(axis=1, keepdims=True)
    return out


def _triu(ndof: int) -> tuple[list[int], list[int]]:
    rows, cols = [], []
    for i in range(ndof):
        for j in range(i, ndof):
            rows.append(i)
            cols.append(j)
    return rows, cols


def normal_equations_from_sums(sums: Any, shift: Any, kind: Kind, ndof: int) -> tuple[Any, Any, Any, Any, Any]:
    """``H`` (B, ndof, ndof), ``b`` (B, ndof), objective (B,), ``sum |grad g|^2`` and ``sum g^2`` (B,).

    Identical, up to rounding, to :func:`normal_equations` on the dense arrays
    (exact Jacobian of the normalised residual). The last two feed the
    featureless-subvolume test.
    """
    _check(kind)
    xp = xp_of(sums)
    lay = sum_layout(ndof)

    def s(name: str) -> Any:
        return sums[:, lay[name]]

    rows, cols = _triu(ndof)
    B = sums.shape[0]
    H = xp.zeros((B, ndof, ndof), dtype=sums.dtype)
    H[:, rows, cols] = s("JJ")
    H[:, cols, rows] = s("JJ")
    SJ, SvJ, SqJ = s("J"), s("vJ"), s("qJ")
    Sv, Svv, Svq, Sq, Sqq, n, Sabs = (s(k)[:, 0] for k in _SCALARS)
    one = xp.ones_like(n)
    if kind in ("sad", "ssd"):
        b = SvJ - SqJ
        obj = Sabs if kind == "sad" else Svv - 2.0 * Svq + Sqq
    elif kind == "zssd":
        vbar = Sv / xp.maximum(n, one)
        m = SJ / xp.maximum(n, one)[:, None]
        H = H - n[:, None, None] * m[:, :, None] * m[:, None, :]
        b = SvJ - vbar[:, None] * SJ - SqJ + m * Sq[:, None]
        obj = (Svv - n * vbar * vbar) - 2.0 * (Svq - vbar * Sq) + Sqq
    else:
        if kind == "nssd":
            vbar = xp.zeros_like(Sv)
            norm2 = Svv
        else:
            vbar = Sv / xp.maximum(n, one)
            norm2 = Svv - n * vbar * vbar
            m = SJ / xp.maximum(n, one)[:, None]
            H = H - n[:, None, None] * m[:, :, None] * m[:, None, :]
        norm = xp.sqrt(xp.maximum(norm2, 0.0))
        inv = 1.0 / xp.where(norm > 0, norm, 1.0)
        a = (SvJ - vbar[:, None] * SJ) * inv[:, None]                 # sum ghat j
        ghat_q = (Svq - vbar * Sq) * inv                               # sum ghat q
        Jr = a - SqJ + (0.0 if kind == "nssd" else 1.0) * (SJ / xp.maximum(n, one)[:, None]) * Sq[:, None]
        H = (H - a[:, :, None] * a[:, None, :]) * (inv * inv)[:, None, None]
        b = (Jr - a * (1.0 - ghat_q)[:, None]) * inv[:, None]
        obj = _SCALE[kind] * (1.0 - 2.0 * ghat_q + Sqq)
    grad_energy = s("JJ")[:, 0] + s("JJ")[:, ndof] + s("JJ")[:, 2 * ndof - 1]
    sum_g2 = Svv + 2.0 * shift * Sv + n * shift * shift
    return H, b, obj, grad_energy, sum_g2
