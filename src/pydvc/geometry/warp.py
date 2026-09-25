"""Shape functions: 3, 6 and 12 degrees of freedom, in CCPi's parameterisation.

Parameter vector (CCPi ``SearchParams``), angles in radians, strain dimensionless::

    index  0  1  2   3    4    5    6    7    8    9    10   11
    name   u  v  w   phi  the  psi  exx  eyy  ezz  exy  eyz  exz

A template offset ``d`` around centre ``c`` maps to

    x' = c + t + (I + E) R d

where ``t = (u, v, w)``, ``R`` is CCPi's roll-pitch-yaw matrix
(``SearchParams::set_param_vect``) and ``E`` is the symmetric strain tensor.
This matches ``FloatingCloud::affine_to``: translate, then ``rotate_ptvect``
(``d -> R d``), then ``strain_ptvect`` (``d' -> d' + E d'``), both relative to
the moved reference point. ``ndof = 3`` gives ``R = I, E = 0``; ``ndof = 6``
gives ``E = 0``.

The forward-additive solver (CCPi parity) needs ``dx'/dp`` at the current
``p``. That is nonlinear in the angles and is computed analytically per point
and iteration: 3 x ndof values per sample, cheap next to interpolation. The
inverse-compositional solver (M5) needs only ``dx'/dp`` at ``p = 0``,
precomputed once per run from the template.
"""

from __future__ import annotations

from typing import Any

from pydvc._todo import todo
from pydvc.kernels.xp import xp_of

PARAM_NAMES = ("u", "v", "w", "phi", "the", "psi", "exx", "eyy", "ezz", "exy", "eyz", "exz")
DOF_CHOICES = (3, 6, 12)

# (row, col) entries of E set by each strain parameter (exx, eyy, ezz, exy, eyz, exz); tensor shears.
_STRAIN_ENTRIES = (((0, 0),), ((1, 1),), ((2, 2),), ((0, 1), (1, 0)), ((1, 2), (2, 1)), ((0, 2), (2, 0)))


def _check_dof(ndof: int) -> None:
    if ndof not in DOF_CHOICES:
        raise ValueError(f"ndof must be one of {DOF_CHOICES}, got {ndof}")


def _elementary(angles: Any) -> tuple[Any, Any]:
    """Factors of ``R = Rx(psi) Ry(the) Rz(phi)`` and their derivatives, each (3, B, 3, 3)."""
    xp = xp_of(angles)
    phi, the, psi = angles[:, 0], angles[:, 1], angles[:, 2]
    shape = (angles.shape[0], 3, 3)
    mats = xp.zeros((3,) + shape, dtype=angles.dtype)
    ders = xp.zeros((3,) + shape, dtype=angles.dtype)
    # Rz(phi): yaw about z
    c, s = xp.cos(phi), xp.sin(phi)
    mats[0, :, 0, 0], mats[0, :, 0, 1], mats[0, :, 1, 0], mats[0, :, 1, 1], mats[0, :, 2, 2] = c, s, -s, c, 1.0
    ders[0, :, 0, 0], ders[0, :, 0, 1], ders[0, :, 1, 0], ders[0, :, 1, 1] = -s, c, -c, -s
    # Ry(the): pitch about y
    c, s = xp.cos(the), xp.sin(the)
    mats[1, :, 0, 0], mats[1, :, 0, 2], mats[1, :, 2, 0], mats[1, :, 2, 2], mats[1, :, 1, 1] = c, -s, s, c, 1.0
    ders[1, :, 0, 0], ders[1, :, 0, 2], ders[1, :, 2, 0], ders[1, :, 2, 2] = -s, -c, c, -s
    # Rx(psi): roll about x
    c, s = xp.cos(psi), xp.sin(psi)
    mats[2, :, 1, 1], mats[2, :, 1, 2], mats[2, :, 2, 1], mats[2, :, 2, 2], mats[2, :, 0, 0] = c, s, -s, c, 1.0
    ders[2, :, 1, 1], ders[2, :, 1, 2], ders[2, :, 2, 1], ders[2, :, 2, 2] = -s, c, -c, -s
    return mats, ders


def rotation_matrix(angles: Any) -> Any:
    """(B, 3) -> (B, 3, 3), CCPi roll-pitch-yaw.

    ``angles = (phi, the, psi)``: yaw about z, pitch about y, roll about x, composed
    as ``R = Rx(psi) Ry(the) Rz(phi)`` (the aerospace 3-2-1 direction-cosine matrix).
    """
    rz, ry, rx = _elementary(angles)[0]
    return rx @ ry @ rz


def strain_tensor(strains: Any) -> Any:
    """(B, 6) ``(exx, eyy, ezz, exy, eyz, exz)`` -> (B, 3, 3) symmetric."""
    xp = xp_of(strains)
    E = xp.zeros((strains.shape[0], 3, 3), dtype=strains.dtype)
    for k, entries in enumerate(_STRAIN_ENTRIES):
        for i, j in entries:
            E[:, i, j] = strains[:, k]
    return E


def deformation_matrix(params: Any) -> Any:
    """(B, ndof) -> ``(I + E) R`` as (B, 3, 3)."""
    xp = xp_of(params)
    ndof = params.shape[-1]
    _check_dof(ndof)
    eye = xp.broadcast_to(xp.eye(3, dtype=params.dtype), (params.shape[0], 3, 3))
    if ndof == 3:
        return eye.copy()
    R = rotation_matrix(params[:, 3:6])
    if ndof == 6:
        return R
    return (eye + strain_tensor(params[:, 6:12])) @ R


def warp(centres: Any, params: Any, offsets: Any) -> Any:
    """(B, 3), (B, ndof), (M, 3) -> (B, M, 3) warped sample positions."""
    xp = xp_of(centres, params, offsets)
    F = deformation_matrix(params)
    moved = xp.einsum("bij,mj->bmi", F, xp.asarray(offsets, dtype=F.dtype))
    return (centres + params[:, :3])[:, None, :] + moved


def warp_linear_jacobians(params: Any) -> Any:
    """``dx'/dp_k = A_k d`` for the non-translation parameters: (B, ndof) -> (B, ndof - 3, 3, 3).

    Translations have ``dx'/dt = I``. Solvers contract ``A_k`` with the image
    gradient and the template, so the (B, M, 3, ndof) array of
    :func:`warp_jacobian` is never formed.
    """
    xp = xp_of(params)
    ndof = params.shape[-1]
    _check_dof(ndof)
    B = params.shape[0]
    if ndof == 3:
        return xp.zeros((B, 0, 3, 3), dtype=params.dtype)
    mats, ders = _elementary(params[:, 3:6])
    rz, ry, rx = mats
    R = rx @ ry @ rz
    dR = xp.stack([rx @ ry @ ders[0], rx @ ders[1] @ rz, ders[2] @ ry @ rz], axis=1)   # d/dphi, d/dthe, d/dpsi
    if ndof == 6:
        return dR
    eye = xp.eye(3, dtype=params.dtype)
    IE = eye + strain_tensor(params[:, 6:12])
    dE_R = xp.zeros((B, 6, 3, 3), dtype=params.dtype)
    for k, entries in enumerate(_STRAIN_ENTRIES):
        for i, j in entries:
            dE_R[:, k, i, :] += R[:, j, :]
    return xp.concatenate([IE[:, None] @ dR, dE_R], axis=1)


def warp_jacobian(params: Any, offsets: Any) -> Any:
    """``dx'/dp``: (B, ndof), (M, 3) -> (B, M, 3, ndof). Forward-additive solver."""
    xp = xp_of(params, offsets)
    B, ndof = params.shape
    d = xp.asarray(offsets, dtype=params.dtype)
    J = xp.zeros((B, d.shape[0], 3, ndof), dtype=params.dtype)
    for i in range(3):
        J[:, :, i, i] = 1.0
    if ndof > 3:
        J[..., 3:] = xp.einsum("bkij,mj->bmik", warp_linear_jacobians(params), d)
    return J


def warp_jacobian_at_identity(offsets: Any, ndof: int) -> Any:
    """``dx'/dp`` at ``p = 0``: (M, 3, ndof). Inverse-compositional solver."""
    raise todo("M5", "warp_jacobian_at_identity")


def compose_inverse(params: Any, delta: Any) -> Any:
    """IC-GN update ``W(p) <- W(p) o W(delta)^-1``, returned in CCPi parameters."""
    raise todo("M5", "compose_inverse")
