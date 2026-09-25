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

PARAM_NAMES = ("u", "v", "w", "phi", "the", "psi", "exx", "eyy", "ezz", "exy", "eyz", "exz")
DOF_CHOICES = (3, 6, 12)


def rotation_matrix(angles: Any) -> Any:
    """(B, 3) -> (B, 3, 3), CCPi roll-pitch-yaw."""
    raise todo("M1", "rotation_matrix")


def strain_tensor(strains: Any) -> Any:
    """(B, 6) ``(exx, eyy, ezz, exy, eyz, exz)`` -> (B, 3, 3) symmetric."""
    raise todo("M1", "strain_tensor")


def deformation_matrix(params: Any) -> Any:
    """(B, ndof) -> ``(I + E) R`` as (B, 3, 3)."""
    raise todo("M1", "deformation_matrix")


def warp(centres: Any, params: Any, offsets: Any) -> Any:
    """(B, 3), (B, ndof), (M, 3) -> (B, M, 3) warped sample positions."""
    raise todo("M1", "warp")


def warp_jacobian(params: Any, offsets: Any) -> Any:
    """``dx'/dp``: (B, ndof), (M, 3) -> (B, M, 3, ndof). Forward-additive solver."""
    raise todo("M1", "warp_jacobian")


def warp_jacobian_at_identity(offsets: Any, ndof: int) -> Any:
    """``dx'/dp`` at ``p = 0``: (M, 3, ndof). Inverse-compositional solver."""
    raise todo("M5", "warp_jacobian_at_identity")


def compose_inverse(params: Any, delta: Any) -> Any:
    """IC-GN update ``W(p) <- W(p) o W(delta)^-1``, returned in CCPi parameters."""
    raise todo("M5", "compose_inverse")
