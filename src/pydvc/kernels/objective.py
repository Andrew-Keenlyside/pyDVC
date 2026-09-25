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
were ``ssd``, exactly as CCPi does. For ``zssd``/``nssd``/``znssd``, the
Jacobian holds the target mean and norm fixed within an iteration (standard
FA-GN/IC-GN practice). The reported objective is always the exact value above.
"""

from __future__ import annotations

from typing import Any, Literal

from pydvc._todo import todo

Kind = Literal["sad", "ssd", "zssd", "nssd", "znssd"]


def objective(ref: Any, tar: Any, kind: Kind, *, mask: Any = None) -> Any:
    """(B, M), (B, M) -> (B,). ``mask`` excludes samples (e.g. outside the brick) from every sum."""
    raise todo("M1", "objective")


def residuals(ref: Any, tar: Any, kind: Kind, *, mask: Any = None) -> tuple[Any, Any]:
    """(B, M) residuals and (B,) per-point scale (``1/|g~|`` etc.) applied to the target gradient in the Jacobian."""
    raise todo("M1", "residuals")


def reference_stats(ref: Any, kind: Kind) -> Any:
    """Per-point reference mean and norm, computed once per point and reused every iteration."""
    raise todo("M1", "reference_stats")
