"""Marker for scaffolded functionality.

Every stub raises ``todo(<milestone>, <what>)`` so a traceback names the
milestone in docs/MVP_PLAN.md that delivers it, and the test suite reports
tests that hit a stub as skipped rather than failed (see tests/conftest.py).
"""

from __future__ import annotations


def todo(milestone: str, what: str) -> NotImplementedError:
    return NotImplementedError(f"[{milestone}] {what} - see docs/MVP_PLAN.md")
