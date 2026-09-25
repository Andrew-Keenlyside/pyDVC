"""Array-namespace dispatch: numpy on the host, cupy on the device.

Same contract as zarr-vectors' ``_xp``: device arrays are recognised by
``__cuda_array_interface__``, and asking for ``"cuda"`` without cupy is an
error, never a silent host fallback. zarr-vectors' module is internal
upstream, so pyDVC keeps its own.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Literal

import numpy as np

Device = Literal["cpu", "cuda"]


def is_device_array(a: Any) -> bool:
    return hasattr(type(a), "__cuda_array_interface__")


def get_xp(device: Device) -> ModuleType:
    if device == "cpu":
        return np
    try:
        import cupy
    except ImportError as exc:
        raise ImportError("device='cuda' needs cupy: pip install 'pydvc[gpu]'") from exc
    return cupy


def xp_of(*arrays: Any) -> ModuleType:
    """Namespace of the inputs: cupy if any is a device array, else numpy."""
    return get_xp("cuda" if any(is_device_array(a) for a in arrays) else "cpu")
