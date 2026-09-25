"""Run ``fused_gn.cu`` on the host: g++ builds each specialisation against ``cuda_emu.h``.

:class:`EmulatedKernels` has the same ``launch`` interface as
:class:`pydvc.kernels.fused.CudaKernels`, with numpy arrays in place of cupy
ones, so :class:`pydvc.kernels.fused.FusedEngine` runs unchanged on top of
either. Tests use it to check the CUDA source against the numpy reference on
machines without a GPU. Every emulated GPU thread is an OS thread, so keep
problems small (tens of points, hundreds of samples).

Needs a C++20 compiler (``$CXX``, default ``g++``). Libraries are cached in
``$PYDVC_CACHE`` (default ``~/.cache/pydvc``) under a hash of the sources.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np

from pydvc.kernels.cuda import EMU_HEADER, SIGNATURES, SOURCE

_CTYPES = {"int": ctypes.c_int, "float": ctypes.c_float}
_LOCK = threading.Lock()


def compiler() -> str | None:
    return shutil.which(os.environ.get("CXX", "g++"))


def _cache_dir() -> Path:
    root = Path(os.environ.get("PYDVC_CACHE", Path.home() / ".cache" / "pydvc")) / "cuda_emu"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _c_params(name: str, brick_t: str | None) -> list[tuple[str, str]]:
    return [(ctype.replace("T*", f"{brick_t}*") if brick_t else ctype, arg) for ctype, arg in SIGNATURES[name]]


class EmulatedKernels:
    xp = np
    emulated = True

    def __init__(self, block: int = 64) -> None:
        if block % 32 or not 32 <= block <= 256:
            raise ValueError("block must be a multiple of 32 in [32, 256]")
        if compiler() is None:
            raise RuntimeError("the CUDA emulator needs a C++20 compiler (set $CXX)")
        self.block = block
        self._funcs: dict[tuple[str, tuple[str, ...]], Any] = {}

    def _function(self, name: str, targs: tuple[str, ...]) -> Any:
        key = (name, targs)
        with _LOCK:
            if key not in self._funcs:
                self._funcs[key] = self._build(name, targs)
            return self._funcs[key]

    def _build(self, name: str, targs: tuple[str, ...]) -> Any:
        brick_t = targs[-1] if "T*" in "".join(c for c, _ in SIGNATURES[name]) else None
        params = _c_params(name, brick_t)
        symbol = "emu_" + name + "_" + hashlib.sha1(",".join(targs).encode()).hexdigest()[:12]
        decl = ", ".join(f"{c} {a}" for c, a in params)
        call = ", ".join(a for _, a in params)
        wrapper = (
            f'#include "{EMU_HEADER}"\n#include "{SOURCE}"\n'
            f'extern "C" void {symbol}(unsigned grid, unsigned block, {decl}) {{\n'
            f"    emu::launch(grid, block, [&] {{ pydvc::{name}<{', '.join(targs)}>({call}); }});\n}}\n"
        )
        digest = hashlib.sha1((SOURCE.read_text() + EMU_HEADER.read_text() + wrapper).encode()).hexdigest()[:16]
        lib = _cache_dir() / f"{symbol}_{digest}.so"
        if not lib.exists():
            src = lib.with_suffix(".cpp")
            src.write_text(wrapper)
            tmp = lib.with_suffix(f".{os.getpid()}.tmp")
            cmd = [compiler(), "-std=c++20", "-O2", "-shared", "-fPIC", "-pthread", "-x", "c++", str(src), "-o", str(tmp)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"emulator build failed for {name}<{', '.join(targs)}>:\n{proc.stderr[-4000:]}")
            tmp.replace(lib)
        fn = getattr(ctypes.CDLL(str(lib)), symbol)
        fn.restype = None
        fn.argtypes = [ctypes.c_uint, ctypes.c_uint] + [
            _CTYPES.get(c, ctypes.c_void_p) for c, _ in params
        ]
        return fn

    def launch(self, name: str, targs: tuple[str, ...], grid: int, block: int, args: list[Any]) -> None:
        fn = self._function(name, targs)
        conv = []
        keep = []
        for (ctype, _), value in zip(SIGNATURES[name], args):
            if isinstance(value, np.ndarray):
                if not value.flags.c_contiguous:
                    raise ValueError(f"{name}: argument arrays must be C-contiguous")
                keep.append(value)
                conv.append(value.ctypes.data_as(ctypes.c_void_p))
            elif ctype == "float":
                conv.append(ctypes.c_float(float(value)))
            else:
                conv.append(ctypes.c_int(int(value)))
        fn(ctypes.c_uint(int(grid)), ctypes.c_uint(int(block)), *conv)
