"""Run instrumentation: NVTX ranges, per-worker event logs and GPU telemetry.

* :func:`nvtx_range` labels a stage on the Nsight Systems timeline (no-op
  without cupy). Workers mark read / solve / write per tile, and the solver
  marks the reference sampling, each Gauss-Newton iteration and the final
  objective.
* :class:`EventLog` appends one JSON line per event to
  ``<workdir>/events/<run id>/worker-<host>-d<device>-p<pid>.jsonl``: each
  tile's read, solve and write intervals and the compute loop's waits, with
  wall-clock times (``time.time()``) so the workers of a run line up.
  :mod:`pydvc.bench.timeline` turns them into utilisation, stall and
  load-balance figures.
* :class:`GpuTelemetry` samples every GPU (utilisation, memory, SM clock,
  power, temperature) once a second with ``nvidia-smi`` for the duration of a
  run, into ``gpu_telemetry.csv`` next to the event logs.

The run id comes from ``$PYDVC_RUN_ID`` (set by ``coordinator.run``, and
inherited by the spawned workers), so every file of one run shares a folder.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator

RUN_ID_ENV = "PYDVC_RUN_ID"


def new_run_id() -> str:
    """A run id (UTC time, then pid), exported so worker processes inherit it."""
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    os.environ[RUN_ID_ENV] = run_id
    return run_id


def events_dir(workdir: str | Path, run_id: str | None = None) -> Path:
    return Path(workdir) / "events" / (run_id or os.environ.get(RUN_ID_ENV) or "adhoc")


_nvtx: Any = None


def _nvtx_module() -> Any:
    global _nvtx
    if _nvtx is None:
        try:
            from cupy.cuda import nvtx

            _nvtx = nvtx
        except Exception:
            _nvtx = False
    return _nvtx


@contextlib.contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """An NVTX range around the block, visible in Nsight Systems; free when not profiled."""
    nv = _nvtx_module()
    if not nv:
        yield
        return
    nv.RangePush(name)
    try:
        yield
    finally:
        nv.RangePop()


class EventLog:
    """Thread-safe JSON-lines log of one worker's events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)
        self._lock = threading.Lock()

    @classmethod
    def for_worker(cls, workdir: str | Path, device: int) -> EventLog:
        name = f"worker-{socket.gethostname()}-d{device}-p{os.getpid()}.jsonl"
        return cls(events_dir(workdir) / name)

    def record(self, ev: str, **fields: Any) -> None:
        line = json.dumps({"ev": ev, **fields}, default=str)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._fh.close()


class GpuTelemetry:
    """``nvidia-smi`` sampling every GPU at ``interval_ms`` into ``path`` while the context is open."""

    FIELDS = "index,timestamp,utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw,temperature.gpu"

    def __init__(self, path: str | Path, interval_ms: int = 1000) -> None:
        self.path, self.interval_ms = Path(path), interval_ms
        self._proc: subprocess.Popen | None = None
        self._fh: Any = None

    def __enter__(self) -> GpuTelemetry:
        exe = shutil.which("nvidia-smi")
        if exe is None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")
        self._fh.write(self.FIELDS + "\n")
        self._fh.flush()
        self._proc = subprocess.Popen([exe, f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits",
                                       f"-lms={self.interval_ms}"], stdout=self._fh, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._fh is not None:
            self._fh.close()
