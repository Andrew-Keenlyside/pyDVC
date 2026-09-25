"""The run's stages, one per CLI subcommand. Each is safe to re-run.

``prepare``   (1 process)  import a .roi into a zarr-vectors store if needed;
                            check that reference and deformed volumes match in
                            shape; build the template; plan tiles; check memory;
                            ``ResultStore.allocate``; write ``plan.json``.
``seed``      (1 GPU)      strategy-dependent: nothing (rigid); the coarse
                            sub-grid solve plus seed field (coarse); or the whole
                            wavefront solve (parity mode, which needs the problem
                            to fit on one GPU).
``run``       (all GPUs)   ``launch_local`` over this node's tiles, skipping
                            tiles already written.
``repair``    (all GPUs)   ``repair_candidates``, then re-solve and rewrite
                            the affected tiles. Only after ``run`` has finished
                            everywhere.
``finalize``  (1 process)  ``ResultStore.finalize``; ``.stat`` summary;
                            optional ``.disp`` export; optional pyramid for
                            Neuroglancer.
"""

from __future__ import annotations

from pydvc._todo import todo
from pydvc.config import RunConfig
from pydvc.io.ccpi import RunSummary


def prepare(cfg: RunConfig) -> None:
    raise todo("M3", "coordinator.prepare")


def seed(cfg: RunConfig) -> None:
    raise todo("M3", "coordinator.seed")


def run(cfg: RunConfig) -> None:
    raise todo("M4", "coordinator.run")


def repair(cfg: RunConfig) -> None:
    raise todo("M5", "coordinator.repair")


def finalize(cfg: RunConfig, *, export_disp: bool = False, pyramid: bool = False) -> RunSummary:
    raise todo("M4", "coordinator.finalize")
