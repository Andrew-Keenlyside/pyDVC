"""Single-process, whole-volume correlation: the numpy reference path (M1).

Both volumes are read once as whole bricks, the points come from a ``.roi``,
and seeding follows the configured strategy (``rigid`` or CCPi-parity
``wavefront``). It needs the whole problem in memory, so it serves the
accuracy tests, the CCPi parity case and later the M2 GPU parity runs. The
tiled, multi-GPU pipeline (M3/M4) produces the same per-point results from
bricks instead.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pydvc._todo import todo
from pydvc.config import RunConfig
from pydvc.geometry.box import Box
from pydvc.geometry.templates import make_template
from pydvc.io.volume import open_volume
from pydvc.solver import seeding
from pydvc.solver.engines import make_engine
from pydvc.solver.gauss_newton import Backend, solve_batch
from pydvc.status import PointStatus


@dataclass
class Results:
    point_id: np.ndarray       # (N,) int64, input order
    xyz: np.ndarray            # (N, 3)
    status: np.ndarray         # (N,) int8 PointStatus
    objmin: np.ndarray         # (N,)
    params: np.ndarray         # (N, ndof)
    n_iter: np.ndarray         # (N,) uint8
    seed: np.ndarray           # (N, 3)
    seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def displacement(self) -> np.ndarray:
        return self.params[:, :3]

    def status_counts(self) -> dict[int, int]:
        values, counts = np.unique(self.status, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            point_id=self.point_id, xyz=self.xyz, status=self.status, objmin=self.objmin,
            params=self.params, displacement=self.displacement, n_iter=self.n_iter, seed=self.seed,
            seconds=self.seconds,
        )

    @classmethod
    def load(cls, path: str | Path) -> Results:
        with np.load(path) as r:
            return cls(
                point_id=r["point_id"], xyz=r["xyz"], status=r["status"], objmin=r["objmin"],
                params=r["params"], n_iter=r["n_iter"], seed=r["seed"], seconds=float(r["seconds"]),
            )

    def write_disp(self, path: str | Path) -> None:
        from pydvc.io.ccpi import write_disp

        write_disp(path, self.point_id, self.xyz, self.status, self.objmin, self.displacement)


def load_points(cfg: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(point_id, xyz)`` from the configured point cloud (``.roi`` or zarr-vectors store), in point-id order."""
    from pydvc.io.pointcloud import PointCloud, is_store, read_roi

    if is_store(cfg.points):
        xyz, point_id = PointCloud(cfg.points).read_all(device="cpu")
        order = np.argsort(point_id, kind="stable")
        return point_id[order], np.asarray(xyz, dtype=np.float64)[order]
    return read_roi(cfg.points)


def run_in_memory(
    cfg: RunConfig,
    *,
    backend: Backend = "numpy",
    progress: Callable[[str], None] | None = None,
) -> Results:
    """Correlate every point of ``cfg`` with both volumes held in memory."""
    point_id, xyz = load_points(cfg)
    return solve_in_memory(cfg, point_id, xyz, backend=backend, progress=progress)


def solve_in_memory(
    cfg: RunConfig,
    point_id: np.ndarray,
    xyz: np.ndarray,
    *,
    strategy: str | None = None,
    seeds: np.ndarray | None = None,
    backend: Backend = "numpy",
    progress: Callable[[str], None] | None = None,
) -> Results:
    """Correlate the given points against whole-volume bricks.

    ``strategy`` overrides ``cfg.seeding.strategy`` (``rigid`` or ``wavefront``);
    ``seeds`` (N, 3), if given, replaces ``rigid_trans`` for the rigid strategy.
    """
    t0 = time.perf_counter()
    say = progress or (lambda msg: None)
    ref_vol = open_volume(cfg.volumes, "reference")
    def_vol = open_volume(cfg.volumes, "deformed")
    if ref_vol.shape != def_vol.shape:
        raise ValueError(f"reference {ref_vol.shape} and deformed {def_vol.shape} volumes differ in shape")
    whole = Box((0, 0, 0), ref_vol.shape)
    ref = ref_vol.read_brick(whole, device="cpu")
    deformed = def_vol.read_brick(whole, device="cpu")
    template = make_template(cfg.subvolume)
    t_read = time.perf_counter() - t0
    engine = make_engine(backend)

    n, ndof = len(xyz), cfg.search.dof
    res = Results(
        point_id=np.asarray(point_id),
        xyz=np.asarray(xyz, dtype=np.float64),
        status=np.full(n, PointStatus.NOT_SEARCHED, dtype=np.int8),
        objmin=np.full(n, np.nan),
        params=np.zeros((n, ndof)),
        n_iter=np.zeros(n, dtype=np.uint8),
        seed=np.zeros((n, 3)),
    )
    if n == 0:
        return res
    start = cfg.seeding.start_point or tuple(res.xyz[0])
    order = seeding.processing_order(res.xyz, start)
    if cfg.num_points_to_process:
        order = order[: cfg.num_points_to_process]
    todo_mask = np.zeros(n, dtype=bool)
    todo_mask[order] = True

    def solve(idx: np.ndarray, s: np.ndarray) -> None:
        out = solve_batch(ref, deformed, res.xyz[idx], s, template, cfg.search, engine=engine)
        res.params[idx], res.status[idx], res.objmin[idx] = out.params, out.status, out.objmin
        res.n_iter[idx], res.seed[idx] = out.n_iter, out.seed

    t1 = time.perf_counter()
    strategy = strategy or cfg.seeding.strategy
    if strategy == "rigid":
        idx = np.flatnonzero(todo_mask)
        base = np.broadcast_to(cfg.search.rigid_trans, (n, 3)) if seeds is None else np.asarray(seeds)
        solve(idx, base[idx])
    elif strategy == "wavefront":
        neighbours = seeding.knn(res.xyz, cfg.seeding.n_neighbours)
        width = cfg.seeding.shell_width or seeding.median_spacing(res.xyz)
        shells = seeding.wavefront_shells(res.xyz, start, width)
        done = 0
        for k, shell in enumerate(shells):
            shell = shell[todo_mask[shell]]
            if shell.size == 0:
                continue
            s = seeding.seed_from_neighbours(shell, neighbours, res.displacement, res.status, cfg.search.rigid_trans)
            solve(shell, s)
            done += shell.size
            say(f"shell {k + 1}/{len(shells)}: {shell.size} points, {done}/{todo_mask.sum()} done")
    else:
        raise todo("M5", f"{strategy!r} seeding in the in-memory runner (the tiled pipeline runs 'coarse')")
    res.timings = {"read": t_read, "solve": time.perf_counter() - t1}
    res.seconds = time.perf_counter() - t0
    return res
