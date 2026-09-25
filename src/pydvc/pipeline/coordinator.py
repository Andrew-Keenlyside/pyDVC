"""The run's stages, one per CLI subcommand. Each is safe to re-run.

``prepare``   (1 process)  import a .roi into a zarr-vectors store if needed;
                            check that reference and deformed volumes match in
                            shape; build the template; plan tiles; check memory;
                            ``ResultStore.allocate``; write ``plan.json``.
``seed``      (1 device)   strategy-dependent: nothing (rigid); the coarse
                            sub-grid solve plus seed field (coarse); or the whole
                            wavefront solve (parity mode, which needs the problem
                            to fit in memory). Re-plans the tiles' brick boxes
                            from the seeds.
``run``       (devices)    the tile loop, skipping tiles already written. One
                            process today; M4 spreads it over every GPU.
``repair``    (M5)         ``repair_candidates``, then re-solve and rewrite
                            the affected tiles. Only after ``run`` has finished
                            everywhere.
``finalize``  (1 process)  ``ResultStore.finalize``; ``.stat`` summary;
                            optional ``.disp`` export; optional pyramid for
                            Neuroglancer.

Files in ``cfg.workdir``: ``plan.json``, ``points.zarrvectors`` (when the
points came as a ``.roi``), ``seeds.npz`` (coarse seeding), ``run_stats.json``,
``failed_tiles.json`` (if any tile failed twice), ``results.stat`` and the
optional ``results.disp``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pydvc._todo import todo
from pydvc.config import RunConfig
from pydvc.io.ccpi import RunSummary

log = logging.getLogger("pydvc")


def _workdir(cfg: RunConfig) -> Path:
    w = Path(cfg.workdir)
    w.mkdir(parents=True, exist_ok=True)
    return w


def points_store(cfg: RunConfig) -> Path:
    """The zarr-vectors store the pipeline reads: ``cfg.points`` itself, or its import in the workdir."""
    from pydvc.io.pointcloud import is_store

    return Path(cfg.points) if is_store(cfg.points) else Path(cfg.workdir) / "points.zarrvectors"


def _seed_field(cfg: RunConfig) -> Path | None:
    p = Path(cfg.workdir) / "seeds.npz"
    return p if p.exists() else None


def _device_bytes(backend: str) -> int:
    from pydvc.solver.engines import on_device

    if on_device(backend):
        import cupy

        return int(cupy.cuda.runtime.memGetInfo()[1])
    import os

    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def _plan(cfg: RunConfig, *, backend: str) -> dict[str, Any]:
    from pydvc.io.pointcloud import PointCloud
    from pydvc.io.volume import open_volume
    from pydvc.pipeline.tiling import assign_lpt, check_memory, plan_tiles, save_plan

    store = points_store(cfg)
    pc = PointCloud(store)
    tiles = plan_tiles(pc, cfg, seed_field_path=_seed_field(cfg))
    dtype_bytes = open_volume(cfg.volumes, "reference").dtype.itemsize
    mem = check_memory(tiles, cfg, device_total_bytes=_device_bytes(backend), dtype_bytes=dtype_bytes)
    if not mem.fits:
        raise MemoryError(
            f"tiles need {mem.in_flight_bytes + mem.batch_bytes:,} bytes of device memory with prefetch_depth "
            f"{cfg.cluster.prefetch_depth}; {mem.device_bytes:,} usable. Reduce cluster.tile_shape or prefetch_depth."
        )
    from pydvc.pipeline.launch import node_info

    plan_path = Path(cfg.workdir) / "plan.json"
    n_nodes = node_info().n_nodes
    save_plan(plan_path, tiles, assign_lpt(tiles, n_nodes), cfg, points_store=str(store.resolve()), memory=asdict(mem),
              backend=backend)
    return {"tiles": len(tiles), "points": pc.n_points, "memory": asdict(mem), "plan": str(plan_path)}


def prepare(cfg: RunConfig, *, backend: str | None = None) -> dict[str, Any]:
    from pydvc.geometry.templates import make_template
    from pydvc.io.pointcloud import PointCloud, chunk_shape_for, import_points, is_store
    from pydvc.io.results import ResultStore
    from pydvc.io.volume import open_volume
    from pydvc.solver.engines import default_backend

    backend = backend or default_backend()
    _workdir(cfg)
    ref, deformed = open_volume(cfg.volumes, "reference"), open_volume(cfg.volumes, "deformed")
    if ref.shape != deformed.shape:
        raise ValueError(f"reference {ref.shape} and deformed {deformed.shape} volumes differ in shape")
    make_template(cfg.subvolume)
    store = points_store(cfg)
    if not is_store(cfg.points):
        roi = Path(cfg.points)
        if not store.exists() or store.stat().st_mtime < roi.stat().st_mtime:
            import shutil

            shutil.rmtree(store, ignore_errors=True)
            import_points(roi, store, volume_shape_zyx=ref.shape, chunk_shape=chunk_shape_for(cfg.cluster.tile_shape))
    pc = PointCloud(store)
    if Path(cfg.output).exists():
        rs = ResultStore(cfg.output, mode="r")
        if rs.dof != cfg.search.dof or tuple(rs.chunk_shape) != tuple(pc.chunk_shape):
            raise ValueError(f"{cfg.output} holds a run with other settings; remove it or choose another output")
    else:
        ResultStore.allocate(cfg.output, points=pc, dof=cfg.search.dof)
    info = _plan(cfg, backend=backend)
    log.info("prepared %s tiles, %s points", info["tiles"], info["points"])
    return info


def seed(cfg: RunConfig, *, backend: str | None = None) -> dict[str, Any]:
    """Strategy-dependent seeding; see the module docstring."""
    from pydvc.solver.engines import default_backend

    backend = backend or default_backend()
    strategy = cfg.seeding.strategy
    if strategy == "rigid":
        return {"strategy": "rigid"}
    if strategy == "wavefront":
        return _seed_wavefront(cfg, backend)
    if strategy == "coarse":
        return _seed_coarse(cfg, backend)
    raise todo("M5", f"{strategy!r} seeding")


def _all_points(cfg: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    from pydvc.io.pointcloud import PointCloud

    xyz, point_id = PointCloud(points_store(cfg)).read_all(device="cpu")
    order = np.argsort(point_id, kind="stable")
    return np.asarray(point_id)[order], np.asarray(xyz, dtype=np.float64)[order]


def _seed_wavefront(cfg: RunConfig, backend: str) -> dict[str, Any]:
    """Parity mode: the whole problem shell by shell, written straight into the results store."""
    from pydvc.pipeline.inmemory import solve_in_memory

    _require_whole_volumes_fit(cfg, "wavefront seeding")
    point_id, xyz = _all_points(cfg)
    res = solve_in_memory(cfg, point_id, xyz, strategy="wavefront", backend=backend)
    _write_results(cfg, res)
    return {"strategy": "wavefront", "points": len(point_id), "seconds": res.seconds}


def _seed_coarse(cfg: RunConfig, backend: str) -> dict[str, Any]:
    """Solve every ``coarse_stride``-th lattice point (wavefront order), interpolate seeds for all points.

    The coarse pass reads whole volumes at full resolution; the pyramid-level
    pass that scales to 4096^3 is M5.
    """
    from pydvc.pipeline.inmemory import solve_in_memory
    from pydvc.solver.seeding import coarse_subset, interpolate_seed_field

    _require_whole_volumes_fit(cfg, "coarse seeding")
    point_id, xyz = _all_points(cfg)
    sub = coarse_subset(xyz, cfg.seeding.coarse_stride)
    res = solve_in_memory(cfg, point_id[sub], xyz[sub], strategy="wavefront", backend=backend)
    seeds = interpolate_seed_field(res.xyz, res.displacement, res.status, xyz, fallback=cfg.search.rigid_trans)
    np.savez(Path(cfg.workdir) / "seeds.npz", point_id=point_id, seed=seeds,
             coarse_point_id=res.point_id, coarse_status=res.status, coarse_displacement=res.displacement)
    info = _plan(cfg, backend=backend)
    good = float((res.status == 0).mean()) if len(res.status) else 0.0
    return {"strategy": "coarse", "coarse_points": len(sub), "coarse_good": good, "seconds": res.seconds, **info}


def _require_whole_volumes_fit(cfg: RunConfig, what: str) -> None:
    """The wavefront and (MVP) coarse passes hold both whole volumes in host memory; fail early if they cannot."""
    import os

    from pydvc.io.volume import open_volume

    v = open_volume(cfg.volumes, "reference")
    need = 2 * int(np.prod(v.shape)) * v.dtype.itemsize
    have = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    if need > 0.8 * have:
        raise MemoryError(
            f"{what} holds both volumes in memory ({need / 1e9:.0f} GB; {have / 1e9:.0f} GB available). "
            "Use seeding.strategy 'rigid' (fine for displacements within a few voxels of rigid_trans), or the "
            "pyramid-level coarse pass (M5)."
        )


def _write_results(cfg: RunConfig, res: Any) -> None:
    """Write whole-problem results into the store, tile by tile (cells stay disjoint)."""
    from pydvc.io.pointcloud import PointCloud
    from pydvc.io.results import ResultStore
    from pydvc.pipeline.tiling import load_plan

    pc = PointCloud(points_store(cfg))
    store = ResultStore(cfg.output, mode="r+")
    tiles, _ = load_plan(Path(cfg.workdir) / "plan.json")
    order = np.argsort(res.point_id)
    ids = res.point_id[order]
    for tile in tiles:
        pts = pc.read_tile(list(tile.cells), device="cpu")
        rows = order[np.searchsorted(ids, pts.point_id)]
        store.write_tile(pts, {
            "status": res.status[rows], "objmin": res.objmin[rows], "displacement": res.displacement[rows],
            "params": res.params[rows], "n_iter": res.n_iter[rows], "seed": res.seed[rows],
        })


def run(
    cfg: RunConfig,
    *,
    backend: str | None = None,
    devices: tuple[int, ...] | None = None,
    cpu_workers: int = 1,
) -> list[Any]:
    """Solve this node's unwritten tiles: one process per GPU (or ``cpu_workers`` CPU processes).

    Resubmitting after a time-out, a node failure or a killed worker solves
    only the tiles still missing. Tiles that failed twice are listed in
    ``failed_tiles.json``, tiles still incomplete in ``run_stats.json``.
    """
    from pydvc.io.results import ResultStore
    from pydvc.pipeline.launch import launch_local, node_info, node_share
    from pydvc.pipeline.tiling import load_plan
    from pydvc.solver.engines import default_backend

    backend = backend or default_backend()
    workdir = _workdir(cfg)
    plan_path = workdir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"{plan_path}: run `pydvc plan` first")
    info = node_info()
    t0 = time.perf_counter()
    stats = launch_local(cfg, backend=backend, devices=devices, cpu_workers=cpu_workers)
    seconds = time.perf_counter() - t0
    tiles, _ = load_plan(plan_path)
    mine = set(node_share(plan_path, info))
    written = ResultStore(cfg.output).written_cells()
    missing = [t.id for t in tiles if t.id in mine and not written.issuperset(t.cells)]
    errors = [e for s in stats for e in s.errors]
    suffix = f".node{info.node_rank}" if info.n_nodes > 1 else ""
    failed_path = workdir / f"failed_tiles{suffix}.json"
    if missing:
        failed_path.write_text(json.dumps({"missing": missing, "errors": errors}, indent=1))
        log.error("%d tiles are not written; see %s and resubmit", len(missing), failed_path)
    elif failed_path.exists():
        failed_path.unlink()
    (workdir / f"run_stats{suffix}.json").write_text(json.dumps(
        {"seconds": seconds, "backend": backend, "node": asdict(info), "missing_tiles": missing,
         "workers": [asdict(s) for s in stats]}, indent=1, default=str))
    return stats


def repair(cfg: RunConfig) -> None:
    raise todo("M5", "coordinator.repair")


def finalize(cfg: RunConfig, *, export_disp: bool = False, pyramid: bool = False) -> RunSummary:
    from pydvc.io.ccpi import write_disp, write_stat
    from pydvc.io.results import ResultStore

    workdir = _workdir(cfg)
    store = ResultStore(cfg.output, mode="r+")
    data = store.read_all()
    n = len(data["point_id"])
    store.finalize(n_points=n, pyramid=pyramid)
    values, counts = np.unique(data["status"], return_counts=True)
    stats_path = workdir / "run_stats.json"
    seconds = json.loads(stats_path.read_text())["seconds"] if stats_path.exists() else 0.0
    summary = RunSummary(n_points=n, seconds=seconds, counts={int(v): int(c) for v, c in zip(values, counts)})
    write_stat(workdir / "results.stat", cfg, summary)
    if export_disp:
        order = np.argsort(data["point_id"], kind="stable")
        write_disp(workdir / "results.disp", data["point_id"][order], data["xyz"][order], data["status"][order],
                   data["objmin"][order], data["displacement"][order])
    return summary
