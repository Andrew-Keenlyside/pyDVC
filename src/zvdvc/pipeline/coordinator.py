"""The run's stages, one per CLI subcommand. Each is safe to re-run.

``prepare``   (1 process)  import a .roi into a zarr-vectors store if needed;
                            check that reference and deformed volumes match in
                            shape; build the template; plan tiles; check memory;
                            ``ResultStore.allocate`` (or check that an existing
                            store was made from the same volumes, points and
                            settings: :func:`fingerprint`); write ``plan.json``.
``seed``      (1 device)   strategy-dependent: nothing (rigid); the coarse
                            sub-grid solve plus seed field (coarse); or the whole
                            wavefront solve (parity mode, which needs the problem
                            to fit in memory). Re-plans the tiles' brick boxes
                            from the seeds.
``run``       (devices)    the tile loop, skipping tiles already written, one
                            process per GPU. Refuses a config that differs from
                            the plan's in anything that changes results, and a
                            seed stage that has not run.
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

import contextlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from zvdvc._todo import todo
from zvdvc.config import RunConfig
from zvdvc.io.ccpi import RunSummary

log = logging.getLogger("zvdvc")


def _workdir(cfg: RunConfig) -> Path:
    w = Path(cfg.workdir)
    w.mkdir(parents=True, exist_ok=True)
    return w


def points_store(cfg: RunConfig) -> Path:
    """The zarr-vectors store the pipeline reads: ``cfg.points`` itself, or its import in the workdir."""
    from zvdvc.io.pointcloud import is_store

    return Path(cfg.points) if is_store(cfg.points) else Path(cfg.workdir) / "points.zarrvectors"


def _seed_field(cfg: RunConfig) -> Path | None:
    """``seeds.npz``, for the coarse strategy only (a stale one must not seed another strategy's run)."""
    p = Path(cfg.workdir) / "seeds.npz"
    return p if cfg.seeding.strategy == "coarse" and p.exists() else None


def _file_digest(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def result_settings(cfg: RunConfig) -> dict[str, Any]:
    """The settings that change results: all but the volume, points and output paths, ``cluster``,
    ``workdir``, ``uncertainty_seeds`` and ``seeding.repair_passes``."""
    d = cfg.to_dict()
    return {
        "volumes": {k: v for k, v in d["volumes"].items() if k not in ("reference", "deformed")},
        "subvolume": d["subvolume"],
        "search": d["search"],
        "seeding": {k: v for k, v in d["seeding"].items() if k != "repair_passes"},
        "num_points_to_process": d["num_points_to_process"],
    }


def volume_identity(cfg: RunConfig, which: str) -> dict[str, Any]:
    """Resolved path, shape, dtype, and the data file's size and mtime (raw/mhd/npy) or ``zarr.json`` digest (zarr)."""
    from zvdvc.io.volume import is_zarr_uri, open_volume

    uri = getattr(cfg.volumes, which)
    vol = open_volume(cfg.volumes, which)
    out: dict[str, Any] = {"path": uri if "://" in str(uri) else str(Path(uri).resolve()),
                           "shape": list(vol.shape), "dtype": np.dtype(vol.dtype).str}
    if is_zarr_uri(uri):
        meta = next((m for m in (Path(uri) / cfg.volumes.array_path / "zarr.json", Path(uri) / "zarr.json") if m.exists()),
                    None)
        out["zarr_json"] = _file_digest(meta) if meta is not None else None
    else:
        data = Path(getattr(getattr(vol, "inner", vol), "path", uri))       # the .raw behind a .mhd
        st = data.stat()
        out |= {"data": str(data.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        if data.resolve() != Path(uri).resolve():
            out["header"] = _file_digest(uri)
    return out


def points_digest(pc: Any) -> str:
    """Digest of the point ids and positions (in id order); refuses repeated ids, which results are matched by."""
    import hashlib

    from zvdvc.io.pointcloud import check_points

    xyz, point_id = pc.read_all(device="cpu")
    point_id = np.asarray(point_id, dtype="<i8")
    check_points(point_id, np.asarray(xyz), where=pc.path)
    order = np.argsort(point_id, kind="stable")
    h = hashlib.sha256(point_id[order].tobytes())
    h.update(np.asarray(xyz, dtype="<f4")[order].tobytes())
    return h.hexdigest()[:16]


def fingerprint(cfg: RunConfig, pc: Any) -> dict[str, Any]:
    """What a results store's values depend on: settings, both volumes' identities and the points."""
    return {
        "settings": result_settings(cfg),
        "volumes": {w: volume_identity(cfg, w) for w in ("reference", "deformed")},
        "points": {"n_points": int(pc.n_points), "digest": points_digest(pc)},
    }


def _device_bytes(backend: str) -> int:
    from zvdvc.solver.engines import on_device

    if on_device(backend):
        import cupy

        return int(cupy.cuda.runtime.memGetInfo()[1])
    return host_memory()[0]


def host_memory() -> tuple[int, int | None]:
    """(total, available) bytes of host RAM; available is None where the OS does not report it (e.g. macOS)."""
    import os

    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total = 1 << 40                                  # unknown: do not block the run on it
    try:
        avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        avail = None
    return int(total), (int(avail) if avail is not None else None)


def _plan(cfg: RunConfig, *, backend: str, seeds: Path | None, fp: dict[str, Any]) -> Any:
    """Tiles and the memory check (``MemoryError`` if they do not fit); returns a callable that writes ``plan.json``.

    ``seeds`` (``seeds.npz``) shapes the deformed bricks; its digest goes into the plan, where ``run`` checks it.
    """
    from zvdvc.io.pointcloud import PointCloud
    from zvdvc.io.volume import open_volume
    from zvdvc.pipeline.tiling import assign_lpt, check_memory, plan_tiles, save_plan

    store = points_store(cfg)
    pc = PointCloud(store)
    tiles = plan_tiles(pc, cfg, seed_field_path=seeds)
    dtype_bytes = open_volume(cfg.volumes, "reference").dtype.itemsize
    mem = check_memory(tiles, cfg, device_total_bytes=_device_bytes(backend), dtype_bytes=dtype_bytes)
    if not mem.fits:
        raise MemoryError(
            f"tiles need {mem.in_flight_bytes + mem.batch_bytes:,} bytes of device memory with prefetch_depth "
            f"{cfg.cluster.prefetch_depth}; {mem.device_bytes:,} usable. Reduce cluster.tile_shape or prefetch_depth."
        )

    def save() -> dict[str, Any]:
        from zvdvc.pipeline.launch import node_info

        plan_path = Path(cfg.workdir) / "plan.json"
        n_nodes = node_info().n_nodes
        save_plan(plan_path, tiles, assign_lpt(tiles, n_nodes), cfg, points_store=str(store.resolve()),
                  memory=asdict(mem), backend=backend, fingerprint=fp,
                  seeds_digest=_file_digest(seeds) if seeds is not None else None)
        return {"tiles": len(tiles), "points": pc.n_points, "memory": asdict(mem), "plan": str(plan_path)}

    return save


def _import_roi(cfg: RunConfig, shape_zyx: tuple[int, int, int]) -> None:
    """Import ``cfg.points`` (a .roi) into the workdir store unless an import at least as new is there."""
    import shutil

    from zvdvc.io.pointcloud import chunk_shape_for, import_points

    roi, store = Path(cfg.points), points_store(cfg)
    if store.exists() and store.stat().st_mtime >= roi.stat().st_mtime:
        return
    shutil.rmtree(store, ignore_errors=True)
    try:
        import_points(roi, store, volume_shape_zyx=shape_zyx, chunk_shape=chunk_shape_for(cfg.cluster.tile_shape))
    except BaseException:
        shutil.rmtree(store, ignore_errors=True)       # a partial import would pass for a current one next time
        raise


def _planned_seeds(cfg: RunConfig, fp: dict[str, Any]) -> Path | None:
    """``seeds.npz`` if the existing plan was made from it for the same fingerprint (a re-plan keeps the seed stage)."""
    from zvdvc.io.results import fingerprint_diff
    from zvdvc.pipeline.tiling import plan_document

    seeds, plan_path = _seed_field(cfg), Path(cfg.workdir) / "plan.json"
    if seeds is None or not plan_path.exists():
        return None
    doc = plan_document(plan_path)
    same = doc.get("fingerprint") is not None and not fingerprint_diff(doc["fingerprint"], fp)
    return seeds if same and doc.get("seeds_digest") == _file_digest(seeds) else None


def prepare(cfg: RunConfig, *, backend: str | None = None) -> dict[str, Any]:
    from zvdvc.geometry.templates import make_template
    from zvdvc.io.pointcloud import PointCloud, is_store
    from zvdvc.io.results import ResultStore
    from zvdvc.io.volume import open_volume
    from zvdvc.solver.engines import default_backend

    backend = backend or default_backend()
    _workdir(cfg)
    ref, deformed = open_volume(cfg.volumes, "reference"), open_volume(cfg.volumes, "deformed")
    if ref.shape != deformed.shape:
        raise ValueError(f"reference {ref.shape} and deformed {deformed.shape} volumes differ in shape")
    template = make_template(cfg.subvolume)
    if not is_store(cfg.points):
        _import_roi(cfg, ref.shape)
    pc = PointCloud(points_store(cfg))
    fp = fingerprint(cfg, pc)
    exists = Path(cfg.output).exists()
    if exists:
        rs = ResultStore(cfg.output, mode="r")
        if rs.dof != cfg.search.dof or tuple(rs.chunk_shape) != tuple(pc.chunk_shape):
            raise ValueError(f"{cfg.output} holds a run with other settings; remove it or choose another output")
        rs.check_template(template.digest())
        rs.check_prefilter(cfg.volumes.prefilter_sigma)
        rs.check_fingerprint(fp)
    save = _plan(cfg, backend=backend, seeds=_planned_seeds(cfg, fp), fp=fp)     # MemoryError before any store exists
    if exists:
        info = save()
    else:
        ResultStore.allocate(cfg.output, points=pc, dof=cfg.search.dof, template_digest=template.digest(),
                             prefilter_sigma=cfg.volumes.prefilter_sigma, fingerprint=fp)
        try:
            info = save()
        except BaseException:
            import shutil

            shutil.rmtree(cfg.output, ignore_errors=True)
            raise
    log.info("prepared %s tiles, %s points", info["tiles"], info["points"])
    return info


def seed(cfg: RunConfig, *, backend: str | None = None) -> dict[str, Any]:
    """Strategy-dependent seeding; see the module docstring."""
    from zvdvc.solver.engines import default_backend

    backend = backend or default_backend()
    strategy = cfg.seeding.strategy
    if strategy == "rigid":
        return {"strategy": "rigid"}
    from zvdvc.pipeline.tiling import plan_document

    plan_path = Path(cfg.workdir) / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"{plan_path}: run `zvdvc plan` first")
    check_plan(cfg, plan_document(plan_path))
    if strategy == "wavefront":
        return _seed_wavefront(cfg, backend)
    if strategy == "coarse":
        return _seed_coarse(cfg, backend)
    raise todo("M5", f"{strategy!r} seeding")


def _all_points(cfg: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    from zvdvc.io.pointcloud import PointCloud

    xyz, point_id = PointCloud(points_store(cfg)).read_all(device="cpu")
    order = np.argsort(point_id, kind="stable")
    return np.asarray(point_id)[order], np.asarray(xyz, dtype=np.float64)[order]


def _seed_wavefront(cfg: RunConfig, backend: str) -> dict[str, Any]:
    """Parity mode: the whole problem shell by shell, written straight into the results store."""
    from zvdvc.pipeline.inmemory import solve_in_memory

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
    from zvdvc.pipeline.inmemory import solve_in_memory
    from zvdvc.solver.seeding import coarse_subset, interpolate_seed_field

    _require_whole_volumes_fit(cfg, "coarse seeding")
    point_id, xyz = _all_points(cfg)
    sub = coarse_subset(xyz, cfg.seeding.coarse_stride)
    res = solve_in_memory(cfg, point_id[sub], xyz[sub], strategy="wavefront", backend=backend)
    seeds = interpolate_seed_field(res.xyz, res.displacement, res.status, xyz, fallback=cfg.search.rigid_trans)
    np.savez(Path(cfg.workdir) / "seeds.npz", point_id=point_id, seed=seeds,
             coarse_point_id=res.point_id, coarse_status=res.status, coarse_displacement=res.displacement)
    info = _plan(cfg, backend=backend, seeds=_seed_field(cfg), fp=_plan_fingerprint(cfg))()
    good = float((res.status == 0).mean()) if len(res.status) else 0.0
    return {"strategy": "coarse", "coarse_points": len(sub), "coarse_good": good, "seconds": res.seconds, **info}


def _require_whole_volumes_fit(cfg: RunConfig, what: str) -> None:
    """The wavefront and (MVP) coarse passes hold both whole volumes in host memory; fail early if they cannot."""
    from zvdvc.io.volume import open_volume

    v = open_volume(cfg.volumes, "reference")
    need = 2 * int(np.prod(v.shape)) * v.dtype.itemsize
    total, avail = host_memory()
    have = avail if avail is not None else total
    if need > 0.8 * have:
        raise MemoryError(
            f"{what} holds both volumes in memory ({need / 1e9:.0f} GB; {have / 1e9:.0f} GB available). "
            "Use seeding.strategy 'rigid' (fine for displacements within a few voxels of rigid_trans), or the "
            "pyramid-level coarse pass (M5)."
        )


def _plan_fingerprint(cfg: RunConfig) -> dict[str, Any]:
    """The fingerprint ``prepare`` recorded (recomputed for a plan that predates fingerprints)."""
    from zvdvc.io.pointcloud import PointCloud
    from zvdvc.pipeline.tiling import plan_document

    fp = plan_document(Path(cfg.workdir) / "plan.json").get("fingerprint")
    return fp if fp is not None else fingerprint(cfg, PointCloud(points_store(cfg)))


def _write_results(cfg: RunConfig, res: Any) -> None:
    """Write whole-problem results into the store, tile by tile (cells stay disjoint)."""
    from zvdvc.io.pointcloud import PointCloud
    from zvdvc.io.results import ResultStore
    from zvdvc.pipeline.tiling import load_plan

    pc = PointCloud(points_store(cfg))
    store = ResultStore(cfg.output, mode="r+")
    store.reopen_for_writes()
    tiles, _ = load_plan(Path(cfg.workdir) / "plan.json")
    order = np.argsort(res.point_id)
    ids = res.point_id[order]
    for tile in tiles:
        pts = pc.read_tile(list(tile.cells), device="cpu")
        rows = order[np.searchsorted(ids, pts.point_id)]
        values = {
            "status": res.status[rows], "objmin": res.objmin[rows], "displacement": res.displacement[rows],
            "params": res.params[rows], "n_iter": res.n_iter[rows], "seed": res.seed[rows],
        }
        if res.displacement_sd is not None:
            values["displacement_sd"] = res.displacement_sd[rows]
        store.write_tile(pts, values)


class RunStats(list):
    """The workers' :class:`~zvdvc.pipeline.worker.WorkerStats`, plus ``missing``: this node's tiles still unwritten."""

    def __init__(self, stats: list[Any], missing: list[int]) -> None:
        super().__init__(stats)
        self.missing = list(missing)


def check_plan(cfg: RunConfig, doc: dict[str, Any]) -> None:
    """Refuse to run ``cfg`` against a plan (and results store) made from other volumes, points or settings."""
    import warnings

    from zvdvc.io.pointcloud import is_store
    from zvdvc.io.results import fingerprint_diff

    planned = RunConfig.from_dict(doc["config"])
    diff = fingerprint_diff(result_settings(planned), result_settings(cfg))
    for key in ("points", "output"):
        if Path(getattr(planned, key)).resolve() != Path(getattr(cfg, key)).resolve():
            diff.append(key)
    fp = doc.get("fingerprint")
    if fp is None:
        warnings.warn(f"{cfg.workdir}/plan.json records no fingerprint (made before it was stored); cannot check that "
                      "the volumes are the ones planned", stacklevel=2)
    else:
        now = {w: volume_identity(cfg, w) for w in ("reference", "deformed")}
        diff += [f"volumes.{k}" for k in fingerprint_diff(fp["volumes"], now)]
    if diff:
        raise ValueError(f"the config differs from the one `zvdvc plan` used ({Path(cfg.workdir) / 'plan.json'}) in "
                         f"{', '.join(diff)}: run with the original config, or plan again with a new output")
    store = points_store(cfg)
    if not is_store(cfg.points) and store.exists() and Path(cfg.points).stat().st_mtime > store.stat().st_mtime:
        raise ValueError(f"{cfg.points} changed after `zvdvc plan` imported it; run `zvdvc plan` again")


def _check_seeding(cfg: RunConfig, doc: dict[str, Any], tiles: list[Any]) -> None:
    """``run`` after ``seed``: coarse needs the planned ``seeds.npz``; wavefront has already written every tile."""
    from zvdvc.io.results import ResultStore

    strategy = cfg.seeding.strategy
    seeds = Path(cfg.workdir) / "seeds.npz"
    if strategy == "rigid":
        return
    if strategy == "coarse":
        if not seeds.exists() or doc.get("seeds_digest") is None:
            raise FileNotFoundError(f"{seeds}: coarse seeding needs the seed field; run `zvdvc seed` first")
        if _file_digest(seeds) != doc["seeds_digest"]:
            raise ValueError(f"{seeds} changed after `zvdvc seed` planned the tiles; run `zvdvc seed` again")
        return
    if strategy == "wavefront":
        written = ResultStore(cfg.output).written_cells()
        cells = {c for t in tiles for c in t.cells}
        if not cells <= written:
            raise ValueError(f"wavefront seeding solves every point in `zvdvc seed`, but {len(cells - written)} of "
                             f"{len(cells)} cells are unwritten; run `zvdvc seed` first")
        return
    raise todo("M5", f"{strategy!r} seeding")


def run(
    cfg: RunConfig,
    *,
    backend: str | None = None,
    devices: tuple[int, ...] | None = None,
    cpu_workers: int = 1,
    max_tiles: int | None = None,
    tile_ids: list[int] | None = None,
) -> RunStats:
    """Solve this node's unwritten tiles: one process per GPU (or ``cpu_workers`` CPU processes).

    ``max_tiles`` solves only the first tiles of this node's share (short profiling runs);
    ``tile_ids`` names the tiles to solve instead (benchmarks).
    Every run logs per-tile events and, on GPUs, 1 Hz telemetry under
    ``<workdir>/events/<run id>/`` (:mod:`zvdvc.profiling`).

    Resubmitting after a time-out, a node failure or a killed worker solves
    only the tiles still missing. Tiles that failed twice are listed in
    ``failed_tiles.json``, tiles still incomplete in ``run_stats.json`` and
    in the returned ``missing``.
    """
    from zvdvc.io.results import ResultStore
    from zvdvc.pipeline.launch import launch_local, node_info, node_share
    from zvdvc.pipeline.tiling import load_plan, plan_document
    from zvdvc.solver.engines import default_backend

    backend = backend or default_backend()
    workdir = _workdir(cfg)
    plan_path = workdir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"{plan_path}: run `zvdvc plan` first")
    doc = plan_document(plan_path)
    check_plan(cfg, doc)
    _check_seeding(cfg, doc, load_plan(plan_path)[0])
    ResultStore(cfg.output, mode="r+").reopen_for_writes()          # after a partial finalize
    from zvdvc.profiling import GpuTelemetry, events_dir, new_run_id
    from zvdvc.solver.engines import on_device

    info = node_info()
    run_id = new_run_id()
    ev_dir = events_dir(workdir, run_id)
    ids = list(tile_ids) if tile_ids is not None else (node_share(plan_path, info)[:max_tiles] if max_tiles else None)
    t0 = time.perf_counter()
    with GpuTelemetry(ev_dir / "gpu_telemetry.csv") if on_device(backend) else contextlib.nullcontext():
        stats = launch_local(cfg, backend=backend, devices=devices, cpu_workers=cpu_workers, tile_ids=ids)
    seconds = time.perf_counter() - t0
    tiles, _ = load_plan(plan_path)
    mine = set(node_share(plan_path, info))
    written = ResultStore(cfg.output).written_cells()
    if ids is not None:
        mine &= set(ids)
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
         "run_id": run_id, "events": str(ev_dir),
         "workers": [asdict(s) for s in stats]}, indent=1, default=str))
    return RunStats(stats, missing)


def repair(cfg: RunConfig) -> None:
    raise todo("M5", "coordinator.repair")


def finalize(cfg: RunConfig, *, export_disp: bool = False, pyramid: bool = False,
             allow_partial: bool = False) -> RunSummary:
    """Finish the results store and write the summaries; refuses while cells are unwritten.

    With ``allow_partial`` the points of unwritten cells count, and are exported, as ``NOT_SEARCHED``.
    """
    from zvdvc.io.ccpi import write_disp, write_stat
    from zvdvc.io.pointcloud import PointCloud
    from zvdvc.io.results import ResultStore
    from zvdvc.status import PointStatus

    workdir = _workdir(cfg)
    store = ResultStore(cfg.output, mode="r+")
    pc = PointCloud(points_store(cfg))
    cells = set(pc.cells())
    unwritten = sorted(cells - store.written_cells())
    if unwritten and not allow_partial:
        raise ValueError(f"{cfg.output}: {len(unwritten)} of {len(cells)} cells unwritten; run `zvdvc run` again "
                         "(or finalize --allow-partial to export their points as NOT_SEARCHED)")
    data = store.read_all()
    store.finalize(n_points=len(data["point_id"]), pyramid=pyramid)
    if unwritten:
        rest = pc.read_tile(unwritten, device="cpu")
        m = len(rest.point_id)
        log.warning("%d of %d cells unwritten: %d points exported as NOT_SEARCHED", len(unwritten), len(cells), m)
        data = {**data, "point_id": np.concatenate([data["point_id"], rest.point_id]),
                "xyz": np.concatenate([data["xyz"], rest.xyz]),
                "status": np.concatenate([data["status"], np.full(m, PointStatus.NOT_SEARCHED, dtype=np.int8)]),
                "objmin": np.concatenate([data["objmin"], np.full(m, np.nan, dtype=np.float32)]),
                "displacement": np.concatenate([data["displacement"], np.zeros((m, 3), dtype=np.float32)])}
    n = len(data["point_id"])
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
