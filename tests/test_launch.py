"""M4: node discovery, the per-node tile queue across processes, and recovery from a killed worker."""

import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from zvdvc.config import ClusterSpec, RunConfig, SearchSpec, SeedingSpec, SubvolumeSpec
from zvdvc.io.results import ResultStore
from zvdvc.pipeline import coordinator
from zvdvc.pipeline.launch import NodeInfo, node_info, node_share
from zvdvc.synth.phantoms import default_field, make_case

pytest.importorskip("numba")
SHAPE = (80, 80, 80)


def _rank(env):
    info = node_info(env)
    return info.node_rank, info.n_nodes


def test_node_info_from_schedulers():
    assert _rank({"SLURM_NODEID": "2", "SLURM_JOB_NUM_NODES": "4"}) == (2, 4)
    assert _rank({"GROUP_RANK": "1", "WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8"}) == (1, 2)
    assert _rank({"OMPI_COMM_WORLD_RANK": "9", "OMPI_COMM_WORLD_SIZE": "16", "OMPI_COMM_WORLD_LOCAL_SIZE": "8"}) == (1, 2)
    assert _rank({}) == (0, 1)
    with pytest.raises(ValueError):
        node_info({"SLURM_NODEID": "3", "SLURM_JOB_NUM_NODES": "2"})


def test_shared_source_requeues_a_failed_tile_once():
    import queue
    import threading

    from zvdvc.pipeline.launch import _SharedSource

    ids, attempts, failed = queue.Queue(), {}, []
    tile = type("T", (), {"id": 4})()
    src = _SharedSource({4: tile}, ids, attempts, failed, threading.Lock())
    ids.put(4)
    assert src.next() is tile and src.next() is None
    src.done(tile, False)
    assert src.next() is tile and failed == []           # requeued once
    src.done(tile, False)
    assert src.next() is None and failed == [4]
    src.done(tile, True)
    assert failed == [4]


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    root = tmp_path_factory.mktemp("launch")
    config = make_case(
        root / "case", shape_zyx=SHAPE, field=default_field("affine", SHAPE), spacing=6.0, chunk=40, shard=80,
        subvolume=SubvolumeSpec(geometry="sphere", size=16, n_samples=1500), search=SearchSpec(dof=12, disp_max=8.0),
    )
    return root, RunConfig.from_yaml(config)


def _cfg(case, name):
    root, base = case
    return dataclasses.replace(base, output=str(root / f"{name}.zarrvectors"), workdir=str(root / f"run_{name}"),
                               cluster=ClusterSpec(tile_shape=(40, 40, 40)), seeding=SeedingSpec(strategy="rigid"))


def _sorted(path):
    r = ResultStore(path).read_all()
    o = np.argsort(r["point_id"])
    return {k: v[o] for k, v in r.items()}


def test_node_shares_partition_the_plan(case):
    cfg = _cfg(case, "shares")
    coordinator.prepare(cfg, backend="cpu")
    plan = Path(cfg.workdir) / "plan.json"
    shares = [node_share(plan, NodeInfo(r, 3, ())) for r in range(3)]
    assert sorted(i for s in shares for i in s) == list(range(8))


def test_two_worker_processes_match_one(case):
    one, two = _cfg(case, "one"), _cfg(case, "two")
    for cfg, n in ((one, 1), (two, 2)):
        coordinator.prepare(cfg, backend="cpu")
        stats = coordinator.run(cfg, backend="cpu", cpu_workers=n)
        assert not [e for s in stats for e in s.errors]
        assert sum(s.tiles for s in stats) == 8
    a, b = _sorted(one.output), _sorted(two.output)
    for name in ("status", "params", "objmin", "n_iter"):
        np.testing.assert_array_equal(a[name], b[name])


def test_killed_worker_then_resubmit_gives_the_same_store(case):
    """M4 acceptance: kill -9 one worker mid-run; the resubmitted job solves only what is missing."""
    clean, crashed = _cfg(case, "clean"), _cfg(case, "crashed")
    coordinator.prepare(clean, backend="cpu")
    coordinator.run(clean, backend="cpu")
    coordinator.prepare(crashed, backend="cpu")
    cfg_file = Path(crashed.workdir) / "config.yaml"
    crashed.to_yaml(cfg_file)
    job = subprocess.Popen(
        [sys.executable, "-c",
         f"import sys; from zvdvc.cli import main; sys.exit(main(['run', {str(cfg_file)!r}, '--backend', 'cpu', "
         "'--cpu-workers', '2']))"],
    )
    workdir = Path(crashed.workdir)
    pid_file = workdir / "workers" / "slot0.pid"
    store = ResultStore(crashed.output)
    deadline = time.time() + 300
    killed = False
    while time.time() < deadline and job.poll() is None:
        if pid_file.exists() and store.written_cells():
            pid = int(pid_file.read_text())
            events = [json.loads(line) for log in workdir.glob(f"events/*/worker-*-d0-p{pid}.jsonl")
                      for line in log.read_text().splitlines() if line.endswith("}")]
            reads = {e["tile"] for e in events if e["ev"] == "read"}
            if reads - {e["tile"] for e in events if e["ev"] == "write"}:     # slot 0 holds an unwritten tile
                os.kill(pid, signal.SIGKILL)
                killed = True
                break
        time.sleep(0.01)
    code = job.wait(timeout=300)
    assert killed
    first = json.loads((workdir / "run_stats.json").read_text())
    missing = first["missing_tiles"]
    assert missing and code == 3                # the node's run survives a lost worker, and says tiles are missing
    assert any("exited with code -9" in e for w in first["workers"] for e in w["errors"])
    assert json.loads((workdir / "failed_tiles.json").read_text())["missing"] == missing
    assert sum(w["tiles_written"] for w in first["workers"]) == 8 - len(missing)
    stats = coordinator.run(crashed, backend="cpu")
    assert sum(s.tiles for s in stats) == sum(s.tiles_written for s in stats) == len(missing)   # only the missing tiles
    assert stats.missing == [] and json.loads((workdir / "run_stats.json").read_text())["missing_tiles"] == []
    assert not (workdir / "failed_tiles.json").exists()
    a, b = _sorted(clean.output), _sorted(crashed.output)
    assert len(a["point_id"]) == len(b["point_id"])
    for name in ("point_id", "status", "params", "objmin", "n_iter", "seed"):
        np.testing.assert_array_equal(a[name], b[name])
