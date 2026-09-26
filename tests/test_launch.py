"""M4: node discovery, the per-node tile queue across processes, and recovery from a killed worker."""

import dataclasses
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from pydvc.config import ClusterSpec, RunConfig, SearchSpec, SeedingSpec, SubvolumeSpec
from pydvc.io.results import ResultStore
from pydvc.pipeline import coordinator
from pydvc.pipeline.launch import NodeInfo, node_info, node_share
from pydvc.synth.phantoms import default_field, make_case

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
         "from pydvc.config import RunConfig; from pydvc.pipeline import coordinator; "
         f"coordinator.run(RunConfig.from_yaml({str(cfg_file)!r}), backend='cpu', cpu_workers=2)"],
    )
    pid_file = Path(crashed.workdir) / "workers" / "slot0.pid"
    store = None
    deadline = time.time() + 300
    killed = False
    while time.time() < deadline and job.poll() is None:
        if pid_file.exists():
            if store is None and Path(crashed.output).exists():
                store = ResultStore(crashed.output)
            if store is not None and len(store.written_cells()) > 0:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
                killed = True
                break
        time.sleep(0.02)
    assert job.wait(timeout=300) == 0          # the node's run survives a lost worker
    assert killed
    import json

    first = json.loads((Path(crashed.workdir) / "run_stats.json").read_text())
    missing = first["missing_tiles"]
    print(f"killed worker left {len(missing)} of 8 tiles missing")
    stats = coordinator.run(crashed, backend="cpu")
    assert sum(s.tiles for s in stats) == len(missing)                  # only the missing tiles
    assert json.loads((Path(crashed.workdir) / "run_stats.json").read_text())["missing_tiles"] == []
    a, b = _sorted(clean.output), _sorted(crashed.output)
    assert len(a["point_id"]) == len(b["point_id"])
    for name in ("point_id", "status", "params", "objmin", "n_iter", "seed"):
        np.testing.assert_array_equal(a[name], b[name])
