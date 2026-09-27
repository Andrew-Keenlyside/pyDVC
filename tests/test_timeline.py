"""Run instrumentation: event logs, telemetry parsing and the timeline summary."""

import json

import numpy as np
import pytest

from pydvc.bench import timeline
from pydvc.profiling import EventLog, GpuTelemetry


def _worker(path, device, solves, start=0.0, end=10.0):
    log = EventLog(path / f"worker-h-d{device}-p1.jsonl")
    log.record("worker_start", t=start, device=device)
    log.record("first_read", t0=start, t1=start + 1.0)
    for i, (t0, t1) in enumerate(solves):
        log.record("read", tile=i, t0=t0 - 1, t1=t0, bytes=1_000_000)
        log.record("solve", tile=i, t0=t0, t1=t1, points=100, iters=700, good=99)
    log.record("worker_end", t=end, tiles=len(solves))
    log.close()


def test_summary_of_two_workers(tmp_path):
    _worker(tmp_path, 0, [(1.0, 5.0), (5.0, 9.0)])              # 8 s solving of 10
    _worker(tmp_path, 1, [(1.0, 4.0)], end=10.0)                 # 3 s solving, idle from t = 4
    s = timeline.summarise(tmp_path)
    assert s["workers"] == 2 and s["points"] == 300 and s["makespan_s"] == pytest.approx(10.0)
    w0, w1 = sorted(s["per_worker"], key=lambda w: w["device"])
    assert w0["utilisation"] == pytest.approx(0.8) and w1["utilisation"] == pytest.approx(0.3)
    assert w0["tail_idle_s"] == pytest.approx(1.0) and w1["tail_idle_s"] == pytest.approx(6.0)
    assert s["load_imbalance"] == pytest.approx(8.0 / 5.5)
    assert s["tile_solve_s"]["n"] == 3 and "worker" in timeline.markdown(s)


def test_telemetry_parsing_skips_unavailable_fields(tmp_path):
    f = tmp_path / "gpu_telemetry.csv"
    f.write_text(GpuTelemetry.FIELDS + "\n0, t, 90, 40, 1000, 1800, 300, 60\n0, t, 70, 20, 3000, 1700, 250, 65\n"
                 "1, t, [N/A], [N/A], [N/A], [N/A], [N/A], [N/A]\n")
    g = timeline.telemetry(f)
    assert set(g) == {"0"} and g["0"]["util_gpu_mean"] == 80 and g["0"]["memory_used_max_mib"] == 3000


def test_telemetry_is_a_no_op_without_nvidia_smi(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with GpuTelemetry(tmp_path / "t.csv"):
        pass
    assert not (tmp_path / "t.csv").exists()


def test_a_pipeline_run_writes_events_the_timeline_can_read(tmp_path):
    pytest.importorskip("numba")
    import dataclasses

    from pydvc.config import ClusterSpec, RunConfig, SearchSpec, SeedingSpec, SubvolumeSpec
    from pydvc.pipeline import coordinator
    from pydvc.synth.phantoms import default_field, make_case

    shape = (64, 64, 64)
    cfg = RunConfig.from_yaml(make_case(tmp_path / "c", shape_zyx=shape, field=default_field("affine", shape), spacing=8.0,
                                        chunk=32, shard=64, subvolume=SubvolumeSpec(geometry="sphere", size=12, n_samples=300),
                                        search=SearchSpec(dof=6, disp_max=6.0)))
    cfg = dataclasses.replace(cfg, cluster=ClusterSpec(tile_shape=(32, 32, 32)), seeding=SeedingSpec(strategy="rigid"))
    coordinator.prepare(cfg, backend="cpu")
    coordinator.run(cfg, backend="cpu", cpu_workers=2, max_tiles=3)
    from pathlib import Path

    ev_dir = json.loads((Path(cfg.workdir) / "run_stats.json").read_text())["events"]
    s = timeline.summarise(ev_dir)
    assert s["workers"] == 2 and sum(w["tiles"] for w in s["per_worker"]) == 3
    assert s["points"] > 0 and np.isfinite(s["points_per_second"])
