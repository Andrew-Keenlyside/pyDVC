"""The cluster campaign's own logic: tile size, resume and redo, stopping on setup failures, the report."""

import json

import pytest

from zvdvc.bench import campaign
from zvdvc.bench.campaign import Campaign


def _c(tmp_path, **kw):
    return Campaign(out=tmp_path / "out", data_dir=tmp_path / "data", log=lambda m: None, **kw)


@pytest.mark.parametrize("size, devices, edge", [(2048, (1, 2, 4, 8), 512), (4096, (1, 2, 4, 8), 1024),
                                                  (2048, (1, 2, 4), 512), (256, (1,), 128), (2048, (1,), 1024)])
def test_tiles_give_every_gpu_at_least_four(tmp_path, size, devices, edge):
    c = _c(tmp_path, size=size, devices=devices)
    assert c.tile_edge() == edge and (size // edge) ** 3 >= 4 * max(devices)


def test_finished_steps_are_skipped_and_redo_reruns_them(tmp_path, monkeypatch):
    calls = []
    for name in campaign.STEPS:
        monkeypatch.setattr(campaign, f"step_{name}", lambda c, name=name: calls.append(name) or {"x": 1})
    c = _c(tmp_path)
    assert set(campaign.run(c, ["env", "kernel"]).values()) == {"ok"}
    calls.clear()
    status = campaign.run(c, ["env", "kernel"], redo=["kernel"])
    assert status == {"env": "done", "kernel": "ok"} and calls == ["kernel"]
    assert json.loads((c.out / "kernel.json").read_text())["x"] == 1


def test_a_failed_setup_step_stops_the_campaign(tmp_path, monkeypatch):
    def boom(c):
        raise RuntimeError("no GPU")

    monkeypatch.setattr(campaign, "step_check", boom)
    monkeypatch.setattr(campaign, "step_env", lambda c: {})
    status = campaign.run(_c(tmp_path), ["env", "check", "kernel"])
    assert status == {"env": "ok", "check": "failed"} and (tmp_path / "out" / "check.error.txt").exists()


def test_report_renders_the_tables_it_has(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "kernel.json").write_text(json.dumps({"synthetic_uint16": {"sums_us_per_point_iter": 1.5, "sample_us_per_point": 2.0,
                                                                      "rel_err": 1e-6}}))
    row = {"devices": 8, "tiles": 64, "seconds": 10.0, "points_per_second": 1e5, "efficiency": 0.9, "utilisation": 0.85,
           "load_imbalance": 1.1, "tail_idle_s_mean": 0.5, "io_wait_fraction": 0.02}
    (out / "strong.json").write_text(json.dumps({"rows": [row], "warnings": []}))
    md = campaign.report_markdown(out)
    assert "| synthetic_uint16 | 1.50 |" in md and "| 8 | 64 | 10.0 | 100000 | 90 % | 85 % | 1.10 |" in md
    assert "Weak scaling" not in md
