"""The check suite's baseline comparison: pass, warn, fail, skip and notes, on hand-made metrics."""

import numpy as np
import pytest

from pydvc.bench.check import Finding, Step, compare, compare_with_baseline_arrays, junit_counts, overall

BASE = {"metrics": {
    "kernel.case_a.fused.sums_us": {"value": 4.0, "cv": 0.01},
    "e2e.casea_grid24.fused.seconds": {"value": 10.0},
    "casea.fused.ccpi.median": {"value": 0.040},
    "casea.fused.ccpi.status_agreement": {"value": 0.990},
}}


def _one(metrics, **kw):
    kw = {"same_hardware": True, "gpu_busy": False} | kw
    [f] = compare(metrics, BASE, **kw)
    return f


@pytest.mark.parametrize("value, status", [(4.6, "pass"), (5.2, "warn"), (5.8, "fail"), (3.0, "note")])
def test_kernel_timings_warn_at_20_and_fail_at_40_percent(value, status):
    assert _one({"kernel.case_a.fused.sums_us": {"value": value}}).status == status


@pytest.mark.parametrize("value, status", [(12.0, "pass"), (13.0, "warn"), (16.0, "fail")])
def test_end_to_end_timings_warn_at_25_and_fail_at_50_percent(value, status):
    assert _one({"e2e.casea_grid24.fused.seconds": {"value": value}}).status == status


def test_noisy_baselines_widen_the_limits():
    base = {"metrics": {"kernel.x.fused.sums_us": {"value": 4.0, "cv": 0.10}}}     # 3 cv = 30 % > 20 %
    [f] = compare({"kernel.x.fused.sums_us": {"value": 5.0}}, base, same_hardware=True, gpu_busy=False)
    assert f.status == "pass"


def test_timings_are_not_compared_across_hardware_or_on_a_busy_gpu():
    m = {"kernel.case_a.fused.sums_us": {"value": 99.0}}
    assert _one(m, same_hardware=False).status == "skip"
    assert _one(m, gpu_busy=True).status == "skip"


def test_ccpi_agreement_may_not_get_worse_than_the_baseline():
    assert _one({"casea.fused.ccpi.median": {"value": 0.044}}).status == "pass"
    assert _one({"casea.fused.ccpi.median": {"value": 0.046}}).status == "fail"
    assert _one({"casea.fused.ccpi.status_agreement": {"value": 0.9885}}).status == "pass"
    assert _one({"casea.fused.ccpi.status_agreement": {"value": 0.985}}).status == "fail"


def test_no_baseline_or_new_metric_is_a_skip():
    assert compare({"a.seconds": {"value": 1.0}}, None, same_hardware=True, gpu_busy=False)[0].status == "skip"
    assert _one({"new.metric": {"value": 1.0}}).status == "skip"


def test_overall_result():
    ok, casea_skipped = Step("tests"), Step("case A central grid", status="skip")
    assert overall([ok], [], require_case_a=False) == "PASS"
    assert overall([ok, casea_skipped], [], require_case_a=False) == "PASS (partial)"
    assert overall([ok, casea_skipped], [], require_case_a=True) == "FAIL"
    assert overall([ok], [Finding("x", "warn", "")], require_case_a=False) == "WARN"
    assert overall([ok], [Finding("x", "note", "")], require_case_a=False) == "PASS"
    assert overall([Step("t", status="fail")], [], require_case_a=False) == "FAIL"


def test_results_are_compared_with_the_baseline_arrays(tmp_path):
    status = np.zeros(1000, dtype=np.int8)
    disp = np.random.default_rng(0).normal(size=(1000, 3))
    np.savez(tmp_path / "base.npz", status=status, displacement=disp)
    np.savez(tmp_path / "same.npz", status=status, displacement=disp + 1e-5)
    moved = status.copy()
    moved[:5] = -1
    np.savez(tmp_path / "moved.npz", status=moved, displacement=disp)
    assert compare_with_baseline_arrays(tmp_path / "same.npz", tmp_path / "base.npz").status == "pass"
    assert compare_with_baseline_arrays(tmp_path / "moved.npz", tmp_path / "base.npz").status == "fail"
    assert compare_with_baseline_arrays(tmp_path / "same.npz", tmp_path / "nope.npz").status == "skip"


def test_junit_counts(tmp_path):
    (tmp_path / "r.xml").write_text('<testsuites><testsuite tests="5" failures="1" errors="0" skipped="2"/></testsuites>')
    assert junit_counts(tmp_path / "r.xml") == {"tests": 5, "failures": 1, "errors": 0, "skipped": 2}


def test_cpu_timings_are_skipped_when_other_work_loads_the_cpu():
    base = {"metrics": {"kernel.case_a.cpu.sums_us": {"value": 150.0}, "kernel.case_a.fused.sums_us": {"value": 4.0}}}
    m = {"kernel.case_a.cpu.sums_us": {"value": 260.0}, "kernel.case_a.fused.sums_us": {"value": 4.1}}
    found = {f.metric: f.status for f in compare(m, base, same_hardware=True, gpu_busy=False, cpu_busy=True)}
    assert found == {"kernel.case_a.cpu.sums_us": "skip", "kernel.case_a.fused.sums_us": "pass"}
