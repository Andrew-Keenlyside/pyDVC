"""M1 acceptance on small synthetic cases: phantom -> numpy reference -> accuracy (docs/MVP_PLAN.md, Q1).

Case S itself (256^3, ~2.2 k points) is run as a benchmark (docs/benchmarks/); these
are the same pipeline on 96^3 so the suite stays fast.
"""

import numpy as np
import pytest

from pydvc.bench.metrics import against_truth
from pydvc.config import RunConfig, SearchSpec, SubvolumeSpec
from pydvc.pipeline.inmemory import Results, run_in_memory
from pydvc.synth.phantoms import DisplacementField, default_field, make_case

# sinusoid: wavelength long enough for an affine subvolume (k r ~ 0.4 at radius 12)
LONG_SINE = DisplacementField("sinusoid", {"amplitude": 1.0, "wavelength": 192.0, "axis": 0, "component": 0})


@pytest.mark.parametrize(
    "kind, dof, noise, rmse_max",
    [("affine", 12, 0.0, 0.02), ("rigid", 6, 0.0, 0.02), ("sinusoid", 12, 0.0, 0.02), ("affine", 12, 0.02, 0.05)],
)
def test_synthetic_accuracy(tmp_path, kind, dof, noise, rmse_max):
    shape = (96, 96, 96)
    search = SearchSpec(dof=dof, objective="znssd", interpolation="tricubic", disp_max=8.0)
    field = LONG_SINE if kind == "sinusoid" else default_field(kind, shape)
    config = make_case(
        tmp_path / "case", shape_zyx=shape, field=field, spacing=12.0, noise_sigma=noise,
        chunk=32, shard=96, subvolume=SubvolumeSpec(geometry="sphere", size=24, n_samples=1500), search=search,
    )
    cfg = RunConfig.from_yaml(config)
    res = run_in_memory(cfg)
    res.save(tmp_path / "results.npz")
    acc = against_truth(tmp_path / "results.npz", tmp_path / "case" / "truth.npz")
    assert acc.n_points >= 27
    assert acc.frac_good >= 0.99
    assert max(acc.rmse) <= rmse_max


def test_results_round_trip_and_disp_export(tmp_path):
    from pydvc.io.ccpi import read_disp

    res = Results(
        point_id=np.array([1, 2]), xyz=np.ones((2, 3)), status=np.array([0, -1], dtype=np.int8),
        objmin=np.array([0.1, np.nan]), params=np.arange(12.0).reshape(2, 6), n_iter=np.array([3, 1], dtype=np.uint8),
        seed=np.zeros((2, 3)), seconds=1.5,
    )
    res.save(tmp_path / "r.npz")
    back = Results.load(tmp_path / "r.npz")
    np.testing.assert_array_equal(back.params, res.params)
    res.write_disp(tmp_path / "r.disp")
    d = read_disp(tmp_path / "r.disp")
    assert d["u"].tolist() == [0.0, 6.0] and d["status"].tolist() == [0, -1]
