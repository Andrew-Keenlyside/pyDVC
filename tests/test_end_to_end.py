"""M3: synthetic case through every stage on one device. Opt in with ``-m slow``."""

import pytest

from pydvc.bench.metrics import against_truth
from pydvc.config import RunConfig
from pydvc.pipeline import coordinator
from pydvc.synth.phantoms import DisplacementField, make_case

pytestmark = pytest.mark.slow


def test_affine_case_end_to_end(tmp_path):
    field = DisplacementField("affine", {"strain": (0.005, -0.003, 0.002, 0.001, 0.0, -0.001), "translation": (1.2, -0.4, 0.3)})
    config_path = make_case(tmp_path / "case", shape_zyx=(128, 128, 128), field=field, spacing=16.0)
    cfg = RunConfig.from_yaml(config_path)

    coordinator.prepare(cfg)
    coordinator.seed(cfg)
    coordinator.run(cfg)
    coordinator.finalize(cfg)

    acc = against_truth(cfg.output, tmp_path / "case" / "truth.npz")
    assert acc.frac_good >= 0.99
    assert max(acc.rmse) <= 0.05
