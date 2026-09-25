from pathlib import Path

import pytest

from pydvc.config import RunConfig

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_yaml_round_trip(path, tmp_path):
    cfg = RunConfig.from_yaml(path)
    cfg.to_yaml(tmp_path / "copy.yaml")
    assert RunConfig.from_yaml(tmp_path / "copy.yaml") == cfg


def test_types_are_restored():
    cfg = RunConfig.from_yaml(CONFIGS[0])
    assert isinstance(cfg.cluster.tile_shape, tuple)
    assert isinstance(cfg.search.rigid_trans, tuple) and all(isinstance(v, float) for v in cfg.search.rigid_trans)


def test_unknown_key_is_an_error():
    with pytest.raises(ValueError, match="unknown keys"):
        RunConfig.from_dict({"volumes": {"reference": "a", "deformed": "b"}, "points": "p", "output": "o", "serach": {}})


def test_invalid_choice_is_an_error():
    data = {"volumes": {"reference": "a", "deformed": "b"}, "points": "p", "output": "o", "search": {"dof": 7}}
    with pytest.raises(ValueError, match="dof"):
        RunConfig.from_dict(data)


def test_threshold_section():
    data = {
        "volumes": {"reference": "a", "deformed": "b"}, "points": "p", "output": "o",
        "search": {"threshold": {"gray_min": 10, "gray_max": 200}},
    }
    cfg = RunConfig.from_dict(data)
    assert cfg.search.threshold.gray_min == 10.0 and cfg.search.threshold.min_fraction == 0.2
