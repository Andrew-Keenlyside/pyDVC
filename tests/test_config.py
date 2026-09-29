from pathlib import Path

import pytest

from zvdvc.config import RunConfig

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


def test_halo():
    data = {"volumes": {"reference": "a", "deformed": "b"}, "points": "p", "output": "o",
            "subvolume": {"geometry": "sphere", "size": 48}, "search": {"disp_max": 10}}
    assert RunConfig.from_dict(data).halo() == pytest.approx(24 + 10 + 2)       # docs: h = 36 for scenario B
    data["subvolume"] = {"geometry": "cube", "size": 20}
    assert RunConfig.from_dict(data).halo() == pytest.approx(10 * 3 ** 0.5 + 12)


def test_numpy_scalars_serialise(tmp_path):
    import numpy as np

    data = {"volumes": {"reference": "a", "deformed": "b"}, "points": "p", "output": "o",
            "seeding": {"start_point": [1.0, 2.0, 3.0]}}
    cfg = RunConfig.from_dict(data)
    import dataclasses

    cfg = dataclasses.replace(cfg, seeding=dataclasses.replace(cfg.seeding, start_point=tuple(np.float64([1, 2, 3]))))
    cfg.to_yaml(tmp_path / "c.yaml")
    assert RunConfig.from_yaml(tmp_path / "c.yaml").seeding.start_point == (1.0, 2.0, 3.0)


def _base():
    return {"volumes": {"reference": "a", "deformed": "b"}, "points": "p", "output": "o"}


def _with(section, key, value):
    data = _base()
    if section is None:
        data[key] = value
    else:
        data.setdefault(section, {})[key] = value
    return data


@pytest.mark.parametrize("section, key, value, match", [
    ("subvolume", "size", -16, r"config.subvolume.size: -16.0 is out of range"),
    ("subvolume", "size", 0, r"subvolume.size"),
    ("subvolume", "n_samples", 0, r"subvolume.n_samples: 0 is out of range"),
    ("subvolume", "n_samples", 10 ** 9, r"subvolume.n_samples: 1000000000 is out of range; it must be between 1 and 1,000,000"),
    ("subvolume", "aspect", [1, 1, 0], r"subvolume.aspect"),
    ("search", "dof", 5, r"config.search.dof: 5 is not one of \[3, 6, 12\]"),
    ("search", "objective", "ZNSSD", r"search.objective"),
    ("search", "interpolation", "spline", r"search.interpolation"),
    ("search", "disp_max", 0, r"search.disp_max: 0.0 is out of range"),
    ("search", "disp_max", -3, r"search.disp_max"),
    ("search", "disp_max", float("nan"), r"search.disp_max"),
    ("search", "disp_max", "abc", r"config.search.disp_max: expected a number, got 'abc'"),
    ("search", "max_iterations", 0, r"search.max_iterations: 0 is out of range; it must be between 1 and 255"),
    ("search", "max_iterations", 300, r"search.max_iterations: 300"),
    ("search", "obj_tol", -1, r"search.obj_tol"),
    ("search", "disp_tol", -1, r"search.disp_tol"),
    ("search", "threshold", {"gray_min": 100, "gray_max": 10}, r"search.threshold.gray_max: 10.0 is out of range"),
    ("search", "threshold", {"gray_min": 0, "gray_max": 10, "min_fraction": 2}, r"search.threshold.min_fraction"),
    ("search", "method", "icgn", r"config.search.method: 'icgn' is not implemented yet \(M5\)"),
    ("search", "report_convg_fail", "yes", r"search.report_convg_fail: expected true/false"),
    ("seeding", "strategy", "fft", r"config.seeding.strategy: 'fft' is not implemented yet \(M5\)"),
    ("seeding", "n_neighbours", 0, r"seeding.n_neighbours"),
    ("seeding", "coarse_stride", 0, r"seeding.coarse_stride"),
    ("seeding", "shell_width", 0, r"seeding.shell_width"),
    ("seeding", "start_point", [1, 2], r"seeding.start_point: expected 3 values"),
    ("seeding", "start_point", [1, "x", 2], r"config.seeding.start_point\[1\]: expected a number, got 'x'"),
    ("cluster", "tile_shape", [0, 64, 64], r"cluster.tile_shape"),
    ("cluster", "gpu_memory_fraction", 5, r"cluster.gpu_memory_fraction: 5.0 is out of range; it must be in \(0, 1\]"),
    ("cluster", "gpu_memory_fraction", 0, r"cluster.gpu_memory_fraction"),
    ("cluster", "batch_points", 0, r"cluster.batch_points"),
    ("cluster", "prefetch_depth", 0, r"cluster.prefetch_depth"),
    ("cluster", "devices", [-1], r"cluster.devices"),
    ("volumes", "prefilter_sigma", -1, r"volumes.prefilter_sigma"),
    ("volumes", "raw_header_bytes", -1, r"volumes.raw_header_bytes"),
    ("volumes", "raw_dtype", "u3", r"volumes.raw_dtype: 'u3' is not a numpy dtype"),
    (None, "uncertainty_seeds", -1, r"config.uncertainty_seeds"),
    (None, "num_points_to_process", -5, r"config.num_points_to_process"),
    ("subvolume", "size", True, r"subvolume.size: expected a number, got True"),
    ("subvolume", "seed", 1.5, r"subvolume.seed: expected an integer"),
])
def test_out_of_range_values_name_the_key(section, key, value, match):
    with pytest.raises(ValueError, match=match):
        RunConfig.from_dict(_with(section, key, value))


def test_boundary_values_are_accepted():
    cfg = RunConfig.from_dict({**_base(), "subvolume": {"n_samples": 1_000_000}, "search": {"max_iterations": 255, "obj_tol": 0},
                               "cluster": {"gpu_memory_fraction": 1.0, "batch_points": 1}, "num_points_to_process": 0,
                               "seeding": {"strategy": "coarse"}})
    assert cfg.subvolume.n_samples == 1_000_000 and cfg.search.max_iterations == 255


@pytest.mark.parametrize("key", ["volumes", "points", "output"])
def test_missing_required_key_is_a_value_error_naming_it(key):
    data = _base()
    del data[key]
    with pytest.raises(ValueError, match=f"config: missing required key '{key}'"):
        RunConfig.from_dict(data)


def test_missing_nested_key_is_named():
    with pytest.raises(ValueError, match=r"config.volumes: missing required key 'deformed'"):
        RunConfig.from_dict({"volumes": {"reference": "a"}, "points": "p", "output": "o"})


def test_bad_yaml_is_a_value_error(tmp_path):
    (tmp_path / "c.yaml").write_text("volumes: [\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        RunConfig.from_yaml(tmp_path / "c.yaml")
    (tmp_path / "c.yaml").write_text("")
    with pytest.raises(ValueError, match="config: expected a mapping"):
        RunConfig.from_yaml(tmp_path / "c.yaml")


@pytest.mark.parametrize("section, key, value", [
    ("seeding", "repair_passes", 3), ("seeding", "coarse_level", 2),
    ("cluster", "brick_dtype", "float32"), ("cluster", "scheduler", "static"),
])
def test_unused_settings_warn_when_changed(section, key, value, recwarn):
    RunConfig.from_dict(_base())                                   # defaults: silent
    assert not [w for w in recwarn if "not used yet" in str(w.message)]
    with pytest.warns(UserWarning, match=f"config.{section}.{key} = {value!r} is not used yet"):
        cfg = RunConfig.from_dict(_with(section, key, value))
    assert getattr(getattr(cfg, section), key) == value


def test_shipped_configs_load_without_warnings(recwarn):
    for path in CONFIGS:
        RunConfig.from_yaml(path)
    assert not [w for w in recwarn if "not used yet" in str(w.message)]
