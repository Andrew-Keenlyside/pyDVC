"""M0: the CCPi runner, against the real ``dvc`` executable when it is installed."""

import numpy as np
import pytest

from pydvc.bench import ccpi_baseline
from pydvc.config import RunConfig, SearchSpec, SubvolumeSpec
from pydvc.synth.phantoms import DisplacementField, make_case


def _dvc_or_skip():
    try:
        return ccpi_baseline.find_dvc()
    except FileNotFoundError:
        pytest.skip("CCPi dvc not installed (conda ccpi-dvc, or set PYDVC_CCPI_DVC)")


def test_thread_counts():
    assert ccpi_baseline.thread_counts(1) == [1]
    assert ccpi_baseline.thread_counts(6) == [1, 2, 4, 6]
    assert ccpi_baseline.thread_counts(8) == [1, 2, 4, 8]


def test_ccpi_recovers_a_translation_given_rigid_trans(tmp_path):
    _dvc_or_skip()
    u = (1.0, -2.0, 1.0)
    config = make_case(
        tmp_path / "case", shape_zyx=(64, 64, 64), field=DisplacementField("affine", {"translation": u}),
        spacing=12.0, chunk=32, shard=64,
        subvolume=SubvolumeSpec(geometry="sphere", size=20, n_samples=500),
        search=SearchSpec(dof=3, disp_max=4.0, rigid_trans=u),
    )
    cfg = RunConfig.from_yaml(config)
    single = ccpi_baseline.run_ccpi(cfg, workdir=tmp_path / "run", omp_threads=1)
    split = ccpi_baseline.run_ccpi(cfg, workdir=tmp_path / "run", omp_threads=1, processes=2)
    from pydvc.io.ccpi import read_disp

    for r in (single, split):
        d = read_disp(r.disp_path)
        assert r.points == len(d) and (d["status"] == 0).all()
        np.testing.assert_allclose(np.stack([d["u"], d["v"], d["w"]], 1), np.broadcast_to(u, (len(d), 3)), atol=1e-3)
