"""M2: the cupy and fused paths agree with the float64 numpy reference."""

import numpy as np
import pytest

from _fields import wave_field, whole_brick
from pydvc.config import SearchSpec, SubvolumeSpec
from pydvc.geometry.templates import make_template
from pydvc.solver.gauss_newton import solve_batch

pytestmark = pytest.mark.gpu

SHAPE = (64, 64, 64)


@pytest.mark.parametrize("backend", ["cupy", "fused"])
@pytest.mark.parametrize("dof", [3, 6, 12])
@pytest.mark.parametrize("kind", ["ssd", "znssd"])
def test_gpu_matches_numpy(backend, dof, kind):
    import cupy as cp

    u = (0.6, -0.3, 0.2)
    ref_np = wave_field(SHAPE)
    def_np = wave_field(SHAPE, shift_xyz=u)
    rng = np.random.default_rng(7)
    centres = rng.uniform(20.0, 44.0, size=(256, 3))
    seeds = np.zeros((256, 3))
    template = make_template(SubvolumeSpec(geometry="sphere", size=16, n_samples=1000))
    search = SearchSpec(dof=dof, objective=kind, disp_max=3.0)

    expected = solve_batch(whole_brick(ref_np), whole_brick(def_np), centres, seeds, template, search, backend="numpy")
    got = solve_batch(
        whole_brick(cp.asarray(ref_np, dtype=cp.float32)),
        whole_brick(cp.asarray(def_np, dtype=cp.float32)),
        cp.asarray(centres, dtype=cp.float32),
        cp.asarray(seeds, dtype=cp.float32),
        template,
        search,
        backend=backend,
    )

    status = cp.asnumpy(got.status)
    assert (status == expected.status).mean() >= 0.999
    good = expected.status == 0
    np.testing.assert_allclose(cp.asnumpy(got.displacement)[good], expected.displacement[good], atol=1e-3)


def test_gpu_pipeline_matches_the_cpu_engine(tmp_path):
    """M3 on a GPU: bricks read to the device, fused kernels, results store; parity with the CPU engine."""
    import dataclasses

    from pydvc.config import ClusterSpec, RunConfig, SeedingSpec
    from pydvc.io.results import ResultStore
    from pydvc.pipeline import coordinator
    from pydvc.synth.phantoms import default_field, make_case

    shape = (80, 80, 80)
    config = make_case(tmp_path / "case", shape_zyx=shape, field=default_field("affine", shape), spacing=8.0,
                       chunk=40, shard=80, subvolume=SubvolumeSpec(geometry="sphere", size=16, n_samples=500),
                       search=SearchSpec(dof=12, disp_max=8.0))
    base = RunConfig.from_yaml(config)
    out = {}
    for backend in ("cpu", "fused"):
        cfg = dataclasses.replace(base, output=str(tmp_path / f"{backend}.zarrvectors"), workdir=str(tmp_path / backend),
                                  cluster=ClusterSpec(tile_shape=(40, 40, 40)), seeding=SeedingSpec(strategy="rigid"))
        coordinator.prepare(cfg, backend=backend)
        stats = coordinator.run(cfg, backend=backend)
        assert not stats[0].errors
        r = ResultStore(cfg.output).read_all()
        o = np.argsort(r["point_id"])
        out[backend] = {k: v[o] for k, v in r.items()}
    assert (out["cpu"]["status"] == out["fused"]["status"]).mean() >= 0.999
    good = out["cpu"]["status"] == 0
    np.testing.assert_allclose(out["fused"]["displacement"][good], out["cpu"]["displacement"][good], atol=1e-3)
