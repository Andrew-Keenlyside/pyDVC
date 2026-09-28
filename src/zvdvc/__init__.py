"""zvDVC: GPU-based Digital Volume Correlation for very large datasets, backed by Zarr Vectors.

Scaffold: most functions raise ``NotImplementedError`` naming the milestone
(M0-M5, docs/MVP_PLAN.md) that delivers them.

Subpackages, in dependency order:

- ``zvdvc.kernels``   numerics on numpy or cupy arrays (interpolation, objectives, fused GN step)
- ``zvdvc.geometry``  subvolume templates, shape functions, point-cloud generation
- ``zvdvc.solver``    batched Gauss-Newton, coarse search, seeding
- ``zvdvc.io``        OME-Zarr volumes, zarr-vectors point clouds and results, CCPi formats
- ``zvdvc.pipeline``  tiles, bricks, batches, per-GPU workers, launch, coordinator
- ``zvdvc.synth``, ``zvdvc.post``, ``zvdvc.bench``  phantoms, strain, benchmarks
"""

from zvdvc.config import RunConfig
from zvdvc.status import PointStatus

__all__ = ["PointStatus", "RunConfig"]
