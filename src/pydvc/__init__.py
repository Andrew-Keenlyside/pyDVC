"""pyDVC: GPU-based Digital Volume Correlation for very large datasets.

Scaffold: most functions raise ``NotImplementedError`` naming the milestone
(M0-M5, docs/MVP_PLAN.md) that delivers them.

Subpackages, in dependency order:

- ``pydvc.kernels``   numerics on numpy or cupy arrays (interpolation, objectives, fused GN step)
- ``pydvc.geometry``  subvolume templates, shape functions, point-cloud generation
- ``pydvc.solver``    batched Gauss-Newton, coarse search, seeding
- ``pydvc.io``        OME-Zarr volumes, zarr-vectors point clouds and results, CCPi formats
- ``pydvc.pipeline``  tiles, bricks, batches, per-GPU workers, launch, coordinator
- ``pydvc.synth``, ``pydvc.post``, ``pydvc.bench``  phantoms, strain, benchmarks
"""

from pydvc.config import RunConfig
from pydvc.status import PointStatus

__all__ = ["PointStatus", "RunConfig"]
