"""Run configuration.

A frozen dataclass tree describes one run. Where a CCPi DVC input key exists,
it is given in the comment, so a CCPi ``dvc_in`` file maps onto this one to
one (:func:`pydvc.io.ccpi.read_dvc_input`). Axis conventions: points and
vectors are ``(x, y, z)`` in voxels; shapes and boxes of arrays are ``(z, y, x)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydvc._todo import todo

Geometry = Literal["cube", "sphere"]
Objective = Literal["sad", "ssd", "zssd", "nssd", "znssd"]
Interpolation = Literal["nearest", "trilinear", "tricubic"]
Method = Literal["fagn", "icgn"]
SeedStrategy = Literal["rigid", "wavefront", "coarse", "fft"]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class VolumeSpec:
    reference: str                       # reference_filename: OME-Zarr array URI, or .raw/.mhd/.npy/.tif
    deformed: str                        # correlate_filename
    array_path: str = "0"                # multiscale level inside an OME-Zarr group
    # Only for CCPi-style flat inputs:
    raw_shape_xyz: tuple[int, int, int] | None = None   # vol_wide, vol_high, vol_tall
    raw_dtype: str | None = None                         # from vol_bit_depth + vol_endian, e.g. "<u2"
    raw_header_bytes: int = 0                            # vol_hdr_lngth


@dataclass(frozen=True)
class SubvolumeSpec:
    geometry: Geometry = "sphere"        # subvol_geom
    size: float = 80.0                   # subvol_size: cube side or sphere diameter (voxels)
    n_samples: int = 8000                # subvol_npts; a cube rounds up to k^3 as CCPi does
    aspect: Vec3 = (1.0, 1.0, 1.0)       # subvol_aspect
    seed: int = 0                        # sphere sampling RNG; one template is shared by all points


@dataclass(frozen=True)
class ThresholdSpec:                     # subvol_thresh on
    gray_min: float                      # gray_thresh_min
    gray_max: float                      # gray_thresh_max
    min_fraction: float = 0.2            # min_vol_fract


@dataclass(frozen=True)
class SearchSpec:
    dof: Literal[3, 6, 12] = 6           # num_srch_dof
    objective: Objective = "znssd"       # obj_function
    interpolation: Interpolation = "tricubic"   # interp_type
    disp_max: float = 38.0               # disp_max (voxels, measured from the seed)
    rigid_trans: Vec3 = (0.0, 0.0, 0.0)  # rigid_trans
    basin_radius: float = 0.0            # basin_radius; 0 disables the translation grid search
    threshold: ThresholdSpec | None = None
    method: Method = "fagn"              # "fagn": CCPi parity; "icgn": inverse compositional (M5)
    max_iterations: int = 20             # CCPi min_Lev_Mar maxit
    obj_tol: float = 1e-6                # CCPi: stop when |d obj| < obj_tol
    disp_tol: float = 1e-2               # CCPi: stop when |d u| < disp_tol (voxels)
    report_convg_fail: bool = True       # CCPi's LM path never reports Convg_Fail; False reproduces that


@dataclass(frozen=True)
class SeedingSpec:
    strategy: SeedStrategy = "wavefront"
    start_point: Vec3 | None = None      # starting_point; None = first point in the cloud
    n_neighbours: int = 75               # CCPi DataCloud nbr_num_save
    shell_width: float | None = None     # wavefront shell width; None = median point spacing
    coarse_stride: int = 4               # coarse strategy: solve every n-th grid point first
    coarse_level: int = 1                # ...on this OME-Zarr pyramid level (1 = 2x downsampled, 1/8 of the bytes)
    repair_passes: int = 1


@dataclass(frozen=True)
class ClusterSpec:
    tile_shape: tuple[int, int, int] = (1024, 1024, 1024)  # voxels (z, y, x); a multiple of the point-cloud chunk
    devices: tuple[int, ...] | None = None                  # None = every visible GPU
    gpu_memory_fraction: float = 0.8
    batch_points: int | None = None                          # None = sized from free memory
    prefetch_depth: int = 2                                  # bricks in flight per GPU (double buffering)
    brick_dtype: Literal["native", "float32"] = "native"     # native keeps u8/u16 on device
    scheduler: Literal["dynamic", "static"] = "dynamic"      # dynamic = shared tile queue within a node


@dataclass(frozen=True)
class RunConfig:
    volumes: VolumeSpec
    points: str                          # point_cloud_filename: zarr-vectors store, or .roi imported by `plan`
    output: str                          # output_filename: zarr-vectors results store
    subvolume: SubvolumeSpec = field(default_factory=SubvolumeSpec)
    search: SearchSpec = field(default_factory=SearchSpec)
    seeding: SeedingSpec = field(default_factory=SeedingSpec)
    cluster: ClusterSpec = field(default_factory=ClusterSpec)
    num_points_to_process: int | None = None   # num_points_to_process
    workdir: str = "runs/default"        # plan.json, seed field, logs, .stat

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raise todo("M1", "RunConfig.from_yaml")

    @classmethod
    def from_ccpi(cls, path: str | Path) -> RunConfig:
        """Build from a CCPi ``dvc_in`` file (see :mod:`pydvc.io.ccpi`)."""
        raise todo("M4", "RunConfig.from_ccpi")

    def to_yaml(self, path: str | Path) -> None:
        raise todo("M1", "RunConfig.to_yaml")

    def halo(self) -> float:
        """Brick margin around a tile's points, in voxels.

        The subvolume's farthest sample (half-diagonal for a cube, since 6/12-DOF
        warps rotate it), plus ``disp_max``, plus the 2-voxel cubic stencil, plus
        the spread of seeds within the tile (added at plan time).
        """
        raise todo("M3", "RunConfig.halo")
