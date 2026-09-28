API Reference
=============

zvDVC is used mainly through its command line (:doc:`cli`). The Python
package underneath is organised in dependency order: numerics at the bottom,
the pipeline and command line at the top.

.. list-table::
   :widths: 30 70

   * - :doc:`cli`
     - The ``zvdvc`` and ``zvdvc-dvc`` commands.
   * - :doc:`config`
     - ``RunConfig`` and ``PointStatus``, re-exported from ``zvdvc``.
   * - :doc:`pipeline`
     - The stages the CLI runs, and the in-memory solver.
   * - :doc:`io`
     - Volumes, zarr-vectors stores and CCPi / iDVC formats.
   * - :doc:`solver`
     - Gauss–Newton, seeding, coarse search and uncertainty.
   * - :doc:`geometry`
     - Boxes, templates, shape functions and point grids.
   * - :doc:`kernels`
     - Interpolation, objectives and the fused GPU / CPU kernels.
   * - :doc:`post_synth`
     - Strain, and synthetic phantoms.
   * - :doc:`bench`
     - Benchmarks and the local check suite.

The package is a research code base at version |release|. Names documented
here can change between versions; the command line and the on-disk formats
(:doc:`/spec/index`) are the stable surfaces.

GPU modules import CuPy only when a function needs it, so the whole package
imports, and these pages build, on a machine without a GPU.

.. toctree::
   :maxdepth: 1
   :hidden:

   cli
   config
   pipeline
   io
   solver
   geometry
   kernels
   post_synth
   bench
