.. zvDVC documentation master file

.. image:: _static/zvdvc-logo.png
   :width: 60%
   :align: center
   :alt: zvDVC: GPU-based Digital Volume Correlation backed by Zarr Vectors

----

**zvDVC** is GPU-based Digital Volume Correlation (DVC) for very large
tomography datasets. It measures how a material deforms between two 3-D
scans by tracking many small subvolumes from the first scan into the second.

It reimplements the local, subvolume-based method of
`iDVC <https://github.com/TomographicImaging/iDVC>`_, the graphical DVC
application from the Tomographic Imaging / CCPi team at UKRI-STFC, and of the
CCPi DVC engine that iDVC runs (Bay et al., 1999). The method, status codes and
file formats follow iDVC and CCPi, so results can be checked against them and
opened in iDVC's viewer. What changes is how the work is organised: thousands of
points are solved at once on the GPU, and each block of work reads its image
data once.

zvDVC is also a use case for `Zarr Vectors
<https://github.com/AllenInstitute/zarr-vectors-py>`_. Search points and
results live in zarr-vectors stores, and their spatial chunk grid is zvDVC's
unit of work. The project aims to show what zarr-vectors' GPU backend,
including reads over GPUDirect Storage, gains a real HPC workload.

.. container:: zv-status

   **Status.** The single-GPU pipeline is complete and measured. On iDVC's
   example dataset (4 680 points), zvDVC takes 1.6 s on an RTX A2000; CCPi DVC,
   as iDVC runs it, takes 37 min. The multi-GPU launcher is implemented but not
   yet measured on 8× H100, and GPUDirect Storage is not wired in or measured
   yet: every published number uses host reads. See :doc:`benchmarks/index`.

----

| `Source on GitHub <https://github.com/Andrew-Keenlyside/zvDVC>`__

Where to start
--------------

.. list-table::
   :widths: 35 65

   * - :doc:`getting_started/introduction`
     - What DVC is, what iDVC and CCPi DVC do, and what zvDVC changes.
   * - :doc:`getting_started/quickstart`
     - A synthetic volume pair with a known deformation, solved and checked
       in a few commands.
   * - :doc:`getting_started/concepts`
     - The mental model: points, subvolumes, tiles, bricks, batches and seeding.
   * - :doc:`spec/index`
     - The method, the run configuration and the on-disk formats in full.
   * - :doc:`tutorials/idvc_example`
     - iDVC's own example dataset, run with CCPi DVC and with zvDVC side by side.
   * - :doc:`benchmarks/index`
     - What has been measured, on what hardware, and against which baseline.
   * - :doc:`how_to/cite`
     - How to cite zvDVC, iDVC, the CCPi DVC method and Zarr Vectors.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/introduction
   getting_started/installation
   getting_started/quickstart
   getting_started/concepts
   getting_started/faq

.. toctree::
   :maxdepth: 1
   :caption: Specification
   :hidden:

   spec/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/synthetic_first_run
   tutorials/idvc_example
   IDVC
   tutorials/strain_uncertainty
   tutorials/python_api
   CLUSTER

.. toctree::
   :maxdepth: 1
   :caption: How-To Guides
   :hidden:

   how_to/choose_tiles_and_batches
   how_to/gpudirect_storage
   TESTING
   how_to/cite

.. toctree::
   :maxdepth: 1
   :caption: Benchmarks
   :hidden:

   benchmarks/index

.. toctree::
   :maxdepth: 1
   :caption: Design Notes
   :hidden:

   MVP_REVIEW
   ARCHITECTURE
   PERFORMANCE
   MVP_PLAN

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api/index
