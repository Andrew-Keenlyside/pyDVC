# Specification

This section is the reference for what zvDVC computes and what it reads and
writes: the correlation method, every run-configuration field, the image,
point and result formats, the status codes and the CCPi/iDVC file
compatibility. It is written for users who need to know exactly what a
setting does, and for anyone writing a reader or checking results against
another code. Every statement is taken from the source in
[`src/zvdvc`](https://github.com/Andrew-Keenlyside/zvDVC/tree/main/src/zvdvc);
where something is planned but not implemented, the page says so.

Each page follows the same structure: a short **Terms** section that defines
the vocabulary used on that page, a plain-English introduction, and a
technical reference with tables, formulas and examples.

---

## Design goals

**Compatibility of method, statuses and files with CCPi DVC and iDVC.**
zvDVC reimplements the local, subvolume-based DVC method of
[iDVC](https://github.com/TomographicImaging/iDVC) and of the
[CCPi DVC engine](https://github.com/TomographicImaging/DigitalVolumeCorrelation)
that iDVC runs. It uses the same sample templates, 3/6/12-DOF shape functions,
five objectives, tricubic interpolant, stopping rules and status codes 0 to −3.
It reads CCPi `dvc_in` and `.roi` files and writes `.disp` and `.stat` files
that iDVC can open. Where zvDVC differs from CCPi, the difference is listed
on the page concerned and in {doc}`/ARCHITECTURE` §10.

**GPU batching.** CCPi solves one point at a time. zvDVC solves thousands of
points at once: every point in a run shares one sample template, so a batch is
described by its centres, seeds and parameters alone, and a fused kernel
reduces each point's samples to one row of normal-equation sums in a single
pass ({doc}`method`).

**The zarr-vectors chunk grid is the work partition.** Search points and
results are [Zarr Vectors](https://github.com/AllenInstitute/zarr-vectors-py)
point clouds on the same spatial chunk grid. A *tile*, the unit of work handed
to a GPU, is a block of whole chunks, so a tile's input rows and output rows
live in the same cells ({doc}`points_store`, {doc}`results_store`).

**Resumable, lock-free writes.** Tiles never share a cell, so workers write
results without locks, following zarr-vectors' three-phase pattern (allocate,
write disjoint cells, rebuild presence). A cell counts as written only when
every result array holds it, so a resubmitted job solves only what is missing
({doc}`results_store`).

**Native dtype.** Image bricks stay in their stored dtype (u8, u16 or f32) on
the device, and the kernels convert in registers. A u16 brick costs half the
memory and half the host-to-device copy of a float32 one ({doc}`volumes`).

---

## Relationship to other projects and specifications

| Project or specification | Relationship |
|---|---|
| [iDVC](https://github.com/TomographicImaging/iDVC) ([docs](https://tomographicimaging.github.io/iDVC/)) | The graphical DVC application from the Tomographic Imaging / CCPi team at UKRI-STFC (Laura Murgatroyd, Edoardo Pasca; Apache-2.0; latest release v25.0.0, 2025). zvDVC follows its point-cloud conventions and writes the files its viewers read. `zvdvc-dvc` runs in place of CCPi's `dvc` under iDVC ({doc}`ccpi_compat`, {doc}`/IDVC`). |
| [CCPi DVC](https://github.com/TomographicImaging/DigitalVolumeCorrelation) | The C++/OpenMP correlation engine iDVC runs (GPL-3.0), initially developed by Prof. Brian K. Bay and collaborators (Bay et al. 1999, doi:[10.1007/BF02323555](https://doi.org/10.1007/BF02323555); Bay 2008, doi:[10.1243/03093247JSA436](https://doi.org/10.1243/03093247JSA436)). zvDVC's method, parameter vector, objectives and status codes are CCPi's ({doc}`method`, {doc}`status_codes`). |
| [Zarr Vectors](https://github.com/AllenInstitute/zarr_vectors) / [zarr-vectors-py](https://github.com/AllenInstitute/zarr-vectors-py) | The format for search points and results. The specification is by Forest Collman (Allen Institute); zarr-vectors-py is its implementation (BRIDGE Neuroscience). zvDVC uses only `zarr_vectors.api` and `zarr_vectors.building`, through `zvdvc.io`. |
| [OME-Zarr](https://ngff.openmicroscopy.org/) / [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) | The recommended format for the dense image volumes. `zvdvc convert` writes OME-Zarr v0.5 (Zarr v3) with sharding and zstd ({doc}`volumes`). |

---

## Pages

```{toctree}
:maxdepth: 1

method
run_config
volumes
points_store
results_store
status_codes
ccpi_compat
```
