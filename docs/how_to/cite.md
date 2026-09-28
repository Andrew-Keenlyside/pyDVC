# How to cite

zvDVC reimplements a published method and runs inside an existing ecosystem.
This page gives the references to cite when you use it, and which ones apply
to what you did.

In short: cite **zvDVC**, **iDVC**, the **CCPi DVC** engine and the **Bay et
al.** method papers whenever you publish zvDVC results; add **Zarr Vectors**
if you use or discuss the stores; add the **example dataset** if you use it.

---

## zvDVC

```bibtex
@software{zvdvc,
  author       = {Keenlyside, Andrew},
  title        = {{zvDVC: GPU-based Digital Volume Correlation backed by Zarr Vectors}},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/Andrew-Keenlyside/zvDVC},
  license      = {GPL-3.0-or-later},
  note         = {Formerly pyDVC. Reimplements the DVC method of iDVC and the
                  CCPi DVC code (B. K. Bay et al.)}
}
```

Please give the commit or version you used; `plan.json` in a run's work
directory records the configuration and the zarr-vectors commit, and the
container image records the zvDVC commit in its labels.

## iDVC

zvDVC follows iDVC's method, status codes and file formats, and is checked
against it. **Cite iDVC whenever you use zvDVC as iDVC's engine** (through the
`zvdvc-dvc` drop-in, {doc}`/IDVC`), **whenever you compare zvDVC with iDVC or
CCPi DVC**, and whenever your settings or point clouds were prepared in iDVC.

```bibtex
@software{idvc,
  author       = {Murgatroyd, Laura and Pasca, Edoardo},
  title        = {{iDVC}},
  organization = {Tomographic Imaging / CCPi, UKRI-STFC},
  year         = {2025},
  version      = {v25.0.0},
  publisher    = {GitHub},
  url          = {https://github.com/TomographicImaging/iDVC},
  note         = {Documentation: \url{https://tomographicimaging.github.io/iDVC/}.
                  Apache-2.0}
}
```

If you used a different iDVC release, change `version` and `year` to match it.

## CCPi DVC

The correlation engine that iDVC runs, and whose method zvDVC reimplements.
zvDVC's comparisons use version 22.0.0.

```bibtex
@software{ccpi_dvc,
  author       = {Bay, B. K. and others},
  title        = {{CCPi Digital Volume Correlation (DigitalVolumeCorrelation)}},
  organization = {Tomographic Imaging / CCPi, UKRI-STFC},
  publisher    = {GitHub},
  url          = {https://github.com/TomographicImaging/DigitalVolumeCorrelation},
  note         = {Code initially developed by B. K. Bay and collaborators. GPL-3.0}
}
```

Add `version` (and `year`) for the release you ran, for example 22.0.0.

## The DVC method

The method papers behind CCPi DVC, and so behind zvDVC:

```bibtex
@article{bay1999dvc,
  author  = {Bay, B. K. and Smith, T. S. and Fyhrie, D. P. and Saad, M.},
  title   = {Digital volume correlation: three-dimensional strain mapping using X-ray tomography},
  journal = {Experimental Mechanics},
  volume  = {39},
  pages   = {217--226},
  year    = {1999},
  doi     = {10.1007/BF02323555}
}

@article{bay2008dvc,
  author  = {Bay, B. K.},
  title   = {Methods and applications of digital volume correlation},
  journal = {The Journal of Strain Analysis for Engineering Design},
  volume  = {43},
  pages   = {745--760},
  year    = {2008},
  doi     = {10.1243/03093247JSA436}
}
```

## Zarr Vectors

zvDVC stores search points and results with zarr-vectors-py. If your work
uses those stores, or discusses the data layout, cite the library and the
specification. These entries are the ones zarr-vectors-py itself asks for (its
`docs/how_to/cite.md`, in the
[zarr-vectors-py repository](https://github.com/AllenInstitute/zarr-vectors-py)):

```bibtex
@software{zarr_vectors_py,
  author       = {{BRIDGE Neuroscience}},
  title        = {{zarr-vectors-py: Python tools for the Zarr Vectors}},
  year         = {2024},
  publisher    = {GitHub},
  url          = {https://github.com/AllenInstitute/zarr-vectors-py},
  note         = {Aligned to the Zarr Vectors specification by Forest Collman,
                  Allen Institute for Brain Sciences.
                  \url{https://github.com/AllenInstitute/zarr_vectors}}
}

@misc{collman_zarr_vectors,
  author       = {Collman, Forest},
  title        = {{Zarr Vectors: a cloud-native format for spatial vector data}},
  year         = {2023},
  publisher    = {GitHub},
  url          = {https://github.com/AllenInstitute/zarr_vectors}
}
```

## Example dataset

zvDVC's case A, and iDVC's own example, is this dataset:

```bibtex
@misc{lee2022magma,
  author    = {Lee, P. and Lavall{\'e}e, Y. and Bay, B.},
  title     = {{Dynamic X-ray CT of Synthetic magma for Digital Volume Correlation analysis}},
  year      = {2022},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.7363345},
  url       = {https://zenodo.org/records/7363345}
}
```

## Dependency citations

If your work relies on specific functionality from upstream dependencies,
please also cite:

| Functionality | Cite |
|---|---|
| Zarr v3 storage (volumes, stores) | [Zarr development team](https://zenodo.org/record/3773454) |
| OME-Zarr image volumes | [Moore et al., 2021](https://doi.org/10.1007/s00418-021-02029-x) |
| GPU backends (`fused`, `cupy`) | [CuPy](https://cupy.dev) |
| Multi-core CPU backend (`cpu`) | [Numba](https://numba.pydata.org) |

## Acknowledgement text

A suggested sentence for a methods section:

> "Digital volume correlation was performed with zvDVC [Keenlyside, 2026], a
> GPU reimplementation of the local subvolume method of the CCPi DVC engine
> [Bay et al., 1999; Bay, 2008] as driven by iDVC [Tomographic Imaging / CCPi,
> UKRI-STFC, 2025]. Search points and results were stored with Zarr Vectors
> [Collman, 2023] as implemented in `zarr-vectors-py` [BRIDGE Neuroscience,
> 2024]."

If you ran zvDVC from iDVC, say so: for example, "…run from iDVC v25.0.0 with
zvDVC as its correlation engine". If you compared with CCPi DVC, name its
version (22.0.0 for zvDVC's comparisons).
