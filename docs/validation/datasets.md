# Public validation datasets

iDVC's example data (case A) has no ground truth: it shows agreement with
iDVC, not accuracy. These public datasets add a known answer or a noise floor.
None has been run yet; this page ranks them and sets out what to run and what
would count as a pass.

## Ranked shortlist

| Rank | Dataset | What it has | Use it to test |
|---|---|---|---|
| 1 | **DVC Challenge 1.0**: Croom et al., *Exp. Mech.* 61:395–410 (2021), [doi:10.1007/s11340-020-00653-x](https://doi.org/10.1007/s11340-020-00653-x); data [doi:10.18130/V3/1UOVKO](https://doi.org/10.18130/V3/1UOVKO) (CC0, 8.4 GB) | syntactic foam scanned on seven CT systems in six labs; repeat scans with no motion, and 1 mm axial and radial stage moves; the smallest volumes are 350 × 300 × 504 | noise floor (published SD 0.012–0.043 voxel), large rigid displacements, parity with iDVC |
| 2 | **DVC Challenge 2.0**: Tong et al. (2026, preprint) [doi:10.21203/rs.3.rs-9683321/v1](https://doi.org/10.21203/rs.3.rs-9683321/v1); data NIST [doi:10.18434/mds2-4129](https://doi.org/10.18434/mds2-4129) | synthetic bead volumes with imposed translation (0–1 voxel in 0.1 steps), stretch, rotation and sinusoidal fields | accuracy against a known field: interpolation bias, strain, spatial resolution |
| 3 | **Vertebra zero-strain repeats**: Tozzi et al., *J. Mech. Behav. Biomed. Mater.* 67:117–126 (2017), [doi:10.1016/j.jmbbm.2016.12.006](https://doi.org/10.1016/j.jmbbm.2016.12.006); data [doi:10.6084/m9.figshare.4308926.v2](https://doi.org/10.6084/m9.figshare.4308926.v2) (CC BY 4.0) | repeated scans of bone with no load, 16-bit DICOM; published strain errors of a commercial local DVC code | strain noise floor, against a published local code |
| 4 | **Glenoid bone**: Boulanaache et al., *Med. Eng. Phys.* 85:48–54 (2020), [doi:10.1016/j.medengphy.2020.09.009](https://doi.org/10.1016/j.medengphy.2020.09.009); data [doi:10.5281/zenodo.3539508](https://doi.org/10.5281/zenodo.3539508) (CC BY 4.0) | three unloaded, repositioned pairs and one loaded pair; MetaImage u8 | noise floor on real bone; MetaImage input as shipped |
| 5 | **spam sandstone VEC4**: [doi:10.5281/zenodo.3888347](https://doi.org/10.5281/zenodo.3888347) (CC BY 4.0), with spam's local DVC (Stamati et al., *JOSS* 5:2286, 2020) | a rigid offset and a localised shear band | parity with a second, independent local DVC code |

## Plan

1. **DVC Challenge 1.0, system XCT1** (about 100 MB per volume). Run the two
   repeat scans against the first scan on a grid, with 24, 32 and 48 voxel
   subvolumes, in zvDVC and in iDVC's engine. Pass: zvDVC's displacement SD
   within 1.5 × iDVC's on the same pair, and at most 0.05 voxel at a 32 voxel
   subvolume. Then the axial and radial 1 mm moves (about 56 voxels), seeded
   with the nominal move and without it (the coarse search). Pass: every point
   converges, and zvDVC's residual pattern matches iDVC's within 0.05 voxel
   RMS. The paper reports machine distortion of up to 0.5 voxel, which is
   real, not error.
2. **DVC Challenge 2.0, translation and stretch series.** Pass: tricubic bias
   at most 0.02 voxel over the 0–1 voxel translations, and strain error at most
   10⁻³ up to 10 % stretch with 12-DOF.
3. **Vertebra repeats**, then **spam VEC4** for a second independent code.

## Practicalities

* Most of these ship as folders of TIFF or DICOM slices. `zvdvc convert` reads
  a single multi-page TIFF (the `[tiff]` extra), so slice folders need stacking
  first, and DICOM a conversion step; a `convert` option for slice folders
  would remove that step.
* All recommended pairs fit the development workstation's 12 GB GPU.
* The DVC Challenge 2.0 archive's README could not be retrieved when this list
  was compiled, so whether ground-truth field files ship with it is
  unconfirmed; the fields can be rebuilt from their stated parameters.
