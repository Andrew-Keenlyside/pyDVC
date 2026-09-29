# MVP review, 2026-09-29

This page reviews zvDVC against its MVP definition ({doc}`MVP_PLAN`): what is
done and measured, what the review fixed, and what stands between the project
and a Go decision. It is for the project owner and for anyone deciding whether
to rely on zvDVC.

## Verdict

* **The software for the MVP (milestones M0–M4) is complete** and tested on one
  GPU and on CPUs.
* **Accuracy (Q1) passes on real data.** A direct comparison with iDVC's engine
  ([study](benchmarks/2026-09-29-idvc-comparison.md)) shows zvDVC agrees with
  CCPi DVC as closely as CCPi agrees with itself, and, with identical sample
  points run to convergence, to a median 2.1 × 10⁻⁵ voxel: the order of the
  difference between two compilations of CCPi (1.7 × 10⁻⁵).
* **The Go decision cannot be made yet.** Q2 (kernel throughput on H100) and Q3
  (1→8 GPU scaling) have not been measured: every measurement so far is on one
  RTX A2000. Both need the 8 × H100 campaign ({doc}`CLUSTER`).
* **The review found and fixed eight ways a run could silently give wrong
  results or lose data**, plus fourteen crash-or-unclear-error cases, each with
  a test. The test suite grew from 311 to 607 tests.

## Scorecard

"A2000" is the development workstation (RTX A2000 12 GB, 32 cores).

| Item | Status | Evidence | What remains |
|---|---|---|---|
| **Q1** synthetic accuracy | **met** | case S: RMSE 0.0011 voxel noise-free, 0.0103 with 2 % noise, 100 % GOOD ([report](benchmarks/2026-09-25-M0-M1-case-S.md)); self-test RMSE 0.0014 on the GPU, cupy and cpu engines | case S itself on the GPU |
| **Q1** case A against CCPi | **met** (revised criterion); original median test fails by 0.0012 | revised: mean difference ≤ 0.003 voxel per axis, spread ratio 0.98–1.03 ([report](benchmarks/2026-09-26-case-A-real.md)); now explained in full by the [iDVC comparison](benchmarks/2026-09-29-idvc-comparison.md): CCPi against itself gives the same median (0.051) | owner sign-off of the revised criterion, which was adopted after the measurement |
| **Q2** kernel ≥ 100 k pt/s per H100 | implemented, **not measured** | A2000 only, at case A settings: 3.7 µs per point-iteration, about 15 % of FP32 peak by operation count | H100 run at scenario-B settings; Nsight counters |
| **Q3** 1→8 GPU efficiency ≥ 85 % | harness only, **not measured** | `bench/scaling.py`, `bench/campaign.py`, `sge/benchmark.qsub`; the multi-GPU launcher has never run on more than one GPU | the 8 × H100 campaign |
| **Q4** data path at 10⁷ points | partial | zarr-vectors validation passes (case M, 227 k points); I/O wait measured only on the CPU engine | a tiled GPU run with I/O wait; a 10⁷-point store timing |
| **Q5** attribution | partial | CCPi, CPU engine and GPU on case A: 2.1 / 615 / 2 928 points/s | same-case runs at scenario-B scale; rewrite {doc}`PERFORMANCE` from measurements |
| M0 baselines | met | CCPi on S (full sweep) and A (two configurations) | a thread curve for A |
| M1 numpy reference | met | Catmull-Rom = CCPi's interpolant to 1e-12; analytic Jacobians against finite differences | time the numpy engine on case A (done here: 424 s) |
| M2 GPU kernels | met on A2000 | GPU against numpy ≤ 1e-3 on GOOD points; 74 GPU tests pass | one H100 measurement |
| M3 tiled data path | met on CPU engines | tiled = in-memory bit for bit on the CPU engine | the same check on the GPU engine |
| M4 multi-GPU, resume | implemented; CPU workers only | kill a worker, resubmit, identical store (test now asserts something really went missing) | GPU workers; 4096³ end to end under 10 min; the SLURM/SGE chains have never run |

The milestone status blocks in {doc}`MVP_PLAN` date from 2026-09-25; this page
supersedes them.

## Direct comparison with iDVC

[The study](benchmarks/2026-09-29-idvc-comparison.md) runs CCPi DVC 22.0.0 the
way iDVC runs it, and zvDVC in its parity mode, on iDVC's example data. In
short:

| What differs between the two results | Median \|Δu\| (voxel) |
|---|---|
| CCPi against CCPi, iDVC's settings (sphere subvolumes) | 0.0513 |
| zvDVC against CCPi, iDVC's settings | 0.0511 |
| zvDVC against CCPi, identical sample points (cube), default stopping | 0.0010 |
| CCPi release against CCPi rebuilt from source (cube) | 1.7 × 10⁻⁵ |
| zvDVC against CCPi, identical sample points, both run to convergence | 2.1 × 10⁻⁵ |

Bit-for-bit agreement is impossible: CCPi seeds each sphere's random sample
points from the clock, per point and per run, so it cannot reproduce itself.
With cube subvolumes CCPi is deterministic (identical across thread counts),
and zvDVC matches it to within its stopping tolerance, or to rebuild-level
differences when both converge.

## Hardening

A failure-mode review probed invalid inputs, interrupted runs, mixed settings,
numerical edge cases and the CLI. Fixed, each with a test that failed before
the fix:

| # | Was | Now |
|---|---|---|
| 1 | Re-running against an existing output silently reused results made with other settings, volumes or points | Every results store and plan carries a fingerprint of the settings, volume identities and points; `prepare`, `seed` and `run` refuse a mismatch and name what differs |
| 2 | A failed `zvdvc-dvc` rerun truncated the previous `.disp`/`.stat`, and iDVC reported success | Everything is validated before any output is touched; outputs are replaced atomically on success only; one `input file problem:` line, no traceback |
| 3 | `zvdvc convert SRC SRC` wiped the source; existing outputs were overwritten | Refused; `--overwrite` needed; writes go to a temporary sibling and are renamed |
| 4 | Big-endian volumes gave wrong results on the GPU tiled path and failed on the CPU engine | Converted to native byte order on read, for every engine |
| 5 | Duplicate point ids received other points' results | Rejected on import, with non-finite coordinates and empty clouds |
| 6 | A featureless (constant) reference subvolume could be reported GOOD | `SINGULAR` on every engine and objective, as are non-finite results |
| 7 | A wrong raw layout (bit depth, dimensions, header) was accepted and every point came back GOOD | The file size must match the description exactly; the error says by what factor it differs |
| 8 | `zvdvc run` exited 0 with tiles missing, and `finalize` exported a partial result | Exit code 3; `finalize` refuses unwritten cells unless `--allow-partial` |

Also fixed: the seeding strategy is enforced by `run`; wavefront seeding now
stores `displacement_sd`; a failed plan leaves no results store behind; the
CCPi importer goes through full config validation; the CLI prints one-line
errors (exit 2, `--traceback` to debug); MetaImage and OME-Zarr inputs get
clear errors; unstored OME-Zarr chunks are reported; every `.disp` writer
treats failed points the same way; a GPU out-of-memory in the in-memory path
falls back to the CPU engine; `RunConfig` range-checks every field.

Still open:

* The plan-time memory check measures host memory when run on a machine
  without a GPU (for example a login node).
* SGE array jobs are not recognised as separate nodes, and two runs on the same
  output are not locked against each other.
* A killed process can leave its `nvidia-smi` telemetry sampler running.
* The default `workdir` is shared between configs in the same directory (now
  caught by the plan check, but still a trap).
* `num_points_to_process` is honoured by the in-memory path and the drop-in,
  not by the tiled `run`.
* `repair` (M5) is not implemented.

## Tests

607 tests: 533 run everywhere, 74 need a GPU. Added by this review: CLI tests
end to end (`tests/test_cli.py`), worker read/solve/write failure and requeue,
a partially written cell, the M4 kill test made non-vacuous, the fingerprint
and plan checks, config range validation, drop-in failure handling over 17 bad
inputs, volume format checks, big-endian round trips, and constant/non-finite
subvolumes on every engine.

Remaining gaps, in priority order:

1. Tiled GPU against in-memory GPU, bit for bit (M3's criterion on the GPU).
2. A two-GPU spawn and kill test (runs on the cluster).
3. Multi-node shares (`SLURM_NODEID` / `SLURM_JOB_NUM_NODES`).
4. Tighten the end-to-end RMSE test from 0.05 to Q1's 0.02; stop turning every
   `NotImplementedError` into a skip (`tests/conftest.py`), which hides
   regressions that reach a stub.
5. Mark tests that use the GPU incidentally (the prefilter uses it when present).

## Validation datasets

Case A has no ground truth: it can show agreement with CCPi, not accuracy. These
public datasets add ground truth or a noise floor, ranked by value for the
effort:

| Rank | Dataset | What it has | Use it to test |
|---|---|---|---|
| 1 | **DVC Challenge 1.0**: Croom et al., *Exp. Mech.* 61:395–410 (2021), [doi:10.1007/s11340-020-00653-x](https://doi.org/10.1007/s11340-020-00653-x); data [doi:10.18130/V3/1UOVKO](https://doi.org/10.18130/V3/1UOVKO) (CC0, 8.4 GB) | syntactic foam on six labs' CT systems; repeat scans with no motion, and 1 mm axial and radial stage moves; smallest volumes 350 × 300 × 504 | noise floor (published SD 0.012–0.043 voxel), large rigid displacements, parity with CCPi |
| 2 | **DVC Challenge 2.0**: Tong et al. (2026, preprint) [doi:10.21203/rs.3.rs-9683321/v1](https://doi.org/10.21203/rs.3.rs-9683321/v1); data NIST [doi:10.18434/mds2-4129](https://doi.org/10.18434/mds2-4129) | synthetic bead volumes with imposed translation (0–1 voxel), stretch, rotation and sinusoidal fields | accuracy against ground truth: interpolation bias, strain, spatial resolution |
| 3 | **Vertebra zero-strain repeats**: Tozzi et al., *J. Mech. Behav. Biomed. Mater.* 67:117–126 (2017), [doi:10.1016/j.jmbbm.2016.12.006](https://doi.org/10.1016/j.jmbbm.2016.12.006); data [doi:10.6084/m9.figshare.4308926.v2](https://doi.org/10.6084/m9.figshare.4308926.v2) (CC BY) | repeated scans of bone with no load; published strain errors of a local DVC code (DaVis) | strain noise floor against a published local code |
| 4 | **Glenoid bone**: Boulanaache et al., *Med. Eng. Phys.* 85:48–54 (2020), [doi:10.1016/j.medengphy.2020.09.009](https://doi.org/10.1016/j.medengphy.2020.09.009); data [doi:10.5281/zenodo.3539508](https://doi.org/10.5281/zenodo.3539508) (CC BY) | three unloaded repositioned pairs and one loaded pair, MetaImage u8 | noise floor on real bone; MetaImage input as shipped |
| 5 | **spam VEC4 sandstone**: [doi:10.5281/zenodo.3888347](https://doi.org/10.5281/zenodo.3888347) (CC BY), with spam's local DVC (Stamati et al., *JOSS* 5:2286, 2020) | a rigid offset plus a localised shear band | parity with a second, independent local DVC code |

Most of these ship as folders of TIFF or DICOM slices. `zvdvc convert` reads a
single multi-page TIFF (the `[tiff]` extra), so slice folders need stacking
first, and DICOM a conversion step; a `convert` option for slice folders would
remove that step. All recommended pairs fit the A2000.

Suggested plan: DVC Challenge 1.0 system XCT1 first (smallest; noise floor and
rigid moves, zvDVC and CCPi side by side, pass if zvDVC's noise SD is within
1.5× of CCPi's on the same pair and ≤ 0.05 voxel at a 32-voxel subvolume),
then DVC Challenge 2.0's translation and stretch series (pass if tricubic
bias ≤ 0.02 voxel and strain error ≤ 10⁻³ up to 10 % stretch).

**Update, 2026-09-29:** both run, with zvDVC and CCPi side by side
({doc}`validation/datasets`). Every criterion passes; the strain criterion
passes for strain fitted to the whole field, and pointwise strain was not
assessed. The runs also found two CCPi robustness problems: segmentation faults
near the image edge, and wrong points reported GOOD at stretches of 15 % and more.

## What is needed for Go

1. The 8 × H100 campaign: Q2 (kernel at scenario-B settings, with Nsight
   counters), Q3 (1→8 GPU efficiency), and the case L end-to-end run. The
   container, the SGE scripts and the campaign harness exist; none has run on
   the cluster.
2. Sign-off of the revised Q1 criterion, now backed by the direct comparison.
3. A first ground-truth validation (DVC Challenge 1.0 XCT1 and 2.0 translation
   series). Done 2026-09-29 ({doc}`validation/datasets`).
