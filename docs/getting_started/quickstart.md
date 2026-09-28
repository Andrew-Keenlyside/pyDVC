# Quickstart

This page runs zvDVC end to end on a small synthetic case with a known
displacement field, then checks the result against the truth. It needs only
the base install ({doc}`/getting_started/installation`). On the RTX A2000
workstation used for the output shown, all the commands together take about
20 s.

{doc}`/tutorials/synthetic_first_run` is the explained version of the same
run, with the outputs opened and inspected.

---

## 1. Make a synthetic case

```bash
zvdvc synth --shape 128 128 128 --field affine --out data/synth128
```

```text
/…/data/synth128/config.yaml
```

`zvdvc synth` writes a speckle volume, deforms it with a known field
(`affine` here; also `rigid`, `sinusoid` and `inclusion`), and writes
everything a run needs:

| path | contents |
|---|---|
| `ref.ome.zarr`, `def.ome.zarr` | reference and deformed volumes, OME-Zarr v3, `uint16` by default |
| `points.roi`, `points.zarrvectors` | search points on a grid (`--spacing 16` by default), as a CCPi `.roi` file and a zarr-vectors store |
| `truth.npz` | the true displacement at every point |
| `config.yaml` | the run configuration, with absolute paths |

The generated configuration uses a sphere subvolume of 32 voxels with 2 000
samples, 12-DOF, ZNSSD, tricubic interpolation, `disp_max` 8 and `wavefront`
seeding. `zvdvc synth --help` lists the flags that change these.

## 2. Run the stages

```bash
zvdvc plan     data/synth128/config.yaml
zvdvc seed     data/synth128/config.yaml
zvdvc run      data/synth128/config.yaml
zvdvc finalize data/synth128/config.yaml --disp
```

Output on a workstation GPU (trimmed):

```text
{'tiles': 1, 'points': 125, 'memory': {…, 'fits': True}, 'plan': '/…/data/synth128/run/plan.json'}
{'strategy': 'wavefront', 'points': 125, 'seconds': 0.64}
device 0: 0 tiles solved, 1 already written, 0 points, 0.00 GB read, compute 0.0 s, I/O wait n/a, status counts {}, 0 errors
RunSummary(n_points=125, seconds=1.09, counts={0: 125})
```

What each stage did:

* **`plan`** checked that the two volumes match, built the subvolume
  template, cut the points into tiles, checked device memory, allocated the
  results store and wrote `run/plan.json`.
* **`seed`** with `wavefront` seeding (the CCPi-parity mode) solves every
  point, shell by shell from the start point, and writes the results.
* **`run`** is the tiled solve. It skips tiles already in the results store,
  which is why it found nothing left to do here. With `coarse` or `rigid`
  seeding, `seed` only prepares starting displacements and `run` does the
  solving.
* **`finalize`** rebuilt the store's metadata, wrote `run/results.stat` and,
  with `--disp`, a CCPi-format `run/results.disp`. `counts={0: 125}` means
  all 125 points have status 0, `GOOD`.

Each stage picks a backend automatically: `fused` with a GPU, otherwise `cpu`
(numba), otherwise `numpy`. Pass `--backend` to choose.

## 3. Compare with the truth

```bash
zvdvc compare data/synth128/results.zarrvectors data/synth128/truth.npz
```

```text
points 125, GOOD 125 (100.00 %)  [GOOD 125]
rmse (x, y, z)  0.0010  0.0010  0.0010
bias (x, y, z)  +0.0002  +0.0000  -0.0001
|error| median 0.0014  p95 0.0030  p99 0.0035
```

Errors are in voxels: about 0.001 here, against the 0.02 voxel RMSE that
`zvdvc selftest` accepts on a noise-free case. `zvdvc compare` also accepts a CCPi `.disp` file as
the reference.

The first rows of `run/results.disp`, the file iDVC's results viewer reads:

```text
n	x	y	z	status	objmin	u	v	w
1	31.5	31.5	31.5	0	4.39302858e-05	0.672790647	-0.171503559	0.222977489
2	47.5	31.5	31.5	0	1.73230255e-05	0.973328233	-0.0970191434	0.110869206
```

## 4. Try the tiled path

To see `run` do the solving, switch the configuration to `coarse` seeding
(`seeding.strategy: coarse`; `seeding.coarse_stride: 2` suits a grid this
small) and point `output` and `workdir` somewhere new. `seed` then solves a
27-point sub-grid, and `run` solves the tile:

```text
{'strategy': 'coarse', 'coarse_points': 27, 'coarse_good': 1.0, …}
device 0: 1 tiles solved, 0 already written, 125 points, 0.01 GB read, compute 0.1 s, I/O wait 0.0 %, status counts {0: 125}, 0 errors
```

Run `zvdvc run` a second time and it reports `0 tiles solved, 1 already
written`: that is how an interrupted run resumes.

## The same run from Python

The CLI stages are thin wrappers around {py:mod}`zvdvc.pipeline.coordinator`:

```python
from zvdvc import RunConfig
from zvdvc.bench.metrics import against_truth
from zvdvc.pipeline import coordinator

cfg = RunConfig.from_yaml("data/synth128/config.yaml")
coordinator.prepare(cfg)                      # zvdvc plan
coordinator.seed(cfg)                         # zvdvc seed
coordinator.run(cfg)                          # zvdvc run: one process per visible GPU
print(coordinator.finalize(cfg, export_disp=True))
print(against_truth(cfg.output, "data/synth128/truth.npz").summary())
```

`prepare`, `seed` and `run` accept `backend=`, as the CLI's `--backend`. More in
{doc}`/tutorials/python_api`.

## Next steps

* {doc}`/getting_started/concepts`: what points, templates, tiles, bricks
  and seeding modes are.
* {doc}`/spec/run_config`: every field of `config.yaml`.
* {doc}`/tutorials/idvc_example`: iDVC's example dataset, run with CCPi DVC
  and zvDVC side by side.
* {doc}`/IDVC`: run iDVC with zvDVC as its engine.
