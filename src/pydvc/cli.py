"""``pydvc`` command line.

The subcommands map one to one onto :mod:`pydvc.pipeline.coordinator` stages,
plus data preparation, comparison, the CCPi drop-in, and ``solve``: the
whole-volume, single-process numpy reference used until the tiled pipeline
lands (M3). Every stage is
re-runnable, and ``run`` skips tiles that are already written.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pydvc._todo import todo


def _cfg(args: argparse.Namespace):
    from pydvc.config import RunConfig

    return RunConfig.from_yaml(args.config)


def cmd_synth(args: argparse.Namespace) -> None:
    from pydvc.config import SearchSpec, SubvolumeSpec
    from pydvc.synth.phantoms import default_field, make_case

    shape = tuple(args.shape)
    config = make_case(
        args.out,
        shape_zyx=shape,
        field=default_field(args.field, shape),
        spacing=args.spacing,
        dtype=args.dtype,
        noise_sigma=args.noise,
        chunk=args.chunk,
        shard=args.shard,
        subvolume=SubvolumeSpec(geometry=args.geometry, size=args.subvol_size, n_samples=args.subvol_npts),
        search=SearchSpec(dof=args.dof, objective=args.objective, interpolation="tricubic", disp_max=args.disp_max),
        seed=args.seed,
    )
    print(config)


def cmd_convert(args: argparse.Namespace) -> None:
    from pydvc.io.volume import convert_to_ome_zarr

    convert_to_ome_zarr(
        args.src, args.dst, chunk=args.chunk, shard=args.shard,
        shape_xyz=tuple(args.shape_xyz) if args.shape_xyz else None, dtype=args.dtype, header_bytes=args.header,
    )


def cmd_plan(args: argparse.Namespace) -> None:
    from pydvc.pipeline import coordinator

    print(coordinator.prepare(_cfg(args), backend=args.backend))


def cmd_seed(args: argparse.Namespace) -> None:
    from pydvc.pipeline import coordinator

    print(coordinator.seed(_cfg(args), backend=args.backend))


def cmd_run(args: argparse.Namespace) -> None:
    from pydvc.pipeline import coordinator

    devices = tuple(args.devices) if args.devices else None
    for st in coordinator.run(_cfg(args), backend=args.backend, devices=devices, cpu_workers=args.cpu_workers):
        busy = st.seconds_compute + st.seconds_io_wait
        wait = f"{100 * st.seconds_io_wait / busy:.1f} %" if busy else "n/a"
        print(f"device {st.device}: {st.tiles} tiles solved, {st.tiles_skipped} already written, {st.points} points, "
              f"{st.bytes_read / 1e9:.2f} GB read, compute {st.seconds_compute:.1f} s, I/O wait {wait}, "
              f"status counts {st.status_counts}, {len(st.errors)} errors")


def cmd_repair(args: argparse.Namespace) -> None:
    from pydvc.pipeline import coordinator

    coordinator.repair(_cfg(args))


def cmd_finalize(args: argparse.Namespace) -> None:
    from pydvc.pipeline import coordinator

    print(coordinator.finalize(_cfg(args), export_disp=args.disp, pyramid=args.pyramid))


def cmd_solve(args: argparse.Namespace) -> None:
    from pathlib import Path

    from pydvc.io.ccpi import RunSummary, write_stat
    from pydvc.pipeline.inmemory import run_in_memory

    cfg = _cfg(args)
    res = run_in_memory(cfg, backend=args.backend, progress=None if args.quiet else print)
    workdir = Path(cfg.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else workdir / "results.npz"
    res.save(out)
    base = out.with_suffix("")
    write_stat(base.with_suffix(".stat"), cfg, RunSummary(len(res.point_id), res.seconds, res.status_counts()))
    if args.disp:
        res.write_disp(base.with_suffix(".disp"))
    rate = len(res.point_id) / res.seconds
    print(f"{out}: {len(res.point_id)} points in {res.seconds:.1f} s ({rate:.1f} pt/s), status counts {res.status_counts()}")


def cmd_compare(args: argparse.Namespace) -> None:
    from pydvc.bench.metrics import against_disp, against_truth

    compare = against_disp if args.reference.endswith(".disp") else against_truth
    print(compare(args.results, args.reference).summary())


def cmd_selftest(args: argparse.Namespace) -> None:
    from pydvc.bench.smoke import main as selftest

    argv = ["--out", args.out] + (["--backends", *args.backends] if args.backends else []) + (["--no-ccpi"] if args.no_ccpi else [])
    raise SystemExit(selftest(argv))


def cmd_ccpi(args: argparse.Namespace) -> None:
    """Drop-in for CCPi's ``dvc <dvc_in>``: same inputs, same .disp/.stat outputs, same progress lines for iDVC."""
    raise todo("M5", "pydvc ccpi")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pydvc", description="GPU-based Digital Volume Correlation")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write a synthetic case with a known displacement field")
    s.add_argument("--shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    s.add_argument("--field", choices=["rigid", "affine", "sinusoid", "inclusion"], default="affine")
    s.add_argument("--spacing", type=float, default=16.0)
    s.add_argument("--dtype", default="uint16")
    s.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sigma, fraction of full scale (0.02 = 2 %%)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--chunk", type=int, default=128)
    s.add_argument("--shard", type=int, default=1024)
    s.add_argument("--geometry", choices=["sphere", "cube"], default="sphere")
    s.add_argument("--subvol-size", type=float, default=32.0)
    s.add_argument("--subvol-npts", type=int, default=2000)
    s.add_argument("--dof", type=int, choices=[3, 6, 12], default=12)
    s.add_argument("--objective", choices=["sad", "ssd", "zssd", "nssd", "znssd"], default="znssd")
    s.add_argument("--disp-max", type=float, default=8.0)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("solve", help="in-memory single-process solve (numpy reference; whole volumes in memory)")
    s.add_argument("config")
    s.add_argument("--backend", choices=["numpy", "cupy", "fused", "cpu"], default="numpy")
    s.add_argument("--out", help="results .npz (default: <workdir>/results.npz)")
    s.add_argument("--disp", action="store_true", help="also write a CCPi .disp next to the results")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_solve)

    s = sub.add_parser("convert", help="convert raw/mhd/npy/tiff to sharded OME-Zarr")
    s.add_argument("src")
    s.add_argument("dst")
    s.add_argument("--chunk", type=int, default=128)
    s.add_argument("--shard", type=int, default=1024)
    s.add_argument("--shape-xyz", type=int, nargs=3, metavar=("X", "Y", "Z"), help=".raw only: volume size")
    s.add_argument("--dtype", help=".raw only: numpy dtype, e.g. '<u2' or '|u1'")
    s.add_argument("--header", type=int, default=0, help=".raw only: header bytes to skip")
    s.set_defaults(func=cmd_convert)

    for name, func, help_ in [
        ("plan", cmd_plan, "tiles, bricks, memory check, result store allocation"),
        ("seed", cmd_seed, "seed field (coarse pass) or wavefront solve"),
        ("run", cmd_run, "solve this node's tiles, one process per GPU"),
        ("repair", cmd_repair, "re-seed and re-solve failed points"),
    ]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("config")
        if name != "repair":
            s.add_argument("--backend", choices=["fused", "cupy", "cpu", "numpy"], default=None,
                           help="compute engine (default: fused with a GPU, else cpu with numba, else numpy)")
        if name == "run":
            s.add_argument("--devices", type=int, nargs="+", help="GPU ordinals (default: every visible GPU)")
            s.add_argument("--cpu-workers", type=int, default=1, help="CPU backends: worker processes sharing the cores")
        s.set_defaults(func=func)

    s = sub.add_parser("finalize", help="rebuild presence, write metadata and summaries")
    s.add_argument("config")
    s.add_argument("--disp", action="store_true", help="also export CCPi .disp")
    s.add_argument("--pyramid", action="store_true", help="build a zarr-vectors pyramid for viewers")
    s.set_defaults(func=cmd_finalize)

    s = sub.add_parser("compare", help="accuracy against ground truth or a CCPi .disp")
    s.add_argument("results")
    s.add_argument("reference", help="truth.npz or .disp")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("selftest", help="check this installation end to end on a small synthetic case (PASS/FAIL)")
    s.add_argument("--out", default="runs/selftest")
    s.add_argument("--backends", nargs="+", help="default: every backend available here")
    s.add_argument("--no-ccpi", action="store_true", help="skip the CCPi dvc comparison")
    s.set_defaults(func=cmd_selftest)

    s = sub.add_parser("ccpi", help="drop-in replacement for CCPi's `dvc <dvc_in>`")
    s.add_argument("dvc_in")
    s.set_defaults(func=cmd_ccpi)

    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
