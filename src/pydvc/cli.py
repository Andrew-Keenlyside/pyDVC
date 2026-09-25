"""``pydvc`` command line.

The subcommands map one to one onto :mod:`pydvc.pipeline.coordinator` stages,
plus data preparation, comparison and the CCPi drop-in. Every stage is
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
    raise todo("M0", "pydvc synth")


def cmd_convert(args: argparse.Namespace) -> None:
    from pydvc.io.volume import convert_to_ome_zarr

    convert_to_ome_zarr(args.src, args.dst, chunk=args.chunk, shard=args.shard)


def cmd_plan(args: argparse.Namespace) -> None:
    from pydvc.pipeline import coordinator

    coordinator.prepare(_cfg(args))


def cmd_seed(args: argparse.Namespace) -> None:
    from pydvc.pipeline import coordinator

    coordinator.seed(_cfg(args))


def cmd_run(args: argparse.Namespace) -> None:
    from pydvc.pipeline import coordinator

    coordinator.run(_cfg(args))


def cmd_repair(args: argparse.Namespace) -> None:
    from pydvc.pipeline import coordinator

    coordinator.repair(_cfg(args))


def cmd_finalize(args: argparse.Namespace) -> None:
    from pydvc.pipeline import coordinator

    coordinator.finalize(_cfg(args), export_disp=args.disp, pyramid=args.pyramid)


def cmd_compare(args: argparse.Namespace) -> None:
    raise todo("M1", "pydvc compare")


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
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("convert", help="convert raw/mhd/npy/tiff to sharded OME-Zarr")
    s.add_argument("src")
    s.add_argument("dst")
    s.add_argument("--chunk", type=int, default=128)
    s.add_argument("--shard", type=int, default=1024)
    s.set_defaults(func=cmd_convert)

    for name, func, help_ in [
        ("plan", cmd_plan, "tiles, bricks, memory check, result store allocation"),
        ("seed", cmd_seed, "seed field (coarse pass) or wavefront solve"),
        ("run", cmd_run, "solve all tiles on this node's GPUs"),
        ("repair", cmd_repair, "re-seed and re-solve failed points"),
    ]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("config")
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

    s = sub.add_parser("ccpi", help="drop-in replacement for CCPi's `dvc <dvc_in>`")
    s.add_argument("dvc_in")
    s.set_defaults(func=cmd_ccpi)

    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
