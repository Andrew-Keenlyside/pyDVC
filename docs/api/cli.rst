Command line
============

``zvdvc`` has one subcommand per pipeline stage, plus data preparation,
comparison, strain, the check suite and the CCPi drop-in. Every stage can be
re-run, and ``run`` skips tiles that are already written. ``zvdvc-dvc`` is the
same drop-in as ``zvdvc ccpi``, installed under its own name for iDVC
(:doc:`/IDVC`).

This page is generated from the argument parser, so it matches the installed
version.

.. argparse::
   :module: zvdvc.cli
   :func: build_parser
   :prog: zvdvc
