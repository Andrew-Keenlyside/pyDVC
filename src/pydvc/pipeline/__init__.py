"""Execution at scale: tiles -> bricks -> batches, one process per GPU.

``tiling``       plan tiles from the point-cloud chunk grid; brick boxes; memory; LPT assignment
``batching``     batch sizing and spatially coherent ordering inside a tile
``worker``       per-GPU loop: read tile, prefetch next bricks, solve, write
``launch``       rank discovery (SLURM / torchrun / MPI) and process-per-GPU spawning
``coordinator``  prepare / seed / run / repair / finalize, the stages behind the CLI
``inmemory``     whole-volume, single-process runner: the numpy reference path (M1) and parity runs
"""
