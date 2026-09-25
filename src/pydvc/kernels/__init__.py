"""Numerics shared by the numpy reference and the GPU path.

``interpolate`` and ``objective`` are written against an array namespace
(numpy or cupy, :mod:`pydvc.kernels.xp`). They form the reference
implementation and the unfused GPU path. ``fused`` is the production GPU
kernel and is tested against them.
"""
