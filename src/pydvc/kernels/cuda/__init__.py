"""CUDA sources for the fused kernels, and a host emulator that runs them without a GPU."""

from pathlib import Path

SOURCE = Path(__file__).with_name("fused_gn.cu")
EMU_HEADER = Path(__file__).with_name("cuda_emu.h")

# Kernel parameter lists: (C type, name). Shared by the cupy launcher and the emulator's wrappers.
SIGNATURES: dict[str, list[tuple[str, str]]] = {
    "sample_values": [
        ("const T*", "brick"), ("const int*", "geom"), ("const int*", "cint"), ("const float*", "cfrac"),
        ("const float*", "params"), ("const float*", "offsets"), ("int", "B"), ("int", "M"),
        ("float*", "values"), ("unsigned char*", "inside"),
    ],
    "gn_sums": [
        ("const T*", "brick"), ("const int*", "geom"), ("const int*", "cint"), ("const float*", "cfrac"),
        ("const float*", "params"), ("const int*", "active"), ("int", "n_active"), ("const float*", "q"),
        ("const float*", "shift"), ("const float*", "offsets"), ("int", "M"), ("float*", "sums"), ("int*", "outside"),
    ],
    "gn_solve": [
        ("const float*", "sums"), ("const int*", "outside"), ("const int*", "active"), ("int", "n_active"),
        ("const float*", "shift"), ("const float*", "seeds"), ("float*", "params"), ("float*", "prev_obj"),
        ("float*", "objmin"), ("unsigned char*", "n_iter"), ("signed char*", "status"), ("int*", "done"),
        ("float", "obj_tol"), ("float", "disp_tol"), ("float", "disp_max"), ("float", "pivot_tol"),
        ("float", "featureless"),
    ],
}
