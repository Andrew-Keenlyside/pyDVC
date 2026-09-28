# Shared by the sge/*.qsub jobs: run zvDVC inside zvdvc.sif under Grid Engine (UCL CS, e.g. pryor).
# Sourced from the job, which is submitted from the zvDVC checkout (qsub -cwd). Expects SIF and BINDS;
# defines box (run a command in the container) and runs a preflight that fails in seconds, not after
# the queue wait, if the GPUs, the image or the paths are wrong.

echo "job ${JOB_ID:-<interactive>} on $(hostname), $(date -Is)"
echo "slots=${NSLOTS:-1} gpus=${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "sif=$SIF"

[ -f "$SIF" ] || { echo "FATAL: no image at $SIF; build it with containers/build.sh" >&2; exit 1; }
RUNTIME=$(command -v apptainer || command -v singularity) \
    || { echo "FATAL: no apptainer/singularity (try: module load apptainer)" >&2; exit 1; }

# Node-local scratch for CuPy's compiled kernels and zvDVC's caches, never the shared file system:
# the image is read-only, and a shared cache written by concurrent jobs corrupts.
export TMPDIR="${TMPDIR:-/tmp/zvdvc.$$}"
CACHE="$TMPDIR/zvdvc-cache"
mkdir -p "$CACHE/cupy" "$CACHE/zvdvc"

# --cleanenv keeps the host's modules and ~/.local out of the image; what the job needs is passed
# explicitly (APPTAINERENV_* for apptainer, SINGULARITYENV_* for older SingularityCE).
pass() { export "APPTAINERENV_$1=$2" "SINGULARITYENV_$1=$2"; }

# The GPUs Grid Engine gave this job, as its GPU prolog set them. Unset inside a job means the prolog
# allocated none and the job started anyway: every card on the node would then be visible, other
# people's included, so stop instead of using them.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    pass CUDA_VISIBLE_DEVICES "$CUDA_VISIBLE_DEVICES"
elif [ -n "${JOB_ID:-}" ]; then
    echo "FATAL: Grid Engine started this job without allocating it any GPUs (CUDA_VISIBLE_DEVICES unset)." >&2
    echo "  Nothing has run. Submit the job again." >&2
    exit 1
fi
pass TMPDIR             "$TMPDIR"
pass XDG_CACHE_HOME     "$CACHE"
pass CUPY_CACHE_DIR     "$CACHE/cupy"
pass ZVDVC_CACHE        "$CACHE/zvdvc"
pass OMP_NUM_THREADS    "${NSLOTS:-1}"
pass NUMBA_NUM_THREADS  "$(nproc)"
pass ZVDVC_MACHINE      "${ZVDVC_MACHINE:-$(hostname -s)-h100}"
[ -n "${ZVDVC_BASELINE_DIR:-}" ] && pass ZVDVC_BASELINE_DIR "$ZVDVC_BASELINE_DIR"
pass PATH               "/opt/venv/bin:/opt/nsight/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
pass PYTHONNOUSERSITE   "1"
pass PYTHONUNBUFFERED   "1"
pass LANG               "C.UTF-8"
pass PYTHONUTF8         "1"

bind="$TMPDIR"
# Bind each directory and, when it is or sits under a symlink (a home-dir link into /SAN, say), its
# real location too; otherwise the link dangles inside the container.
IFS=, read -r -a bind_dirs <<< "$BINDS"
for d in "${bind_dirs[@]}"; do
    [ -n "$d" ] || continue
    [ -e "$d" ] || { echo "FATAL: BINDS entry $d does not exist" >&2; exit 1; }
    bind="$bind,$d"
    real="$(readlink -f "$d")"
    [ "$real" = "$d" ] || { bind="$bind,$real"; echo "binding $d -> $real"; }
done
# Optional: a zvDVC checkout used instead of the copy baked into the image (code edits without a rebuild).
if [ -n "${ZVDVC_DEV_SRC:-}" ]; then
    ZVDVC_DEV_SRC="$(cd "$ZVDVC_DEV_SRC" && pwd -P)"
    [ -d "$ZVDVC_DEV_SRC/src/zvdvc" ] || { echo "FATAL: ZVDVC_DEV_SRC=$ZVDVC_DEV_SRC has no src/zvdvc" >&2; exit 1; }
    bind="$bind,$ZVDVC_DEV_SRC:/opt/zvdvc-dev:ro"
    pass PYTHONPATH "/opt/zvdvc-dev/src"
    pass ZVDVC_SRC  "/opt/zvdvc-dev"
    echo "using zvDVC source from $ZVDVC_DEV_SRC"
fi

box() { "$RUNTIME" exec --nv --cleanenv --bind "$bind" "$SIF" "$@"; }

# Every GPU the job holds must be usable (an exclusive-mode card held by someone else only fails later),
# and CuPy must compile a kernel with the image's NVRTC.
box python - <<'PY'
import sys
import cupy as cp
import zvdvc
n = cp.cuda.runtime.getDeviceCount()
if n == 0:
    sys.exit("FATAL: cupy sees no GPU inside the container (is --nv working?)")
print(f"zvdvc from {zvdvc.__file__}; cupy {cp.__version__}; {n} GPU(s): "
      f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}; driver {cp.cuda.runtime.driverGetVersion()}")
for i in range(n):
    try:
        with cp.cuda.Device(i):
            assert int((cp.arange(1000, dtype=cp.float32) ** 2).sum()) > 0
            assert bool(cp.ones((4096, 64), dtype=bool).all(axis=1).all())      # a CUB reduction (NVRTC + headers)
    except Exception as exc:
        sys.exit(f"FATAL: GPU {i} of {n} cannot be used: {str(exc).splitlines()[0]}")
PY
