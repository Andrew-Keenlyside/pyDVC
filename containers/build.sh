#!/bin/bash
# Build zvdvc.sif on a Linux machine with internet access, for copying to the cluster.
#
#   bash containers/build.sh                          # CUDA 12.4, ./zvdvc.sif
#   bash containers/build.sh --cuda 12.1 --out /data/images/zvdvc.sif
#
# Picking --cuda: on a cluster GPU node (qrsh), run nvidia-smi and read the
# "Driver Version" in its header. CUDA 12.x runs on any driver >= 525 (minor
# version compatibility: CuPy compiles to native cubins, so no newer PTX JIT is
# needed), but pick the newest the driver supports:
#   driver >= 570   12.8        driver >= 560   12.6
#   driver >= 550   12.4        driver >= 525   12.1
#
# Needs: Linux x86_64, apptainer (or singularity), internet access, ~20 GB free
# under the build tmp dir, and root, sudo or --fakeroot. The build copies this
# working tree (uncommitted edits included) into the image; its commit is
# recorded in the image's labels.
set -euo pipefail

usage() {
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'
Options:
  --cuda VER    12.1, 12.4 (default), 12.6 or 12.8
  --out PATH    where to write the image (default: ./zvdvc.sif)
  --mode MODE   how to get root: auto (default), root, sudo or fakeroot
  --force       overwrite an existing image
  -h, --help    this text
EOF
}
say() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

CUDA=12.4
OUT=zvdvc.sif
MODE=auto
FORCE=0
CUPY=14.2.0
while [ $# -gt 0 ]; do
    case "$1" in
        --cuda)  CUDA="${2:?--cuda needs a value}"; shift 2 ;;
        --out)   OUT="${2:?--out needs a value}"; shift 2 ;;
        --mode)  MODE="${2:?--mode needs a value}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done
case "$CUDA" in
    12.1) BASE=nvidia/cuda:12.1.1-devel-ubuntu22.04 ;;
    12.4) BASE=nvidia/cuda:12.4.1-devel-ubuntu22.04 ;;
    12.6) BASE=nvidia/cuda:12.6.3-devel-ubuntu22.04 ;;
    12.8) BASE=nvidia/cuda:12.8.1-devel-ubuntu22.04 ;;
    *) die "--cuda must be 12.1, 12.4, 12.6 or 12.8, not '$CUDA'" ;;
esac

say "checking this machine"
[ "$(uname -s)" = Linux ]  || die "this has to run on Linux (found $(uname -s))"
[ "$(uname -m)" = x86_64 ] || die "the cluster is x86_64; building on $(uname -m) would not run there"
RUNTIME=$(command -v apptainer || command -v singularity || true)
[ -n "$RUNTIME" ] || die "no apptainer or singularity on PATH (https://apptainer.org/docs/admin/main/installation.html)"
echo "runtime  $("$RUNTIME" --version)"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEF="$REPO/containers/zvdvc.def"
for f in "$DEF" "$REPO/pyproject.toml" "$REPO/src/zvdvc/__init__.py" "$REPO/scripts/get_ccpi_dvc.sh"; do
    [ -e "$f" ] || die "missing $f; run this from a full zvDVC checkout"
done
case "$REPO" in *[[:space:]]*) die "the checkout path has a space in it ($REPO); move it" ;; esac
COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
git -C "$REPO" diff --quiet 2>/dev/null || COMMIT="$COMMIT-dirty"
echo "source   $REPO ($COMMIT)"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
mkdir -p "$(dirname "$OUT")"
[ ! -e "$OUT" ] || [ "$FORCE" = 1 ] || die "$OUT already exists; pass --force to replace it"
echo "output   $OUT"

BUILD_TMP="${APPTAINER_TMPDIR:-${SINGULARITY_TMPDIR:-${TMPDIR:-/tmp}}}"
mkdir -p "$BUILD_TMP"
free_gb=$(df -Pk "$BUILD_TMP" | awk 'NR==2 {print int($4/1048576)}')
echo "tmp      $BUILD_TMP (${free_gb} GB free)"
[ "$free_gb" -ge 20 ] || die "the build needs about 20 GB under $BUILD_TMP; point APPTAINER_TMPDIR somewhere bigger"
export APPTAINER_TMPDIR="$BUILD_TMP" SINGULARITY_TMPDIR="$BUILD_TMP"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$BUILD_TMP/apptainer-cache}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"

for url in https://pypi.org/simple/ https://github.com https://conda.anaconda.org https://registry-1.docker.io/v2/; do
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$url" || true)
    [ "$code" != 000 ] || die "cannot reach $url; the build downloads from it"
done
echo "network  ok"

if [ "$MODE" = auto ]; then
    if [ "$(id -u)" = 0 ]; then MODE=root
    elif sudo -n true 2>/dev/null; then MODE=sudo
    else MODE=fakeroot
    fi
fi
case "$MODE" in
    root)     BUILD=("$RUNTIME" build) ;;
    sudo)     BUILD=(sudo -E "$RUNTIME" build) ;;
    fakeroot) BUILD=("$RUNTIME" build --fakeroot) ;;
    *) die "--mode must be auto, root, sudo or fakeroot, not '$MODE'" ;;
esac
echo "mode     $MODE"

say "writing the definition"
FILLED="$BUILD_TMP/zvdvc.$$.def"
trap 'rm -f "$FILLED"' EXIT
sed -e "s|@BASE_IMAGE@|$BASE|g" -e "s|@CUDA@|$CUDA|g" -e "s|@REPO@|$REPO|g" \
    -e "s|@COMMIT@|$COMMIT|g" -e "s|@CUPY@|$CUPY|g" "$DEF" > "$FILLED"
grep -q '@[A-Z_]*@' "$FILLED" && die "unfilled placeholder in $FILLED"

say "building $OUT (CUDA $CUDA, base $BASE); this takes 10-20 minutes"
FORCE_FLAG=()
[ "$FORCE" = 1 ] && FORCE_FLAG=(--force)
"${BUILD[@]}" "${FORCE_FLAG[@]}" "$OUT" "$FILLED"

say "checking the image"
"$RUNTIME" exec "$OUT" python -c "import zvdvc, cupy, numba; print('zvdvc ok; cupy', cupy.__version__)"
"$RUNTIME" exec "$OUT" sh -c 'command -v nsys >/dev/null && echo "nsys ok" || echo "nsys missing (profiling runs will be skipped)"'
if command -v nvidia-smi >/dev/null 2>&1; then
    "$RUNTIME" exec --nv "$OUT" zvdvc check quick >/dev/null 2>&1 && echo "zvdvc check quick: PASS" \
        || echo "zvdvc check quick failed here; run it by hand: $RUNTIME exec --nv $OUT zvdvc check gpu"
fi
ls -lh "$OUT"
cat <<EOF

Next: copy $OUT to the cluster's shared storage, e.g.
  rsync -avP $OUT pryor:/home/akeenlys/storage_main/bridge_project_data/containers/
and follow docs/CLUSTER.md.
EOF
