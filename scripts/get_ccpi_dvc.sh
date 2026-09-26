#!/bin/bash
# Unpack CCPi's `dvc` (the engine iDVC runs) from its conda package, without conda.
#
#   scripts/get_ccpi_dvc.sh                 # 22.0.0 into ~/.local/opt/ccpi-dvc-22.0.0
#   eval "$(scripts/get_ccpi_dvc.sh)"       # ... and export PYDVC_CCPI_DVC
#
# Linux x86-64 only; the binary links the system's libgomp and libstdc++ (GCC >= 9).
# Use 22.0.0: the 25.0.0 build's tricubic interpolation is broken (docs/benchmarks).
set -euo pipefail
VERSION=${1:-22.0.0}
BUILD=${2:-0}
DEST=${3:-$HOME/.local/opt/ccpi-dvc-$VERSION}
URL=https://conda.anaconda.org/ccpi/linux-64/ccpi-dvc-$VERSION-$BUILD.tar.bz2

if [ ! -x "$DEST/bin/dvc" ]; then
  mkdir -p "$DEST"
  curl -fsSL "$URL" | tar xj -C "$DEST"
fi
if ! (cd "$(mktemp -d)" && "$DEST/bin/dvc" >/dev/null 2>&1); then
  echo "error: $DEST/bin/dvc does not run; check its libraries with: ldd $DEST/bin/dvc" >&2
  exit 1
fi
echo "export PYDVC_CCPI_DVC=$DEST/bin/dvc"
