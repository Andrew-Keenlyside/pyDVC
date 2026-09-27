#!/bin/bash
# Make iDVC run pyDVC instead of CCPi's dvc: write an executable named `dvc` that runs pydvc-dvc in
# the pyDVC conda environment, in a directory you put first on PATH when starting iDVC.
#
#   bash scripts/idvc/install_dvc_shim.sh                 # ~/.local/pydvc-dvc/bin/dvc, env pydvc-gpu
#   bash scripts/idvc/install_dvc_shim.sh DIR ENV_NAME
#   PATH=~/.local/pydvc-dvc/bin:$PATH idvc                # then start iDVC like this
#
# `conda run` activates the environment (CuPy needs CONDA_PREFIX for its CUDA headers) and
# --no-capture-output streams the progress lines iDVC reads.
set -euo pipefail
DIR=${1:-$HOME/.local/pydvc-dvc/bin}
ENV_NAME=${2:-pydvc-gpu}
CONDA=$(command -v conda || true)
[ -n "$CONDA" ] || { echo "ERROR: conda not on PATH" >&2; exit 1; }
"$CONDA" run -n "$ENV_NAME" pydvc-dvc version >/dev/null \
    || { echo "ERROR: pydvc-dvc not found in conda env '$ENV_NAME' (pip install -e . there first)" >&2; exit 1; }
mkdir -p "$DIR"
cat > "$DIR/dvc" <<SHIM
#!/bin/bash
# pyDVC drop-in for CCPi's dvc (written by pyDVC scripts/idvc/install_dvc_shim.sh)
exec "$CONDA" run -n "$ENV_NAME" --no-capture-output pydvc-dvc "\$@"
SHIM
chmod +x "$DIR/dvc"
echo "wrote $DIR/dvc"
"$DIR/dvc" version
echo "start iDVC with:  PATH=$DIR:\$PATH idvc"
