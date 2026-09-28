#!/bin/bash
# Make iDVC run zvDVC instead of CCPi's dvc: write an executable named `dvc` that runs zvdvc-dvc in
# the zvDVC conda environment, in a directory you put first on PATH when starting iDVC.
#
#   bash scripts/idvc/install_dvc_shim.sh                 # ~/.local/zvdvc-dvc/bin/dvc, env zvdvc-gpu
#   bash scripts/idvc/install_dvc_shim.sh DIR ENV_NAME
#   PATH=~/.local/zvdvc-dvc/bin:$PATH idvc                # then start iDVC like this
#
# `conda run` activates the environment (CuPy needs CONDA_PREFIX for its CUDA headers) and
# --no-capture-output streams the progress lines iDVC reads.
set -euo pipefail
DIR=${1:-$HOME/.local/zvdvc-dvc/bin}
ENV_NAME=${2:-zvdvc-gpu}
CONDA=$(command -v conda || true)
[ -n "$CONDA" ] || { echo "ERROR: conda not on PATH" >&2; exit 1; }
"$CONDA" run -n "$ENV_NAME" zvdvc-dvc version >/dev/null \
    || { echo "ERROR: zvdvc-dvc not found in conda env '$ENV_NAME' (pip install -e . there first)" >&2; exit 1; }
mkdir -p "$DIR"
cat > "$DIR/dvc" <<SHIM
#!/bin/bash
# zvDVC drop-in for CCPi's dvc (written by zvDVC scripts/idvc/install_dvc_shim.sh)
exec "$CONDA" run -n "$ENV_NAME" --no-capture-output zvdvc-dvc "\$@"
SHIM
chmod +x "$DIR/dvc"
echo "wrote $DIR/dvc"
"$DIR/dvc" version
echo "start iDVC with:  PATH=$DIR:\$PATH idvc"
