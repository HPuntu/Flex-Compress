#!/bin/bash
# One-off: create a light `flexc` conda env.  Not needed on Isambard-AI, where the
# project's dynamode-gh200 env (CUDA PyTorch) is used by default.
#
#   bash sbatch/setup_env.sh                 # CPU-only (numpy backend)
#   FLEXC_TORCH=1 bash sbatch/setup_env.sh   # + CUDA PyTorch (cu128 aarch64 wheels, as BriCS documents)
#   FLEXC_SITE=htc bash sbatch/setup_env.sh
#
# Reusing another env also works: export FLEXC_ENV=<name or path>.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXC_SITE="${FLEXC_SITE:-isambard}"
# shellcheck disable=SC1090
source "$here/sites/$FLEXC_SITE.sh"
flexc_conda_init
if ! command -v conda >/dev/null 2>&1; then
  arch=$(uname -m)
  cat >&2 <<EOF
No conda found (looked for ${FLEXC_CONDA:-site setup}/bin/activate).  Install Miniforge first, as BriCS recommends:
  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh
  bash Miniforge3-Linux-${arch}.sh -b -p ${FLEXC_CONDA:-\$HOME/miniforge3}
then re-run this script.
EOF
  exit 1
fi
name="${FLEXC_ENV:-flexc}"
conda create -y -n "$name" -c conda-forge --override-channels \
  "python=3.10" "numpy>=1.24,<2" "scipy>=1.10" "zarr>=2.16,<3" "numcodecs>=0.11,<0.16" \
  "pyyaml>=6" "matplotlib>=3.7" "pytest>=8"
set +u
conda activate "$name"
set -u
if [[ "${FLEXC_TORCH:-0}" == 1 ]]; then
  pip install torch --index-url "${FLEXC_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
fi
python -c "import numpy, scipy, zarr, numcodecs, yaml, matplotlib, platform; print('ok', platform.machine(), numpy.__version__, scipy.__version__, zarr.__version__)"
echo "Created env '$name'.  Test from the repo root:  PYTHONPATH=src:DynaMode-dev/src python -m pytest -q"
