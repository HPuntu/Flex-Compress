#!/bin/bash
# Shared environment for Flex-Compress jobs (sourced, not executed).
# Mirrors DynaMode-dev's sbatch conventions: site setup from $HOME/hpc_env.sh,
# then the DynaMode conda env (numpy/scipy/zarr<3 are all that is needed).
#   FLEXC_ENV     conda env path or name (default: $DYNAMODE_ENV, then the name dynamode-v100)
set -euo pipefail
if [[ -f "$HOME/hpc_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/hpc_env.sh"
fi
# A name is resolved by conda through its envs dirs (including CONDA_ENVS_PATH).
env_path="${FLEXC_ENV:-${DYNAMODE_ENV:-dynamode-v100}}"
if command -v conda >/dev/null 2>&1; then
  set +u
  conda activate "$env_path"
  set -u
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
unset PYTHONHOME PYTHONUSERBASE
# One target per task: let BLAS use the task's cores; keep Blosc decoding single-threaded.
threads="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1 MALLOC_ARENA_MAX=2
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/flexc-mpl-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
