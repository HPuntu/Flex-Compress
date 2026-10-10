#!/bin/bash
# Shared environment for Flex-Compress jobs (sourced, not executed).
#   FLEXC_SITE   site profile in sites/ (default: isambard; also: htc)
#   FLEXC_ENV    conda env name or path (default: first that activates from the
#                site's candidate list, e.g. $DYNAMODE_ENV, flexc, dynamode)
# The profile sets data/run paths, conda setup, per-job parallelism and the
# Slurm resource arguments used by submit.sh / resume.sh.
set -euo pipefail
_flexc_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXC_SITE="${FLEXC_SITE:-isambard}"
_flexc_profile="$_flexc_here/sites/$FLEXC_SITE.sh"
[[ -f "$_flexc_profile" ]] || { echo "flexc: unknown site '$FLEXC_SITE' (no $_flexc_profile)" >&2; return 2 2>/dev/null || exit 2; }
# shellcheck disable=SC1090
source "$_flexc_profile"
export FLEXC_SITE

flexc_conda_init
FLEXC_ENV_ACTIVE=""
if command -v conda >/dev/null 2>&1; then
  for _flexc_env in "${FLEXC_ENV_CANDIDATES[@]}"; do
    [[ -n "$_flexc_env" ]] || continue
    set +u
    if conda activate "$_flexc_env" >/dev/null 2>&1; then
      set -u
      FLEXC_ENV_ACTIVE="$_flexc_env"
      break
    fi
    set -u
  done
fi
if [[ -z "$FLEXC_ENV_ACTIVE" ]]; then
  echo "flexc: no conda env activated (tried: ${FLEXC_ENV_CANDIDATES[*]}); using $(command -v python || echo 'no python')" >&2
fi
python -c "import numpy, scipy, zarr, yaml" 2>/dev/null || {
  echo "flexc: $(command -v python) lacks numpy/scipy/zarr<3/pyyaml; set FLEXC_ENV or run sbatch/setup_env.sh" >&2
  return 2 2>/dev/null || exit 2; }

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
unset PYTHONHOME PYTHONUSERBASE
# The worker sets per-process BLAS threads itself when FLEXC_JOBS > 1.
export OMP_NUM_THREADS="$FLEXC_THREADS" OPENBLAS_NUM_THREADS="$FLEXC_THREADS" MKL_NUM_THREADS="$FLEXC_THREADS"
export NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1 MALLOC_ARENA_MAX=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/flexc-mpl-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"

# Optional CUDA MPS daemon (FLEXC_MPS=1): lets many worker processes' kernels
# run concurrently on the one H100 instead of time-slicing.  Falls back to
# plain sharing if the daemon cannot start; stopped when the job script exits.
flexc_start_mps() {
  if [[ "${FLEXC_MPS:-0}" != 1 || "${FLEXC_DEVICE:-auto}" == numpy ]]; then
    return 0
  fi
  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "flexc: FLEXC_MPS=1 but nvidia-cuda-mps-control not found; GPU shared by time-slicing"
    return 0
  fi
  export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/flexc-mps-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
  export CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY/log"
  mkdir -p "$CUDA_MPS_LOG_DIRECTORY"
  if nvidia-cuda-mps-control -d; then
    echo "flexc: CUDA MPS started ($CUDA_MPS_PIPE_DIRECTORY)"
    trap 'echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true' EXIT
  else
    echo "flexc: CUDA MPS unavailable; GPU shared by time-slicing"
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  fi
}
