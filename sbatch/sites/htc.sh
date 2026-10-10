#!/bin/bash
# shellcheck disable=SC2034  # variables are used by flexc_env.sh, submit.sh and launch.sh
# Site profile: Oxford stats HTC cluster (stat-opig).  Select with FLEXC_SITE=htc.
# Sourced by sbatch/flexc_env.sh.  Every value can be overridden from the environment.
# CPU allocations are per task, so many small one-target-at-a-time worker tasks.

FLEXC_ROOT="${FLEXC_ROOT:-/data/stat-opig/$USER}"
export DYNAMODE_ATLAS_ARCHIVE="${DYNAMODE_ATLAS_ARCHIVE:-$FLEXC_ROOT/DATA/ATLAS/compressed_bb_zarr}"
export DYNAMODE_MDCATH_ARCHIVE="${DYNAMODE_MDCATH_ARCHIVE:-$FLEXC_ROOT/DATA/mdCATH/compressed_bb_zarr}"
FLEXC_RUNS_ROOT="${FLEXC_RUNS_ROOT:-$FLEXC_ROOT/DATA/flex-compress-runs}"

FLEXC_CONDA="${FLEXC_CONDA:-}"
FLEXC_ENV_CANDIDATES=("${FLEXC_ENV:-}" "${DYNAMODE_ENV:-}" "${CONDA_ENVS_PATH:-}/dynamode-v100" dynamode-v100 flexc)

flexc_conda_init() {
  # Same site setup as DynaMode-dev's launchers.
  if [[ -f "$HOME/hpc_env.sh" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$HOME/hpc_env.sh"
    set -u
  fi
  if [[ -n "$FLEXC_CONDA" && -f "$FLEXC_CONDA/bin/activate" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$FLEXC_CONDA/bin/activate"
    set -u
  fi
}

FLEXC_DEVICE="${FLEXC_DEVICE:-numpy}"   # CPU partition; set auto/cuda with a GPU request
FLEXC_MPS="${FLEXC_MPS:-0}"
FLEXC_JOBS="${FLEXC_JOBS:-1}"
FLEXC_MEM_BUDGET_GB="${FLEXC_MEM_BUDGET_GB:-0}"       # 0 = 85% of the job's cgroup limit
FLEXC_GPU_MEM_BUDGET_GB="${FLEXC_GPU_MEM_BUDGET_GB:-0}"
FLEXC_THREADS="${FLEXC_THREADS:-${FLEXC_CPUS:-4}}"
# Worker tasks per round, all pulling from the run's shared queue; a run
# continues round by round (at most FLEXC_MAX_ROUNDS) until every target is done.
FLEXC_SHARDS_PILOT="${FLEXC_SHARDS_PILOT:-8}"
FLEXC_SHARDS_FULL="${FLEXC_SHARDS_FULL:-64}"
FLEXC_MAX_ROUNDS="${FLEXC_MAX_ROUNDS:-6}"

flexc_sbatch_args=(--partition="${FLEXC_PARTITION:-long}" --nodes=1 --cpus-per-task="${FLEXC_CPUS:-4}"
                   --mem="${FLEXC_MEM:-24G}" --time="${FLEXC_TIME:-24:00:00}")
flexc_agg_args=(--partition="${FLEXC_PARTITION:-long}" --nodes=1 --cpus-per-task=4 --mem=32G
                --time="${FLEXC_AGG_TIME:-04:00:00}")
