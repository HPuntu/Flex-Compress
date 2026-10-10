#!/bin/bash
#SBATCH --job-name=flexc-work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
# Processes one shard of a configured run using the run's frozen code snapshot.
# Submitted by sbatch/submit.sh from the copy inside the run directory.
set -euo pipefail
run=${1:?usage: worker.sbatch RUN_DIR}
printf '%s job=%s shard=%s host=%s run=%s\n' "$(date -Is)" "${SLURM_JOB_ID:-local}" \
  "${SLURM_ARRAY_TASK_ID:-0}" "$(hostname)" "$run"
# shellcheck disable=SC1091
source "$run/flexc_env.sh"
export PYTHONPATH="$run/code/src"
exec python -u -m flexcompress.run worker --run-dir "$run" --shard "${SLURM_ARRAY_TASK_ID:-0}"
