#!/bin/bash
# Configure a frozen run over a dataset's whole training split and submit it.
# The run continues round by round on its own until every target is done:
#
#   bash sbatch/submit.sh configs/stage01_atlas.yaml          # full training split (default)
#   bash sbatch/submit.sh configs/stage01_mdcath.yaml
#   bash sbatch/submit_train.sh                               # both datasets' training splits
#   bash sbatch/submit.sh configs/stage01_atlas.yaml pilot    # optional smoke test: 32 domains
#   FLEXC_SITE=htc bash sbatch/submit.sh configs/stage01_atlas.yaml
#
# Each round is a worker array whose tasks all pull from the run's shared
# queue (largest targets first), followed by a finalize job that submits the
# next round while targets remain and aggregates what is done (launch.sh).
#
# Run from the Flex-Compress checkout on a login node.  The site profile
# (sbatch/sites/$FLEXC_SITE.sh, default isambard) supplies data and run paths,
# the conda env, per-task parallelism and Slurm resources.  Common overrides:
#   FLEXC_SITE         isambard (default) | htc
#   FLEXC_ROOT         site root (isambard: /projects/u6xq/hew)
#   FLEXC_RUNS_ROOT    parent of run directories (isambard: $FLEXC_ROOT/flex-compress-runs)
#   FLEXC_ENV          conda env name or path (isambard default: dynamode-gh200)
#   FLEXC_SHARDS       worker tasks per round (isambard: 16 GH200s full / 1 pilot; htc: 64 / 8)
#   FLEXC_MAX_ROUNDS   rounds before a run stops on its own (isambard: 4 x 24 h; htc: 6)
#   FLEXC_JOBS, FLEXC_THREADS   max proteins in flight per task x BLAS threads (isambard: 32 x 2)
#   FLEXC_MEM_BUDGET_GB, FLEXC_GPU_MEM_BUDGET_GB   memory for admitting proteins (isambard: 95 / 80)
#   FLEXC_DEVICE       compute backend: auto (isambard: H100 if the env has CUDA torch) | numpy | cuda
#   FLEXC_MPS          1 = start CUDA MPS inside each task so worker processes share the GPU concurrently
#   FLEXC_CONCURRENCY  max simultaneous tasks of a round (default: all)
#   FLEXC_TIME, FLEXC_PARTITION, FLEXC_CPUS, FLEXC_MEM   Slurm resources (see profile)
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
usage="usage: submit.sh CONFIG.yaml [full|pilot] [--set key=value ...]"
config=${1:?$usage}
shift
mode=full
if [[ $# -gt 0 && $1 != --* ]]; then
  mode=$1
  shift
fi
case "$mode" in pilot|full) ;; *) echo "$usage" >&2; exit 2 ;; esac
[[ -f "$repo/DynaMode-dev/src/dynamode/dataloader/compressed_backbone.py" ]] || {
  echo "DynaMode-dev submodule (dev branch) missing: git submodule update --init" >&2; exit 2; }

# shellcheck disable=SC1091
source "$repo/sbatch/flexc_env.sh"
name=$(basename "$config" .yaml)
run="$FLEXC_RUNS_ROOT/$name-$mode-$(date +%Y%m%d-%H%M%S)"
if [[ $mode == full ]]; then default_shards=$FLEXC_SHARDS_FULL; else default_shards=$FLEXC_SHARDS_PILOT; fi
shards=${FLEXC_SHARDS:-$default_shards}
echo "site=$FLEXC_SITE env=${FLEXC_ENV_ACTIVE:-none} device=$FLEXC_DEVICE atlas=$DYNAMODE_ATLAS_ARCHIVE mdcath=$DYNAMODE_MDCATH_ARCHIVE"

PYTHONPATH="$repo/src:$repo/DynaMode-dev/src" python -m flexcompress.run configure \
  --config "$config" --run-dir "$run" --mode "$mode" --shards "$shards" "$@"
# Freeze the site choices so workers, later rounds, resumes and re-aggregations
# match this submission.  FLEXC_* variables set in the environment still win.
# shellcheck disable=SC2016  # the \${VAR:-frozen} text is meant literally
{
  printf 'FLEXC_SITE=${FLEXC_SITE:-%q}\n' "$FLEXC_SITE"
  printf 'FLEXC_ENV=${FLEXC_ENV:-%q}\n' "${FLEXC_ENV_ACTIVE:-${FLEXC_ENV:-}}"
  printf 'FLEXC_JOBS=${FLEXC_JOBS:-%q}\nFLEXC_THREADS=${FLEXC_THREADS:-%q}\n' "$FLEXC_JOBS" "$FLEXC_THREADS"
  printf 'FLEXC_DEVICE=${FLEXC_DEVICE:-%q}\nFLEXC_MPS=${FLEXC_MPS:-%q}\n' "$FLEXC_DEVICE" "$FLEXC_MPS"
  printf 'FLEXC_MEM_BUDGET_GB=${FLEXC_MEM_BUDGET_GB:-%q}\n' "$FLEXC_MEM_BUDGET_GB"
  printf 'FLEXC_GPU_MEM_BUDGET_GB=${FLEXC_GPU_MEM_BUDGET_GB:-%q}\n' "$FLEXC_GPU_MEM_BUDGET_GB"
  printf 'FLEXC_MAX_ROUNDS=${FLEXC_MAX_ROUNDS:-%q}\n' "$FLEXC_MAX_ROUNDS"
  printf 'FLEXC_ROOT=${FLEXC_ROOT:-%q}\nFLEXC_CONDA=${FLEXC_CONDA:-%q}\n' "$FLEXC_ROOT" "${FLEXC_CONDA:-}"
} > "$run/site.env"

# shellcheck disable=SC1091
source "$run/launch.sh"
FLEXC_SERIES="submit-$(date +%Y%m%d-%H%M%S)"
export FLEXC_SERIES
PYTHONPATH="$run/code/src" python -m flexcompress.run status --run-dir "$run" --round 0 >/dev/null
flexc_launch "$run" 1
echo "progress: PYTHONPATH=$run/code/src python -m flexcompress.run status --run-dir $run"
