#!/bin/bash
# Configure a frozen run and submit its shard array plus the aggregation job.
#
#   bash sbatch/submit.sh configs/stage01_atlas.yaml pilot
#   bash sbatch/submit.sh configs/stage01_mdcath.yaml full --set folds.max_folds=2
#
# Run from the Flex-Compress checkout on the cluster login node.  Requires the
# archive variable named in the config (DYNAMODE_ATLAS_ARCHIVE /
# DYNAMODE_MDCATH_ARCHIVE).  Tunables (environment):
#   FLEXC_RUNS_ROOT    parent of run directories (default /data/stat-opig/$USER/DATA/flex-compress-runs)
#   FLEXC_SHARDS       array size (default 8 pilot / 64 full)
#   FLEXC_CONCURRENCY  max simultaneous array tasks (default 16)
#   FLEXC_PARTITION    Slurm partition (default long)
#   FLEXC_CPUS / FLEXC_MEM / FLEXC_TIME   per-task resources (4 / 24G / 24:00:00)
#   FLEXC_ENV          conda env path or name (default $DYNAMODE_ENV, then dynamode-v100)
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
config=${1:?usage: submit.sh CONFIG.yaml pilot|full [--set key=value ...]}
mode=${2:?usage: submit.sh CONFIG.yaml pilot|full [--set key=value ...]}
shift 2
case "$mode" in pilot|full) ;; *) echo "mode must be pilot or full" >&2; exit 2 ;; esac
[[ -f "$repo/DynaMode-dev/src/dynamode/dataloader/compressed_backbone.py" ]] || {
  echo "DynaMode-dev submodule (dev branch) missing: git submodule update --init" >&2; exit 2; }

# shellcheck disable=SC1091
source "$repo/sbatch/flexc_env.sh"
name=$(basename "$config" .yaml)
root=${FLEXC_RUNS_ROOT:-/data/stat-opig/$USER/DATA/flex-compress-runs}
run="$root/$name-$mode-$(date +%Y%m%d-%H%M%S)"
default_shards=8
[[ $mode == full ]] && default_shards=64
shards=${FLEXC_SHARDS:-$default_shards}

PYTHONPATH="$repo/src:$repo/DynaMode-dev/src" python -m flexcompress.run configure \
  --config "$config" --run-dir "$run" --mode "$mode" --shards "$shards" "$@"
n=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_shards'])" "$run/targets.json")

common=(--partition="${FLEXC_PARTITION:-long}")
work=$(sbatch --parsable "${common[@]}" --array="0-$((n - 1))%${FLEXC_CONCURRENCY:-16}" \
  --cpus-per-task="${FLEXC_CPUS:-4}" --mem="${FLEXC_MEM:-24G}" --time="${FLEXC_TIME:-24:00:00}" \
  --job-name="flexc-$name" --output="$run/logs/worker-%A_%a.out" --error="$run/logs/worker-%A_%a.err" \
  "$run/worker.sbatch" "$run")
work=${work%%;*}
# afterany: aggregate even if some shards hit their time limit; failures are
# listed in report/summary.json and the worker can be re-submitted to resume.
agg=$(sbatch --parsable "${common[@]}" --dependency="afterany:$work" --job-name="flexc-$name-agg" \
  --output="$run/logs/aggregate-%j.out" --error="$run/logs/aggregate-%j.err" \
  "$run/aggregate.sbatch" "$run")
agg=${agg%%;*}
printf 'run=%s\nworker_array=%s (%s shards)\naggregate=%s\n' "$run" "$work" "$n" "$agg" | tee "$run/submission.txt"
