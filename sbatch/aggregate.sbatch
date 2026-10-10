#!/bin/bash
#SBATCH --job-name=flexc-agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Finalize a round (submitted by launch.sh after the round's worker array):
#   1. void the ended array's leftover claims and record progress (rounds.json);
#   2. submit the next round if targets remain: at most FLEXC_MAX_ROUNDS, and
#      only if this round finished something new;
#   3. aggregate everything finished so far into report/ (tables, summary.json
#      with a coverage block, figures).
# By hand, without a round, it only aggregates:  bash RUN/aggregate.sbatch RUN
set -euo pipefail
run=${1:?usage: aggregate.sbatch RUN_DIR [ROUND WORKER_ARRAY_ID]}
round=${2:-}
array=${3:-}
# shellcheck disable=SC1091
[[ -f "$run/site.env" ]] && source "$run/site.env"
# shellcheck disable=SC1091
source "$run/flexc_env.sh"
export PYTHONPATH="$run/code/src"

if [[ -n "$round" ]]; then
  status_args=(--run-dir "$run" --round "$round" --max-rounds "${FLEXC_MAX_ROUNDS:-4}")
  [[ -n "$array" ]] && status_args+=(--dead-job "$array")
  out=$(python -u -m flexcompress.run status "${status_args[@]}") || out="action=error"
  printf '%s\n' "$out"
  action=$(tail -n 1 <<< "$out")
  echo "flexc: series ${FLEXC_SERIES:-default} round $round -> $action"
  case "$action" in
    action=resubmit)
      unset FLEXC_RETRY_SINCE  # later rounds leave new failures alone; resume.sh retries them
      # shellcheck disable=SC1091
      source "$run/launch.sh"
      flexc_launch "$run" $((round + 1)) \
        || echo "flexc: could not submit round $((round + 1)) from this job; on a login node run: bash sbatch/resume.sh $run"
      ;;
    action=finished)
      echo "flexc: every target of the split is done or failed (failures: errors/, retry with sbatch/resume.sh)" ;;
    action=waiting)
      echo "flexc: the remaining targets are held by another live array; its finalize job continues" ;;
    *)
      echo "flexc: not continuing ($action); check logs/ and errors/, then: bash sbatch/resume.sh $run" ;;
  esac
fi
exec python -u -m flexcompress.run aggregate --run-dir "$run"
