#!/bin/bash
# Resubmit an existing run: workers skip finished targets (same config hash),
# so this resumes after time-outs or node failures, then re-aggregates.
#
#   bash sbatch/resume.sh /path/to/run              # every shard
#   bash sbatch/resume.sh /path/to/run 3,17,42      # only these shards
#
# Uses the run's frozen code snapshot and scripts; FLEXC_* resources as in submit.sh.
set -euo pipefail
run=$(cd "${1:?usage: resume.sh RUN_DIR [SHARD_LIST]}" && pwd)
[[ -f "$run/targets.json" && -f "$run/worker.sbatch" ]] || { echo "not a configured run: $run" >&2; exit 2; }
n=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['n_shards'])" "$run/targets.json")
array=${2:-0-$((n - 1))}
common=(--partition="${FLEXC_PARTITION:-long}")
work=$(sbatch --parsable "${common[@]}" --array="${array}%${FLEXC_CONCURRENCY:-16}" \
  --cpus-per-task="${FLEXC_CPUS:-4}" --mem="${FLEXC_MEM:-24G}" --time="${FLEXC_TIME:-24:00:00}" \
  --job-name="flexc-resume" --output="$run/logs/worker-%A_%a.out" --error="$run/logs/worker-%A_%a.err" \
  "$run/worker.sbatch" "$run")
work=${work%%;*}
agg=$(sbatch --parsable "${common[@]}" --dependency="afterany:$work" --job-name="flexc-resume-agg" \
  --output="$run/logs/aggregate-%j.out" --error="$run/logs/aggregate-%j.err" \
  "$run/aggregate.sbatch" "$run")
agg=${agg%%;*}
printf '%s resume worker_array=%s (%s) aggregate=%s\n' "$(date -Is)" "$work" "$array" "${agg}" | tee -a "$run/submission.txt"
