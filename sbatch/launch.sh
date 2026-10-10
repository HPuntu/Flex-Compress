#!/bin/bash
# shellcheck disable=SC2154  # flexc_sbatch_args, flexc_agg_args, FLEXC_*: from the site profile
# Submit one round of a run (sourced after flexc_env.sh by submit.sh, resume.sh
# and the finalize job; configure copies it into each run directory):
#   - a worker array of FLEXC_SHARDS tasks (default: the run's n_shards), all
#     pulling targets from the run's shared queue, largest first;
#   - a finalize job (afterany) that voids the array's leftover claims, records
#     progress, submits the next round while targets remain (at most
#     FLEXC_MAX_ROUNDS, and only if the round finished something), and
#     aggregates everything done so far into report/.

# sbatch without the calling job's SLURM_* variables, so a round submitted from
# inside a finalize job does not inherit that job's allocation.
flexc_sbatch() {
  local -a drop=()
  local name
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    while IFS= read -r name; do
      case $name in
        SLURM_CONF) ;;
        SLURM_* | CUDA_MPS_*) drop+=(-u "$name") ;;
      esac
    done < <(compgen -e)
  fi
  env ${drop[@]+"${drop[@]}"} sbatch "$@"
}

# flexc_launch RUN_DIR ROUND
flexc_launch() {
  local run=$1 round=${2:-1}
  local n name work fin
  n=${FLEXC_SHARDS:-$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_shards'])" "$run/targets.json")}
  name=$(basename "$run")
  work=$(flexc_sbatch --parsable "${flexc_sbatch_args[@]}" --array="0-$((n - 1))%${FLEXC_CONCURRENCY:-$n}" \
    --job-name="flexc-$name-r$round" --output="$run/logs/worker-r$round-%A_%a.out" \
    --error="$run/logs/worker-r$round-%A_%a.err" "$run/worker.sbatch" "$run") || return 1
  work=${work%%;*}
  fin=$(flexc_sbatch --parsable "${flexc_agg_args[@]}" --dependency="afterany:$work" \
    --job-name="flexc-$name-fin$round" --output="$run/logs/finalize-r$round-%j.out" \
    --error="$run/logs/finalize-r$round-%j.err" "$run/aggregate.sbatch" "$run" "$round" "$work") \
    || { scancel "$work" 2>/dev/null; echo "flexc: finalize job not accepted; cancelled array $work" >&2; return 1; }
  fin=${fin%%;*}
  printf '%s series=%s round=%s/%s site=%s workers=%s (%s tasks x %s jobs) finalize=%s run=%s\n' "$(date -Is)" \
    "${FLEXC_SERIES:-default}" "$round" "${FLEXC_MAX_ROUNDS:-4}" "$FLEXC_SITE" "$work" "$n" "$FLEXC_JOBS" "$fin" "$run" \
    | tee -a "$run/submission.txt"
}
