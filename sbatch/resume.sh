#!/bin/bash
# Continue a run from a login node once it has stopped (round limit, a round
# with no progress, a fixed problem in the data): a new series of rounds over
# the targets that are not done yet, this time also retrying targets that
# failed.  While jobs of the run are still queued or running it refuses, since
# the run continues on its own; FLEXC_FORCE=1 starts a second series anyway
# (safe: workers share the run's queue and never take a held target).
#
#   bash sbatch/resume.sh /path/to/run
#   FLEXC_SHARDS=32 bash sbatch/resume.sh /path/to/run      # more workers this time
#   FLEXC_JOBS=16 bash sbatch/resume.sh /path/to/run        # fewer proteins per GH200 (after OOM kills)
#
# Uses the run's frozen code, scripts and site choices (run/site.env); FLEXC_*
# variables set now override them, as in submit.sh.
set -euo pipefail
run=$(cd "${1:?usage: resume.sh RUN_DIR}" && pwd)
for f in targets.json worker.sbatch aggregate.sbatch launch.sh; do
  [[ -f "$run/$f" ]] || { echo "not a configured run (no $f): $run" >&2; exit 2; }
done
# shellcheck disable=SC1091
[[ -f "$run/site.env" ]] && source "$run/site.env"  # frozen choices; FLEXC_* set now win
# shellcheck disable=SC1091
source "$run/flexc_env.sh"
# shellcheck disable=SC1091
source "$run/launch.sh"
name=$(basename "$run")
if [[ "${FLEXC_FORCE:-0}" != 1 ]] && command -v squeue >/dev/null 2>&1; then
  active=$(squeue -h -u "${USER:-$(id -un)}" -o %j 2>/dev/null | grep -c -F "flexc-$name-" || true)
  if (( ${active:-0} > 0 )); then
    echo "flexc: $active jobs of $name are still queued or running, and the run continues on its own." >&2
    echo "       Progress: PYTHONPATH=$run/code/src python -m flexcompress.run status --run-dir $run" >&2
    echo "       FLEXC_FORCE=1 starts a second series of rounds alongside it." >&2
    exit 2
  fi
fi
PYTHONPATH="$run/code/src" python -m flexcompress.run status --run-dir "$run" | sed -n 1p
FLEXC_SERIES="resume-$(date +%Y%m%d-%H%M%S)"
export FLEXC_SERIES
FLEXC_RETRY_SINCE=$(date +%s)
export FLEXC_RETRY_SINCE
PYTHONPATH="$run/code/src" python -m flexcompress.run status --run-dir "$run" --round 0 >/dev/null
flexc_launch "$run" 1
