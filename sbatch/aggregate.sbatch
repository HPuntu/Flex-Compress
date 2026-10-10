#!/bin/bash
#SBATCH --job-name=flexc-agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
# Merges all shard results into report/ (tables, summary.json, figures).
# Safe to re-run at any time; it only reads results/ and descriptors/.
set -euo pipefail
run=${1:?usage: aggregate.sbatch RUN_DIR}
# shellcheck disable=SC1091
source "$run/flexc_env.sh"
export PYTHONPATH="$run/code/src"
exec python -u -m flexcompress.run aggregate --run-dir "$run"
