#!/bin/bash
#SBATCH --job-name=flexc-bench
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=flexc-bench-%j.out
# Time one synthetic ATLAS-sized target (N=300, 3 x 10,000 frames) on the CPU
# (numpy) and on the H100, and check they agree; or (--throughput) measure
# proteins/hour of the real worker at several concurrencies.  Isambard-AI:
#   sbatch sbatch/bench.sbatch                                   # from the repo root
#   sbatch sbatch/bench.sbatch --throughput --devices cuda --jobs-list 8,16,32,48
#   FLEXC_MPS=1 sbatch sbatch/bench.sbatch --throughput --devices cuda --jobs-list 16,32,48
# Extra arguments go to `python -m flexcompress.run bench` (later flags win).
set -euo pipefail
repo="${FLEXC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source "$repo/sbatch/flexc_env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
flexc_start_mps
export PYTHONPATH="$repo/src:$repo/DynaMode-dev/src"
python -u -m flexcompress.run bench --devices numpy,cuda --workdir "${TMPDIR:-/tmp}/flexc-bench-${SLURM_JOB_ID:-local}" "$@"
