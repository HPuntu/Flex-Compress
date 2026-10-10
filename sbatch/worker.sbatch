#!/bin/bash
#SBATCH --job-name=flexc-work
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One worker of a run: pulls targets from the run's shared queue (largest
# first, never one another live worker holds) until none is left or the
# walltime is near, using the run's frozen code snapshot.  Resources (one
# GH200 on Isambard-AI; CPUs/memory on HTC) and the array size come from the
# site profile via launch.sh.  FLEXC_RETRY_SINCE=<unix time> (set by resume.sh)
# also retries targets whose failure was recorded before that time.
set -euo pipefail
run=${1:?usage: worker.sbatch RUN_DIR}
# shellcheck disable=SC1091
[[ -f "$run/site.env" ]] && source "$run/site.env"
# shellcheck disable=SC1091
source "$run/flexc_env.sh"
FLEXC_DEVICE="${FLEXC_DEVICE:-auto}"
printf '%s job=%s shard=%s host=%s site=%s env=%s device=%s jobs=%s threads=%s run=%s\n' "$(date -Is)" \
  "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:-0}" "$(hostname)" "$FLEXC_SITE" "${FLEXC_ENV_ACTIVE:-none}" \
  "$FLEXC_DEVICE" "$FLEXC_JOBS" "$FLEXC_THREADS" "$run"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())" \
  2>/dev/null || echo "torch not importable: running the numpy backend"

flexc_start_mps  # FLEXC_MPS=1: concurrent kernels from all worker processes (see flexc_env.sh)

extra=()
[[ -n "${FLEXC_RETRY_SINCE:-}" ]] && extra+=(--retry-since "$FLEXC_RETRY_SINCE")
export PYTHONPATH="$run/code/src"
python -u -m flexcompress.run worker --run-dir "$run" --shard "${SLURM_ARRAY_TASK_ID:-0}" \
  --jobs "$FLEXC_JOBS" --threads "$FLEXC_THREADS" --device "$FLEXC_DEVICE" \
  --mem-budget-gb "${FLEXC_MEM_BUDGET_GB:-0}" --gpu-mem-budget-gb "${FLEXC_GPU_MEM_BUDGET_GB:-0}" \
  ${extra[@]+"${extra[@]}"}
