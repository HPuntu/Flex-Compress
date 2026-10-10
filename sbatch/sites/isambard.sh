#!/bin/bash
# shellcheck disable=SC2034  # variables are used by flexc_env.sh, submit.sh and launch.sh
# Site profile: Isambard-AI phase 2 (BriCS), project u6xq.  Login: ssh u6xq.aip2.isambard
# Sourced by sbatch/flexc_env.sh.  Every value can be overridden from the environment.
#
# Isambard-AI allocates whole GH200 superchips: `--nodes=1 --gpus=1` gives one
# Grace CPU (72 Arm cores, ~115 GB usable memory) plus its H100.  Each job runs
# FLEXC_JOBS targets at once (FLEXC_THREADS BLAS threads each); with a CUDA
# PyTorch env (dynamode-gh200) the heavy linear algebra of every process runs on
# the shared H100 (FLEXC_DEVICE=auto), and archive decoding stays on the CPU.
# Limits (docs.isambard.ac.uk): workq default partition, 24 h walltime,
# <= 1000 array tasks, 256 running jobs per user.

FLEXC_ROOT="${FLEXC_ROOT:-/projects/u6xq/hew}"
export DYNAMODE_ATLAS_ARCHIVE="${DYNAMODE_ATLAS_ARCHIVE:-$FLEXC_ROOT/DATA/ATLAS/compressed_bb_zarr}"
export DYNAMODE_MDCATH_ARCHIVE="${DYNAMODE_MDCATH_ARCHIVE:-$FLEXC_ROOT/DATA/mdCATH/compressed_bb_zarr}"
FLEXC_RUNS_ROOT="${FLEXC_RUNS_ROOT:-$FLEXC_ROOT/flex-compress-runs}"

# Project Miniforge install (no `conda init`), as recommended by BriCS.
FLEXC_CONDA="${FLEXC_CONDA:-$FLEXC_ROOT/miniforge3}"
# First environment that activates wins: explicit choice, then the project's
# GH200 DynaMode env (CUDA PyTorch), then the CPU-only env from setup_env.sh.
FLEXC_ENV_CANDIDATES=("${FLEXC_ENV:-}" "${DYNAMODE_ENV:-}" dynamode-gh200 flexc)

flexc_conda_init() {
  if [[ -f "$FLEXC_CONDA/bin/activate" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$FLEXC_CONDA/bin/activate"
    set -u
  fi
}

# Compute backend: auto = H100 when the env has CUDA PyTorch, else numpy on the CPU.
FLEXC_DEVICE="${FLEXC_DEVICE:-auto}"
# CUDA MPS lets the worker processes' kernels run concurrently on the H100
# instead of time-slicing; worth it with many processes.  Opt-in (1) until
# `FLEXC_MPS=1 sbatch sbatch/bench.sbatch --throughput` confirms it on Isambard.
FLEXC_MPS="${FLEXC_MPS:-0}"
# Proteins in flight per GH200.  With the heavy linear algebra on the H100 each
# process needs little CPU, so up to 32 run at once (2 BLAS threads each, 64 of
# 72 cores; the rest decode archives).  The worker admits a protein only when
# its estimated memory fits the budgets below (largest first, small ones
# backfill), so big ATLAS chains never all start together.
FLEXC_JOBS="${FLEXC_JOBS:-32}"
FLEXC_THREADS="${FLEXC_THREADS:-2}"
# One GH200 = ~115 GB usable host memory + 96 GB on the H100.  Leave headroom.
FLEXC_MEM_BUDGET_GB="${FLEXC_MEM_BUDGET_GB:-95}"
FLEXC_GPU_MEM_BUDGET_GB="${FLEXC_GPU_MEM_BUDGET_GB:-80}"

# Worker tasks (GH200s) per round.  They all pull from the run's shared queue,
# so this sets how fast a training split finishes, not whether it does: a run
# continues round by round (24 h each, at most FLEXC_MAX_ROUNDS) until every
# target is done.  16 + 16 GH200s run the ATLAS and mdCATH splits side by side.
FLEXC_SHARDS_PILOT="${FLEXC_SHARDS_PILOT:-1}"
FLEXC_SHARDS_FULL="${FLEXC_SHARDS_FULL:-16}"
FLEXC_MAX_ROUNDS="${FLEXC_MAX_ROUNDS:-4}"

flexc_sbatch_args=(--nodes=1 --gpus=1 --time="${FLEXC_TIME:-24:00:00}")
flexc_agg_args=(--nodes=1 --gpus=1 --time="${FLEXC_AGG_TIME:-04:00:00}")
if [[ -n "${FLEXC_PARTITION:-}" ]]; then
  flexc_sbatch_args+=(--partition="$FLEXC_PARTITION")
  flexc_agg_args+=(--partition="$FLEXC_PARTITION")
fi
