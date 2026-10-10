#!/bin/bash
# Submit the whole training split of every dataset: one frozen run per config
# (default: ATLAS and mdCATH), each continuing round by round until all its
# targets are done.  Validation and test splits stay untouched.
#
#   bash sbatch/submit_train.sh
#   bash sbatch/submit_train.sh --set spectral.codecs=[subspace_local]     # passed to every config
#   FLEXC_TRAIN_CONFIGS="configs/stage01_atlas.yaml configs/stage01_atlas_1ns.yaml" bash sbatch/submit_train.sh
#
# Same environment overrides as submit.sh (FLEXC_SHARDS, FLEXC_JOBS, ...).
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
read -r -a configs <<< "${FLEXC_TRAIN_CONFIGS:-configs/stage01_atlas.yaml configs/stage01_mdcath.yaml}"
for config in "${configs[@]}"; do
  echo "== $config: full training split"
  bash "$repo/sbatch/submit.sh" "$config" full "$@"
done
