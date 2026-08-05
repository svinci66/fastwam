#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_PREFIX="${RUN_PREFIX:-robotwin_hanging_mug_qood_actual_pair_collection_20260805}"
SEEDS_CSV="${SEEDS:-4800001,4800002,4800003,4800004,4800005}"

IFS=',' read -r -a seeds <<< "${SEEDS_CSV}"
for seed in "${seeds[@]}"; do
  manifest="${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_hanging_mug_seed${seed}_20260805.json"
  [[ -f "${manifest}" ]] || {
    printf 'Missing single-seed manifest: %s\n' "${manifest}" >&2
    exit 1
  }
  printf '[segmented-qood-collection] seed=%s\n' "${seed}"
  env \
    "RUN_NAME=${RUN_PREFIX}_seed${seed}" \
    "EPISODES=1" \
    "VARIANTS=baseline,imagination" \
    "SEED_MANIFEST_PATH=${manifest}" \
    "SAVE_BASELINE_TRANSITIONS=true" \
    "SAVE_RESIDUAL_TRANSITIONS=true" \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_hanging_mug_qood_unlimited_pair.sh"
done

printf '[segmented-qood-collection] all requested seeds complete\n'
