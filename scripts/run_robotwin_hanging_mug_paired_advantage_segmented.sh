#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_4task_paired_v2_iql_20260803}"
RUN_PREFIX="${RUN_PREFIX:-robotwin_hanging_mug_paired_advantage_actor_aligned_v4_segmented_20260805}"
PAIRED_CHECKPOINT="${PAIRED_CHECKPOINT:-${RUN_ROOT}/iql_20epoch_imagination_paired_gate_actor_aligned_v4/checkpoint.pt}"
SEEDS_CSV="${SEEDS:-4800001,4800002,4800003,4800004,4800005}"

IFS=',' read -r -a seeds <<< "${SEEDS_CSV}"
for seed in "${seeds[@]}"; do
  manifest="${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_hanging_mug_seed${seed}_20260805.json"
  [[ -f "${manifest}" ]] || {
    printf 'Missing single-seed manifest: %s\n' "${manifest}" >&2
    exit 1
  }
  printf '[segmented-paired-eval] seed=%s\n' "${seed}"
  env \
    "RUN_ROOT=${RUN_ROOT}" \
    "RUN_NAME=${RUN_PREFIX}_seed${seed}" \
    "PAIRED_CHECKPOINT=${PAIRED_CHECKPOINT}" \
    "EPISODES=1" \
    "VARIANTS=imagination" \
    "SEED_MANIFEST_PATH=${manifest}" \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_hanging_mug_paired_advantage_pair.sh"
done

printf '[segmented-paired-eval] all requested seeds complete\n'
