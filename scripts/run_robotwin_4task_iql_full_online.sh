#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_4task_paired_v2_iql_20260803}"
TASKS="${TASKS:-adjust_bottle,hanging_mug,open_microwave,place_can_basket}"
EPISODES="${EPISODES:-5}"
BASE_SEED="${BASE_SEED:-47}"
ENVIRONMENT_START_SEED="${ENVIRONMENT_START_SEED:-4800000}"
TRIAL_OFFSET="${TRIAL_OFFSET:-0}"
RUN_PREFIX="${RUN_PREFIX:-robotwin_4task_iql_full_heldout5_20260803}"

COMMON_ENV=(
  "TASKS=${TASKS}"
  "EPISODES=${EPISODES}"
  "BASE_SEED=${BASE_SEED}"
  "ENVIRONMENT_START_SEED=${ENVIRONMENT_START_SEED}"
  "TRIAL_OFFSET=${TRIAL_OFFSET}"
  "IMAGINATION_CHECKPOINT=${RUN_ROOT}/iql_5epoch_imagination/checkpoint.pt"
  "NO_IMAGINATION_CHECKPOINT=${RUN_ROOT}/iql_5epoch_no_imagination/checkpoint.pt"
  "RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-local-20260803"
  "SAVE_RESIDUAL_TRANSITIONS=false"
  "SAVE_BASELINE_TRANSITIONS=false"
)

env \
  "${COMMON_ENV[@]}" \
  "RUN_NAME=${RUN_PREFIX}_ungated" \
  "VARIANTS=baseline,imagination" \
  "RESIDUAL_Q_GATE_ENABLED=false" \
  "RESIDUAL_SUPPORT_INDEX_PATH=none" \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"

env \
  "${COMMON_ENV[@]}" \
  "RUN_NAME=${RUN_PREFIX}_qood" \
  "VARIANTS=imagination" \
  "RESIDUAL_Q_GATE_ENABLED=true" \
  "RESIDUAL_Q_GATE_MARGIN=0.0" \
  "RESIDUAL_Q_GATE_MAX_DISAGREEMENT=0.05" \
  "RESIDUAL_Q_GATE_CRITIC_SOURCE=target" \
  "RESIDUAL_SUPPORT_INDEX_PATH=${RUN_ROOT}/support_index_imagination_q95_local" \
  "RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=true" \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"

printf 'Full four-task online evaluation complete: %s\n' "${RUN_PREFIX}"
