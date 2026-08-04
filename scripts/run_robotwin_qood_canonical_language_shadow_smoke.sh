#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_4task_paired_v2_iql_20260803}"
RUN_NAME="${RUN_NAME:-robotwin_4task_qood_canonical_language_shadow1_20260804}"
IMAGINATION_CHECKPOINT="${IMAGINATION_CHECKPOINT:-${RUN_ROOT}/iql_5epoch_imagination/checkpoint.pt}"
NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_CHECKPOINT:-${RUN_ROOT}/iql_5epoch_no_imagination/checkpoint.pt}"
RESIDUAL_SUPPORT_INDEX_PATH="${RESIDUAL_SUPPORT_INDEX_PATH:-${RUN_ROOT}/support_index_imagination_q95_local}"

env \
  "TASKS=adjust_bottle,hanging_mug,open_microwave,place_can_basket" \
  "EPISODES=1" \
  "VARIANTS=imagination" \
  "BASE_SEED=47" \
  "ENVIRONMENT_START_SEED=4800000" \
  "TRIAL_OFFSET=0" \
  "RUN_NAME=${RUN_NAME}" \
  "INFERENCE_STEPS=10" \
  "REPLAN_STEPS=24" \
  "TEXT_CFG_SCALE=1.0" \
  "TASK_CONFIG=demo_clean" \
  "INSTRUCTION_TYPE=unseen" \
  "INSTRUCTION_MODE=official" \
  "PAPER_ALIGNED=true" \
  "STRICT_PAIRED=true" \
  "SEED_MANIFEST_PATH=${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_4task_heldout1_expert_seeds_20260804.json" \
  "IMAGINATION_CHECKPOINT=${IMAGINATION_CHECKPOINT}" \
  "NO_IMAGINATION_CHECKPOINT=${NO_IMAGINATION_CHECKPOINT}" \
  "RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-local-20260803" \
  "RESIDUAL_LANGUAGE_MODE=training_canonical" \
  "RESIDUAL_Q_GATE_ENABLED=true" \
  "RESIDUAL_Q_GATE_MARGIN=0.0" \
  "RESIDUAL_Q_GATE_MAX_DISAGREEMENT=0.05" \
  "RESIDUAL_Q_GATE_CRITIC_SOURCE=target" \
  "RESIDUAL_SUPPORT_INDEX_PATH=${RESIDUAL_SUPPORT_INDEX_PATH}" \
  "RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=true" \
  "RESIDUAL_SHADOW_MODE=true" \
  "SAVE_RESIDUAL_TRANSITIONS=false" \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
