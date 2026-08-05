#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_corrected_posttrain_20260805}"
RUN_NAME="${RUN_NAME:-robotwin_4task_corrected_qood_margin003_paired5_20260805}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/iql_corrected_imagination_20epoch_paired_gate/checkpoint.pt}"
NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_CHECKPOINT:-${RUN_ROOT}/iql_corrected_imagination_20epoch/checkpoint.pt}"
SUPPORT_INDEX="${SUPPORT_INDEX:-${RUN_ROOT}/support_index_corrected_q95}"
SEED_MANIFEST="${SEED_MANIFEST:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_4task_heldout5_expert_seeds_20260804.json}"
TRIAL_OFFSET="${TRIAL_OFFSET:-0}"

for path in "${CHECKPOINT}" "${NO_IMAGINATION_CHECKPOINT}" "${SUPPORT_INDEX}/metadata.json" "${SEED_MANIFEST}"; do
  [[ -e "${path}" ]] || {
    printf 'Missing required artifact: %s\n' "${path}" >&2
    exit 1
  }
done

env \
  RUN_NAME="${RUN_NAME}" \
  VARIANTS="${VARIANTS:-baseline,imagination}" \
  TASKS="${TASKS:-adjust_bottle,hanging_mug,open_microwave,place_can_basket}" \
  EPISODES="${EPISODES:-5}" \
  TRIAL_OFFSET="${TRIAL_OFFSET}" \
  IMAGINATION_CHECKPOINT="${CHECKPOINT}" \
  NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_CHECKPOINT}" \
  RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-local-20260803 \
  RESIDUAL_Q_GATE_ENABLED=true \
  RESIDUAL_Q_GATE_MARGIN=0.003 \
  RESIDUAL_Q_GATE_MAX_DISAGREEMENT=0.05 \
  RESIDUAL_Q_GATE_CRITIC_SOURCE=target \
  RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
  RESIDUAL_SUPPORT_INDEX_PATH="${SUPPORT_INDEX}" \
  RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=true \
  RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=true \
  RESIDUAL_OUTCOME_CONFIRMATION_MIN_PROGRESS=0.0 \
  RESIDUAL_OUTCOME_CONFIRMATION_REANCHOR_REPLANS=1 \
  RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
  RESIDUAL_LANGUAGE_MODE=training_canonical \
  SAVE_RESIDUAL_TRANSITIONS=true \
  SAVE_BASELINE_TRANSITIONS=false \
  PAPER_ALIGNED=true \
  STRICT_PAIRED=true \
  SEED_MANIFEST_PATH="${SEED_MANIFEST}" \
  INSTRUCTION_MODE=official \
  INSTRUCTION_TYPE=unseen \
  INFERENCE_STEPS=10 \
  REPLAN_STEPS=24 \
  TEXT_CFG_SCALE=1.0 \
  EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-true}" \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
