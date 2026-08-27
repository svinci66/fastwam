#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ROOT="${TRAIN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_open_microwave_wan_head_awr_smoke_seed42_20260827/training}"
NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_CHECKPOINT:-${TRAIN_ROOT}/seed42/no_imagination/checkpoint.pt}"
IMAGINATION_CHECKPOINT="${IMAGINATION_CHECKPOINT:-${TRAIN_ROOT}/seed42/with_imagination/checkpoint.pt}"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_4task_heldout5_expert_seeds_20260804.json}"
RUN_NAME="${RUN_NAME:-robotwin_open_microwave_wan_head_awr_online_pair_5ep_20260827}"
EPISODES="${EPISODES:-5}"

for checkpoint in "${NO_IMAGINATION_CHECKPOINT}" "${IMAGINATION_CHECKPOINT}"; do
  [[ -s "${checkpoint}" ]] || {
    printf '[wan-head-online] missing checkpoint: %s\n' "${checkpoint}" >&2
    exit 1
  }
done
[[ -s "${TRAIN_ROOT}/paired_training_audit.json" ]] || {
  printf '[wan-head-online] missing paired training audit\n' >&2
  exit 1
}

# The underlying runner also supports historical Q/OOD experiments.  Pin every
# such switch off here so this comparison measures only the AWR residual actor.
env \
  RUN_NAME="${RUN_NAME}" \
  VARIANTS=baseline,no_imagination,imagination \
  TASKS=open_microwave \
  EPISODES="${EPISODES}" \
  BASE_SEED=47 TRIAL_OFFSET=0 \
  INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
  TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
  PAPER_ALIGNED=true STRICT_PAIRED=true \
  DETERMINISTIC_INSTRUCTION_BY_SEED=true EXPERT_CHECK=true \
  SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH}" \
  NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_CHECKPOINT}" \
  IMAGINATION_CHECKPOINT="${IMAGINATION_CHECKPOINT}" \
  RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-modelscope-local-v1 \
  RESIDUAL_LANGUAGE_MODE=policy_instruction \
  RESIDUAL_Q_GATE_ENABLED=false \
  RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
  RESIDUAL_SUPPORT_INDEX_PATH=none \
  RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
  RESIDUAL_SHADOW_MODE=false \
  RESIDUAL_INTERVENTION_REPLANS=all \
  RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
  RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=false \
  RESIDUAL_SOFT_SCALE_ENABLED=false \
  SAVE_BASELINE_TRANSITIONS=false SAVE_RESIDUAL_TRANSITIONS=false \
  EVAL_VIDEO_LOG=true \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"

printf '[wan-head-online] complete run=%s\n' "${RUN_NAME}"
