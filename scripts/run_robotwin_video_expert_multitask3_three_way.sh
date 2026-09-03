#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${SEED:-42}"
TASKS="${TASKS:-open_microwave,hanging_mug,place_can_basket}"
EPISODES="${EPISODES:-5}"
FEATURE_VERSION="fastwam_video_expert_final_token_mean_l2_v1"

REPLAY_DIR="${REPLAY_DIR:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_weight025_epochs3_seed42_20260903/replay}"
NO_IMAGINATION_CONFIG="${NO_IMAGINATION_CONFIG:-${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_video_expert_multitask3_no_imagination_smoke.yaml}"
NO_IMAGINATION_ROOT="${NO_IMAGINATION_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_no_imagination_epochs3_seed${SEED}_20260903}"
NO_IMAGINATION_TRAIN_DIR="${NO_IMAGINATION_TRAIN_DIR:-${NO_IMAGINATION_ROOT}/training/seed${SEED}/no_imagination}"
IMAGINATION_TRAIN_DIR="${IMAGINATION_TRAIN_DIR:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_weight025_epochs3_seed42_20260903/training/seed42/with_imagination}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-robotwin_wan_head_multitask3_awr_formal_block1_5ep_20260901}"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${BASELINE_RUN_NAME}/prevalidated_seed_manifest.json}"
ONLINE_RUN_NAME="${ONLINE_RUN_NAME:-robotwin_video_expert_multitask3_three_way_epoch003_5ep_20260903}"

for path in "${REPLAY_DIR}/manifest.json" "${NO_IMAGINATION_CONFIG}" \
  "${IMAGINATION_TRAIN_DIR}/checkpoint.pt" "${SEED_MANIFEST_PATH}"; do
  [[ -s "${path}" ]] || { printf '[video-expert-three-way] missing: %s\n' "${path}" >&2; exit 1; }
done

# Train the only missing arm. The shared entry point skips it on safe resume when
# both checkpoint.pt and history.json already exist.
PHASE=train \
CONFIG="${NO_IMAGINATION_CONFIG}" \
RUN_ROOT="${NO_IMAGINATION_ROOT}" \
REPLAY_DIR="${REPLAY_DIR}" \
TRAIN_DIR="${NO_IMAGINATION_TRAIN_DIR}" \
bash "${PROJECT_ROOT}/scripts/run_robotwin_video_expert_multitask3_weight025.sh"

env \
  RUN_NAME="${ONLINE_RUN_NAME}" \
  VARIANTS=baseline,no_imagination,imagination \
  TASKS="${TASKS}" EPISODES="${EPISODES}" BASE_SEED=47 TRIAL_OFFSET=0 \
  INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
  TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
  PAPER_ALIGNED=true STRICT_PAIRED=true \
  DETERMINISTIC_INSTRUCTION_BY_SEED=true EXPERT_CHECK=true \
  SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH}" \
  NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_TRAIN_DIR}/checkpoint.pt" \
  IMAGINATION_CHECKPOINT="${IMAGINATION_TRAIN_DIR}/checkpoint.pt" \
  RESIDUAL_ENCODER_PATH=none \
  RESIDUAL_ENCODER_VERSION="${FEATURE_VERSION}" \
  RESIDUAL_LANGUAGE_MODE=policy_instruction \
  RESIDUAL_Q_GATE_ENABLED=false RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
  RESIDUAL_SUPPORT_INDEX_PATH=none RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
  RESIDUAL_SHADOW_MODE=false RESIDUAL_INTERVENTION_REPLANS=all \
  RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
  RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=false RESIDUAL_SOFT_SCALE_ENABLED=false \
  SAVE_BASELINE_TRANSITIONS=false SAVE_RESIDUAL_TRANSITIONS=false \
  EVAL_VIDEO_LOG=true \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"

touch "${NO_IMAGINATION_ROOT}/THREE_WAY_COMPLETE"
printf '[video-expert-three-way] complete summary=%s\n' \
  "${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${ONLINE_RUN_NAME}/summary.json"
