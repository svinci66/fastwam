#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_4task_paired_v2_iql_20260803}"
RUN_NAME="${RUN_NAME:-robotwin_hanging_mug_qood_outcome_softscale_paired5_20260804}"
EPISODES="${EPISODES:-5}"
TRIAL_OFFSET="${TRIAL_OFFSET:-0}"
VARIANTS="${VARIANTS:-baseline,imagination}"
if [[ -z "${SEED_MANIFEST_PATH:-}" ]]; then
  case "${EPISODES}" in
    1)
      SEED_MANIFEST_PATH="${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_4task_heldout1_expert_seeds_20260804.json"
      ;;
    5)
      SEED_MANIFEST_PATH="${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_4task_heldout5_expert_seeds_20260804.json"
      ;;
    *)
      printf 'SEED_MANIFEST_PATH is required when EPISODES is not 1 or 5\n' >&2
      exit 1
      ;;
  esac
fi

# Paper-aligned, strictly paired test of closed-loop result confirmation.
# Initial observation hashes are logged for both variants. Future-video
# inference runs only after an actually applied residual, so the FastWAM action
# is fixed first and baseline evaluation remains efficient and unconfounded.
env \
  "TASKS=hanging_mug" \
  "EPISODES=${EPISODES}" \
  "BASE_SEED=47" \
  "ENVIRONMENT_START_SEED=4800000" \
  "TRIAL_OFFSET=${TRIAL_OFFSET}" \
  "RUN_NAME=${RUN_NAME}" \
  "VARIANTS=${VARIANTS}" \
  "INFERENCE_STEPS=10" \
  "REPLAN_STEPS=24" \
  "TEXT_CFG_SCALE=1.0" \
  "TASK_CONFIG=demo_clean" \
  "EVAL_VIDEO_LOG=false" \
  "INSTRUCTION_TYPE=unseen" \
  "INSTRUCTION_MODE=official" \
  "PAPER_ALIGNED=true" \
  "STRICT_PAIRED=true" \
  "SEED_MANIFEST_PATH=${SEED_MANIFEST_PATH}" \
  "IMAGINATION_CHECKPOINT=${RUN_ROOT}/iql_20epoch_imagination_canonical_probe/checkpoint.pt" \
  "NO_IMAGINATION_CHECKPOINT=${RUN_ROOT}/iql_5epoch_no_imagination/checkpoint.pt" \
  "RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-local-20260803" \
  "RESIDUAL_LANGUAGE_MODE=training_canonical" \
  "RESIDUAL_Q_GATE_ENABLED=true" \
  "RESIDUAL_Q_GATE_MARGIN=0.0" \
  "RESIDUAL_Q_GATE_MAX_DISAGREEMENT=0.05" \
  "RESIDUAL_Q_GATE_RISK_SCALE=0.0" \
  "RESIDUAL_Q_GATE_RISK_DECAY=1.0" \
  "RESIDUAL_SOFT_SCALE_ENABLED=true" \
  "RESIDUAL_SOFT_SCALE_Q_FULL_ADVANTAGE=0.005" \
  "RESIDUAL_SOFT_SCALE_SUPPORT_FULL_MARGIN=0.25" \
  "RESIDUAL_Q_GATE_CRITIC_SOURCE=target" \
  "RESIDUAL_SUPPORT_INDEX_PATH=${RUN_ROOT}/support_index_imagination_20epoch_q95_probe" \
  "RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=true" \
  "RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none" \
  "RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=true" \
  "RESIDUAL_OUTCOME_CONFIRMATION_MIN_PROGRESS=0.0" \
  "RESIDUAL_OUTCOME_CONFIRMATION_REANCHOR_REPLANS=1" \
  "SAVE_RESIDUAL_TRANSITIONS=false" \
  "SAVE_BASELINE_TRANSITIONS=false" \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
