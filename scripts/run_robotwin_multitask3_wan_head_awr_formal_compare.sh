#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
TASKS="${TASKS:-open_microwave,hanging_mug,place_can_basket}"
EPISODES="${EPISODES:-5}"
RUN_NAME="${RUN_NAME:-robotwin_wan_head_multitask3_awr_formal_block1_5ep_20260901}"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_4task_heldout5_expert_seeds_20260804.json}"
TREATMENT_DIR="${TREATMENT_DIR:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_wan_head_multitask3_awr_seed42_20260901/training/formal}"
CONTROL_DIR="${CONTROL_DIR:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_wan_head_multitask3_no_imagination_awr_seed42_20260901/training/formal}"
AUDIT_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}"
DRIVER_LOG="${AUDIT_ROOT}/driver.log"

mkdir -p "${AUDIT_ROOT}"
exec > >(tee -a "${DRIVER_LOG}") 2>&1

[[ -s "${TREATMENT_DIR}/checkpoint.pt" ]] || {
  printf '[formal-compare] missing treatment checkpoint: %s\n' "${TREATMENT_DIR}/checkpoint.pt" >&2
  exit 1
}
[[ -s "${SEED_MANIFEST_PATH}" ]] || {
  printf '[formal-compare] missing seed manifest: %s\n' "${SEED_MANIFEST_PATH}" >&2
  exit 1
}

if [[ ! -s "${CONTROL_DIR}/checkpoint.pt" ]]; then
  printf '[formal-compare] stage=train_matched_control\n'
  bash "${PROJECT_ROOT}/scripts/train_robotwin_multitask3_no_imagination_control.sh"
fi

printf '[formal-compare] stage=audit_checkpoint_ablation\n'
conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_awr_checkpoint_ablation.py" \
  --control-dir "${CONTROL_DIR}" --treatment-dir "${TREATMENT_DIR}" \
  --output-json "${AUDIT_ROOT}/checkpoint_ablation_audit.json"

printf '[formal-compare] stage=online variants=baseline,no_imagination,imagination tasks=%s episodes=%s\n' \
  "${TASKS}" "${EPISODES}"
env \
  RUN_NAME="${RUN_NAME}" \
  VARIANTS=baseline,no_imagination,imagination \
  TASKS="${TASKS}" EPISODES="${EPISODES}" \
  BASE_SEED=47 TRIAL_OFFSET=0 \
  INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
  TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
  PAPER_ALIGNED=true STRICT_PAIRED=true \
  DETERMINISTIC_INSTRUCTION_BY_SEED=true EXPERT_CHECK=true \
  SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH}" \
  NO_IMAGINATION_CHECKPOINT="${CONTROL_DIR}/checkpoint.pt" \
  IMAGINATION_CHECKPOINT="${TREATMENT_DIR}/checkpoint.pt" \
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

touch "${AUDIT_ROOT}/COMPLETE"
printf '[formal-compare] complete summary=%s\n' "${AUDIT_ROOT}/summary.json"
