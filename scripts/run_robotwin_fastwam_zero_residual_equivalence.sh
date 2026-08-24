#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-robotwin_fastwam_zero_residual_equivalence_4task1ep_20260824}"
TASKS="${TASKS:-adjust_bottle,hanging_mug,open_microwave,place_can_basket}"
EPISODES="${EPISODES:-1}"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_4task_heldout1_expert_seeds_20260804.json}"
RESIDUAL_CHECKPOINT="${RESIDUAL_CHECKPOINT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_corrected_posttrain_20260805/iql_corrected_imagination_20epoch_paired_gate/checkpoint.pt}"
RESIDUAL_ENCODER_VERSION="${RESIDUAL_ENCODER_VERSION:-siglip-so400m-patch14-384-local-20260803}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
AUDIT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}"

[[ -f "${SEED_MANIFEST_PATH}" ]] || {
  printf 'Missing seed manifest: %s\n' "${SEED_MANIFEST_PATH}" >&2
  exit 1
}
[[ -f "${RESIDUAL_CHECKPOINT}" ]] || {
  printf 'Missing residual checkpoint: %s\n' "${RESIDUAL_CHECKPOINT}" >&2
  exit 1
}

env \
  RUN_NAME="${RUN_NAME}" \
  TASKS="${TASKS}" \
  EPISODES="${EPISODES}" \
  VARIANTS=baseline,imagination \
  BASE_SEED=47 \
  PAPER_ALIGNED=true \
  STRICT_PAIRED=true \
  INSTRUCTION_MODE=official \
  INSTRUCTION_TYPE=unseen \
  INFERENCE_STEPS=10 \
  REPLAN_STEPS=24 \
  TEXT_CFG_SCALE=1.0 \
  SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH}" \
  DETERMINISTIC_INSTRUCTION_BY_SEED=true \
  EXPERT_CHECK=true \
  EVAL_VIDEO_LOG=false \
  TILED=false \
  CAPTURE_DECODE_TILED=false \
  IMAGINATION_CHECKPOINT="${RESIDUAL_CHECKPOINT}" \
  NO_IMAGINATION_CHECKPOINT="${RESIDUAL_CHECKPOINT}" \
  RESIDUAL_ENCODER_VERSION="${RESIDUAL_ENCODER_VERSION}" \
  RESIDUAL_LANGUAGE_MODE=policy_instruction \
  RESIDUAL_Q_GATE_ENABLED=false \
  RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
  RESIDUAL_SUPPORT_INDEX_PATH=none \
  RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
  RESIDUAL_SHADOW_MODE=true \
  SAVE_BASELINE_TRANSITIONS=false \
  SAVE_RESIDUAL_TRANSITIONS=false \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"

conda run --no-capture-output -n robotwin_fastwam python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_zero_residual_equivalence.py" \
  --baseline-dir "${RESULT_BASE}/${RUN_NAME}_baseline" \
  --shadow-dir "${RESULT_BASE}/${RUN_NAME}_imagination" \
  --tasks "${TASKS}" \
  --output-json "${AUDIT_DIR}/zero_residual_equivalence_audit.json" \
  --require-exact

printf 'Zero-residual equivalence PASS: %s\n' \
  "${AUDIT_DIR}/zero_residual_equivalence_audit.json"
