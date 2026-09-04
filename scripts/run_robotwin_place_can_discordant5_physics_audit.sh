#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-robotwin_place_can_discordant5_three_way_physics_audit_20260904}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_place_can_basket_discordant5_20260904.json}"
TRAIN_ROOT="${TRAIN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_place_can_basket_single_task_epoch003_seed42_20260903/training/seed42}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${RUN_NAME}}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

NO_IMAGINATION_CHECKPOINT="${TRAIN_ROOT}/no_imagination/checkpoint.pt"
IMAGINATION_CHECKPOINT="${TRAIN_ROOT}/with_imagination/checkpoint.pt"
PHYSICS_AUDIT_ROOT="${ARTIFACT_ROOT}/physics_audit"
SUMMARY="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}/summary.json"

for path in "${MANIFEST}" "${NO_IMAGINATION_CHECKPOINT}" "${IMAGINATION_CHECKPOINT}"; do
  [[ -s "${path}" ]] || { printf '[discordant5-audit] missing: %s\n' "${path}" >&2; exit 1; }
done
[[ "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || {
  printf '[discordant5-audit] MAX_ATTEMPTS must be positive\n' >&2
  exit 1
}

mkdir -p "${ARTIFACT_ROOT}"
exec > >(tee -a "${ARTIFACT_ROOT}/driver.log") 2>&1

run_three_way() {
  env \
    RUN_NAME="${RUN_NAME}" \
    VARIANTS=baseline,no_imagination,imagination \
    TASKS=place_can_basket EPISODES=5 BASE_SEED=47 TRIAL_OFFSET=0 \
    INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
    TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
    PAPER_ALIGNED=true STRICT_PAIRED=true \
    DETERMINISTIC_INSTRUCTION_BY_SEED=true EXPERT_CHECK=true \
    SEED_MANIFEST_PATH="${MANIFEST}" \
    NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_CHECKPOINT}" \
    IMAGINATION_CHECKPOINT="${IMAGINATION_CHECKPOINT}" \
    RESIDUAL_ENCODER_PATH=none \
    RESIDUAL_ENCODER_VERSION=fastwam_video_expert_final_token_mean_l2_v1 \
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
    PHYSICS_AUDIT_ENABLED=true \
    PHYSICS_AUDIT_ACTOR_ATTR=can \
    PHYSICS_AUDIT_OUTPUT_ROOT="${PHYSICS_AUDIT_ROOT}" \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
}

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  printf '[discordant5-audit] attempt=%d/%d\n' "${attempt}" "${MAX_ATTEMPTS}"
  if run_three_way; then
    break
  fi
  if (( attempt == MAX_ATTEMPTS )); then
    printf '[discordant5-audit] exhausted attempts\n' >&2
    exit 1
  fi
done

[[ -s "${SUMMARY}" ]] || {
  printf '[discordant5-audit] missing summary: %s\n' "${SUMMARY}" >&2
  exit 1
}
conda run --no-capture-output -n robotwin_fastwam python -u \
  "${PROJECT_ROOT}/experiments/robotwin/analyze_place_can_physics_audit.py" \
  --audit-root "${PHYSICS_AUDIT_ROOT}" \
  --online-summary "${SUMMARY}" \
  --manifest "${MANIFEST}" \
  --output-json "${ARTIFACT_ROOT}/physics_audit_summary.json"
touch "${ARTIFACT_ROOT}/COMPLETE"
printf '[discordant5-audit] complete: %s\n' "${SUMMARY}"
