#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${SEED:-44}"
EPISODES="${EPISODES:-10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
TASKS="open_microwave,hanging_mug,place_can_basket"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_balanced_pairs_seed${SEED}_20260904}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_balanced_heldout10_20260904/heldout_manifest.json}"
ONLINE_RUN_NAME="${ONLINE_RUN_NAME:-robotwin_video_expert_multitask3_balanced_seed${SEED}_formal10_20260904}"
CONTROL_CHECKPOINT="${RUN_ROOT}/training/seed${SEED}/no_imagination/checkpoint.pt"
TREATMENT_CHECKPOINT="${RUN_ROOT}/training/seed${SEED}/with_imagination/checkpoint.pt"
SUMMARY_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${ONLINE_RUN_NAME}"
SUMMARY="${SUMMARY_DIR}/summary.json"
FEATURE_VERSION="fastwam_video_expert_final_token_mean_l2_v1"

for path in "${MANIFEST}" "${CONTROL_CHECKPOINT}" "${TREATMENT_CHECKPOINT}"; do
  [[ -s "${path}" ]] || { printf '[balanced-formal] missing: %s\n' "${path}" >&2; exit 1; }
done
[[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]] || { printf 'EPISODES must be positive\n' >&2; exit 1; }
[[ "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || { printf 'MAX_ATTEMPTS must be positive\n' >&2; exit 1; }
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/formal10_online.log") 2>&1

run_formal() {
  env \
    RUN_NAME="${ONLINE_RUN_NAME}" \
    VARIANTS=baseline,no_imagination,imagination \
    TASKS="${TASKS}" EPISODES="${EPISODES}" BASE_SEED=47 TRIAL_OFFSET=0 \
    INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
    TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
    PAPER_ALIGNED=true STRICT_PAIRED=true \
    DETERMINISTIC_INSTRUCTION_BY_SEED=true EXPERT_CHECK=true \
    SEED_MANIFEST_PATH="${MANIFEST}" \
    NO_IMAGINATION_CHECKPOINT="${CONTROL_CHECKPOINT}" \
    IMAGINATION_CHECKPOINT="${TREATMENT_CHECKPOINT}" \
    RESIDUAL_ENCODER_PATH=none RESIDUAL_ENCODER_VERSION="${FEATURE_VERSION}" \
    RESIDUAL_LANGUAGE_MODE=policy_instruction \
    RESIDUAL_Q_GATE_ENABLED=false RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
    RESIDUAL_SUPPORT_INDEX_PATH=none RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
    RESIDUAL_SHADOW_MODE=false RESIDUAL_INTERVENTION_REPLANS=all \
    RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
    RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=false RESIDUAL_SOFT_SCALE_ENABLED=false \
    SAVE_BASELINE_TRANSITIONS=false SAVE_RESIDUAL_TRANSITIONS=false \
    EVAL_VIDEO_LOG=true \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
}

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  printf '[balanced-formal] attempt=%d/%d\n' "${attempt}" "${MAX_ATTEMPTS}"
  if run_formal; then
    break
  fi
  if (( attempt == MAX_ATTEMPTS )); then
    printf '[balanced-formal] exhausted attempts\n' >&2
    exit 1
  fi
  printf '[balanced-formal] retrying; completed task markers will be skipped\n' >&2
done

for task in open_microwave hanging_mug place_can_basket; do
  conda run --no-capture-output -n robotwin_fastwam python -u \
    "${PROJECT_ROOT}/experiments/robotwin/audit_wan_head_heldout_pair.py" \
    --summary "${SUMMARY}" --task "${task}" --expected-pairs "${EPISODES}" \
    --output-json "${SUMMARY_DIR}/audit_${task}.json"
done

touch "${RUN_ROOT}/FORMAL10_COMPLETE"
printf '[balanced-formal] complete: %s\n' "${SUMMARY}"
