#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-robotwin_open_microwave_wan_head_discordant7_trajectory_segmented_20260830}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_open_microwave_discordant7_diagnostic_20260830.json}"
WEIGHT_ROOT="${WEIGHT_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_open_microwave_wan_head_weight_sweep_20260827/weight025/training/seed42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${RUN_NAME}}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${OUTPUT_ROOT}/driver.log") 2>&1

for variant in no_imagination imagination; do
  for (( episode=0; episode<7; episode++ )); do
    segment="$(printf '%02d' "${episode}")"
    segment_run="${RUN_NAME}_segment${segment}"
    printf '[open-microwave-diagnostic] variant=%s episode=%s segment_run=%s\n' \
      "${variant}" "${episode}" "${segment_run}"
    env \
      RUN_NAME="${segment_run}" VARIANTS="${variant}" \
      TASKS=open_microwave EPISODES=1 BASE_SEED=47 TRIAL_OFFSET="${episode}" \
      INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
      TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
      PAPER_ALIGNED=true STRICT_PAIRED=true \
      DETERMINISTIC_INSTRUCTION_BY_SEED=true EXPERT_CHECK=true \
      SEED_MANIFEST_PATH="${MANIFEST}" \
      NO_IMAGINATION_CHECKPOINT="${WEIGHT_ROOT}/no_imagination/checkpoint.pt" \
      IMAGINATION_CHECKPOINT="${WEIGHT_ROOT}/with_imagination/checkpoint.pt" \
      RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-modelscope-local-v1 \
      RESIDUAL_LANGUAGE_MODE=policy_instruction \
      RESIDUAL_Q_GATE_ENABLED=false RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
      RESIDUAL_SUPPORT_INDEX_PATH=none RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
      RESIDUAL_SHADOW_MODE=false RESIDUAL_INTERVENTION_REPLANS=all \
      RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
      RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=false RESIDUAL_SOFT_SCALE_ENABLED=false \
      SAVE_BASELINE_TRANSITIONS=false SAVE_RESIDUAL_TRANSITIONS=true EVAL_VIDEO_LOG=true \
      bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
  done
done

conda run --no-capture-output -n robotwin_fastwam python -u \
  "${PROJECT_ROOT}/experiments/robotwin/analyze_open_microwave_progress_diagnostic.py" \
  --result-base "${RESULT_BASE}" --run-name "${RUN_NAME}" \
  --output-json "${OUTPUT_ROOT}/progress_diagnostic.json"

touch "${OUTPUT_ROOT}/COMPLETE"
printf '[open-microwave-diagnostic] complete: %s\n' "${OUTPUT_ROOT}/progress_diagnostic.json"
