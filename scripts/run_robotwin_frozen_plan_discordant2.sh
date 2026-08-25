#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_RUN_NAME="${BASE_RUN_NAME:-robotwin_frozen_plan_discordant2_v2_fixed_20260825}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_imagination_reward_formal5_20260825.json}"
VAE_PATH="${VAE_PATH:-/home/ubuntu/sj/fastwam/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
ARTIFACT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${BASE_RUN_NAME}"
mkdir -p "${ARTIFACT_DIR}"
exec > >(tee -a "${ARTIFACT_DIR}/driver.log") 2>&1

cases=(
  "hanging_mug 4800004 1 3 hanging_seed4800004"
  "place_can_basket 4800001 0 2 place_seed4800001"
)
validation_files=()

for case_spec in "${cases[@]}"; do
  read -r task seed episode_offset intervention case_tag <<< "${case_spec}"
  run_name="${BASE_RUN_NAME}_${case_tag}"
  printf '[discordant2] start task=%s seed=%s intervention=%s\n' \
    "${task}" "${seed}" "${intervention}"
  RUN_NAME="${run_name}" TASK="${task}" \
    ENVIRONMENT_START_SEED="${seed}" INTERVENTION_REPLAN="${intervention}" \
    ENVIRONMENT_EPISODE_OFFSET="${episode_offset}" \
    MANIFEST="${MANIFEST}" \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_frozen_plan_trajectory_smoke.sh"

  clean_root="${RESULT_BASE}/${run_name}_clean/${task}/imagination_transitions/${task}/policy/episode_0000"
  corrupt_root="${RESULT_BASE}/${run_name}_corrupted/${task}/imagination_transitions/${task}/controlled_corrupt_0.050/episode_0000"
  correct_root="${RESULT_BASE}/${run_name}_corrected/${task}/imagination_transitions/${task}/controlled_correct_0.050/episode_0000"
  validation_json="${ARTIFACT_DIR}/${case_tag}_vae_reward.json"
  conda run --no-capture-output -n robotwin_fastwam python -u \
    "${PROJECT_ROOT}/experiments/robotwin/validate_frozen_plan_vae_reward.py" \
    --clean-root "${clean_root}" --corrupt-root "${corrupt_root}" \
    --correct-root "${correct_root}" --replan "${intervention}" \
    --shuffle-replan "$((intervention + 1))" --vae-path "${VAE_PATH}" \
    --device cuda --dtype bf16 --output-json "${validation_json}"
  validation_files+=("${validation_json}")
  printf '[discordant2] complete task=%s seed=%s validation=%s\n' \
    "${task}" "${seed}" "${validation_json}"
done

summary_args=()
for path in "${validation_files[@]}"; do
  summary_args+=(--input-json "${path}")
done
conda run --no-capture-output -n robotwin_fastwam python -u \
  "${PROJECT_ROOT}/experiments/robotwin/summarize_frozen_plan_vae_validations.py" \
  "${summary_args[@]}" --output-json "${ARTIFACT_DIR}/summary.json"
printf '[discordant2] PASS summary=%s\n' "${ARTIFACT_DIR}/summary.json"
