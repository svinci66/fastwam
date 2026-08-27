#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIRST_RESULT="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_low_success_pair_screen_2task3ep_20260827_wan_vae_reward/wan_vae_pair_rewards.json"
SECOND_RUN="robotwin_low_success_pair_expansion_2task5ep_seed7_20260827"
THIRD_RUN="robotwin_open_microwave_expansion_5ep_seed14_20260827"
SECOND_OUTPUT="${SECOND_RUN}_open_microwave_wan_vae_reward"
THIRD_OUTPUT="${THIRD_RUN}_open_microwave_wan_vae_reward"
SECOND_RESULT="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${SECOND_OUTPUT}/wan_vae_pair_rewards.json"
THIRD_RESULT="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${THIRD_OUTPUT}/wan_vae_pair_rewards.json"
SUMMARY_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_open_microwave_9pair_wan_vae_reward_20260827"
SUMMARY_JSON="${SUMMARY_DIR}/summary.json"

[[ -f "${FIRST_RESULT}" ]] || { printf 'Missing first result: %s\n' "${FIRST_RESULT}" >&2; exit 1; }
mkdir -p "${SUMMARY_DIR}"
exec > >(tee -a "${SUMMARY_DIR}/driver.log") 2>&1

env SOURCE_RUN_NAME="${SECOND_RUN}" OUTPUT_NAME="${SECOND_OUTPUT}" \
  TASKS=open_microwave bash "${PROJECT_ROOT}/scripts/run_robotwin_natural_failure_vae_reward.sh"

env SOURCE_RUN_NAME="${THIRD_RUN}" OUTPUT_NAME="${THIRD_OUTPUT}" \
  TASKS=open_microwave bash "${PROJECT_ROOT}/scripts/run_robotwin_natural_failure_vae_reward.sh"

conda run --no-capture-output -n robotwin_fastwam \
  env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src" PYTHONUNBUFFERED=1 \
  python -u "${PROJECT_ROOT}/experiments/robotwin/summarize_natural_failure_vae_rewards.py" \
  --inputs "${FIRST_RESULT}" "${SECOND_RESULT}" "${THIRD_RESULT}" \
  --task open_microwave --output-json "${SUMMARY_JSON}"

printf '[open-microwave-vae] complete summary=%s\n' "${SUMMARY_JSON}"
