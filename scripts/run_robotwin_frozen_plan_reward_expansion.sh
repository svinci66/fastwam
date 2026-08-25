#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_RUN_NAME="${BASE_RUN_NAME:-robotwin_frozen_plan_reward_expansion_20260825}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_frozen_plan_reward_expansion_20260825.json}"
VAE_PATH="${VAE_PATH:-/home/ubuntu/sj/fastwam/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
ARTIFACT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${BASE_RUN_NAME}"
TARGET_NEW_PAIRS="${TARGET_NEW_PAIRS:-6}"
mkdir -p "${ARTIFACT_DIR}"
exec > >(tee -a "${ARTIFACT_DIR}/driver.log") 2>&1

STATUS_TSV="${ARTIFACT_DIR}/status.tsv"
if [[ ! -f "${STATUS_TSV}" ]]; then
  printf 'timestamp\ttask\tseed\tepisode_offset\ttrial_offset\tintervention_replan\tclean\tcorrupted\tcorrected\tdecision\n' \
    > "${STATUS_TSV}"
fi

# Cases are ordered by current-protocol evidence first. Historical hanging_mug
# successes are fallbacks and are rechecked by a fresh clean branch before use.
# task seed manifest_offset trial_offset intervention_replan case_tag
cases=(
  "place_can_basket 4800003 0 0 2 place_seed4800003"
  "hanging_mug 4800003 0 0 3 hanging_seed4800003"
  "hanging_mug 4800005 1 1 3 hanging_seed4800005"
  "adjust_bottle 4800000 0 0 2 adjust_seed4800000"
  "adjust_bottle 4800001 1 1 2 adjust_seed4800001"
  "adjust_bottle 4800002 2 2 2 adjust_seed4800002"
  "adjust_bottle 4800003 3 3 2 adjust_seed4800003"
  "adjust_bottle 4800004 4 4 2 adjust_seed4800004"
  "stack_blocks_two 4800000 0 0 2 stack_seed4800000_red"
  "stack_blocks_two 4800001 1 1 9 stack_seed4800001_green"
  "stack_blocks_two 4800002 2 2 2 stack_seed4800002_red"
  "stack_blocks_two 4800003 3 3 9 stack_seed4800003_green"
  "stack_blocks_two 4800004 4 4 2 stack_seed4800004_red"
  "hanging_mug 4800007 2 2 3 hanging_seed4800007"
  "hanging_mug 4800008 3 3 3 hanging_seed4800008"
  "hanging_mug 4800011 4 4 3 hanging_seed4800011"
  "hanging_mug 4800015 5 5 3 hanging_seed4800015"
)

record_status() {
  local task="$1" seed="$2" episode_offset="$3" trial_offset="$4"
  local intervention="$5" clean="$6" corrupt="$7" correct="$8" decision="$9"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "${task}" "${seed}" "${episode_offset}" \
    "${trial_offset}" "${intervention}" "${clean}" "${corrupt}" \
    "${correct}" "${decision}" >> "${STATUS_TSV}"
}

result_value() {
  local run_name="$1" branch="$2" task="$3"
  local path="${RESULT_BASE}/${run_name}_${branch}/${task}/_result.txt"
  [[ -s "${path}" ]] || return 1
  awk 'NF { value=$NF } END { print value }' "${path}"
}

is_success() {
  awk -v value="$1" 'BEGIN { exit !(value + 0.0 >= 0.5) }'
}

run_branches() {
  local run_name="$1" task="$2" seed="$3" episode_offset="$4"
  local trial_offset="$5" intervention="$6" branches="$7" audit="$8"
  RUN_NAME="${run_name}" TASK="${task}" \
    ENVIRONMENT_START_SEED="${seed}" INTERVENTION_REPLAN="${intervention}" \
    ENVIRONMENT_EPISODE_OFFSET="${episode_offset}" TRIAL_OFFSET="${trial_offset}" \
    MANIFEST="${MANIFEST}" BRANCHES="${branches}" RUN_AUDIT="${audit}" \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_frozen_plan_trajectory_smoke.sh"
}

valid_pairs="$(find "${ARTIFACT_DIR}" -maxdepth 1 -type f -name '*_vae_reward.json' | wc -l)"
printf '[reward-expansion] resume valid_new_pairs=%s target=%s\n' \
  "${valid_pairs}" "${TARGET_NEW_PAIRS}"

for case_spec in "${cases[@]}"; do
  if (( valid_pairs >= TARGET_NEW_PAIRS )); then
    printf '[reward-expansion] target reached; stop screening\n'
    break
  fi
  read -r task seed episode_offset trial_offset intervention case_tag <<< "${case_spec}"
  run_name="${BASE_RUN_NAME}_${case_tag}"
  validation_json="${ARTIFACT_DIR}/${case_tag}_vae_reward.json"
  if [[ -s "${validation_json}" ]]; then
    printf '[reward-expansion] skip validated case=%s\n' "${case_tag}"
    continue
  fi

  printf '[reward-expansion] corrupt screen case=%s task=%s seed=%s replan=%s\n' \
    "${case_tag}" "${task}" "${seed}" "${intervention}"
  run_branches "${run_name}" "${task}" "${seed}" "${episode_offset}" \
    "${trial_offset}" "${intervention}" corrupted false
  corrupt_result="$(result_value "${run_name}" corrupted "${task}")"
  if is_success "${corrupt_result}"; then
    record_status "${task}" "${seed}" "${episode_offset}" "${trial_offset}" \
      "${intervention}" not_run "${corrupt_result}" not_run corrupt_still_succeeds
    printf '[reward-expansion] reject case=%s reason=corrupt_still_succeeds\n' "${case_tag}"
    continue
  fi

  printf '[reward-expansion] corrupt failed; confirm clean case=%s\n' "${case_tag}"
  run_branches "${run_name}" "${task}" "${seed}" "${episode_offset}" \
    "${trial_offset}" "${intervention}" clean false
  clean_result="$(result_value "${run_name}" clean "${task}")"
  if ! is_success "${clean_result}"; then
    record_status "${task}" "${seed}" "${episode_offset}" "${trial_offset}" \
      "${intervention}" "${clean_result}" "${corrupt_result}" not_run clean_not_reproduced
    printf '[reward-expansion] reject case=%s reason=clean_not_reproduced\n' "${case_tag}"
    continue
  fi

  printf '[reward-expansion] strict candidate; run corrected case=%s\n' "${case_tag}"
  run_branches "${run_name}" "${task}" "${seed}" "${episode_offset}" \
    "${trial_offset}" "${intervention}" corrected false
  correct_result="$(result_value "${run_name}" corrected "${task}")"
  if ! is_success "${correct_result}"; then
    record_status "${task}" "${seed}" "${episode_offset}" "${trial_offset}" \
      "${intervention}" "${clean_result}" "${corrupt_result}" "${correct_result}" correction_did_not_recover
    printf '[reward-expansion] reject case=%s reason=correction_did_not_recover\n' "${case_tag}"
    continue
  fi

  # All branches now exist. The full invocation performs fail-closed action and
  # trajectory audits while skipping already complete rollouts.
  run_branches "${run_name}" "${task}" "${seed}" "${episode_offset}" \
    "${trial_offset}" "${intervention}" clean,corrupted,corrected true

  episode_dir="episode_$(printf '%04d' "${trial_offset}")"
  clean_root="${RESULT_BASE}/${run_name}_clean/${task}/imagination_transitions/${task}/policy/${episode_dir}"
  corrupt_root="${RESULT_BASE}/${run_name}_corrupted/${task}/imagination_transitions/${task}/controlled_corrupt_0.050/${episode_dir}"
  correct_root="${RESULT_BASE}/${run_name}_corrected/${task}/imagination_transitions/${task}/controlled_correct_0.050/${episode_dir}"
  conda run --no-capture-output -n robotwin_fastwam python -u \
    "${PROJECT_ROOT}/experiments/robotwin/validate_frozen_plan_vae_reward.py" \
    --clean-root "${clean_root}" --corrupt-root "${corrupt_root}" \
    --correct-root "${correct_root}" --replan "${intervention}" \
    --shuffle-replan "$((intervention + 1))" --vae-path "${VAE_PATH}" \
    --device cuda --dtype bf16 --output-json "${validation_json}"
  valid_pairs=$((valid_pairs + 1))
  record_status "${task}" "${seed}" "${episode_offset}" "${trial_offset}" \
    "${intervention}" "${clean_result}" "${corrupt_result}" "${correct_result}" strict_pair_validated
  printf '[reward-expansion] accepted case=%s valid_new_pairs=%s/%s\n' \
    "${case_tag}" "${valid_pairs}" "${TARGET_NEW_PAIRS}"
done

validation_files=(
  "${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_frozen_plan_discordant2_v2_fixed_20260825/"*_vae_reward.json
  "${ARTIFACT_DIR}/"*_vae_reward.json
)
existing_validation_files=()
for path in "${validation_files[@]}"; do
  [[ -s "${path}" ]] && existing_validation_files+=("${path}")
done
if (( ${#existing_validation_files[@]} > 0 )); then
  summary_args=()
  for path in "${existing_validation_files[@]}"; do
    summary_args+=(--input-json "${path}")
  done
  conda run --no-capture-output -n robotwin_fastwam python -u \
    "${PROJECT_ROOT}/experiments/robotwin/summarize_frozen_plan_vae_validations.py" \
    "${summary_args[@]}" --output-json "${ARTIFACT_DIR}/summary.json"
fi

if (( valid_pairs < TARGET_NEW_PAIRS )); then
  printf '[reward-expansion] STOP_STANDARD_NOT_MET valid_new_pairs=%s target=%s; do not train actor\n' \
    "${valid_pairs}" "${TARGET_NEW_PAIRS}"
  exit 3
fi
printf '[reward-expansion] PASS valid_new_pairs=%s target=%s summary=%s\n' \
  "${valid_pairs}" "${TARGET_NEW_PAIRS}" "${ARTIFACT_DIR}/summary.json"
