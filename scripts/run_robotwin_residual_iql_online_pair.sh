#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/sj/fastwam/RoboTwin-upstream}"
SIGLIP_PATH="${SIGLIP_PATH:-/home/ubuntu/sj/fastwam/checkpoints/siglip-so400m-patch14-384-modelscope}"
RESIDUAL_ROOT="${RESIDUAL_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_expert10_residual_iql_20260729}"
NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_CHECKPOINT:-${RESIDUAL_ROOT}/iql_balanced_no_imagination/checkpoint.pt}"
IMAGINATION_CHECKPOINT="${IMAGINATION_CHECKPOINT:-${RESIDUAL_ROOT}/iql_balanced_imagination/checkpoint.pt}"
RESIDUAL_ENCODER_VERSION="${RESIDUAL_ENCODER_VERSION:-siglip-so400m-patch14-384-local-20260729}"
RUN_NAME="${RUN_NAME:-robotwin_residual_iql_online_pair_3task5ep_20260729}"
TASKS_CSV="${TASKS:-adjust_bottle,open_laptop,stack_blocks_two}"
EPISODES="${EPISODES:-5}"
BASE_SEED="${BASE_SEED:-42}"
TRIAL_OFFSET="${TRIAL_OFFSET:-0}"
INFERENCE_STEPS="${INFERENCE_STEPS:-4}"
GPU_ID="${GPU_ID:-0}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
SUMMARY_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}"

for path in "${CHECKPOINT}" "${DATASET_STATS}" \
  "${NO_IMAGINATION_CHECKPOINT}" "${IMAGINATION_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { printf 'Missing file: %s\n' "${path}" >&2; exit 1; }
done
for path in "${ROBOTWIN_ROOT}" "${SIGLIP_PATH}"; do
  [[ -d "${path}" ]] || { printf 'Missing directory: %s\n' "${path}" >&2; exit 1; }
done
[[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]] || { printf 'EPISODES must be positive\n' >&2; exit 1; }

instruction_for_task() {
  case "$1" in
    adjust_bottle)
      printf '%s' 'Pick up the bottle from the table and keep it upright.'
      ;;
    open_laptop)
      printf '%s' 'Open the laptop completely.'
      ;;
    stack_blocks_two)
      printf '%s' \
        'Move the red and green blocks to the center and stack the green block on the red block.'
      ;;
    *)
      printf 'No fixed instruction configured for task: %s\n' "$1" >&2
      return 1
      ;;
  esac
}

variants=(baseline no_imagination imagination)
checkpoints=(none "${NO_IMAGINATION_CHECKPOINT}" "${IMAGINATION_CHECKPOINT}")
IFS=',' read -r -a task_names <<< "${TASKS_CSV}"
environment_start_seed="$(( 100000 * (1 + BASE_SEED) + TRIAL_OFFSET ))"
mkdir -p "${SUMMARY_DIR}"

completed_log_for_task() {
  local run_dir="$1"
  local task_name="$2"
  local latest_log
  latest_log="$(find "${run_dir}" -maxdepth 1 -type f \
    -name "eval_${task_name}_*.log" -print | sort | tail -n 1)"
  [[ -n "${latest_log}" ]] || return 1
  rg -q 'Success rate:' "${latest_log}" || return 1
  printf '%s' "${latest_log}"
}

for variant_index in "${!variants[@]}"; do
  variant="${variants[$variant_index]}"
  residual_checkpoint="${checkpoints[$variant_index]}"
  run_dir="${RESULT_BASE}/${RUN_NAME}_${variant}"
  mkdir -p "${run_dir}"
  for task_name in "${task_names[@]}"; do
    marker="${run_dir}/.${task_name}_${EPISODES}ep_complete"
    if [[ -f "${marker}" ]] && completed_log_for_task "${run_dir}" "${task_name}" >/dev/null; then
      printf '[robotwin-online-pair] skip complete variant=%s task=%s\n' \
        "${variant}" "${task_name}"
      continue
    fi
    instruction="$(instruction_for_task "${task_name}")"
    action_mode=policy
    residual_args=()
    if [[ "${variant}" != "baseline" ]]; then
      action_mode=residual
      residual_args=(
        "EVALUATION.residual_checkpoint=${residual_checkpoint}"
        "EVALUATION.residual_encoder_path=${SIGLIP_PATH}"
        "EVALUATION.residual_encoder_version=${RESIDUAL_ENCODER_VERSION}"
        "EVALUATION.residual_encoder_dtype=bf16"
      )
    fi
    printf '[robotwin-online-pair] variant=%s task=%s episodes=%s env_seed=%s\n' \
      "${variant}" "${task_name}" "${EPISODES}" "${environment_start_seed}"
    conda run --no-capture-output -n "${CONDA_ENV}" \
      env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      MPLCONFIGDIR=/tmp/matplotlib_robotwin PYTHONUNBUFFERED=1 \
      python -u "${PROJECT_ROOT}/experiments/robotwin/eval_robotwin_single.py" \
      "ckpt=${CHECKPOINT}" \
      "seed=${BASE_SEED}" \
      "gpu_id=${GPU_ID}" \
      "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}" \
      "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
      "EVALUATION.task_name=${task_name}" \
      EVALUATION.task_config=demo_clean \
      "EVALUATION.eval_num_episodes=${EPISODES}" \
      "EVALUATION.trial_offset=${TRIAL_OFFSET}" \
      "EVALUATION.environment_start_seed=${environment_start_seed}" \
      "EVALUATION.environment_episode_offset=${TRIAL_OFFSET}" \
      "EVALUATION.num_inference_steps=${INFERENCE_STEPS}" \
      EVALUATION.replan_steps=24 \
      "EVALUATION.action_mode=${action_mode}" \
      "${residual_args[@]}" \
      "EVALUATION.fixed_instruction=${instruction}" \
      EVALUATION.timing_enabled=true \
      EVALUATION.save_imagination_transitions=false \
      "EVALUATION.output_dir=${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}_${variant}"
    completed_log_for_task "${run_dir}" "${task_name}" >/dev/null || {
      printf 'RoboTwin returned without a valid success-rate log: variant=%s task=%s\n' \
        "${variant}" "${task_name}" >&2
      exit 1
    }
    touch "${marker}"
  done
done

conda run --no-capture-output -n "${CONDA_ENV}" python \
  "${PROJECT_ROOT}/experiments/robotwin/summarize_residual_iql_online_pair.py" \
  --result-base "${RESULT_BASE}" \
  --run-name "${RUN_NAME}" \
  --tasks "${TASKS_CSV}" \
  --output-json "${SUMMARY_DIR}/summary.json"

printf 'RoboTwin residual-IQL online pair complete: %s\n' "${SUMMARY_DIR}/summary.json"
