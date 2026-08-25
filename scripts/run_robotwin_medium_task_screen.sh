#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/sj/fastwam/RoboTwin-upstream}"
RUN_NAME="${RUN_NAME:-robotwin_imagination_medium_screen_5ep_20260825}"
TASKS_CSV="${TASKS:-hanging_mug,place_can_basket,stack_blocks_two,open_microwave,adjust_bottle}"
EPISODES="${EPISODES:-5}"
BASE_SEED="${BASE_SEED:-47}"
ENVIRONMENT_START_SEED="${ENVIRONMENT_START_SEED:-4800000}"
GPU_ID="${GPU_ID:-0}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
RUN_DIR="${RESULT_BASE}/${RUN_NAME}"
ARTIFACT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${RUN_NAME}"

for path in "${CHECKPOINT}" "${DATASET_STATS}"; do
  [[ -f "${path}" ]] || { printf 'Missing file: %s\n' "${path}" >&2; exit 1; }
done
[[ -d "${ROBOTWIN_ROOT}" ]] || { printf 'Missing RoboTwin root: %s\n' "${ROBOTWIN_ROOT}" >&2; exit 1; }
[[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]] || { printf 'EPISODES must be positive\n' >&2; exit 1; }
nvidia-modprobe -u -c=0
nvidia-smi -L >/dev/null

mkdir -p "${RUN_DIR}" "${ARTIFACT_DIR}"
exec > >(tee -a "${ARTIFACT_DIR}/driver.log") 2>&1
IFS=',' read -r -a tasks <<< "${TASKS_CSV}"

completed_log_for_task() {
  local task="$1"
  local latest_log
  latest_log="$(find "${RUN_DIR}" -maxdepth 1 -type f -name "eval_${task}_*.log" -print | sort | tail -n 1)"
  [[ -n "${latest_log}" ]] || return 1
  rg -q 'Success rate:' "${latest_log}" || return 1
  [[ "$(find "${RUN_DIR}/${task}" -maxdepth 1 -type f -name 'episode*.mp4' 2>/dev/null | wc -l)" -eq "${EPISODES}" ]]
}

for task in "${tasks[@]}"; do
  marker="${RUN_DIR}/.${task}_${EPISODES}ep_complete"
  if [[ -f "${marker}" ]] && completed_log_for_task "${task}"; then
    printf '[medium-screen] skip complete task=%s episodes=%s\n' "${task}" "${EPISODES}"
    continue
  fi
  printf '[medium-screen] start task=%s episodes=%s seed_start=%s\n' \
    "${task}" "${EPISODES}" "${ENVIRONMENT_START_SEED}"
  conda run --no-capture-output -n "${CONDA_ENV}" \
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    MPLCONFIGDIR=/tmp/matplotlib_robotwin PYTHONUNBUFFERED=1 \
    python -u "${PROJECT_ROOT}/experiments/robotwin/eval_robotwin_single.py" \
    "ckpt=${CHECKPOINT}" \
    "seed=${BASE_SEED}" \
    "gpu_id=${GPU_ID}" \
    "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}" \
    "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
    "EVALUATION.task_name=${task}" \
    EVALUATION.task_config=demo_clean \
    EVALUATION.instruction_type=unseen \
    "EVALUATION.eval_num_episodes=${EPISODES}" \
    EVALUATION.eval_video_log=true \
    EVALUATION.num_inference_steps=10 \
    EVALUATION.replan_steps=24 \
    EVALUATION.text_cfg_scale=1.0 \
    EVALUATION.action_mode=policy \
    EVALUATION.action_noise_std=0.0 \
    EVALUATION.action_hold_probability=0.0 \
    EVALUATION.gripper_close_delay_steps=0 \
    "EVALUATION.environment_start_seed=${ENVIRONMENT_START_SEED}" \
    EVALUATION.environment_episode_offset=0 \
    EVALUATION.environment_seed_manifest_path=null \
    EVALUATION.deterministic_instruction_by_seed=true \
    EVALUATION.expert_check=true \
    EVALUATION.fixed_instruction=null \
    EVALUATION.paper_aligned=true \
    EVALUATION.strict_paired=false \
    EVALUATION.save_imagination_transitions=false \
    EVALUATION.timing_enabled=true \
    "EVALUATION.output_dir=${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}"
  completed_log_for_task "${task}" || {
    printf 'Task did not produce a complete log/video set: %s\n' "${task}" >&2
    exit 1
  }
  touch "${marker}"
done

conda run --no-capture-output -n "${CONDA_ENV}" \
  python -u "${PROJECT_ROOT}/experiments/robotwin/summarize_medium_task_screen.py" \
  --run-dir "${RUN_DIR}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --tasks "${TASKS_CSV}"

printf '[medium-screen] complete summary=%s\n' "${ARTIFACT_DIR}/screen_summary.json"
