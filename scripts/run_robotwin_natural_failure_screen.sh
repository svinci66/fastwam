#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/sj/fastwam/RoboTwin-upstream}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/robotwin_official_success/extracted}"
TASKS_CSV="${TASKS:-hanging_mug,place_can_basket}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
RUN_NAME="${RUN_NAME:-robotwin_natural_failure_screen_2task10ep_20260826}"
GPU_ID="${GPU_ID:-0}"
JOB_COOLDOWN_SECONDS="${JOB_COOLDOWN_SECONDS:-60}"

ARTIFACT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${RUN_NAME}"
RUN_DIR="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384/${RUN_NAME}"
ONLINE_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}"
MANIFEST="${ARTIFACT_DIR}/expert_source_manifest.json"
CASES_JSONL="${ARTIFACT_DIR}/expert_source_cases.jsonl"
LOCK_FILE="${ARTIFACT_DIR}/collector.lock"

for path in "${CHECKPOINT}" "${DATASET_STATS}"; do
  [[ -f "${path}" ]] || { printf 'Missing file: %s\n' "${path}" >&2; exit 1; }
done
[[ -d "${ROBOTWIN_ROOT}" ]] || { printf 'Missing RoboTwin root: %s\n' "${ROBOTWIN_ROOT}" >&2; exit 1; }
[[ -d "${DATASET_ROOT}" ]] || { printf 'Missing dataset root: %s\n' "${DATASET_ROOT}" >&2; exit 1; }
[[ "${EPISODES_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || { printf 'EPISODES_PER_TASK must be positive\n' >&2; exit 1; }

mkdir -p "${ARTIFACT_DIR}" "${ONLINE_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[natural-failure] another collector owns %s\n' "${LOCK_FILE}" >&2
  exit 73
fi
exec > >(tee -a "${ARTIFACT_DIR}/driver.log") 2>&1

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/prepare_natural_failure_screen.py" \
  --dataset-root "${DATASET_ROOT}" \
  --tasks "${TASKS_CSV}" \
  --episodes-per-task "${EPISODES_PER_TASK}" \
  --manifest "${MANIFEST}" \
  --cases-jsonl "${CASES_JSONL}"

completed_task() {
  local task="$1" marker="${ARTIFACT_DIR}/.${task}_complete" latest_log
  [[ -f "${marker}" ]] || return 1
  latest_log="$(find "${RUN_DIR}" -maxdepth 1 -type f -name "eval_${task}_*.log" -print 2>/dev/null | sort | tail -n 1)"
  [[ -n "${latest_log}" ]] || return 1
  rg -q 'Success rate:' "${latest_log}" || return 1
  [[ "$(find "${RUN_DIR}/${task}" -maxdepth 1 -type f -name 'episode*.mp4' 2>/dev/null | wc -l)" -eq "${EPISODES_PER_TASK}" ]]
}

IFS=',' read -r -a tasks <<< "${TASKS_CSV}"
for task in "${tasks[@]}"; do
  task="$(printf '%s' "${task}" | xargs)"
  [[ -n "${task}" ]] || continue
  if completed_task "${task}"; then
    printf '[natural-failure] skip complete task=%s\n' "${task}"
    continue
  fi
  printf '[natural-failure] evaluate task=%s episodes=%s split=seen no_artificial_corruption=true\n' \
    "${task}" "${EPISODES_PER_TASK}"
  conda run --no-capture-output -n "${CONDA_ENV}" \
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    MPLCONFIGDIR=/tmp/matplotlib_robotwin PYTHONUNBUFFERED=1 \
    python -u "${PROJECT_ROOT}/experiments/robotwin/eval_robotwin_single.py" \
    "ckpt=${CHECKPOINT}" seed=47 "gpu_id=${GPU_ID}" \
    "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}" \
    "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
    "EVALUATION.task_name=${task}" EVALUATION.task_config=demo_clean \
    EVALUATION.instruction_type=seen \
    "EVALUATION.eval_num_episodes=${EPISODES_PER_TASK}" \
    EVALUATION.eval_video_log=true EVALUATION.num_inference_steps=10 \
    EVALUATION.replan_steps=24 EVALUATION.text_cfg_scale=1.0 \
    EVALUATION.skip_get_obs_within_replan=false \
    EVALUATION.action_mode=policy EVALUATION.action_noise_std=0.0 \
    EVALUATION.action_hold_probability=0.0 EVALUATION.gripper_close_delay_steps=0 \
    EVALUATION.environment_start_seed=0 EVALUATION.environment_episode_offset=0 \
    "EVALUATION.environment_seed_manifest_path=${MANIFEST}" \
    EVALUATION.deterministic_instruction_by_seed=true EVALUATION.expert_check=false \
    EVALUATION.fixed_instruction=null EVALUATION.paper_aligned=false \
    EVALUATION.strict_paired=false EVALUATION.save_imagination_transitions=true \
    EVALUATION.deterministic_algorithms=true EVALUATION.deterministic_warn_only=false \
    EVALUATION.timing_enabled=true "EVALUATION.output_dir=${ONLINE_DIR}"
  touch "${ARTIFACT_DIR}/.${task}_complete"
  sync -d "${ARTIFACT_DIR}/.${task}_complete" 2>/dev/null || true
  if (( JOB_COOLDOWN_SECONDS > 0 )); then
    printf '[natural-failure] task complete; cooldown=%ss\n' "${JOB_COOLDOWN_SECONDS}"
    sleep "${JOB_COOLDOWN_SECONDS}"
  fi
done

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/summarize_natural_failure_screen.py" \
  --cases-jsonl "${CASES_JSONL}" \
  --run-dir "${RUN_DIR}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --tasks "${TASKS_CSV}" \
  --fps 20

printf '[natural-failure] complete summary=%s\n' "${ARTIFACT_DIR}/summary.json"
