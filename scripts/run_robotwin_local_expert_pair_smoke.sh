#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/sj/fastwam/RoboTwin-upstream}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
RUN_NAME="${RUN_NAME:-robotwin_local_expert_pair_smoke_2task3ep_20260827}"
GPU_ID="${GPU_ID:-0}"
TASKS_CSV="${TASKS:-hanging_mug,place_can_basket}"
HANGING_MUG_SEEDS="${HANGING_MUG_SEEDS:-0,2,3}"
PLACE_CAN_BASKET_SEEDS="${PLACE_CAN_BASKET_SEEDS:-0,1,2}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-30}"

ARTIFACT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${RUN_NAME}"
LOCAL_SOURCE_ROOT="${ARTIFACT_DIR}/local_expert_source"
RUN_DIR="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384/${RUN_NAME}"
ONLINE_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}"
MANIFEST="${ARTIFACT_DIR}/local_expert_manifest.json"
CASES_JSONL="${ARTIFACT_DIR}/local_expert_cases.jsonl"
LOCK_FILE="${ARTIFACT_DIR}/collector.lock"

for path in "${CHECKPOINT}" "${DATASET_STATS}"; do
  [[ -f "${path}" ]] || { printf 'Missing file: %s\n' "${path}" >&2; exit 1; }
done
[[ -d "${ROBOTWIN_ROOT}" ]] || { printf 'Missing RoboTwin root: %s\n' "${ROBOTWIN_ROOT}" >&2; exit 1; }
[[ "${COOLDOWN_SECONDS}" =~ ^[0-9]+$ ]] || { printf 'COOLDOWN_SECONDS must be non-negative\n' >&2; exit 1; }

mkdir -p "${ARTIFACT_DIR}" "${ONLINE_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[local-pair] another collector owns %s\n' "${LOCK_FILE}" >&2
  exit 73
fi
exec > >(tee -a "${ARTIFACT_DIR}/driver.log") 2>&1

seed_csv_for_task() {
  case "$1" in
    hanging_mug) printf '%s' "${HANGING_MUG_SEEDS}" ;;
    place_can_basket) printf '%s' "${PLACE_CAN_BASKET_SEEDS}" ;;
    *) printf 'No seed list configured for task %s\n' "$1" >&2; return 1 ;;
  esac
}

valid_video() {
  local video="$1" duration
  [[ -s "${video}" ]] || return 1
  duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "${video}" 2>/dev/null)"
  [[ -n "${duration}" ]] || return 1
  awk -v duration="${duration}" 'BEGIN { exit !(duration > 0) }'
}

IFS=',' read -r -a tasks <<< "${TASKS_CSV}"
episode_count=""
for task in "${tasks[@]}"; do
  task="$(printf '%s' "${task}" | xargs)"
  [[ -n "${task}" ]] || continue
  bundle="${LOCAL_SOURCE_ROOT}/${task}/local_current_clean"
  mkdir -p "${bundle}"
  seed_csv="$(seed_csv_for_task "${task}")"
  IFS=',' read -r -a seeds <<< "${seed_csv}"
  if [[ -z "${episode_count}" ]]; then
    episode_count="${#seeds[@]}"
  elif [[ "${episode_count}" -ne "${#seeds[@]}" ]]; then
    printf 'All tasks must use the same episode count\n' >&2
    exit 1
  fi
  seed_file_text=""
  for episode in "${!seeds[@]}"; do
    seed="$(printf '%s' "${seeds[$episode]}" | xargs)"
    [[ "${seed}" =~ ^[0-9]+$ ]] || { printf 'Invalid seed: %s\n' "${seed}" >&2; exit 1; }
    seed_file_text+="${seed} "
    marker="${ARTIFACT_DIR}/.${task}_expert_episode$(printf '%03d' "${episode}")_complete"
    metadata="${bundle}/pair_metadata/episode${episode}.json"
    expert_hdf5="${bundle}/data/episode${episode}.hdf5"
    expert_video="${bundle}/video/episode${episode}.mp4"
    if [[ -f "${marker}" && -s "${metadata}" && -s "${expert_hdf5}" ]] && valid_video "${expert_video}"; then
      printf '[local-pair] skip expert task=%s episode=%s seed=%s\n' "${task}" "${episode}" "${seed}"
      continue
    fi
    printf '[local-pair] collect expert task=%s episode=%s seed=%s\n' "${task}" "${episode}" "${seed}"
    conda run --no-capture-output -n "${CONDA_ENV}" \
      env CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
      python -u "${PROJECT_ROOT}/experiments/robotwin/collect_local_expert_pair_episode.py" \
      --robotwin-root "${ROBOTWIN_ROOT}" --task "${task}" --task-config demo_clean \
      --seed "${seed}" --episode-index "${episode}" --output-bundle "${bundle}"
    [[ -s "${metadata}" && -s "${expert_hdf5}" ]] || {
      printf '[local-pair] missing expert artifact task=%s episode=%s\n' "${task}" "${episode}" >&2
      exit 1
    }
    valid_video "${expert_video}" || {
      printf '[local-pair] invalid expert video task=%s episode=%s\n' "${task}" "${episode}" >&2
      exit 1
    }
    touch "${marker}"
    sync -d "${marker}" 2>/dev/null || true
    if (( COOLDOWN_SECONDS > 0 )); then sleep "${COOLDOWN_SECONDS}"; fi
  done
  printf '%s\n' "${seed_file_text}" > "${bundle}/seed.txt"
done

[[ -n "${episode_count}" && "${episode_count}" -gt 0 ]] || { printf 'No episodes configured\n' >&2; exit 1; }
conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/prepare_natural_failure_screen.py" \
  --dataset-root "${LOCAL_SOURCE_ROOT}" --tasks "${TASKS_CSV}" \
  --episodes-per-task "${episode_count}" --manifest "${MANIFEST}" \
  --cases-jsonl "${CASES_JSONL}"

for task in "${tasks[@]}"; do
  task="$(printf '%s' "${task}" | xargs)"
  [[ -n "${task}" ]] || continue
  for (( episode=0; episode<episode_count; episode++ )); do
    marker="${ARTIFACT_DIR}/.${task}_fastwam_episode$(printf '%03d' "${episode}")_complete"
    policy_video="${RUN_DIR}/${task}/episode${episode}.mp4"
    policy_current="${RUN_DIR}/${task}/imagination_transitions/${task}/policy/episode_$(printf '%04d' "${episode}")/replan_0000/current.png"
    if [[ -f "${marker}" && -s "${policy_current}" ]] && valid_video "${policy_video}"; then
      printf '[local-pair] skip FastWAM task=%s episode=%s\n' "${task}" "${episode}"
      continue
    fi
    printf '[local-pair] evaluate FastWAM task=%s episode=%s expert_check=true\n' "${task}" "${episode}"
    conda run --no-capture-output -n "${CONDA_ENV}" \
      env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
      MPLCONFIGDIR=/tmp/matplotlib_robotwin PYTHONUNBUFFERED=1 \
      python -u "${PROJECT_ROOT}/experiments/robotwin/eval_robotwin_single.py" \
      "ckpt=${CHECKPOINT}" seed=47 "gpu_id=${GPU_ID}" \
      "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}" \
      "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
      "EVALUATION.task_name=${task}" EVALUATION.task_config=demo_clean \
      EVALUATION.instruction_type=seen EVALUATION.eval_num_episodes=1 \
      "EVALUATION.trial_offset=${episode}" \
      EVALUATION.eval_video_log=true EVALUATION.num_inference_steps=10 \
      EVALUATION.replan_steps=24 EVALUATION.text_cfg_scale=1.0 \
      EVALUATION.skip_get_obs_within_replan=false \
      EVALUATION.action_mode=policy EVALUATION.action_noise_std=0.0 \
      EVALUATION.action_hold_probability=0.0 EVALUATION.gripper_close_delay_steps=0 \
      EVALUATION.environment_start_seed=0 "EVALUATION.environment_episode_offset=${episode}" \
      "EVALUATION.environment_seed_manifest_path=${MANIFEST}" \
      EVALUATION.deterministic_instruction_by_seed=true EVALUATION.expert_check=true \
      EVALUATION.fixed_instruction=null EVALUATION.paper_aligned=false \
      EVALUATION.strict_paired=false EVALUATION.save_imagination_transitions=true \
      EVALUATION.deterministic_algorithms=true EVALUATION.deterministic_warn_only=false \
      EVALUATION.timing_enabled=true "EVALUATION.output_dir=${ONLINE_DIR}"
    valid_video "${policy_video}" || {
      printf '[local-pair] invalid policy video task=%s episode=%s\n' "${task}" "${episode}" >&2
      exit 1
    }
    [[ -s "${policy_current}" ]] || { printf 'Missing policy initial: %s\n' "${policy_current}" >&2; exit 1; }
    touch "${marker}"
    sync -d "${marker}" 2>/dev/null || true
    if (( COOLDOWN_SECONDS > 0 )); then sleep "${COOLDOWN_SECONDS}"; fi
  done
done

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/validate_local_expert_fastwam_pairs.py" \
  --cases-jsonl "${CASES_JSONL}" --run-dir "${RUN_DIR}" \
  --output-json "${ARTIFACT_DIR}/strict_pair_audit.json" --require-valid

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/summarize_natural_failure_screen.py" \
  --cases-jsonl "${CASES_JSONL}" --run-dir "${RUN_DIR}" \
  --artifact-dir "${ARTIFACT_DIR}" --tasks "${TASKS_CSV}" --fps 20

printf '[local-pair] complete audit=%s summary=%s\n' \
  "${ARTIFACT_DIR}/strict_pair_audit.json" "${ARTIFACT_DIR}/summary.json"
