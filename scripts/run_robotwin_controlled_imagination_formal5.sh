#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/sj/fastwam/RoboTwin-upstream}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_imagination_reward_formal5_20260825.json}"
RUN_NAME="${RUN_NAME:-robotwin_imagination_reward_formal5_20260825}"
MAX_ABS_DELTA="${MAX_ABS_DELTA:-0.05}"
GPU_ID="${GPU_ID:-0}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
ARTIFACT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${RUN_NAME}"

for path in "${CHECKPOINT}" "${DATASET_STATS}" "${MANIFEST}"; do
  [[ -f "${path}" ]] || { printf 'Missing file: %s\n' "${path}" >&2; exit 1; }
done
[[ -d "${ROBOTWIN_ROOT}" ]] || { printf 'Missing RoboTwin root: %s\n' "${ROBOTWIN_ROOT}" >&2; exit 1; }
mkdir -p "${ARTIFACT_DIR}"
exec > >(tee -a "${ARTIFACT_DIR}/driver.log") 2>&1
nvidia-modprobe -u -c=0
nvidia-smi -L

tasks=(hanging_mug place_can_basket)
episodes=(3 2)
environment_starts=(4800003 4800001)
intervention_replans=(3 2)
branch_tags=(clean corrupted corrected)
action_modes=(policy controlled_corrupt controlled_correct)
mode_tags=(policy controlled_corrupt_0.050 controlled_correct_0.050)

branch_complete() {
  local task="$1" count="$2" intervention="$3" tag="$4" mode_tag="$5"
  local root log_file episode
  root="${RESULT_BASE}/${RUN_NAME}_${tag}/${task}"
  log_file="$(find "${RESULT_BASE}/${RUN_NAME}_${tag}" -maxdepth 1 -type f -name "eval_${task}_*.log" -print 2>/dev/null | sort | tail -n 1)"
  [[ -n "${log_file}" ]] && rg -q 'Success rate:' "${log_file}" || return 1
  [[ "$(find "${root}" -maxdepth 1 -type f -name 'episode*.mp4' 2>/dev/null | wc -l)" -eq "${count}" ]] || return 1
  for (( episode=0; episode<count; episode++ )); do
    [[ -f "${root}/imagination_transitions/${task}/${mode_tag}/episode_$(printf '%04d' "${episode}")/replan_$(printf '%04d' "${intervention}")/metadata.json" ]] || return 1
  done
}

run_branch() {
  local task="$1" count="$2" environment_start="$3" intervention="$4"
  local tag="$5" mode="$6" mode_tag="$7" noise_std
  noise_std="${MAX_ABS_DELTA}"
  [[ "${mode}" == policy ]] && noise_std=0.0
  if branch_complete "${task}" "${count}" "${intervention}" "${tag}" "${mode_tag}"; then
    printf '[formal5] skip complete task=%s branch=%s\n' "${task}" "${tag}"
    return
  fi
  printf '[formal5] start task=%s episodes=%s branch=%s intervention=%s\n' \
    "${task}" "${count}" "${tag}" "${intervention}"
  timeout --verbose --signal=TERM --kill-after=30s 7200s \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    MPLCONFIGDIR=/tmp/matplotlib_robotwin PYTHONUNBUFFERED=1 \
    python -u "${PROJECT_ROOT}/experiments/robotwin/eval_robotwin_single.py" \
    "ckpt=${CHECKPOINT}" seed=47 "gpu_id=${GPU_ID}" \
    "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}" \
    "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
    "EVALUATION.task_name=${task}" EVALUATION.task_config=demo_clean \
    EVALUATION.instruction_type=unseen "EVALUATION.eval_num_episodes=${count}" \
    EVALUATION.eval_video_log=true EVALUATION.num_inference_steps=10 \
    EVALUATION.replan_steps=24 EVALUATION.text_cfg_scale=1.0 \
    "EVALUATION.action_mode=${mode}" "EVALUATION.action_noise_std=${noise_std}" \
    EVALUATION.action_noise_seed=20260825 \
    "EVALUATION.action_noise_replans=${intervention}" \
    EVALUATION.action_hold_probability=0.0 EVALUATION.gripper_close_delay_steps=0 \
    EVALUATION.trial_offset=0 "EVALUATION.environment_start_seed=${environment_start}" \
    "EVALUATION.environment_seed_manifest_path=${MANIFEST}" \
    EVALUATION.environment_episode_offset=0 \
    EVALUATION.deterministic_instruction_by_seed=true EVALUATION.expert_check=true \
    EVALUATION.fixed_instruction=null EVALUATION.paper_aligned=false \
    EVALUATION.strict_paired=false EVALUATION.save_imagination_transitions=true \
    EVALUATION.timing_enabled=true \
    "EVALUATION.output_dir=${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}_${tag}"
  branch_complete "${task}" "${count}" "${intervention}" "${tag}" "${mode_tag}" || {
    printf '[formal5] incomplete task=%s branch=%s\n' "${task}" "${tag}" >&2
    exit 65
  }
  printf '[formal5] complete task=%s branch=%s\n' "${task}" "${tag}"
}

for task_index in "${!tasks[@]}"; do
  task="${tasks[task_index]}"
  count="${episodes[task_index]}"
  environment_start="${environment_starts[task_index]}"
  intervention="${intervention_replans[task_index]}"
  for branch_index in "${!branch_tags[@]}"; do
    run_branch "${task}" "${count}" "${environment_start}" "${intervention}" \
      "${branch_tags[branch_index]}" "${action_modes[branch_index]}" "${mode_tags[branch_index]}"
  done
done

mkdir -p "${ARTIFACT_DIR}/audits" "${ARTIFACT_DIR}/videos"
for task_index in "${!tasks[@]}"; do
  task="${tasks[task_index]}"
  count="${episodes[task_index]}"
  intervention="${intervention_replans[task_index]}"
  for (( episode=0; episode<count; episode++ )); do
    episode_dir="episode_$(printf '%04d' "${episode}")"
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
      "${PROJECT_ROOT}/experiments/robotwin/audit_controlled_imagination_triplet.py" \
      --clean-root "${RESULT_BASE}/${RUN_NAME}_clean/${task}/imagination_transitions/${task}/policy/${episode_dir}" \
      --corrupt-root "${RESULT_BASE}/${RUN_NAME}_corrupted/${task}/imagination_transitions/${task}/controlled_corrupt_0.050/${episode_dir}" \
      --correct-root "${RESULT_BASE}/${RUN_NAME}_corrected/${task}/imagination_transitions/${task}/controlled_correct_0.050/${episode_dir}" \
      --intervention-replan "${intervention}" \
      --output-json "${ARTIFACT_DIR}/audits/${task}_${episode_dir}.json"
    mkdir -p "${ARTIFACT_DIR}/videos/${task}/${episode_dir}"
    for tag in "${branch_tags[@]}"; do
      ln -sfn "${RESULT_BASE}/${RUN_NAME}_${tag}/${task}/episode${episode}.mp4" \
        "${ARTIFACT_DIR}/videos/${task}/${episode_dir}/${tag}.mp4"
    done
  done
done
printf '[formal5] PASS audits=%s\n' "${ARTIFACT_DIR}/audits"
