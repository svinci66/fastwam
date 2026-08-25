#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/sj/fastwam/RoboTwin-upstream}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_imagination_reward_smoke_20260825.json}"
RUN_NAME="${RUN_NAME:-robotwin_frozen_plan_trajectory_v2_smoke_20260825}"
TASK="${TASK:-hanging_mug}"
INTERVENTION_REPLAN="${INTERVENTION_REPLAN:-3}"
MAX_ABS_DELTA="${MAX_ABS_DELTA:-0.05}"
GPU_ID="${GPU_ID:-0}"
ARTIFACT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${RUN_NAME}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"

for path in "${CHECKPOINT}" "${DATASET_STATS}" "${MANIFEST}"; do
  [[ -f "${path}" ]] || { printf 'Missing file: %s\n' "${path}" >&2; exit 1; }
done
[[ -d "${ROBOTWIN_ROOT}" ]] || { printf 'Missing RoboTwin root: %s\n' "${ROBOTWIN_ROOT}" >&2; exit 1; }
mkdir -p "${ARTIFACT_DIR}"
exec > >(tee -a "${ARTIFACT_DIR}/driver.log") 2>&1

nvidia-modprobe -u -c=0
nvidia-smi -L

transition_root() {
  local tag="$1" mode_tag="$2"
  printf '%s' "${RESULT_BASE}/${RUN_NAME}_${tag}/${TASK}/imagination_transitions/${TASK}/${mode_tag}/episode_0000"
}

branch_complete() {
  local tag="$1" mode_tag="$2" root log_file video metadata
  root="${RESULT_BASE}/${RUN_NAME}_${tag}/${TASK}"
  log_file="$(find "${RESULT_BASE}/${RUN_NAME}_${tag}" -maxdepth 1 -type f -name "eval_${TASK}_*.log" -print 2>/dev/null | sort | tail -n 1)"
  video="${root}/episode0.mp4"
  metadata="$(transition_root "${tag}" "${mode_tag}")/replan_$(printf '%04d' "${INTERVENTION_REPLAN}")/metadata.json"
  [[ -n "${log_file}" ]] && rg -q 'Success rate:' "${log_file}" \
    && [[ -s "${video}" ]] \
    && [[ -f "${metadata}" ]] \
    && rg -q 'robotwin_imagination_trajectory_v2' "${metadata}"
}

run_branch() {
  local tag="$1" mode="$2" noise_std="$3" mode_tag="$4"
  if branch_complete "${tag}" "${mode_tag}"; then
    printf '[trajectory-smoke] skip complete branch=%s\n' "${tag}"
    return
  fi
  printf '[trajectory-smoke] start branch=%s mode=%s replan=%s max_abs_delta=%s\n' \
    "${tag}" "${mode}" "${INTERVENTION_REPLAN}" "${noise_std}"
  timeout --verbose --signal=TERM --kill-after=30s 2400s \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    MPLCONFIGDIR=/tmp/matplotlib_robotwin PYTHONUNBUFFERED=1 \
    python -u "${PROJECT_ROOT}/experiments/robotwin/eval_robotwin_single.py" \
    "ckpt=${CHECKPOINT}" seed=47 "gpu_id=${GPU_ID}" \
    "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}" \
    "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
    "EVALUATION.task_name=${TASK}" EVALUATION.task_config=demo_clean \
    EVALUATION.instruction_type=unseen EVALUATION.eval_num_episodes=1 \
    EVALUATION.eval_video_log=true EVALUATION.num_inference_steps=10 \
    EVALUATION.replan_steps=24 EVALUATION.text_cfg_scale=1.0 \
    "EVALUATION.action_mode=${mode}" "EVALUATION.action_noise_std=${noise_std}" \
    EVALUATION.action_noise_seed=20260825 \
    "EVALUATION.action_noise_replans=${INTERVENTION_REPLAN}" \
    EVALUATION.action_hold_probability=0.0 EVALUATION.gripper_close_delay_steps=0 \
    EVALUATION.trial_offset=0 EVALUATION.environment_start_seed=4800003 \
    "EVALUATION.environment_seed_manifest_path=${MANIFEST}" \
    EVALUATION.environment_episode_offset=0 \
    EVALUATION.deterministic_instruction_by_seed=true EVALUATION.expert_check=true \
    EVALUATION.fixed_instruction=null EVALUATION.paper_aligned=false \
    EVALUATION.strict_paired=false EVALUATION.save_imagination_transitions=true \
    EVALUATION.timing_enabled=true \
    "EVALUATION.output_dir=${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}_${tag}"
  branch_complete "${tag}" "${mode_tag}" || {
    printf '[trajectory-smoke] incomplete branch=%s\n' "${tag}" >&2
    exit 65
  }
  printf '[trajectory-smoke] complete branch=%s\n' "${tag}"
}

run_branch clean policy 0.0 policy
run_branch corrupted controlled_corrupt "${MAX_ABS_DELTA}" "controlled_corrupt_$(printf '%.3f' "${MAX_ABS_DELTA}")"
run_branch corrected controlled_correct "${MAX_ABS_DELTA}" "controlled_correct_$(printf '%.3f' "${MAX_ABS_DELTA}")"

CLEAN_ROOT="$(transition_root clean policy)"
CORRUPT_ROOT="$(transition_root corrupted "controlled_corrupt_$(printf '%.3f' "${MAX_ABS_DELTA}")")"
CORRECT_ROOT="$(transition_root corrected "controlled_correct_$(printf '%.3f' "${MAX_ABS_DELTA}")")"

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_controlled_imagination_triplet.py" \
  --clean-root "${CLEAN_ROOT}" --corrupt-root "${CORRUPT_ROOT}" \
  --correct-root "${CORRECT_ROOT}" --intervention-replan "${INTERVENTION_REPLAN}" \
  --output-json "${ARTIFACT_DIR}/action_triplet_audit.json"

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_frozen_plan_trajectory_triplet.py" \
  --clean-root "${CLEAN_ROOT}" --corrupt-root "${CORRUPT_ROOT}" \
  --correct-root "${CORRECT_ROOT}" --intervention-replan "${INTERVENTION_REPLAN}" \
  --output-json "${ARTIFACT_DIR}/trajectory_triplet_audit.json"

mkdir -p "${ARTIFACT_DIR}/videos"
for tag in clean corrupted corrected; do
  ln -sfn "${RESULT_BASE}/${RUN_NAME}_${tag}/${TASK}/episode0.mp4" \
    "${ARTIFACT_DIR}/videos/${tag}.mp4"
done
printf '[trajectory-smoke] PASS audit=%s\n' "${ARTIFACT_DIR}/trajectory_triplet_audit.json"
