#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/sj/fastwam/RoboTwin-upstream}"
SIGLIP_PATH="${SIGLIP_PATH:-/home/ubuntu/sj/fastwam/checkpoints/siglip-so400m-patch14-384-modelscope}"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
OUTPUT_NAME="${OUTPUT_NAME:-robotwin_move_can_controlled_corruption_$(date +%Y%m%d_%H%M%S)}"
EPISODES="${EPISODES:-5}"
MAX_EPISODES_PER_PROCESS="${MAX_EPISODES_PER_PROCESS:-4}"
BASE_SEED="${BASE_SEED:-42}"
INFERENCE_STEPS="${INFERENCE_STEPS:-4}"
GPU_ID="${GPU_ID:-0}"
ACTION_CORRUPTION_SEED="${ACTION_CORRUPTION_SEED:-20260729}"

task_name="move_can_pot"
instruction="Pick up the can and place it beside the pot."
modes=(policy hold hold gripper_delay)
hold_probabilities=(0.0 0.25 0.75 0.0)
gripper_delay_steps=(0 0 0 24)
mode_tags=(policy hold_0.250 hold_0.750 gripper_delay_024)

[[ -f "${CHECKPOINT}" ]] || { printf 'Missing checkpoint: %s\n' "${CHECKPOINT}" >&2; exit 1; }
[[ -f "${DATASET_STATS}" ]] || { printf 'Missing dataset stats: %s\n' "${DATASET_STATS}" >&2; exit 1; }
[[ -d "${ROBOTWIN_ROOT}" ]] || { printf 'Missing RoboTwin root: %s\n' "${ROBOTWIN_ROOT}" >&2; exit 1; }
[[ -d "${SIGLIP_PATH}" ]] || { printf 'Missing SigLIP path: %s\n' "${SIGLIP_PATH}" >&2; exit 1; }

output_dir="${PROJECT_ROOT}/evaluate_results/robotwin/${OUTPUT_NAME}"
raw_output_dir="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384/${OUTPUT_NAME}"

episode_is_complete() {
  local mode_tag="$1"
  local trial="$2"
  local episode_dir policy_current
  episode_dir="${raw_output_dir}/${task_name}/imagination_transitions/${task_name}/${mode_tag}/episode_$(printf '%04d' "${trial}")"
  python3 "${PROJECT_ROOT}/experiments/robotwin/check_episode_complete.py" \
    "${episode_dir}" || return 1
  if [[ "${mode_tag}" != "policy" ]]; then
    policy_current="${raw_output_dir}/${task_name}/imagination_transitions/${task_name}/policy/episode_$(printf '%04d' "${trial}")/replan_0000/current.png"
    cmp -s "${episode_dir}/replan_0000/current.png" "${policy_current}" || return 1
  fi
}

for index in "${!modes[@]}"; do
  mode="${modes[$index]}"
  hold_probability="${hold_probabilities[$index]}"
  delay_steps="${gripper_delay_steps[$index]}"
  mode_tag="${mode_tags[$index]}"
  trial_offset=0
  while (( trial_offset < EPISODES )); do
    remaining="$(( EPISODES - trial_offset ))"
    batch_episodes="${MAX_EPISODES_PER_PROCESS}"
    (( batch_episodes > remaining )) && batch_episodes="${remaining}"
    batch_complete=true
    for (( trial=trial_offset; trial<trial_offset+batch_episodes; trial++ )); do
      episode_is_complete "${mode_tag}" "${trial}" || batch_complete=false
    done
    if [[ "${batch_complete}" == true ]]; then
      printf '[robotwin-controlled] skip complete mode=%s trials=%d..%d\n' \
        "${mode_tag}" "${trial_offset}" "$(( trial_offset + batch_episodes - 1 ))"
    else
      environment_start_seed="$(( 100000 * (1 + BASE_SEED) + trial_offset ))"
      printf '[robotwin-controlled] mode=%s trials=%d..%d env_seed=%d\n' \
        "${mode_tag}" "${trial_offset}" \
        "$(( trial_offset + batch_episodes - 1 ))" "${environment_start_seed}"
      env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
        conda run -n "${CONDA_ENV}" python -u \
        "${PROJECT_ROOT}/experiments/robotwin/eval_robotwin_single.py" \
        "ckpt=${CHECKPOINT}" \
        "seed=${BASE_SEED}" \
        "gpu_id=${GPU_ID}" \
        "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}" \
        "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
        "EVALUATION.task_name=${task_name}" \
        EVALUATION.task_config=demo_clean \
        "EVALUATION.eval_num_episodes=${batch_episodes}" \
        "EVALUATION.trial_offset=${trial_offset}" \
        "EVALUATION.environment_start_seed=${environment_start_seed}" \
        "EVALUATION.environment_episode_offset=${trial_offset}" \
        "EVALUATION.num_inference_steps=${INFERENCE_STEPS}" \
        EVALUATION.replan_steps=24 \
        "EVALUATION.action_mode=${mode}" \
        EVALUATION.action_noise_std=0.0 \
        "EVALUATION.action_hold_probability=${hold_probability}" \
        "EVALUATION.gripper_close_delay_steps=${delay_steps}" \
        "EVALUATION.action_corruption_seed=${ACTION_CORRUPTION_SEED}" \
        "EVALUATION.fixed_instruction=${instruction}" \
        EVALUATION.save_imagination_transitions=true \
        "EVALUATION.output_dir=${output_dir}"
    fi
    trial_offset="$(( trial_offset + batch_episodes ))"
  done
done

printf 'Raw collection complete: %s\n' "${raw_output_dir}"
env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
  conda run -n "${CONDA_ENV}" python \
  "${PROJECT_ROOT}/experiments/robotwin/analyze_imagination_rewards.py" \
  --input-dir "${raw_output_dir}" \
  --encoder-path "${SIGLIP_PATH}" \
  --output-dir "${raw_output_dir}/reward_audit" \
  --device cuda \
  --batch-size 12 \
  --minimum-paired-trials "${EPISODES}"
printf 'Reward audit complete: %s\n' "${raw_output_dir}/reward_audit/reward_audit_summary.json"
