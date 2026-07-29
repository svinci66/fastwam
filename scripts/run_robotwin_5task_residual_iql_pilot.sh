#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
SIGLIP_PATH="${SIGLIP_PATH:-/home/ubuntu/sj/fastwam/checkpoints/siglip-so400m-patch14-384-modelscope}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/home/ubuntu/sj/fastwam/checkpoints}"
ZIP_ROOT="${ZIP_ROOT:-${PROJECT_ROOT}/data/robotwin_official_success/zips}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/robotwin_official_success/extracted}"
PREVIOUS_ROOT="${PREVIOUS_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_expert10_residual_iql_20260729}"
HIGH_CONTROLLED_ROOT="${HIGH_CONTROLLED_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_controlled_corruption_3task5ep_20260729}"
LOW_CONTROLLED_ROOT="${LOW_CONTROLLED_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_reward_validation_3task5ep_paired_v2_20260728}"
RUN_NAME="${RUN_NAME:-robotwin_5task_residual_iql_pilot_20260729}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/${RUN_NAME}}"
LOW_TASKS="${LOW_TASKS:-blocks_ranking_size,hanging_mug}"
EPISODES_PER_LOW_TASK="${EPISODES_PER_LOW_TASK:-10}"
EPOCHS="${EPOCHS:-20}"
REWARD_ENCODER_VERSION="${REWARD_ENCODER_VERSION:-siglip-so400m-patch14-384-local-20260729}"

IFS=',' read -r -a low_tasks <<< "${LOW_TASKS}"
if ((${#low_tasks[@]} != 2)); then
  printf 'LOW_TASKS must contain exactly the two low-success pilot tasks\n' >&2
  exit 1
fi
for task in "${low_tasks[@]}"; do
  case "${task}" in
    blocks_ranking_size|hanging_mug) ;;
    *) printf 'Unsupported low-success task: %s\n' "${task}" >&2; exit 1 ;;
  esac
done
[[ "${EPISODES_PER_LOW_TASK}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'EPISODES_PER_LOW_TASK must be positive\n' >&2
  exit 1
}
[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'EPOCHS must be positive\n' >&2
  exit 1
}

for path in "${CHECKPOINT}" "${DATASET_STATS}"; do
  [[ -f "${path}" ]] || { printf 'Missing file: %s\n' "${path}" >&2; exit 1; }
done
for path in "${SIGLIP_PATH}" "${MODEL_BASE_PATH}" \
  "${PREVIOUS_ROOT}/expert_transitions" "${HIGH_CONTROLLED_ROOT}"; do
  [[ -d "${path}" ]] || { printf 'Missing directory: %s\n' "${path}" >&2; exit 1; }
done

mkdir -p "${DATASET_ROOT}" "${RUN_ROOT}"
for task in "${low_tasks[@]}"; do
  zip_path="${ZIP_ROOT}/${task}_aloha-agilex_clean_50.zip"
  marker="${DATASET_ROOT}/${task}/.extracted_ok"
  [[ -f "${zip_path}" ]] || { printf 'Missing expert zip: %s\n' "${zip_path}" >&2; exit 1; }
  [[ -d "${LOW_CONTROLLED_ROOT}/${task}" ]] || {
    printf 'Missing controlled transition directory: %s\n' \
      "${LOW_CONTROLLED_ROOT}/${task}" >&2
    exit 1
  }
  if [[ ! -f "${marker}" ]]; then
    unzip -tq "${zip_path}" >/dev/null
    mkdir -p "${DATASET_ROOT}/${task}"
    unzip -q -o "${zip_path}" -d "${DATASET_ROOT}/${task}"
    touch "${marker}"
  fi
done

expert_dir="${RUN_ROOT}/expert_transitions_low2"
if [[ ! -f "${expert_dir}/expert_export_summary.json" ]]; then
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/export_expert_imagination_transitions.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${expert_dir}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-stats "${DATASET_STATS}" \
    --model-base-path "${MODEL_BASE_PATH}" \
    --tasks "${LOW_TASKS}" \
    --episodes-per-task "${EPISODES_PER_LOW_TASK}" \
    --replan-steps 24 \
    --action-horizon 32 \
    --num-inference-steps 4 \
    --seed 42 \
    --device cuda \
    --mixed-precision bf16
fi

replay_dir="${RUN_ROOT}/replay_5task"
if [[ ! -f "${replay_dir}/manifest.json" ]]; then
  input_args=(
    --input-dir "${HIGH_CONTROLLED_ROOT}"
    --input-dir "${PREVIOUS_ROOT}/expert_transitions"
    --input-dir "${LOW_CONTROLLED_ROOT}/blocks_ranking_size"
    --input-dir "${LOW_CONTROLLED_ROOT}/hanging_mug"
    --input-dir "${expert_dir}"
  )
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_residual_rl_replay.py" \
    "${input_args[@]}" \
    --output-dir "${replay_dir}" \
    --encoder-path "${SIGLIP_PATH}" \
    --reward-encoder-version "${REWARD_ENCODER_VERSION}" \
    --reward-config "${PROJECT_ROOT}/configs/rl/robotwin_residual_iql_smoke.yaml" \
    --device cuda \
    --batch-size 12
fi

for variant in no_imagination imagination; do
  if [[ "${variant}" == "imagination" ]]; then
    config="${PROJECT_ROOT}/configs/rl/robotwin_residual_iql_smoke.yaml"
  else
    config="${PROJECT_ROOT}/configs/rl/robotwin_residual_iql_smoke_no_imagination.yaml"
  fi
  output_dir="${RUN_ROOT}/iql_5task_${variant}"
  if [[ ! -f "${output_dir}/checkpoint.pt" ]]; then
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
      conda run --no-capture-output -n "${CONDA_ENV}" python -u \
      "${PROJECT_ROOT}/scripts/train_libero_residual_iql.py" \
      --config "${config}" \
      --replay-dir "${replay_dir}" \
      --output-dir "${output_dir}" \
      --device cuda \
      --epochs "${EPOCHS}" \
      --seed 42
  fi
done

audit_path="${RUN_ROOT}/offline_audit_5task.json"
if [[ ! -f "${audit_path}" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/analyze_residual_iql_smoke.py" \
    --replay-dir "${replay_dir}" \
    --no-imagination-checkpoint "${RUN_ROOT}/iql_5task_no_imagination/checkpoint.pt" \
    --imagination-checkpoint "${RUN_ROOT}/iql_5task_imagination/checkpoint.pt" \
    --output-json "${audit_path}" \
    --device cuda
fi

support_dir="${RUN_ROOT}/support_index_imagination_q95_local"
if [[ ! -f "${support_dir}/metadata.json" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_residual_support_index.py" \
    --replay-dir "${replay_dir}" \
    --checkpoint "${RUN_ROOT}/iql_5task_imagination/checkpoint.pt" \
    --output-dir "${support_dir}" \
    --calibration-fraction 0.25 \
    --quantile 0.95 \
    --neighbors 10 \
    --score-neighbors 3 \
    --language-similarity-threshold 0.99 \
    --seed 42 \
    --device cuda
fi

printf 'RoboTwin five-task residual-IQL pilot completed: %s\n' "${RUN_ROOT}"
