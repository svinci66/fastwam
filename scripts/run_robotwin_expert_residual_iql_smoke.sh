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
CONTROLLED_ROOT="${CONTROLLED_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_controlled_corruption_3task5ep_20260729}"
RUN_NAME="${RUN_NAME:-robotwin_expert10_residual_iql_20260729}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/${RUN_NAME}}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
EPOCHS="${EPOCHS:-20}"
TASKS="${TASKS:-adjust_bottle,open_laptop,stack_blocks_two}"
REWARD_ENCODER_VERSION="${REWARD_ENCODER_VERSION:-siglip-so400m-patch14-384-local-20260729}"

mkdir -p "${DATASET_ROOT}" "${RUN_ROOT}"

for task in adjust_bottle open_laptop stack_blocks_two; do
  zip_path="${ZIP_ROOT}/${task}_aloha-agilex_clean_50.zip"
  marker="${DATASET_ROOT}/${task}/.extracted_ok"
  [[ -f "${zip_path}" ]] || { printf 'Missing expert zip: %s\n' "${zip_path}" >&2; exit 1; }
  if [[ ! -f "${marker}" ]]; then
    unzip -tq "${zip_path}" >/dev/null
    mkdir -p "${DATASET_ROOT}/${task}"
    unzip -q -o "${zip_path}" -d "${DATASET_ROOT}/${task}"
    touch "${marker}"
  fi
done

expert_dir="${RUN_ROOT}/expert_transitions"
if [[ ! -f "${expert_dir}/expert_export_summary.json" ]]; then
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/export_expert_imagination_transitions.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${expert_dir}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-stats "${DATASET_STATS}" \
    --model-base-path "${MODEL_BASE_PATH}" \
    --tasks "${TASKS}" \
    --episodes-per-task "${EPISODES_PER_TASK}" \
    --replan-steps 24 \
    --action-horizon 32 \
    --num-inference-steps 4 \
    --seed 42 \
    --device cuda \
    --mixed-precision bf16
fi

replay_dir="${RUN_ROOT}/replay"
if [[ ! -f "${replay_dir}/manifest.json" ]]; then
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_residual_rl_replay.py" \
    --input-dir "${CONTROLLED_ROOT}" \
    --input-dir "${expert_dir}" \
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
  output_dir="${RUN_ROOT}/iql_balanced_${variant}"
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

audit_path="${RUN_ROOT}/offline_audit_balanced.json"
if [[ ! -f "${audit_path}" ]]; then
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/analyze_residual_iql_smoke.py" \
    --replay-dir "${replay_dir}" \
    --no-imagination-checkpoint \
    "${RUN_ROOT}/iql_balanced_no_imagination/checkpoint.pt" \
    --imagination-checkpoint \
    "${RUN_ROOT}/iql_balanced_imagination/checkpoint.pt" \
    --output-json "${audit_path}" \
    --device cuda
fi

printf 'RoboTwin expert residual-IQL smoke completed: %s\n' "${RUN_ROOT}"
