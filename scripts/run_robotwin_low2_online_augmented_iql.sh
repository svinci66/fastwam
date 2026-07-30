#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
RUN_NAME="${RUN_NAME:-robotwin_low2_online_augmented_iql_20260730}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/${RUN_NAME}}"
BASE_ROOT="${BASE_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_5task_residual_iql_pilot_20260729}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/robotwin_official_success/extracted}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/home/ubuntu/sj/fastwam/checkpoints}"
SIGLIP_PATH="${SIGLIP_PATH:-/home/ubuntu/sj/fastwam/checkpoints/siglip-so400m-patch14-384-modelscope}"
REWARD_ENCODER_VERSION="${REWARD_ENCODER_VERSION:-siglip-so400m-patch14-384-local-20260729}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
EPOCHS="${EPOCHS:-20}"

expert_dir="${RUN_ROOT}/expert_transitions_low2_50"
expert_increment="${RUN_ROOT}/expert_increment_replay_5to49"
online_increment="${RUN_ROOT}/online_increment_replay"
base_replay="${BASE_ROOT}/replay_5task"
merged_replay="${RUN_ROOT}/replay_5task_online_expert50"
normalization_manifest="${base_replay}/manifest.json"

for path in "${CHECKPOINT}" "${DATASET_STATS}" "${normalization_manifest}" \
  "${online_increment}/manifest.json"; do
  [[ -f "${path}" ]] || { printf 'Missing required file: %s\n' "${path}" >&2; exit 1; }
done
for path in "${DATASET_ROOT}" "${MODEL_BASE_PATH}" "${SIGLIP_PATH}"; do
  [[ -d "${path}" ]] || { printf 'Missing required directory: %s\n' "${path}" >&2; exit 1; }
done
[[ "${EPISODES_PER_TASK}" == "50" ]] || {
  printf 'This incremental recipe expects exactly the clean_50 expert set.\n' >&2
  exit 1
}

expert_export_complete() {
  local summary="${expert_dir}/expert_export_summary.json"
  [[ -f "${summary}" ]] || return 1
  conda run -n "${CONDA_ENV}" python -c \
    'import json,sys; p=json.load(open(sys.argv[1])); expected={"blocks_ranking_size","hanging_mug"}; assert set(p["tasks"])==expected; assert all(p["tasks"][task]["episodes"]==50 for task in expected)' \
    "${summary}" >/dev/null 2>&1
}

mkdir -p "${RUN_ROOT}"
if ! expert_export_complete; then
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/export_expert_imagination_transitions.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${expert_dir}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-stats "${DATASET_STATS}" \
    --model-base-path "${MODEL_BASE_PATH}" \
    --tasks blocks_ranking_size,hanging_mug \
    --episodes-per-task 50 \
    --replan-steps 24 \
    --action-horizon 32 \
    --num-inference-steps 4 \
    --seed 42 \
    --device cuda \
    --mixed-precision bf16
fi

# Episodes 0..4 already exist in the base replay.  Encoding only 5..49 avoids
# silently oversampling the first five demonstrations during the merge.
if [[ ! -f "${expert_increment}/manifest.json" ]]; then
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_residual_rl_replay.py" \
    --input-dir "${expert_dir}" \
    --output-dir "${expert_increment}" \
    --encoder-path "${SIGLIP_PATH}" \
    --reward-encoder-version "${REWARD_ENCODER_VERSION}" \
    --reward-config "${PROJECT_ROOT}/configs/rl/robotwin_residual_iql_smoke.yaml" \
    --camera-normalization-manifest "${normalization_manifest}" \
    --min-trial-index 5 \
    --max-trial-index 49 \
    --device cuda \
    --batch-size 24
fi

if [[ ! -f "${merged_replay}/manifest.json" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/merge_residual_replays.py" \
    --input-replay "${base_replay}" \
    --input-replay "${online_increment}" \
    --input-replay "${expert_increment}" \
    --output-dir "${merged_replay}"
fi

for variant in no_imagination imagination; do
  if [[ "${variant}" == "imagination" ]]; then
    config="${PROJECT_ROOT}/configs/rl/robotwin_residual_iql_smoke.yaml"
  else
    config="${PROJECT_ROOT}/configs/rl/robotwin_residual_iql_smoke_no_imagination.yaml"
  fi
  output_dir="${RUN_ROOT}/iql_online_expert50_${variant}"
  if [[ ! -f "${output_dir}/checkpoint.pt" ]]; then
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
      conda run --no-capture-output -n "${CONDA_ENV}" python -u \
      "${PROJECT_ROOT}/scripts/train_libero_residual_iql.py" \
      --config "${config}" \
      --replay-dir "${merged_replay}" \
      --output-dir "${output_dir}" \
      --device cuda \
      --epochs "${EPOCHS}" \
      --seed 42
  fi
done

audit_path="${RUN_ROOT}/offline_audit_online_expert50.json"
if [[ ! -f "${audit_path}" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/analyze_residual_iql_smoke.py" \
    --replay-dir "${merged_replay}" \
    --no-imagination-checkpoint "${RUN_ROOT}/iql_online_expert50_no_imagination/checkpoint.pt" \
    --imagination-checkpoint "${RUN_ROOT}/iql_online_expert50_imagination/checkpoint.pt" \
    --output-json "${audit_path}" \
    --device cuda
fi

support_dir="${RUN_ROOT}/support_index_imagination_q95_local"
if [[ ! -f "${support_dir}/metadata.json" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_residual_support_index.py" \
    --replay-dir "${merged_replay}" \
    --checkpoint "${RUN_ROOT}/iql_online_expert50_imagination/checkpoint.pt" \
    --output-dir "${support_dir}" \
    --calibration-fraction 0.25 \
    --quantile 0.95 \
    --neighbors 10 \
    --score-neighbors 3 \
    --language-similarity-threshold 0.99 \
    --seed 42 \
    --device cuda
fi

printf 'RoboTwin online-augmented matched IQL pipeline completed: %s\n' "${RUN_ROOT}"
