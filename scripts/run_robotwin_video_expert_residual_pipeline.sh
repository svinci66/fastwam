#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
PHASE="${PHASE:-feature-smoke}"
TASKS="${TASKS:-open_microwave,hanging_mug,place_can_basket}"
SEED="${SEED:-42}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/home/ubuntu/sj/fastwam/checkpoints}"
REWARD_JSON="${REWARD_JSON:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_wan_head_multitask4_smoke2_20260901_wan_vae_head_reward/wan_vae_pair_rewards.json}"
FEATURE_VERSION="fastwam_video_expert_final_token_mean_l2_v1"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_awr_seed${SEED}_20260903}"
BACKFILL_DIR="${RUN_ROOT}/feature_backfill"
REPLAY_DIR="${RUN_ROOT}/replay"
CONTROL_DIR="${RUN_ROOT}/training/seed${SEED}/no_imagination"
TREATMENT_DIR="${RUN_ROOT}/training/seed${SEED}/with_imagination"
CONTROL_CONFIG="${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_video_expert_multitask3_no_imagination_smoke.yaml"
TREATMENT_CONFIG="${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_video_expert_multitask3_imagination_smoke.yaml"

case "${PHASE}" in
  feature-smoke|train-smoke) ;;
  *) printf 'PHASE must be feature-smoke or train-smoke, got %s\n' "${PHASE}" >&2; exit 2 ;;
esac

for path in "${CHECKPOINT}" "${DATASET_STATS}" "${MODEL_BASE_PATH}" \
  "${REWARD_JSON}" "${CONTROL_CONFIG}" "${TREATMENT_CONFIG}"; do
  [[ -e "${path}" ]] || { printf '[video-expert-residual] missing: %s\n' "${path}" >&2; exit 1; }
done

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/driver.log") 2>&1

limit_args=()
if [[ "${PHASE}" == "feature-smoke" ]]; then
  limit_args=(--limit 1)
fi
conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/backfill_video_expert_features.py" \
  --reward-json "${REWARD_JSON}" --output-dir "${BACKFILL_DIR}" \
  --checkpoint "${CHECKPOINT}" --dataset-stats "${DATASET_STATS}" \
  --model-base-path "${MODEL_BASE_PATH}" --tasks "${TASKS}" \
  --device cuda --mixed-precision bf16 --num-inference-steps 10 \
  "${limit_args[@]}"

if [[ "${PHASE}" == "feature-smoke" ]]; then
  printf '[video-expert-residual] feature smoke complete: %s\n' "${BACKFILL_DIR}"
  exit 0
fi

if [[ ! -s "${REPLAY_DIR}/manifest.json" ]]; then
  [[ ! -e "${REPLAY_DIR}" ]] || {
    printf '[video-expert-residual] refusing incomplete replay: %s\n' "${REPLAY_DIR}" >&2
    exit 1
  }
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_wan_vae_head_awr_replay.py" \
    --reward-json "${REWARD_JSON}" --output-dir "${REPLAY_DIR}" \
    --actor-observation-source fastwam_video_expert \
    --observation-encoder-version "${FEATURE_VERSION}" \
    --reward-config "${TREATMENT_CONFIG}" --tasks "${TASKS}" \
    --minimum-pairwise-accuracy 0.90
fi

declare -A configs=(
  [no_imagination]="${CONTROL_CONFIG}"
  [with_imagination]="${TREATMENT_CONFIG}"
)
declare -A outputs=(
  [no_imagination]="${CONTROL_DIR}"
  [with_imagination]="${TREATMENT_DIR}"
)
for variant in no_imagination with_imagination; do
  output_dir="${outputs[${variant}]}"
  if [[ -s "${output_dir}/checkpoint.pt" && -s "${output_dir}/history.json" ]]; then
    printf '[video-expert-residual] skip complete variant=%s\n' "${variant}"
    continue
  fi
  [[ ! -e "${output_dir}" ]] || {
    printf '[video-expert-residual] refusing incomplete output: %s\n' "${output_dir}" >&2
    exit 1
  }
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
    --config "${configs[${variant}]}" --replay-dir "${REPLAY_DIR}" \
    --output-dir "${output_dir}" --seed "${SEED}" \
    --timeout-bootstrap-value 0.0
done

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_awr_training_pair.py" \
  --output-root "${RUN_ROOT}/training" --seeds "${SEED}" \
  --output-json "${RUN_ROOT}/training/paired_training_audit.json"

touch "${RUN_ROOT}/TRAIN_SMOKE_COMPLETE"
printf '[video-expert-residual] train smoke complete: %s\n' "${RUN_ROOT}"
