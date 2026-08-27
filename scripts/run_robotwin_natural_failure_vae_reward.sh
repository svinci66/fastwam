#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-robotwin_low_success_pair_screen_2task3ep_20260827}"
OUTPUT_NAME="${OUTPUT_NAME:-${SOURCE_RUN_NAME}_wan_vae_reward}"
TASKS="${TASKS:-}"
REWARD_CAMERAS="${REWARD_CAMERAS:-head,left_wrist,right_wrist}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/home/ubuntu/sj/fastwam/checkpoints}"
VAE_PATH="${VAE_PATH:-${MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors}"
ARTIFACT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${SOURCE_RUN_NAME}"
CASES_JSONL="${CASES_JSONL:-${ARTIFACT_DIR}/status.jsonl}"
FASTWAM_RUN_DIR="${FASTWAM_RUN_DIR:-${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384/${SOURCE_RUN_NAME}}"
OUTPUT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${OUTPUT_NAME}"
EXPERT_ROOT="${OUTPUT_DIR}/expert_imagination"
RESULT_JSON="${OUTPUT_DIR}/wan_vae_pair_rewards.json"

for path in "${CHECKPOINT}" "${DATASET_STATS}" "${VAE_PATH}" "${CASES_JSONL}"; do
  [[ -f "${path}" ]] || { printf 'Missing file: %s\n' "${path}" >&2; exit 1; }
done
[[ -d "${MODEL_BASE_PATH}" ]] || { printf 'Missing model base: %s\n' "${MODEL_BASE_PATH}" >&2; exit 1; }
[[ -d "${FASTWAM_RUN_DIR}" ]] || { printf 'Missing FastWAM run: %s\n' "${FASTWAM_RUN_DIR}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/driver.log") 2>&1

nvidia-modprobe -u -c=0
nvidia-smi -L

conda run --no-capture-output -n "${CONDA_ENV}" \
  env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  PYTHONUNBUFFERED=1 \
  python -u "${PROJECT_ROOT}/experiments/robotwin/export_paired_expert_imagination_trajectories.py" \
  --cases-jsonl "${CASES_JSONL}" --output-dir "${EXPERT_ROOT}" \
  --tasks "${TASKS}" \
  --checkpoint "${CHECKPOINT}" --dataset-stats "${DATASET_STATS}" \
  --model-base-path "${MODEL_BASE_PATH}" --num-inference-steps 10 \
  --replan-steps 24 --seed 47 --device cuda --mixed-precision bf16

conda run --no-capture-output -n "${CONDA_ENV}" \
  env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  PYTHONUNBUFFERED=1 \
  python -u "${PROJECT_ROOT}/experiments/robotwin/score_natural_failure_vae_pairs.py" \
  --cases-jsonl "${CASES_JSONL}" --expert-root "${EXPERT_ROOT}" \
  --tasks "${TASKS}" \
  --fastwam-run-dir "${FASTWAM_RUN_DIR}" --vae-path "${VAE_PATH}" \
  --reward-cameras "${REWARD_CAMERAS}" \
  --device cuda --dtype bf16 --output-json "${RESULT_JSON}"

printf '[natural-failure-vae] complete result=%s\n' "${RESULT_JSON}"
