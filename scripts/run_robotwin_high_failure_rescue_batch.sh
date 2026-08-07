#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BATCH_NAME="${BATCH_NAME:-robotwin_high_failure_rescue_batch_20260806}"
MAX_CANDIDATES_PER_SEED="${MAX_CANDIDATES_PER_SEED:-5}"
RESUME="${RESUME:-false}"
PLACE_SEEDS="${PLACE_SEEDS:-4800000,4800004}"
OPEN_SEEDS="${OPEN_SEEDS:-4800000}"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
ENCODER_PATH="${ENCODER_PATH:-/home/ubuntu/sj/fastwam/checkpoints/siglip-so400m-patch14-384-modelscope}"
OUTPUT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin_residual_pairs"
DRIVER_ROOT="${OUTPUT_BASE}/${BATCH_NAME}"
PLACE_RUN="${BATCH_NAME}_place_can_basket"
OPEN_RUN="${BATCH_NAME}_open_microwave"

mkdir -p "${DRIVER_ROOT}"

printf '[high-failure-batch] stage=place_can_basket seeds=%s\n' "${PLACE_SEEDS}"
env \
  RUN_NAME="${PLACE_RUN}" \
  TASK=place_can_basket \
  SEEDS="${PLACE_SEEDS}" \
  MAX_CANDIDATES_PER_SEED="${MAX_CANDIDATES_PER_SEED}" \
  RESUME="${RESUME}" \
  SEED_MANIFEST_TAG=20260806 \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_stratified_single_intervention_collection.sh" \
  >> "${DRIVER_ROOT}/place_can_basket.log" 2>&1

printf '[high-failure-batch] stage=open_microwave seeds=%s\n' "${OPEN_SEEDS}"
env \
  RUN_NAME="${OPEN_RUN}" \
  TASK=open_microwave \
  SEEDS="${OPEN_SEEDS}" \
  MAX_CANDIDATES_PER_SEED="${MAX_CANDIDATES_PER_SEED}" \
  RESUME="${RESUME}" \
  SEED_MANIFEST_TAG=20260806 \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_stratified_single_intervention_collection.sh" \
  >> "${DRIVER_ROOT}/open_microwave.log" 2>&1

for run_name in "${PLACE_RUN}" "${OPEN_RUN}"; do
  printf '[high-failure-batch] stage=score run=%s\n' "${run_name}"
  conda run --no-capture-output -n "${CONDA_ENV}" python \
    "${PROJECT_ROOT}/experiments/robotwin/score_single_intervention_pairs.py" \
    --pair-root "${OUTPUT_BASE}/${run_name}" \
    --encoder-path "${ENCODER_PATH}" \
    --device cuda \
    --encoder-dtype bf16 \
    --batch-size 24 \
    --q-margin 0.003 \
    --output-dir "${OUTPUT_BASE}/${run_name}/statistics" \
    > "${DRIVER_ROOT}/${run_name}_score.log" 2>&1
done

printf '[high-failure-batch] complete output=%s\n' "${DRIVER_ROOT}"
