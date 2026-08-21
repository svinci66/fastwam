#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
OBS_ROOT="${ROBOMIMIC_DEPLOYABLE_OBS_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_deployable_observations}"
SOURCE_ROOT="${ROBOMIMIC_SYMMETRIC_SOURCE_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_symmetric_gradient_posttrain}"
PCA_DIM="${ROBOMIMIC_VISION_PCA_DIM:-}"
if [[ -n "${PCA_DIM}" ]]; then
    DEFAULT_OUTPUT_ROOT="${PROJECT_ROOT}/evaluate_results/robomimic_deployable_q_pca${PCA_DIM}_posttrain"
else
    DEFAULT_OUTPUT_ROOT="${PROJECT_ROOT}/evaluate_results/robomimic_deployable_q_posttrain"
fi
OUTPUT_ROOT="${ROBOMIMIC_DEPLOYABLE_Q_OUTPUT:-${DEFAULT_OUTPUT_ROOT}}"
ENCODER_PATH="${ROBOMIMIC_SIGLIP_PATH:-/home/ubuntu/sj/fastwam/checkpoints/siglip-so400m-patch14-384-modelscope}"
SEEDS_CSV="${ROBOMIMIC_DEPLOYABLE_Q_SEEDS:-20260820,20260821,20260822}"
ACTION_PRIOR_INIT="${ROBOMIMIC_ACTION_PRIOR_INIT:-0}"
TEACHER_REGULARIZATION="${ROBOMIMIC_Q_TEACHER_REGULARIZATION:-1.0}"

OBS_HDF5="${OBS_ROOT}/can_wrist384_proprio.hdf5"
OBS_FEATURES="${OBS_ROOT}/can_wrist_siglip_proprio.npz"
Q_DATASET="${OUTPUT_ROOT}/can_symmetric_pairwise_q.npz"
ACTOR_DATASET="${OUTPUT_ROOT}/can_symmetric_residual_actor.npz"

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/encode_can_deployable_observations.py" \
    --observations "${OBS_HDF5}" \
    --encoder-path "${ENCODER_PATH}" \
    --output "${OBS_FEATURES}" \
    --report-json "${OBS_ROOT}/encoding_report.json"

PREPARE_EXTRA_ARGS=()
if [[ -n "${PCA_DIM}" ]]; then
    PREPARE_EXTRA_ARGS+=(
        --vision-pca-dim "${PCA_DIM}"
        --projection-output "${OUTPUT_ROOT}/vision_pca_projection.npz"
    )
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/prepare_can_deployable_training_datasets.py" \
    --q-source "${SOURCE_ROOT}/can_symmetric_pairwise_q.npz" \
    --actor-source "${SOURCE_ROOT}/can_symmetric_residual_actor.npz" \
    --observations "${OBS_FEATURES}" \
    --q-output "${Q_DATASET}" \
    --actor-output "${ACTOR_DATASET}" \
    --report-json "${OUTPUT_ROOT}/preparation_report.json" \
    "${PREPARE_EXTRA_ARGS[@]}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
for SEED in "${SEEDS[@]}"; do
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/train_can_pairwise_q.py" \
        --dataset "${Q_DATASET}" \
        --output-dir "${OUTPUT_ROOT}/action_only_seed${SEED}" \
        --state-mode action_only \
        --seed "${SEED}"
    FULL_EXTRA_ARGS=()
    if [[ "${ACTION_PRIOR_INIT}" == "1" ]]; then
        FULL_EXTRA_ARGS+=(
            --initialize-action-checkpoint "${OUTPUT_ROOT}/action_only_seed${SEED}/checkpoint.pt"
            --teacher-regularization "${TEACHER_REGULARIZATION}"
        )
    fi
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/train_can_pairwise_q.py" \
        --dataset "${Q_DATASET}" \
        --output-dir "${OUTPUT_ROOT}/full_state_seed${SEED}" \
        --state-mode full \
        --seed "${SEED}" \
        "${FULL_EXTRA_ARGS[@]}"
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/summarize_can_pairwise_q.py" \
    --output-root "${OUTPUT_ROOT}" \
    --output-json "${OUTPUT_ROOT}/q_comparison_multiseed.json"
