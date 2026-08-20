#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${ROBOMIMIC_WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
ROBOMIMIC_SOURCE="${ROBOMIMIC_SOURCE:-${WORKSPACE_ROOT}/robomimic-upstream}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-${WORKSPACE_ROOT}/robosuite-v1.5.1-src}"
SEED="${ROBOMIMIC_SYMMETRIC_SEED:-20260820}"
TRAIN_STATES="${ROBOMIMIC_SYMMETRIC_TRAIN_STATES:-1200}"
VALID_STATES="${ROBOMIMIC_SYMMETRIC_VALID_STATES:-300}"

COLLECTION_ROOT="${ROBOMIMIC_SYMMETRIC_OUTPUT_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_symmetric_collection_expanded}"
COLLECTION="${COLLECTION_ROOT}/can_symmetric_expanded_${TRAIN_STATES}train_${VALID_STATES}valid_seed${SEED}.hdf5"
SYMMETRIC_ROOT="${ROBOMIMIC_SYMMETRIC_SOURCE_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_symmetric_gradient_expanded}"
OBS_ROOT="${ROBOMIMIC_DEPLOYABLE_OBS_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_deployable_observations_expanded}"
Q_ROOT="${ROBOMIMIC_DEPLOYABLE_Q_OUTPUT:-${PROJECT_ROOT}/evaluate_results/robomimic_deployable_q_expanded_posttrain}"

mkdir -p "${COLLECTION_ROOT}" "${SYMMETRIC_ROOT}" "${OBS_ROOT}" "${Q_ROOT}" /tmp/fastwam-matplotlib
export PYTHONPATH="${ROBOMIMIC_SOURCE}:${ROBOSUITE_SOURCE}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fastwam-matplotlib}"

echo "[stage 1/4] collect ${TRAIN_STATES} train and ${VALID_STATES} validation states"
ROBOMIMIC_SYMMETRIC_OUTPUT_ROOT="${COLLECTION_ROOT}" \
ROBOMIMIC_SYMMETRIC_TRAIN_STATES="${TRAIN_STATES}" \
ROBOMIMIC_SYMMETRIC_VALID_STATES="${VALID_STATES}" \
    bash "${PROJECT_ROOT}/scripts/run_robomimic_symmetric_collection.sh" expanded

echo "[stage 2/4] prepare trajectory-split symmetric datasets"
"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/prepare_can_symmetric_training_datasets.py" \
    --collection "${COLLECTION}" \
    --q-output "${SYMMETRIC_ROOT}/can_symmetric_pairwise_q.npz" \
    --actor-output "${SYMMETRIC_ROOT}/can_symmetric_residual_actor.npz" \
    --report-json "${SYMMETRIC_ROOT}/preparation_report.json" \
    --residual-scale 0.1 \
    --actor-target-mode symmetric_gradient

echo "[stage 3/4] replay deployable wrist RGB and proprioception"
"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/collect_can_deployable_observations.py" \
    --collection "${COLLECTION}" \
    --output "${OBS_ROOT}/can_wrist384_proprio.hdf5" \
    --summary-json "${OBS_ROOT}/can_wrist384_proprio.summary.json"

echo "[stage 4/4] encode frozen SigLIP and run the unchanged three-seed Q gate"
ROBOMIMIC_SYMMETRIC_SOURCE_ROOT="${SYMMETRIC_ROOT}" \
ROBOMIMIC_DEPLOYABLE_OBS_ROOT="${OBS_ROOT}" \
ROBOMIMIC_DEPLOYABLE_Q_OUTPUT="${Q_ROOT}" \
ROBOMIMIC_VISION_PCA_DIM= \
    bash "${PROJECT_ROOT}/scripts/run_robomimic_deployable_q_posttrain.sh"
