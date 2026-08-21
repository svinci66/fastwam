#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${ROBOMIMIC_WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
ROBOMIMIC_SOURCE="${ROBOMIMIC_SOURCE:-${WORKSPACE_ROOT}/robomimic-upstream}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-${WORKSPACE_ROOT}/robosuite-v1.5.1-src}"
DATASET="${ROBOMIMIC_PAIRED_DATASET:-${WORKSPACE_ROOT}/datasets/robomimic_hf/v1.5/can/paired/low_dim_v15.hdf5}"
OUTPUT="${ROBOMIMIC_EXPERT_REPLAY_AUDIT_OUTPUT:-${PROJECT_ROOT}/evaluate_results/robomimic_expert_replay_audit/valid20.json}"

mkdir -p "$(dirname "${OUTPUT}")" /tmp/fastwam_numba_cache
export PYTHONPATH="${ROBOMIMIC_SOURCE}:${ROBOSUITE_SOURCE}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/fastwam_numba_cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/audit_can_expert_episode_replay.py" \
    --dataset "${DATASET}" \
    --split valid \
    --output-json "${OUTPUT}"
