#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${ROBOMIMIC_WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"
DATASET="${ROBOMIMIC_PAIRED_DATASET:-${WORKSPACE_ROOT}/datasets/robomimic_hf/v1.5/can/paired/low_dim_v15.hdf5}"
ROBOMIMIC_SOURCE="${ROBOMIMIC_SOURCE:-${WORKSPACE_ROOT}/robomimic-upstream}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-${WORKSPACE_ROOT}/robosuite-v1.5.1-src}"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
OUTPUT_DIR="${ROBOMIMIC_PHASE1_OUTPUT:-${PROJECT_ROOT}/evaluate_results/robomimic_phase1}"
PAIR_INDEX="${ROBOMIMIC_PAIR_INDEX:-0}"

for required in "${DATASET}" "${ROBOMIMIC_SOURCE}/robomimic" "${ROBOSUITE_SOURCE}/robosuite" "${PYTHON_BIN}"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required path: ${required}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROBOMIMIC_SOURCE}:${ROBOSUITE_SOURCE}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fastwam-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" -c 'import robomimic, robosuite; assert robomimic.__version__ == "0.5.0"; assert robosuite.__version__ == "1.5.1"; print(f"robomimic={robomimic.__version__} robosuite={robosuite.__version__}")'
"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/audit_can_paired_dataset.py" \
    --dataset "${DATASET}" \
    --output-json "${OUTPUT_DIR}/can_paired_audit.json"
"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/validate_can_state_branching.py" \
    --dataset "${DATASET}" \
    --pair-index "${PAIR_INDEX}" \
    --output-json "${OUTPUT_DIR}/can_state_branching_pair_${PAIR_INDEX}.json"

echo "RoboMimic Phase-1 smoke passed. Reports: ${OUTPUT_DIR}"
