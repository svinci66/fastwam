#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${ROBOMIMIC_WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
ROBOMIMIC_SOURCE="${ROBOMIMIC_SOURCE:-${WORKSPACE_ROOT}/robomimic-upstream}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-${WORKSPACE_ROOT}/robosuite-v1.5.1-src}"
DATASET="${ROBOMIMIC_TERMINAL_SOURCE_DATASET:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_q_holdout/seed20260825/rollouts.hdf5}"
ACTOR_DATASET="${ROBOMIMIC_TERMINAL_ACTOR_DATASET:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_q_holdout/seed20260825/deployable/frozen_train_new_valid_actor.npz}"
Q_ROOT="${ROBOMIMIC_TERMINAL_Q_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_residual_posttrain/deployable_q_pca16_action_prior}"
ACTOR_ROOT="${ROBOMIMIC_TERMINAL_ACTOR_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_residual_posttrain/residual_actor_pca16_action_prior}"
ACTOR_SEED="${ROBOMIMIC_TERMINAL_ACTOR_SEED:-20260820}"
OUTPUT_ROOT="${ROBOMIMIC_TERMINAL_OUTPUT_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_terminal_safety_seed20260825}"
DEMOS_CSV="${ROBOMIMIC_TERMINAL_DEMOS:-demo_0,demo_1,demo_2,demo_3,demo_4,demo_5,demo_6,demo_7,demo_8,demo_9,demo_10,demo_11,demo_12,demo_13,demo_14,demo_15,demo_16,demo_17,demo_18,demo_19}"

mkdir -p "${OUTPUT_ROOT}" /tmp/fastwam_numba_cache
export PYTHONPATH="${ROBOMIMIC_SOURCE}:${ROBOSUITE_SOURCE}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/fastwam_numba_cache}"

Q_ARGS=()
for Q_SEED in 20260820 20260821 20260822; do
    Q_ARGS+=(--q-checkpoint "${Q_ROOT}/full_state_seed${Q_SEED}/checkpoint.pt")
done

IFS=',' read -r -a DEMOS <<< "${DEMOS_CSV}"
for DEMO in "${DEMOS[@]}"; do
    OUTPUT_JSON="${OUTPUT_ROOT}/${DEMO}.json"
    if [[ -f "${OUTPUT_JSON}" ]] && "${PYTHON_BIN}" -c \
        'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("complete") and d.get("rows") and "state" in d["rows"][0] else 1)' \
        "${OUTPUT_JSON}"
    then
        echo "Skipping complete ${DEMO}"
        continue
    fi
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/audit_can_terminal_intervention_tails.py" \
        --dataset "${DATASET}" \
        --demo "${DEMO}" \
        --actor-dataset "${ACTOR_DATASET}" \
        --actor-checkpoint "${ACTOR_ROOT}/actor_seed${ACTOR_SEED}/checkpoint.pt" \
        "${Q_ARGS[@]}" \
        --output-json "${OUTPUT_JSON}" \
        2>&1 | tee "${OUTPUT_ROOT}/${DEMO}.log"
done

AUDIT_ARGS=()
for DEMO in "${DEMOS[@]}"; do
    AUDIT_ARGS+=(--audit "${OUTPUT_ROOT}/${DEMO}.json")
done
"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/prepare_can_terminal_safety_dataset.py" \
    "${AUDIT_ARGS[@]}" \
    --source-dataset "${DATASET}" \
    --output "${OUTPUT_ROOT}/terminal_safety_pairwise.npz" \
    --report-json "${OUTPUT_ROOT}/terminal_safety_pairwise.summary.json"
