#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${ROBOMIMIC_WORKSPACE_ROOT:-$(dirname "${PROJECT_ROOT}")}"
PYTHON_BIN="${ROBOMIMIC_PYTHON:-/home/ubuntu/miniconda3/envs/dexmimicgen/bin/python}"
ROBOMIMIC_SOURCE="${ROBOMIMIC_SOURCE:-${WORKSPACE_ROOT}/robomimic-upstream}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-${WORKSPACE_ROOT}/robosuite-v1.5.1-src}"
TRAIN_ROOT="${ROBOMIMIC_BC_RNN_TRAIN_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_base/train_seed20260821}"
SEED="${ROBOMIMIC_TERMINAL_HARD_SEED:-20260826}"
EPISODES="${ROBOMIMIC_TERMINAL_HARD_EPISODES:-20}"
OUTPUT_ROOT="${ROBOMIMIC_TERMINAL_HARD_OUTPUT:-${PROJECT_ROOT}/evaluate_results/robomimic_terminal_hard_negative_seed${SEED}}"
ACTOR_DATASET="${ROBOMIMIC_TERMINAL_ACTOR_DATASET:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_q_holdout/seed20260825/deployable/frozen_train_new_valid_actor.npz}"
Q_ROOT="${ROBOMIMIC_TERMINAL_Q_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_residual_posttrain/deployable_q_pca16_action_prior}"
ACTOR_ROOT="${ROBOMIMIC_TERMINAL_ACTOR_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_bc_rnn_residual_posttrain/residual_actor_pca16_action_prior}"
ACTOR_SEEDS_CSV="${ROBOMIMIC_TERMINAL_HARD_ACTOR_SEEDS:-20260820,20260821,20260822}"
ROLLOUTS="${OUTPUT_ROOT}/rollouts.hdf5"

if [[ -z "${ROBOMIMIC_BC_RNN_CHECKPOINT:-}" ]]; then
    CHECKPOINT="$(find "${TRAIN_ROOT}" -path '*/models/model_epoch_275_*.pth' -print -quit)"
else
    CHECKPOINT="${ROBOMIMIC_BC_RNN_CHECKPOINT}"
fi
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
    echo "BC-RNN checkpoint not found" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}" /tmp/fastwam_numba_cache
export PYTHONPATH="${ROBOMIMIC_SOURCE}:${ROBOSUITE_SOURCE}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/fastwam_numba_cache}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/collect_can_bc_rnn_rollouts.py" \
    --checkpoint "${CHECKPOINT}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --horizon 400 \
    --valid-every 5 \
    --branch-warmup 10 \
    --branch-horizon 20 \
    --branch-stride 5 \
    --output "${ROLLOUTS}" \
    --state-index "${OUTPUT_ROOT}/state_index.npz" \
    --summary-json "${OUTPUT_ROOT}/rollouts.summary.json"

mapfile -t DEMOS < <(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/select_can_rollout_demos.py" \
        --dataset "${ROLLOUTS}" \
        --require-success
)
if [[ "${#DEMOS[@]}" -eq 0 ]]; then
    echo "No successful baseline demos were collected" >&2
    exit 1
fi

Q_ARGS=()
for Q_SEED in 20260820 20260821 20260822; do
    Q_ARGS+=(--q-checkpoint "${Q_ROOT}/full_state_seed${Q_SEED}/checkpoint.pt")
done

AUDIT_ARGS=()
IFS=',' read -r -a ACTOR_SEEDS <<< "${ACTOR_SEEDS_CSV}"
for ACTOR_SEED in "${ACTOR_SEEDS[@]}"; do
    ACTOR_OUTPUT="${OUTPUT_ROOT}/actor_seed${ACTOR_SEED}"
    mkdir -p "${ACTOR_OUTPUT}"
    for DEMO in "${DEMOS[@]}"; do
        OUTPUT_JSON="${ACTOR_OUTPUT}/${DEMO}.json"
        if [[ -f "${OUTPUT_JSON}" ]] && "${PYTHON_BIN}" -c \
            'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("complete") else 1)' \
            "${OUTPUT_JSON}"
        then
            echo "Skipping complete actor_seed${ACTOR_SEED}/${DEMO}"
        else
            "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/audit_can_terminal_intervention_tails.py" \
                --dataset "${ROLLOUTS}" \
                --demo "${DEMO}" \
                --actor-dataset "${ACTOR_DATASET}" \
                --actor-checkpoint "${ACTOR_ROOT}/actor_seed${ACTOR_SEED}/checkpoint.pt" \
                "${Q_ARGS[@]}" \
                --output-json "${OUTPUT_JSON}" \
                2>&1 | tee "${ACTOR_OUTPUT}/${DEMO}.log"
        fi
        AUDIT_ARGS+=(--audit "${OUTPUT_JSON}")
    done
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robomimic/prepare_can_terminal_safety_dataset.py" \
    "${AUDIT_ARGS[@]}" \
    --source-dataset "${ROLLOUTS}" \
    --output "${OUTPUT_ROOT}/terminal_hard_negative_pairwise.npz" \
    --report-json "${OUTPUT_ROOT}/terminal_hard_negative_pairwise.summary.json"
