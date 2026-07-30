#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-robotwin_blocks_seed4300003_replan_counterfactual_20260730}"
TASK="${TASK:-blocks_ranking_size}"
BASE_SEED="${BASE_SEED:-42}"
TRIAL_OFFSET="${TRIAL_OFFSET:-3}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
SUMMARY_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}"
PILOT_ROOT="${PILOT_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_5task_residual_iql_pilot_20260729}"
IMAGINATION_CHECKPOINT="${IMAGINATION_CHECKPOINT:-${PILOT_ROOT}/iql_5task_imagination/checkpoint.pt}"
NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_CHECKPOINT:-${PILOT_ROOT}/iql_5task_no_imagination/checkpoint.pt}"
SUPPORT_INDEX_PATH="${SUPPORT_INDEX_PATH:-${PILOT_ROOT}/support_index_imagination_q95_local}"

if [[ "${TASK}" != "blocks_ranking_size" ]]; then
  printf 'This counterfactual is pre-registered for blocks_ranking_size, got %s\n' \
    "${TASK}" >&2
  exit 1
fi

run_condition() {
  local label="$1"
  local shadow_mode="$2"
  local intervention_replans="$3"
  printf '[robotwin-counterfactual] condition=%s shadow=%s replans=%s\n' \
    "${label}" "${shadow_mode}" "${intervention_replans}"
  env \
    RUN_NAME="${RUN_NAME}_${label}" \
    VARIANTS=imagination \
    TASKS="${TASK}" \
    EPISODES=1 \
    BASE_SEED="${BASE_SEED}" \
    TRIAL_OFFSET="${TRIAL_OFFSET}" \
    TILED=false \
    CAPTURE_DECODE_TILED=true \
    RESIDUAL_DEVICE=cpu \
    NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_CHECKPOINT}" \
    IMAGINATION_CHECKPOINT="${IMAGINATION_CHECKPOINT}" \
    RESIDUAL_Q_GATE_ENABLED=true \
    RESIDUAL_Q_GATE_MARGIN=0.0 \
    RESIDUAL_Q_GATE_MAX_DISAGREEMENT=0.05 \
    RESIDUAL_Q_GATE_CRITIC_SOURCE=target \
    RESIDUAL_SUPPORT_INDEX_PATH="${SUPPORT_INDEX_PATH}" \
    RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=true \
    RESIDUAL_SHADOW_MODE="${shadow_mode}" \
    RESIDUAL_INTERVENTION_REPLANS="${intervention_replans}" \
    RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=2 \
    SAVE_RESIDUAL_TRANSITIONS=true \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
}

run_condition shadow true all
run_condition replan9 false 9
run_condition replan11 false 11
run_condition replan9_11 false 9,11

mkdir -p "${SUMMARY_DIR}"
conda run --no-capture-output -n "${CONDA_ENV:-robotwin_fastwam}" python \
  "${PROJECT_ROOT}/experiments/robotwin/summarize_residual_counterfactual.py" \
  --result-base "${RESULT_BASE}" \
  --run-name "${RUN_NAME}" \
  --task "${TASK}" \
  --expected-seed 4300003 \
  --output-json "${SUMMARY_DIR}/summary.json"

printf 'RoboTwin counterfactual complete: %s\n' "${SUMMARY_DIR}/summary.json"
