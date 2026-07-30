#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-robotwin_low2_online_expert50_strict_pair_5ep_20260730}"
RUN_PREFIX="${RUN_PREFIX:-robotwin_low2_online_expert50_strict_seed_matrix_5ep_20260730}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
SUMMARY_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_PREFIX}"
RL_ROOT="${RL_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_low2_online_augmented_iql_20260730}"
NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_CHECKPOINT:-${RL_ROOT}/iql_online_expert50_no_imagination/checkpoint.pt}"
IMAGINATION_CHECKPOINT="${IMAGINATION_CHECKPOINT:-${RL_ROOT}/iql_online_expert50_imagination/checkpoint.pt}"
RESIDUAL_SUPPORT_INDEX_PATH="${RESIDUAL_SUPPORT_INDEX_PATH:-${RL_ROOT}/support_index_imagination_q95_local}"

blocks_seeds=(4300000 4300001 4300002 4300003 4300004)
hanging_seeds=(4300002 4300003 4300006 4300008 4300010)

run_task_variant() {
  local task="$1"
  local variant="$2"
  shift 2
  local seeds=("$@")
  local episode_index seed segment_name
  for episode_index in "${!seeds[@]}"; do
    seed="${seeds[$episode_index]}"
    segment_name="${RUN_PREFIX}__${task}_episode${episode_index}"
    env \
      RUN_NAME="${segment_name}" \
      VARIANTS="${variant}" \
      TASKS="${task}" \
      EPISODES=1 \
      BASE_SEED=42 \
      TRIAL_OFFSET="${episode_index}" \
      ENVIRONMENT_START_SEED="${seed}" \
      NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_CHECKPOINT}" \
      IMAGINATION_CHECKPOINT="${IMAGINATION_CHECKPOINT}" \
      RESIDUAL_SUPPORT_INDEX_PATH="${RESIDUAL_SUPPORT_INDEX_PATH}" \
      RESIDUAL_Q_GATE_ENABLED=true \
      RESIDUAL_Q_GATE_MARGIN=0.0 \
      RESIDUAL_Q_GATE_MAX_DISAGREEMENT=0.05 \
      RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=2 \
      SAVE_RESIDUAL_TRANSITIONS=true \
      SAVE_BASELINE_TRANSITIONS=false \
      bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
  done
}

for variant in no_imagination imagination; do
  run_task_variant blocks_ranking_size "${variant}" "${blocks_seeds[@]}"
  run_task_variant hanging_mug "${variant}" "${hanging_seeds[@]}"
done

mkdir -p "${SUMMARY_DIR}"
conda run --no-capture-output -n robotwin_fastwam python -u \
  "${PROJECT_ROOT}/experiments/robotwin/summarize_residual_iql_seed_matrix.py" \
  --result-base "${RESULT_BASE}" \
  --baseline-run-name "${BASELINE_RUN_NAME}" \
  --segment-run-prefix "${RUN_PREFIX}" \
  --output-json "${SUMMARY_DIR}/summary.json"

printf 'RoboTwin strict seed matrix complete: %s\n' "${SUMMARY_DIR}/summary.json"
