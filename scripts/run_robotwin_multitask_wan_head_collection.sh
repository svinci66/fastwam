#!/usr/bin/env bash
set -euo pipefail

# Formal four-task collection for testing whether the frozen Wan-head reward
# generalizes beyond open_microwave. This reuses the exact local-expert pairing
# collector and never selects trajectories by their reward value.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
TASKS="${TASKS:-open_microwave,hanging_mug,place_can_basket,blocks_ranking_size}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-20}"
MIN_FAILURES_PER_TASK="${MIN_FAILURES_PER_TASK:-8}"
START_CANDIDATE_SEED="${START_CANDIDATE_SEED:-100}"
MAX_CANDIDATE_SEED="${MAX_CANDIDATE_SEED:-500}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-30}"
GPU_ID="${GPU_ID:-0}"
RUN_NAME="${RUN_NAME:-robotwin_wan_head_multitask4_collection_20260901}"
REWARD_RUN_NAME="${REWARD_RUN_NAME:-${RUN_NAME}_wan_vae_head_reward}"
PROTOCOL_MANIFEST="${PROTOCOL_MANIFEST:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_wan_head_multitask4_collection_20260901.json}"
MIN_MACRO_PAIRWISE_ACCURACY="${MIN_MACRO_PAIRWISE_ACCURACY:-0.60}"
MIN_POSITIVE_MARGIN_TASKS="${MIN_POSITIVE_MARGIN_TASKS:-3}"
REQUIRE_TRAINING_READY="${REQUIRE_TRAINING_READY:-true}"

ARTIFACT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${RUN_NAME}"
REWARD_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${REWARD_RUN_NAME}"
SCREEN_SUMMARY="${ARTIFACT_DIR}/summary.json"
REWARD_JSON="${REWARD_DIR}/wan_vae_pair_rewards.json"
AUDIT_JSON="${ARTIFACT_DIR}/multitask_collection_audit.json"

[[ -s "${PROTOCOL_MANIFEST}" ]] || {
  printf 'Frozen protocol manifest missing: %s\n' "${PROTOCOL_MANIFEST}" >&2; exit 1;
}

[[ "${EPISODES_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'EPISODES_PER_TASK must be positive\n' >&2; exit 1;
}
[[ "${MIN_FAILURES_PER_TASK}" =~ ^[0-9]+$ ]] || {
  printf 'MIN_FAILURES_PER_TASK must be non-negative\n' >&2; exit 1;
}
(( MIN_FAILURES_PER_TASK <= EPISODES_PER_TASK )) || {
  printf 'MIN_FAILURES_PER_TASK cannot exceed EPISODES_PER_TASK\n' >&2; exit 1;
}
case "${REQUIRE_TRAINING_READY}" in
  true|false) ;;
  *) printf 'REQUIRE_TRAINING_READY must be true or false\n' >&2; exit 1 ;;
esac

IFS=',' read -r -a task_values <<< "${TASKS}"
if (( ${#task_values[@]} != 4 )); then
  printf 'Formal collection requires exactly four tasks: %s\n' "${TASKS}" >&2
  exit 1
fi
declare -A seen_tasks=()
for task in "${task_values[@]}"; do
  task="$(printf '%s' "${task}" | xargs)"
  case "${task}" in
    open_microwave|hanging_mug|place_can_basket|blocks_ranking_size) ;;
    *) printf 'Task is outside the frozen four-task protocol: %s\n' "${task}" >&2; exit 1 ;;
  esac
  [[ -z "${seen_tasks[${task}]:-}" ]] || {
    printf 'Duplicate task: %s\n' "${task}" >&2; exit 1;
  }
  seen_tasks["${task}"]=1
done

printf '[multitask-wan-head] stage=collect tasks=%s episodes_per_task=%s minimum_failures=%s\n' \
  "${TASKS}" "${EPISODES_PER_TASK}" "${MIN_FAILURES_PER_TASK}"
env RUN_NAME="${RUN_NAME}" TASKS="${TASKS}" \
  EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
  START_CANDIDATE_SEED="${START_CANDIDATE_SEED}" \
  MAX_CANDIDATE_SEED="${MAX_CANDIDATE_SEED}" \
  COOLDOWN_SECONDS="${COOLDOWN_SECONDS}" GPU_ID="${GPU_ID}" \
  CONDA_ENV="${CONDA_ENV}" \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_local_expert_pair_smoke.sh"

[[ -s "${SCREEN_SUMMARY}" ]] || {
  printf 'Collection summary missing: %s\n' "${SCREEN_SUMMARY}" >&2; exit 1;
}

printf '[multitask-wan-head] stage=score reward_cameras=head\n'
env SOURCE_RUN_NAME="${RUN_NAME}" OUTPUT_NAME="${REWARD_RUN_NAME}" \
  TASKS="${TASKS}" REWARD_CAMERAS=head CONDA_ENV="${CONDA_ENV}" \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_natural_failure_vae_reward.sh"

[[ -s "${REWARD_JSON}" ]] || {
  printf 'Wan-head reward output missing: %s\n' "${REWARD_JSON}" >&2; exit 1;
}

audit_args=(
  --screen-summary "${SCREEN_SUMMARY}"
  --reward-json "${REWARD_JSON}"
  --tasks "${TASKS}"
  --episodes-per-task "${EPISODES_PER_TASK}"
  --minimum-failures-per-task "${MIN_FAILURES_PER_TASK}"
  --minimum-macro-pairwise-accuracy "${MIN_MACRO_PAIRWISE_ACCURACY}"
  --minimum-positive-margin-tasks "${MIN_POSITIVE_MARGIN_TASKS}"
  --output-json "${AUDIT_JSON}"
)
if [[ "${REQUIRE_TRAINING_READY}" == true ]]; then
  audit_args+=(--require-training-ready)
fi

printf '[multitask-wan-head] stage=audit output=%s\n' "${AUDIT_JSON}"
conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_multitask_wan_head_collection.py" \
  "${audit_args[@]}"

printf '[multitask-wan-head] complete audit=%s\n' "${AUDIT_JSON}"
