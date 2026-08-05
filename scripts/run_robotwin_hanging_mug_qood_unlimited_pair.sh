#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_4task_paired_v2_iql_20260803}"
RUN_NAME="${RUN_NAME:-${1:-robotwin_hanging_mug_qood_canonical_20epoch_unlimited_paired5_20260804}}"
EPISODES="${EPISODES:-5}"
RESIDUAL_Q_GATE_RISK_SCALE="${RESIDUAL_Q_GATE_RISK_SCALE:-${2:-0.0}}"
RESIDUAL_Q_GATE_RISK_DECAY="${RESIDUAL_Q_GATE_RISK_DECAY:-${3:-1.0}}"
VARIANTS_CSV="${VARIANTS:-baseline,imagination}"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_4task_heldout5_expert_seeds_20260804.json}"
SAVE_RESIDUAL_TRANSITIONS="${SAVE_RESIDUAL_TRANSITIONS:-false}"
SAVE_BASELINE_TRANSITIONS="${SAVE_BASELINE_TRANSITIONS:-false}"

# hanging_mug is the current medium-difficulty task: the matched baseline
# succeeds on roughly 40-60% of held-out episodes and Q+OOD has non-zero
# intervention coverage.  Keep all per-replan safety gates, but deliberately
# remove the episode-level intervention-count budget.
env \
  "TASKS=hanging_mug" \
  "EPISODES=${EPISODES}" \
  "BASE_SEED=47" \
  "ENVIRONMENT_START_SEED=4800000" \
  "TRIAL_OFFSET=0" \
  "RUN_NAME=${RUN_NAME}" \
  "VARIANTS=${VARIANTS_CSV}" \
  "INFERENCE_STEPS=10" \
  "REPLAN_STEPS=24" \
  "TEXT_CFG_SCALE=1.0" \
  "TASK_CONFIG=demo_clean" \
  "EVAL_VIDEO_LOG=false" \
  "INSTRUCTION_TYPE=unseen" \
  "INSTRUCTION_MODE=official" \
  "PAPER_ALIGNED=true" \
  "STRICT_PAIRED=true" \
  "SEED_MANIFEST_PATH=${SEED_MANIFEST_PATH}" \
  "IMAGINATION_CHECKPOINT=${RUN_ROOT}/iql_20epoch_imagination_canonical_probe/checkpoint.pt" \
  "NO_IMAGINATION_CHECKPOINT=${RUN_ROOT}/iql_5epoch_no_imagination/checkpoint.pt" \
  "RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-local-20260803" \
  "RESIDUAL_LANGUAGE_MODE=training_canonical" \
  "RESIDUAL_Q_GATE_ENABLED=true" \
  "RESIDUAL_Q_GATE_MARGIN=0.0" \
  "RESIDUAL_Q_GATE_MAX_DISAGREEMENT=0.05" \
  "RESIDUAL_Q_GATE_RISK_SCALE=${RESIDUAL_Q_GATE_RISK_SCALE}" \
  "RESIDUAL_Q_GATE_RISK_DECAY=${RESIDUAL_Q_GATE_RISK_DECAY}" \
  "RESIDUAL_Q_GATE_CRITIC_SOURCE=target" \
  "RESIDUAL_SUPPORT_INDEX_PATH=${RUN_ROOT}/support_index_imagination_20epoch_q95_probe" \
  "RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=true" \
  "RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none" \
  "SAVE_RESIDUAL_TRANSITIONS=${SAVE_RESIDUAL_TRANSITIONS}" \
  "SAVE_BASELINE_TRANSITIONS=${SAVE_BASELINE_TRANSITIONS}" \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
