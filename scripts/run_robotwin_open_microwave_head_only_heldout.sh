#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-robotwin_open_microwave_head_only_heldout_10ep_seed19_20260827}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
START_CANDIDATE_SEED="${START_CANDIDATE_SEED:-19}"
MAX_CANDIDATE_SEED="${MAX_CANDIDATE_SEED:-60}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-30}"
OUTPUT_NAME="${OUTPUT_NAME:-${RUN_NAME}_wan_vae_head_reward}"

cd "${PROJECT_ROOT}"

env RUN_NAME="${RUN_NAME}" TASKS=open_microwave \
  EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
  START_CANDIDATE_SEED="${START_CANDIDATE_SEED}" \
  MAX_CANDIDATE_SEED="${MAX_CANDIDATE_SEED}" \
  COOLDOWN_SECONDS="${COOLDOWN_SECONDS}" \
  bash scripts/run_robotwin_local_expert_pair_smoke.sh

env SOURCE_RUN_NAME="${RUN_NAME}" OUTPUT_NAME="${OUTPUT_NAME}" \
  TASKS=open_microwave REWARD_CAMERAS=head \
  bash scripts/run_robotwin_natural_failure_vae_reward.sh

printf '[head-only-heldout] complete run=%s reward=%s\n' \
  "${RUN_NAME}" \
  "${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${OUTPUT_NAME}/wan_vae_pair_rewards.json"
