#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-robotwin_video_expert_place_can_basket_fresh15_seed42_20260904}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${RUN_NAME}}"

env \
  COUNT=15 \
  START_SEED=4800140 \
  END_SEED=4800169 \
  POOL="${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_place_can_basket_fresh15_candidate_pool_20260904.json" \
  TRAIN_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_place_can_basket_single_task_epoch003_seed42_20260903/training/seed42" \
  ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  RUN_NAME="${RUN_NAME}" \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_video_expert_place_can_basket_heldout20.sh"
