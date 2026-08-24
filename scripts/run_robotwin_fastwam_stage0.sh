#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/robotwin_fastwam_stage0_20260824"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/driver.log") 2>&1

printf '[stage0] starting 4-task x 1-seed zero-residual equivalence smoke\n'
env \
  RUN_NAME=robotwin_fastwam_zero_residual_equivalence_4task1ep_20260824 \
  EPISODES=1 \
  SEED_MANIFEST_PATH="${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_4task_heldout1_expert_seeds_20260804.json" \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_fastwam_zero_residual_equivalence.sh"

printf '[stage0] smoke passed; starting 4-task x 5-seed hard equivalence audit\n'
env \
  RUN_NAME=robotwin_fastwam_zero_residual_equivalence_4task5ep_20260824 \
  EPISODES=5 \
  SEED_MANIFEST_PATH="${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_4task_heldout5_expert_seeds_20260804.json" \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_fastwam_zero_residual_equivalence.sh"

printf '[stage0] complete: smoke and full audits passed\n'
