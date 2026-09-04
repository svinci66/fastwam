#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_place_can_basket_single_task_epoch003_seed42_20260903"
EVAL_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_place_can_basket_fresh15_seed42_20260904"
RUN_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_place_can_basket_single_task_epoch003_seed43_20260904"
ONLINE_RUN_NAME="robotwin_video_expert_place_can_basket_fresh15_seed43_20260904"
SUMMARY="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${ONLINE_RUN_NAME}/summary.json"

for path in "${SOURCE_ROOT}/replay/manifest.json" \
  "${SOURCE_ROOT}/replay/arrays.npz" "${EVAL_ROOT}/heldout_manifest.json"; do
  [[ -s "${path}" ]] || { printf '[place-can-seed43] missing: %s\n' "${path}" >&2; exit 1; }
done

env \
  SEED=43 \
  EPISODES=15 \
  VARIANTS=no_imagination,imagination \
  RUN_ROOT="${RUN_ROOT}" \
  REPLAY_DIR="${SOURCE_ROOT}/replay" \
  SEED_MANIFEST_PATH="${EVAL_ROOT}/heldout_manifest.json" \
  ONLINE_RUN_NAME="${ONLINE_RUN_NAME}" \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_video_expert_place_can_basket_single_task.sh"

conda run --no-capture-output -n robotwin_fastwam python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_wan_head_heldout_pair.py" \
  --summary "${SUMMARY}" --task place_can_basket --expected-pairs 15 \
  --output-json "${RUN_ROOT}/heldout_audit.json"

touch "${RUN_ROOT}/PAIR_COMPLETE"
printf '[place-can-seed43] complete: %s\n' "${RUN_ROOT}/heldout_audit.json"
