#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_PREFIX="${RUN_PREFIX:-robotwin_4task_corrected_qood_margin003_segmented_20260805}"

# adjust_bottle episodes 0-3 completed in the original five-episode residual
# process.  Episode 4 triggered a late VAE CUDA OOM.  Every remaining pair is
# isolated in a fresh process so CUDA memory cannot accumulate across episodes.
segments=(
  adjust_bottle:4
  hanging_mug:0 hanging_mug:1 hanging_mug:2 hanging_mug:3 hanging_mug:4
  open_microwave:0 open_microwave:1 open_microwave:2 open_microwave:3 open_microwave:4
  place_can_basket:0 place_can_basket:1 place_can_basket:2 place_can_basket:3 place_can_basket:4
)

for segment in "${segments[@]}"; do
  task="${segment%%:*}"
  offset="${segment##*:}"
  run_name="${RUN_PREFIX}_${task}_episode${offset}"
  printf '[robotwin-segmented-recovery] task=%s offset=%s run=%s\n' \
    "${task}" "${offset}" "${run_name}"

  env \
    RUN_NAME="${run_name}" \
    VARIANTS=baseline,imagination \
    TASKS="${task}" \
    EPISODES=1 \
    TRIAL_OFFSET="${offset}" \
    EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-true}" \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_corrected_qood_margin003_expanded4.sh"
done

printf 'Segmented Q+OOD recovery evaluation complete: %s\n' "${RUN_PREFIX}"
