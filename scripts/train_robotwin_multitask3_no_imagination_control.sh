#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
TASKS="${TASKS:-open_microwave,hanging_mug,place_can_basket}"
SEED="${SEED:-42}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_wan_head_multitask3_awr_seed${SEED}_20260901}"
REPLAY_DIR="${REPLAY_DIR:-${SOURCE_ROOT}/replay}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_wan_head_multitask3_no_imagination_formal.yaml}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_wan_head_multitask3_no_imagination_awr_seed${SEED}_20260901}"
TRAINING_DIR="${RUN_ROOT}/training/formal"
LOG_FILE="${RUN_ROOT}/driver.log"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

for path in "${REPLAY_DIR}/manifest.json" "${CONFIG}"; do
  [[ -s "${path}" ]] || { printf '[multitask3-control] missing: %s\n' "${path}" >&2; exit 1; }
done

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
  --config "${CONFIG}" --replay-dir "${REPLAY_DIR}" \
  --output-dir /tmp/unused-robotwin-multitask3-control-validation \
  --seed "${SEED}" --timeout-bootstrap-value 0.0 --validate-only

if [[ ! -s "${RUN_ROOT}/FORMAL_COMPLETE" ]]; then
  [[ ! -e "${TRAINING_DIR}" ]] || {
    printf '[multitask3-control] refusing incomplete output: %s\n' "${TRAINING_DIR}" >&2
    exit 1
  }
  printf '[multitask3-control] stage=formal_train epochs=20 imagination_weight=0\n'
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
    --config "${CONFIG}" --replay-dir "${REPLAY_DIR}" \
    --output-dir "${TRAINING_DIR}" --seed "${SEED}" \
    --timeout-bootstrap-value 0.0
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/audit_awr_training_health.py" \
    --training-dir "${TRAINING_DIR}" --replay-dir "${REPLAY_DIR}" \
    --expected-tasks "${TASKS}" --expected-epochs 20 \
    --expected-imagination-weight 0.0 \
    --maximum-saturation-fraction 0.50 \
    --output-json "${TRAINING_DIR}/health_audit.json"
  touch "${RUN_ROOT}/FORMAL_COMPLETE"
else
  printf '[multitask3-control] stage=formal_train status=skip_complete\n'
fi

touch "${RUN_ROOT}/COMPLETE"
printf '[multitask3-control] complete run_root=%s\n' "${RUN_ROOT}"
