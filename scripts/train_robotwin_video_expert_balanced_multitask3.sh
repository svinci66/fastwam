#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
SEED="${SEED:-44}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_balanced_pairs_seed${SEED}_20260904}"
REPLAY_DIR="${REPLAY_DIR:-${RUN_ROOT}/replay}"
CONTROL_CONFIG="${CONTROL_CONFIG:-${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_video_expert_multitask3_no_imagination_smoke.yaml}"
TREATMENT_CONFIG="${TREATMENT_CONFIG:-${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_video_expert_multitask3_imagination_smoke.yaml}"
TRAIN_ROOT="${RUN_ROOT}/training"
CONTROL_DIR="${TRAIN_ROOT}/seed${SEED}/no_imagination"
TREATMENT_DIR="${TRAIN_ROOT}/seed${SEED}/with_imagination"

for path in "${REPLAY_DIR}/manifest.json" "${CONTROL_CONFIG}" "${TREATMENT_CONFIG}"; do
  [[ -s "${path}" ]] || { printf '[balanced-multitask3] missing: %s\n' "${path}" >&2; exit 1; }
done
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/training.log") 2>&1

train_if_missing() {
  local variant="$1" config="$2" output_dir="$3"
  if [[ -s "${output_dir}/checkpoint.pt" && -s "${output_dir}/history.json" ]]; then
    printf '[balanced-multitask3] skip complete variant=%s\n' "${variant}"
    return
  fi
  [[ ! -e "${output_dir}" ]] || {
    printf '[balanced-multitask3] refusing incomplete output: %s\n' "${output_dir}" >&2
    exit 1
  }
  printf '[balanced-multitask3] train variant=%s seed=%s\n' "${variant}" "${SEED}"
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
    --config "${config}" --replay-dir "${REPLAY_DIR}" \
    --output-dir "${output_dir}" --seed "${SEED}" \
    --timeout-bootstrap-value 0.0
}

train_if_missing no_imagination "${CONTROL_CONFIG}" "${CONTROL_DIR}"
train_if_missing with_imagination "${TREATMENT_CONFIG}" "${TREATMENT_DIR}"

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_awr_training_pair.py" \
  --output-root "${TRAIN_ROOT}" --seeds "${SEED}" \
  --output-json "${TRAIN_ROOT}/paired_training_audit.json"

touch "${RUN_ROOT}/TRAINING_COMPLETE"
printf '[balanced-multitask3] complete: %s\n' "${TRAIN_ROOT}"
