#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
REPLAY_DIR="${REPLAY_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/awr_pair}"
SEEDS_CSV="${SEEDS:-42,43,44}"
TIMEOUT_BOOTSTRAP_VALUE="${TIMEOUT_BOOTSTRAP_VALUE:-0.0}"

[[ -n "${REPLAY_DIR}" ]] || {
  printf 'REPLAY_DIR must point to a built RoboTwin ReplayBuffer directory.\n' >&2
  exit 2
}
[[ -f "${REPLAY_DIR}/manifest.json" ]] || {
  printf 'Replay manifest not found: %s/manifest.json\n' "${REPLAY_DIR}" >&2
  exit 2
}

declare -A CONFIGS=(
  [no_imagination]="${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_no_imagination.yaml"
  [with_imagination]="${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_with_imagination.yaml"
)
IFS=',' read -r -a seeds <<< "${SEEDS_CSV}"
mkdir -p "${OUTPUT_ROOT}"

for seed in "${seeds[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || {
    printf 'Invalid training seed: %s\n' "${seed}" >&2
    exit 2
  }
  for variant in no_imagination with_imagination; do
    output_dir="${OUTPUT_ROOT}/seed${seed}/${variant}"
    if [[ -s "${output_dir}/checkpoint.pt" && -s "${output_dir}/history.json" ]]; then
      printf '[robotwin-awr] skip complete seed=%s variant=%s\n' "${seed}" "${variant}"
      continue
    fi
    if [[ -e "${output_dir}" ]]; then
      printf 'Refusing to overwrite incomplete output: %s\n' "${output_dir}" >&2
      exit 1
    fi
    printf '[robotwin-awr] train seed=%s variant=%s replay=%s\n' \
      "${seed}" "${variant}" "${REPLAY_DIR}"
    conda run --no-capture-output -n "${CONDA_ENV}" \
      python -u "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
      --config "${CONFIGS[${variant}]}" \
      --replay-dir "${REPLAY_DIR}" \
      --output-dir "${output_dir}" \
      --seed "${seed}" \
      --timeout-bootstrap-value "${TIMEOUT_BOOTSTRAP_VALUE}"
  done
done

conda run --no-capture-output -n "${CONDA_ENV}" \
  python -u "${PROJECT_ROOT}/experiments/robotwin/audit_awr_training_pair.py" \
  --output-root "${OUTPUT_ROOT}" \
  --seeds "${SEEDS_CSV}" \
  --output-json "${OUTPUT_ROOT}/paired_training_audit.json"

printf '[robotwin-awr] complete: %s\n' "${OUTPUT_ROOT}/paired_training_audit.json"
