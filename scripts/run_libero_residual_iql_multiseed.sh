#!/usr/bin/env bash
set -euo pipefail

# Run matched no-imagination / imagination IQL training sequentially on one GPU.
# Each seed uses identical initialization within its pair, while different seeds
# provide independent optimization replicates.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPLAY_DIR=""
OUTPUT_ROOT=""
SEEDS="42,43,44"
DEVICE="cuda"
EPOCHS=""
RESUME="0"

usage() {
  printf '%s\n' "Run matched residual-IQL pairs for multiple seeds on one GPU."
  printf '%s\n' "Required: --replay-dir PATH --output-root PATH"
  printf '%s\n' "Optional: --seeds CSV --device cuda|cpu --epochs N --resume"
}

require_value() {
  if [[ -z "${2:-}" ]]; then
    printf 'Error: %s requires a value.\n' "$1" >&2
    exit 2
  fi
}

while (($# > 0)); do
  case "$1" in
    --replay-dir) require_value "$1" "${2:-}"; REPLAY_DIR="$2"; shift 2 ;;
    --output-root) require_value "$1" "${2:-}"; OUTPUT_ROOT="$2"; shift 2 ;;
    --seeds) require_value "$1" "${2:-}"; SEEDS="$2"; shift 2 ;;
    --device) require_value "$1" "${2:-}"; DEVICE="$2"; shift 2 ;;
    --epochs) require_value "$1" "${2:-}"; EPOCHS="$2"; shift 2 ;;
    --resume) RESUME="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Error: unknown option %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "${REPLAY_DIR}" ]] || { printf 'Missing replay: %s\n' "${REPLAY_DIR}" >&2; exit 1; }
[[ -n "${OUTPUT_ROOT}" ]] || { printf 'Error: --output-root is required.\n' >&2; exit 2; }
if [[ -e "${OUTPUT_ROOT}" && "${RESUME}" != "1" ]]; then
  printf 'Output root exists; pass --resume or choose a new path: %s\n' \
    "${OUTPUT_ROOT}" >&2
  exit 1
fi

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
((${#SEED_ARRAY[@]} > 0)) || { printf 'Error: --seeds must not be empty.\n' >&2; exit 2; }
for seed in "${SEED_ARRAY[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || {
    printf 'Error: seeds must be non-negative integers, got %s\n' "${seed}" >&2
    exit 2
  }
done

REPLAY_DIR="$(readlink -f "${REPLAY_DIR}")"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(readlink -f "${OUTPUT_ROOT}")"
STAGE_DIR="${OUTPUT_ROOT}/.stages"
mkdir -p "${STAGE_DIR}"

printf '%s\n' \
  "replay_dir=${REPLAY_DIR}" \
  "seeds=${SEEDS}" \
  "device=${DEVICE}" \
  "epochs=${EPOCHS:-config_default}" \
  > "${OUTPUT_ROOT}/multiseed_config.txt"

for seed in "${SEED_ARRAY[@]}"; do
  stage="${STAGE_DIR}/seed_${seed}.done"
  seed_output="${OUTPUT_ROOT}/seed_${seed}"
  if [[ -f "${stage}" ]]; then
    printf '[skip] completed seed %s: %s\n' "${seed}" "${seed_output}"
    continue
  fi
  if [[ -e "${seed_output}" ]]; then
    printf 'Incomplete seed output already exists; inspect it before resuming: %s\n' \
      "${seed_output}" >&2
    exit 1
  fi
  args=(
    --replay-dir "${REPLAY_DIR}"
    --output-root "${seed_output}"
    --device "${DEVICE}"
    --seed "${seed}"
  )
  if [[ -n "${EPOCHS}" ]]; then
    args+=(--epochs "${EPOCHS}")
  fi
  PYTHON_BIN="${PYTHON_BIN:-python}" \
    bash "${PROJECT_ROOT}/scripts/run_libero_residual_iql_pair.sh" "${args[@]}"
  date --iso-8601=seconds > "${stage}"
done

printf 'All requested seeds completed. Results: %s\n' "${OUTPUT_ROOT}"
