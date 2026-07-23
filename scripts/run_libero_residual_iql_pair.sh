#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPLAY_DIR=""
OUTPUT_ROOT=""
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="cuda"
EPOCHS=""
SEED=""

usage() {
  printf '%s\n' "Train matched no-imagination / imagination residual-IQL actors."
  printf '%s\n' "Required: --replay-dir PATH --output-root PATH"
  printf '%s\n' "Optional: --device cuda|cpu --epochs N --seed N"
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
    --device) require_value "$1" "${2:-}"; DEVICE="$2"; shift 2 ;;
    --epochs) require_value "$1" "${2:-}"; EPOCHS="$2"; shift 2 ;;
    --seed) require_value "$1" "${2:-}"; SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Error: unknown option %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "${REPLAY_DIR}" ]] || { printf 'Missing replay: %s\n' "${REPLAY_DIR}" >&2; exit 1; }
[[ -n "${OUTPUT_ROOT}" ]] || { printf 'Error: --output-root is required.\n' >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  printf 'Output root already exists: %s\n' "${OUTPUT_ROOT}" >&2
  exit 1
}

REPLAY_DIR="$(readlink -f "${REPLAY_DIR}")"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(readlink -f "${OUTPUT_ROOT}")"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

COMMON_ARGS=(--replay-dir "${REPLAY_DIR}" --device "${DEVICE}")
if [[ -n "${EPOCHS}" ]]; then
  COMMON_ARGS+=(--epochs "${EPOCHS}")
fi
if [[ -n "${SEED}" ]]; then
  COMMON_ARGS+=(--seed "${SEED}")
fi

for variant in no_imagination imagination; do
  config="configs/rl/libero_residual_iql_multitask_global_camera_norm_no_imagination.yaml"
  output_dir="${OUTPUT_ROOT}/train_no_imagination"
  if [[ "${variant}" == "imagination" ]]; then
    config="configs/rl/libero_residual_iql_multitask_global_camera_norm.yaml"
    output_dir="${OUTPUT_ROOT}/train_with_imagination"
  fi
  "${PYTHON_BIN}" scripts/train_libero_residual_iql.py \
    --config "${config}" \
    --output-dir "${output_dir}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/${variant}.log"
done

"${PYTHON_BIN}" -c \
  "import hashlib,json,math,pathlib,torch; root=pathlib.Path('${OUTPUT_ROOT}'); paths=[root/'train_no_imagination',root/'train_with_imagination']; payloads=[torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False) for path in paths]; histories=[json.loads((path/'history.json').read_text()) for path in paths]; assert all(item['format']=='fastwam_residual_iql_v1' for item in payloads); assert payloads[0]['summary']['initialization_sha256']==payloads[1]['summary']['initialization_sha256']; assert all(math.isfinite(float(v)) for history in histories for row in history for v in row.values()); actor_hashes=[]; [actor_hashes.append(hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for _,t in sorted(item['actor'].items()))).hexdigest()) for item in payloads]; assert actor_hashes[0]!=actor_hashes[1]; print({'formats':[item['format'] for item in payloads],'epochs':[len(item) for item in histories],'actor_hashes':actor_hashes,'last':[item[-1] for item in histories]})" \
  | tee "${OUTPUT_ROOT}/pair_check.log"

printf 'Complete. Results: %s\n' "${OUTPUT_ROOT}"
