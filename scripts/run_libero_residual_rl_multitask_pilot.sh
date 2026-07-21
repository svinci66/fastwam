#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT="${FASTWAM_CKPT:-}"
DATASET_STATS_PATH="${DATASET_STATS:-}"
SIGLIP_MODEL_PATH="${SIGLIP_PATH:-}"
SIGLIP_VERSION="${REWARD_ENCODER_VERSION:-}"
LIBERO_ROOT="${LIBERO_ROOT:-}"
MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-}"
OUTPUT_ROOT="${MULTITASK_PILOT_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="0"
SEED="42"
STATE_INDICES="0,1,2,3,4"
TASK_IDS="0 1 2 3 4 5 6 7 8 9"
REPLAN_STEPS="8"
NUM_INFERENCE_STEPS="4"
NOISE_STD="0.075"
SIGLIP_BATCH_SIZE="32"
TIMEOUT_BOOTSTRAP_VALUE="0.0"
RESUME="0"

usage() {
  printf '%s\n' "Run the resumable 10-task LIBERO residual-AWR pilot."
  printf '%s\n' "Required: --checkpoint PATH --dataset-stats PATH --siglip-path PATH"
  printf '%s\n' "Optional: --output-root PATH --libero-root PATH --model-base-path PATH"
  printf '%s\n' "          --gpu-id N --state-indices CSV --inference-steps N --resume"
}

require_value() {
  if [[ -z "${2:-}" ]]; then
    printf 'Error: %s requires a value.\n' "$1" >&2
    exit 2
  fi
}

while (($# > 0)); do
  case "$1" in
    --checkpoint) require_value "$1" "${2:-}"; CHECKPOINT="$2"; shift 2 ;;
    --dataset-stats) require_value "$1" "${2:-}"; DATASET_STATS_PATH="$2"; shift 2 ;;
    --siglip-path) require_value "$1" "${2:-}"; SIGLIP_MODEL_PATH="$2"; shift 2 ;;
    --siglip-version) require_value "$1" "${2:-}"; SIGLIP_VERSION="$2"; shift 2 ;;
    --output-root) require_value "$1" "${2:-}"; OUTPUT_ROOT="$2"; shift 2 ;;
    --libero-root) require_value "$1" "${2:-}"; LIBERO_ROOT="$2"; shift 2 ;;
    --model-base-path) require_value "$1" "${2:-}"; MODEL_BASE_PATH="$2"; shift 2 ;;
    --gpu-id) require_value "$1" "${2:-}"; GPU_ID="$2"; shift 2 ;;
    --state-indices) require_value "$1" "${2:-}"; STATE_INDICES="$2"; shift 2 ;;
    --inference-steps) require_value "$1" "${2:-}"; NUM_INFERENCE_STEPS="$2"; shift 2 ;;
    --resume) RESUME="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Error: unknown option %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for path_spec in "checkpoint:${CHECKPOINT}" "dataset stats:${DATASET_STATS_PATH}" "SigLIP path:${SIGLIP_MODEL_PATH}"; do
  label="${path_spec%%:*}"
  path="${path_spec#*:}"
  if [[ -z "${path}" ]]; then
    printf 'Error: %s is required.\n' "${label}" >&2
    exit 2
  fi
done

[[ -f "${CHECKPOINT}" ]] || { printf 'Missing checkpoint: %s\n' "${CHECKPOINT}" >&2; exit 1; }
[[ -f "${DATASET_STATS_PATH}" ]] || { printf 'Missing dataset stats: %s\n' "${DATASET_STATS_PATH}" >&2; exit 1; }
[[ -d "${SIGLIP_MODEL_PATH}" ]] || { printf 'Missing SigLIP path: %s\n' "${SIGLIP_MODEL_PATH}" >&2; exit 1; }
[[ -d "${LIBERO_ROOT}" ]] || { printf 'Missing LIBERO root: %s\n' "${LIBERO_ROOT}" >&2; exit 1; }
[[ -d "${MODEL_BASE_PATH}" ]] || { printf 'Missing model base path: %s\n' "${MODEL_BASE_PATH}" >&2; exit 1; }

CHECKPOINT="$(readlink -f "${CHECKPOINT}")"
DATASET_STATS_PATH="$(readlink -f "${DATASET_STATS_PATH}")"
SIGLIP_MODEL_PATH="$(readlink -f "${SIGLIP_MODEL_PATH}")"
LIBERO_ROOT="$(readlink -f "${LIBERO_ROOT}")"
MODEL_BASE_PATH="$(readlink -f "${MODEL_BASE_PATH}")"
if [[ -z "${SIGLIP_VERSION}" ]]; then
  SIGLIP_VERSION="google/siglip-so400m-patch14-384@$(basename "${SIGLIP_MODEL_PATH}")"
fi
if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${PROJECT_ROOT}/../runs/libero_goal_multitask_pilot_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -e "${OUTPUT_ROOT}" && "${RESUME}" != "1" ]]; then
  printf 'Output root exists; pass --resume or choose a new path: %s\n' "${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(readlink -f "${OUTPUT_ROOT}")"
LOG_DIR="${OUTPUT_ROOT}/logs"
STAGE_DIR="${OUTPUT_ROOT}/.stages"
mkdir -p "${LOG_DIR}" "${STAGE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONNOUSERSITE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
GIT_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  printf 'Refusing to run with an uncommitted worktree.\n' >&2
  git status --short >&2
  exit 1
fi

IFS=',' read -r -a STATE_INDEX_ARRAY <<< "${STATE_INDICES}"
TRIAL_INDICES="[${STATE_INDICES}]"
NUM_TRIALS="${#STATE_INDEX_ARRAY[@]}"

stage_done() { [[ -f "${STAGE_DIR}/$1.done" ]]; }
mark_done() { date --iso-8601=seconds > "${STAGE_DIR}/$1.done"; }
run_logged() {
  local name="$1"
  shift
  printf '[run] %s\n' "$*"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}
verify_raw_task_mode() {
  local root="$1"
  local expected="$2"
  local metadata_count
  local arrays_count
  metadata_count="$(find "${root}" -type f -name metadata.json | wc -l)"
  arrays_count="$(find "${root}" -type f -name rollout_arrays.npz | wc -l)"
  if ((metadata_count <= 0 || metadata_count != arrays_count || metadata_count < expected)); then
    printf 'Invalid raw transition count under %s: metadata=%s arrays=%s expected_at_least=%s\n' \
      "${root}" "${metadata_count}" "${arrays_count}" "${expected}" >&2
    exit 1
  fi
}

printf '%s\n' "git_commit=${GIT_COMMIT}" "output_root=${OUTPUT_ROOT}" \
  "task_ids=${TASK_IDS}" "state_indices=${STATE_INDICES}" \
  "inference_steps=${NUM_INFERENCE_STEPS}" "noise_std=${NOISE_STD}" \
  "checkpoint=${CHECKPOINT}" "siglip_version=${SIGLIP_VERSION}" \
  > "${OUTPUT_ROOT}/run_config.txt"

run_logged cuda_check "${PYTHON_BIN}" -c \
  "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))"

COMMON_EVAL_OVERRIDES=(
  "task=libero_uncond_2cam224_1e-4"
  "ckpt=${CHECKPOINT}"
  "gpu_id=0"
  "mixed_precision=bf16"
  "seed=${SEED}"
  "EVALUATION.device=cuda"
  "EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}"
  "EVALUATION.task_suite_name=libero_goal"
  "EVALUATION.num_trials=${NUM_TRIALS}"
  "EVALUATION.trial_indices=${TRIAL_INDICES}"
  "EVALUATION.num_steps_wait=30"
  "EVALUATION.action_horizon=32"
  "EVALUATION.replan_steps=${REPLAN_STEPS}"
  "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}"
  "EVALUATION.rand_device=cpu"
  "EVALUATION.visualize_future_video=true"
  "EVALUATION.save_imagination_transitions=true"
  "EVALUATION.save_rollout_video=false"
  "EVALUATION.save_prediction_videos=false"
  "EVALUATION.imagination_use_direct_action=true"
  "EVALUATION.binarize_gripper=true"
  "EVALUATION.use_action_ensembler=false"
)

for task_id in ${TASK_IDS}; do
  for mode in policy noise; do
    stage="collect_task${task_id}_${mode}"
    output_dir="${OUTPUT_ROOT}/raw/task$(printf '%02d' "${task_id}")/${mode}"
    transition_dir="${output_dir}/libero_goal/imagination_transitions"
    if ! stage_done "${stage}"; then
      mode_args=("EVALUATION.action_mode=${mode}")
      if [[ "${mode}" == "noise" ]]; then
        mode_args+=("EVALUATION.action_noise_std=${NOISE_STD}")
      fi
      run_logged "${stage}" "${PYTHON_BIN}" experiments/libero/eval_libero_single.py \
        "${COMMON_EVAL_OVERRIDES[@]}" "EVALUATION.task_id=${task_id}" \
        "EVALUATION.output_dir=${output_dir}" "${mode_args[@]}"
      verify_raw_task_mode "${transition_dir}" "${NUM_TRIALS}"
      mark_done "${stage}"
    else
      verify_raw_task_mode "${transition_dir}" "${NUM_TRIALS}"
    fi
  done
done

TASK_ID_ARGS=()
for task_id in ${TASK_IDS}; do TASK_ID_ARGS+=("${task_id}"); done
TRIAL_INDEX_ARGS=()
for trial_idx in "${STATE_INDEX_ARRAY[@]}"; do TRIAL_INDEX_ARGS+=("${trial_idx}"); done

if ! stage_done audit_raw; then
  run_logged audit_raw "${PYTHON_BIN}" experiments/libero/audit_multitask_collection.py \
    --collection-root "${OUTPUT_ROOT}/raw" \
    --task-ids "${TASK_ID_ARGS[@]}" \
    --trial-indices "${TRIAL_INDEX_ARGS[@]}" \
    --output-json "${OUTPUT_ROOT}/raw_collection_audit.json"
  mark_done audit_raw
fi

REPLAY_DIR="${OUTPUT_ROOT}/replay"
if ! stage_done build_replay; then
  INPUT_DIR_ARGS=()
  for task_id in ${TASK_IDS}; do
    task_dir="${OUTPUT_ROOT}/raw/task$(printf '%02d' "${task_id}")"
    INPUT_DIR_ARGS+=(--input-dir "${task_dir}/policy/libero_goal/imagination_transitions")
    INPUT_DIR_ARGS+=(--input-dir "${task_dir}/noise/libero_goal/imagination_transitions")
  done
  run_logged build_replay "${PYTHON_BIN}" experiments/libero/build_residual_rl_replay.py \
    "${INPUT_DIR_ARGS[@]}" --output-dir "${REPLAY_DIR}" \
    --encoder-path "${SIGLIP_MODEL_PATH}" \
    --reward-encoder-version "${SIGLIP_VERSION}" \
    --reward-config configs/rl/libero_residual_awr_multitask.yaml \
    --device cuda --batch-size "${SIGLIP_BATCH_SIZE}" --agent-weight 0.5 --wrist-weight 0.5
  mark_done build_replay
fi

if ! stage_done validate; then
  for variant in no_imagination imagination; do
    config="configs/rl/libero_residual_awr_multitask_no_imagination.yaml"
    if [[ "${variant}" == "imagination" ]]; then
      config="configs/rl/libero_residual_awr_multitask.yaml"
    fi
    run_logged "validate_${variant}" "${PYTHON_BIN}" scripts/train_libero_residual_awr.py \
      --config "${config}" --replay-dir "${REPLAY_DIR}" \
      --output-dir "${OUTPUT_ROOT}/unused_validate_${variant}" \
      --timeout-bootstrap-value "${TIMEOUT_BOOTSTRAP_VALUE}" --validate-only
  done
  mark_done validate
fi

for variant in no_imagination imagination; do
  config="configs/rl/libero_residual_awr_multitask_no_imagination.yaml"
  output_dir="${OUTPUT_ROOT}/train_no_imagination"
  if [[ "${variant}" == "imagination" ]]; then
    config="configs/rl/libero_residual_awr_multitask.yaml"
    output_dir="${OUTPUT_ROOT}/train_with_imagination"
  fi
  stage="train_${variant}"
  if ! stage_done "${stage}"; then
    run_logged "${stage}" "${PYTHON_BIN}" scripts/train_libero_residual_awr.py \
      --config "${config}" --replay-dir "${REPLAY_DIR}" --output-dir "${output_dir}" \
      --timeout-bootstrap-value "${TIMEOUT_BOOTSTRAP_VALUE}"
    mark_done "${stage}"
  fi
done

run_logged final_check "${PYTHON_BIN}" -c \
  "import json, math, pathlib, torch; root=pathlib.Path('${OUTPUT_ROOT}'); paths=[root/'train_no_imagination', root/'train_with_imagination']; payloads=[torch.load(path/'checkpoint.pt', map_location='cpu', weights_only=False) for path in paths]; histories=[json.loads((path/'history.json').read_text()) for path in paths]; assert all(histories); assert all(math.isfinite(float(value)) for history in histories for row in history for value in row.values()); assert payloads[0]['summary']['initialization_sha256']==payloads[1]['summary']['initialization_sha256']; print({'checkpoint_formats':[item['format'] for item in payloads], 'epochs':[len(item) for item in histories], 'last':[item[-1] for item in histories]})"

mark_done complete
printf 'Complete. Results: %s\n' "${OUTPUT_ROOT}"
