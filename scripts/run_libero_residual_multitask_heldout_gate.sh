#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT="${FASTWAM_CKPT:-}"
DATASET_STATS_PATH="${DATASET_STATS:-}"
SIGLIP_MODEL_PATH="${SIGLIP_PATH:-}"
SIGLIP_VERSION="${REWARD_ENCODER_VERSION:-}"
NO_IMAGINATION_CHECKPOINT=""
IMAGINATION_CHECKPOINT=""
LIBERO_ROOT="${LIBERO_ROOT:-}"
MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-}"
OUTPUT_ROOT="${MULTITASK_GATE_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="0"
STATE_INDICES="5"
TASK_IDS="0 1 2 3 4 5 6 7 8 9"
NUM_INFERENCE_STEPS="4"
SEED="42"
RESUME="0"

usage() {
  printf '%s\n' "Run a resumable held-out gate for the ten-task residual actors."
  printf '%s\n' "Required: --checkpoint PATH --dataset-stats PATH --siglip-path PATH"
  printf '%s\n' "          --siglip-version ID --no-imagination-checkpoint PATH"
  printf '%s\n' "          --imagination-checkpoint PATH"
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
    --no-imagination-checkpoint)
      require_value "$1" "${2:-}"; NO_IMAGINATION_CHECKPOINT="$2"; shift 2 ;;
    --imagination-checkpoint)
      require_value "$1" "${2:-}"; IMAGINATION_CHECKPOINT="$2"; shift 2 ;;
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

for path_spec in \
  "checkpoint:${CHECKPOINT}" \
  "dataset stats:${DATASET_STATS_PATH}" \
  "SigLIP path:${SIGLIP_MODEL_PATH}" \
  "no-imagination checkpoint:${NO_IMAGINATION_CHECKPOINT}" \
  "imagination checkpoint:${IMAGINATION_CHECKPOINT}"; do
  label="${path_spec%%:*}"
  path="${path_spec#*:}"
  [[ -n "${path}" ]] || { printf 'Error: %s is required.\n' "${label}" >&2; exit 2; }
done
[[ -n "${SIGLIP_VERSION}" ]] || { printf 'Error: --siglip-version is required.\n' >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { printf 'Missing checkpoint: %s\n' "${CHECKPOINT}" >&2; exit 1; }
[[ -f "${DATASET_STATS_PATH}" ]] || { printf 'Missing dataset stats: %s\n' "${DATASET_STATS_PATH}" >&2; exit 1; }
[[ -d "${SIGLIP_MODEL_PATH}" ]] || { printf 'Missing SigLIP path: %s\n' "${SIGLIP_MODEL_PATH}" >&2; exit 1; }
[[ -f "${NO_IMAGINATION_CHECKPOINT}" ]] || {
  printf 'Missing no-imagination checkpoint: %s\n' "${NO_IMAGINATION_CHECKPOINT}" >&2; exit 1;
}
[[ -f "${IMAGINATION_CHECKPOINT}" ]] || {
  printf 'Missing imagination checkpoint: %s\n' "${IMAGINATION_CHECKPOINT}" >&2; exit 1;
}
[[ -d "${LIBERO_ROOT}" ]] || { printf 'Missing LIBERO root: %s\n' "${LIBERO_ROOT}" >&2; exit 1; }
[[ -d "${MODEL_BASE_PATH}" ]] || { printf 'Missing model base path: %s\n' "${MODEL_BASE_PATH}" >&2; exit 1; }

CHECKPOINT="$(readlink -f "${CHECKPOINT}")"
DATASET_STATS_PATH="$(readlink -f "${DATASET_STATS_PATH}")"
SIGLIP_MODEL_PATH="$(readlink -f "${SIGLIP_MODEL_PATH}")"
NO_IMAGINATION_CHECKPOINT="$(readlink -f "${NO_IMAGINATION_CHECKPOINT}")"
IMAGINATION_CHECKPOINT="$(readlink -f "${IMAGINATION_CHECKPOINT}")"
LIBERO_ROOT="$(readlink -f "${LIBERO_ROOT}")"
MODEL_BASE_PATH="$(readlink -f "${MODEL_BASE_PATH}")"
if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${PROJECT_ROOT}/../runs/libero_goal_multitask_heldout_gate_$(date +%Y%m%d_%H%M%S)"
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
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

printf '%s\n' \
  "git_commit=${GIT_COMMIT}" \
  "task_ids=${TASK_IDS}" \
  "state_indices=${STATE_INDICES}" \
  "inference_steps=${NUM_INFERENCE_STEPS}" \
  "seed=${SEED}" \
  "checkpoint=${CHECKPOINT}" \
  "no_imagination_checkpoint=${NO_IMAGINATION_CHECKPOINT}" \
  "imagination_checkpoint=${IMAGINATION_CHECKPOINT}" \
  "siglip_version=${SIGLIP_VERSION}" \
  > "${OUTPUT_ROOT}/run_config.txt"

run_logged cuda_check "${PYTHON_BIN}" -c \
  "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))"

COMMON_OVERRIDES=(
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
  "EVALUATION.replan_steps=8"
  "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}"
  "EVALUATION.rand_device=cpu"
  "EVALUATION.visualize_future_video=false"
  "EVALUATION.save_imagination_transitions=false"
  "EVALUATION.save_rollout_video=false"
  "EVALUATION.save_prediction_videos=false"
  "EVALUATION.imagination_use_direct_action=false"
  "EVALUATION.binarize_gripper=true"
  "EVALUATION.use_action_ensembler=false"
  "EVALUATION.deterministic_env=true"
  "EVALUATION.deterministic_algorithms=true"
  "EVALUATION.deterministic_warn_only=false"
  "EVALUATION.record_action_hashes=true"
)

for task_id in ${TASK_IDS}; do
  for variant in baseline no_imagination imagination; do
    stage="eval_task${task_id}_${variant}"
    output_dir="${OUTPUT_ROOT}/${variant}/task$(printf '%02d' "${task_id}")"
    result_file="${output_dir}/libero_goal/gpu0_task${task_id}_results.json"
    if stage_done "${stage}"; then
      [[ -f "${result_file}" ]] || { printf 'Missing completed result: %s\n' "${result_file}" >&2; exit 1; }
      printf '[skip] %s\n' "${stage}"
      continue
    fi

    variant_overrides=("EVALUATION.action_mode=policy")
    if [[ "${variant}" != "baseline" ]]; then
      residual_checkpoint="${NO_IMAGINATION_CHECKPOINT}"
      if [[ "${variant}" == "imagination" ]]; then
        residual_checkpoint="${IMAGINATION_CHECKPOINT}"
      fi
      variant_overrides=(
        "EVALUATION.action_mode=residual"
        "EVALUATION.residual_checkpoint=${residual_checkpoint}"
        "EVALUATION.residual_encoder_path=${SIGLIP_MODEL_PATH}"
        "EVALUATION.residual_encoder_version=${SIGLIP_VERSION}"
        "EVALUATION.residual_encoder_dtype=no"
      )
    fi

    run_logged "${stage}" "${PYTHON_BIN}" experiments/libero/eval_libero_single.py \
      "${COMMON_OVERRIDES[@]}" \
      "EVALUATION.task_id=${task_id}" \
      "EVALUATION.output_dir=${output_dir}" \
      "${variant_overrides[@]}"
    [[ -f "${result_file}" ]] || { printf 'Missing result: %s\n' "${result_file}" >&2; exit 1; }
    mark_done "${stage}"
  done
done

mark_done complete
printf '[complete] held-out multi-task gate passed\noutput_root=%s\n' "${OUTPUT_ROOT}"
