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
OUTPUT_ROOT="${SMOKE_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="0"
SEED="42"
TASK_ID="3"
NUM_TRIALS="1"
REPLAN_STEPS="8"
NUM_INFERENCE_STEPS="4"
NOISE_STD="0.075"
SIGLIP_BATCH_SIZE="16"
AGENT_WEIGHT="0.5"
WRIST_WEIGHT="0.5"
TIMEOUT_BOOTSTRAP_VALUE="0.0"
RUN_UNIT_TESTS="1"
RUN_TRAINING="1"
RESUME="0"

usage() {
  cat <<'EOF'
Run the single-GPU LIBERO residual-RL smoke pipeline.

Required (flags or matching environment variables):
  --checkpoint PATH       FASTWAM_CKPT
  --dataset-stats PATH    DATASET_STATS
  --siglip-path PATH      SIGLIP_PATH

Common options:
  --output-root PATH      Fresh output root (default: evaluate_results timestamp)
  --libero-root PATH      Optional LIBERO repository root added to PYTHONPATH
  --model-base-path PATH  Optional DIFFSYNTH_MODEL_BASE_PATH override
  --siglip-version TEXT   Immutable reward encoder version identifier
  --gpu-id N              Physical GPU exposed as logical cuda:0 (default: 0)
  --seed N                Shared policy/noise seed (default: 42)
  --task-id N             libero_goal task id (default: 3)
  --num-trials N          Episodes per behavior mode (default: 1)
  --inference-steps N     Diffusion steps for smoke collection (default: 4)
  --noise-std X           Controlled behavior noise (default: 0.075)
  --python PATH           Python executable after environment activation
  --skip-unit-tests       Skip the repository CPU unit tests
  --no-train              Stop after replay validation
  --resume                Skip stages with completion markers in output root
  -h, --help              Show this help

Example:
  bash scripts/run_libero_residual_rl_smoke.sh \
    --checkpoint /server/checkpoints/libero_uncond_2cam224.pt \
    --dataset-stats /server/checkpoints/libero_uncond_2cam224_dataset_stats.json \
    --siglip-path /server/checkpoints/siglip/snapshots/REVISION \
    --output-root /server/runs/fastwam_rl_smoke
EOF
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    echo "Error: ${option} requires a value." >&2
    exit 2
  fi
}

while (($# > 0)); do
  case "$1" in
    --checkpoint)
      require_value "$1" "${2:-}"
      CHECKPOINT="$2"
      shift 2
      ;;
    --dataset-stats)
      require_value "$1" "${2:-}"
      DATASET_STATS_PATH="$2"
      shift 2
      ;;
    --siglip-path)
      require_value "$1" "${2:-}"
      SIGLIP_MODEL_PATH="$2"
      shift 2
      ;;
    --siglip-version)
      require_value "$1" "${2:-}"
      SIGLIP_VERSION="$2"
      shift 2
      ;;
    --output-root)
      require_value "$1" "${2:-}"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --libero-root)
      require_value "$1" "${2:-}"
      LIBERO_ROOT="$2"
      shift 2
      ;;
    --model-base-path)
      require_value "$1" "${2:-}"
      MODEL_BASE_PATH="$2"
      shift 2
      ;;
    --gpu-id)
      require_value "$1" "${2:-}"
      GPU_ID="$2"
      shift 2
      ;;
    --seed)
      require_value "$1" "${2:-}"
      SEED="$2"
      shift 2
      ;;
    --task-id)
      require_value "$1" "${2:-}"
      TASK_ID="$2"
      shift 2
      ;;
    --num-trials)
      require_value "$1" "${2:-}"
      NUM_TRIALS="$2"
      shift 2
      ;;
    --inference-steps)
      require_value "$1" "${2:-}"
      NUM_INFERENCE_STEPS="$2"
      shift 2
      ;;
    --noise-std)
      require_value "$1" "${2:-}"
      NOISE_STD="$2"
      shift 2
      ;;
    --python)
      require_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-unit-tests)
      RUN_UNIT_TESTS="0"
      shift
      ;;
    --no-train)
      RUN_TRAINING="0"
      shift
      ;;
    --resume)
      RESUME="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for pair in \
  "--checkpoint:${CHECKPOINT}" \
  "--dataset-stats:${DATASET_STATS_PATH}" \
  "--siglip-path:${SIGLIP_MODEL_PATH}"; do
  option="${pair%%:*}"
  value="${pair#*:}"
  if [[ -z "${value}" ]]; then
    echo "Error: ${option} is required." >&2
    usage >&2
    exit 2
  fi
done

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Error: FastWAM checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_STATS_PATH}" ]]; then
  echo "Error: dataset stats not found: ${DATASET_STATS_PATH}" >&2
  exit 1
fi
if [[ ! -d "${SIGLIP_MODEL_PATH}" ]]; then
  echo "Error: SigLIP model directory not found: ${SIGLIP_MODEL_PATH}" >&2
  exit 1
fi
if [[ -n "${LIBERO_ROOT}" && ! -d "${LIBERO_ROOT}" ]]; then
  echo "Error: LIBERO root not found: ${LIBERO_ROOT}" >&2
  exit 1
fi
if [[ -n "${MODEL_BASE_PATH}" && ! -d "${MODEL_BASE_PATH}" ]]; then
  echo "Error: model base path not found: ${MODEL_BASE_PATH}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if ! [[ "${GPU_ID}" =~ ^[0-9]+$ && "${SEED}" =~ ^[0-9]+$ && "${TASK_ID}" =~ ^[0-9]+$ ]]; then
  echo "Error: gpu-id, seed, and task-id must be non-negative integers." >&2
  exit 2
fi
if ! [[ "${NUM_TRIALS}" =~ ^[1-9][0-9]*$ && "${NUM_INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: num-trials and inference-steps must be positive integers." >&2
  exit 2
fi
if ! [[ "${NOISE_STD}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: noise-std must be a non-negative decimal number." >&2
  exit 2
fi
if [[ "${REPLAN_STEPS}" != "8" ]]; then
  echo "Error: this smoke protocol is frozen to replan_steps=8." >&2
  exit 2
fi

CHECKPOINT="$(readlink -f "${CHECKPOINT}")"
DATASET_STATS_PATH="$(readlink -f "${DATASET_STATS_PATH}")"
SIGLIP_MODEL_PATH="$(readlink -f "${SIGLIP_MODEL_PATH}")"
if [[ -n "${LIBERO_ROOT}" ]]; then
  LIBERO_ROOT="$(readlink -f "${LIBERO_ROOT}")"
fi
if [[ -n "${MODEL_BASE_PATH}" ]]; then
  MODEL_BASE_PATH="$(readlink -f "${MODEL_BASE_PATH}")"
  export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"
fi
if [[ -z "${SIGLIP_VERSION}" ]]; then
  SIGLIP_VERSION="google/siglip-so400m-patch14-384@$(basename "${SIGLIP_MODEL_PATH}")"
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${PROJECT_ROOT}/evaluate_results/residual_rl_smoke_$(date +%Y%m%d_%H%M%S)"
fi
OUTPUT_PREEXISTED="0"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  OUTPUT_PREEXISTED="1"
fi
if [[ -e "${OUTPUT_ROOT}" && "${RESUME}" != "1" ]]; then
  echo "Error: output root already exists; choose a fresh path or pass --resume: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(readlink -f "${OUTPUT_ROOT}")"
LOG_DIR="${OUTPUT_ROOT}/logs"
STAGE_DIR="${OUTPUT_ROOT}/.stages"
mkdir -p "${LOG_DIR}" "${STAGE_DIR}"

POLICY_ROOT="${OUTPUT_ROOT}/policy"
NOISE_ROOT="${OUTPUT_ROOT}/noise_${NOISE_STD}"
POLICY_TRANSITIONS="${POLICY_ROOT}/libero_goal/imagination_transitions"
NOISE_TRANSITIONS="${NOISE_ROOT}/libero_goal/imagination_transitions"
REPLAY_DIR="${OUTPUT_ROOT}/replay"
NO_IMAG_RUN="${OUTPUT_ROOT}/train_no_imagination"
WITH_IMAG_RUN="${OUTPUT_ROOT}/train_with_imagination"
NO_IMAG_EVAL="${OUTPUT_ROOT}/eval_no_imagination"
WITH_IMAG_EVAL="${OUTPUT_ROOT}/eval_with_imagination"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${LIBERO_ROOT:+:${LIBERO_ROOT}}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
if [[ -z "${GIT_COMMIT}" ]]; then
  echo "Error: project root is not a readable Git checkout: ${PROJECT_ROOT}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Error: source checkout has uncommitted or untracked files; refusing a non-reproducible smoke run." >&2
  git status --short --untracked-files=normal >&2
  exit 1
fi

emit_frozen_config() {
  echo "git_commit=${GIT_COMMIT}"
  echo "checkpoint=${CHECKPOINT}"
  echo "dataset_stats=${DATASET_STATS_PATH}"
  echo "siglip_path=${SIGLIP_MODEL_PATH}"
  echo "siglip_version=${SIGLIP_VERSION}"
  echo "libero_root=${LIBERO_ROOT}"
  echo "model_base_path=${MODEL_BASE_PATH}"
  echo "gpu_id=${GPU_ID}"
  echo "seed=${SEED}"
  echo "task_suite=libero_goal"
  echo "task_id=${TASK_ID}"
  echo "num_trials=${NUM_TRIALS}"
  echo "replan_steps=${REPLAN_STEPS}"
  echo "num_inference_steps=${NUM_INFERENCE_STEPS}"
  echo "noise_std=${NOISE_STD}"
  echo "siglip_batch_size=${SIGLIP_BATCH_SIZE}"
  echo "agent_weight=${AGENT_WEIGHT}"
  echo "wrist_weight=${WRIST_WEIGHT}"
  echo "timeout_bootstrap_value=${TIMEOUT_BOOTSTRAP_VALUE}"
}

FROZEN_CONFIG_PATH="${OUTPUT_ROOT}/smoke_config.env"
PROPOSED_CONFIG_PATH="${OUTPUT_ROOT}/.smoke_config.proposed"
emit_frozen_config >"${PROPOSED_CONFIG_PATH}"
if [[ "${OUTPUT_PREEXISTED}" == "1" && "${RESUME}" == "1" ]]; then
  if [[ ! -f "${FROZEN_CONFIG_PATH}" ]]; then
    echo "Error: cannot resume an output root without smoke_config.env: ${OUTPUT_ROOT}" >&2
    exit 1
  fi
  if ! cmp -s "${FROZEN_CONFIG_PATH}" "${PROPOSED_CONFIG_PATH}"; then
    echo "Error: resume parameters differ from the frozen smoke configuration." >&2
    diff -u "${FROZEN_CONFIG_PATH}" "${PROPOSED_CONFIG_PATH}" || true
    exit 1
  fi
  rm -f "${PROPOSED_CONFIG_PATH}"
else
  mv "${PROPOSED_CONFIG_PATH}" "${FROZEN_CONFIG_PATH}"
fi

stage_done() {
  [[ -f "${STAGE_DIR}/$1.done" ]]
}

mark_done() {
  date --iso-8601=seconds >"${STAGE_DIR}/$1.done"
}

run_logged() {
  local log_name="$1"
  shift
  echo "[run] $*"
  "$@" 2>&1 | tee "${LOG_DIR}/${log_name}.log"
}

verify_transition_root() {
  local root="$1"
  local metadata_count
  local array_count
  metadata_count="$(find "${root}" -type f -name metadata.json | wc -l)"
  array_count="$(find "${root}" -type f -name rollout_arrays.npz | wc -l)"
  if ((metadata_count <= 0 || metadata_count != array_count)); then
    echo "Error: invalid transition record counts under ${root}: metadata=${metadata_count} arrays=${array_count}" >&2
    exit 1
  fi
  echo "[check] ${root}: metadata=${metadata_count} arrays=${array_count}"
}

echo "============================================================"
echo "FastWAM residual-RL single-GPU smoke"
echo "project_root=${PROJECT_ROOT}"
echo "git_commit=${GIT_COMMIT}"
echo "checkpoint=${CHECKPOINT}"
echo "dataset_stats=${DATASET_STATS_PATH}"
echo "siglip_path=${SIGLIP_MODEL_PATH}"
echo "siglip_version=${SIGLIP_VERSION}"
echo "physical_gpu=${GPU_ID} logical_gpu=0"
echo "task=libero_goal/${TASK_ID} seed=${SEED} trials_per_mode=${NUM_TRIALS}"
echo "K=${REPLAN_STEPS} inference_steps=${NUM_INFERENCE_STEPS} noise_std=${NOISE_STD}"
echo "output_root=${OUTPUT_ROOT}"
echo "============================================================"

run_logged nvidia_smi nvidia-smi
run_logged cuda_check "${PYTHON_BIN}" -c \
  "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('visible_gpus=', torch.cuda.device_count()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1"
run_logged import_check "${PYTHON_BIN}" -c \
  "import fastwam, mujoco, robosuite; from libero.libero import benchmark; print('FastWAM/LIBERO imports OK')"

if [[ "${RUN_UNIT_TESTS}" == "1" ]]; then
  if ! stage_done unit_tests; then
    run_logged unit_tests "${PYTHON_BIN}" -m pytest -q tests
    mark_done unit_tests
  else
    echo "[skip] unit_tests"
  fi
fi

COMMON_EVAL_OVERRIDES=(
  "task=libero_uncond_2cam224_1e-4"
  "ckpt=${CHECKPOINT}"
  "gpu_id=0"
  "mixed_precision=bf16"
  "seed=${SEED}"
  "EVALUATION.device=cuda"
  "EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}"
  "EVALUATION.task_suite_name=libero_goal"
  "EVALUATION.task_id=${TASK_ID}"
  "EVALUATION.num_trials=${NUM_TRIALS}"
  "EVALUATION.num_steps_wait=30"
  "EVALUATION.action_horizon=32"
  "EVALUATION.replan_steps=${REPLAN_STEPS}"
  "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}"
  "EVALUATION.rand_device=cpu"
  "EVALUATION.visualize_future_video=true"
  "EVALUATION.save_imagination_transitions=true"
  "EVALUATION.imagination_use_direct_action=true"
  "EVALUATION.binarize_gripper=true"
  "EVALUATION.use_action_ensembler=false"
)

if ! stage_done collect_policy; then
  run_logged collect_policy "${PYTHON_BIN}" experiments/libero/eval_libero_single.py \
    "${COMMON_EVAL_OVERRIDES[@]}" \
    "EVALUATION.action_mode=policy" \
    "EVALUATION.output_dir=${POLICY_ROOT}"
  verify_transition_root "${POLICY_TRANSITIONS}"
  mark_done collect_policy
else
  echo "[skip] collect_policy"
  verify_transition_root "${POLICY_TRANSITIONS}"
fi

if ! stage_done collect_noise; then
  run_logged collect_noise "${PYTHON_BIN}" experiments/libero/eval_libero_single.py \
    "${COMMON_EVAL_OVERRIDES[@]}" \
    "EVALUATION.action_mode=noise" \
    "EVALUATION.action_noise_std=${NOISE_STD}" \
    "EVALUATION.output_dir=${NOISE_ROOT}"
  verify_transition_root "${NOISE_TRANSITIONS}"
  mark_done collect_noise
else
  echo "[skip] collect_noise"
  verify_transition_root "${NOISE_TRANSITIONS}"
fi

if ! stage_done build_replay; then
  if [[ -e "${REPLAY_DIR}" ]]; then
    echo "Error: incomplete replay path exists without a completion marker: ${REPLAY_DIR}" >&2
    exit 1
  fi
  run_logged build_replay "${PYTHON_BIN}" experiments/libero/build_residual_rl_replay.py \
    --input-dir "${POLICY_TRANSITIONS}" \
    --input-dir "${NOISE_TRANSITIONS}" \
    --output-dir "${REPLAY_DIR}" \
    --encoder-path "${SIGLIP_MODEL_PATH}" \
    --reward-encoder-version "${SIGLIP_VERSION}" \
    --reward-config configs/rl/libero_residual_awr_mvp.yaml \
    --device cuda \
    --batch-size "${SIGLIP_BATCH_SIZE}" \
    --agent-weight "${AGENT_WEIGHT}" \
    --wrist-weight "${WRIST_WEIGHT}"
  test -f "${REPLAY_DIR}/manifest.json"
  test -f "${REPLAY_DIR}/arrays.npz"
  test -f "${REPLAY_DIR}/transitions.jsonl"
  mark_done build_replay
else
  echo "[skip] build_replay"
fi

if ! stage_done validate_replay; then
  run_logged validate_with_imagination "${PYTHON_BIN}" scripts/train_libero_residual_awr.py \
    --config configs/rl/libero_residual_awr_mvp.yaml \
    --replay-dir "${REPLAY_DIR}" \
    --output-dir "${OUTPUT_ROOT}/unused_validate_with_imagination" \
    --timeout-bootstrap-value "${TIMEOUT_BOOTSTRAP_VALUE}" \
    --validate-only
  run_logged validate_no_imagination "${PYTHON_BIN}" scripts/train_libero_residual_awr.py \
    --config configs/rl/libero_residual_awr_no_imagination.yaml \
    --replay-dir "${REPLAY_DIR}" \
    --output-dir "${OUTPUT_ROOT}/unused_validate_no_imagination" \
    --timeout-bootstrap-value "${TIMEOUT_BOOTSTRAP_VALUE}" \
    --validate-only
  mark_done validate_replay
else
  echo "[skip] validate_replay"
fi

if [[ "${RUN_TRAINING}" == "0" ]]; then
  echo "[complete] collection, replay construction, and validation passed; training skipped."
  echo "output_root=${OUTPUT_ROOT}"
  exit 0
fi

if ! stage_done train_no_imagination; then
  if [[ -e "${NO_IMAG_RUN}" ]]; then
    echo "Error: incomplete no-imagination run exists without a completion marker: ${NO_IMAG_RUN}" >&2
    exit 1
  fi
  run_logged train_no_imagination "${PYTHON_BIN}" scripts/train_libero_residual_awr.py \
    --config configs/rl/libero_residual_awr_no_imagination.yaml \
    --replay-dir "${REPLAY_DIR}" \
    --output-dir "${NO_IMAG_RUN}" \
    --timeout-bootstrap-value "${TIMEOUT_BOOTSTRAP_VALUE}"
  test -f "${NO_IMAG_RUN}/checkpoint.pt"
  test -f "${NO_IMAG_RUN}/history.json"
  mark_done train_no_imagination
else
  echo "[skip] train_no_imagination"
fi

if ! stage_done train_with_imagination; then
  if [[ -e "${WITH_IMAG_RUN}" ]]; then
    echo "Error: incomplete with-imagination run exists without a completion marker: ${WITH_IMAG_RUN}" >&2
    exit 1
  fi
  run_logged train_with_imagination "${PYTHON_BIN}" scripts/train_libero_residual_awr.py \
    --config configs/rl/libero_residual_awr_mvp.yaml \
    --replay-dir "${REPLAY_DIR}" \
    --output-dir "${WITH_IMAG_RUN}" \
    --timeout-bootstrap-value "${TIMEOUT_BOOTSTRAP_VALUE}"
  test -f "${WITH_IMAG_RUN}/checkpoint.pt"
  test -f "${WITH_IMAG_RUN}/history.json"
  mark_done train_with_imagination
else
  echo "[skip] train_with_imagination"
fi

export NO_IMAG_HISTORY="${NO_IMAG_RUN}/history.json"
export WITH_IMAG_HISTORY="${WITH_IMAG_RUN}/history.json"
run_logged verify_histories "${PYTHON_BIN}" -c \
  "import json, math, os, pathlib, torch; paths=[os.environ['NO_IMAG_HISTORY'], os.environ['WITH_IMAG_HISTORY']]; histories=[json.load(open(path)) for path in paths]; assert all(histories); bad=[(path, i, key, value) for path, history in zip(paths, histories) for i, row in enumerate(history) for key, value in row.items() if isinstance(value, (int, float)) and not math.isfinite(value)]; checkpoints=[torch.load(pathlib.Path(path).with_name('checkpoint.pt'), map_location='cpu', weights_only=False) for path in paths]; init=[item['summary']['initialization_sha256'] for item in checkpoints]; print('epochs=', [len(history) for history in histories]); print('last=', [history[-1] for history in histories]); print('initialization_sha256=', init); print('non_finite=', bad); assert not bad; assert init[0] == init[1], init"

if ! stage_done eval_no_imagination; then
  run_logged eval_no_imagination "${PYTHON_BIN}" experiments/libero/eval_libero_single.py \
    "${COMMON_EVAL_OVERRIDES[@]}" \
    "EVALUATION.action_mode=residual" \
    "EVALUATION.residual_checkpoint=${NO_IMAG_RUN}/checkpoint.pt" \
    "EVALUATION.residual_encoder_path=${SIGLIP_MODEL_PATH}" \
    "EVALUATION.residual_encoder_version=${SIGLIP_VERSION}" \
    "EVALUATION.residual_encoder_dtype=no" \
    "EVALUATION.output_dir=${NO_IMAG_EVAL}"
  mark_done eval_no_imagination
else
  echo "[skip] eval_no_imagination"
fi

if ! stage_done eval_with_imagination; then
  run_logged eval_with_imagination "${PYTHON_BIN}" experiments/libero/eval_libero_single.py \
    "${COMMON_EVAL_OVERRIDES[@]}" \
    "EVALUATION.action_mode=residual" \
    "EVALUATION.residual_checkpoint=${WITH_IMAG_RUN}/checkpoint.pt" \
    "EVALUATION.residual_encoder_path=${SIGLIP_MODEL_PATH}" \
    "EVALUATION.residual_encoder_version=${SIGLIP_VERSION}" \
    "EVALUATION.residual_encoder_dtype=no" \
    "EVALUATION.output_dir=${WITH_IMAG_EVAL}"
  mark_done eval_with_imagination
else
  echo "[skip] eval_with_imagination"
fi

echo "============================================================"
echo "[complete] single-GPU residual-RL smoke passed"
echo "output_root=${OUTPUT_ROOT}"
echo "replay=${REPLAY_DIR}"
echo "no_imagination_checkpoint=${NO_IMAG_RUN}/checkpoint.pt"
echo "with_imagination_checkpoint=${WITH_IMAG_RUN}/checkpoint.pt"
echo "baseline_result=${POLICY_ROOT}/libero_goal/gpu0_task${TASK_ID}_results.json"
echo "no_imagination_result=${NO_IMAG_EVAL}/libero_goal/gpu0_task${TASK_ID}_results.json"
echo "with_imagination_result=${WITH_IMAG_EVAL}/libero_goal/gpu0_task${TASK_ID}_results.json"
echo "============================================================"
