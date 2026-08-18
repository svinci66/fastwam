#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_incremental_iql_20260818}"
PAIR_ROOT="${PAIR_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_pairs}"
SIGLIP_PATH="${SIGLIP_PATH:-/home/ubuntu/sj/fastwam/checkpoints/siglip-so400m-patch14-384-modelscope}"
OLD_ROOT="${OLD_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_corrected_posttrain_20260805}"
OLD_REPLAY="${OLD_REPLAY:-${OLD_ROOT}/replay_corrected_bf16}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${OLD_ROOT}/iql_corrected_imagination_20epoch_paired_gate/checkpoint.pt}"
REWARD_ENCODER_VERSION="${REWARD_ENCODER_VERSION:-siglip-so400m-patch14-384-local-20260803}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-24}"
RESCUE_REPEAT="${RESCUE_REPEAT:-4}"
IQL_EPOCHS="${IQL_EPOCHS:-5}"
IQL_SEED="${IQL_SEED:-20260818}"

SELECTION="${RUN_ROOT}/source_selection.json"
RESCUE_REPLAY="${RUN_ROOT}/replay_rescue_pairs_bf16"
NEGATIVE_REPLAY="${RUN_ROOT}/replay_hard_negative_pairs_bf16"
MERGED_REPLAY="${RUN_ROOT}/replay_backbone_rescue4x_hard_negative"
TRAIN_DIR="${RUN_ROOT}/iql_warm_start_${IQL_EPOCHS}epoch"

for value in "${ENCODER_BATCH_SIZE}" "${RESCUE_REPEAT}" "${IQL_EPOCHS}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'Batch size, rescue repeat, and epochs must be positive integers\n' >&2
    exit 1
  }
done
for path in "${PAIR_ROOT}" "${SIGLIP_PATH}" "${OLD_REPLAY}"; do
  [[ -d "${path}" ]] || {
    printf 'Missing required directory: %s\n' "${path}" >&2
    exit 1
  }
done
[[ -f "${INIT_CHECKPOINT}" ]] || {
  printf 'Missing initialization checkpoint: %s\n' "${INIT_CHECKPOINT}" >&2
  exit 1
}

mkdir -p "${RUN_ROOT}"
conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/select_paired_replay_sources.py" \
  --pair-root "${PAIR_ROOT}" \
  --output "${SELECTION}"

build_bucket() {
  local bucket="$1"
  local output_dir="$2"
  [[ -f "${output_dir}/manifest.json" ]] && return 0

  local source_rows=()
  mapfile -t source_rows < <(
    conda run --no-capture-output -n "${CONDA_ENV}" python -c \
      "import json,pathlib; x=json.loads(pathlib.Path('${SELECTION}').read_text()); [print(f\"{s['path']}\\t{s['environment_seed']}\") for s in x['buckets']['${bucket}']['sources']]"
  )
  local input_args=()
  local seed_args=()
  local row path seed
  for row in "${source_rows[@]}"; do
    IFS=$'\t' read -r path seed <<< "${row}"
    input_args+=(--input-dir "${path}")
    seed_args+=(--env-seed-override "${path}=${seed}")
  done
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_residual_rl_replay.py" \
    "${input_args[@]}" \
    "${seed_args[@]}" \
    --output-dir "${output_dir}" \
    --encoder-path "${SIGLIP_PATH}" \
    --reward-encoder-version "${REWARD_ENCODER_VERSION}" \
    --reward-config "${PROJECT_ROOT}/configs/rl/robotwin_residual_iql_smoke.yaml" \
    --camera-normalization-manifest "${OLD_REPLAY}/manifest.json" \
    --device cuda \
    --encoder-dtype bf16 \
    --batch-size "${ENCODER_BATCH_SIZE}"
}

build_bucket rescue "${RESCUE_REPLAY}"
build_bucket hard_negative "${NEGATIVE_REPLAY}"

if [[ ! -f "${MERGED_REPLAY}/manifest.json" ]]; then
  merge_args=(--input-replay "${OLD_REPLAY}")
  for ((index = 0; index < RESCUE_REPEAT; index += 1)); do
    merge_args+=(--input-replay "${RESCUE_REPLAY}")
  done
  merge_args+=(--input-replay "${NEGATIVE_REPLAY}")
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/merge_residual_replays.py" \
    "${merge_args[@]}" \
    --output-dir "${MERGED_REPLAY}"
fi

if [[ ! -f "${TRAIN_DIR}/checkpoint.pt" ]]; then
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/scripts/train_libero_residual_iql.py" \
    --config "${PROJECT_ROOT}/configs/rl/robotwin_residual_iql_smoke.yaml" \
    --replay-dir "${MERGED_REPLAY}" \
    --output-dir "${TRAIN_DIR}" \
    --init-checkpoint "${INIT_CHECKPOINT}" \
    --device cuda \
    --epochs "${IQL_EPOCHS}" \
    --seed "${IQL_SEED}"
fi

for name in old new; do
  checkpoint="${INIT_CHECKPOINT}"
  [[ "${name}" == new ]] && checkpoint="${TRAIN_DIR}/checkpoint.pt"
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/analyze_residual_q_gate.py" \
    --replay-dir "${MERGED_REPLAY}" \
    --checkpoint "${checkpoint}" \
    --output-json "${RUN_ROOT}/q_gate_audit_${name}.json" \
    --device cuda \
    --batch-size 256
done

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/analyze_paired_incremental_iql.py" \
  --selection "${SELECTION}" \
  --rescue-replay "${RESCUE_REPLAY}" \
  --hard-negative-replay "${NEGATIVE_REPLAY}" \
  --old-checkpoint "${INIT_CHECKPOINT}" \
  --new-checkpoint "${TRAIN_DIR}/checkpoint.pt" \
  --output-json "${RUN_ROOT}/paired_intervention_audit.json" \
  --device cuda \
  --q-margin 0.005 \
  --max-q-disagreement 0.02

printf 'Incremental RoboTwin IQL artifacts are ready: %s\n' "${RUN_ROOT}"
