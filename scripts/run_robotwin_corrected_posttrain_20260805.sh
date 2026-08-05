#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
SIGLIP_PATH="${SIGLIP_PATH:-/home/ubuntu/sj/fastwam/checkpoints/siglip-so400m-patch14-384-modelscope}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_corrected_posttrain_20260805}"
REWARD_ENCODER_VERSION="${REWARD_ENCODER_VERSION:-siglip-so400m-patch14-384-local-20260803}"
IQL_EPOCHS="${IQL_EPOCHS:-20}"
GATE_EPOCHS="${GATE_EPOCHS:-30}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-24}"

REPLAY_DIR="${RUN_ROOT}/replay_corrected_bf16"
IQL_DIR="${RUN_ROOT}/iql_corrected_imagination_${IQL_EPOCHS}epoch"
GATE_DIR="${RUN_ROOT}/iql_corrected_imagination_${IQL_EPOCHS}epoch_paired_gate"
SUPPORT_DIR="${RUN_ROOT}/support_index_corrected_q95"

RAW_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
RL_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl"

input_dirs=(
  "${RAW_ROOT}/robotwin_10step_failure_collection_4task5ep_paired_v2_20260731/open_microwave"
  "${RAW_ROOT}/robotwin_10step_failure_collection_4task5ep_paired_v2_20260731/hanging_mug"
  "${RAW_ROOT}/robotwin_10step_failure_collection_4task5ep_paired_v2_20260731/place_can_basket"
  "${RAW_ROOT}/robotwin_10step_failure_collection_4task5ep_paired_v2_20260731/adjust_bottle"
  "${RL_ROOT}/robotwin_10step_selected_expert10_20260730/expert_transitions"
  "${RAW_ROOT}/robotwin_low2_online_expert50_strict_pair_5ep_20260730_baseline/blocks_ranking_size"
  "${RAW_ROOT}/robotwin_low2_online_expert50_strict_seed_matrix_5ep_20260730__blocks_ranking_size_episode0_imagination/blocks_ranking_size"
  "${RAW_ROOT}/robotwin_low2_online_expert50_strict_seed_matrix_5ep_20260730__blocks_ranking_size_episode1_imagination/blocks_ranking_size"
  "${RAW_ROOT}/robotwin_low2_online_expert50_strict_seed_matrix_5ep_20260730__blocks_ranking_size_episode2_imagination/blocks_ranking_size"
  "${RAW_ROOT}/robotwin_low2_online_expert50_strict_seed_matrix_5ep_20260730__blocks_ranking_size_episode3_imagination/blocks_ranking_size"
  "${RAW_ROOT}/robotwin_low2_online_expert50_strict_seed_matrix_5ep_20260730__blocks_ranking_size_episode4_imagination/blocks_ranking_size"
  "${RAW_ROOT}/robotwin_hanging_mug_qood_actual_pair_collection_20260805_seed4800001_baseline/hanging_mug"
  "${RAW_ROOT}/robotwin_hanging_mug_qood_actual_pair_collection_20260805_seed4800001_imagination/hanging_mug"
  "${RAW_ROOT}/robotwin_hanging_mug_qood_actual_pair_collection_recovery_20260805_seed4800002_baseline/hanging_mug"
  "${RAW_ROOT}/robotwin_hanging_mug_qood_actual_pair_collection_recovery_20260805_seed4800002_imagination/hanging_mug"
  "${RAW_ROOT}/robotwin_hanging_mug_qood_actual_pair_collection_recovery_20260805_seed4800003_baseline/hanging_mug"
  "${RAW_ROOT}/robotwin_hanging_mug_qood_actual_pair_collection_recovery_20260805_seed4800003_imagination/hanging_mug"
  "${RAW_ROOT}/robotwin_hanging_mug_qood_actual_pair_collection_recovery_20260805_seed4800004_baseline/hanging_mug"
  "${RAW_ROOT}/robotwin_hanging_mug_qood_actual_pair_collection_recovery_20260805_seed4800004_imagination/hanging_mug"
  "${RAW_ROOT}/robotwin_hanging_mug_qood_actual_pair_collection_recovery_20260805_seed4800005_baseline/hanging_mug"
  "${RAW_ROOT}/robotwin_hanging_mug_qood_actual_pair_collection_recovery_20260805_seed4800005_imagination/hanging_mug"
)

target_seed_overrides=(
  "${input_dirs[11]}=4800001"
  "${input_dirs[12]}=4800001"
  "${input_dirs[13]}=4800002"
  "${input_dirs[14]}=4800002"
  "${input_dirs[15]}=4800003"
  "${input_dirs[16]}=4800003"
  "${input_dirs[17]}=4800004"
  "${input_dirs[18]}=4800004"
  "${input_dirs[19]}=4800005"
  "${input_dirs[20]}=4800005"
)

[[ "${IQL_EPOCHS}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'IQL_EPOCHS must be positive\n' >&2
  exit 1
}
[[ "${GATE_EPOCHS}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'GATE_EPOCHS must be positive\n' >&2
  exit 1
}
[[ "${ENCODER_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'ENCODER_BATCH_SIZE must be positive\n' >&2
  exit 1
}
[[ -d "${SIGLIP_PATH}" ]] || {
  printf 'Missing SigLIP directory: %s\n' "${SIGLIP_PATH}" >&2
  exit 1
}
for input_dir in "${input_dirs[@]}"; do
  [[ -d "${input_dir}" ]] || {
    printf 'Missing raw transition directory: %s\n' "${input_dir}" >&2
    exit 1
  }
done

mkdir -p "${RUN_ROOT}"

if [[ ! -f "${REPLAY_DIR}/manifest.json" ]]; then
  input_args=()
  for input_dir in "${input_dirs[@]}"; do
    input_args+=(--input-dir "${input_dir}")
  done
  seed_args=()
  for override in "${target_seed_overrides[@]}"; do
    seed_args+=(--env-seed-override "${override}")
  done
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_residual_rl_replay.py" \
    "${input_args[@]}" \
    "${seed_args[@]}" \
    --output-dir "${REPLAY_DIR}" \
    --encoder-path "${SIGLIP_PATH}" \
    --reward-encoder-version "${REWARD_ENCODER_VERSION}" \
    --reward-config "${PROJECT_ROOT}/configs/rl/robotwin_residual_iql_smoke.yaml" \
    --device cuda \
    --encoder-dtype bf16 \
    --batch-size "${ENCODER_BATCH_SIZE}"
fi

conda run --no-capture-output -n "${CONDA_ENV}" python -c \
  "import json,pathlib; p=pathlib.Path('${REPLAY_DIR}/manifest.json'); m=json.loads(p.read_text()); assert m['num_transitions']==3627, m['num_transitions']; assert m['provenance']['encoder_dtype']=='bfloat16', m['provenance']; print({'replay':str(p.parent),'transitions':m['num_transitions'],'encoder_dtype':m['provenance']['encoder_dtype']})"

if [[ ! -f "${IQL_DIR}/checkpoint.pt" ]]; then
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/scripts/train_libero_residual_iql.py" \
    --config "${PROJECT_ROOT}/configs/rl/robotwin_residual_iql_smoke.yaml" \
    --replay-dir "${REPLAY_DIR}" \
    --output-dir "${IQL_DIR}" \
    --device cuda \
    --epochs "${IQL_EPOCHS}" \
    --seed 42
fi

if [[ ! -f "${GATE_DIR}/checkpoint.pt" ]]; then
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib_robotwin \
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/scripts/train_robotwin_paired_advantage_gate.py" \
    --replay-dir "${REPLAY_DIR}" \
    --base-checkpoint "${IQL_DIR}/checkpoint.pt" \
    --output-dir "${GATE_DIR}" \
    --device cuda \
    --epochs "${GATE_EPOCHS}" \
    --seed 42 \
    --include-residual-equal-outcomes-as-negative
fi

if [[ ! -f "${SUPPORT_DIR}/metadata.json" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_residual_support_index.py" \
    --replay-dir "${REPLAY_DIR}" \
    --checkpoint "${GATE_DIR}/checkpoint.pt" \
    --output-dir "${SUPPORT_DIR}" \
    --calibration-fraction 0.25 \
    --quantile 0.95 \
    --neighbors 10 \
    --score-neighbors 3 \
    --language-similarity-threshold 0.99 \
    --seed 42 \
    --device cuda
fi

conda run --no-capture-output -n "${CONDA_ENV}" python -c \
  "import json,math,pathlib; p=pathlib.Path('${SUPPORT_DIR}/metadata.json'); m=json.loads(p.read_text()); keys=('state_threshold','action_threshold','state_increase_threshold'); assert all(math.isfinite(float(m[k])) for k in keys), {k:m[k] for k in keys}; assert int(m['num_language_prototypes']) >= len(m['task_names']); print({'support_index':str(p.parent),'task_count':len(m['task_names']),'language_prototypes':m['num_language_prototypes'],'thresholds':{k:m[k] for k in keys}})"

printf 'Corrected RoboTwin post-training artifacts are ready: %s\n' "${RUN_ROOT}"
