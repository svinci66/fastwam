#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
REWARD_JSON="${REWARD_JSON:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_open_microwave_head_only_heldout_10ep_seed19_20260827_wan_vae_head_reward/wan_vae_pair_rewards.json}"
ENCODER_PATH="${ENCODER_PATH:-${PROJECT_ROOT}/../checkpoints/siglip-so400m-patch14-384-modelscope}"
ENCODER_VERSION="${ENCODER_VERSION:-siglip-so400m-patch14-384-modelscope-local-v1}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_open_microwave_wan_head_awr_smoke_seed42_20260827}"
REPLAY_DIR="${RUN_ROOT}/replay"
OUTPUT_ROOT="${RUN_ROOT}/training"
LOG_FILE="${RUN_ROOT}/driver.log"
SEED="${SEED:-42}"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

control="${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_wan_head_no_imagination_smoke.yaml"
treatment="${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_wan_head_with_imagination_smoke.yaml"

if [[ ! -s "${REPLAY_DIR}/manifest.json" ]]; then
  [[ ! -e "${REPLAY_DIR}" ]] || {
    printf '[wan-head-awr] refusing incomplete replay: %s\n' "${REPLAY_DIR}" >&2
    exit 1
  }
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_wan_vae_head_awr_replay.py" \
    --reward-json "${REWARD_JSON}" \
    --output-dir "${REPLAY_DIR}" \
    --encoder-path "${ENCODER_PATH}" \
    --observation-encoder-version "${ENCODER_VERSION}" \
    --reward-config "${treatment}" \
    --device cuda --encoder-dtype bf16 --batch-size 48
fi

for config in "${control}" "${treatment}"; do
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
    --config "${config}" --replay-dir "${REPLAY_DIR}" \
    --output-dir /tmp/unused-wan-head-awr-validation --seed "${SEED}" \
    --timeout-bootstrap-value 0.0 --validate-only
done

declare -A configs=(
  [no_imagination]="${control}"
  [with_imagination]="${treatment}"
)
for variant in no_imagination with_imagination; do
  output_dir="${OUTPUT_ROOT}/seed${SEED}/${variant}"
  if [[ -s "${output_dir}/checkpoint.pt" && -s "${output_dir}/history.json" ]]; then
    printf '[wan-head-awr] skip complete variant=%s\n' "${variant}"
    continue
  fi
  [[ ! -e "${output_dir}" ]] || {
    printf '[wan-head-awr] refusing incomplete output: %s\n' "${output_dir}" >&2
    exit 1
  }
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
    --config "${configs[${variant}]}" --replay-dir "${REPLAY_DIR}" \
    --output-dir "${output_dir}" --seed "${SEED}" \
    --timeout-bootstrap-value 0.0
done

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_awr_training_pair.py" \
  --output-root "${OUTPUT_ROOT}" --seeds "${SEED}" \
  --output-json "${OUTPUT_ROOT}/paired_training_audit.json"

touch "${RUN_ROOT}/COMPLETE"
printf '[wan-head-awr] complete: %s\n' "${RUN_ROOT}"
