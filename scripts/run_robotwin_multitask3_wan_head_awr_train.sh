#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
TASKS="${TASKS:-open_microwave,hanging_mug,place_can_basket}"
SEED="${SEED:-42}"
REWARD_JSON="${REWARD_JSON:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_wan_head_multitask4_smoke2_20260901_wan_vae_head_reward/wan_vae_pair_rewards.json}"
ENCODER_PATH="${ENCODER_PATH:-${PROJECT_ROOT}/../checkpoints/siglip-so400m-patch14-384-modelscope}"
ENCODER_VERSION="${ENCODER_VERSION:-siglip-so400m-patch14-384-modelscope-local-v1}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_wan_head_multitask3_awr_seed${SEED}_20260901}"
REPLAY_DIR="${RUN_ROOT}/replay"
SMOKE_DIR="${RUN_ROOT}/training/smoke"
FORMAL_DIR="${RUN_ROOT}/training/formal"
SMOKE_CONFIG="${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_wan_head_multitask3_smoke.yaml"
FORMAL_CONFIG="${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_wan_head_multitask3_formal.yaml"
LOG_FILE="${RUN_ROOT}/driver.log"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

for path in "${REWARD_JSON}" "${ENCODER_PATH}" "${SMOKE_CONFIG}" "${FORMAL_CONFIG}"; do
  [[ -e "${path}" ]] || { printf '[multitask3-awr] missing: %s\n' "${path}" >&2; exit 1; }
done

if [[ ! -s "${REPLAY_DIR}/manifest.json" ]]; then
  [[ ! -e "${REPLAY_DIR}" ]] || {
    printf '[multitask3-awr] refusing incomplete replay: %s\n' "${REPLAY_DIR}" >&2
    exit 1
  }
  printf '[multitask3-awr] stage=build_replay tasks=%s\n' "${TASKS}"
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_wan_vae_head_awr_replay.py" \
    --reward-json "${REWARD_JSON}" --output-dir "${REPLAY_DIR}" \
    --encoder-path "${ENCODER_PATH}" \
    --observation-encoder-version "${ENCODER_VERSION}" \
    --reward-config "${FORMAL_CONFIG}" --tasks "${TASKS}" \
    --minimum-pairwise-accuracy 0.90 \
    --device cuda --encoder-dtype bf16 --batch-size 48
else
  printf '[multitask3-awr] stage=build_replay status=skip_complete\n'
fi

for config in "${SMOKE_CONFIG}" "${FORMAL_CONFIG}"; do
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
    --config "${config}" --replay-dir "${REPLAY_DIR}" \
    --output-dir /tmp/unused-robotwin-multitask3-awr-validation \
    --seed "${SEED}" --timeout-bootstrap-value 0.0 --validate-only
done

if [[ ! -s "${RUN_ROOT}/SMOKE_COMPLETE" ]]; then
  [[ ! -e "${SMOKE_DIR}" ]] || {
    printf '[multitask3-awr] refusing incomplete smoke output: %s\n' "${SMOKE_DIR}" >&2
    exit 1
  }
  printf '[multitask3-awr] stage=smoke_train epochs=3\n'
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
    --config "${SMOKE_CONFIG}" --replay-dir "${REPLAY_DIR}" \
    --output-dir "${SMOKE_DIR}" --seed "${SEED}" \
    --timeout-bootstrap-value 0.0
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/audit_awr_training_health.py" \
    --training-dir "${SMOKE_DIR}" --replay-dir "${REPLAY_DIR}" \
    --expected-tasks "${TASKS}" --expected-epochs 3 \
    --output-json "${SMOKE_DIR}/health_audit.json"
  touch "${RUN_ROOT}/SMOKE_COMPLETE"
else
  printf '[multitask3-awr] stage=smoke_train status=skip_complete\n'
fi

if [[ ! -s "${RUN_ROOT}/FORMAL_COMPLETE" ]]; then
  [[ ! -e "${FORMAL_DIR}" ]] || {
    printf '[multitask3-awr] refusing incomplete formal output: %s\n' "${FORMAL_DIR}" >&2
    exit 1
  }
  printf '[multitask3-awr] stage=formal_train epochs=20\n'
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
    --config "${FORMAL_CONFIG}" --replay-dir "${REPLAY_DIR}" \
    --output-dir "${FORMAL_DIR}" --seed "${SEED}" \
    --timeout-bootstrap-value 0.0
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/audit_awr_training_health.py" \
    --training-dir "${FORMAL_DIR}" --replay-dir "${REPLAY_DIR}" \
    --expected-tasks "${TASKS}" --expected-epochs 20 \
    --maximum-saturation-fraction 0.50 \
    --output-json "${FORMAL_DIR}/health_audit.json"
  touch "${RUN_ROOT}/FORMAL_COMPLETE"
else
  printf '[multitask3-awr] stage=formal_train status=skip_complete\n'
fi

touch "${RUN_ROOT}/COMPLETE"
printf '[multitask3-awr] complete run_root=%s\n' "${RUN_ROOT}"
