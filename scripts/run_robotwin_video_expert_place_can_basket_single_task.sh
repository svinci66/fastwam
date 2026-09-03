#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
SEED="${SEED:-42}"
EPISODES="${EPISODES:-5}"
MAX_ONLINE_ATTEMPTS="${MAX_ONLINE_ATTEMPTS:-5}"
TASK="place_can_basket"
FEATURE_VERSION="fastwam_video_expert_final_token_mean_l2_v1"

REWARD_JSON="${REWARD_JSON:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_wan_head_multitask4_smoke2_20260901_wan_vae_head_reward/wan_vae_pair_rewards.json}"
NO_IMAGINATION_CONFIG="${NO_IMAGINATION_CONFIG:-${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_video_expert_place_can_basket_no_imagination.yaml}"
IMAGINATION_CONFIG="${IMAGINATION_CONFIG:-${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_video_expert_place_can_basket_weight025.yaml}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_place_can_basket_single_task_epoch003_seed${SEED}_20260903}"
REPLAY_DIR="${REPLAY_DIR:-${RUN_ROOT}/replay}"
NO_IMAGINATION_TRAIN_DIR="${NO_IMAGINATION_TRAIN_DIR:-${RUN_ROOT}/training/seed${SEED}/no_imagination}"
IMAGINATION_TRAIN_DIR="${IMAGINATION_TRAIN_DIR:-${RUN_ROOT}/training/seed${SEED}/with_imagination}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-robotwin_wan_head_multitask3_awr_formal_block1_5ep_20260901}"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${BASELINE_RUN_NAME}/prevalidated_seed_manifest.json}"
ONLINE_RUN_NAME="${ONLINE_RUN_NAME:-robotwin_video_expert_place_can_basket_single_task_three_way_epoch003_5ep_20260903}"

for path in "${REWARD_JSON}" "${NO_IMAGINATION_CONFIG}" \
  "${IMAGINATION_CONFIG}" "${SEED_MANIFEST_PATH}"; do
  [[ -s "${path}" ]] || { printf '[place-can-single] missing: %s\n' "${path}" >&2; exit 1; }
done
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/driver.log") 2>&1

if [[ ! -s "${REPLAY_DIR}/manifest.json" ]]; then
  [[ ! -e "${REPLAY_DIR}" ]] || {
    printf '[place-can-single] refusing incomplete replay: %s\n' "${REPLAY_DIR}" >&2
    exit 1
  }
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_wan_vae_head_awr_replay.py" \
    --reward-json "${REWARD_JSON}" --output-dir "${REPLAY_DIR}" \
    --actor-observation-source fastwam_video_expert \
    --observation-encoder-version "${FEATURE_VERSION}" \
    --reward-config "${IMAGINATION_CONFIG}" --tasks "${TASK}" \
    --minimum-pairwise-accuracy 1.0
fi

train_if_missing() {
  local config="$1"
  local output_dir="$2"
  if [[ -s "${output_dir}/checkpoint.pt" && -s "${output_dir}/history.json" ]]; then
    printf '[place-can-single] skip complete training: %s\n' "${output_dir}"
    return
  fi
  [[ ! -e "${output_dir}" ]] || {
    printf '[place-can-single] refusing incomplete training: %s\n' "${output_dir}" >&2
    exit 1
  }
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
    --config "${config}" --replay-dir "${REPLAY_DIR}" \
    --output-dir "${output_dir}" --seed "${SEED}" \
    --timeout-bootstrap-value 0.0
}

train_if_missing "${NO_IMAGINATION_CONFIG}" "${NO_IMAGINATION_TRAIN_DIR}"
train_if_missing "${IMAGINATION_CONFIG}" "${IMAGINATION_TRAIN_DIR}"

run_online_pair() {
  env \
    RUN_NAME="${ONLINE_RUN_NAME}" \
    VARIANTS=baseline,no_imagination,imagination \
    TASKS="${TASK}" EPISODES="${EPISODES}" BASE_SEED=47 TRIAL_OFFSET=0 \
    INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
    TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
    PAPER_ALIGNED=true STRICT_PAIRED=true \
    DETERMINISTIC_INSTRUCTION_BY_SEED=true EXPERT_CHECK=true \
    SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH}" \
    NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_TRAIN_DIR}/checkpoint.pt" \
    IMAGINATION_CHECKPOINT="${IMAGINATION_TRAIN_DIR}/checkpoint.pt" \
    RESIDUAL_ENCODER_PATH=none RESIDUAL_ENCODER_VERSION="${FEATURE_VERSION}" \
    RESIDUAL_LANGUAGE_MODE=policy_instruction \
    RESIDUAL_Q_GATE_ENABLED=false RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
    RESIDUAL_SUPPORT_INDEX_PATH=none RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
    RESIDUAL_SHADOW_MODE=false RESIDUAL_INTERVENTION_REPLANS=all \
    RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
    RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=false RESIDUAL_SOFT_SCALE_ENABLED=false \
    SAVE_BASELINE_TRANSITIONS=false SAVE_RESIDUAL_TRANSITIONS=false \
    EVAL_VIDEO_LOG=true \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
}

for ((attempt = 1; attempt <= MAX_ONLINE_ATTEMPTS; attempt++)); do
  printf '[place-can-single] online attempt=%d/%d\n' \
    "${attempt}" "${MAX_ONLINE_ATTEMPTS}"
  if run_online_pair; then
    break
  fi
  if (( attempt == MAX_ONLINE_ATTEMPTS )); then
    printf '[place-can-single] exhausted online attempts\n' >&2
    exit 1
  fi
  printf '[place-can-single] online attempt failed; resuming completed variants\n' >&2
done

touch "${RUN_ROOT}/COMPLETE"
printf '[place-can-single] complete summary=%s\n' \
  "${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${ONLINE_RUN_NAME}/summary.json"
