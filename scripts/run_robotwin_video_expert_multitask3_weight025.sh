#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
PHASE="${PHASE:-all}"
SEED="${SEED:-42}"
TASKS="${TASKS:-open_microwave,hanging_mug,place_can_basket}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/home/ubuntu/sj/fastwam/checkpoints}"
REWARD_JSON="${REWARD_JSON:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_wan_head_multitask4_smoke2_20260901_wan_vae_head_reward/wan_vae_pair_rewards.json}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-robotwin_wan_head_multitask3_awr_formal_block1_5ep_20260901}"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${BASELINE_RUN_NAME}/prevalidated_seed_manifest.json}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_video_expert_multitask3_imagination_smoke.yaml}"
FEATURE_VERSION="fastwam_video_expert_final_token_mean_l2_v1"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_weight025_epochs3_seed${SEED}_20260903}"
BACKFILL_DIR="${BACKFILL_DIR:-${RUN_ROOT}/feature_backfill}"
REPLAY_DIR="${REPLAY_DIR:-${RUN_ROOT}/replay}"
TRAIN_DIR="${TRAIN_DIR:-${RUN_ROOT}/training/seed${SEED}/with_imagination}"
ONLINE_RUN_NAME="${ONLINE_RUN_NAME:-robotwin_video_expert_multitask3_weight025_epoch003_5ep_20260903}"

case "${PHASE}" in
  all|backfill|train|eval|train-eval) ;;
  *) printf 'PHASE must be all, backfill, train, eval, or train-eval; got %s\n' "${PHASE}" >&2; exit 2 ;;
esac

for path in "${CHECKPOINT}" "${DATASET_STATS}" "${MODEL_BASE_PATH}" \
  "${REWARD_JSON}" "${SEED_MANIFEST_PATH}" "${CONFIG}"; do
  [[ -e "${path}" ]] || { printf '[video-expert-multitask3] missing: %s\n' "${path}" >&2; exit 1; }
done

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/driver.log") 2>&1

run_backfill_and_replay() {
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/backfill_video_expert_features.py" \
    --reward-json "${REWARD_JSON}" --output-dir "${BACKFILL_DIR}" \
    --checkpoint "${CHECKPOINT}" --dataset-stats "${DATASET_STATS}" \
    --model-base-path "${MODEL_BASE_PATH}" --tasks "${TASKS}" \
    --device cuda --mixed-precision bf16 --num-inference-steps 10

  if [[ ! -s "${REPLAY_DIR}/manifest.json" ]]; then
    [[ ! -e "${REPLAY_DIR}" ]] || {
      printf '[video-expert-multitask3] refusing incomplete replay: %s\n' "${REPLAY_DIR}" >&2
      exit 1
    }
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
      "${PROJECT_ROOT}/experiments/robotwin/build_wan_vae_head_awr_replay.py" \
      --reward-json "${REWARD_JSON}" --output-dir "${REPLAY_DIR}" \
      --actor-observation-source fastwam_video_expert \
      --observation-encoder-version "${FEATURE_VERSION}" \
      --reward-config "${CONFIG}" --tasks "${TASKS}" \
      --minimum-pairwise-accuracy 0.90
  fi
}

run_train() {
  if [[ ! -s "${TRAIN_DIR}/checkpoint.pt" || ! -s "${TRAIN_DIR}/history.json" ]]; then
    [[ ! -e "${TRAIN_DIR}" ]] || {
      printf '[video-expert-multitask3] refusing incomplete training output: %s\n' "${TRAIN_DIR}" >&2
      exit 1
    }
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
      "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
      --config "${CONFIG}" --replay-dir "${REPLAY_DIR}" \
      --output-dir "${TRAIN_DIR}" --seed "${SEED}" \
      --timeout-bootstrap-value 0.0
  fi
}

run_eval() {
  env \
    RUN_NAME="${ONLINE_RUN_NAME}" VARIANTS=imagination \
    TASKS="${TASKS}" EPISODES=5 BASE_SEED=47 TRIAL_OFFSET=0 \
    INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
    TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
    PAPER_ALIGNED=true STRICT_PAIRED=true \
    DETERMINISTIC_INSTRUCTION_BY_SEED=true EXPERT_CHECK=true \
    SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH}" \
    NO_IMAGINATION_CHECKPOINT="${TRAIN_DIR}/checkpoint.pt" \
    IMAGINATION_CHECKPOINT="${TRAIN_DIR}/checkpoint.pt" \
    RESIDUAL_ENCODER_PATH=none \
    RESIDUAL_ENCODER_VERSION="${FEATURE_VERSION}" \
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

case "${PHASE}" in
  all)
    run_backfill_and_replay
    run_train
    run_eval
    ;;
  backfill) run_backfill_and_replay ;;
  train) run_train ;;
  eval) run_eval ;;
  train-eval)
    run_train
    run_eval
    ;;
esac

touch "${RUN_ROOT}/${PHASE^^}_COMPLETE"
printf '[video-expert-multitask3] phase=%s complete run_root=%s\n' "${PHASE}" "${RUN_ROOT}"
