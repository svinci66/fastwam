#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-robotwin_single_intervention_pairs_smoke_20260806}"
TASKS="${TASKS:-hanging_mug}"
INTERVENTION_REPLANS="${INTERVENTION_REPLANS:-5}"
EPISODES="${EPISODES:-1}"
BASE_SEED="${BASE_SEED:-42}"
TRIAL_OFFSET="${TRIAL_OFFSET:-0}"
IMAGINATION_CHECKPOINT="${IMAGINATION_CHECKPOINT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_corrected_posttrain_20260805/iql_corrected_imagination_20epoch_paired_gate/checkpoint.pt}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
PAIR_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_residual_pairs/${RUN_NAME}"
COMPUTE_IMAGINATION_REWARD="${COMPUTE_IMAGINATION_REWARD:-false}"
ENCODER_PATH="${ENCODER_PATH:-/home/ubuntu/sj/fastwam/checkpoints/siglip-so400m-patch14-384-modelscope}"
REWARD_DEVICE="${REWARD_DEVICE:-cuda}"
REWARD_ENCODER_DTYPE="${REWARD_ENCODER_DTYPE:-bf16}"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
RESIDUAL_DEVICE="${RESIDUAL_DEVICE:-same}"
RESIDUAL_ENCODER_VERSION="${RESIDUAL_ENCODER_VERSION:-siglip-so400m-patch14-384-local-20260803}"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_4task_heldout5_expert_seeds_20260804.json}"

[[ -f "${IMAGINATION_CHECKPOINT}" ]] || {
  printf 'Missing residual checkpoint: %s\n' "${IMAGINATION_CHECKPOINT}" >&2
  exit 1
}
[[ -f "${SEED_MANIFEST_PATH}" ]] || {
  printf 'Missing strict seed manifest: %s\n' "${SEED_MANIFEST_PATH}" >&2
  exit 1
}
[[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'EPISODES must be positive\n' >&2
  exit 1
}

IFS=',' read -r -a replans <<< "${INTERVENTION_REPLANS}"
for raw_replan in "${replans[@]}"; do
  replan="${raw_replan//[[:space:]]/}"
  [[ "${replan}" =~ ^[0-9]+$ ]] || {
    printf 'Invalid intervention replan: %s\n' "${raw_replan}" >&2
    exit 1
  }
  segment_name="${RUN_NAME}_replan${replan}"
  printf '[single-intervention] task=%s replan=%s episodes=%s run=%s\n' \
    "${TASKS}" "${replan}" "${EPISODES}" "${segment_name}"

  env \
    RUN_NAME="${segment_name}" \
    VARIANTS=baseline,imagination \
    TASKS="${TASKS}" \
    EPISODES="${EPISODES}" \
    BASE_SEED="${BASE_SEED}" \
    TRIAL_OFFSET="${TRIAL_OFFSET}" \
    PAPER_ALIGNED=true \
    STRICT_PAIRED=true \
    INSTRUCTION_MODE=official \
    INSTRUCTION_TYPE=unseen \
    INFERENCE_STEPS=10 \
    REPLAN_STEPS=24 \
    TEXT_CFG_SCALE=1.0 \
    SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH}" \
    EVAL_VIDEO_LOG=false \
    TILED=false \
    CAPTURE_DECODE_TILED=true \
    RESIDUAL_DEVICE="${RESIDUAL_DEVICE}" \
    RESIDUAL_ENCODER_VERSION="${RESIDUAL_ENCODER_VERSION}" \
    IMAGINATION_CHECKPOINT="${IMAGINATION_CHECKPOINT}" \
    RESIDUAL_Q_GATE_ENABLED=false \
    RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
    RESIDUAL_SUPPORT_INDEX_PATH=none \
    RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
    RESIDUAL_SHADOW_MODE=false \
    RESIDUAL_INTERVENTION_REPLANS="${replan}" \
    RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=1 \
    SAVE_BASELINE_TRANSITIONS=true \
    SAVE_RESIDUAL_TRANSITIONS=true \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"

  baseline_dir="${RESULT_BASE}/${segment_name}_baseline"
  residual_dir="${RESULT_BASE}/${segment_name}_imagination"
  output_dir="${PAIR_ROOT}/replan${replan}"
  reward_args=()
  if [[ "${COMPUTE_IMAGINATION_REWARD}" == "true" ]]; then
    reward_dir="${output_dir}/reward_audit"
    conda run --no-capture-output -n "${CONDA_ENV}" python \
      "${PROJECT_ROOT}/experiments/robotwin/analyze_imagination_rewards.py" \
      --input-dir "${baseline_dir}" \
      --input-dir "${residual_dir}" \
      --encoder-path "${ENCODER_PATH}" \
      --encoder-dtype "${REWARD_ENCODER_DTYPE}" \
      --device "${REWARD_DEVICE}" \
      --batch-size 24 \
      --minimum-paired-trials 1 \
      --output-dir "${reward_dir}"
    reward_args=(--reward-jsonl "${reward_dir}/transition_rewards.jsonl")
  fi
  conda run --no-capture-output -n "${CONDA_ENV}" python \
    "${PROJECT_ROOT}/experiments/robotwin/build_single_intervention_pairs.py" \
    --baseline-dir "${baseline_dir}" \
    --residual-dir "${residual_dir}" \
    --intervention-replans "${replan}" \
    --output-dir "${output_dir}" \
    --require-accepted \
    "${reward_args[@]}"
done

printf 'Single-intervention pair collection complete: %s\n' "${PAIR_ROOT}"
