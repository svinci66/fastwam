#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_corrected_posttrain_20260805}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/iql_corrected_imagination_20epoch_paired_gate/checkpoint.pt}"
SUPPORT_INDEX="${SUPPORT_INDEX:-${RUN_ROOT}/support_index_corrected_q95}"
SIGLIP_PATH="${SIGLIP_PATH:-/home/ubuntu/sj/fastwam/checkpoints/siglip-so400m-patch14-384-modelscope}"
RUN_PREFIX="${RUN_PREFIX:-robotwin_hanging_mug_corrected_qood_margin003_pair}"
SEEDS="${SEEDS:-4800001 4800003}"
VARIANTS="${VARIANTS:-baseline,imagination}"
EPISODES="${EPISODES:-1}"

[[ -f "${CHECKPOINT}" ]] || {
  printf 'Missing corrected checkpoint: %s\n' "${CHECKPOINT}" >&2
  exit 1
}
[[ -f "${SUPPORT_INDEX}/metadata.json" ]] || {
  printf 'Missing corrected support index: %s\n' "${SUPPORT_INDEX}" >&2
  exit 1
}
[[ -d "${SIGLIP_PATH}" ]] || {
  printf 'Missing SigLIP directory: %s\n' "${SIGLIP_PATH}" >&2
  exit 1
}

for seed in ${SEEDS}; do
  manifest="${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_hanging_mug_seed${seed}_20260805.json"
  [[ -f "${manifest}" ]] || {
    printf 'Missing fixed-seed manifest: %s\n' "${manifest}" >&2
    exit 1
  }

  env \
    RUN_NAME="${RUN_PREFIX}_seed${seed}" \
    VARIANTS="${VARIANTS}" \
    TASKS=hanging_mug \
    EPISODES="${EPISODES}" \
    IMAGINATION_CHECKPOINT="${CHECKPOINT}" \
    NO_IMAGINATION_CHECKPOINT="${CHECKPOINT}" \
    SIGLIP_PATH="${SIGLIP_PATH}" \
    RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-local-20260803 \
    RESIDUAL_Q_GATE_ENABLED=true \
    RESIDUAL_Q_GATE_MARGIN=0.003 \
    RESIDUAL_Q_GATE_MAX_DISAGREEMENT=0.05 \
    RESIDUAL_Q_GATE_CRITIC_SOURCE=target \
    RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
    RESIDUAL_SUPPORT_INDEX_PATH="${SUPPORT_INDEX}" \
    RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=true \
    RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=true \
    RESIDUAL_OUTCOME_CONFIRMATION_MIN_PROGRESS=0.0 \
    RESIDUAL_OUTCOME_CONFIRMATION_REANCHOR_REPLANS=1 \
    RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
    RESIDUAL_LANGUAGE_MODE=training_canonical \
    SAVE_RESIDUAL_TRANSITIONS=true \
    SAVE_BASELINE_TRANSITIONS=false \
    PAPER_ALIGNED=true \
    STRICT_PAIRED=true \
    SEED_MANIFEST_PATH="${manifest}" \
    INSTRUCTION_MODE=official \
    INSTRUCTION_TYPE=unseen \
    INFERENCE_STEPS=10 \
    REPLAN_STEPS=24 \
    TEXT_CFG_SCALE=1.0 \
    EVAL_VIDEO_LOG=true \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
done

printf 'Corrected Q+OOD margin-0.003 paired validation complete. Prefix: %s\n' "${RUN_PREFIX}"
