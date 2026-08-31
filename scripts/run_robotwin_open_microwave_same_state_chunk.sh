#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
RUN_NAME="${RUN_NAME:-robotwin_open_microwave_same_state_chunk_smoke_20260830}"
TRIAL_OFFSET="${TRIAL_OFFSET:-4}"
TARGET_OPEN_RATIO="${TARGET_OPEN_RATIO:-0.5}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_open_microwave_discordant7_diagnostic_20260830.json}"
WEIGHT_ROOT="${WEIGHT_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_open_microwave_wan_head_weight_sweep_20260827/weight025/training/seed42}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
ARTIFACT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${RUN_NAME}"
EPISODE_DIR="episode_$(printf '%04d' "${TRIAL_OFFSET}")"

[[ "${TRIAL_OFFSET}" =~ ^[0-6]$ ]] || {
  printf 'TRIAL_OFFSET must select one of the seven diagnostic episodes (0..6)\n' >&2
  exit 2
}
for path in "${MANIFEST}" \
  "${WEIGHT_ROOT}/no_imagination/checkpoint.pt" \
  "${WEIGHT_ROOT}/with_imagination/checkpoint.pt"; do
  [[ -f "${path}" ]] || { printf 'Missing file: %s\n' "${path}" >&2; exit 2; }
done
mkdir -p "${ARTIFACT_DIR}"
exec > >(tee -a "${ARTIFACT_DIR}/driver.log") 2>&1

common_env=(
  TASKS=open_microwave EPISODES=1 BASE_SEED=47 TRIAL_OFFSET="${TRIAL_OFFSET}"
  INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0
  TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official
  PAPER_ALIGNED=true STRICT_PAIRED=true DETERMINISTIC_INSTRUCTION_BY_SEED=true
  EXPERT_CHECK=true SEED_MANIFEST_PATH="${MANIFEST}"
  NO_IMAGINATION_CHECKPOINT="${WEIGHT_ROOT}/no_imagination/checkpoint.pt"
  IMAGINATION_CHECKPOINT="${WEIGHT_ROOT}/with_imagination/checkpoint.pt"
  RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-modelscope-local-v1
  RESIDUAL_LANGUAGE_MODE=policy_instruction RESIDUAL_Q_GATE_ENABLED=false
  RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false RESIDUAL_SUPPORT_INDEX_PATH=none
  RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false RESIDUAL_SOFT_SCALE_ENABLED=false
  RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=false ACTION_HOLD_PROBABILITY=0.0
  EVAL_VIDEO_LOG=true TILED=false CAPTURE_DECODE_TILED=false
)

run_variant() {
  local variant="$1" intervention="$2" max_interventions="$3"
  printf '[same-state-chunk] start variant=%s intervention=%s\n' \
    "${variant}" "${intervention}"
  timeout --verbose --signal=TERM --kill-after=30s 2400s \
    env "${common_env[@]}" RUN_NAME="${RUN_NAME}" VARIANTS="${variant}" \
      RESIDUAL_SHADOW_MODE=false RESIDUAL_INTERVENTION_REPLANS="${intervention}" \
      RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE="${max_interventions}" \
      SAVE_BASELINE_TRANSITIONS=true SAVE_RESIDUAL_TRANSITIONS=true \
      bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
  printf '[same-state-chunk] complete variant=%s\n' "${variant}"
}

baseline_root="${RESULT_BASE}/${RUN_NAME}_baseline/open_microwave/imagination_transitions/open_microwave/policy/${EPISODE_DIR}"
no_imagination_root="${RESULT_BASE}/${RUN_NAME}_no_imagination/open_microwave/imagination_transitions/open_microwave/residual/${EPISODE_DIR}"
imagination_root="${RESULT_BASE}/${RUN_NAME}_imagination/open_microwave/imagination_transitions/open_microwave/residual/${EPISODE_DIR}"

run_variant baseline all none

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/select_open_microwave_chunk_anchor.py" \
  --transition-root "${baseline_root}" --target-open-ratio "${TARGET_OPEN_RATIO}" \
  --output-json "${ARTIFACT_DIR}/anchor.json" \
  --output-replan "${ARTIFACT_DIR}/anchor_replan.txt"
read -r anchor_replan < "${ARTIFACT_DIR}/anchor_replan.txt"
[[ "${anchor_replan}" =~ ^[0-9]+$ ]] || {
  printf 'Invalid selected anchor replan: %s\n' "${anchor_replan}" >&2
  exit 3
}
printf '[same-state-chunk] selected replan=%s target_ratio=%s\n' \
  "${anchor_replan}" "${TARGET_OPEN_RATIO}"

run_variant no_imagination "${anchor_replan}" 1
run_variant imagination "${anchor_replan}" 1

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_open_microwave_chunk_triplet.py" \
  --baseline-root "${baseline_root}" \
  --no-imagination-root "${no_imagination_root}" \
  --imagination-root "${imagination_root}" \
  --intervention-replan "${anchor_replan}" \
  --output-json "${ARTIFACT_DIR}/triplet_audit.json" --require-accepted

touch "${ARTIFACT_DIR}/COMPLETE"
printf '[same-state-chunk] complete audit=%s\n' "${ARTIFACT_DIR}/triplet_audit.json"
