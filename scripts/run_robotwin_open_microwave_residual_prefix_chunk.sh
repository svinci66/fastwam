#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
RUN_NAME="${RUN_NAME:-robotwin_open_microwave_residual_prefix_formal_20260831_ep04}"
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
  EVAL_VIDEO_LOG=true TILED=false
  CAPTURE_DECODE_TILED="${CAPTURE_DECODE_TILED:-false}"
  SAVE_BASELINE_TRANSITIONS=false SAVE_RESIDUAL_TRANSITIONS=true
)

run_prefix_branch() {
  local branch_name="$1" intervention_replans="$2"
  local override_checkpoint="$3" override_replans="$4"
  printf '[residual-prefix-chunk] start branch=%s interventions=%s override=%s\n' \
    "${branch_name}" "${intervention_replans}" "${override_replans}"
  timeout --verbose --signal=TERM --kill-after=30s 2400s \
    env "${common_env[@]}" RUN_NAME="${RUN_NAME}_${branch_name}" \
      VARIANTS=no_imagination RESIDUAL_SHADOW_MODE=false \
      RESIDUAL_INTERVENTION_REPLANS="${intervention_replans}" \
      RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
      RESIDUAL_ACTOR_OVERRIDE_CHECKPOINT="${override_checkpoint}" \
      RESIDUAL_ACTOR_OVERRIDE_REPLANS="${override_replans}" \
      bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
  printf '[residual-prefix-chunk] complete branch=%s\n' "${branch_name}"
}

reference_root="${RESULT_BASE}/${RUN_NAME}_reference_no_imagination/open_microwave/imagination_transitions/open_microwave/residual/${EPISODE_DIR}"
baseline_root="${RESULT_BASE}/${RUN_NAME}_baseline_at_anchor_no_imagination/open_microwave/imagination_transitions/open_microwave/residual/${EPISODE_DIR}"
imagination_root="${RESULT_BASE}/${RUN_NAME}_imagination_at_anchor_no_imagination/open_microwave/imagination_transitions/open_microwave/residual/${EPISODE_DIR}"

# The ordinary residual actor creates the prefix state and also supplies the
# ordinary-residual branch at the selected chunk.
run_prefix_branch reference all none none

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/select_open_microwave_chunk_anchor.py" \
  --transition-root "${reference_root}" --target-open-ratio "${TARGET_OPEN_RATIO}" \
  --output-json "${ARTIFACT_DIR}/anchor.json" \
  --output-replan "${ARTIFACT_DIR}/anchor_replan.txt"
read -r anchor_replan < "${ARTIFACT_DIR}/anchor_replan.txt"
[[ "${anchor_replan}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'Formal residual-prefix anchor must be after replan 0, got %s\n' \
    "${anchor_replan}" >&2
  exit 3
}
prefix_replans="$(seq -s, 0 "$((anchor_replan - 1))")"
through_target="${prefix_replans},${anchor_replan}"
printf '[residual-prefix-chunk] selected replan=%s prefix=%s\n' \
  "${anchor_replan}" "${prefix_replans}"

# Baseline-at-anchor keeps the identical ordinary-residual prefix but disables
# the residual only for the selected chunk.
run_prefix_branch baseline_at_anchor "${prefix_replans}" none none

# The imagination branch keeps the same ordinary-residual prefix and hot-swaps
# only the selected chunk's actor.  The shared SigLIP encoder avoids duplicate
# vision-model memory.
run_prefix_branch imagination_at_anchor "${through_target}" \
  "${WEIGHT_ROOT}/with_imagination/checkpoint.pt" "${anchor_replan}"

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_open_microwave_chunk_triplet.py" \
  --baseline-root "${baseline_root}" \
  --no-imagination-root "${reference_root}" \
  --imagination-root "${imagination_root}" \
  --intervention-replan "${anchor_replan}" \
  --prefix-replans all_before_target --require-imagination-override \
  --output-json "${ARTIFACT_DIR}/triplet_audit.json" --require-accepted

touch "${ARTIFACT_DIR}/COMPLETE"
printf '[residual-prefix-chunk] complete audit=%s\n' \
  "${ARTIFACT_DIR}/triplet_audit.json"
