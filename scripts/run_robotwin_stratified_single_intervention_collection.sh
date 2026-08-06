#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-robotwin_stratified_single_intervention_20260806}"
SEEDS_CSV="${SEEDS:-4800001,4800002}"
MAX_CANDIDATES_PER_SEED="${MAX_CANDIDATES_PER_SEED:-4}"
Q_MARGIN="${Q_MARGIN:-0.003}"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_corrected_posttrain_20260805}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/iql_corrected_imagination_20epoch_paired_gate/checkpoint.pt}"
SUPPORT_INDEX="${SUPPORT_INDEX:-${RUN_ROOT}/support_index_corrected_q95}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
OUTPUT_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_residual_pairs/${RUN_NAME}"

for path in "${CHECKPOINT}" "${SUPPORT_INDEX}/metadata.json"; do
  [[ -f "${path}" ]] || { printf 'Missing artifact: %s\n' "${path}" >&2; exit 1; }
done
[[ "${MAX_CANDIDATES_PER_SEED}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'MAX_CANDIDATES_PER_SEED must be positive\n' >&2
  exit 1
}

IFS=',' read -r -a seeds <<< "${SEEDS_CSV}"
for raw_seed in "${seeds[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  [[ "${seed}" =~ ^[0-9]+$ ]] || { printf 'Invalid seed: %s\n' "${raw_seed}" >&2; exit 1; }
  manifest="${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_hanging_mug_seed${seed}_20260805.json"
  [[ -f "${manifest}" ]] || { printf 'Missing seed manifest: %s\n' "${manifest}" >&2; exit 1; }

  shadow_name="${RUN_NAME}_seed${seed}_shadow"
  printf '[stratified-pairs] stage=shadow seed=%s run=%s\n' "${seed}" "${shadow_name}"
  env \
    RUN_NAME="${shadow_name}" \
    VARIANTS=imagination \
    TASKS=hanging_mug \
    EPISODES=1 \
    IMAGINATION_CHECKPOINT="${CHECKPOINT}" \
    NO_IMAGINATION_CHECKPOINT="${CHECKPOINT}" \
    RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-local-20260803 \
    RESIDUAL_Q_GATE_ENABLED=true \
    RESIDUAL_Q_GATE_MARGIN="${Q_MARGIN}" \
    RESIDUAL_Q_GATE_MAX_DISAGREEMENT=0.05 \
    RESIDUAL_Q_GATE_CRITIC_SOURCE=target \
    RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
    RESIDUAL_SUPPORT_INDEX_PATH="${SUPPORT_INDEX}" \
    RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=true \
    RESIDUAL_SHADOW_MODE=true \
    RESIDUAL_INTERVENTION_REPLANS=all \
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
    EVAL_VIDEO_LOG=false \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"

  shadow_dir="${RESULT_BASE}/${shadow_name}_imagination"
  seed_output="${OUTPUT_ROOT}/seed${seed}"
  plan_json="${seed_output}/candidate_plan.json"
  plan_tsv="${seed_output}/candidate_plan.tsv"
  conda run --no-capture-output -n "${CONDA_ENV}" python \
    "${PROJECT_ROOT}/experiments/robotwin/select_single_intervention_candidates.py" \
    --input-dir "${shadow_dir}" \
    --q-margin "${Q_MARGIN}" \
    --max-per-episode "${MAX_CANDIDATES_PER_SEED}" \
    --output-json "${plan_json}" \
    --output-tsv "${plan_tsv}"

  while IFS=$'\t' read -r task environment_seed trial_idx replan stratum; do
    [[ "${task}" == "task_name" ]] && continue
    [[ "${environment_seed}" == "${seed}" ]] || {
      printf 'Candidate seed mismatch: expected=%s actual=%s\n' "${seed}" "${environment_seed}" >&2
      exit 1
    }
    force_name="${RUN_NAME}_seed${seed}_${stratum}_replan${replan}"
    printf '[stratified-pairs] stage=force seed=%s replan=%s stratum=%s\n' \
      "${seed}" "${replan}" "${stratum}"
    env \
      RUN_NAME="${force_name}" \
      VARIANTS=imagination \
      TASKS="${task}" \
      EPISODES=1 \
      IMAGINATION_CHECKPOINT="${CHECKPOINT}" \
      NO_IMAGINATION_CHECKPOINT="${CHECKPOINT}" \
      RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-local-20260803 \
      RESIDUAL_Q_GATE_ENABLED=false \
      RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
      RESIDUAL_SUPPORT_INDEX_PATH=none \
      RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
      RESIDUAL_SHADOW_MODE=false \
      RESIDUAL_INTERVENTION_REPLANS="${replan}" \
      RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=1 \
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
      EVAL_VIDEO_LOG=false \
      bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"

    force_dir="${RESULT_BASE}/${force_name}_imagination"
    pair_dir="${seed_output}/${stratum}_replan${replan}"
    conda run --no-capture-output -n "${CONDA_ENV}" python \
      "${PROJECT_ROOT}/experiments/robotwin/build_single_intervention_pairs.py" \
      --baseline-dir "${shadow_dir}" \
      --baseline-action-mode residual \
      --residual-dir "${force_dir}" \
      --intervention-replans "${replan}" \
      --output-dir "${pair_dir}"
  done < "${plan_tsv}"
done

printf 'Stratified single-intervention collection complete: %s\n' "${OUTPUT_ROOT}"
