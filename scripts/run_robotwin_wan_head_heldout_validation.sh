#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/sj/fastwam/RoboTwin-upstream}"
POOL="${POOL:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_open_microwave_heldout_candidate_pool_20260828.json}"
WEIGHT_ROOT="${WEIGHT_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_open_microwave_wan_head_weight_sweep_20260827/weight025/training/seed42}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_open_microwave_wan_head_heldout10_20260828}"
BUNDLE="${ARTIFACT_ROOT}/expert_feasibility/open_microwave"
MANIFEST="${ARTIFACT_ROOT}/heldout_manifest.json"
RUN_NAME="${RUN_NAME:-robotwin_open_microwave_wan_head_weight025_heldout10_20260828}"
SUMMARY="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}/summary.json"
COUNT=10
START_SEED=4800100
END_SEED=4800129

mkdir -p "${ARTIFACT_ROOT}" "${BUNDLE}"
exec > >(tee -a "${ARTIFACT_ROOT}/driver.log") 2>&1

next_seed="${START_SEED}"
for (( episode=0; episode<COUNT; episode++ )); do
  marker="${ARTIFACT_ROOT}/.expert_episode$(printf '%03d' "${episode}")_complete"
  metadata="${BUNDLE}/pair_metadata/episode${episode}.json"
  if [[ -f "${marker}" && -s "${metadata}" ]]; then
    seed="$(conda run --no-capture-output -n "${CONDA_ENV}" python -c \
      'import json,sys; print(int(json.load(open(sys.argv[1]))["seed"]))' "${metadata}")"
    next_seed="$(( seed + 1 ))"
    printf '[wan-head-heldout] skip expert-feasible episode=%s seed=%s\n' "${episode}" "${seed}"
    continue
  fi
  selected=false
  seed="${next_seed}"
  while (( seed <= END_SEED )); do
    printf '[wan-head-heldout] expert screen episode=%s candidate_seed=%s\n' "${episode}" "${seed}"
    if conda run --no-capture-output -n "${CONDA_ENV}" \
      env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
      python -u "${PROJECT_ROOT}/experiments/robotwin/collect_local_expert_pair_episode.py" \
      --robotwin-root "${ROBOTWIN_ROOT}" --task open_microwave --task-config demo_clean \
      --seed "${seed}" --episode-index "${episode}" --output-bundle "${BUNDLE}"; then
      selected=true
      break
    else
      status="$?"
      if [[ "${status}" -eq 20 || "${status}" -eq 21 ]]; then
        printf '[wan-head-heldout] reject infeasible seed=%s status=%s\n' "${seed}" "${status}"
        seed="$(( seed + 1 ))"
        continue
      fi
      exit "${status}"
    fi
  done
  [[ "${selected}" == true ]] || { printf 'Candidate pool exhausted\n' >&2; exit 1; }
  touch "${marker}"
  next_seed="$(( seed + 1 ))"
done

if [[ ! -s "${MANIFEST}" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_expert_feasible_seed_manifest.py" \
    --candidate-pool "${POOL}" --metadata-dir "${BUNDLE}/pair_metadata" \
    --task open_microwave --count "${COUNT}" --output-json "${MANIFEST}"
fi

env \
  RUN_NAME="${RUN_NAME}" VARIANTS=no_imagination,imagination \
  TASKS=open_microwave EPISODES="${COUNT}" BASE_SEED=47 TRIAL_OFFSET=0 \
  INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
  TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
  PAPER_ALIGNED=true STRICT_PAIRED=true \
  DETERMINISTIC_INSTRUCTION_BY_SEED=true EXPERT_CHECK=true \
  SEED_MANIFEST_PATH="${MANIFEST}" \
  NO_IMAGINATION_CHECKPOINT="${WEIGHT_ROOT}/no_imagination/checkpoint.pt" \
  IMAGINATION_CHECKPOINT="${WEIGHT_ROOT}/with_imagination/checkpoint.pt" \
  RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-modelscope-local-v1 \
  RESIDUAL_LANGUAGE_MODE=policy_instruction \
  RESIDUAL_Q_GATE_ENABLED=false RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
  RESIDUAL_SUPPORT_INDEX_PATH=none RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
  RESIDUAL_SHADOW_MODE=false RESIDUAL_INTERVENTION_REPLANS=all \
  RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
  RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=false RESIDUAL_SOFT_SCALE_ENABLED=false \
  SAVE_BASELINE_TRANSITIONS=false SAVE_RESIDUAL_TRANSITIONS=false EVAL_VIDEO_LOG=true \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_wan_head_heldout_pair.py" \
  --summary "${SUMMARY}" --output-json "${ARTIFACT_ROOT}/heldout_audit.json"

touch "${ARTIFACT_ROOT}/COMPLETE"
printf '[wan-head-heldout] complete: %s\n' "${ARTIFACT_ROOT}/heldout_audit.json"
