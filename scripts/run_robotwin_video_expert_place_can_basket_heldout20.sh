#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/sj/fastwam/RoboTwin-upstream}"
TASK="place_can_basket"
COUNT="${COUNT:-20}"
START_SEED="${START_SEED:-4800100}"
END_SEED="${END_SEED:-4800139}"
POOL="${POOL:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_place_can_basket_heldout_candidate_pool_20260903.json}"
TRAIN_ROOT="${TRAIN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_place_can_basket_single_task_epoch003_seed42_20260903/training/seed42}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_place_can_basket_heldout20_20260903}"
BUNDLE="${ARTIFACT_ROOT}/expert_feasibility/${TASK}"
MANIFEST="${ARTIFACT_ROOT}/heldout_manifest.json"
RUN_NAME="${RUN_NAME:-robotwin_video_expert_place_can_basket_heldout20_20260903}"
SUMMARY="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}/summary.json"
FEATURE_VERSION="fastwam_video_expert_final_token_mean_l2_v1"
MAX_ONLINE_ATTEMPTS="${MAX_ONLINE_ATTEMPTS:-5}"
CAPTURE_DECODE_TILED="${CAPTURE_DECODE_TILED:-false}"

[[ "${COUNT}" =~ ^[1-9][0-9]*$ ]] || { printf 'COUNT must be positive\n' >&2; exit 1; }
[[ "${START_SEED}" =~ ^[0-9]+$ ]] || { printf 'START_SEED must be non-negative\n' >&2; exit 1; }
[[ "${END_SEED}" =~ ^[0-9]+$ ]] || { printf 'END_SEED must be non-negative\n' >&2; exit 1; }
(( START_SEED <= END_SEED )) || { printf 'START_SEED must not exceed END_SEED\n' >&2; exit 1; }
for path in "${POOL}" "${TRAIN_ROOT}/no_imagination/checkpoint.pt" \
  "${TRAIN_ROOT}/with_imagination/checkpoint.pt"; do
  [[ -s "${path}" ]] || { printf '[place-can-heldout20] missing: %s\n' "${path}" >&2; exit 1; }
done

mkdir -p "${ARTIFACT_ROOT}" "${BUNDLE}"
exec > >(tee -a "${ARTIFACT_ROOT}/driver.log") 2>&1

next_seed="${START_SEED}"
for ((episode = 0; episode < COUNT; episode++)); do
  marker="${ARTIFACT_ROOT}/.expert_episode$(printf '%03d' "${episode}")_complete"
  metadata="${BUNDLE}/pair_metadata/episode${episode}.json"
  if [[ -f "${marker}" && -s "${metadata}" ]]; then
    seed="$(conda run --no-capture-output -n "${CONDA_ENV}" python -c \
      'import json,sys; print(int(json.load(open(sys.argv[1]))["seed"]))' "${metadata}")"
    next_seed="$((seed + 1))"
    printf '[place-can-heldout20] skip expert-feasible episode=%s seed=%s\n' \
      "${episode}" "${seed}"
    continue
  fi
  selected=false
  seed="${next_seed}"
  while (( seed <= END_SEED )); do
    printf '[place-can-heldout20] expert screen episode=%s candidate_seed=%s\n' \
      "${episode}" "${seed}"
    set +e
    conda run --no-capture-output -n "${CONDA_ENV}" \
      env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
      python -u "${PROJECT_ROOT}/experiments/robotwin/collect_local_expert_pair_episode.py" \
      --robotwin-root "${ROBOTWIN_ROOT}" --task "${TASK}" --task-config demo_clean \
      --seed "${seed}" --episode-index "${episode}" --output-bundle "${BUNDLE}"
    status=$?
    set -e
    if [[ "${status}" -eq 0 ]]; then
      selected=true
      break
    fi
    if [[ "${status}" -eq 20 || "${status}" -eq 21 ]]; then
      printf '[place-can-heldout20] reject infeasible seed=%s status=%s\n' \
        "${seed}" "${status}"
      seed="$((seed + 1))"
      continue
    fi
    exit "${status}"
  done
  [[ "${selected}" == true ]] || { printf 'Candidate pool exhausted\n' >&2; exit 1; }
  touch "${marker}"
  next_seed="$((seed + 1))"
done

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/build_expert_feasible_seed_manifest.py" \
  --candidate-pool "${POOL}" --metadata-dir "${BUNDLE}/pair_metadata" \
  --instructions-dir "${BUNDLE}/instructions" --task "${TASK}" \
  --count "${COUNT}" --output-json "${MANIFEST}"

run_online_pair() {
  env \
    RUN_NAME="${RUN_NAME}" VARIANTS=no_imagination,imagination \
    TASKS="${TASK}" EPISODES="${COUNT}" BASE_SEED=47 TRIAL_OFFSET=0 \
    INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
    TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
    PAPER_ALIGNED=true STRICT_PAIRED=true \
    DETERMINISTIC_INSTRUCTION_BY_SEED=true EXPERT_CHECK=true \
    SEED_MANIFEST_PATH="${MANIFEST}" \
    NO_IMAGINATION_CHECKPOINT="${TRAIN_ROOT}/no_imagination/checkpoint.pt" \
    IMAGINATION_CHECKPOINT="${TRAIN_ROOT}/with_imagination/checkpoint.pt" \
    RESIDUAL_ENCODER_PATH=none RESIDUAL_ENCODER_VERSION="${FEATURE_VERSION}" \
    RESIDUAL_LANGUAGE_MODE=policy_instruction \
    RESIDUAL_Q_GATE_ENABLED=false RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
    RESIDUAL_SUPPORT_INDEX_PATH=none RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
    RESIDUAL_SHADOW_MODE=false RESIDUAL_INTERVENTION_REPLANS=all \
    RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
    RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=false RESIDUAL_SOFT_SCALE_ENABLED=false \
    SAVE_BASELINE_TRANSITIONS=false SAVE_RESIDUAL_TRANSITIONS=false \
    EVAL_VIDEO_LOG=true CAPTURE_DECODE_TILED="${CAPTURE_DECODE_TILED}" \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
}

for ((attempt = 1; attempt <= MAX_ONLINE_ATTEMPTS; attempt++)); do
  printf '[place-can-heldout20] online attempt=%d/%d\n' \
    "${attempt}" "${MAX_ONLINE_ATTEMPTS}"
  if run_online_pair; then
    break
  fi
  if (( attempt == MAX_ONLINE_ATTEMPTS )); then
    printf '[place-can-heldout20] exhausted online attempts\n' >&2
    exit 1
  fi
done

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_wan_head_heldout_pair.py" \
  --summary "${SUMMARY}" --task "${TASK}" --expected-pairs "${COUNT}" \
  --output-json "${ARTIFACT_ROOT}/heldout_audit.json"

touch "${ARTIFACT_ROOT}/COMPLETE"
printf '[place-can-heldout20] complete: %s\n' "${ARTIFACT_ROOT}/heldout_audit.json"
