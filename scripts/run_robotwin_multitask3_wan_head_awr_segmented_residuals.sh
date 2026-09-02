#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
RUN_NAME="${RUN_NAME:-robotwin_wan_head_multitask3_awr_formal_block1_5ep_20260901}"
SEGMENT_PREFIX="${SEGMENT_PREFIX:-${RUN_NAME}_segmented}"
TASKS="${TASKS:-open_microwave,hanging_mug,place_can_basket}"
VARIANTS="${VARIANTS:-no_imagination,imagination}"
EPISODES="${EPISODES:-5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}/prevalidated_seed_manifest.json}"
CONTROL_DIR="${CONTROL_DIR:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_wan_head_multitask3_no_imagination_awr_seed42_20260901/training/formal}"
TREATMENT_DIR="${TREATMENT_DIR:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_wan_head_multitask3_awr_seed42_20260901/training/formal}"
AUDIT_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}"
CRASH_MARKER_DIR="${AUDIT_ROOT}/segmented_crash_failures"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"

[[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]] || { printf 'EPISODES must be positive\n' >&2; exit 1; }
[[ "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || { printf 'MAX_ATTEMPTS must be positive\n' >&2; exit 1; }
for path in "${SEED_MANIFEST_PATH}" "${CONTROL_DIR}/checkpoint.pt" "${TREATMENT_DIR}/checkpoint.pt"; do
  [[ -s "${path}" ]] || { printf '[segmented-residuals] missing: %s\n' "${path}" >&2; exit 1; }
done
mkdir -p "${CRASH_MARKER_DIR}"

IFS=',' read -r -a variants <<< "${VARIANTS}"
IFS=',' read -r -a tasks <<< "${TASKS}"
for variant in "${variants[@]}"; do
  for task in "${tasks[@]}"; do
    for ((offset = 0; offset < EPISODES; offset++)); do
      terminal_marker="${CRASH_MARKER_DIR}/${variant}__${task}__episode${offset}"
      segment_complete=false
      for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
        segment_run="${SEGMENT_PREFIX}__${variant}__${task}__episode${offset}__attempt${attempt}"
        run_dir="${RESULT_BASE}/${segment_run}_${variant}"
        complete_marker="${run_dir}/.${task}_1ep_complete"
        if [[ -f "${complete_marker}" ]]; then
          printf '[segmented-residuals] skip complete variant=%s task=%s episode=%s attempt=%s\n' \
            "${variant}" "${task}" "${offset}" "${attempt}"
          segment_complete=true
          break
        fi
        printf '[segmented-residuals] start variant=%s task=%s episode=%s attempt=%s/%s\n' \
          "${variant}" "${task}" "${offset}" "${attempt}" "${MAX_ATTEMPTS}"
        set +e
        env \
          RUN_NAME="${segment_run}" VARIANTS="${variant}" TASKS="${task}" \
          EPISODES=1 BASE_SEED=47 TRIAL_OFFSET="${offset}" \
          INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
          TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
          PAPER_ALIGNED=true STRICT_PAIRED=true \
          DETERMINISTIC_INSTRUCTION_BY_SEED=true EXPERT_CHECK=true \
          SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH}" \
          NO_IMAGINATION_CHECKPOINT="${CONTROL_DIR}/checkpoint.pt" \
          IMAGINATION_CHECKPOINT="${TREATMENT_DIR}/checkpoint.pt" \
          RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-modelscope-local-v1 \
          RESIDUAL_LANGUAGE_MODE=policy_instruction \
          RESIDUAL_Q_GATE_ENABLED=false \
          RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
          RESIDUAL_SUPPORT_INDEX_PATH=none \
          RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
          RESIDUAL_SHADOW_MODE=false \
          RESIDUAL_INTERVENTION_REPLANS=all \
          RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
          RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=false \
          RESIDUAL_SOFT_SCALE_ENABLED=false \
          SAVE_BASELINE_TRANSITIONS=false SAVE_RESIDUAL_TRANSITIONS=false \
          EVAL_VIDEO_LOG=true \
          bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
        return_code=$?
        set -e
        if [[ "${return_code}" == 0 && -f "${complete_marker}" ]]; then
          segment_complete=true
          break
        fi
        printf '[segmented-residuals] retry variant=%s task=%s episode=%s attempt=%s return_code=%s\n' \
          "${variant}" "${task}" "${offset}" "${attempt}" "${return_code}" >&2
      done
      if [[ "${segment_complete}" != true ]]; then
        touch "${terminal_marker}"
        printf '[segmented-residuals] classify_runtime_failure variant=%s task=%s episode=%s attempts=%s\n' \
          "${variant}" "${task}" "${offset}" "${MAX_ATTEMPTS}" >&2
      fi
    done
  done
done

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/summarize_multitask3_segmented_compare.py" \
  --result-base "${RESULT_BASE}" --baseline-run-name "${RUN_NAME}" \
  --segment-prefix "${SEGMENT_PREFIX}" --seed-manifest "${SEED_MANIFEST_PATH}" \
  --tasks "${TASKS}" --variants "${VARIANTS}" --episodes "${EPISODES}" \
  --max-attempts "${MAX_ATTEMPTS}" --crash-marker-dir "${CRASH_MARKER_DIR}" \
  --output-json "${AUDIT_ROOT}/summary.json"

touch "${AUDIT_ROOT}/SEGMENTED_RESIDUALS_COMPLETE"
printf '[segmented-residuals] complete summary=%s\n' "${AUDIT_ROOT}/summary.json"
