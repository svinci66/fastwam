#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/sj/fastwam/RoboTwin-upstream}"
SIGLIP_PATH="${SIGLIP_PATH:-/home/ubuntu/sj/fastwam/checkpoints/siglip-so400m-patch14-384-modelscope}"
RESIDUAL_ENCODER_PATH="${RESIDUAL_ENCODER_PATH:-${SIGLIP_PATH}}"
RESIDUAL_ROOT="${RESIDUAL_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_expert10_residual_iql_20260729}"
NO_IMAGINATION_CHECKPOINT="${NO_IMAGINATION_CHECKPOINT:-${RESIDUAL_ROOT}/iql_balanced_no_imagination/checkpoint.pt}"
IMAGINATION_CHECKPOINT="${IMAGINATION_CHECKPOINT:-${RESIDUAL_ROOT}/iql_balanced_imagination/checkpoint.pt}"
RESIDUAL_ENCODER_VERSION="${RESIDUAL_ENCODER_VERSION:-siglip-so400m-patch14-384-local-20260729}"
RUN_NAME="${RUN_NAME:-robotwin_residual_iql_online_pair_3task5ep_20260729}"
VARIANTS_CSV="${VARIANTS:-baseline,no_imagination,imagination}"
TASKS_CSV="${TASKS:-adjust_bottle,open_laptop,stack_blocks_two}"
EPISODES="${EPISODES:-5}"
BASE_SEED="${BASE_SEED:-42}"
TRIAL_OFFSET="${TRIAL_OFFSET:-0}"
INFERENCE_STEPS="${INFERENCE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
TEXT_CFG_SCALE="${TEXT_CFG_SCALE:-1.0}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-true}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
INSTRUCTION_MODE="${INSTRUCTION_MODE:-fixed}"
PAPER_ALIGNED="${PAPER_ALIGNED:-false}"
STRICT_PAIRED="${STRICT_PAIRED:-false}"
DETERMINISTIC_INSTRUCTION_BY_SEED="${DETERMINISTIC_INSTRUCTION_BY_SEED:-${STRICT_PAIRED}}"
EXPERT_CHECK="${EXPERT_CHECK:-true}"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-none}"
PHYSICS_AUDIT_ENABLED="${PHYSICS_AUDIT_ENABLED:-false}"
PHYSICS_AUDIT_ACTOR_ATTR="${PHYSICS_AUDIT_ACTOR_ATTR:-can}"
PHYSICS_AUDIT_OUTPUT_ROOT="${PHYSICS_AUDIT_OUTPUT_ROOT:-none}"
GPU_ID="${GPU_ID:-0}"
TILED="${TILED:-false}"
CAPTURE_DECODE_TILED="${CAPTURE_DECODE_TILED:-false}"
RESIDUAL_DEVICE="${RESIDUAL_DEVICE:-same}"
RESIDUAL_Q_GATE_ENABLED="${RESIDUAL_Q_GATE_ENABLED:-false}"
RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED="${RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED:-false}"
RESIDUAL_PAIRED_ADVANTAGE_THRESHOLD="${RESIDUAL_PAIRED_ADVANTAGE_THRESHOLD:-none}"
RESIDUAL_PAIRED_ADVANTAGE_MAX_DISAGREEMENT="${RESIDUAL_PAIRED_ADVANTAGE_MAX_DISAGREEMENT:-none}"
RESIDUAL_Q_GATE_MARGIN="${RESIDUAL_Q_GATE_MARGIN:-0.0}"
RESIDUAL_Q_GATE_MAX_DISAGREEMENT="${RESIDUAL_Q_GATE_MAX_DISAGREEMENT:-0.05}"
RESIDUAL_Q_GATE_RISK_SCALE="${RESIDUAL_Q_GATE_RISK_SCALE:-0.0}"
RESIDUAL_Q_GATE_RISK_DECAY="${RESIDUAL_Q_GATE_RISK_DECAY:-1.0}"
RESIDUAL_SOFT_SCALE_ENABLED="${RESIDUAL_SOFT_SCALE_ENABLED:-false}"
RESIDUAL_SOFT_SCALE_Q_FULL_ADVANTAGE="${RESIDUAL_SOFT_SCALE_Q_FULL_ADVANTAGE:-0.005}"
RESIDUAL_SOFT_SCALE_SUPPORT_FULL_MARGIN="${RESIDUAL_SOFT_SCALE_SUPPORT_FULL_MARGIN:-0.25}"
RESIDUAL_Q_GATE_CRITIC_SOURCE="${RESIDUAL_Q_GATE_CRITIC_SOURCE:-target}"
RESIDUAL_SUPPORT_INDEX_PATH="${RESIDUAL_SUPPORT_INDEX_PATH:-none}"
RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED="${RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED:-true}"
RESIDUAL_SHADOW_MODE="${RESIDUAL_SHADOW_MODE:-false}"
RESIDUAL_INTERVENTION_REPLANS="${RESIDUAL_INTERVENTION_REPLANS:-all}"
RESIDUAL_ACTOR_OVERRIDE_CHECKPOINT="${RESIDUAL_ACTOR_OVERRIDE_CHECKPOINT:-none}"
RESIDUAL_ACTOR_OVERRIDE_REPLANS="${RESIDUAL_ACTOR_OVERRIDE_REPLANS:-none}"
RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE="${RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE:-none}"
RESIDUAL_OUTCOME_CONFIRMATION_ENABLED="${RESIDUAL_OUTCOME_CONFIRMATION_ENABLED:-false}"
RESIDUAL_OUTCOME_CONFIRMATION_MIN_PROGRESS="${RESIDUAL_OUTCOME_CONFIRMATION_MIN_PROGRESS:-0.0}"
RESIDUAL_OUTCOME_CONFIRMATION_REANCHOR_REPLANS="${RESIDUAL_OUTCOME_CONFIRMATION_REANCHOR_REPLANS:-1}"
RESIDUAL_LANGUAGE_MODE="${RESIDUAL_LANGUAGE_MODE:-policy_instruction}"
SAVE_RESIDUAL_TRANSITIONS="${SAVE_RESIDUAL_TRANSITIONS:-false}"
SAVE_BASELINE_TRANSITIONS="${SAVE_BASELINE_TRANSITIONS:-false}"
ACTION_HOLD_PROBABILITY="${ACTION_HOLD_PROBABILITY:-0.0}"
ACTION_HOLD_REPLANS="${ACTION_HOLD_REPLANS:-all}"
ACTION_CORRUPTION_SEED="${ACTION_CORRUPTION_SEED:-20260729}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
SUMMARY_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}"

for path in "${CHECKPOINT}" "${DATASET_STATS}" \
  "${NO_IMAGINATION_CHECKPOINT}" "${IMAGINATION_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { printf 'Missing file: %s\n' "${path}" >&2; exit 1; }
done
for path in "${ROBOTWIN_ROOT}"; do
  [[ -d "${path}" ]] || { printf 'Missing directory: %s\n' "${path}" >&2; exit 1; }
done
if [[ "${RESIDUAL_ENCODER_PATH}" != "none" && "${RESIDUAL_ENCODER_PATH}" != "null" ]]; then
  [[ -d "${RESIDUAL_ENCODER_PATH}" ]] || {
    printf 'Missing residual encoder directory: %s\n' "${RESIDUAL_ENCODER_PATH}" >&2
    exit 1
  }
fi
[[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]] || { printf 'EPISODES must be positive\n' >&2; exit 1; }
[[ "${INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]] || { printf 'INFERENCE_STEPS must be positive\n' >&2; exit 1; }
[[ "${REPLAN_STEPS}" =~ ^[1-9][0-9]*$ ]] || { printf 'REPLAN_STEPS must be positive\n' >&2; exit 1; }
if [[ "${INSTRUCTION_MODE}" != "fixed" && "${INSTRUCTION_MODE}" != "official" ]]; then
  printf 'INSTRUCTION_MODE must be fixed or official\n' >&2
  exit 1
fi
if [[ "${RESIDUAL_LANGUAGE_MODE}" != "policy_instruction" \
  && "${RESIDUAL_LANGUAGE_MODE}" != "training_canonical" ]]; then
  printf 'RESIDUAL_LANGUAGE_MODE must be policy_instruction or training_canonical\n' >&2
  exit 1
fi
if [[ "${STRICT_PAIRED}" == "true" ]]; then
  [[ "${PAPER_ALIGNED}" == "true" ]] || {
    printf 'STRICT_PAIRED=true requires PAPER_ALIGNED=true\n' >&2
    exit 1
  }
  [[ -f "${SEED_MANIFEST_PATH}" ]] || {
    printf 'Strict seed manifest not found: %s\n' "${SEED_MANIFEST_PATH}" >&2
    exit 1
  }
fi
if [[ "${PAPER_ALIGNED}" == "true" ]]; then
  [[ "${INFERENCE_STEPS}" == "10" ]] || { printf 'Paper-aligned runs require INFERENCE_STEPS=10\n' >&2; exit 1; }
  [[ "${REPLAN_STEPS}" == "24" ]] || { printf 'Paper-aligned runs require REPLAN_STEPS=24\n' >&2; exit 1; }
  [[ "${TEXT_CFG_SCALE}" == "1.0" ]] || { printf 'Paper-aligned runs require TEXT_CFG_SCALE=1.0\n' >&2; exit 1; }
  [[ "${INSTRUCTION_TYPE}" == "unseen" ]] || { printf 'Paper-aligned runs require INSTRUCTION_TYPE=unseen\n' >&2; exit 1; }
  [[ "${INSTRUCTION_MODE}" == "official" ]] || { printf 'Paper-aligned runs require INSTRUCTION_MODE=official\n' >&2; exit 1; }
fi

instruction_for_task() {
  case "$1" in
    adjust_bottle)
      printf '%s' 'Pick up the bottle from the table and keep it upright.'
      ;;
    blocks_ranking_size)
      printf '%s' 'Rank the blocks by size.'
      ;;
    hanging_mug)
      printf '%s' 'Hang the mug on the mug rack.'
      ;;
    open_microwave)
      printf '%s' 'Use one arm to open the microwave.'
      ;;
    place_can_basket)
      printf '%s' 'Pick up the can, put it into the basket, then lift the basket.'
      ;;
    open_laptop)
      printf '%s' 'Open the laptop completely.'
      ;;
    stack_blocks_two)
      printf '%s' \
        'Move the red and green blocks to the center and stack the green block on the red block.'
      ;;
    *)
      printf 'No fixed instruction configured for task: %s\n' "$1" >&2
      return 1
      ;;
  esac
}

IFS=',' read -r -a variants <<< "${VARIANTS_CSV}"
IFS=',' read -r -a task_names <<< "${TASKS_CSV}"
environment_start_seed="${ENVIRONMENT_START_SEED:-$(( 100000 * (1 + BASE_SEED) + TRIAL_OFFSET ))}"
mkdir -p "${SUMMARY_DIR}"

completed_log_for_task() {
  local run_dir="$1"
  local task_name="$2"
  local latest_log
  latest_log="$(find "${run_dir}" -maxdepth 1 -type f \
    -name "eval_${task_name}_*.log" -print | sort | tail -n 1)"
  [[ -n "${latest_log}" ]] || return 1
  rg -q 'Success rate:' "${latest_log}" || return 1
}

for variant in "${variants[@]}"; do
  case "${variant}" in
    baseline) residual_checkpoint=none ;;
    no_imagination) residual_checkpoint="${NO_IMAGINATION_CHECKPOINT}" ;;
    imagination) residual_checkpoint="${IMAGINATION_CHECKPOINT}" ;;
    *) printf 'Unsupported variant: %s\n' "${variant}" >&2; exit 1 ;;
  esac
  run_dir="${RESULT_BASE}/${RUN_NAME}_${variant}"
  mkdir -p "${run_dir}"
  for task_name in "${task_names[@]}"; do
    marker="${run_dir}/.${task_name}_${EPISODES}ep_complete"
    if [[ -f "${marker}" ]] && completed_log_for_task "${run_dir}" "${task_name}"; then
      printf '[robotwin-online-pair] skip complete variant=%s task=%s\n' \
        "${variant}" "${task_name}"
      continue
    fi
    instruction_args=()
    if [[ "${INSTRUCTION_MODE}" == "fixed" ]]; then
      instruction="$(instruction_for_task "${task_name}")"
      instruction_args=("EVALUATION.fixed_instruction='${instruction}'")
    else
      instruction_args=("EVALUATION.fixed_instruction=null")
    fi
    action_mode=policy
    save_transitions="${SAVE_BASELINE_TRANSITIONS}"
    residual_args=()
    physics_audit_output_dir="${PHYSICS_AUDIT_OUTPUT_ROOT}"
    if [[ "${PHYSICS_AUDIT_ENABLED}" == "true" && "${PHYSICS_AUDIT_OUTPUT_ROOT}" != "none" ]]; then
      physics_audit_output_dir="${PHYSICS_AUDIT_OUTPUT_ROOT}/${variant}/${task_name}"
    fi
    if [[ "${variant}" != "baseline" ]]; then
      action_mode=residual
      save_transitions="${SAVE_RESIDUAL_TRANSITIONS}"
      residual_language_instruction=null
      if [[ "${RESIDUAL_LANGUAGE_MODE}" == "training_canonical" ]]; then
        residual_language_instruction="$(instruction_for_task "${task_name}")"
      fi
      residual_args=(
        "EVALUATION.residual_checkpoint=${residual_checkpoint}"
        "EVALUATION.residual_encoder_path=${RESIDUAL_ENCODER_PATH}"
        "EVALUATION.residual_encoder_version=${RESIDUAL_ENCODER_VERSION}"
        "EVALUATION.residual_encoder_dtype=bf16"
        "EVALUATION.residual_device=${RESIDUAL_DEVICE}"
        "EVALUATION.residual_q_gate_enabled=${RESIDUAL_Q_GATE_ENABLED}"
        "EVALUATION.residual_paired_advantage_gate_enabled=${RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED}"
        "EVALUATION.residual_paired_advantage_threshold=${RESIDUAL_PAIRED_ADVANTAGE_THRESHOLD}"
        "EVALUATION.residual_paired_advantage_max_disagreement=${RESIDUAL_PAIRED_ADVANTAGE_MAX_DISAGREEMENT}"
        "EVALUATION.residual_q_gate_margin=${RESIDUAL_Q_GATE_MARGIN}"
        "EVALUATION.residual_q_gate_max_disagreement=${RESIDUAL_Q_GATE_MAX_DISAGREEMENT}"
        "EVALUATION.residual_q_gate_risk_scale=${RESIDUAL_Q_GATE_RISK_SCALE}"
        "EVALUATION.residual_q_gate_risk_decay=${RESIDUAL_Q_GATE_RISK_DECAY}"
        "EVALUATION.residual_soft_scale_enabled=${RESIDUAL_SOFT_SCALE_ENABLED}"
        "EVALUATION.residual_soft_scale_q_full_advantage=${RESIDUAL_SOFT_SCALE_Q_FULL_ADVANTAGE}"
        "EVALUATION.residual_soft_scale_support_full_margin=${RESIDUAL_SOFT_SCALE_SUPPORT_FULL_MARGIN}"
        "EVALUATION.residual_q_gate_critic_source=${RESIDUAL_Q_GATE_CRITIC_SOURCE}"
        "EVALUATION.residual_support_index_path=${RESIDUAL_SUPPORT_INDEX_PATH}"
        "EVALUATION.residual_support_circuit_breaker_enabled=${RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED}"
        "EVALUATION.residual_shadow_mode=${RESIDUAL_SHADOW_MODE}"
        "EVALUATION.residual_intervention_replans='${RESIDUAL_INTERVENTION_REPLANS}'"
        "EVALUATION.residual_actor_override_checkpoint=${RESIDUAL_ACTOR_OVERRIDE_CHECKPOINT}"
        "EVALUATION.residual_actor_override_replans='${RESIDUAL_ACTOR_OVERRIDE_REPLANS}'"
        "EVALUATION.residual_max_interventions_per_episode=${RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE}"
        "EVALUATION.residual_outcome_confirmation_enabled=${RESIDUAL_OUTCOME_CONFIRMATION_ENABLED}"
        "EVALUATION.residual_outcome_confirmation_min_progress=${RESIDUAL_OUTCOME_CONFIRMATION_MIN_PROGRESS}"
        "EVALUATION.residual_outcome_confirmation_reanchor_replans=${RESIDUAL_OUTCOME_CONFIRMATION_REANCHOR_REPLANS}"
        "EVALUATION.residual_language_instruction='${residual_language_instruction}'"
      )
    fi
    printf '[robotwin-online-pair] variant=%s task=%s episodes=%s env_seed=%s inference_steps=%s instruction_mode=%s residual_language_mode=%s paper_aligned=%s strict_paired=%s\n' \
      "${variant}" "${task_name}" "${EPISODES}" "${environment_start_seed}" \
      "${INFERENCE_STEPS}" "${INSTRUCTION_MODE}" "${RESIDUAL_LANGUAGE_MODE}" \
      "${PAPER_ALIGNED}" "${STRICT_PAIRED}"
    conda run --no-capture-output -n "${CONDA_ENV}" \
      env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
      MPLCONFIGDIR=/tmp/matplotlib_robotwin PYTHONUNBUFFERED=1 \
      python -u "${PROJECT_ROOT}/experiments/robotwin/eval_robotwin_single.py" \
      "ckpt=${CHECKPOINT}" \
      "seed=${BASE_SEED}" \
      "gpu_id=${GPU_ID}" \
      "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}" \
      "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
      "EVALUATION.task_name=${task_name}" \
      "EVALUATION.task_config=${TASK_CONFIG}" \
      "EVALUATION.instruction_type=${INSTRUCTION_TYPE}" \
      "EVALUATION.eval_num_episodes=${EPISODES}" \
      "EVALUATION.eval_video_log=${EVAL_VIDEO_LOG}" \
      "EVALUATION.trial_offset=${TRIAL_OFFSET}" \
      "EVALUATION.environment_start_seed=${environment_start_seed}" \
      "EVALUATION.environment_episode_offset=${TRIAL_OFFSET}" \
      "EVALUATION.num_inference_steps=${INFERENCE_STEPS}" \
      "EVALUATION.text_cfg_scale=${TEXT_CFG_SCALE}" \
      "EVALUATION.tiled=${TILED}" \
      "EVALUATION.capture_decode_tiled=${CAPTURE_DECODE_TILED}" \
      "EVALUATION.replan_steps=${REPLAN_STEPS}" \
      "EVALUATION.action_mode=${action_mode}" \
      "EVALUATION.action_hold_probability=${ACTION_HOLD_PROBABILITY}" \
      "EVALUATION.action_hold_replans='${ACTION_HOLD_REPLANS}'" \
      "EVALUATION.action_corruption_seed=${ACTION_CORRUPTION_SEED}" \
      "${residual_args[@]}" \
      "${instruction_args[@]}" \
      "EVALUATION.environment_seed_manifest_path=${SEED_MANIFEST_PATH}" \
      "EVALUATION.deterministic_instruction_by_seed=${DETERMINISTIC_INSTRUCTION_BY_SEED}" \
      "EVALUATION.physics_audit_enabled=${PHYSICS_AUDIT_ENABLED}" \
      "EVALUATION.physics_audit_actor_attr=${PHYSICS_AUDIT_ACTOR_ATTR}" \
      "EVALUATION.physics_audit_output_dir=${physics_audit_output_dir}" \
      "EVALUATION.expert_check=${EXPERT_CHECK}" \
      "EVALUATION.paper_aligned=${PAPER_ALIGNED}" \
      "EVALUATION.strict_paired=${STRICT_PAIRED}" \
      EVALUATION.timing_enabled=true \
      "EVALUATION.save_imagination_transitions=${save_transitions}" \
      "EVALUATION.output_dir=${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}_${variant}"
    completed_log_for_task "${run_dir}" "${task_name}" || {
      printf 'RoboTwin returned without a valid success-rate log: variant=%s task=%s\n' \
        "${variant}" "${task_name}" >&2
      exit 1
    }
    touch "${marker}"
  done
done

conda run --no-capture-output -n "${CONDA_ENV}" python \
  "${PROJECT_ROOT}/experiments/robotwin/summarize_residual_iql_online_pair.py" \
  --result-base "${RESULT_BASE}" \
  --run-name "${RUN_NAME}" \
  --variants "${VARIANTS_CSV}" \
  --tasks "${TASKS_CSV}" \
  --output-json "${SUMMARY_DIR}/summary.json"

printf 'RoboTwin residual-IQL online pair complete: %s\n' "${SUMMARY_DIR}/summary.json"
