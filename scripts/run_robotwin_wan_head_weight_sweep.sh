#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
REPLAY_DIR="${REPLAY_DIR:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_open_microwave_wan_head_awr_smoke_seed42_20260827/replay}"
SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_4task_heldout5_expert_seeds_20260804.json}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_open_microwave_wan_head_weight_sweep_20260827}"
CONTROL_CONFIG="${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_wan_head_no_imagination_smoke.yaml"
DRIVER_LOG="${RUN_ROOT}/driver.log"
SEED=42

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${DRIVER_LOG}") 2>&1

train_pair() {
  local label="$1" candidate_config="$2" train_root="$3" variant config output_dir
  declare -A configs=(
    [no_imagination]="${CONTROL_CONFIG}"
    [with_imagination]="${candidate_config}"
  )
  for variant in no_imagination with_imagination; do
    config="${configs[${variant}]}"
    output_dir="${train_root}/seed${SEED}/${variant}"
    if [[ -s "${output_dir}/checkpoint.pt" && -s "${output_dir}/history.json" ]]; then
      printf '[wan-head-weight] skip trained label=%s variant=%s\n' "${label}" "${variant}"
      continue
    fi
    [[ ! -e "${output_dir}" ]] || {
      printf '[wan-head-weight] refusing incomplete train output: %s\n' "${output_dir}" >&2
      exit 1
    }
    conda run --no-capture-output -n "${CONDA_ENV}" python -u \
      "${PROJECT_ROOT}/scripts/train_robotwin_residual_awr.py" \
      --config "${config}" --replay-dir "${REPLAY_DIR}" \
      --output-dir "${output_dir}" --seed "${SEED}" \
      --timeout-bootstrap-value 0.0
  done
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/audit_awr_training_pair.py" \
    --output-root "${train_root}" --seeds "${SEED}" \
    --output-json "${train_root}/paired_training_audit.json"
}

online_pair() {
  local label="$1" train_root="$2" online_run_name="$3"
  env \
    RUN_NAME="${online_run_name}" VARIANTS=no_imagination,imagination \
    TASKS=open_microwave EPISODES=5 BASE_SEED=47 TRIAL_OFFSET=0 \
    INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
    TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
    PAPER_ALIGNED=true STRICT_PAIRED=true \
    DETERMINISTIC_INSTRUCTION_BY_SEED=true EXPERT_CHECK=true \
    SEED_MANIFEST_PATH="${SEED_MANIFEST_PATH}" \
    NO_IMAGINATION_CHECKPOINT="${train_root}/seed${SEED}/no_imagination/checkpoint.pt" \
    IMAGINATION_CHECKPOINT="${train_root}/seed${SEED}/with_imagination/checkpoint.pt" \
    RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-modelscope-local-v1 \
    RESIDUAL_LANGUAGE_MODE=policy_instruction \
    RESIDUAL_Q_GATE_ENABLED=false RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
    RESIDUAL_SUPPORT_INDEX_PATH=none RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
    RESIDUAL_SHADOW_MODE=false RESIDUAL_INTERVENTION_REPLANS=all \
    RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
    RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=false RESIDUAL_SOFT_SCALE_ENABLED=false \
    SAVE_BASELINE_TRANSITIONS=false SAVE_RESIDUAL_TRANSITIONS=false \
    EVAL_VIDEO_LOG=true \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"
  printf '[wan-head-weight] online complete label=%s run=%s\n' "${label}" "${online_run_name}"
}

run_candidate() {
  local label="$1" weight="$2" config="$3" retry_on_tie="$4"
  local candidate_root="${RUN_ROOT}/${label}"
  local train_root="${candidate_root}/training"
  local online_run_name="robotwin_open_microwave_wan_head_${label}_online_pair_5ep_20260827"
  local summary="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${online_run_name}/summary.json"
  local decision="${candidate_root}/decision.json"
  train_pair "${label}" "${config}" "${train_root}"
  online_pair "${label}" "${train_root}" "${online_run_name}"
  decision_args=()
  if [[ "${retry_on_tie}" == "true" ]]; then decision_args+=(--retry-on-tie); fi
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/decide_wan_head_weight_candidate.py" \
    --summary "${summary}" --weight "${weight}" \
    "${decision_args[@]}" --output-json "${decision}"
}

weight025_config="${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_wan_head_weight025_smoke.yaml"
run_candidate weight025 0.25 "${weight025_config}" true

if rg -q '"decision": "retry_lower_weight"' "${RUN_ROOT}/weight025/decision.json"; then
  weight010_config="${PROJECT_ROOT}/configs/rl/robotwin_residual_awr_wan_head_weight010_smoke.yaml"
  run_candidate weight010 0.1 "${weight010_config}" false
  cp "${RUN_ROOT}/weight010/decision.json" "${RUN_ROOT}/final_decision.json"
else
  cp "${RUN_ROOT}/weight025/decision.json" "${RUN_ROOT}/final_decision.json"
fi

touch "${RUN_ROOT}/COMPLETE"
printf '[wan-head-weight] sweep complete: %s\n' "${RUN_ROOT}/final_decision.json"
