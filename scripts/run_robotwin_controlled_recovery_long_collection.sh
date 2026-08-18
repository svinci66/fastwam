#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_residual_rl/robotwin_corrected_posttrain_20260805"
CHECKPOINT="${RUN_ROOT}/iql_corrected_imagination_20epoch_paired_gate/checkpoint.pt"
SUPPORT_INDEX="${RUN_ROOT}/support_index_corrected_q95"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
MANIFEST="${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_hanging_mug_expansion10_20260817.json"
OUTPUT_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_residual_pairs/robotwin_controlled_recovery_long_20260817"
DRIVER_LOG="${OUTPUT_ROOT}/driver.log"
STATUS_TSV="${OUTPUT_ROOT}/case_status.tsv"

# seed, manifest offset, hold probability, deterministic corruption seed.
# Every selected Clean reference is already a successful paper-aligned rollout.
CASES=(
  "4800007 0 0.10 20260818"
  "4800007 0 0.10 20260819"
  "4800007 0 0.10 20260821"
  "4800008 1 0.10 20260820"
  "4800008 1 0.10 20260821"
  "4800008 1 0.10 20260823"
  "4800011 4 0.10 20260818"
  "4800011 4 0.10 20260820"
  "4800011 4 0.10 20260821"
  "4800015 7 0.10 20260821"
  "4800015 7 0.10 20260822"
  "4800015 7 0.10 20260824"
)

cd "${PROJECT_ROOT}" || exit 1
mkdir -p "${OUTPUT_ROOT}"
if [[ ! -s "${STATUS_TSV}" ]]; then
  printf 'seed\toffset\thold_probability\tcorruption_seed\tstatus\tdetail\n' > "${STATUS_TSV}"
fi

run_online() {
  local run_name="$1"
  local offset="$2"
  local hold_probability="$3"
  local corruption_seed="$4"
  local shadow_mode="$5"
  local replan="$6"
  local q_gate="$7"
  local support_path="$8"
  local max_interventions="$9"

  env \
    CUDA_VISIBLE_DEVICES=0 \
    RUN_NAME="${run_name}" \
    VARIANTS=imagination \
    TASKS=hanging_mug \
    EPISODES=1 \
    TRIAL_OFFSET="${offset}" \
    IMAGINATION_CHECKPOINT="${CHECKPOINT}" \
    NO_IMAGINATION_CHECKPOINT="${CHECKPOINT}" \
    RESIDUAL_ENCODER_VERSION=siglip-so400m-patch14-384-local-20260803 \
    RESIDUAL_Q_GATE_ENABLED="${q_gate}" \
    RESIDUAL_Q_GATE_MARGIN=0.003 \
    RESIDUAL_Q_GATE_MAX_DISAGREEMENT=0.05 \
    RESIDUAL_Q_GATE_CRITIC_SOURCE=target \
    RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
    RESIDUAL_SUPPORT_INDEX_PATH="${support_path}" \
    RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED="${q_gate}" \
    RESIDUAL_SHADOW_MODE="${shadow_mode}" \
    RESIDUAL_INTERVENTION_REPLANS="${replan}" \
    RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE="${max_interventions}" \
    RESIDUAL_LANGUAGE_MODE=training_canonical \
    SAVE_RESIDUAL_TRANSITIONS=true \
    ACTION_HOLD_PROBABILITY="${hold_probability}" \
    ACTION_CORRUPTION_SEED="${corruption_seed}" \
    PAPER_ALIGNED=false \
    STRICT_PAIRED=false \
    DETERMINISTIC_INSTRUCTION_BY_SEED=true \
    SEED_MANIFEST_PATH="${MANIFEST}" \
    INSTRUCTION_MODE=official \
    INSTRUCTION_TYPE=unseen \
    INFERENCE_STEPS=10 \
    REPLAN_STEPS=24 \
    TEXT_CFG_SCALE=1.0 \
    EVAL_VIDEO_LOG=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    RESIDUAL_DEVICE=cpu \
    CAPTURE_DECODE_TILED=true \
    TILED=false \
    bash scripts/run_robotwin_residual_iql_online_pair.sh >> "${DRIVER_LOG}" 2>&1
}

run_online_retry() {
  local attempt
  for attempt in 1 2 3; do
    printf '[controlled-long] stage=online attempt=%s/3 run=%s\n' "${attempt}" "$1" | tee -a "${DRIVER_LOG}"
    if ! nvidia-smi >> "${DRIVER_LOG}" 2>&1; then
      nvidia-modprobe -u -c 0 >> "${DRIVER_LOG}" 2>&1 || true
    fi
    run_online "$@" && return 0
    sleep 5
  done
  return 1
}

for case_spec in "${CASES[@]}"; do
  read -r seed offset probability corruption_seed <<< "${case_spec}"
  probability_tag="$(printf '%03d' "$(awk -v p="${probability}" 'BEGIN { print int(p * 100 + 0.5) }')")"
  case_tag="seed${seed}_hold${probability_tag}_cs${corruption_seed}"
  case_dir="${OUTPUT_ROOT}/${case_tag}"
  corrupt_name="robotwin_controlled_long_20260817_${case_tag}_shadow"
  corrupt_dir="${RESULT_BASE}/${corrupt_name}_imagination"
  corrupt_summary="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${corrupt_name}/summary.json"
  clean_dir="${RESULT_BASE}/robotwin_hanging_mug_expansion_20260817_seed${seed}_shadow_imagination"
  plan_json="${case_dir}/candidate_plan.json"
  plan_tsv="${case_dir}/candidate_plan.tsv"
  mkdir -p "${case_dir}"

  if [[ -f "${case_dir}/CASE_COMPLETE" ]]; then
    printf '[controlled-long] case=%s stage=skip reason=complete\n' "${case_tag}" | tee -a "${DRIVER_LOG}"
    continue
  fi

  printf '[controlled-long] case=%s stage=corrupted-shadow\n' "${case_tag}" | tee -a "${DRIVER_LOG}"
  if [[ ! -s "${corrupt_summary}" ]] && ! run_online_retry \
      "${corrupt_name}" "${offset}" "${probability}" "${corruption_seed}" \
      true all true "${SUPPORT_INDEX}" none; then
    printf '%s\t%s\t%s\t%s\tonline_error\tcorrupted_shadow\n' \
      "${seed}" "${offset}" "${probability}" "${corruption_seed}" >> "${STATUS_TSV}"
    touch "${case_dir}/CASE_FAILED"
    continue
  fi

  if ! conda run --no-capture-output -n robotwin_fastwam python \
      /tmp/select_controlled_recovery_candidates.py \
      --shadow-dir "${corrupt_dir}" \
      --summary-json "${corrupt_summary}" \
      --output-json "${plan_json}" \
      --output-tsv "${plan_tsv}" \
      --max-candidates 1 >> "${DRIVER_LOG}" 2>&1; then
    printf '%s\t%s\t%s\t%s\taudit_error\tcandidate_selection\n' \
      "${seed}" "${offset}" "${probability}" "${corruption_seed}" >> "${STATUS_TSV}"
    touch "${case_dir}/CASE_FAILED"
    continue
  fi

  selected_count="$(sed -n '2,$p' "${plan_tsv}" | wc -l)"
  if [[ "${selected_count}" == 0 ]]; then
    skip_reason="$(conda run --no-capture-output -n robotwin_fastwam python -c \
      "import json; print(json.load(open('${plan_json}'))['skip_reason'])")"
    printf '%s\t%s\t%s\t%s\tno_recovery\t%s\n' \
      "${seed}" "${offset}" "${probability}" "${corruption_seed}" "${skip_reason}" >> "${STATUS_TSV}"
    touch "${case_dir}/CASE_COMPLETE"
    continue
  fi

  while IFS=$'\t' read -r replan candidate_rms q_advantage; do
    [[ "${replan}" == replan_idx ]] && continue
    recovery_name="robotwin_controlled_long_20260817_${case_tag}_replan${replan}"
    recovery_dir="${RESULT_BASE}/${recovery_name}_imagination"
    recovery_summary="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${recovery_name}/summary.json"
    pair_dir="${case_dir}/replan${replan}"
    printf '[controlled-long] case=%s stage=recovery replan=%s\n' "${case_tag}" "${replan}" | tee -a "${DRIVER_LOG}"

    if [[ ! -s "${recovery_summary}" ]] && ! run_online_retry \
        "${recovery_name}" "${offset}" "${probability}" "${corruption_seed}" \
        false "${replan}" false none 1; then
      printf '%s\t%s\t%s\t%s\tonline_error\trecovery_replan_%s\n' \
        "${seed}" "${offset}" "${probability}" "${corruption_seed}" "${replan}" >> "${STATUS_TSV}"
      touch "${case_dir}/CASE_FAILED"
      continue 2
    fi

    conda run --no-capture-output -n robotwin_fastwam python \
      experiments/robotwin/build_single_intervention_pairs.py \
      --baseline-dir "${corrupt_dir}" \
      --baseline-action-mode residual \
      --residual-dir "${recovery_dir}" \
      --intervention-replans "${replan}" \
      --output-dir "${pair_dir}" >> "${DRIVER_LOG}" 2>&1 || {
        touch "${case_dir}/CASE_FAILED"
        continue 2
      }

    if conda run --no-capture-output -n robotwin_fastwam python \
        /tmp/audit_controlled_recovery_pair.py \
        --clean-dir "${clean_dir}" \
        --corrupted-dir "${corrupt_dir}" \
        --recovery-dir "${recovery_dir}" \
        --pairing-summary "${pair_dir}/pairing_summary.json" \
        --intervention-replan "${replan}" \
        --output-json "${pair_dir}/controlled_prefix_audit.json" >> "${DRIVER_LOG}" 2>&1; then
      label="$(conda run --no-capture-output -n robotwin_fastwam python -c \
        "import json; print(next(iter(json.load(open('${pair_dir}/pairing_summary.json'))['label_counts'])))")"
      printf '%s\t%s\t%s\t%s\tusable_pair\t%s_replan_%s\n' \
        "${seed}" "${offset}" "${probability}" "${corruption_seed}" "${label}" "${replan}" >> "${STATUS_TSV}"
      touch "${case_dir}/CASE_COMPLETE"
    else
      printf '%s\t%s\t%s\t%s\tpair_quarantined\treplan_%s\n' \
        "${seed}" "${offset}" "${probability}" "${corruption_seed}" "${replan}" >> "${STATUS_TSV}"
      touch "${case_dir}/CASE_FAILED"
    fi
  done < "${plan_tsv}"
done

usable_count="$(find "${OUTPUT_ROOT}" -name controlled_prefix_audit.json -print0 | xargs -0 -r rg -l '"usable": true' | wc -l)"
rescue_count="$(find "${OUTPUT_ROOT}" -name pairing_summary.json -print0 | xargs -0 -r rg -l '"rescue": 1' | wc -l)"
complete_count="$(find "${OUTPUT_ROOT}" -name CASE_COMPLETE | wc -l)"
failed_count="$(find "${OUTPUT_ROOT}" -name CASE_FAILED | wc -l)"
printf '[controlled-long] complete cases=%s usable_pairs=%s rescues=%s failed_cases=%s output=%s\n' \
  "${complete_count}" "${usable_count}" "${rescue_count}" "${failed_count}" "${OUTPUT_ROOT}" | tee -a "${DRIVER_LOG}"
