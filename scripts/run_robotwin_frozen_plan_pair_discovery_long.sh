#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_RUN_NAME="${BASE_RUN_NAME:-robotwin_frozen_plan_pair_discovery_long_20260825}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_frozen_plan_reward_expansion_20260825.json}"
CASES_FILE="${CASES_FILE:-}"
NOISE_SEEDS_CSV="${NOISE_SEEDS_CSV:-}"
CLEAN_PRESCREEN="${CLEAN_PRESCREEN:-false}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
ARTIFACT_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${BASE_RUN_NAME}"
REVIEW_DIR="${ARTIFACT_DIR}/manual_review"
TARGET_STRICT_PAIRS="${TARGET_STRICT_PAIRS:-8}"
MAX_SCREEN_RUNS="${MAX_SCREEN_RUNS:-64}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"
MAX_ABS_DELTA="${MAX_ABS_DELTA:-0.05}"
GPU_ID="${GPU_ID:-0}"
JOB_COOLDOWN_SECONDS="${JOB_COOLDOWN_SECONDS:-30}"
SEGMENT_SCREEN_RUNS="${SEGMENT_SCREEN_RUNS:-4}"
SEGMENT_COOLDOWN_SECONDS="${SEGMENT_COOLDOWN_SECONDS:-120}"
GPU_TELEMETRY_INTERVAL_SECONDS="${GPU_TELEMETRY_INTERVAL_SECONDS:-2}"
STRICT_BRANCH_RETRIES="${STRICT_BRANCH_RETRIES:-1}"
mkdir -p "${ARTIFACT_DIR}" "${REVIEW_DIR}"
exec > >(tee -a "${ARTIFACT_DIR}/driver.log") 2>&1

for value_name in JOB_COOLDOWN_SECONDS SEGMENT_SCREEN_RUNS \
  SEGMENT_COOLDOWN_SECONDS GPU_TELEMETRY_INTERVAL_SECONDS; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    printf '[pair-discovery] %s must be a non-negative integer, got %q\n' \
      "${value_name}" "${value}" >&2
    exit 64
  fi
done
if [[ "${CLEAN_PRESCREEN}" != true && "${CLEAN_PRESCREEN}" != false ]]; then
  printf '[pair-discovery] CLEAN_PRESCREEN must be true or false, got %q\n' \
    "${CLEAN_PRESCREEN}" >&2
  exit 64
fi
if [[ -n "${CASES_FILE}" && ! -s "${CASES_FILE}" ]]; then
  printf '[pair-discovery] CASES_FILE is missing or empty: %s\n' \
    "${CASES_FILE}" >&2
  exit 64
fi

# A second collector would double GPU load and can also corrupt the status file.
# flock is automatically released after a process or machine restart.
LOCK_FILE="${ARTIFACT_DIR}/collector.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[pair-discovery] another collector owns lock=%s\n' "${LOCK_FILE}" >&2
  exit 73
fi

STATUS_TSV="${ARTIFACT_DIR}/status.tsv"
if [[ ! -f "${STATUS_TSV}" ]]; then
  printf 'timestamp\tcombo_key\ttask\tseed\treplan\tnoise_seed\tclean\tcorrupted\tcorrected\tdecision\treview_dir\n' \
    > "${STATUS_TSV}"
fi
PRESCREEN_TSV="${ARTIFACT_DIR}/clean_prescreen.tsv"
if [[ ! -f "${PRESCREEN_TSV}" ]]; then
  printf 'timestamp\tcase_tag\ttask\tseed\tclean\tdecision\n' > "${PRESCREEN_TSV}"
fi

TELEMETRY_CSV="${ARTIFACT_DIR}/gpu_telemetry.csv"
TELEMETRY_ERROR_LOG="${ARTIFACT_DIR}/gpu_telemetry_errors.log"
GPU_TELEMETRY_PID=''

start_gpu_telemetry() {
  (( GPU_TELEMETRY_INTERVAL_SECONDS > 0 )) || return 0
  if [[ ! -s "${TELEMETRY_CSV}" ]]; then
    printf 'sample_time,boot_id,gpu_index,temperature_c,power_draw_w,power_limit_w,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_total_mib,pstate\n' \
      > "${TELEMETRY_CSV}"
  fi
  local boot_id
  boot_id="$(< /proc/sys/kernel/random/boot_id)"
  (
    while :; do
      sample_time="$(date --iso-8601=seconds)"
      if gpu_rows="$(nvidia-smi -i "${GPU_ID}" \
        --query-gpu=index,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.total,pstate \
        --format=csv,noheader,nounits 2>> "${TELEMETRY_ERROR_LOG}")"; then
        while IFS= read -r gpu_row; do
          [[ -n "${gpu_row}" ]] || continue
          printf '%s,%s,%s\n' "${sample_time}" "${boot_id}" "${gpu_row}"
        done <<< "${gpu_rows}"
      else
        printf '%s telemetry query failed\n' "${sample_time}" \
          >> "${TELEMETRY_ERROR_LOG}"
      fi
      sleep "${GPU_TELEMETRY_INTERVAL_SECONDS}"
    done
  ) >> "${TELEMETRY_CSV}" &
  GPU_TELEMETRY_PID=$!
  printf '[pair-discovery] gpu telemetry pid=%s interval=%ss output=%s\n' \
    "${GPU_TELEMETRY_PID}" "${GPU_TELEMETRY_INTERVAL_SECONDS}" "${TELEMETRY_CSV}"
}

stop_gpu_telemetry() {
  if [[ -n "${GPU_TELEMETRY_PID}" ]] && kill -0 "${GPU_TELEMETRY_PID}" 2>/dev/null; then
    kill "${GPU_TELEMETRY_PID}" 2>/dev/null || true
    wait "${GPU_TELEMETRY_PID}" 2>/dev/null || true
  fi
}

trap stop_gpu_telemetry EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
start_gpu_telemetry

printf '[pair-discovery] safety config job_cooldown=%ss segment_size=%s segment_cooldown=%ss telemetry_interval=%ss gpu=%s boot_id=%s\n' \
  "${JOB_COOLDOWN_SECONDS}" "${SEGMENT_SCREEN_RUNS}" \
  "${SEGMENT_COOLDOWN_SECONDS}" "${GPU_TELEMETRY_INTERVAL_SECONDS}" \
  "${GPU_ID}" "$(< /proc/sys/kernel/random/boot_id)"

# Each case has already succeeded under the current FastWAM protocol. Replans
# are pre-registered from grasp/contact/place phases, not selected by reward.
# task seed manifest_offset trial_offset replan_csv case_tag
cases=(
  "place_can_basket 4800003 0 0 2,5,8 place_seed4800003"
  "hanging_mug 4800003 0 0 2,3,4,9,10 hanging_seed4800003"
  "hanging_mug 4800005 1 1 2,3,4,9,10 hanging_seed4800005"
  "adjust_bottle 4800000 0 0 1,2,3 adjust_seed4800000"
  "adjust_bottle 4800001 1 1 1,2,3 adjust_seed4800001"
  "adjust_bottle 4800002 2 2 1,2,3 adjust_seed4800002"
  "adjust_bottle 4800003 3 3 1,2,3 adjust_seed4800003"
  "adjust_bottle 4800004 4 4 1,2,3 adjust_seed4800004"
  "stack_blocks_two 4800000 0 0 2,3,4,8,9,10 stack_seed4800000"
  "stack_blocks_two 4800001 1 1 2,3,4,8,9,10 stack_seed4800001"
  "stack_blocks_two 4800002 2 2 2,3,4,8,9,10 stack_seed4800002"
  "stack_blocks_two 4800003 3 3 2,3,4,8,9,10 stack_seed4800003"
  "stack_blocks_two 4800004 4 4 2,3,4,8,9,10 stack_seed4800004"
  "hanging_mug 4800008 3 0 2,3,4,9,10 hanging_seed4800008"
  "hanging_mug 4800011 4 0 2,3,4,9,10 hanging_seed4800011"
  "hanging_mug 4800015 5 0 2,3,4,9,10 hanging_seed4800015"
)
if [[ -n "${CASES_FILE}" ]]; then
  mapfile -t cases < <(
    awk 'NF && $1 !~ /^#/ { print }' "${CASES_FILE}"
  )
  (( ${#cases[@]} > 0 )) || {
    printf '[pair-discovery] CASES_FILE has no runnable cases: %s\n' \
      "${CASES_FILE}" >&2
    exit 64
  }
fi

# These seeds define deterministic bounded joint-action perturbation directions.
# They are discovery conditions, not an independently reported test set.
noise_seeds=(20260826 20260827 20260828 20260829 20260830 20260831 20260832 20260833)
if [[ -n "${NOISE_SEEDS_CSV}" ]]; then
  IFS=',' read -r -a noise_seeds <<< "${NOISE_SEEDS_CSV}"
  for noise_seed in "${noise_seeds[@]}"; do
    [[ "${noise_seed}" =~ ^[0-9]+$ ]] || {
      printf '[pair-discovery] invalid noise seed: %q\n' "${noise_seed}" >&2
      exit 64
    }
  done
fi

declare -A done_combos=()
declare -A accepted_cases=()
declare -A invalid_cases=()
declare -A prescreened_cases=()
declare -A prescreen_invalid_cases=()
while IFS=$'\t' read -r _timestamp combo_key _task _seed _replan _noise_seed \
  _clean _corrupt _correct decision _review_dir; do
  [[ "${combo_key}" == combo_key || -z "${combo_key}" ]] && continue
  done_combos["${combo_key}"]=1
  case_tag="${combo_key%%__r*}"
  if [[ "${decision}" == strict_pair_ready ]]; then
    accepted_cases["${case_tag}"]=1
  elif [[ "${decision}" == clean_not_reproduced || "${decision}" == strict_seed_rejected ]]; then
    invalid_cases["${case_tag}"]=1
  fi
done < "${STATUS_TSV}"
while IFS=$'\t' read -r _timestamp case_tag _task _seed _clean decision; do
  [[ "${case_tag}" == case_tag || -z "${case_tag}" ]] && continue
  if [[ "${decision}" == clean_prescreen_pass ]]; then
    prescreened_cases["${case_tag}"]=1
  else
    prescreen_invalid_cases["${case_tag}"]=1
  fi
done < "${PRESCREEN_TSV}"

record_status() {
  local combo_key="$1" task="$2" seed="$3" replan="$4" noise_seed="$5"
  local clean="$6" corrupt="$7" correct="$8" decision="$9" review_path="${10}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "${combo_key}" "${task}" "${seed}" \
    "${replan}" "${noise_seed}" "${clean}" "${corrupt}" "${correct}" \
    "${decision}" "${review_path}" >> "${STATUS_TSV}"
  # Persist each completed decision immediately so a hard reset only repeats
  # the branch that was in flight, never earlier completed combinations.
  sync -d "${STATUS_TSV}" 2>/dev/null || true
  done_combos["${combo_key}"]=1
}

record_prescreen() {
  local case_tag="$1" task="$2" seed="$3" clean="$4" decision="$5"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "${case_tag}" "${task}" "${seed}" \
    "${clean}" "${decision}" >> "${PRESCREEN_TSV}"
  sync -d "${PRESCREEN_TSV}" 2>/dev/null || true
  if [[ "${decision}" == clean_prescreen_pass ]]; then
    prescreened_cases["${case_tag}"]=1
  else
    prescreen_invalid_cases["${case_tag}"]=1
  fi
}

result_value() {
  local run_name="$1" branch="$2" task="$3"
  local path="${RESULT_BASE}/${run_name}_${branch}/${task}/_result.txt"
  [[ -s "${path}" ]] || return 1
  awk 'NF { value=$NF } END { print value }' "${path}"
}

is_success() {
  awk -v value="$1" 'BEGIN { exit !(value + 0.0 >= 0.5) }'
}

run_branches() {
  local run_name="$1" task="$2" seed="$3" episode_offset="$4"
  local trial_offset="$5" replan="$6" noise_seed="$7" branches="$8" audit="$9"
  local rc=0
  if RUN_NAME="${run_name}" TASK="${task}" ENVIRONMENT_START_SEED="${seed}" \
      ENVIRONMENT_EPISODE_OFFSET="${episode_offset}" TRIAL_OFFSET="${trial_offset}" \
      INTERVENTION_REPLAN="${replan}" ACTION_NOISE_SEED="${noise_seed}" \
      MAX_ABS_DELTA="${MAX_ABS_DELTA}" MANIFEST="${MANIFEST}" GPU_ID="${GPU_ID}" \
      BRANCHES="${branches}" RUN_AUDIT="${audit}" \
      bash "${PROJECT_ROOT}/scripts/run_robotwin_frozen_plan_trajectory_smoke.sh"; then
    rc=0
  else
    rc=$?
  fi
  if (( JOB_COOLDOWN_SECONDS > 0 )); then
    printf '[pair-discovery] gpu job finished rc=%s; cooldown=%ss\n' \
      "${rc}" "${JOB_COOLDOWN_SECONDS}"
    sleep "${JOB_COOLDOWN_SECONDS}"
  fi
  return "${rc}"
}

LAST_BRANCH_STRICT_ERROR=0
run_branches_with_strict_retry() {
  local run_name="$1" task="$2" seed="$3" episode_offset="$4"
  local trial_offset="$5" replan="$6" noise_seed="$7" branches="$8" audit="$9"
  local retry=0 rc=0 log_start=0
  local case_log="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${run_name}/driver.log"
  while true; do
    log_start=0
    [[ -f "${case_log}" ]] && log_start="$(wc -l < "${case_log}")"
    LAST_BRANCH_STRICT_ERROR=0
    if run_branches "${run_name}" "${task}" "${seed}" "${episode_offset}" \
      "${trial_offset}" "${replan}" "${noise_seed}" "${branches}" "${audit}"; then
      return 0
    else
      rc=$?
    fi
    if [[ -s "${case_log}" ]] && tail -n "+$((log_start + 1))" "${case_log}" | rg -q \
      'Strict environment seed .* (is not expert-feasible|failed expert validation|became unstable)'; then
      LAST_BRANCH_STRICT_ERROR=1
    fi
    if (( LAST_BRANCH_STRICT_ERROR == 0 || retry >= STRICT_BRANCH_RETRIES )); then
      return "${rc}"
    fi
    retry=$((retry + 1))
    printf '[pair-discovery] transient strict-seed rejection branch=%s retry=%s/%s combo_run=%s\n' \
      "${branches}" "${retry}" "${STRICT_BRANCH_RETRIES}" "${run_name}"
  done
}

new_screen_runs=0
last_segment_pause_at=-1
maybe_segment_cooldown() {
  (( SEGMENT_SCREEN_RUNS > 0 )) || return 0
  (( new_screen_runs > 0 )) || return 0
  (( new_screen_runs % SEGMENT_SCREEN_RUNS == 0 )) || return 0
  (( last_segment_pause_at != new_screen_runs )) || return 0
  last_segment_pause_at="${new_screen_runs}"
  sync -d "${STATUS_TSV}" 2>/dev/null || true
  if (( SEGMENT_COOLDOWN_SECONDS > 0 )); then
    printf '[pair-discovery] segment complete new_screens=%s; cooldown=%ss\n' \
      "${new_screen_runs}" "${SEGMENT_COOLDOWN_SECONDS}"
    sleep "${SEGMENT_COOLDOWN_SECONDS}"
  fi
}

strict_pairs="${#accepted_cases[@]}"
screen_runs="${#done_combos[@]}"
printf '[pair-discovery] resume strict_pairs=%s target=%s screen_runs=%s max=%s\n' \
  "${strict_pairs}" "${TARGET_STRICT_PAIRS}" "${screen_runs}" "${MAX_SCREEN_RUNS}"
printf '[pair-discovery] search cases=%s noise_seeds=%s clean_prescreen=%s cases_file=%s\n' \
  "${#cases[@]}" "${#noise_seeds[@]}" "${CLEAN_PRESCREEN}" \
  "${CASES_FILE:-builtin}"

stop_requested=false
for noise_seed in "${noise_seeds[@]}"; do
  for stage_slot in 0 1 2 3 4 5; do
    for case_spec in "${cases[@]}"; do
      read -r task seed episode_offset trial_offset replan_csv case_tag <<< "${case_spec}"
      [[ -n "${accepted_cases[${case_tag}]:-}" ]] && continue
      [[ -n "${invalid_cases[${case_tag}]:-}" ]] && continue
      [[ -n "${prescreen_invalid_cases[${case_tag}]:-}" ]] && continue
      IFS=',' read -r -a replans <<< "${replan_csv}"
      (( stage_slot < ${#replans[@]} )) || continue
      replan="${replans[${stage_slot}]}"
      combo_key="${case_tag}__r${replan}__ns${noise_seed}"
      [[ -n "${done_combos[${combo_key}]:-}" ]] && continue

      if (( strict_pairs >= TARGET_STRICT_PAIRS )); then
        printf '[pair-discovery] target reached\n'
        stop_requested=true
        break 3
      fi
      if (( screen_runs >= MAX_SCREEN_RUNS )); then
        printf '[pair-discovery] max screen runs reached\n'
        stop_requested=true
        break 3
      fi

      if [[ "${CLEAN_PRESCREEN}" == true \
          && -z "${prescreened_cases[${case_tag}]:-}" ]]; then
        prescreen_run_name="${BASE_RUN_NAME}_prescreen_${case_tag}"
        printf '[pair-discovery] clean prescreen case=%s task=%s seed=%s\n' \
          "${case_tag}" "${task}" "${seed}"
        if ! run_branches_with_strict_retry "${prescreen_run_name}" "${task}" "${seed}" \
          "${episode_offset}" "${trial_offset}" "${replan}" "${noise_seed}" \
          clean false; then
          if (( LAST_BRANCH_STRICT_ERROR == 1 )); then
            record_prescreen "${case_tag}" "${task}" "${seed}" not_run \
              clean_prescreen_strict_seed_rejected
            continue
          fi
          printf '[pair-discovery] fatal clean prescreen error case=%s\n' \
            "${case_tag}" >&2
          exit 70
        fi
        prescreen_result="$(result_value "${prescreen_run_name}" clean "${task}")"
        if ! is_success "${prescreen_result}"; then
          record_prescreen "${case_tag}" "${task}" "${seed}" \
            "${prescreen_result}" clean_prescreen_failed
          printf '[pair-discovery] reject case=%s reason=clean_prescreen_failed\n' \
            "${case_tag}"
          continue
        fi
        record_prescreen "${case_tag}" "${task}" "${seed}" \
          "${prescreen_result}" clean_prescreen_pass
        printf '[pair-discovery] clean prescreen pass case=%s\n' "${case_tag}"
      fi

      free_gb="$(df -BG --output=avail "${PROJECT_ROOT}" | tail -n 1 | tr -dc '0-9')"
      if [[ -z "${free_gb}" ]] || (( free_gb < MIN_FREE_GB )); then
        printf '[pair-discovery] storage stop free_gb=%s min_free_gb=%s\n' \
          "${free_gb:-unknown}" "${MIN_FREE_GB}"
        stop_requested=true
        break 3
      fi

      screen_runs=$((screen_runs + 1))
      new_screen_runs=$((new_screen_runs + 1))
      run_name="${BASE_RUN_NAME}_${combo_key}"
      printf '[pair-discovery] screen %s/%s combo=%s free_gb=%s\n' \
        "${screen_runs}" "${MAX_SCREEN_RUNS}" "${combo_key}" "${free_gb}"
      if ! run_branches_with_strict_retry "${run_name}" "${task}" "${seed}" "${episode_offset}" \
        "${trial_offset}" "${replan}" "${noise_seed}" corrupted false; then
        if (( LAST_BRANCH_STRICT_ERROR == 1 )); then
          record_status "${combo_key}" "${task}" "${seed}" "${replan}" \
            "${noise_seed}" not_run not_run not_run strict_seed_rejected ''
          invalid_cases["${case_tag}"]=1
          maybe_segment_cooldown
          continue
        fi
        printf '[pair-discovery] fatal evaluation error combo=%s\n' "${combo_key}" >&2
        exit 70
      fi

      corrupt_result="$(result_value "${run_name}" corrupted "${task}")"
      if is_success "${corrupt_result}"; then
        record_status "${combo_key}" "${task}" "${seed}" "${replan}" \
          "${noise_seed}" not_run "${corrupt_result}" not_run corrupt_still_succeeds ''
        maybe_segment_cooldown
        continue
      fi

      printf '[pair-discovery] corrupt failed; confirm strict triplet combo=%s\n' "${combo_key}"
      if ! run_branches_with_strict_retry "${run_name}" "${task}" "${seed}" \
        "${episode_offset}" "${trial_offset}" "${replan}" "${noise_seed}" \
        clean false; then
        if (( LAST_BRANCH_STRICT_ERROR == 1 )); then
          record_status "${combo_key}" "${task}" "${seed}" "${replan}" \
            "${noise_seed}" not_run "${corrupt_result}" not_run strict_seed_unstable_clean ''
          invalid_cases["${case_tag}"]=1
          maybe_segment_cooldown
          continue
        fi
        printf '[pair-discovery] fatal clean confirmation error combo=%s\n' \
          "${combo_key}" >&2
        exit 70
      fi
      clean_result="$(result_value "${run_name}" clean "${task}")"
      if ! is_success "${clean_result}"; then
        record_status "${combo_key}" "${task}" "${seed}" "${replan}" \
          "${noise_seed}" "${clean_result}" "${corrupt_result}" not_run clean_not_reproduced ''
        invalid_cases["${case_tag}"]=1
        maybe_segment_cooldown
        continue
      fi

      if ! run_branches_with_strict_retry "${run_name}" "${task}" "${seed}" \
        "${episode_offset}" "${trial_offset}" "${replan}" "${noise_seed}" \
        corrected false; then
        if (( LAST_BRANCH_STRICT_ERROR == 1 )); then
          record_status "${combo_key}" "${task}" "${seed}" "${replan}" \
            "${noise_seed}" "${clean_result}" "${corrupt_result}" not_run \
            strict_seed_unstable_corrected ''
          invalid_cases["${case_tag}"]=1
          maybe_segment_cooldown
          continue
        fi
        printf '[pair-discovery] fatal corrected confirmation error combo=%s\n' \
          "${combo_key}" >&2
        exit 70
      fi
      correct_result="$(result_value "${run_name}" corrected "${task}")"
      if ! is_success "${correct_result}"; then
        record_status "${combo_key}" "${task}" "${seed}" "${replan}" \
          "${noise_seed}" "${clean_result}" "${corrupt_result}" "${correct_result}" correction_did_not_recover ''
        maybe_segment_cooldown
        continue
      fi

      if ! run_branches_with_strict_retry "${run_name}" "${task}" "${seed}" \
        "${episode_offset}" "${trial_offset}" "${replan}" "${noise_seed}" \
        clean,corrupted,corrected true; then
        if (( LAST_BRANCH_STRICT_ERROR == 1 )); then
          record_status "${combo_key}" "${task}" "${seed}" "${replan}" \
            "${noise_seed}" "${clean_result}" "${corrupt_result}" "${correct_result}" \
            strict_seed_unstable_audit ''
          invalid_cases["${case_tag}"]=1
          maybe_segment_cooldown
          continue
        fi
        printf '[pair-discovery] fatal triplet audit error combo=%s\n' \
          "${combo_key}" >&2
        exit 70
      fi

      review_name="$(printf 'candidate_%02d_%s' "$((strict_pairs + 1))" "${combo_key}")"
      review_path="${REVIEW_DIR}/${review_name}"
      mkdir -p "${review_path}"
      for branch in clean corrupted corrected; do
        ln -sfn "${RESULT_BASE}/${run_name}_${branch}/${task}/episode0.mp4" \
          "${review_path}/${branch}.mp4"
      done
      ln -sfn "${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${run_name}/action_triplet_audit.json" \
        "${review_path}/action_triplet_audit.json"
      ln -sfn "${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${run_name}/trajectory_triplet_audit.json" \
        "${review_path}/trajectory_triplet_audit.json"
      record_status "${combo_key}" "${task}" "${seed}" "${replan}" \
        "${noise_seed}" "${clean_result}" "${corrupt_result}" "${correct_result}" \
        strict_pair_ready "${review_path}"
      accepted_cases["${case_tag}"]=1
      strict_pairs=$((strict_pairs + 1))
      printf '[pair-discovery] accepted strict_pair=%s/%s review=%s\n' \
        "${strict_pairs}" "${TARGET_STRICT_PAIRS}" "${review_path}"
      maybe_segment_cooldown
    done
  done
done

printf '[pair-discovery] finished strict_pairs=%s target=%s screen_runs=%s stop_requested=%s\n' \
  "${strict_pairs}" "${TARGET_STRICT_PAIRS}" "${screen_runs}" "${stop_requested}"
if (( strict_pairs < TARGET_STRICT_PAIRS )); then
  printf '[pair-discovery] DISCOVERY_TARGET_NOT_MET; inspect status before expanding search\n'
  exit 3
fi
printf '[pair-discovery] PASS manual_review=%s\n' "${REVIEW_DIR}"
