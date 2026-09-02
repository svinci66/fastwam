#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-robotwin_wan_head_multitask3_awr_formal_block1_5ep_20260901}"
SEGMENT_PREFIX="${SEGMENT_PREFIX:-${RUN_NAME}_segmented}"
TASKS="${TASKS:-open_microwave,hanging_mug,place_can_basket}"
VARIANTS="${VARIANTS:-no_imagination,imagination}"
EPISODES="${EPISODES:-5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
SERVICE_NAME="${SERVICE_NAME:-fastwam-robotwin-compare.service}"
AUDIT_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${RUN_NAME}"
RESULT_BASE="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384"
CRASH_MARKER_DIR="${AUDIT_ROOT}/segmented_crash_failures"
MONITOR_LOG="${AUDIT_ROOT}/monitor.log"

mkdir -p "${AUDIT_ROOT}"
exec 9>"${AUDIT_ROOT}/.monitor.lock"
flock -n 9 || exit 0

completed=0
terminal_failures=0
total=0
IFS=',' read -r -a variants <<< "${VARIANTS}"
IFS=',' read -r -a tasks <<< "${TASKS}"
for variant in "${variants[@]}"; do
  for task in "${tasks[@]}"; do
    for ((offset = 0; offset < EPISODES; offset++)); do
      total=$((total + 1))
      found=false
      for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
        segment_run="${SEGMENT_PREFIX}__${variant}__${task}__episode${offset}__attempt${attempt}"
        marker="${RESULT_BASE}/${segment_run}_${variant}/.${task}_1ep_complete"
        if [[ -f "${marker}" ]]; then
          completed=$((completed + 1))
          found=true
          break
        fi
      done
      if [[ "${found}" != true && -f "${CRASH_MARKER_DIR}/${variant}__${task}__episode${offset}" ]]; then
        terminal_failures=$((terminal_failures + 1))
      fi
    done
  done
done

timestamp="$(date --iso-8601=seconds)"
service_state="$(systemctl --user is-active "${SERVICE_NAME}" 2>/dev/null || true)"
overall_complete=false
[[ -f "${AUDIT_ROOT}/COMPLETE" ]] && overall_complete=true

{
  printf '[%s] completed=%s/%s terminal_runtime_failures=%s overall_complete=%s service=%s\n' \
    "${timestamp}" "${completed}" "${total}" "${terminal_failures}" \
    "${overall_complete}" "${service_state:-unknown}"
  if [[ -f "${AUDIT_ROOT}/driver.log" ]]; then
    tail -n 3 "${AUDIT_ROOT}/driver.log" | sed 's/^/[driver] /'
  fi
} >> "${MONITOR_LOG}" 2>&1

if [[ "${overall_complete}" != true && "${service_state}" != active && "${service_state}" != activating ]]; then
  printf '[%s] restarting inactive comparison service\n' "${timestamp}" >> "${MONITOR_LOG}"
  systemctl --user restart "${SERVICE_NAME}" >> "${MONITOR_LOG}" 2>&1
fi

