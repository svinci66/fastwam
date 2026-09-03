#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_ROOT}/scripts/run_robotwin_video_expert_multitask3_three_way.sh}"
RUNNER_PATTERN="${RUNNER_PATTERN:-$(basename "${RUN_SCRIPT}")}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_no_imagination_epochs3_seed42_20260903}"
ONLINE_RUN_NAME="${ONLINE_RUN_NAME:-robotwin_video_expert_multitask3_three_way_epoch003_5ep_20260903}"
DRIVER_LOG="${DRIVER_LOG:-${RUN_ROOT}/three_way_driver.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${RUN_ROOT}/three_way_watchdog.log}"
COMPLETE_MARKER="${COMPLETE_MARKER:-${RUN_ROOT}/THREE_WAY_COMPLETE}"
SUMMARY_JSON="${SUMMARY_JSON:-${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${ONLINE_RUN_NAME}/summary.json}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STALE_SECONDS="${STALE_SECONDS:-600}"
LOCK_FILE="${LOCK_FILE:-/tmp/fastwam_video_expert_three_way_watchdog.lock}"

mkdir -p "${RUN_ROOT}"
exec 9>"${LOCK_FILE}"
flock -n 9 || { printf '[watchdog] another watchdog holds %s\n' "${LOCK_FILE}"; exit 0; }

log() {
  printf '[%s] [watchdog] %s\n' "$(date '+%F %T')" "$*" | tee -a "${WATCHDOG_LOG}"
}

runner_alive() {
  pgrep -f "bash .*${RUNNER_PATTERN}" >/dev/null
}

evaluation_alive() {
  pgrep -f "eval_robotwin_single.py.*${ONLINE_RUN_NAME}" >/dev/null
}

result_complete() {
  [[ -f "${COMPLETE_MARKER}" && -s "${SUMMARY_JSON}" ]]
}

start_runner() {
  log 'starting or resuming three-way evaluation'
  (
    cd "${PROJECT_ROOT}"
    bash "${RUN_SCRIPT}" >>"${DRIVER_LOG}" 2>&1
  ) &
}

terminate_stalled_evaluation() {
  local -a pids=()
  mapfile -t pids < <(pgrep -f "eval_robotwin_single.py.*${ONLINE_RUN_NAME}" || true)
  if (( ${#pids[@]} == 0 )); then
    return
  fi
  log "stalled evaluation detected; sending TERM to pids=${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep 10
  mapfile -t pids < <(pgrep -f "eval_robotwin_single.py.*${ONLINE_RUN_NAME}" || true)
  if (( ${#pids[@]} > 0 )); then
    log "evaluation ignored TERM; sending KILL to pids=${pids[*]}"
    kill -KILL "${pids[@]}" 2>/dev/null || true
  fi
}

log "monitoring run=${ONLINE_RUN_NAME} poll=${POLL_SECONDS}s stale=${STALE_SECONDS}s"
while ! result_complete; do
  if ! runner_alive; then
    start_runner
    sleep "${POLL_SECONDS}"
    continue
  fi

  if evaluation_alive && [[ -f "${DRIVER_LOG}" ]]; then
    now="$(date +%s)"
    modified="$(stat -c %Y "${DRIVER_LOG}")"
    age="$((now - modified))"
    if (( age >= STALE_SECONDS )); then
      terminate_stalled_evaluation
    fi
  fi
  sleep "${POLL_SECONDS}"
done

log "complete summary=${SUMMARY_JSON}"
