#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${ROBOMIMIC_DEPLOYABLE_EXPANSION_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_deployable_expansion}"
SESSION_NAME="${ROBOMIMIC_DEPLOYABLE_EXPANSION_SESSION:-fastwam_robomimic_deployable_expansion}"
LOG_FILE="${OUTPUT_ROOT}/driver.log"
mkdir -p "${OUTPUT_ROOT}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "pipeline already running in tmux session ${SESSION_NAME}" >&2
    exit 1
fi

tmux new-session \
    -d \
    -s "${SESSION_NAME}" \
    -c "${PROJECT_ROOT}" \
    "exec bash scripts/run_robomimic_deployable_expansion.sh > '${LOG_FILE}' 2>&1"
PANE_PID="$(tmux list-panes -t "${SESSION_NAME}" -F '#{pane_pid}')"
echo "started tmux session ${SESSION_NAME} (pane PID ${PANE_PID})"
echo "log: ${LOG_FILE}"
