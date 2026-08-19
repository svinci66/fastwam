#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${ROBOMIMIC_COUNTERFACTUAL_OUTPUT_ROOT:-${PROJECT_ROOT}/evaluate_results/robomimic_counterfactual}"
mkdir -p "${OUTPUT_ROOT}"

LOG_FILE="${OUTPUT_ROOT}/long_collection.log"
SESSION_NAME="${ROBOMIMIC_COUNTERFACTUAL_TMUX_SESSION:-fastwam_robomimic_counterfactual}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "collection already running in tmux session ${SESSION_NAME}" >&2
    exit 1
fi

tmux new-session \
    -d \
    -s "${SESSION_NAME}" \
    -c "${PROJECT_ROOT}" \
    "exec bash scripts/run_robomimic_counterfactual_collection.sh long > '${LOG_FILE}' 2>&1"
PANE_PID="$(tmux list-panes -t "${SESSION_NAME}" -F '#{pane_pid}')"
echo "started tmux session ${SESSION_NAME} (pane PID ${PANE_PID})"
echo "log: ${LOG_FILE}"
