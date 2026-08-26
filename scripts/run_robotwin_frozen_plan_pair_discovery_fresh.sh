#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export BASE_RUN_NAME="${BASE_RUN_NAME:-robotwin_frozen_plan_pair_discovery_long_20260825}"
export MANIFEST="${MANIFEST:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_frozen_plan_fresh_candidates_20260826.json}"
export CASES_FILE="${CASES_FILE:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_frozen_plan_fresh_cases_20260826.tsv}"
export NOISE_SEEDS_CSV="${NOISE_SEEDS_CSV:-20260842,20260843,20260844,20260845,20260846,20260847,20260848,20260849}"
export CLEAN_PRESCREEN="${CLEAN_PRESCREEN:-true}"
export TARGET_STRICT_PAIRS="${TARGET_STRICT_PAIRS:-8}"
# The shared status currently contains 67 screens. Keep the previously
# registered cumulative ceiling of 96, so this batch can add at most 29.
export MAX_SCREEN_RUNS="${MAX_SCREEN_RUNS:-96}"
export MAX_ABS_DELTA="${MAX_ABS_DELTA:-0.05}"

exec bash "${PROJECT_ROOT}/scripts/run_robotwin_frozen_plan_pair_discovery_long.sh"
