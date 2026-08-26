#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export BASE_RUN_NAME="${BASE_RUN_NAME:-robotwin_frozen_plan_pair_discovery_long_20260825}"
export MANIFEST="${MANIFEST:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_frozen_plan_targeted_candidates_20260826.json}"
export CASES_FILE="${CASES_FILE:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_frozen_plan_targeted_cases_20260826.tsv}"
export NOISE_SEEDS_CSV="${NOISE_SEEDS_CSV:-20260834,20260835,20260836,20260837,20260838,20260839,20260840,20260841}"
export CLEAN_PRESCREEN="${CLEAN_PRESCREEN:-true}"
export TARGET_STRICT_PAIRS="${TARGET_STRICT_PAIRS:-8}"
# The shared status already contains 64 screens. This permits exactly 32 new
# targeted screens before the search must be redesigned again.
export MAX_SCREEN_RUNS="${MAX_SCREEN_RUNS:-96}"
export MAX_ABS_DELTA="${MAX_ABS_DELTA:-0.05}"

exec bash "${PROJECT_ROOT}/scripts/run_robotwin_frozen_plan_pair_discovery_long.sh"
