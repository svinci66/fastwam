#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
RUN_NAME="${RUN_NAME:-robotwin_place_can_natural_pair_expansion5_20260904}"
TASK="place_can_basket"
TARGET_FAILURES="${TARGET_FAILURES:-5}"
INITIAL_CANDIDATES="${INITIAL_CANDIDATES:-10}"
MAX_CANDIDATES="${MAX_CANDIDATES:-20}"
CANDIDATE_INCREMENT="${CANDIDATE_INCREMENT:-2}"
START_CANDIDATE_SEED="${START_CANDIDATE_SEED:-700}"
MAX_CANDIDATE_SEED="${MAX_CANDIDATE_SEED:-749}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-5}"
ARTIFACT_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${RUN_NAME}"
STATUS_JSONL="${ARTIFACT_ROOT}/status.jsonl"
STRICT_AUDIT="${ARTIFACT_ROOT}/strict_pair_audit.json"
SELECTED_JSONL="${ARTIFACT_ROOT}/selected_natural_failures.jsonl"
SELECTION_AUDIT="${ARTIFACT_ROOT}/selection_audit.json"

for value in "${TARGET_FAILURES}" "${INITIAL_CANDIDATES}" \
  "${MAX_CANDIDATES}" "${CANDIDATE_INCREMENT}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    printf '[place-can-expansion] invalid positive integer: %s\n' "${value}" >&2
    exit 1
  }
done
(( INITIAL_CANDIDATES <= MAX_CANDIDATES )) || {
  printf '[place-can-expansion] initial candidates exceed maximum\n' >&2
  exit 1
}

mkdir -p "${ARTIFACT_ROOT}"
candidates="${INITIAL_CANDIDATES}"
while true; do
  printf '[place-can-expansion] collect candidates=%s target_failures=%s\n' \
    "${candidates}" "${TARGET_FAILURES}"
  env RUN_NAME="${RUN_NAME}" TASKS="${TASK}" \
    EPISODES_PER_TASK="${candidates}" \
    START_CANDIDATE_SEED="${START_CANDIDATE_SEED}" \
    MAX_CANDIDATE_SEED="${MAX_CANDIDATE_SEED}" \
    COOLDOWN_SECONDS="${COOLDOWN_SECONDS}" CONDA_ENV="${CONDA_ENV}" \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_local_expert_pair_smoke.sh"

  failure_count="$(conda run --no-capture-output -n "${CONDA_ENV}" python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(int(d["per_task"][sys.argv[2]]["natural_failures"]))' \
    "${ARTIFACT_ROOT}/summary.json" "${TASK}")"
  printf '[place-can-expansion] strict natural failures=%s/%s\n' \
    "${failure_count}" "${candidates}"
  if (( failure_count >= TARGET_FAILURES )); then
    break
  fi
  if (( candidates >= MAX_CANDIDATES )); then
    printf '[place-can-expansion] insufficient natural failures at maximum candidates\n' >&2
    exit 1
  fi
  candidates="$((candidates + CANDIDATE_INCREMENT))"
  if (( candidates > MAX_CANDIDATES )); then candidates="${MAX_CANDIDATES}"; fi
done

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/select_natural_failure_pairs.py" \
  --status-jsonl "${STATUS_JSONL}" --strict-audit "${STRICT_AUDIT}" \
  --task "${TASK}" --count "${TARGET_FAILURES}" \
  --output-jsonl "${SELECTED_JSONL}" --output-audit "${SELECTION_AUDIT}"

touch "${ARTIFACT_ROOT}/COLLECTION_COMPLETE"
printf '[place-can-expansion] complete: %s\n' "${SELECTION_AUDIT}"
