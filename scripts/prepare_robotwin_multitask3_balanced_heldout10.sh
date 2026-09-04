#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/sj/fastwam/RoboTwin-upstream}"
COUNT="${COUNT:-10}"
TASKS="open_microwave hanging_mug place_can_basket"
POOL="${POOL:-${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_multitask3_balanced_heldout10_candidate_pool_20260904.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_balanced_heldout10_20260904}"
MANIFEST="${ARTIFACT_ROOT}/heldout_manifest.json"

[[ "${COUNT}" =~ ^[1-9][0-9]*$ ]] || { printf 'COUNT must be positive\n' >&2; exit 1; }
[[ -s "${POOL}" ]] || { printf 'Missing candidate pool: %s\n' "${POOL}" >&2; exit 1; }
mkdir -p "${ARTIFACT_ROOT}"
exec > >(tee -a "${ARTIFACT_ROOT}/driver.log") 2>&1

valid_video() {
  local video="$1" duration
  [[ -s "${video}" ]] || return 1
  duration="$(ffprobe -v error -show_entries format=duration \
    -of default=nw=1:nk=1 "${video}" 2>/dev/null)"
  [[ -n "${duration}" ]] || return 1
  awk -v duration="${duration}" 'BEGIN { exit !(duration > 0) }'
}

single_manifests=()
for task in ${TASKS}; do
  case "${task}" in
    open_microwave) start_seed=4800200; end_seed=4800239 ;;
    hanging_mug) start_seed=4800240; end_seed=4800279 ;;
    place_can_basket) start_seed=4800280; end_seed=4800319 ;;
    *) exit 2 ;;
  esac
  bundle="${ARTIFACT_ROOT}/expert_feasibility/${task}"
  mkdir -p "${bundle}"
  next_seed="${start_seed}"
  for ((episode = 0; episode < COUNT; episode++)); do
    marker="${ARTIFACT_ROOT}/.${task}_expert_episode$(printf '%03d' "${episode}")_complete"
    metadata="${bundle}/pair_metadata/episode${episode}.json"
    expert_hdf5="${bundle}/data/episode${episode}.hdf5"
    expert_video="${bundle}/video/episode${episode}.mp4"
    instruction_file="${bundle}/instructions/episode${episode}.json"
    if [[ -f "${marker}" && -s "${metadata}" && -s "${expert_hdf5}" \
      && -s "${instruction_file}" ]] && valid_video "${expert_video}"; then
      seed="$(conda run --no-capture-output -n "${CONDA_ENV}" python -c \
        'import json,sys; print(int(json.load(open(sys.argv[1]))["seed"]))' "${metadata}")"
      next_seed="$((seed + 1))"
      printf '[heldout10] skip task=%s episode=%s seed=%s\n' "${task}" "${episode}" "${seed}"
      continue
    fi
    selected=false
    seed="${next_seed}"
    while (( seed <= end_seed )); do
      printf '[heldout10] expert screen task=%s episode=%s candidate_seed=%s\n' \
        "${task}" "${episode}" "${seed}"
      if conda run --no-capture-output -n "${CONDA_ENV}" \
        env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
        python -u "${PROJECT_ROOT}/experiments/robotwin/collect_local_expert_pair_episode.py" \
        --robotwin-root "${ROBOTWIN_ROOT}" --task "${task}" --task-config demo_clean \
        --seed "${seed}" --episode-index "${episode}" --output-bundle "${bundle}"; then
        selected=true
        break
      else
        status="$?"
        if [[ "${status}" -eq 20 || "${status}" -eq 21 ]]; then
          printf '[heldout10] reject task=%s seed=%s status=%s\n' "${task}" "${seed}" "${status}"
          seed="$((seed + 1))"
          continue
        fi
        exit "${status}"
      fi
    done
    [[ "${selected}" == true ]] || { printf 'Candidate pool exhausted for %s\n' "${task}" >&2; exit 1; }
    [[ -s "${metadata}" && -s "${expert_hdf5}" && -s "${instruction_file}" ]] || {
      printf '[heldout10] incomplete expert artifact task=%s episode=%s\n' "${task}" "${episode}" >&2
      exit 1
    }
    valid_video "${expert_video}" || {
      printf '[heldout10] invalid expert video task=%s episode=%s\n' "${task}" "${episode}" >&2
      exit 1
    }
    touch "${marker}"
    next_seed="$((seed + 1))"
  done
  single_manifest="${ARTIFACT_ROOT}/${task}_heldout_manifest.json"
  conda run --no-capture-output -n "${CONDA_ENV}" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_expert_feasible_seed_manifest.py" \
    --candidate-pool "${POOL}" --metadata-dir "${bundle}/pair_metadata" \
    --instructions-dir "${bundle}/instructions" --task "${task}" \
    --count "${COUNT}" --output-json "${single_manifest}"
  single_manifests+=("${single_manifest}")
done

conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/merge_expert_feasible_manifests.py" \
  --inputs "${single_manifests[@]}" \
  --tasks open_microwave,hanging_mug,place_can_basket \
  --output-json "${MANIFEST}"

touch "${ARTIFACT_ROOT}/HELDOUT_MANIFEST_COMPLETE"
printf '[heldout10] complete: %s\n' "${MANIFEST}"
