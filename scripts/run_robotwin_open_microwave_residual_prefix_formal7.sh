#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_PREFIX="${RUN_PREFIX:-robotwin_open_microwave_residual_prefix_formal7_20260831}"
ARTIFACT_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${RUN_PREFIX}"
EPISODE_ORDER="${EPISODE_ORDER:-4,5,3,1,2,0,6}"

mkdir -p "${ARTIFACT_ROOT}"
exec > >(tee -a "${ARTIFACT_ROOT}/driver.log") 2>&1
IFS=',' read -r -a episodes <<< "${EPISODE_ORDER}"
for episode in "${episodes[@]}"; do
  episode_run="${RUN_PREFIX}_ep$(printf '%02d' "${episode}")"
  output_dir="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${episode_run}"
  if [[ -f "${output_dir}/COMPLETE" && -s "${output_dir}/triplet_audit.json" ]]; then
    printf '[formal7] skip complete episode=%s\n' "${episode}"
    continue
  fi
  printf '[formal7] start episode=%s run=%s\n' "${episode}" "${episode_run}"
  env RUN_NAME="${episode_run}" TRIAL_OFFSET="${episode}" TARGET_OPEN_RATIO=0.5 \
    bash "${PROJECT_ROOT}/scripts/run_robotwin_open_microwave_residual_prefix_chunk.sh"
  printf '[formal7] complete episode=%s\n' "${episode}"
done

env FORMAL_ROOT="${ARTIFACT_ROOT}" \
  conda run --no-capture-output -n "${CONDA_ENV:-robotwin_fastwam}" python -u -c '
import json
import os
from pathlib import Path
root = Path(os.environ["FORMAL_ROOT"])
rows = []
for path in sorted(root.parent.glob(root.name + "_ep*/triplet_audit.json")):
    payload = json.loads(path.read_text())
    rows.append({
        "path": str(path.resolve()),
        "accepted": payload["accepted"],
        "seed": payload["branches"]["baseline"]["environment_seed"],
        "replan": payload["intervention_replan"],
        **payload["causal_open_ratio_delta"],
    })
summary = {
    "schema_version": "robotwin_open_microwave_residual_prefix_formal7_v1",
    "accepted_count": sum(row["accepted"] for row in rows),
    "episode_count": len(rows),
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
'

touch "${ARTIFACT_ROOT}/COMPLETE"
printf '[formal7] complete summary=%s\n' "${ARTIFACT_ROOT}/summary.json"
