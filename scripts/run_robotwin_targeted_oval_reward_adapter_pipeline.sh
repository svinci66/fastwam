#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-robotwin_fastwam}"
SEED="${SEED:-44}"
COLLECTION_RUN="${COLLECTION_RUN:-robotwin_place_can_targeted_oval_pairs_20260906}"
COLLECTION_ROOT="${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/${COLLECTION_RUN}"
CASES_JSONL="${COLLECTION_ROOT}/selected_natural_failures.jsonl"
POLICY_RUN_DIR="${PROJECT_ROOT}/evaluate_results/robotwin/robotwin_uncond_3cam_384/${COLLECTION_RUN}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/home/ubuntu/sj/fastwam/checkpoints}"
VAE_PATH="${VAE_PATH:-/home/ubuntu/sj/fastwam/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors}"
BASE_REWARD_JSON="${BASE_REWARD_JSON:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_balanced_pairs_seed44_20260904/merged_wan_vae_head_rewards.json}"
BASE_RESIDUAL_CHECKPOINT="${BASE_RESIDUAL_CHECKPOINT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_paired_rank_seed44_20260905/training/seed44/no_imagination/checkpoint.pt}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/evaluate_results/robotwin_imagination_restart/robotwin_targeted_oval_adapter_pair_seed${SEED}_20260906}"
NEW_REWARD_ROOT="${RUN_ROOT}/targeted_reward"
EXPERT_ROOT="${NEW_REWARD_ROOT}/expert_imagination"
NEW_REWARD_JSON="${NEW_REWARD_ROOT}/wan_vae_head_pair_rewards.json"
BACKFILL_DIR="${NEW_REWARD_ROOT}/video_expert_backfill"
MERGED_REWARD_JSON="${RUN_ROOT}/merged_52pair_wan_vae_head_rewards.json"
REPLAY_DIR="${RUN_ROOT}/replay"
CONTROL_DIR="${RUN_ROOT}/training/no_imagination"
TREATMENT_DIR="${RUN_ROOT}/training/with_imagination"
CONTROL_CONFIG="${PROJECT_ROOT}/configs/rl/robotwin_imagination_adapter_spatial_no_imagination.yaml"
TREATMENT_CONFIG="${PROJECT_ROOT}/configs/rl/robotwin_imagination_adapter_spatial_paired_rank025.yaml"
TRAINING_AUDIT="${RUN_ROOT}/training/paired_training_audit.json"
DEV_MANIFEST="${PROJECT_ROOT}/experiments/robotwin/manifests/robotwin_place_can_basket_adapter_dev3_20260906.json"
DEV_RUN_NAME="robotwin_targeted_oval_adapter_pair_seed${SEED}_dev3_20260906"
DEV_SUMMARY_DIR="${PROJECT_ROOT}/evaluate_results/robotwin_residual_online/${DEV_RUN_NAME}"
DEV_SUMMARY="${DEV_SUMMARY_DIR}/summary.json"

for path in "${COLLECTION_ROOT}/COLLECTION_COMPLETE" "${CASES_JSONL}" \
  "${CHECKPOINT}" "${DATASET_STATS}" "${MODEL_BASE_PATH}" "${VAE_PATH}" \
  "${BASE_REWARD_JSON}" "${BASE_RESIDUAL_CHECKPOINT}" "${CONTROL_CONFIG}" \
  "${TREATMENT_CONFIG}" "${DEV_MANIFEST}"; do
  [[ -e "${path}" ]] || { printf '[targeted-pipeline] missing: %s\n' "${path}" >&2; exit 1; }
done

mkdir -p "${RUN_ROOT}" "${NEW_REWARD_ROOT}"
exec > >(tee -a "${RUN_ROOT}/driver.log") 2>&1

if [[ ! -s "${EXPERT_ROOT}/expert_export_summary.json" ]]; then
  printf '[targeted-pipeline] stage=export_expert_imagination\n'
  conda run --no-capture-output -n "${CONDA_ENV}" \
    env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/export_paired_expert_imagination_trajectories.py" \
    --cases-jsonl "${CASES_JSONL}" --tasks place_can_basket \
    --output-dir "${EXPERT_ROOT}" --checkpoint "${CHECKPOINT}" \
    --dataset-stats "${DATASET_STATS}" --model-base-path "${MODEL_BASE_PATH}" \
    --replan-steps 24 --action-horizon 32 --num-inference-steps 10 \
    --seed 47 --device cuda --mixed-precision bf16
else
  printf '[targeted-pipeline] stage=export_expert_imagination status=skip_complete\n'
fi

if [[ ! -s "${NEW_REWARD_JSON}" ]]; then
  printf '[targeted-pipeline] stage=score_head_wan_vae\n'
  conda run --no-capture-output -n "${CONDA_ENV}" \
    env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/score_natural_failure_vae_pairs.py" \
    --cases-jsonl "${CASES_JSONL}" --tasks place_can_basket \
    --expert-root "${EXPERT_ROOT}" --fastwam-run-dir "${POLICY_RUN_DIR}" \
    --vae-path "${VAE_PATH}" --device cuda --dtype bf16 \
    --reward-cameras head --output-json "${NEW_REWARD_JSON}"
else
  printf '[targeted-pipeline] stage=score_head_wan_vae status=skip_complete\n'
fi

# This is the direct reward falsification gate.  The selected batch contains
# expert successes and natural FastWAM failures, so all five must rank in the
# correct direction before the reward is allowed to train either adapter.
conda run --no-capture-output -n "${CONDA_ENV}" \
  env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src" python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["pair_count"]==5, d["pair_count"]; assert d["correctly_ranked_count"]==5, d["correctly_ranked_count"]; assert d["pairwise_accuracy"]==1.0, d["pairwise_accuracy"]; print("[targeted-pipeline] reward_gate=pass pairwise=5/5")' \
  "${NEW_REWARD_JSON}"

if [[ ! -s "${BACKFILL_DIR}/video_expert_backfill_summary.json" ]]; then
  printf '[targeted-pipeline] stage=backfill_spatial_video_expert\n'
  conda run --no-capture-output -n "${CONDA_ENV}" \
    env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/backfill_video_expert_features.py" \
    --reward-json "${NEW_REWARD_JSON}" --output-dir "${BACKFILL_DIR}" \
    --checkpoint "${CHECKPOINT}" --dataset-stats "${DATASET_STATS}" \
    --model-base-path "${MODEL_BASE_PATH}" --tasks place_can_basket \
    --device cuda --mixed-precision bf16 --num-inference-steps 10 \
    --include-head-spatial-feature
else
  printf '[targeted-pipeline] stage=backfill_spatial_video_expert status=skip_complete\n'
fi

if [[ ! -s "${MERGED_REWARD_JSON}" ]]; then
  printf '[targeted-pipeline] stage=merge_reward old47_plus_new5\n'
  conda run --no-capture-output -n "${CONDA_ENV}" \
    env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/merge_natural_failure_vae_rewards.py" \
    --inputs "${BASE_REWARD_JSON}" "${NEW_REWARD_JSON}" \
    --tasks open_microwave,hanging_mug,place_can_basket \
    --output-json "${MERGED_REWARD_JSON}"
else
  printf '[targeted-pipeline] stage=merge_reward status=skip_complete\n'
fi

if [[ ! -s "${REPLAY_DIR}/manifest.json" ]]; then
  [[ ! -e "${REPLAY_DIR}" ]] || {
    printf '[targeted-pipeline] incomplete replay exists: %s\n' "${REPLAY_DIR}" >&2
    exit 1
  }
  printf '[targeted-pipeline] stage=build_paired_rank_replay\n'
  conda run --no-capture-output -n "${CONDA_ENV}" \
    env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src" python -u \
    "${PROJECT_ROOT}/experiments/robotwin/build_wan_vae_head_awr_replay.py" \
    --reward-json "${MERGED_REWARD_JSON}" --output-dir "${REPLAY_DIR}" \
    --actor-observation-source fastwam_video_expert \
    --observation-encoder-version fastwam_video_expert_final_token_mean_l2_v1 \
    --reward-config "${TREATMENT_CONFIG}" \
    --reward-calibration paired_rank_discount_normalized_v1 \
    --tasks open_microwave,hanging_mug,place_can_basket \
    --minimum-pairwise-accuracy 0.90
else
  printf '[targeted-pipeline] stage=build_paired_rank_replay status=skip_complete\n'
fi

train_adapter() {
  local variant="$1" config="$2" output="$3"
  if [[ -s "${output}/checkpoint.pt" && -s "${output}/adapter_audit.json" ]]; then
    printf '[targeted-pipeline] stage=train variant=%s status=skip_complete\n' "${variant}"
    return
  fi
  [[ ! -e "${output}" ]] || {
    printf '[targeted-pipeline] incomplete training output exists: %s\n' "${output}" >&2
    exit 1
  }
  printf '[targeted-pipeline] stage=train variant=%s\n' "${variant}"
  conda run --no-capture-output -n "${CONDA_ENV}" \
    env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src" python -u \
    "${PROJECT_ROOT}/scripts/train_robotwin_imagination_adapter_awr.py" \
    --config "${config}" --replay-dir "${REPLAY_DIR}" \
    --base-checkpoint "${BASE_RESIDUAL_CHECKPOINT}" --output-dir "${output}" \
    --spatial-reward-json "${MERGED_REWARD_JSON}" --seed "${SEED}" \
    --timeout-bootstrap-value 0.0
}

train_adapter no_imagination "${CONTROL_CONFIG}" "${CONTROL_DIR}"
train_adapter with_imagination "${TREATMENT_CONFIG}" "${TREATMENT_DIR}"

conda run --no-capture-output -n "${CONDA_ENV}" \
  env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_imagination_adapter_training_pair.py" \
  --control-checkpoint "${CONTROL_DIR}/checkpoint.pt" \
  --treatment-checkpoint "${TREATMENT_DIR}/checkpoint.pt" \
  --output-json "${TRAINING_AUDIT}"
touch "${RUN_ROOT}/TRAINING_COMPLETE"

printf '[targeted-pipeline] stage=online_dev3\n'
env RUN_NAME="${DEV_RUN_NAME}" VARIANTS=no_imagination,imagination \
  TASKS=place_can_basket EPISODES=3 BASE_SEED=47 TRIAL_OFFSET=0 \
  INFERENCE_STEPS=10 REPLAN_STEPS=24 TEXT_CFG_SCALE=1.0 \
  TASK_CONFIG=demo_clean INSTRUCTION_TYPE=unseen INSTRUCTION_MODE=official \
  PAPER_ALIGNED=true STRICT_PAIRED=true DETERMINISTIC_INSTRUCTION_BY_SEED=true \
  EXPERT_CHECK=true SEED_MANIFEST_PATH="${DEV_MANIFEST}" \
  NO_IMAGINATION_CHECKPOINT="${CONTROL_DIR}/checkpoint.pt" \
  IMAGINATION_CHECKPOINT="${TREATMENT_DIR}/checkpoint.pt" \
  RESIDUAL_ENCODER_PATH=none \
  RESIDUAL_ENCODER_VERSION=fastwam_video_expert_final_token_mean_l2_v1 \
  RESIDUAL_LANGUAGE_MODE=policy_instruction RESIDUAL_Q_GATE_ENABLED=false \
  RESIDUAL_PAIRED_ADVANTAGE_GATE_ENABLED=false \
  RESIDUAL_SUPPORT_INDEX_PATH=none \
  RESIDUAL_SUPPORT_CIRCUIT_BREAKER_ENABLED=false \
  RESIDUAL_SHADOW_MODE=false RESIDUAL_INTERVENTION_REPLANS=all \
  RESIDUAL_MAX_INTERVENTIONS_PER_EPISODE=none \
  RESIDUAL_OUTCOME_CONFIRMATION_ENABLED=false RESIDUAL_SOFT_SCALE_ENABLED=false \
  SAVE_BASELINE_TRANSITIONS=false SAVE_RESIDUAL_TRANSITIONS=false \
  EVAL_VIDEO_LOG=true \
  bash "${PROJECT_ROOT}/scripts/run_robotwin_residual_iql_online_pair.sh"

conda run --no-capture-output -n "${CONDA_ENV}" \
  env PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src" python -u \
  "${PROJECT_ROOT}/experiments/robotwin/audit_targeted_adapter_dev_gate.py" \
  --summary "${DEV_SUMMARY}" --output-json "${DEV_SUMMARY_DIR}/frozen_dev_gate.json"
touch "${RUN_ROOT}/DEV_GATE_PASSED"
printf '[targeted-pipeline] complete_and_dev_gate_passed run_root=%s\n' "${RUN_ROOT}"
