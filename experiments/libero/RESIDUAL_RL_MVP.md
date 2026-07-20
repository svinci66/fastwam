# LIBERO residual-RL MVP

## Scope

This is a resource-bounded first RL implementation for testing whether the frozen
FastWAM imagination-progress signal improves control. It is deliberately split into
three jobs:

```text
frozen FastWAM + LIBERO collector
                -> raw action-chunk records
frozen SigLIP reward encoder
                -> checksummed compact replay shard
small residual actor + value critic
                -> offline AWR learner
```

FastWAM is never passed to the learner. On a single GPU, collect a rollout batch,
release the large inference model, and only then start the learner. The local LIBERO
source tree does not need to be modified.

This MVP is not full IQL and does not update FastWAM's ActionDiT. It trains a bounded
MLP residual on top of the immutable released action chunk. The gripper residual is
zero in the first configuration, so FastWAM retains control of the discontinuous
gripper command.

## Implemented safety and integrity constraints

- Every transition represents the action chunk that was actually executed, not the
  unexecuted remainder of the 32-step FastWAM prediction.
- The first protocol uses `target_k=8`, aligned to the `t+8` predicted video frame.
- `effective_k`, raw per-step simulator rewards, next proprio, `terminated`, and
  `truncated` are stored separately.
- A partial chunk is not marked imagination-aligned unless it has a matched predicted
  goal. Invalid alignment receives zero imagination shaping.
- Policy, predictor, and reward-encoder versions plus environment, goal, and action
  seeds are mandatory replay fields.
- Replay arrays and metadata have SHA-256 checksums and a schema version.
- A replay shard cannot mix reward-encoder versions, action horizons, or tensor shapes.
- Timeout is never silently treated as a true terminal. Monte Carlo return generation
  requires an explicit bootstrap value for every truncated episode.
- Raw LIBERO reward is retained for audit, but its default coefficient is zero because
  LIBERO's reward is itself the success indicator. This prevents success being counted
  both as raw environment reward and as the configured success bonus.
- Cumulative absolute imagination shaping is capped at at most half of one success
  bonus per episode.
- The same immutable replay can be relabeled with or without imagination reward. This
  enables a matched reward ablation without recollecting different trajectories.

## Server preparation

Install the repository environment, LIBERO, the released FastWAM checkpoint, its
dataset statistics, and the same frozen SigLIP checkpoint used for reward validation.
Record an immutable identifier for each. A path such as `latest` is not a sufficient
version identifier; use a commit, model revision, or content checksum.

Before collection, require all of the following:

```bash
nvidia-smi
conda run -n fastwam python -c "import torch; assert torch.cuda.is_available()"
```

## 1. Collect raw K=8 chunks

Run the existing single-task evaluation with future-video generation and transition
saving enabled. The direct FastWAM action remains the action baseline while the joint
inference path supplies the predicted future goal. Collect both policy behavior and a
controlled noisy behavior with the same task and environment seeds. A policy-only
replay has zero residual targets and cannot teach a residual actor how reward ranks
alternative actions.

Policy behavior:

```bash
conda run -n fastwam env PYTHONPATH=.:./src \
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/server/checkpoints/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=/server/checkpoints/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_goal \
  EVALUATION.task_id=3 \
  EVALUATION.num_trials=10 \
  EVALUATION.replan_steps=8 \
  EVALUATION.visualize_future_video=true \
  EVALUATION.save_imagination_transitions=true \
  EVALUATION.imagination_use_direct_action=true \
  EVALUATION.action_mode=policy \
  EVALUATION.output_dir=/server/rollouts/task3_seed42_policy \
  seed=42
```

Matched noisy behavior:

```bash
conda run -n fastwam env PYTHONPATH=.:./src \
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/server/checkpoints/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=/server/checkpoints/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_goal \
  EVALUATION.task_id=3 \
  EVALUATION.num_trials=10 \
  EVALUATION.replan_steps=8 \
  EVALUATION.visualize_future_video=true \
  EVALUATION.save_imagination_transitions=true \
  EVALUATION.imagination_use_direct_action=true \
  EVALUATION.action_mode=noise \
  EVALUATION.action_noise_std=0.15 \
  EVALUATION.output_dir=/server/rollouts/task3_seed42_noise015 \
  seed=42
```

The first server smoke test can use this one noise scale. A formal run should freeze a
small exploration mixture on a development split; do not select its scale from final
success results. The replay records `behavior_mode` and `action_noise_std` so mixtures
remain auditable.

Each saved transition contains lossless current/goal/actual images and
`rollout_arrays.npz` with:

```text
proprio
next_proprio
baseline_actions
planned_executed_actions
executed_actions
environment_rewards
```

`executed_actions`, rather than `planned_executed_actions`, is authoritative when an
action ensembler or early termination changes what reaches the simulator.

## 2. Build a compact replay shard

Run frozen SigLIP feature extraction after collection:

```bash
conda run -n fastwam env PYTHONPATH=.:./src \
python experiments/libero/build_residual_rl_replay.py \
  --input-dir /server/rollouts/task3_seed42_policy/libero_goal/imagination_transitions \
  --input-dir /server/rollouts/task3_seed42_noise015/libero_goal/imagination_transitions \
  --output-dir /server/replays/task3_seed42_siglip_v1 \
  --encoder-path /server/checkpoints/siglip-so400m-patch14-384 \
  --reward-encoder-version siglip-so400m-patch14-384@IMMUTABLE_REVISION \
  --reward-config configs/rl/libero_residual_awr_mvp.yaml \
  --device cuda \
  --agent-weight 0.5 \
  --wrist-weight 0.5
```

Equal camera weights reproduce the frozen validation metric. If camera weights are
changed, select and freeze them on a separate development split before building the
formal replay. Do not tune weights on the final evaluation task and then report that
same task as held-out evidence.

## 3. Validate without training

The validation command loads the replay, verifies checksums and shapes, relabels its
reward, constructs actor/critic shapes, and computes returns. It performs no optimizer
step and writes no checkpoint:

```bash
conda run -n fastwam env PYTHONPATH=.:./src \
python scripts/train_libero_residual_awr.py \
  --config configs/rl/libero_residual_awr_mvp.yaml \
  --replay-dir /server/replays/task3_seed42_siglip_v1 \
  --output-dir /server/runs/unused_validate_only \
  --timeout-bootstrap-value 0.0 \
  --validate-only
```

Using zero for a timeout bootstrap is an explicit conservative MVP choice, not a claim
that timeout is a true terminal. Once a stable target critic exists, provide per-episode
bootstrap values with `--timeout-bootstrap-json` instead.

## 4. Run the matched reward ablation on the server

Train the control without imagination reward:

```bash
conda run -n fastwam env PYTHONPATH=.:./src \
python scripts/train_libero_residual_awr.py \
  --config configs/rl/libero_residual_awr_no_imagination.yaml \
  --replay-dir /server/replays/task3_seed42_siglip_v1 \
  --output-dir /server/runs/task3_seed42_no_imagination \
  --timeout-bootstrap-value 0.0
```

Train the matched learner with imagination reward:

```bash
conda run -n fastwam env PYTHONPATH=.:./src \
python scripts/train_libero_residual_awr.py \
  --config configs/rl/libero_residual_awr_mvp.yaml \
  --replay-dir /server/replays/task3_seed42_siglip_v1 \
  --output-dir /server/runs/task3_seed42_with_imagination \
  --timeout-bootstrap-value 0.0
```

The two jobs use identical transitions, actor/critic capacity, optimizer settings,
seeds, and update counts. Their only intended reward difference is
`imagination_weight: 0.0` versus `1.0`.

The frozen FastWAM policy remains the no-RL baseline. A server rollout adapter that
loads the residual checkpoint and performs fixed-seed A/B/C evaluation is the next
implementation step; do not interpret offline actor loss as a task-success result.

## Outputs

The learner writes:

```text
checkpoint.pt
history.json
run_config.json
```

`checkpoint.pt` contains actor and critic weights, their exact configurations, the
reward/AWR configurations, a replay-manifest checksum, and dataset summary. It does not
contain or modify the released FastWAM checkpoint.

## Current non-goals

- No local training was used to validate this implementation.
- No PPO or GRPO is implemented.
- No full IQL Q/V pair is claimed.
- No end-to-end gradient passes through the 5B Video Expert.
- No ActionDiT AWFM update is claimed; this first learner is a separate residual head.
- No online learner reads weights while a collector is in the middle of an episode.
