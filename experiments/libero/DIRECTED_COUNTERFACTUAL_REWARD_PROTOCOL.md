# LIBERO directed exact-state imagination-reward protocol

## Status and scope

This document freezes the protocol before inspecting any directed-counterfactual
reward result. The experiment does not train or update FastWAM and does not modify
LIBERO. It reuses the ten validated anchor states and FastWAM imagined goals from:

```text
evaluate_results/exact_state_seed2042_final/manifest.json
```

The task is `libero_goal` task 3:

```text
open the top drawer and put the bowl inside
```

The experiment tests the initial approach phase only. MuJoCo state inspection in the
source experiment showed that the bowl and drawer remain stationary during the first
eight actions, while all ten policy branches move the end effector toward the bowl.
Therefore end-effector-to-bowl distance progress is the predeclared local oracle.

## Fixed counterfactuals

For each of ten anchors, use the first eight source policy actions and collect four
branches from the identical MuJoCo and controller state:

1. `policy`: unchanged source FastWAM action.
2. `toward_bowl`: replace only translation with the anchor end-effector-to-bowl
   direction.
3. `away_from_bowl`: exact opposite of `toward_bowl` translation.
4. `zero`: LIBERO no-op `[0, 0, 0, 0, 0, 0, -1]`.

For every time step, `toward_bowl` and `away_from_bowl` must have:

```text
identical translation magnitude
identical policy rotation command
identical policy gripper command
translation vectors that are exact opposites
```

The translation magnitude is frozen before collection as:

```text
min(norm(policy_translation), 0.25)
```

The `0.25` cap keeps the test local and is not tuned on a reward result. There is no
new model inference: current frames and imagined goals are copied losslessly from the
source manifest. Every branch current and endpoint is rendered eight times.

## Frozen reward candidates

The primary metric remains the previously frozen equal-camera direction reward:

```text
raw_dual = 0.5 * agent_delta_alignment + 0.5 * wrist_delta_alignment
```

Secondary diagnostics are:

```text
agent
wrist
concat_progress
dual_progress
historical frozen camera weight (diagnostic only)
```

One combined candidate is frozen prospectively:

```text
normalized_equal_hybrid = 0.5 * raw_dual / 0.23146070776166341
                        + 0.5 * concat_progress / 0.009989142145974928
```

The two scales are the source seed-2042 policy means recorded before directed action
collection. They are positive-scale normalizers without centering. No weight or scale
will be changed after reading the new result.

## Integrity and manipulation checks

The analyzer must reject the collection if:

- branch names or action shapes differ from the fixed protocol;
- policy actions differ from the source actions;
- toward and away translation vectors are not exact opposites;
- their per-step translation magnitudes differ;
- rotation or gripper commands differ between the paired branches;
- exact-state reconstruction error exceeds `1e-10`;
- branch-current feature stability exceeds cosine distance `1e-4` or feature L2
  `0.015`;
- stored geometry progress is inconsistent with initial and final simulator distance.

The physical manipulation must also pass, on ten anchors:

```text
toward progress > 0: at least 9/10
away progress < 0:   at least 9/10
toward progress > away progress: 10/10
```

## Primary acceptance gates

`raw_dual` supports the directed imagination-reward hypothesis only if all gates pass:

1. Exactly ten independent anchors.
2. Physical manipulation checks above pass.
3. `reward(toward) > reward(away)` in at least `8/10` anchors.
4. The paired-bootstrap 95% lower bound of `reward(toward) - reward(away)` is greater
   than zero.
5. Within-anchor Spearman correlation between reward and oracle distance progress is
   positive in at least `8/10` anchors, using `policy/toward/away/zero`.
6. The paired-anchor bootstrap 95% lower bound of mean Spearman is greater than zero.
7. Mean absolute zero reward is at most `5%` of mean absolute policy reward.

Ten anchors are the independent statistical units. Eight render repeats, four
branches, and simulator action steps are not independent samples. Bootstrap uses
10,000 anchor resamples with seed 2042.

Secondary candidates are reported against the same statistics, but none becomes the
primary metric retrospectively. A secondary candidate may motivate a separately
preregistered validation; it cannot turn a failed primary experiment into a pass.

## Commands

Collection:

```bash
MUJOCO_GL=egl \
PYTHONPATH=/home/ubuntu/sj/LIBERO:$PWD/src:$PWD \
conda run -n fastwam env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python \
  experiments/libero/collect_directed_state_counterfactuals.py \
  --source-manifest evaluate_results/exact_state_seed2042_final/manifest.json \
  --output-dir evaluate_results/directed_state_seed2042_final \
  --num-anchors 10 \
  --translation-magnitude-cap 0.25
```

Analysis:

```bash
PYTHONPATH=$PWD/src:$PWD \
conda run -n fastwam python \
  experiments/libero/analyze_directed_state_counterfactuals.py \
  --manifest evaluate_results/directed_state_seed2042_final/manifest.json \
  --encoder-path /home/ubuntu/zhumj/code/dsrl/ckpt/siglip2-so400m-patch14-224 \
  --calibration-summary evaluate_results/exact_state_seed2042_final/analysis/summary.json \
  --camera-calibration-json evaluate_results/imagination_reward_v2_offline/camera_weight_calibration.json \
  --output-dir evaluate_results/directed_state_seed2042_final/analysis \
  --device cuda \
  --batch-size 32 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 2042
```

Formal output is ignored locally because it contains lossless images and feature
caches. The protocol, code, tests, and final result report are committed to Git.
