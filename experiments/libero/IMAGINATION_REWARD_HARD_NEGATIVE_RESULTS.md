# LIBERO Reward V2 same-phase hard-negative validation

## Scope

This experiment replaces the original distant wrong-goal control with a more
controlled hard negative.  It uses the existing 1156 Reward V2 transitions and cached
camera features.  It does not start LIBERO, encode images, collect rollouts, tune
camera weights, or train a policy.

The previously selected reward is frozen:

```text
agent weight: 0.25
wrist weight: 0.75
agent scale: 0.5458950427
wrist scale: 0.3160181364
```

## Hard-negative protocol

For each transition, the candidate pool must have:

```text
same seed
same task
same action mode
same early/middle/late phase
different trial
```

Selection then proceeds in two fixed steps:

1. Find the five candidates with the smallest equal-dual camera distance between
   current observations.
2. Among those five, choose the candidate whose predicted goal has the largest
   equal-dual distance from the correct predicted goal.

The selector sees current and goal features only.  It never sees the executed actual
frame, reward value, episode success, or correct-vs-wrong result.  All 1156 transitions
have a valid hard negative; no fallback to the old distant selector is needed.

## Did the selector produce a harder control?

Yes.  Current-feature cosine distance is lower for the new negative:

| Seed | New mean current distance | Old mean current distance | New is closer than old |
| ---: | ---: | ---: | ---: |
| 42 | 0.03307 | 0.05833 | 86.95% |
| 1042 | 0.03367 | 0.06158 | 92.19% |

The new goal is also less trivially different from the correct goal:

| Seed | New mean goal distance | Old mean goal distance |
| ---: | ---: |
| 42 | 0.06479 | 0.07476 |
| 1042 | 0.06470 | 0.08752 |

Thus the new comparison removes much of the current-state and task-phase mismatch.
Although the selector chooses the most different goal among five local neighbors, that
goal remains closer than the old cross-phase distant goal on average.

## All-transition result

The fraction of transitions where the correct goal receives more reward than the hard
negative is:

| Reward | Seed 42 | Seed 1042 |
| --- | ---: | ---: |
| Agent camera | 63.67% | 65.53% |
| Wrist camera | 59.44% | 59.93% |
| Raw equal dual | 64.20% | 65.53% |
| Frozen normalized `0.25 / 0.75` | 63.32% | 62.82% |

The predeclared all-transition `>= 70%` gate fails.  The formal decision remains:

```text
do_not_use_for_rl
```

## Why the all-transition number falls

Action-mode stratification reveals that almost the entire failure comes from zero
actions.

### Policy and noise transitions only

| Reward | Seed 42 | Seed 1042 |
| --- | ---: | ---: |
| Agent camera | 75.71% | 76.99% |
| Wrist camera | 68.45% | 65.49% |
| Raw equal dual | 77.29% | 78.17% |
| Frozen normalized `0.25 / 0.75` | 73.82% | 73.16% |

### Zero/no-op transitions only

| Reward | Seed 42 | Seed 1042 |
| --- | ---: | ---: |
| Agent camera | 48.40% | 50.00% |
| Wrist camera | 48.00% | 52.40% |
| Raw equal dual | 47.60% | 48.40% |
| Frozen normalized `0.25 / 0.75` | 50.00% | 48.80% |

For a no-op transition:

```text
actual_delta = feature(actual) - feature(current) approximately 0
```

The direction reward is therefore approximately zero for every possible imagined
goal.  It has no information with which to identify one goal direction over another,
so correct-vs-wrong accuracy should approach chance.  Requiring zero actions to exceed
70% goal identity is mathematically inconsistent with this reward definition.

This is not a no-op reward exploit: previous validation shows the frozen reward's
seed-1042 episode mean is `0.0040` for zero, versus `0.6435` for policy.  Zero receives
almost no reward, which is the desired behavior.  It simply cannot be used to measure
goal-direction specificity.

The original all-transition gate is retained because changing it after seeing these
results would be post-hoc.  The policy/noise result is reported as a diagnostic, not as
a retrospective pass.

## Camera-weight finding

The frozen wrist-heavy weight is not robust to the improved negative protocol.

On action-producing seed-42 transitions:

```text
raw equal dual:       77.29%
frozen wrist-heavy:   73.82%
```

On action-producing seed-1042 transitions:

```text
raw equal dual:       78.17%
frozen wrist-heavy:   73.16%
```

The previous `0.25 agent / 0.75 wrist` calibration optimized discrimination against
more distant negatives.  Once current state and phase are matched, the agent camera is
more informative and the wrist-heavy candidate loses `3.47` points on seed 42 and
`5.01` points on seed 1042 relative to raw equal dual.  Therefore that fixed weight is
rejected as a final reward choice; no replacement is selected on seed 1042.

## Interpretation

The hard-negative test changes the diagnosis in two useful ways:

1. Reward V2 contains state-specific goal information for actual policy/noise changes.
   The frozen reward exceeds 70% on both seeds, and raw equal dual reaches about
   77--78%.
2. Goal identity is undefined for a no-op under a change-direction reward.  Goal
   specificity and no-op suppression must be evaluated with separate metrics.

This supports the action-result-versus-imagination hypothesis more strongly than the
old distant-negative test, but it does not authorize RL yet because the corrected
acceptance rule was discovered on already inspected data.

The next prospective validation should predeclare two separate gates before collecting
a new seed or task:

```text
action-producing transitions:
  correct hard goal > wrong hard goal >= 70%

zero transitions:
  reward magnitude remains near zero relative to policy/noise
```

A third unseen seed is required to validate that corrected rule without reusing seed
1042 for another selection decision.

## Reproduction

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/fastwam/bin/python \
  experiments/libero/validate_same_phase_hard_negatives.py \
  --input-jsonl evaluate_results/imagination_reward_v2_offline/reward_v2_transitions.jsonl \
  --feature-cache evaluate_results/imagination_reward_v2_offline/camera_siglip_features.npz \
  --camera-calibration-json evaluate_results/imagination_reward_v2_offline/camera_weight_calibration.json \
  --output-json evaluate_results/imagination_reward_v2_offline/same_phase_hard_negative_validation.json \
  --nearest-k 5 \
  --goal-specificity-threshold 0.70
```

The full summary and per-transition selections remain in the ignored local
`evaluate_results/imagination_reward_v2_offline/` directory.
