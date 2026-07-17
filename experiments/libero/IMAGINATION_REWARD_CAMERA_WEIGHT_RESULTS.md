# LIBERO Reward V2 camera-weight calibration

## Question

The first Reward V2 diagnostic averaged the agent and wrist camera rewards with equal
coefficients.  Because the cameras observe different content and have different reward
scales, this experiment asks whether calibration-only normalization and unequal fixed
weights improve the reward without using the validation seed for tuning.

This remains an offline diagnostic.  It reuses the existing Reward V2 transition JSONL
and starts neither LIBERO nor a neural model.  No rollout, training, or feature encoding
is performed.

## Frozen protocol

The protocol was fixed before reading the weight results:

```text
calibration seed: 42
validation seed: 1042
camera scale: seed-42 policy/noise 90th percentile of absolute camera reward
agent-weight candidates: 0.25, 0.40, 0.50, 0.60, 0.75
wrist weight: 1 - agent weight
goal-specificity gate: >= 70%
episode success AUC gate: >= 0.90
paired action-quality gate: policy > noise > zero in 5/5 trials
success/failure gate: successful mode > failed mode in every comparable pair
```

The per-camera reward is divided by its positive scale without centering:

```text
r_normalized =
    w_agent * r_agent / scale_agent
    +
    w_wrist * r_wrist / scale_wrist
```

This preserves reward sign and the zero-action reference.  Zero transitions are
excluded from scale estimation so that abundant no-ops cannot collapse the scale.

Only candidates passing all seed-42 gates are eligible.  The eligible candidate with
the largest correct-goal-vs-wrong-goal fraction is selected; remaining ties are broken
by paired ordering, success-pair ordering, AUC, and closeness to equal weights.  The
seed-1042 rows are not consulted until after selection.

## Calibration result: seed 42

The fixed seed-42 scales are:

```text
agent scale: 0.5458950427
wrist scale: 0.3160181364
```

The different scales confirm that equal raw coefficients do not imply equal semantic
or numerical contribution.

| Agent weight | Wrist weight | Correct > wrong | Success AUC | Paired order | Success > failure pairs | Passes calibration |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.25 | 0.75 | 71.78% | 0.982 | 5/5 | 10/10 | yes |
| 0.40 | 0.60 | 71.60% | 0.964 | 5/5 | 10/10 | yes |
| 0.50 | 0.50 | 70.90% | 0.964 | 5/5 | 10/10 | yes |
| 0.60 | 0.40 | 69.84% | 0.964 | 5/5 | 10/10 | no |
| 0.75 | 0.25 | 69.14% | 0.964 | 5/5 | 10/10 | no |

The calibration rule selects:

```text
w_agent = 0.25
w_wrist = 0.75
```

After scale normalization, placing more weight on the wrist camera improves target
specificity on seed 42 while retaining all action-quality orderings.  This is consistent
with the wrist view contributing local gripper/object information that is weaker in the
global agent view.

## Frozen validation result: seed 1042

The seed-42 scales and `0.25 / 0.75` weights were frozen before this evaluation.

| Metric | Frozen seed-1042 result | Required | Pass |
| --- | ---: | ---: | :---: |
| Correct goal > wrong goal | 69.44% | >= 70% | no |
| Episode success ROC AUC | 0.982 | >= 0.90 | yes |
| `policy > noise > zero` | 5/5 | 5/5 | yes |
| Successful mode > failed mode | 10/10 | 10/10 | yes |

The episode means remain well separated:

```text
policy: 0.643515
noise:  0.459982
zero:   0.004027
```

The critical seed-1042 trial 2 remains fixed:

```text
policy (success): 0.675353
noise  (success): 0.495362
zero   (failure): 0.007066
```

Compared with the previous unnormalized equal-dual candidate, seed-1042 goal
specificity rises from `68.93%` to `69.44%`, an improvement of about `0.51` percentage
points.  It still misses the predeclared threshold by about `0.56` percentage points.

## Decision

The experiment confirms the user's concern: camera scale and viewpoint should not be
treated as interchangeable.  Seed-42 calibration favors the wrist camera after scale
normalization, and the frozen weighting slightly improves target specificity on the
independent seed while preserving action-quality rankings.

However, the selected candidate fails the frozen validation goal-specificity gate.
The decision remains:

```text
do_not_use_for_rl
```

No alternative weight is selected using seed 1042, even if another candidate might
look better there.  That would turn the validation set into another tuning set.

The remaining failure is unlikely to be solved reliably by finer weight scanning.  The
next minimum diagnostic should improve the wrong-goal protocol: compare correct goals
against same-task, same-phase hard negatives with similar current observations.  This
will determine whether the remaining error comes from camera fusion or from an
ill-posed wrong-goal comparison.

## Reproduction

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/fastwam/bin/python \
  experiments/libero/calibrate_camera_reward_weights.py \
  --input-jsonl evaluate_results/imagination_reward_v2_offline/reward_v2_transitions.jsonl \
  --output-json evaluate_results/imagination_reward_v2_offline/camera_weight_calibration.json \
  --calibration-seed 42 \
  --validation-seed 1042 \
  --agent-weights 0.25 0.40 0.50 0.60 0.75 \
  --scale-quantile 0.90 \
  --goal-specificity-threshold 0.70 \
  --success-auc-threshold 0.90
```

The full result JSON remains in the ignored local `evaluate_results/` directory.
