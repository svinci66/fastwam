# RoboMimic deployable-observation Q results

## Purpose

The earlier Q-guided residual proof used the simulator's 71-dimensional state.
That state is useful for checking the learning method, but it is privileged and
cannot be supplied to a deployed robot. This stage replaces it with observations
that are available at inference time:

- a frozen SigLIP SO400M embedding of the `robot0_eye_in_hand` wrist camera;
- 16 proprioceptive values (joint position, end-effector position and
  quaternion, and gripper position);
- the unchanged base FastWAM action chunk.

The exact simulator states from the symmetric counterfactual collection were
replayed once per source demonstration. All 500 requested observations were
rendered at 384 x 384, their replayed state L-infinity error was zero, every
image was non-constant, and every proprioceptive vector was finite. Train and
validation remain split by source trajectory.

## Strict comparison

The preregistered Q gate was kept unchanged:

- full-observation balanced accuracy and AUC must each be at least 0.60 for
  every seed;
- the full model must beat the action-only control for every seed;
- mean balanced-accuracy gain over action-only must be at least 0.02.

With the raw 1,152-dimensional frozen SigLIP feature plus 16-dimensional
proprioception, the three validation seeds produced:

| seed | full balanced accuracy | action-only balanced accuracy | gain | full AUC |
|---:|---:|---:|---:|---:|
| 20260820 | 0.7309 | 0.6994 | +0.0314 | 0.7782 |
| 20260821 | 0.7128 | 0.6986 | +0.0142 | 0.7517 |
| 20260822 | 0.7102 | 0.6978 | +0.0124 | 0.7500 |

Mean balanced-accuracy gain was **+0.01937**. This misses the fixed +0.02 gate
by 0.00063, so the result is recorded as **not passed**. No threshold was
changed after observing the result, and a deployable residual actor was not
trained from this Q ensemble.

## PCA diagnostic

A train-only PCA projection to 64 vision dimensions was tested as a
predeclared-data-safe capacity diagnostic. It reduced mean full balanced
accuracy to 0.6419, compared with 0.6986 for the action-only control, for a
mean gain of -0.0567. Therefore the PCA representation discards information
needed for this comparison and is not the default pipeline.

## Decision

This stage verifies that wrist vision plus proprioception carries useful signal,
but the evidence is not yet strong enough to authorize Q-guided residual
training. The next experiment should increase trajectory-level observation and
counterfactual coverage while preserving the same held-out split and gate.
Only after that gate passes should the residual actor and online branch
evaluation resume.

Run the raw-feature comparison with:

```bash
bash scripts/run_robomimic_deployable_q_posttrain.sh
```

The failed PCA diagnostic can be reproduced explicitly with:

```bash
ROBOMIMIC_VISION_PCA_DIM=64 \
  bash scripts/run_robomimic_deployable_q_posttrain.sh
```
