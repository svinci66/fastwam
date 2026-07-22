# Global Per-Camera Reward Normalization — 2026-07-22

## Scope

This run validates the task-balanced global, per-camera normalization proposed for
the multi-task imagination reward. It reuses the same 1,573 schema-v3 transitions
from the ten-task pilot. No new LIBERO trajectory was collected and no online policy
evaluation was run.

Code under test: `6d0399e` on `feat/libero-residual-rl-mvp`.

Run directory:

```text
/home/ubuntu/sj/fastwam/runs/libero_goal_global_camera_norm_6d0399e_20260722_152000
```

## Transform

The replay builder gives every task equal total statistical weight, then fits one
median and IQR for each camera over valid aligned training transitions. The task ID
does not select a different transform. The normalized camera scores are combined
with the configured camera weights and smoothly bounded:

```text
z_agent = (raw_agent - median_agent) / IQR_agent
z_wrist = (raw_wrist - median_wrist) / IQR_wrist
r_imagination = 0.1 * tanh(0.5 * z_agent + 0.5 * z_wrist)
```

The frozen statistics are recorded in replay provenance and copied into both
learner checkpoints. They are not fitted or applied during online action inference.

## Fitted statistics

Statistics use 1,490 valid aligned transitions from all ten tasks:

| Camera | Median | Q25 | Q75 | IQR |
| --- | ---: | ---: | ---: | ---: |
| agent | 0.389910 | 0.241573 | 0.540158 | 0.298585 |
| wrist | 0.260781 | 0.166474 | 0.379608 | 0.213133 |

The different centers and scales confirm that the two viewpoints should not share
one raw normalization statistic.

## Reward distribution

| Statistic | Normalized imagination reward |
| --- | ---: |
| minimum | -0.091074 |
| Q05 | -0.080313 |
| Q25 | -0.041481 |
| median | -0.002523 |
| mean | -0.004084 |
| Q75 | 0.032389 |
| Q95 | 0.073520 |
| maximum | 0.095569 |
| positive fraction | 49.01% |
| negative fraction | 50.99% |
| fraction at either ±0.1 boundary | 0.00% |

The old `delta_alignment_v1` pilot placed most rewards at the positive clipping
boundary. This transform removes that saturation while preserving useful ordering:

| Episode label | Transitions | Reward mean | Reward median |
| --- | ---: | ---: | ---: |
| successful | 1,273 | +0.005740 | +0.006997 |
| failed | 300 | -0.045767 | -0.062114 |

This is an episode-level association, not causal evidence that the shaping reward
improves control.

## Matched AWR training

Both variants trained for 20 epochs from identical actor and critic initializations,
with the same replay, task-balanced sampler, seed 42, and timeout bootstrap value
0.0. The only reward ablation is `imagination_weight`:

| Variant | Imagination weight | Actor SHA-256 | Predicted residual RMS |
| --- | ---: | --- | ---: |
| control | 0.0 | `7dcce704276e7dca...` | 0.008284 |
| normalized imagination | 1.0 | `532ad7cee65edabc...` | 0.008883 |

Across the replay observations, the paired actor-output delta RMS is 0.003917 and
all transitions differ by more than 1e-6 in at least one action output. Therefore
the normalized imagination reward reaches the optimizer and materially changes the
learned actor.

## Verification and conclusion

- Full unit suite: 77 passed.
- Replay construction: passed, 1,573 transitions written and checksummed.
- Both pre-training validations: passed with identical initialization hashes.
- Both 20-epoch training runs: passed without NaNs or optimizer failures.
- Replay provenance in both checkpoints contains the same frozen per-camera stats.

The normalization implementation is accepted as the formal multi-task reward
variant: it solves the clipping problem without introducing task-specific scales.
This run does not establish an online success-rate improvement. The next experiment
is a matched deterministic LIBERO evaluation of the control and normalized-reward
actors on the same held-out initial states.
