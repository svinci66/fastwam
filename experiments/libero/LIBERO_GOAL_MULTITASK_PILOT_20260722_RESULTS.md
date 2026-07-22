# LIBERO Goal multi-task residual-AWR pilot (2026-07-22)

## Decision

The resumable ten-task pilot completed end to end:

```text
100 online episodes
  -> 1,573 schema-v3 transitions
  -> structural and conditioning audit
  -> matched no-imagination / imagination AWR training
  -> two finite v2 residual-policy checkpoints
```

This run validates the multi-task engineering and training path. It does **not** yet
show that adding the imagination reward improves the deployed policy. The collected
behavior is strongly success-heavy, controlled noise did not make the policy worse,
and the configured imagination reward is heavily clipped.

## Frozen protocol

```text
code commit:
  d5d8181b77cde049499128054c2bb30fb8627588

suite/tasks:
  libero_goal / task 0..9

initial-state indices:
  0,1,2,3,4 for every task and behavior mode

behavior modes:
  frozen FastWAM policy
  frozen FastWAM policy + Gaussian action noise (std=0.075)

episodes:
  10 tasks x 5 states x 2 modes = 100

FastWAM action horizon / executed replan horizon K:
  32 / 8

diffusion inference steps:
  4

reward cameras and weights:
  agent=0.5, wrist=0.5

AWR epochs / seed:
  20 / 42
```

Local output root:

```text
/home/ubuntu/sj/fastwam/runs/
libero_goal_multitask_pilot_1577965_20260721_210905
```

The run resumed after an interrupted first attempt. All final collection stages were
produced by the fixed evaluator at `d5d8181`; the incomplete pre-fix task-0 directory
was moved outside the final `raw/` input tree.

## Collection and conditioning audit

- All 20 task/mode cells contain trials `0..4`.
- Structural audit: passed, with no errors.
- Replay transitions: `1,573`.
- Replay schema: `3`.
- Task-language features: 10 distinct task vectors.
- Language encoder: frozen Wan UMT5-XXL masked mean, 4096 dimensions.
- Observation features: frozen SigLIP, two cameras, 2304 dimensions.
- Executed residual RMS over the replay: `0.04834697`.
- Raw data size: about 495 MB; complete run size: about 579 MB.

Per-task transition coverage:

| Task | Policy | Noise | Total successes / 10 |
| ---: | ---: | ---: | ---: |
| 0 | 76 | 75 | 10 |
| 1 | 53 | 54 | 10 |
| 2 | 94 | 54 | 9 |
| 3 | 141 | 125 | 9 |
| 4 | 55 | 54 | 10 |
| 5 | 117 | 111 | 8 |
| 6 | 62 | 75 | 10 |
| 7 | 47 | 47 | 10 |
| 8 | 46 | 47 | 10 |
| 9 | 121 | 119 | 8 |

Across modes, the frozen policy succeeded on `46/50` episodes and the noisy behavior
succeeded on `48/50`. Of the 50 matched initial-state pairs, both modes succeeded in
46, only noise succeeded in 2, and both failed in 2. There were no policy-only
successes. Consequently, this batch does not provide the intended broad set of
lower-quality noisy trajectories.

## Imagination-reward diagnostic

The replay uses `delta_alignment_v1`: the actual visual change after K actions is
compared with the direction and magnitude of FastWAM's imagined visual change. The
following analysis uses the raw, pre-clip score unless stated otherwise.

| Behavior | Raw mean | Agent mean | Wrist mean | Applied mean | At +0.1 clip |
| --- | ---: | ---: | ---: | ---: | ---: |
| Policy | 0.301903 | 0.355525 | 0.248282 | 0.088819 | 83.50% |
| Noise | 0.318045 | 0.370583 | 0.265508 | 0.092198 | 89.62% |

At the matched episode level, policy minus noise was:

```text
raw score mean:    -0.000644
raw score median:  +0.002628
policy higher:     28/50 pairs
```

The score therefore did not separate the policy from `std=0.075` noise. Transition
level correlation between the raw score and executed residual RMS was only `0.0503`.

There is a descriptive outcome relationship:

```text
successful episodes: 94, mean raw score 0.345136
failed episodes:      6, mean raw score 0.171770
```

This is useful preliminary evidence that imagined/actual agreement contains outcome
information. It is not clean causal evidence: there are only six failures, failures
are concentrated in a few tasks, and task difficulty can affect both the score and
success.

The main configuration issue is clipping. Most transitions receive the maximum
`+0.1` imagination contribution, including 92.07% of transitions from successful
episodes. In this batch, the shaping term behaves too much like a nearly constant
positive step reward and has limited ability to rank actions.

The agent view has a materially larger average raw score than the wrist view, and
their transition-level correlation is only `0.3611`. This confirms that the two
cameras carry different signals; equal weighting is a pilot setting, not a validated
final choice.

## Matched AWR training

Both actors and critics used identical seed-42 initialization hashes. Both runs use
task balancing, baseline-action conditioning, frozen language conditioning, success
bonus `10`, and imitation weight `0.1`. The only reward difference is imagination
weight `0` versus `1`.

| Metric | No imagination | With imagination |
| --- | ---: | ---: |
| Epochs | 20 | 20 |
| Initial critic loss | 6.364841 | 6.034978 |
| Final critic loss | 0.971081 | 0.876523 |
| Final actor loss | 0.001803 | 0.001819 |
| Final masked action MSE | 0.002287 | 0.002272 |
| Replay reward mean | 0.538406 | 0.628860 |
| Replay return mean | 4.589526 | 5.141407 |
| Checkpoint format | fastwam_residual_awr_v2 | fastwam_residual_awr_v2 |

All recorded values are finite. The loss curves show that both offline optimization
runs are numerically healthy. The higher reward/return in the imagination run follows
directly from adding a mostly positive reward term and is not an evaluation metric.
The small final loss differences likewise do not establish better online behavior.

Artifact hashes:

```text
replay manifest:
  5e96ebb5063cbf8dbd664928b91d0b662f9b522d3026468d1d0cb9bdd0302b71

no-imagination checkpoint:
  a5ffe4d85120f5880d5a9f3279b45daf9f31d510bb03a506f87aae3ff5999dca

imagination checkpoint:
  71679d8c657088518d0ca2ec49d14422e769797f91b54c1af9779022b60797f8
```

## Bounded conclusion and next gate

The formal multi-task path is ready for server-scale use: data collection, replay
provenance, ten-task language conditioning, task-balanced AWR, and checkpoint writing
all passed locally.

The reward hypothesis remains plausible only in the bounded form originally chosen:
imagined/actual agreement can be one reward component, alongside action imitation and
terminal success. This pilot does not support making it the whole reward, and the
current clipping/noise protocol is not discriminative enough to claim policy benefit.

Before expanding training data further, the two new residual checkpoints should be
evaluated online against frozen FastWAM on held-out initial states. That comparison is
the next gate for deciding whether to keep the current reward or first recalibrate its
scale, camera weights, and negative-data collection.
