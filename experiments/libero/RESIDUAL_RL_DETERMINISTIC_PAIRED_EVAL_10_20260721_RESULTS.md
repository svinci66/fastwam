# Strict-deterministic LIBERO residual-RL paired 10-trial evaluation (2026-07-21)

## Decision

The frozen FastWAM baseline, residual actor trained without imagination reward, and
residual actor trained with `delta_alignment_v1` were reevaluated on the same ten
explicit LIBERO initial states after fixing the SigLIP CUDA-SDPA nondeterminism.

The primary pass produced:

```text
frozen FastWAM:                    9/10
residual, no imagination reward:  8/10
residual, delta_alignment_v1:      9/10
```

An independent second process reproduced every result bit-for-bit for all three
policies. The two passes had identical success vectors, policy-step counts, all
per-replan observation and action hashes, and all final MuJoCo state hashes.

Compared directly with the no-imagination actor, the imagination-trained actor
recovered trials 8 and 9 but lost trial 6, for a net gain of one success. This is a
directionally positive signal for the imagination reward on these ten initial states,
but the exact paired test is non-significant (`p=1.0`) and the capped mean step count
is worse. The result does not yet establish an effectiveness improvement.

## Frozen protocol

```text
code:
  23196926b34db15d9c378483c5f5a63b633d5e0c

branch:
  feat/libero-residual-rl-mvp

output root:
  /home/ubuntu/sj/fastwam/runs/
  local_residual_deterministic_paired_eval_2319692_20260721_191233

suite/task:
  libero_goal / task 3

language:
  open the top drawer and put the bowl inside

explicit initial-state indices:
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

seed:
  42

FastWAM action horizon / executed replan horizon K:
  32 / 8

diffusion inference steps:
  4

action path:
  action-only; visualize_future_video=false

action ensembler:
  disabled

environment count:
  1
```

The strict-deterministic settings were:

```text
EVALUATION.deterministic_env=true
EVALUATION.deterministic_algorithms=true
EVALUATION.deterministic_warn_only=false
EVALUATION.record_action_hashes=true
CUBLAS_WORKSPACE_CONFIG=:4096:8
cuDNN benchmark=false
cuDNN deterministic=true
TF32=false
Flash SDPA=false
memory-efficient SDPA=false
math SDPA=true
```

The FastWAM checkpoint, dataset statistics, SigLIP encoder, and residual checkpoints
match the corrected smoke and determinism-audit reports. The environment uses the
model-preserving soft-reset mode documented in
`RESIDUAL_RL_DETERMINISM_AUDIT_20260721_RESULTS.md`.

## Primary paired result

Failures are represented as the 400-step task limit.

| Trial | Frozen FastWAM | No-imagination residual | Imagination residual |
| ---: | ---: | ---: | ---: |
| 0 | success / `173` | success / `173` | success / `169` |
| 1 | success / `182` | success / `177` | success / `177` |
| 2 | success / `180` | success / `170` | success / `170` |
| 3 | success / `169` | success / `196` | success / `189` |
| 4 | success / `186` | success / `177` | success / `299` |
| 5 | fail / `400` | success / `168` | success / `346` |
| 6 | success / `164` | success / `168` | fail / `400` |
| 7 | success / `263` | success / `184` | success / `179` |
| 8 | success / `393` | fail / `400` | success / `177` |
| 9 | success / `179` | fail / `400` | success / `231` |

Aggregate metrics:

| Metric | Frozen FastWAM | No-imagination residual | Imagination residual |
| --- | ---: | ---: | ---: |
| Success | `9/10` | `8/10` | `9/10` |
| Success rate | `90%` | `80%` | `90%` |
| Wilson 95% interval | `[59.58%, 98.21%]` | `[49.02%, 94.33%]` | `[59.58%, 98.21%]` |
| Mean steps, failures capped at 400 | `228.9` | `221.3` | `233.7` |
| Mean steps among successes only | `209.89` | `176.63` | `215.22` |
| Median steps among successes only | `180` | `175` | `179` |
| Mean residual RMS | n/a | `0.02559741` | `0.02223240` |

The imagination actor's mean residual RMS is `13.15%` lower than the matched
no-imagination actor. This repeats the earlier finding that the imagination reward
changes the controller toward smaller corrections. Smaller corrections still do not
imply better task efficiency: its capped mean is 12.4 steps higher (`233.7` versus
`221.3`).

Success-only means are censored by different failure sets. In particular, the
no-imagination actor's hard trials 8 and 9 are omitted from its success-only mean, so
the capped mean is the safer overall efficiency summary.

## Paired success analysis

The exact McNemar comparisons are:

| Comparison | First failure -> second success | First success -> second failure | Exact two-sided p |
| --- | ---: | ---: | ---: |
| Frozen -> no imagination | `1` | `2` | `1.000` |
| Frozen -> imagination | `1` | `1` | `1.000` |
| No imagination -> imagination | `2` | `1` | `1.000` |

For the reward ablation itself:

```text
trials favoring imagination success:     [8, 9]
trials favoring no-imagination success:  [6]
net success difference:                  +1/10 for imagination
```

Both residual actors succeeded on trials `[0, 1, 2, 3, 4, 5, 7]`. The paired
imagination-minus-no-imagination step differences on these common successes were:

```text
[-4, 0, 0, -7, +122, +178, -5]
```

Their mean is `+40.57` steps and median is `0`. The imagination actor was faster on
three non-tied common successes and slower on two, but its large slowdowns on trials
4 and 5 dominate the mean. There is no consistent efficiency improvement.

## Full repeatability proof

The second pass was launched as three fresh Python processes with exactly the same
configuration. It is a repeatability check, not ten additional independent initial
states.

| Policy | Pass 1 | Pass 2 | Steps equal | Entire action audit equal |
| --- | ---: | ---: | ---: | ---: |
| Frozen FastWAM | `9/10` | `9/10` | yes | yes |
| No-imagination residual | `8/10` | `8/10` | yes | yes |
| Imagination residual | `9/10` | `9/10` | yes | yes |

The equality comparison covered:

```text
frozen FastWAM:        291 replans
no-imagination actor:  280 replans
imagination actor:     298 replans
total per pass:        869 replans
```

At every replan, the audit includes agent and wrist images, proprioception, FastWAM
baseline action, corrected and executed action chunks, and—when applicable—SigLIP
observation features and residual actions. All three final-state sequences also
matched. Therefore this table is no longer confounded by the previously identified
fused-SDPA nondeterminism.

Process durations were close across repetitions:

| Policy | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| Frozen FastWAM | `132.44 s` | `133.67 s` |
| No-imagination residual | `154.12 s` | `153.05 s` |
| Imagination residual | `160.71 s` | `160.67 s` |

## Relationship to the previous 10-trial table

The earlier metric-complete table was:

```text
frozen FastWAM:                    8/10
residual, no imagination reward:  9/10
residual, delta_alignment_v1:      7/10
```

The new strict-deterministic table is:

```text
frozen FastWAM:                    9/10
residual, no imagination reward:  8/10
residual, delta_alignment_v1:      9/10
```

The old and new rows must not be combined. The strict path changes the SigLIP SDPA
backend and the environment-reset protocol, and the old residual evaluation was
shown not to be bitwise repeatable. The new table supersedes the old table for
deciding the next experiment, while the old files remain valid provenance records.

## Result files

Each path below is relative to the output root:

```text
pass1/baseline/libero_goal/gpu0_task3_results.json
pass1/no_imagination/libero_goal/gpu0_task3_results.json
pass1/with_imagination/libero_goal/gpu0_task3_results.json
pass2/baseline/libero_goal/gpu0_task3_results.json
pass2/no_imagination/libero_goal/gpu0_task3_results.json
pass2/with_imagination/libero_goal/gpu0_task3_results.json
logs/pass1_baseline.log
logs/pass1_no_imagination.log
logs/pass1_with_imagination.log
logs/pass2_baseline.log
logs/pass2_no_imagination.log
logs/pass2_with_imagination.log
```

All six rollout-video directories are preserved. All six processes exited with
status zero. Robosuite's known ignored `EGL_NOT_INITIALIZED` destructor warning was
emitted after result writing and did not affect the outputs.

## Conclusion and next gate

The deterministic rerun supports the following bounded conclusions:

```text
the strict single-GPU evaluation path is repeatable:                   supported
delta_alignment_v1 changes the actor and lowers residual RMS:          supported
delta_alignment_v1 has a positive success direction vs no imagination: observed (+1/10)
delta_alignment_v1 significantly improves success:                    not supported
delta_alignment_v1 improves overall or paired completion efficiency:   not supported
delta_alignment_v1 outperforms frozen FastWAM on this task:            not supported
```

The next experiment should not tune reward weights on these same ten states. The
minimal scientifically useful gate is to evaluate the two frozen residual actors on
additional held-out task-3 initial states if available, or on a predeclared set of
other LIBERO Goal tasks using the same strict protocol. Only after confirming the
direction on held-out states should reward weights or training be changed.
