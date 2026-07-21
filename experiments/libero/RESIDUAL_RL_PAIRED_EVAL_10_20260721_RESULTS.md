# LIBERO residual-RL paired 10-trial evaluation (2026-07-21)

## Decision

The requested paired evaluation completed for frozen FastWAM, the residual actor
trained without imagination reward, and the residual actor trained with
`delta_alignment_v1`. All policies used the same ordered LIBERO task-3 initial states
and the same action-only evaluation protocol.

The metric-complete primary pass produced:

```text
frozen FastWAM:                    8/10
residual, no imagination reward:  9/10
residual, delta_alignment_v1:      7/10
```

This run does **not** support the claim that imagination reward improves task success.
The imagination-trained actor produced smaller residuals and was faster on most
initial states where both residual actors succeeded, but it also failed more initial
states. Exact paired tests are not significant at this sample size.

An initial pass before a metrics-recording fix produced `8/10`, `9/10`, and `9/10`.
The fix only records action steps and residual RMS outside the video-visualization
path; it does not enter model inference or action execution. Frozen FastWAM and the
no-imagination actor reproduced their success vectors exactly, while the
imagination-trained actor changed from `9/10` to `7/10`. This is evidence that the
current action diffusion/simulator path is not strictly repeatable on marginal
initial states despite the shared configured seed. Both passes are retained; the
less favorable result is not discarded.

## Frozen protocol

```text
code for metric-complete pass:
  83b125f (metric fix committed immediately after online verification)

base code and checkpoints:
  8e477f1229ba58afa497e3866ff7cc609d10e35e

output root:
  /home/ubuntu/sj/fastwam/runs/local_residual_rl_paired_eval_10_8e477f1_20260721_162643

suite/task:
  libero_goal / task 3

language:
  open the top drawer and put the bowl inside

ordered initial states:
  LIBERO task initial states 0 through 9

seed:
  42

action path:
  model.infer_action (action-only; visualize_future_video=false)

FastWAM action horizon / residual horizon:
  32 / 8

diffusion inference steps:
  4

action ensembler:
  disabled

prediction-video and transition export:
  disabled
```

The FastWAM checkpoint, dataset statistics, SigLIP encoder, residual checkpoints, and
encoder provenance are identical to the corrected one-trial smoke documented in
`RESIDUAL_RL_CORRECTED_SMOKE_20260721_RESULTS.md`.

Turning future-video visualization off changes this evaluation to the normal
action-only `infer_action` path. The results are internally paired across all three
policies, but action-step values must not be directly compared to the preceding smoke
that called `infer_joint` before the direct action inference.

## Metric recording defect found during evaluation

The first pass correctly saved success outcomes and rollout videos, but returned zero
`episode_policy_steps` and null `episode_residual_rms` whenever
`visualize_future_video=false`. Those diagnostics were incorrectly reconstructed
only from predicted-video clips.

Commit `83b125f` fixes this by:

1. counting every action actually sent after the environment warm-up independently
   of video generation;
2. collecting residual RMS at every residual-policy replan independently of predicted
   clips; and
3. returning both diagnostics directly from the episode runner.

The change passed `py_compile`, `git diff --check`, all 58 unit tests, and three online
10-trial reruns. The reruns wrote nonzero per-episode steps and finite residual RMS.

## Primary metric-complete result

Failures are assigned the 400-step task limit for the capped-step summary.

| Trial | Frozen FastWAM | No imagination residual | Imagination residual |
| ---: | ---: | ---: | ---: |
| 0 | fail / `400` | success / `173` | success / `168` |
| 1 | success / `182` | success / `180` | success / `176` |
| 2 | success / `180` | success / `170` | success / `171` |
| 3 | success / `167` | success / `199` | success / `161` |
| 4 | success / `186` | success / `370` | success / `176` |
| 5 | fail / `400` | success / `169` | fail / `400` |
| 6 | success / `164` | fail / `400` | success / `167` |
| 7 | success / `254` | success / `227` | success / `180` |
| 8 | success / `238` | success / `176` | fail / `400` |
| 9 | success / `175` | success / `316` | fail / `400` |

Aggregate results:

| Metric | Frozen FastWAM | No imagination residual | Imagination residual |
| --- | ---: | ---: | ---: |
| Success | `8/10` | `9/10` | `7/10` |
| Success rate | `80%` | `90%` | `70%` |
| Wilson 95% interval | `[49.02%, 94.33%]` | `[59.58%, 98.21%]` | `[39.68%, 89.22%]` |
| Mean steps, failures capped at 400 | `234.6` | `238.0` | `239.9` |
| Mean steps among successes only | `193.25` | `220.00` | `171.29` |
| Median steps among successes only | `181` | `180` | `171` |
| Mean episode residual RMS | n/a | `0.02563512` | `0.02209227` |

Success-only speed is censored by policy failures: the imagination actor's hard
states 5, 8, and 9 are absent from its success-only mean. The capped mean is therefore
the safer overall efficiency description, and it shows no advantage over the
no-imagination actor (`239.9` versus `238.0`).

The imagination actor's mean residual RMS is `13.82%` lower. This confirms that the
reward changes the learned controller toward smaller corrections, but smaller does
not imply better task completion.

## Paired success analysis

The exact McNemar comparisons are:

| Comparison | First policy failure -> second success | First success -> second failure | Exact two-sided p |
| --- | ---: | ---: | ---: |
| Frozen -> no imagination | `2` | `1` | `1.000` |
| Frozen -> imagination | `1` | `2` | `1.000` |
| No imagination -> imagination | `1` | `3` | `0.625` |

For the direct imagination-reward ablation, trial 6 favors imagination, while trials
5, 8, and 9 favor no imagination. Ten paired initial states are too few to establish a
success-rate difference, and the observed direction is not favorable to the current
imagination reward in the metric-complete pass.

Both residual actors succeeded on trials `[0, 1, 2, 3, 4, 7]`. On these six common
successes, imagination-minus-no-imagination step differences were:

```text
[-5, -4, +1, -38, -194, -47]
```

The imagination actor was faster on five of six common successes, with mean
`-47.83` steps and median `-21.5` steps. The exact two-sided sign-test p-value is
`0.21875`. This is an efficiency hint, not confirmatory evidence, and trial 4 accounts
for much of the mean difference.

## Repeatability audit

The initial pass and metric-complete rerun used the same task, configured seed,
ordered initial states, checkpoint paths, and action-only protocol.

| Pass | Frozen FastWAM | No imagination residual | Imagination residual |
| --- | ---: | ---: | ---: |
| Initial pass | `8/10` | `9/10` | `9/10` |
| Metric-complete rerun | `8/10` | `9/10` | `7/10` |

Success vectors:

```text
frozen, both passes:
  [0, 1, 1, 1, 1, 0, 1, 1, 1, 1]

no imagination, both passes:
  [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]

imagination, initial pass:
  [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]

imagination, metric-complete rerun:
  [1, 1, 1, 1, 1, 0, 1, 1, 0, 0]
```

The two passes reuse the same ten initial states, so their totals must not be reported
as 20 independent initial-state trials. Descriptively, however, the repeat totals are
`16/20`, `18/20`, and `16/20`. They reinforce that there is no current success-rate
advantage for the imagination-trained actor.

## Result files

Metric-complete JSON files:

```text
baseline_metrics_fixed/libero_goal/gpu0_task3_results.json
no_imagination_metrics_fixed/libero_goal/gpu0_task3_results.json
with_imagination_metrics_fixed/libero_goal/gpu0_task3_results.json
```

The corresponding first-pass JSON files, all six logs, and all rollout MP4 files are
preserved under the same output root. Robosuite emitted its known ignored
`EGL_NOT_INITIALIZED` destructor warning after each result was written. All six
evaluation processes exited with status zero.

## Conclusion and next gate

The evidence now supports a narrower conclusion than the one-trial smoke:

```text
residual adaptation can change success outcomes: supported
delta_alignment_v1 changes the actor and reduces residual magnitude: supported
delta_alignment_v1 improves success over the matched no-imagination reward: not supported
delta_alignment_v1 may improve speed conditional on success: weak, non-significant hint
strict repeatability of marginal outcomes under the current seed setup: not established
```

Before collecting a larger effectiveness table or tuning reward weights, the next
minimal engineering experiment should audit determinism. Log a hash of the first
FastWAM action chunk and residual chunk for every trial, enable deterministic PyTorch
settings where supported, and repeat a few marginal states (especially 5, 8, and 9).
If action hashes match but outcomes differ, investigate MuJoCo/environment
determinism; if hashes differ, isolate diffusion/CUDA RNG behavior. Reward changes
should not be selected on these same ten task-3 states.
