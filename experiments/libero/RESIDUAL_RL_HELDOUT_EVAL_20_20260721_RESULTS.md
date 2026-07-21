# LIBERO residual-RL held-out 20-state evaluation (2026-07-21)

## Decision

The frozen FastWAM baseline, no-imagination residual actor, and
`delta_alignment_v1` residual actor were evaluated once on task-3 initial states
`10..29`. These states were not used to train the current residual checkpoints and
were not part of the preceding development comparison on states `0..9`.

The held-out result is:

```text
frozen FastWAM:                    20/20
residual, no imagination reward:  19/20
residual, delta_alignment_v1:      20/20
```

The imagination-trained actor recovered trial 10, the only state failed by the
no-imagination actor, and introduced no held-out failure. This preserves the positive
success direction observed on the development states. However, there is only one
discordant residual-policy pair (`p=1.0` by exact McNemar), and frozen FastWAM already
solved all 20 states. The imagination actor therefore avoided a residual-induced
regression; it did not improve success over the base policy.

Task efficiency also did not improve. Frozen FastWAM averaged `186.05` steps, the
no-imagination actor averaged `205.05` with its failure capped at 400, and the
imagination actor averaged `208.90`. The evidence supports continuing to investigate
the reward, but not claiming an overall policy improvement.

## Frozen protocol

```text
code:
  85c90598e4ce05e90fe1f9758408025b73a78897

branch:
  feat/libero-residual-rl-mvp

output root:
  /home/ubuntu/sj/fastwam/runs/
  local_residual_heldout_eval_20_85c9059_20260721_194109

suite/task:
  libero_goal / task 3

language:
  open the top drawer and put the bowl inside

held-out initial-state indices:
  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
   20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

untouched final-test indices:
  [30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
   40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

seed:
  42

FastWAM action horizon / executed replan horizon K:
  32 / 8

diffusion inference steps:
  4

action path:
  action-only; visualize_future_video=false

action ensembler / environment count:
  disabled / 1
```

Strict determinism remained enabled:

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

Per the explicit efficiency request, each policy was run once. No repeat pass was
counted or launched. The strict path itself had already passed a full two-process,
10-state, per-replan hash audit in
`RESIDUAL_RL_DETERMINISTIC_PAIRED_EVAL_10_20260721_RESULTS.md`.

The current residual replay contains 72 transitions from task-3 trial 0 only, under
policy and noisy behavior. Thus states `10..29` are held out from both current
residual training and the development evaluation on `0..9`.

## Per-state result

| Trial | Frozen FastWAM | No-imagination residual | Imagination residual |
| ---: | ---: | ---: | ---: |
| 10 | success / `168` | fail / `400` | success / `170` |
| 11 | success / `178` | success / `172` | success / `176` |
| 12 | success / `180` | success / `173` | success / `173` |
| 13 | success / `164` | success / `170` | success / `221` |
| 14 | success / `172` | success / `171` | success / `172` |
| 15 | success / `185` | success / `181` | success / `183` |
| 16 | success / `300` | success / `213` | success / `212` |
| 17 | success / `173` | success / `334` | success / `198` |
| 18 | success / `177` | success / `179` | success / `180` |
| 19 | success / `242` | success / `228` | success / `348` |
| 20 | success / `174` | success / `171` | success / `170` |
| 21 | success / `168` | success / `167` | success / `337` |
| 22 | success / `169` | success / `166` | success / `239` |
| 23 | success / `180` | success / `173` | success / `169` |
| 24 | success / `173` | success / `171` | success / `169` |
| 25 | success / `174` | success / `222` | success / `162` |
| 26 | success / `174` | success / `181` | success / `218` |
| 27 | success / `178` | success / `170` | success / `173` |
| 28 | success / `181` | success / `291` | success / `343` |
| 29 | success / `211` | success / `168` | success / `165` |

## Aggregate metrics

Failures are assigned the 400-step task limit in the capped mean.

| Metric | Frozen FastWAM | No-imagination residual | Imagination residual |
| --- | ---: | ---: | ---: |
| Success | `20/20` | `19/20` | `20/20` |
| Success rate | `100%` | `95%` | `100%` |
| Wilson 95% interval | `[83.89%, 100%]` | `[76.39%, 99.11%]` | `[83.89%, 100%]` |
| Mean steps, failures capped at 400 | `186.05` | `205.05` | `208.90` |
| Mean steps among successes only | `186.05` | `194.79` | `208.90` |
| Median steps among successes only | `175.5` | `173` | `178` |
| Mean residual RMS | n/a | `0.02555736` | `0.02217416` |

The imagination actor's residual RMS is `13.24%` lower. This closely matches the
`13.15%` reduction on development states `0..9`, confirming that
`delta_alignment_v1` consistently changes the learned controller toward smaller
corrections beyond the viewed development states.

Smaller residuals again do not imply faster completion. Despite having no failure,
the imagination actor's mean is 3.85 steps worse than the no-imagination capped mean
and 22.85 steps worse than frozen FastWAM.

## Paired analysis

Exact paired success comparisons:

| Comparison | First failure -> second success | First success -> second failure | Exact two-sided p |
| --- | ---: | ---: | ---: |
| Frozen -> no imagination | `0` | `1` | `1.000` |
| Frozen -> imagination | `0` | `0` | `1.000` |
| No imagination -> imagination | `1` | `0` | `1.000` |

The only residual-policy success disagreement is trial 10, favoring imagination.
One discordant pair is insufficient for statistical evidence even though its
direction matches the development comparison.

Both residual actors succeeded on trials `11..29`. On these 19 common successes,
imagination-minus-no-imagination step differences were:

```text
[+4, 0, +51, +1, +2, -1, -136, +1, +120, -1,
 +170, +73, -4, -2, -60, +37, +3, +52, -3]
```

Here negative means the imagination actor was faster. Summary:

```text
imagination faster:  7
no imagination faster: 11
tie: 1
mean difference:   +16.16 steps
median difference:  +1 step
exact sign-test p:   0.481
```

The imagination reward has no consistent paired efficiency advantage. Large
slowdowns on trials 19, 21, and 22 are partly offset by a large speedup on trial 17,
which illustrates why the mean and median should both be reported.

Frozen FastWAM and the imagination actor succeeded on all 20 states. Their paired
imagination-minus-baseline step difference has mean `+22.85` and median `-1`. The
median is essentially tied, while a few large imagination slowdowns raise the mean.
The two-sided sign test is `p=1.0`.

## Relationship to the development result

Development states `0..9` produced:

```text
frozen FastWAM:                    9/10
residual, no imagination reward:  8/10
residual, delta_alignment_v1:      9/10
```

Held-out states `10..29` produced:

```text
frozen FastWAM:                    20/20
residual, no imagination reward:  19/20
residual, delta_alignment_v1:      20/20
```

The direction is consistent in both sets: imagination has one more success than the
matched no-imagination actor. Across the two sets descriptively, residual-policy
discordances favor imagination on trials 8, 9, and 10 and favor no imagination on
trial 6. Because `0..9` served as a repeatedly viewed development set, the held-out
`10..29` result remains the primary validation evidence; the sets should not be
presented as one untouched 30-state test.

The held-out set also exposes a ceiling effect: frozen FastWAM is already `20/20`.
This task slice cannot demonstrate a success-rate improvement over the base policy.
Its useful finding is narrower: imagination reward prevented the one held-out
regression introduced by the no-imagination residual actor.

## Result files

Relative to the output root:

```text
baseline/libero_goal/gpu0_task3_results.json
no_imagination/libero_goal/gpu0_task3_results.json
with_imagination/libero_goal/gpu0_task3_results.json
logs/baseline.log
logs/no_imagination.log
logs/with_imagination.log
```

All rollout videos and action-audit hashes are retained under the same root. The
three processes exited with status zero. Robosuite's known ignored
`EGL_NOT_INITIALIZED` destructor warning occurred after writing results and did not
affect process status.

## Conclusion and next gate

The held-out evidence supports these bounded conclusions:

```text
delta_alignment_v1 lowers residual RMS on unseen task-3 states:         supported
imagination avoids the no-imagination failure on this held-out set:      observed
the positive success direction repeats beyond the development states:   observed
imagination significantly improves residual-policy success:             not supported
imagination improves completion efficiency:                              not supported
either residual actor improves frozen FastWAM success:                   not supported
```

States `30..49` remain untouched and should stay reserved until a model or reward
configuration is selected. Running the unchanged policies there immediately would
consume the final test set without resolving the current ceiling effect. The next
efficient engineering step should be to understand why the current residual actor
slows some successful trajectories, or predeclare a harder task slice where frozen
FastWAM has failures before using the final task-3 states.
