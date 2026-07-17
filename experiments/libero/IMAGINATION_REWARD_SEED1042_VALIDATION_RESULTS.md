# LIBERO imagination-reward independent validation (seed 1042)

## Scope

This experiment independently validates the task-3 reward candidate selected on seed 42. The
FastWAM checkpoint, SigLIP encoder, reward match weight, and task-specific zero-action reference
were frozen before collecting the validation data.

```text
suite: libero_goal
task_id: 3
task: open the top drawer and put the bowl inside
validation seed: 1042
paired initial-state indices: 5
action modes: policy, noise(std=0.15), zero
match weight: 0.05 (fixed from seed 42)
zero-action distance reference: 0.04124292854086781 (fixed from seed 42)
valid transitions: 589
```

The seed-42 and seed-1042 first-frame hashes differ, confirming that the validation observations
are not byte-identical repeats. No policy training or reward retuning was performed.

## Environment outcomes

| Action mode | Successes | Mean executed steps | Valid transitions | Future-video PSNR |
| --- | ---: | ---: | ---: | ---: |
| policy | 5/5 | 198.8 | 122 | 28.1195 dB |
| noise | 2/5 | 349.4 | 217 | 26.4592 dB |
| zero | 0/5 | 400.0 | 250 | 24.7160 dB |

The environment and video metrics reproduce the expected aggregate ordering.

## Frozen reward outcome

The frozen candidate is:

```text
r_combined = r_progress + 0.05 * (0.04124292854086781 - distance_after)
```

| Metric | Seed-42 tuning result | Seed-1042 validation result | Required |
| --- | ---: | ---: | ---: |
| Correct reward > wrong-goal reward | 72.84% | 61.29% | >= 70% |
| Paired `policy > noise > zero` | 5/5 | 3/5 | 5/5 |
| Successful mean > failed mean | yes | yes | yes |

At fixed weight 0.05, the validation-set episode means are:

| Group | Mean episode reward |
| --- | ---: |
| policy | 0.00872367 |
| noise | 0.00108915 |
| zero | -0.00013116 |
| successful episode | 0.00677759 |
| failed episode | 0.00012064 |

The aggregate ordering and success association remain strong, and the zero-action mean remains
small relative to the policy mean. However, the predeclared goal-specificity and all-pairs
criteria both fail.

## Per-trial ordering

| Trial | Policy reward | Noise reward | Zero reward | Full ordering |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 0.00818855 | 0.00139413 | -0.00065834 | yes |
| 1 | 0.01252098 | 0.00074450 | -0.00016621 | yes |
| 2 | 0.00550160 | -0.00056004 | 0.00041972 | no |
| 3 | 0.01254255 | -0.00051766 | 0.00009765 | no |
| 4 | 0.00486467 | 0.00438484 | -0.00034862 | yes |

Trial 2 is the most important counterexample: the noisy policy completed the task while the zero
policy did not, but the reward ranked zero above noise. This is evidence that the current visual
distance reward can mis-rank behavior on individual states even when its aggregate correlation
looks strong.

## Decision

The core observation remains supported at the aggregate level: better actions produce better
environment outcomes, higher video similarity, and a much higher mean imagination reward.
However, the particular `lambda_match = 0.05` formulation does not generalize reliably enough
to unseen seed-1042 states and is rejected for policy optimization.

Do not start RL with this reward and do not tune a replacement weight on seed 1042. The next
diagnostic should explain the failed trial-2 and trial-3 rankings, with emphasis on camera/object
features and phase-dependent reward behavior, before proposing another reward formula.

## Reproduction

```bash
PYTHONPATH=. python experiments/libero/scan_imagination_reward_weights.py \
  --input-jsonl evaluate_results/imagination_validation_seed1042/analysis/imagination_rewards.jsonl \
  --output-json evaluate_results/imagination_validation_seed1042/analysis/imagination_reward_fixed_lambda_validation.json \
  --weights 0 0.05 \
  --goal-specificity-threshold 0.70 \
  --selection-margin 0.02 \
  --fixed-zero-reference 0.04124292854086781
```

Raw videos, aligned PNG triplets, transition JSONL, and the full validation JSON remain in the
ignored local `evaluate_results/imagination_validation_seed1042/` directory.
