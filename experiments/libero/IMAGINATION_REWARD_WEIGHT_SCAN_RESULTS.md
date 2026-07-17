# LIBERO imagination-reward weight scan

## Scope

This is an offline reward-design experiment. It reuses the 567 aligned transitions from the
5-episode-per-action-mode validation on `libero_goal/task_3`. It does not rerun LIBERO, update
FastWAM, or train a policy.

The candidate reward is:

```text
r_combined = r_progress + lambda_match * r_match
r_progress = distance_before - distance_after
r_match = median_zero_action_distance_after_per_task - distance_after
```

The zero-action median centers the absolute-match component around a no-op reference. For this
task the fixed reference was `0.04124293`. This avoids introducing a large constant positive
reward for merely remaining in the episode.

## Selection criteria

A candidate must satisfy all three conditions:

1. Correct-goal reward beats the paired wrong-goal reward in at least 70% of transitions.
2. All five paired trials retain `policy > noise > zero` by mean episode reward.
3. Successful episodes retain a higher mean reward than failed episodes.

The reported recommendation also requires a two-percentage-point buffer above the 70% target,
so a weight that only crosses the threshold by one transition is not selected.

## Results

| Match weight | Correct > wrong | Paired `policy > noise > zero` | Success > failure | Decision |
| ---: | ---: | ---: | :---: | :--- |
| 0.000 | 56.44% | 5/5 | yes | baseline; goal specificity fails |
| 0.030 | 69.14% | 5/5 | yes | below threshold |
| 0.033 | 70.02% | 5/5 | yes | mathematical minimum; threshold edge |
| 0.040 | 70.72% | 5/5 | yes | passes |
| 0.050 | 72.84% | 5/5 | yes | provisional recommendation |
| 0.060 | 73.54% | 5/5 | yes | passes |
| 0.070 | 74.43% | 5/5 | yes | passes |
| 0.080 | 75.31% | 4/5 | yes | action ordering starts to degrade |
| 0.250 | 82.36% | 2/5 | yes | rejected |

At the provisional `lambda_match = 0.05`:

| Group | Mean episode reward |
| --- | ---: |
| policy | 0.00865797 |
| noise | 0.00211873 |
| zero | -0.00003668 |
| successful episode | 0.00707660 |
| failed episode | 0.00052049 |

The combined reward improves correct-versus-wrong goal discrimination by 16.40 percentage
points (`56.44% -> 72.84%`) while preserving all five paired action-quality orderings. The
zero-action mean remains close to zero.

## Decision and limitation

`lambda_match = 0.05` is the provisional candidate for the next validation, not a final RL
hyperparameter. Larger values improve goal specificity but eventually make the absolute
matching term dominate and break the desired action ordering. The observed useful region on
this dataset is approximately `0.04-0.07`.

The same task-3 data was used to choose and score the weight, so this is a tuning result rather
than independent confirmation. Before policy updates, freeze `lambda_match = 0.05` and validate
it on unseen transitions from another LIBERO task or a new task-3 seed. Do not retune the weight
on that validation set.

## Independent-validation update

The frozen `lambda_match = 0.05` candidate was subsequently evaluated on seed 1042 without
retuning. It did not pass: correct-versus-wrong discrimination reached only 61.29%, and the full
paired action ordering held in 3/5 trials. The provisional candidate is therefore rejected for
RL use. See `IMAGINATION_REWARD_SEED1042_VALIDATION_RESULTS.md` for the complete result.

## Reproduction

```bash
PYTHONPATH=. python experiments/libero/scan_imagination_reward_weights.py \
  --input-jsonl evaluate_results/imagination_validation_5eps/analysis/imagination_rewards.jsonl \
  --output-json evaluate_results/imagination_validation_5eps/analysis/imagination_reward_weight_scan_final.json \
  --weights 0 0.01 0.02 0.03 0.031 0.032 0.033 0.034 0.035 0.036 0.037 0.038 0.039 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.25 0.5 1 2 \
  --goal-specificity-threshold 0.70 \
  --selection-margin 0.02
```

The full JSON scan remains in the ignored local `evaluate_results/` directory. This compact
record and the reusable scan code are committed to Git.
