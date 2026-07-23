# Residual IQL held-out paired evaluation — 2026-07-23

## Decision

The residual-IQL deployment and safety gate passes, but the imagination-reward
performance hypothesis does not pass this diagnostic:

```text
no-imagination IQL: 50 / 50 successes (100%)
imagination IQL:    49 / 50 successes ( 98%)
paired difference:  -1 / 50 episodes (-2 percentage points)
```

The single discordant episode is insufficient evidence of a true regression
(two-sided exact McNemar `p = 1.0`). More importantly, there is no observed success
improvement to support a positive claim. The lower offline training loss of the
imagination variant therefore did not translate into better held-out success in this
success-heavy replay.

This result does not reject imagination alignment as an auxiliary reward. It shows
that the current replay does not contain enough hard, failed, and recovery behavior
for the reward to demonstrate a useful policy-ranking effect.

## Evaluation boundary

Output:

```text
/home/ubuntu/sj/fastwam/runs/
libero_goal_iql_heldout_5to9_4a7e3d2_20260723
```

Configuration:

```text
suite:                  LIBERO Goal
tasks:                  0..9
held-out states:        5, 6, 7, 8, 9
episodes per variant:   50
evaluation variants:    no-imagination IQL, imagination IQL
FastWAM inference steps: 4
base seed:              42
independent per-episode policy/action seeds: enabled
deterministic PyTorch algorithms: enabled
```

The original FastWAM baseline was not rerun because this gate asks whether adding
the imagination term changes the IQL actor relative to its matched no-imagination
control. Both actors came from the same 1,573-transition replay and identical
initialization; only the reward ablation differs.

## Per-task results

| Task | No imagination | Imagination | Mean steps, no imagination | Mean steps, imagination |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5/5 | 5/5 | 123.0 | 125.0 |
| 1 | 5/5 | 5/5 | 85.8 | 85.8 |
| 2 | 5/5 | 5/5 | 85.0 | 85.2 |
| 3 | 5/5 | 4/5 | 200.2 | 215.6 |
| 4 | 5/5 | 5/5 | 82.6 | 81.4 |
| 5 | 5/5 | 5/5 | 132.2 | 130.8 |
| 6 | 5/5 | 5/5 | 106.6 | 103.8 |
| 7 | 5/5 | 5/5 | 69.6 | 70.0 |
| 8 | 5/5 | 5/5 | 71.4 | 71.2 |
| 9 | 5/5 | 5/5 | 146.0 | 131.2 |
| **Total** | **50/50** | **49/50** | **110.24** | **110.00** |

Wilson 95% intervals are `[92.87%, 100%]` and `[89.50%, 99.65%]`,
respectively. They overlap substantially.

The imagination actor's mean step count among its 49 successful episodes is
`104.08`, versus `110.24` for the 50 control successes. That comparison is
selection-biased because it excludes the imagination failure. Restricting the
comparison to the 49 pairs where both actors succeed gives an
imagination-minus-control mean of `-4.82` policy steps and median `0`; this is a
secondary diagnostic, not evidence of improved task performance.

## Pairing and determinism audit

All 50 episode pairs have:

- identical recorded trial indices and derived policy/action RNG seeds;
- identical serialized LIBERO initial-state hashes;
- the same action-noise stream;
- deterministic PyTorch settings recorded as enabled.

Exact first-observation and first-baseline-action hashes match in 43 of 50 pairs.
The other seven begin from the same serialized state but differ at the byte-hash
level after the 30-step MuJoCo settling period. Exact hashes are sensitive to very
small simulator/render differences, so these pairs remain useful for aggregate
evaluation but should not be described as byte-identical trajectories.

The only success-discordant pair is task 3, state 5:

```text
no-imagination: success in 176 policy steps
imagination:    failure at the 400-step limit
```

For this pair, the policy/action seeds, serialized initial state, first agent image,
first wrist image, first proprioception, observation feature, language feature, and
first FastWAM baseline action chunk all match exactly. The corrected and residual
action hashes differ, as expected. The failure is therefore attributable to the
different learned residual policies rather than an unmatched initial rollout.

## Action magnitude

Mean episode residual RMS:

```text
no-imagination: 0.007180
imagination:    0.006905
```

Both actors remain small corrections to FastWAM and complete 99 of 100 combined
episodes. This is enough to pass the deployment/safety gate: checkpoint loading,
language conditioning, bounded residual execution, and multi-task rollout all work.

## Consequence for the next stage

Do not tune the reward against states `5..9`; they are now validation data. Do not
claim an imagination-reward success improvement from this run.

Proceed to the pre-registered stage-2 collection on training states
`0..4,10..34`, using the FastWAM policy plus action-noise levels `0.075` and `0.15`.
Start with task 3 as a resumable collection pilot because it produced the only
discordant failure, then continue the same output root over all ten tasks if the raw
collection audit shows adequate failures and valid imagined/observed pairs.

The stage-2 runner supports `--collect-only` so a task subset can be collected and
audited before replay construction without contaminating held-out states or
prematurely training on a partial replay.
