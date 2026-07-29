# RoboTwin Online Residual-IQL Paired Evaluation (2026-07-29)

## Scope

This is the first online comparison of the frozen FastWAM policy and the two
matched residual-IQL actors trained from 30 official successful demonstrations
plus the controlled-failure replay. The comparison uses the same five clean
environment seeds for every policy on each of three training tasks.

Policies:

1. Frozen FastWAM baseline.
2. Residual IQL without the imagination reward.
3. Residual IQL with the imagination reward.

Tasks are `adjust_bottle`, `open_laptop`, and `stack_blocks_two`. Each
policy/task pair runs five episodes, for 45 episodes total.

## Results

| Policy | adjust_bottle | open_laptop | stack_blocks_two | Overall |
| --- | ---: | ---: | ---: | ---: |
| Frozen FastWAM | 5/5 | 5/5 | 5/5 | 15/15 (100.0%) |
| IQL without imagination | 5/5 | 5/5 | 3/5 | 13/15 (86.7%) |
| IQL with imagination | 5/5 | 3/5 | 2/5 | 10/15 (66.7%) |

All online residual logs confirm that the correction was actually applied. The
maximum absolute residual remained below 0.05 and the two gripper residuals
remained exactly zero.

## Interpretation

The current residual actors do not improve online success over the frozen
FastWAM baseline. The no-imagination actor loses two `stack_blocks_two`
episodes, while the imagination actor loses two `open_laptop` and three
`stack_blocks_two` episodes. The imagination actor is therefore worse than
both the frozen baseline and the matched no-imagination actor in this small
paired evaluation.

This result rejects the current checkpoint as a policy improvement. It does
not by itself reject imagination-based shaping in general: the baseline is
already saturated on these five clean seeds, the training set is small, and
the residual actor is applied at every replan even when the baseline action is
already correct. A safer next experiment should test residual gating or a
smaller correction scale on seeds/tasks where frozen FastWAM has non-saturated
success, rather than simply expanding this checkpoint's evaluation.

## Recovery audit

The computer restarted during the final `imagination / stack_blocks_two`
batch. Eight completed policy/task batches (40 episodes) were retained. The
resumable launcher verified their final success-rate logs, skipped them, and
reran only the incomplete five-episode batch. The final summary contains nine
complete policy/task batches and no incomplete rows.

Machine-readable summary:

```text
evaluate_results/robotwin_residual_online/robotwin_residual_iql_online_pair_3task5ep_20260729/summary.json
```

Reproduction command:

```bash
bash scripts/run_robotwin_residual_iql_online_pair.sh
```
