# RoboTwin dual-Q residual retention check (2026-07-29)

## Question

Can the Q networks already trained by residual IQL be reused at inference time
to avoid applying harmful residual actions, while retaining useful corrections?

## Gate

For every FastWAM replan, the imagination-trained actor proposes a corrected
action chunk. The two frozen target-Q critics compare it with the unmodified
FastWAM chunk under the same observation, proprioception, language instruction,
and baseline action conditioning:

```text
advantage_i = Q_i(s, residual_actor(s, a_fastwam)) - Q_i(s, a_fastwam)
```

The correction is applied only when both conditions hold:

- minimum dual-Q advantage is at least `0.0025`;
- absolute disagreement between the two Q advantages is at most `0.02`.

Otherwise the policy executes the exact FastWAM baseline chunk. Target critics
are used instead of the online critics.

These thresholds were fixed before online evaluation. They were selected by an
offline scan of the existing 1,159-transition training replay. At the selected
threshold, the imagination model's gate applied to 8.54% of all replay
transitions and 10.73% of expert transitions, while reducing expert action MSE
from `0.00323073` to `0.00320254` (0.87%).

## Online protocol

- Tasks: `adjust_bottle`, `open_laptop`, `stack_blocks_two`
- Episodes: five per task, 15 total
- Initial-state seed schedule: the same schedule as the earlier baseline and
  ungated residual-IQL comparison
- Residual actor: imagination-reward IQL checkpoint
- FastWAM checkpoint and all simulator settings unchanged

## Result

| Policy | adjust_bottle | open_laptop | stack_blocks_two | Overall |
|---|---:|---:|---:|---:|
| Frozen FastWAM | 5/5 | 5/5 | 5/5 | 15/15 (100.0%) |
| Ungated imagination IQL | 5/5 | 3/5 | 2/5 | 10/15 (66.7%) |
| **Dual-Q-gated imagination IQL** | **5/5** | **4/5** | **5/5** | **14/15 (93.3%)** |

The gate rejected every residual replan on `adjust_bottle` and
`stack_blocks_two`, preserving their 10/10 baseline successes. On `open_laptop`
it applied on 37/64 replans (57.81%) and produced 4/5 successes.

## Failure audit

The single failed `open_laptop` rollout was environment seed `4300000`. The gate
applied the residual on 28/30 replans in that rollout. In the four successful
rollouts it applied on only `2/8`, `2/14`, `1/5`, and `4/7` replans. Thus the
failure is associated with a sustained out-of-distribution region in which both
critics remained overconfident; dual-Q agreement alone is not a complete OOD
detector.

## Conclusion

This minimal check supports reusing the trained Q networks as a conservative
inference gate. It recovers four of the five successes lost by the ungated
imagination actor and retains 93.3% overall success, which is sufficiently close
to the 100% frozen baseline for moving to the intended improvement experiments
on low-success tasks.

It does not yet prove that the residual actor improves task success. The next
experiment should train and evaluate on `hanging_mug` (initial FastWAM 1/5) and
`blocks_ranking_size` (3/5), while keeping the three retention tasks in every
paired evaluation. The online results above must not be reused to tune the gate
threshold.

Raw local summaries:

- `evaluate_results/robotwin_residual_online/robotwin_residual_iql_online_pair_3task5ep_20260729/summary.json`
- `evaluate_results/robotwin_residual_online/robotwin_residual_q_gate_imagination_3task5ep_20260729/summary.json`

