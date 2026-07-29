# RoboTwin residual safety-gate audit (2026-07-29)

## Question

The Q-gated residual policy reduced high-success-task retention from the frozen
FastWAM baseline's 15/15 to 14/15.  The single failure was `open_laptop`,
environment seed `4300000`.  This audit asks whether the failure is caused by
out-of-distribution (OOD) inputs and whether a support gate can prevent it.

## Evaluation protocol

- Freeze FastWAM, the residual actor, and both IQL critics.
- Build the support index from successful replay episodes only.
- Split by whole episode: 33 reference episodes / 301 transitions and 12
  calibration episodes / 114 transitions.
- Use task-balanced calibration quantiles; do not tune thresholds on the online
  failure result.
- Compare state support (three-camera SigLIP feature, proprioception, and
  baseline action chunk), candidate residual-action support, language match,
  both IQL critics, and an episode-latched support circuit breaker.
- Use shadow execution and exact-prefix counterfactual runs from the same initial
  seed to separate candidate scoring from actual intervention effects.

## Calibrated support gate

The accepted local-density index is:

`evaluate_results/robotwin_residual_rl/robotwin_expert10_residual_iql_20260729/support_index_imagination_q95_local`

Calibration thresholds are 1.796079 for state support, 2.193999 for residual
action support, and 0.633792 for the post-intervention state-score increase.
Whole-replay joint acceptance is 87.36% for successful samples and 78.59% for
failed samples.  This is useful as a hard extrapolation check, but not accurate
enough to identify all harmful in-distribution actions.

## Failed-seed audit

All runs use `open_laptop`, seed `4300000`.

| Run | Actual residual interventions | Result |
| --- | ---: | ---: |
| Frozen FastWAM / support shadow | 0 | 1/1 |
| Q + support, unrestricted | 28/30 replans | 0/1 |
| Only replan 2 eligible | 1 | 1/1 |
| Only replan 3 eligible | 1 | 1/1 |
| Only replans 2 and 3 eligible | 2 | 1/1 |
| Q + support, maximum 2 interventions | 2 | 1/1 |

The unrestricted failed trajectory stayed inside the calibrated support region,
and the circuit breaker did not trigger.  Therefore this observed failure is
not classical distance-based OOD.  Each first intervention is harmless alone,
and the first two are harmless together; harm appears after continued residual
intervention changes the closed-loop trajectory.  Two critics and a kNN support
check cannot by themselves detect this accumulation error.

## Implemented safety policy

`residual_max_interventions_per_episode` limits only residual chunks that are
actually executed.  Q/support rejection, shadow candidates, and counterfactual
eligibility filtering do not consume the budget.  The policy resets the counter
at every episode and logs the count, remaining budget, and exhaustion state.

The value 2 is a provisional development setting supported by the failed-seed
counterfactual.  It must be held fixed and evaluated on additional initial
states; it is not yet a final globally optimal hyperparameter.

## Fixed-budget retention result

With the value fixed at 2, the same three high-success tasks and five episodes
per task produced 15/15 successes:

| Task | Success | Maximum interventions in an episode | Gate approval rate |
| --- | ---: | ---: | ---: |
| `adjust_bottle` | 5/5 | 1 | 8.00% |
| `open_laptop` | 5/5 | 2 | 39.02% |
| `stack_blocks_two` | 5/5 | 0 | 0.00% |

The paired frozen FastWAM result on these initial states was 15/15, while the
previous unrestricted Q/support-gated residual result was 14/15.  The budgeted
policy therefore passes this small retention check.  It does not yet establish
improvement: in particular, `stack_blocks_two` executed no residual at all.

Machine-readable summary:

`evaluate_results/robotwin_residual_online/robotwin_budget2_retention_3task5ep_20260729/summary.json`

## Decision rule before low-success-task training

The fixed safety policy passed the initial 15-episode retention check.  The next
stage may collect/train on lower-success tasks, but model selection must use
development seeds separate from the final paired evaluation seeds.  Measure
both retention and improvement, do not change the budget or support thresholds
using final seeds, and report success counts plus paired per-seed changes rather
than using critic value or reward ranking as a substitute for simulator success.
