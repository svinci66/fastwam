# Terminal safety outcome-weighting result

Date: 2026-08-24

## Question

Can a fixed 20x training weight on rare terminal outcome changes make the
history-aware full-state critic reject harmful residual interventions?

## Protocol

- Dataset: 542 terminal-tail pairs from 20 trajectory-disjoint RoboMimic Can
  rollouts (440 train and 102 validation).
- Rare events: 13 terminal outcome changes: five gains and eight losses. Ten
  events are in train and three are in validation.
- Compare matched full-state and action-only pairwise critics over seeds
  `20260820`, `20260821`, and `20260822`.
- Initialize each full-state critic from its action-only model and retain the
  fixed teacher regularization of 1.0.
- Weight the ten training rows with `terminal_outcome_changed=1` by 20x. The
  validation distribution and decision threshold are unchanged.

## Result

| Metric (validation mean) | Full-state | Action-only | State gain |
| --- | ---: | ---: | ---: |
| Balanced accuracy | 0.6028 | 0.6028 | 0.0000 |
| AUC | 0.6265 | 0.6265 | 0.0000 |

All three selected full-state checkpoints remained at epoch zero and exactly
preserved their corresponding action-only critic. On the three validation
events from `demo_4`:

- Step 33 (success loss): rejected by only one of three seeds.
- Step 42 (success gain): accepted by all three seeds.
- Step 48 (success loss): accepted incorrectly by all three seeds.

The full-state ensemble therefore gets only one of the three event types
correct across all seeds. Outcome weighting does not pass the safety gate and
must not be used for online residual evaluation.

## Decision

Do not tune more loss weights against the same three validation events. The
next dataset uses new initial states and all three frozen residual actors, and
audits only trajectories on which the baseline succeeds. This targets terminal
success-loss hard negatives while preserving a trajectory-disjoint validation
split. The benefit Q and terminal safety decision should be evaluated as
separate roles once that dataset is complete.
