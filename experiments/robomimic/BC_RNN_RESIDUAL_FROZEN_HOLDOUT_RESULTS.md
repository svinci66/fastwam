# BC-RNN residual frozen holdout results

Date: 2026-08-21

## Protocol

- Freeze the PCA-16 projection, three action-prior initialized Q critics, and
  three Q-guided residual actors before examining this holdout.
- Collect 20 new closed-loop BC-RNN trajectories with seed `20260825`; 14/20
  succeed. Select 100 states from four trajectory-disjoint validation
  trajectories for the branch evaluation.
- Render live wrist observations, encode with the frozen SigLIP, and apply the
  PCA projection fitted only on the old 300-state training split.
- Calibrate Q and OOD thresholds using only the old training references.
- Re-run the baseline inside each branch comparison to eliminate replay drift.

## Frozen Q ranking

The new 100-state holdout contains 428 decisive action pairs.

| Seed | Full-state balanced accuracy | Action-only balanced accuracy | Gain |
| --- | ---: | ---: | ---: |
| 20260820 | 0.5896 | 0.5810 | +0.0086 |
| 20260821 | 0.5647 | 0.5586 | +0.0060 |
| 20260822 | 0.5754 | 0.5698 | +0.0055 |
| Mean | 0.5765 | 0.5698 | +0.0067 |

The full-state critic generalizes above chance and consistently beats the
action-only control, but the margin is substantially smaller than on the old
validation split. It does not satisfy the original +0.02 state-gain gate.

## Gated short-horizon branches

| Actor seed | Accepted | Improved | Tied | Worsened | Mean score delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260820 | 36/100 | 19 | 80 | 1 | +0.0000561 |
| 20260821 | 31/100 | 16 | 83 | 1 | +0.0000490 |
| 20260822 | 33/100 | 19 | 79 | 2 | +0.0000550 |

All actors preserve the gripper dimension, stay below the 0.03 residual bound,
and have exact environment restoration. This supports a limited claim: the
frozen residual plus Q/OOD gate produces more local improvements than local
regressions on trajectory-disjoint states.

## Complete online episodes

The preselected actor seed `20260820` was then evaluated on all four new valid
episodes with live wrist rendering and SigLIP/PCA inference.

- Baseline successes: 2/4.
- Residual successes: 1/4.
- Success gains: 0; success losses: 1.
- Mean shaped-reward delta: +0.1268; three episodes improve in shaped reward.
- Interventions: 110/343 decisions (32.1%).
- Residual bound, gripper preservation, and deterministic reset checks pass.

The online safety gate **fails** because `demo_4` changes from success to
failure after 28 accepted interventions. Positive dense reward and positive
short-horizon branch scores therefore do not guarantee task completion under
repeated intervention. This version must not be presented as a success-rate
improvement or promoted to broader online evaluation.

## Conclusion

The current result isolates the next bottleneck: the residual learns useful
local corrections, but the critic does not estimate the long-horizon cost of
repeated corrections reliably enough. The next change should target temporal
credit and accumulated intervention risk, while retaining the existing
state-aware OOD rejection and zero-initialized bounded residual actor.

Local artifacts are under
`evaluate_results/robomimic_bc_rnn_q_holdout/seed20260825/deployable/`.
