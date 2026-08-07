# RoboTwin direct intervention-gate diagnostic (2026-08-07)

## Objective

This stage tests whether the unreliable inference-time Q gate can eventually
be replaced by a classifier trained directly on causal residual-versus-FastWAM
outcomes.  IQL still retains its Q critics during training.  The proposed
change affects only the deployment decision that authorizes a residual chunk.

## Implementation

- Outcome confirmation no longer requires inference-time Q critics.  It can
  re-anchor execution to FastWAM behind a paired gate and OOD support gate.
- The new data loader consumes only the exact replan where a residual was
  forced.  It does not broadcast an episode outcome over unrelated replans.
- `rescue` is the positive label; `regression` is a full-weight negative;
  outcome ties are conservative, lower-weight negatives.
- A terminal success with a short stored action prefix is restored only from
  its matched shadow record when the exporter hashes and the complete stored
  prefixes agree.  It is never zero padded.
- Diagnostic checkpoints carry `paired_advantage_deployment_ready=false`, and
  online loading rejects them.  Q gating remains available for controlled
  ablations rather than being physically deleted.

## Data audit

Thirty unique, accepted single-intervention pairs are available:

- 3 rescues;
- 6 regressions;
- 21 outcome ties;
- all 3 rescues come from `place_can_basket`, seed `4800002`;
- the 27 negative examples cover six task/seed groups.

Because positive support occupies only one task/seed group, an independent
seed split with both classes is impossible.  Random transition splitting would
leak nearly identical episode context and is intentionally not reported.

## Diagnostic result

A two-model action-conditioned gate was fitted for 30 epochs on CPU.  This is
a training-set capacity smoke test, not held-out evidence.  Its conservative
minimum probability separated the fitted pairs with 3/3 rescues accepted and
0/27 negatives accepted at the smoke-only threshold.

On the six outcome-changing `place_can_basket` causal probes:

| Gate | Rescues approved | Regressions approved |
|---|---:|---:|
| Current twin-Q | 0 / 3 | 1 / 3 |
| OOD only | 3 / 3 | 3 / 3 |
| Direct paired + OOD, training overlap | 3 / 3 | 0 / 3 |

The comparison confirms that the direct gate has enough representational
capacity to encode the desired decision and reconfirms that Q and OOD alone do
not solve the current selection problem.  It does **not** establish
generalization because the paired model was trained on these probes.

Artifacts:

- Diagnostic checkpoint and fit summary:
  `evaluate_results/robotwin_residual_rl/robotwin_corrected_posttrain_20260805/intervention_gate_diagnostic_20260807/`
- Gate comparison:
  `evaluate_results/robotwin_residual_rl/robotwin_corrected_posttrain_20260805/intervention_gate_diagnostic_20260807/gate_comparison.json`

## Decision and next requirement

No online episode was launched with the diagnostic gate.  Its data audit
correctly blocks deployment.  The next data step is to obtain actor-aligned
rescues from at least one additional task/seed group, while retaining
regressions and OOD negatives.  The gate can then be trained on one positive
group and evaluated on another before comparing `Q + OOD` against
`paired + OOD` online.
