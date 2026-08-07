# RoboTwin high-failure single-intervention analysis (2026-08-07)

## Scope

This report audits and analyzes the paper-aligned, fixed-seed single-residual-
intervention batch:

- `place_can_basket`: seeds `4800002` and `4800004`, five intervention probes
  per seed.
- `open_microwave`: seed `4800000`, five intervention probes.
- FastWAM inference: 10 denoising steps, 24-step replanning, official fixed
  instruction and deterministic seed manifest.
- Residual actor: corrected 20-epoch IQL checkpoint.
- Candidate safety metadata: target twin-Q margin `0.003` and Q95 OOD support
  index.

The 15 probes share only three baseline episodes. They are causal
intervention-time probes, not 15 independent evaluation episodes, so their
counts must not be reported as task success rates.

## Audit corrections

### Terminal-prefix pair

`place_can_basket / seed 4800002 / replan 21` was initially quarantined because
the shadow branch stored 24 actions while the residual branch terminated
successfully after five actions. The full planned FastWAM action hashes are
identical, the intervention observations and proprioception are identical, and
the five stored common-prefix actions have zero maximum absolute difference.
The pair is therefore a valid terminal-prefix counterfactual and is labeled as
a rescue. The pair builder now accepts this case only when:

1. the full baseline-action hashes agree;
2. at least one branch is terminal;
3. the stored action dimensions are compatible; and
4. the complete common prefix matches within tolerance.

### Frozen camera normalization

The first automatic score fitted normalization on the evaluation probes. The
formal score instead freezes the global per-camera statistics from the exact
training replay manifest. This removes evaluation-set leakage and preserves
cross-task comparability. Terminal transitions whose effective horizon differs
from the imagined horizon remain valid for final success labels but are
excluded from local imagination-progress statistics.

Formal outputs:

- `evaluate_results/robotwin_residual_pairs/robotwin_high_failure_rescue_batch_20260806_place_can_basket/statistics_frozen_global/`
- `evaluate_results/robotwin_residual_pairs/robotwin_high_failure_rescue_batch_20260806_open_microwave/statistics_frozen_global/`

## Outcome results

### `place_can_basket`

| Baseline seed | Baseline outcome | Replan | Gate stratum | Forced residual outcome | Causal label |
|---|---:|---:|---|---:|---|
| 4800002 | failure | 0 | Q rejected | failure | local worse |
| 4800002 | failure | 7 | Q rejected | success | rescue |
| 4800002 | failure | 14 | Q rejected | success | rescue |
| 4800002 | failure | 21 | Q rejected | success | rescue, terminal prefix |
| 4800002 | failure | 28 | Q rejected | failure | local improve only |
| 4800004 | success | 0 | Q rejected | failure | regression |
| 4800004 | success | 4 | Q approved | failure | regression |
| 4800004 | success | 7 | Q rejected | failure | regression |
| 4800004 | success | 11 | Q rejected | success | neutral |
| 4800004 | success | 18 | Q rejected | success | local improve |

The actor can produce useful corrections: three of five probed intervention
times rescue the failed seed. It can also destroy an otherwise successful
trajectory at three of five probed times. Residual usefulness is therefore
strongly state- and timing-dependent; an always-on actor is not acceptable.

Residual magnitude alone cannot separate the cases. Rescue RMS values span
approximately `0.0040` to `0.0260`, while regression values span approximately
`0.0053` to `0.0193`.

### `open_microwave`

All five candidate interventions were action-OOD and were rejected by the
support gate. Forced execution produced no task success:

- one local improvement;
- one local neutral result;
- three local degradations;
- mean frozen-normalized local delta: `-0.01124`.

The candidate action-support scores (`3.89` to `33.43`) all exceed the frozen
threshold (`2.412`). This is useful evidence that the OOD gate is protecting
the base policy in this seed rather than merely being over-conservative.

## Q-gate analysis

For `place_can_basket`, the only Q-approved candidate caused a regression. All
three rescues were Q-rejected. Comparing the three rescue Q values with the
three regression Q values gives a small-sample ranking AUC of `0.333`; the
Pearson correlation between minimum twin-Q advantage and the causal outcome is
`-0.181`.

These numbers are descriptive, not statistically conclusive, but they reject
the operational assumption that the current Q margin reliably selects helpful
interventions. Merely lowering the margin is insufficient because harmful and
helpful Q values overlap and are not monotonically ordered.

The corrected training replay explains this weakness. It contains policy,
expert and controlled-corruption transitions for `place_can_basket` and
`open_microwave`, but no residual-behavior transitions for either task. The Q
critics therefore have little direct coverage of the actor's candidate action
distribution on these tasks. `open_microwave` confirms this directly through
the action-OOD scores.

## Imagination-reward analysis

Using the frozen global per-camera normalization:

- `place_can_basket`: nine horizon-aligned pairs have mean local delta
  `+0.00140`.
- `open_microwave`: five aligned pairs have mean local delta `-0.01124`.
- Among the five horizon-aligned outcome-changing `place_can_basket` pairs,
  the sign of local progress agrees with rescue versus regression only two
  times (`2/5`).
- The correlation between local progress and causal outcome on those valid
  `place_can_basket` rows is approximately `0.002`.

There are direct semantic counterexamples: one rescue has negative local
progress, while the Q-approved regression has positive local progress. The
three camera deltas also frequently disagree. Consequently, imagined-versus-
actual visual alignment is useful as a dense auxiliary signal, but it cannot
serve as the sole reward or as a deployment gate.

## Bottleneck diagnosis

The immediate bottleneck is data coverage and critic/gate calibration, not a
demonstrated lack of actor capacity:

1. The current pooled-SigLIP MLP actor already generates three exact paired
   rescues, proving that useful actions exist within its present hypothesis
   class.
2. The current Q gate rejects every rescue and approves one regression.
3. The replay contains no direct residual behavior for the two analyzed tasks.
4. The OOD gate correctly identifies unsupported `open_microwave` candidate
   actions and forced probes do not contradict it.

The actor architecture can still become a later bottleneck. Dual-wrist patch
features and temporal recurrence remain well-motivated for fine manipulation
and repeated interventions, but changing them before retraining on the new
counterfactual data would confound the current diagnosis.

## Recommended next step

1. Add the accepted single-intervention pairs to a task-balanced training set,
   preserving `rescue`, `regression`, neutral and OOD-negative labels.
2. Retrain/calibrate the twin Q critics and paired candidate-vs-baseline gate
   on these causal labels; do not solve the result with a margin-only change.
3. Add supported positive residual examples for `open_microwave`; its current
   probes supply useful OOD negatives but no positive actor target.
4. Use zero initialization for a newly trained actor output layer and retain
   the current per-dimension maximum scale of `0.05`; do not increase it to
   `0.1` while regressions remain common.
5. Re-run the same fixed-seed probes as a diagnostic, then evaluate on new
   held-out seeds. Only after this data-controlled retraining should a dual-
   wrist patch encoder and LSTM/GRU be introduced as an architecture ablation.

The next formal online evaluation must report independent episodes and preserve
high-success baseline capability; the probe counts in this report are not a
replacement for that evaluation.
