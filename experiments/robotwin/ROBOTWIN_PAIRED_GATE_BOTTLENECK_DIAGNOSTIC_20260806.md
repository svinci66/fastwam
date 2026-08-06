# RoboTwin paired-gate bottleneck diagnostic (2026-08-06)

## Question

The expanded paper-aligned online evaluation regressed `hanging_mug` from 3/5
with frozen FastWAM to 1/5 with the corrected residual policy.  This diagnostic
asks whether the immediate limitation is model capacity, paired-data coverage,
or missing short temporal context.  It does not launch the simulator.

## Data and leakage controls

Only actual residual episodes with a clean FastWAM episode under the same task
and environment seed are eligible.  Expert and controlled-corruption episodes
are excluded because behavior source would otherwise be an easy proxy for the
episode outcome.  The resulting diagnostic set contains 339 transitions from
10 complete episodes: four rescue/improvement positives and six regression or
non-improvement negatives across `hanging_mug` and `blocks_ranking_size`.

All splits operate on complete episode IDs.  Four-fold cross-validation is
stratified by the episode outcome label, and three independent split/training
seeds (42, 43, and 44) are reported.  Individual transitions from one episode
can never appear in both training and validation.

The label is still an episode-level outcome copied to each transition.  It
therefore supports a generalization audit but cannot identify which one of
several interventions causally changed the outcome.

## Results

### 1. Current-model overfit check

The current 256x2 paired gate fits all ten actual residual episodes perfectly
under every seed:

- Episode balanced accuracy: 1.000 for seeds 42, 43, and 44.
- Episode AUROC: 1.000 for all three seeds.
- Brier error: approximately `5.7e-9` on average.
- Both ensemble members finish with sampled training accuracy 1.000.

The current architecture therefore has enough capacity to memorize the
available paired residual labels.  Failure to fit the training examples is not
the immediate bottleneck.

### 2. Fixed-data capacity comparison

The table reports mean episode-level out-of-fold metrics across the three
independent seeds.  Chance AUROC is 0.5.

| Hidden layers | Balanced accuracy | AUROC |
| --- | ---: | ---: |
| 128x2 | 0.389 | 0.236 |
| **256x2 (current)** | **0.361** | **0.195** |
| 512x4 | 0.403 | 0.264 |
| 1024x4 | 0.389 | 0.236 |

All sizes generalize poorly, and performance does not improve monotonically
with parameter count.  The apparent 512x4 gain is fewer than one correctly
classified episode on this ten-episode set and is not evidence for a larger
model.  Enlarging the current MLP is not supported as the next change.

### 3. Fixed-model data-fraction curve

| Fraction of available training episodes | Balanced accuracy | AUROC |
| --- | ---: | ---: |
| 25% | 0.361 | 0.375 |
| 50% | 0.375 | 0.361 |
| 75% | 0.431 | 0.278 |
| 100% | 0.361 | 0.195 |

The curve is non-monotonic and unstable.  This does **not** mean that additional
independent data would hurt.  Every point is a subset of the same ten episodes,
so adding another memorized seed can strengthen a spurious episode-level rule
instead of adding independent coverage.  The curve is itself evidence that the
current set is too small to estimate a learning trend.

### 4. Short temporal context

History length three uses the current visual/proprioceptive context plus two
within-episode lagged state deltas.  Missing history is zero and history never
crosses episode boundaries.

| Context | Balanced accuracy | AUROC | False-positive rate | Rescue recall |
| --- | ---: | ---: | ---: | ---: |
| Current state only | 0.361 | 0.195 | 0.444 | 0.167 |
| Current + two lagged deltas | 0.417 | 0.181 | 0.167 | 0.000 |

The short history makes the classifier more conservative, but it rejects every
held-out rescue.  It does not learn useful intervention timing.  Temporal
context may still be necessary, but the present data cannot demonstrate its
value.

## Conclusion

The experiment rules out a simple claim that the 256x2 gate is too small:
it memorizes the actual paired residual set perfectly, while models up to
1024x4 do not improve held-out ranking.  The immediate validated bottleneck is
the effective supervision set: only ten episode-level paired outcomes are
available, and they do not provide intervention-level causal labels.  The
held-out predictions are often inverted, which is consistent with learning
seed-, object-, instruction-, or trajectory-specific correlations.

This result does not prove that the current inputs are sufficient.  It says
that capacity and temporal architecture cannot be compared reliably until the
gate receives more independent, actor-aligned supervision.  The next data
collection should use isolated single interventions followed by a return to
FastWAM, paired against a no-intervention rollout from the same initial state.
Each candidate can then receive a direct short-horizon progress and terminal
outcome label.  Rescue, regression, and tie examples must be balanced by task
and split by complete environment seed.

The current residual actor, IQL checkpoint, and imagination reward remain
frozen during this diagnostic.  No online success claim is made.

## Artifacts

- Diagnostic implementation:
  `experiments/robotwin/diagnose_paired_gate_bottleneck.py`
- Seed-42 result:
  `evaluate_results/robotwin_residual_rl/robotwin_corrected_posttrain_20260805/bottleneck_diagnostic_20260806/summary.json`
- Seed-43 and seed-44 replications:
  `summary_seed43.json` and `summary_seed44.json` in the same directory.

