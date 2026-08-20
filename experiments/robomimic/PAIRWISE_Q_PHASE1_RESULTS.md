# RoboMimic Phase 1 pairwise-Q results

## Question

Can a small shared `Q(state, action_chunk)` rank which of two exact-start
counterfactual branches obtains the higher measured return on source
trajectories that were never used for training? Does state provide information
beyond recognizing the action distribution alone?

## Data

Source collection:
`evaluate_results/robomimic_counterfactual/can_long_prefix_5000_seed20260819.hdf5`

Cleaning and reconstruction produced:

- 5,000 input branch pairs;
- 4,693 non-tied pairs retained;
- 307 ties removed;
- zero non-finite, state-restore, branch-start, or action-reconstruction errors;
- 4,239 training pairs from 180 official training demonstrations;
- 454 validation pairs from 20 official validation demonstrations;
- zero source-demonstration overlap across the splits;
- state shape `(4693, 71)`;
- action-chunk shape `(4693, 3, 7)`.

The three-action chunk is required because collection perturbed the first three
actions. It is deterministically reconstructed from the source trajectory and
saved per-sample random seed; every reconstructed first action was checked
against the action stored during online branch collection.

## Model and objective

The shared Q network is a 123,905-parameter MLP with hidden dimensions
`256 -> 256 -> 128`. The model evaluates the base and candidate chunks with
shared weights. Training minimizes class-balanced binary cross-entropy on

`Q(state, candidate_chunk) - Q(state, base_chunk)`.

The positive class means the candidate branch achieved the higher actual
branch score. Early stopping selects validation balanced accuracy. All state
and action normalization statistics are computed on training trajectories
only.

The action-only control uses the same training objective but excludes state.
This detects a trivial solution based only on perturbation magnitude or action
manifold proximity.

## Held-out results

All values below use the same 454 pairs from the 20 official validation
demonstrations.

| Seed | Full balanced accuracy | Full AUC | Action-only balanced accuracy | Action-only AUC | Balanced-accuracy gain |
|---:|---:|---:|---:|---:|---:|
| 20260820 | 76.42% | 82.46% | 70.79% | 75.92% | +5.63 pp |
| 20260821 | 76.08% | 82.70% | 70.82% | 76.51% | +5.26 pp |
| 20260822 | 76.38% | 82.86% | 70.62% | 76.58% | +5.76 pp |
| Mean | 76.29% | 82.67% | 70.74% | 76.34% | +5.55 pp |

The full-state accuracy mean is 77.68%, compared with a 67.18% majority-class
accuracy. The balanced-accuracy standard deviation across seeds is 0.15
percentage points. Full state beats action-only for every seed.

The pre-registered Phase-1 gate passes:

- every seed has full-state balanced accuracy and AUC above 60%;
- the mean balanced-accuracy gain over action-only exceeds 2 percentage points;
- full state beats action-only for every seed.

## Interpretation and limit

This is positive evidence that same-start counterfactual data supplies a
learnable, state-conditioned action ranking signal. It also rejects the claim
that all observed performance comes from the action prior, although the
70.74% action-only result shows that the action prior remains substantial.

This result does **not** yet show that a residual policy improves online task
success. Most labels come from shaped short-horizon progress; only 12 of the
5,000 raw branches changed terminal success. The next experiment should train
a zero-initialized residual proposal on the retained better actions, use this Q
only to compare candidate versus base chunks, and evaluate retention and
improvement online. OOD/support calibration must be fit using training
trajectories only before Q is allowed to intervene.

## Reproduce

```bash
bash scripts/run_robomimic_pairwise_q_phase1.sh
```

Machine-readable results are written to
`evaluate_results/robomimic_pairwise_q_phase1/comparison_multiseed.json`.

The subsequent residual-proposal experiment is documented in
`experiments/robomimic/RESIDUAL_ACTOR_PHASE1_RESULTS.md`. Pairwise Q passes, but
the first sparse-candidate residual actor does not pass the online-readiness
gate.
