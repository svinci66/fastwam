# RoboMimic Phase 1 residual-actor results

## Objective

Train a zero-initialized residual proposal from the successful pairwise-Q
dataset, then determine whether train-only KNN support and Q-advantage gates
make the proposal safe and selective enough for online evaluation.

## Target construction

The 4,693 non-tied branch pairs contain 3,887 unique `(demo, step)` states.
Repeated candidates are aggregated before training:

- if any candidate improved measured return, use the highest-return residual;
- otherwise use an exact zero residual target;
- clip each residual component to `[-0.1, 0.1]`;
- keep the official source-trajectory train/validation split.

This produces 3,521 training states and 366 held-out validation states. The
training split has 1,301 improvement and 2,220 zero targets; validation has 133
improvement and 233 zero targets. There is no source-demo overlap.

## Actor

The actor is a 126,485-parameter MLP with hidden dimensions
`256 -> 256 -> 128`. It consumes the 71-D state and base `3 x 7` action chunk,
then predicts a bounded `3 x 7` residual. The final layer is explicitly
zero-initialized and an exact-zero forward pass is asserted before training.
Class-balanced Huber regression prevents the more numerous zero targets from
dominating.

Three seeds were trained. Early stopping selected epochs 1, 1, and 2,
respectively; later epochs fit training candidates but worsened held-out
targets.

## Actor validation

| Metric | Three-seed mean |
|---|---:|
| Positive-target residual cosine | 0.123 |
| Positive direction alignment | 74.19% |
| Zero-target predicted norm | 0.0122 |

The actor stays close to the base action, but the positive residual direction
is only weakly predictable. This is consistent with the dataset having only
1.20 sampled candidates per unique state on average: pairwise Q can rank a
candidate that it is shown, while the actor must infer a useful direction from
very sparse local action exploration.

## OOD and Q gate

The support index uses `K=5` and the training-only 95th-percentile distance.
References contain training base chunks and retained better chunks. Results:

- 98.91% of held-out actor proposals remain in support;
- 100% of uniformly random action chunks are rejected;
- OOD detection therefore passes its component check.

Using `Q(candidate) - Q(base) > 0` is unsafe: it accepts roughly 95% of actor
proposals, including zero-target states. The corrected gate calibrates its Q
threshold at the 95th percentile of training zero-target advantages. Across
three seeds:

| Metric | Mean | Range |
|---|---:|---:|
| Q advantage AUC for improvement states | 0.585 | 0.529–0.614 |
| Intervention on improvement targets | 3.26% | 2.26–5.26% |
| Intervention on zero targets | 2.15% | 1.72–2.58% |
| Intervention separation | +1.11 pp | -0.32–+3.12 pp |

Threshold sweeps confirm there is no stable operating point: lowering the
threshold increases coverage but also intervenes on roughly as many zero-target
states. The Q model successfully ranked collected candidates in the previous
experiment, but it does not reliably distinguish these newly generated actor
proposals.

## Decision

`ready_for_online = false`.

Passing components:

- exact zero initialization and bounded output;
- train-only KNN OOD rejection.

Failing components:

- positive residual direction cosine;
- Q discrimination of actor proposals;
- intervention separation between improvement and zero-target states.

An online run is intentionally not started, because a negative result would be
expected from the offline gate and could unnecessarily damage the base policy.

The next data experiment should sample a controlled local action neighborhood:
approximately 8–16 symmetric candidate directions from each selected state,
instead of about one random candidate. This will estimate a local improvement
direction that a deterministic residual actor can learn. The existing 5,000
branches remain useful for pairwise-Q training and do not need to be discarded.

## Reproduce

```bash
bash scripts/run_robomimic_residual_actor_phase1.sh
```

Machine-readable summary:
`evaluate_results/robomimic_residual_actor_phase1/summary_multiseed.json`.
