# Residual IQL formal replay audit — 2026-07-23

## Decision

The formal all-task replay is accepted for matched residual-IQL training.

This gate validates the dataset, reward signal, and paired training
configuration. It does **not** claim that an IQL actor improves online LIBERO
success; that claim still requires training and held-out online evaluation.

## Collection

The resumable stage-2 pipeline collected:

```text
suite:                 LIBERO Goal
tasks:                 0..9
training states/task:  0..4 and 10..34 (30)
behaviors/state:       policy, noise 0.075, noise 0.15
episodes:              10 x 30 x 3 = 900
successful episodes:   804
failed episodes:       96
raw records:           15,836
aligned records:       15,133
```

States `5..9` remain reserved for validation. States `35..49` remain untouched
for the final comparison.

Behavior outcomes were:

| Behavior | Episodes | Successes | Replay transitions |
| --- | ---: | ---: | ---: |
| FastWAM policy | 300 | 289 | 4,482 |
| Gaussian noise 0.075 | 300 | 272 | 5,142 |
| Gaussian noise 0.15 | 300 | 243 | 6,212 |

The 804 successful episodes produce 804 terminated transitions. The 96 failed
episodes produce 96 truncated transitions. Every episode has exactly one of
these outcomes.

## Replay integrity

Replay location:

```text
/home/ubuntu/sj/fastwam/runs/
libero_goal_iql_stage2_formal_20260723/replay
```

The schema-v3 replay contains 15,836 transitions and occupies about 415 MB:

```text
arrays.npz:        413,937,098 bytes
transitions.jsonl:  21,298,910 bytes
```

All arrays are finite and match the manifest shapes. Each of the ten tasks has
exactly one frozen 4,096-dimensional UMT5 instruction feature, and the ten
instruction vectors are distinct. Observation, next-observation, and imagined
goal features are 2,304-dimensional fused two-camera SigLIP features.

Task transition counts vary from 914 to 2,925 because failed trajectories are
longer. Both IQL configs therefore retain task-balanced minibatches.

## Camera normalization

Normalization was fit only on the 15,133 aligned training transitions. Tasks
contribute equal total weight, while the two cameras use separate robust
statistics:

| Camera | Median center | IQR scale |
| --- | ---: | ---: |
| Agent view | 0.348270 | 0.321107 |
| Wrist view | 0.239494 | 0.216751 |

The final reward combines the separately normalized camera values at equal
weight and maps them through a bounded tanh with magnitude below 0.1.

The earlier boundary pile-up is removed:

| Threshold | Fraction of all transitions |
| --- | ---: |
| `abs(r_imagination) >= 0.090` | 0.4231% |
| `abs(r_imagination) >= 0.095` | 0.0316% |
| `abs(r_imagination) >= 0.099` | 0.0000% |

There are 14,440 distinct applied imagination rewards after rounding to six
decimal places. The 703 unaligned suffix transitions account for the 4.439%
exact-zero fraction.

## Reward direction

The transition-weighted imagination means decrease as behavior is degraded:

```text
FastWAM policy:  +0.00946
noise 0.075:     +0.00127
noise 0.15:      -0.00883
```

Episode-level comparisons use the same task and the same initial-state index.
There are 300 paired episodes per contrast:

| Contrast | Mean difference | 95% bootstrap CI | Tasks with positive mean |
| --- | ---: | ---: | ---: |
| policy − noise 0.15 | +0.01271 | [+0.01035, +0.01513] | 10/10 |
| noise 0.075 − noise 0.15 | +0.00873 | [+0.00629, +0.01119] | 10/10 |
| policy − noise 0.075 | +0.00398 | [+0.00172, +0.00626] | 8/10 |

The mean per-transition imagination reward within a successful episode is
`+0.01552`; within a failed episode it is `-0.03203`.

These results support the limited hypothesis needed for training: the
imagination term provides a non-saturated local preference signal correlated
with action degradation and episode outcome. They do not show that this term
is sufficient as a standalone objective.

## Composite reward

The accepted configs keep the imagination signal as only one component:

```text
goal completion:  +10 at success
action imitation:  0.1 x normalized residual penalty
imagination:       weight 0 or 1 for the paired ablation
```

LIBERO's sparse environment-return field is zero in this collected replay;
goal completion is represented explicitly by the success component. The
imitation component has mean `-0.22460`. The applied imagination component has
mean `-0.00037` and standard deviation `0.04498`, so it refines local ranking
without replacing the terminal goal signal.

## Matched IQL validation

Both configs load the formal replay successfully:

```text
no-imagination reward range: [-1.11441, 10.00000]
with-imagination range:      [-1.13591, 10.08471]
network context dimension:   2,312
language conditioning:       enabled
baseline-action conditioning: enabled
task balancing:              enabled
```

For seed 42, actor, both Q critics, and the value critic have identical
initialization hashes across the two variants. The only intended difference is
`imagination_weight: 0.0` versus `1.0`.

Seed-override validation additionally confirmed that both seed-43 variants
share the same initialization hashes, while those hashes differ from seed 42.

## Verification

```text
formal replay build:             passed
no-imagination validate-only:    passed
with-imagination validate-only:  passed
seed-43 matched-init check:      passed
all replay arrays finite:        passed
manifest shapes:                 passed
repository tests:                83 passed
Python compilation:              passed
shell syntax checks:             passed
git diff --check:                passed
```

## Next gate

1. Copy the 415 MB replay to the GPU server.
2. Run seeds 42, 43, and 44 for both matched reward variants.
3. Reject any run with non-finite metrics or a mismatched within-seed
   initialization hash.
4. Evaluate paired checkpoints first on validation states `5..9`.
5. Use states `35..49` only after the training and selection rule is frozen.

The multi-seed wrapper runs all six training jobs sequentially on one GPU:

```bash
PYTHON_BIN=/path/to/fastwam/bin/python \
bash scripts/run_libero_residual_iql_multiseed.sh \
  --replay-dir /path/to/replay \
  --output-root /path/to/iql_formal_3seed \
  --seeds 42,43,44 \
  --device cuda
```
