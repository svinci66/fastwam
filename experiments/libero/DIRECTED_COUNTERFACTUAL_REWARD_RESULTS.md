# LIBERO directed exact-state imagination-reward result

## Decision

The corrected, cross-source-aligned experiment passes every preregistered primary
gate. The frozen equal-camera `raw_dual` reward contains statistically reliable local
direction information when action quality is defined by real MuJoCo progress rather
than Gaussian noise magnitude.

This does **not** approve `raw_dual` as a complete RL reward. The wrong-direction
branch still receives positive reward because it retains the policy rotation and
produces non-static visual change. The result supports imagination consistency as a
bounded auxiliary term alongside FastWAM imitation and environment success.

The protocol and initial implementation were committed before reward inspection:

```text
branch: feat/libero-directed-counterfactual-reward
preregistration commit: 3c4d41018e7ec0c75cd6dfc026f12e98ed9939ed
cross-source integrity fix: d065c3e
```

## Invalid first attempt and correction

The first attempted formal collection reproduced flattened MuJoCo states but omitted
the source collector's one LIBERO soft reset per anchor. That changed model-level
camera state, which is absent from the flattened state, for anchors 1--9. Because the
experiment reused source imagined goals, the first reward result was invalidated.

Before corrected collection, the collector was changed to replay exactly one soft
reset per anchor and the analyzer was strengthened with two cross-source gates:

1. New policy final MuJoCo state must reproduce the source policy final state.
2. New policy endpoint feature ensembles must reproduce source policy endpoints.

No reward formula, action cap, anchor, candidate, statistical threshold, or bootstrap
setting changed. The invalid local output is retained separately as:

```text
evaluate_results/directed_state_seed2042_invalid_camera_sequence/
```

All numbers below come only from the corrected `directed_state_seed2042_final` run.

## Protocol

The experiment reuses the ten validated seed-2042 anchor states, source policy action
chunks, and FastWAM imagined goals from the previous exact-state experiment. It makes
zero new model inferences and performs no training.

```text
suite: libero_goal
task: task 3, open the top drawer and put the bowl inside
anchors: 10
executed actions per branch: 8
render repeats per current/endpoint: 8
translation magnitude cap: 0.25
branches: policy / toward_bowl / away_from_bowl / zero
```

At every step, `toward_bowl` and `away_from_bowl` use identical translation
magnitudes, policy rotations, and policy gripper commands. Translation is the only
paired difference and the translation vectors are exact opposites. The local oracle
is end-effector-to-bowl distance progress from MuJoCo state.

The complete protocol and audit amendment are in
`experiments/libero/DIRECTED_COUNTERFACTUAL_REWARD_PROTOCOL.md`.

## Integrity and physical manipulation

All structural, exact-state, and cross-source checks pass:

| Check | Result | Limit |
| --- | ---: | ---: |
| Anchors | `10` | exactly `10` |
| Branches | `40` | exactly `40` |
| Simulator action steps | `320` | exactly `320` |
| New FastWAM inferences | `0` | exactly `0` |
| Maximum branch-anchor cosine distance | `3.982e-5` | `1.0e-4` |
| Maximum branch-anchor feature L2 | `0.008934` | `0.015` |
| Maximum new/source policy state difference | `7.498e-9` | `1.0e-7` |
| Maximum new/source policy feature cosine distance | `2.331e-5` | `1.0e-4` |
| Maximum new/source policy feature L2 | `0.006823` | `0.015` |
| Exact paired-action construction | passed | required |

The first policy endpoint PNG for both cameras is pixel-identical to the source at all
ten anchors. Feature gates use all eight renders rather than relying on that single
pixel check.

The counterfactual manipulation is strong and symmetric:

| Branch | Mean end-effector-to-bowl progress |
| --- | ---: |
| Policy | `+16.361 mm` |
| Toward bowl | `+24.825 mm` |
| Away from bowl | `-24.267 mm` |
| Zero | `+0.0016 mm` |

Geometry gates:

```text
toward progress > 0: 10/10
away progress < 0: 10/10
toward progress > away progress: 10/10
mean paired geometric difference: +49.092 mm
95% paired-bootstrap CI: [+49.061, +49.122] mm
```

## Primary result

The prospectively frozen primary reward is:

```text
raw_dual = 0.5 * agent_delta_alignment + 0.5 * wrist_delta_alignment
```

Mean reward by branch:

| Branch | Mean raw dual reward |
| --- | ---: |
| Policy | `0.230884` |
| Toward bowl | `0.108678` |
| Away from bowl | `0.059766` |
| Zero | `0.003303` |

Directed paired result:

```text
reward(toward) > reward(away): 8/10 anchors
mean toward-minus-away reward: +0.048912
95% paired-bootstrap CI: [+0.019048, +0.080404]
```

Within each anchor, reward was ranked against actual MuJoCo distance progress across
`policy/toward/away/zero`:

```text
positive reward-vs-geometry Spearman: 8/10 anchors
mean Spearman: 0.52
95% paired-bootstrap CI: [0.34, 0.68]
```

No-op suppression remains below the preregistered limit:

```text
mean absolute zero / mean absolute policy reward = 3.59%
required <= 5%
```

## Preregistered gates

| Gate | Result | Pass |
| --- | ---: | :---: |
| Exactly ten anchors | `10` | yes |
| Toward geometry positive | `10/10`, required `>=9/10` | yes |
| Away geometry negative | `10/10`, required `>=9/10` | yes |
| Toward geometry exceeds away | `10/10` | yes |
| Reward toward exceeds away | `8/10`, required `>=8/10` | yes |
| Toward-minus-away reward CI lower > 0 | `0.019048` | yes |
| Positive reward/geometry Spearman | `8/10`, required `>=8/10` | yes |
| Mean Spearman CI lower > 0 | `0.34` | yes |
| Zero/policy absolute reward <= 5% | `3.59%` | yes |
| Source policy endpoint feature alignment | passed | yes |

Formal decision:

```text
supports_directed_imagination_reward_hypothesis
```

The two count gates pass exactly at their `8/10` thresholds, not with a wide count
margin. The positive paired-difference and Spearman confidence intervals provide the
stronger aggregate evidence.

## Camera and secondary diagnostics

| Metric | Policy | Toward | Away | Zero | Toward > away | Mean difference 95% CI | Positive geometry rho | Zero/policy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw equal dual | 0.23088 | 0.10868 | 0.05977 | 0.00330 | 8/10 | `[0.01905, 0.08040]` | 8/10 | 3.59% |
| Agent | 0.30308 | 0.14645 | 0.06303 | -0.00427 | 8/10 | `[0.02248, 0.14518]` | 8/10 | 4.27% |
| Wrist | 0.15869 | 0.07091 | 0.05651 | 0.01088 | 6/10 | `[-0.00157, 0.03178]` | 6/10 | 6.86% |
| Concat progress | 0.00990 | 0.00528 | -0.00891 | -0.00063 | 10/10 | `[0.00824, 0.02055]` | 10/10 | 9.70% |
| Dual progress | 0.01500 | -0.00023 | 0.00155 | 0.00081 | 5/10 | `[-0.00593, 0.00161]` | 5/10 | 7.41% |
| Frozen old camera weight | 0.51541 | 0.23535 | 0.16297 | 0.02387 | 8/10 | `[0.02732, 0.11875]` | 8/10 | 4.90% |
| Frozen normalized hybrid | 0.99452 | 0.49897 | -0.31669 | -0.02441 | 10/10 | `[0.48035, 1.16413]` | 10/10 | 6.53% |

Equal dual remains the primary metric. Agent supplies most of its directional margin;
wrist alone does not pass the paired-direction or no-op thresholds. Equal dual still
passes because wrist adds useful state information on some anchors without erasing
the stronger agent signal. No new camera weight is selected on these results.

`concat_progress` is the best pure geometric-order diagnostic, but its zero/policy
ratio is `9.70%`. The frozen normalized hybrid gets perfect directed and geometry
counts, but its no-op ratio is `6.53%`, above the fixed `5%` gate. It is therefore not
promoted post hoc. A future independently validated reward may combine direction and
progress, but this dataset is not used to select a new formula.

## The two paired counterexamples

Anchors 2 and 3 are retained as real primary failures.

Anchor 2:

```text
geometry: toward +24.384 mm, away -24.636 mm
agent reward: toward -0.002532, away 0.046408
wrist reward: toward 0.124417, away 0.135537
raw dual: toward 0.060942, away 0.090973
paired difference: -0.030030
```

Both cameras prefer the wrong direction, although concat progress correctly prefers
toward. This is the strongest remaining counterexample to using direction alignment
alone.

Anchor 3:

```text
geometry: toward +24.650 mm, away -24.411 mm
agent reward: toward 0.062756, away 0.134510
wrist reward: toward 0.128820, away 0.062298
raw dual: toward 0.095788, away 0.098404
paired difference: -0.002616
```

Here the cameras disagree and equal averaging yields a near tie slightly favoring
away. No per-anchor weighting or threshold is added after inspection.

## Interpretation for RL

The corrected experiment resolves the main ambiguity in the previous Gaussian-noise
severity test. When action quality is controlled by a real physical direction,
`raw_dual` contains a reproducible local signal for the correct direction.

However, mean reward ordering is:

```text
policy > toward > away > zero
```

while local geometry ordering is:

```text
toward > policy > zero > away
```

This difference is fundamental:

- the imagined goal was generated together with the source policy, so policy changes
  naturally match the imagination best;
- toward and away retain policy rotation, so wrong translation can still produce
  positively aligned visual change;
- direction alignment recognizes relative direction on most anchors but does not make
  every task-worsening action negative;
- two strong/near-tie counterexamples remain;
- optimizing imagination consistency alone could reward motion that looks partially
  policy-like without improving the task.

Therefore the evidence supports only the combined structure:

```text
R_total = R_success + lambda_imit * R_imitation
                      + lambda_imag * R_imagination
```

It does not support replacing environment success or FastWAM imitation with
`R_imagination`.

## Limitations and next decision

The validation covers one task, one early approach phase, one source seed, and ten
anchors. End-effector-to-bowl distance is valid here because the bowl and drawer do
not move during the eight-step window; it is not a universal reward after grasping or
during drawer manipulation. The experiment validates a local auxiliary signal, not
an RL improvement or a full-episode return.

The next engineering step can move to a minimal frozen-FastWAM residual policy, but
the first comparison must retain the causal ablation:

```text
A: frozen FastWAM, no RL
B: FastWAM residual RL + imitation + environment success
C: B + small bounded raw_dual imagination auxiliary
```

Success rate and interaction efficiency, not imagination return, remain primary. The
auxiliary contribution must be bounded so its full-episode sum cannot overwhelm one
true task success.

## Reproduction

Use the commands in
`experiments/libero/DIRECTED_COUNTERFACTUAL_REWARD_PROTOCOL.md`. The corrected formal
ignored output is:

```text
evaluate_results/directed_state_seed2042_final/
```

It contains the manifest, exact states, matched actions, repeated lossless renders,
frozen SigLIP feature cache, reward rows, and `analysis/summary.json`.

Corrected summary SHA-256:

```text
7ad07cdeb00cac13c8cee83abe0dcc1fdc7f155579457e04b9c40fa8d1de71b5
```
