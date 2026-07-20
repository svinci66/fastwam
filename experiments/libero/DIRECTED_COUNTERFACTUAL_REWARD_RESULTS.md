# LIBERO directed exact-state imagination-reward result

> **Temporarily invalidated:** the first attempted formal collection did not replay
> the source collector's per-anchor LIBERO soft-reset sequence. Its flattened MuJoCo
> states matched, but anchors 1--9 used mismatched model-level camera state relative
> to the reused imagined goals. The numerical results below are retained only as an
> audit trail and must not be cited. A corrected, cross-source-gated run is pending.

## Decision

The frozen primary `raw_dual` reward passes every preregistered directed test. This
supports the narrow hypothesis that execution-versus-imagination direction alignment
contains task-relevant local direction information.

It does **not** approve `raw_dual` as a complete RL reward. The wrong-direction branch
still receives positive reward because it retains the policy rotation and produces
non-static visual change. The result instead supports using imagination consistency as
a bounded auxiliary term alongside FastWAM imitation and environment success.

The protocol and implementation were committed and pushed before the formal reward
analysis:

```text
branch: feat/libero-directed-counterfactual-reward
preregistration commit: 3c4d41018e7ec0c75cd6dfc026f12e98ed9939ed
```

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
paired difference and the two translation vectors are exact opposites. The local
oracle is end-effector-to-bowl distance progress from MuJoCo state.

The complete preregistration is in
`experiments/libero/DIRECTED_COUNTERFACTUAL_REWARD_PROTOCOL.md`.

## Integrity and physical manipulation

All structural and exact-state checks pass:

| Check | Result |
| --- | ---: |
| Anchors | `10` |
| Branches | `40` |
| Simulator action steps | `320` |
| New FastWAM inferences | `0` |
| Maximum branch-anchor cosine distance | `1.323e-5` |
| Maximum branch-anchor feature L2 | `0.005143` |
| Exact action pairing | passed |
| Exact MuJoCo reconstruction | passed |

The counterfactual manipulation is strong and symmetric:

| Branch | Mean end-effector-to-bowl progress |
| --- | ---: |
| Policy | `+16.361 mm` |
| Toward bowl | `+24.825 mm` |
| Away from bowl | `-24.267 mm` |
| Zero | `+0.0016 mm` |

The geometry gates pass:

```text
toward progress > 0: 10/10
away progress < 0: 10/10
toward progress > away progress: 10/10
mean paired geometric difference: +49.092 mm
95% paired-bootstrap CI: [+49.061, +49.122] mm
```

## Primary result

The frozen primary reward is:

```text
raw_dual = 0.5 * agent_delta_alignment + 0.5 * wrist_delta_alignment
```

Mean reward by branch:

| Branch | Mean raw dual reward |
| --- | ---: |
| Policy | `0.145799` |
| Toward bowl | `0.082061` |
| Away from bowl | `0.043583` |
| Zero | `0.005396` |

The directed paired result is:

```text
reward(toward) > reward(away): 9/10 anchors
mean toward-minus-away reward: +0.038478
95% paired-bootstrap CI: [+0.015510, +0.069141]
```

Within each anchor, reward was ranked against the actual MuJoCo distance progress of
`policy/toward/away/zero`:

```text
positive reward-vs-geometry Spearman: 9/10 anchors
mean Spearman: 0.56
95% paired-bootstrap CI: [0.42, 0.66]
```

No-op suppression remains below the preregistered limit:

```text
mean absolute zero / mean absolute policy reward = 4.31%
required <= 5%
```

## Preregistered gates

| Gate | Result | Pass |
| --- | ---: | :---: |
| Exactly ten anchors | `10` | yes |
| Toward geometry positive | `10/10`, required `>=9/10` | yes |
| Away geometry negative | `10/10`, required `>=9/10` | yes |
| Toward geometry exceeds away | `10/10` | yes |
| Reward toward exceeds away | `9/10`, required `>=8/10` | yes |
| Toward-minus-away reward CI lower > 0 | `0.015510` | yes |
| Positive reward/geometry Spearman | `9/10`, required `>=8/10` | yes |
| Mean Spearman CI lower > 0 | `0.42` | yes |
| Zero/policy absolute reward <= 5% | `4.31%` | yes |

Formal decision:

```text
supports_directed_imagination_reward_hypothesis
```

## Camera and secondary diagnostics

| Metric | Policy | Toward | Away | Zero | Toward > away | Mean difference 95% CI | Positive geometry rho | Zero/policy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw equal dual | 0.14580 | 0.08206 | 0.04358 | 0.00540 | 9/10 | `[0.01551, 0.06914]` | 9/10 | 4.31% |
| Agent | 0.12480 | 0.10954 | 0.06136 | 0.00768 | 7/10 | `[0.01218, 0.09619]` | 7/10 | 8.55% |
| Wrist | 0.16679 | 0.05459 | 0.02581 | 0.00311 | 8/10 | `[0.01114, 0.04785]` | 8/10 | 3.36% |
| Concat progress | -0.01329 | 0.00169 | -0.00664 | 0.00008 | 7/10 | `[0.00187, 0.01547]` | 4/10 | 6.68% |
| Dual progress | 0.00768 | 0.00115 | -0.00119 | 0.00002 | 7/10 | `[-0.00231, 0.00706]` | 7/10 | 15.22% |
| Frozen old camera weight | 0.45300 | 0.17971 | 0.08935 | 0.01090 | 9/10 | `[0.03895, 0.14991]` | 9/10 | 3.40% |
| Frozen normalized hybrid | -0.35021 | 0.26204 | -0.23818 | 0.01564 | 8/10 | `[0.15803, 0.87711]` | 7/10 | 9.66% |

Equal dual is the only prospectively primary metric. The historical wrist-heavy
candidate performs well here but remains diagnostic because earlier controlled hard
negatives showed its calibration is not generally robust. The normalized hybrid is
rejected as a replacement: despite separating the paired directions, its geometry
Spearman interval crosses zero and its no-op ratio exceeds the fixed limit.

The cameras are complementary. Agent alone has the larger average paired margin but
fails three individual pairs and the no-op gate. Wrist alone passes eight pairs and
the no-op gate. Their equal mean passes nine pairs and all gates; no new weight is
selected on these results.

## The one paired counterexample

Anchor 1 is the only primary pair with `toward <= away`:

```text
geometry: toward +24.910 mm, away -24.184 mm
agent reward: toward 0.031835, away 0.043987
wrist reward: toward 0.051462, away 0.045336
raw dual: toward 0.041649, away 0.044662
paired difference: -0.003013
```

The error is small and the two cameras disagree: wrist chooses the correct direction,
while agent prefers away strongly enough to flip the equal mean. It is retained as a
real counterexample; no per-anchor weighting or post-hoc threshold is introduced.

## Interpretation for RL

This experiment resolves the ambiguity in the previous Gaussian-severity test. When
action quality is defined by a controlled, real physical direction rather than by
noise magnitude, the primary imagination reward consistently contains the correct
local direction signal.

However, the mean reward ordering is:

```text
policy > toward > away > zero
```

while the local geometry ordering is:

```text
toward > policy > zero > away
```

The difference is meaningful:

- the imagined goal was generated together with the source policy, so policy changes
  naturally match the imagination best;
- toward and away retain the same policy rotation, so even the wrong translation can
  produce positively aligned visual change;
- the reward recognizes relative direction but does not make every wrong task action
  negative;
- optimizing it alone could still reward motion that looks partially policy-like
  without improving the task.

Therefore the result supports this reward structure:

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

The next engineering step may now move to a minimal frozen-FastWAM residual policy,
but the first comparison must retain the causal ablation:

```text
A: frozen FastWAM, no RL
B: FastWAM residual RL + imitation + environment success
C: B + small bounded raw_dual imagination auxiliary
```

Success rate and interaction efficiency, not imagination return, remain the primary
outcomes. The auxiliary contribution must be bounded so its full-episode sum cannot
overwhelm one true task success.

## Reproduction

Use the commands frozen in
`experiments/libero/DIRECTED_COUNTERFACTUAL_REWARD_PROTOCOL.md`. The formal ignored
output is:

```text
evaluate_results/directed_state_seed2042_final/
```

It contains the manifest, exact states, matched actions, repeated lossless renders,
frozen SigLIP feature cache, reward rows, and `analysis/summary.json`.
