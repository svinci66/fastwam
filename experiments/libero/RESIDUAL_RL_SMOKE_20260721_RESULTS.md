# LIBERO residual-RL local smoke result (2026-07-21)

## Decision

This run validates the end-to-end engineering path from frozen FastWAM collection
through replay construction and residual-AWR checkpoint export. It does **not**
establish that imagination shaping improves LIBERO success.

The run is retained as a pipeline smoke and must not be used as the formal reward
ablation because:

1. actor and critic seeds were applied after model construction, so the two learners
   were not guaranteed to start from identical weights; and
2. the replay used cosine-distance progress (`progress_v1`), while the strongest
   prospective directed-counterfactual evidence supports equal-camera delta alignment
   (`delta_alignment_v1`).

Both limitations were identified before the next corrected run. The original output
directory is preserved without modification.

## Frozen run identity

```text
output root:
  /home/ubuntu/sj/fastwam/runs/local_residual_rl_smoke_release_69bdb37_20260721_141100

git commit:
  69bdb37253ae93b66d4457778d9363b7b1cf9609

FastWAM checkpoint:
  /home/ubuntu/sj/fastwam/checkpoints/fastwam_release_clean/libero_uncond_2cam224.pt

dataset statistics:
  /home/ubuntu/sj/fastwam/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json

reward encoder:
  google/siglip-so400m-patch14-384@sha256:ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a
```

Frozen protocol:

```text
suite/task: libero_goal / task 3
language: open the top drawer and put the bowl inside
seed: 42
trials per behavior: 1
K / residual horizon: 8
FastWAM action horizon: 32
diffusion inference steps: 4
noise standard deviation: 0.075
camera weights: agent 0.5 / wrist 0.5
timeout bootstrap: explicit 0.0
```

## Collection result

| Behavior | Success | Simulator policy steps | Saved transitions | Valid K-aligned transitions |
| --- | ---: | ---: | ---: | ---: |
| Frozen FastWAM policy | `0/1` | `400` | `50` | `50` |
| FastWAM + Gaussian noise 0.075 | `1/1` | `173` | `22` | `21` |

The 72 transitions come from only two independent episodes. They must not be treated
as 72 independent experimental units. In particular, the only successful trajectory
is the noisy trajectory, so the success bonus strongly favors that one sampled noise
sequence.

Replay schema:

```text
num transitions: 72
observation feature: 2304
proprio: 8
action chunk: 8 x 7
behavior counts: policy 50 / noise 22
overall executed residual RMS: 0.0369495460
noise-only executed residual RMS: 0.0672455245
```

## Reward relabeling result

| Metric | Success + imitation | Success + imitation + progress_v1 |
| --- | ---: | ---: |
| Mean transition reward | `0.10387459` | `0.10733734` |
| Mean return | `1.23881972` | `1.25393617` |
| Minimum return | `0.00000000` | `-0.03467863` |
| Maximum return | `9.95110130` | `9.95110130` |

The imagination term changes mean return by `+0.01511645` (about `+1.22%`). It is
therefore auxiliary relative to the success bonus, as intended.

Per-behavior stored imagination diagnostics:

| Behavior | Mean raw imagination | Positive fraction | Episode applied sum |
| --- | ---: | ---: | ---: |
| Policy | `0.00192307` | `52.0%` | `0.09615343` |
| Noise 0.075 | `0.00728692` | `77.27%` | `0.15316523` |

This descriptive ordering is consistent with the sampled noisy episode succeeding,
but the two branches leave the shared initial state after their first actions. It is
not a same-state causal reward comparison.

## Learner result

Both learners completed 20 epochs and produced finite checkpoints and histories.

| Final metric | No imagination | With progress_v1 |
| --- | ---: | ---: |
| Actor loss | `0.00215708` | `0.00255819` |
| Critic loss | `2.68655527` | `3.05516028` |
| Mean action MSE | `0.00199375` | `0.00205973` |
| Predicted residual RMS on replay | `0.01952421` | `0.02179084` |
| Predicted residual maximum absolute value | `0.07811765` | `0.08492722` |

The RMS difference between the two final actor residuals on the same replay is
`0.00891922` (maximum absolute difference `0.03093718`). Because initial model weights
were not controlled before construction, this difference cannot be attributed solely
to the reward ablation.

## Engineering checks

The original run passed all tests present at commit `69bdb37`:

```text
52 passed
no non-finite history values
replay checksums valid
both checkpoints written
```

EGL emitted ignored destructor-time `EGL_NOT_INITIALIZED` messages after result files
were saved. Both collectors returned success and the messages do not invalidate the
recorded simulator trajectories.

## Required correction before interpretation

The next run must:

1. seed Python/NumPy/PyTorch before actor and critic construction and record their
   initial state hashes;
2. version the imagination formula and use the prospectively supported equal-camera
   delta-alignment auxiliary;
3. bind the residual checkpoint to the exact encoder/preprocessor provenance;
4. evaluate frozen FastWAM, no-imagination residual, and with-imagination residual on
   identical initial states; and
5. continue treating a one-episode run as a smoke rather than effectiveness evidence.
