# Corrected LIBERO residual-RL smoke result (2026-07-21)

## Decision

The corrected single-GPU smoke passed end to end. The run verifies that the frozen
FastWAM collector, versioned two-camera imagination reward, replay builder, matched
residual-AWR training, checkpoint provenance checks, and online residual evaluation
work together.

The online result contains a small positive directional signal for the imagination
reward: both residual policies succeeded, while the imagination-trained policy used
three fewer simulator policy steps, a smaller residual, and had slightly higher
future-video PSNR. This is **not effectiveness evidence** because each condition has
only one episode, the learner replay has only two source episodes, and evaluation
reuses the replay task and initial-state seed.

The defensible conclusion is therefore:

```text
engineering path: passed
matched reward ablation: passed
imagination reward changes the learned policy: confirmed
imagination reward improves LIBERO success: not yet established
next gate: repeated paired evaluation before scaling training
```

The superseded pipeline-only run remains documented in
`RESIDUAL_RL_SMOKE_20260721_RESULTS.md`; its output directory was not modified.

## Frozen run identity

```text
output root:
  /home/ubuntu/sj/fastwam/runs/local_residual_rl_corrected_4ebe877

git commit:
  4ebe8774e7deba38572c4376cdfe10494e2ce2e0

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
trials per collection/evaluation condition: 1
K / residual horizon: 8
FastWAM action horizon: 32
diffusion inference steps: 4
noise standard deviation: 0.075
camera weights: agent 0.5 / wrist 0.5
camera image size: 224
feature fusion: per_camera_l2_then_agent_wrist_concat_l2_v1
imagination reward: delta_alignment_v1
timeout bootstrap: explicit 0.0
```

The run's complete frozen arguments are stored in `smoke_config.env` under the output
root. Every completed stage also has a marker under `.stages` and an independent log
under `logs`.

## Collection and replay result

| Behavior | Success | Simulator policy steps | Saved transitions | Valid K-aligned transitions | Future PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen FastWAM policy | `0/1` | `400` | `50` | `50` | `25.22964` |
| FastWAM + Gaussian noise 0.075 | `1/1` | `169` | `22` | `21` | `27.45523` |

Replay schema and provenance validation passed:

```text
schema version: 2
num transitions: 72
observation feature: 2304
proprio: 8
action chunk: 8 x 7
behavior counts: policy 50 / noise 22
overall executed residual RMS: 0.0368895548
imagination reward type: delta_alignment_v1
camera order: agent, wrist
camera weights: 0.5, 0.5
```

These 72 transitions are not 72 independent trials. They come from two trajectories,
and the only successful replay trajectory is the noisy one. The learner can therefore
memorize properties of one favorable exploration sample.

## Matched learner result

Both learners used the same replay, seed, architecture, optimizer settings, balanced
policy/noise minibatches, and 20 epochs. The only intended reward difference was
`imagination_weight=0.0` versus `imagination_weight=1.0`.

Initialization was verified after moving all seeding before model construction:

```text
actor initialization SHA-256:
  8884e88dfdbcbaaf8bf952f671bfa7881248ff47250022b04e2eafd442b60079

critic initialization SHA-256:
  3775617ba1fd354b002d757a5e0d063c4ef7b60bfacef94fdf338ebe08ac8d3b
```

The hashes match exactly across the two jobs.

| Reward/return metric | No imagination | With delta alignment | Difference |
| --- | ---: | ---: | ---: |
| Mean transition reward | `0.10354552` | `0.19856940` | `+0.09502389` |
| Minimum transition reward | `-0.17183402` | `-0.07183403` | `+0.10000000` |
| Maximum transition reward | `9.92159748` | `9.92159748` | `0.00000000` |
| Mean return | `1.23519850` | `2.08185196` | `+0.84665346` |
| Minimum return | `0.00000000` | `0.10000000` | `+0.10000000` |
| Maximum return | `9.92159748` | `9.92159748` | `0.00000000` |

The imagination signal remains auxiliary: it is clipped to `[-0.1, 0.1]` per
transition and episode-level absolute shaping is capped relative to the success
bonus. Its cumulative effect can nevertheless change AWR advantage ordering across a
long unsuccessful trajectory, which explains the visible mean-return change.

| Final epoch metric | No imagination | With delta alignment |
| --- | ---: | ---: |
| Actor loss | `0.00290108` | `0.00263795` |
| Critic loss | `1.07829899` | `1.11799127` |
| Mean action MSE | `0.00183071` | `0.00169066` |
| Maximum advantage weight | `15.34920454` | `7.64664268` |

Both histories contain 20 finite epochs. On all 72 replay contexts:

```text
no-imagination predicted residual RMS: 0.02600000
with-imagination predicted residual RMS: 0.02245226
RMS difference between actor outputs: 0.00915589
maximum absolute actor-output difference: 0.02327673
final actor parameter relative L2 difference: 0.01319490
```

The matching initialization and nonzero final differences confirm that reward
relabeling changes the learned policy. They do not establish which policy generalizes
better.

## Online evaluation result

All three conditions used task 3, seed 42, and the same initial-state protocol.

| Online policy | Success | Simulator policy steps | Residual RMS | Future PSNR |
| --- | ---: | ---: | ---: | ---: |
| Frozen FastWAM | `0/1` | `400` | n/a | `25.22964` |
| Residual AWR, no imagination | `1/1` | `172` | `0.02581645` | `27.25461` |
| Residual AWR, delta alignment | `1/1` | `169` | `0.02238325` | `27.54162` |

Relative to the no-imagination residual, the imagination-trained residual used three
fewer policy steps (`-1.74%`), had residual RMS lower by `0.00343321` (`-13.30%`), and
future-video PSNR higher by `0.28701 dB`. These are descriptive single-episode
differences, not confidence-bearing estimates. PSNR is a model-prediction diagnostic,
not a substitute for task success.

The baseline failure plus noisy-behavior success indicates that this initial state is
sensitive to small action changes. Both learned residual policies may mainly imitate
the one successful noise trajectory. In addition, evaluation uses the same task and
seed represented in the replay, so this smoke does not test held-out generalization.

## Engineering checks

```text
58 unit tests passed
replay array and metadata checksums passed
schema/reward/provenance validation passed
matched actor and critic initialization hashes passed
both 20-epoch histories contain no NaN or Inf
both checkpoints passed strict online encoder provenance validation
all three evaluation result JSON files were written
smoke orchestration exited with status 0
```

Robosuite emitted ignored `EGL_NOT_INITIALIZED` exceptions while destructing an
already-finished render context. They occurred after result JSON and videos were
saved; the orchestration process returned zero, so they do not invalidate the runs.

## Minimal next experiment

Do not enlarge the actor, update FastWAM, or begin a long online RL run yet. The next
experiment should reuse these frozen checkpoints and run repeated paired evaluation:

1. compare frozen FastWAM, no-imagination residual, and imagination residual;
2. use at least 10 trials on task 3 with an explicitly frozen evaluation seed list;
3. report paired success outcomes first, then successful-episode step count and
   residual RMS as secondary metrics;
4. do not rebuild the replay or retrain between evaluation conditions; and
5. interpret the imagination reward as promising only if its advantage is repeated
   across multiple initial states, rather than being driven by this single seed.

If the two residual policies remain tied on success, add more replay trajectories and
held-out seeds before changing reward weights. The present smoke supports proceeding
to that repeated evaluation, but it does not justify a claim that imagination reward
already improves LIBERO performance.
