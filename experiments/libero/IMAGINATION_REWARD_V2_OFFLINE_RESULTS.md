# LIBERO camera-aware imagination Reward V2: offline diagnostic

## Scope

This is the minimum offline experiment following the rejected seed-1042 Reward V1
candidate.  It reuses the two existing five-trial datasets and does **not** launch
LIBERO, collect new rollouts, train a policy, or tune a reward on seed 1042.

```text
task: libero_goal task 3, open the top drawer and put the bowl inside
seeds: 42 and 1042
action modes per seed: policy / noise(std=0.15) / zero
episodes: 30
alignment-valid transitions: 1156
frozen encoder: local SigLIP2 So400m patch14 224
camera layout: horizontal agent view | wrist view, 224x224 each
```

The evaluation code confirms that the left half is `agentview_image` and the right
half is `robot0_eye_in_hand_image`.  The diagnostic splitter rejects any image that
is not exactly two square horizontal views, rather than silently accepting a changed
layout.

## Reward definition

For each camera, define the actual and imagined feature changes from the same current
observation:

```text
delta_actual = feature(actual_t+K) - feature(current_t)
delta_goal   = feature(imagined_t+K) - feature(current_t)
```

The camera-level Reward V2 diagnostic is:

```text
r_delta = cosine(delta_actual, delta_goal)
          * min(norm(delta_actual) / norm(delta_goal), 1)
```

The cosine tests whether execution changed the scene in the imagined direction.  The
magnitude ratio makes a visually static transition approach zero without introducing
a dataset-tuned no-op threshold.

Four candidates were fixed before reading the results:

1. `concat_progress`: original distance progress on the concatenated image.
2. `agent_delta_alignment`: Reward V2 on the external agent camera.
3. `dual_delta_alignment`: equal mean of agent and wrist Reward V2.
4. `concat_progress_plus_dual_delta`: progress plus a fixed `0.01` direction term.

The `0.01` coefficient is only a predeclared diagnostic scale.  It was not scanned or
selected using seed 1042.

## Main result

The episode-level results are:

| Candidate | Seed | Policy | Noise | Zero | `policy > noise > zero` | Success AUC | Correct goal > wrong goal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| concat progress | 42 | 0.008490 | 0.002505 | 0.000018 | 5/5 | 0.946 | 53.79% |
| concat progress | 1042 | 0.008602 | 0.001193 | 0.000010 | 4/5 | 0.857 | 53.82% |
| agent delta | 42 | 0.379977 | 0.232269 | 0.004500 | 5/5 | 0.982 | 66.84% |
| agent delta | 1042 | 0.383518 | 0.218861 | 0.002712 | 5/5 | 1.000 | 66.38% |
| dual delta | 42 | 0.288117 | 0.192338 | 0.002845 | 5/5 | 0.964 | 69.49% |
| dual delta | 1042 | 0.290331 | 0.185222 | 0.001943 | 5/5 | 1.000 | 68.93% |
| progress + dual delta | 42 | 0.011371 | 0.004429 | 0.000046 | 5/5 | 0.964 | 55.56% |
| progress + dual delta | 1042 | 0.011505 | 0.003045 | 0.000029 | 5/5 | 0.946 | 56.20% |

Across both seeds, the dual-camera delta candidate obtains:

```text
paired policy > noise > zero: 10/10 trials
successful action mode > failed action mode: 20/20 comparable pairs
episode success ROC AUC: 0.9866
correct goal > selected wrong goal: 69.20%
```

Its zero-action episode mean is about `0.00239`, compared with `0.18878` for noisy
actions and `0.28922` for policy actions.  The magnitude factor therefore suppresses
the no-op shortcut on these trajectories without a fitted threshold.

## Seed-1042 counterexamples

Trial 2 was the decisive Reward V1 counterexample: noise succeeded and zero failed,
while concatenated progress still ranked noise below zero.

| Candidate | Policy (success) | Noise (success) | Zero (failure) | Correct order |
| --- | ---: | ---: | ---: | :---: |
| concat progress | 0.005476 | -0.000325 | -0.000014 | no |
| agent delta | 0.347746 | 0.259137 | 0.005340 | yes |
| dual delta | 0.282604 | 0.208929 | 0.003644 | yes |
| progress + dual delta | 0.008302 | 0.001765 | 0.000022 | yes |

In trial 3, pure concatenated progress already ranks policy, noise, and zero correctly;
the previously rejected Reward V1 `progress + 0.05 * match` formula was the component
that broke that ordering.  Both delta candidates retain a large separation:

```text
trial 3 agent delta: policy 0.347791, noise 0.208220, zero 0.001831
trial 3 dual delta:  policy 0.284031, noise 0.192319, zero 0.001385
```

The saved transition curves show that progress repeatedly changes sign and stays near
zero for good noisy actions, whereas delta alignment remains positive through most of
their trajectory.  Zero-action transitions stay concentrated near zero in both
cameras.

## Interpretation and decision

This result supports the minimum hypothesis under test: comparing the **change caused
by execution** with the **change imagined from the same current state** is a better
action-quality signal than asking only whether the final image became closer to the
imagined image.  The improvement reproduces independently on seed 42 and seed 1042,
including the seed-1042 trial-2 failure case.

It does not yet justify policy optimization:

- The experiment covers one LIBERO task, one checkpoint, two seeds, and 30 already
  collected episodes.  It demonstrates association, not a causal RL improvement.
- The dual candidate reaches `69.49%` and `68.93%` correct-vs-wrong discrimination on
  the two seeds, narrowly below the previously declared 70% threshold.
- The fixed progress-plus-direction mixture fixes action ordering but damages goal
  discrimination, so it is rejected rather than reweighted on seed 1042.
- The current wrong goal is a deterministic distant goal from another episode.  It is
  not yet a controlled hard negative with matched current state and task phase.

Therefore `dual_delta_alignment` is the only candidate worth carrying into the next
diagnostic, but it is **not** approved as an RL reward yet.  The minimum next check is
to test it against controlled same-phase wrong goals and repeated/no-op frames using
the existing feature cache, before collecting data for another task.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. \
  /home/ubuntu/miniconda3/envs/fastwam/bin/python \
  experiments/libero/diagnose_camera_delta_rewards.py \
  --input-dir evaluate_results/imagination_validation_5eps \
  --input-dir evaluate_results/imagination_validation_seed1042 \
  --encoder-path /home/ubuntu/zhumj/code/dsrl/ckpt/siglip2-so400m-patch14-224 \
  --output-dir evaluate_results/imagination_reward_v2_offline \
  --feature-cache evaluate_results/imagination_reward_v2_offline/camera_siglip_features.npz \
  --device cuda \
  --batch-size 16 \
  --direction-weight 0.01 \
  --diagnostic-seed 1042 \
  --diagnostic-trials 2 3
```

The ignored local output directory contains the feature cache, transition and episode
JSONL files, the full JSON summary, and the seed-1042 trial-2/trial-3 PNG curves.
