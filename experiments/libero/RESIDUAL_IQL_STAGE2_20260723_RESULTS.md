# Residual IQL stage-2 implementation and offline smoke — 2026-07-23

## Decision

The second-stage offline-RL path is implemented and passes its first matched
training gate:

```text
accepted 1,573-transition normalized replay
  -> matched no-imagination / imagination IQL initialization
  -> twin action-value critics + expectile value critic
  -> 20 epochs of finite offline optimization
  -> two loadable fastwam_residual_iql_v1 actor checkpoints
```

This proves that the normalized imagination reward reaches a stronger offline-RL
learner and changes its deployed residual actor. It does **not** prove an online
success-rate improvement. The current replay remains success-heavy.

The subsequent local held-out rollout is recorded in
`RESIDUAL_IQL_HELDOUT_20260723_RESULTS.md`: the matched no-imagination and
imagination actors achieved `50/50` and `49/50`, respectively. The deployment gate
passed, but the success-improvement hypothesis did not.

## Algorithm boundary

FastWAM is frozen. The IQL learner contains:

- the existing bounded deterministic residual actor;
- two action-conditioned Q critics, each conditioned on observation, proprioception,
  language, frozen FastWAM baseline actions, and executed actions;
- an expectile value critic;
- chunk-aware discounts `gamma ** effective_k`;
- task-balanced minibatches;
- advantage-weighted behavior regression for actor extraction.

Padded suffixes after `effective_k` are replaced with FastWAM baseline actions before
being passed to a Q critic. Timeouts are treated as terminal in this first version
because the replay does not contain a valid FastWAM baseline action for a state after
the time limit.

Frozen parameters:

```text
gamma:                  0.99
expectile:              0.7
advantage temperature:  3.0
maximum actor weight:   100
target update tau:      0.005
batch size:             64
epochs:                 20
seed:                   42
```

## Replay and reward ablation

Replay:

```text
/home/ubuntu/sj/fastwam/runs/
libero_goal_global_camera_norm_6d0399e_20260722_152000/replay
```

It contains 1,573 schema-v3 transitions from all ten LIBERO Goal tasks, including
frozen UMT5 language features and the accepted
`delta_alignment_global_camera_norm_v1` values.

The two variants share the replay and identical actor/Q/value initialization hashes.
The only intended reward difference is:

```text
control imagination_weight:  0.0
test imagination_weight:     1.0
```

An attempted run against the earlier `delta_alignment_v1` replay was rejected before
training. This is the intended provenance guard: a config may not reinterpret raw
imagination values as globally normalized values.

## 20-epoch result

Output:

```text
/home/ubuntu/sj/fastwam/runs/
libero_goal_iql_pair_20ep_20260723_1120
```

All recorded metrics are finite:

| Metric | No imagination | With imagination |
| --- | ---: | ---: |
| Initial Q loss | 12.023766 | 12.045361 |
| Final Q loss | 0.753128 | 0.732471 |
| Initial value loss | 0.202573 | 0.202649 |
| Final value loss | 0.021425 | 0.018891 |
| Final actor loss | 0.002681 | 0.002308 |
| Final masked action MSE | 0.002376 | 0.002387 |
| Final mean Q | 1.705867 | 1.740901 |
| Predicted residual RMS | 0.007305 | 0.006822 |

The replay-wide paired actor-output delta RMS is `0.007003`. Actor state hashes are
different, while all initial actor/Q/value hashes are identical. The reward therefore
materially changes IQL's learned policy rather than only changing logged losses.

The final loss difference is not a task-performance metric. Online held-out evaluation
is still required.

## Verification

```text
targeted IQL/AWR/checkpoint tests: 18 passed
repository tests directory:        83 passed
Python compilation:                passed
shell syntax checks:               passed
git diff --check:                  passed
```

The new IQL checkpoint format is accepted by the existing online residual evaluator;
legacy AWR v1/v2 loading remains supported.

## Formal data pipeline

`scripts/run_libero_residual_iql_stage2.sh` implements the resumable formal default:

```text
tasks:       LIBERO Goal 0..9
train states per task:
             0..4 and 10..34 (30 states)
behaviors:   policy, Gaussian noise 0.075, Gaussian noise 0.15
episodes:    10 x 30 x 3 = 900
validation:  states 5..9 remain outside the collection
final test:  states 35..49 remain untouched
```

Each episode derives policy and action RNG seeds from
`(base_seed, task_id, trial_index, stream)`. Reordering or resuming collection
therefore does not change the stochastic sequence assigned to a state. Paired
behaviors share a FastWAM policy seed but use distinct action-noise streams. The raw
collection audit verifies both properties before replay construction.

The script then builds the globally normalized two-camera replay, validates both IQL
configs, and trains the matched pair. It is ready for a GPU server, but the 900
episodes have not been collected by this local implementation run.

## Next gate

1. Keep held-out states `5..9` fixed and do not use them for reward tuning.
2. The task-3 collection pilot passed with 28 failures in 90 episodes and no
   structural audit errors.
3. Resume the same output root over the remaining nine tasks.
4. Build and audit the globally normalized all-task replay.
5. Train at least three seeds and reserve states `35..49` for the final comparison.

Do not claim IQL or imagination-reward success improvement until the paired online
gate is complete.
