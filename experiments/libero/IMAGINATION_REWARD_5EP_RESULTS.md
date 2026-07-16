# LIBERO imagination-reward 5-episode validation

## Scope

This is a reward-validation experiment, not policy training. The released direct FastWAM checkpoint was frozen. For each replan, the released `infer_action()` output was used as the policy action while the video expert generated a separate `t+8` goal. A frozen local SigLIP vision encoder measured cosine distance.

```text
suite: libero_goal
task_id: 3
task: open the top drawer and put the bowl inside
K: 8
seed: 42
paired LIBERO initial states: 5
action modes: policy, noise(std=0.15), zero
```

## Environment outcomes

| Action mode | Successes | Mean executed steps | Valid transitions | Future-video PSNR |
| --- | ---: | ---: | ---: | ---: |
| policy | 5/5 | 193.0 | 118 | 28.2921 dB |
| noise | 2/5 | 320.8 | 199 | 26.2938 dB |
| zero | 0/5 | 400.0 | 250 | 24.3735 dB |

## Imagination-progress outcomes

| Action mode | Mean progress per episode transition | Mean episode imagination return |
| --- | ---: | ---: |
| policy | 0.0084587 | 0.1956174 |
| noise | 0.0024573 | 0.0890563 |
| zero | 0.0000173 | 0.0008639 |

All five paired initial states satisfied:

```text
policy progress > noise progress > zero progress
```

Across success labels:

| Episode outcome | Mean progress per transition | Mean episode imagination return |
| --- | ---: | ---: |
| success | 0.0069537 | 0.1611662 |
| failure | 0.0007488 | 0.0374406 |

## Wrong-goal control

Wrong goals were selected from the same task and a different episode, preferring the same action mode and the most distant available replan index.

```text
correct progress > wrong-goal progress: 56.44%
actual outcome closer to correct goal than wrong goal: 89.42%
```

The absolute target-matching check is strong, but the relative-progress comparison does not meet the provisional 70% threshold. This difference is expected to matter: relative progress includes each goal's different starting distance, while the absolute comparison asks the simpler question of which goal the actual outcome matches.

## Initial decision

The reward passes the action-quality and stationary-policy smoke tests:

- `policy > noise > zero` held for all five paired initial states.
- Success rate degraded in the expected order: `100% > 40% > 0%`.
- Successful episodes had substantially higher progress than failed episodes.
- Zero actions produced progress close to zero.

The result is therefore sufficient to continue reward design, but not yet sufficient to start reinforcement-learning updates. Before using the reward for policy optimization, the goal-specific component should resolve the weak `correct progress > wrong progress` result, for example by reporting or lightly incorporating absolute goal matching while keeping environment success dominant.

Raw videos, aligned PNG triplets, transition JSONL, and episode JSONL remain under the ignored local `evaluate_results/imagination_validation_5eps/` directory. They are intentionally not committed because of their size.
