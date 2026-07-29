# RoboTwin Expert Residual-IQL Smoke Test (2026-07-29)

## Scope

This experiment checks whether a small residual actor can learn useful corrections
from a mixture of official successful demonstrations and the existing controlled
failure rollouts. It is an offline training smoke test, not yet an online success-rate
claim.

## Data

- Tasks: `adjust_bottle`, `open_laptop`, and `stack_blocks_two`.
- Official demonstrations: 10 successful episodes per task, downloaded from the
  [official RoboTwin 2.0 dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
- Expert transitions: 289 total (`66 / 89 / 134` by task).
- Existing controlled transitions: 870 total, including policy, action-hold, and
  gripper-delay behaviors.
- Combined replay: 1,159 transitions, of which 1,075 have a valid K=24 alignment.
- Observation features: SigLIP features from head, left-wrist, and right-wrist
  cameras, normalized globally with separate statistics for each camera.
- Conditioning: task language embedding, proprioception, and the frozen FastWAM
  baseline action chunk.

The replay manifest is:

```text
evaluate_results/robotwin_residual_rl/robotwin_expert10_residual_iql_20260729/replay/manifest.json
SHA-256: 4d4e0b34149115aafe67734689006dfa51b20cd0e3c25b77af96c17cfd95aaac
```

## Matched training setup

Two IQL residual actors were trained for 20 epochs with seed 42 and identical
initial actor, critic, and value-network weights:

1. `no_imagination`: environment/task-completion and action-imitation terms only.
2. `imagination`: the same setup plus the imagined-goal alignment reward.

Sampling is balanced jointly by task and behavior type so that the more numerous
hold failures do not dominate the official successful demonstrations. The residual
limit is 0.05 on arm action dimensions; gripper residual scale is zero, so gripper
commands remain exactly owned by frozen FastWAM.

## Offline results

| Metric on the 289 expert transitions | Frozen FastWAM | IQL without imagination | IQL with imagination |
| --- | ---: | ---: | ---: |
| Expert-action MSE | 0.0032307 | 0.0031296 | 0.0031383 |
| MSE reduction vs. frozen baseline | - | 3.13% | 2.86% |
| Predicted residual RMS | - | 0.01141 | 0.01182 |
| Maximum absolute residual | - | 0.04965 | 0.04971 |
| Maximum gripper residual | - | 0.0 | 0.0 |

The two trained actors differ by RMS 0.00552 (maximum absolute difference 0.04687),
so the imagination term materially changes the learned policy. However, it does not
improve this offline expert-action metric: the no-imagination actor is slightly better
(3.13% versus 2.86% MSE reduction).

Checkpoint hashes:

```text
no_imagination: bfd17a7318760e5e227b5f655e4a954932e2c0368ded163ccf956789122d7c49
imagination:    cfb59df6834d38a2ca43280bd09f95555a37d1f66bab803c1306b9dbc3887a21
```

The complete machine-readable audit is in
`evaluate_results/robotwin_residual_rl/robotwin_expert10_residual_iql_20260729/offline_audit_balanced.json`.

## Interpretation

The smoke test passes its narrow technical objective: with only 30 official success
episodes and the controlled failures, the residual actor learns a small correction
that moves FastWAM actions toward expert actions without changing gripper commands.

This result does **not** establish that residual IQL improves RoboTwin task success,
and it does **not** show an advantage from the imagination reward. The required next
comparison is an online, paired-seed evaluation of frozen FastWAM, residual IQL
without imagination, and residual IQL with imagination. The RoboTwin residual-policy
deployment adapter must be completed before that comparison can run.

## Reproduction

The full resumable pipeline is:

```bash
bash scripts/run_robotwin_expert_residual_iql_smoke.sh
```

It extracts the downloaded official demonstrations, exports aligned FastWAM/expert
transitions, builds the combined replay, and trains both matched IQL variants.

An early 5-epoch, task-only-balanced run was rejected because controlled failures
dominated the batches and zero-scale gripper dimensions were still affected by final
action clipping. The reported results above are from the corrected 20-epoch,
task-and-behavior-balanced rerun.
