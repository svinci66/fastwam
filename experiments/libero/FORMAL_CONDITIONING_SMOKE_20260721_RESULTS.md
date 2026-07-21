# Formal residual-conditioning smoke test (2026-07-21)

This run verifies the local engineering path introduced in commit `aa759a0`:

```text
FastWAM rollout + frozen UMT5 task feature
  -> schema-v3 replay
  -> language/baseline-action-conditioned residual AWR
  -> v2 checkpoint
  -> strict online checkpoint load and one real two-camera correction
```

It uses only `libero_goal/task3`, one policy episode and one controlled-noise
episode. It is therefore a pipeline smoke test, not evidence for cross-task language
conditioning or for an imagination-reward improvement.

## Data and replay

- Git commit: `aa759a0d2994696fbb6dba7570cf88464924cd1a`
- GPU: NVIDIA GeForce RTX 5090 (32 GB)
- Task: `libero_goal/task3`
- Instruction: `open the top drawer and put the bowl inside`
- Policy collection: `0/1` success, 50 transitions
- Controlled noise (`std=0.075`): `1/1` success, 22 transitions
- Replay transitions: 72 (`policy=50`, `noise=22`)
- Replay schema: 3
- Observation feature: two-camera SigLIP, 2304 dimensions
- Language feature: frozen Wan UMT5-XXL masked mean, 4096 dimensions
- All 72 language arrays were finite and identical within the single task, as expected
- Replay manifest SHA-256: `52819efdebc3a59ae668aecea7fc4dbb58377fedb8bcde592d166ef675a59894`

Local output root:

```text
/home/ubuntu/sj/fastwam/runs/local_formal_conditioning_smoke_aa759a0_20260721_205425
```

## Training validation

The formal imagination config was used:

```text
configs/rl/libero_residual_awr_multitask.yaml
```

- Actor parameters: 2,731,192
- Critic parameters: 2,702,977
- Actor language input: 4096 dimensions, projected to 256
- Actor/critic baseline-action projection: 128 dimensions
- Executed residual RMS in replay: `0.03694955`
- Epochs: 20
- All recorded metrics: finite
- Actor loss: `0.00353136 -> 0.00236150`
- Critic loss: `5.36630535 -> 1.98638266`
- Mean masked action MSE: `0.00196163 -> 0.00148141`
- Checkpoint format: `fastwam_residual_awr_v2`
- Checkpoint SHA-256: `75b9b4b1e6423c2ce0fd3c30ae6ad42afff8a737dae1164857cf127586564c65`

The checkpoint was then loaded with strict SigLIP and UMT5 provenance checks. On a
saved real two-camera transition it produced a finite `(8, 7)` corrected action
chunk with residual RMS `0.03027575` and maximum absolute residual `0.07308116`.

## Conclusion

The new formal conditioning path runs locally end to end. The next experiment must
collect multiple tasks—initially all ten `libero_goal` tasks—because this single-task
run cannot test whether the policy actually uses language to distinguish tasks.
