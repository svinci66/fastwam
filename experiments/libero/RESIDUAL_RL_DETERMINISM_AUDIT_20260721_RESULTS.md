# LIBERO residual-policy determinism audit (2026-07-21)

## Decision

The paired-evaluation instability was reproduced, localized, and removed from the
strict audit path.

The first divergent value was the frozen SigLIP observation feature. For the first
audited episode, all three fresh processes received byte-identical initial states,
agent images, wrist images, proprioception, and FastWAM baseline action chunks, but
produced different SigLIP feature hashes at replan zero. The residual action and
executed action therefore diverged immediately.

Transformers 4.49 automatically selected `SiglipSdpaAttention`. PyTorch had enabled
the fused CUDA Flash and memory-efficient scaled-dot-product-attention backends.
`torch.use_deterministic_algorithms(True)` alone did not make those SigLIP outputs
bitwise repeatable across fresh processes.

Strict audit mode now disables both fused SDPA backends and enables the CUDA math
SDPA backend. Two fresh-process evaluations after this change produced exactly equal
per-replan audit records, including the final MuJoCo state:

```text
post-fix episode_action_audit equality: true
post-fix success vectors:               [true, false, true] == [true, false, true]
post-fix policy steps:                  [186, 400, 168] == [186, 400, 168]
```

This establishes a deterministic evaluation mode for the audited single-GPU,
single-environment protocol. It does not turn the preceding 10-trial result into
evidence that imagination reward improves success.

## Code under audit

```text
branch:
  feat/libero-residual-rl-mvp

audit instrumentation commit:
  f8af59d (feat: add LIBERO action determinism audit)

FastWAM checkpoint:
  /home/ubuntu/sj/fastwam/checkpoints/fastwam_release_clean/libero_uncond_2cam224.pt

residual checkpoint:
  /home/ubuntu/sj/fastwam/runs/local_residual_rl_corrected_4ebe877/
  train_with_imagination/checkpoint.pt

SigLIP encoder:
  /home/ubuntu/sj/fastwam/checkpoints/siglip-so400m-patch14-384-modelscope

SigLIP immutable version:
  google/siglip-so400m-patch14-384@
  sha256:ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a

output root:
  /home/ubuntu/sj/fastwam/runs/
  local_residual_determinism_audit_f8af59d_20260721_185055
```

## Frozen protocol

```text
suite/task:                 libero_goal / task 3
language:                   open the top drawer and put the bowl inside
explicit initial states:    [5, 8, 9]
seed:                       42
independent processes:      3 before the SDPA fix, 2 after the fix
action mode:                with-imagination residual actor
FastWAM action horizon:     32
executed/replan horizon K:  8
diffusion inference steps:  4
action ensembler:           disabled
future-video inference:     disabled
environment instances:     1
MuJoCo reset mode:          model-preserving soft reset
PyTorch deterministic:      enabled, warn_only=false
CUBLAS_WORKSPACE_CONFIG:    :4096:8
cuDNN benchmark:            false
cuDNN deterministic:        true
TF32 matmul:                false
```

The environment audit mode seeds Python and NumPy and sets
`env.env.hard_reset=false`. This matches the repository's exact-state
counterfactual collector: a hard reset reloads the XML and can change model-level
camera or geometry state that is not represented by LIBERO's flattened MuJoCo state.

Each episode records SHA-256 values for:

1. the requested LIBERO initial-state array;
2. every replan's agent image, wrist image, proprioception, FastWAM baseline action,
   SigLIP observation feature, residual action, corrected action, and executed action
   prefix; and
3. the final flattened MuJoCo state, success flag, and executed policy-step count.

The hash includes the NumPy dtype and shape as well as contiguous array bytes, so a
match means bitwise equality with matching representation rather than merely close
floating-point values.

## Pre-fix reproduction

The three original strict runs all completed successfully as processes. No strict
PyTorch deterministic-algorithm exception was raised, but their trajectories were
not bitwise repeatable.

| Run | Success | Trial 5 steps | Trial 8 steps | Trial 9 steps |
| --- | ---: | ---: | ---: | ---: |
| `repeat1` | `3/3` | `186` | `263` | `164` |
| `repeat2` | `3/3` | `186` | `255` | `163` |
| `repeat3` | `3/3` | `186` | `255` | `166` |

For trial 5, which is the first episode in every process, replan zero isolates the
first divergence without contamination from an earlier trajectory:

| Value at trial 5 / replan 0 | Run 1 hash prefix | Run 2 hash prefix | Run 3 hash prefix | Equal? |
| --- | --- | --- | --- | --- |
| Initial state | `d64a4ab0dd7c` | `d64a4ab0dd7c` | `d64a4ab0dd7c` | yes |
| Agent image | `5a88d494f0f0` | `5a88d494f0f0` | `5a88d494f0f0` | yes |
| Wrist image | `ba9877edbfc3` | `ba9877edbfc3` | `ba9877edbfc3` | yes |
| Proprioception | `2e78e8c34df5` | `2e78e8c34df5` | `2e78e8c34df5` | yes |
| FastWAM action chunk | `81f6372f8e55` | `81f6372f8e55` | `81f6372f8e55` | yes |
| SigLIP observation feature | `b04e0be66296` | `8525eba48347` | `670c2c2c07e6` | **no** |
| Residual action | `a4d2dc706e85` | `46dbca17d2a9` | `890059333403` | **no** |
| Corrected action | `f0894009fe1b` | `4b858c523d0a` | `ba95eb79c08e` | **no** |

This rules out the initial-state loader, the first rendered observations, and the
FastWAM action diffusion path as the first cause in this protocol. The residual actor
itself is a deterministic feed-forward network once its inputs are fixed; the
different observation-feature hashes place the first observed divergence in the
SigLIP encoder path.

Trial 8 and trial 9 sometimes already had different first observations by the third
process. They run after trial 5 in the same environment instance, so the earlier
different actions can change model state that is not reset by the flattened state.
That is a downstream cascade and is not used to identify the root cause.

## Fix and post-fix proof

When `EVALUATION.deterministic_algorithms=true`, the evaluator now additionally
configures:

```text
torch.backends.cuda.enable_flash_sdp(false)
torch.backends.cuda.enable_mem_efficient_sdp(false)
torch.backends.cuda.enable_math_sdp(true)
```

The selected backend state is written to every result JSON. Both post-fix results
reported:

```json
{
  "cuda_flash_sdp_enabled": false,
  "cuda_mem_efficient_sdp_enabled": false,
  "cuda_math_sdp_enabled": true
}
```

Post-fix outcomes:

| Run | Success | Trial 5 | Trial 8 | Trial 9 | Duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| `math_sdp_repeat1` | `2/3` | success / `186` | fail / `400` | success / `168` | `80.20 s` |
| `math_sdp_repeat2` | `2/3` | success / `186` | fail / `400` | success / `168` | `80.25 s` |

A complete Python equality comparison of the two `episode_action_audit` values
returned `true`. Therefore all recorded hashes matched at all 95 replans
(`24 + 50 + 21`), and all three final state hashes, success flags, and step counts
matched. The math-only attention path added about ten seconds to these three-episode
runs relative to the approximately 70-second fused-attention runs.

The post-fix `2/3` must not be interpreted as a measured degradation from the
pre-fix `3/3`. There are only three reused initial states, and changing a numerical
kernel changes low-order outputs and can send a contact-rich simulator down a
different trajectory. The valid conclusion is repeatability, not comparative task
performance.

## Verification

```text
py_compile:
  passed

unit tests:
  66 passed in 0.88 s

git diff --check:
  passed

online strict audits:
  5 independent evaluation processes, all exited with status 0
```

Robosuite emitted its known ignored `EGL_NOT_INITIALIZED` destructor warning after
writing the result files. It did not change any process exit code or result JSON.

## Result files

Relative to the output root:

```text
repeat1/libero_goal/gpu0_task3_results.json
repeat2/libero_goal/gpu0_task3_results.json
repeat3/libero_goal/gpu0_task3_results.json
math_sdp_repeat1/libero_goal/gpu0_task3_results.json
math_sdp_repeat2/libero_goal/gpu0_task3_results.json
logs/repeat1.log
logs/repeat2.log
logs/repeat3.log
logs/math_sdp_repeat1.log
logs/math_sdp_repeat2.log
```

All rollout videos were retained under each run's `libero_goal/videos` directory.

## Consequence for the reward experiment

The earlier observation that the imagination residual actor changed from `9/10` to
`7/10` under the same configured seed is now explained by an uncontrolled numerical
source in its SigLIP observation encoder. The old two totals remain valid records of
what ran, but they are not an acceptable deterministic A/B comparison.

The next effectiveness evaluation should run all policies under this strict audit
configuration and should use a validation set that was not used to choose reward
weights. The current audit proves that this evaluation path is repeatable; it does
not yet prove that `delta_alignment_v1` improves the policy.
