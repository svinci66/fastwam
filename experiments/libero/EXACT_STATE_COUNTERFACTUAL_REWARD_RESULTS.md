# LIBERO exact-state counterfactual Reward V2 result

## Conclusion

This prospective seed-2042 experiment does **not** pass all predeclared gates, so
Reward V2 is not yet approved as an RL reward.

The result is nevertheless informative:

- Reward V2 reliably separates action-producing branches from a zero-action branch.
- It suppresses zero-action absolute reward to `3.80%` of policy reward, below the
  predeclared `5%` limit.
- It does not reliably rank progressively noisier versions of the same policy action
  at each anchor state. Only `6/10` anchor Spearman correlations are positive and the
  paired bootstrap interval includes zero.

In plain language, the current reward can tell that the robot moved in a broadly
useful direction, but it cannot yet consistently tell which of two similar moving
actions is better. That is insufficient for stable policy optimization.

## Final protocol

The experiment uses the first ten initial states of LIBERO Goal task 3:

```text
open the top drawer and put the bowl inside
```

For every anchor:

1. Wait 30 simulator steps.
2. Run FastWAM exactly once to obtain one direct action chunk and one imagined goal.
3. Keep the first `K_exec = 8` actions.
4. Draw one Gaussian noise direction and scale that same direction by `0.075`,
   `0.15`, and `0.30`.
5. Execute `policy`, the three noise levels, and `zero` from the same physical state.
6. Preserve the policy gripper action in all noisy branches and clip only non-gripper
   dimensions to `[-1, 1]`.
7. Render every current and final state eight times, average frozen SigLIP features,
   and compute Reward V2 separately for the agent and wrist cameras.

The primary reward was fixed before data collection:

```text
raw_dual = 0.5 * agent_delta_alignment + 0.5 * wrist_delta_alignment
```

The frozen historical `0.25 agent / 0.75 wrist` calibration is retained only as a
secondary diagnostic. It is not selected or tuned on seed 2042.

The 10 anchors are the independent statistical units. The 8 render repeats, 5
branches, and 400 total simulator action steps are not treated as independent samples.

## Why controller reset is necessary

A flattened MuJoCo state is not the complete LIBERO execution state. It omits robot
controller goals and interpolation buffers. Restoring only `sim_state` caused a
supposed zero branch to inherit controller state from an earlier moving branch.

Calling normal `env.reset()` is also unsuitable for counterfactuals. LIBERO defaults
to hard resets that reload the model, while soft resets still run placement samplers.
Both can change model-level geometry or fixture positions that are absent from the
flattened state.

The validated branch initialization therefore uses one unchanged MuJoCo model and:

```text
reset robot controller only
-> restore task initial state
-> replay the same 30 wait actions
-> restore the exact saved MuJoCo anchor state
-> execute one branch
```

Before scoring, the analyzer checks that every reconstructed branch anchor matches the
shared anchor in frozen feature space. The final run passed:

| Integrity check | Result | Limit |
| --- | ---: | ---: |
| Maximum anchor cosine distance | `2.75e-5` | `1.00e-4` |
| Maximum anchor feature L2 | `0.00740` | `0.01500` |
| Collection structure | 10 anchors / 50 branches / 400 steps | exact |

Earlier audit outputs made before this controller/model distinction are invalid and
are not used in the final statistics. The only final local result directory is:

```text
evaluate_results/exact_state_seed2042_final/
```

## Primary result

Mean raw equal-dual Reward V2:

| Branch | Mean reward |
| --- | ---: |
| Policy | `0.23146` |
| Noise `0.075` | `0.20943` |
| Noise `0.150` | `0.20707` |
| Noise `0.300` | `0.19187` |
| Zero | `0.00360` |

The population means decrease with noise, but the per-anchor ordering is not stable.
The high-noise branch exceeds policy in `3/10` anchors and exceeds the middle-noise
branch in `5/10` anchors.

## Predeclared gates

| Gate | Result | Pass |
| --- | ---: | :---: |
| Positive per-anchor Spearman | `6/10` (required `>= 8/10`) | No |
| Mean Spearman, paired bootstrap 95% CI | `0.18 [-0.32, 0.64]` | No |
| Policy reward greater than zero | `10/10` (required `>= 9/10`) | Yes |
| Policy minus zero, bootstrap 95% CI | `0.22786 [0.19456, 0.25862]` | Yes |
| Noise `0.150` greater than zero | `10/10` (required `>= 8/10`) | Yes |
| Noise `0.150` minus zero, bootstrap 95% CI | `0.20346 [0.16291, 0.24478]` | Yes |
| Mean absolute zero / policy reward | `3.80%` (required `<= 5%`) | Yes |

Five of the seven boolean checks pass. The two failed checks are the two parts of the
same monotonic-severity requirement. The formal decision is:

```text
does_not_yet_support_exact_state_reward_hypothesis
do_not_use_for_rl_yet
```

More precisely, the broad action-versus-imagination hypothesis receives partial
support, while the stronger claim required for RL—reward should consistently improve
as action quality improves at a fixed state—does not.

## Reproduction

Run the collector from the repository root. This performs 10 model inferences and
does not train or update FastWAM:

```bash
conda run -n fastwam env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  NUMBA_DISABLE_JIT=1 \
  MPLCONFIGDIR=/tmp/fastwam-matplotlib \
  MUJOCO_GL=egl \
  DIFFSYNTH_MODEL_BASE_PATH=/home/ubuntu/sj/fastwam/checkpoints \
  PYTHONPATH=/home/ubuntu/sj/LIBERO:$PWD/src:$PWD \
  python experiments/libero/collect_exact_state_counterfactuals.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/home/ubuntu/sj/fastwam/checkpoints/fastwam_release_clean/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_goal \
  EVALUATION.task_id=3 \
  EVALUATION.num_steps_wait=30 \
  EVALUATION.replan_steps=8 \
  EVALUATION.visualize_future_video=true \
  EVALUATION.imagination_use_direct_action=true \
  EVALUATION.output_dir=./evaluate_results/exact_state_seed2042_final \
  seed=2042
```

Then score the frozen images. The generated feature cache is reused unless
`--rebuild-feature-cache` is explicitly passed:

```bash
conda run -n fastwam env \
  PYTHONPATH=$PWD/src:$PWD \
  python experiments/libero/analyze_exact_state_counterfactuals.py \
  --manifest evaluate_results/exact_state_seed2042_final/manifest.json \
  --encoder-path /home/ubuntu/zhumj/code/dsrl/ckpt/siglip2-so400m-patch14-224 \
  --camera-calibration-json evaluate_results/imagination_reward_v2_offline/camera_weight_calibration.json \
  --output-dir evaluate_results/exact_state_seed2042_final/analysis \
  --device cuda \
  --batch-size 32 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 2042
```

The ignored local output contains `manifest.json`, lossless images, exact states,
actions, the shared noise direction, `analysis/features.npz`, per-branch reward rows,
and the final `analysis/summary.json`.
