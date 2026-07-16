# LIBERO imagination-reward smoke test

This test does not train or update FastWAM. It saves lossless, time-aligned triplets:

```text
current observation at t
predicted goal at t+8
actual observation at t+8
```

It then uses a separately loaded, frozen SigLIP vision encoder to compute:

```text
imagination_progress =
    cosine_distance(current, predicted_goal)
  - cosine_distance(actual, predicted_goal)
```

## 1. Collect one policy episode

Run from the repository root. The local LIBERO initialization files and FastWAM checkpoint are trusted inputs, so the command uses PyTorch's compatibility switch for pre-2.6 serialized files.

```bash
conda run -n fastwam env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  NUMBA_DISABLE_JIT=1 \
  MPLCONFIGDIR=/tmp/fastwam-matplotlib \
  MUJOCO_GL=egl \
  DIFFSYNTH_MODEL_BASE_PATH=/home/ubuntu/sj/fastwam/checkpoints \
  python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/home/ubuntu/sj/fastwam/checkpoints/fastwam_release_clean/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=/home/ubuntu/sj/fastwam/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_goal \
  EVALUATION.task_id=3 \
  EVALUATION.num_trials=1 \
  EVALUATION.replan_steps=8 \
  EVALUATION.visualize_future_video=true \
  EVALUATION.save_imagination_transitions=true \
  EVALUATION.imagination_use_direct_action=true \
  EVALUATION.action_mode=policy \
  EVALUATION.output_dir=./evaluate_results/imagination_policy \
  seed=42
```

For the stationary control, change only:

```text
EVALUATION.action_mode=zero
EVALUATION.output_dir=./evaluate_results/imagination_zero
```

The optional noisy control uses `EVALUATION.action_mode=noise` and `EVALUATION.action_noise_std=0.15`.

## 2. Score the saved transitions

The encoder path must be a local Hugging Face-compatible SigLIP vision checkpoint. Repeat `--input-dir` to compare policy and control runs in one summary.

```bash
conda run -n fastwam python experiments/libero/analyze_imagination_rewards.py \
  --input-dir ./evaluate_results/imagination_policy/libero_goal/imagination_transitions \
  --input-dir ./evaluate_results/imagination_zero/libero_goal/imagination_transitions \
  --encoder-path /path/to/local/siglip-checkpoint \
  --output-dir ./evaluate_results/imagination_analysis \
  --device cuda \
  --batch-size 16
```

The output contains one JSONL record per valid `t -> t+8` transition and one aggregate JSON summary. The cyclic next-transition wrong-goal number is only a smoke diagnostic; adjacent goals can be semantically similar, so it is not a formal matched-wrong-goal test.

The first decision is deliberately small: policy progress should be clearly larger than the zero-action control. A single episode only verifies the pipeline and does not establish statistical effectiveness.
