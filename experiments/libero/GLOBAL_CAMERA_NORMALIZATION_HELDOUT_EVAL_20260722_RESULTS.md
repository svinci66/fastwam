# Global Per-Camera Normalization Held-Out Evaluation — 2026-07-22

## Protocol

The globally normalized imagination-reward actor and its matched no-imagination
control were evaluated on all ten `libero_goal` tasks. Each task used held-out
initial-state indices 5 through 9, for 50 episodes per actor and 100 episodes total.

Both actors used the same frozen FastWAM checkpoint, SigLIP encoder, deterministic
settings, seed 42, four inference steps, and ordered initial states. The recorded
initial-state hashes match for all 50 actor pairs.

Code: `dc53405` on `feat/libero-residual-rl-mvp`.

Run directory:

```text
/home/ubuntu/sj/fastwam/runs/libero_goal_global_camera_norm_heldout_5to9_dc53405_20260722
```

## Results

| Task | Control successes | Normalized imagination successes | Control mean steps | Normalized mean steps |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5/5 | 5/5 | 118.6 | 119.6 |
| 1 | 5/5 | 5/5 | 85.0 | 85.0 |
| 2 | 5/5 | 5/5 | 83.6 | 83.8 |
| 3 | 5/5 | 5/5 | 196.6 | 184.8 |
| 4 | 5/5 | 5/5 | 83.0 | 83.0 |
| 5 | 5/5 | 5/5 | 131.2 | 133.2 |
| 6 | 5/5 | 4/5 | 99.8 | 151.0 |
| 7 | 5/5 | 5/5 | 72.6 | 72.8 |
| 8 | 5/5 | 5/5 | 72.4 | 72.2 |
| 9 | 5/5 | 5/5 | 131.4 | 132.2 |
| **Total** | **50/50 (100%)** | **49/50 (98%)** | **107.42** | **111.76** |

The normalized actor's only failure was task 6, initial-state index 6. It reached
the 400-step cap. The matched control succeeded on that state in 145 steps.

Wilson 95% intervals are approximately 92.9–100% for the 50/50 control and
89.5–99.6% for the 49/50 normalized actor. There is only one discordant pair, so
the observed two-percentage-point difference is not statistically significant.

Among the 49 state pairs where both actors succeeded, the normalized actor used
0.78 fewer steps on average and the paired median difference was zero. Therefore
there is no meaningful completion-efficiency difference after excluding the one
failure. When failures are capped at 400 steps, the normalized actor is 4.34 steps
slower on average because of that failure.

The normalized actor's mean online residual RMS was 0.008434, versus 0.007839 for
the control, an increase of 7.58%.

## Comparison with the previous reward

On the same held-out state set, the prior results were:

| Policy | Successes | Capped mean steps |
| --- | ---: | ---: |
| Frozen FastWAM baseline | 49/50 | 114.10 |
| Previous no-imagination actor | 49/50 | 110.10 |
| Previous clipped `delta_alignment_v1` actor | 49/50 | 110.66 |
| New normalized no-imagination control | 50/50 | 107.42 |
| New globally normalized imagination actor | 49/50 | 111.76 |

The new normalization fixes reward saturation and produces a valid training signal,
but its measured online success rate remains 98%, equal to the old imagination actor
and the frozen baseline on this state set. It does not outperform its matched control.

## Conclusion

The current evidence does not support a success-rate improvement from the globally
normalized imagination reward. The best interpretation is neutral: normalization
solves the reward-distribution defect without causing a statistically established
degradation, but the imagination term has not yet translated into better online
control. The single task-6/state-6 regression should be examined before increasing
the training or evaluation scale.
