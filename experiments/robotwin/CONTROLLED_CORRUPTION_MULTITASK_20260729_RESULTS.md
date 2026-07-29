# RoboTwin controlled-corruption multi-task validation (2026-07-29)

## Purpose

Extend the successful `move_can_pot` controlled-corruption test to distinct
manipulation skills and test whether the imagination-reward ordering generalizes.

## Setup

- Tasks: `adjust_bottle`, `open_laptop`, and `stack_blocks_two`
- Behaviors: policy, 25% action-chunk hold, 75% action-chunk hold, and a 24-step
  first-gripper-close delay
- Five paired trials per task and behavior: 60 episodes total
- Same initial observation across all four behaviors in each paired trial
- FastWAM checkpoint: `robotwin_uncond_3cam_384.pt`
- Reward encoder: local SigLIP SO400M Patch14-384
- Reward normalization: task-balanced global statistics, computed separately per camera

## Task-level results

Success counts:

| Task | Policy | 25% hold | 75% hold | Gripper delay |
| --- | ---: | ---: | ---: | ---: |
| `adjust_bottle` | 5/5 | 5/5 | 2/5 | 5/5 |
| `open_laptop` | 5/5 | 4/5 | 1/5 | 5/5 |
| `stack_blocks_two` | 5/5 | 3/5 | 0/5 | 5/5 |
| **All tasks** | **15/15** | **12/15** | **3/15** | **15/15** |

Mean imagination reward:

| Task | Policy | 25% hold | 75% hold | Gripper delay |
| --- | ---: | ---: | ---: | ---: |
| `adjust_bottle` | 0.06757 | 0.03345 | -0.00819 | 0.06432 |
| `open_laptop` | 0.04509 | 0.02386 | -0.01007 | 0.04353 |
| `stack_blocks_two` | 0.05898 | 0.03694 | -0.00812 | 0.05906 |
| **All tasks** | **0.05722** | **0.03142** | **-0.00879** | **0.05564** |

The corruption masks confirm that all interventions were active:

| Task | Mild corrupted/total chunks | Strong corrupted/total chunks | Gripper-delay corrupted/total chunks |
| --- | ---: | ---: | ---: |
| `adjust_bottle` | 14/38 | 62/80 | 10/30 |
| `open_laptop` | 25/79 | 105/131 | 6/41 |
| `stack_blocks_two` | 32/111 | 132/170 | 9/64 |

The increasing effect of the same hold probability on longer tasks shows that a
fixed per-replan corruption probability is not an equal-strength intervention across
tasks. This is acceptable for this ordering test, but future perturbation calibration
should use the realized number or duration of corrupted chunks.

## Reward audit

- Transitions: 870 total, 815 temporally aligned
- Complete paired trials: 15
- Paired `policy > mild hold > strong hold`: 15/15 (100%)
- Per-task paired ordering: 5/5 for every task
- Correct imagined goal beats a shuffled goal: 78.90%
  - `adjust_bottle`: 77.78%
  - `open_laptop`: 80.81%
  - `stack_blocks_two`: 78.01%
- Mean reward for successful episodes: 0.04685
- Mean reward for failed episodes: -0.00506
- Initial-state match: 100%
- Temporal record integrity: 100%
- Reward clipping saturation: 1.23%
- All seven primary audit gates passed

RoboTwin rejected some invalid `adjust_bottle` initializations. Two trial slots
therefore resolved to the same valid initial image, so the 15 paired trials contain
14 distinct initial images. This does not affect within-trial matching but slightly
reduces the effective state diversity.

## Gripper-delay finding

The 24-step gripper delay remained successful in all 15 episodes. Its aggregate mean
reward was slightly below policy, but policy beat it in only 8/15 paired trials
(53.33%). The endpoint reward at `K=24` therefore does **not** reliably detect this
short, recoverable disturbance across tasks. This intervention should be treated as
a boundary result, not as evidence for reward ordering.

## Conclusion

The main hypothesis generalizes across these three tasks: the imagination reward
consistently ranks unmodified, mildly held, and strongly held behavior in the same
order as execution quality. The result is considerably stronger than the earlier
Gaussian-noise test because the perturbations produce a controlled success gradient
and the ordering holds independently on every task.

The result still supports imagination reward as one dense component rather than the
whole objective. Task-success reward remains necessary, and an action imitation or
residual regularization term is needed to constrain the learned residual actor.
Undiscounted reward sums can also favor long episodes, so training and reporting
should use per-step/discounted reward together with explicit success and efficiency
metrics.

## Reproduction

```bash
OUTPUT_NAME=robotwin_controlled_corruption_3task5ep_20260729 \
TASKS=adjust_bottle,open_laptop,stack_blocks_two \
EPISODES=5 \
bash scripts/run_robotwin_controlled_corruption_validation.sh
```

Raw local summary (gitignored):

`evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_controlled_corruption_3task5ep_20260729/reward_audit/reward_audit_summary.json`
