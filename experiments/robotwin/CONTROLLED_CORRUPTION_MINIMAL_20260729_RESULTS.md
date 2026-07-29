# RoboTwin controlled-corruption minimal validation (2026-07-29)

## Purpose

Test whether the current imagination reward ranks behavior quality on a task that
FastWAM can solve reliably, before training a residual policy.

## Setup

- Task: `move_can_pot`, `demo_clean`
- Instruction: `Pick up the can and place it beside the pot.`
- Checkpoint: `robotwin_uncond_3cam_384.pt`
- Five paired initial states per behavior (20 episodes total)
- Replan horizon: 24 environment steps
- Reward encoder: local SigLIP SO400M Patch14-384
- Reward normalization: global statistics with per-camera median/IQR
- Behaviors:
  - unmodified FastWAM policy
  - 25% probability of holding the non-gripper targets for one action chunk
  - 75% probability of holding the non-gripper targets for one action chunk
  - delay each gripper's first close command by 24 steps

All four behaviors used identical initial observations for every paired trial.

## Results

| Behavior | Success | Mean reward | Mean episode return |
| --- | ---: | ---: | ---: |
| Policy | 5/5 (100%) | 0.04569 | 0.26451 |
| Gripper close delay | 5/5 (100%) | 0.04078 | 0.25013 |
| 25% action hold | 4/5 (80%) | 0.02192 | 0.18492 |
| 75% action hold | 1/5 (20%) | -0.01272 | -0.20349 |

The action-corruption masks confirm that the intervention was active:

| Behavior | Corrupted action chunks | Total action chunks |
| --- | ---: | ---: |
| Policy | 0 | 33 |
| 25% action hold | 17 | 54 |
| 75% action hold | 65 | 84 |
| Gripper close delay | 10 | 36 |

## Reward audit

- Paired `policy > mild hold > strong hold`: 5/5 (100%)
- Paired `policy > gripper delay`: 4/5 (80%)
- Correct imagined goal beats a shuffled goal: 81.68%
- Mean transition reward, successful episodes: 0.03521
- Mean transition reward, failed episodes: -0.00997
- Valid temporal records: 100%
- Reward clipping saturation: 0%
- Initial-state pairing match: 100%
- All seven audit gates passed

## Conclusion

This minimal experiment supports the working hypothesis: the imagination reward is
useful as one dense component of the RL reward. It consistently ranks increasingly
damaged behavior below the original policy, and it separates successful from failed
episodes on average. The successful gripper-delay rollouts also show that the reward
can penalize a recoverable process disturbance without treating it as equivalent to
task failure.

The experiment does **not** establish that imagination reward should be the entire
RL objective. Residual-policy training should retain action-imitation/regularization
and environment task-success reward alongside this dense term. It also covers only
one task and five paired states, so task-level generalization remains to be tested.

## Reproduction

```bash
OUTPUT_NAME=robotwin_move_can_controlled_corruption_5ep_20260729 \
EPISODES=5 \
bash scripts/run_robotwin_controlled_corruption_validation.sh
```

Raw local summary (gitignored):

`evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_move_can_controlled_corruption_5ep_20260729/reward_audit/reward_audit_summary.json`
