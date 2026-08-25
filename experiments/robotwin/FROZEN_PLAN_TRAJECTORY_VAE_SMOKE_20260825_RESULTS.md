# Frozen-plan trajectory reward: Wan VAE smoke result

Date: 2026-08-25

## Outcome

The trajectory-v2 collection and both fail-closed audits passed on the registered
`hanging_mug` seed `4800003`, intervention replan `3`.

- FastWAM setting: 10 denoising steps, 24-action chunks, unseen instruction.
- Branches: clean, bounded controlled corruption (`0.05`), and exact inverse correction.
- Saved offsets: `0, 4, 8, 12, 16, 20, 24`.
- All branches used byte-identical frozen predicted trajectories at the intervention.
- Clean and corrected executed the same actions and produced byte-identical actual trajectories.
- The corrupted branch changed both the action and actual visual trajectory.

The first Wan-native reward diagnostic encoded every saved composite frame with the
frozen Wan2.2 VAE, split the resulting `48 x 24 x 20` latent into the registered head,
left-wrist and right-wrist regions, computed change-direction cosine alignment at all
six future offsets, and averaged time within each camera and then the three cameras
equally.

| Branch | Matched frozen plan | Shuffled plan | Matched - shuffled |
|---|---:|---:|---:|
| clean | 0.4037 | -0.0789 | +0.4826 |
| corrupt | 0.3519 | -0.0079 | +0.3597 |
| corrected | 0.4037 | -0.0789 | +0.4826 |

All three cameras independently scored clean above corrupt. Clean and corrected
scores were exactly equal. No camera-specific scale or weight was fitted.

## Interpretation and stopping boundary

This smoke supports the local mechanism: a fixed FastWAM plan contains information
about the direction of the actual within-chunk visual change, and the multi-time Wan
VAE score decreases under a controlled action perturbation. It also rejects the
specific shuffled reference used in this smoke.

It does **not** yet prove that the reward predicts task success or improves a policy:
all three complete episodes succeeded and this is one seed. Residual training remains
blocked until the same ordering is validated on enough outcome-discordant pairs.

The next experiment keeps this implementation frozen and scores pre-registered
discordant controlled pairs. If matched plans do not consistently beat shuffled plans,
or successful/corrected branches are not ranked above failed/corrupted branches, the
project returns to feature/time alignment analysis instead of training an actor or
tuning camera weights.

## Artifacts

- `evaluate_results/robotwin_imagination_restart/robotwin_frozen_plan_trajectory_v2_smoke_20260825/action_triplet_audit.json`
- `evaluate_results/robotwin_imagination_restart/robotwin_frozen_plan_trajectory_v2_smoke_20260825/trajectory_triplet_audit.json`
- `evaluate_results/robotwin_imagination_restart/robotwin_frozen_plan_trajectory_v2_smoke_20260825/vae_reward_validation_replan3.json`
