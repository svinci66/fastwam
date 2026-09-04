# Place-can imagination-reward training-seed replication (2026-09-04)

## Question

Does the positive `place_can_basket` effect observed for the seed-42 AWR
training run reproduce after changing only the training seed?

## Controlled setup

- Training seeds: 42 and 43.
- One immutable replay for every run; replay manifest SHA-256:
  `f34fbbbfb5edd1cce8d7fa857a95e38e28b342b588d4394c602889d942cb4f50`.
- For each seed, the no-imagination and imagination runs have identical Actor
  and Critic initialization hashes. The only allowed configuration differences
  are the experiment name and imagination weight (`0` versus `0.25`).
- Both training seeds use three AWR epochs and the frozen FastWAM Video Expert
  representation.
- Evaluation reuses the same predeclared 15 expert-feasible environment seeds,
  frozen official instructions, and exact initial observations.
- No Q/OOD gate or other inference-time protection is active.

The Actor and Critic initialization hashes differ between seed 42 and seed 43,
confirming that this is a genuine independent training initialization rather
than a repeated checkpoint.

## Online results

| Training seed | No imagination | With imagination | Paired wins | Paired losses | Difference |
|---:|---:|---:|---:|---:|---:|
| 42 | 4/15 (26.7%) | 7/15 (46.7%) | 3 | 0 | +20.0 pp |
| 43 | 4/15 (26.7%) | 6/15 (40.0%) | 4 | 2 | +13.3 pp |
| **Pooled** | **8/30 (26.7%)** | **13/30 (43.3%)** | **7** | **2** | **+16.7 pp** |

The pooled exact one-sided sign-test probability for seven wins and two losses
is `0.08984375`. This is encouraging but does not cross a conventional 0.05
threshold. The pooled rows are also repeated evaluations of the same 15
environment states under two independently trained policy pairs, so they
should not be presented as 30 independent environment conditions.

All 60 videos across the two training-seed comparisons are present and
readable. The seed-43 logs contain no traceback, CUDA out-of-memory error, or
runtime error. Its training-pair audit and online initial-state/protocol audits
all pass exactly.

## Decision

The minimal replication criterion passes: the imagination treatment improves
online success for two different training initializations, while the replay,
evaluation states, and all non-reward settings remain fixed. This is stronger
evidence than the earlier single-seed result and makes it reasonable to move to
a second task or a broader task set.

It is not yet a paper-level statistical claim. Before reporting a robust
average effect, add seed 44 or evaluate a preregistered larger set, and report
results by training seed rather than treating all repeated environment states
as independent samples.

## Artifacts

- Seed-42 audit:
  `evaluate_results/robotwin_imagination_restart/robotwin_video_expert_place_can_basket_fresh15_seed42_20260904/heldout_audit.json`
- Seed-43 training audit:
  `evaluate_results/robotwin_imagination_restart/robotwin_video_expert_place_can_basket_single_task_epoch003_seed43_20260904/training/paired_training_audit.json`
- Seed-43 online audit:
  `evaluate_results/robotwin_imagination_restart/robotwin_video_expert_place_can_basket_single_task_epoch003_seed43_20260904/heldout_audit.json`
- Seed-43 entry point:
  `scripts/run_robotwin_video_expert_place_can_basket_seed43_fresh15.sh`
