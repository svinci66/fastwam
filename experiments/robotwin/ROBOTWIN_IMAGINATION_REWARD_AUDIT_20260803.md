# RoboTwin imagination-reward audit (2026-08-03)

This audit uses the cleaned 4-task paired-v2 collection. The input passed the
strict initial-state audit for all 60 behavior-policy pairs, including a
single-hash check across every retained transition in each episode.

## Data and result

- Transitions: 2,409 total, 2,329 alignment-valid.
- Episodes: 80 total; 20 complete policy/mild/strong paired trials.
- Temporal record integrity: 1.000.
- Initial-state match fraction: 1.000.
- Policy > mild hold > strong hold: 0.850.
- Correct goal reward > shuffled goal reward: 0.754.
- Policy reward > gripper-delay reward: 0.550.
- Mean reward for successful episodes: +0.03755.
- Mean reward for failed episodes: +0.00781.
- Clip saturation fraction: 0.0275.
- All seven configured validation gates passed.

## Behavior summary

| Behavior | Successes | Success rate | Mean imagination reward |
|---|---:|---:|---:|
| policy | 11/20 | 0.55 | +0.03607 |
| hold 0.25 | 10/20 | 0.50 | +0.02227 |
| hold 0.75 | 6/20 | 0.30 | -0.00494 |
| gripper delay 24 | 10/20 | 0.50 | +0.03286 |

The result supports using imagination consistency as one reward component: it
tracks hold-corruption severity and separates successful from failed episodes.
The weaker 0.55 ordering against gripper delay also confirms that it should not
be the sole reward; action imitation and terminal task success remain necessary.

Local machine-readable outputs are stored in:

`evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_10step_failure_collection_4task5ep_paired_v2_20260731/reward_audit_after_pairing_cleanup_20260803/`

The directory contains `transition_rewards.jsonl`, `episode_rewards.jsonl`, and
`reward_audit_summary.json`.
