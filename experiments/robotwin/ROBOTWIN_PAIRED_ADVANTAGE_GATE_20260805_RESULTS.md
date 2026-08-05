# RoboTwin paired residual-advantage gate

## Change

The new gate directly classifies whether a candidate trajectory outperformed a clean FastWAM trajectory with the same task and environment seed. Candidate-success/baseline-failure episodes are positive; candidate-failure/baseline-success episodes are negative; equal outcomes are excluded from the primary label. Splits are made by complete environment seed rather than individual transitions.

The IQL residual actor remains frozen. Two action-conditioned classifiers were trained and attached to its checkpoint. Inference requires the conservative minimum probability from the two classifiers and then independently applies the existing OOD support check.

## Training and offline validation

- Replay: 2919 transitions over four tasks.
- Informative paired data: 24 episodes and 647 transitions.
- Training split: 18 episodes / 487 transitions.
- Held-out seed split: 6 episodes / 160 transitions.
- Held-out negative-transition false-positive rate at the calibrated threshold: 0%.
- Held-out positive-transition true-positive rate: 93.1%.
- Calibrated minimum probability threshold: 0.998478.
- Actor candidates approved offline before OOD support filtering: 560 / 2919 (19.2%).

The trained checkpoint is `evaluate_results/robotwin_residual_rl/robotwin_4task_paired_v2_iql_20260803/iql_20epoch_imagination_paired_gate_v1/checkpoint.pt`.

## Paper-aligned online result

Task: `hanging_mug`, five strictly paired expert-feasible seeds, official unseen instructions, 10 denoising steps, and 24 executed actions per replan.

| Policy | Successes | Success rate |
| --- | ---: | ---: |
| FastWAM baseline | 3 / 5 | 60% |
| FastWAM + paired advantage gate + OOD gate | 3 / 5 | 60% |

All five seeds, instructions, and initial-observation hashes matched exactly. Paired outcomes were 3 both-success and 2 both-failure, with zero improvements and zero regressions.

The paired classifier approved one of 119 online replans, but the independent OOD support gate rejected it. Therefore no residual action was executed. This first version successfully preserves baseline behavior but does not yet demonstrate task improvement. The limiting factor is now coverage of real residual candidates: expert/corruption pairs calibrate safety, but additional paired rollouts containing actual residual actions are required to learn when a useful residual should be authorized.

Summary: `evaluate_results/robotwin_residual_online/robotwin_hanging_mug_paired_advantage_paired5_20260805/summary.json`.
