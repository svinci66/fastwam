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

## Actor-aligned v4 follow-up

The original 2919-transition replay was merged with 447 transitions from
strictly paired `blocks_ranking_size` rollouts in which the learned residual
actor was actually executed.  The merged replay contains 3366 transitions over
five tasks.  For executed residual episodes only, an outcome tie is now a
non-improving negative; equal controlled-corruption outcomes remain excluded.
This supplies two actor-aligned improvement episodes and three actor-aligned
non-improvement episodes instead of exposing the gate to actor positives only.

The validation split contains seven complete episodes and 210 transitions.
The calibrated conservative threshold is 0.999242 with a maximum ensemble
disagreement of 0.000194.  It yields 0% transition false positives and 48.3%
true positives on the held-out split.  The calibration threshold uses the next
representable float32 value above the largest held-out negative rather than a
fixed 0.999 cap or an oversized additive margin.

The paper-aligned `hanging_mug` online evaluation was rerun as five independent,
restart-safe single-seed jobs after two machine interruptions.  Every seed,
official unseen instruction, and initial-observation hash exactly matches the
previous 3/5 FastWAM baseline.  The actor-aligned v4 policy also succeeds on
3/5 episodes, with 0 improvements and 0 regressions.  Across 119 replans, the
paired gate approves zero candidates and executes zero residual actions.

This is a safe but negative result: actor-aligned data from
`blocks_ranking_size` does not transfer enough intervention coverage to
`hanging_mug`.  The next data collection should therefore obtain strictly
paired, paper-aligned actual-residual outcomes on the target task, containing
both rescues and regressions, rather than relaxing the validated safety
threshold.

Checkpoint:
`evaluate_results/robotwin_residual_rl/robotwin_4task_paired_v2_iql_20260803/iql_20epoch_imagination_paired_gate_actor_aligned_v4/checkpoint.pt`.

Segment summaries:
`evaluate_results/robotwin_residual_online/robotwin_hanging_mug_paired_advantage_actor_aligned_v4_segmented_20260805_seed*/summary.json`.

## Target-task actor-aligned v5 follow-up

To test whether v4 was conservative because it had no target-task residual
outcomes, five strictly paired `hanging_mug` captures were collected with the
actual Q+OOD residual policy.  The baseline and residual runs used the same
fixed seed, official instruction, and initial observation for every pair.  The
outcomes were:

| Seed | FastWAM baseline | Actual Q+OOD residual | Pair label |
| --- | --- | --- | --- |
| 4800001 | failure | success | improvement |
| 4800002 | failure | success | improvement |
| 4800003 | success | failure | regression |
| 4800004 | success | failure | regression |
| 4800005 | success | failure | regression |

The ten captures contain 261 transitions (119 policy and 142 residual).  The
single-seed `trial_idx=0` values were relabeled to their real environment seeds
when building the replay, so paired-advantage grouping cannot accidentally
match different episodes.  The new shard was merged with the 3366-transition
actor-aligned replay, producing 3627 transitions.  The replay builder now
supports repeatable `--env-seed-override INPUT_DIR=SEED` arguments for this
restart-safe capture layout.

The v5 gate was trained with equal outcomes from actual residual episodes
treated as negative.  Its training split contains 26 episodes / 738
transitions; the held-out split contains 8 episodes / 248 transitions.  The
strict validation calibration gives 0% transition false positives, 41.4% true
positives, threshold `0.999329`, and maximum ensemble disagreement `0.000243`.

The paper-aligned v5 online evaluation still succeeds on 3/5 episodes, exactly
matching the frozen FastWAM baseline (0 improvements, 0 regressions).  It
approved only 1 of 119 replans and applied one residual intervention.  The
remaining target-task online predictions have ensemble disagreements around
0.89--0.99, far above the offline calibration range, so the OOD part of the
gate continues to reject them.  Thus adding five target episodes made the
classifier see both rescues and regressions, but did not yet provide enough
coverage for reliable online authorization; the result is safe but not an
improvement claim.

Checkpoint:
`evaluate_results/robotwin_residual_rl/robotwin_4task_paired_v2_iql_20260803/iql_20epoch_imagination_paired_gate_hanging_actual_v5/checkpoint.pt`.

Replay:
`evaluate_results/robotwin_residual_rl/robotwin_4task_paired_v2_iql_20260803/replay_with_actor_aligned_hanging_actual_pairs_20260805/`.

Segment summaries:
`evaluate_results/robotwin_residual_online/robotwin_hanging_mug_paired_advantage_hanging_actual_v5b_segmented_20260805_seed*/summary.json`.

## Deterministic preprocessing audit (2026-08-05)

Before using the five target-task pairs for another training run, the saved
`hanging_mug` seed-4800001 capture was replayed through the residual inference
inputs.  It contains 38 replan records and preserves the online initial-state
hash.  The audit found that the replay builder did not explicitly document the
online camera preprocessing and precision: the online path resizes each split
camera view to 224x224 before SigLIP, and the evaluation checkpoint uses bf16.
Those details matter because the paired gate is evaluated at a very high
threshold.

The replay path now performs the same 224x224 per-camera resize, records the
SigLIP precision in replay provenance, checks that precision when loading new
checkpoints, and retains per-camera normalization before feature fusion.  The
changes are pushed as commits `13fb76b` and `ed458c6`; 37 focused regression
tests pass.  Existing v5 checkpoints and replay shards were deliberately not
declared valid after this audit, because they were built before the precision
metadata was available.  A corrected full replay and a fresh gate/actor
training run must therefore precede any new success-rate claim.

The machine reboot also left the NVIDIA driver unavailable (`nvidia-smi`
cannot communicate with the driver).  CPU bf16 is suitable for a small code
path smoke check but is too slow for rebuilding the complete replay; the exact
bf16 rebuild is queued until CUDA is restored.  No threshold relaxation or
additional data collection was started during this audit.

## Imagination-reward sanity check (pre-correction diagnostic)

On the previously built five-pair shard, the reward is informative only when
normalized per replan.  The residual-minus-baseline mean reward was positive on
both rescue pairs (`+0.0210`, `+0.0171`) and negative on all three regression
pairs (`-0.0205`, `-0.0298`, `-0.0151`), giving the expected sign on 5/5 pairs.
Raw episode sums are not a valid substitute: failed episodes execute more
replans than successful episodes, so the sum is confounded by episode length.
This is a useful preliminary signal, not a final result—the shard was encoded
before the preprocessing/precision audit and must be recomputed on the
corrected replay before it is used as evidence for training.
