# RoboTwin high-failure residual rescue search (2026-08-06)

## Candidate tasks

The paper-aligned strict five-episode audit used official unseen instructions,
10 denoising steps, 24-step replanning, and exact seed/instruction pairing.
The matched FastWAM baseline failure rates were:

- `open_microwave`: 5/5 failures.
- `place_can_basket`: 4/5 failures.
- `hanging_mug`: 2/5 failures.
- `adjust_bottle`: 0/5 failures.

The next rescue-data collection should therefore prioritize
`open_microwave` and `place_can_basket`, while retaining `hanging_mug` as a
known task with a terminal rescue example.

## Confirmed terminal rescue example

The paper-aligned paired run
`robotwin_hanging_mug_qood_actual_pair_collection_20260805_seed4800001`
holds the environment seed and official instruction fixed. FastWAM failed and
the Q+OOD residual branch succeeded. The initial-observation hash matched
exactly across branches (`29ef4c4f...b5d81a4d`). This is the current clean
terminal rescue example, although it used the earlier canonical 20-epoch
residual actor and zero Q margin rather than the corrected actor.

## Corrected-actor search on `open_microwave`

Task/seed: `open_microwave`, seed `4800004`.

Protocol: paper-aligned strict pairing, official frozen instruction, 10
denoising steps, 24-step replanning, and the corrected paired-gate actor.

The Q+OOD shadow rollout failed and produced 62 residual candidates:

- 50 OOD-rejected candidates.
- 4 Q-rejected candidates.
- 8 other rejected candidates.
- 0 candidates approved by the complete gate.

Six phase-diverse candidates at replans `0, 11, 23, 34, 46, 60` were forced
one at a time. All six branches and the shadow baseline failed; all six pairs
passed exact pre-intervention alignment and none were quarantined.

Global per-camera-normalized SigLIP scoring found:

- 4 local improvements and 2 local regressions.
- Mean local progress delta: `+0.012664`.
- Replan 46 was the cleanest local positive: Q predicted positive advantage,
  the OOD gate rejected it, and all three camera progress deltas were positive.
- Replan 34 was a useful counterexample: Q predicted positive advantage, but
  local progress became worse.

A five-intervention window at replans `44-48` also failed. Therefore these
results support only the claim that the actor contains locally useful
corrections on a high-failure task; they do not establish a corrected-actor
terminal rescue on `open_microwave`.

## Artifacts

- Pair root: `evaluate_results/robotwin_residual_pairs/robotwin_open_microwave_rescue_search_seed4800004_20260806/`
- Statistics: `evaluate_results/robotwin_residual_pairs/robotwin_open_microwave_rescue_search_seed4800004_20260806/statistics/statistics.json`
- Scored pairs: `evaluate_results/robotwin_residual_pairs/robotwin_open_microwave_rescue_search_seed4800004_20260806/statistics/scored_pairs.jsonl`
- Window summary: `evaluate_results/robotwin_residual_online/robotwin_open_microwave_rescue_window44to48_seed4800004_20260806/summary.json`

## Decision

Do not expand the same single-intervention search blindly. Build the next
training/validation slice from (1) the confirmed `hanging_mug` rescue,
(2) locally positive and negative `open_microwave` pairs, and (3) new strict
paired failures from `place_can_basket`. The next online check should evaluate
short, phase-aware residual windows learned from those labels rather than
forcing every residual or merely relaxing the OOD threshold.
