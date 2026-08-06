# RoboTwin Single-Intervention Pair Smoke (2026-08-06)

## Purpose

Validate a causal-supervision data path before scaling collection.  Each pair
uses the same paper-aligned RoboTwin seed and official instruction.  The
residual branch may intervene at exactly one registered FastWAM replan; all
other replans execute the frozen FastWAM action.

## Protocol

- Task: `hanging_mug`
- Environment seed: `4800001`
- Instruction source: official unseen instruction, selected deterministically
- FastWAM inference steps: `10`
- Replan steps: `24`
- Intervention replan: `2`
- Maximum residual interventions: `1`
- Videos disabled; transition feedback capture enabled

The exporter accepted the pair only after all of the following matched at the
intervention boundary:

- initial observation hash;
- current observation hash;
- proprioception (`max_abs_difference = 0.0`);
- frozen FastWAM baseline action chunk (`max_abs_difference = 0.0`).

## Result

The pair passed every consistency check and no sample was quarantined.

| Branch | Episode outcome | Imagination progress at replan 2 |
|---|---:|---:|
| FastWAM baseline | success | `0.018976` |
| One residual intervention | failure | `0.047166` |

The exported label is `regression`.  The residual candidate RMS was
`0.003140`; despite a positive local imagination-progress difference of
`+0.028191`, it changed the final outcome from success to failure.

This single example is not a performance estimate.  It validates the paired
data pipeline and provides a concrete counterexample to using imagination
progress as the whole reward: local imagined-goal alignment must remain a
shaping term subordinate to terminal success and no-harm supervision.

## Artifacts

- Pair summary: `evaluate_results/robotwin_residual_pairs/robotwin_single_intervention_pairs_smoke_20260806/replan2/pairing_summary.json`
- Accepted pair: `evaluate_results/robotwin_residual_pairs/robotwin_single_intervention_pairs_smoke_20260806/replan2/accepted_pairs.jsonl`
- Reward audit: `evaluate_results/robotwin_residual_pairs/robotwin_single_intervention_pairs_smoke_20260806/replan2/reward_audit/`

## Next collection gate

Before retraining, collect intervention pairs across multiple seeds and
candidate strata, including current-gate approvals and rejections.  Keep
`rescue`, `regression`, `local_improve`, `local_worse`, and `neutral` labels;
exclude every pre-intervention mismatch from training.
