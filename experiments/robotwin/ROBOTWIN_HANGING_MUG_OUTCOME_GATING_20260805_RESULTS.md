# RoboTwin hanging_mug outcome-confirmed residual evaluation

## Protocol

- Task: `hanging_mug`, `demo_clean`, five strictly paired expert-feasible seeds.
- Paper-aligned FastWAM inference: 10 denoising steps, 24 executed actions per replan, text CFG 1.0, official unseen instructions.
- Candidate: imagination IQL residual with conservative twin-Q gate, OOD support circuit breaker, confidence-weighted residual scaling, and one FastWAM-only re-anchor after a negative imagined outcome.
- Cumulative-risk penalty was disabled and there was no global intervention-count limit.

The baseline and candidate matched exactly on all five environment seeds, official instructions, and initial-observation hashes. A first recovery attempt used a different generated instruction and was excluded. The recovery manifest now records the original official instruction so interrupted runs can be resumed without changing the protocol.

## Result

| Policy | Successes | Success rate |
| --- | ---: | ---: |
| Paper-aligned FastWAM baseline | 3 / 5 | 60% |
| FastWAM + IQL residual + Q/OOD + outcome confirmation | 0 / 5 | 0% |

Paired outcomes: 0 improved, 3 regressed, 0 both successful, and 2 both failed.

Candidate diagnostics across the five episodes:

- 190 replans and 34 applied residual interventions (17.9%).
- 24 of 34 post-intervention imagination-progress measurements were non-negative (70.6%).
- Mean post-intervention imagination progress was +0.00528.
- Negative result confirmation forced 10 FastWAM-only re-anchor replans.
- Mean scale of applied residuals was 0.626; only 45.8% of replans were classified in-distribution.

## Conclusion

The implementation behaves as designed, but this configuration is rejected as a performance-improving policy. Result confirmation and soft scaling do not prevent the residual from destroying baseline successes. Most locally positive imagination-progress measurements coexist with zero episode successes, so short-horizon imagined visual progress is not sufficiently aligned with task completion to authorize repeated interventions by itself.

The valid logs are:

- Baseline: `evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_hanging_mug_qood_outcome_softscale_paired5_20260804_baseline/eval_hanging_mug_20260804_214323.log`
- Candidate episodes 0-3: `evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_hanging_mug_qood_outcome_softscale_paired5_20260804_imagination/eval_hanging_mug_20260804_214848.log`
- Candidate episode 4 exact recovery: `evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_hanging_mug_qood_outcome_softscale_paired5_20260804_episode4_exact_imagination/eval_hanging_mug_20260805_103913.log`
