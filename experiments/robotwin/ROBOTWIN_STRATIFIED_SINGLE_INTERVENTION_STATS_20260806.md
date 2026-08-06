# RoboTwin Stratified Single-Intervention Statistics (2026-08-06)

## Protocol

- Task: `hanging_mug`
- Paper-aligned expert-feasible seeds: `4800001`, `4800002`
- Frozen FastWAM shadow baseline followed by independent, single-intervention forks
- Four candidates selected per seed across Q-approved, Q-rejected, and OOD-rejected strata
- Exact pre-intervention observation, proprioception, and FastWAM-action matching required
- Eight accepted pairs; zero quarantined pairs
- SigLIP bf16 scoring with one task-balanced global normalization, fitted separately per camera
- Local label threshold: absolute paired progress difference greater than `0.01`

## Aggregate result

| Label | Count |
|---|---:|
| `regression` | 2 |
| `local_improve` | 2 |
| `local_worse` | 1 |
| `neutral` | 3 |
| `rescue` | 0 |

Every shadow baseline succeeded (`8/8`).  The single-intervention branches
succeeded in `6/8`, so this batch measures retention/no-harm rather than rescue.
The descriptive paired success change is `100% -> 75%`; with only two
discordant pairs, this is not a statistically conclusive success-rate estimate.

## Per-pair result

| Seed | Replan | Shadow stratum | min Q advantage | Outcome label | Local progress delta |
|---:|---:|---|---:|---|---:|
| 4800001 | 0 | OOD rejected | -0.010754 | regression | +0.012242 |
| 4800001 | 7 | Q rejected | -0.004718 | neutral | +0.006862 |
| 4800001 | 8 | approved | +0.003416 | local improve | +0.013084 |
| 4800001 | 12 | approved | +0.050172 | neutral | -0.003829 |
| 4800002 | 1 | OOD rejected | -0.000737 | local worse | -0.018719 |
| 4800002 | 6 | Q rejected | -0.000444 | neutral | +0.003322 |
| 4800002 | 10 | approved | +0.003245 | regression | -0.018646 |
| 4800002 | 12 | Q rejected | -0.013385 | local improve | +0.019164 |

## Gate audit

| Shadow stratum | Pairs | Residual successes | Labels | Mean local delta |
|---|---:|---:|---|---:|
| approved | 3 | 2 | 1 improve, 1 neutral, 1 regression | -0.003130 |
| Q rejected | 3 | 3 | 1 improve, 2 neutral | +0.009783 |
| OOD rejected | 2 | 1 | 1 local worse, 1 regression | -0.003238 |

The min-Q-advantage/local-progress Pearson correlation is `-0.332` on eight
deliberately stratified samples.  This small, non-random sample cannot establish
a population correlation, but it does falsify the stronger claim that the
current Q ordering is already reliable: a Q-approved action regressed, while a
strongly Q-rejected action locally improved without losing terminal success.

The two OOD-rejected candidates were both harmful under the registered labels.
This is directionally consistent with keeping the support/OOD circuit breaker,
although two samples are far too few to calibrate its threshold.

One regression had positive imagination progress (`+0.012242`).  Therefore the
imagination signal cannot be used as a standalone success/no-harm label; terminal
success must override local shaping in the paired supervision.

## Immediate implication

Do not retrain the actor or gate from these eight rows alone.  The next
collection should add baseline-failure states so that `rescue` is observable,
while retaining baseline-success states to measure regression.  Continue
stratified sampling rather than collecting only current-gate approvals.

Machine-readable outputs:

- `evaluate_results/robotwin_residual_pairs/robotwin_stratified_single_intervention_batch1_20260806/statistics_global_camera/statistics.json`
- `evaluate_results/robotwin_residual_pairs/robotwin_stratified_single_intervention_batch1_20260806/statistics_global_camera/scored_pairs.jsonl`
