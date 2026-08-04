# RoboTwin four-task residual-IQL online evaluation (2026-08-04)

## Setup

- Base policy: released FastWAM RoboTwin checkpoint.
- Residual policy: the five-epoch, language-conditioned IQL checkpoint trained with imagination reward.
- Variants: frozen FastWAM baseline, ungated residual, and the same residual behind dual-Q plus OOD support gates.
- Tasks: `adjust_bottle`, `hanging_mug`, `open_microwave`, and `place_can_basket`.
- Nominal evaluation size: five held-out episodes per task and variant, 60 episodes total.
- Environment seed range started at 4,800,000. RoboTwin expert-check skipped an invalid seed in two tasks, so both raw and common-seed results are reported.
- Interrupted runs were resumed from task-level completion markers. The final summarizer ignores incomplete logs and reconstructs each episode outcome from its accepted environment seed.

## Raw results

| Task | FastWAM | Ungated residual | Q+OOD residual | Q+OOD gate approval |
|---|---:|---:|---:|---:|
| `adjust_bottle` | 5/5 | 5/5 | 5/5 | 20.00% |
| `hanging_mug` | 2/5 | 0/5 | 3/5 | 7.63% |
| `open_microwave` | 0/5 | 1/5 | 0/5 | 0.00% |
| `place_can_basket` | 3/5 | 0/5 | 3/5 | 1.09% |
| **Overall** | **10/20 (50%)** | **6/20 (30%)** | **11/20 (55%)** | — |

The accepted seed sets match exactly for `adjust_bottle` and `place_can_basket`. They differ by one seed for `hanging_mug` and `open_microwave`, so raw success rates alone are not a fully paired comparison.

## Common-seed paired results

| Task | Common seeds | FastWAM | Ungated residual | Q+OOD residual |
|---|---:|---:|---:|---:|
| `adjust_bottle` | 5 | 5/5 | 5/5 | 5/5 |
| `hanging_mug` | 4 | 2/4 | 0/4 | 3/4 |
| `open_microwave` | 4 | 0/4 | 1/4 | 0/4 |
| `place_can_basket` | 5 | 3/5 | 0/5 | 3/5 |
| **Overall** | **18** | **10/18 (55.6%)** | **6/18 (33.3%)** | **11/18 (61.1%)** |

Paired episode transitions relative to FastWAM:

| Variant | Improved failures | Regressed successes | Both succeeded | Both failed |
|---|---:|---:|---:|---:|
| Ungated residual | 1 | 5 | 5 | 7 |
| Q+OOD residual | 1 | 0 | 10 | 7 |

## Interpretation

1. The learned residual is not safe when it is always applied. It fixes one common-seed failure but destroys five baseline successes, reducing the matched success rate from 55.6% to 33.3%.
2. Dual-Q and OOD support gating removes the observed regressions in this sample. It preserves all ten matched baseline successes and fixes one `hanging_mug` failure.
3. The gate is behaving conservatively rather than merely shrinking the action everywhere. It approves 20% of replans on `adjust_bottle`, 7.63% on `hanging_mug`, 1.09% on `place_can_basket`, and none on `open_microwave`.
4. The `open_microwave` support in-distribution rate is only 2.54%, and the final Q+OOD approval rate is zero. The gated policy therefore falls back to FastWAM and cannot reproduce the one success obtained by the unsafe ungated actor.
5. The result supports the deployment hypothesis—selective Q+OOD gating is necessary to retain FastWAM's existing ability—while providing only preliminary evidence of improvement. A one-episode matched gain is not statistically sufficient for a success-rate claim.
6. This run does not isolate the imagination reward. Both residual variants use the same imagination-trained actor; the difference between them is gating. A matched no-imagination checkpoint comparison is still required to attribute any gain specifically to imagination reward.

The machine-readable reconstruction, including every accepted seed and episode outcome, is stored at `evaluate_results/robotwin_residual_online/robotwin_4task_iql_full_heldout5_20260803_final_summary.json`.
