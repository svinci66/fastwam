# Paired-rank imagination reward: reward-credit repair and development results

Date: 2026-09-05

## Decision

The reward-credit bug is real and has been repaired at the replay/return level,
but the repaired reward does not yet improve the shared three-task residual
actor over the no-imagination control. Do not spend a fresh held-out set or
scale to more tasks yet.

The next optimization should preserve the successful no-imagination residual
as a frozen anchor and learn a small, zero-initialized, spatially aware
imagination adapter under an explicit trust region. Another global reward
weight sweep or more epochs is not the preferred next step.

## Diagnosed failure mechanism

The original reward passed episode-mean Wan-VAE ranking on 46/47 pairs, but its
exact AWR return targets failed a stricter audit:

- only 13/47 pairs had a nonshrinking expert-minus-failure return gap;
- `open_microwave`: 1/17 nonshrinking;
- `hanging_mug`: 7/15 nonshrinking;
- `place_can_basket`: 5/15 nonshrinking;
- for the only oval-basket training pair (`seed=619`), the failed policy
  received `0.15077` total imagination shaping while the expert received only
  `0.07055`; its expert-minus-failure initial return gap shrank by `0.00442`.

This proves that episode-average ranking was not faithfully preserved after
per-chunk normalization, differing episode lengths, action-step discounting,
and AWR return construction.

Two formal `place_can_basket` regressions (`4800282`, `4800283`) were then
reproduced deterministically. Both were caused by the residual actor during
replans 0--5: the can missed the oval basket during release, while gripper
residuals remained exactly zero. Bidirectional actor swaps closed the causal
loop:

- no-imagination actor with old imagination actor on replans 0--5: `0/2`;
- old imagination actor with no-imagination actor on replans 0--5: `2/2`.

## Implemented reward repair

New versioned reward type:

```text
wan_vae_head_trajectory_paired_rank_discount_norm_v1
```

For each expert/failure pair:

1. the positive episode-level Wan-VAE margin is converted to a bounded global
   confidence using the median positive margin across all selected tasks;
2. the expert episode receives a positive target imagination return and the
   failed policy episode receives the equal-magnitude negative target;
3. the target is divided by the exact action-step discount mass, rather than
   raw chunk count, so different episode lengths cannot change its initial
   return magnitude;
4. a nonpositive world-model margin is zeroed instead of being forced into a
   reversed label;
5. the old globally normalized reward remains available unchanged for exact
   reproduction.

The new reward-credit audit computes the exact control/treatment returns used
by AWR and rejects a candidate if per-task expert-minus-failure gaps shrink or
the required `place_can_basket/seed619` diagnostic gap does not strictly
increase.

## Offline gate results

The paired-rank replay contains the same 2,934 transitions and 47 behavior
pairs as the previous balanced replay.

At imagination weight `0.25`:

- all 47/47 pair gaps are nonshrinking;
- 46/46 originally positive pairs strictly improve;
- `place_can_basket`: 15/15 strictly improve;
- `seed=619`: gap `0.70233 -> 0.71197` (`+0.00964`).

At imagination weight `0.10`:

- all 47/47 pair gaps are nonshrinking;
- `seed=619`: gap `0.70233 -> 0.70619` (`+0.00386`).

The zero-weight control trained on the new replay is bitwise identical to the
previous no-imagination actor:

```text
actor SHA-256: fc674f28ac43a53a0a05f88d2ce60445b8e18f2286aebc0bcdb6c6b3bd0c61d5
```

This confirms that the replay schema change has no effect when imagination
weight is zero.

## Staged online results

### Two known oval-basket regressions

| Model | Seeds 4800282/4800283 |
|---|---:|
| No-imagination control | 2/2 |
| Old global reward, weight 0.25 | 0/2 |
| Paired-rank reward, weight 0.25 | 1/2 |
| Paired-rank reward, weight 0.10 | **2/2** |

The paired-rank repair reduced the early action-window distance from the
control by about 42--44% at weight 0.25 and recovered one regression. Reducing
the weight to 0.10 recovered both.

Single-chunk swaps on the remaining weight-0.25 failure showed that neither the
early prefix nor release chunk was independently invalid: either side could be
replaced with the control and recover success. Their combination crossed the
narrow placement tolerance, demonstrating accumulation of small closed-loop
corrections rather than one gross command or gripper error.

### Thirty-state development set

The already-observed formal states were deliberately reused as a development
set. They are no longer eligible as fresh held-out evidence.

| Model | Overall | open_microwave | hanging_mug | place_can_basket |
|---|---:|---:|---:|---:|
| FastWAM baseline | 11/30 | 2/10 | 3/10 | 6/10 |
| No-imagination residual | **12/30** | 2/10 | 2/10 | **8/10** |
| Old global reward 0.25 | 11/30 | **3/10** | 2/10 | 6/10 |
| Paired-rank reward 0.10 | 11/30 | 2/10 | 2/10 | 7/10 |

Paired-rank 0.10 versus the no-imagination control:

- `open_microwave`: 0 wins / 0 losses;
- `hanging_mug`: 0 wins / 0 losses;
- `place_can_basket`: 0 wins / 1 loss;
- overall: 0 wins / 1 loss.

It repaired the two selected oval-basket failures but introduced a different
failure on `seed=4800286`. A targeted check confirmed a weight trade-off:

- paired-rank weight 0.25 succeeds on `4800286`;
- paired-rank weight 0.10 fails on `4800286`;
- on the two oval-basket seeds, weight 0.25 scores 1/2 while weight 0.10 scores
  2/2.

No single tested global imagination weight dominates the no-imagination actor.

## What was learned

1. Correct episode ranking alone is insufficient; the exact AWR return gap
   must be audited after all reward transformations.
2. Fixing the sign and length bias is necessary: it reduces the diagnosed
   action drift and repairs selected regressions.
3. It is not sufficient for multi-task improvement: global weight 0.10 trades
   one corrected state for a new regression and loses the old
   `open_microwave` gain.
4. The actor observation remains a likely bottleneck. Initial
   `place_can_basket` Video Expert token-mean features have median pairwise
   cosine similarity `0.99934`, so fine basket geometry and release tolerance
   are poorly separated.
5. The full residual actor should not be moved globally by imagination reward
   when the no-imagination actor already solves 12/30 and preserves more
   `place_can_basket` states.

## Next optimization direction

Freeze the no-imagination residual actor and add a small zero-initialized
imagination adapter that predicts only a bounded residual-of-residual. Feed the
adapter spatial Video Expert patch/token features rather than the global token
mean, and regularize its output toward zero. This creates an explicit trust
region around the best current actor while allowing state-dependent corrections
where imagination evidence is discriminative.

The next minimum test should use the three diagnostic states only:

- oval baskets: `4800282`, `4800283`;
- perforated basket trade-off: `4800286`.

Required result before another 30-state run: `3/3`, no change in gripper
commands, and no persistent release-window joint bias relative to the frozen
control. If that fails, collect 5--10 targeted oval/perforated-basket pairs with
both explicit and implicit arm selection before changing actor capacity.

## Authoritative artifacts

- Legacy reward-credit audit:
  `evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_paired_rank_seed44_20260905/audits/legacy_credit_audit.json`
- Paired-rank 0.25 audit:
  `evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_paired_rank_seed44_20260905/audits/paired_rank_credit_audit.json`
- Paired-rank 0.10 audit:
  `evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_paired_rank_seed44_20260905/audits/paired_rank_weight010_credit_audit.json`
- New replay:
  `evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_paired_rank_seed44_20260905/replay`
- Training outputs:
  `evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_paired_rank_seed44_20260905/training/seed44`
- Weight-0.10 two-seed result:
  `evaluate_results/robotwin_residual_online/robotwin_place_can_basket_paired_rank_weight010_seed44_regression2_20260905/summary.json`
- Weight-0.10 development result:
  `evaluate_results/robotwin_residual_online/robotwin_video_expert_multitask3_paired_rank_weight010_seed44_dev10_20260905/summary.json`
- Weight-0.25 seed-4800286 result:
  `evaluate_results/robotwin_residual_online/robotwin_place_can_basket_paired_rank_weight025_seed4800286_20260905/summary.json`
