# RoboTwin targeted oval-basket collection results (2026-09-06)

## Decision

The pre-registered collection stopped before training.  It found only two
strict-valid natural FastWAM failures, below the required five-pair threshold.
No replay merge, adapter training, development-gate evaluation, or final-test
evaluation was started.  The shortfall is retained as a negative result rather
than extending the seed range or weakening the selection criteria after seeing
outcomes.

## Formal collection result

The frozen protocol is documented in
`ROBOTWIN_TARGETED_OVAL_DATA_PROTOCOL_20260906.md`.  Candidate seeds were
examined once in the registered range `4800400..4800479`.  A scene was eligible
before policy execution only when it satisfied all of the following:

- task `place_can_basket`, configuration `demo_clean`;
- target `basket_id=0`;
- actual manipulation arm alternating left/right;
- exact successful expert planning and deterministic replay.

The result was:

| Quantity | Result |
|---|---:|
| Eligible expert states | 12 |
| Left/right arm coverage | 6 / 6 |
| Exact expert/FastWAM initial-state audits | 12 / 12 |
| Released FastWAM successes | 10 / 12 |
| Natural FastWAM failures | 2 / 12 |
| Required failures for training | 5 |

The two natural failures were selected mechanically in candidate order, without
reading any imagination-reward value:

- episode 2, seed `4800408`, left arm;
- episode 7, seed `4800429`, right arm.

The observed natural-failure rate was `16.7%`.  Therefore `basket_id=0` plus arm
choice is not a sufficient description of the prior residual regressions.  A
finer factor such as object/basket pose, approach geometry, or placement/release
timing is likely required.

## Exploratory reward audit (not a training gate)

After the formal collection stopped, the head-camera Wan-VAE reward was
computed on all two available failures.  This audit is explicitly exploratory:
it does not replace the registered five-pair gate and cannot authorize training.

| Seed | Arm | Expert reward | FastWAM-failure reward | Margin |
|---:|---|---:|---:|---:|
| 4800408 | left | 0.479995 | 0.376601 | +0.103394 |
| 4800429 | right | 0.491311 | 0.414663 | +0.076648 |

The reward ranked both pairs correctly (`2/2`, mean margin `+0.090021`).  This
supports only the narrow statement that the reward direction generalizes to
these two newly collected failures.  It does not establish that AWR will assign
the right chunk-level credit or improve online success.

## What this rules out and what comes next

This run does not support repeatedly adding generic oval-basket examples.  The
next data hypothesis must first distinguish the two failures from the ten
successes using information available before reward computation, preferably:

1. initial can/basket relative pose and actual arm;
2. first-chunk FastWAM action and approach geometry;
3. the first placement/release divergence measured from simulator state.

Only after a reproducible failure stratum is identified should a new candidate
pool and sample count be registered.  The existing two pairs may be retained as
diagnostic data, but they must not be promoted into the planned five-pair
training batch post hoc.

## Authoritative artifacts

- Collection root:
  `evaluate_results/robotwin_imagination_restart/robotwin_place_can_targeted_oval_pairs_20260906`
- Strict initial-state audit:
  `evaluate_results/robotwin_imagination_restart/robotwin_place_can_targeted_oval_pairs_20260906/strict_pair_audit.json`
- Mechanical two-pair selection audit:
  `evaluate_results/robotwin_imagination_restart/robotwin_place_can_targeted_oval_pairs_20260906/selection_audit_available2.json`
- Exploratory reward result:
  `evaluate_results/robotwin_imagination_restart/robotwin_targeted_oval_available2_reward_audit_20260906/wan_vae_head_pair_rewards.json`
