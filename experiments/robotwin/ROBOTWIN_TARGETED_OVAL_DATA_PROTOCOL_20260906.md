# RoboTwin targeted oval-basket data protocol (2026-09-06)

## Why this experiment exists

The balanced three-task replay contains 15 `place_can_basket` expert/policy
pairs.  Reset metadata shows the following coverage:

- `basket_id=1`: 12 pairs.
- `basket_id=0`: 3 pairs.
- `basket_id=0, arm_tag=left`: 1 pair.
- `basket_id=0, arm_tag=right`: 2 pairs.

Both protected development regressions (environment seeds `4800282` and
`4800283`) are `basket_id=0, arm_tag=left`.  This experiment tests the specific
hypothesis that the spatial adapter regression is caused by missing support for
that reset-time scene stratum.  It is not, by itself, evidence that the
imagination reward is effective.

## Frozen collection rule

The machine-readable protocol is
`manifests/robotwin_place_can_basket_targeted_oval_pairs_20260906.json`.

1. Use fresh candidate seeds `4800400..4800479` in order.
2. Keep only `basket_id=0` expert-feasible scenes, alternating the actual
   manipulation arm left then right.
3. Evaluate the released FastWAM policy with 10 denoising steps and 24-action
   replanning, without artificial corruption.
4. Select the first five strict-pair-valid natural FastWAM failures.
5. Selection must not read imagination rewards or residual outcomes.
6. Stop after five pairs; do not extend the same collection because a learned
   variant loses an evaluation.

Seed `4800320` was used only for an end-to-end collector smoke test and is
excluded from formal collection.  The final-test candidate range
`4800500..4800559` was registered before training and must remain untouched
unless the development gate passes.

## Frozen comparison and rejection rule

Train two spatial adapters on exactly the same augmented replay, initialization
seed, sampler, optimizer, and three epochs:

- no-imagination adapter (`imagination_weight=0.0`);
- paired-rank imagination adapter (`imagination_weight=0.25`).

Keep the released FastWAM policy and the frozen ordinary residual as references.
Reject the current imagination candidate without collecting more data when any
of the following occurs:

- an offline exact-pair or trust-region audit fails;
- it breaks an originally successful protected development case that the
  no-imagination comparator passes;
- it does not improve over the no-imagination adapter on the complete frozen
  development gate.

After rejection, the next research action is to revise chunk-level temporal
credit assignment or the learning objective, not to select more favorable
samples from the same stratum.
