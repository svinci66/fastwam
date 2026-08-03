# RoboTwin paired-v2 cleanup (2026-08-03)

Collection:
`evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_10step_failure_collection_4task5ep_paired_v2_20260731`

## Result

- Repaired 8 behavior episodes whose initial observation did not match the
  corresponding policy episode.
- Final exact-byte audit: **60/60 behavior-policy pairs passed**.
- Per task: `open_microwave` 15/15, `hanging_mug` 15/15,
  `place_can_basket` 15/15, and `adjust_bottle` 15/15.
- All 20 policy episodes and all 60 behavior episodes have terminal metadata.
- A follow-up full-transition audit quarantined 37 stale `replan_*` tail
  directories left by shorter replacement rollouts; every retained episode now
  contains exactly one initial-state hash across all of its transitions.
- No collection process remained after the audit.

The local machine-readable reports are stored under the collection's
`pairing_audit/` directory as `exact_pairing_audit.json` and
`exact_pairing_audit.md`.

## Root cause and correction

RoboTwin's upstream evaluator performs an expert motion-planning check before
policy evaluation. A nondeterministic planning failure advances `now_seed`, so
the same requested trial can silently use different scene seeds in different
corruption modes.

The FastWAM compatibility entrypoint now makes this expert check configurable.
Controlled corruption collection disables it, requires a fixed instruction,
records the accepted environment seed, quarantines the previous episode
directory before any retry, and validates exact initial-image pairing after
every successful subprocess. The RoboTwin upstream checkout is not modified.

The repair reused the formal run settings `BASE_SEED=45` and
`ACTION_CORRUPTION_SEED=20260731`. The final audit was run with:

```bash
python3 experiments/robotwin/audit_paired_initial_states.py \
  --input-dir evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_10step_failure_collection_4task5ep_paired_v2_20260731 \
  --output-json evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_10step_failure_collection_4task5ep_paired_v2_20260731/pairing_audit/exact_pairing_audit.json \
  --output-markdown evaluate_results/robotwin/robotwin_uncond_3cam_384/robotwin_10step_failure_collection_4task5ep_paired_v2_20260731/pairing_audit/exact_pairing_audit.md \
  --require-exact
```

Regression tests: 12 passed across the compatibility wrapper, pairing audit,
imagination-reward utilities, and online-summary tests.
