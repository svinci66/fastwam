# Place-can discordant-seed physics audit (2026-09-04)

## Question

Do the five `place_can_basket` outcome flips observed in the held-out-20
evaluation reproduce under exact seed/instruction pairing, and do the physics
traces show a plausible manipulation-stage difference rather than an invalid
simulation?

The five seeds were selected because the no-imagination and imagination
policies disagreed in the prior evaluation. They are therefore diagnostic
examples, not an unbiased success-rate estimate.

## Protocol

- Variants: frozen FastWAM baseline, residual without imagination reward, and
  residual trained with imagination reward weight `0.25`.
- Same five seeds and frozen official instructions for all variants.
- Paper-aligned FastWAM inference: 10 denoising steps and 24 executed actions
  per replan.
- No Q gate, OOD support gate, circuit breaker, intervention limit, or soft
  scaling.
- Per-action can pose, velocity, contacts, action, and anomaly flags recorded
  without modifying the environment state.

## Results

| Seed | FastWAM | No imagination | With imagination | Residual comparison |
|---:|:---:|:---:|:---:|:---|
| 4800104 | fail | success | fail | imagination loss |
| 4800107 | success | success | fail | imagination loss |
| 4800108 | fail | fail | success | imagination win |
| 4800117 | fail | fail | success | imagination win |
| 4800124 | success | fail | success | imagination win |
| **Total** | **2/5** | **2/5** | **3/5** | **3 wins / 2 losses** |

All five residual outcomes exactly reproduced the corresponding subset of the
held-out-20 run. Initial observation hashes, seeds, instructions, and initial
can poses matched across variants. All 15 videos were present and readable;
the evaluation logs contained no traceback, CUDA out-of-memory error, or
runtime error.

## Mechanistic reading

- On seeds `4800117` and `4800124`, the no-imagination policy lost gripper-can
  contact near actions 81--82 and never produced lift evidence. The imagination
  policy maintained contact, lifted the can, contacted the basket, and
  succeeded.
- On seed `4800108`, both residual policies lifted and reached the basket. The
  no-imagination trajectory subsequently sent the can to the ground, whereas
  the imagination trajectory completed normally.
- On seeds `4800104` and `4800107`, the reverse failure mode appeared: the
  imagination policy lost gripper-can contact early and failed to lift or
  place the can, while the no-imagination policy succeeded.
- The residual gripper dimensions remained exactly zero. These differences are
  caused by residual arm motion changing grasp retention and placement, not by
  directly commanding the gripper to open or close.
- Large-motion anomaly flags occur after failed placements in several traces
  and are consistent with policy-induced object ejection/falling. There is no
  NaN or evidence of spontaneous simulator corruption in the audited runs.

## Conclusion and next decision

The experiment supports a narrow claim: the imagination reward causes a real,
reproducible change in manipulation behavior and produced a net `+1/5` outcome
on this deliberately disagreement-enriched set. It does not yet establish a
general success-rate improvement because the set is selected and small.

The next experiment should hold the task, replay, architecture, epoch count,
and reward weight fixed; train no-imagination and imagination variants under
three training seeds; and evaluate every pair on a fresh, predeclared seed
manifest. Proceed to more tasks only if the imagination treatment has a
positive paired effect for most training seeds and does not introduce a
recurrent grasp-loss failure. If that condition fails, revise the reward or
training objective before collecting more tasks.

## Reproduction

```bash
bash scripts/run_robotwin_place_can_discordant5_physics_audit.sh
```

Structured output is written to
`evaluate_results/robotwin_imagination_restart/robotwin_place_can_discordant5_three_way_physics_audit_20260904/physics_audit_summary.json`.
