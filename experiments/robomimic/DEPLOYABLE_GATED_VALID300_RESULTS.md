# Deployable Q/OOD-gated residual: fresh-paired valid-300 result

## Protocol

- actor seed fixed before evaluation: `20260820`;
- 300 held-out states from trajectory-disjoint validation demonstrations;
- observation: frozen wrist-camera SigLIP feature plus 16-dimensional proprioception;
- three frozen deployable Q critics;
- conservative advantage: critic mean minus one ensemble standard deviation;
- Q threshold: 95th percentile on training zero-target states;
- OOD support: train-only KNN with `k=5` and the 95th-percentile threshold;
- residual component bound: 0.03, with the gripper component fixed to zero;
- score margin: 0.0001.

Every state was given a fresh baseline rollout in the same evaluation process.
An accepted residual was then rolled out from the exact same restored state. A
rejected proposal reused the fresh baseline result by definition. This avoids
confounding residual effects with cross-process replay drift.

## Result

- 77/300 proposals were accepted (25.67% intervention rate);
- among accepted proposals: 74 improved, one tied, and two worsened;
- accepted-branch mean score delta: +0.00066877;
- accepted-branch median score delta: +0.00075703;
- whole-set mean score delta: +0.00017165;
- all 223 rejected proposals had exactly zero executed delta;
- no success gains and no success losses occurred within the short branch horizon;
- maximum restore and branch-initial-state error: zero;
- maximum executed residual component: 0.02955;
- maximum gripper residual: zero.

All preregistered conditions passed: positive mean delta, improvements
outnumbering regressions, no success losses, deterministic restoration, bounded
residuals, and preserved gripper actions.

The two regressions are concentrated in one held-out trajectory:
`demo_98` at steps 73 and 76. Their score deltas were -0.00224905 and
-0.00229010. Both were in KNN support and had high conservative Q advantages,
so they are genuine local Q misrankings rather than OOD rejections or replay
noise. They must not be used to tune this validation threshold after the fact;
they should instead be carried into the next independent safety analysis.

## Superseded result

The earlier valid-300 file under
`evaluate_results/robomimic_deployable_gated_validation_valid300` compared new
rollouts against baseline scores saved during the older collection process.
One rejected state showed a large apparent regression despite executing no
residual. That result is superseded and must not be cited. The canonical local
artifact is:

`evaluate_results/robomimic_deployable_gated_fresh_pair_valid300/seed20260820_valid300.json`
