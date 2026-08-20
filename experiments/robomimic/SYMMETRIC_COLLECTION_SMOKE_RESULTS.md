# RoboMimic symmetric local-action collection smoke

## Purpose

The first residual actor failed its online-readiness gate because the previous
dataset contained only 1.20 sampled candidates per unique state. This
collector measures a local action neighborhood instead of adding more unrelated
full episodes.

For every selected state, it generates four deterministic orthogonal directions
in the first three pose actions. Each direction is evaluated at `+0.1` and
`-0.1`, producing eight candidates. Gripper commands and the remaining action
tail are held fixed. Every branch restores the episode initial state and
replays the exact controller prefix before intervention.

## Smoke configuration

- 40 official training states;
- 10 official validation states;
- eight candidates per state (`4 x +/-`);
- intervention length: three actions;
- branch horizon: 20 actions;
- 450 total rollouts including one base branch per state;
- incremental commit every five states.

## Result

The smoke passed its structural and signal-quality gate:

- 50/50 states committed;
- all 23 HDF5 fields have length 50;
- action-candidate tensor shape `(50, 8, 3, 7)`;
- 40/50 states (80%) contain an improving candidate;
- 165/200 symmetric direction pairs (82.5%) produce a non-tied score;
- all 400 candidate branches diverge from the base final state;
- maximum restore error: zero;
- maximum base/candidate branch-start error: zero;
- all stored values are finite.

The smoke took 470 seconds, or 9.4 seconds per state. No candidate changed
terminal success in this small run; the useful signal is short-horizon shaped
progress, as intended for local direction estimation.

## Long collection

After the smoke passed, a persistent tmux run was started with 400 training and
100 validation states. At eight candidates per state, it will contain 4,000
candidate branches plus 500 base branches. Output is committed after every five
states and is resumable.

```bash
bash scripts/start_robomimic_symmetric_long.sh
```

The long-run output is
`evaluate_results/robomimic_symmetric_collection/can_symmetric_long_400train_100valid_seed20260820.hdf5`.
