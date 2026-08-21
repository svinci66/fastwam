# RoboMimic Expert Episode Replay Audit

## Purpose

Before expanding the deployable residual policy from short counterfactual branches to
complete online episodes, we audited whether the saved expert action sequence is a
reliable full-episode base policy in the current RoboSuite 1.5.1 environment.

## Setup

- Dataset: `datasets/robomimic_hf/v1.5/can/paired/low_dim_v15.hdf5`
- Split: all 20 validation demonstrations
- Initial state: restored from each demonstration's first simulator state
- Base behavior: replay every stored expert action open loop
- Success: environment task-success signal during the replay

## Result

- Stored successful demonstrations: 20 / 20
- Replayed successful episodes: 10 / 20 (50%)
- Initial-state restore error: exactly 0
- Maximum deviation from the next stored simulator state: 28.4711 (L-infinity)
- Failed replays: `demo_4`, `demo_28`, `demo_38`, `demo_44`, `demo_52`,
  `demo_80`, `demo_88`, `demo_98`, `demo_104`, and `demo_188`

The one-episode live SigLIP + residual + Q/OOD smoke selected `demo_38`. Its
open-loop baseline and residual replay both failed, so its positive shaped-reward
delta is not evidence of an online task-success improvement.

## Decision

Stored expert actions are suitable for restored-state short-branch comparisons, but
they are not a reliable closed-loop full-episode base policy. We therefore stop the
open-loop online expansion here. The next valid experiment is:

1. train or load a RoboMimic BC-RNN base policy on the matching RoboSuite 1.5.1 data;
2. verify that this closed-loop policy succeeds online without the residual;
3. run the deployable residual actor and Q/OOD gate around the BC-RNN actions;
4. collect new counterfactual data around the actual BC-RNN state-action distribution.

This does not invalidate the 300-state fresh-paired short-branch validation. It only
prevents that result from being incorrectly promoted to a full-episode success claim
using a drifting open-loop controller.
