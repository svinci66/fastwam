# BC-RNN Residual Data Smoke Results

## Online trajectory export

- Base checkpoint: BC-RNN epoch 275
- Episodes: 5 (4 train, 1 validation)
- Successful episodes: 4 (the small-sample rate is not a baseline estimate)
- Branchable states: 141
- Stored low-dimensional observations: complete
- Incremental HDF5 and state-index writes: complete

All five trajectories were replayed from their initial simulator states. Every stored
simulator state matched exactly (`L-infinity error = 0`), and all success/failure
outcomes reproduced. The new trajectories are therefore suitable for exact-state
counterfactual branching, unlike the older external expert demonstrations.

## Symmetric branch smoke

- States: 6 (4 train, 2 validation)
- Candidates per state: 8, arranged as four positive/negative direction pairs
- States with an improving candidate: 4 / 6
- Decisive symmetric pairs: 18 / 24
- Candidate branches that diverged from the base: 48 / 48
- Restore and branch-initial-state error: 0
- Success outcome changes: 0 (expected for this very small 20-step smoke)

The quality gate passed. This establishes that short counterfactual perturbations around
the BC-RNN policy's actual online actions provide a finite, deterministic, and rankable
learning signal for the new residual/Q/OOD pipeline.

## Next stage

Collect 40 closed-loop BC-RNN episodes, then sample 300 training and 100 validation
states for symmetric branching. The stage is resumable after interruption. Its output
will be used to render wrist observations and retrain the deployable residual actor,
Q ensemble, and OOD support model.
