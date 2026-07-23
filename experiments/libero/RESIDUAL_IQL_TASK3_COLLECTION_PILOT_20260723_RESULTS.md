# Residual IQL task-3 collection pilot — 2026-07-23

## Decision

The formal stage-2 data-collection pilot passes. Task 3 now has a useful mixture of
successful and failed behavior rather than another success-only replay:

```text
episodes:             90
successes:            62
failures:             28
raw transitions:   2,925
valid transitions: 2,874
structural errors:      0
```

This is sufficient to continue the pre-registered collection over the remaining
LIBERO Goal tasks. It is not yet a training result: no replay was built and no IQL
checkpoint was trained from this partial task subset.

## Collection boundary

Output:

```text
/home/ubuntu/sj/fastwam/runs/
libero_goal_iql_stage2_formal_20260723
```

Training states:

```text
0..4 and 10..34
```

Held-out validation states `5..9` and final-test states `35..49` were not used.
Each of the 30 training states was collected with:

1. the frozen FastWAM policy;
2. independent Gaussian action noise with standard deviation `0.075`;
3. independent Gaussian action noise with standard deviation `0.15`.

Policy seeds are derived independently per task and state. The two noise behaviors
use distinct action-seed streams while retaining the same state-specific FastWAM
policy seed.

## Results by behavior

| Behavior | Success | Failure | Mean policy steps | Raw transitions | Valid transitions | Future PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Policy | 27/30 | 3 | 204.43 | 777 | 753 | 28.3301 |
| Noise 0.075 | 28/30 | 2 | 213.70 | 812 | 791 | 27.3195 |
| Noise 0.15 | 7/30 | 23 | 355.87 | 1,336 | 1,330 | 26.8825 |
| **Total** | **62/90** | **28** | — | **2,925** | **2,874** | — |

The `0.075` behavior slightly outperformed the unperturbed policy on these 30
states. Noise must therefore not be treated as an automatic negative label: a
perturbation can occasionally rescue a trajectory. IQL should rank the observed
outcomes through Q/value learning, using the actual completion, imitation, and
imagination-alignment rewards.

The `0.15` behavior supplies the intended hard coverage: 23 failures and much longer
trajectories. This addresses the main weakness of the earlier 1,573-transition
replay, whose overwhelmingly successful behavior left too little return variation
for the imagination term to affect task success.

## Audit

`audit_multitask_collection.py` passed with:

```text
expected task IDs:                 present
all 30 states in all 3 behaviors: present
missing states:                    none
metadata.json files:               2,925
rollout_arrays.npz files:          2,925
structural errors:                 none
distinct task language features:   1
aggregate executed residual RMS:   0.100598
```

The evaluator reported 2,874 transitions with valid imagined/observed alignment.
The 51 excluded transitions are terminal-boundary records where a complete aligned
future/observed pair is unavailable; the files themselves are structurally valid
and retained for provenance.

Disk usage for the pilot root is approximately `911 MiB`.

## Next stage

Resume the same collection root over tasks `0..9`. Task 3 will be skipped through
its completed stage markers, so the pilot is part of the formal dataset rather than
throwaway work. After all tasks are present:

1. run the full raw collection audit;
2. build one globally normalized, camera-specific replay over all tasks;
3. train matched no-imagination and imagination IQL variants with at least three
   seeds;
4. select only on validation states `5..9`;
5. use states `35..49` once for the final comparison.

