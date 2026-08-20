# RoboMimic Phase 1: same-state counterfactual smoke test

## Purpose

This phase does not train or fine-tune FastWAM. It tests whether RoboMimic can
provide the data primitive that was missing from the earlier RoboTwin residual
experiments: two outcomes generated from the same recoverable simulator state.

The intended downstream sample is

`(state, base action, candidate action, base return, candidate return)`.

This supports a pairwise value target, such as
`Q(state, better action) > Q(state, worse action)`, without asking a regressor
to infer absolute success from unrelated episodes.

## Pinned runtime

- RoboMimic: `0.5.0` (`d309eaecc18acf4152a830a895a6984b8ac71b05`)
- RoboSuite: `1.5.1` (`a071383d53568ab798eb315c0e95357911be922d`)
- Local Python environment: `dexmimicgen`
- Dataset: RoboMimic v1.5 Can Paired, low-dimensional observations

The smoke script uses source checkouts through `PYTHONPATH`; it does not modify
the existing RoboTwin or LIBERO environments.

## Result on 2026-08-19

The dataset audit passed:

- 200 demonstrations and 19,795 transitions;
- 100 successful and 100 failed demonstrations;
- 100 consecutive success/failure pairs;
- all 100 pairs share the same initial state within `1e-10`;
- maximum measured initial-state error: `2.220446049250313e-16`;
- train/validation masks keep complete pairs together.

The downloaded Can MG sparse dataset was also checked and contains 3,900
rollouts (585,000 transitions): 718 successful and 3,182 failed. MG is a later
coverage pool, not an exact-pair substitute; its episodes must not be treated
as if they shared initial states.

The environment replay smoke on pair 0 also passed:

- restoring the saved state produced zero numerical error;
- replaying the same successful action sequence twice produced identical final
  states and rewards;
- the successful branch completed the Can task;
- the paired failed branch did not complete it;
- the two final states diverged (`L_inf = 7.212139450127829`).

The same replay check also passed on the last pair (pair 99), ruling out a
result that only works for the first HDF5 entry.

This establishes feasibility only. It does not yet show that a learned Q
function generalizes, that a residual improves FastWAM, or that the approach
transfers to RoboTwin.

## Reproduce

Download the two low-dimensional Phase-1 datasets:

```bash
bash scripts/download_robomimic_phase1_data.sh
```

Run the structural audit and exact-state replay:

```bash
bash scripts/run_robomimic_phase1_smoke.sh
```

Reports are written to `evaluate_results/robomimic_phase1/`.

## Next gate

Before training a larger model, build short-horizon branches from multiple
intermediate states and check three properties on held-out state groups:

1. the branch return gives a stable better/worse label;
2. a small pairwise Q model ranks held-out branches above chance;
3. the residual remains near zero when the candidate action has no measured
   advantage.

Only after these pass should the same collection mechanism be ported back to
RoboTwin and attached to frozen FastWAM features.

The held-out pairwise-Q gate has now passed. See
`experiments/robomimic/PAIRWISE_Q_PHASE1_RESULTS.md` for the three-seed result
and the action-only control.

## Short-branch collection

The next data-generation step is now available as an incremental, resumable
collector. MuJoCo state alone does not contain the OSC controller's internal
target history. Therefore, each branch starts by restoring the episode's exact
initial state and replaying the same action prefix. It then executes either the
recorded action or a bounded pose-action perturbation for the first three
steps. The remaining action tail is held fixed, so the measured score
difference is attributable to the intervention while controller history is
identical between branches.

Dense RoboSuite reward ranks short branches; task success receives a separate
large bonus and is stored explicitly. Pose actions are perturbed but the
gripper command is preserved by default. Samples retain their official
`train` or `valid` source split to prevent trajectory leakage during learning.

Run the 20-sample quality gate:

```bash
bash scripts/run_robomimic_counterfactual_collection.sh smoke
```

Start or resume the 5,000-sample collection in a persistent tmux session:

```bash
bash scripts/start_robomimic_counterfactual_long.sh
```

The quality gate additionally checks that successful source segments replay as
successful base branches. The long run writes a committed sample count and
summary every ten samples; an interrupted run resumes deterministically from
that count.
