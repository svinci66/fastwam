# Fixed natural-60 reward and AWR diagnostic

Status: authorized development diagnostic, frozen before new scores on 2026-09-07.
No simulator rollouts, parameter updates, task expansion, or weight selection.

## Population and provenance

Use all 20 policy episodes for each of open_microwave, hanging_mug, and
place_can_basket in `robotwin_wan_head_multitask4_smoke2_20260901`.
There are 18 successes and 42 failures. These are seen-instruction development
data, overlapping the training-source failures, not fresh paper-aligned evidence.
Freeze metadata, array, and image hashes in an inventory before encoding.

The primary horizon is the first 16/13/5 complete 24-action chunks respectively
(384/312/120 actions). Every episode reaches its task's horizon before termination.
Full-episode means are secondary and explicitly confounded by termination length.
Use only complete aligned chunks; report incomplete and degenerate records.

## Scores and controls

Reuse the existing Wan2.2 VAE single-image encoder, bf16, full 384x320 composite,
head latent region, flattened delta cosine, offsets 4/8/12/16/20/24 relative to 0.
Do not alter reward signs with outcomes. Encode on GPU with inference only.
Persist float32 head features by image SHA256 and model/encoder provenance.

Primary content control: for every actual chunk, average the score against every
OTHER episode's predicted delta trajectory in the SAME task and replan index.
The primary horizon has all 19 donors available. This avoids choosing one lucky
shuffle or using labels for matching. Full-horizon donor availability varies and
is diagnostic only. Secondary time control reverses the six predicted offsets
while keeping the initial reference fixed. Nuisance controls: real latent motion
RMS, executed joint-action variation RMS, baseline-action magnitude, and negative
episode duration (the last is an explicit leakage sentinel, never a reward).

## Statistics and decisions

Report per-task AUC and equal-task macro AUC, with ties worth 0.5. Use 10,000
within-task, outcome-stratified episode bootstrap replicates, seed 20260907.
Use the same draws for paired score contrasts. Recompute donor averages within
each bootstrap draw, excluding all copies of the target identity. Intervals are
percentile 95%, conditional on these three tasks and observed class counts.
Do not count chunks as independent episodes. No task or threshold is selected
after scores. Pooled AUC is descriptive only.

Promising development signal requires primary macro AUC >=0.65, CI lower >0.5,
matched-minus-donor macro AUC >=0.05, its CI lower >0, at least two tasks with
matched AUC >0.5, and no task with a supported reverse direction (CI upper <0.5).
This can authorize proposing a causal test, not training automatically.
If content delta CI upper <=0, reject identifiable incremental content on this
diagnostic; if matched AUC CI upper <0.65, exclude the registered useful primary
effect. Otherwise unmet gates are inconclusive. Never increase the fixed sample.
Prefix negatives do not reject possible late-stage rewards.

Only microwave has recorded simulator progress. Report per-episode rank
correlation with chunk progress and setback windows (ratio drop >=0.02).
These are setbacks, NOT certified first irreversible failures. For the other
tasks, state-based failure localization is unavailable; do not fabricate labels.

## AWR transmission

Use the original 2,934-transition balanced replay and frozen seed44 ordinary,
raw-weight025, paired-rank025, paired-rank010 checkpoints/replays.
Verify record identity, nonreward arrays, and file hashes. Recompute exact
reward components and action-step Monte Carlo returns using checkpoint configs
and explicit zero timeout bootstrap, matching the prior training convention.
Evaluate frozen final critics/actors on CPU. Recreate the three deterministic
training-epoch sampler index schedules, but evaluate them with FINAL checkpoints;
these weights are a frozen-checkpoint diagnostic, NOT historical training weights.
Report clipping, ESS, task/behavior/phase weight and loss mass, local action-output
gradient magnitude (not parameter gradients), residual target feasibility, and
return/value/advantage shifts. Cross-evaluate treatment returns with the control
critic to separate return changes from critic adaptation. No optimizer exists in
the diagnostic. Label-only ranking is not promoted to a main reward.
