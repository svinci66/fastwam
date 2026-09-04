# Balanced three-task Video Expert residual AWR: formal-10 results

Date: 2026-09-04

## Decision

Stop scaling the current three-task recipe. The head-camera Wan-VAE imagination
reward at weight `0.25` did not outperform the otherwise identical
no-imagination residual on the 30 fresh held-out states. It helped
`open_microwave`, was neutral relative to the control on `hanging_mug`, and
regressed `place_can_basket`.

This result rejects the current **training recipe**, not the reward signal in
isolation. The reward still ranked expert success above natural FastWAM failure
on 46/47 training pairs. The remaining problem is that this ranking signal did
not produce a uniformly better shared residual actor.

## Data and protocol audit

- Training reward pairs: `open_microwave=17`, `hanging_mug=15`,
  `place_can_basket=15` (`47` pairs total).
- Reward ranking: `46/47 = 97.87%` overall; the newly collected five
  `place_can_basket` pairs ranked correctly `5/5`.
- Replay: `2,934` action-chunk transitions from `94` separate expert/policy
  episodes.
- Actor observation: frozen FastWAM Video Expert feature, 3,072 dimensions.
- Language observation: frozen FastWAM UMT5 masked-mean feature.
- Reward observation: head-camera Wan2.2 VAE trajectory agreement with a
  frozen reference throughout each action chunk and offsets
  `[0, 4, 8, 12, 16, 20, 24]`.
- Reward normalization: one global episode-balanced median/IQR transform fit
  over all three selected tasks (`center=0.361653`, `scale=0.296635`).
- Control and treatment used the same replay, initialization, seed `44`, three
  AWR epochs, task-balanced sampling, architecture, and optimizer settings.
  Their only configuration difference was imagination weight `0` versus
  `0.25`.
- Formal evaluation used ten new expert-feasible states per task. The 30 seeds
  were disjoint across tasks and had zero overlap with the training replay.
- All three variants used the same seeds, initial-observation hashes, and
  official unseen instructions. Evaluation was paper-aligned with ten diffusion
  steps and 24 executed actions per replan. Q/OOD gates and other legacy gating
  modules were disabled.

## Smoke result

The two-state-per-task smoke test verified deployment and exact pairing:

| Variant | Overall | open_microwave | hanging_mug | place_can_basket |
|---|---:|---:|---:|---:|
| FastWAM baseline | 0/6 | 0/2 | 0/2 | 0/2 |
| Residual, no imagination | 0/6 | 0/2 | 0/2 | 0/2 |
| Residual, imagination 0.25 | 2/6 | 0/2 | 2/2 | 0/2 |

This was a positive smoke signal but was not treated as efficacy evidence
because the sample was very small and five of six task/state combinations had
no outcome variation.

## Formal result

| Variant | Overall | open_microwave | hanging_mug | place_can_basket |
|---|---:|---:|---:|---:|
| FastWAM baseline | 11/30 (36.7%) | 2/10 | 3/10 | 6/10 |
| Residual, no imagination | 12/30 (40.0%) | 2/10 | 2/10 | 8/10 |
| Residual, imagination 0.25 | 11/30 (36.7%) | 3/10 | 2/10 | 6/10 |

Paired treatment-versus-control outcomes:

| Task | Treatment wins | Treatment losses | Interpretation |
|---|---:|---:|---|
| open_microwave | 1 | 0 | Positive, but only one discordant pair |
| hanging_mug | 0 | 0 | Identical to no-imagination control |
| place_can_basket | 0 | 2 | Rejected; treatment regressed two control successes |
| **Overall** | **1** | **2** | Net negative versus the control |

Paired comparisons to the frozen FastWAM baseline:

- No-imagination residual: two improvements and one regression overall, for a
  net success change of `+1`.
- Imagination residual: three improvements and three regressions overall, for a
  net success change of `0`.
- On `place_can_basket`, the no-imagination residual improved two baseline
  failures with no regressions. The imagination residual also improved two
  baseline failures but regressed two different baseline successes.

## Stop rule

Do not expand this exact recipe to more tasks, seeds, epochs, or server-scale
training unless both conditions hold on held-out paired evaluation:

1. the imagination treatment has more total successes than the no-imagination
   control and more paired wins than losses; and
2. no selected task is rejected by the paired task audit (candidate successes
   lower than control or paired losses greater than paired wins).

The current run fails both conditions: `11 < 12` overall, paired wins/losses are
`1/2`, and `place_can_basket` is rejected. Increasing epochs is not the next
action because earlier three-task runs also degraded from epoch 3 to epoch 5.

The next experiment should first diagnose why the imagination-labelled actor
changes the two `place_can_basket` success states in the wrong direction. A
small, fixed-state analysis comparing control and treatment residual chunks on
those two regressions is justified; another broad data collection or training
scale-up is not.

## Authoritative artifacts

- Merged rewards:
  `evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_balanced_pairs_seed44_20260904/merged_wan_vae_head_rewards.json`
- Replay build summary:
  `evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_balanced_pairs_seed44_20260904/replay/build_summary.json`
- Training pair audit:
  `evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_balanced_pairs_seed44_20260904/training/paired_training_audit.json`
- Held-out manifest:
  `evaluate_results/robotwin_imagination_restart/robotwin_video_expert_multitask3_balanced_heldout10_20260904/heldout_manifest.json`
- Smoke summary:
  `evaluate_results/robotwin_residual_online/robotwin_video_expert_multitask3_balanced_seed44_smoke2_20260904/summary.json`
- Formal summary:
  `evaluate_results/robotwin_residual_online/robotwin_video_expert_multitask3_balanced_seed44_formal10_20260904/summary.json`
- Per-task formal audits:
  `evaluate_results/robotwin_residual_online/robotwin_video_expert_multitask3_balanced_seed44_formal10_20260904/audit_{open_microwave,hanging_mug,place_can_basket}.json`
