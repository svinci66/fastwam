# RoboTwin head-only Wan-VAE AWR smoke (2026-08-27)

## Outcome

- Built one immutable replay with 640 action-chunk transitions from 7 matched
  `open_microwave` expert-success / natural-FastWAM-failure pairs.
- Kept the residual actor input unchanged: three-camera frozen SigLIP features,
  proprioception, language, and baseline action chunk.
- Used only the head camera for reward shaping.  The reward compares the frozen
  FastWAM imagined trajectory with actual observations at action offsets
  `0, 4, 8, 12, 16, 20, 24` in Wan2.2 VAE latent space.
- Fitted one episode-balanced global median/IQR normalization and applied
  `0.1 * tanh((score - median) / IQR)`.  The resulting labels span
  `[-0.074309, 0.096434]`; 13 incomplete terminal chunks receive zero shaping.
- Completed a one-seed, three-epoch AWR control/treatment smoke.  The audit
  confirms identical replay, seed, and actor/critic initialization; the only
  configuration differences are the experiment name and imagination weight.

## Training health

| Variant | Critic loss epoch 0 -> 2 | Final actor SHA-256 |
| --- | ---: | --- |
| no imagination | 2.6952 -> 1.3362 | `168d75963f71ceafaf957acf8c74464c83775bee9bfeef904a2651107169939d` |
| head-only imagination | 2.6645 -> 1.2243 | `2726c186e8e54355fc2507f35986072f607e0734c27124af5c5bb6018ebbb0b6` |

Both actors moved away from the identical zero-initialized actor SHA-256
`2abe4bd02bf8423425f45c26bda125b900e64817c763d344a841a88ea41bb51c`.

## Artifacts and interpretation

The local run is under
`evaluate_results/robotwin_imagination_restart/robotwin_open_microwave_wan_head_awr_smoke_seed42_20260827`.
`training/paired_training_audit.json` reports `all_exact: true`.

This smoke validates data construction, reward relabeling, optimization, and a
strict offline ablation.  It does **not** yet establish an online success-rate
gain.  The next experiment is a matched-seed online comparison of the frozen
FastWAM baseline, no-imagination residual, and head-imagination residual.
