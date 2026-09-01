# RoboTwin three-task Wan-head residual AWR training (2026-09-01)

## Outcome

- Restricted training to `open_microwave`, `hanging_mug`, and
  `place_can_basket`; retained `blocks_ranking_size` for later capability
  retention evaluation.
- Built one immutable replay from 42 natural-failure/expert-success pairs.  The
  selected head-camera Wan-VAE reward ranks 41/42 pairs correctly (97.62%).
- Encoded 2,729 action-chunk transitions with frozen three-camera SigLIP
  observations and the FastWAM UMT5 language feature.
- Passed a three-epoch smoke health gate and automatically completed the
  pre-registered 20-epoch formal AWR run.
- FastWAM remains frozen.  The residual actor is zero-initialized, uses a
  per-joint maximum scale of 0.1, and leaves both gripper dimensions unchanged.

## Replay coverage

| Task | Reward pairs | Action-chunk transitions |
| --- | ---: | ---: |
| `open_microwave` | 17 | 1,529 |
| `hanging_mug` | 15 | 789 |
| `place_can_basket` | 10 | 411 |

The AWR loader uses task-uniform sampling so the larger `open_microwave` shard
does not dominate optimizer steps.

## Training health

| Run | Epochs | Critic loss | Actor loss | Residual RMS | Saturation fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 2.8612 → 0.4058 | 0.000625 → 0.000481 | 0.00485 | 0.000% |
| formal | 20 | 2.8612 → 0.0415 | 0.000625 → 0.000254 | 0.00881 | 0.039% |

Both audits confirm finite optimization histories, a changed actor relative to
the common zero initialization, exact frozen gripper dimensions, nonzero expert
residual targets, expected task coverage, head-only Wan reward labels, and no
residual saturation.

## Artifacts

- Run root:
  `evaluate_results/robotwin_imagination_restart/robotwin_wan_head_multitask3_awr_seed42_20260901`
- Replay: `replay/`
- Smoke checkpoint and audit: `training/smoke/`
- Formal checkpoint and audit: `training/formal/`
- Formal checkpoint SHA-256:
  `d271f2074e413102db87df5409db95f08227574d106a70203a839002c381d372`

This run establishes healthy multi-task offline optimization.  It does not yet
establish an online success-rate improvement.  The next required experiment is
a fixed-seed paired online comparison against frozen FastWAM on the three
training tasks, with `blocks_ranking_size` used only as a retention check.
