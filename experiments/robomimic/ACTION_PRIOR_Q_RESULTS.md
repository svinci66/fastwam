# Action-prior initialized deployable Q diagnostic

Date: 2026-08-21

## Question

The original deployable full-state Q underperformed an action-only control on
the held-out trajectory split. This diagnostic tests whether the full-state
model can retain the strong action prior while learning a smaller
state-conditioned correction.

## Method

- Input: train-only PCA-16 wrist SigLIP features plus proprioception (32 state
  dimensions total).
- Initialize the full-state `PairwiseQ` from the matching action-only Q. The
  state columns of the first layer are zero initialized, so epoch 0 exactly
  reproduces the action-only Q.
- Fine-tune with the original pairwise binary objective plus an MSE teacher
  penalty of 1.0 against the frozen action-only logits.
- Preserve the original three seeds and the pre-registered strict gate.

## Results

| Seed | Full-state balanced accuracy | Action-only balanced accuracy | Gain |
| --- | ---: | ---: | ---: |
| 20260820 | 0.6617 | 0.6398 | +0.0219 |
| 20260821 | 0.6508 | 0.6373 | +0.0135 |
| 20260822 | 0.6480 | 0.6334 | +0.0146 |
| Mean | 0.6535 | 0.6368 | +0.0167 |

Mean full-state AUC was 0.6666 versus 0.6629 for action-only. The full-state
model beat the action-only control for every seed, but the mean balanced
accuracy gain did not reach the pre-registered +0.02 requirement. The strict
gate therefore remains **failed** and no residual actor should be promoted
from this validation result.

The associated state-shuffle audit showed that the PCA-16 Q does use the
state: globally shuffling held-out states reduced mean balanced accuracy by
about three points. However, selecting blend weights or further
hyperparameters on this same split would bias the result. The next experiment
should freeze this configuration and evaluate it on newly collected,
trajectory-disjoint counterfactual states.

## Reproduction

```bash
ROBOMIMIC_VISION_PCA_DIM=16 \
ROBOMIMIC_ACTION_PRIOR_INIT=1 \
ROBOMIMIC_Q_TEACHER_REGULARIZATION=1.0 \
bash scripts/run_robomimic_deployable_q_posttrain.sh
```

Local artifacts are under
`evaluate_results/robomimic_bc_rnn_residual_posttrain/deployable_q_pca16_action_prior/`.
