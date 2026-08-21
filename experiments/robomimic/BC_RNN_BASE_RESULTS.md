# RoboMimic Closed-Loop BC-RNN Base Results

## Why this stage was necessary

The saved expert action sequence reproduced only 10 of 20 successful validation
episodes when replayed open loop. It therefore could not serve as a reliable base for
full-episode residual evaluation. We trained a policy that reads the current simulator
observation at every step instead.

## Training setup

- Dataset: RoboMimic Can paired low-dimensional data, RoboSuite 1.5.1
- Split: 180 training demonstrations and 20 validation demonstrations
- Policy: official RoboMimic BC-RNN-GMM configuration
- Inputs: end-effector position/quaternion, gripper position, and object state
- Recurrent model: 2-layer LSTM, hidden dimension 400, sequence length 10
- Output: 5-mode GMM over 7-dimensional actions
- Optimization: 500 epochs, 100 updates per epoch, batch size 100
- Evaluation: 10 online rollouts every 25 epochs after warm-up, horizon 400

The RoboMimic low-noise GMM forces its evaluation standard deviation to `1e-4`.
Consequently, validation NLL is artificially large and was not used for checkpoint
selection. Checkpoints were selected by online task success.

## Training-time online results

The 19 evaluated checkpoints had the following success rates:

`50:0.0, 75:0.0, 100:0.1, 125:0.1, 150:0.3, 175:0.5, 200:0.4,
225:0.4, 250:0.4, 275:0.6, 300:0.4, 325:0.4, 350:0.1, 375:0.2,
400:0.1, 425:0.3, 450:0.1, 475:0.0, 500:0.0`.

The best training-time checkpoint was epoch 275 (6/10). Later checkpoints overfit,
so `last.pth` must not be used as the base policy.

## Independent fixed-seed evaluation

- Checkpoint: epoch 275
- Seed: 20260822
- Episodes: 50
- Successes: 21
- Success rate: 42%
- Mean horizon: 284.26 steps
- Checkpoint SHA-256:
  `c99311c9e7f7b8836aa48c03b4fec536a5a19f8255e2da7a793809f63d2dc6a9`

Successful episodes generally terminated in roughly 100--170 steps; failures reached
the 400-step limit. This is a suitable medium-difficulty base distribution: the policy
is genuinely closed loop and succeeds often enough to evaluate preservation, while
leaving substantial failure space for a residual policy to improve.

## Decision

Use epoch 275 as the current RoboMimic base. Do not directly attach the residual actor
trained around expert action chunks. First collect short counterfactual branches around
the BC-RNN policy's actual online state-action distribution, then retrain the residual,
Q ensemble, and OOD support model on that distribution before paired online evaluation.
