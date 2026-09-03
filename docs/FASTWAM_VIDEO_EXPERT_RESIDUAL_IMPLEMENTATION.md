# FastWAM Video Expert Residual Implementation

## 结论

Residual actor 的视觉输入已由独立 SigLIP 编码改为 FastWAM 原生 Video Expert 表征。在线推理时，基础动作和 residual 视觉特征来自同一次 FastWAM Video Expert prefill；新训练入口不再加载第二套视觉编码器。

## 数据流

```text
三相机复合图像 + 指令 + proprio
          |
          v
      Wan VAE 首帧 latent
          |
          v
Video Expert pre_dit + 30 层 transformer
          |                         |
          | 每层 K/V cache          | 最后一层空间 token [B, S, 3072]
          v                         v
Action Expert 去噪得到基础 action   token mean + L2 normalize
          |                         |
          +------------+------------+
                       v
 residual actor(native feature + proprio + UMT5 instruction + base action chunk)
                       |
                       v
       clipped(base action + 0.1 * residual)
```

Action Expert 原本就通过 MoT 使用 Video Expert 每一层的 K/V cache。Residual 使用同一次 prefill 的最后一层 token，经无参数的均值池化和 L2 归一化得到 3072 维向量。它不是额外视觉模型，也不会引入 SigLIP/FastWAM 两套表示不一致的问题。

## 训练输入与目标

- 视觉：`fastwam_video_expert_final_token_mean_l2_v1`，3072 维。
- 本体状态：RoboTwin 14 维 proprio。
- 语言：FastWAM 同一条指令的冻结 UMT5 embedding，经 residual 内部投影到 128 维。
- 基础动作：FastWAM 的完整 action chunk，经投影到 128 维。
- 输出：与基础 action chunk 同形状的有界 residual；输出层保持零初始化，部署缩放仍为 `0.1`。
- 学习：当前仍是 AWR；成功奖励是最终锚点，Wan VAE 头部相机想象一致性只作为附加奖励。当前实验不启用 Q 门控、OOD 门控或 goal conditioning。

## 离线 replay 与在线一致性

旧轨迹中的图像、动作、成功标签和想象奖励可以继续使用，但需要运行一次 feature backfill，用冻结 FastWAM 对保存的当前图像提取 Video Expert 特征。每条记录保存：

- 特征版本；
- 特征维度；
- 基础 FastWAM checkpoint 的 SHA-256。

训练 checkpoint 会继承这些 provenance。在线加载时必须使用相同 FastWAM checkpoint；哈希或维度不一致会直接拒绝运行，避免离线训练特征和在线特征静默错位。

## 入口

最小特征验证：

```bash
PHASE=feature-smoke bash scripts/run_robotwin_video_expert_residual_pipeline.sh
```

完整回填并训练匹配的无想象/有想象 AWR smoke 模型：

```bash
PHASE=train-smoke bash scripts/run_robotwin_video_expert_residual_pipeline.sh
```

旧 SigLIP checkpoint 仍可加载，但旧脚本必须显式设置 `--actor-observation-source siglip`；新 Video Expert checkpoint 不接受 `residual_encoder_path`。

## 当前边界

本实现使用最后一层 token 的全局均值，是一个稳定、无额外参数的第一版。Action Expert 实际使用的是所有层的 Video Expert K/V cache，因此该 3072 维向量是原生视觉状态的压缩摘要，并不等价于完整 K/V cache。下一步应先用同一数据和种子比较无想象与有想象 AWR，而不是立即增加新的门控或复杂池化模块。
