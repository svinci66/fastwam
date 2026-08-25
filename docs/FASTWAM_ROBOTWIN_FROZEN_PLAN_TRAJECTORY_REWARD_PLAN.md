# FastWAM RoboTwin 冻结规划轨迹奖励执行方案

更新日期：2026-08-25
状态：本轮想象奖励的冻结执行方案；在本方案通过前，不训练 Residual actor。

## 1. 研究动机

FastWAM 已经用 Wan Video Expert 联合预测未来视频和动作。我们的目标不是再训练一个独立的视觉奖励模型，而是把 FastWAM 自己的规划作为参考：如果真实执行产生的视觉变化与这段规划一致，就给 Residual 正向辅助奖励；如果执行偏离规划，就给负向辅助奖励。这样希望让模型的规划能力和实际执行能力更贴合。

想象奖励只负责提供过程信号。环境终局成功奖励始终是最终锚点，防止世界模型本身的错误想象误导 Residual。

## 2. 本轮固定设计

每次 FastWAM replan 生成一个 24-action chunk，同时生成对应的未来视频。该次推理产生的整段想象在 chunk 内冻结，执行中不得重新生成或替换参考目标。

- 动作时刻：`0, 4, 8, 12, 16, 20, 24`。
- 预测轨迹：一次 `infer_joint` 得到并冻结的 7 帧。
- 实际轨迹：从相同起点执行动作后，在上述时刻读取并保存 7 帧。
- 比较方式：比较多个中间时刻的变化，不再只比较 chunk 终点。
- 相机：`head`、`left_wrist`、`right_wrist` 分别计算。
- 相机权重：首轮统一使用等权，不在现有 5 个样本上调权重。
- 视觉表征顺序：先验证时序和配对链路，再尝试 Wan VAE latent，最后尝试 Video Expert token；SigLIP 全局特征只保留为旧代理基线。
- 终局奖励：成功奖励保持独立且权重大于想象辅助项。

概念上的奖励形式为：

```text
r_total = r_success + lambda_imagination * r_imagination - lambda_action * ||residual||^2

r_imagination = mean_camera(mean_time(alignment(delta_actual, delta_imagined)))
```

这里 `delta` 都以同一个 chunk 起点为参照。首轮只验证 `r_imagination` 的排序能力，不调 `lambda_imagination`，也不开始 actor 训练。

## 3. 分阶段执行

### 阶段 1：冻结轨迹采集链路

扩展 RoboTwin 统一评测入口，在每个 transition 中保存：

- 原有的 `current.png`、`predicted_goal.png`、`actual.png`，保持旧工具兼容。
- `predicted_trajectory/`：7 个预测帧及对应 action offset。
- `actual_trajectory/`：7 个实际帧及对应 action offset。
- `trajectory_alignment_valid`、帧数、offset、schema version 和 action audit。

先对同一 seed 运行 `clean / controlled_corrupt / controlled_correct` 三分支。三个分支必须共享干预前观测和同一段 FastWAM 参考想象；`corrected` 必须恢复并执行 clean action。

停止标准：

- 如果 7 个预期时刻不完整、offset 错位、三分支参考想象 hash 不一致，立即停止，不计算新奖励。
- 如果 `clean` 与 `corrected` 的动作或实际轨迹不一致，立即停止，先修复确定性或采集副作用。
- 只有 fail-closed 审计全部通过，才进入阶段 2。

### 阶段 2：验证世界模型参考是否包含有效信息

在不训练 actor 的情况下，先比较：

1. 匹配的预测轨迹与实际轨迹。
2. 打乱 seed/replan 后的错误预测轨迹与同一实际轨迹。
3. clean、corrupt、correct 三分支在每个中间时刻的轨迹一致性。

停止标准：

- 匹配参考不能稳定优于打乱参考：说明当前表征或时间对齐无效，不进入训练。
- corrupt 失败样本的轨迹分数不低于对应 clean/correct：说明奖励不能排序当前受控变化，不进入训练。
- 先检查时间对齐和表征，不通过时不得用调相机权重掩盖问题。

### 阶段 3：使用 FastWAM/Wan 原生视觉表征

按风险从低到高依次验证：

1. Wan VAE latent：复用 FastWAM 视频编码/解码空间，先建立稳定的逐时刻特征接口。
2. Wan Video Expert token：在固定层和固定时间步抽取时空 token，按相机区域分别池化。

每次只替换视觉表征，数据、时间点、相机等权、归一化和评价集保持不变。不得同时改表征、奖励公式和数据集。

通过标准：

- 匹配参考相对 shuffled reference 有稳定正 margin。
- 结果不同的配对样本中，成功分支分数更高的比例至少 65%。
- 合并 ROC-AUC 至少 0.65，且每个任务至少 0.60。
- 至少包含 8 个结果不同的有效 pair；不足只判定为样本不足。

### 阶段 4：小规模 Residual 训练

只有阶段 3 通过后，才用同一 replay 训练两组零初始化、bounded residual：

- `no_imagination`：成功奖励 + 动作模仿/残差正则。
- `with_imagination`：完全相同设置，再加入已冻结的想象辅助奖励。

不加入 Q 门控、OOD 门控、累计风险或 FastWAM 主干微调。训练方法先保持统一 AWR，使唯一变量是是否加入想象奖励。

准入标准：`with_imagination` 在 held-out 配对在线测试中必须优于 `no_imagination` 和纯 FastWAM，且 rescue 多于 regression；否则停止扩大规模并回到奖励或数据覆盖分析。

## 4. 当前明确不做的事项

- 不在 5 个样本上学习或手调三相机权重。
- 不用最终成功标签反向挑选奖励公式。
- 不把想象奖励作为全部奖励。
- 不恢复 Twin-Q、Q/OOD gate 或多次门控干预。
- 不立即大规模采集或训练。
- 不微调 FastWAM/Wan 主干。

## 5. 当前执行点

阶段 1 已完成，trajectory-v2 的动作、轨迹、seed、instruction 和 trial offset 审计均已通过。Wan VAE latent 首轮 2 组严格 outcome-discordant pair 也得到正确排序：`clean/corrected` 分数均高于 `corrupt`，且 matched reference 均明显优于 shuffled reference；当前只能视为正向 smoke，尚未达到样本量准入线。

当前继续阶段 3 的固定协议扩展：保持 `0.05` 最大归一化动作扰动、三个相机等权和 `0/4/8/12/16/20/24` 七个时刻不变。候选只依据对齐配置下的 baseline 成功记录和录像中的任务阶段预注册，不查看 Wan VAE 奖励后选样本。

为减少无效计算，扩展按以下顺序自动执行：

1. 先只运行单次 `controlled_corrupt` 分支。
2. 扰动后仍成功的候选直接排除，不补跑另外两支。
3. 扰动后失败时，补跑 `clean`；若 clean 不能复现成功，则排除该 seed。
4. clean 成功后补跑 `corrected`；只有 corrected 恢复成功且三分支审计通过，才计算 Wan VAE reward。
5. 累计达到 8 个严格 pair 后才计算正式排序指标并判断是否进入阶段 4；候选用尽仍不足 8 个时，停止训练并扩大独立 baseline-success seed，而不是加大扰动或调相机权重。
