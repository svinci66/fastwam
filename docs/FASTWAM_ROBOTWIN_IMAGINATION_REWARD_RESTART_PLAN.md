# FastWAM RoboTwin 想象奖励重新验证方案

更新日期：2026-08-24

状态：总体实验边界说明。当前奖励实现和执行停止标准以
[`FASTWAM_ROBOTWIN_FROZEN_PLAN_TRAJECTORY_REWARD_PLAN.md`](FASTWAM_ROBOTWIN_FROZEN_PLAN_TRAJECTORY_REWARD_PLAN.md)
为准；此前以 Twin-Q、OOD gate、因果门控和 IQL 为核心的路线暂停，不参与本轮实验。

## 1. 本轮只回答两个问题

1. FastWAM 的想象奖励能否在中等难度 RoboTwin 任务上，稳定区分更好和更差的动作结果？
2. 在完全相同的数据、Residual actor、训练随机种子和在线评测协议下，加入想象奖励能否在小规模评测中提高成功率？

本轮不研究门控，不证明大规模泛化，也不追求论文级统计显著性。只有上述两个问题都得到正向结果，才扩大任务、数据和评测规模。

## 2. 系统边界

本轮系统只有以下部分：

```text
观测 + 官方语言指令
          │
          ▼
   冻结的 FastWAM ─────► imagined future（只用于训练奖励）
          │
          ▼
   FastWAM baseline action
          │
          ▼
   零初始化 Residual actor
          │
          ▼
 baseline + bounded residual
```

明确移出本轮主实验的模块：

- Twin-Q 和任何 Q 门控。
- IQL Twin-Q、IQL value network 和基于 Q 的 checkpoint 选择。
- OOD gate、因果门控、outcome confirmation、累计风险预算。
- 根据模型置信度动态决定是否干预的任何逻辑。
- FastWAM 主干微调。

旧代码和 checkpoint 保留以便复盘，但新实验脚本不得加载它们。在线比较采用预先固定的单次干预位置，因此不需要门控，也不会把“什么时候干预”和“想象奖励是否有效”混在一起。

## 3. 统一、不可变的 FastWAM 设置

所有阶段统一使用：

- 官方 FastWAM RoboTwin 权重。
- 10 次去噪。
- 每 24 个动作重新调用 FastWAM。
- 官方 unseen instruction。
- 同一任务不同方法使用完全相同的环境 seed、指令、初始状态和推理随机性。
- RoboTwin 官方指令候选会受单次运行的 episode 数量影响，因此筛选完成后必须同时冻结实际接受的 `seed + instruction` manifest；后续配对不能只复用 seed。
- Residual 最大尺度首轮固定为 `0.05`，gripper residual 固定为零。
- 每个 episode 只在预注册的一个 replan 执行一次 residual，其余时间完全执行 FastWAM。

任何方法不得单独修改语言、动作步数、相机输入、去噪次数、episode 长度或成功判定。

## 4. 阶段 A：重新筛选中等难度任务

不直接沿用历史难度结论，先在当前论文对齐设置下重新筛选。

### 候选任务

首轮候选为：

- `hanging_mug`
- `place_can_basket`
- `stack_blocks_two`
- `open_microwave`
- `adjust_bottle`

每个任务先跑 5 个固定 seed 的纯 FastWAM baseline；结果接近边界时再补到 10 个 seed。

### 入选规则

选择两个任务作为主实验任务：

- 10 回合成功率优先落在 30%--70%。
- 至少同时出现 3 个成功和 3 个失败。
- 失败不是环境崩溃、超时加载或评测脚本错误。
- 如果没有两个任务满足条件，选择成功率最接近 50% 且同时存在成功和失败的任务。

成功率高于 80% 的任务只能作为能力保持检查，成功率低于 20% 的任务本轮不用于验证奖励带来的提升。

### 阶段 A 停止标准

- 找到两个合格任务：进入阶段 B。
- 所有候选任务都处于大于 80% 或小于 20%：停止训练，先扩展任务候选或调整任务配置。
- baseline 无法稳定复现相同 seed：停止全部后续实验，先修复评测协议。

## 5. 失败 rollout 视频和人工分析

从阶段 A 开始，所有方法的失败 episode 都必须保存视频，不能只保存成功率 JSON。

### 保存内容

- 保存可用相机视角的同步拼接视频；至少包含主视角和腕部视角。
- 文件名包含 `task`、`seed`、`variant`、干预 replan 和最终结果。
- 同目录保存逐 replan 元数据：FastWAM 动作、residual、实际动作、想象奖励总分和各相机分量。
- 每个方法额外保存至少两个成功视频，作为人工对照。

统一目录：

```text
evaluate_results/robotwin_imagination_restart/<run_name>/
├── summaries/
├── metadata/
└── videos/
    ├── failures/<task>/<variant>/
    └── success_examples/<task>/<variant>/
```

### 人工标注

每轮评测生成 `failure_review.csv`，至少填写：

- 第一次明显偏离发生在哪个 replan。
- 失败阶段：定位、接近、抓取、运输、放置、释放、碰撞或超时。
- 失败主要来自 FastWAM，还是 residual 使原本合理的轨迹偏离。
- 想象奖励在偏离前后是否出现与人工判断相反的变化。
- 视频是否足以判断；不能判断的样本不得强行分类。

在人工复核完成以前，不允许因为成功率下降就直接增加模型或门控模块。

## 6. 阶段 B：只验证想象奖励是否有效

本阶段不训练 actor。目标是先确认奖励本身能否排序动作结果。

### 数据

对每个入选任务使用 10 个训练/验证 seed。每个 seed 从相同 FastWAM 状态产生两条匹配 rollout：

1. `baseline`：执行 FastWAM 原动作。
2. `perturbed`：在预注册 replan 执行一次有界 residual 扰动，之后重新进入 FastWAM 闭环。

扰动从固定随机种子生成，幅度不超过后续 actor 的 `0.05` 上限，不修改 gripper。两条分支必须匹配初始状态、指令、干预前观测和 FastWAM 动作；不完整的 pair 直接丢弃。

### 奖励计算

不再把 SigLIP 终点 `delta_alignment` 当作正式奖励。每次 replan 的 FastWAM
想象在完整 action chunk 内冻结，并在 `0, 4, 8, 12, 16, 20, 24` 多个时刻与
实际轨迹比较；三个相机分别计算并暂时等权。表征优先验证 Wan VAE latent，随后验证
Wan Video Expert token。详细 schema、审计要求和停止标准见冻结规划轨迹奖励执行方案。

- 各相机使用训练 seed 的全局统计分别归一化。
- episode 想象奖励累计绝对值裁剪到 `1.0`。
- 终局成功标签仍作为最终锚点，不由想象奖励替代。

### 评价指标

- 成功 rollout 与失败 rollout 的想象奖励分布。
- 预测成功/失败的 ROC-AUC。
- 结果不同的匹配 pair 中，奖励是否把成功分支排在失败分支之前。
- 两个任务分别统计，不能只报告混合后的总体数字。
- 结合失败视频检查奖励是否只偏好“大幅视觉运动”，而不是任务进展。

### 阶段 B 通过标准

必须同时满足：

- 合并 ROC-AUC 至少为 `0.65`。
- 每个任务 ROC-AUC 至少为 `0.60`，方向不能相反。
- 终局结果不同的 pair 中，成功分支奖励更高的比例至少为 `65%`。
- 至少有 8 个结果不同的有效 pair；否则只判定样本不足，不能判定奖励有效。
- 人工视频复核没有发现奖励主要由无关镜头运动、遮挡或机械臂大幅移动驱动。

### 阶段 B 失败后的处理

- 有效 discordant pair 少于 8：只补数据，不修改模型。
- 指标接近随机：停止 actor 训练，优先修正时间对齐、相机归一化或视觉变化定义。
- 一个任务有效、另一个任务方向相反：不能报告普遍有效；只允许定位失败原因后重新做一次预注册验证。
- 第二次独立验证仍未达到标准：停止想象奖励主线，不进入阶段 C。

## 7. 阶段 C：无 Q 的小规模 Residual 训练

阶段 B 通过后，才训练 residual。为了完全隔离想象奖励的贡献，使用不学习动作价值 Q 的 Monte-Carlo AWR。

### 公平对照

在同一份 replay 上训练两个 residual actor：

1. `residual_no_imagination`

```text
R = 10 * terminal_success - 0.1 * normalized_residual_MSE
```

2. `residual_with_imagination`

```text
R = 10 * terminal_success
    - 0.1 * normalized_residual_MSE
    + normalized_imagination_reward
```

想象奖励整回合绝对值仍不得超过 `1.0`，保证它只是辅助项，不覆盖终局成功。

两组必须保持：

- 相同 replay、train/validation seed 划分。
- 相同 actor 结构和零初始化。
- 相同 batch、epoch、优化器和训练随机种子。
- 相同 checkpoint 选择规则。
- 三个训练 seed。

AWR 首先从实际奖励计算 Monte-Carlo return，再训练一个仅依赖状态的 `V(s)` 作为方差降低基线，并用 `exp((return - V(s)) / beta)` 对行为回归加权。这个训练期 `V(s)` 不接收 residual candidate，不是 `Q(s,a)`，不参与在线推理或干预批准。整个实验不训练 Twin-Q，也不使用任何 Q 或 V 门控。验证集 checkpoint 只根据 held-out 行为损失和真实 episode return 选择。

## 8. 阶段 D：小规模在线成功率验证

对两个中等难度任务各使用 10 个完全 held-out seed，逐 seed 配对运行：

1. `FastWAM baseline`
2. `FastWAM + residual_no_imagination`
3. `FastWAM + residual_with_imagination`

两个 residual 版本都只在相同的预注册 replan 执行一次，不使用任何 gate。每个失败都保存视频并完成人工归因。

### 小规模成功标准

20 个任务回合合并统计时，`with_imagination` 必须同时满足：

- 比 `no_imagination` 至少多成功 2 回合。
- 比纯 FastWAM 至少多成功 2 回合。
- 两个任务都不能净下降。
- `rescue` 数量严格多于 `regression`。
- 在 FastWAM 原本成功的回合中，最多造成 1 个 regression。
- 三个训练 seed 中至少两个呈现同方向提升。

这是继续扩大规模的工程准入标准，不等于论文级统计结论。

### 小规模失败后的结论和动作

- 阶段 B 通过、但 `with_imagination` 不优于 `no_imagination`：结论是奖励有相关性但尚未带来策略收益；优先检查奖励信用分配和 replay 覆盖，不加入 Q 门控。
- 两个 residual 都不优于 FastWAM：检查 residual 数据是否包含可学习的成功修正，以及 actor 是否只学到接近零或持续偏移。
- `with_imagination` 的 regression 明显更多：结合失败视频定位奖励误导的阶段；停止扩大训练。
- 只有单个训练 seed 提升：判定结果不稳定，不能进入正式实验。

## 9. 本轮最终交付物

- 中等难度任务筛选表和固定 seed manifest。
- 三组方法的严格配对成功率表。
- 奖励 ROC-AUC、pairwise ranking accuracy 和按任务统计。
- 所有失败 rollout 视频及 `failure_review.csv`。
- 无想象/有想象两个 residual checkpoint，包含三个训练 seed。
- 一页结论：奖励是否有效、是否带来小规模提升、失败发生在哪些阶段。

## 10. 执行顺序

严格按以下顺序推进，任何阶段不通过就停止，不自动增加 Q 或其他模块：

```text
A. 重新筛选中等难度任务
        ↓
B. 离线/配对验证想象奖励
        ↓ 仅在通过时
C. 同数据训练 no-imagination / with-imagination residual
        ↓
D. 两任务 × 10 held-out seed 三组在线配对评测
        ↓
人工复核全部失败视频并形成结论
```
