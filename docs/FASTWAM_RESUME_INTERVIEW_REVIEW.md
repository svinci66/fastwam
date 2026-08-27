# FastWAM Residual RL 简历核验与面试复习手册

更新日期：2026-08-27

适用简历条目：**预训练 VLA 的离线强化学习后训练与安全残差控制**

本文有两个目的：

1. 对照当前源码、配置和实验记录，核验简历中的技术表述是否真实；
2. 把简历中出现的每个知识点整理成可用于面试复习的“原理—实现—证据—局限—追问”材料。

---

## 1. 核验结论

### 1.1 总体结论

技术内容**基本符合项目实际情况**，可以写入简历，但需要保留三个边界：

- 项目确实实现过完整的 Residual IQL、Twin-Q、expectile value、优势加权 actor、想象奖励、Q gate、kNN 支持域检测和 episode 干预预算。
- 这些模块不等于已经形成了具有理论保证的“安全控制器”。严格配对实验暴露过 Q 排序错误、kNN 无法识别累积闭环风险等问题，因此更稳妥的措辞是“安全约束与分布外干预分析”或“保守残差控制”。
- 2026-08-24 之后的当前主实验暂时移除了 IQL、Q/OOD gate 和预算，回到更小的研究问题：先独立验证 FastWAM 想象奖励，再用无 Q 的 AWR 做公平对照。因此面试时应把 IQL 与门控描述为“已完成的框架与阶段性探索”，不要说成当前已经验证成功的最终方案。

公司、岗位归属和任职日期不属于代码仓库可验证范围；本文只核验技术事实。

### 1.2 逐条核验

| 简历表述 | 判断 | 项目证据与需要注意的边界 |
|---|---|---|
| 面向预训练 FastWAM 策略搭建 Residual IQL 后训练框架 | 符合 | `src/fastwam/rl/iql_trainer.py` 实现完整 IQL，FastWAM 主干不参与该优化过程。|
| 冻结 VLA 主干 | 符合 | Residual actor、Q critic 和 value critic单独训练；FastWAM 作为冻结动作先验与想象来源。|
| 以多相机观测、语言指令、本体状态和基础动作为输入 | 符合，但应说清输入是冻结特征 | RoboTwin 使用 head、left-wrist、right-wrist 三相机的冻结视觉特征；语言使用 FastWAM UMT5 masked-mean 特征；actor 还接收 proprio 与 FastWAM action chunk。不是直接把原始图像和文本送入一个新训练的端到端网络。|
| 学习有界动作残差 | 符合 | Actor 输出经过 `tanh × residual_scale`，与 baseline action 相加后再受动作上下界约束；gripper residual 可固定为零。|
| 实现双 Q Critic、Expectile Value Learning | 符合 | 两个 action-conditioned Q、一个 V、target-Q 软更新和 `min(Q1,Q2)` 均已实现。|
| 实现优势加权回归 | 符合 | Actor 对数据动作做加权行为回归，权重来自 `exp(temperature × (Q−V))` 并裁剪。严格说是 IQL 中的 advantage-weighted behavioral cloning，而不是另起一个在线 policy-gradient 算法。|
| 基于成功、失败和受控扰动轨迹构建离线数据 | 符合 | Replay 支持 policy、expert、noise、hold、gripper-delay、residual 等行为；做过成功/失败与受控破坏配对分析。|
| 设计任务奖励与未来状态一致性奖励 | 符合，但建议换名 | 更准确的名称是“基于 FastWAM 想象未来与真实后继观测对齐的辅助奖励”。当前正式研究正在重新验证其时间对齐、表征与任务成功相关性，不能说已经证明它能提升策略。|
| 引入 Q-value improvement gate | 符合 | 在线比较 candidate 与 baseline 的两个 Q advantage，并使用最小优势和分歧阈值决定是否放行。|
| 引入 kNN 状态/动作支持域检测 | 符合 | 支持索引联合视觉、proprio、baseline action，并检查候选 residual 在邻近成功数据中的动作支持。|
| 引入 episode 级干预预算 | 符合 | 只对实际执行的 residual chunk 计数，每个 episode 重置；曾在小规模保能力测试中使用。它是经验性保护机制，不是普适最优预算。|
| 用于分析 OOD 价值高估、触发边界和连续修正退化 | 符合 | 严格实验观察到 Twin-Q 对 actor 候选的错误排序，以及连续干预使轨迹逐步偏离；也确认 kNN 只能发现缺少支持，不能判断所有有害动作。|

---

## 2. 推荐的简历版本

### 2.1 推荐标题

优先推荐：

> **预训练 VLA 的离线强化学习后训练与保守残差控制**

如果必须保留“安全”关键词，建议写成：

> **预训练 VLA 的离线强化学习后训练与残差安全约束**

“安全残差控制”容易让面试官理解为已经获得形式化安全保证、真实机器人安全认证或经过大规模稳定性验证，而当前项目属于部署保护机制与失败边界研究。

### 2.2 推荐正文

> **目标与职责：** 面向预训练 FastWAM 策略搭建 Residual IQL 后训练框架；冻结 VLA 主干，融合三相机冻结视觉特征、FastWAM UMT5 语言特征、本体状态和基础动作块，学习零初始化、幅度受限的动作残差。
>
> **算法链路：** 实现 Twin-Q Critic、expectile value learning、target-Q 软更新与优势加权行为回归；基于成功、失败及受控扰动轨迹构建离线 replay，并设计终局成功、动作偏差约束和想象未来—真实后继观测对齐的复合奖励。
>
> **部署约束与诊断：** 实现 candidate-vs-baseline Q improvement gate、kNN 状态/动作支持域检测和 episode 级干预预算；通过同 seed 配对与单次干预反事实实验，分析离线 Q 在 actor 候选上的 OOD 高估、门控触发边界及连续 residual 引起的闭环退化。

这版比原文更准确的地方：

- 明确“多相机和语言”使用的是冻结特征，而非端到端微调；
- 明确 actor 是零初始化且有界；
- 把“优势加权回归”限定为行为回归，避免被误解为 PPO/SAC 式在线更新；
- 将“安全门控”改成“部署约束与诊断”，不暗示已解决安全问题；
- 强调同 seed 配对和单次干预实验，这是本项目比简单堆模块更有价值的工程研究部分。

---

## 3. 30 秒项目介绍

> FastWAM 是一个能够同时预测机器人动作和未来视频的 VLA。我冻结它作为基础策略，只训练一个幅度很小的 residual actor，在多相机视觉特征、语言、本体状态和 FastWAM 原动作条件下修正 action chunk。训练侧实现了离线 IQL，包括 Twin-Q、expectile V 和优势加权行为回归，并把终局成功、动作模仿约束及想象未来与真实观测的对齐信号组合为奖励。部署侧进一步做了 Q improvement、kNN 支持域和干预预算实验。后续严格配对测试发现，双 Q 一致仍可能在 actor 候选上共同高估，kNN 也无法识别所有连续干预风险，所以当前研究重点是先把奖励有效性和同状态候选数据验证扎实，而不是把门控包装成已经解决的问题。

---

## 4. 系统全链路

```text
当前三相机图像 ──► 冻结视觉编码器 ──► 多相机 observation feature ─┐
语言指令 ────────► FastWAM UMT5 ────► language feature ──────────┤
机器人关节状态 ─────────────────────► proprio ────────────────────┤
                                                                  ▼
冻结 FastWAM ──► baseline action chunk ──────────────► Residual actor
        │                                                   │
        └──► imagined future                                ▼
                                                bounded residual action
                                                          │
                                    candidate = baseline + residual
                                                          │
                         ┌────────────────────────────────┼────────────────────┐
                         ▼                                ▼                    ▼
                    Twin-Q ΔQ                       kNN support         episode budget
                         └────────────────────────────────┼────────────────────┘
                                                          ▼
                                             candidate 或 baseline
                                                          │
                                                          ▼
                                              环境执行并再次闭环重规划
```

需要主动说明：这张图是“实现过的完整探索链路”。当前想象奖励重启实验为了隔离变量，暂时关闭了 Twin-Q、OOD gate 和 episode budget。

---

## 5. 知识点一：FastWAM、VLA 与后训练目标

### 一句话理解

FastWAM 是基础能力很强但并非完美的预训练 VLA；本项目不从头训练它，而是把它当作冻结动作先验，在外部增加一个小型 residual 分支做后训练。

### 为什么选择冻结主干

- 本地和单机算力难以稳定全量微调视频生成与动作模型；
- 直接更新主干容易破坏已有能力，难以判断提升来自哪里；
- Residual 将学习目标从“重新学会机器人控制”缩小为“在必要状态下做小修正”；
- zero residual 天然回退为原始 FastWAM，便于做严格 baseline 对照。

### 项目里的真实边界

- FastWAM 生成完整 baseline action chunk；
- residual actor 不替代 FastWAM，只修改被授权的动作维度；
- FastWAM 的 imagined future 只作为训练奖励信息，不是部署时必须执行的第二套规划器；
- 当前没有把 residual 能力蒸馏或合并回 FastWAM 主干参数。

### 常见追问

**问：这算微调 FastWAM 吗？**

答：广义上属于对预训练 VLA 的 post-training，但参数层面没有更新 FastWAM 主干；训练的是外接 residual actor 和价值模型。更准确的说法是“冻结基础策略的 Residual RL 后训练”。

**问：为什么不直接 LoRA 微调？**

答：第一阶段希望最小化灾难性遗忘并建立可解释对照。Residual 能保持 base action prior，并能通过 residual=0 精确回退。只有证明系统级修正确实有稳定净收益后，再考虑 LoRA 或 adapter 合并更合理。

---

## 6. 知识点二：Residual actor 的结构与有界控制

### 输入

项目中的 actor context 主要包含：

- 三相机冻结视觉特征的融合结果；
- proprioception；
- 可选的 FastWAM imagined-goal feature；
- 单独投影后的语言特征；
- 单独投影后的 FastWAM baseline action chunk。

RoboTwin 三个相机为：

- `head`；
- `left_wrist`；
- `right_wrist`。

融合不是简单把原始像素丢给 MLP，而是先对每个相机的冻结特征做归一化，再按固定顺序拼接并归一化。

### 输出

Actor 的核心形式为：

```text
r = tanh(MLP(context, language, baseline_action)) × residual_scale
a_candidate = clip(a_fastwam + r, action_low, action_high)
```

其中 `residual_scale` 按动作维度设置。对于不希望 residual 接管的 gripper 维度，scale 设为 0，输出严格保持 FastWAM 原动作。

### 为什么零初始化输出层

最后一层权重和偏置初始化为 0，所以训练开始时：

```text
r ≈ 0
a_candidate ≈ a_fastwam
```

意义是：

- 初始策略不会随机破坏 base policy；
- 训练早期行为更稳定；
- zero-residual equivalence 更容易审计；
- 体现“基础策略负责主要能力，residual 只负责必要修正”的归纳偏置。

### 局限

- 当前 MLP 使用压缩后的全局视觉特征，可能丢失精细接触和局部几何信息；
- 没有 LSTM 时，时序状态主要依赖当前观测和 action chunk，连续接触任务可能部分可观测；
- 有界只限制单次动作幅度，不能保证多次小修正不会累计成大偏离。

### 源码入口

- `src/fastwam/rl/models.py::ResidualActorConfig`
- `src/fastwam/rl/models.py::ResidualActor`
- `experiments/robotwin/build_residual_rl_replay.py::combine_camera_features`

---

## 7. 知识点三：为什么使用离线强化学习

### 一句话理解

离线 RL 使用已经收集好的轨迹学习，不需要在训练的每一步持续与昂贵、易崩溃的仿真器交互。

### 本项目适合离线 RL 的原因

- FastWAM 推理和 RoboTwin/LIBERO 仿真都较重，在线采样成本高；
- 可以复用 baseline、专家、失败和受控扰动轨迹；
- 离线训练便于固定数据做 no-imagination/with-imagination 公平对照；
- 避免在线探索把高成功率基础能力迅速破坏。

### 离线 RL 的核心困难

训练数据只覆盖行为策略执行过的动作。如果 actor 在部署时提出数据中从未出现的候选，critic 会发生外推：它可能给 OOD 动作异常高的 Q。

这正是项目后来遇到的问题：两个 Q 在相同缺失数据上可能产生相关偏差，所以“双 Q 都同意”并不能证明候选动作安全或更好。

### 常见追问

**问：既然训练是离线的，为什么还要在线评测？**

答：离线 loss、Q 值和奖励排序都只是代理指标。最终要证明 residual 是否改善真实闭环控制，仍然必须在仿真环境中按相同 seed 在线 rollout。离线训练不等于离线完成最终验证。

---

## 8. 知识点四：IQL 的整体原理

### IQL 解决什么问题

标准 actor-critic 在 Bellman backup 或策略更新时可能显式评估数据集外动作。IQL 通过 expectile value learning，从数据集动作的 Q 分布中学习一个偏向高价值区域的 V，而不需要在 target 中对 actor 产生的新动作求最大值。

### 项目中的三个学习对象

1. 两个动作价值函数：`Q1(s,a)`、`Q2(s,a)`；
2. 一个状态价值函数：`V(s)`；
3. 一个 residual actor：输出 `a_fastwam + r`。

这里的 `s` 还包含 FastWAM baseline action 和语言条件；`a` 是实际执行的 action chunk。

### 训练顺序

每个 batch 内依次：

1. 用 target-Q 的保守最小值训练 V；
2. 用 `r + γ^K V(s')` 训练两个 Q；
3. 用 `Q−V` 生成优势权重，训练 actor 回归数据动作；
4. 对 target-Q 做 Polyak 软更新。

### 为什么折扣是 `γ^K`

一条 replay transition 对应一个执行了 `K` 个低层动作的 action chunk，而不是单个控制步。因此下一状态发生在 K 步之后，折扣应为 `γ^K`。最后不足完整 chunk 时使用 `effective_k`。

### 源码入口

- `src/fastwam/rl/iql_trainer.py::IQLConfig`
- `src/fastwam/rl/iql_trainer.py::compute_iql_losses`
- `src/fastwam/rl/iql_trainer.py::train_residual_iql`

---

## 9. 知识点五：Twin-Q Critic

### 为什么需要两个 Q

单个 Q 容易因 bootstrap 和函数逼近产生高估。项目使用两个独立 critic，并在 value target 中取：

```text
Qmin(s,a) = min(Q1_target(s,a), Q2_target(s,a))
```

这是一种保守估计，降低只由一个 critic 偶然高估造成的问题。

### Q 的输入为什么同时包含 baseline 和 executed action

Q 网络分别编码：

- 状态 context；
- FastWAM baseline action chunk；
- 实际执行或候选 action chunk；
- 语言特征。

这样 Q 学的是“在 FastWAM 原建议为某动作时，把它修改成另一个动作的价值”，不用从像素中重新推断 base action prior。

### 双 Q 不能解决什么

- 两个 Q 使用同一 replay，可能共享系统性数据偏差；
- actor 候选若超出 replay 支持，两个 Q 可能共同错误自信；
- Q 学到的是相关性与 Bellman 价值，不是严格的干预因果效应；
- 两个 Q 的分歧小，只能说明它们意见一致，不能说明意见正确。

### 项目观察

严格单次干预 pair 中曾出现：真实 rescue 被 Q 拒绝，而被 Q 批准的候选导致 regression。这促使项目把 Q 从“独立放行依据”降级为辅助或诊断量。

### 源码入口

- `src/fastwam/rl/models.py::ActionValueCritic`
- `src/fastwam/rl/iql_trainer.py::compute_iql_losses`

---

## 10. 知识点六：Expectile Value Learning

### 公式

令：

```text
δ = Qmin_target(s,a_dataset) - V(s)
```

expectile loss 为非对称平方损失：

```text
L_V = E[ |τ - I(δ < 0)| × δ² ]
```

代码等价地在 `δ ≥ 0` 时使用权重 `τ`，在 `δ < 0` 时使用 `1−τ`。

### τ 的含义

- `τ = 0.5` 接近普通均方回归；
- `τ > 0.5` 更重视 V 低估高 Q 数据动作的误差；
- 项目默认 `τ = 0.7`，使 V 靠近数据集动作 Q 分布的较高 expectile；
- τ 过高会更偏向极少数高 Q 样本，也更容易受 Q 噪声影响。

### 为什么不是 max Q

IQL 不直接构造数据外动作并求 `max_a Q(s,a)`，而是在已有数据动作的 Q 分布上做 expectile regression，从而减少 target 端的 OOD 动作查询。

### 常见追问

**问：expectile 和 quantile 有什么区别？**

答：quantile regression 常用非对称绝对误差，对应分位点；expectile 使用非对称平方误差，对大误差更敏感，得到的是 expectile，不等同于分位数。

---

## 11. 知识点七：优势加权行为回归

### 项目公式

优势为：

```text
A(s,a) = Qmin_target(s,a_dataset) - V(s)
```

权重为：

```text
w = clip(exp(temperature × A), w_max)
```

Actor loss 为：

```text
L_actor = E[w × masked_MSE(a_actor, a_dataset)]
```

只对 `effective_k` 内真正执行的动作计算 MSE，padding 后缀不参与训练。

### 直观解释

- 数据里比当前 V 更好的动作得到更大权重；
- 较差动作仍在数据支持内，但影响更小；
- actor 仍然是监督式回归，不通过 Q 对动作做直接梯度上升；
- 这比直接最大化 Q 更保守，适合离线数据。

### “AWR”和“IQL actor update”的关系

两者都使用优势加权回归思想，但优势来源不同：

- Monte-Carlo AWR 可用 return-to-go 与 V 的差；
- 当前 IQL actor 使用 `Q−V`；
- 2026-08-24 后的奖励重启实验为了隔离 Q，计划使用无 Q 的 Monte-Carlo AWR；
- 简历可以写“优势加权回归”，但面试时要说明自己实现的是哪一种。

---

## 12. 知识点八：离线 replay 的构造

### Transition 中保存什么

每条 chunk-level transition 包括：

- 当前和下一时刻的视觉特征；
- imagined-goal feature；
- 当前和下一 proprio；
- FastWAM baseline action chunk；
- 实际执行 action chunk；
- `effective_k`；
- 环境奖励、终局成功、terminated/truncated；
- 行为模式和扰动强度；
- 语言特征及其 encoder version；
- 想象奖励、相机信息和对齐有效性；
- task、episode、transition index 和 seed/provenance。

### 数据来源

- FastWAM policy rollout：代表基础策略分布；
- expert success：提供成功动作与终局信号；
- natural failure：提供真实失败分布；
- noise/hold/gripper-delay：构造可控破坏强度；
- residual rollout：覆盖 actor 实际会提出的候选；
- 同 seed 单次干预 pair：用于区分 rescue、regression 和 neutral。

### 为什么仅有专家成功轨迹不够

如果数据只有专家动作：

- critic 不知道坏动作应得到什么值；
- actor 只学模仿专家，无法学习何时不应修正；
- Q gate 无法比较 baseline 与 residual candidate；
- 部署候选容易处在数据支持之外。

### 为什么受控扰动也不够

人工 hold 或 noise 往往比真实 residual 粗糙。奖励能区分强破坏，不代表能区分幅度很小但改变接触结果的 residual。因此最终仍需要 actor-aligned 的真实候选和同状态配对结果。

### 数据划分原则

- 按完整 environment seed/episode 划分 train、validation、test；
- 不能把同一 episode 的不同 transition 分到不同集合；
- 同一个 episode 的成功标签复制到多个 transition 会制造伪样本量，应避免用 transition 数冒充独立轨迹数；
- 相机归一化与门控阈值只能由训练/验证数据拟合，不能看最终测试 seed。

### 源码入口

- `src/fastwam/rl/replay_buffer.py`
- `experiments/robotwin/build_residual_rl_replay.py`
- `experiments/robotwin/merge_residual_replays.py`

---

## 13. 知识点九：复合奖励设计

### 项目奖励结构

可概括为：

```text
R = w_env × R_env
  + w_success × success_bonus × I(success)
  - w_imitation × normalized_MSE(a_exec, a_fastwam)
  + bounded_imagination_shaping
  - step_penalty × K
```

### 各项作用

- 终局成功：最终任务目标，必须占主导；
- imitation/动作偏差约束：限制 residual 偏离基础动作，兼具行为先验与保能力作用；
- imagination shaping：缓解终局成功稀疏，提供局部进展信号；
- step penalty：可鼓励效率，但不应在未校准时改变任务含义；
- environment reward：接口保留，部分配置中权重可以为零。

### 为什么要 episode shaping budget

即使单步想象奖励被 clip，长 episode 累加后仍可能超过终局成功奖励，导致策略优化“视觉上像在进步”而不是完成任务。

项目用 `EpisodeShapingBudget` 限制一个 episode 内想象奖励绝对值累计上限：

```text
budget = success_bonus × success_weight × max_imagination_to_success_ratio
```

代码要求比例不超过 0.5，从设计上保证终局成功保持主导。当前重启实验使用了更严格的规划上限：想象项整回合绝对值不超过 1.0，而成功奖励为 10。

### 源码入口

- `src/fastwam/rl/rewards.py::CompositeRewardConfig`
- `src/fastwam/rl/rewards.py::EpisodeShapingBudget`
- `src/fastwam/rl/rewards.py::compute_composite_reward`

---

## 14. 知识点十：未来状态一致性/想象奖励

### 简历里如何准确解释

不要只说“实际结果和想象越像，奖励越高”，因为静态终点相似度可能受到背景、遮挡和机械臂位置影响。更准确的表述是：

> 比较真实观测变化方向与 FastWAM 冻结想象轨迹的变化方向是否一致，并将其作为低权重、相机感知、时间对齐的辅助奖励。

### 历史实现：特征变化方向对齐

对每个相机：

```text
Δactual   = f(actual_after_K) - f(current)
Δimagined = f(imagined_goal)  - f(current)

direction = cosine(Δactual, Δimagined)
magnitude_ratio = min(||Δactual|| / ||Δimagined||, 1)
r_camera = direction × magnitude_ratio
```

然后各相机分别用训练集全局统计归一化，再融合并裁剪。

### 当前重验证：冻结规划轨迹、多时间点、Wan 表征

当前方案不再只比较一个终点，而是在同一个 action chunk 内冻结 FastWAM imagined plan，在 `0, 4, 8, 12, 16, 20, 24` 等多个时间点与真实轨迹比较，并优先验证 Wan VAE latent，之后再验证 Video Expert token。

一次 smoke 中，clean/corrected 分支得分高于受控 corruption，且三个相机方向一致；但这只是单 seed 的局部机制证据，尚不能证明成功率提升。

### 为什么不能作为全部奖励

- 视觉变化一致不保证抓取稳定；
- 模型想象本身可能错误；
- 腕部相机运动会制造大视觉变化；
- 局部一步进展不等于最终闭环成功；
- 机械臂可能朝正确方向运动，但接触姿态或时机已经不可恢复。

所以想象奖励只能是 shaping，必须由终局成功锚定，并与动作偏差约束共同使用。

### 源码与实验入口

- `src/fastwam/rl/rewards.py::compute_imagination_reward`
- `experiments/robotwin/imagination_reward_utils.py`
- `experiments/robotwin/FROZEN_PLAN_TRAJECTORY_VAE_SMOKE_20260825_RESULTS.md`
- `docs/FASTWAM_ROBOTWIN_FROZEN_PLAN_TRAJECTORY_REWARD_PLAN.md`

---

## 15. 知识点十一：Q-value improvement gate

### 计算方式

在完全相同的状态与 FastWAM baseline 条件下，两个 Q 分别比较候选动作和原动作：

```text
ΔQ_i = Q_i(s, a_candidate) - Q_i(s, a_fastwam)
```

历史 gate 的核心条件为：

```text
min(ΔQ_1, ΔQ_2) ≥ margin
abs(ΔQ_1 - ΔQ_2) ≤ max_disagreement
```

否则执行原 FastWAM 动作。

### 为什么比较 improvement 而不是绝对 Q

- 绝对 Q 在不同任务和状态间标度可能不同；
- candidate 与 baseline 在同一状态下作差，可以抵消部分公共偏差；
- 本项目目标是判断 residual 是否优于当前 FastWAM 建议，不是判断该状态本身价值高不高。

### 为什么仍然会失败

- candidate action 可能不在 replay 支持内；
- Q 的系统性误差在作差后不一定消失；
- 两个 Q 共享数据，可能共同高估；
- Bellman Q 学到的长期价值不等于同状态单次干预的真实净收益；
- 连续被 gate 放行会改变后续状态分布，单步 ΔQ 无法自然表示累计风险。

### 面试时的正确结论

> Q improvement gate 是一个合理的保守启发式和诊断工具，但严格实验表明它不能单独作为安全放行器。项目价值之一正是通过在线 pair 发现了训练分布与 actor 候选分布不一致的问题。

### 源码入口

- `src/fastwam/rl/online_policy.py::OnlineResidualPolicy.act`
- `experiments/robotwin/analyze_residual_q_gate.py`
- `experiments/robotwin/Q_GATED_RESIDUAL_RETENTION_20260729_RESULTS.md`

---

## 16. 知识点十二：kNN 状态/动作支持域检测

### 目标

kNN support gate 不预测“动作好不好”，而是回答：

> 当前状态和候选 residual 是否与成功 replay 中见过的状态/动作足够接近？

### 状态支持

状态距离联合：

- L2 归一化视觉特征距离；
- robust-normalized proprio RMS 距离；
- robust-normalized FastWAM baseline action chunk RMS 距离。

三者平方平均后开根号得到联合距离，再除以参考样本的局部半径得到密度归一化 score。

### 动作支持

先找到相近的受支持状态，再比较 candidate residual 与这些状态中实际执行 residual 的距离。动作各维会按允许的 `residual_scale` 归一化，避免不同动作维度尺度不一致。

### 语言路由

支持索引保存每个任务的语言 prototype。在线语言特征先用 cosine similarity 匹配任务；相似度不足时直接视为不受支持，避免拿一个任务的数据为另一个指令放行。

### 为什么只从成功 episode 建索引

目的是把“受支持”定义为靠近成功行为分布，而不是简单靠近任意收集数据。但代价是成功数据稀少时覆盖率可能很低，gate 会过度保守。

### 能做与不能做

能做：

- 拒绝明显远离训练分布的状态；
- 拒绝训练中未出现过的大 residual；
- 在不确定时回退到 FastWAM。

不能做：

- 判断一个 in-support 动作是否会导致失败；
- 证明邻近成功样本就一定安全；
- 识别所有连续小干预的累积闭环风险；
- 主动产生 rescue。

### 项目观察

`open_laptop` 的失败轨迹仍处于校准支持域内，因此失败不是经典距离型 OOD，而是连续 residual 改变闭环轨迹后的累积误差。这个结果说明 kNN 是 support detector，不是 outcome predictor。

### 源码入口

- `src/fastwam/rl/support_gate.py::ResidualSupportIndex`
- `experiments/robotwin/build_residual_support_index.py`
- `experiments/robotwin/OOD_SUPPORT_GATE_AUDIT_20260729_RESULTS.md`

---

## 17. 知识点十三：episode 级干预预算

### 机制

设置每个 episode 最多实际执行多少个 residual chunk：

```text
if intervention_count < max_interventions:
    candidate 可继续参与最终仲裁
else:
    回退 baseline
```

只有 residual 真正被执行时才计数；Q/support 拒绝、shadow candidate 和不符合干预位置的候选不消耗预算。每个新 episode 重置。

### 为什么需要

有界 residual 只保证单次修正小，不保证多次修正的总影响小。一次 residual 可能无害，连续 20 次小修正却可能让状态分布逐渐偏离 FastWAM 的闭环吸引域。

### 历史结果

某次 `open_laptop` audit 中：

- 不限干预时一条原本成功的 baseline 轨迹失败；
- 单次或前两次干预仍成功；
- 固定最多两次干预后，小规模高成功任务保能力测试达到 15/15。

但“2 次”只由一个失败 seed 支持，是开发期启发式，不是全任务最优结论。后续又发现固定次数会压低 rescue recall，所以当前不能把预算描述为最终解决方案。

### 与 reward shaping budget 的区别

- intervention budget：限制在线执行 residual 的次数；
- shaping budget：限制训练奖励中想象 shaping 的累计绝对值；
- 两者都在 episode 级工作，但控制对象完全不同。

### 源码入口

- `src/fastwam/rl/online_policy.py`
- `src/fastwam/rl/rewards.py::EpisodeShapingBudget`

---

## 18. 知识点十四：Q 门控是否是因果门控

答案：**不是。**

Q gate 估计的是离线模型中的价值差：

```text
Q(s, candidate) - Q(s, baseline)
```

因果问题需要比较同一前缀状态下两个干预分支的真实终局结果：

```text
P(success | do(candidate)) - P(success | do(baseline))
```

项目使用相同 environment seed、指令、初始观测、proprio、干预前观测和 FastWAM 动作哈希建立匹配 pair，并在固定 replan 只干预一次，再让两条分支各自重新进入 FastWAM 闭环。

标签包括：

- rescue：baseline 失败，candidate 成功；
- regression：baseline 成功，candidate 失败；
- neutral success：两者成功；
- neutral failure：两者失败。

需要保留的严谨边界：RoboTwin 当前并非对 simulator memory 做任意时刻的精确状态克隆，所以这是严格审计后的配对反事实近似，不应声称形式化证明了因果效应。

---

## 19. 知识点十五：为什么连续修正会退化

### 闭环分布漂移

FastWAM 是闭环策略。Residual 在时刻 t 改变动作后，不只改变当前一步，还改变后续观测：

```text
s_t --residual--> s_(t+1)' --FastWAM--> a_(t+1)'
```

后续 FastWAM 看到的是不同状态，输出也会变化。如果每次 replan 都继续 residual，误差可能自我强化。

### 为什么单步幅度限制不够

`||r_t||` 很小不意味着：

```text
Σ_t effect(r_t)
```

很小。尤其在接触、抓取和临界几何状态中，微小动作可能造成不可逆后果。

### 项目尝试过的控制手段

- Q advantage margin；
- Twin-Q disagreement；
- kNN support；
- soft residual scaling；
- cumulative risk heuristic；
- episode intervention budget；
- residual 后强制 re-anchor/outcome confirmation；
- 单次干预的严格配对数据。

最终经验是：启发式门控可以减少部分 regression，但如果训练数据中缺少 actor-aligned rescue/regression，继续调阈值无法根治问题。

---

## 20. 知识点十六：实验设计与指标

### 为什么必须 paired evaluation

机器人任务成功受初始物体位置、语言模板、推理随机性和仿真 seed 影响。若 baseline 与 residual 使用不同初始状态，成功率差异可能只是随机难度差异。

因此必须固定并审计：

- environment seed；
- 实际使用的语言 instruction；
- 初始观测/proprio；
- FastWAM checkpoint 与推理设置；
- 干预前 FastWAM action；
- action chunk 长度和 episode horizon。

### 关键指标

- overall success rate；
- rescue 数量；
- regression 数量；
- paired net gain = rescue − regression；
- gate approval rate；
- regression false-approval rate；
- rescue recall；
- gate AUROC/balanced accuracy；
- 不同训练 seed 的稳定性；
- 按任务而不是只看混合总体结果。

### 为什么 reward/Q 排序不能代替在线成功率

Reward 和 Q 都是训练或选择代理。最终系统目标是环境成功，必须通过在线闭环、同 seed 配对结果验证。只报告 loss 下降、Q 增大或 reward 更高不能证明策略改进。

---

## 21. 已有实验结论应如何陈述

### 可以说的

- Residual actor 确实能产生非零且在少数严格 pair 中造成真实 rescue，说明候选空间中存在有用修正；
- always-on residual 会破坏 FastWAM 原本成功的轨迹，说明必须控制干预时机；
- Q+OOD 在一组四任务小样本中把 common-seed 结果从 baseline 10/18 提到 11/18，并消除了该样本中的 regression，但证据不足以形成显著提升结论；
- 更严格的任务测试中 Q/OOD 发生过明显 regression，说明上述小样本结果不能泛化；
- 双 Q 一致不能消除共同 OOD 偏差；
- kNN 支持域可以作为回退保护，但不能判断所有 in-support 候选的真实结果；
- 想象奖励能识别部分受控破坏，但仍需在 outcome-discordant pair 上证明与任务成功一致；
- 当前最可信的瓶颈不是单纯模型参数量，而是 actor-aligned 数据、奖励信用分配和门控校准。

### 不能说的

- “Residual IQL 已稳定提升 FastWAM 成功率”；
- “Q+OOD 已解决离线价值高估”；
- “kNN gate 能保证动作安全”；
- “未来状态一致性奖励已经证明可提升策略”；
- “episode 最多两次 residual 是全任务最优设置”；
- “项目已完成论文级统计验证”。

---

## 22. 当前研究状态

截至 2026-08-27，当前主路线是：

1. 保持论文对齐的 FastWAM 设置：10 次去噪、24-action chunk、官方 unseen instruction；
2. 在中等难度任务上冻结一次 FastWAM imagined trajectory；
3. 对 clean、bounded corruption 和 correction 分支做多时间点轨迹对齐；
4. 优先用 Wan VAE latent 验证想象奖励，再考虑 Video Expert token；
5. 只有当奖励在足够多 outcome-discordant pair 上通过预注册阈值，才训练 residual；
6. 训练时使用同一 replay 的 no-imagination/with-imagination 公平对照；
7. 当前计划使用无 Q 的 Monte-Carlo AWR 隔离奖励贡献，暂不把 Q/OOD gate 混入该实验。

这不否定简历里的 Residual IQL 工作，而是说明项目经过失败分析后主动缩小问题、重建可证伪实验链路。

---

## 23. 高频面试问题与参考回答

### 23.1 你的核心贡献是什么

> 我完成的不只是一个 residual MLP，而是从数据采集、版本化 replay、复合奖励、Residual IQL、在线门控到严格 paired evaluation 的完整链路。更重要的是，我通过反例定位到 Twin-Q 与 kNN 的能力边界：两个 Q 在 actor 候选缺少覆盖时会共同高估，距离型 OOD 也识别不了所有连续干预风险。之后我把实验收缩为先独立验证想象奖励和 actor-aligned 配对数据，避免继续调门控阈值掩盖数据问题。

### 23.2 为什么 IQL 比 CQL 或 SAC 更适合这一步

> SAC 更偏在线，直接对 actor 动作最大化 Q，在离线数据下容易利用 Q 的外推误差。CQL 显式压低数据外动作价值，但实现和调参更复杂，也可能过度保守。IQL 不需要在 Bellman target 中显式评估数据外动作，通过 expectile V 和优势加权回归保持在数据支持附近，适合作为资源可控的第一版离线 RL。项目结果也说明 IQL 仍不自动解决 actor 候选覆盖，因此后面才增加支持域与 paired audit。

### 23.3 Twin-Q 与 Double DQN 有什么区别

> 两者都在处理高估，但机制不同。这里是两个连续动作 Q critic，对相同 `(s,a)` 取较小值，接近 clipped double Q；Double DQN 则分离动作选择与动作评估，最初用于离散动作。不能简单把项目里的 Twin-Q 称为 Double DQN。

### 23.4 为什么 Q 的 target 使用 V 而不是下一状态 actor 动作的 Q

> 这是 IQL 的关键。使用 `V(s')` 可以避免在 target 里查询 actor 生成的潜在 OOD 动作 Q；V 是从数据动作的 Q 分布通过 expectile 学出来的。

### 23.5 为什么 actor 还要输入 baseline action

> 相同视觉状态下，residual 的含义依赖 FastWAM 原本打算做什么。输入 baseline action 后，actor 可以学习“针对当前 base proposal 修正多少”，而不是仅从视觉重新学习一套完整策略。

### 23.6 想象奖励与世界模型 reward 有什么不同

> 它不是额外训练一个 reward model，而是利用 FastWAM 已经生成的 imagined future，比较真实变化和想象变化的一致性。它是一种由预训练世界知识派生的 shaping signal，最终仍由环境成功标签锚定。

### 23.7 为什么按相机分别归一化

> 头部相机和腕部相机的视角、运动幅度和遮挡分布不同，原始 reward 标度不能直接比较。按相机用训练集全局统计归一化后再统一融合，既保留统一模型，又避免某个天然方差更大的相机支配奖励。

### 23.8 kNN 为什么同时看状态和动作

> 状态在分布内不代表候选动作在分布内。例如在熟悉场景中 actor 仍可能输出训练中没见过的 residual。状态支持和动作支持都通过，才说明该 `(s,a)` 组合有一定数据覆盖。

### 23.9 episode budget 为什么不是最好的最终方案

> 它能限制连续干预风险，但不知道哪一次干预最关键，也可能把后期真正有用的 rescue 拒绝掉。它适合开发期保守保护和失败定位，最终更希望由有 actor-aligned 因果监督的 gate 决定干预时机。

### 23.10 最近的问题都是数据问题吗

> 不能简单归结为数据量。更准确的是数据分布和标签质量问题：历史数据有专家和受控破坏，但缺少当前 actor 候选在同状态下的 rescue/regression；部分标签又是终局结果复制到多 transition，信用分配较粗。模型结构也可能限制细粒度接触判断，但容量实验没有显示单纯扩大 MLP能解决 held-out 排序，所以当前优先修复数据与实验协议。

### 23.11 为什么现在又从 IQL 回到 AWR

> 不是认为 IQL 没价值，而是当前要单独回答“想象奖励是否有增益”。如果同时引入 Twin-Q、Q gate 和 OOD gate，结果变化无法归因。Monte-Carlo AWR 不训练 action Q，也不在部署时门控，适合用同一数据做 no-imagination/with-imagination 的最小公平实验。奖励成立后再重新引入 IQL 更合理。

### 23.12 项目最大的负面结果是什么

> 更严格的 paired test 中出现了 Q 拒绝真实 rescue、却批准 regression 的反例，而且某些连续干预失败仍处在 kNN 支持域内。这否定了“双 Q 一致加 kNN 就足够安全”的早期假设，也推动我把 Q 降级为辅助信息，并重新设计 actor-aligned 单次干预数据和奖励验证协议。

---

## 24. 面试前速记卡

### IQL 三个 loss

```text
V: expectile regression to min target-Q on dataset actions
Q: MSE to r + γ^K × bootstrap_mask × V(next_state)
Actor: exp(temperature × (Q−V)) weighted behavior cloning
```

### Residual actor

```text
input = visual feature + proprio + language + FastWAM action chunk
output = tanh(MLP) × per-dimension scale
candidate = clip(base + residual)
zero output init; gripper residual = 0
```

### 三种保护机制

```text
Q gate:      预测 candidate 是否比 baseline 价值高
kNN support: 检查 state/action 是否有数据支持
budget:      限制一回合实际 residual 次数
```

### 三者局限

```text
Q gate ≠ 因果收益
kNN support ≠ 成功预测
intervention budget ≠ 最优时机选择
```

### 项目最重要的研究判断

```text
有界 residual 不等于闭环安全；
双 Q 一致不等于价值正确；
受控破坏可排序不等于细粒度修正可排序；
离线指标必须由同 seed 在线成功率验证。
```

---

## 25. 源码与实验索引

| 主题 | 主要文件 |
|---|---|
| Residual actor / Q / V 结构 | `src/fastwam/rl/models.py` |
| IQL loss 与训练循环 | `src/fastwam/rl/iql_trainer.py` |
| AWR 与 context 构造 | `src/fastwam/rl/awr_trainer.py` |
| Replay schema 与 reward relabel | `src/fastwam/rl/replay_buffer.py` |
| 复合奖励与 shaping budget | `src/fastwam/rl/rewards.py` |
| 在线 Q gate、预算和 residual 仲裁 | `src/fastwam/rl/online_policy.py` |
| kNN 支持域 | `src/fastwam/rl/support_gate.py` |
| 单次干预分类门控 | `src/fastwam/rl/intervention_gate.py` |
| RoboTwin replay 构建 | `experiments/robotwin/build_residual_rl_replay.py` |
| 支持索引构建和校准 | `experiments/robotwin/build_residual_support_index.py` |
| RoboTwin 在线部署接入 | `experiments/robotwin/fastwam_policy/deploy_policy.py` |
| 四任务 IQL 在线结果 | `experiments/robotwin/ROBOTWIN_4TASK_IQL_FULL_ONLINE_20260804_RESULTS.md` |
| OOD 与累计干预审计 | `experiments/robotwin/OOD_SUPPORT_GATE_AUDIT_20260729_RESULTS.md` |
| 当前奖励重启方案 | `docs/FASTWAM_ROBOTWIN_IMAGINATION_REWARD_RESTART_PLAN.md` |
| 当前冻结想象轨迹方案 | `docs/FASTWAM_ROBOTWIN_FROZEN_PLAN_TRAJECTORY_REWARD_PLAN.md` |

---

## 26. 本次核验执行记录

核验时检查了上述核心源码、当前实验规划、历史在线结果和相关单元测试。使用 RoboTwin FastWAM 环境执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  /home/ubuntu/miniconda3/envs/robotwin_fastwam/bin/python -m pytest -q \
  tests/test_residual_iql.py \
  tests/test_rl_rewards.py \
  tests/test_support_gate.py \
  tests/test_online_residual_policy.py \
  tests/test_residual_language_routing.py
```

结果：`34 passed`。

这说明简历涉及的核心实现目前仍可通过对应单元测试；它不等同于已经证明在线成功率提升或部署安全。
