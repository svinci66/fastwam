# FastWAM 强化学习后训练：阶段总结、问题复盘与后续方案

> **路线更新（2026-08-24）：** 下文关于 Twin-Q、OOD gate 和因果门控的设计保留为历史复盘，但不再是当前主实验。当前先在中等难度 RoboTwin 任务上重新验证想象奖励，并用相同数据训练 `no-imagination` / `with-imagination` 两个无 Q residual actor，随后做小规模在线配对评测和失败视频人工分析。当前执行方案见 [FASTWAM_ROBOTWIN_IMAGINATION_REWARD_RESTART_PLAN.md](./FASTWAM_ROBOTWIN_IMAGINATION_REWARD_RESTART_PLAN.md)。

更新日期：2026-08-24

## 第一部分：展示用要点摘要

### 1. 项目目标

- 以官方 FastWAM 为冻结基础策略，使用 Residual RL 对动作块进行小幅修正。
- 在 FastWAM 失败的中等难度任务上获得额外成功，同时不破坏其原有成功能力。
- 利用 FastWAM 的未来想象构造辅助奖励，并用因果门控和 OOD 检测控制 residual 的在线干预风险。
- 第一阶段先证明 `FastWAM + Residual` 系统级改进有效；后续再考虑将能力蒸馏或合并回 FastWAM 主模型。

### 2. 当前系统结构

- **FastWAM baseline**：保持冻结，负责生成主要动作块和持续闭环重规划。
- **Residual actor**：根据当前视觉、proprio、语言和 FastWAM 原动作，输出幅度受限的动作修正。
- **IQL critic**：离线学习状态、FastWAM 动作和 residual 候选动作的长期价值。
- **想象奖励**：比较 FastWAM 想象的未来与动作执行后的真实观测，作为密集辅助信号。
- **直接因果门控**：使用同状态单次干预的 `rescue/regression` 配对，预测 residual 候选是否比 FastWAM 原动作带来更高的净成功概率，是部署时的主要批准依据。
- **OOD gate**：作为不可绕过的安全否决器，拒绝训练支持范围外的状态或 residual 动作，使系统回退到 FastWAM。
- **Twin-Q 辅助判断**：估计 candidate-vs-baseline 长期价值差，只作为门控特征、置信度或诊断量，不再独立批准 residual。
- **动作仲裁器**：仅在因果门控批准且 OOD 检查通过时执行 residual，否则保持 FastWAM 原动作。

### 3. 已经得到的关键结论

- Residual actor 能产生真实有效的修正，并非完全没有动作能力。
- Residual 是否有效高度依赖当前状态和干预时机，不能始终开启。
- 现有 Q 门控在 actor 在线候选分布上泛化较差，曾拒绝真实 rescue、批准 regression。
- 想象奖励可以识别明显动作破坏，但不能单独代表最终任务成功。
- OOD gate 能阻止明显不受支持的动作，但只能保护 baseline，不能主动创造提升。
- 当前证据不支持优先扩大 actor；更直接的瓶颈是 actor-aligned 因果数据、信用分配和门控校准。

### 4. 之前失败的主要原因

1. **早期评测协议不完全一致**：推理步数、语言指令、seed、初始状态和预处理曾出现不一致。
2. **训练与部署动作分布不一致**：训练数据主要是专家和人工破坏动作，缺少 residual actor 在线候选。
3. **奖励目标不完全一致**：局部视觉进步不一定带来最终任务成功。
4. **重复干预存在累积风险**：单次修正可能有效，连续修正可能让轨迹逐渐偏离 FastWAM。
5. **有效标签不独立**：部分历史样本来自少数 seed 或把 episode 标签复制给多个 transition，容易产生过拟合。

### 5. 修正后的核心方案

- 先验证 FastWAM 与 zero-residual wrapper 在每次重规划上 100% 等价。
- 从完全相同的 FastWAM 状态建立 baseline 和单次 residual 两条分支。
- 一个动作块后，两条分支都重新交还给实时 FastWAM 闭环运行到结束。
- 直接标注 `rescue`、`regression` 和 neutral，而不是只使用间接 Q 值或局部视觉奖励。
- 用这些 actor-aligned 因果数据重新训练 IQL actor、critic 和 candidate-vs-baseline 门控。
- 先通过高成功率任务的安全保留测试，再扩大到中等难度任务的成功率提升测试。

### 6. 硬停止标准

- 协议、初始状态或 FastWAM 动作无法 100% 配对：该批实验无效，修复后重跑。
- 40 对有效因果样本后仍没有跨 seed rescue：停止该任务采集，修改候选动作或任务选择。
- 门控验证 AUROC 低于 0.70，或 regression 错误批准率超过 10%：禁止进入在线部署。
- 10 回合准入测试中 regression 不少于 rescue：停止扩大评测，回到数据和门控训练。
- 高成功率任务损失超过 5 个百分点：当前方法不满足“不损失原能力”的目标。
- 只有工程异常或协议错误才原样重跑；指标失败不能只换随机 seed 重复同一配置。

### 7. 希望导师重点指导的问题

1. 第一阶段应定位为“FastWAM + 安全 Residual-RL 系统”，还是必须最终把 residual 合并回 FastWAM 参数？
2. 论文贡献应更突出想象奖励，还是突出 actor-aligned 因果数据和安全干预门控？
3. 每任务 30 个严格配对 seed 是否足以作为初步结果，还是应直接扩大到 50--100 个 seed？
4. 在已有 actor 能产生 rescue 的情况下，是否认可先解决数据和选择问题，再做 LSTM、patch token 或更大模型？

---

## 第二部分：项目背景与研究动机

### 1. 为什么选择 FastWAM

FastWAM 同时生成机器人动作和与任务相关的未来视频想象。相比只预测动作的普通 VLA，它额外提供了一个可利用的世界模型信号：模型不仅给出“下一步怎么做”，还表达了“执行任务后未来可能是什么样子”。

本项目最初的想法是：如果动作执行后的真实结果与 FastWAM 想象的未来更接近，说明动作可能正在沿着模型认为正确的方向推进；如果偏差较大，则可能需要降低该动作的价值。因此，可以将想象与真实结果的差异作为强化学习奖励的一部分。

但 FastWAM 本身已经具有较强能力。直接替换其动作、全量微调主干或始终施加 residual 都可能引入灾难性退化。因此，本项目选择冻结 FastWAM，只训练一个幅度较小的 residual 分支，并在部署时增加安全门控。

### 2. 研究问题

项目需要回答三个问题：

1. FastWAM 的未来想象能否提供与任务进展相关的密集奖励？
2. 离线强化学习能否学到比 FastWAM 原动作更好的小幅修正？
3. 如何只在 residual 真正有帮助时干预，同时保护 FastWAM 已有成功能力？

### 3. 当前方法的总体数据流

```text
当前三相机观测 + proprio + 语言指令
                  │
                  ├──────────────► 冻结 FastWAM ──► baseline action chunk
                  │                         │
                  │                         └──────► imagined future video
                  │
                  └──► Residual actor + baseline action ──► candidate action chunk
                                                         │
                                   因果门控 + OOD + Q 辅助判断
                                                         │
                        ┌────────────────────────────────┴──────────────┐
                        │                                               │
                     批准候选                                        拒绝候选
                        │                                               │
               执行 FastWAM + residual                          执行原 FastWAM
                        │                                               │
                        └──────────► 下一次重新调用 FastWAM ◄──────────┘
```

---

## 第三部分：已经完成的工作

### 1. 奖励设计与验证

已经实现并验证了以下奖励组成：

- 终局任务成功奖励。
- 与 FastWAM 动作保持接近的模仿/偏差约束。
- 想象未来和实际未来之间的视觉进展奖励。
- 多相机分别统计、全局统一使用的归一化参数。
- episode 级想象奖励预算，避免密集 shaping 覆盖最终成功奖励。

受控 hold corruption 实验表明，想象奖励能够稳定区分正常动作、轻度破坏和重度破坏，说明它确实包含一定的任务进展信息。

### 2. Residual actor 与 IQL

当前 residual actor：

- 输入视觉特征、proprio、语言特征和 FastWAM 原动作块。
- 使用有界 MLP 预测动作残差。
- 输出层零初始化，使训练开始时 residual 接近零。
- residual 最大尺度保持较小，gripper 维度由 FastWAM 保留控制权。

当前 IQL 实现包含：

- 两个 action-value critic。
- 一个 expectile value critic。
- target Q 软更新。
- 使用 `Q - V` 优势权重训练 residual actor。
- Bellman target 不需要直接查询数据分布外动作。

### 3. Q、OOD 和因果门控

已经实现并测试：

- Twin-Q candidate-vs-baseline advantage。
- 两个 Q 的分歧阈值。
- 状态和 residual 动作的 kNN 支持度检查。
- OOD circuit breaker。
- residual 软缩放和累积风险启发式。
- 单次干预后的 FastWAM re-anchor。
- 直接使用严格单次干预 pair 训练的 candidate-vs-baseline 分类门控。

三者不是同一个模块：

- Q 门控比较两个 IQL critic 对 `FastWAM + residual` 与 FastWAM 原动作的预测价值；它依赖 replay 覆盖，可能在 OOD 候选上共同高估，因此不是因果门控。
- 因果门控直接使用同一前缀下的 `rescue/regression/neutral` 终局差异训练，目标是判断“执行这次 residual 相对不执行是否产生净收益”。
- OOD gate 不负责发现 rescue，只负责否决没有数据支持的状态和动作。

历史版本曾以 `Twin-Q + OOD` 作为主要在线批准链路。由于后续严格 pair 出现“Q 拒绝 rescue、批准 regression”的反例，当前方案已将 Q 降级为辅助信息：高 Q 不能绕过因果门控或 OOD 拒绝，两个 Q 意见一致也不能单独作为安全证明。

当前 Stage 1 单次干预采集会关闭 Q、因果和 OOD 门控，在预注册位置强制执行一次 residual，专门生成门控训练需要的正反例；离线准入通过后，因果门控和 OOD gate 才会重新接入正式在线评测。

### 4. 工程和评测修正

已经修正或增加：

- 论文对齐的 10 次去噪和 24 步重规划。
- 官方 unseen instruction。
- deterministic instruction by seed。
- 版本化环境 seed manifest。
- 初始观测、指令和动作哈希审计。
- 修复 expert feasibility check 导致的 seed 静默推进。
- 清理不完整和 stale replay 目录。
- 统一 online/replay 的图像尺寸、相机切分和 bf16 编码。
- 归一化统计只由训练 replay 计算并冻结，避免测试泄漏。

---

## 第四部分：RoboTwin 失败结果的详细复盘

### 1. 早期结果为什么不能全部用于结论

早期部分实验存在以下协议漂移：

- FastWAM 推理曾使用少于论文设置的去噪步数。
- 部分 residual 评测使用固定或生成指令，而 baseline 使用官方指令。
- RoboTwin expert check 可能推进随机 seed，造成表面相同但实际初始场景不同。
- replay 和 online 图像预处理曾不完全一致。
- 部分 probe 共享同一个 baseline episode，不能当作独立成功率样本。

这些问题后来已经逐项修复。因此，当前判断主要依据修正后、论文对齐、初始状态严格配对的实验。

### 2. Ungated residual 的失败

在四任务共同 seed 评测中：

| 方法 | 成功 | Rescue | Regression |
|---|---:|---:|---:|
| FastWAM | 10/18 | — | — |
| 始终执行 residual | 6/18 | 1 | 5 |
| Q+OOD residual | 11/18 | 1 | 0 |

该结果说明：

- actor 偶尔可以修复 FastWAM 失败。
- always-on residual 会破坏更多原本成功的轨迹，因此不可部署。
- Q+OOD 在这组小样本中保护了 baseline，但只有一个新增成功，不能形成稳定提升结论。

### 3. 更严格评测暴露了门控不稳定

在严格配对的 `hanging_mug` 五 seed 实验中：

- FastWAM baseline：3/5。
- Residual + Q/OOD + outcome confirmation：0/5。
- 三个 FastWAM 成功回合全部发生 regression。

这说明之前四任务中的单个提升不能泛化。即使加入 Q、OOD、软缩放和局部结果确认，也没有可靠识别有害修正。

### 4. Actor 有能力，但 Q 排序方向错误

在 `place_can_basket` 严格单次干预实验中：

- 同一个 FastWAM 失败 seed，在三个不同 replan 上被 residual 修复为成功。
- 另一个 FastWAM 成功 seed，在三个 replan 上被 residual 破坏为失败。
- 三个 rescue 全部被当前 Q 拒绝。
- 唯一被 Q 批准的候选造成了 regression。
- 小样本 rescue-vs-regression Q 排序 AUC 为 0.333。

因此，当前 actor 的假设空间中已经存在有用动作，问题主要是 Q 和 gate 没有学会在正确状态批准它们。

### 5. 为什么 Q 会失败

旧 replay 主要包含：

- FastWAM policy transitions。
- 专家成功轨迹。
- hold、gripper delay 等人工破坏轨迹。

但目标任务中缺少当前 residual actor 真正提出并执行的 candidate action。因此 critic 在训练时看到的动作分布与部署时不同。两个 Q 网络即使结构独立，也可能在同一份缺少覆盖的数据上产生相关的错误自信。

这也是为什么只调整 Q margin、增加第二个 Q 或扩大 critic 参数量不能从根本上解决问题。

### 6. 想象奖励为什么不能单独使用

受控破坏实验中，想象奖励可以识别大幅错误动作；但在细粒度 residual 修正上存在直接反例：

- 某个单次 residual 的局部想象进度为正。
- 但它将原本成功的 FastWAM 轨迹变成失败。

想象奖励衡量的是短期视觉变化是否朝向模型想象，而机器人任务成功还取决于：

- 抓取是否稳定。
- 接触力和姿态是否正确。
- 动作发生的具体时机。
- 后续轨迹是否仍可恢复。
- 多相机视角中哪些变化真正与任务目标相关。

因此，想象奖励应保留为低权重密集 shaping；终局成功和同状态 FastWAM 对照结果必须占主导。

### 7. 为什么现在不优先扩大 actor

对当前 gate 的固定数据容量实验显示：

- 当前小网络可以完全记忆训练数据。
- 扩大到更深、更宽网络后，held-out 排序没有稳定改善。
- 当前状态输入加入短历史后更保守，但拒绝了全部 held-out rescue。

这不能证明当前视觉表示永远足够，但可以排除“当前失败只是网络太小”的简单解释。在得到更多独立、actor-aligned 的因果监督前，比较 MLP、LSTM 或更大网络没有可靠基础。

---

## 第五部分：修正后的实验假设

### 1. 旧假设

旧实验隐含假设是：

> 在专家和人工破坏 replay 上学到的 IQL Q 值，可以直接判断 residual actor 在线提出的候选动作是否优于 FastWAM。

严格实验已经表明，这个假设目前不成立。

### 2. 新假设

修正后的假设是：

> 如果训练数据直接覆盖 residual actor 在 FastWAM 闭环中提出的候选动作，并通过相同状态下的单次干预对照获得 rescue/regression 因果标签，那么 residual actor 和安全门控可以学会选择少量真正有帮助的修正，同时拒绝会破坏 FastWAM 的修正。

### 3. 想象奖励在新假设中的位置

想象奖励不再承担“直接决定是否执行 residual”的责任，而承担：

- 在终局成功奖励稀疏时提供局部进展信息。
- 在相同终局结果的 pair 中辅助区分更合理的中间状态。
- 为后续研究 FastWAM 世界模型是否能改善 RL sample efficiency 提供可消融组件。

---

## 第六部分：详细实验方案

完整、可执行的实验方案和所有阈值保存在：

`docs/FASTWAM_ROBOTWIN_CAUSAL_RESIDUAL_EXPERIMENT_PLAN.md`

### 阶段 0：FastWAM baseline 等价审计

运行 FastWAM baseline 和 residual shadow wrapper：

- 官方 FastWAM checkpoint。
- 10 次去噪。
- 24 步重规划。
- 官方 unseen instruction。
- 相同环境 seed manifest。

逐 replan 比较：

- instruction 哈希。
- 当前观测哈希。
- FastWAM baseline action 哈希。
- 实际执行 action 哈希。
- 最终成功结果。

只有所有字段 100% 一致才进入后续阶段。

### 阶段 1：actor-aligned 单次干预采集

在同一个 FastWAM 状态建立：

- Baseline：执行 FastWAM 原动作块。
- Candidate：执行 FastWAM + 一个 residual 动作块。

执行一个动作块后，两条分支都重新交给 FastWAM 闭环，不允许继续回放干预前保存的未来动作。

每个主要任务的数据目标：

- 至少 60 个有效 pair。
- 至少 8 个 rescue，覆盖至少 4 个环境 seed。
- 至少 8 个 regression。
- 至少 15 个 neutral/OOD 样本。

### 阶段 2：重新训练

Residual actor 保持零初始化、小尺度和 gripper ownership，不先扩大结构。

奖励优先级：

1. 终局成功奖励。
2. 相对 FastWAM 的因果 rescue/regression 监督。
3. 动作偏差约束。
4. 低权重想象奖励。

门控部署顺序：

1. 直接因果 candidate-vs-baseline gate。
2. OOD 支持检查。
3. Twin-Q 辅助置信度。

### 阶段 3：10 回合准入测试

任务：

- 高成功率保能力任务：`adjust_bottle`。
- 中等难度目标任务：`hanging_mug`、`place_can_basket`。
- OOD 诊断任务：`open_microwave`。

只有满足以下条件才扩大：

- 高成功率任务最多一个 regression。
- 两个目标任务合计至少两个 rescue。
- 两个目标任务合计最多一个 regression。
- `rescue - regression > 0`。

### 阶段 4：正式评测

- 每个任务至少 30 个独立 held-out seed。
- 两个目标任务合计成功率至少提高 10 个百分点。
- 每个目标任务均不能净下降。
- Baseline 成功轨迹 regression 率不超过 5%。
- 高成功率任务下降不超过 5 个百分点。
- 论文级结果再扩大到每任务 50--100 个 seed，并报告置信区间。

---

## 第七部分：当前执行状态

当前正在执行阶段 0：

- 已增加逐 replan 的轻量哈希记录，不需要保存昂贵的想象视频。
- 先运行四任务各一个 seed 的 smoke。
- Smoke 100% 通过后自动扩大到四任务各五个 seed。
- 任一状态、动作、指令或结果不一致都会硬停止，不会自动进入数据采集。

该阶段的意义是确保后续所有 residual 结果都相对于一个没有被 wrapper、语言路由或评测代码改变的官方 FastWAM baseline。

---

## 第八部分：希望导师帮助决策的研究方向

### 问题 1：最终贡献形态

两种可能路线：

- **系统级贡献**：冻结 FastWAM，加 Residual-RL 和安全因果门控。
- **模型级贡献**：系统级验证后，将 residual 蒸馏为 FastWAM adapter、residual head 或 LoRA 后训练。

希望导师判断第一种是否足以作为阶段性论文主线，还是必须完成第二种才能形成完整贡献。

### 问题 2：论文技术重点

目前有两个候选重点：

- FastWAM imagination-guided reward。
- Actor-aligned causal residual learning and safe intervention gating。

现有结果显示，想象奖励更适合作为辅助创新点，而因果数据和安全门控是决定方法是否真正提升 FastWAM 的核心。希望导师判断论文叙事是否应以第二点为主、第一点为辅助。

### 问题 3：实验规模

希望确认：

- 初步验证是否接受每任务 10 个 seed。
- 阶段性正式结果是否接受每任务 30 个严格配对 seed。
- 最终投稿是否应固定为每任务 50--100 个 seed，并扩大到更多 RoboTwin 任务。

### 问题 4：何时修改模型结构

当前 actor 已经产生过真实 rescue，而 gate 缺少跨 seed 泛化。希望确认是否认可以下顺序：

1. 先补充 actor-aligned 因果数据。
2. 重新训练并验证 gate。
3. 数据达到门槛后再比较 MLP、patch token、LSTM/GRU。
4. 最后再决定是否微调 FastWAM 主干。

---

## 第九部分：口头展示建议

如果只有两分钟，可以按照以下顺序：

1. **目标**：用安全 residual-RL 改进 FastWAM，同时不损失原能力。
2. **现象**：actor 能产生 rescue，但错误时机也会造成 regression。
3. **根因**：训练数据不覆盖 actor 在线候选，Q 和想象奖励无法可靠判断最终结果。
4. **修正**：收集相同状态下 FastWAM-vs-residual 的单次干预因果 pair。
5. **验证**：先保能力，再测提升；regression 不少于 rescue 就停止。
6. **请导师判断**：贡献定位、论文重点和所需实验规模。

一句话版本：

> 当前已经证明 residual actor 能修复 FastWAM 的个别失败，但现有 Q 和想象奖励不能可靠判断干预时机；下一步将在严格论文设置下收集 FastWAM 原生、actor-aligned 的单次干预因果数据，重新训练安全门控，并用“目标任务净提升且高成功率任务不退化”作为硬标准。
