# FastWAM RoboTwin 因果 Residual-RL 实验方案

更新日期：2026-08-24

> **状态：已暂停，不再作为当前执行方案。** 2026-08-24 起，项目先回到最小研究问题：在中等难度 RoboTwin 任务上重新验证想象奖励，并比较不加/加入想象奖励的 Residual actor。Twin-Q、IQL critic、OOD gate 和因果门控均退出本轮实验。当前方案见 [FASTWAM_ROBOTWIN_IMAGINATION_REWARD_RESTART_PLAN.md](./FASTWAM_ROBOTWIN_IMAGINATION_REWARD_RESTART_PLAN.md)。下文仅作为历史设计保留。

## 1. 实验目标

在论文对齐的 RoboTwin 闭环中，验证由冻结 FastWAM、Residual actor 和安全门控组成的策略能修复部分 FastWAM 失败，同时基本不破坏 FastWAM 已成功的轨迹。

本阶段先证明系统级的 `FastWAM + Residual` 有效，不立即微调 FastWAM 主干。只有系统级结果通过正式标准后，才考虑把 residual 能力蒸馏为 FastWAM adapter、residual head 或 LoRA 后训练。

## 2. 当前系统结构与门控职责

完整推理结构如下：

```text
三相机观测 + proprio + 官方语言指令
                    │
                    ▼
              冻结的 FastWAM
                    │
                    ├──► baseline action chunk
                    └──► imagined future（仅用于训练辅助奖励）
                    │
                    ▼
              Residual actor
                    │
                    └──► candidate = baseline + bounded residual
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
               直接因果门控        OOD gate          Twin-Q
               主要批准依据         安全否决器         辅助置信度
                    └─────────────────┼─────────────────┘
                                      ▼
                         执行 candidate 或回退 baseline
                                      │
                                      ▼
                              下一次 FastWAM replan
```

各模块的边界必须明确：

- **Residual actor** 只生成有界候选修正，不决定是否执行；运动维度最大修正为 `0.05`，gripper residual 固定为零。
- **直接因果门控** 是部署时的主要批准依据。它使用同状态单次干预 pair 的 `rescue`、`regression` 和 neutral 结果，学习 candidate 相对 baseline 的净成功收益。
- **OOD gate** 是独立安全否决器。状态或 residual 动作缺少训练支持时必须回退到 FastWAM，即使其他模块给出高分也不能绕过。
- **Twin-Q** 估计长期价值差，作为门控输入、置信度或诊断量保留，但不得单独批准 residual，也不得覆盖因果门控和 OOD 拒绝。
- **动作仲裁器** 只有在因果门控批准且 OOD 检查通过时才允许执行 residual；其余情况保持原始 FastWAM 动作。

### Q 门控为什么不是因果门控

现有 Twin-Q 计算近似的 candidate-vs-baseline value advantage：

```text
delta_Q = min_i [Q_i(s, a_fastwam + r) - Q_i(s, a_fastwam)]
```

它来自离线 replay 的 Bellman 学习，只表示模型预测的长期价值差。两个 Q 同时给出高值，仍可能是共同的数据偏差或 OOD 外推；此前实验已经出现 Q 拒绝真实 rescue、批准 regression 的反例。因此“双 Q 一致”不等于“干预具有因果收益”。

直接因果门控使用严格匹配的两条闭环分支近似估计：

```text
delta_success = P(success | do(candidate)) - P(success | do(baseline))
```

RoboTwin 当前不是模拟器内存级状态克隆，所以这里属于经过初始观测、指令、proprio、干预观测和 FastWAM 动作哈希审计的匹配反事实近似，而不是无条件的形式化因果证明。任何前缀不一致的 pair 都必须隔离，不能训练门控。

### 不同阶段的门控开关

- **阶段 1 数据采集**：关闭 Q、因果和 OOD 门控，在预注册 replan 强制执行且只执行一次 residual，以观察真实 `rescue/regression`；否则无法获得门控会拒绝的反例。
- **阶段 2 离线训练与验证**：训练直接因果门控，单独评估 OOD 安全性，并把 Twin-Q 作为辅助信息做消融和校准。
- **阶段 3/4 在线部署**：只有通过离线准入的因果门控和 OOD gate 可以决定执行；Twin-Q 不具有独立放行权。

## 3. 任务和数据划分

任务分为三类：

- 保能力任务：`adjust_bottle`，检查 residual 是否破坏 FastWAM 的高成功率能力。
- 主要改进任务：`hanging_mug`、`place_can_basket`，二者既有 FastWAM 成功样本，也保留失败空间。
- OOD 诊断任务：`open_microwave`，当前只检查安全回退，不要求本轮必须提升。

环境 seed 必须在采集前划分并写入版本化 manifest：

- 60% 训练。
- 20% 验证，用于门控阈值、奖励权重和训练轮数选择。
- 20% 最终测试；训练结束前不得查看结果。
- 同一环境 seed 不得跨集合出现。

## 4. 阶段 0：冻结并审计 FastWAM 基线

统一使用：

- 官方 FastWAM RoboTwin 权重。
- 10 次去噪。
- 每 24 个动作重新调用 FastWAM。
- 官方 unseen instruction。
- 固定 seed manifest。
- 相同 checkpoint、语言指令、初始观测和推理随机性。

先在每个任务至少 5 个相同 seed 上运行 FastWAM 与 zero-residual wrapper，并逐 replan 比较 FastWAM 动作、初始状态和最终结果。

硬停止标准：

- 初始状态、指令和动作一致率必须为 100%。
- 最终成功结果必须完全一致。
- 任一不一致都会使该批后续结果无效；必须先修复协议，然后从基线重新执行。

## 5. 阶段 1：收集 FastWAM 原生因果配对数据

不再以普通专家轨迹为主要数据，而是收集 actor-aligned 单次干预配对。

在 FastWAM 到达某个 replan 状态后，从同一状态建立两条分支：

1. Baseline 分支执行 FastWAM 原动作块。
2. Residual 分支执行 FastWAM 动作块加当前 actor 给出的一个 residual。

约束：

- 干预前图像、proprio、语言指令和 FastWAM 动作哈希完全相同。
- Residual 分支每回合只允许一次强制干预。
- 一个动作块结束后，两条分支都重新调用实时 FastWAM，继续闭环执行到任务结束。
- 不得继续回放干预前保存的未来动作。
- 每个 seed 最多选择早期、中期、后期各一个 replan，防止单一轨迹制造大量伪独立样本。

每对数据记录：

- FastWAM 和 residual 分支的终局成功结果。
- `rescue`：FastWAM 失败、Residual 成功。
- `regression`：FastWAM 成功、Residual 失败。
- `neutral_success`：两者都成功。
- `neutral_failure`：两者都失败。
- 干预阶段、相机特征、proprio、语言特征。
- FastWAM 原动作、residual 候选动作和实际执行动作。
- 想象奖励及各相机分量、Q 值和 OOD 距离。

### 数据就绪标准

每个主要改进任务至少需要：

- 60 对完全有效的单次干预数据。
- 至少 8 个 `rescue`，来自至少 4 个不同环境 seed。
- 至少 8 个 `regression`。
- 至少 15 个 neutral/OOD 样本。
- 整批有效配对率至少 95%；单个不完全配对样本必须丢弃。

提前失败标准：

- 收集 40 对有效数据后仍少于 3 个 rescue，或所有 rescue 都来自同一个 seed，则停止该任务继续采集。
- 此时不得直接训练或重复同一 seed；应调整 actor 候选、干预阶段或目标任务。
- 强制 residual 完全不能产生 rescue 时，才进入 actor 结构/候选产生方式的修改。
- actor 能产生 rescue 但 Q 全部拒绝时，优先修复 critic/gate 数据，而不是扩大 actor。

## 6. 阶段 2：训练 Residual actor、IQL 和门控

### Residual actor

首轮保持当前结构：

- 冻结 FastWAM。
- 三相机分别统计并应用全局归一化。
- 语言和 FastWAM 原动作作为条件。
- MLP residual actor。
- 输出层零初始化。
- residual 每维最大尺度保持 `0.05`。
- gripper residual 固定为零。

当前 actor 已经产生过严格配对的 rescue，因此在新的因果数据证明表示能力不足以前，不加入 LSTM，也不扩大网络。

### IQL 奖励

终局结果必须主导：

```text
r = r_success + r_imitation + r_imagination
```

- 最终成功：`+10`。
- 失败或超时：`0`。
- 动作偏差：`-0.1 * normalized_MSE(executed, FastWAM)`。
- 想象奖励按相机分别使用训练集全局统计归一化，整个回合累计绝对值不超过 `1.0`，即成功奖励的 10%。
- 想象奖励不得覆盖终局结果。
- `rescue/regression` 用于训练 candidate-vs-baseline 因果门控，不把 episode 标签复制到所有 transition。

训练设置：

- 三个独立训练 seed。
- 初始上限 20 epochs。
- 连续 5 epochs 验证指标不再提升时 early stop。
- 训练、验证按完整环境 seed 隔离。
- checkpoint 不能只按 Q 自己预测的收益选择，必须使用 held-out 因果配对指标。

### 部署门控

部署批准顺序：

1. 直接因果门控判断 candidate 是否比当前 FastWAM 动作更可能成功。
2. OOD gate 对训练支持范围外的状态和 residual 动作执行不可绕过的否决。
3. Twin-Q 仅作为因果门控的辅助特征、置信度或诊断量，不拥有独立放行权。

最终执行条件为“因果门控批准且 OOD 检查通过”；不能用降低 Q 阈值代替因果门控验证，也不能让高 Q 覆盖 OOD 拒绝。

## 7. 离线准入标准

在完全未参与训练的验证 seed 上，必须同时满足：

- Rescue 与非 rescue AUROC 至少 0.70。
- Balanced accuracy 至少 0.65。
- Regression 错误批准率不超过 10%。
- Rescue recall 至少 50%。
- 三个训练 seed 中至少两个达到上述标准。
- 不同训练 seed 的批准率差异不超过 15 个百分点。

失败后的动作：

- Regression 错误批准率过高：补充 regression 和边界 hard negative，重训门控。
- Rescue recall 过低但安全性良好：补充跨 seed rescue，重训 actor 和门控。
- 训练很好、验证接近随机：增加独立 seed，禁止先扩大网络。
- 三个训练 seed 差异过大：判定数据覆盖不足，不进入在线评测。
- 两轮数据修正和重新训练后仍不通过：停止当前 gate 结构，再评估 patch token、LSTM/GRU 或新的时序输入。

## 8. 阶段 3：10 回合在线准入测试

每个任务使用 10 个相同 held-out seed 分别运行：

- FastWAM。
- FastWAM + Residual + Gate。

允许扩大评测的条件：

- `adjust_bottle` 最多损失 1 个 FastWAM 成功回合。
- 两个目标任务合计至少 2 个 rescue。
- 两个目标任务合计最多 1 个 regression。
- `rescue - regression` 必须为正。
- seed、指令和初始状态必须 100% 配对。

任一条件出现即停止，不扩大评测：

- 高成功率任务出现至少 2 个 regression。
- 目标任务 regression 数量不少于 rescue。
- 门控大量批准 OOD 动作。
- Residual 连续干预造成轨迹漂移。

对应修复：

- 有 rescue 但 regression 过多：回收 regression，主要重训门控。
- 几乎没有批准动作：补充 actor-aligned rescue，不能直接降低阈值。
- 批准很多但没有 rescue：判定 Q/gate 排序失效，重做门控训练。
- 强制 residual 也没有 rescue：回到 actor 和候选数据设计。

## 9. 阶段 4：正式配对评测

小规模测试通过后，每个任务扩大到至少 30 个独立 held-out seed。

正式通过标准：

- 高成功率任务下降不超过 5 个百分点。
- 每个主要目标任务均不能净下降。
- 两个主要目标任务合计成功率至少提高 10 个百分点。
- Baseline 成功轨迹的 regression 比例不超过 5%。
- Rescue 数量明显多于 regression。
- 配对 bootstrap 或单侧 McNemar 检验显示净提升方向稳定。

论文级最终结果应继续扩大到每任务 50--100 个独立 seed，并报告置信区间。

## 10. 全流程停止和重新执行规则

成功停止要求：

- 协议和配对审计 100% 通过。
- actor 在多个独立 seed 上产生 rescue。
- 离线门控满足安全性和 recall 标准。
- 10 回合准入测试净收益为正。
- 正式评测提高目标任务且基本保留高成功率任务。
- 改进来自真实 residual 干预，而不是门控完全退化为 FastWAM。

失败并重新设计的条件：

- 40 对有效样本后仍没有跨 seed rescue。
- 数据量达标后门控验证 AUROC 仍接近 0.5。
- 两轮重训后 regression 错误批准率仍超过 10%。
- 在线测试中 regression 不少于 rescue。
- 高成功率任务下降超过 5 个百分点。

只有协议错误、进程异常或产物损坏时才原样重新执行。指标不达标时禁止只更换随机 seed 重复同一配置，必须按失败类型重新采集数据、训练对应模块或修改模型。
