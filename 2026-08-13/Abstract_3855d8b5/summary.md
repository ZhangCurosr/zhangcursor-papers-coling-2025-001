---
title: "Abstract"
source: https://aclanthology.org/2025.coling-main.1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:29:34"
field: "大语言模型智能体规划"
keywords: ["LLM Agent", "Planning", "ReAct", "Prediction", "AgentBench", "Chain-of-Thought", "Tree-of-Thought"]
innovations: ["提出PreAct框架，将预测机制引入LLM Agent规划，增强推理多样性与策略方向性", "证明PreAct与Reflexion/TOT正交可组合，历史预测具有持续正向累积效应", "提出Diversity与Directional Strategy两项规划过程评估指标"]
benchmarks: ["AgentBench-HH", "AgentBench-OS", "AgentBench-DB", "AgentBench-LTP", "HotpotQA"]
---

# 论文速读：PreAct: Predict Future with Reasoning and Action

## 一句话总结
本文提出 PreAct 框架，通过在每个规划步骤中加入"预测未来观测与应对措施"的机制，增强 LLM Agent 的推理多样性与策略方向性，在 AgentBench 多个复杂任务上显著优于 ReAct，且可与 Reflexion、TOT 等记忆/选择策略正交组合。

---

## 研究问题与动机

1. **核心问题**：现有 LLM Agent 的 action sampling 方法（如 ReAct）仅生成直接因果推理路径，容易重复相同动作，缺乏探索多样性和战略导向性，难以应对复杂关系任务。
2. **ReAct 不足**：ReAct 将 thought-action-observation 序列作为 history，但缺少对未来的前瞻性预测，导致 Agent 在面对不确定环境时难以调整策略方向。
3. **现有工作偏差**：当前研究多聚焦于 state selection 策略优化（如 TOT、GOT、RAP），忽视了对 action sampling 方法本身的改进。
4. **灵感来源**：受 Task-Oriented Dialogue 中"预测未来"工作启发，将预测机制引入 Agent 规划，使模型能够提前构想行动后果并制定应对方案。

---

## 核心贡献（创新点）

1. **提出 PreAct 框架**：首次将 prediction 与 reasoning、action 同步整合到 LLM Agent 中，每个步骤要求模型预测未来观测及相应应对措施；与 ReAct 的本质区别在于引入前瞻性预测机制，而非仅依赖历史轨迹。
2. **验证 PreAct 与现有策略正交性**：证明 PreAct 可独立于 Reflexion（长期记忆）和 TOT（动作选择策略）发挥作用，两者组合后性能进一步提升；与已有工作的本质区别在于 PreAct 改进的是 action sampling 本身，而 Reflexion/TOT 改进的是 history 质量或动作选择。
3. **发现历史预测的持续正向效应**：通过消融实验表明，保留越多历史预测，规划性能越稳定提升；与已有工作不同的是，本文量化了预测信息的累积价值，而非仅验证单步效果。
4. **提出两项规划评估指标**：推理多样性（Diversity）与策略方向性（Directional Strategy），为后续强化学习过程级奖励函数设计提供依据。

---

## 方法详解

### 2.1 预备知识

- **Agent 环境交互范式**：在第 k 步，Agent 基于历史和上一步观测选择动作 $a_k = \pi_{agent}(o_{k-1}, history)$，环境根据状态转移函数 $o_k = \pi_{env}(o_{k-1}, a_k)$ 返回新观测。
- **ReAct**：以 $\{o_0, t_1, a_1, o_1, ..., t_{k-1}, a_{k-1}\}$ 作为 history，结合 CoT 推理与行动。
- **Reflexion**：任务失败后生成反思 ref，更新 history 以避免重复错误。
- **TOT**：每步采样多个候选动作 $\{a_{k1}, ..., a_{kn}\}$，再通过选择策略 $\pi_{selection}$ 选出最优动作。

### 2.2 PreAct 框架

PreAct 与 ReAct 的两个核心差异：

1. **Action Policy（$\pi_{agent}$）改进**：每步要求 LLM 生成对未来观测和应对措施的预测 $p$，并在实际观测与预测不符时反思并调整方向。
2. **History 扩展**：将预测信息加入历史，定义四种模式：

| 模式 | History 构成 |
|------|-------------|
| **Permanent** | $\{o_0, t_1, a_1, p_1, o_1, ..., t_{k-1}, a_{k-1}, p_{k-1}\}$ |
| **Immediate** | $\{o_0, t_1, a_1, o_1, ..., t_{k-1}, a_{k-1}, p_{k-1}\}$（仅保留最新预测） |
| **Reflexion** | $\{ref, o_0, t_1, a_1, p_1, o_1, ..., t_{k-1}, a_{k-1}, p_{k-1}\}$ |
| **TOT** | 结合 TOT 选择策略，history 为 Permanent 或 Reflexion 模式 |

### 关键 Prompt 设计（以 HH 为例）

```
THOUGHT: 思考当前状态与计划
ACTION: 执行动作
PREDICTED FEEDBACK:
  1. 可能的反馈类型及应对措施
  2. 另一种可能的反馈类型及应对措施
  ...
```

若实际反馈与预测不符，需在下一轮 THOUGHT 中分析差异原因并调整策略。

---

## 实验与结果

### 数据集与设置

- **AgentBench** 四个子数据集：Householding (HH)、Operating System (OS)、Database (DB)、Lateral Thinking Puzzles (LTP)
- **HotpotQA**：用于验证与 TOT 的组合效果
- 模型：GPT-3.5-turbo-1106 与 GPT-4-1106-preview

### 主要结果（Permanent 模式）

| 模型 | HH Test | OS Test | DB Test | LTP Test |
|------|---------|---------|---------|----------|
| ReAct (3.5) | 10.0 | 16.7 | 39.3 | 11.0 |
| **PreAct (3.5)** | **18.0** | **20.1** | **45.7** | 14.1 |
| ReAct (4) | 68.0 | 37.5 | 51.3 | 29.0 |
| **PreAct (4)** | **78.0** | **43.1** | 51.3 | 24.9 |

- **HH**：PreAct 较 ReAct 提升约 **20%**
- **OS**：平均提升 **12%**
- **DB**：平均提升 **6%**
- **LTP**：受 GPT 安全机制拒绝影响，提升有限

### Reflexion 模式结果

| 模型 | HH Test | OS Test | DB Test |
|------|---------|---------|---------|
| ReAct (3.5) | 18.0 | 21.5 | 45.6 |
| **PreAct (3.5)** | **20.0** | **24.3** | **55.3** |
| ReAct (4) | 78.0 | 48.6 | 58.0 |
| **PreAct (4)** | **80.0** | **50.0** | **58.3** |

### 与 TOT 组合（HotpotQA）

| 模型 | 100样本平均 | 1000样本 |
|------|------------|----------|
| ReAct+TOT | 64.9 | 64.9 |
| **PreAct+TOT** | **70.8** | **70.8** |

- PreAct+TOT 比 ReAct+TOT 高约 **5%**，证明两者正交可叠加。

### 历史预测影响（RQ2）

- 保留更多历史预测 → 成功率持续提升（HH: 66%→70%→74%；OS: 40.9%→42.3%→43.1%）
- LTP 数据集中 Permanent 模式因安全拒绝导致性能下降。

### 内在机制分析（RQ3）

| 指标 | ReAct (GPT3.5) | PreAct (GPT3.5) | ReAct (GPT4) | PreAct (GPT4) |
|------|----------------|-----------------|--------------|---------------|
| Diversity | 0.69 | **0.84** | 1.89 | **2.29** |
| Directional Strategy | - | - | - | - |
| Strategy Score (HH) | - | - | - | 至少高 **20%** |

- **多样性**：至少 45% 样本中 PreAct 多样性优于 ReAct，反向不超过 34%
- **策略方向性**：PreAct 评分至少比 ReAct 高 20%
- **案例研究**：PreAct 能纠正错误（如 DB 任务中修正列名），而 ReAct 重复错误。

---

## 相关工作脉络

1. **ReAct (Yao et al., 2022)**：开创性地将 CoT 推理与行动结合，本文在其基础上引入预测机制，本质区别在于 PreAct 增加前瞻性规划能力。
2. **Reflexion (Shinn et al., 2023)**：通过反思改进历史质量，本文证明 PreAct 与 Reflexion 正交，可组合使用。
3. **TOT (Yao et al., 2023)**：通过多动作采样与选择提升规划质量，本文证明 PreAct 与 TOT 相互独立，组合后性能提升约 5%。
4. **RAP (Hao et al., 2023)、LATS (Zhou et al., 2023)**：基于搜索算法的动作选择策略，本文从 prompt 层面改进 action sampling，与搜索策略形成互补。
5. **Task-Oriented Dialogue 预测工作**：Qi et al. (2020) ProphetNet、Zeng et al. (2022, 2023) FutureTOD 启发本文，但本文首次将预测机制引入通用 Agent 规划而非对话系统。
6. **GOT (Besta et al., 2023)、Think-on-Graph (Sun et al., 2023)**：扩展推理结构，本文从"预测"维度切入，与图结构方法无直接竞争关系。

---

## 局限性与未来方向

1. **短期记忆限制**：当前 PreAct 主要依赖 short-time memory（history），未来需探索与更广泛的 long-term memory（如 example memory、insight memory）的交互。
2. **仅基于 prompt 验证**：未进行 fine-tuning，未来可通过 PreAct 轨迹微调模型以挖掘更深层机制。
3. **幻觉敏感性问题**：Table 5 显示当预测完全幻觉时性能下降明显，需进一步提升预测鲁棒性。
4. **LTP 数据集受限**：受 GPT 安全机制影响，部分样本被拒绝，导致评估不充分。

---

## 研究启发与可借鉴点

1. **预测机制的可迁移性**：将"预测-比对-调整"循环引入其他 Agent 任务（如代码生成、机器人控制），可能同样提升规划质量。
2. **正交策略组合思想**：PreAct 与 Reflexion/TOT 的正交性提示我们，不同改进维度可叠加使用，为多层 Agent 架构设计提供思路。
3. **规划过程评估指标**：提出的 Diversity 和 Directional Strategy 指标可用于 RL 训练中的过程级奖励设计，替代仅依赖最终结果的稀疏奖励。
4. **历史信息的累积价值量化**：通过控制历史预测数量验证持续正向效应，这种消融设计值得借鉴于其他记忆机制研究。
5. **Prompt 工程中的"前瞻性"设计**：在 Prompt 中明确要求模型预测未来状态及应对措施，是一种简单有效的规划增强手段，可应用于多种 LLM 应用场景。

---

## 关键术语表

**PreAct**：本文提出的 Agent 框架，在每个规划步骤中要求 LLM 预测未来观测及应对措施，以增强推理多样性与策略方向性。

**ReAct**：Yao 等人提出的 LLM Agent 框架，结合 Chain-of-Thought 推理与实际行动，交替输出 thought-action-observation。

**Reflexion**：Shinn 等人提出的长期记忆策略，通过反思失败轨迹来改进后续决策。

**TOT (Tree-of-Thought)**：Yao 等人提出的动作选择策略，每步采样多个候选动作并通过评估选择最优路径。

**Diversity（推理多样性）**：本文提出的评估指标，衡量 Agent 在不同步骤中思考与行动的差异化程度。

**Directional Strategy（策略方向性）**：本文提出的评估指标，衡量 Agent 每一步规划方向对任务完成的促进程度。

**AgentBench**：评估 LLM 作为 Agent 能力的基准测试平台，包含多个交互式任务数据集。

**Permanent/Immediate/Reflexion 模式**：PreAct 中历史预测信息的三种保留策略，分别对应全量保留、仅保留最新预测、结合反思信息。

---

## 可复现要素

- **数据集**：AgentBench（HH/OS/DB/LTP）、HotpotQA；**未明确说明是否公开**，但 AgentBench 有 GitHub 代码
- **代码/权重**：论文未提及开源代码，实验基于 GPT API（gpt-3.5-turbo-1106、gpt-4-1106-preview、gpt-4-turbo-2024-04-09）
- **关键超参**：
  - HH/OS/DB 最大交互轮次分别为 35、8、5
  - LTP 最大 50 轮，过滤连续 3 轮拒绝的样本
  - HOTpotQA TOT 实验：100 样本运行 5 次，1000 样本运行 1 次
- **Prompt**：详见附录 B，包含 HH/OS/DB/LTP 的 PreAct 与 ReAct 对比 Prompt

---
