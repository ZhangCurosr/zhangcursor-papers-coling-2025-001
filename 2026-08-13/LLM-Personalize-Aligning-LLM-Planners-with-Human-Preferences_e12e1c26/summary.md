---
title: "LLM-Personalize-Aligning-LLM-Planners-with-Human-Preferences"
source: https://aclanthology.org/2025.coling-main.98.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:41:30"
field: "具身AI与机器人规划"
keywords: ["LLM planning", "personalization", "reinforced self-training", "household robotics", "scene graph", "imitation learning"]
innovations: ["结合模仿学习bootstrap与迭代ReST的LLM规划器个性化对齐方法", "动态场景图+迭代重规划解决多房间部分可观测长视界规划"]
benchmarks: ["Housekeep"]
---

# 论文速读：LLM-Personalize-Aligning-LLM-Planners-with-Human-Preferences

## 一句话总结
本文提出 **LLM-Personalize** 框架，通过**模仿学习+迭代强化自训练（ReST）**将LLM规划器个性化对齐到用户偏好，在多房间、部分可观测的家庭环境中实现长视界物品整理任务，在Housekeep基准上相比现有LLM规划器成功率提升超过30%。

## 研究问题与动机
1. **个性化偏好对齐缺口**：通用LLM知识反映的是"普遍偏好"，而不同家庭的个性化偏好差异显著（如咖啡杯应放在餐桌还是厨房橱柜），现有方法缺乏对个性化家庭偏好的对齐能力。
2. **部分可观测性挑战**：家庭场景中机器人只能获得局部视角观测（egocentric observation），需维护内部状态表示并完成多房间长视界探索与决策。
3. **长视界计划难以标注**：直接将ReST应用于家庭机器人任务时，LLM可能在一个长计划中同时包含正确和错误放置动作，导致难以自动提取干净的正例训练样本。
4. **可执行性难题**：LLM规划器难以从复杂上下文（动态场景图）中精确提取信息并生成可执行的连续动作序列，容易产生命名不完整或动作序贯错误的计划。

## 核心贡献（创新点）
1. **LLM-Personalize三组件框架**：整合Context Generator（动态场景图构建）、LLM Planner（迭代规划）和Controller，解决多房间部分可观测环境下的长视界规划问题；与SayPlan的静态场景图本质区别在于动态增量更新。
2. **模仿学习引导的ReST优化流程**：先通过示范数据bootstrap LLM规划器，使其理解复杂上下文、生成可执行计划并初步对齐偏好；与直接ReST的本质区别在于示范阶段保证了每个计划目标单一，便于后续阶段准确标注。
3. **面向长视界规划的ReST适配**：通过构造"探索房间"和"整理单件物品"两类示范，确保每个响应可被清晰标注为是否偏好匹配；与机器翻译等单步生成任务的ReST本质区别在于处理了多动作序列的正负混杂问题。
4. **Housekeep基准上的显著性能提升**：在4个场景的测试集上，经IL+2轮ST后成功率较基线提升超30%，同时具备跨场景泛化能力；与仅做grounding的基线（如SayCan、SayPlan）本质区别在于引入了用户偏好驱动的微调闭环。

## 方法详解
**架构三组件**：
- **Context Generator**：从局部观测 $o_t$ 构建并动态更新场景图 $G_t$，包含房间、容器（receptacle）、物体节点及关系边；prompt包含 $G_t$ 的自然语言描述、当前持有的物体、任务指令及两个in-context学习示例。
- **LLM Planner**：接收prompt后生成包含前10个高层动作（high-level actions）的计划序列 $\omega \in \{go\ to,\ look\ at,\ pick\ up,\ place\ on\}$；采用**迭代重规划**策略——每轮执行完一个计划后再重新规划，而非单步迭代。
- **Controller**：使用Housekeep模拟器自带off-the-shelf控制器，将高阶层动作翻译为低层控制动作序列。

**优化流程（两阶段）**：
1. **模仿学习（IL）阶段**：构造演示器（demonstrator）生成示范计划（探索房间 / 整理单个物品），收集数据集 $\mathcal{D}_{demo} = \{(x^i, y^i)\}$，用NLL损失微调：
   $$\mathcal{L}_{NLL} = -\mathbb{E}_{(x,y)\in\mathcal{D}_{demo}}\left[\sum_{\tau=1}^{|y|}\log P_\theta(y_\tau | y_{1:\tau-1}, x)\right]$$
2. **迭代强化自训练（ReST）阶段**：每轮收集 $M$ 条交互样本 $\{(x^i, y^i, out^i)\}$，根据偏好奖励 $r^i \in \{1,0,-1\}$ 过滤出正样本 $\mathcal{D}_{self-train} = \{(x^i, y^i)|r^i>0\}$，再次用NLL损失微调，迭代进行。

## 实验与结果
**数据集**：Housekeep（Kant et al., 2022），4个不同布局的3D仿真家庭场景，每任务含5-10个物体、3-7个错放物品，每场景随机采样10个演示任务、20个训练任务、5个测试任务。

**评估基线**：SayCan（Ahn et al., 2022）、SayPlan（Rana et al., 2023）、LLM-Planner（Song et al., 2023，作为base model）；均使用GPT-3.5-turbo（temperature=1）。

**主要结果**：
- **最强结果**：Scene 3测试集上，LLM-Personalize (ST iter=2) 达到 **43.3±4.3%** 成功率，显著优于所有基线（基线多为负值）。
- **整体提升**：Scene 1测试集从基线的 -3.6% → IL后17.6% → ST iter=1后25.8% → ST iter=2后 **29.6%**，累计提升约33个百分点；Across scenes平均成功率提升超过30%。
- **跨场景泛化**（Table 2）：在Scene 2训练后测试Scene 1，ST1达到19.4±2.9%，验证了个性化偏好的可迁移性。
- **消融发现**：IL阶段大幅改善可执行性（Fig.6a）；ST1阶段探索性更强（unique placements增加），ST2阶段转向exploit已学偏好。

## 相关工作脉络
1. **LLM-Planner (Song et al., 2023)**：允许LLM迭代规划以应对可扩展性，本文沿用其迭代规划思想但加入了动态场景图和个人化偏好对齐。
2. **SayPlan (Rana et al., 2023)**：使用静态场景图解决长视界规划，但场景图不随探索动态更新，且无个性化偏好学习机制。
3. **SayCan (Ahn et al., 2022)**：通过affordance grounding LLM计划，适用于小场景和有限物体词汇表，难以扩展到多房间大规模家庭环境。
4. **ReST (Gulcehre et al., 2023)**：原应用于机器翻译等单步生成任务，本文首次将其适配到长视界多步骤机器人规划，并通过IL bootstrap解决正负混杂标注难题。
5. **Tidybot (Wu et al., 2023)**：使用LLM推理个性化偏好规则，但不直接优化规划器策略，本文方法更端到端、直接微调规划行为。

## 局限性与未来方向
1. **实验规模受限**：受限于LLM API预算，仅在4个仿真场景中验证，未来需在更大规模、更多样化的家庭环境中评估可扩展性与鲁棒性。
2. **缺乏真机验证**：目前仅在仿真环境验证，尚未在真实物理机器人和家庭场景中测试，未来方向为实机部署。
3. **对齐方法的扩展空间**：仅使用基于奖励过滤的SFT，未探索DPO等偏好对齐方法（受限于API不支持梯度访问），未来可研究其他feedback形式的fine-tuning。

## 研究启发与可借鉴点
1. **IL-Bootstrap + ReST的两阶段设计**：对于需要用户偏好对齐的长视界规划任务，可复用"先用结构化示范数据bootstrap、再用正反馈数据自训练"的流程，避免直接ReST中因正负混杂导致的训练信号噪声。
2. **动态场景图+迭代重规划**：用增量更新的场景图维护部分可观测状态，配合"执行完整计划后再重规划"的策略（非单步迭代），可有效平衡规划连贯性与环境信息获取。
3. **跨场景泛化评估设计**：训练/测试使用不同场景的布置组合（Table 2），能验证个性化偏好的可迁移性而非过拟合，值得在类似个性化研究中借鉴。
4. **奖励驱动的过滤式SFT替代DPO**：当无法访问模型梯度或使用在线API时，可通过收集交互、基于奖励过滤正样本、再进行SFT的方式实现偏好对齐，绕开DPO的工程限制。

## 关键术语表
**ReST（Reinforced Self-Training）**：一种迭代式LLM对齐方法，通过grow步骤收集LLM生成的多响应，再用人类偏好/奖励筛选正样本后进行监督微调。
**Context Generator**：负责从局部观测构建和动态更新场景图，并将环境状态、指令和示例整合为prompt传入LLM规划器。
**High-level Action**：LLM规划器输出的抽象动作（如go to、pick up、place on），由底层控制器翻译为具体的机器人控制指令序列。
**Success Rate**：Housekeep评估指标，定义为最终正确放置物品数与初始错放物品数的比值百分比，负值表示放置错误多于正确。
**Iterative Re-planning**：每执行完一个完整的高层计划后，基于最新观测重新生成下一个计划，而非每步都重新规划。
**DPO（Direct Preference Optimization）**：一种偏好对齐方法，需要正负配对样本并访问模型梯度，本文因API限制未采用。

## 可复现要素
- **数据集**：Housekeep（Kant et al., 2022）公开基准，论文使用4个场景的train/test划分
- **代码/权重**：论文未提供开源代码与微调后的权重（仅声明使用OpenAI fine-tune API）
- **关键超参**：GPT-3.5-turbo，temperature=1；每任务采集5条episode经验；IL阶段使用10个演示任务；ST每轮收集M条交互；取规划序列前10个高阶层动作
- **评估**：mean ± standard error across 5 runs per task，任务集划分见Section 4.1
