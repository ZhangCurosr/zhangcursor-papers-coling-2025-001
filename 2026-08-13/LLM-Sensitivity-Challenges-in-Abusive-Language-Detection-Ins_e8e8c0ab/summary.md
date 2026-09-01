---
title: "LLM-Sensitivity-Challenges-in-Abusive-Language-Detection-Ins"
source: https://aclanthology.org/2025.coling-main.188.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:57:28"
field: "大语言模型公平性与鲁棒性评估"
keywords: ["Abusive Language Detection", "LLM Bias", "Zero-shot Evaluation", "Instruction Tuning", "RLHF", "Prediction Bias", "Prompt Engineering"]
innovations: ["首次系统量化并对比了指令调优与RLHF模型在零样本辱骂检测中的相反预测偏差倾向", "提出将预测偏差指标(bias/bias_agg)作为F1分数的互补评估，揭示高准确率下可能隐藏的分布不公", "探索了通过提示词注入标签分布信息来校准RLHF模型过预测偏差的初步可行性"]
benchmarks: ["Civil-Comments", "HASOC-2019", "HatEval", "OLID", "Davidson-2017", "HateXplain"]
---

# 论文速读：LLM-Sensitivity-Challenges-in-Abusive-Language-Detection-Ins

## 一句话总结
本文系统评估了四种主流 LLM 在零样本辱骂语言检测任务中的**预测偏差（prediction bias）**，发现指令调优模型倾向于系统性**低估（under-predict）**辱骂类别，而 RLHF 对齐模型则倾向于**高估（over-predict）**；并探索了在提示词中注入标签分布信息以缓解偏差的初步方法。

## 研究问题与动机
1. **现有评估盲区**： abusive language detection 领域的 LLM 研究多聚焦于 $F_1$ 等准确率指标，忽视了预测分布本身是否偏离真实标签比例这一**公平性与校准性**问题。
2. **微调策略引发系统性偏差**： 指令调优（instruction-tuning）数据通常包含大量不平衡的辱骂数据集，可能导致模型对少数类（辱骂）不敏感；而强化学习人类反馈（RLHF）出于安全对齐的保守性，可能使模型过度触发安全护栏，导致误判。
3. **实用场景需求**： LLM 常被直接作为**自动标注器**（zero-shot annotator）用于特定社区，若存在未校正的预测偏差，将直接污染后续监督学习的数据集，影响下游模型公平性。
4. **缓解方法未知**： 目前缺乏对 LLM 此类分布感知偏差的系统测量框架，也鲜有工作探索通过提示工程（如告知标签分布）来校准其输出。

## 核心贡献（创新点）
1. **首次系统量化对比 instruction-tuned 与 RLHF 模型在 zero-shot 辱骂检测中的预测偏差方向**：揭示了前者因训练数据不平衡导致**under-prediction**，后者因安全对齐导致**over-prediction**的根本性差异，填补了 LLM 公平性评估中针对具体 NLP 任务细分的研究空白。
2. **提出并采用 `bias` 与 `bias_agg` 指标作为 $F_1$ 的互补评估**：明确指出高 $F_1$ 不等于无偏差（如一个模型将所有样本预测为多数类也可能获得高多数类 $F_1$），强调评估应同时关注**分类准确性**与**预测分布校准性**。
3. **探索性地提出通过提示词注入标签分布信息来引导/校准 LLM 输出**：证明了对于存在严重 over-prediction 的 RLHF 模型，该方法能有效降低偏差并提升整体 macro-$F_1$；但对于已经 under-predict 或轻度偏差的模型，该方法无效甚至有害，揭示了模型可调性的边界。

## 方法详解
1.  **评估设置**：采用 **zero-shot prompting**，不使用任何辱骂检测特定领域的微调。对四个 LLM（Flan-T5-XL, OPT-IML-1.3B, LLaMA 2-Chat-7B, GPT-3.5）测试其在七个英文数据集（4个二分类，3个多分类）上的表现。
2.  **偏差度量公式**：
    *   对于类别 $c$，定义偏差 $\text{bias}_c = \frac{(\text{TP}_c + \text{FP}_c) - (\text{TP}_c + \text{FN}_c)}{\text{TP}_c + \text{FN}_c}$。其中分子是模型预测为 $c$ 的比例减去真实为 $c$ 的比例，分母是真实为 $c$ 的比例。$\text{bias}_c > 0$ 表示高估（over-prediction），$\text{bias}_c < 0$ 表示低估（under-prediction）。
    *   聚合偏差 $\text{bias}_{\text{agg}} = \frac{1}{|C|} \sum_{c \in C} |\text{bias}_c|$，衡量模型整体预测分布的平均绝对偏差程度。
3.  **提示词设计**：
    *   **Base Prompt**：直接要求模型将文本分类为 NORMAL/OFFENSIVE/HATESPEECH 等。
    *   **Numeric 变体**：在提示中明确告知训练集的精确标签百分比（e.g., "16.8% labels are NORMAL, 77.4% labels are OFFENSIVE..."）。
    *   **Word 变体**：用定性描述标签频率顺序（e.g., "OFFENSIVE occurs more frequently than NORMAL..."）。
    *   **Feedback 变体**（附录C）：基于开发集性能，直接告知模型其预测分布与真实分布的差异（e.g., "You wrongly predicted less NORMAL... but much more OFFENSIVE..."），利用类似 self-correct 的思路进行引导。
4.  **实验控制**：为消除提示词中**标签顺序**带来的位置偏见，对每个模型在开发集上测试了多种标签排列组合，选取平均 macro-$F_1$ 最优的提示格式。对 temperature 和 top-p 进行网格搜索。每个实验运行3个随机种子取平均。

## 实验与结果
*   **数据集**：
    *   二分类：Civil-Comments (toxic), HASOC-2019 task1 (HOF), HatEval (hate speech), OLID (offensive)。
    *   多分类：Davidson-2017 (NORMAL/OFFENSIVE/HATESPEECH), HateXplain, HASOC-2019 task2 (hate/offensive/profane)。
*   **主要发现（Table 2）**：
    *   **Instruction-tuned 模型** (Flan-T5, OPT-IML) 普遍存在**负偏差**（低估 abusive 类）。例如 Flan-T5 XL 在 binary datasets 上 abusive class bias 为 **-23.55**，OPT-IML 为 **-11.76**。
    *   **RLHF 模型** (LLaMA 2-Chat, GPT-3.5) 普遍存在**正偏差**（高估 abusive 类）。LLaMA 2-Chat 在 binary datasets 上 abusive class bias 高达 **138.76**，GPT-3.5 为 **20.59**。
    *   这种偏差方向在不同数据集间具有高度一致性，表明是模型微调策略的**系统性倾向**，而非数据集偏差。
*   **缓解实验（Table 3, 4, 5, 6）**：
    *   对于存在**严重 over-prediction** 的 LLaMA 2-Chat，在提示中加入训练集标签分布（numeric 或 word 方式）显著降低了其 $\text{bias}_{\text{agg}}$（从 77.82 降至 47.38 和 35.39），并提升了 macro-$F_1$。
    *   对于 GPT-3.5，**feedback 提示**（基于其自身在开发集上的错误模式进行纠正）效果最佳，将 binary datasets 上的 $\text{bias}_{\text{agg}}$ 从 17.32 大幅降至 **4.42**。
    *   对于 already under-predicting 或 bias 较小的模型（如 Flan-T5, OPT-IML），上述分布注入方法**无效或有害**。
    *   即使使用**虚构的均衡分布**（50%/50% 或 equal frequencies）作为提示，也能让 RLHF 模型的性能达到接近使用真实训练分布的水平，表明模型对分布信息具有响应能力。
*   **最强结果**：LLaMA 2-Chat 7B 在使用 word (train) 提示后，在 binary datasets 上的 $\text{bias}_{\text{agg}}$ 降至 **35.39**（相对 base 的 77.82 降低了约 **54.5%**），macro-$F_1$ 提升至 63.75。GPT-3.5 使用 feedback 提示后，$\text{bias}_{\text{agg}}$ 降至 **4.42**（相对 base 的 17.32 降低了约 **74.5%**）。

## 相关工作脉络
1.  **传统分类不平衡学习** (Buda et al., 2018; Zhang et al., 2024)：本文借鉴了传统方法中通过阈值调整或重采样来补偿先验概率分布的思想，但将其应用于 zero-shot LLM 的提示工程语境中。
2.  **LLM 提示敏感性研究** (Zhu et al., 2023; Pezeshkpour and Hruschka, 2023; Shu et al., 2024)：本文确认并利用了 LLM 对提示词细节（如标签顺序、分布信息）敏感的特性，通过仔细选择提示格式和控制变量来确保评估的可靠性。
3.  **RLHF 的安全保守性** (Touvron et al., 2023)：本文的工作直接延伸并验证了 LLaMA 2 报告中关于 RLHF 安全微调可能导致对非对抗性提示**过度拒绝**的观察，将其具体量化到辱骂检测任务的预测偏差上。
4.  **自动标注与 LLM-as-a-Judge** (Plaza-del arco et al., 2023; Li et al., 2024)：本文警示了直接使用 LLM 作为自动标注器的潜在风险——其固有的预测偏差会被转移到生成的伪标签数据中，进而影响依赖这些数据的下游模型。
5.  **文本分类偏差度量** (Dixon et al., 2018)：本文采用了 Dixon 等人提出的计算预测分布与真实分布差异的核心思想，并将其形式化为 $\text{bias}_c$ 和 $\text{bias}_{\text{agg}}$ 指标，专门用于 LLM 的 zero-shot 评估。
6.  **自我修正/反馈提示** (Madaan et al., 2023)：附录 C 中的 feedback 提示策略受到 Self-refine 等自我修正方法的启发，通过向模型提供其自身错误的分布信息来引导其调整输出。

## 局限性与未来方向
1.  **语言与文化局限**：实验仅限于**英语**语料和标注，以避免跨文化偏见。然而，辱骂的定义和分布可能因文化而异，结论在其他语言和社区中的普适性有待验证。
2.  **缓解方法的局限性**：提出的标签分布注入方法**效果不一致**，对过预测模型有效，但对欠预测或已校准模型无效甚至有害。且依赖训练集分布信息，在实际无标签场景中可能需要使用假设分布（如均衡分布）。
3.  **缺乏根因分析与深度干预**：研究停留在现象观察和浅层提示工程缓解，未深入探索偏差产生的内部机制（如具体哪些训练数据或 RLHF 奖励模型组件导致了偏差），也未尝试模型内部的干预（如微调）。
4.  **细粒度类别偏差复杂性**：在多类别任务（如 HASOC-2019 task2）中，不同类别间的偏差方向不一，简单的全局分布提示难以精细调控每一类的预测比例，甚至可能引发类别间的此消彼长（如 GPT-3.5 在 HateXplain 上的表现）。
5.  **评估指标扩展**：当前主要关注二元/多元分类的宏观 $F_1$ 和偏差，未涉及连续置信度评分的校准度（calibration）评估，也未考虑不同代价矩阵下的决策偏差。

## 研究启发与可借鉴点
1.  **评估范式迁移**：任何将 LLM 用于**自动化分类/标注**的场景（如内容审核、情感分析、实体识别），都应**同步报告预测偏差指标**（$\text{bias}_c$, $\text{bias}_{\text{agg}}$），而不仅是准确率/$F_1$。这有助于揭示模型在分布层面的系统性不公。
2.  **提示工程策略**：当目标 LLM 存在明显的**过预测**倾向时，可以尝试在 system prompt 或 few-shot 示例中**显式声明目标标签的先验分布**（无论是精确比例还是相对顺序），作为一种低成本校准手段。需警惕其对欠预测模型的副作用。
3.  **模型选择依据**：在需要**高召回率**（不想漏掉任何辱骂）的任务中，应谨慎使用 RLHF 模型，其 over-prediction 倾向可能导致大量误报。在需要**高精确率**（不想冤枉用户）的任务中，指令调优模型可能更合适，但需关注其漏报问题。
4.  **数据污染风险意识**：本研究揭示了“LLM 零样本标注 -> 训练监督模型”流水线中潜在的**偏差传递与放大**风险。后续工作可在本团队方向中探索如何检测或校正这种由标注器引入的分布偏差。
5.  **消融实验设计**：本文严格控制了提示词标签顺序、temperature/top-p 超参、随机种子，并进行了多数据集平均以抵消单个数据集的偏差。这种严谨的实验设计范式值得借鉴。

## 关键术语表
*   **Instruction-tuned LLMs**: 在大量指令-响应配对数据上进行微调的 LLM，旨在提升其遵循自然语言指令完成各种任务的能力（如 Flan-T5, OPT-IML）。
*   **RLHF (Reinforcement Learning from Human Feedback)**: 一种通过对模型输出进行人类偏好排序并训练奖励模型，再用强化学习优化策略模型以对齐人类价值观（如 helpfulness, safety）的微调技术（如 LLaMA-Chat, GPT-3.5）。
*   **Over-prediction / Under-prediction**: 模型预测某个类别的正样本比例高于/低于该类别在真实数据集中的实际比例。
*   **Prediction Bias ($\text{bias}_c$)**: 本文定义的衡量模型对类别 $c$ 预测偏倚程度的指标，计算公式为 (预测比例 - 真实比例) / 真实比例。
*   **Aggregated Bias ($\text{bias}_{\text{agg}}$)**: 所有类别预测偏差绝对值的平均，用于综合衡量模型预测分布的整体偏离程度。
*   **Zero-shot Prompting**: 在不提供任何特定任务的训练示例的情况下，仅通过自然语言指令让 LLM 执行分类任务。
*   **Label Distribution Prompting**: 在提示词中额外提供目标数据集中各类别标签占比信息，以引导 LLM 调整其输出分布。
*   **Macro-$F_1$**: 对每个类别单独计算 $F_1$ 分数后取算术平均值，对少数类性能给予与多数类相同的权重。

## 可复现要素
*   **数据集**: 均使用公开数据集：Civil-Comments, Davidson-2017, HASOC-2019, HatEval, HateXplain, OLID。论文声明对 Civil-Comments 进行了 5000 条子采样。
*   **代码**: 论文声明代码和实验输出已开源在 GitHub (链接见原文 footnote 5，此处未提供)。
*   **模型**: Flan-T5-XL (3B, open source), OPT-IML-1.3B, LLaMA 2-Chat-7B (open source), GPT-3.5 (API)。
*   **关键超参数**: 各模型的最优 (temperature, top_p) 组合见 Table 1：Flan-T5 (0.7, 0.7), OPT-IML (0.9, 0.7), GPT-3.5 (0.9, 0.3), LLaMA 2-Chat (0.1, 1.0)。每个实验运行 3 个随机种子 (0, 21, 42) 取平均。
*   **提示词选择**: 对每个模型和每个数据集，在开发集上测试多种提示格式（包括标签顺序排列）和分布注入方式，选取 macro-$F_1$ 最高的作为最终报告结果（见 Appendix A/Table 7）。
