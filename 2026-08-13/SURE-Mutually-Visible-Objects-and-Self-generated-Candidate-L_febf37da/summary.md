---
title: "SURE-Mutually-Visible-Objects-and-Self-generated-Candidate-L"
source: https://aclanthology.org/2025.coling-main.31.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:59:35"
field: "信息抽取"
keywords: ["关系抽取", "候选标签标记", "双向注意力", "Pipeline模型", "CLMs", "M-DOS", "信息抽取"]
innovations: ["提出CLMs机制将关系抽取从填空式标签生成转化为选择题式标签选择", "设计M-DOS机制使对象标记相互可见以捕捉间接关系", "两阶段训练策略结合降序排列CLMs提升语义理解"]
benchmarks: ["ACE04", "ACE05", "SciERC"]
---

# 论文速读：SURE-Mutually-Visible-Objects-and-Self-generated-Candidate-L

## 一句话总结
本文提出 SURE（Shanxi University Relation Extraction system），一种基于 Candidate Label Markers（CLMs）的两阶段关系抽取方法，通过将传统的"填空"式标签生成转化为"选择题"式的标签选择，结合 M-DOS（对象间相互可见的方向注意力）机制增强实体对交互，在 ACE04、ACE05 和 SciERC 三个标准基准上达到 SOTA。

## 研究问题与动机
1. **Pipeline 模型忽略标签选择过程**：PURE、PL-Marker 等先进 pipeline 模型专注于生成正确标签，但忽略了标签的主动选择机制，导致模型难以区分正负样本。
2. **实体对交互捕捉不足**：现有方法如 PL-Marker 中，相同 subject 对应的多个 object markers 互不可见，无法捕捉对象间的间接关系（如 Figure 1 中 Apple enthusiasts 与 Apple Park 的关联）。
3. **计算复杂度与精度权衡困难**：Joint 模型虽能避免错误传播但计算复杂，而 PURE (Full) 有 O(N²) 复杂度，PURE (Approx.) 虽快但精度下降，需在效率与性能间取得平衡。
4. **填空题 vs 选择题的认知差异**：基于教育心理学研究（Medawela et al., 2017），同一问题在多项选择（MCQ）下的平均得分显著高于填空题（FITB），作者认为关系抽取亦可借鉴此原理。

## 核心贡献（创新点）
1. **提出 CLMs 机制实现标签从生成到选择**：通过 Stage 1 筛选 Top-n 最优标签和 Bottom-m 最差标签作为候选，将 RE 任务由 FITB 转化为 MCQ，与已有方法仅做标签生成的本质区别在于引入了负样本对比学习。
2. **设计 M-DOS 增强对象间交互感知**：使 levitated object markers 相互可见，允许模型通过中介对象推断实体对的间接关系（如通过 Apple enthusiasts 连接 Steve Jobs 与 Apple Park），区别于 PL-Marker 仅关注 subject-object 直接关系。
3. **两阶段训练策略与降序排列设计**：Stage 2 单独训练以建立基础语义理解，随后两阶段联合训练；CLMs 按概率降序排列而非随机打乱，经 ablation 验证结构化学习优于随机化，提升模型性能。
4. **即插即用的通用增强模块**：CLMs 可无缝集成至现有 RE 模型（如 PURE 和 PL-Marker），在不改变 Encoder 架构的前提下分别带来 1.8% 和 1.0% 的 Rel+ 提升。

## 方法详解
**整体架构**：基于 PL-Marker 构建的两阶段 pipeline 方法，NER 结果作为 RE 输入。

**Stage 1 (St.1) — CLMs 生成**：
- 无梯度传播的推理阶段，对每个实体对 $(s_i, s_j)$，计算表示 $\theta(s_i, s_j) = [h_{a-1}; h_{b+1}; h_c; h_d]$（分别为 subject 的 solid markers 嵌入和 object 的 levitated markers 嵌入）。
- 通过 Feed-Forward Network 得到关系类型概率分布 $P_r(r|s_i, s_j) = [p_0, p_1, ..., p_K]$。
- 按概率降序排列后，选取 Top-n（正标签）和 Bottom-m（负标签），转换为 CLMs 标记：
  ```
  CLMs = [Pos: t₀], [Pos: t₁], ..., [Pos: tₙ₋₁], [Neg: tₖ₋ₘ₊₁], ..., [Neg: tₖ]
  ```
- 将 CLMs 拼接到原文序列后形成新输入 $X = [\hat{X}; CLMs]$。

**Stage 2 (St.2) — CLMs 选择**：
- 带梯度传播的训练阶段，模型从 CLMs 中选择最可能关系。
- 引入 M-DOS（Mutual Directional Attention in Objects）机制，修改方向注意力矩阵，使 object marker O1 可与 O2 双向交互（PL-Marker 中 $(O1, O2) = 0$，SURE 中设为 1）。

**基本理解阶段**：初期仅用 St.2 训练数个 epoch，使模型积累足够知识后再引入两阶段训练，确保 CLMs 的有效性。

## 实验与结果
**数据集**：ACE04（4087 关系/6 类）、ACE05（7070 关系/6 类）、SciERC（4648 关系/7 类），使用官方划分或 5-fold 交叉验证。

**评估指标**：Boundary F1（Rel，span 边界+关系类型正确）和 Strict F1（Rel+，额外要求 entity type 正确）。

**基线模型**：DYGIE++、TriMF、UniRE、PURE、PL-Marker、Recollect、HGERE。

**主要结果**（Table 1）：
- **SciERC（SciBERT）**：SURE Rel+ = **44.3%**，超越 PL-Marker（41.6%）**+2.7%**，超越 HGERE（41.8%）**+2.5%**。
- **ACE05（BERT_BASE）**：SURE Rel+ = **67.6%**，超越 PL-Marker（66.5%）**+1.1%**。
- **ACE04（BERT_BASE）**：SURE Rel+ = **64.1%**，超越 PL-Marker（62.6%）**+1.5%**。
- **ACE05（ALBERT_XLARGE）**：SURE Rel+ = **71.6%**，超越 PL-Marker（66.5%）**+5.1%**；ACE04 Rel+ = **67.8%**，超越 PL-Marker（66.5%）**+1.3%**。
- 使用 HGERE NER 结果时，SciERC Rel 达 **56.9%**，为当前 SOTA。

**推理速度**（Table 2）：SURE 在 SciERC 上达 **210.4 sents/s**，较 PURE (Full)（92.8）提速约 2x，且精度高于 PURE (Approx.)（52.8 Rel → SURE 53.8 Rel）。

## 相关工作脉络
1. **PURE（Zhong & Chen, 2021）**：开创性引入 marker 机制的 pipeline 模型，处理复杂度 O(N²)；本文与其定位差异在于 PURE 是单对处理，本文通过 CLMs 实现多选一机制。
2. **PL-Marker（Ye et al., 2022）**：结合 PURE Full 与 Approx. 的混合策略，O(N) 复杂度，fix 一个 subject 处理多个 objects；本文在此基础上增加 M-DOS 使 objects 互见，并引入 CLMs 增强语义理解。
3. **HGERE（Yan et al., 2023）**：基于超图神经网络和高召回剪枝器的 joint 模型；本文将其作为 NER 源提供高质量实体边界，进一步提升 RE 性能。
4. **TPLinker（Wang et al., 2020b）**：单阶段 table-filling 联合抽取方法，规避 exposure bias；与本文 pipeline 思路不同，但共同目标是避免错误传播。
5. **UniRE（Wang et al., 2021b）**：统一标签空间方法处理对称关系；本文采用 directed relational instances 处理对称关系，与 UniRE 定位互补。
6. **Recollect（Wu et al., 2024）**：基于记忆流 tuning 的方法；本文 CLMs 为 self-generated candidate labels，无需额外参数 tuning。

## 局限性与未来方向
1. **训练需两次前向传播**：St.1 无梯度推理 + St.2 有梯度训练，计算开销增加；作者建议可通过增加 St.2 单独训练的 epoch 比例加速。
2. **CLMs 数量依赖任务标签规模**：最佳 (n, m) 组合因数据集而异（SciERC 用 (2,4)，ACE05 用 (3,2)，ACE04 用 (3,1)），缺乏自适应选择机制。
3. **仅验证于关系抽取任务**：尚未扩展至其他 IE 子任务（如事件抽取、情感分析）或视觉领域。
4. **依赖高质量 NER 输入**：本文承认若使用更准确的 NER 结果，性能仍有提升空间，端到端性能未充分验证。

## 研究启发与可借鉴点
1. **MCQ 范式迁移**：将语言理解任务从"填空"转化为"选择"的思路可迁移至其他 IE 子任务（如事件抽取中的 event type 判别），值得探索。
2. **负样本对比增强**：显式引入 Bottom-m 负标签并区分 Pos/Neg 标记，可视为一种轻量级对比学习策略，适用于任何需要细粒度分类的任务。
3. **双向交互注意力机制**：M-DOS 打破传统单向注意力限制，使对象间可互见，对共指消解、多跳推理等需要跨实体建模的任务有借鉴价值。
4. **结构化输入排序优于随机化**：ablation 显示降序排列 CLMs 优于随机打乱（+0.6%），提示在需要引入辅助信息时，保持语义结构可能比增强鲁棒性更重要。
5. **两阶段渐进训练策略**：先用 St.2 单独训练建立基础理解再引入完整流程，可作为复杂双阶段模型避免初期不稳定的一种通用技巧。

## 关键术语表
**CLMs（Candidate Label Markers）**：候选标签标记，由 Stage 1 筛选出的最优 n 个正标签和最差 m 个负标签转换而成的特殊 token，拼接至输入文本供 Stage 2 选择。

**M-DOS（Mutual Directional Attention in Objects）**：对象间相互可见的方向注意力机制，修改注意力矩阵使 levitated object markers 可双向交互，捕捉对象间的间接关系。

**PL-Marker**：Ye et al. (2022) 提出的 pipeline RE 模型，结合 subject-fixed 和 levitated markers 设计，O(N) 复杂度处理多对象关系。

**PURE**：Zhong & Chen (2021) 开创性 pipeline 模型，通过在实体对前后插入 marker 来聚焦关系预测，有 Full（O(N²)）和 Approx.（O(1)）两种推理模式。

**Rel / Rel+**：关系抽取评估指标，Rel 要求 span 边界和关系类型正确，Rel+ 在此基础上额外要求 entity type 正确（strict 版本）。

**FITB vs MCQ**：Fill-In-The-Blank（填空）与 Multiple Choice Question（多选）的认知对比框架，本文论证 MCQ 范式可提升模型标签预测能力。

## 可复现要素
- **数据集**：ACE04、ACE05、SciERC，均为公开数据集。
- **代码/权重**：论文声明代码已公开（"SURE, along with our code, is publicly available"），具体仓库链接见论文末尾。
- **关键超参**：
  - CLMs 数量：SciERC 用 (n=2, m=4)，ACE05 用 (n=3, m=2)，ACE04 用 (n=3, m=1)。
  - 学习率：BERT_BASE 用 2e-5，ALBERT_XLARGE 用 1e-5。
  - Epochs：SciERC 训练 20 轮，ACE04/ACE05 训练 30 轮。
  - Warm-up ratio：ACE04/ACE05 设为 0.33。
  - 随机种子：5 个（42, 43, 44, 45, 46）。
