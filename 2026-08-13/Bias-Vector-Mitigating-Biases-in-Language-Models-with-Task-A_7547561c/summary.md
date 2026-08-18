---
title: "Bias-Vector-Mitigating-Biases-in-Language-Models-with-Task-A"
source: https://aclanthology.org/2025.coling-main.190.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:00:45"
---

# 论文速读：Bias-Vector-Mitigating-Biases-in-Language-Models-with-Task-A

## 一句话总结
本文受 Task Arithmetic 启发，提出 Bias Vector 方法：通过将预训练语言模型在偏见语料上持续训练后的权重与原始权重作差得到“偏见向量”，再直接减去该向量实现去偏见。该方法无需人工构造去偏见数据，在 BERT/ALBERT/RoBERTa 上平均降低 SEAT 偏见效应值 0.177，且 GLUE 下游性能未受损。

## 研究问题与动机
- 语言模型会继承训练数据中的社会偏见与刻板印象，已引发实际公平性与歧视问题。
- 现有针对 Transformer 架构 LMs 的去偏见方法多依赖持续训练配合人工标注的反偏见数据，构建成本高、难以扩展。
- 词嵌入层面的去偏见技术（如主成分减法、零空间投影）作用于固定表示空间，无法直接迁移到参数驱动的 Transformer LMs。
- 任务算术表明模型权重差可编码特定任务/属性信息，启发本文探索“仅通过参数减法剥离偏见”的轻量路径。

## 核心贡献（创新点）
- 提出 Bias Vector 方法：构造预训练模型与偏见持续训练模型之间的权重差，并直接对预训练参数执行减法去偏。与现有依赖反事实语料或复杂投影空间的方法本质不同，彻底摆脱了人工去偏见数据的构建成本。
- 系统验证去偏见有效性：在 BERT、ALBERT、RoBERTa 上通过 SEAT 基准验证，平均 Cohen's d 效应值改善 0.177；在 GLUE 上证实去偏后模型下游任务性能保持（平均提升 0.23%）。
- 揭示缩放因子 λ 的临界效应：通过大范围扫描 λ 发现，适度缩放可有效降偏，但过大 λ 会引发预训练表示坍塌（Representation Collapse），使模型丧失区分刻板与反刻板信息的能力，而非真正完成去偏。

## 方法详解
- **持续训练构建偏见模型**：采用 Masked Language Modeling (MLM) 任务，使用 StereoSet intrasentence 数据集中仅填充 stereotype 选项的样本（共 8,498 句，覆盖 race/profession/gender/religion），对预训练 LM 进行 30 epoch 持续训练，使参数向偏见方向偏移，得到 $\theta_{bias}$。
- **构造 Bias Vector**：假设架构一致，直接逐参数计算差值：$V_{bias} = \theta_{bias} - \theta_{org}$，其中 $\theta_{org}$ 为原始预训练权重，$V_{bias}$ 即为编码偏见方向的参数向量。
- **去偏见参数更新**：$\theta_{debias} = \theta_{org} - \lambda V_{bias}$，$\lambda$ 为控制去除力度的缩放因子。Layer Normalization 层参数因仅负责分布归一化、不携带偏见语义，被显式排除在减法运算之外。
- **评估指标**：偏见幅度采用 SEAT 基准的 Cohen's d 效应值（计算目标句与属性句余弦相似度差异的标准化均值，越接近 0 表示偏见越低）；任务保持能力采用 GLUE 基准微调后的准确率/相关系数。

## 实验与结果
- **数据集与基准**：持续训练使用 StereoSet intrasentence；偏见评估使用 SEAT（含 SEAT-6/6b/7/7b/8/8b 等性别、种族子集）；下游任务保持使用 GLUE。
- **SEAT 结果（λ=1）**：BERT 效应值从 0.672 降至 0.447（↓0.225），ALBERT 从 0.675 降至 0.534（↓0.141），RoBERTa 从 0.733 降至 0.570（↓0.163），三者平均改善 0.177；BV(all, 1) 综合效果优于单类偏见训练。
- **GLUE 结果（λ=1）**：BERT/ALBERT/RoBERTa 平均分数分别由 0.776/0.779/0.794 变为 0.779/0.785/0.792，平均提升 0.23%，证明去偏未损害下游表示能力。
- **λ 敏感性分析**：λ 增大时 SEAT 效应值单调趋近于 0，但 λ≥10 时 ALBERT、λ≥100 时 BERT/RoBERTa 的 GLUE 性能显著下滑（如 BERT λ=100 时 avg 跌至 0.367）。作者证实大 λ 导致模型丧失区分能力（坍塌），而非真正消除偏见。
- **基线对比（性别偏见 SEAT-8）**：BV(gender, 100) 效应值达 0.188，优于 CDA/Dropout/SentDebias，仅次于 INLP，验证了方法在特定偏见维度上的竞争力。

## 相关工作脉络
- **Task Arithmetic (Ilharco et al., 2023)**：提出用任务向量编辑模型权重，本文首次将其思想迁移至“负向属性剥离”（偏见消除）任务，区别于原有的正向任务增强范式。
- **词嵌入去偏见 (Bolukbasi et al., 2016; Ravfogel et al., 2020 INLP)**：前者减去偏见主成分，后者通过迭代零空间投影剥离嵌入向量中的偏见信息；两者作用于固定输出表示层，而 Bias Vector 作用于 Transformer 全参数空间，无需训练专用投影器。
- **LM 持续训练去偏见 (Zmigrod et al., 2019 CDA; Liang et al., 2020 SentDebias; Webster et al., 2020 Dropout)**：依赖人工构造的反事实语料或引入额外对抗/投影损失；本文仅利用现有偏见语料反向推导权重差，完全无需反偏见数据。
- **偏见评测基准体系 (Meade et al., 2022; Nadeem et al., 2021 StereoSet; May et al., 2019 SEAT)**：本文严格沿用 Meade 的系统评测协议与开源代码，使用 StereoSet 作训练源、SEAT 作评估源以避免数据泄露，确保与已有工作的公平对比。
- **模型权重融合 (Wortsman et al., 2022; Matena & Raffel, 2022)**：关注多模型参数平均以提升鲁棒性；本文聚焦单一模型权重的有向减法操作，实现特定属性的精准剥离。

## 局限性与未来方向
- **文化与语料局限**：StereoSet 反映的是美国标注者的偏见认知，方法对其他文化/地域语境泛化性待验证。
- **评测范围受限**：受算力限制，未在 GPT-2 等解码器架构上验证；GLUE 仅评测了 λ=1/10/100 三个点，未覆盖中间 λ 对下游任务的连续影响。
- **性别偏见缓解较弱**：性别词汇在预训练语料中高频出现，且性别类偏见训练样本仅 996 句，导致 Bias Vector 捕获的信号有限。
- **未来方向**：将方法扩展至大规模 LLM（如 LLaMA、GPT 系列）；设计防重叠 Bias Vector 机制以解耦多偏见耦合；系统建模
