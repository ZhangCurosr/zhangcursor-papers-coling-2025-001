---
title: "Looks-can-be-Deceptive-Distinguishing-Repetition-Disfluency"
source: https://aclanthology.org/2025.coling-main.15.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:43:33"
field: "多语言语音处理与自然语言处理交叉"
keywords: ["speech disfluency", "reduplication", "repetition", "sequence labeling", "multilingual NLP", "Indian languages"]
innovations: ["提出基于RiR语言学结构的新颖建模方法以区分形态重复与不流畅重复", "发布首个面向印地语、泰卢固语和马拉地语的词级红uplicated/repetition多语言数据集IndicRedRep"]
benchmarks: ["IndicRedRep (自建数据集)"]
---

# 论文速读：Looks-can-be-Deceptive-Distinguishing-Repetition-Disfluency

## 一句话总结
本文首次大规模研究了语音中“词形重复（reduplication）”与“重复性不流畅（repetition disfluency）”的区分问题，提出了多语言数据集 IndicRedRep 及基于 reparandum-interregnum-repair 结构的新方法，显著提升了多类词级分类性能。

## 研究问题与动机
- **问题核心**：在自发语音中，“词重复（repetition）”是一种无意义的不流畅现象，而“全重复（reduplication）”是一种具有语法/语义功能的形态学过程，两者形式相似但功能截然不同。
- **现有不足**：以往工作多关注不流畅检测或形态学分析，缺乏将两者进行区分的公开数据集与系统研究；现有语音语料库（如朗读语音）中真正的重复现象稀少。
- **应用价值**：准确区分二者有助于提升 ASR 系统的词错误率（WER），并改善下游机器翻译等任务的性能。
- **多语言挑战**：需验证该方法在词汇形态差异显著的多种语言（印地语、泰卢固语、马拉地语）中的泛化能力。

## 核心贡献（创新点）
1. **发布首个多语言标注数据集**：构建并公开 IndicRedRep，包含逾 4.5K 印地语、1.6K 泰卢固语和 1.6K 马拉地语的句子，提供词级（token-level）的 reduplication 和 repetition 标注。
2. **提出 RiR 结构建模方法**：首次将语音不流畅的 Reparandum-Interregnum-Repair 语言学结构引入模型输入，通过显式注入上下文结构信息来提升分类判别力。
3. **系统的跨语言对比评估**：在三种语言上全面评估从经典序列标注模型到多元 Transformer 模型的性能，揭示了语言特性对模型表现的差异化影响。

## 方法详解
- **数据集构建**：
  - 印地语数据源自 GramVaani 自发电话语音语料库，经启发式过滤（相邻词重复）和人工转录修正后，由三位语言学家进行标注，Fleiss' kappa 达 83.29%。
  - 泰卢固语和马拉地语数据通过 Gemma 指令微调模型将印地语句翻译/生成，再经母语者二次过滤与校正。
- **核心框架：RiR 结构**：
  - 定义：一个不流畅片段包含 **Reparandum**（将被重复或更正的词）、可选的 **Interregnum**（填充词、停顿或话语标记）和 **Repair**（最终更正/接续的词）。
  - 实现：使用正则表达式从输入文本中提取这三部分，并通过 `[SEP]` 标记拼接为额外特征输入模型。
- **建模方式**：
  - 任务形式化为序列标注（Sequence Labeling），采用 BIO 标注体系。
  - 输入为去除标点后的语音转写文本，输出为每个 token 的标签（B-Repetition, I-Repetition, B-Reduplication, I-Reduplication, O）。
  - 实验对比了 Logistic Regression、BiLSTM-CRF 基线，以及 bert-base-multilingual、XLMR、mT0、BloomZ、Gemma、ChatGPT 等多种模型，并评估 RiR 结构带来的增益。
  - 训练设置：batch size 为 8，最大 5 epochs，使用 AdamW 优化器，学习率 1e-5。

## 实验与结果
- **数据集规模**：印地语训练/验证/测试集分别为 3622/453/453 句；泰卢固语和马拉地语各约 1289/161/161 句。总重复（repetition）实例 3263 个，全重复（reduplication）实例 2340 个。
- **主要结果（Macro F1）**：
  - **最强结果**：ChatGPT (gpt-3.5-turbo) + RiR 在三种语言上取得平均 Macro F1 **89.04%**。
  - **XLMR-large + RiR** 取得优异且高效的性能，平均 Macro F1 为 **84.80%**，其中印地语达到 **85.62%**，泰卢固语 **83.95%**，马拉地语 **84.82%**。
  - **RiR 贡献**：在所有模型和语言上均带来稳定提升，例如 XLMR-base 从 81.11% 提升至 81.93%，bert-base-multilingual 从 77.13% 提升至 79.67%，平均提升约 **3%**。
  - **基线对比**：BiLSTM-CRF + RiR 平均 F1 为 55.71%，远低于 Transformer 模型，凸显了预训练语言模型的优势。
- **结论**：结合语言学先验的 RiR 结构能有效辅助 Transformer 模型更准确地区分两种易混淆现象。

## 相关工作脉络
1. **Reduplication as MWEs**：以往计算形态学研究（如 Chakraborty & Bandyopadhyay, 2010; Kulkarni et al., 2012）聚焦于单一语言的红uplicated词识别，未涉及与不流畅重复的区分，也未提供大规模词级标注数据集。
2. **Repetition as Disfluency**：传统方法（如 Liu et al., 2006; Zayats et al., 2016）主要解决不流畅检测问题，目标是将不流畅单元整体标记，而非细粒度区分其内部是“有意义的重复”还是“无意义的重复”。
3. **Shriberg (1994) 不流畅结构理论**：本文借鉴了其 Reparandum-Interregnum-Repair 分析框架，并将其操作化为模型可处理的输入特征，这是以往工作未深入探索的方向。
4. **现有语音数据集**：如 Common Voice、Shrutilipi 等多为朗读语音，其中重复现象极少；本文刻意选用自发语音语料（GramVaani）以确保数据的真实性。

## 局限性与未来方向
- **语言范围有限**：仅涵盖印地语、泰卢固语、马拉地语三种印度语言，方法的跨语言泛化能力有待在其他语系中验证。
- **未探索其他子词表示**：未使用 ELMo 等上下文子词嵌入，可能遗漏其他表示学习带来的收益。
- **RiR 提取依赖规则**：当前通过正则表达式提取 RiR 结构，可能对形态更复杂的语言或噪声更大的转写文本不够鲁棒。
- **未来方向**：扩展至更多语言；探索将 RiR 结构与其他语言学特征结合；结合大语言模型进行交互式 refine。

## 研究启发与可借鉴点
1. **语言学结构可作为可微/非微特征注入模型**：RiR 这种显式的语言学分析框架可以通过字符串操作（如正则、分隔符）无缝整合进神经网络输入，为其他语言学任务（如语调边界识别、句法歧义消解）提供了可复用的范式。
2. **低资源语言的“翻译+人工校正”数据构建策略**：对于缺乏标注资源的语言，利用高质量源语言数据和强大的指令微调 LLM 进行数据生成与平行化，再辅以母语者质量控制，是一种高效可行的路径。
3. **关注自发语音中的细微语言学现象**：许多现有数据集偏向书面语或朗读语音，忽略了自发口语中的复杂现象（如形态重复 vs. 不流畅重复），这提示在数据采集阶段需特别关注数据源的“自然度”。
4. **多模型对比评估的完整性**：同时评测从传统机器学习到各类前沿多语言 Transformer 及闭源 LLM 的表现，能更全面地反映任务难度和方法的相对优势。

## 关键术语表
- **Reduplication（全重复/重叠）**：一种形态学过程，通过完整重复词干来表达复数、强度、体貌等语法或语义意义。
- **Repetition（重复性不流畅）**：一种语音不流畅现象，指无意识的词语重复，通常反映认知加工、犹豫或纠错过程。
- **Reparandum-Interregnum-Repair (RiR)**：描述不流畅结构的语言学框架，分别指代“待重复/更正的词”、“可选的填充/停顿”、“最终的更正/接续部分”。
- **IndicRedRep**：本文发布的多语言数据集名称，包含印地语、泰卢固语和马拉地语的 reduplication 与 repetition 词级标注。
- **Sequence Labeling（序列标注）**：将标签序列（如 BIO）分配给输入序列中每个 token 的任务形式，常用于 NER、词性标注等。
- **Macro F1**：评估指标，对每个类别的 F1 分数求算术平均，能均衡反映少数类与多数类的性能，适合类别不平衡场景。

## 可复现要素
- **数据集**：IndicRedRep 数据集已公开，地址：https://github.com/arifahmad-py/IndicRedRep/
- **代码**：论文声明代码开源，详见上述链接。
- **关键超参**：batch size = 8，max epochs = 5，optimizer = AdamW，learning rate = 1e-5。
- **基线模型**：bert-base-multilingual，XLMR-base/large，mT0，BloomZ，Gemma，ChatGPT (gpt-3.5-turbo)，BiLSTM-CRF。
- **评估指标**：Precision, Recall, F1-score（Macro F1 averaged over 5 runs）。
