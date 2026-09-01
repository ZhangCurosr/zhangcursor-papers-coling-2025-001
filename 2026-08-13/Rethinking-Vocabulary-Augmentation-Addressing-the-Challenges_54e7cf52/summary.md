---
title: "Rethinking-Vocabulary-Augmentation-Addressing-the-Challenges"
source: https://aclanthology.org/2025.coling-main.197.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:39:21"
field: "多语言自然语言处理"
keywords: ["词汇扩充", "低资源语言", "多语言模型", "熵一致性", "语义分布"]
innovations: ["提出基于语义分布与频率分布一致性的词汇选择方法ECWS", "引入类别定义词距离构建语义分布熵以弥补纯频率排序的缺陷", "在多种低资源语言分类任务上验证了该方法的有效性与稳定性"]
benchmarks: ["AfriSenti-SemEval", "GLUECoS", "Hate Speech (Bengali)", "IITP Product Review", "Headline Prediction (Gujarati)"]
---

# 论文速读：Rethinking Vocabulary Augmentation: Addressing the Challenges of Low-Resource Languages in Multilingual Models

## 一句话总结
本文针对多语言语言模型（MLLMs）在低资源语言（LRL）上因词汇表覆盖不足导致性能下降的问题，提出了**熵一致性词选择（ECWS）**方法，通过联合衡量词汇的语义分布熵与频率分布熵的一致性来筛选并扩充词汇表，显著提升了多语言分类任务的性能。

## 研究问题与动机
1.  **低资源语言表征缺陷**：现有MLLMs（如mBERT、XLM-R）在预训练阶段受限于语料不平衡，导致LRL词汇未能充分学习，或在下游任务中因子词切分过碎而无法被模型有效理解。
2.  **现有词汇扩充方法的局限**：主流的词汇扩充方法（如EVALM）主要依赖**词频排序**来选择需补充的脆弱词汇，但这忽略了词汇在模型内部表示分布与外部频率分布之间的本质差异。
3.  **语义与频率的偏差**：仅靠词频无法准确反映词汇是否被模型“真正掌握”。高频词可能因切分合理而表征良好，而某些在语义空间中分散、且频率分布与实际任务类别关联度低的词汇，才是真正需要扩充的“脆弱词”。

## 核心贡献（创新点）
1.  **提出了熵一致性词选择（ECWS）框架**：创新性地将基于语义空间的分布熵与基于下游任务频率的分布熵相结合，用于评估词汇的选择优先级，而非单纯依赖词频。
2.  **设计了语义分布熵（SEC）计算机制**：通过计算候选词汇与任务类别定义词（category-defining words）在模型嵌入空间的距离（余弦相似度倒数），量化词汇在语义类别间的离散程度。
3.  **实现了端到端的词汇优化验证**：在多种低资源语言（印地语、孟加拉语、古吉拉特语及12种非洲语言）的分类任务上，ECWS在平均性能上超越了包括FLOTA、FOCUS及EVALM在内的所有基线方法，达到了SOTA。

## 方法详解
ECWS方法包含三个核心步骤：
1.  **语义分布熵计算 (SEC)**：
    *   利用MLLM的Tokenizer对候选词 $w$ 进行子词切分，获取其语义表示向量 $e_w$（取[CLS]向量）。
    *   计算 $e_w$ 与各任务类别定义词向量 $e_c^l$ 的距离 $q_c$（定义为余弦相似度的倒数）。
    *   将距离归一化后视为概率分布，计算该词的语义分布熵 $H_{sd}(w) = -\sum q'_c \log(q'_c)$。熵值越高，说明该词在语义空间中与各类别定义词的距离越均匀（即缺乏明确的类别特异性）。
2.  **频率分布熵计算 (FEC)**：
    *   统计候选词 $w$ 在下游任务各类别 $c$ 中的出现频率 $n(w,c)$，构建多项分布 $p(c|w)$。
    *   计算频率分布熵 $H_f(w) = -\sum p(c|w) \log(p(c|w))$。
3.  **一致性计算与词选择 (CCWS)**：
    *   定义一致性得分 $r(w) = -\frac{H_{sd}(w) - H_f(w)}{H_{sd}(w)}$。
    *   **筛选逻辑**：一致性得分越低（即语义熵与频率熵差异越大，分布越不均衡），说明该词在频率统计上看似有用，但在模型语义空间中表征混乱，最需要被加入词汇表。
    *   过滤掉低频词后，选取一致性得分最低的 $Z$ 个词加入原词汇表 $V$，形成新词表 $V_{new}$ 进行微调。新词Embedding初始化为现有LRL子词或其对应英文翻译子词的加权组合。

## 实验与结果
*   **数据集**：IITP Product Review (Hindi), Hate Speech (Bengali), Headline Prediction (Gujarati), GLUECoS Sentiment (Hindi-English混合), AfriSenti-SemEval (12种非洲语言)。
*   **基线对比**：Fine-tune, FLOTA (分词优化), FOCUS (Embedding初始化), EVALM (基于频率的词汇扩充)。
*   **主要结果**：
    *   **整体表现**：ECWS在全部5个任务上的平均Macro F1达到 **70.28**，Accuracy达到 **71.48**，均优于第二名EVALM（平均F1 69.71，Acc 70.86），分别提升了 **0.57** 和 **0.62** 分。
    *   **具体任务**：在孟加拉语仇恨言论检测中，ECWS（F1 68.16）较EVALM（F1 67.20）提升明显；在印地语产品评论中达到72.28。
    *   **消融实验**：移除SEC（仅用频率）或FEC（仅用语义）均导致性能下降，证明两者互补，其中SEC的贡献略大。
    *   **词表规模分析**：随着扩充词数量增加，ECWS性能呈单调线性增长，而EVALM在某些任务上出现性能波动。

## 相关工作脉络
1.  **EVALM (Nag et al., 2023)**：本文最直接的对比基线，同样关注LRL词汇扩充，但EVALM主要基于子词熵和频率分布，未引入模型语义空间的距离度量。
2.  **FLOTA (Hofmann et al., 2022)**：属于分词器优化路线，通过调整分词策略保留形态完整性，不涉及词汇表扩充，本文将其作为非词汇扩充类的对比基线。
3.  **FOCUS (Dobler and de Melo, 2023)**：属于Embedding初始化路线，依赖静态词向量对齐，在小样本任务上表现不佳，仅在大样本AfriSenti任务中与ECWS有一定竞争。
4.  **Cross-lingual Vocabulary Adaptation (Yamaguchi et al., 2024)**：探索跨语言词汇适应，本文与之定位不同，侧重于通过一致性度量精选词汇以提升下游分类性能。

## 局限性与未来方向
1.  **模型范围限制**：目前仅验证了基于Encoder的模型（mBERT、XLM-R），未扩展到大规模Decoder类大语言模型（LLMs）。
2.  **类别定义词的局限性**：当前使用显式的类别关键词（如"positive", "negative"）作为语义锚点，这存在一定偏差，未来可探索更多样化的语义锚点生成方式。
3.  **应用场景单一**：主要验证了文本分类任务，在其他NLP任务（如NER、机器翻译）上的泛化能力尚待验证。

## 研究启发与可借鉴点
1.  **多维度一致性评估视角**：将“统计频率”与“模型语义分布”相结合来评估数据/特征质量，这一思路可迁移至其他低资源领域适配或特征选择任务中。
2.  **细粒度的Embedding初始化策略**：针对新加入的LRL词，结合现有目标语言子词与源语言（英文）对应子词进行初始化，是一种高效解决冷启动问题的技巧。
3.  **基于熵的损失/筛选函数设计**：利用熵来量化分布的不确定性或混乱程度，进而指导词汇选择，为模型压缩或蒸馏中的知识筛选提供了新的参考维度。

## 关键术语表
**Low-Resource Languages (LRL)**：低资源语言，指在预训练阶段可用文本数据极少，导致模型表征能力受限的语言。
**Category-defining words**：类别定义词，用于代表下游任务某一特定类别的典型词汇（如情感分析中的“positive”）。
**Semantic-distribution-based Entropy (SEC)**：语义分布熵，基于候选词与类别定义词在模型隐空间的距离计算得出的熵，反映词汇语义特征的分散程度。
**Frequency-based Entropy (FEC)**：频率分布熵，基于词汇在下游任务各类别中的统计频率计算得出的熵。
**Consistency Score**：一致性得分，用于衡量词汇在语义空间分布与频率统计分布之间的一致性，得分越低表示越需要被扩充。
**Vocabulary Augmentation**：词汇扩充，指在预训练词表基础上，针对特定下游任务或领域向模型词典中添加新词汇的过程。

## 可复现要素
*   **数据集**：IITP Product Review, Hate Speech (Bengali), Headline Prediction (Gujarati), GLUECoS Sentiment, AfriSenti-SemEval 均为公开数据集。
*   **代码/权重**：论文未明确声明代码开源仓库链接（截至阅读时），建议联系作者获取。基础模型mBERT和XLM-RoBERTa-base官方权重可公开获取。
*   **关键超参**：学习率 2e-5，最大序列长度 128，Batch size 16，Epoch数 15，Embedding初始化标准差 0.02，GPU为NVIDIA A100。
