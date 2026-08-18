---
title: "Acquiring-Bidirectionality-via-Large-and-Small-Language-Mode"
source: https://aclanthology.org/2025.coling-main.116.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:29:34"
field: "token级序列标注"
keywords: ["命名实体识别", "双向语言模型", "单向语言模型", "少样本学习", "Token分类", "反向语言模型"]
innovations: ["通过拼接前向与反向LM表示实现伪双向性，无需微调大模型本身", "提出异构配置策略（大前向+小反向）兼顾性能与效率", "在少样本和罕见域场景下显著超越单向UniLM并优于BERT"]
benchmarks: ["CoNLL-2003", "Few-NERD"]
---

# 论文速读：Acquiring-Bidirectionality-via-Large-and-Small-Language-Mode

## 一句话总结
本文提出通过训练一个小规模反向语言模型（backward LM），将其表示与前向语言模型的表示拼接，在不微调大型前向模型本身的情况下赋予其伪双向能力，在token级分类任务上实现了超越BERT的效果，尤其在少样本和罕见域场景下提升显著。

## 研究问题与动机
- 大型单向语言模型（UniLM，如Llama-2、GPT系列）在生成任务上表现优异，但在token级分类任务（如NER、POS、chunking）上仍远逊于BERT等双向语言模型（BiLM），2024年CoNLL-2003 NER和DocRED的Top-3模型均基于BiLM。
- 现有弥补方案LLM2Vec需针对每个UniLM单独微调（移除因果注意力掩码后重新训练），在快速演化的UniLM生态中无法复用，成本高昂。
- 需要一个"one-for-many"的通用方案，使大型UniLM仅通过后处理即可获得双向表征能力，而无需访问其内部参数或梯度。

## 核心贡献（创新点）
- **伪双向性（Pseudo Bidirectionality）机制**：训练独立的小规模反向LM，将前向与后向表示拼接即可模拟BiLM，无需修改大模型本身；与LLM2Vec需要逐模型微调的本质区别在于该方案可跨模型复用。
- **异构配置兼容性**：允许前向模型与反向模型规模差异极大（$|θ| \gg |θ'|$），仅需共享vocabulary和tokenizer；区别于同类方法要求同构架构或同规模参数。
- **黑盒可用**：仅需获取UniLM的最终表示输出，无需访问内部层或反向传播；而LLM2Vec等方法要求完整模型可访问并修改。
- **少样本与罕见域突破**：在CoNLL-2003 K-shot设置（每类4K样本）及Few-NERD罕见域上，该方法稳定超越单向UniLM且在某些设置下优于BERT，填补了few-shot到many-shot之间的性能谷地。

## 方法详解
- **前向与后向表示计算**：前向LM计算 $\overrightarrow{\mathbf{h}}_i = \overrightarrow{UniLM}_\theta(x_i | \mathbf{x}_{<i})$，反向LM计算 $\overleftarrow{\mathbf{h}}_i = \overleftarrow{UniLM}_{\theta'}(x_i | \mathbf{x}_{>i})$，两者语义对齐但上下文方向相反。
- **拼接融合**：最终token表示为 $\mathbf{h}_i = \mathrm{Concat}[\overrightarrow{\mathbf{h}}_i, \overleftarrow{\mathbf{h}}_i]$，维度为两模型hidden dimension之和。
- **反向LM训练**：以GPT-2 architecture为基础，从随机初始化开始，在BookCorpus和Wikitext-103上shuffle文档级concatenation数据；用前向模型tokenizer做subword分词，截取1024-token片段后反转序列，以next-token prediction为训练目标；batch size=512，lr=2e-5，cosine scheduler。
- **下游任务适配**：冻结前后向LM参数，仅训练两层线性分类头（$\mathbf{W}_1 \in \mathbb{R}^{d \times d}$, $\mathbf{W}_2 \in \mathbb{R}^{c \times d}$），输入 $\mathbf{h}_i$，损失为标准cross-entropy；batch size=32，lr=1e-3线性衰减至0，AdamW优化。
- **Few-shot设置**：CoNLL-2003中每实体类型抽取K个样本，共4K训练数据；在验证集上搜索top-3超参组合（lr∈{9e-3,...,1e-4}，seed∈{10..19}，dropout∈{0,0.1,0.2,0.3}），报告三组平均test F1。

## 实验与结果
- **数据集**：CoNLL-2003（chunking、POS、NER）、Few-NERD（罕见域NER）。
- **基线模型**：BERT（bert-base-uncased, 109M）、GPT-2（base, 124M）、Llama-2（7B），以及对应的Concat变体（GPT-2+124M反向、Llama-2+124M反向）。
- **主要结果**：
  - 全量数据集上，Concat方法在所有三个任务上均提升；CoNLL-2003 NER F1较单向UniLM提升超过10分，且达到或超越BERT性能。
  - Few-NERD上Llama-2 Concat优于GPT-2 Concat，印证大前向模型对罕见域更有效。
  - Few-shot K-shot实验中，当训练样本不足16 shots/实体时，所提方法显著优于BERT；填补了zero-/few-shot与many-shot之间的性能谷地。
- **案例研究**：句子"Jones Medical completes acquisition."中，单向UniLM将Jones误标为B-PER，Concat方法正确识别为B-ORG；"and"连接的并列实体（Lotte/Hyundai/Samsung）场景下，前向模型难以判断首个实体类型，Concat利用反向上下文纠正。

## 相关工作脉络
- **Bidirectional LMs（BERT）**：本文方法的对比基准，证明单向+反向拼接可达到或超越BERT，且无需BERT式的掩码预训练。
- **LLM2Vec（BehnamGhader et al., 2024）**：通过移除causal mask后微调UniLM获得双向能力；本文方案不修改大模型、可复用反向LM且支持黑盒调用。
- **ELMo / BiLSTM**：早期通过双向编码获取上下文信息的思路；本文将其思想在大模型时代重新验证，并解决算力成本问题。
- **Meet-in-the-Middle（Nguyen et al., 2023; Li et al., 2023b）**：在生成过程中结合双向概率；本文聚焦表征质量而非生成。
- **Label Supervised LLaMA finetuning（Li et al., 2023a）、Looking Right（Dukic & Snajder, 2024）**：需完整微调UniLM；本文仅需后向小模型，对资源受限场景更友好。

## 局限性与未来方向
- 实验仅在英语token级分类任务（chunking/POS/NER）和两个前向模型（GPT-2 124M、Llama-2 7B）上进行，尚未验证文本分类（需pooling）或更大规模UniLM的效果。
- 与LLM2Vec未做定量对比（NER评测分数定义不同：span-level vs word-level），未来需统一评估协议。
- 反向LM本身有额外推理开销，但可通过复用已有前向模型部署来分摊成本。
- 未来方向包括：拓展至低资源/跨语言场景、探索BiLM替代反向LM提供后向上下文、使用更多参数规模的前向模型验证缩放规律。

## 研究启发与可借鉴点
- **"向后读取"的简单有效性**：仅需额外训练一个小型反向LM并以拼接方式融合表示，即可弥合单向模型在编码任务上的缺陷，实现接近BiLM的性能，方法论上极为轻量。
- **异构配置策略**：大前向模型+小反向模型的组合既保留大模型知识容量又控制训练成本，可迁移至其他需要双向上下文的编码场景（如关系抽取、句法分析）。
- **填补少样本性能谷地**：该方法在K-shot（4K~64K样本）区间表现尤为突出，为"需要适度标注但不足以全量训练"的实际场景提供了高性价比方案。
- **黑盒可访问性**：仅需API输出表征即可接入，适用于闭源商业模型的下游适配，降低了大模型部署门槛。

## 关键术语表
- **UniLM（Unidirectional Language Model）**：仅从前向上下文（历史token）计算表示的语言模型，如GPT系列，无法利用后续信息。
- **BiLM（Bidirectional Language Model）**：同时从前向和后向上下文计算表示的语言模型，如BERT，适合需要完整上下文的编码任务。
- **Pseudo Bidirectionality（伪双向性）**：通过拼接前向与后向表示模拟BiLM的双向能力，而非真正修改注意力机制。
- **Backward LM（反向语言模型）**：从句子末尾开始生成/预测的小型语言模型，提供后向上下文表示。
- **LLM2Vec**：移除causal mask后微调UniLM使其具备双向能力的已有方法，需逐模型微调且要求完整模型访问。
- **K-shot（少样本设置）**：从每实体类型中抽取K个样本（共4K）进行训练，评估模型在极少量标注数据下的泛化能力。
- **Span-level F1**：基于命名实体跨度（首尾token匹配）计算的F1分数，是NER任务的标准评测指标。

## 可复现要素
- **数据集**：CoNLL-2003、Few-NERD、BookCorpus、Wikitext-103；论文未明确声明是否全部公开，但均为公开数据集。
- **代码/权重**：论文已开源反向LM及训练代码（标注为GitHub链接，见注释²）。
- **关键超参**：反向LM训练：batch size=512，lr=2e-5，cosine scheduler，序列长度=1024；下游分类：batch size=32，lr=1e-3线性衰减至0，AdamW，两层线性分类头（隐藏维度=d，输出维度=c）；Few-shot：batch size=4，lr搜索范围{9e-3,...,1e-4}，dropout{0,0.1,0.2,0.3}，seed{10..19}。
- **模型配置**：GPT-2 base（124M）和Llama-2-7b作为前向模型；反向LM均为124M参数（GPT-2 architecture），随机初始化训练。
