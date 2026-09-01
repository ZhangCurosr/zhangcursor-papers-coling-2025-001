---
title: "ROUGE-SciQFS-A-ROUGE-based-Method-to-Automatically-Create-Da"
source: https://aclanthology.org/2025.coling-main.149.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:58:53"
field: "科学自然语言处理"
keywords: ["Scientific Query-Focused Summarization", "ROUGE", "Dataset Construction", "TF-IDF", "Embedding Models", "Citation Analysis"]
innovations: ["基于引用关系的自动化 Sci-QFS 数据集构建方法", "发现传统 TF-IDF 方法在科学查询摘要任务上优于预训练深度学习模型"]
benchmarks: ["Average Precision", "ROC AUC"]
---

# 论文速读：ROUGE-SciQFS-A-ROUGE-based-Method-to-Automatically-Create-Da

## 一句话总结
本文提出了一种基于 ROUGE 的新方法论，利用学术文献中的引用关系自动构建科学查询摘要（Sci-QFS）任务的数据集，并发现经典 TF-IDF 方法在该任务上优于预训练的深度学习模型，揭示了该任务对现代 NLP 模型仍具挑战性。

## 研究问题与动机
1. **数据匮乏问题**：科学查询摘要（Sci-QFS）任务的发展落后于其他科学自然语言处理领域，主要原因是缺乏大规模训练数据
2. **人工标注成本高昂**：让领域专家阅读并总结长文档或文档集合是一项复杂且耗时的过程
3. **现有方法局限**：已有 QFS 数据集（如 DUC）缺乏多样性，且查询与文档集合的主题集中度过高
4. **模型性能瓶颈**：尽管大语言模型在许多基准上表现优异，但 Sci-QFS 任务对它们来说仍然困难

## 核心贡献（创新点）
1. **自动化数据集构建方法论**：利用学术论文中的引用关系隐式构建 QFS 训练样本，与需要人工标注的方法本质不同
2. **大规模 Sci-QFS 数据集**：构建了一个包含 8,965 个示例的新数据集，覆盖 AI 和 NLP 领域的科学文献
3. **反直觉的基准测试结果**：发现经典 TF-IDF 方法在某些指标上显著优于预训练的深度学习模型
4. **ROUGE 增强的数据优化**：使用 ROUGE 分数作为启发式方法，自动识别和添加更多相关句子
5. **全面的模型评估**：系统比较了 12 种不同嵌入模型和分类器的组合，为后续研究提供了详细基准

## 方法详解
1. **内容提取阶段**：使用 Science-Parse（AllenAI 开发的 LSTM 基础工具）从 PDF 文件中提取文本、元数据和引用信息，输出 JSON 格式
2. **数据表构建**：将原始 JSON 文件合并、清洗、去重后生成三个表：
   - Papers 表：包含 paper_id、title、abstract、text
   - References 表：包含 reference_id、title、total_citations、abstract
   - Citations 表：包含 paper_id、reference_id、internal_reference_id、context、start_offset、end_offset
3. **查询定义**：将被引用论文的摘要作为查询（query），被引用论文的完整文本作为文档集合
4. **标注生成**：通过检查论文中的句子是否在引用上下文中出现，为每个句子生成 True/False 相关性标签
5. **数据增强**：使用贪婪算法，基于 ROUGE 分数逐个添加句子到摘要中，直到分数不再提升
6. **模型架构**：将查询和句子嵌入到欧几里得空间，拼接后输入二进制分类器，使用二元交叉熵损失训练

## 实验与结果
1. **数据集规模**：1,365 篇 PDF 文件，最终获得 8,965 个示例，平均文档长度 353 个句子，正标签比例 3.9%
2. **最佳模型**：
   - TF-IDF Word unigrams + Cosine Similarity 分类器：Average Precision 0.197，ROC AUC 0.765
   - SPECTER embeddings + LSTM 分类器：Average Precision 0.208，ROC AUC 0.691
3. **关键发现**：
   - 传统方法（TF-IDF）在 ROC AUC 上优于现代预训练模型
   - LSTM 在序列感知分类器中表现优于 Transformer
   - SPECTER 嵌入在大多数情况下优于 Sentence-BERT
4. **性能瓶颈**：没有任何模型在 Average Precision 上超过 0.21，表明任务对自动方法仍然困难

## 相关工作脉络
1. **DUC 共享任务**（Dang, 2005-2007）：早期查询摘要基准，但存在主题集中度过高的问题
2. **CL-SciSumm 共享任务**（Jaidka et al., 2019）：科学文档摘要任务，通过启发式方法筛选重要论文
3. **HighlightROUGE**（Collins et al., 2017）：使用 ROUGE 自动提取科学论文摘要的方法
4. **SPECTER 嵌入**（Cohan et al., 2020）：基于引用的科学文档级表示学习方法
5. **QuOTeS 系统**（Ramirez-Orta et al., 2023）：交互式科学查询摘要系统，但数据集仅 23 个示例
6. **LMGQS 数据集**（Xu et al., 2023）：使用 InstructGPT 合成的 110 万示例 QFS 数据集

## 局限性与未来方向
1. **数据质量限制**：Science-Parse 可能无法提取所有引用，导致遗漏潜在相关句子
2. **硬件要求高**：数据增强过程计算成本高昂，需要 32GB+ RAM 和工作站
3. **文档长度限制**：长文档难以训练序列感知模型（LSTM、Transformer）
4. **未来方向**：
   - 研究为什么当前模型在该任务上表现不佳
   - 探索 LLM（如 GPT-4）在 Sci-QFS 任务上的行为
   - 使用 LLM 验证和增强现有数据集

## 研究启发与可借鉴点
1. **引用关系的数据挖掘价值**：学术论文中的引用结构可作为隐式标注信号，无需人工标注
2. **传统方法的复兴机会**：在特定专业领域，简单的统计方法可能优于复杂的深度学习模型
3. **ROUGE 作为优化目标**：ROUGE 分数可用于数据增强和模型评估，但需注意其与人类判断的差异
4. **领域适应嵌入的重要性**：SPECTER 在科学文献上优于通用嵌入模型（Sentence-BERT）
5. **任务难度基准测试**：该研究揭示了 Auto-QFS 任务的实际难度，为后续研究提供了 realistic baseline

## 关键术语表
**Scientific Query-Focused Summarization (Sci-QFS)**：针对科学文献的查询摘要任务，根据用户查询从多篇文档中提取相关信息生成摘要
**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**：基于召回的摘要自动评估指标，计算 n-gram 重叠度
**TF-IDF (Term Frequency-Inverse Document Frequency)**：词频-逆文档频率，经典的文本加权统计方法
**SPECTER**：基于引文信息的科学文档级 Transformer 嵌入模型
**Sentence-BERT**：使用 Siamese BERT 网络生成句子嵌入的预训练模型
**Average Precision**：衡量分类器在所有阈值下的平均精度，对类别不平衡敏感
**ROC AUC**：受试者工作特征曲线下的面积，衡量排序能力

## 可复现要素
1. **数据集**：8,965 个示例，训练集 7,172，开发集 897，测试集 897
2. **代码开源**：https://github.com/jarobyte91/rouge_sciqfs
3. **嵌入模型**：TF-IDF (word unigrams, char trigrams)、Sentence-BERT (all-MiniLM-L6-v2)、SPECTER (allenai-specter)
4. **分类器**：Cosine Similarity、Euclidean Distance、MLP、LSTM、Transformer
5. **超参数**：使用 Random Search 调优，学习率 10^-4，Adam 优化器，L2 正则化
