---
title: "Enhancing-Long-range-Dependency-with-State-Space-Model-and-K"
source: https://aclanthology.org/2025.coling-main.148.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:40:48"
field: "细粒度情感分析"
keywords: ["Aspect-Based Sentiment Analysis", "State Space Model", "Mamba", "Kolmogorov-Arnold Networks", "Graph Convolutional Network", "Long-range Dependency"]
innovations: ["首次将选择性状态空间模型Mamba引入ABSA任务以增强长距离依赖建模", "提出基于KAN的门控融合机制自适应集成语法与语义特征", "构建MambaFormer模块协同处理长短距离依赖"]
benchmarks: ["Restaurant14", "Laptop14", "Twitter"]
---

# 论文速读：Enhancing-Long-range-Dependency-with-State-Space-Model-and-K

## 一句话总结
本文提出MambaForGCN框架，将选择性状态空间模型（Mamba）与图卷积网络（GCN）结合，首次将Mamba引入ABSA任务以增强方面词与意见词之间的长距离依赖建模，并通过Kolmogorov-Arnold Networks（KAN）门控融合自适应集成语法与语义特征，在三个基准数据集上达到SOTA性能。

## 研究问题与动机
- **注意力机制的二次复杂度限制**：标准注意力机制在处理方面词与意见词之间的长距离依赖时计算开销大，且易误判无关上下文词，限制模型对短距离依赖的有效性。
- **现有方法未能有效融合语法与语义信息**：虽然部分研究尝试结合语义与句法方法，但缺乏有效整合两种信息的机制，导致性能次优。
- **隐含意见词的处理困难**：传统模型过度依赖显式的方面-意见词对，难以识别未明确陈述但对情感分析有贡献的隐含意见词。

## 核心贡献（创新点）
1. **首次将选择性状态空间模型（Mamba）引入ABSA任务**：通过Mamba模块捕捉方面词与意见词之间的长距离依赖，突破注意力机制的二次复杂度瓶颈。
2. **提出基于Kolmogorov-Arnold Networks的门控融合机制**：利用KAN的可学习激活函数自适应地整合SynGCN与MambaFormer的特征表示，有效过滤无关噪声信息。
3. **构建MambaFormer模块协同处理长短距离依赖**：将Multihead Attention（MHA）与Mamba块结合，分别捕捉短距离与长距离依赖，实现互补建模。
4. **在三个人气数据集上验证有效性并超越SOTA基线**：在Restaurant14、Laptop14和Twitter数据集上均取得最佳或接近最佳的准确率与F1分数。

## 方法详解
- **输入嵌入模块**：使用BiLSTM或BERT对句子进行编码，提取方面词对应的子序列作为初始节点表示；BERT采用"[CLS] sentence [SEP] aspect [SEP]"格式输入。
- **基于语法的图卷积网络模块（SynGCN）**：利用LAL-Parser生成所有可能依存弧的概率矩阵，构建句法邻接矩阵$A^{\mathrm{syn}} \in \mathbb{R}^{n \times n}$，通过图卷积层（式15）学习句法结构表示$H^{\mathrm{syn}}$，降低依赖解析误差的影响。
- **MambaFormer模块**：包含MHA块与Mamba块两个子模块。MHA块通过标准Transformer注意力机制捕捉短距离依赖（式16-19）；Mamba块通过线性投影、1D卷积、SiLU激活和SSM模块（式20-24）捕捉长距离依赖，并利用乘法门控机制融合两条投影路径的输出。
- **KAN门控融合模块**：通过KAN网络生成门控映射$\mathrm{Gate}^{\mathrm{syn}}$和$\mathrm{Gate}^{\mathrm{sem}}$（式25），利用门控加法融合机制（式26-27）自适应整合语法与语义特征表示，然后通过均值池化与线性分类器输出情感极性概率（式28-29）。
- **训练目标**：采用标准交叉熵损失函数（式30），在所有句子-方面词对上计算预测情感极性的负对数似然。

## 实验与结果
- **数据集**：SemEval 2014 Task的三个公开数据集——Restaurant14、Laptop14、Twitter，统计信息见表1。
- **评估指标**：准确率（Acc.）与宏观平均F1分数。
- **主要结果**：
  - MambaForGCN在Restaurant14上达到84.38%准确率与77.47% F1，在Laptop14上达到78.64%准确率与76.61% F1，在Twitter上达到75.96%准确率与74.77% F1，全面超越无BERT的SOTA基线模型。
  - 结合BERT后，MambaForGCN+BERT在Laptop14上达到81.80%准确率与78.59% F1，在Twitter上达到77.67%准确率与76.88% F1，表现优异。
  - 消融实验（表3）表明：移除MHA导致Restaurant/Laptop/Twitter准确率分别下降1.59%/1.43%/1.13%；移除Mamba导致下降1.71%/1.68%/1.41%；替换KAN门控融合为全连接网络导致下降1.90%/1.58%/1.81%。
- **最强结果**：MambaForGCN+BERT在Laptop14上达到81.80%准确率，较次优模型DGGCN+BERT（81.30%）提升0.5个百分点；在Twitter上达到77.67%准确率，较次优模型IA-HiNET+BERT（77.59%）提升0.08个百分点。

## 相关工作脉络
- **注意力机制类方法**：IAN、MGAN、Coattention-Memnet等通过交互式或多重注意力机制捕捉方面-上下文交互，但受限于二次复杂度和短距离建模能力。
- **句法增强GCN方法**：CDT、ASGCN-DT、Sentic-GCN、EK-GCN等将依存句法信息融入图卷积网络，但未有效处理长距离依赖。
- **知识感知图网络**：KDGN、DGGCN等整合领域知识与情感知识增强ABSA，但在隐式意见词识别上仍有不足。
- **Transformer-Mamba混合架构**：Jamba、Block-state Transformers等工作探索注意力与SSM的混合，本文首次在ABSA任务中验证该架构的有效性。
- **KAN网络应用**：KAN作为新兴架构在数学函数拟合中展现优势，本文首次将其引入情感分析的门控特征融合任务。

## 局限性与未来方向
- **泛化能力受限**：模型在三种不同语言模式、领域特定术语或未登录词的真實世界数据上可能表现波动。
- **隐含意见词识别仍具挑战**：虽然KAN能够捕捉非线性依赖关系，但对于高度隐晦或文化背景依赖的意见词推理仍需改进。
- **未来可探索更复杂的依赖图构建策略**以及跨语言/跨领域的迁移学习机制。

## 研究启发与可借鉴点
- **Mamba在序列建模中的效率优势**：SSM的线性复杂度特性可推广至其他需要长距离依赖建模的NLP任务，如文本分类、命名实体识别等。
- **KAN门控融合的架构设计**：利用可学习激活函数替代传统Sigmoid/Softmax门控，为多模态/多源特征融合提供新思路。
- **句法概率矩阵的鲁棒性设计**：采用依存弧概率分布而非硬决策可提升句法特征的容错性，值得在图神经网络任务中借鉴。
- **长短距离依赖的显式分离建模**：MHA与Mamba的分层设计可启发其他任务中多尺度依赖建模的架构选择。

## 关键术语表
- **Aspect-Based Sentiment Analysis (ABSA)**：细粒度情感分析，针对文本中特定方面词（aspect）识别情感极性（正面/负面/中性）。
- **State Space Model (SSM)**：状态空间模型，一种用于序列建模的连续系统离散化方法，Mamba是其选择性版本，具备线性复杂度。
- **Kolmogorov-Arnold Networks (KANs)**：基于Kolmogorov-Arnold表示定理的网络架构，用可学习的一维B样条函数替代固定激活函数，增强非线性拟合能力。
- **Graph Convolutional Network (GCN)**：图卷积网络，通过图结构上的消息传递聚合节点邻域信息，用于捕捉句法依存关系。
- **MambaFormer**：本文提出的混合模块，结合Multihead Attention与Mamba块，分别处理短距离与长距离依赖。
- **Selective State Space Model**：Mamba的核心机制，通过输入依赖的选择性扫描动态过滤无关信息，提升序列建模效率。
- **LAL-Parser**：基于自注意力机制的依存句法解析器，输出所有可能依存弧的概率分布。
- **Gated Fusion**：门控融合机制，通过可学习门控信号加权整合多源特征表示。

## 可复现要素
- **数据集**：SemEval 2014 Task 4的Restaurant14、Laptop14、Twitter三个公开数据集，可从竞赛网站获取。
- **代码开源情况**：论文未提及代码开源。
- **关键超参数**：Glove 300维预训练词向量，BiLSTM隐藏维度50，dropout 0.7，SynGCN与MambaFormer各2层，MHA 4头，Mamba层16维状态向量，学习率0.002，batch size 16，训练50轮。
- **依赖解析工具**：LAL-Parser（Mrini et al., 2019）。
- **实现框架**：PyTorch。
