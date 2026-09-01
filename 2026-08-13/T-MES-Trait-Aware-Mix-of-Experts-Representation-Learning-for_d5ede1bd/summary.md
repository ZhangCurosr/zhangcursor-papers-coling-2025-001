---
title: "T-MES-Trait-Aware-Mix-of-Experts-Representation-Learning-for"
source: https://aclanthology.org/2025.coling-main.81.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:01:50"
field: "自然语言处理-教育文本分析"
keywords: ["自动作文评分", "多特征评分", "混合专家", "预训练语言模型", "对比学习"]
innovations: ["首个将MoE架构引入多特征作文评分，通过门控机制实现高效多任务表示学习", "提出评分多样性正则化（MHS）、特征表示相关性正则化和特征相关性损失三种策略联合优化"]
benchmarks: ["ASAP/ASAP++", "QWK"]
---

# 论文速读：T-MES: Trait-Aware Mix-of-Experts Representation Learning for Multi-trait Essay Scoring

## 一句话总结
本文提出基于特征感知混合专家（MoE）表示学习的多特征作文评分方法 **T-MES**，通过单一编码器结合门控MoE架构同时预测多个写作特征分数，并设计三种正则化策略（评分多样性正则化、特征表示相关性正则化、特征相关性损失）来增强表示多样性与特征间关联建模，在 ASAP/ASAP++ 数据集上实现优于现有基线的 QWK 成绩，同时显著降低计算开销。

## 研究问题与动机
1. **多特征评分的教育价值**：现有 AES 研究多聚焦整体评分或单一特征，但教育场景中需多维度反馈以帮助学生有针对性改进。
2. **预训练模型的效率瓶颈**：直接将 Transformer 应用于多特征评分需复制多个编码器，导致训练与推理效率低下。
3. **单一表示的局限性**：现有方法使用统一的 [CLS] 文档级表示处理所有特征，无法捕捉不同特征所需的差异化文本表示。
4. **MSE 损失忽略特征关联**：传统 MSE 损失仅优化单特征预测，忽视了不同写作特征之间的内在相关性和协同关系。

## 核心贡献（创新点）
1. **首个将 MoE 架构引入多特征作文评分**：通过门控机制让不同专家学习各特征的专属表示，单次前向传播即可输出所有特征分数，大幅提升多任务训练效率。
2. **评分多样性正则化（Scoring Diversity Regularization）**：利用最小超球面分离（MHS）方法最大化不同专家权重向量在单位超球面上的距离，促使各专家专注于目标特征。
3. **特征表示相关性正则化（Representation Correlation Regularization）**：基于对比学习拉近高度相关特征表示、推远低相关特征表示，增强跨特征协同学习能力。
4. **特征相关性损失（Trait Correlation Loss）**：基于真实标签的 Pearson 相关系数构建约束，引导预测分数分布与真实特征相关性保持一致。

## 方法详解
### 整体框架
- 编码器：采用 `RoBERTa_base`（129M 参数）获取 token 级上下文表示
- MoE 层：n 个专家并行全连接层，门控网络输出 softmax 权重控制各 token 对特征的贡献
- 特征表示：$F_i = \sum_{i=1}^{n} g_i \cdot f_i$，其中 $g_i$ 为门控分数，$f_i$ 为专家输出
- 评分头：$\hat{y}^k = \text{Sigmoid}(W_y^k F_k + b_y^k)$，输出归一化至 [0,1]

### 三种正则化策略
1. **评分多样性正则化 $L_{SD}$**：将专家权重投影到单位超球面 $S^{d-1}$，最大化不同专家间的最短距离：$\max \min_{i \neq j} d(\hat{w}_i, \hat{w}_j)$
2. **特征表示相关性正则化 $L_{RC}$**：构造对比损失，正样本对为余弦相似度最高的特征表示对，负样本对为相似度最低的，温度参数 $\tau=0.1$
3. **特征相关性损失 $L_{TC}$**：对真实标签 Pearson 相关系数 $P(y_j, y_k) \geq \delta$（阈值 $\delta=0.8$）的特征对，最小化预测向量间余弦距离

### 总损失函数
$$L_{Final} = \lambda L_{MSE} + (1-\lambda) L_{TC} + \alpha L_{RC} + \beta L_{SD}$$
关键超参：$\lambda=0.7, \alpha=0.1, \beta=0.0001, \delta=0.8, \tau=0.1$，dropout=0.1，lr=1e-5，batch_size=32，max_length=512，epochs=50

## 实验与结果
**数据集**：ASAP/ASAP++ 公开数据集，含 8 个提示（P1–P8），共约 13,978 篇作文，10 个评分特征（Content、Organization、Word Choice、Sentence Fluency、Conventions、Prompt Adherence、Language、Narrativity、Style、Voice）。

**评估指标**：Quadratic Weighted Kappa（QWK）

**主要结果**：
- T-MES 平均 QWK 达 **0.719**，较 BERT 基线提升约 **1.8%**（0.701→0.719）
- 较 Multi-task 基线 DualTrans（0.707）提升 **1.2%**
- 在 10 个特征中的 8 个上优于 DualTrans，仅在 Overall（0.774 vs 0.778）和 Voice（0.590 vs 0.619）上略低
- Voice 特征表现受限归因于样本极少（仅 726 篇）

**效率优势**：
- 参数量仅 **129M**，约为 DualTrans（277M）的一半
- 单次推理仅需 **0.014 秒**，远低于 DuplexTrans 的 0.030 秒
- 消融实验验证三种正则化均有效，全移除时 AVG QWK 降至 0.702

## 相关工作脉络
1. **MTL-BiLSTM（Kumar et al., 2022）**：基于 CNN-LSTM 的多任务学习方法，用辅助多特征任务辅助整体评分；T-MES 改用预训练 Transformer + MoE，效率更高且直接预测各特征。
2. **DualTrans（Cho et al., 2024）**：双尺度 Transformer 多特征学习方法；T-MES 参数量为其一半，且在多数特征上超越。
3. **Do et al.（2023a, 2023b）**：提示与特征关系感知的跨提示作文评分；本文聚焦同一提示下的多特征联合建模。
4. **传统 AES 方法（HISK、STL-LSTM）**：基于手工特征或浅层深度学习；T-MES 充分利用预训练语言模型的表征能力。
5. **MoE 在 NLP 中的应用**：本文首次将 MoE 引入多特征作文评分，解决多任务表示多样性与效率的平衡问题。

## 局限性与未来方向
1. **小样本特征性能下降**：Voice 等样本量严重不足（<600）的特征评分效果较差，需改进跨特征协同预测能力。
2. **缺乏多尺度表示**：当前方法仅学习 token 级表示，未融合句子级和段落级结构化信息。
3. **未来方向**：探索更丰富的特征协同策略、多尺度特征表示学习，以及预训练模型在教育文本评估中的深入应用。

## 研究启发与可借鉴点
1. **MoE 门控机制用于多任务表示学习**：通过可学习的门控权重让不同专家聚焦不同任务维度，适用于各类多标签/多输出回归任务。
2. **超球面多样性正则化（MHS）**：将权重投影到单位超球面并最大化分离距离，可有效防止专家坍缩，值得迁移至其他多专家场景。
3. **基于标签相关性的对比正则化**：利用真实标签的 Pearson 相关系数构建正负样本对，比单纯基于预测结果的对比学习更具语义指导意义。
4. ** plug-and-play 设计思想**：T-MES 可无缝接入任意 encoder-based 预训练模型（BERT/RoBERTa 实验结果相近），提供通用多任务扩展范式。

## 关键术语表
**Automated Essay Scoring (AES)**：利用人工智能技术对作文进行自动评分的研究方向。
**Multi-trait Scoring**：从多个维度（内容、组织、词汇等）对作文进行独立评分的任务。
**Mixture-of-Experts (MoE)**：由多个专用专家网络和一个门控网络组成的架构，根据输入动态选择专家组合。
**Quadratic Weighted Kappa (QWK)**：衡量预测评分与人工评分一致性的加权 Kappa 指标，广泛应用于 AES 评测。
**Minimum Hypersphere Separation (MHS)**：通过在单位超球面上最大化点间最小距离来增强模型组件多样性的正则化方法。
**Trait Correlation Loss (L_TC)**：基于真实标签 Pearson 相关系数约束预测分数分布一致性的损失函数。
**Contrastive Regularization (L_RC)**：通过拉近相关特征表示、推远不相关特征表示的对比学习正则化策略。

## 可复现要素
- **数据集**：ASAP/ASAP++（公开可用，https://www.kaggle.com/c/asap-asessments）
- **代码/权重**：论文未提及开源声明
- **关键超参**：$\lambda=0.7, \alpha=0.1, \beta=0.0001, \delta=0.8, \tau=0.1$；AdamW 优化器，lr=1e-5，batch_size=32，max_length=512，dropout=0.1，训练 50 epochs
- **预训练模型**：RoBERTa_base（129M 参数），亦兼容 BERT
