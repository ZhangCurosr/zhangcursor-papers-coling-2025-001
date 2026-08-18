---
title: "Enhancing-Reranking-for-Recommendation-with-LLMs-through-Use"
source: https://aclanthology.org/2025.coling-main.45.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:30:50"
field: "LLM增强推荐系统"
keywords: ["推荐系统", "重排序", "大语言模型", "检索增强", "对比学习", "用户偏好"]
innovations: ["提出候选物品驱动的transformer检索器过滤LLM生成偏好", "设计对比学习与偏好-物品匹配双预训练目标对齐LLM与推荐模型"]
benchmarks: ["MovieLens-1M", "Amazon-book", "Steam"]
---

# 论文速读：Enhancing-Reranking-for-Recommendation-with-LLMs-through-Use

## 一句话总结
本文提出 UR4Rec 框架，通过小型 transformer-based 用户偏好检索器从 LLM 生成的用户偏好与物品知识中过滤出与候选物品相关的关键信息，并设计对比学习与偏好-物品匹配两个预训练目标对齐 LLM 与推荐模型的嵌入空间，从而增强传统推荐系统的重排序效果。

## 研究问题与动机
1. **LLM 生成信息冗余**：现有 LLM 增强推荐方法直接从 LLM 提取的用户偏好中包含大量与候选物品无关的无用信息（如图 1 所示），这些冗余噪声会干扰推荐性能。
2. **语义鸿沟问题**：LLM 未针对推荐任务训练，其生成的文本表示与传统推荐模型的向量空间存在语义差距，难以直接融合。
3. **传统方法的局限性**：传统推荐方法仅依赖 ID 特征建模用户序列，缺乏对显式用户需求和世界知识的利用。
4. **重排序任务的特殊性**：重排序阶段需要精确匹配候选物品与用户偏好的细粒度相关性，LLM 生成的粗粒度偏好难以直接适用。

## 核心贡献（创新点）
1. **提出 UR4Rec 框架**：通过用户偏好检索器构建 LLM 与推荐系统之间的桥梁，实现 LLM-to-recommendation 的对齐；与直接使用 LLM 生成内容的方法本质区别在于引入了检索与对齐机制。
2. **设计候选物品驱动的检索器**：以候选物品作为查询，通过 cross-attention 机制从 LLM 生成的偏好中过滤出与候选物品相关的必要信息；与 KAR 等方法直接使用 Hybrid-expert Adaptor 的本质区别在于检索器显式建模候选-偏好交互。
3. **引入两个预训练目标**：对比学习（CL）对齐 LLM 嵌入与推荐嵌入空间，偏好-物品匹配（PIM）通过二分类任务增强辨别能力；区别于先前工作仅使用单一对齐损失，双目标协同提升了检索质量。
4. **三阶段训练策略**：LLM 生成（冻结）→ Retriever 预训练 → 联合微调 Retriever 与推荐模型；与端到端微调 LLM 的方法相比，计算效率更高且无需修改 LLM 参数。

## 方法详解
**整体架构**：UR4Rec 由三部分组成：LLM 生成器、用户偏好检索器、推荐系统（骨干模型）。

**LLM 生成器**：
- **用户偏好生成**：设计 prompt 模板 $f_u$，以用户历史交互序列 $H = [i_1, i_2, ..., i_m]$ 的标题和类别信息为输入，调用 Llama2-Chat 生成用户偏好文本 $s_u$。
- **物品知识生成**：对每个历史物品设计 prompt 模板 $f_i$，生成物品知识文本 $s_i$。

**用户偏好检索器**：
- 基于 transformer 的小网络，每个 block 包含自注意力（MHAtt）、交叉注意力（CrossAtt）和前馈网络（FFN）。
- 引入 $K$ 个代理嵌入 $\mathbf{P}$ 作为可学习参数，与候选物品表示共同作为检索器输入。
- LLM 生成的文本通过冻结的 BERT Encoder 编码为向量后聚合：$\mathbf{e}_u^{\text{aggr}} = \text{Concat}(\mathbf{e}_u, \mathbf{e}_{i_1}, ..., \mathbf{e}_{i_m})$。

**预训练目标**：
1. **对比学习损失（$\mathcal{L}_{\text{CL}}$）**：使用 InfoNCE 损失，最大化正样本与过滤后偏好向量的相似度，最小化负样本相似度：
$$\mathcal{L}_{\text{CL}} = -\sum_{j=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{e}_j^{\text{pref}}, \mathbf{e}_j^{\text{pos}})/\tau)}{\sum_{k=1}^{M} \exp(\text{sim}(\mathbf{e}_j^{\text{pref}}, \mathbf{e}_j^{\text{neg}_k})/\tau)}$$
其中 $\text{sim}(\cdot)$ 计算代理嵌入与物品嵌入的最大余弦相似度。

2. **偏好-物品匹配损失（$\mathcal{L}_{\text{CF}}$）**：将代理嵌入与物品表示拼接后输入检索器，通过线性层预测匹配概率，使用 Binary Cross Entropy 损失：
$$\mathcal{L}_{\text{CF}} = -\sum_{j=1}^{N}[y_j \log \hat{y}_j + (1-y_j)\log(1-\hat{y}_j)]$$

**总预训练损失**：$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{CL}} + \alpha \cdot \mathcal{L}_{\text{CF}}$

**增强推荐系统**：将检索器输出的代理位置嵌入聚合为增强向量 $\mathbf{e}_i^{\text{aug}}$，拼接至骨干模型的输入中进行联合训练，优化目标为 $\mathcal{L}_{\text{RS}}$。

## 实验与结果
**数据集**：MovieLens-1M（6,040 用户/3,883 物品）、Amazon-book（11,906/17,332）、Steam（6,959/2,889），8:1:1 划分。

**基线模型**：
- 增强方法：BERT-aug、Llama2-Chat-aug、KAR
- 先进推荐器：Recformer、LRURec
- 骨干模型：DLCM、PRM、SetRank、GRU4Rec、SASRec

**主要结果**（Table 2, NDCG@5 示例）：
- **MovieLens-1M + DLCM**：UR4Rec 达 0.631，相对 KAR（0.424）提升 48.8%。
- **Amazon-book + PRM**：UR4Rec 达 0.433，相对 KAR（0.245）提升 76.7%。
- **Steam + SetRank**：UR4Rec 达 0.329，相对 KAR（0.284）提升 15.8%。
- 在全部 15 组实验（5 骨干 × 3 数据集）中均显著优于所有基线（p<0.05）。

**消融实验**（Table 3）：
- 移除检索器（w/o Retr）导致性能大幅下降（如 MovieLens-1M+PRM：0.618→0.348）。
- 移除 LLM（w/o LLM）性能显著降低，验证 LLM 知识增强的有效性。
- 预训练目标 ablation：w/o PIM 影响大于 w/o CL，说明偏好-物品匹配任务对对齐更重要。

**参数敏感性**：代理数量 K 的最优值因骨干模型历史长度而异（PRM 为 8，GRU4Rec 为 16）。

## 相关工作脉络
1. **KAR (Xi et al., 2023)**：使用 Hybrid-expert Adaptor 将 LLM 生成的知识转换为增强向量；UR4Rec 的区别在于通过检索器以候选物品为查询进行精确过滤，而非直接转换。
2. **LLM-as-Recommender (ChatRec, LLaMA-Rec)**：直接让 LLM 生成推荐结果；UR4Rec 属于增强传统推荐器的路线，保留原有推荐模型结构。
3. **U-BERT (Qiu et al., 2021)**：使用 BERT 编码用户偏好；UR4Rec 使用 LLM 生成更丰富的语义信息并通过检索器进行过滤。
4. **Rella (Lin et al., 2023)**：检索增强 LLM 用于序列推荐；UR4Rec 聚焦重排序场景，通过代理嵌入实现对齐。
5. **TallRec (Bao et al., 2023)**：通过指令微调对齐 LLM 与推荐；UR4Rec 不微调 LLM，保持其冻结状态以降低计算成本。

## 局限性与未来方向
1. **可解释性不足**：LLM 作为黑盒模型，无法提供推荐结果的可解释依据；未来可开发可解释组件。
2. **数据集规模有限**：仅在三个公开数据集上验证，未测试大规模工业场景。
3. **仅覆盖重排序任务**：框架针对 reranking 设计，在召回阶段的适用性待验证。
4. **LLM 依赖**：生成阶段需要调用 LLM，推理延迟可能影响在线服务。

## 研究启发与可借鉴点
1. **代理嵌入（Proxy Embeddings）机制**：将可学习代理作为检索器输入，通过 cross-attention 与 LLM 生成的文本交互，是一种高效的检索-对齐范式，可迁移至其他 LLM 增强场景。
2. **双目标预训练策略**：对比学习 + 匹配任务的组合有效弥合了 LLM 与领域模型间的语义鸿沟，该思路可扩展至多模态推荐。
3. **离线-在线分离训练**：LLM 生成与预训练可离线完成，仅检索器和骨干模型在线联合训练，兼顾性能与效率，适合工业部署。
4. **候选驱动的信息过滤**：以候选物品为查询过滤 LLM 生成内容的设计模式，可推广至问答、对话推荐等需要精准信息抽取的场景。

## 关键术语表
**UR4Rec**：本文提出的用户偏好检索增强推荐框架。
**Proxy Embeddings**：可学习的代理嵌入向量，作为检索器与 LLM 生成内容的交互接口。
**Contrastive Learning (CL)**：通过 InfoNCE 损失对齐 LLM 嵌入与推荐嵌入空间的预训练目标。
**Preference-Item Matching (PIM)**：二分类任务，预测过滤后的偏好是否与候选物品匹配。
**InfoNCE Loss**：对比学习常用的损失函数，最大化正样本对相似度，最小化负样本对相似度。
**Cross-Attention**：检索器中用于交互代理/物品与 LLM 生成文本的注意力机制。

## 可复现要素
- **数据集**：MovieLens-1M、Amazon-book（5-core）、Steam，均为公开数据集。
- **代码/权重**：论文未提供开源链接。
- **关键超参**：代理数量 K=8（DLCM/PRM/SetRank）或 K=16（SASRec/GRU4Rec）；批次大小 32；学习率 1e-4（前三骨干）或 1e-3（后两骨干）；嵌入维度 768；负样本数 10；候选物品数 100；历史长度 10 或 150。
