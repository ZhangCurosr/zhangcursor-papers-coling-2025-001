---
title: "LTRS-Improving-Word-Sense-Disambiguation-via-Learning-to-Ran"
source: https://aclanthology.org/2025.coling-main.132.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:42:03"
field: "词义消歧与词汇语义"
keywords: ["Word Sense Disambiguation", "Learning to Rank", "ListNet", "ListMLE", "Low-resource NLP", "Chinese WSD"]
innovations: ["将 listwise learning-to-rank 引入 WSD，通过排序扩展 sense 定义列表增强低频 sense 学习", "使用 BGE 句向量构建连续相似度 ground truth 替代离散标签", "构造统一候选定义集实现跨词知识迁移并提升训练并行效率"]
benchmarks: ["FiCLS", "MiCLS", "CCD sense inventory"]
---

# 论文速读：LTRS-Improving-Word-Sense-Disambiguation-via-Learning-to-Ran

## 一句话总结
本文提出 **LTRS**（Learning to Rank Senses）方法，通过将词义消歧（WSD）转化为对扩展 sense 定义列表的排序问题，使模型能够从更广泛的实例中学习相似 sense 的表示与区分；该方法在中文 WSD 上达到 **79.6% F1** 的 SOTA，在低资源设置（LFS、few-shot、zero-shot）下展现强鲁棒性，且训练收敛速度优于基线。

## 研究问题与动机
1. **低频 sense（LFS）学习不足**：现有监督 WSD 方法通常仅针对目标词的预定义 sense 进行分类，LFS 因训练样本稀少而性能显著下降，MFS 与 LFS 之间存在明显性能差距。
2. **相似 sense 的信息未被充分利用**：语言学上，具有相似 sense 的词往往出现在相似语境中，现有方法忽略了利用其他词的相似 sense 实例来辅助当前 sense 的学习。
3. **Loss reweighting 方法的局限性**：虽然部分工作通过 loss 重加权缓解类别不平衡，但在数据极度稀疏时易导致过拟合。
4. **传统 pointwise/pairwise 方法的不足**：Pointwise 忽略对象间关系，Pairwise 忽略全局位置，难以同时建模"最相关 sense"与"相似 sense"之间的相对次序。

## 核心贡献（创新点）
1. **提出 LTRS 框架**：将 WSD 转化为 listwise learning-to-rank 问题，通过对扩展 sense 定义列表排序来增强 sense 表示学习，与已有 bi-encoder/classifier 方法本质不同。
2. **扩展候选定义集**：将 mini-batch 中所有目标词的 sense 定义合并为统一候选集 $D_W$，使模型能够从跨词的相似 sense 实例中学习，这是与仅使用目标词自身定义的基线方法的本质区别。
3. **引入 BGE 构建 ground truth 排序**：使用 SOTA 句向量模型 BGE 计算 sense 定义间的语义相似度，构建连续的真实评分向量 $S_w^T$，替代传统离散标签，使排序信号更精细。
4. **适配 ListNet 与 ListMLE 至 WSD**：将两种 listwise LTR 损失函数引入 WSD 任务，通过温度参数 $\tau$ 控制分布平滑度，实现对 sense 全局排序的学习，与点/对方法形成对比。
5. **高效训练设计**：通过统一定义集解决不同词汇 sense 数量不均导致的并行效率问题，单 epoch 耗时从 BEM 的 24.2 分钟降至约 9.6–9.8 分钟。

## 方法详解
**任务形式化**：给定多义词 $w$ 及语境 $c_w$，从定义集 $D_w = \{d_i\}_{i=1}^{l}$ 中选择最优 sense 定义：$\hat{d} = \arg\max_d f(w,d)$。

**编码器**：采用双编码器架构，上下文编码器 $E_c$ 与定义编码器 $E_d$ 均初始化为 BERT；输入中目标词被替换为 [MASK]，$\mathbf{r}_w$ 取自 [MASK] 位置输出，$\mathbf{r}_d$ 取自 [CLS] 位置输出。

**统一候选集**：对 mini-batch $W=\{w_i\}$，构造 $D_W = \bigcup_i D_{w_i}$，预测分数列表 $S_w = [\phi(\mathbf{r}_w, \mathbf{r}_{d_i})]_{i=1}^{|D_W|}$，其中 $\phi$ 为余弦相似度。

**Ground truth 构建**：以正确定义 $d^*$ 为锚，使用 BGE 编码器 $E$ 计算 $S_w^T = [\phi(E(d^*), E(d_i))]_{i=1}^{|D_W|}$，得到连续相似度真值。

**ListNet 损失**：
$$\mathcal{L}_{\text{ListNet}} = -\sum_{i=1}^{|S_w|} P_{S_w^T}(i) \log P_{S_w}(i), \quad P_S(i) = \frac{e^{s_i/\tau}}{\sum_j e^{s_j/\tau}}$$
对 $S_w$ 和 $S_w^T$ 使用不同温度 $\tau_1, \tau_2$。

**ListMLE 损失**：
$$\mathcal{L}_{\text{ListMLE}} = -\log \prod_{i=1}^{k} \frac{e^{s_{\pi^T(i)}/\tau_3}}{\sum_{j=i}^{|S_w^T|} e^{s_{\pi^T(j)}/\tau_3}}$$
其中 $k<|S_w^T|$ 为截断超参，聚焦 top-k 排序；对高位排名赋予更大权重以强化高优 sense 的学习。

**优化配置**：$\tau_1=\tau_2=\tau_3=0.05$，$k=5$，AdamW 优化，学习率 5e-5，最多 20 epoch，batch size 在 {64, 128, 256} 中搜索。

## 实验与结果
**数据集**：融合 FiCLS 与 MiCLS，共 96,829 实例，覆盖 CCD 中 88.1% 多义词和 77.9% sense；按 7:1:2 划分训练/验证/测试集。

**基线**：MFS、BERT、GlossBERT、BEM、FormBERT、ESCHER。

**主要结果（F1%）**：

| 方法 | Valid | Test ALL |
|---|---|---|
| FormBERT | 78.3 | 78.1 |
| ESCHER | 78.3 | 77.9 |
| BEM | 75.8 | 75.5 |
| **LTRS_ListNet** | **80.2** | **79.6** |
| **LTRS_ListMLE** | 79.7 | 79.3 |

- LTRS_ListNet 超越 BEM（相同 bi-encoder 架构）**+1.5 F1**；超越 FormBERT **+1.5 F1**；超越 ESCHER **+1.7 F1**。
- 在所有词性（Noun/Verb/Adj./Adv.）上均获最佳。

**低资源子集（F1%）**：

| 方法 | MFS | LFS | Zero-shot |
|---|---|---|---|
| BEM | 86.3 | 71.6 | 62.3 |
| ESCHER | 87.1 | 70.8 | 57.6 |
| LTRS_ListNet | 85.6 | **75.3** | **70.0** |
| LTRS_ListMLE | 85.9 | 74.6 | 69.3 |

- LFS 提升 **+3.7 / +3.4 F1**；Zero-shot 提升 **+7.7 / +6.7 F1**，鲁棒性显著。

**Few-shot**：每 sense 仅 3 个训练实例时，LTRS 已达到 BEM 无限制数据的相近性能。

**训练效率**：LTRS_ListNet 单 epoch 9.6 分钟 vs BEM 24.2 分钟；100 分钟内即达最优验证性能。

**Batch Size 分析**：256 > 128 > 64，更大 batch 提供更高阶 sense 相似性知识。

## 相关工作脉络
1. **GlossBERT (Huang et al., 2019)**：将 sense 定义拼接至输入，通过 cross-encoder 学习匹配分数；LTRS 采用 bi-encoder + LTR 损失，支持跨词候选扩展与 listwise 全局排序。
2. **BEM (Blevins & Zettlemoyer, 2020)**：双编码器结构，用点积计算相似度并做分类；LTRS 继承 bi-encoder 形式但引入排序学习，利用跨 batch 相似 sense 定义构建连续真值信号。
3. **FormBERT (Zheng et al., 2021)**：融入形态学知识的 WSD 方法；本文与 FormBERT 在同一融合数据集上对比，LTRS 不依赖形态特征而通过排序机制提升 LFS。
4. **ESCHER (Barba et al., 2021a)**：提取式 sense comprehension 方法；LTRS 为生成式/表示式排序框架，架构更轻量（仅 110M 参数 vs ESCHER 的 larger 变体）。
5. **Z-reweighting (Su et al., 2022)**：针对 rare/zero-shot WSD 的 loss 重加权策略；LTRS 通过排序信号从相似 sense 实例中迁移知识，避免对稀缺数据的直接重加权过拟合风险。
6. **ListNet/ListMLE (Cao et al., 2007; Xia et al., 2008)**：经典 listwise LTR 方法，广泛应用于信息检索；本文首次将其引入 WSD 任务，并适配连续相似度真值替代离散 rank 标签。

## 局限性与未来方向
1. **数据集依赖性**：性能优势源于 lexical sample 数据集中较高比例的 LFS/zero-shot sense；在低频 sense 占比低的数据集上优势可能减弱。
2. **依赖外部句向量模型**：ground truth 排序需 BGE 等 SOTA 句向量模型，在低资源语言中句向量质量下降会削弱 LTRS 效果。
3. **细粒度 sense 区分不足**：对高度相近的细粒度 sense（如"观览1 vs 观览2"）无法获得显著提升，BGE 输出的相似度分过于接近导致排序信号模糊。
4. **未来方向**：将在更多语言（尤其是低资源语言）上评估；探索结合额外词汇语义和句法知识以改进细粒度消歧。

## 研究启发与可借鉴点
1. **Listwise LTR 引入分类任务**：将 listwise ranking 思想迁移到 sense 分类中，通过扩展候选集和连续真值信号增强表示学习，该思路可迁移至其他词汇语义任务（如 frame 析取、语义角色标注）。
2. **跨词/跨样本知识迁移**：在 mini-batch 内构建统一候选池，使模型能从同类词的相似 sense 实例中迁移学习，这一设计可推广至多词联合消歧或跨语言 WSD。
3. **BGE 构建 soft ground truth**：用预训练句向量模型生成连续相似度真值替代硬标签，提供了更丰富的监督信号；该方法可用于任何需要度量语义相似性的判别任务。
4. **训练效率优化**：通过统一候选集解决 variable-size 并行问题，单 epoch 提速约 2.5×；对 sense inventory 大小差异大的语言/任务有直接参考价值。
5. **Few-shot 场景的有效利用**：LTRS 在每 sense 仅 3 实例时即可达到强基线性能，其排序学习机制对样本效率的提升值得在低资源 NLP 任务中进一步探索。

## 关键术语表
- **Word Sense Disambiguation (WSD)**：词义消歧，根据语境确定多义词的特定含义。
- **Learning to Rank Senses (LTRS)**：本文提出的方法，通过让模型对 sense 定义列表排序来学习 sense 表示与消歧。
- **Listwise LTR**：列表级排序学习，同时考虑所有候选对象的全局排序顺序，而非独立打分或两两比较。
- **MFS / LFS**：Most Frequent Sense（最频繁 sense）/ Less Frequent Sense（低频 sense），后者因训练数据稀疏而更难消歧。
- **BGE**：BAAI General Embedding，SOTA 中文句向量模型，用于计算 sense 定义间的语义相似度。
- **Zero-shot WSD**：测试集中出现训练集未见过的 sense，评估模型对全新词义的理解能力。
- **Bi-encoder**：双编码器架构，分别编码 query 和 candidate 后计算相似度，相比 cross-encoder 更高效。
- **WrdInv / MorInv**：MiCLS 提供的词汇库存和形态库存，用于统一 sense 定义来源。

## 可复现要素
- **数据集**：融合 FiCLS + MiCLS（论文未单独开源融合集，但源数据集公开：FiCLS 与 MiCLS 均可从原论文获取）；Sense inventory 来自 WrdInv/MorInv（MiCLS 附录）。
- **代码**：论文未明确声明开源链接（ACL Anthology 页面链接为 https://aclanthology.org/2025.coling-main.132.pdf），代码可用性需进一步确认。
- **预训练模型**：chinese-bert-base-wwm-ext（110M）、bge-large-zh-v1.5（326M），均为公开模型。
- **关键超参**：$\tau_1=\tau_2=\tau_3=0.05$，$k=5$，batch size=256，学习率=5e-5，epoch=20，AdamW 优化。
- **硬件**：单卡 NVIDIA RTX 3090（43GB）。
