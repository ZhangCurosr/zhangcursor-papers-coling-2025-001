---
title: "CON-RECALL-Detecting-Pre-training-Data-in-LLMs-via-Contrasti"
source: https://aclanthology.org/2025.coling-main.68.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:01:32"
field: "大语言模型隐私与安全"
keywords: ["Membership Inference Attack", "Contrastive Decoding", "LLM Privacy", "Pre-training Data Detection", "WikiMIA"]
innovations: ["首次将对比解码引入成员推断，利用成员与非成员前缀的非对称分布偏移", "提出仅灰盒访问的高效评分公式，无需参考模型", "提出基于外部LLM的历史事件补全成员数据近似策略"]
benchmarks: ["WikiMIA", "MIMIR"]
---

# 论文速读：CON-RECALL-Detecting-Pre-training-Data-in-LLMs-via-Contrasti

## 一句话总结
本文提出 CON-RECALL，一种基于对比解码的预训练数据检测方法，通过同时利用成员与非成员上下文的非对称分布偏移来增强大语言模型的成员推断能力，在 WikiMIA 和 MIMIR 基准上均达到最新最优性能。

## 研究问题与动机
- **核心问题**：检测 LLM 是否"记忆"了特定训练数据（成员推断攻击），以应对隐私泄露、版权侵权和基准测试污染等安全风险。
- **现有方法局限一**：Loss、Min-K%、Min-K%++ 等方法仅分析目标文本自身分布，未利用上下文信息。
- **现有方法局限二**：ReCall 等基于参考的方法仅利用非成员上下文，忽略了成员上下文的潜在判别信号。
- **动机**：先前研究认为成员上下文带来的分布偏移微小、价值有限，但本文通过实验发现，成员与非成员上下文对目标文本的影响存在显著**非对称性**，对比利用二者可放大微小差异。

## 核心贡献（创新点）
1. **对比解码框架用于成员推断**：将对比解码思想从文本生成迁移到成员推断领域，首次系统性利用成员与非成员前缀的对比信号。
   — 与 ReCall 仅使用非成员前缀的本质区别在于引入双路对比，形成更强的分布偏移信号。

2. **非对称分布偏移的发现与量化**：通过 signed Wasserstein distance 揭示成员数据对非成员前缀敏感（负向偏移大），而非成员数据对成员前缀敏感（负向偏移大）的非对称现象。
   — 区别于以往仅关注 log-likelihood 绝对值的方法，本文聚焦于上下文引入的相对偏移方向与强度。

3. **仅灰盒访问即可运行**：CON-RECALL 仅需 token 概率（graybox），无需额外参考模型。
   — 相比 Ref 等方法需要完整参考模型，显著降低实际应用门槛。

4. **成员数据近似策略**：当无法获取真实成员数据时，利用外部 LLM 生成历史事件描述并截断，引导目标模型补全以模拟成员上下文。
   — 突破了依赖真实训练子集的局限，扩展了方法的适用场景。

## 方法详解
- **核心思想**：利用目标文本分别被成员前缀（$P_{member}$）和非成员前缀（$P_{non-member}$）修饰时的对数似然差异作为判别信号。
- **评分公式**：
  $$s(x, \mathcal{M}) = \frac{LL(x | P_{non-member}) - \gamma \cdot LL(x | P_{member})}{LL(x)}$$
  其中 $LL(\cdot)$ 表示对数似然，$\gamma$ 为对比强度超参数（默认优化范围 0.1–1.0）。
- **分布偏移量化**：引入 signed Wasserstein distance 度量前后缀条件下的分布变化：
  $$W_{signed}(P, Q) = \text{sign}(\mathbb{E}_Q[X] - \mathbb{E}_P[X]) \cdot W(P, Q)$$
- **实现细节**：WikiMIA 固定使用 7-shot 前缀；MIMIR 从 1 到 10-shot 搜索最优值；所有 prefix 样本从评估集中剔除以保证公平比较。

## 实验与结果
- **数据集**：WikiMIA（3 个子集：32/64/128 token）和 MIMIR（含 Wikipedia、GitHub、PileCC 等 7 个子集，7-gram 和 13-gram 设置）。
- **模型**：Mamba-1.4B、Pythia-6.9B/2.8B/12B、GPT-NeoX-20B、LLaMA-30B。
- **基线**：Loss、Ref、Zlib、Neighbor、Min-K%、Min-K%++、ReCall（7 种）。
- **WikiMIA 主要结果**（AUC 平均值）：CON-RECALL 达 95.8（32-tok）、97.7（64-tok）、95.9（128-tok），分别较 SOTA ReCall 提升 **7.4%、6.6%、5.7%**；TPR@5%FPR 平均提升 **30.8%**。
- **MIMIR 结果**：在 7-gram 设置下 CON-RECALL 在多数数据集上取得最高 AUC；在更难区分的数据子集（如 GitHub）上仍保持竞争力。
- **鲁棒性**：面对随机删除（10%/15%/20%）、同义词替换和改写（paraphrase）等文本扰动，CON-RECALL 性能下降最小，显著优于所有基线。
- **消融实验**：γ=0 时退化为 ReCall；γ 在 0.3–0.7 范围内表现稳定；shot 数增加带来性能提升。
- **成员近似**：零成员访问（zero access）场景下 AUC 仍达 87.5（WikiMIA-32），优于除 ReCall 外的所有基线。

## 相关工作脉络
- **Yeom et al. (2018) Loss**：直接用目标序列的负对数似然作为评分，无上下文对比。
- **Carlini et al. (2022) Ref**：需要额外训练参考模型校准损失，属黑盒/灰盒混合方法。
- **Shi et al. (2024a) Min-K%**：基于最小 k% token 概率均值，仅依赖目标文本自身。
- **Zhang et al. (2024) Min-K%++**：对 Min-K% 做 token 级 log-probability 归一化改进。
- **Xie et al. (2024) ReCall**：计算带非成员前缀与不带前缀的条件对数似然比值，是当前 SOTA，但仅用单路上下文。
- **定位差异**：CON-RECALL 在 ReCall 基础上引入成员前缀对比，形成双路不对称信号，是方法论上的进一步升级而非简单叠加。

## 局限性与未来方向
- **灰盒限制**：仅适用于可获取 token 概率的场景，无法直接应用于纯 API 黑盒调用（如商用 ChatGPT）。
- **前缀选择依赖**：性能受成员/非成员前缀质量影响，自动化最优前缀选择仍是开放问题。
- **对抗鲁棒性边界**：仅验证了基础文本扰动，面对更复杂的对抗性规避手段（如针对性扰动设计）尚需进一步检验。
- **伦理风险**：技术可能被滥用以提取敏感信息，需配套负责任使用指南。

## 研究启发与可借鉴点
1. **对比信号在分布差异微小场景下的放大价值**：即使单个上下文带来的偏移微弱，通过精心设计的对比结构仍可提取强判别信号，该思路可迁移到其他模型解释性任务。
2. **Signed Wasserstein distance 作为分布偏移度量**：相比 KL 散度等对称度量，带符号版本能保留偏移方向信息，适合用于分析上下文对模型输出的因果性影响。
3. **成员数据近似策略的通用性**：用外部 LLM 生成"伪成员"的思路可推广至其他需要成员样本的攻击或评测场景（如数据指纹检测）。
4. **消融实验设计**：通过 γ=0 精确锚定基线方法，清晰展示增量贡献，实验设计严谨值得借鉴。

## 关键术语表
**Membership Inference Attack (MIA)**：判断特定数据点是否属于模型训练集的攻击方法，是评估模型隐私风险的核心工具。
**Graybox Access**：攻击者无法获取模型权重，但可查询输入对应的 token 概率分布，是现实中最常见的模型交互形式。
**Contrastive Decoding**：原为文本生成技术，通过对比不同条件（如相关/无关上下文）下的模型输出分布来引导生成；本文首次将其迁移至 MIA。
**Signed Wasserstein Distance**：Wasserstein 距离的有符号变体，既度量分布差异幅度又保留偏移方向（正/负）。
**Shots**：构成前缀的样本数量；本文固定 WikiMIA 用 7-shot，MIMIR 在 1–10 间搜索。
**TPR@5%FPR**：在假正率 5% 约束下的真正率，衡量低误报场景下的检测能力，比 AUC 更贴近实际应用需求。
**Non-member Prefix**：取自模型训练集之外的文本片段，用于构造对照组以放大成员/非成员分布差异。

## 可复现要素
- **数据集**：WikiMIA（公开）和 MIMIR（公开），均来自已有开源基准。
- **代码**：论文未提供代码仓库链接，但所有方法和超参均有详细公式与实现说明。
- **关键超参**：γ ∈ {0.1, 0.2, …, 1.0} 网格搜索；WikiMIA 固定 7-shot；MIMIR 在 1–10-shot 间搜索最优。
- **模型来源**：HuggingFace，部署于 4×NVIDIA RTX 3090。
- **成员近似**：使用 GPT-4o 生成历史事件描述，prompt 细节见 Appendix C。
