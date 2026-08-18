---
title: "Learning-Transition-Patterns-by-Large-Language-Models-for-Se"
source: https://aclanthology.org/2025.coling-main.171.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:42:31"
field: "大模型推荐系统"
keywords: ["Sequential Recommendation", "Large Language Model", "Item Linear Projection", "Transition Pattern", "Parameter-Efficient Tuning", "Generative Recommendation"]
innovations: ["将序列文本映射为序列item ID，通过multi-query捕获item间转换模式", "引入ILP线性投影替代item embedding table，支持全排序高效推理", "Efficient ILP Tuning冻结LLM仅训线性层，实现跨场景低开销适配"]
benchmarks: ["Amazon Review (Scientific, Instruments, Arts, Office, Games, Pet)", "MovieLens", "Steam"]
---

# 论文速读：Learning-Transition-Patterns-by-Large-Language-Models-for-Se

## 一句话总结
本文提出 ST2SI 框架，将序列推荐任务重构为"将 item 文本序列映射为 item ID 序列"，通过 multi-query input 和 item linear projection (ILP) 捕获用户行为序列中的 item 间转换模式，并通过 efficient ILP tuning 实现跨场景高效适配。

## 研究问题与动机
1. **现有 LLM 推荐方法忽略 item 间转换模式**：Recformer、LLM-Rec 等方法仅将序列文本整体映射到目标 item，未建模 $A \to B \to C \to D$ 中的动态转换关系。
2. **文本表示学习效率低下**：Recformer 需两阶段微调；LLM-Rec 需为每个目标 item 采样 1000 个负样本，计算开销大且效果次优。
3. **生成式推荐存在幻觉与低效问题**：直接生成 item 文本的开域 NLP 生成方式在推理时耗时且易产生幻觉（hallucination）。
4. **PLM 提取的 item 表示不适合推荐任务**：预训练语言模型主要面向 NLP 任务设计，提取的特征粒度粗（句子级），难以捕捉细粒度（词级）用户偏好。

## 核心贡献（创新点）
1. **提出 ST2SI 框架**：将 open domain 的语言建模目标重构为 item 建模目标，通过 multi-query input 捕获交互序列中 item 间的转换模式，区别于 Recformer/LLM-Rec 的"整体序列→目标item"范式。
2. **引入 Item Linear Projection (ILP)**：将最后一层 hidden representation 线性映射到 item ID 概率分布，无需维护 item embedding table，区别于传统 generative recommendation 基于 vocab 生成 token 的方式。
3. **提出 ID Alignment 机制**：通过 instruction tuning 解决 item text 与 item ID 之间的语义不对齐问题，显式强化 LLM 对"item 文本→item ID"映射的理解。
4. **设计 Efficient ILP Tuning**：冻结全部 LLM 参数，仅训练 per-scenario 的 ILP 线性层即可适配新场景，计算资源消耗远低于 full parameter tuning。
5. **系统性实验验证**：在 Amazon Review 六个下游数据集上，相比最佳基线平均提升 NDCG@10 (7.33%)、Recall@10 (4.65%)、MRR (8.42%)。

## 方法详解
1. **Multi-Query Input**：在序列 $S_u = \{v_1, v_2, ..., v_n\}$ 对应的文本序列中，每段 item text 后插入查询 token [Q]，构造 $X_q = \{T_1, [Q], T_2, [Q], ..., T_n, [Q]\}$，使模型在每个 [Q] 位置预测下一个 item。
2. **Item Linear Projection (ILP)**：将 LLM 最后一层在 [Q] 位置的 hidden state $h_{qi}^l \in \mathbb{R}^{d_k}$ 通过权重矩阵 $W \in \mathbb{R}^{d_k \times N}$ 投影并 Softmax，得到目标 item ID 的概率分布 $p_i$；训练目标为 $\mathcal{L}(v) = \sum_i \log P(v_{i+1}|T_1^q, ..., T_i^q; \Theta)$。
3. **ID Alignment (Instruction Tuning)**：构造指令如"Please give the ID of the following item: {item_text} [Q]"，目标为对应 item ID，强化 LLM 对 item text→item ID 映射的理解。
4. **训练目标**：联合优化 multi-query 预测损失与 instruction tuning 损失，均采用交叉熵损失：$\mathcal{L} = -\sum_{i=1}^{j} y_i \log(p_i)$。
5. **Efficient ILP Tuning**：预训练阶段在源域数据上训练 ILP；下游目标场景仅替换/微调对应 ILP 权重，冻结全部 LLM 参数，实现场景灵活切换。

## 实验与结果
- **数据集**：Amazon Review 共 8 个类别（2 个预训练域：Food、Cell；6 个下游目标域：Scientific、Instruments、Arts、Office、Games、Pet）。
- **基线**：GRU4Rec、SASRec、BERT4Rec、FDSA、S³-Rec、ZESRec、UniSRec、Recformer、LLM-Rec。
- **评估指标**：NDCG@10、Recall@10、MRR，full ranking leave-one-out 评估。
- **核心结果**：ST2SI（full parameter）在全部 6 个数据集上均取得最优；相对第二佳基线平均提升 NDCG@10 **7.33%**、Recall@10 **4.65%**、MRR **8.42%**。ST2SI_ILP（仅训 ILP）已优于多数传统基线。
- **ILP Accuracy**：Acc@1 在 Scientific (0.953)、Arts (0.979)、Games (0.945) 等数据集上均超过 94%，证明 item text→ID 映射的有效性。
- **消融**：移除 Multi-Query 或 ID Alignment 均导致性能下降；pre-training 对 ILP tuning 有正向增益，但对 full parameter tuning 反而不利。
- **多属性扩展**：添加 brand 属性普遍有益，添加 category 在部分数据集上带来额外提升（也带来下降，说明信息选择需谨慎）。

## 相关工作脉络
1. **Recformer (Li et al., 2023)**：将 item 编码为 key-value 属性对，两阶段预训练+微调学习语言表示；本文认为其范式低效且未建模 item 间转换模式。
2. **LLM-Rec (Tang et al., 2023)**：直接利用 LLM 进行多域行为建模，但需 1000 负样本采样，计算量大；本文以更高效的 ILP 替代负采样策略。
3. **P5 / InstructRec / KP4SR**：生成式推荐方法将推荐转为开域 NLG 任务，存在效率与幻觉问题；本文直接生成 item ID 规避上述缺陷。
4. **ZESRec / UniSRec**：使用 PLM 提取 item 文本特征后接入独立序列模型；本文统一在 LLM 内完成语言理解与序列建模，表征更细粒度。
5. **TALLRec / LLaRA**：针对全排序推荐场景，TALLRec 需大量负采样；LLaRA 需 beam search 解码耗时大；本文 ILP 投影机制天然支持全排序高效推理（单实例推理仅 0.02s）。

## 局限性与未来方向
1. **规模扩展性**：当 item 数量达到百万/亿级时，ILP 的 softmax 计算与存储开销巨大；作者提出可引入 hierarchical softmax 或多 ID 分配策略缓解。
2. **仅利用文本模态**：当前框架未融合图像等多模态信息，未来计划引入 item image 以增强多模态偏好挖掘。
3. **预训练数据敏感**：全参数微调时预训练反而损害下游性能（domain mismatch），说明 LLM 作为通用 foundation model 的潜力尚未完全释放。
4. **更大模型的 ILP 训练难度**：实验显示 opt-2.7b 与 llama-7b 在仅训练 ILP 时表现不及 opt-125m，可能存在过拟合或优化困难，需进一步研究。

## 研究启发与可借鉴点
1. **Multi-Query 序列建模范式**：将"整体序列→目标"改为"每个 item 后插入 query token→预测下一 item"的思路可迁移至任意序列决策任务（如对话管理、路径规划）。
2. **ID Linear Projection 替代 Embedding Table**：在 LLM 输出端用轻量线性层直接预测 item ID 概率分布，避免维护大规模 item embedding table，可借鉴于大item库推荐系统。
3. **Instruction Tuning 对齐语义不对齐问题**：通过显式 instruction 解决 item text 与 item ID 之间的映射缺失，可作为 LLM 适配推荐任务的通用接口设计参考。
4. **Pre-training 与下游 Tuning 的正交分析**：发现 pre-training 对 ILP tuning 有益但对 full tuning 有害的结论，提醒团队在 foundation model 微调策略上需区分参数更新范围再决定是否引入域内预训练。

## 关键术语表
**ST2SI**：Sequence Text to Sequence ID 的缩写，本文提出的将 item 文本序列映射为 item ID 序列的推荐框架。
**Multi-Query Input**：在每个 item text 后插入特殊查询 token [Q]，使模型能在每个位置独立预测下一个 item。
**Item Linear Projection (ILP)**：将 LLM 最后一层 hidden state 线性映射到 item ID 的概率分布，替代传统 item embedding lookup。
**ID Alignment**：通过 instruction tuning 显式训练 LLM 理解 item text 与 item ID 之间的对应关系。
**Efficient ILP Tuning**：冻结 LLM 全部参数，仅训练 per-scenario ILP 线性层，实现低成本跨场景适配。
**Full Ranking Evaluation**：在整个 item 全集上进行排序评估，区别于 sample-based 负采样评估。
**Leave-One-Out**：将每个用户最后一个交互作为测试集，其余作为训练/验证集的经典评估策略。

## 可复现要素
- **数据集**：Amazon Review（Food、Cell 作预训练；Scientific、Instruments、Arts、Office、Games、Pet 作评测），公开可获取，论文未提供预处理脚本链接。
- **代码/权重**：论文未明确声明开源，作者在补充材料提及 GitHub 仓库提供训练日志，但代码未直接公开。
- **关键超参**：max sequence length=50，max tokens per attribute=32，batch size=32，gradient accumulation=4；full tuning lr=5e-5，ILP tuning lr=3e-4（无预训练）/1e-4（有预训练）；optimizer=AdamW，early stop patience=5。
- **模型底座**：opt-125m（主要实验），补充实验涵盖 opt-350m/1.3b/2.7b 与 llama-7b。
- **硬件**：2×Tesla V100（opt-125m），4×V100（opt-2.7b），8×V100（llama-7b，出现 OOM）。
