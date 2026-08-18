---
title: "MoSLD-An-Extremely-Parameter-Efficient-Mixture-of-Shared-LoR"
source: https://aclanthology.org/2025.coling-main.111.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:44:32"
field: "大语言模型参数高效微调"
keywords: ["LoRA", "Mixture of Experts", "Parameter-Efficient Fine-tuning", "Multi-Task Learning", "Model Sharing", "Dropout"]
innovations: ["共享LoRA上投影矩阵A以解耦跨任务通用特征与特定特征", "在共享A矩阵上引入dropout策略缓解过拟合与优化不平衡"]
benchmarks: ["OBQA", "CSQA", "Race", "MCTest", "Arc-e", "Arc-c", "MMLU", "GSM8K"]
---

# 论文速读：MoSLD-An-Extremely-Parameter-Efficient-Mixture-of-Shared-LoR

## 一句话总结
MoSLD 提出了一种参数高效的混合共享 LoRA（Mixture-of-Shared-LoRAs）多任务微调方法：通过共享 LoRA 上投影矩阵 A 来捕捉跨任务通用特征，并在 A 上引入 dropout 策略缓解过拟合与优化不平衡，在单任务与多任务混合设置下均优于现有基线。

## 研究问题与动机
1. **LoRA 在多任务混合数据下性能下降**：不同任务数据存在异质性与不均衡，导致知识冲突与干扰，使 plain LoRA 在混合训练时表现劣于单任务，甚至低于 FP-tuning。
2. **现有 MoE-LoRA 方法忽视跨域数据冲突**：MoLoRA、SiRA、MoLA、MixLoRA 等方法主要通过引入多个独立 LoRA 专家并结合路由机制来融合多任务，但过于强调各专家独特性，忽略了跨任务通用知识迁移。
3. **MoE 架构引入大量额外参数**：多个 LoRA 模块使训练参数量显著膨胀（如 MoLA 约为 plain LoRA 的 5 倍），计算成本高昂，以牺牲效率换取效果。
4. **LoRA 本身存在过拟合风险**：通用特征矩阵更新频率高于特定特征矩阵，导致优化不平衡，加剧多任务场景下的参数冗余与过拟合。

## 核心贡献（创新点）
1. **共享机制（Sharing Mechanism）**：将 LoRA 的 A 矩阵作为通用特征矩阵在所有专家间共享，B 矩阵保留为各专家专属特定特征矩阵；与 MoLA/MixLoRA 等完全独立专家方案本质区别在于显式解耦通用/特定特征并强制跨任务知识共享。
2. **A 矩阵 Dropout 策略**：在共享的通用特征矩阵 A 上按概率 p 随机丢弃更新，以平衡 A/B 矩阵的优化节奏；与常规 dropout 应用位置不同，本文首次将 dropout 精准作用于 MoE 共享层以缓解冗余与过拟合。
3. **极致的参数效率**：在 LLaMA2-7B 上 MoSLD 训练参数量仅为 FP-tuning 的 20.6%，且比 MoLA 减少约 128 个 A 矩阵，同时混合设置平均分数高出 MoLA 1.56%。
4. **系统性的多场景验证**：覆盖单任务、混合任务、跨域泛化（MMLU）、混合数学推理（GSM8K）及模型缩放（7B/13B/33B），全面证明方法的有效性与可扩展性。

## 方法详解
- **LoRA 低秩分解基础**：$\Delta W = BA$，其中 $A \in \mathbb{R}^{r \times d_{out}}$ 为上投影矩阵（初始化自随机高斯），$B \in \mathbb{R}^{d_{in} \times r}$ 为下投影矩阵（初始化为零）。
- **共享机制**：对 L 层 Transformer，每层分配 $N_l$ 个专家，所有专家共享同一个 $A_l$，仅保留各自的 $B_{i,l}$；每层仅需存储一个 A 矩阵，大幅削减参数。
- **Top-K 路由**：门控权重 $W_l \in \mathbb{R}^{d_{in} \times N_l}$，输出 softmax 后取 Top-K 得分并归一化：
  $$S_l^k(x) = \frac{\mathrm{TopK}(\mathrm{softmax}(W_l x), K)_k}{\sum_{k=1}^K \mathrm{TopK}(\mathrm{softmax}(W_l x), K)_k}$$
- **前向传播**（作用于 $W_q, W_v$）：
  $$h_l = W_0 x + \frac{\alpha}{r}\sum_{k=1}^K S_l^k(x) A_{k,l} B_{k,l} x$$
- **A 矩阵 Dropout**：每步训练以概率 $p$ 对 A 生成 Bernoulli 掩码 $\mathrm{Mask}$，得到 $\tilde{A}_l' = (\mathrm{Mask} \odot A_l)/(1-p)$，防止 A 过度拟合。
- **负载平衡损失**：$L_b = \sum_{k=1}^K c_k \cdot s^k$，其中 $c_k$ 为分配给专家 k 的 token 数，$s^k$ 为专家 k 的平均门控得分，抑制专家利用不均衡。

## 实验与结果
- **数据集**：6 个常识推理数据集 OBQA、CSQA、Race、MCTest、Arc-e、Arc-c（答案准确率评估）；外加 MMLU（跨域泛化）、GSM8K（数学推理混合）、OpenOrca（通用数据混合）。
- **基线**：FP-tuning、Prefix-tuning、LoRA、MoLoRA、SiRA、MoLA、MixLoRA。
- **最强结果（混合设置，6 数据集平均）**：
  - **MoSLD：71.56%**（单任务 70.20%）
  - 对比 MoLA 70.00%（+1.56%）、LoRA 69.15%（+2.41%）、FP-tuning 70.38%（+1.18%）
  - 消融：去掉 dropout 的 MoSL 混合设置 70.20%（-1.36%），验证 dropout 关键作用。
- **跨域泛化（MMLU）**：MoSLD 在单任务与混合设置下 consistently 优于 LoRA/MoLA/MoSL。
- **数学推理混合（GSM8K）**：plain LoRA 混合较单任务下降 5.61%，MoSLD 反而提升 0.67%，证明有效缓解数据冲突。
- **模型缩放（LLaMA2-7B/13B/33B）**：单→混合提升分别为 1.36%、1.61%、1.91%，体现良好 scaling 能力。
- **效率**：MoSLD 可训练参数 1.389B，MoLA 为 2.228B，MoSLD 减少约 38% 训练参数。

## 相关工作脉络
1. **MoE 架构**（Shazeer et al., 2017; Fedus et al., 2022）：从样本级到 token 级路由演进，本文将其引入 PEFT 场景。
2. **LoRA-MoE 结合先驱**（MoLoRA、SiRA、MixLoRA）：训练多个独立 LoRA 模块后路由融合，本文核心区别在于共享 A 矩阵而非完全独立专家。
3. **MoLA**（Gao et al., 2024）：按层分配不同数量 LoRA 专家，但专家间无参数共享；本文在其基础上引入共享机制并降低参数量。
4. **MixLoRA**（Li et al., 2024，同期工作）：在 FFN 层融合多 LoRA + 共享 FFN + self-attention 使用 plain LoRA；本文方案更轻量且直接在 attention 层实现共享。
5. **LoraHub**（Zhang et al., 2023; Huang et al., 2024）：线性组合已训练 LoRA；本文属于端到端多任务联合训练范式，非后验组合。

## 局限性与未来方向
1. 仅在 self-attention 层的 $W_q, W_v$ 上实验，尚未扩展到 FFN 层。
2. 未系统研究多任务混合数据比例对 MoSLD 的影响。
3. 通用特征与特定特征的解耦缺乏可视化分析，未来需进一步验证。
4. 仅验证了 6 个 QA 类数据集，在生成、对话、指令微调等更复杂任务上的表现待探索。

## 研究启发与可借鉴点
1. **"A/B 矩阵语义解耦"思路可迁移**：将 LoRA 上下投影矩阵分别视为通用/特定特征载体，可在其他 PEFT 方法（如 QLoRA、DoRA）中借鉴类似共享设计。
2. **Dropout 作用于共享层的有效性**：在 MoE 共享参数上施加 dropout 以缓解过拟合与优化失衡，为一类新的正则化策略提供了可行范式。
3. **按层差异化专家分配的超参设计**：高层分配更多专家、低层分配较少专家的策略具有直觉合理性，可复用于其他 MoE-PEFT 方法。
4. **混合 vs 单任务差距分析**：本文以混合-单任务分数差作为关键评估指标，揭示了方法在多任务鲁棒性上的优劣，值得纳入后续评测框架。
5. **数学推理等异构任务混合验证**：证明方法可处理低共性的任务混合，拓宽了 MoE-LoRA 的应用边界。

## 关键术语表
**MoSLD**：Mixture-of-Shared-LoRAs with Dropout，本文提出的参数高效多任务微调模型。
**LoRA (Low-Rank Adaptation)**：通过低秩矩阵分解 $BA$ 近似权重更新的高效微调方法。
**MoE (Mixture of Experts)**：由多个专家网络与路由机制组成的集成架构，仅激活部分专家处理输入。
**Top-K Routing**：路由函数选择得分最高的 K 个专家参与当前输入的计算。
**General-feature Matrix (A)**：LoRA 上投影矩阵，本文视其为跨任务共享的通用特征载体。
**Specific-feature Matrix (B)**：LoRA 下投影矩阵，保留为各专家专属的任务特定特征。
**Load Balancing Loss**：鼓励路由器均匀分配 token 至各专家的正则化损失。
**Out-of-domain Generalization**：模型在训练分布之外的领域数据上的泛化能力。

## 可复现要素
- **数据集**：OBQA、CSQA、Race、MCTest、Arc-e、Arc-c 均为公开数据集；GSM8K、OpenOrca、MMLU 亦公开。
- **代码/权重**：论文未明确声明代码开源，未提及权重发布（需关注作者后续更新）。
- **关键超参**：base model LLaMA2-7B；$r=8, \alpha=16$；各层专家数 $(8,6,4,2)$（第 1-8/9-16/17-24/25-32 层）；$K=2$；dropout ratio $p=0.1$；epoch=10；learning rate=3e-4；batch size=128；16×A100 40GB。
