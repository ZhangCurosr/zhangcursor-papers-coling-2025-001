---
title: "Enhancing-One-Shot-Pruned-Pre-trained-Language-Models-throug"
source: https://aclanthology.org/2025.coling-main.117.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:30:38"
field: "语言模型压缩与高效推理"
keywords: ["模型剪枝", "预训练语言模型", "一次性剪枝", "稀疏正则化", "权重分布优化", "后训练压缩"]
innovations: ["提出三阶段 Sparse-Dense-Sparse 后训练剪枝框架", "设计融合残差稀疏先验、数据正则化与权重正则化的多维稀疏正则化策略", "引入基于 absmin 的软稀疏掩码进行二次剪枝权重调整"]
benchmarks: ["Raw-Wikitext2 Perplexity", "Zero-shot COPA/Lambada/OpenBookQA/PIQA/RTE/StoryCloze/Winogrande"]
---

# 论文速读：Enhancing-One-Shot-Pruned-Pre-trained-Language-Models-through-Sparse-Dense-Sparse-Mechanism

## 一句话总结
本文提出 SDS（Sparse-Dense-Sparse）三阶段剪枝框架，通过对权重分布进行稀疏正则化优化，使一次性剪枝后的预训练语言模型在相同稀疏配置下显著优于 SparseGPT 和 Wanda，在 LLaMA-3-8B 的 2:4 稀疏度下相比 Wanda 将 Raw-Wikitext2 困惑度降低 5.16、多任务零样本准确率平均提升 3.86%。

## 研究问题与动机
1. **PLMs 规模带来高成本**：预训练语言模型参数规模庞大，推理延迟与存储开销显著，剪枝是有效压缩手段，但传统一次性剪枝在紧凑模型（如 OPT-350M）和高稀疏度（60%-80%）下性能下降严重。
2. **预训练缺乏剪枝感知**：PLMs 在预训练阶段未引入剪枝相关正则化，导致直接剪枝时权重分布不"剪枝友好"，表现为剪枝后产生过于尖锐的双峰分布。
3. **知识损失可恢复**：实验发现仅需少量校准样本（如 C4 中 128 段文本）即可将剪枝模型的 perplexity 恢复至接近原始密集模型水平，说明剪枝损失可在有限资源下重建。
4. **生物神经元启发**：大脑神经元呈现"稀疏→密集→稀疏"的连接演化模式，本文借鉴此思路设计 Sparse-Dense-Sparse 框架，寻找更优的权重分布与剪枝策略。

## 核心贡献（创新点）
1. **提出 SDS 三阶段框架**：以初始剪枝（SparseGPT/Wanda）、密集重构（带稀疏正则化）、二次剪枝与权重调整为核心流程，从权重分布优化视角系统性提升一次性剪枝效果；本质区别于直接剪枝，强调通过重构步骤主动塑造"剪枝友好"分布。
2. **设计多维稀疏正则化策略**：融合残差稀疏特征先验、基于数据的正则化（使用高 loss 稀疏样本激活）与基于权重的 L1/L2 正则化，三者共同作用使重构后的密集模型呈现更明显的三峰分布；区别于 Han et al. 的 DSD（训练阶段使用），本文针对后训练一次性剪枝设计。
3. **引入软稀疏掩码权重调整**：二次剪枝后通过 absmin 动态选择的软掩码进行权重微调，目标对齐原始密集模型而非中间密集模型，避免前期剪枝信息损失；相比直接对稀疏模型做一阶优化，显著缓解局部最优。
4. **验证 SDS 的强泛化性**：不仅适用于 SparseGPT/Wanda，还可与 ADMM、PrunerZero、DS∅T 等剪枝方法结合，在 LLaMA-3-8B 上均取得稳定提升。

## 方法详解
**整体流程（三阶段）：**
- **第一阶段：初始剪枝（Initial Pruning）**
  以 SparseGPT 或 Wanda 为基底，对 PLM 的各全连接层（self-attn 与 FFN 模块）执行一次性剪枝，得到稀疏权重 $\mathbf{W}^{\text{sparse}}$。

- **第二阶段：密集重构（Re-dense Weight Reconstruction）**
  采用层级知识蒸馏方式，以 128 个样本输入，逐层对齐密集模型输出。损失函数为：
  $$L_{\text{base}} = \|\mathbf{W}^{\text{dense}}\mathbf{X} - \mathbf{W}^{\text{sparse}}\mathbf{X}\|_2^2, \quad L_{\text{reg}} = \lambda_1\|\mathbf{W}\|_1 + \lambda_2\|\mathbf{W}\|_2^2$$
  其中 $\lambda_1:\lambda_2=0.1$，迭代 200 轮，学习率 0.1。稀疏正则化促使重构后权重出现更强的三峰分布（中心峰值更尖锐），增强剪枝友好性。

- **第三阶段：二次剪枝与软掩码调整（Second Pruning）**
  对重构后的密集模型再次执行同类型剪枝，随后利用软稀疏掩码 $\mathbf{M}^{\text{soft}}$（按 $|\mathbf{W}^{\text{sparse-2nd}}|$ 的 absmin 动态选择）进行权重微调：
  $$\widehat{\mathbf{W}}^{\text{SDS}} = \mathbf{M}^{\text{soft}} \odot \arg\min_{\mathbf{W}} \|\mathbf{W}^{\text{dense}}\mathbf{X} - \mathbf{W}\mathbf{X}\|_2^2$$
  当第二次一次性剪枝已超越初始剪枝时，触发 early-exit 跳过此步。

**数据选择策略**：实验表明使用稀疏模型激活（SD-data）作为重构输入，比密集数据（DD-data）或蒸馏感知数据（KD-data）更适合 SDS 流程，因其兼具正则化效应与适中学习难度。

## 实验与结果
- **模型与数据集**：OPT（125M–13B）、LLaMA-1/2/3（7B–70B）；校准集为 C4 中 128 段 2048 token；评测集为 Raw-Wikitext2（perplexity）及 COPA、Lambada、OpenBookQA、PIQA、RTE、StoryCloze、Winogrande 七项零样本任务。
- **50% 稀疏度**：SDS-SparseGPT 在 OPT-350M 上 PPL 降至 29.36（SparseGPT 为 31.58）；LLaMA-3-8B 上 SDS-Wanda 平均准确率 69.60%（Wanda 为 68.06%）。
- **2:4 稀疏度**：SDS-SparseGPT 在 LLaMA-3-8B 上 PPL 降至 15.43（SparseGPT 为 17.88），平均准确率 63.19%（+0.19%）；SDS-Wanda 相比 Wanda 将 PPL 从 24.29 降至 19.13，平均准确率从 57.17% 提升至 61.03%（**+3.86%**）。
- **高稀疏度（60%-80%）**：在 70% 稀疏度下，SDS-Wanda 于 LLaMA-3-70B 上将 PPL 从 25.35 大幅降至 16.96；在 80% 稀疏度下，SDS-SparseGPT 于 LLaMA-2-70B 上准确率达到 60.40%，而 Wanda 已严重退化。
- **加速效果**：在 AMD R7 Pro CPU 上，50% 稀疏模型实现 1.19×–1.87× 推理加速；在 NVIDIA A100 GPU 上 2:4 稀疏 kernel 加速达 1.46×–2.12×。
- **消融实验**：SDS 全流程效果最优；残差稀疏特征与数据正则化起主导作用，权重正则化为辅助；使用统一校准集即可，无需额外样本。

## 相关工作脉络
1. **SparseGPT (Frantar & Alistarh, 2023)**：基于二阶信息的列级剪枝方法，是本文初始剪枝的基线之一；本文以 SDS 对其做后处理优化。
2. **Wanda (Sun et al., 2023)**：结合权重与激活幅度的无更新剪枝策略；本文同样以其为基线，证明 SDS 的通用增强能力。
3. **Dense-Sparse-Dense (Han et al., 2017)**：训练阶段的三阶段稀疏化方法；本文受其启发但应用于后训练一次性剪枝，无需重新预训练。
4. **SPDF (Thangarasa et al., 2023)**：稀疏预训练+密集微调范式；本文聚焦于后训练压缩，不涉及预训练阶段改动。
5. **Prune-and-Tune (Syed et al., 2023)**：在剪枝后加入少量迭代微调；本文 SDS 通过分布优化达到类似效果但保持零-shot 免训练特性。
6. **OWL (Yin et al., 2024)、DS∅T (Zhang et al., 2024)、PrunerZero (Dong et al., 2024)**：后续提出的剪枝或微调方法；本文验证 SDS 同样可与之结合进一步提升。

## 局限性与未来方向
1. **额外计算开销**：SDS 优化需 200 轮迭代与额外前向传播，尽管相比完整预训练代价较小，但仍高于单次一次性剪枝。
2. **未深入探索与其他压缩方法的结合**：作者指出未来将研究 SDS 与量化（Quantization）等技术的协同优化。
3. **校准数据依赖**：虽仅用 128 样本，但数据质量仍影响效果（Appendix A.5 显示不同校准集略有差异）；如何设计更鲁棒的校准策略有待研究。
4. **早期退出策略的普适性**：仅在小模型（<3B）或特定稀疏度下触发 early-exit，更大规模模型仍需完整流程。

## 研究启发与可借鉴点
1. **"中间重构"思想可迁移**：SDS 通过密集重构重塑权重分布的思路可推广至后训练量化（PTQ）等其他压缩场景，形成类似的优化框架。
2. **多维稀疏正则化组合**：残差稀疏先验+数据正则+权重正则的组合策略具有通用性，可设计针对不同层/模块的差异化正则方案。
3. **软掩码动态调整机制**：基于 absmin 的软稀疏掩码在后续微调中保护异常值（outliers）的思路，对量化中的 outlier 平滑有参考价值。
4. **多 GPU 并行优化设计**：论文采用层间并行、层内串行的优化策略，将 7B 模型 SDS 优化从 12.9 小时降至 1.7 小时，该工程经验值得借鉴。
5. **与下游团队方向的结合机会**：若团队关注低资源场景下的模型压缩或边缘部署，SDS 的免重训特性与硬件加速效果可直接应用；同时可探索 SDS 与动态稀疏（Deja Vu 类）方法的结合。

## 关键术语表
**SDS（Sparse-Dense-Sparse）**：一种三阶段后训练剪枝框架，通过稀疏化→密集重构→再稀疏化的流程优化预训练语言模型的权重分布。
**One-shot pruning**：一次性剪枝，指无需重新训练即可对预训练模型进行压缩的方法，如 SparseGPT、Wanda。
**Pruning-friendly weight distribution**：剪枝友好型权重分布，指具有更尖锐零值峰值、更均匀分布特征的权重结构，便于后续识别和剔除不重要连接。
**Sparse regularization**：稀疏正则化，通过 L1/L2 等惩罚项促使权重向稀疏化方向演化，增强模型对剪枝的适应能力。
**Soft sparse mask**：软稀疏掩码，根据当前权重幅值的 absmin 动态选择的掩码，用于限制微调仅在保留的稀疏位置进行。
**Calibration samples**：校准样本，用于后训练压缩（剪枝/量化）时估计激活分布的小规模无标签数据。
**Absmin**：绝对值最小准则，用于选择应被保留的权重位置，作为动态掩码的筛选依据。
**Early-exit**：早期退出机制，当第二次一次性剪枝已超越初始结果时跳过权重调整步骤以节省计算。

## 可复现要素
- **数据集**：C4（校准，公开）、Raw-Wikitext2（评测，公开）、OPT/LLaMA 系列（公开权重）；COPA、Lambada、OpenBookQA、PIQA、RTE、StoryCloze、Winogrande（公开零样本评测基准）。
- **代码/权重**：论文声明基于 PyTorch 和 HuggingFace Transformers/Datasets 实现；代码开源情况论文未明确声明（需查看论文附属仓库）。
- **关键超参**：校准样本数 128 段（每段 2048 token）；重构阶段迭代 200 轮，学习率 0.1；L1:L2 正则化比例 0.1；输出层并行优化策略。
- **硬件**：NVIDIA V100 32GB（单卡串行优化）、多卡并行优化；AMD R7 Pro 5850U CPU 推理加速测试；NVIDIA A100 PCIe 40GB GPU kernel 加速测试。
