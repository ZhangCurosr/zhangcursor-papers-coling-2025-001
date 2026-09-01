---
title: "ProSparse-Introducing-and-Enhancing-Intrinsic-Activation-Spa"
source: https://aclanthology.org/2025.coling-main.180.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:39:01"
field: "大语言模型推理效率优化"
keywords: ["激活稀疏性", "ReLUfication", "大语言模型加速", "L1正则化", "模型压缩", "推理加速"]
innovations: ["提出渐进式稀疏正则化，通过正弦曲线平滑递增L1正则化因子以实现高稀疏度并保持性能", "构建三步ReLUFication框架（激活函数替换+渐进稀疏正则化+阈值偏移），实现89%+内生生激活稀疏度", "提供稀疏度与正则化因子的定量关系公式，实现稀疏度的可控调节"]
benchmarks: ["HumanEval", "MBPP", "MMLU", "GSM8K", "BBH", "PIQA", "HellaSwag", "WinoGrande", "BoolQ", "TyDi QA"]
---

# 论文速读：ProSparse-Introducing-and-Enhancing-Intrinsic-Activation-Spa

## 一句话总结
本文提出了 **ProSparse**，一种简单有效的 ReLUfication 方法，通过三步策略将使用 GELU/Swish 等非 ReLU 激活函数的大语言模型（LLM）转换为具有内生生激活稀疏性的 ReLU 模型，在保持与原始 Swish 版本相当性能的同时，实现了 LLaMA2-7B/13B 高达 89.32%/88.80% 的激活稀疏度，并获得了最高 4.52× 的实际推理加速。

## 研究问题与动机
- **核心问题**：如何为采用非 ReLU 激活函数（GELU/Swish）的现有 LLM 引入内生激活稀疏性，同时避免性能退化。
- **现有方法的不足**：
  - 早期直接替换为 ReLU 的方法（vanilla ReLU）受限于原始密集激活分布，无法达到满意的稀疏度。
  - 插入式/偏移 ReLU（shifted ReLU）通过剧烈转移激活分布来提升稀疏度，但存在性能退化风险，且论文复现结果远低于其声称的 ~95%。
  - 使用固定 $L_1$ 正则化因子容易引发激活分布的剧烈偏移，导致下游任务性能大幅下降。
  - GELU/Swish 本身缺乏内生稀疏性，现有非内生的阈值搜索方法会丢失神经元输出并损害性能。

## 核心贡献（创新点）
1. **提出 ProSparse 三步 ReLUfication 框架**：依次执行激活函数替换、渐进式稀疏正则化和激活阈值偏移，在保持性能的同时实现高内生稀疏度。
2. **渐进式稀疏正则化（Progressive Sparsity Regularization）**：首次将 $L_1$ 正则化因子在多个阶段沿平滑正弦曲线渐进增加（含 warmup 阶段），避免激活分布剧烈偏移。
3. **端到端的高稀疏度模型开源**：获得了 LLaMA2-7B/13B 和 MiniCPM-1B 的稀疏版，稀疏度达 87~89%，性能与原始 Swish 版本相当，是目前开源 LLaMA 中稀疏度最高的版本。
4. **系统级稀疏 GPU 算子与加速验证**：实现了利用输入侧和输出侧稀疏性的两个精确加速 GPU 算子，并在 PowerInfer 上验证了最高 4.52× 的实际推理加速。

## 方法详解
**三步骤整体流程：**

1. **激活函数替换**：将 FFN 中的激活函数 $\sigma$（GELU/Swish）替换为 $\text{ReLU}(x)=\max(x,0)$，并进行持续训练，使模型初步适应新的 ReLU 激活。
2. **渐进式稀疏正则化**：对 FFN 中间输出 $\mathbf{x}_1$ 施加 $L_1$ 正则化损失 $\mathcal{L}_{reg}^i(\lambda)=\lambda\|\mathbf{x}_1\|_1$。关键设计是因子 $\lambda$ 不在全程固定，而是在多个阶段中**沿平滑正弦曲线渐进增加**：
   - **Warmup 阶段**：设置较小的常数 $\lambda$，为分布适应留出时间。
   - **增量阶段**：每个阶段的 $\lambda$ 沿正弦曲线从谷底递增到峰值，利用正弦在极值点处导数接近零的特性，使因子在转折点附近平缓变化，避免激活分布的剧烈突变。整体优化目标为 $\mathcal{L}_{lm}+\mathcal{L}_{reg}(\lambda)$。
3. **激活阈值偏移**：将 ReLU 替换为 FATReLU（$\sigma(x)=x$ when $x\geq t$，否则为 0），通过偏移阈值 $t>0$ 剪除低影响神经元以提升稀疏度，论文选择 $t=0.01$ 以平衡稀疏度与性能。

**可预测稀疏度的定量关系**：最终激活稀疏度主要取决于最后一阶段正则化因子 $\lambda_S$，二者呈负指数关系（对 ProSparse-7B 近似为 $100-\exp(-1.76\cdot\lambda_S^{0.30}+3.49)$），使得稀疏度可控。

## 实验与结果
- **模型与数据集**：ProSparse 应用于 LLaMA2-7B、LLaMA2-13B 和 MiniCPM-1B；训练数据包含语言建模数据集（StarCoder、Wikipedia、Pile 等）和指令微调数据集（UltraChat、Flan 等）。
- **评估基准**：Code Generation（HumanEval、MBPP）、Commonsense Reasoning（PIQA、SIQA、HellaSwag、WinoGrande、COPA）、Reading Comprehension（BoolQ、LAMBADA、TyDi QA）、以及其他（GSM8K、MMLU、BBH、AGI-Eval）。
- **主要结果**：

| 模型 | 稀疏度 | 平均性能 | 对比基线（稀疏度） |
|------|--------|----------|-------------------|
| ProSparse-7B | **89.32%** | 38.46% | ReluLLaMA-7B（66.98%，37.62%）|
| ProSparse-13B | **88.80%** | 44.90% | ReluLLaMA-13B（71.56%，42.74%）|
| ProSparse-1B | **87.89%** | 44.72% | 原始 MiniCPM-1B（44.44%）|

- **加速效果**：
  - 使用 PowerInfer 近似加速算法：**最高 4.52×** 加速（ProSparse-13B*，相对于 Dense 基准）。
  - 使用精确 GPU 算子：Step (2) 最高 **2.44×** 加速，Step (3) 最高 **1.70×** 加速。
  - 激活稀疏度越高，激活召回率、预测稀疏度和推理速度均显著提升。
- **最强结果**：LLaMA2-13B 在 PowerInfer 上获得 **4.52×** 加速；ProSparse-7B 达到 **89.32%** 稀疏度，为开源 LLaMA 版本中最高的内生生稀疏度。

## 相关工作脉络
1. **ReluLLaMA / ReluFalcon**（Zhang et al., 2024）：仅通过激活函数替换实现 ReLUfication，稀疏度有限（~67-72%），未处理原始密集分布的限制。ProSparse 在此基础上引入正则化突破此限制。
2. **ReLU Strikes Back**（Mirzadeh et al., 2023）：插入并偏移 ReLU 以激进地转移激活分布，声称 ~95% 稀疏度但论文复现失败；ProSparse 以更平滑的渐进策略达到可比稀疏度且性能更优。
3. **FATReLU**（Kurtz et al., 2020）：阈值偏移激活函数，本文将其作为第三步用于进一步剪除低影响神经元。
4. **PowerInfer**（Song et al., 2023）：基于激活预测器的近似稀疏推理加速框架，本文将其作为加速评估基准，证明更高稀疏度带来更好的预测可塑性和加速效果。
5. **Deja Vu**（Liu et al., 2023）：利用上下文稀疏性进行 LLM 推理加速，与 ProSparse 正交，均可受益于内生激活稀疏性。
6. **IncReg**（Wang et al., 2019）：引入增量正则化因子用于卷积网络剪枝，但与 ProSparse 在场景（Transformer LLM vs 卷积网络）、目标和策略上完全不同。

## 局限性与未来方向
- **大规模模型验证缺失**：论文仅在 1B~13B 尺度上验证，70B 及以上规模模型的适用性有待探索。
- **仅优化 FFN 层**：仅针对 FFN 的 step (2)(3) 进行稀疏加速，注意力层及 FFN step (1)（矩阵乘 $\mathbf{x}\mathbf{W}_s^T$）的稀疏化尚未涉及，大量 LLM 计算未被优化。
- **训练成本增加**：相比无正则化的 ReLUfication，需要额外约 54.53B tokens 的训练（占 LLaMA2 预训练 token 的约 2.73%），成本虽可接受但对极端资源受限场景仍为负担。
- **未来方向**：探索注意力层的稀疏化方法、优化 FFN step (1) 的稀疏计算（如剪枝、低秩分解）、以及将 ProSparse 扩展到更大规模模型。

## 研究启发与可借鉴点
1. **渐进式 $L_1$ 正则化策略**：将固定因子改为沿平滑曲线（正弦）递增，是平衡稀疏度与性能的有效通用思路，可迁移到其他激活稀疏化或模型压缩场景。
2. **稀疏度的可控性定量公式**：发现最终稀疏度与最终阶段正则化因子之间的负指数关系，提供了以目标稀疏度反推超参数的方法，减少实验搜索成本。
3. **稀疏 GPU 算子的工程实现范式**：输入侧稀疏算子（稀疏矩阵-向量乘）和输出侧稀疏算子（融合 ReLU+稀疏 matmul+逐元素乘）的设计思路，对后续稀疏推理引擎开发有参考价值。
4. **指令微调阶段正则化因子的调整策略**：SFT 阶段需使用比预训练阶段更小的正则化因子以适应新数据分布，这一经验对稀疏模型的下游微调具有指导意义。
5. **激活稀疏度的数据集差异性分析**：发现格式化的指令微调数据上稀疏度更高（如 QA 数据集 > 对话数据 > 纯文本），这对实际部署场景中稀疏加速收益的预估提供了依据。

## 关键术语表
**ReLUfication**：将非 ReLU 激活的 LLM 转换为 ReLU 激活 LLM 的过程，以引入内生激活稀疏性。
**Intrinsic Activation Sparsity（内生生激活稀疏性）**：由激活函数本身（如 ReLU）自然产生的零元素，不依赖外部阈值搜索。
**Progressive Sparsity Regularization（渐进式稀疏正则化）**：将 $L_1$ 正则化因子在多个训练阶段沿平滑曲线逐步增加的策略，以避免激活分布的剧烈偏移。
**FATReLU**：带正阈值偏移的 ReLU（$\max(x-t,0)$ with $t>0$），可进一步剪除低影响激活元素。
**Activation Recall（激活召回率）**：预测器正确预测的真实激活元素占所有真实激活元素的比例，衡量预测准确性。
**Predicted Sparsity（预测稀疏度）**：预测器标记为激活值为零的元素占总元素的比例，反映可跳过计算的规模。
**Approximate Acceleration Algorithm（近似加速算法）**：基于激活预测器进行硬件资源调度和计算跳过的方法，可能存在推理误差。
**Accurate Acceleration Algorithm（精确加速算法）**：基于稀疏 GPU 算子直接跳过零元素的计算，保证推理结果精确无误。

## 可复现要素
- **数据集**：训练数据包含 StarCoder、Wikipedia、Pile 等语言建模数据和 UltraChat、Flan、Super-Natural Instructions 等指令微调数据；评估基准包括 HumanEval、MBPP、PIQA、SIQA、GSM8K、MMLU 等（论文 Appendix E 详述）。
- **代码/权重**：论文声明稀疏版 LLaMA2 模型"available"（标注 †），代码/权重开源声明在论文中标注了可用链接。
- **关键超参**：
  - LLaMA2-7B：峰值学习率 3e-5，AdamW，8×A100 80GB，约 10 天，总训练 token 34.60B（最终 $\lambda_S=0.2$）。
  - LLaMA2-13B：峰值学习率 5e-5，32×A100 80GB，约 20-30 天，总训练 token ~134B（最终 $\lambda_S=0.02$）。
  - MiniCPM-1B：沿用 MiniCPM-1B 超参，另含 decay 和 SFT 阶段（SFT 阶段 $\lambda=1\times10^{-2}$）。
  - 阈值 $t=0.01$，上下文长度 4096。
