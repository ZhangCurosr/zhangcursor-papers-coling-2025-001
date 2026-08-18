---
title: "MLLM-I2W-Harnessing-Multimodal-Large-Language-Model-for-Zero"
source: https://aclanthology.org/2025.coling-main.125.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:44:18"
field: "多模态检索与跨模态对齐"
keywords: ["Zero-Shot Composed Image Retrieval", "Multimodal Large Language Model", "Pseudo-word Mapping", "Modality Gap", "Uncertainty Modeling", "Mixture of Experts"]
innovations: ["基于MLLM的自适应主语词选择与文本增强编码模块", "不确定性建模缓解图文模态鸿沟", "原型学习与MOE映射网络联合提取深层融合特征"]
benchmarks: ["COCO", "CIRR", "Fashion-IQ"]
---

# 论文速读：MLLM-I2W: Harnessing Multimodal Large Language Model for Zero-Shot Composed Image Retrieval

## 一句话总结
本文提出 MLLM-I2W，一种基于多模态大语言模型（MLLM）的零样本组合图像检索（ZS-CIR）方法，通过将描述相关的图像信息自适应转换为伪词标记，在 COCO、CIRR 和 Fashion-IQ 三个基准上均优于现有最先进方法。

## 研究问题与动机
1. **ZS-CIR 核心挑战**：现有零样本方法需在图像-文本联合嵌入空间中学习将伪词映射到图像，但文本与图像特征常分布在不同的聚类中，存在模态鸿沟（modal gap）。
2. **监督方法成本高昂**：传统 CIR 需人工标注 triple（参考图、文本描述、目标图），标注成本高且泛化性差。
3. **现有 ZS-CIR 方法不足**：
   - Pic2Word 仅映射全局图像特征，缺乏对关键视觉区域的聚焦能力；
   - Context-I2W 使用 Spacy 词性标注器选择主语，过于简单且不能保证选中最相关的视觉主体；
   - 文本生成常忽略视觉上下文，引入无关噪声。
4. **模态权重分配不合理**： prior 工作通常对图像和文本特征等权融合，但实验表明不同数据集最优权重差异显著（COCO 最优为文本 0.6/图像 0.4，CIRR 为 0.8/0.2）。

## 核心贡献（创新点）
1. **MLLM 增强编码模块**：设计 MRT（主语词选择）和 MET（文本增强）两个组件，利用 MLLM 的图像理解能力替代固定词性标注规则，精准定位图像相关主语并生成丰富描述。**与 Context-I2W 的本质区别**在于不再依赖 spaCy 规则，而是通过 MLLM 的 latent world knowledge 自适应选择最优主语词。
2. **不确定性建模（Uncertainty Modeling）**：向图像和文本特征注入高斯噪声以增大特征空间中的分散度，促进不同模态特征的交叉重叠，缓解模态鸿沟。**与 prior work 的本质区别**在于将不确定性建模同时应用于视觉和文本双分支，并结合多头注意力增强特征表示。
3. **自适应模态融合模块**：通过注意力机制学习图像与文本的重要性权重 α（用 Sigmoid 计算），实现非均匀加权融合。**与 prior work 的本质区别**在于摒弃等权平均策略，根据查询贡献动态分配模态权重。
4. **原型学习与 MOE 映射网络**：设计 K=6 个可学习原型作为 query 提取深层融合特征；将映射网络从单层 MLP 替换为 4 专家 MOE 结构。**与 prior work 的本质区别**在于引入原型学习捕捉深层语义，并用 MOE 提升泛化能力而非单一投影层。

## 方法详解
**整体框架**（Figure 2）：MLLM-I2W 包含五个核心模块：(1) MLLM 增强编码 → (2) 不确定性建模 → (3) 自适应模态融合 → (4) 原型学习 → (5) MOE 映射网络。

**3.2 MLLM 增强编码**：
- **MRT（Main Regard Text）**：提示语 "Please replace the most relevant nouns in the <replace> text with learnable markers [replace]"，让 MLLM 基于图像内容选择最相关主语词替换为可学习标记 `[replace]`。
- **MET（Multi-modal Enhancement Text）**：提示语 "Please generate a brief description of the image"，生成更丰富的图像描述文本。
- 处理后的文本输入 CLIP frozen text encoder，提取 [CLS] token embedding $t \in \mathbb{R}^{d \times 1}$ 作为文本引导信号。

**3.3 不确定性建模**：
- 向原始特征分布的目标特征添加高斯噪声，计算噪声的均值和标准差。
- 引入多头注意力增强视觉和文本特征的表示能力，使不同模态特征在嵌入空间中更大程度重叠。

**3.4 自适应模态融合**：
- 对文本特征 $t$ 和视觉特征 $v$ 分别进行 self-attention：
  $$\mathbf{t_{att}} = softmax\left(\frac{QQ^T}{\sqrt{d}}\right)V, \quad \mathbf{v_{att}} = softmax\left(\frac{KK^T}{\sqrt{d}}\right)V$$
- 经过 FFN 残差连接得到 $\tilde{t}, \tilde{v}$，再经 MLP 得到 $t^*, v^*$。
- 计算文本重要性权重：$\alpha = Sigmoid(FC(t^*, v^*))$。
- 最终融合特征：$\mathbf{f} = \alpha \cdot t^* + (1-\alpha) \cdot v^*$。

**3.5 原型学习建模**：
- 初始化 $K$ 个原型 $P = [p_1, ..., p_K] \in \mathbb{R}^{d \times K}$，每个原型含不同语义信息。
- 原型作为 query，融合特征 $f$ 作为 key/value，通过转换器层提取深层融合特征：
  $$\tilde{f} = Concat(g_1(p_1, f), ..., g_K(p_K, f)), \quad g_i(p_i, f) = W_k(MHA(p_i, f, f))$$

**3.6 MOE 映射网络**：
- 4 个专家，每个专家为 3 层 MLP（隐藏维度 512）。
- 路由机制将输入分配到前两个专家，设置专家容量限制防止过载。
- 映射输出伪词标记 $S^*$，与文本 token 结合后输入 CLIP text encoder 生成最终表征。

**3.7 损失函数**：
- 对比损失：$\mathcal{L} = \mathcal{L}_{t2i}(\hat{s}, v) + \mathcal{L}_{i2t}(\hat{s}, v)$
- 温度参数 $\tau$ 控制 hard negative 惩罚强度。

**3.8 推理阶段**：
- 将伪词 $S^*$ 替换文本中的 `[replace]` 标记，与视觉特征组合后进行相似度计算。
- 不同任务设计不同 prompt：对象/场景组合 "a photo of [replace], [obj₁ tag]..."; 句子操作 "a photo of [replace], [sentence]."

## 实验与结果
**数据集**：
- **COCO**：4766 查询图 / 4766 候选图，评估对象组合任务。
- **CIRR**：4148 查询图 / 2315 候选图，评估对象/场景操作任务。
- **Fashion-IQ**：Dress(2017/3817)、Shirt(2038/6346)、Toptee(1961/5373)，评估属性操作任务。
- 训练数据：Conceptual Captions（300 万图像-文本对）。

**基线方法**：Image-only、Text-only、Image+Text、Pic2word、Context-I2W、CIReVL、PM、Searle-xl、PALAVRA、Combiner（ supervised）、TIRG、ARTEMIS、CIRPLANT、MAAF。

**主要结果**：
- **COCO（Table 3）**：MLLM-I2W R1=15.7, R5=31.2, R10=40.9，超越 SoTA Context-I2W（R10=40.9 vs 38.1，平均提升 2.57%）。
- **CIRR（Table 4）**：MLLM-I2W R1=28.3, R5=57.9, R10=70.2, R50=93.9，超越 SoTA Context-I2W（R10=68.5，平均提升 2.83%），优于部分监督方法 Combiner（R10=73.2）。
- **Fashion-IQ（Table 5）**：MLLM-I2W 在 Dress/Shirt/Toptee 三任务上 R10 平均 30.3，超越 SoTA CIReVL（平均 28.6，提升 1.61%）。
- **统计显著性**：所有数据集的 paired t-test p 值均 < 0.05，结果显著。

**消融实验（Table 6，CIRR）**：
- Baseline（R10=57.6）→ +MRT+MET（60.5）→ +Uncertainty（63.7）→ +Adaptive Fusion（64.5）→ +Prototype（65.2）→ +MOE（68.8）→ Full model（70.2），逐模块验证有效性。
- **权重敏感性实验（Table 1）**：COCO 最优文本权重 0.6，CIRR 最优文本权重 0.8，验证自适应融合的必要性。

## 相关工作脉络
1. **Pic2Word**（Saito et al., 2023）：仅用图像训练，通过 VLM 将图像映射为语言标记。本文与其区别在于：Pic2Word 缺乏查询图像与文本的交互，无法利用文本细粒度信息；本文通过 MLLM 实现图文交互增强。
2. **Context-I2W**（Tang et al., 2024）：使用词性标注器选择主语词并结合上下文约束。本文与其区别在于：Context-I2W 的主语选择规则固定（替换第一个名词），不能保证选中最相关主体；本文通过 MLLM 基于图像内容自适应选择。
3. **LinCIR**（Gu et al., 2024）：纯文本方法，使用自掩码投影（SMP）生成新文本。本文与其区别在于：LinCIR 完全忽略查询图像的视觉信息；本文显式建模图文融合。
4. **CIReVL**（Karthik et al., 2023）：使用 VLM 生成参考图像描述，再通过 LLM 重组装。本文与其区别在于：CIReVL 侧重文本生成；本文聚焦伪词映射网络设计。
5. **Combiner**（Baldrati et al., 2022）：监督方法，需 triple 标注。本文定位：作为零样本方法，无需标注数据即可达到或超越部分监督方法性能。
6. **CLIP-based CIR**：多篇工作基于 CLIP 进行图文检索。本文定位：在 CLIP 基础上引入 MLLM 增强编码和不确定性建模，解决模态鸿沟问题。

## 局限性与未来方向
1. **映射网络结构局限**：当前 MOE 映射网络仍基于全连接层，在共享嵌入空间中因特征分属不同聚类导致信息损失。未来需设计更有效的非线性优化映射问题。
2. **MLLM 依赖**：增强编码模块依赖预训练 MLLM 的能力，可能引入推理延迟和计算开销。
3. **通用性验证**：仅在三个 benchmark 上验证，未在其他领域（如 medical image retrieval）测试泛化性。

## 研究启发与可借鉴点
1. **MLLM 作为语义增强器**：可将 MLLM 的 subject word selection 机制迁移到其他 cross-modal 任务（如 VQA、图像描述生成），替代固定规则的 NLP 工具（如 spaCy）。
2. **不确定性建模用于模态对齐**：高斯噪声注入策略可推广至其他多模态对比学习场景，缓解模态鸿沟问题。
3. **自适应权重融合**：基于注意力机制的动态模态权重设计可复用于多模态融合任务（如多模态分类、检索）。
4. **原型学习提取深层特征**：将原型作为 query 的 Cross-Attention 结构可迁移至少样本学习、聚类分析等任务。
5. **MOE 结构替代 MLP 投影**：在需要提升泛化能力的映射模块中，可用 MOE 替换单一 MLP，值得在其他 zero-shot 跨模态任务中验证。

## 关键术语表
**Zero-Shot Composed Image Retrieval (ZS-CIR)**：无需任务特定标注数据，仅用图像或图像-文本对训练的组合图像检索方法，通过伪词映射实现图文联合查询。

**Pseudo-word**：在 CLIP 等预训练模型的联合嵌入空间中，通过映射网络生成的、与图像特征相近的文本标记，用于替代原始查询文本中的关键信息。

**Modality Gap**：图像和文本特征在多模态对比学习中分布于不同聚类区域的现象，导致跨模态相似度计算不准确。

**Uncertainty Modeling**：通过向特征添加高斯噪声增加特征分散度，促进不同模态特征在嵌入空间中重叠的技术手段。

**Adaptive Modal Fusion**：根据图像和文本特征对查询任务的贡献度动态分配权重，而非简单平均的模态融合策略。

**Mixture of Experts (MOE)**：由多个子专家网络和路由机制组成的模型结构，可根据输入动态选择激活的专家，提升模型容量和泛化能力。

**Conceptual Captions**：包含 300 万图像-文本对的开源数据集，常用于训练和评估多模态预训练模型。

## 可复现要素
- **训练数据集**：Conceptual Captions（300 万图像-文本对），公开可用。
- **评测数据集**：COCO、CIRR、Fashion-IQ，均已公开。
- **代码开源状态**：论文未提及代码开源计划。
- **模型权重**：使用预训练 CLIP ViT-L/14，公开可用；MLLM-I2W 自身权重论文未声明开源。
- **关键超参**：
  - 共享原型数 K = 6
  - MOE 专家数 = 4，每个专家为 3 层 MLP，隐藏维度 512
  - tanh-gating 初始化标量 = 0
  - 优化器：AdamW，lr = 5×10⁻⁶，weight decay = 0.1
  - Warmup：10,000 steps 线性预热
  - Batch size = 512
  - 硬件：4× Tesla A6000 (48G) GPU
