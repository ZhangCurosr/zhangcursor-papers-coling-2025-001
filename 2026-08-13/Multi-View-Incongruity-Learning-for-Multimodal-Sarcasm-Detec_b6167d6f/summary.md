---
title: "Multi-View-Incongruity-Learning-for-Multimodal-Sarcasm-Detec"
source: https://aclanthology.org/2025.coling-main.119.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:44:55"
field: "多模态情感分析"
keywords: ["多模态讽刺检测", "虚假相关性", "不一致学习", "对比学习", "多视角融合", "数据增强"]
innovations: ["三视角不一致学习（token-patch/entity-object/sentiment）联合建模讽刺线索", "双向数据增强+对比学习缓解模态偏置和虚假相关性", "基于Beta分布置信度的多视角融合与SPMSD对抗测试集"]
benchmarks: ["MMSD", "SPMSD"]
---

# 论文速读：Multi-View-Incongruity-Learning-for-Multimodal-Sarcasm-Detection

## 一句话总结
本文针对多模态讽刺检测（MSD）中模型依赖虚假相关性的问题，提出了 **MICL** 方法：通过三种视角的不一致学习（token-patch、entity-object、sentiment）结合对比学习与双向数据增强，有效缓解模态偏置并提升模型泛化能力；同时构建了专门测试虚假相关性的 **SPMSD** 测试集。

## 研究问题与动机
- **现有 MSD 模型过度依赖文本编码器**：文本信息量远高于图像，导致改变图像不影响预测结果，暴露出严重的模态偏置（图1a）。
- **模型错误依赖非关键文本特征而非关键情感特征**：修改关键情感词对结果无影响，而修改不影响语义的次要描述反而导致误判（图1b）。
- **现有工作仅关注 token-patch 级不一致**，忽略了实体-对象关系和显式情感极性在讽刺识别中的重要作用，难以捕捉讽刺所需的完整不一致线索。
- **现有数据增强策略偏向文本侧**，忽视了图像增强的价值，可能进一步加剧模态偏置。

## 核心贡献（创新点）
1. **提出 MICL，首次系统性地从三个视角学习不一致信息**：token-patch（交叉注意力）、entity-object（语义图+GAT）、sentiment（SenticNet情感极性），与仅关注 token-patch 的工作（如 MIL-Net）本质不同。
2. **设计双向互补的数据增强策略**：文本侧用 ChatGPT 进行情感反转/同义改写，图像侧采用裁剪、互换、风格迁移、文本生成四路增强，突破了仅增强文本的局限（对比 DMSD-CL 仅做文本增强）。
3. **构建 SPMSD 测试集以量化虚假相关性影响**：1000 条精心构造的样本，涵盖多种虚假相关场景，用于系统评估模型的泛化鲁棒性，这是一个新的评测基准贡献。
4. **引入 Beta 分布置信度加权的多视角融合模块**：利用证据理论估计各视角可信度，以自注意力进行融合，解决不同样本中各视角贡献差异大的问题（参考 TMC 和 Subjective Logic）。

## 方法详解
**整体架构**分为三个模块：多模态特征编码 → 多视角不一致学习 → 多视角融合，并辅以双向数据增强和对比学习。

1. **多模态特征编码**
   - **文本编码**：使用 GLM-4V 对图像进行高质量 OCR 提取与翻译，得到 OCR-text $\mathcal{O}$，与原推文文本 $\mathcal{T}$ 拼接后输入 RoBERTa：
     $H^t = \text{Self\_Att}(\text{RoBERTa}(\mathcal{T} \oplus \mathcal{O}))$
   - **图像编码**：ViT-base-patch32 处理图像，输出 patch embeddings $H^v$。

2. **多视角不一致学习**
   - **Token-patch 不一致**：设计混合注意力机制（三种输入配置：混合文本+图像、文本为Query图像为KV、图像为Query文本为KV），避免传统单方向交叉注意力导致的模态偏置。
   - **Entity-object 不一致**：文本侧用 spaCy 构建实体依存图，图像侧用 Region Proposal 构建对象图（余弦相似度>0.6建边），通过 GAT 学习图表示后 pooling 得到 $f_e$。
   - **Sentiment 不一致**：通过 SenticNet 分别提取原文和 OCR-text 的情感极性，结合文本隐藏状态经 MLP 得到 $f_s$。

3. **多视角融合**
   - 基于证据理论（Subjective Logic / Beta 分布），将各视角分类器输出作为证据 $e^m$，计算置信度 $c^m = \frac{e_0^m + e_1^m}{(e_0^m+1)+(e_1^m+1)}$，再用自注意力网络融合三个视角的 embedding。

4. **数据增强与对比学习**
   - **文本增强**：① 反转情感词/实体生成反标签样本；② 同义改写保持原标签。由 GPT-3.5 生成，比例1:1。
   - **图像增强**：随机裁剪（3份）、同标签图像互换（3份）、风格迁移（2份）、GLM-4V 生成标题后用 Stable Diffusion 生成新图（2份）。
   - **对比损失**：同标签样本为正对，计算 t→v 和 v→t 双向对比损失 $\mathcal{L}_{cl}$。
   - **总损失**：$\mathcal{L} = \mathcal{L}_{ce} + \lambda \mathcal{L}_{cl}$，其中 $\lambda=1$，温度参数 $\tau=0.07$。

## 实验与结果
- **数据集**：公共 MMSD（训练集19816、验证集2410、测试集2409），以及本文构建的 SPMSD（1000样本，573讽刺+427非讽刺）。
- **评估指标**：Accuracy、Precision、Recall、F1（Binary-Average 和 Macro-Average）。
- **基线**：单模态（TextCNN、Bi-LSTM、BERT、RoBERTa、Image、ViT）和多模态（HFM、D&RNet、Res-BERT、Att-BERT、CMGCN、Multi-View CLIP、MIL-Net、DMSD-CL、G²SAM）。

**主要结果（MMSD）**：MICL 在全部指标上均超过最优基线 G²SAM*（已用 RoBERTa 重跑的公平对比）：
  - Accuracy：**92.08%** vs G²SAM* 91.07%（+1.01%）
  - Binary F1：**90.33%** vs 89.17%（+1.16%）
  - Macro F1：**91.81%** vs 90.78%（+1.03%）

**SPMSD 结果（虚假相关性泛化测试）**：MICL 准确率 **68.7%**，显著优于所有基线，较 DMSD-CL（60.6%）提升 **8.1%**，绝对提升幅度大于 MMSD 上的提升，证明了抗虚假相关性的有效性。消融实验中，移除 sentiment 模块后 SPMSD F1 从 73.94% 降至 65.79%，证明 sentiment 视角对缓解虚假相关性最关键。

## 相关工作脉络
1. **MIL-Net (Qiao et al., 2023)**：首个同时建模局部和全局不一致的 MSD 方法，但仅关注 token-patch 级别不一致，未涉及实体和情绪层面。
2. **DMSD-CL (Jia et al., 2024)**：利用反事实增强和对比学习改进 OOD 泛化，但仅做文本侧数据增强，且缺少实体-对象不一致建模。
3. **G²SAM (Wei et al., 2024)**：基于全局语义aware的图方法，当前 SOTA，但同样未系统解决虚假相关问题。
4. **InCrossMGs (Liang et al., 2021)**：早期基于图神经网络建模模态内/间交互的工作，为本研究引入图结构提供了思路来源。
5. **抵抗虚假相关性的方法论**：包括 DRO 框架（Wen & Li, 2021）、ERM+finetune（Kirichenko et al., 2023）、反事实VQA增强（Niu et al., 2021），本文从模型和数据进行双重干预，定位更为综合。

## 局限性与未来方向
- **SPMSD 上准确率仅 68.7%**，说明仍有较大改进空间，未能完全消除虚假相关性影响。
- 部分虚假相关性可能偶然提升训练集性能，文中未对此类"有益虚假相关性"进行深入讨论。
- **模型复杂度较高**：混合注意力、图注意力网络、GLM-4V OCR、Stable Diffusion 图像生成等多组件叠加，可能影响可扩展性和推理效率。
- 未来可探索在更大规模、更复杂分布的数据集上验证，以及简化模型架构以提升效率。

## 研究启发与可借鉴点
1. **三视角不一致学习框架**（token-patch + entity-object + sentiment）可迁移至其他多模态理解任务，如虚假新闻检测、跨模态情感分析，作为通用的不一致建模范式。
2. **双向数据增强策略**（文本+图像同时增强）的思路值得借鉴：现有工作普遍偏重文本增强，本文证明图像增强对缓解模态偏置同样关键。
3. **基于证据理论的 Beta 分布置信度融合**为多视角/多模态融合提供了一种可解释的权重分配方案，可复用于其他需要动态融合的任务。
4. **SPMSD 的构建思路**：针对特定模型缺陷（虚假相关性）设计对抗性测试集，是一种有效的模型评估和诊断方法，可推广到其他领域。
5. **OCR-text 作为补充模态**：利用 LVLM 改进 OCR 质量并翻译非英语文本，增强了图像中文字信息的利用，对社交媒体场景具有普适价值。

## 关键术语表
- **Multimodal Sarcasm Detection (MSD)**：利用文本和图像等多模态信息联合识别社交媒体中的讽刺内容。
- **Spurious Correlation**：模型在训练过程中学习的非泛化性统计关联（非因果特征），导致在分布外数据上表现下降。
- **Incongruity（不一致）**：讽刺的核心语言学特征，指文本与图像、字面意义与实际意图、情感表达与实际语境之间的不匹配。
- **MICL**：本文提出的 Multimodal Incongruities via Contrastive Learning 方法，通过多视角不一致学习和对比学习缓解虚假相关性。
- **SPMSD**：本文构建的包含潜在虚假相关模式的测试集（1000样本），用于系统性评估模型的泛化鲁棒性。
- **Beta分布置信度融合**：基于证据理论和主观逻辑（Subjective Logic），用 Beta 分布参数估计各视角预测的可信度并加权融合。
- **SenticNet**：一个用于情感分析和常识推理的知识库工具，本文用于提取文本的显式情感极性。
- **混合注意力机制**：同时以文本为Query和以图像为Query的双向交叉注意力，避免传统单方向注意力导致的模态偏置。

## 可复现要素
- **数据集**：MMSD（公开）；SPMSD（论文声明基于 MMSD 构建，详细信息见附录A，是否公开需进一步确认）。
- **代码/权重**：论文未明确声明开源，需查看论文或作者主页确认。
- **关键超参**：特征维度 $d=768$，温度参数 $\tau=0.07$，对比学习权重 $\lambda=1$，学习率 $1\times10^{-5}$，优化器 Adam，线性衰减至0。
- **模型配置**：文本编码器 RoBERTa-base，图像编码器 ViT-base-patch32-224，spacy en_core_web_trf 提取实体依存，图像区域余弦相似度阈值>0.6 建边。
- **训练环境**：单卡 Nvidia RTX 4090 (24G)。
- **关键工具**：GLM-4V（OCR/翻译）、GPT-3.5-turbo（文本增强）、Stable Diffusion（图像增强）。
