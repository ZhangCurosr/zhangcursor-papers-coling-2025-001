---
title: "Noise-powered-Multi-modal-Knowledge-Graph-Representation-Fra"
source: https://aclanthology.org/2025.coling-main.11.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:45:13"
field: "多模态知识表示学习"
keywords: ["多模态知识图谱", "知识图谱补全", "实体对齐", "噪声注入", "Transformer", "多模态融合"]
innovations: ["首次提出GMNM主动引入高斯模态噪声提升MMKG鲁棒性", "首个同时支持MKGC与MMEA的统一轻量Transformer框架（13M参数）"]
benchmarks: ["DB15K", "MKG-W", "MKG-Y", "Multi-modal DBP15K", "MMEA-UMVM"]
---

# 论文速读：Noise-powered-Multi-modal-Knowledge-Graph-Representation-Fra

## 一句话总结
本文提出SNAG，一个统一的多模态知识图谱（MMKG）表示学习框架，首次同时支持MKGC（多模态知识图谱补全）和MMEA（多模态实体对齐）两大任务；其核心创新在于**主动引入高斯模态噪声**而非抗拒噪声，结合Transformer架构实现高效的实体级多模态交互融合。

## 研究问题与动机
- 现有MMKG方法大多从传统KG视角出发，未将MMKG表示学习与下游/预训练任务解耦，缺乏统一框架同时支撑MKGC与MMEA。
- 已有工作主要聚焦于"拒绝和对抗噪声"，但真实场景中多模态数据天然存在噪声与缺失，如何使模型适应噪声环境成为关键问题。
- 多模态预训练大模型需要将结构化知识有效嵌入，当前MMKG表示能力不足，易导致知识误解与多模态幻觉。
- 现有方法多依赖大规模PLM（如BERT、ViT），参数量庞大（>200M），训练成本高，缺乏轻量级可适配方案。

## 核心贡献（创新点）
1. **提出SNAG统一框架**：首个同时支持MKGC与MMEA任务的MMKG表示学习框架，通过任务特定目标灵活适配不同下游场景。
2. **Gauss Modality Noise Masking (GMNM)**：首次在MMKG领域主动引入模态级高斯噪声（按概率ρ注入比例ε的噪声），模拟真实世界噪声分布，提升模型鲁棒性。
3. **Entity-Level Modality Interaction模块**：基于Transformer的MHCA机制实现实例级模态置信度评估与动态加权融合，优于传统的早期/晚期融合策略。
4. **轻量级参数高效设计**：模型仅13M参数，远小于PLM-based方法（>200M），在DB15K等基准上实现SOTA且兼容多种基线增强。
5. **系统性消融验证**：揭示噪声参数（ρ=0.2, ε=0.7）的最优配置，证明模态掩码比Dropout更适合MMKG场景。

## 方法详解
**整体架构**：SNAG采用Transformer-based架构，包含模态嵌入、GMNM噪声注入、实体级模态交互三大核心组件，通过MKGC或MMEA特定目标进行训练。

**模态嵌入**：
- 图结构嵌入 $h_i^g$：MKGC使用FC层变换；MMEA使用GAT（2层2头）捕获邻域结构
- 关系/属性嵌入 $h_i^m$：MKGC使用RotatE（维度d/2）；MMEA使用BoW特征+FC层
- 视觉嵌入 $h_i^v$：冻结预训练编码器（VGG/BEiT/ResNet-152/CLIP），缺失图像用正态分布随机特征填补
- 表面嵌入 $h_i^s$：Sentence-BERT提取[CLS]特征

**GMNM（公式1-2）**：
每轮训练开始时，以概率ρ对模态特征 $x_i^m$ 注入噪声：
- 以概率(1-ρ)保持原特征
- 以概率ρ执行 $\hat{x}_i^m = (1-\epsilon)x_i^m + \epsilon \tilde{x}_i^m$，其中 $\tilde{x}_i^m = \varphi_m \odot z + \mu_m$（z~N(0,I)）
噪声统计量（均值μ_m、标准差φ_m）来自同模态非噪声数据的分布，确保噪声与现实缺失场景一致

**实体级模态交互**：
- MHCA：多头交叉模态注意力，计算模态间相关性 $\beta_{mj}$
- FFN：两层线性+ReLU进一步处理
- ILC：实例级置信度 $\tilde{w}^m$ 归一化后用于MMEA的训练目标加权

**任务特定目标**：
- MKGC：使用RotatE评分函数 $\mathcal{F}(e^h,r,e^t)=||\bar{h}^{head}\circ x^r - \bar{h}^{tail}||$，配合自对抗负采样损失 $\mathcal{L}_{kgc}$
- MMEA：结合GMI全局融合表示 + ECIA显式置信度增强 + IIR隐式跨模态细化，总损失 $\mathcal{L}_{ea}=\mathcal{L}_{GMI}+\mathcal{L}_{ECIA}+\mathcal{L}_{IIR}$

## 实验与结果
**数据集**：
- MKGC：DB15K（12.8K实体）、MKG-W（Wikidata子集）、MKG-Y（YAGO子集）
- MMEA：Multi-modal DBP15K（ZH-EN/JA-EN/FR-EN）、MMEA-UMVM（EN-FR/EN-DE/D-W-V1/D-W-V2），含不同图像缺失比例（R_img=0.4/0.6/1.0）

**主要结果**：
- **MKGC**：SNAG在DB15K上MRR=0.363（vs AdaMF 0.325）、H@1=0.274（vs 0.213）；MKG-W上MRR=0.373（vs 0.343）；MKG-Y上MRR=0.395（vs 0.381），**三项全部SOTA**
- **MMEA**：在7个标准数据集上全面SOTA，如DBP15K_ZH-EN的H@1=0.798（vs MEAformer 0.776）、MRR=0.858（vs 0.840）
- **GMNM增强效果**：作为插件可稳定提升EVA、MCLEA、MEAformer等基线性能
- **消融**：移除GMNM后DB15K MRR从0.363降至0.357；仅用图结构（无其他模态）MRR仅0.293

**关键超参**：ρ=0.2（噪声应用概率），ε=0.7（噪声比例），隐藏维度d=256（MKGC）/300（MMEA），batch=1024（MKGC）/3500（MMEA）

## 相关工作脉络
- **IKRL**（Xie et al., 2017）：早期多模态KG嵌入，使用TransE变体融合图像特征，但缺乏跨模态精细交互机制。
- **MoSE**（Zhao et al., 2022）：模态分割集成方法，训练独立模型后集成输出；SNAG通过单一Transformer统一融合，参数更轻量。
- **MKGformer**（Chen et al., 2022）：基于BERT的MLM方法，将KG三元组转为序列；参数量>200M，SNAG仅13M且不需预训练PLM。
- **EVA**（Liu et al., 2021）：MMEA基线，使用注意力调制模态重要性；SNAG在其基础上引入GMNM与实例级置信度显著超越。
- **MCLEA**（Lin et al., 2022）：对比学习MMEA方法；SNAG的GMI+ECIA+IIR三目标联合优化更全面。
- **AdaMF**（Zhang et al., 2024）：近期MKGC SOTA，利用不平衡模态信息；SNAG在其基础上通过噪声注入进一步提升鲁棒性。

## 局限性与未来方向
- 实验仅覆盖DB15K及MKG系列基准，未包含FB15K-237、WN18等传统KG数据集，难以验证纯结构任务的泛化能力。
- 排除MoSE等Ensemble方法和MKGformer等PLM-based方法因参数量差异（公平性考量），但未能直接对比其绝对性能上限。
- 仅验证MKGC与MMEA两个任务，未探索框架在知识预训练、RAG、多模态LLM微调等更广泛场景的潜力。
- 噪声参数（ρ, ε）依赖手动调优，缺乏自适应噪声强度学习机制。
- 视觉特征缺失时用正态分布随机特征替代，可能与真实缺失模式存在偏差。

## 研究启发与可借鉴点
- **噪声即信号**：GMNM的核心理念（主动引入噪声模拟真实缺失）可迁移至其他多模态表示学习任务，如多模态RAG、视觉语言模型的知识注入。
- **实例级置信度评估**：ILC模块输出的 $\tilde{w}^m$ 可作为模态质量指标，用于下游任务的动态权重分配或异常检测。
- **轻量统一框架设计**：13M参数实现双任务SOTA，为资源受限场景下的MMKG研究提供范式参考。
- **实验设计借鉴**：GMNM作为插件式模块可无缝集成至现有MMEA/MKGC方法（见Table 2），这种"即插即用"验证策略值得推广。
- **噪声参数敏感性分析**：Table 3展示了ρ和ε的网格搜索结果（ρ=0.1~0.5, ε=0.2~0.8），为后续噪声注入类方法提供了调参基线。

## 关键术语表
**MMKG（Multi-modal Knowledge Graph）**：包含文本、图像、音频等多种模态数据关联的知识图谱，实体可携带多模态属性值。
**MKGC（Multi-modal Knowledge Graph Completion）**：利用多模态特征补全KG中缺失的三元组，本文聚焦实体预测（Link Prediction）子任务。
**MMEA（Multi-modal Entity Alignment）**：在多模态KG间识别指向同一现实实体的实体对，是KG融合的核心任务。
**GMNM（Gauss Modality Noise Masking）**：在高斯分布约束下主动向模态特征注入噪声的掩码策略，模拟真实数据缺失场景。
**MHCA（Multi-Head Cross-Modal Attention）**：Transformer中的多头交叉注意力机制，用于评估不同模态间的语义相关性。
**GMI（Global Modality Integration）**：全局模态融合策略，通过可学习权重拼接多模态嵌入，保留原始特征区分度。
**ECIA（Explicit Confidence-augmented Intra-modal Alignment）**：在模态内对齐损失中引入实例级置信度加权，抑制低质量特征干扰。
**IIR（Implicit Inter-modal Refinement）**：利用Transformer输出隐藏状态直接进行跨模态对齐细化，自适应调整注意力分数。

## 可复现要素
- **数据集**：DB15K、MKG-W、MKG-Y、Multi-modal DBP15K、MMEA-UMVM（OpenEA基准），均为公开数据集
- **代码**：已开源，GitHub: https://github.com/zjukg/SNAG
- **权重**：论文未提及公开预训练权重
- **关键超参**：
  - GMNM: ρ=0.2, ε=0.7
  - MKGC: d=256, K=32（负样本数）, λ=12（margin）, τ_kgc=2.0, lr=1e-4, batch=1024
  - MMEA: d=300, N_h=1, batch=3500, 500 epochs + 500 iterative epochs
  - 视觉编码器：DB15K用VGG/ResNet-152，MKG-W/Y用BEiT，OpenEA用CLIP
  - 文本编码器：Sentence-BERT（MKGC），BoW（MMEA）
