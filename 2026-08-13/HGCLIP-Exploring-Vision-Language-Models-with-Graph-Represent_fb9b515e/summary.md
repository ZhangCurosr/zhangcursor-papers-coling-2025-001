---
title: "HGCLIP-Exploring-Vision-Language-Models-with-Graph-Represent"
source: https://aclanthology.org/2025.coling-main.19.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:33:38"
---

# 论文速读：HGCLIP-Exploring-Vision-Language-Models-with-Graph-Represent

## 一句话总结
提出 HGCLIP 框架，将 CLIP 与图表示学习深度融合，通过多模态层级提示与原型驱动的图注意力机制，系统性挖掘类别树状结构信息，在 11 个视觉分层分类基准上刷新最优性能。

## 研究问题与动机
1. 传统分层图像分类方法仅依赖单模态图像特征，无法利用层级标签的天然文本描述，在复杂/新类别场景下泛化受限。
2. 已有 VLM+层级方法（如 CHiLS、Hierarchy-CLIP）均为免训练（training-free）策略，对细粒度数据易产生过度稀疏提示，且仅做表面语义扩充，未深度建模跨层级拓扑依赖。
3. 现有工作难以同时适配从粗粒度到细粒度的多级别预测，且在数据集未提供权威层级时缺乏自适应构建能力。

## 核心贡献（创新点）
1. **提出 HGCLIP 分层适配框架**：首次将图表示学习系统引入 VLM 的分层图像分类任务，实现文本与图像特征的双向层级增强，突破单一模态或浅层提示的局限。
2. **多模态深层提示机制**：在视觉与文本分支的深层 Transformer 块同时注入可学习 prompt token，使模型在更高语义层次捕获跨粒度上下文表征。
3. **原型驱动的图注意力特征融合**：利用训练集类原型构建视觉图并聚合层级消息，再通过注意力重加权将结构化类感知特征注入原始空间特征图，实现“类级结构引导+像素级细化”。
4. **强泛化与容错设计**：支持 ChatGPT 自动查询生成缺失层级，并在领域偏移（Domain Generalization）与子群偏移（Subpopulation Shift） setting 下保持显著增益。

## 方法详解
- **多模态层级提示（Multi-modal Hierarchical Prompt）**：在 CLIP 视觉与文本分支的前 9 个 Transformer 层中分别插入可学习 token $\mathbf{P}^V$ 与 $\mathbf{P}^T$（各 4 个），生成带层级上下文的文本特征 $\tilde{\mathbf{F}}_t$ 与空间/全局视觉特征 $\tilde{\mathbf{F}}_s, \tilde{\mathbf{f}}_v$。
- **文本图编码**：将类别层次树转化为图 $G=(V,E)$，节点初始特征为 $\tilde{\mathbf{f}}_n^t$，经多层 GAT 执行消息聚合，输出融合层级结构的文本特征 $\hat{\mathbf{F}}_t$。
- **图像图编码与注意力融合**：从训练集每类提取全局特征作为类别原型 $\mathbf{F}_v^*$，同样经图编码器得到 $\hat{\mathbf{F}}_{v*}$；计算空间特征图 $\hat{\mathbf{F}}_s$ 与原型间的相似度注意力图 $\psi=\hat{\mathbf{F}}_s\hat{\mathbf{F}}_{v*}^T$，经温度缩放后更新 $\hat{\mathbf{F}}_s=\text{SoftMax}(\psi/\alpha)\hat{\mathbf{F}}_{v*}$，池化得 $\hat{\mathbf{f}}_v$。
- **分类 logits 与损失**：最终输出为两部分加权叠加 $\text{logits}=\lambda_1\tilde{\mathbf{f}}_v\hat{\mathbf{F}}_t^T+\lambda_2\hat{\mathbf{f}}_v\hat{\mathbf{F}}_t^T$（$\lambda_1=1, \lambda_2=0.2$）；总损失为各层级交叉熵加权和 $\mathcal{L}=\sum_{i=1}^h w_i\cdot\mathcal{L}_{CE}(GT_i, logits_i)$。
- **层级构建**：若数据集无现成层次，按固定 prompt 调用 ChatGPT 生成 $h$ 层树状标签结构。

## 实验与结果
- **数据集**：11 个公开基准，涵盖通用（CIFAR-100, Caltech-101）、细粒度（FGVC-Aircraft, StanfordCars, Food-101, Fruits-360, OxfordPets-37）、场景（SUN397）、纹理（DTD）与卫星图像（EuroSAT），层级深度 2-4 层；额外在 ImageNet-V2/Sketch/A/R 评估领域泛化，在 BREEDS 子集评估子群偏移。
- **基线**：CLIP, CoOp, CoCoOp, VPT, MaPLe, KgCoOp, PromptSRC，以及特征适配方法（CLIP-Adapter, Tip-Adapter, CALIP）和纯视觉 FGVC 方法。
- **核心结果**：HGCLIP 在全部 11 个数据集的各层级均超越最强基线 MaPLe 与 PromptSRC，平均绝对提升 2.2% 与 5.7%。代表性成绩：CIFAR-100 $l_1$: 91.87%, $l_2$: 86.55%；FGVC-Aircraft $l_3$: 61.33%（较 MaPLe 提升 8.75%）；ETHEC $l_4$: 60.39%。
- **泛化验证**：在 4 个 ImageNet 变体上持续领先（如 ImageNet-R 达 76.07%）；在 BREEDS 子群偏移任务中保持稳定增益；消融表明 GAT（3 层）为最优图编码器，四模块全开效果最佳。

## 相关工作脉络
1. **CHiLS (Novack et al., 2023)**：将类别映射为子类别列表作为 prompt，细粒度数据下提示过于稀疏；本文用图消息传递实现深度结构化融合，避免浅层列表拼接导致的上下文淹没。
2. **Hierarchy-CLIP (Ge et al., 2023)**：基于 WordNet 做标签表面语义扩充，仅增强同层语义关联；本文引入图编码器对节点特征递归聚合，显式建模跨层拓扑依赖。
3. **MaPLe / PromptSRC**：参数量高效提示学习代表，但仅在文本或浅层多模态空间操作；本文在深层同时注入双模态提示，并额外引入图注意力重构空间特征。
4. **传统分层/细粒度分类 (PMG, FGN, GHORD, TFGIC 等)**：依赖单模态图像特征或定制专用分支；本文直接复用预训练 VLM 对齐能力，通过图提示实现多模态协同升级，避免从头训练。
5. **图表示学习 (GCN/GAT/GraphSAGE)**：经典方法用于社交/知识图谱节点分类；本文将其迁移至视觉-语言类别节点的嵌入空间，解决非欧几里得层级结构到连续特征表示的映射问题。

## 局限性与未来方向
1. 仅使用轻量级图编码器（GCN/GAT/GraphSAGE），未探索图 Transformer 或动态拓扑学习，表征容量存在上限。
2. 文本与视觉图
