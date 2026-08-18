---
title: "Integrating-Group-based-Preferences-from-Coarse-to-Fine-for"
source: https://aclanthology.org/2025.coling-main.153.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:41:26"
---

# 论文速读：Integrating-Group-based-Preferences-from-Coarse-to-Fine-for

## 一句话总结
本文针对交叉领域推荐（CDR）中的冷启动用户问题，提出 GRAPECDR 模型，将用户偏好系统性地拆分为类别、品牌、方面三个从粗到细的粒度，利用外部记忆网络捕获相似重叠用户的群体偏好信号，并通过元网络生成个性化跨域转移函数，最终融合多维度偏好评分实现高精度冷启动推荐。

## 研究问题与动机
- 现有 CDR 方法通常将用户偏好建模为单一的统一向量，忽略了真实决策过程中偏好存在类别、品牌、方面三个不同粒度的层次结构，且各粒度对用户决策的影响权重并不相同。
- 传统方法多依赖用户自身历史购买/评论记录直接生成转移函数，忽视了相似用户之间存在的偏好共性（group-based similarities），导致源域表征在跨域迁移时信息增益有限。
- 已有个性化转移工作（如 PTUPCDR）虽为每个用户生成独立迁移参数，但未引入群体先验；而共享转移函数方法（如 CMF、EMCDR）又无法适配用户间的个性化差异，二者均难以兼顾泛化性与个性化。
- 当前基于方面（aspect）的跨域方法（如 CatN）仅关注文本级细粒度，缺乏与类别、品牌等语义层级的联合建模，整体表征仍偏单向度。

## 核心贡献（创新点）
- 提出从粗到细的多粒度偏好整合框架 GRAPECDR，首次在 CDR 冷启动场景中联合显式建模类别、品牌、方面三层偏好。与仅使用统一用户向量的方法相比，本文实现了不同语义层级偏好独立生成、独立迁移与自适应融合。
- 引入外部记忆网络（EMN）在各粒度上构建基于群体的偏好表征，通过 erase-add 机制动态学习与聚合重叠用户的典型特征。与直接复用个体历史表征的方法相比，该设计将相似用户的群体信号转化为可微的记忆先验。
- 设计基于元网络（Meta-Network）的个性化跨域转移函数，以群体偏好表征为输入生成每个用户专属的迁移权重矩阵。与全局固定桥接函数相比，实现了真正的参数级个性化跨域映射。
- 提出多维度偏好评分融合机制，利用用户表征学习类别/品牌/方面评分的自适应权重。与等权拼接或固定权重融合相比，更贴合实际用户在不同决策场景下的偏好侧重。

## 方法详解
- **多粒度内在表征生成**：在源域 $\mathcal{D}^s$ 中，分别统计用户历史购买 item 的类别频次 $\mathbf{n}_{e_u^s}$、品牌频次 $\mathbf{n}_{i_u^s}$ 以及评论中方面词频次 $\mathbf{n}_{v_u^s}$，通过 softmax 加权嵌入矩阵得到三类内在表征：$\mathbf{e}_u^s = \text{softmax}(\mathbf{n}_{e_u^s})\mathbf{E}^s$、$\mathbf{i}_u^s = \text{softmax}(\mathbf{n}_{i_u^s})\mathbf{I}^s$、$\mathbf{v}_u^s = \text{softmax}(\mathbf{n}_{v_u^s})\mathbf{V}^s$。
- **基于群体的表征生成（EMN）**：为每个粒度维护一个记忆矩阵 $\mathbf{M}^e, \mathbf{M}^i, \mathbf{M}^v$，存储重叠用户的典型特征。通过软注意力计算用户与各记忆槽的相似度权重 $z_{u,k}$，得到群体增强表征 $\mathbf{e}_{u^*}^s = \sum_k z_{u,k} \mathbf{m}_k^e$。训练时采用 NTM 风格的 erase-add 更新：先计算擦除向量 $\mathbf{erase}_u = \sigma(\mathbf{W}_{erase}\mathbf{e}_u^s + \mathbf{b}_{erase})$，执行 $\mathbf{m}_u^e \leftarrow \mathbf{m}_u^e \odot (1-\mathbf{erase}_u)$，再计算添加向量 $\mathbf{add}_u = \tanh(\mathbf{W}_{add}\mathbf{e}_u^s + \mathbf{b}_{add})$ 并执行 $\mathbf{m}_u^e \leftarrow \mathbf{m}_u^e + \mathbf{add}_u$，实现典型特征的遗忘与强化。
- **元网络个性化转移**：对每个粒度，先将内在表征投影回嵌入空间 $\mathbf{e}_{\hat{u}}^s = \text{softmax}(\mathbf{e}_u^s (\mathbf{E}^s)^T)\mathbf{E}^s$，再输入两层全连接元网络 $g(\cdot;\theta)$ 生成用户专属的转移权重矩阵 $\mathbf{W}_u^e$。群体表征经该用户专属转移函数映射至目标域：$\mathbf{e}_u^t = f_u^e(\mathbf{e}_{u^*}^s) = \mathbf{e}_{u^*}^s \mathbf{W}_u^e$，品牌与方面同理得到 $\mathbf{i}_u^t, \mathbf{v}_u^t$。
- **多维度评分预测与融合**：
  - **类别评分**：提取 item 类别集合 $\mathbf{e}_i^t$，通过注意力 $\phi_{u,i,k}$ 聚合 item 类别嵌入，再与用户目标域类别表征经两层 MLP 得 $r_{u,i}^e$。
  - **品牌评分**：直接拼接用户品牌表征与 item 品牌嵌入，经 MLP 得 $r_{u,i}^i$。
  - **方面评分**：从 item 评论中抽取方面-极性对 $(\mathbf{v}_{i,k}^t, s_{i,k}^t)$，计算用户关注度 $\sigma_{u,i,k}$ 后与极性值（$(-1,1)$）加权求和得 $r_{u,i}^v$。
  - **最终融合**：学习用户自适应权重 $\mathbf{w}_f = \text{sigmoid}(\mathbf{w}_w[\mathbf{e}_u^t \oplus \mathbf{i}_u^t \oplus \mathbf{v}_u^t] + \mathbf{b}_w)$，对三类评分加权融合：$\hat{r}_{u,i} = \mathbf{w}_f [r_{u,i}^e \oplus r_{u,i}^i \oplus r_{u,i}^v]$。
- **优化策略**：仅在重叠用户集 $\mathcal{U}^o$ 上使用 SmoothL1 Loss 训练：$\mathcal{L}_{main} = -\sum_{u \in \mathcal{U}^o, i \in \mathcal{I}^t} l(y_{u,i}, \hat{r}_{u,i})$，其中 $l$ 为 Huber 形式的分段损失。

## 实验与结果
- **数据集**：Amazon Reviews 三个垂直领域对：Movies→Music（Task 1）、Books→Movies（Task 2）、Books→Music（Task 3）；重叠用户数分别为 18,031、37,388、16,738。
- **评估基线**：TGT、CMF、SSCDR、EMCDR、PTUPCDR、REMIT、MIMNet；指标为 MAE
