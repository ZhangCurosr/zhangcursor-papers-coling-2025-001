---
title: "LogiGraph-Logical-Reasoning-with-Contrastive-Learning-and-Li"
source: https://aclanthology.org/2025.coling-main.72.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:43:28"
---

# 论文速读：LogiGraph-Logical-Reasoning-with-Contrastive-Learning-and-Li

## 一句话总结
本文提出 LogiGraph，通过构建基于连词与标点的对偶图、设计轻量化图神经网络并结合对比学习生成负样本，有效兼顾上下文语义与显式逻辑结构，在 ReClor 与 LogiQA 逻辑推理数据集上取得 SOTA 性能。

## 研究问题与动机
- 现有 MRC 逻辑推理方法往往偏重单向表征：要么依赖预训练语言模型的上下文语义匹配，要么仅抓取实体共现关系，难以平衡语义理解与显式逻辑依赖。
- 传统 GCN 在文本图节点更新中易引发过度平滑，导致节点特征趋同；且多层特征变换、对称归一化等组件徒增参数量与推理耗时，缺乏面向逻辑推理的轻量化适配。
- 现有方法多聚焦正向逻辑链路学习，缺乏对负向逻辑关系（如条件反转、逻辑否定）的显式建模，限制了模型在复杂论证场景下的泛化与辨别能力。

## 核心贡献（创新点）
- 提出基于连词与标点的对偶图构建策略，分别刻画显式逻辑联结与句法结构边界，区别于传统方法仅依赖实体共现或单一语义边的稀疏图表示。
- 设计轻量化 GCN 节点更新机制，剔除特征变换层、对称归一化项与自连接，仅保留线性邻域聚合与后置 ReLU，区别于传统 GCN 因深层非线性变换导致的节点平滑与推理延迟问题。
- 引入句法规则驱动的对比学习负样本生成机制，通过删除、条件反转与逻辑否定构造负向上下文，区别于 MERIt 等依赖外部知识图谱或大模型反事实改写可能引入知识污染的做法。
- 构建分类损失与对比损失联合优化的端到端推理框架，在低参数开销下同步提升 ReClor 与 LogiQA 的测试精度与题型覆盖能力。

## 方法详解
- **输入与图构建**：以 RoBERTa-Large 为 Token 编码器，依据显式连词（如 because, but, so）切分节点构建连词图 $G_{conj}$，依据标点符号（如 ., ,）切分节点构建结构图 $G_{struct}$，形成对偶图 $(N_i, E_i)$。
- **节点编码**：节点内 Token 向量求和融合获得初始特征 $\mathbf{s}_i$，叠加位置编码后得到节点表示 $\mathbf{n}_i$；双图结构有效缓解单 Token 节点的表征稀疏性。
- **轻量化图网络**：消息聚合简化为 $\tilde{\mathbf{n}}_k = \sum_{j \in \mathcal{N}_i} \mathbf{A}^{R_{ji}} \mathbf{n}_j$（$R \in \{R_c, R_s\}$），经 $l$ 层高阶聚合后叠加残差并施加 ReLU，最终通过全局平均池化得到图表示 $\mathbf{N}_g$，再与 PLM 的 [CLS] 向量拼接为 $\tilde{\mathbf{N}}_g$。
- **对比学习模块**：利用 NLTK 对原始上下文执行句法扰动生成负样本 $S_n^-$，正样本 $S_n^+$ 与负样本共享同一图编码架构；对比损失 $\mathcal{L}_C = -\sum \log \frac{e^{s'(S_n, S_n^+)}}{e^{s'(S_n, S_n^+)} + e^{s'(S_n, S_n^-)}}$ 拉大正负表示距离。
- **答案预测**：图全局表示 $\tilde{\mathbf{N}}_g$ 与上下文/问题经 Cross-Attention 交互后拼接，送入前馈网络与 Softmax 输出选项概率；总损失 $\mathcal{L}_{total} = \mathcal{L}_{CE} + \mathcal{L}_{C}$。

## 实验与结果
- **数据集**：ReClor（Dev/Test/Test-E/Test-H）与 LogiQA（Dev/Test），均为公开机器阅读理解逻辑推理基准。
- **评估基线**：BERT-Large、XLNet、RoBERTa-Large、DAGN、FocalReasoner、MERIt、LoCSGN、Logiformer 及 ChatGPT (Turbo 3.5)。
- **主要结果**：ReClor 测试集 65.0%（较次优 Logiformer 提升 3.51%）；LogiQA 测试集 44.42%（较 Logiformer 提升 4.39%），均超越同期 SOTA 及通用大模型。替换为 ALBERT-xxlarge-v2 与 DeBERTa-v2-xlarge 后仍保持一致性增益。
- **消融结论**：连词图与结构图分别带来 ReClor 测试集 +4.7% 与 +4.39% 的精度提升；轻量化 GCN 相比标准 GCN 推理时间缩短约 14%（0.1918s vs 0.2231s）且精度更高；对比学习模块贡献稳定；模型在 Necessary Assumption 与 Principle 等复杂题型上提升最为显著。

## 相关工作脉络
- **DAGN / Logiformer**：前者用 GCN 聚合话语特征，后者用图 Transformer 提取特征，均依赖重参数图网络；本文以极简线性聚合替代复杂图算子，并在结构之外额外注入对比学习以弥补负向逻辑缺失。
- **FocalReasoner**：聚焦事实单元抽取构建超图，仅覆盖单一逻辑视角；本文通过连词+标点对偶图同时捕获显式联结与句法边界
