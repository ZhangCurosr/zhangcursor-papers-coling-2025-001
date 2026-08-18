---
title: "Knowledge-Graph-Entity-Typing-with-Curriculum-Contrastive-Le"
source: https://aclanthology.org/2025.coling-main.38.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:41:23"
field: "知识图谱表示学习"
keywords: ["知识图谱实体类型预测", "课程对比学习", "图神经网络", "语义融合", "对比学习"]
innovations: ["Enhanced-MLP融合语义与结构嵌入", "课程对比学习策略（动态噪声调度）", "多视图对比损失函数设计"]
benchmarks: ["FB15kET", "YAGO43kET"]
---

# 论文速读：Knowledge-Graph-Entity-Typing-with-Curriculum-Contrastive-Le

## 一句话总结
本文提出CCLET模型，结合课程对比学习策略解决知识图谱实体类型预测（KGET）任务，通过PLM与LightGCN分别编码语义与结构信息，利用Enhanced-MLP融合两者，并通过动态噪声调节的课程学习机制从简单到复杂逐步训练，在FB15kET和YAGO43kET数据集上取得SOTA性能。

## 研究问题与动机
- **核心问题**：知识图谱中实体类型标注不完整（如FB15kET中10%标为"/music/artist"的实体缺少"/people/person"类型），且实体类型高度多样化（47.4%的实体拥有超过10个类型）。
- **现有方法局限一**：大多数近期模型仅关注KG的结构信息或文本语义信息其中之一，缺乏两者的有效融合（如CompoundE、MCLET、MiNer）。
- **现有方法局限二**：训练过程中所有特征被同时学习，忽略了难易程度差异，导致训练效率低（如SSET、TET）。

## 核心贡献（创新点）
- **Enhanced-MLP融合架构**：设计了包含Batch Normalization、Dropout和残差连接的增强型MLP，将语义嵌入与结构嵌入融合至统一维度空间；与已有工作相比，该方法通过可学习嵌入与GCN编码的多视图结构显式解耦语义与结构信息后再融合，避免了简单的拼接或相加。
- **课程对比学习策略**：提出通过线性递增高斯噪声水平自动控制训练难度的课程学习机制；本质区别在于将课程学习应用于对比学习框架，通过噪声调度实现从易到难的渐进式表征学习，而非传统课程学习中基于样本难度排序的策略。
- **新型对比损失函数**：设计包含视图内对比与视图间对比的联合损失函数；与标准InfoNCE相比，该方法在原始嵌入与噪声嵌入上分别计算对比损失并组合，增强了模型对细微特征差异的区分能力。
- **多视图图结构构建**：将KG转换为包含entity-type、cluster-type、entity-cluster三种边类型的三级子图结构；相比仅使用triples或types单视角的方法，该方法显式建模了不同粒度视角的知识。

## 方法详解
**整体架构**：CCLET由两大部分组成——知识融合模块（左侧）与课程对比学习模块（右侧）。

**结构信息处理**：
- 将KG的triples（$G_{triples}$）和types（$G_{types}$）转换为三级图结构，生成entity-type、cluster-type、entity-cluster三种边类型
- 使用LightGCN分别编码三个子图，获取结构嵌入$Struct_e$、$Struct_c$、$Struct_t$

**语义信息处理**：
- 使用BERT编码实体名称、描述、类型及聚类信息
- 为每个实体、聚类、类型维护一组可学习的语义嵌入$Sem_e$、$Sem_c$、$Sem_t$

**Enhanced-MLP融合**：
- 第一层：$y_1 = Dropout(ELU(BN_1(W_1 x + b_1)), p)$
- 第二层（含残差）：$y_2 = BN_2(W_2 y_1 + b_2) + x$
- L2归一化后得到统一的语义和结构嵌入，最终$hybrid = Struct + Sem$

**课程对比学习**：
- 噪声调度策略：$\sigma(E) = \sigma_0 + (\sigma_{max} - \sigma_0) \cdot \frac{E}{E_{max}}$（线性递增）
- 数据扰动：$\tilde{x} = x + N(0, \sigma^2)$
- 视图定义：entity-cluster视图（粗粒度）与entity-type视图（细粒度）
- 对比样本划分：锚点、加噪正样本、其他节点负样本
- 损失函数：$L_{joint} = mean[\sum(L_{orig} + L_{noise} + L_{orig-noise})]$
- 总损失：$L = L_{ET} + \lambda L_{joint} + \gamma\|\Theta\|_2^2$（$L_{ET}$为SFNA损失）

## 实验与结果
**数据集**：FB15kET（14,951实体、1,345关系、3,584类型、1,081聚类）和YAGO43kET（42,335实体、37关系、45,182类型、1,124聚类），文本信息通过Wikidata API获取。

**评估指标**：MR、MRR、Hits@k（k=1,3,10）。

**主要结果**：
- **FB15kET**：CCLET在所有五维指标上均达SOTA，Hit@1=70.2%（较第二名提升3.4%），Hit@3=81.1%，Hit@10=90.1%，MR=11
- **YAGO43kET**：MR=176（较SSET提升68位），Hit@1=44.8%（第二），但Hit@10=55.0%低于SSET的57.6%
- 扩大评测范围至H@50~H@200时，CCLET在H@100上提升3.1%，其他指标提升至少2.4%

**训练效率**：FB15kET耗时2.8小时（较SSET减少75%），YAGO43kET耗时11.8小时（减少70%），使用单张RTX 3090 GPU训练500 epoch。

**消融实验**：
- 仅结构：Hit@1=66.5%；结构+实体语义：70.2%；加入对比学习：69.5%；加入课程学习：70.2%（完整模型最优）
- 消融表明对比学习带来约3%的Hit@1提升，课程学习显著改善泛化能力

**关键案例**：在"educational television"、"multiple sclerosis"等少样本类型上，正确排名提升幅度高达2264位。

## 相关工作脉络
- **与Embedding-based方法（ETE、ConnectE、CompoundE）的关系**：此类方法仅依赖结构化嵌入，缺乏语义信息利用；CCLET通过PLM引入丰富上下文语义，弥补单一结构表征的不足。
- **与GNN-based方法（MiNer、MCLET）的差异**：MiNer挖掘邻居共现关系，MCLET采用多视角对比但无课程调度；CCLET的创新在于将课程学习与跨视图对比结合，并显式建模三级图结构。
- **与Transformer-based方法（TET）的对比**：TET整合局部、全局与上下文信息但未融合结构化GNN；CCLET通过LightGCN捕获图谱拓扑与BERT语义形成互补。
- **与Hybrid-based方法（SSET）的定位差异**：SSET同样融合语义与结构，但采用一次性训练所有特征；CCLET通过课程对比学习实现渐进式训练，显著提升收敛速度与鲁棒性。
- **与对比学习工作（GCL等）的联系与区别**：GCL等通过数据增强进行对比，CCLET通过可控噪声注入与课程调度，使对比学习更具目标导向性。

## 局限性与未来方向
- **语义融合可进一步优化**：当前Enhanced-MLP的融合方式较为直接，可探索更深层、更精细的语义特征挖掘机制。
- **大规模数据集表现不佳**：在YAGO43kET等大尺度数据集上，冗余特征与噪声干扰训练，导致部分指标落后于SSET。
- **未来方向**：改进语义与结构信息的融合策略；针对大规模KG设计抗噪更强的表征学习方法；探索更复杂的课程调度机制。

## 研究启发与可借鉴点
- **课程学习+对比学习的结合范式**：通过可控噪声线性递增实现"由易到难"的训练策略，可有效提升对比学习的稳定性与泛化能力，可迁移至其他图表示学习任务。
- **多视图对比损失的精细化设计**：将原始嵌入与噪声嵌入分别计算对比损失并联合优化，增强了模型对细微特征差异的区分力，该设计思路适用于多视角表征学习场景。
- **三级图结构构建策略**：将KG划分为entity-type、entity-cluster、cluster-type三种边类型，显式建模不同粒度视角，为多粒度知识融合提供了新思路。
- **消融实验设计的系统性**：从结构/语义分别消融到各模块逐一验证，清晰地量化了各组件贡献，值得在模型评估中借鉴。

## 关键术语表
**KGET（Knowledge Graph Entity Typing）**：知识图谱实体类型预测任务，旨在推断KG中实体缺失的类型标注。
**Curriculum Learning（课程学习）**：受人类学习启发的训练策略，按难度递增顺序安排样本，提升模型泛化能力。
**Contrastive Learning（对比学习）**：通过拉近正样本对、推远负样本对来学习判别性表征的自监督方法。
**Enhanced-MLP**：包含Batch Normalization、Dropout和残差连接的增强型多层感知机，用于融合语义与结构嵌入。
**LightGCN**：简化版图卷积网络，专为推荐系统设计，通过邻居聚合生成节点嵌入。
**SFNA（Semantic Fusion with Neighborhood Aggregation）**：本文采用的实体类型预测损失函数。
**PLM（Pre-trained Language Model）**：预训练语言模型，本文使用BERT编码实体文本信息。
**Coarse-grained Clustering（粗粒度聚类）**：将实体划分为抽象聚类，用于构建三级图结构中的中间层次。

## 可复现要素
- **数据集**：FB15kET和YAGO43kET为标准公开数据集；实体描述通过Wikidata API获取。
- **代码/权重**：论文未提及开源情况。
- **关键超参**：学习率[0.1, 0.01, 0.001]，batch size[32, 64, 128, 256]，语义嵌入维度768，结构嵌入维度[50, 100, 200]，权重α∈[0.3, 0.5, 0.7]，λ∈[0.0001, 0.001, 0.01]，L2正则γ∈[1e-6, 1e-5, 1e-4]，训练500 epoch，单卡RTX 3090。
- **预训练模型**：BERT（语义编码器）。
