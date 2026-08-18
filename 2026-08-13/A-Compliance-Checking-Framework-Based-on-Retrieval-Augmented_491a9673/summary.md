---
title: "A-Compliance-Checking-Framework-Based-on-Retrieval-Augmented"
source: https://aclanthology.org/2025.coling-main.178.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:58:08"
field: "法律与自然语言处理交叉"
keywords: ["合规检查", "RAG", "事件图", "道义命题", "法律NLP", "知识图谱", "大语言模型"]
innovations: ["提出静态-动态-计算三层RAG合规检查框架，融合结构化知识图谱与大模型推理", "设计以道义命题为核心的无监督事件图抽取算法，用于结构化表达法规知识", "展示跨领域ICL迁移能力，仅需源域少量标注即可迁移至新领域"]
benchmarks: ["EU2UK", "GDPR-13", "CONTRACT", "CSSCD"]
---

# 论文速读：A Compliance Checking Framework Based on Retrieval-Augmented Generation

## 一句话总结
本文提出一种基于检索增强生成（RAG）的合规检查框架，通过静态层、动态层和计算层三层架构，将结构化法规知识与大语言模型相结合，在欧盟-英国、GDPR、合同一致性、中国社会保险四个数据集上均取得最优性能。

## 研究问题与动机
- **合规检查的有效性需求**：企业需验证业务流程是否违反法律法规，传统人工检查成本高且易遗漏，需自动化手段辅助。
- **现有方法的局限**：基于逻辑的方法（如Petri网、一阶谓词逻辑）推理精确但缺乏可扩展性；基于语义嵌入的方法灵活泛化强，但丢失结构化信息和全局上下文，影响准确性与可解释性。
- **法规知识的特殊性**：法规文档中的知识并非以实体关系为中心，而是以行为、状态和道义命题（deontic propositions）为中心，传统知识图谱难以直接建模。
- **全局推理能力不足**：局部合规可能在全局视角下存在冲突，现有方法难以捕捉需要跨多条法规联合推理的合规冲突。

## 核心贡献（创新点）
- **提出三层RAG合规检查框架**：将静态事实知识层、动态法规/业务过程层与计算推理层分离，实现结构化知识与LLM参数知识的融合。
- **设计面向道义命题的事件图（eventic graph）来结构化表达法规信息**：节点主要描述行为与状态而非实体，关系类型涵盖义务（Duty）、禁止（Prohibited）、权利（Right）等，契合法规语义结构。
- **提出无监督的道义命题中心信息抽取算法**：利用LLM API直接从法规文档中抽取"主体-道义词-行为"三元组，无需人工标注数据。
- **在中文和英文多个数据集上实现SOTA**：在EU2UK（文档级）、GDPR-13、CONTRACT和CSSCD四个数据集上，MCC指标均显著优于既有基线方法。
- **展示跨领域迁移能力与全局推理优势**：仅需源领域少量标注即可迁移至新领域；能检测到需要结合多条法规才能识别的全局合规冲突。

## 方法详解
- **静态层（Static Layer）**：存储三类事实知识——实体中心型（CN-DBpedia/DBpedia）、概念中心型（OpenConcepts/ConceptNet）和术语定义型（通过BigBird+CRF联合抽取模型从法规/业务文档中提取术语-定义对，构建TDKnow）；三者合并为知识图谱$\mathcal{G}_{static}$。
- **动态层-事件图模块**：对法规文档构建事件图$\mathcal{G}_{eventic}$，节点包括组织、个人、法规文档、类别、行为和状态；关系类型包括Publish、WorkFor、Duty、Prohibited、Right、ClassifiedTo、Cite；使用Algorithm 1无监督抽取三元组。
- **动态层-块向量模块**：将业务流程文档分段为chunks，用SBERT编码为语义向量矩阵$\mathbb{C}$，存入Faiss向量库以便快速相似度检索。
- **计算层检索与推理流程**：
  - 步骤1：将每个chunk向量$\vec{c}_i$与事件图$\mathcal{G}_{eventic}$中每个节点$u_j$的嵌入向量计算余弦相似度，超过阈值$\lambda$则标记为命中，得到子图$\mathcal{G}_{sub}$。
  - 步骤2：将$\mathcal{G}_{sub}$的节点与$\mathcal{G}_{static}$的节点求交集$\mathcal{P}$。
  - 步骤3：获取$\mathcal{P}$的邻接节点集合$\mathcal{N}$，并取包含$\mathcal{N}$的最大连通子图$\mathcal{G}_{fus}$（异构知识子图）。
  - 步骤4：将chunk $c_i$与知识子图$\mathcal{G}_{fus}$包装入指令模板，输入LLM（中文用ChatGLM-3-6b，英文用LLaMa-2）生成合规检查结果及解释。

## 实验与结果
- **数据集**：EU2UK（文档级合规，英）、GDPR-13（隐私政策合规，英）、CONTRACT（合同一致性，英）、CSSCD（中国社会保障合规，中文）。
- **评估指标**：Matthews Correlation Coefficient（MCC），适配正负样本严重不平衡场景。
- **基线方法**：Doc2Doc、CLS、Offsets、NeuralConflict、TER-PLM、TER-Inner、TER-GraphAtt，以及多个开源LLM。
- **主要结果**（MCC值）：
  - EU2UK：Our framework **0.778**（第二名TER-GraphAtt 0.730，提升+0.048）
  - GDPR-13：**0.652**（第二名TER-Inner 0.620，提升+0.032）
  - CONTRACT：**0.730**（第二名TER-GraphAtt 0.713，提升+0.017）
  - CSSCD：**0.680**（第二名NeuralConflict 0.643，提升+0.037）
- **消融实验**：逐层移除静态层知识图谱后性能持续下降，证明各组件均不可缺；完全移除静态层后性能低于所有基线。
- **跨域迁移**：以CSSCD为源域，通过ICL零样本迁移至其他三个域，MCC分别为EU2UK 0.743、GDPR-13 0.603、CONTRACT 0.714，保持有效性能。
- **全局推理测试**：设计需跨多条法规联合推理的案例，仅Moonshot-v1-128k和GPT-4正确识别，说明全局推理是当前研究的难点。
- **超参分析**：阈值$\lambda$在0.7~0.8时最优；不同SBERT替换模型影响甚微；ChatGLM-3-6b在中文任务最优，LLaMa-2在英文任务最优。

## 相关工作脉络
- **逻辑推理方法**（Governatori et al., 2006; Zhang & El-Gohary, 2017; Rojas et al., 2016）：使用一阶逻辑、Petri网等进行显式推理，优势是精确可解释，但依赖专家且可扩展性差；本文与之的区别是不依赖形式化建模，借助RAG实现灵活推理。
- **语义嵌入方法**（Aires et al., 2018, 2021; Huang et al., 2024）：用向量偏移或分类网络做合规判断，泛化性强但丢失结构化信息；本文用SBERT+事件图保留了法规的结构化语义。
- **信息检索方法**（Chalkidis et al., 2021, Doc2Doc）：在EU2UK上效果较好，但本质仍是文档相似度匹配；本文引入知识子图检索增强，在文档级任务上超越Doc2Doc 0.048。
- **文本蕴涵方法**（Wehnert et al., 2022; Sun et al., 2017; Chen et al., 2019）：TER系列被纳入基线，在部分数据集上表现接近；本文框架因融合全局知识而综合最优。
- **知识图谱辅助合规**（Guo et al., 2021; Zheng et al., 2022; Fitkau & Hartmann, 2024）：利用本体或BIM建模；本文的创新在于提出eventic图（以行为/状态为核心节点）并结合RAG范式，而非纯本体方法。
- **大模型在法律/NLP中的应用**：本文探索了ChatGLM和LLaMa-2在合规推理中的能力边界，并为后续Agent-based检索优化提供了方向。

## 局限性与未来方向
- **检索阈值的静态性**：当前阈值$\lambda$为人工设定，当静态层或动态层知识更新后，原阈值可能失效，影响检索质量。未来可探索LLM Agent驱动的自适应知识检索机制。
- **事件图抽取依赖商业LLM API**：Algorithm 1使用商业LLM进行无监督抽取，存在成本和隐私考虑；未来可优化为开源模型或微调专用抽取器。
- **事件图本体设计的领域局限性**：六种实体类型和七种关系类型可能不足以覆盖所有行业法规，扩展性有待验证。
- **大规模知识图谱维护成本**：动态层法规库需持续更新，知识图谱的增量构建与版本管理尚未详细讨论。
- **中英双语模型性能差异未深入分析**：ChatGLM和LLaMa-2分别在各自语言表现最优，但模型选择与任务适配的深层原因未充分探讨。

## 研究启发与可借鉴点
- **静态-动态分层架构设计**：将"不变事实知识"与"可变法规/业务信息"分离存储，再通过检索融合，这一分层思路可迁移至其他法律NLP任务（如法规问答、合同审查）。
- **道义命题为中心的无监督抽取算法**：利用LLM指令模板进行规则驱动的信息抽取（无需标注），在缺少领域标注数据的新场景下极具借鉴价值。
- **事件图（eventic graph）的知识表示形式**：将行为/状态作为核心节点而非传统实体，对法律法规、政策文档、标准规范的结构化建模具有通用性。
- **跨域ICL迁移策略**：仅需在源域提供少量标注示例，即可让LLM在新领域进行零样本术语-定义抽取，降低了领域迁移的数据门槛。
- **全局合规冲突检测的评估范式**：设计了需跨多条法规联合推理的测试案例，可作为衡量LLM或RAG系统"全局推理能力"的有效基准。

## 关键术语表
- **Compliance Checking（合规检查）**：利用NLP技术验证企业业务流程文档是否符合相关法律法规和行业标准。
- **Eventic Graph（事件图）**：一种以行为（Action）和状态（State）为核心节点、以道义关系（Duty/Prohibited/Right等）为边的知识图谱，用于结构化表达法规知识。
- **Deontic Proposition（道义命题）**：表达义务、禁止或权利的规范性陈述，是法规知识的核心语义单元。
- **RAG（Retrieval-Augmented Generation，检索增强生成）**：在LLM生成过程中引入外部知识检索，将检索到的相关知识作为上下文输入以提升生成质量和准确性。
- **MCC（Matthews Correlation Coefficient）**：适用于二分类不平衡数据集的性能评估指标，取值范围[-1, 1]，值越高表示分类性能越好。
- **SBERT（Sentence-BERT）**：基于BERT的句子级语义嵌入模型，可生成高质量的低维语义向量用于相似度计算。
- **Faiss**：Facebook开发的稠密向量相似度检索库，支持大规模向量的高效近似最近邻搜索。
- **ICL（In-Context Learning，上下文学习）**：利用LLM在输入中给定少量示例即可适应新任务的能力，无需微调参数。

## 可复现要素
- **数据集**：EU2UK（Chalkidis et al., 2021）、GDPR-13（Liu et al., 2021）、CONTRACT（Aires et al., 2019）、CSSCD（Huang et al., 2024）；均为公开数据集。
- **代码/权重**：论文未明确声明开源仓库；模型部分使用了ChatGLM-3-6b和LLaMa-2等开源模型，BigBird抽取模型及5500条术语-定义标注数据未公开。
- **关键超参**：检索相似度阈值$\lambda$取0.7~0.8；BigBird模型最大输入长度5000 tokens，batch size为4，学习率2e-5，AdamW优化器，最多20 epoch并启用early stopping。
- **知识图谱资源**：静态层使用CN-DBpedia、OpenConcepts（中文）和DBpedia、ConceptNet（英文）等开源图谱。
- **LLM**：中文任务用ChatGLM-3-6b，英文任务用LLaMa-2；Eventic图抽取使用商业LLM API。
