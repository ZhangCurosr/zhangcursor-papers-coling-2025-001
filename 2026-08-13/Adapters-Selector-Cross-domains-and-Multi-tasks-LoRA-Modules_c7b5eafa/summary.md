---
title: "Adapters-Selector-Cross-domains-and-Multi-tasks-LoRA-Modules"
source: https://aclanthology.org/2025.coling-main.40.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:29:34"
field: "大语言模型高效微调"
keywords: ["PEFT", "LoRA", "多任务学习", "跨领域适配", "Adapter", "Selector"]
innovations: ["提出解耦的Adapter-Selector框架实现跨领域多任务动态路由", "基于Kmeans的LLM Embedding向量数据选择方法", "支持Add/Balance/Delete的轻量级Selector在线更新机制"]
benchmarks: ["MedNLP", "BioInstruct", "Fin-GPT", "Legalbench"]
---

# 论文速读：Adapters-Selector-Cross-domains-and-Multi-tasks-LoRA-Modules

## 一句话总结
论文提出 Adapters Selector (AS) 框架，通过训练一个轻量级 Selector 识别输入数据的领域与任务，从而动态切换对应的 LoRA Adapter，实现跨领域多任务场景下多个 PEFT 模块的高效集成与推理。

## 研究问题与动机
1. **单领域单任务 PEFT 的局限性**：现有参数高效微调（PEFT）方法在单一领域任务上表现优异，但缺乏对多个 Adapter 进行有效集成与切换的机制。
2. **跨领域多任务的冲突问题**：不同领域/任务的 LoRA 模块同时作用于同一基座模型时会产生知识冲突，难以兼顾各领域专属性能。
3. **现有方法的不足**：多任务 PEFT 方法（如 MFTCoder、MoELoRA）依赖复杂的数据构建与同步训练，带来额外训练成本与推理延迟，且需针对 MoE 等特殊结构进行定制训练。
4. **模块化集成的空白**：如何让不同来源、不同参数的 LoRA 模块在不重新联合训练的前提下，在单一模型上保持各自领域专属能力并实现灵活切换，尚待探索。

## 核心贡献（创新点）
1. **提出 AS 框架**：将 Adapter 训练与 Selector 训练解耦，通过 Selector 实现"输入→领域/任务→Adapter"的动态路由，避免多任务联合训练带来的性能退化。
2. **Kmeans 数据选择方法**：利用基座模型 Embedding 层输出向量计算样本间距离（支持 L2/IP/COS），为每个任务选取代表性子集用于 Selector 训练，降低训练成本。
3. **支持三种 Selector 更新操作**：新增（Add）、平衡（Balance）、删除（Delete）任务，通过混合新旧数据二次微调实现 Selector 的灵活迭代。
4. **验证跨模型通用性**：在 Qwen2、Gemma、Yi、Internlm2.5、Deepseek、Llama3 等不同架构与规模（1.5B–9B）的基座模型上验证 AS 的有效性，小模型亦能取得出色表现。

## 方法详解
### 框架流程
- **步骤一**：对每个领域-任务数据集，使用 LoRA（或其变体）独立微调基座模型，获得专属 Adapter 权重。
- **步骤二**：对每个任务的数据集应用 Kmeans 数据选择，生成等量规模的表示数据集，混合后用于训练 Selector。
- **步骤三**：推理时，输入先经 Selector 识别领域与任务，再切换对应 Adapter 进行生成。

### Kmeans 数据选择（Algorithm 1）
1. 使用基座模型 Embedding 层（或 m3e）对每条输入生成向量。
2. 初始化 k 个聚类中心，通过迭代更新中心（按最近邻分配样本后重新计算均值）直至收敛。
3. 选取距离各聚类中心最近的样本构成该任务的代表性子集。
4. 三种距离度量：L2（欧氏距离）、IP（内积）、COS（余弦相似度）。

### Selector 训练
- 输入格式：`input:{text}`
- 输出格式：领域 + 任务的分类标签
- 使用 PEFT（LoRA）在小型混合数据集上微调基座模型得到 Selector。

### 推理流程（Figure 3）
1. **Adapter Selection**：输入经基座模型 + Selector 识别出领域与任务。
2. **Adapter Switching**：断开当前 Adapter，加载目标 Adapter。
3. **Domain Inference**：结合领域/任务模板提示词，执行推理生成。
4. 推理完成后基座模型重新连接 Selector，等待下一请求。

### 更新策略
- **Add**：保留原数据（比例 10%/30%/50%）+ 新任务数据，融合后二次微调。
- **Balance**：对边界模糊任务增广、对易区分任务降采样，平衡 Selector 性能。
- **Delete**：移除难区分任务，仅保留少量原始数据 + 剩余任务数据重新微调。

## 实验与结果
### 数据集
- **医疗（Med）**：MedNLP（MC）、BioInstruct（IE/TG/QA）
- **金融（Fin）**：Fin-GPT（RE/SA/HA/CM）
- **法律（Legal）**：Legalbench（MC/IE/JW/JI）

### 评估基线
- **MFTCoder**：多任务 LoRA 联合训练，结合多损失函数处理数据不平衡。
- **MoELoRA**：基于 MoE 结构的 LoRA 多任务训练。

### 主要结果
- **Selector 精度**（Table 1, 12任务）：使用 Embedding 层 + L2/COS 距离、k=1000 时 Selector 准确率最高达 **0.989**；L2 距离表现稳定最优。
- **跨领域任务保持率**（Table 2）：12 任务场景下，Legal/Fin 多数任务保持率 >90%；Med 中 TG/QA 因任务模糊性较低（<60%），影响同域 IE 至约 65%。
- **对比 SOTA**（Table 1 底部）：AS（Selector 14，Embedding 层+L2）在 Fin SA 上达 **0.975**，显著优于 MFTCoder（0.936）与 MoELoRA（0.728）。
- **LoRA 变体适配**（Figure 4）：r=4 时 rsLoRA 收敛更快，r=8 时 DoRA 表现最优；LoRA/DoRA 最佳参数为 r=4, α=32，rsLoRA/PiSSA 为 r=8, α=16。
- **更新效果**（Table 3）：Add 操作（保留 50% 原数据）精度仅下降 0.5%；Balance 操作难以处理模糊任务（IE 下降 33%）；Delete 操作在仅保留 0.1% 原数据时精度仍达 98.9%。
- **跨模型通用性**（Table 4）：Internlm2.5-7B 综合表现最佳（acc 0.881）；Gemma-2B 小模型效果甚至优于 Yi1.5-6B 与 Qwen2-7B。

## 相关工作脉络
1. **LoRA 及其变体**（Hu et al., 2021; Kalajdzievski, 2023; Liu et al., 2024b; Meng et al., 2024）：AS 采用 LoRA 作为 Adapter 训练基础，并验证了 rsLoRA/DoRA/PiSSA 等变体的兼容性。
2. **多任务 PEFT**（Crawshaw, 2020; Liu et al., 2024a; Gao et al., 2024）：MFTCoder 和 MoELoRA 通过联合训练或多专家结构处理多任务，AS 通过解耦训练+动态切换避免此类方法的数据构建与同步训练开销。
3. **Adapter 集成方法**（Huang et al., 2023; Pfeiffer et al., 2020）：LoraHub 和 AdapterFusion 关注权重融合，AS 则通过 Selector 实现运行时动态选择，保留各 Adapter 独立能力。
4. **MoE-LoRA 方法**（Dou et al., 2024; Feng et al., 2024; Li et al., 2024）：需要定制 MoE 训练结构，AS 以轻量 Selector 替代复杂 MoE，推理时仅加载单一 Adapter，降低显存压力。
5. **数据选择方法**（Chen et al., 2023; Wang et al., 2024a）：AS 借鉴 Kmeans 聚类的数据选择思想，创新性地使用 LLM Embedding 层向量替代独立 embedding 模型，提升效率。

## 局限性与未来方向
1. **推理延迟**：Selector 引入额外 prefill 时间（约 3 个 token 解码时间），双次切换带来轻微延迟，需在实际部署中优化。
2. **模糊任务区分困难**：文本生成（TG）和问答（QA）等边界模糊的任务导致 Selector 准确率偏低，进而影响同域其他任务性能。
3. **任务数量上限**：单模型部署超过 12 个任务时会面临显存与负载均衡问题，实际应用需进一步探索可扩展性。
4. **未探索场景**：目前仅在 NLP 领域验证，视觉/多模态任务的跨域 Adapter 集成尚未涉及。

## 研究启发与可借鉴点
1. **解耦训练范式**：Adapter 独立训练 + Selector 动态路由的设计思路可迁移至多模态、代码生成等领域，避免多任务联合训练的灾难性干扰。
2. **Embedding 层复用**：直接使用 LLM 自身 Embedding 层进行数据选择，替代独立 embedding 模型（如 m3e），节省显存与计算开销，且高维特征（3584 vs 768）提升聚类质量。
3. **轻量更新机制**：Add/Balance/Delete 三类更新操作的实验设计为动态任务管理提供了可参考的工程实践方案，尤其 Delete 操作的高保留率（98.9%）对任务剔除场景有指导意义。
4. **跨模型验证策略**：在 7 种不同架构/规模模型上统一验证框架通用性，为后续研究的多模型适配实验提供了可复现的实验设计模板。

## 关键术语表
- **PEFT（Parameter-Efficient Fine-Tuning）**：参数高效微调，仅更新模型少量参数即可适配下游任务的微调策略。
- **LoRA（Low-Rank Adaptation）**：通过注入低秩分解矩阵实现高效微调的主流 PEFT 方法。
- **Selector**：AS 框架中用于识别输入领域与任务的轻量适配器，实现动态路由。
- **Adapter**：针对单一领域-任务独立训练的 LoRA 权重模块，推理时可插入基座模型。
- **Kmeans 数据选择**：基于聚类中心的代表性样本筛选方法，用于构建 Selector 训练集。
- **L2/IP/COS 距离**：分别指欧氏距离、内积、余弦相似度，用于衡量样本向量间相似性。
- **Add/Balance/Delete**：Selector 的三种更新操作，分别对应新增任务、平衡任务分布、删除任务。
- **Bertscore-F1**：基于 BERT 的文本生成质量评估指标，衡量生成文本与参考文本的语义相似度。

## 可复现要素
- **数据集**：MedNLP、BioInstruct、Fin-GPT、Legalbench（均为公开数据集，附录 B 提供示例）
- **代码开源**：https://github.com/tirant35/TASA
- **权重开源**：论文未明确提及权重开源地址
- **关键超参**：
  - LoRA: r=4/8, α=16/32, dropout=0.05
  - 学习率：4e-4
  - 训练轮数：5 epochs
  - 最大序列长度：512
  - Kmeans 聚类数 k：100/500/1000
  - Embedding：m3e（768维）或 LLM 自身 Embedding 层（3584维）
- **硬件**：3× NVIDIA 4090（24GB）
