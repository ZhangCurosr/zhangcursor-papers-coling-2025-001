---
title: "Taxonomy-Guided-Zero-Shot-Recommendations-with-LLMs"
source: https://aclanthology.org/2025.coling-main.102.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:00:21"
field: "LLM-based Recommendation"
keywords: ["zero-shot recommendation", "large language models", "taxonomy", "LLM prompting", "recommender systems"]
innovations: ["提出 TAXREC 框架通过分类学字典将 LLM 内在知识注入零样本推荐", "设计两阶段流程：一次性分类学构建 + LLM 特征匹配推荐", "证明结构化规则匹配优于 BERT 语义相似度在分类学场景中"]
benchmarks: ["MovieLens-100k", "BookCrossing"]
---

# 论文速读：Taxonomy-Guided Zero-Shot Recommendations with LLMs

## 一句话总结
论文提出 **TAXREC** 框架，通过利用 LLM 内在知识生成分类学字典，将物品池结构化为分类特征，在零样本场景下实现高效、可控的大语言模型推荐，显著优于传统方法和直接 LLM 推荐基线。

## 研究问题与动机
- **提示长度受限**：推荐系统中候选物品池可达百万级，每项以数十 token 表示，远超 LLM 的 prompt 长度限制；即使能放入全部物品，长上下文也会导致解码性能下降（lost-in-the-middle 效应）。
- **物品信息模糊且非结构化**：商品标题和描述由商家自由编写，如"1984"可指年份、书籍或电影，"Emma"可指人名或书名，缺乏足够上下文时 LLM 难以理解其真实含义。
- **生成不受约束**：纯文本生成方式易产生不在候选池中的虚构物品（如生成"Punch-Out!!!"），且无法对全量候选品计算排序分数，计算成本极高。

## 核心贡献（创新点）
- **分类学字典结构化框架**：首次系统性地利用 LLM 内在知识生成领域分类学字典，并将物品池组织为结构化的分类特征表示，缓解标题模糊问题。
- **TAXREC 两阶段零样本推荐框架**：提出一次性分类学构建 + LLM 推荐的两步流程，全程无需微调、无需额外用户交互数据，实现真正的零样本推荐。
- **特征匹配替代语义相似度**：设计基于分类学规则的匹配机制（`Score = |i^C ∩ F|`），证明结构化规则匹配优于 BERT 等学习式语义相似度方法。

## 方法详解
**整体架构（Figure 2）**：TAXREC 包含两个阶段：(a) 一次性分类学分类，(b) 基于 LLM 的推荐。

**阶段一：一次性分类学分类（Section 3.2）**
- 通过 `Taxonomy Generation Prompt`（Table 1）从 LLM 提取领域内分类学字典 $\mathcal{T}$，例如书籍领域可得到 Genre、Theme、Language 等属性。
- 对每个物品 $i$，使用 `Categorization Prompt` 结合 $\mathcal{T}$ 生成结构化分类表示 $i^C = [f_1, f_2, ..., f_{|i^C|}]$，形成分类物品池 $\mathcal{I}^C$。
- 例："1984"被细化为 "Type: Book, Genre: Fiction, Theme: Power..."，提供丰富上下文。

**阶段二：基于 LLM 的推荐（Section 3.3）**
- 将用户历史交互 $\mathcal{H}$ 映射为分类历史 $\mathcal{H}^C$，与分类学字典 $\mathcal{T}$ 共同构成 `Recommendation Prompt`。
- LLM 生成特征列表 $F = [f_1, f_2, ..., f_{|F|}]$，格式受 $\mathcal{T}$ 约束，避免幻觉生成。
- 通过公式 (5) 计算匹配分数：$\text{Score}_i = |i^C \cap F|$，取 Top-k 高分物品返回。

**关键设计**：用分类学字典 $\mathcal{T}$ 替代完整物品池输入 prompt，大幅节省 token；LLM 输出为结构化特征而非原始文本，便于精确匹配。

## 实验与结果
**数据集**：
- **Movie**：MovieLens-100k 子集，1,682 物品，2,000 测试序列，取前 10 次交互为历史。
- **Book**：BookCrossing 子集，4,389 物品，2,000 测试序列，随机采样历史交互。

**基线方法**：RecFormer、UniSRec、ZESRec、Popularity、AverageEmb、DirectLLMRec（无分类学直接推荐）。

**主要结果（Table 2）**：
- TAXREC-GPT4 在 Movie 上 R@10=**0.300**，N@10=**0.157**；在 Book 上 R@10=**0.240**，N@10=**0.138**，均显著优于所有基线（p<0.05）。
- DirectRec-GPT4 在 Book 上仅 R@10=0.025，TAXREC-GPT4 提升近 **10 倍**，证明分类学在未知领域尤为关键。
- TAXREC-LLaMA2 同样显著提升 DirectRec-LLaMA2（Movie: 0.190 vs 0.085；Book: 0.150 vs 0.015）。
- 消融实验（Table 3）：移除分类学（w/o Tax）在 Book 上性能跌至 0.025（R@10），移除匹配机制（w/o Match）亦有明显下降，验证两组件必要性。

## 相关工作脉络
- **RecFormer**（Li et al., 2023）：将物品编码为句子、历史交互为序列进行判别式推荐，使用预训练权重零样本推理，但未解决物品文本模糊问题。
- **UniSRec**（Hou et al., 2022）：利用预训练语言模型提取文本表示并通过 MoE 适配器域适应，本文保持预训练参数不做微调，更贴近纯零场景。
- **ZESRec**（Ding et al., 2022）：用预训练 BERT 编码物品文本作为特征，使用冻结 BERT 嵌入公平对比，未涉及 LLM 知识检索。
- **DirectLLMRec**：本文提出的变体，直接将历史物品喂给 LLM 生成推荐，作为对比验证分类学框架的价值。
- **ChatRec**（Gao et al., 2023）：交互式 LLM 增强推荐系统，支持多轮对话，但需微调，非纯零样本场景。
- **RecMind**（Wang et al., 2023b）：结合数据库与 LLM 作为自主 Agent，使用外部工具缩小候选池，本文完全依赖 LLM 内在知识压缩候选。

## 局限性与未来方向
- 分类学生成仅依赖 LLM prompt，可能存在更有效的分类方法以捕捉更细粒度物品特征。
- LLM 领域知识不足时（如 Book 数据集表现弱于 Movie），分类学和推荐质量均受影响。
- 自动生成分类学缺乏科学严谨性和完整性，未来需引入更系统化的分类标准。
- 仅验证了 Movie 和 Book 两个数据集，未覆盖更多领域（如 POI、新闻等）。

## 研究启发与可借鉴点
- **知识蒸馏式提示设计**：通过 prompt 让 LLM 输出结构化分类知识，再用于下游任务，是一种无需微调的"知识注入"范式，可迁移至其他 LLM 应用。
- **结构化约束替代语义匹配**：在特征表示已结构化的场景下，规则匹配（集合交集）优于 BERT 语义相似度，提示我们在特定任务中重新评估"复杂模型"的必要性。
- **两阶段解耦思路**：先构建静态知识表示（分类学+物品分类），再用于动态推荐，可分离计算与推理阶段，降低在线延迟。
- **Prompt 设计敏感性的系统探索**：Table 4 展示了不同 prompt 组合效果差异，提示后续工作应更精细地调优输入/输出格式。

## 关键术语表
- **Zero-shot Recommendation**：零样本推荐，指无需用户-物品交互历史数据进行模型微调即可进行推荐。
- **Taxonomy Dictionary**：分类学字典，由 LLM 生成的领域特征分类体系，用于结构化和压缩物品表示。
- **TAXREC**：Taxonomy-guided Recommendation 框架，本文提出的两阶段零样本推荐方法。
- **Categorized Item Pool**：分类物品池，将原始物品标题通过分类学字典转化为结构化特征列表。
- **Feature Matching Score**：特征匹配分数，通过计算分类物品特征与 LLM 生成特征的交集大小进行排序。
- **DirectLLMRec**：直接 LLM 推荐基线，不经过分类学处理，直接将历史交互喂给 LLM 生成推荐。

## 可复现要素
- **数据集**：MovieLens-100k（Movie 子集）、BookCrossing（Book 子集），来源于公开数据集。
- **代码**：已开源，地址 https://github.com/yueqingliang1/TaxRec。
- **模型**：使用 GPT-4（OpenAI API）和 LLaMA-2-7b（预训练参数），均需访问相应资源。
- **关键超参**：历史交互长度固定为 10；prompt 设计见 Table 1；分类学特征数量实验探索 5/10/15/20；匹配策略以集合交集为主。
