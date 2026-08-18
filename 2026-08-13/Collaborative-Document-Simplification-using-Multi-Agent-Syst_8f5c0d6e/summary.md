---
title: "Collaborative-Document-Simplification-using-Multi-Agent-Syst"
source: https://aclanthology.org/2025.coling-main.60.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:02:14"
field: "文本简化与可读性"
keywords: ["文本简化", "文档级简化", "多智能体系统", "大语言模型", "Prompt Engineering"]
innovations: ["提出 AgentSimp，首个将隐喻分析与术语解释同时引入文档简化的多智能体框架", "系统对比管道式与同步通信策略、直接拼接与迭代拼接重建策略的四种组合", "无需监督训练，基于 GPT-4 的 AgentSimp 在 Newsela 上 SARI 达 41.58，超越所有微调开源基线"]
benchmarks: ["Newsela", "GuoFeng Webnovel"]
---

# 论文速读：Collaborative-Document-Simplification-using-Multi-Agent-Syst

## 一句话总结
本文提出 AgentSimp，一种基于大型语言模型的多智能体协作框架，用于**文档级文本简化**。该框架模拟人类专家团队的分工协作流程，通过引入隐喻分析与术语解释等细粒度简化角色，在自动评测和人工评测中均优于现有单模型/单遍生成方法及微调开源模型。

## 研究问题与动机
1. **文档级简化数据稀缺**：高质量并行训练/微调语料极少，现有句/词级方法难以直接推广至长文档。
2. **小参数模型质量受限**：现有端到端简化模型存在语法错误、忠实性不足、流畅性差等问题，且无法处理长文本的细粒度需求（隐喻、高级情感、专业术语、篇章连贯性）。
3. **单遍生成易产出摘要**：LLM 在单次提示下倾向于将长文档简化为摘要而非平行简化版本。
4. **多智能体系统在简化任务中尚未探索**：现有 LLM 多智能体工作（Agent-Verse、AutoGen 等）针对通用任务，不适配文档简化这一特定需求。

## 核心贡献（创新点）
1. **提出 AgentSimp 多智能体文档简化框架**：首次将隐喻分析（Metaphorical Analyst）与术语解释（Terminology Interpreter）同时引入文档简化任务，与现有单模型或两阶段方法形成本质区别。
2. **探索两种智能体通信策略与两种文档重建策略**：系统对比管道式与同步通信策略、直接拼接与迭代拼接重建策略，为文档简化工作流设计提供新视角。
3. **无需监督训练即可超越微调开源模型**：基于 GPT-4 的 AgentSimp 在 Newsela 上 SARI 达 41.58，超越微调后的 Llama-2（36.51）和 Mistral（37.20），证明无需大规模平行语料也能取得优越简化效果。
4. **全面的自动评测与人工评测验证**：在 Newsela 和 GuoFeng（网络小说）两个数据集上进行评测，证明方法在通用新闻与文学领域均有效。

## 方法详解
**AgentSimp 框架包含三大组件：**

### 1. 角色分配（Role Allocation）
共 9 个智能体，模拟人类专家团队：
- **Project Director（项目主管）**：阅读原文后生成简化指南（含摘要、关键词、文化背景、风格、目标受众）。
- **Article Logic Analyst（文章逻辑分析师）**：构建结构化大纲（主标题+子标题），指导后续重建。
- **Content Simplifier（内容简化器）**：执行初始简化——拆分/合并/删除/重排句子，替换复杂词汇。
- **Simplify Supervisor（简化监督员）**：审查简化结果并给出改进建议。
- **Metaphorical Analyst（隐喻分析师）**：为隐喻和复杂情感表达提供清晰解释。
- **Terminology Interpreter（术语解释器）**：为专业术语提供通俗化解释。
- **Content Integrator（内容整合器）**：仅同步策略中使用，融合多个角度的简化结果。
- **Article Architect（文章架构师）**：根据指南和大纲，将简化段落重建为连贯完整文档。
- **Proofreader（校对员）**：最终审查语法、拼写、事实错误，输出最终简化文档。

### 2. 四步整体流程
- **步骤 1（总体规划）**：Project Director 生成简化指南，Article Logic Analyst 构建结构大纲。
- **步骤 2（初始简化）**：按段落切块后，Content Simplifier 做基础简化，Simplify Supervisor 反馈改进意见。
- **步骤 3（精细简化）**：引入 Metaphorical Analyst 和 Terminology Interpreter，对隐喻和术语进行专项简化。
- **步骤 4（重建与修订）**：Article Architect 将简化段落块重建为完整文档，Proofreader 做最终审校。

### 3. 通信策略
- **管道式通信（Pipeline）**：对每个段落依次经过 Content Simplifier → Metaphorical Analyst → Terminology Interpreter 串行处理。
- **同步通信（Synchronous）**：三个简化代理并行处理同一输入，由 Content Integrator 融合结果。

### 4. 文档重建策略
- **直接拼接（Direct Combination）**：将所有简化段落块直接合并后重建，适合短文档，保留结构准确。
- **迭代拼接（Iterative Combination）**：每次取 C=2 个段落块进行重建，上一步输出末尾段落作为下一步输入的首段，适合长/复杂文档。

### 关键超参
- Temperature 设为 **0.6**
- Llama-2/Mistral 微调：LoRA，learning rate 5e-5，30 epochs，max_samples 500

## 实验与结果
**数据集**：
- **Newsela**（200 篇文档，4,850 段落，平均 1,103 词/文档）：通用新闻领域，有参考简化文档。
- **GuoFeng Webnovel**（100 篇文档，6,056 段落，平均 1,758 词/文档）：网络小说文学领域，无简化参考，仅做人评。

**评估指标**：SARI↑、D-SARI↑、BARTScore(BART-S)↓、FKGL↓ + 人工评测（连贯性/简化度/忠实度/偏好度）。

**主要结果（Newsela 数据集，基于 GPT-4）**：

| 方法 | SARI | D-SARI | BART-S | FKGL |
|---|---|---|---|---|
| Llama-2（微调） | 36.51 | 26.92 | -3.13 | 6.77 |
| Mistral（微调） | 37.20 | 27.98 | -2.84 | 5.29 |
| GPT-4（单提示） | 33.61 | 22.67 | -2.78 | 7.58 |
| **AgentSimp PL†** | **41.58** | **30.72** | **-2.53** | 8.42 |
| AgentSimp PL | 40.37 | 28.85 | -2.76 | 8.16 |
| AgentSimp SYNC† | 40.53 | 28.22 | -2.42 | 8.96 |

**最强结果**：AgentSimp 基于 GPT-4 + 管道式通信 + 迭代重建（PL†），SARI = **41.58**，D-SARI = **30.72**，相对微调 Llama-2 提升 **+5.07** SARI，相对微调 Mistral 提升 **+4.38** SARI，显著优于所有基线。

**消融结论**：
- 术语解释器（TI）和隐喻分析师（MA）贡献最大（移除 TI 导致 SARI 从 40.37 降至 38.75，移除 MA 降至 36.43）。
- 管道式通信略优于同步通信；迭代重建略优于直接重建。
- 校对员和文章架构师虽略微降低自动指标（因增加细节），但显著提升人工评价中的连贯性和忠实度。
- 基于 GPT-3.5 的版本成本更低且性能接近 GPT-4 版本，是性价比更优选择。

## 相关工作脉络
1. **KIS（Keep it Simple, Laban et al., 2021）**：多段落无监督简化方法，作为传统基线，AgentSimp 在 SARI 等指标上大幅超越。
2. **BART-SWIPE（Laban et al., 2023）**：基于 Wikipedia 简化数据集的微调模型，适合较短文档；AgentSimp 无需微调即可在长文档上取得更好效果。
3. **PG（Cripwell et al., 2023）**：计划引导的句子级简化方法，通过预测操作作为控制 token；AgentSimp 通过多角色协作实现细粒度简化，而非依赖操作预测。
4. **Sun et al. (2023a)**：连续预训练策略用于句子级简化，但不适合长文本；AgentSimp 通过分段-重建策略天然适配长文档。
5. **BooookScore（Chang et al., 2024）**：书籍级摘要方法，涉及 chunking 和重建策略，但目标是摘要而非简化；AgentSimp 借鉴了其重建思路但面向简化任务。
6. **Agent-Verse（Chen et al., 2024）**：通用多智能体框架；AgentSimp 明确说明通用框架不适配简化任务，因此设计了针对简化的定制化角色与流程。
7. **Simsum（Blinova et al., 2023）**：先摘要再简化两阶段方法；AgentSimp 不依赖摘要作为中间步骤，而是并行多视角简化后重建。

## 局限性与未来方向
1. **非通用框架**：AgentSimp 仅适用于文档简化任务，不能直接迁移至其他 NLP 任务（论文自述限制）。
2. **数据集匮乏且风格单一**：现有并行简化数据集数量有限，且写作风格同质化；未来需构建更多样化的并行文档简化数据集。
3. **人工评测主观性**：受限于成本和时间，人工评测仅 6 位非英语母语研究生参与、30 篇文档，样本量偏小，未来需扩大评测规模。
4. **FKGL 指标偏差**：因简化过程中加入更多解释性内容，FKGL 分数偏高，作者承认该指标不适用于简化任务评估，仅作为参考。
5. **未探索自监督/强化学习**：当前完全基于 LLM prompt，未来可探索结合 RL 优化策略。

## 研究启发与可借鉴点
1. **多角色专业化设计范式**：将简化任务拆解为语义简化、隐喻解析、术语解释等独立子任务，各由专责 Agent 完成——此范式可迁移至翻译、摘要、改写等其他文本生成任务。
2. **通信策略与重建策略的组合搜索**：系统性对比管道式/同步通信和直接/迭代重建四种组合，为多智能体工作流设计提供了可复用的实验框架。
3. **避免 LLM 单遍生成偏向摘要**：通过分段处理和重建机制，有效规避了 GPT 类模型在长文本简化任务中"偷懒"生成摘要的问题，该策略对长文本任务具有普适参考价值。
4. **跨领域泛化验证**：同时测试通用新闻（Newsela）和文学小说（GuoFeng），证明了框架在不同文体下的鲁棒性——这种跨领域验证设计值得借鉴。
5. **低成本替代方案**：GPT-3.5 版本的 AgentSimp 在保留大部分性能的同时显著降低成本，为实际部署提供了可行的工程路径。

## 关键术语表
**Text Simplification（文本简化）**：通过降低词汇、句法复杂度使文本更易理解，同时保持原意不变的自然语言处理任务。
**Document-level Simplification（文档级简化）**：面向整篇文档（而非单句或词）的简化任务，需同时考虑局部简化与全局连贯性。
**Multi-Agent System（多智能体系统）**：由多个具有不同职责的 AI 智能体协同完成复杂任务的方法框架，模拟人类团队协作。
**SARI（System Aspect for Assessment of Ranking Inputs）**：基于 n-gram 编辑操作的简化质量评估指标，衡量被添加、删除、保留词汇的效果。
**D-SARI（Document-level SARI）**：针对文档级别简化场景改进的 SARI 变体，对三种编辑操作施加惩罚。
**BARTScore（BART-S）**：基于预训练 BART 模型的文本生成质量评估指标，衡量生成文本与参考文本的似然度。
**FKGL（Flesch-Kincaid Grade Level）**：衡量英文文本阅读难度级别的指标，分数越低表示文本越简单。
**Pipeline-style Communication（管道式通信）**：智能体按顺序串行处理任务，前一个智能体的输出作为下一个智能体的输入。
**Synchronous Communication（同步通信）**：多个智能体并行处理同一输入，结果由整合器合并。

## 可复现要素
- **数据集**：Newsela（公开）、GuoFeng Webnovel（公开，使用英文版）；论文采样了各数据集的测试子集，原始数据集可公开获取。
- **代码/权重**：**论文未提及代码开源**。
- **关键超参**：Temperature = 0.6；Llama-2/Mistral 微调使用 LLaMA-Factory，LoRA，learning rate 5e-5，30 epochs，batch_size 2，gradient_accumulation 4，warmup_ratio 0.1。
- **模型调用**：GPT-3.5-turbo-0125（16,385 tokens）、GPT-4-0125-preview（128,000 tokens）、Llama-3-70b（8,000 tokens）、Mistral-7b、Mixtral-8x7B。
