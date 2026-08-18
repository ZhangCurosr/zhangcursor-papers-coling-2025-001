---
title: "MiMoTable-A-Multi-scale-Spreadsheet-Benchmark-with-Meta-Oper"
source: https://aclanthology.org/2025.coling-main.173.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:44:38"
field: "表格推理与基准评测"
keywords: ["table reasoning", "benchmark", "large language models", "spreadsheet", "meta operations", "difficulty evaluation"]
innovations: ["提出多尺度真实电子表格基准MiMoTable（428表格/1719题，7领域中英双语）", "定义六类元操作（Lookup/Edit/Calculate/Compare/Visualize/Reasoning）及复合难度评分公式", "首次用统一元操作标准跨benchmark量化难度并验证性能-难度单调性"]
benchmarks: ["MiMoTable", "WikiTableQuestions", "WikiSQL"]
---

# 论文速读：MiMoTable-A-Multi-scale-Spreadsheet-Benchmark-with-Meta-Oper

## 一句话总结
论文提出了**MiMoTable**，一个包含428个真实场景多尺度电子表格（含中英文、简单/复杂表头、单/多sheet/多文件/多表格）的新基准，并设计了基于**六类元操作（Meta Operations）**的细粒度难度评估标准，全面评测LLM在复杂表格推理任务上的能力。

## 研究问题与动机
1. **现有基准表格类型单一**：大多数benchmark仅使用单表+简单表头（单行/单列），无法反映真实场景中复杂的表格结构（如分层表头、多sheet、多文件、单sheet内多表格并存）。
2. **现有基准无法衡量任务内难度差异**：当前benchmark按任务类型（TableQA、Table2Text等）分类，但同一任务下不同数据集的难度（如WikiSQL vs WikiTableQuestions）缺乏统一的量化标准。
3. **LLM在真实复杂表格上仍有明显局限**：尽管LLM在现有benchmark上表现优异，但面对更贴近现实的表格数据时，其理解与推理能力仍有较大提升空间。

## 核心贡献（创新点）
1. **构建多尺度电子表格基准MiMoTable**：收集428个真实Excel文件（中英双语、7个领域），覆盖四种维度变化（表头类型、sheet数量、文件数量、单sheet内表格数量），构建1719个(表格,问题,答案)三元组，弥补了现有benchmark在表格多样性上的不足。
2. **提出基于元操作的难度量化标准**：定义Lookup、Edit、Calculate、Compare、Visualize、Reasoning六类互斥元操作及三级难度评分（1-3级），引入复合难度公式量化每个问题及整个数据集的难度，为跨benchmark横向对比提供统一标尺。
3. **系统性评测与基准对标验证**：评测16个主流LLM在MiMoTable上的表现（最佳Claude-3.5-Sonnet仅77.4%），并将元操作标准应用于WikiSQL和WikiTableQuestions等已有基准，验证了"难度越高→模型准确率越低"的趋势，证明标准的有效性和泛化性。

## 方法详解
**1. 表格类型与难度分级**
- 四个维度：表头类型（Simple/Complex）、Sheet数量（Single/Multiple）、文件数量（Single/Multiple）、单Sheet内表格数量（Single/Multiple）
- 难度分为Simple（144个，33.6%）、Medium（139个，32.5%）、Hard（145个，33.9%），覆盖范围均衡

**2. 六类元操作定义**

| 元操作 | 说明 | 难度 |
|--------|------|------|
| Lookup | 定位特定目标位置 | 1 |
| Edit | 修改/删除/添加表格内容 | 1 |
| Calculate | 数值计算（求和、平均、最值等） | 2 |
| Compare | 比较两个及以上目标 | 2 |
| Visualize | 生成图表可视化 | 2 |
| Reasoning | 推断表格中未明确包含的信息 | 3 |

**3. 问题难度计算公式**
- 每个问题$ q_i $关联$ K_i $个元操作，对应难度序列$ S_{q_i} = [s_1, s_2, \ldots, s_{K_i}] $，$ s_k \in \{1,2,3\} $
- 记$ ms_{q_i} = \max(S_{q_i}) $为该问题涉及的最高难度等级
- 问题难度分：$ qs_i = ms_{q_i} + (\sum_{1}^{K_i} s_i - ms_{q_i}) / M_{ms_{q_i}} $，其中$ M_{ms} \in \{1, 6, 8\} $（对应$ ms=1,2,3 $）
- 数据集整体难度：$ ds = \frac{\sum_{1}^{N} qs_i}{N} $，MiMoTable整体难度为2.2，分数区间[1, 4]
- 难度分布：[1,2)占18.1%，[2,3)占66.9%，[3,4]占15.0%

**4. 数据构建流程**
- **表格收集**：中文来源（百度文库）、英文来源（Google搜索），覆盖建筑、金融、办公、教育、会计、电商、制造七领域；去噪+匿名化（GPT替换敏感信息）
- **问题生成**：GPT-4o将表格转为markdown后生成问题，要求包含不同元操作组合；针对多sheet/多文件生成跨表/跨文件问题；人工审核过滤
- **答案生成**：GPT-4o+Code Interpreter插件生成Python代码执行（处理编辑/可视化类问题）；多次推理取众数答案；10位专家双重标注，Cohen's Kappa=0.83

## 实验与结果
**数据集与基线**：MiMoTable（1719题）、WikiTableQuestions、WikiSQL；16个LLM（含开源/闭源/TableLlama）

**主要结果（MiMoTable整体）**：
- **最佳模型**：Claude-3.5-Sonnet，**77.4%**（English 79.0%，Chinese 76.2%）
- **次佳**：GPT-4o-CI（代码解释器）**69.2%**，GPT-4o-TXT（纯文本输入）**69.0%**
- 最差：TableLlama仅21.1%，Gemma-7B仅23.3%

**分维度分析**：
- **按表格难度**：所有模型在Simple（~81%）表现远好于Medium（~51-75%）和Hard（~49-72%）
- **按元操作**：Lookup任务最高（Claude 89.0%），Reasoning最低（Claude 63.3%）
- **中英差异**：Claude-3.5-Sonnet英文>中文；Qwen2-72B相反

**元操作有效性验证**：
- WikiSQL（难度1.5）准确率显著高于WikiTableQuestions（难度2.0）；MiMoTable-Simple（难度2.2）进一步下降
- 三个基准的性能-难度呈单调递减关系，证明元操作难度标准的有效性和跨benchmark通用性

**两种方法对比（GPT-4o）**：代码解释器（GPT-4o-CI）在简单/中等表格的计算类问题上优于纯文本方式；纯文本方式在困难表格及Lookup/Reasoning任务上表现更好（受context window限制较小）

## 相关工作脉络
1. **WikiTableQuestions / WikiSQL / FetaQA**：经典TableQA基准，但均采用Wikipedia来源的简单表头单表，缺乏真实场景复杂性
2. **HiTab / AIT-QA**：涉及分层表头的benchmark，但仅覆盖单表场景，未扩展至多sheet/多文件
3. **ToTTo / Text2Analysis**：分别聚焦Table2Text和混合任务，表格类型单一，无法统一衡量跨任务难度差异
4. **SPREADSHEETBENCH / WikiTableEdit**：涉及电子表格操作，但未引入元操作粒度分析与多尺度难度标准
5. **DAEval / DS-1000 / Text2Analysis**：面向高级数据分析任务，表格多为简化格式，缺乏真实多尺度电子表格
6. **MiMoTable定位**：首个同时覆盖四大任务类型（TableQA/Table2Text/Table Manipulation/Advanced Data Analysis）且具备多尺度真实电子表格与元操作难度标准的benchmark

## 局限性与未来方向
1. **未进行SFT实验**：仅做zero-shot/few-shot评测，未来可通过SFT分析各类元操作的提升空间
2. **提示词未针对模型优化**：除TableLlama外所有模型使用同一套prompt，未做逐模型调优
3. **超参数统一默认**：未对不同模型调整超参数，可能影响部分模型性能发挥
4. **缺乏长上下文场景**：未专门设计需要超长context的表格推理benchmark
5. **未来方向**：SFT细化分析各类元操作、开发长上下文表格推理基准、研究in-context learning在表格推理中的应用

## 研究启发与可借鉴点
1. **元操作分类法可直接迁移**：六类元操作（Lookup/Edit/Calculate/Compare/Visualize/Reasoning）及其难度分级可作为通用分析框架，应用于其他表格/数据推理benchmark的细粒度分析
2. **多尺度表格设计范式**：从表头、sheet、文件、子表格四个维度构建难度矩阵的思路，可推广至文档理解、报表分析等类似场景的benchmark构建
3. **GPT-4o+Code Interpreter混合答案生成**：自动生成初始答案+多次推理众数筛选+人工专家校验的流程，在代码/表格类任务中具有复现价值，可有效控制答案质量
4. **跨benchmark难度对标验证**：用同一套难度标准对多个已有benchmark打分并验证"难度-性能单调性"的实验设计，可作为新benchmark有效性的强有力支撑
5. **中英双语双域覆盖**：同时构建中英文版本（本文中文占57.6%问题量），为多语言表格推理研究提供数据基础

## 关键术语表
**Meta Operations（元操作）**：将表格推理问题分解为六类基本操作（Lookup/Edit/Calculate/Compare/Visualize/Reasoning），每类有对应难度等级（1-3级），问题可涉及多个元操作的组合
**多尺度电子表格（Multi-scale Spreadsheets）**：从表头类型、sheet数量、文件数量、单sheet内表格数量四个维度刻画表格复杂度的分类体系
**难度评分（Difficulty Score）**：基于元操作难度计算的量化指标（范围[1,4]），用于衡量单个问题或整个benchmark的推理难度
**Code Interpreter（代码解释器）**：GPT-4o的插件功能，允许模型生成并执行Python代码，适用于表格编辑、数值计算和图表生成类任务
**Tablellama**：针对表格任务微调的开源通用表格模型，在本文基准上表现最差（21.1%），反映专用模型在复杂真实场景下的局限

## 可复现要素
- **数据集**：1719个(表格,问题,答案)三元组，428个.xlsx文件（中英双语），已开源
- **代码**：https://github.com/jasonNLP/MiMoTable
- **模型**：16个LLM（Llama3.1/3、Qwen2/1.5、Mistral、Deepseek-Coder、Gemma开源系列；GPT-4o、Claude-3.5-Sonnet、Gemini-1.5-Pro闭源；TableLlama）
- **关键超参**：GPT-4o生成问题/答案；10位专家双重标注；人工过滤初始生成样本；Prompt模板见附录（Table 8/9/10）；评估使用GPT-4o判断正确性（开放式问题0-1分）
- **语言**：中英文混合（中文246表/1048题，英文182表/671题）
- **评估指标**：Accuracy（自动由GPT-4o评判）
