---
title: "MCS-SQL-Leveraging-Multiple-Prompts-and-Multiple-Choice-Sele"
source: https://aclanthology.org/2025.coling-main.24.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:43:55"
field: "Text-to-SQL与自然语言接口"
keywords: ["Text-to-SQL", "In-Context Learning", "Prompt Engineering", "Schema Linking", "Large Language Models", "Multiple-Choice Selection"]
innovations: ["多提示Schema Linking：通过随机打乱顺序生成多个prompt并取并集，显著提升链接召回率", "双策略动态Few-Shot选择：融合问题相似度和masked问题相似度生成多个SQL生成prompt", "置信度过滤+LLM多选决策：先按执行一致性过滤候选再用LLM选择，优于多数投票"]
benchmarks: ["BIRD", "Spider"]
---

# 论文速读：MCS-SQL-Leveraging-Multiple-Prompts-and-Multiple-Choice-Sele

## 一句话总结
本文提出MCS-SQL方法，通过多提示策略（multiple prompts）在schema linking、SQL生成和选择三个阶段探索更广阔的搜索空间，并利用LLM进行多选决策（Multiple-Choice Selection）聚合候选查询；在BIRD和Spider基准上均取得ICL方法的最优性能，BIRD达到EX 65.5%（提升5.9%）、VES 71.4%，Spider达到EX 89.6%。

## 研究问题与动机
- **ICL方法在复杂benchmark上仍显著落后于人类**：BIRD基准（复杂schema和查询）上ICL方法准确率不超过60%，而人类可达93.0%，差距明显。
- **LLM对prompt高度敏感**：即使是语义相同的prompt，因句子顺序、few-shot示例选择及排列顺序等不同，LLM输出会显著变化。
- **现有方法的局限性**：自一致性解码（self-consistency）、树思维（ToT）等方法仅依赖单一prompt探索推理路径，未能有效缓解LLM的prompt敏感性，搜索空间受限。
- **现有ICL文本到SQL方法依赖固定手工示例**：DIN-SQL、MAC-SQL等使用固定few-shot示例集，性能随示例选择波动大；DAIL-SQL虽动态选择但只聚焦单一最优策略。

## 核心贡献（创新点）
- **多提示Schema Linking**：通过随机打乱表/列顺序生成$p_t$和$p_c$个不同prompt，对每个prompt采样$n$次响应并取并集，以高召回率鲁棒地完成表链接和列链接；与固定schema输入方式的本质区别在于主动利用LLM的顺序敏感性来扩大候选覆盖。
- **双策略动态Few-Shot示例选择**：结合问题相似度和masked问题相似度两种检索策略生成$p_q$个不同prompt，覆盖更多SQL生成路径；与DAIL-SQL等单一相似度策略的本质区别在于并行融合多种检索信号以提升多样性。
- **基于置信度过滤与LLM多选决策（MCS）的选择机制**：先将候选SQL按执行结果分组、保留最快查询，并依据执行一致性计算confidence score进行阈值过滤，再由LLM在多选题形式中对候选进行排序选择；与多数投票（majority vote）的本质区别在于引入LLM的语义理解能力而非仅依赖频率统计。
- **系统化端到端流水线**：将上述三个步骤串联，形成统一的ICL文本到SQL框架，在BIRD基准上刷新ICL方法SOTA（EX +5.9%，VES +3.7%）。

## 方法详解
方法包含三个串联阶段：

1. **Schema Linking（表链接+列链接）**
   - **表链接**：给定DB schema和问题，要求LLM以JSON格式输出选择理由和表列表（受zero-shot-CoT启发）。为消除表顺序影响，随机打乱表顺序生成$p_t=3$个prompt，每个prompt用temperature=1.0采样$n=20$次响应，最终输出取所有$p_t \times n$个响应的并集（漏选代价高于多选的代价）。
   - **列链接**：仅在链接到的子schema上进行，要求LLM以`table_name.column_name`格式输出，同样随机打乱后生成$p_c=3$个prompt并取并集。
   
2. **Multiple SQL Generation**
   - **Few-shot示例选择**：
     - 基于问题相似度：选取训练集中与测试问题embedding最近的$k=20$个示例。
     - 基于masked问题相似度：用LLM将问题中的表名、列名、值替换为特殊token后计算相似度，减少schema特有内容干扰。
     - 融合两者生成$p_q=5$个不同prompt（含纯问题相似度、纯masked、混合组合）。
   - **SQL生成prompt**：包含few-shot示例（不含目标DB schema）、精简后的DB schema（仅linking得到的表和列）、样本表数据（CSV格式，帮助LLM理解列值分布）、自然语言问题，并要求LLM输出推理步骤和SQL（JSON格式）。每个prompt采样$n=20$次，共得到$p_q \times n = 100$个候选查询。
   
3. **Selection**
   - **候选过滤**：
     - 排除语法错误或超时的查询（timeout=180s）。
     - 按执行结果分组，每组仅保留执行时间最短的查询（公式2）。
     - 计算每个查询的confidence score：$confidence(q_i) = \frac{1}{N}\sum_{j=1}^{N}[\text{exec}(q_j)=\text{exec}(q_i)]$（公式1），剔除低于阈值$T=0.2$的查询（公式3）。
   - **Multiple-Choice Selection（MCS）**：将剩余候选按confidence降序排列，以多选题形式呈现给LLM，要求LLM给出选择理由和最终SQL；采样$n=20$次后通过多数投票确定最终查询。

## 实验与结果
- **数据集**：BIRD（95个真实大型数据库，12,751问答对，含外部知识推理需求）和Spider（200个数据库，10,181个问题，跨领域）。
- **评估指标**：执行准确率（EX）和BIRD特有的有效效率分数（VES）。
- **基线**：GPT-4 zero-shot、DIN-SQL、DAIL-SQL、MAC-SQL（均基于GPT-4）。
- **主要结果**：
  - **BIRD test**：MCS-SQL+GPT-4达到EX 65.5%、VES 71.4%，较前SOTA（MAC-SQL）提升+5.86%（EX）和+3.67%（VES），刷新ICL方法记录。
  - **Spider test**：达到EX 89.6%，较前SOTA（MAC-SQL）提升约3.0%（dev 89.5% vs 86.8%）。
- **消融实验关键发现**：
  - Schema linking贡献+2.1%（BIRD）、+3.3%（Spider）。
  - 加入sample table contents贡献+2.4%（BIRD）、+3.0%（Spider）。
  - 动态few-shot示例（masked相似度）贡献最大：+4.8%（BIRD）、+5.8%（Spider）。
  - 多提示vs单提示：BIRD上schema linking的recall从77.1%提升至89.8%（+12.7%），证明多prompt对复杂schema尤为重要。
  - MCS（含置信度过滤）较多数投票提升+0.6%（BIRD）、+0.3%（Spider）；若去掉置信度过滤则MCS效果显著下降（BIRD从63.4%降至62.5%）。

## 相关工作脉络
- **Self-Consistency / CoT相关**：Wang et al. (2022) 自一致性解码通过多数投票聚合单次prompt的多采样输出；本文利用多prompt扩展搜索空间并结合LLM语义选择，突破单一prompt限制。
- **DIN-SQL**（Pourreza & Rafiei, 2023a）：按问题复杂度分类并使用固定CoT示例；本文动态检索示例并采用多prompt策略，减少对人工示例的依赖。
- **DAIL-SQL**（Gao et al., 2023）：结合问题和查询相似度动态选择few-shot示例并引入自一致性；本文进一步引入masked问题相似度和多prompt并行策略，并增加LLM多选决策环节。
- **MAC-SQL**（Wang et al., 2023a）：多智能体分解子问题并顺序生成SQL；本文聚焦于单次生成的多路径探索与聚合，而非问题分解。
- **Schema Linking研究**：Lei et al. (2020)、ResDSQL等；本文强调利用LLM的顺序敏感性通过多prompt并集提升linking召回率。
- **Prompt敏感性研究**：Webson & Pavlick (2022)、Jang & Lukasiewicz (2023) 等指出LLM对句子顺序和结构敏感；本文将其转化为优势，系统性利用该特性。

## 局限性与未来方向
- **计算成本较高**：需要3-4次连续LLM API调用，且每个prompt需生成多个采样（$p_t \times n$、$p_c \times n$、$p_q \times n$），导致推理开销大。
- **LLM成本依赖**：当前方法高度依赖GPT-4等闭源大模型，未来可探索更轻量级模型或微调策略以降低成本。
- **Gold答案质量问题**：错误分析显示BIRD和Spider分别有62%和73%的"失败"案例实为gold答案不准确或评估方法局限所致，提示需要更精确的gold标准和评估体系。
- **未来方向**：开发更经济的prompt聚合方法、适配更小规模开源模型、改进评估协议以更好处理gold答案歧义。

## 研究启发与可借鉴点
- **利用模型敏感性而非抑制它**：将LLM对输入顺序的敏感性转化为搜索多样性工具（随机打乱+多prompt），这一思路可迁移至其他需要稳定输出的NLP任务。
- **Confidence-based预过滤提升LLM选择质量**：先通过执行一致性和时间筛选候选，再用LLM做最终决策，比直接让LLM从大量候选中选择效果更好；这种"粗筛+精判"的两阶段策略适用于其他候选生成场景。
- **Masked相似度检索**：将实体/ID类token遮蔽后再计算问题相似度，能更好地捕捉查询模式相似性而非表面词汇相似性，可迁移至代码生成、模板匹配等任务。
- **Sample table contents作为schema补充**：在prompt中嵌入少量真实行数据（CSV格式）帮助LLM理解列值分布，这一低成本技巧可普遍应用于需要理解数据内容的LLM应用。
- **并集操作保障高召回**：在schema linking等关键前置步骤中，采用并集而非交集来聚合多模型/多prompt输出，以牺牲少量冗余为代价换取高召回，对后续生成至关重要。

## 关键术语表
- **In-Context Learning (ICL)**：利用大语言模型在prompt中提供少量示例即可完成学习任务，无需微调参数。
- **Schema Linking**：从数据库模式（表、列）中识别并筛选出与自然语言问题相关的子集，是Text-to-SQL的关键前置步骤。
- **Execution Accuracy (EX)**：模型生成的SQL与gold SQL执行结果完全一致的比例，是Text-to-SQL的主流评估指标。
- **Valid Efficiency Score (VES)**：BIRD基准提出的指标，综合考虑查询正确性和执行效率（时间），无效查询得分为0。
- **Multiple-Choice Selection (MCS)**：将候选SQL以多选题形式呈现给LLM，要求模型给出选择理由并投票决定最终查询的方法。
- **Self-Consistency**：从LLM多次采样推理路径，通过多数投票选择最终答案的解码策略。
- **Masked Question Similarity**：先将问题中的表名、列名、值等特殊token遮蔽，再计算问题间embedding相似度的检索策略。
- **Confidence Score**：候选池中执行结果与某查询相同的查询所占比例，用于衡量该查询被多路径一致支持的程度。

## 可复现要素
- **数据集**：Spider和BIRD均为公开基准，可从官方渠道获取。
- **代码/权重**：论文未明确声明开源仓库地址，但提及详细prompt模板见附录B。
- **关键超参**：
  - $p_t = 3$（表链接prompt数），$p_c = 3$（列链接prompt数），$p_q = 5$（SQL生成prompt数）
  - $n = 20$（每个prompt的采样响应数）
  - $k = 20$（few-shot示例数量）
  - temperature = 1.0
  - 置信度阈值 $T = 0.2$
  - 执行超时时间 180s
- **模型**：GPT-4 8K（via Azure OpenAI API），嵌入模型text-embedding-ada-002，FAISS用于相似度搜索。
