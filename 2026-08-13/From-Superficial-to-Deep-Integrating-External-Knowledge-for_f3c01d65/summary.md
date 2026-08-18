---
title: "From-Superficial-to-Deep-Integrating-External-Knowledge-for"
source: https://aclanthology.org/2025.coling-main.55.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:32:47"
field: "自然语言生成-问题生成"
keywords: ["后续问题生成", "知识图谱", "大语言模型", "外部知识增强", "对话系统"]
innovations: ["提出Recognition-Selection-Fusion三阶段外部知识增强框架，通过在线KG构建和知识融合续写生成信息更丰富的后续问题", "设计双维节点选择策略（PageRank重要性+BERT语义相似度）从KG中选择最优外部知识节点", "创新知识融合操作：让LLM基于上下文续写Wiki知识，实现内外部知识的深度融合"]
benchmarks: ["FOLLOWUPQG"]
---

# 论文速读：From-Superficial-to-Deep-Integrating-External-Knowledge-for-Follow-up-Question-Generation-Using-Knowledge-Graph-and-LLM

## 一句话总结
提出一种三阶段外部知识增强的后续问题生成方法，通过在线构建知识图谱（KG）检索Wikipedia背景知识并结合LLM的知识融合操作，生成信息更丰富、认知复杂度更高且更贴近人类提问水平的后续问题。

## 研究问题与动机
- **核心问题**：现有后续问题生成方法生成的问题停留在"表面语境相关"层面，缺乏深度和创意，与人类利用常识知识进行高阶认知推理的能力存在显著差距。
- **现有方法不足**：传统方法（模板填充、PLM微调）依赖预设模式或仅利用上下文表面信息，生成的问题多为改写或泛泛而谈；即使LLM也缺乏主动引入外部知识的机制，难以实现类比、因果推理等高阶认知策略。
- **人类优势**：人类提问时可借助未明确提及的背景知识（如讨论"眼睛颜色"时联想到"黑色素""虹膜"），通过联想和类比引导对话深入，而机器仅能依赖上下文字面信息。
- **挑战**：如何在保持语境相关性的同时，有效引入外部常识知识并提升问题的信息量和认知复杂度。

## 核心贡献（创新点）
- **三阶段外部知识增强框架**：提出Recognition-Selection-Fusion三阶段流程，将多源知识（上下文关键词 + Wikipedia背景知识 + LLM内部知识）整合到后续问题生成中，相比仅依赖上下文的基线方法，显著提升问题的信息丰富度和深度。
- **在线KG构建的知识选择策略**：利用llmgraph工具在线构建以Wikipedia页面实体为中心的知识图谱，结合PageRank节点重要性（$I_i$）和BERT语义相似度（$S_i$）的复合得分（$R_i = I'_i + \beta \times S_i$）选择最优外部知识节点，使引入的背景知识既符合常识又与上下文高度相关。
- **知识融合续写操作**：设计创新的文本续写提示（text continuation task），要求LLM基于上下文对Wiki知识进行续写，既激发LLM内部世界知识的激活，又通过自然语言生成实现外部知识与上下文的深度融合，避免外部知识喧宾夺主。
- **系统验证与深入分析**：在FOLLOWUPQG数据集上进行大量实验，定量（MI、Distinct-n、TTR等）和定性（人类评估、案例分析）证明该方法在语境一致性、信息量、多样性和认知复杂度上全面优于PLM和LLM基线；消融实验验证了各模块的有效性。

## 方法详解
**整体框架**：三阶段流水线（Recognition → Selection → Fusion），输入为QA对，输出为后续问题。

**Stage 1: Recognition（识别）**
- LLM从历史QA对中抽取1个Topic（高层意图）和n个Keywords（细粒度细节），本文n=3。
- 迭代检索Wikipedia：先用Topic搜索页面集合C，再逐个用Keywords缩小范围，找到最匹配的页面（如有唯一匹配则提前终止）。
- **重排序模块**：用T5作为reranker，计算query $Q$（Topic+Keywords拼接）在页面实体定义$p_i$下的条件概率：
  $$P(Q|p_i) = \frac{1}{|Q|} \prod_t P(Q_t|Q_{<t}, p_i; \Theta)$$
  取概率最高的页面作为话题实体。

**Stage 2: Selection（选择）**
- 用开源工具llmgraph基于Wikipedia页面在线构建KG，以页面实体为中心节点。
- **节点重要性评分**：结合PageRank权重$w_i$和随机游走访问次数$n_i$（游走步数=100）：
  $$I_i = w_i \times n_i$$
- **语义相似度评分**：用all-MiniLM-L6-v2编码query $q$和实体定义$e_i$，计算余弦相似度：
  $$S_i = \frac{q \cdot e_i}{\|q\| \|e_i\|}$$
- **复合得分**（max-min归一化$I_i$后）：
  $$R_i = I'_i + \beta \times S_i, \quad \beta=1.0$$
  选择$R_i$最高的实体定义作为引入的Wiki知识。

**Stage 3: Fusion（融合）**
- **知识续写**：Prompt设计——"Given a QA pair: [Question], [Answer]. Please continue writing the following sentences with a few sentences based on the question-answer pair to reflect the association with it."，要求LLM续写Wiki知识，激活LLM内部常识并与上下文融合。
- **后续问题生成**：最终Prompt——"Given: [Question], [Answer], [Related knowledge]. Based on this, raise a follow-up question that is relevant and thoughtful."，输出最终后续问题。
- 所有LLM统一使用gpt-3.5-turbo，温度=1.0，通过vLLM加速推理。

## 实验与结果
- **数据集**：FOLLOWUPQG（来源于Reddit ELI5子版块），含3,790条三元组（初始问题、回答、后续问题），涵盖关联、因果推理等高阶认知技能。
- **评估指标**：Topic Consistency（LDA主题共现）、Mutual Information（MI，越低越好）、Distinct-1/2、TTR（类型-符号比）。
- **基线模型**：
  - PLMs：BART、T5、GPT-Neo（均在训练集上fine-tune）
  - LLMs：gpt-3.5-turbo、LLaMA3、Qwen2、ChatGLM4（标准prompt）
- **主要结果**（Table 1）：
  - **MI最低**：Ours = 0.7515（优于gpt-3.5-turbo的0.7677），表明生成的后续问题包含更多新信息。
  - **Distinct-1最高**：Ours = 33.84%（vs. gpt-3.5-turbo 31.73%，LLaMA3 31.63%）。
  - **TTR最高**：Ours = 97.08%（vs. gpt-3.5-turbo 96.65%）。
  - **注意**：PLMs的Topic Consistency更高（BART 62.11% vs. Ours 54.42%），但作者指出这是因为PLMs过拟合训练数据、倾向于复述原文，并非质量更好。
- **人类评估**（100个测试样本，5名英语熟练评审）（Table 2）：
  - **Complexity**：Ours = 0.42，显著优于所有基线（最高LLaMA3仅0.21），提升**至少18%**。
  - **Informativeness**：Ours = 1.98，优于gpt-3.5-turbo的1.66。
  - **Relevance**：所有模型均维持高水平（≈0.97-0.98）。
  - **用户偏好投票**（Figure 3）：Ours生成的问题最受用户青睐，最接近人类提问水平。
- **消融实验**（Table 3，语义距离分析）：
  - 去掉re-ranker：$dis(Wiki_k, q)$下降1.00%，$dis(Wiki_k, fq)$下降8.02%，说明重排序对知识相关性至关重要。
  - 随机选择KG节点：$dis(Wiki_k, q)$下降26.60%，$dis(Wiki_k, fq)$下降9.79%，验证重要性+相似度选择策略的必要性。
  - 去掉知识融合（直接用Wiki知识生成问题）：$dis(Wiki_k, fq)$激增10.73%，$dis(q, fq)$下降16.09%，说明续写操作能有效防止外部知识喧宾夺主，保持语境聚焦。
- **β参数分析**（Table 4）：$\beta=1$时BLEU-1/2和Perplexity最优，Topic Consistency在$\beta=1.5$略高，综合判定$\beta=1$为最佳，即节点重要性与语义相似度权重相等。
- **最强结果**：在信息量（MI最低）、多样性（Distinct-1/TTR最高）、认知复杂度（人类评估+18%）三项核心指标上全面最优，且用户偏好度最高。

## 相关工作脉络
- **传统问题生成**（Du et al., 2017; Pan et al., 2019）：答案从源文本中可直接获取，与本文"探索未知信息"的后续问题生成目标根本不同。
- **基于规则的后续问题生成**（Soni & Roberts, 2019; Oh et al., 2015）：模板填充限制问题类型多样性，无法生成个性化问题。
- **PLM-based方法**（Kumar & Joshi, 2017; Su et al., 2018; Ge et al., 2023）：Ge et al. (2023) 提出知识驱动框架，但仅选择知识实体-关系对，未利用图谱结构进行知识选择和融合。
- **LLM-based方法**（Meng et al., 2023）：发现PLM和LLM生成的问题在信息量和复杂度上均远落后于人类，但本文指出此前缺乏针对LLM的后续问题生成方法——本文通过外部知识注入弥补这一空白。
- **知识图谱增强生成**（Agrawal et al., 2024; Louis et al., 2024）：现有工作多用于QA或特定领域，本文首次将在线KG构建+LLM知识融合应用于开放式后续问题生成。
- **定位差异**：本文区别于纯prompt-based LLM方法（如直接让gpt-3.5-turbo生成）和纯PLM微调方法，核心创新在于**在线KG构建+双维节点选择+知识融合续写**的三段式设计，实现了"从表面语境到深度知识"的跨越。

## 局限性与未来方向
- **知识库局限性**：依赖Wikipedia作为外部知识源，在某些垂直领域（如法律、医疗）中准确性和覆盖度不足（论文Section 7明确提及）。
- **实时性瓶颈**：在线构建KG耗时较长，可能限制其在需要低延迟的对话系统中的应用（论文Section 7）。
- **未来方向**：如何平衡知识准确性与工作效率；探索更适合垂直领域的知识源；优化KG构建和查询的效率。

## 研究启发与可借鉴点
- **在线KG构建作为外部知识检索的增强手段**：传统RAG依赖向量检索，本文通过在线构建KG并利用图结构（PageRank+随机游走）进行节点重要性评估，为知识选择提供了更结构化的视角，可迁移到Open-Domain QA、知识增强对话等任务。
- **双维知识选择策略（重要性+语义相似度）**：复合得分$R_i = I'_i + \beta \times S_i$的设计兼顾了"常识显著性"和"语境相关性"，这种权衡机制可推广到其他需要外部知识筛选的场景。
- **知识融合续写（Text Continuation）操作**：通过让LLM续写外部知识而非直接拼接，既激活了LLM内部世界知识，又实现了知识的语境化融合，避免了"知识堆砌"问题——这一提示工程设计对任何涉及外部知识注入的生成任务均有参考价值。
- **PLM与LLM在后续问题生成中的表现差异**：实验发现PLM在Topic Consistency上优于LLM（因过拟合复述），但LLM在多样性上更强，这一发现对模型选型和混合架构设计有启发意义。
- **结合本团队方向的创新机会**：若团队从事垂直领域（如法律、医疗）对话系统，可将本文的KG构建+知识融合框架迁移到领域知识库（如Medical KG、Legal KG），并结合领域适配的LLM，探索"领域知识增强的主动问答"方向。

## 关键术语表
- **Follow-up Question Generation（后续问题生成）**：基于对话上下文生成能够引导对话深入、探索新信息的后续问题，其答案无法从前文直接获取。
- **Knowledge Graph（KG，知识图谱）**：以实体和关系构成的结构化知识表示，本文通过llmgraph工具在线构建，用于组织和检索外部背景知识。
- **Mutual Information（MI，互信息）**：衡量两个变量之间信息共享程度的指标，在本文中用于评估初始问题与后续问题之间的信息重叠，MI越低表示后续问题包含越多新信息。
- **Distinct-n / TTR（多样性指标）**：Distinct-n衡量n-gram的唯一比例，TTR衡量词类型与符号的比值，二者均用于评估生成文本的词汇多样性。
- **Topic Consistency（主题一致性）**：基于LDA主题模型计算上下文与生成问题之间的主题共现频率，衡量生成的后续问题与当前对话话题的关联程度。
- **Re-ranker（重排序器）**：用T5模型对搜索引擎返回的候选Wikipedia页面进行重排序，通过计算query在页面实体定义上的条件概率选择最相关页面。
- **Knowledge Fusion（知识融合）**：本文设计的文本续写操作，要求LLM基于上下文对Wiki知识进行续写，实现外部知识与LLM内部知识的深度融合。
- **llmgraph**：开源工具，利用LLM从Wikipedia页面自动构建知识图谱，本文用于在线KG构建。

## 可复现要素
- **数据集**：FOLLOWUPQG（来源于Reddit ELI5），3,790条样本，论文未明确说明是否公开，但引用了Meng et al. (2023)，可参考原论文获取。
- **代码/权重**：论文未提供开源代码；使用llmgraph开源工具；llm使用gpt-3.5-turbo（通过API调用，非本地权重）。
- **关键超参**：Keywords数量n=3；随机游走步数=100；嵌入模型all-MiniLM-L6-v2；β=1.0；温度=1.0；reranker使用T5。
- **硬件环境**：NVIDIA 4090 24GB GPU集群；推理加速使用vLLM。
