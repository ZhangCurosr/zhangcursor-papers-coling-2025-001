---
title: "LLMs-Know-What-They-Need-Leveraging-a-Missing-Information-Gu"
source: https://aclanthology.org/2025.coling-main.163.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:41:46"
field: "检索增强生成与复杂推理"
keywords: ["RAG", "检索增强生成", "多跳问答", "缺失信息", "LLM推理"]
innovations: ["验证LLM识别缺失信息的能力并用于引导多跳检索", "句子级重排序过滤替代摘要压缩降低token消耗", "参数知识兜底机制提升检索失败时的鲁棒性"]
benchmarks: ["WikiHop", "HotpotQA", "Musique", "NQ", "TriviaQA", "StrategyQA"]
---

# 论文速读：LLMs-Know-What-They-Need-Leveraging-a-Missing-Information-Gu

## 一句话总结
本文验证了LLM具备从检索文档中提取有用信息并识别"缺失信息"的能力，据此提出MIGRES框架——利用缺失信息生成定向查询以引导后续多跳检索，并结合句子级重排序过滤与LLM参数知识补充，显著提升复杂多跳问答的RAG效果。

## 研究问题与动机
1. **多跳查询的检索困境**：复杂多跳问题（如"电影《Oh Billy, Behave》导演的出生地"）所需的中间信息未在查询中显式给出，传统RAG难以直接检索到相关文档。
2. **文档噪声干扰**：从海量候选中检索到的文档常包含大量无关内容，降低信息提取质量并增加token消耗。
3. **现有CoT方法的局限**：传统思维链（CoT）依赖任务特定的演示示例，且在推理过程中容易产生幻觉，缺乏对知识缺口的动态感知。
4. **LLM能力的再发现**：受人类推理启发——获取线索后逐步搜索缺失信息，作者验证LLM是否具备类似的"已知-缺失"判断能力。

## 核心贡献（创新点）
1. **实验验证LLM的双能力**：首次在多篇实证中证明LLM在零样本下可从文档中提取有用信息（精度90.6%）并精准识别缺失信息（准确率95.6%），为后续框架设计奠定基础。
2. **MIGRES缺失信息引导范式**：提出Retrieve-Extraction-Solving循环，用缺失信息生成单跳子查询驱动多轮检索，区别于CoT的事先生成全部子问题或ReAct的盲目追问。
3. **句子级重排序过滤策略**：将段落拆分为句子后基于相关性打分过滤噪声，相比SUMM/SNIPPET等摘要方法无需额外LLM调用即可降低外部知识token消耗。
4. **LLM参数知识兜底机制**：当检索无果时，提示LLM直接生成相关参数知识作为补充，提升迭代效率与鲁棒性。
5. **Memory模块防重复机制**：记录历史检索知识与查询，避免生成重复子查询和引入hard negative passages。

## 方法详解
**MIGRES包含四个核心模块：**

1. **Main Module（主控）**：接收查询q与已知信息集I，判断是否能回答；若能则输出答案a与解释E；否则输出"unanswerable"及缺失信息$I_{miss}$。

2. **Retrieval Module（检索）**：
   - Query Generator：基于$I_{miss}$生成不超过3个新子查询$[q_{t+1}^1,...,q_{t+1}^m]$
   - Retriever：使用BM25（k1=0.9, b=0.4）检索外部知识K
   - Knowledge Filter：两级过滤
     - 段落级：用BGE-reranker-base计算相关性分数，低于阈值$\delta$的直接丢弃
     - 句子级：用NLTK分句后逐句打分过滤，若整段相关性高于各句子则保留原段落
     - 空结果兜底：当无高相关文档时，提示LLM生成参数知识

3. **Leaf Module（信息提取）**：结合子查询与知识K，指令LLM提取带引用的有用信息$I'$，并用NLI模型（T5-xxl-nli）验证引用段落是否蕴含$I'$，过滤非蕴含内容。

4. **Memory Module（记忆）**：记录历史检索知识与生成的查询，防止重复生成与检索hard negative passages。

**迭代流程**：初始检索→信息提炼→缺失判断→子查询生成→再检索，循环直至回答或达到最大迭代步数T。

## 实验与结果
**数据集**：WikiHop、HotpotQA、Musique（多跳QA）；NQ、TriviaQA（开放域QA）；StrategyQA（常识推理）。

**评估指标**：EM（精确匹配）与Acc†（GPT-3.5-1106评估）。

**主要结果**（零样本，Table 3）：
- WikiHop：MIGRES EM=33.6 vs RERANK 28.5（+5.1）；Acc†=58.5 vs 56.0
- HotpotQA：MIGRES EM=38.0 vs ITER-RETGEN 44.1（少样本）；Acc†=66.6
- Musique：MIGRES EM=18.6 vs ReAct 23.4（少样本）；Acc†=32.8
- NQ：MIGRES EM=43.0 vs VANILLA 33.5（+9.5）；Acc†=80.0
- StrategyQA：MIGRES EM=73.4 vs 所有基线最优

**强检索基线**（Oracle知识池）：MIGRES†在WikiHop达到EM=40.1，表明部分错误源于检索缺失而非模型缺陷。

**少样本增强**（MIGRES*）：WikiHop EM提升至47.6-54.0，证明演示可引导生成更精准子查询。

**效率对比**（Table 4）：MIGRES平均每问题仅处理1.3-3.1个段落（总API调用8-12次），远低于ReAct/Self-Ask的14-16个段落。

**消融验证**：句子级过滤较摘要方法（SUMM/SNIPPET）在保持性能的同时大幅降低token消耗（WikiHop 733 vs 1249 tokens）。

## 相关工作脉络
1. **CoT分解方法**（Zhou et al., 2022; Press et al., 2023）：静态一次性分解全部子问题，缺乏对检索结果的动态感知；MIGRES则基于实际检索结果判断知识缺口，实现动态适应。
2. **迭代检索生成方法**（Shao et al., 2023; Feng et al., 2023）：简单拼接检索与生成内容，未明确识别每轮的知识缺口；MIGRES通过Missing Information生成精准子查询。
3. **Verify方法**（Sun et al., 2024; Yao et al., 2023）：缺乏信息验证步骤或仅验证句子级；MIGRES结合NLI模型验证提取信息是否被引用段落蕴含。
4. **Tree-of-Thought**（Kim et al., 2023）：树状分解耗时较长；MIGRES采用线性迭代且限制每轮最多3个子查询，效率更高。
5. **知识压缩方法**（Gao et al., 2023; Xu et al., 2024）：SUMM/SNIPPET需额外LLM调用；句子级重排序无需额外成本且能保留原始文本。

## 局限性与未来方向
1. **小模型验证不足**：实验主要在SOTA LLM上进行，仅用Llama-2-13B做了初步验证，更小模型的推理能力尚需系统评估。
2. **强检索器性能下降**：使用BGE密集检索器反而导致性能下降，因密集检索召回语义相关但信息无关的hard negative噪声。
3. **Musique表现不佳**：该数据集问题更晦涩模糊，检索器难以召回关键知识，导致EM仅18.6。
4. **检索策略依赖**：BM25优于混合检索，未来需探索更适合多跳推理的检索器设计。

## 研究启发与可借鉴点
1. **"缺失信息"作为检索导向信号**：将知识缺口显式建模并用于生成子查询，比CoT的事先分解更灵活，可迁移至其他需要多轮检索的任务（如对话系统、代码补全）。
2. **句子级细粒度过滤替代摘要压缩**：在降低token消耗的同时保留原始信息完整性，值得在长文档RAG中借鉴。
3. **NLI验证提取信息的可证性**：在Leaf Module引入 entailment 检验，可有效抑制幻觉，可复用于其他需要证据支撑的生成任务。
4. **参数知识兜底机制**：当外部检索失败时回退到LLM自身知识，兼顾开放域与闭源知识，为设计鲁棒RAG系统提供思路。
5. **零样本有效性**：MIGRES在零样本下即优于多数少样本基线，说明其设计本身已隐含了足够的推理结构，减少了对 demonstrations 的依赖。

## 关键术语表
**MIGRES**：Missing Information Guided Retrieve-Extraction-Solving，一种利用缺失信息引导检索-提取-求解循环的RAG框架。

**Multi-hop QA**：多跳问答，需要多个推理步骤和多次检索才能回答的问题类型。

**Retrieval-Augmented Generation (RAG)**：检索增强生成，通过外部文档检索补充LLM知识以提升生成质量的范式。

**Hard Negative Passage**：高相关性得分但不含有用信息的检索段落，对模型产生误导。

**NLI (Natural Language Inference)**：自然语言推理，判断前提文本是否蕴含假设文本的任务，本文用于验证提取信息的可信度。

**Sentence-Level Re-ranking**：句子级重排序，将段落拆分为句子后独立打分过滤的噪声消除策略。

**Acc†**：使用GPT-3.5-1106评估模型输出正确性的指标，弥补传统EM在多值答案场景下的不足。

## 可复现要素
- **数据集**：WikiHop、HotpotQA、Musique、NQ、TriviaQA、StrategyQA（公开数据集）
- **代码**：已开源，见 https://github.com/AdelWang/MIGRES
- **权重**：BGE-reranker-base（开源）、T5-xxl-nli（开源）、BM25（开源）
- **关键超参**：max iteration T=5（多跳QA）/3（ODQA）；relevance threshold δ=3.0/5.0/0.0；top-k passages=5；每查询检索50篇
