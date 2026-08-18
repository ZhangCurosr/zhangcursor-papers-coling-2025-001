---
title: "MAC-SQL-A-Multi-Agent-Collaborative-Framework-for-Text-to-SQ"
source: https://aclanthology.org/2025.coling-main.36.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:43:48"
field: "Text-to-SQL 与大语言模型"
keywords: ["Text-to-SQL", "多智能体协作", "大语言模型", "数据库查询", "链式推理", "代码生成"]
innovations: ["提出 Selector-Decomposer-Refiner 三智能体协作框架，引入外部 SQL 执行器反馈闭环实现自我纠错", "基于 Agent-Instruct 数据集（10K条）对 Code Llama 7B 进行多任务指令微调，使开源小模型逼近 GPT-4 性能", "在 BIRD holdout test 上以 MAC-SQL+GPT-4 达到 59.59% EX，刷新当时 SOTA"]
benchmarks: ["BIRD", "Spider"]
---

# 论文速读：MAC-SQL-A-Multi-Agent-Collaborative-Framework-for-Text-to-SQL

## 一句话总结
本文提出 MAC-SQL，一个基于大语言模型的多智能体协作框架，通过 Selector（数据库精简）、Decomposer（问题分解+CoT推理）和 Refiner（SQL纠错）三个智能体的协同工作，解决 Text-to-SQL 在大库和复杂问题上的性能下降问题；MAC-SQL+GPT-4 在 BIRD holdout test 上达到 59.59% 执行准确率，刷新当时 SOTA。

## 研究问题与动机
- **大库干扰**：现有 LLM-based Text-to-SQL 方法在面对"巨大"数据库时，输入大量无关 schema 会引入噪声，导致生成 SQL 偏离目标。
- **复杂多步推理不足**：用户问题常需多步推理（如嵌套子查询、跨表连接、数值计算），单次直接生成难以胜任，已有方法多集中于 prompt 策略微调，缺乏显式分解机制。
- **忽视外部工具与模型协作**：大多数方法仅依赖模型内部知识，未利用 SQL 执行器作为外部工具进行自我检验与纠错，缺乏容错闭环。
- **开源模型能力差距大**：开源模型在 Text-to-SQL 任务上远低于 GPT-4 等闭源模型，缺乏针对性的智能体指令微调数据与框架。

## 核心贡献（创新点）
1. **提出 MAC-SQL 多智能体协作框架**：将 Text-to-SQL 任务拆解为数据库精简（Selector）、问题分解+SQL生成（Decomposer）和 SQL 纠错（Refiner）三个协作环节，引入工具调用与迭代修正机制；与 DIN-SQL/DAIL-SQL 等方法仅依赖静态 prompt 的本质区别在于**多阶段动态协作与外部工具反馈闭环**。
2. **构建 Agent-Instruct 指令数据集并开源 SQL-Llama（7B）**：基于 BIRD 和 Spider 训练集，用 GPT-4 驱动 MAC-SQL 流程生成 10,000 条覆盖三个智能体任务的高质量指令数据，对 Code Llama 7B 进行监督微调；与已有开源模型的区别在于**专门针对多智能体 Text-to-SQL 任务进行指令微调**，而非通用代码或对话微调。
3. **在 BIRD benchmark 上达到当时 SOTA（59.59% EX）**：MAC-SQL+GPT-4 超越第二名 DAIL-SQL+GPT-4（57.41%）约 2.18 个百分点；开源版 SQL-Llama（43.94%）与 GPT-4 基线（46.35%）差距仅约 2.4%，展示了小模型通过多智能体框架可逼近大模型的潜力。
4. **系统性的消融与错误分析**：验证了 Selector/Decomposer/Refiner 各自在简单/中等/困难问题上的贡献，并分析 Gold Error、Semantic Correct、Schema Linking Error 等错误类型，揭示了现有数据集中标注质量问题的重要性。

## 方法详解
MAC-SQL 由三个核心智能体串联协作，算法流程如 Algorithm 1 所示：

- **Selector（数据库精简器）**：当数据库 schema prompt 的 token 数超过阈值（如 GPT-4-32k 下 >25k）时激活，根据用户问题 q 和外部知识 kg，从原始 schema S 中筛选出最小相关子集 S'={T',C'}，保留每张表至少 6 列，以 JSON 格式输出，减少无关信息干扰并控制上下文长度。函数形式：$S' = f_{selector}(q, S, k | \mathcal{M})$。

- **Decomposer（分解器，核心生成器）**：采用动态判断+Few-shot Chain-of-Thought（CoT）策略：先判断问题难度，简单问题直接生成 SQL；复杂问题则分解为 1~5 个子问题及其对应 SQL，逐步递进生成最终 SQL。生成过程建模为：
  $P_{\mathcal{M}}(y|q, S', k) = \prod_{j=1}^{L} P_{\mathcal{M}}(y^j|y^{<j}; q^j, S', k)$
  选用 CoT 而非 Least-to-Most 的原因：后者需迭代调用且需设定停止标准，效率低；CoT 可在一次生成中完成多步推理。

- **Refiner（精炼器/纠错器）**：调用外部 SQL 执行器对 Decomposer 生成的 SQL 进行执行，若返回语法错误、schema 幻觉（不存在的表/列名）或空结果，则 Refiner 接收错误信息 e 和原 SQL y'，指导 LLM 修正，最多迭代 3 轮。函数形式：$y = f_{refiner}(e, y', q, S', k | \mathcal{M})$。注意：若 SQL 执行无报错且非空结果，即使语义不符，Refiner 也不进一步修正（论文自述的局限）。

- **SQL-Llama 模型训练**：基于 Code Llama 7B，使用 Agent-Instruct 数据集（N=3 类任务：数据库精简、问题分解+SQL生成、SQL纠错）进行多任务监督微调，损失函数为：
  $\mathcal{L} = -\sum_{i=1}^{N} \mathbb{E}_{q, S^i, k, y^i \sim \mathcal{D}}[\log P(y^i|q, S^i, k; \mathcal{M})]$

## 实验与结果
- **数据集**：BIRD（95 个大库、33.4GB 数据、37 个专业领域，侧重真实大规模库）；Spider（200 个库、138 个领域、7000 训练对，侧重跨域泛化）。
- **评估指标**：执行准确率（EX）、精确匹配准确率（EM）、有效效率分（VES）。
- **主要结果（BIRD）**：
  - MAC-SQL+GPT-4：Dev EX=59.39%，Test EX=**59.59%**（SOTA）；超越 DIN-SQL+GPT-4（Dev 50.72%/Test 55.90%）和 DAIL-SQL+GPT-4（Dev 54.76%/Test 57.41%）。
  - MAC-SQL+SQL-Llama（7B）：Dev EX=**43.94%**，与 vanilla GPT-4（46.35%）仅差 2.41。
  - 加上 Oracle Schema（ ground truth 子库）后，MAC-SQL+SQL-Llama Dev EX 可达 51.43%，MAC-SQL+GPT-4 可达 70.28%。
- **主要结果（Spider）**：MAC-SQL+GPT-4 Dev EX=86.75%，Test EX=82.80%，超过 DAIL-SQL+GPT-4（Dev 84.40%/Test 86.60%）。
- **消融实验（BIRD Dev）**：去除 Selector（整体 -2.11%）、去除 Decomposer（-3.85%）、去除 Refiner（-4.63%），三组件在 Challenging 难度上贡献最大。
- **Few-shot 影响**：0→2 shot 在 BIRD 和 Spider 上均稳定提升；受限于 GPT-4 调用成本，最多评测到 2-shot（Spider dev 约 6M tokens，BIRD dev 约 10M tokens）。

## 相关工作脉络
1. **DIN-SQL（Pourreza & Rafiei, 2023）**：将 Text-to-SQL 拆分为多个子任务分别 prompt GPT-4；本文与其差异在于引入外部工具执行反馈闭环和三个独立智能体的动态协作，而非单链式子任务。
2. **DAIL-SQL（Gao et al., 2023）**：基于骨架相似度选 few-shot 示例并去除跨域知识；本文更侧重多智能体流水线而非单纯 prompt 工程优化。
3. **C3-SQL（Dong et al., 2023）**：针对 Spider 设计 schema linking 过滤和校准偏差 prompt；本文框架面向更复杂的 BIRD 大库场景，且具备自我纠错能力。
4. **AutoGPT/OpenAgents/AutoGen**：均为通用多智能体框架，不针对 Text-to-SQL；本文是首个专为 Text-to-SQL 设计的多智能体协作框架。
5. **StructGPT（Jiang et al., 2023）**：让 LLM 通过零样本方式推理结构化数据；本文引入显式分解+执行反馈，更贴合 SQL 生成的语法和执行约束。
6. **Code Llama（Rozière et al., 2023）**：本文的 SQL-Llama 基于 Code Llama 7B 微调，扩展其 Text-to-SQL 多智能体指令遵循能力。

## 局限性与未来方向
- **Prompt 非最优**：智能体 prompt 可能仍有优化空间，当前版本未必代表最佳设计。
- **Refiner 的语义盲区**：当 SQL 执行无报错且返回非空结果但与问题语义不符时，Refiner 不做修正，这类"语法正确但语义错误"的场景有待解决。
- **仅评测到 2-shot**：受 GPT-4 API 成本限制（dev 集百万级 token 消耗），未能探索更多 few-shot 示例的效果。
- **模型规模受限**：SQL-Llama 仅基于 7B 模型微调，作者认为使用更大模型有望进一步提升性能。
- **错误分析揭示数据质量问题**：Gold Error（BIRD 占 30%、Spider 占 22%）说明标注本身存在噪声，需更高质量的数据集支持。

## 研究启发与可借鉴点
1. **多智能体流水线设计可迁移**：Selector-Decomposer-Refiner 的"精简→分解→纠错"三段式结构可迁移到其他结构化输出任务（如 Text-to-SPARQL、Text-to-NL2BASH）。
2. **CoT + Few-shot 结合的实践细节**：动态难度判断（简单问题跳过分解）避免了不必要的计算开销，这一策略值得在复杂推理任务中借鉴。
3. **外部工具反馈闭环**：将 SQL 执行器作为工具供 Refiner 获取反馈，实现了"生成-检验-修正"的容错闭环，对代码生成、公式生成等需要验证的任务有参考价值。
4. **Oracle Schema 消融揭示了瓶颈**：加 Oracle Schema 后 SQL-Llama 从 43.94% 提升到 51.43%，说明数据库精简（Selector）是当前开源模型的主要短板，可作为后续改进的明确方向。
5. **开源指令数据集的价值**：Agent-Instruct 数据集（10,000 条）的公开为社区提供了可直接复用的多智能体 Text-to-SQL 微调数据，有利于推动开源模型在该领域的竞争。

## 关键术语表
- **Text-to-SQL**：将自然语言问题自动翻译为可执行的 SQL 查询语句的任务。
- **MAC-SQL**：Multi-Agent Collaborative SQL，本文提出的三智能体协作 Text-to-SQL 框架。
- **Selector**：负责从大数据库中筛选与问题相关的子 schema，减少无关信息干扰的智能体。
- **Decomposer**：将复杂问题分解为若干子问题并逐个子问题生成 SQL（CoT 方式）的核心智能体。
- **Refiner**：利用外部 SQL 执行器获取反馈，对语法/执行错误进行自动修正的智能体。
- **Execution Accuracy (EX)**：预测 SQL 与 gold SQL 执行结果完全一致的比例，是本文主要评估指标。
- **Valid Efficiency Score (VES)**：衡量有效 SQL（执行结果与 gold 一致）的查询效率的指标。
- **Agent-Instruct**：本文构建的 10,000 条多智能体指令微调数据集，覆盖数据库精简、问题分解、SQL 生成与纠错三类任务。

## 可复现要素
- **数据集**：BIRD（公开）、Spider（公开）；Agent-Instruct 微调数据集论文声明已开源。
- **代码/模型**：SQL-Llama（7B，基于 Code Llama 7B 微调）已开源；MAC-SQL 框架代码论文未明确说明开源状态，但提到会 open-source。
- **关键超参**：Selector 激活阈值：token 数 > 0.8 × max_sequence_length（GPT-4-32k 下 >25k）；Decomposer 最多分解 5 个子问题；Refiner 最多迭代 3 轮纠错；Few-shot 最多 2-shot。
- **基线模型**：GPT-4（32k context）、GPT-3.5-Turbo（16k context）、Code Llama 7B。
