---
title: "Efficient-Tool-Use-with-Chain-of-Abstraction-Reasoning"
source: https://aclanthology.org/2025.coling-main.185.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:30:14"
field: "大语言模型推理与工具使用"
keywords: ["Chain-of-Abstraction", "tool-augmented LLM", "multi-step reasoning", "tool use planning", "reasoning efficiency"]
innovations: ["提出 CoA 方法，通过抽象占位符解耦通用推理与领域知识获取", "设计基于开源自回归模型的数据重写流水线，无需闭源模型蒸馏", "实现推理与工具调用的跨样本并行化架构，提升推理效率约 1.4×"]
benchmarks: ["GSM8K", "ASDiv", "HotpotQA", "SVAMP", "MAWPS", "NaturalQuestions", "TriviaQA"]
---

# 论文速读：Efficient-Tool-Use-with-Chain-of-Abstraction-Reasoning

## 一句话总结
本文提出 **Chain-of-Abstraction (CoA)** 推理方法，通过让 LLM 先生成含抽象占位符的多步推理链、再由领域工具一次性填充具体知识，解耦了通用推理与领域知识获取，在数学推理和 Wiki QA 两个任务上分别取得约 7.5% 和 4.5% 的绝对准确率提升，同时推理速度提升约 1.4×。

---

## 研究问题与动机

1. **现有工具增强 LLM 在多步推理中规划能力弱**：如 Toolformer 等方法将文本生成与 API 调用交错（interleaved），缺乏对多步推理链中工具调用之间内在关联的显式建模，导致推理准确性不足。
2. **推理效率低下**：交错执行文本生成与 API 调用会引入大量"等待时间"——模型必须等待工具响应后才能继续解码，多步推理场景下这一问题尤其严重。
3. **领域知识漂移鲁棒性差**：传统 CoT 推理链中直接嵌入具体数值或事实，当问题涉及的知识分布发生变化时，模型难以泛化。
4. **缺乏通用的多步推理规划机制**：现有方法（如 ReAct、FireAct）依赖冗长的 thought-action-observation 循环，推理速度慢，且仍未解决工具调用的效率问题。

---

## 核心贡献（创新点）

1. **提出 Chain-of-Abstraction (CoA) 推理框架**：将 LLM 的推理过程解耦为两步——首先生成含抽象占位符（如 y₁, y₂）的多步推理链，再用领域工具一次性填充具体知识。与已有工作的本质区别在于：CoA 让模型聚焦于学习通用推理策略而非实例特定知识，从而提升分布外泛化能力。

2. **设计简单的微调数据自动构建流水线**：利用 LLaMa-70B 将现有 QA 数据的回答重写为含抽象变量的推理链，并通过领域工具（方程求解器/Wiki 搜索引擎）验证重写结果的正确性，无需从闭源模型蒸馏数据。与已有工作的本质区别在于：该流水线完全基于开源模型和规则验证，避免了对 GPT-4 等闭源模型的依赖。

3. **实现推理与工具调用的并行化架构**：CoA 允许模型在生成当前样本的抽象推理链的同时，对前一样本执行工具调用，通过流水线方式摊销总体推理时间。与已有工作的本质区别在于：不同于 Toolformer/ReAct 的串行 interleaving，CoA 实现了跨样本的解码-工具调用并行。

4. **在数学推理和 Wiki QA 两个异构领域均取得显著提升**：平均约 6% 绝对准确率提升，推理速度提升约 1.4×，且 Human Evaluation 显示推理错误率减少约 8%。与已有工作的本质区别在于：CoA 方法在保持高效率的同时兼顾了两个差异巨大的推理域。

---

## 方法详解

### 3.1 Chain-of-Abstraction (CoA) 推理框架

CoA 将多步推理分解为三个阶段：

**阶段一：生成抽象推理链**
- LLM 被微调以生成包含抽象占位符（如 `y₁, y₂, y₃`）的多步推理链
- 占位符不影响推理流，仅作为待填充的变量索引
- 示例（数学）：`[20 + 35 = y₁]`, `[90 - y₁ = y₂]`（注意同一中间结果 `y₁` 在多处出现时共享同一占位符，显式连接多个推理步骤）

**阶段二：工具填充（Reification）**
- 领域专用工具一次性填充所有占位符
- 数学域：方程求解器（SymPy）提取 CoA 中的推导方程，组建方程组求解各变量真实值
- Wiki QA 域：Wikipedia 搜索引擎（BM25 + Sentence-BERT 重排序）检索相关文章；SpaCy NER 提取桥接实体

**阶段三：基于填充后的推理链生成最终答案**

**关键设计原则：**
- 解耦通用推理与领域知识，使模型专注于学习泛化性强的推理策略
- 抽象推理链允许模型在不同样本间并行切换：生成下一样本的 CoA 时，可对当前样本执行工具调用
- 工具调用在完整推理链生成后统一执行，支持同一链内工具的并行调用

### 3.2 微调数据构建

构建流程（图 2）：
1. 收集开源 QA 数据集（GSM8K、ASDiv、HotpotQA）
2. 用 LLaMa-70B 将每个问题的金标准回答重写为含抽象变量的推理链
3. 标记金标准回答中与知识操作对应的片段（数学推导、Wikipedia 引用陈述）
4. 用领域工具验证重写的正确性：
   - 数学：方程求解器执行标记操作，验证抽象链能否被正确填充得到最终答案
   - Wiki QA：验证 WikiSearch 返回的 article 是否与金标准 article 匹配
5. 仅保留通过验证的 CoA 推理链作为微调数据

验证通过率：数学域约 76.6% 的 CoA 推理链通过方程求解器验证。

---

## 实验与结果

### 数据集
- **数学推理**：GSM8K（训练 7473 题，测试 1319 题）、ASDiv（单步，691 题）、SVAMP（OOD 测试 1000 题）、MAWPS（OOD 测试 2065 题，含 AddSub/SingleEQ/SingleOp/MultiArith 四子集）
- **Wiki QA**：HotpotQA Bridge QA（微调 8956 题，测试 5918 题）、Comparison QA（微调 5405 题，测试 1487 题）、WebQuestions/NaturalQuestions/TriviaQA（OOD 测试）

### 评估基线
- **CoT-FSP**：8-shot few-shot prompting
- **CoT-FT**：原始 CoT 数据微调
- **Toolformer**：在 CCNet 上微调的工具使用模型
- **Toolformer-Math/Wiki**：本文在领域 CoT 数据上重新微调的 Toolformer
- **FireAct**：基于 GPT-4 蒸馏的 ReAct 轨迹微调（Wiki QA 独有）
- **CoA (no Tool)**：用同 backbone 模型替代方程求解器的 ablation
- **PAL / DECLARATIVE**：基于代码的提示方法

### 主要结果

**数学推理（LLaMa-2-Chat-7B，表 4）：**
| 方法 | GSM8K | ASDiv | SVAMP | MAWPS-All |
|------|-------|-------|-------|-----------|
| CoT-FSP | 24.03 | 54.14 | 51.30 | 76.32 |
| CoT-FT | 35.41 | 59.00 | 46.90 | 73.37 |
| Toolformer | 23.65 | 50.85 | 48.80 | 70.85 |
| Toolformer-Math | 36.01 | 59.18 | 47.60 | 74.43 |
| **CoA** | **38.29** | **59.57** | **54.20** | **82.13** |

CoA 相对 CoT-FT 平均提升约 **+7.5%** 绝对准确率，OOD 泛化优势尤其明显（SVAMP +7.3%，MAWPS +8.76%）。

**数学推理（LLaMa-2-Chat-70B）：**
CoA 在 GSM8K 上达到 62.32%（最优），MAWPS-All 达到 91.91%（最优）。

**Wiki QA（LLaMa-2-Chat-7B，表 7）：**
| 方法 | HotpotQA-Both | WQ | NQ | TriviaQA |
|------|-------------|-----|-----|----------|
| CoT-FSP | 18.47 | 34.65 | 30.91 | 53.48 |
| FireAct | 26.20 | 36.02 | 35.87 | 52.96 |
| **CoA** | **28.22** | 35.97 | **38.67** | **57.90** |

CoA 在 Bridge QA 上达到 21.00%（优于所有基线），平均约 **+4.5%** 绝对准确率提升。

**推理效率（GSM8K，图 4）：**
- CoA 比 Toolformer 快约 **1.47×**（数学）和 **1.33×**（Wiki QA）
- 随推理步数增加，CoA 的耗时增长曲线更平缓，扩展性更好

**自一致性解码（表 6，16 次采样多数投票）：**
CoA 达到 40.79%，优于所有基线（CoT-FT: 39.12%，Toolformer-Math: 35.25%）。

**人类评估（表 5，200 个 GSM8K 样本）：**
| 方法 | 算术错误率 | 推理错误率 |
|------|-----------|-----------|
| CoT-FSP | 17.3% | 70.3% |
| CoT-FT | 25.2% | 67.8% |
| **CoA** | **0.0%** | **60.4%** |

CoA 将算术错误降至零（借助方程求解器），推理错误减少约 8%。

**Ablation（表 4）：**
- CoA (no Tool) 显著低于 CoA，证明专用工具的重要性
- 但 CoA (no Tool) 仍在 OOD 上优于所有基线，证明抽象推理链本身贡献了鲁棒性
- 移除 ASDiv 单步数据后，CoA 在单步数据集上性能显著下降（表 B），证明推理步数分布平衡的重要性

---

## 相关工作脉络

1. **Toolformer (Schick et al., 2023)**：通过微调让 LLM 学会调用外部 API，但采用串行 interleaving 方式，推理效率低，且未显式建模多步工具调用间的关联。CoA 在规划效率和推理准确性上均超越 Toolformer 及其领域微调版本。

2. **ReAct (Yao et al., 2023b) / FireAct (Chen et al., 2023)**：将 LLM 与工具集成到 thought-action-observation 闭环中，推理链冗长且仍为串行交互。CoA 通过抽象链解耦推理与工具调用，效率更高且无需 GPT-4 蒸馏数据。

3. **Program of Thoughts / PAL (Chen et al., 2022; Gao et al., 2023)**：提示 LLM 生成程序化推理并与代码解释器交互，但严重依赖闭源代码模型（Codex），仅适用于程序化算术推理。CoA 不依赖编码模型，且可泛化至开放域 QA。

4. **HuggingGPT / Chameleon / Meta-Tool**：关注多工具的高层序列规划以处理多领域混合任务。CoA 关注的是通用 CoT 推理链规划中对领域工具的意识，两者目标不同。

5. **ToolChain\* (Zhuang et al., 2023)**：结合工具使用规划与树搜索推理，适用于程序化任务。CoA 专注于自然语言推理链的抽象化，面向数学和 Wiki QA 等通用多步推理场景。

---

## 局限性与未来方向

1. **评估领域的覆盖有限**：仅在数学推理和 Wikipedia QA 两个领域验证，且以英语为主要测试语言，不能代表所有现实推理场景。
2. **全参数微调计算成本高**：目前实验基于完整 LLM 微调，未来可探索 LoRA 等高效微调方案。
3. **依赖 LLaMa-70B 进行数据重写**：数据构建阶段需要大型模型参与，虽可通过蒸馏降低依赖，但尚未研究。
4. **Wiki QA 需要两个模型**：一个生成抽象推理链，另一个基于填充后的链生成最终答案，增加了部署复杂度。

---

## 研究启发与可借鉴点

1. **解耦推理与知识获取的思路极具迁移价值**：将"推理策略学习"与"领域知识填充"分离的设计，可推广至代码生成、科学计算、法律推理等多个需要多步推理且依赖外部知识的领域。

2. **基于大型开源自回归模型的数据重写流水线**：用 LLaMa-70B 将现有 CoT 数据自动重写为抽象推理链，再通过领域工具验证正确性的方案，为大规模推理数据的低成本构建提供了可复用范式，无需依赖 GPT-4 蒸馏。

3. **跨样本的解码-工具并行化架构**：CoA 的流水线并行策略（生成下一样本 CoA 的同时执行当前样本工具调用）是提升 Agent 系统吞吐量的有效设计，可直接应用于多轮对话 Agent 和批处理推理系统。

4. **推理步数分布平衡对微调至关重要**：实验表明同时包含单步和多步推理的训练数据对模型鲁棒性不可忽视，这对后续工作构建微调数据集提供了重要的数据工程指导。

5. **抽象占位符显式连接多步推理**：在同一中间结果多处出现时使用相同占位符的设计，增强了多步推理链的结构化程度，可启发未来工作研究更丰富的推理图结构表示。

---

## 关键术语表

**Chain-of-Abstraction (CoA)**：一种 LLM 推理方法，先让模型生成含抽象占位符的多步推理链，再用领域工具一次性填充具体知识，从而解耦通用推理与领域知识获取。

**Reification**：将抽象推理链中的占位符替换为领域工具返回的具体知识值的过程。

**Interleaved execution**：文本生成与 API 调用交替执行的推理模式，如 Toolformer 和 ReAct 所采用的方式，会导致推理等待延迟。

**Bridge QA**：HotpotQA 中的一种问答类型，需要通过识别中间实体来连接问题与答案，要求至少两步连续的推理。

**Self-consistency decoding**：通过多次采样推理链并使用多数投票聚合结果的解码策略，可进一步提升推理准确性。

**Toolformer - Math/Wiki**：本文针对特定领域 CoT 数据重新微调的 Toolformer 版本，作为更公平的比较基线。

**GSM8K**：Grade School Math 8K，包含 7473 道小学水平多步数学应用题的标准数据集。

**HotpotQA**：包含 11.3 万多跳 QA 对的开放域问答数据集，分为 Bridge QA 和 Comparison QA 两种类型。

---

## 可复现要素

- **数据集**：GSM8K（公开）、ASDiv（公开）、HotpotQA（公开）、SVAMP（公开）、MAWPS（公开）、WebQuestions/NaturalQuestions/TriviaQA（公开）
- **代码/权重**：论文未提及代码开源声明
- **关键超参**：
  - 优化器：AdamW（β₁=0.9, β₂=0.95, ε=1e-8），weight decay=0.1
  - 学习率：7B 模型 2e-5，70B 模型 1e-5
  - Cosine learning rate scheduler，warm-up steps=10
  - Batch size=8
  - 数学微调：总 400 steps，7B 最佳 checkpoint 在 step 240，70B 在 step 200
  - Wiki QA 微调：总 500 steps，7B 最佳 checkpoint 在 step 450，70B 在 step 300
  - 自一致性解码：temperature=1.0, top-k=40, top-p=0.5, 16 次采样
