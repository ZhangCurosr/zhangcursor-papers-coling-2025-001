---
title: "Improving-Explainable-Fact-Checking-with-Claim-Evidence-Corr"
source: https://aclanthology.org/2025.coling-main.108.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:34:22"
field: "可解释自然语言处理"
keywords: ["explainable fact-checking", "claim-evidence correlation", "large language models", "CorFEVER", "FEVER", "LLM reasoning"]
innovations: ["提出两阶段相关性推理框架CorXFact，显式建模声明-证据相关性以提升核查准确性与可解释性", "构建CorFEVER测试集，同步评测相关性识别与核查性能，覆盖封闭域与现实世界场景", "揭示无关证据对小型LLM的负面影响，并提供有效的证据过滤与多任务混合微调策略"]
benchmarks: ["FEVER", "CorFEVER (closed-domain)", "CorFEVER (real-world)"]
---

# 论文速读：Improving-Explainable-Fact-Checking-with-Claim-Evidence-Corr

## 一句话总结
本文提出 **CorXFact** 框架，通过让大语言模型（LLM）先评估每个证据与声明之间的“相关性”（包括支持/反驳、程度及间接性），再综合权衡这些相关性进行核查判决并生成解释，从而模拟人类核查者的推理原则，提升事实核查的准确性与可解释性；同时构建了包含封闭域与现实世界场景的 **CorFEVER** 测试集用于全面评测。

## 研究问题与动机
- 现有基于LLM的事实核查系统虽能达到人类水平性能，但其决策过程仍是“黑盒”，生成的解释无法清晰揭示背后的推理原则。
- 人类核查者在验证声明时，会系统地评估每条证据与声明的相关性（支持/反驳强度与程度），并综合权衡所有证据的相关性后做出判决；当前LLM事实核查方法未能显式模拟这一原理。
- 现实世界场景中，检索到的证据往往多样化、非结构化且可能包含无关或矛盾信息，如何有效处理这类证据并保证核查的鲁棒性仍待探索。
- 缺乏专门用于评测LLM在“声明‑证据相关性识别”与“基于证据的核查”两方面能力的统一测试集。

## 核心贡献（创新点）
1. **提出CorXFact两阶段推理框架**：先由LLM识别声明‑证据相关性（relevance & degree），再由LLM基于这些相关性进行核查判决并生成透明解释；与直接进行核查的基线相比，本质区别在于显式引入并利用了“证据‑声明相关性”作为中间推理链。
2. **构建CorFEVER测试集**：包含封闭域（源自FEVER，人工标注相关性）与现实世界（从公开网站检索，模拟真实场景）两部分；与现有仅关注判决准确率的基准不同，该集同时评测相关性识别能力与核查性能。
3. **揭示无关证据对小型LLM的影响并验证框架有效性**：实验表明，去除Label 6（无关）证据可显著提升小型Llama‑2模型的性能，而GPT‑4因参数量大、知识丰富反而能更好利用该类信息；CorXFact在封闭域显著超越BEVERS等强基线。

## 方法详解
CorXFact采用链式思维（CoT）策略，分为两个阶段：
- **阶段一：声明‑证据相关性推理（correlation identifier）**  
  输入声明 $c$ 和证据集合 $E=\{e_1, e_2, ..., e_n\}$，LLM被指令以“声明作为证据上下文”的方式，逐条判断每条证据与声明的相关性，输出标签 $r_i$。标签体系包含7类（表1）：  
  Label 0：证据 Definitely supports；Label 1：Definitely contradicts；Label 2：Indirectly supports；Label 3：Indirectly contradicts；Label 4：Partially supports；Label 5：Partially contradicts；Label 6：No relation。  
  Few‑shot prompt 强制模型将声明视为证据的上下文，以处理证据中出现的代词或缩写。
- **阶段二：基于相关性的核查判决（fact checker）**  
  输入声明、证据及其对应的相关性标签，LLM综合权衡所有相关性，输出核查标签 $L \in \{\text{SUPPORTS, REFUTES, NOT ENOUGH INFO}\}$ 及简短的自然语言解释 $J$。Prompt 鼓励模型优先给出明确判决，避免轻易使用 NEI。
- **模型适配与微调**  
  框架可适配任意LLM。研究中使用了 GPT‑4、Llama‑2‑7b‑chat 以及基于 GPT‑4 生成的合成数据（12,142条核查样本、19,998条相关性识别样本）微调的 Llama‑2。微调策略比较表明，将两项能力混合训练（Fact‑check & Cor）效果最佳（Table 7）。

## 实验与结果
- **数据集**：FEVER dev set（3,000条，每类各1/3）、CorFEVER closed‑domain（1,000条，人工标注相关性）、CorFEVER real‑world（同1,000条声明，从Google/Bing检索证据，平均每条10条证据）。
- **基线**：BEVERS（传统三模块SoTA）、Llama‑2‑7b‑chat、Fine‑tuned Llama‑2、GPT‑4。
- **封闭域结果（Table 3）**：所有引入相关性的CorXFact变体均优于对应基线。最佳为 Fine‑tuned Llama‑2 + GPT‑4 提供相关性 → Acc = 96.50%，SUPPORT F₁ = 95.25%，REFUTE F₁ = 98.81%，NEI F₁ = 95.39%，超过同配置下 GPT‑4 自身（95.90%）。
- **现实世界结果（Table 4）**：整体性能低于封闭域，但CorXFact仍能提升多数设置。去除无关证据（Label 6）后，小型模型提升明显；GPT‑4 则反常地更能容忍无关证据。
- **FEVER dev set 结果（Table 5）**：CorXFact（Fine‑tuned Llama‑2）Acc = 93.57%，较 Fine‑tuned Llama‑2 基线（91.94%）提升1.63%，较BEVERS（81.84%）提升11.73%。错误纠正统计（Figure 2）显示约88%的REFUTE误判与33%的SUPPORT误判可被修正。

## 相关工作脉络
- **BEVERS**（DeHaven & Scott, 2023）：经典检索‑选证‑核查三模块 pipeline；本文与其对比显示LLM+相关性推理能进一步拉开差距。
- **QACheck / FACTIFY‑5WQA** 等基于问答的无证据核查方法：依赖LLM内部知识，易产生幻觉；本文聚焦有证据场景，通过显式相关性评估规避幻觉。
- **ProofVer、EXFAKT** 等早期可解释事实核查：侧重自然语言摘要或知识图谱解释；本文创新在于将“证据‑声明相关性”作为显式推理单元，使解释更贴近人类核查逻辑。
- **LLM‑based explainable fact‑checking**（Tan et al., 2023; Zhang & Gao, 2023; Kim et al., 2024）：同样使用LLM生成解释，但缺乏结构化相关性评估；本文方法提供了可量化、可评测的相关性识别能力。
- **SciFact‑open**（Wadden et al., 2022）：面向开放域科学声明核查；本文CorFEVER real‑world 部分亦模拟开放域场景，但更强调证据多样性与冲突处理。

## 局限性与未来方向
1. **幻觉问题未根除**：即使引入相关性推理，LLM仍可能产生错误相关性判断，尤其对罕见或间接关系（Label 2‑5）识别率较低。
2. **语言与领域局限**：仅评估英语声明；FEVER声明结构简洁，与社会媒体噪声声明存在差距，泛化至多语言及更嘈杂场景需进一步研究。
3. **解释性质仍为局部事后解释**：CorXFact的解释属于针对单一输入的事后归因（local post‑hoc），并非内置可证明的推理链，距离理想的“忠实解释”尚有距离。
4. **现实世界证据检索质量依赖外部API**：Google Custom Search 返回的片段可能存在语义模糊或截断，影响相关性判断。

## 研究启发与可借鉴点
- **相关性作为中间表示**：将“证据‑声明相关性”显式建模为可识别、可度量的标签，可迁移至其他需要证据权衡的任务（如法律推理、医学诊断）。
- **小规模模型 + 强模型协同**：用GPT‑4生成合成数据微调小型模型，使小型模型获得接近强模型的相关性识别能力，性价比高。
- **消融无关证据**：在开放域核查中，主动过滤或标记无关证据（Label 6）可显著提升小型模型的鲁棒性；该策略可用于预处理模块设计。
- **双能力混合微调优于链式微调**：实验表明将核查与相关性识别任务混合训练（而非顺序链式）效果更好，提示在多任务学习中任务耦合方式需谨慎设计。

## 关键术语表
- **CorXFact**：Correlation‑Enhanced Explainable Fact‑Checking，本文提出的两阶段LLM事实核查框架。
- **CorFEVER**：包含封闭域与现实世界子集的测试集，人工标注声明‑证据相关性，用于全面评测相关性识别与核查能力。
- **Claim‑Evidence Correlation**：声明与证据之间的相关性，定义7个标签刻画支持/反驳的确定性与程度。
- **Chain‑of‑Thought (CoT)**：通过分步推理提示激发LLM复杂推理能力的技术；本文用于引导相关性识别与核查两阶段。
- **BEVERS**：基于传统三模块（检索‑选证‑核查）的事实核查系统，本文的主要对比基线之一。
- **Post‑hoc Explanation**：事后解释，指模型做出预测后再生成的解释；CorXFact的解释属于此类。
- **LLM Fine‑tuning with LoRA**：采用低秩适配对LLama‑2进行微调，文中用于赋予其相关性识别与核查双重能力。
- **Real‑world Evidence**：从公开网站（Google/Bing）检索得到的多样化、非结构化证据，用于模拟真实核查场景。

## 可复现要素
- **数据集**：FEVER（公开）；CorFEVER（论文声明代码与数据将开源，见Footnote 1）。
- **代码/权重**：作者声明所有代码和数据将向研究社区发布。
- **关键超参**：LLM生成 cutoff = 256，temperature = 0.2，top_p = 0.9，top_k = 1，presence_penalty = 1.1，frequency_penalty = 2；微调使用LoRA，3 epochs。
