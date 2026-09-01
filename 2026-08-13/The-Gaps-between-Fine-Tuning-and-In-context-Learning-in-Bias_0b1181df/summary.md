---
title: "The-Gaps-between-Fine-Tuning-and-In-context-Learning-in-Bias"
source: https://aclanthology.org/2025.coling-main.187.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:00:46"
field: "语言模型公平性与偏见评估"
keywords: ["bias evaluation", "in-context learning", "fine-tuning", "debiasing", "catastrophic forgetting", "intrinsic/extrinsic bias"]
innovations: ["首次系统对比FT与ICL在偏见评估一致性和下游性能保持上的差异，证明ICL显著优于FT/PEFT", "提出通过输出状态余弦相似度量化去偏见方法对模型行为的扰动程度", "揭示ICL范式下预训练阶段内在偏见评估可更可靠地预测下游任务偏见水平"]
benchmarks: ["CrowS-Pairs", "StereoSet", "MBE", "BBQ", "BNLI", "WinoBias", "RACE", "ANLI", "OntoNotes v5.0"]
---

# 论文速读：The-Gaps-between-Fine-Tuning-and-In-context-Learning-in-Bias

## 一句话总结
本文系统对比了微调（FT）与上下文学习（ICL）在偏见评估和去偏见中的表现差异，发现 ICL 去偏见方法在内在偏见与外在偏见评分之间的相关性、以及下游任务性能保持方面均显著优于 FT/PEFT 方法。

## 研究问题与动机
- **FT 去偏分的"预测鸿沟"问题**：已有研究表明，微调后模型的内在偏见（预训练阶段评估）与外在偏见（下游任务评估）评分间相关性极低，导致预训练阶段的偏见评估结果难以可靠预测下游任务中的真实偏差水平。
- **FT 去偏分的灾难性遗忘**：无论是全参数微调（FT）还是参数高效微调（PEFT），去偏见操作均会显著降低下游任务性能（如 RACE、ANLI、OntoNotes），其根源在于参数更新改变了模型输出分布。
- **ICL 的去偏见效果缺乏系统性对比**：ICL 无需更新模型参数即可通过提示引导模型输出，理论上可能同时缓解上述两个问题，但此前未见系统分析。
- **研究目标**：填补这一空白，回答两个核心问题——（1）ICL 去偏见后内在/外在偏见相关性是否高于 FT？（2）ICL 去偏见的下游性能退化是否小于 FT？

## 核心贡献（创新点）
1. **首次系统性地对比 FT 与 ICL 在偏见评估一致性上的差异**：本文证明 ICL 去偏见后，内在偏见评估（CP/SS/MBE）与外在偏见评估（BBQ/BNLI/WB）的 Pearson 相关系数显著高于 FT 和 PEFT（如 CP→BBQ 从 FT 的 0.23 提升至 ICL 的 0.42，p<0.01）。
2. **首次量化比较 FT/PEFT/ICL 去偏见对下游任务性能的冲击**：通过控制去偏见强度一致的条件，本文表明 FT/PEFT 导致的下游性能下降显著高于 ICL，且 ICL 去偏见后的模型输出状态与原始模型余弦相似度更高。
3. **重新审视偏见评估的实践指导意义**：指出在 ICL 范式下，预训练阶段的内在偏见评估结果可更可靠地反映下游任务的偏见水平，为基于提示的去偏见实践提供了理论支撑。
4. **揭示了"参数更新幅度"是造成上述差距的根本机制**：通过测量去偏见前后模型输出状态的余弦相似度，证明 ICL 因不更新参数而保持了更稳定的输出分布，从而同时降低了灾难性遗忘和偏见评估不一致的风险。

## 方法详解
- **实验设置**：使用 LaMini 系列蒸馏小模型（LaMini-T5-61M/223M、LaMini-GPT-124M 等共 8 个模型），在四块 NVIDIA A100 GPU 上完成全部实验，训练和推理均在 24 小时内完成。
- **内在偏见评估数据集**：
  - **CrowS-Pairs (CP)**：通过比较正面刻板印象句和反面刻板印象句的似然度来评估社会偏见。
  - **StereoSet (SS)**：类似 CP，评估语言模型的刻板印象偏见。
  - **Multilingual Bias Evaluation (MBE)**：跨语言评估性别偏见；本文仅使用英语性别偏见子集。
- **外在偏见评估数据集（下游任务）**：
  - **BBQ**：问答任务中的社会偏见基准，判断模型在模糊 vs. 消歧输入下是否预测刻板印象答案。
  - **BNLI**：自然语言推理中的性别偏见基准，比较前提和假设中职业/性别词变化对预测的影响。
  - **WinoBias (WB)**：指代消解任务中的性别偏见基准。
- **FT/PEFT 去偏见方法**：
  - **CDA（Counterfactual Data Augmentation）**：将训练数据中的性别词替换生成反事实样本（如 "She is a nurse" → "He is a nurse"）以平衡数据。
  - **ALT（All-Layer Token-level debiasing）**：对所有层的 MLM 嵌入进行正交投影，消除与性别和职业术语相关的偏差方向。
  - **PEFT（Adapter）**：在 feed-forward 子层后插入单个 adapter 模块，降维比设为 16。
- **ICL 去偏见方法**：
  - **ZSD（Zero-Shot Debiasing）**：使用固定提示 "Please ensure that your answer is unbiased and does not rely on stereotypes."
  - **FSD（Few-Shot Debiasing）**：提供 16 个从下游任务数据集随机采样的反事实陈述示例。
- **关键度量**：
  - Pearson 相关系数 r：衡量内在偏见评分与下游任务偏见评分之间的一致性。
  - 下游任务性能下降比例：去偏见后相对原始模型在 RACE/ANLI/OntoNotes 上的 F1 变化。
  - 输出状态余弦相似度：cossim(e_i^o, e_i^f) 和 cossim(e_i^o, e_i^c)，其中 e_i^o 为原始模型、e_i^f 为 FT 去偏见模型、e_i^c 为 ICL 去偏见模型的最后层隐藏状态。
- **公平比较策略**：为确保可比性，将 FT 去偏分的去偏见效果控制在与 ZSD/FSD 去偏见效果相差 ±0.005 的范围内，再进行下游性能对比。

## 实验与结果
- **数据集**：内在偏见 — CP、SS、MBE；外在偏见 — BBQ、BNLI、WB；下游任务性能 — RACE（约 10 万道阅读理解题）、ANLI（约 17 万对 NLI 样本）、OntoNotes v5.0（1.3 万句指代消解）。
- **基线**：CDA+FT、ALT+FT、CDA+PEFT、ALT+PEFT、ZSD（ICL）、FSD（ICL）。
- **内在/外在偏见相关性（Table 1）**：
  - FT 相关性最低（CP→BBQ: 0.23，SS→BNLI: 0.15，MBE→WB: 0.12）；PEFT 次之（0.26、0.24、0.20）；ICL 最高（0.42†、0.44†、0.42†），且 FT vs. PEFT 仅在 2/9 个情况下有显著差异，而 PEFT vs. ICL 在全部 9 个情况下均显著（p<0.01）。
  - 关键数字：**ICL 相关性约为 FT 的 1.7–2.8 倍**。
- **下游性能下降（Figure 2 & Table 2）**：
  - 去偏见后模型输出状态余弦相似度：FT 方法为 0.51–0.66，PEFT 为 0.59–0.70，ICL 为 0.73–0.87，**ICL 明显高于 FT/PEFT**，表明 ICL 对模型行为扰动更小。
  - ZSD 在 ANLI 上的性能保持最优（0.83†），显著优于所有 FT/PEFT 方法。
  - 当以 ZSD 为对照标准等化去偏见程度时，FT/PEFT 的性能下降更大（因为 ZSD 本身去偏见幅度较小）。
- **最强结果**：ICL 方法的偏见相关性（最高 0.44，SS→BNLI）较 FT 提升约 193%（0.15→0.44），下游任务输出相似度（最高 0.87，ZSD on OntoNotes）较 FT 提升约 31%–71%。

## 相关工作脉络
1. **Kaneko et al. (2022a)**：发现了 FT 模型内在/外在偏见相关性极低的现象，本文将其推广至 ICL 场景并证明该现象在 ICL 下显著缓解。
2. **Goldfarb-Tarrant et al. (2021)**：首次报告预训练偏见指标与下游应用偏见间的相关性可忽略，本文在此基础上进一步区分 FT 和 ICL 两种范式下的不同表现。
3. **Oba et al. (2023)**：提出了 FSD ICL 去偏见方法，但未系统分析其与 FT 的差距；本文首次将其与 FT/PEFT 进行全面对比。
4. **Meade et al. (2022)**：实证比较了多种去偏见方法，发现 FT 去偏见普遍导致下游性能下降；本文补充了 ICL 方案作为更优替代。
5. **Xie & Lukasiewicz (2023)**：研究了 PEFT 去偏见的有效性；本文在此基础上进一步证明即使 PEFT 也难以接近 ICL 的效果。
6. **Schick et al. (2021)**：提出 Self-Diagnosis and Self-Debiasing 方法；本文关注更广泛的 ICL 提示策略（ZSD/FSD）的系统性对比。

## 局限性与未来方向
- **模型规模受限**：实验仅使用 LaMini 蒸馏小模型（最大 223M 参数），未在 LLaMA 等大型模型上验证，未来需用更大模型复现。
- **下游任务覆盖有限**：仅评估了 QA、NLI、指代消解三类任务，需扩展到更多存在偏见评估数据的任务类型。
- **偏见类型单一**：仅关注英语中的二元性别偏见，未涉及种族、宗教等社会偏见，也未考虑非二元性别。
- **语言单一**：仅使用英语（形态较简单），需扩展到多语言场景以验证结论的普适性。
- **ICL 提示设计**：ZSD 和 FSD 的提示效果可能依赖于特定设计，未来需探索更鲁棒的提示策略。

## 研究启发与可借鉴点
1. **ICL 作为去偏见的优选范式**：对于资源受限且需要保持下游性能的场景，ICL 去偏见（尤其是 FSD）比 FT/PEFT 更具实用性，可直接迁移到团队的 bias mitigation 研究管线中。
2. **"等化去偏见强度再比较下游性能"的实验设计**：本文通过将 FT 去偏见效果控制在与 ICL 方法 ±0.005 范围内再进行性能对比，消除了去偏见程度不一致带来的混淆，该控制变量的思路值得借鉴。
3. **输出状态余弦相似度作为灾难性遗忘的代理度量**：相比直接报告下游 F1 变化，使用 cossim(e^o, e^d) 提供更细粒度的模型行为扰动度量，可作为后续研究的补充评估手段。
4. **偏见评估相关性的范式依赖**：本文揭示了偏见评估结论高度依赖于使用 FT 还是 ICL，提醒我们在引用先前基于 FT 的结论时需谨慎，未来研究应明确标注所采用的推理范式。
5. **小模型快速验证大型方法的可行性**：使用 LaMini 系列小模型在单天内完成全部实验，证明了快速迭代验证的可行性，适合在早期研究阶段筛选方法方向。

## 关键术语表
**Intrinsic Bias（内在偏见）**：在预训练阶段通过语言模型 likelihood 比较（如 CP/SS/MBE）衡量的偏见程度，不涉及具体下游任务。
**Extrinsic Bias（外在偏见）**：在实际下游任务中（如 BBQ/BNLI/WB）评估的语言模型偏见程度，反映模型在真实应用场景中的偏差行为。
**Catastrophic Forgetting（灾难性遗忘）**：微调过程导致模型在下游任务上的性能系统性下降的现象。
**Zero-Shot Debiasing (ZSD)**：通过单一提示（如 "ensure your answer is unbiased"）引导模型输出去偏见结果的 ICL 方法。
**Few-Shot Debiasing (FSD)**：通过提供多个反事实示例提示来引导模型去偏见的 ICL 方法。
**Adapter（适配器）**：一种 PEFT 方法，在 Transformer 子层之间插入小型可训练模块（本文降维比设为 16）。
**Pearson Correlation r**：用于量化内在偏见评分与外在偏见评分之间线性相关程度的统计指标。
**Output State Cosine Similarity**：去偏见前后模型最后层隐藏向量的余弦相似度，用于衡量参数/提示扰动对模型输出的影响程度。

## 可复现要素
- **数据集**：CP、SS、MBE、BBQ、BNLI、WinoBias、RACE、ANLI、OntoNotes v5.0 — 均为公开数据集。
- **代码/模型**：LaMini 模型为开源蒸馏模型；实验使用 HuggingFace Transformers 库实现；论文未提供独立代码仓库声明。
- **关键超参**：Adapter 降维比 = 16；FSD 使用 16 个随机采样示例；去偏见强度等化容差 = ±0.005；硬件为 4× NVIDIA A100 GPU。
- **模型列表**：LaMini-T5-61M、LaMini-T5-223M、LaMini-GPT-124M、LaMini-Cerebras-111M、LaMini-Cerebras-256M、LaMini-Flan-T5-77M、LaMini-Flan-T5-248M、LaMini-Neo-125M。
