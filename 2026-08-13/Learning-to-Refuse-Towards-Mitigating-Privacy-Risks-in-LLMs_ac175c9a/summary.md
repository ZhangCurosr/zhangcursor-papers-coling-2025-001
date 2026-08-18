---
title: "Learning-to-Refuse-Towards-Mitigating-Privacy-Risks-in-LLMs"
source: https://aclanthology.org/2025.coling-main.114.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:42:40"
field: "大模型隐私安全"
keywords: ["Machine Unlearning", "Privacy Protection", "LLM", "Name-Aware Refusal", "Data Augmentation", "Personal Data"]
innovations: ["提出姓名感知的拒绝回答机制，帮助模型精准识别需保护的个体", "对称式对比数据增强扩展遗忘集与保留集分布，提升泛化能力", "首个面向真实人物隐私保护的 MU 评估基准 RETURN"]
benchmarks: ["RETURN"]
---

# 论文速读：Learning-to-Refuse-Towards-Mitigating-Privacy-Risks-in-LLMs

## 一句话总结
本文针对大语言模型（LLMs）意外 memorize 个人隐私数据的风险，提出首个面向真实场景的隐私保护评估基准 RETURN（包含 2,492 个真实名人的 QA 对），并设计了 Name-Aware Unlearning Framework (NAUF) 方法，使模型能够在遗忘特定目标人物信息的同时保持对其他无关人物的问答能力。实验表明 NAUF 的平均遗忘得分比最优基线高 5.65 分。

## 研究问题与动机
1. **核心问题**：如何在不重新训练的前提下，使 LLM 保护特定个体的隐私数据，满足"被遗忘权"（Right To Be Forgotten, RTBF）的需求。
2. **现有评测缺失**：已有 MU 方法评测多使用 QA 数据集、虚构传记或版权内容，缺乏面向真实世界中真实个体的隐私保护评估基准。
3. **梯度上升方法的局限**：基于梯度上升（GA）的方法对超参数敏感，且难以有效区分遗忘集与保留集，易导致模型崩溃或破坏保留集性能。
4. **RGD 方法的不足**：Relabeled Gradient Descent (RGD) 将遗忘集重标记为"我不知道"等无知回答，虽能有效遗忘目标数据，但会显著损害保留集性能，使模型拒绝回答本应正常回答的问题。

## 核心贡献（创新点）
1. **提出 RETURN 数据集**：首个包含 2,492 个真实名人及每人 20× QA 对的评估基准，用于评测真实场景下保护个人数据的 MU 方法；与以往基于虚构传记（如 Tofu）或通用 QA 的方法形成对比，覆盖真实人物与真实 Wiki 背景信息。
2. **提出 Name-Aware Refusal Answer 机制**：设计 100 条含目标姓名的拒绝回答模板，使模型学会区分"哪些人物的信息需要保护"；区别于 RGD 使用通用"我不知道"模板，该方法提供语义明确的姓名感知拒绝信号。
3. **提出 Contrastive Data Augmentation (CDA)**：通过对称的对比数据增强策略扩展遗忘集和保留集的分布——遗忘集通过替换名字生成负样本，保留集通过原始模型预测生成正样本——从而提升方法的泛化能力；相比直接小规模微调，显式扩展数据分布。
4. **系统性评测与分析**：在 LLaMA-3-8B-Instruct 上对比 GA、NPO、RGD、RDPO 四类遗忘方法与 GD/KLD 两类正则化，揭示 CDA 对不同正则化策略的互补作用，以及遗忘/保留比例对性能的影响规律。

## 方法详解
**整体框架 NAUF** 由两部分组成，目标是在遗忘集 $\mathcal{D}^F$ 上执行遗忘、在保留集 $\mathcal{D}^R$ 上施加正则化。

- **Name-Aware Refusal Answer**：
  - 为遗忘集中的每个样本 $(x, y)$ 重标记为姓名感知的拒绝回答 $y^{refuse}$，例如"I'm afraid I can't help with inquiries about [NAME]"。
  - 通过 GPT-4 构造 100 条拒绝回答模板 $\mathcal{D}^{refuse}$。
  - 在此基础上执行梯度下降（类似 RGD，但拒绝回答包含具体姓名）：
    $$\mathcal{L}_{NAUF}(\mathcal{D}^F, \mathcal{M}_u) = -\mathbb{E}_{(x,y)\sim\mathcal{D}^F, y^{refuse}\sim\mathcal{D}^{refuse}}[\log \mathcal{M}_u(y^{refuse}|x)]$$

- **Contrastive Data Augmentation (CDA)**：
  - **遗忘集增强**：从遗忘集或保留集中随机采样其他样本的 QA 对，将名字替换为目标个体名字后，使用姓名感知拒绝回答重标记，以此扩展遗忘集分布。
  - **保留集增强**：同样替换名字，但使用原始模型 $\mathcal{M}_o$ 对该问题的预测作为重标记答案，维持保留集知识。
  - 实验中遗忘集与保留集均扩充至原规模的 2 倍。

- **正则化策略**（可选）：
  - **GD 正则化**：$\mathcal{L}_{GD}(\mathcal{D}^R, \mathcal{M}_u) = -\mathbb{E}[\log \mathcal{M}_u(y|x)]$，直接最大化保留集正确回答的概率。
  - **KLD 正则化**：$\mathcal{L}_{KLD}(\mathcal{D}^R, \mathcal{M}_u, \mathcal{M}_o) = \mathbb{E}[KL(\mathcal{M}_o(y|x) \| \mathcal{M}_u(y|x))]$，最小化原始模型与遗忘后模型在保留集上的分布差异。
  - 计算预算约束：每 epoch 从保留集采样的样本数等于整个遗忘集大小。

- **训练设置**：AdamW，学习率 $1\text{e}{-5}$，batch size=32，5 epochs，2× NVIDIA A100-40GB。

## 实验与结果
- **数据集**：RETURN（2,492 个真实人物，每人 20× QA 对，基于 Wikipedia + GPT-4 生成）。
- **基线方法**：GA、NPO、RGD、RDPO（遗忘集）× GD、KLD 正则化；Oracle 为无遗忘的原模型。
- **评估指标**：Forget Score（遗忘得分）、Retain Score（保留得分）、Average Unlearning Score（二者均值）、下游任务准确率（WinoGrande、PIQA、LogiQA、LAMBADA、ARC-c）。
- **最强结果**：
  - NAUF + KLD 正则化：Forget=96.95，Retain=63.23，Avg=**80.09**，下游任务平均 60.80。
  - 较最佳基线 RGD+GD（Avg=72.83）提升 **5.65 分**。
  - 即使去除 CDA，NAUF（KLD）仍达 Avg=77.38，优于所有基线。
- **关键结论**：
  - 无正则化时除 GA 外所有方法平均得分约 50，说明正则化对保留下游能力至关重要。
  - GD 正则化在遗忘/保留间更均衡（得分差仅 5 分），KLD 正则化遗忘得分更高但保留得分较低（差 26 分）。
  - RGD 遗忘得分高但 Retain Score 极低（3.16），因"我不知道"过于泛化，破坏了保留集推理能力。
  - CDA 在 GD 下可提升 Forget Score 约 10 分，在 KLD 下可提升 Retain Score 约 4 分。
  - 随 epoch 增加，NAUF 持续改善，而 RGD 在 5→10 epoch 几乎停滞且下游能力下降。
  - 遗忘/保留比例增大（遗忘集占比提高）可提升保留得分，NAUF 在 20:80 时 Avg 达 82.43。

## 相关工作脉络
1. **Machine Unlearning (Cao & Yang, 2015; Bourtoule et al., 2021)**：MU 核心定义——在不重新训练的前提下消除 undesirable data 影响；本文将其应用于真实隐私保护场景。
2. **Gradient Ascent (GA)**（Jang et al., 2022）：直接在遗忘集上梯度上升以降低正确预测概率；本文指出 GA 对超参敏感且易导致模型崩溃（实验中也出现 NS——无意义输出）。
3. **Negative Preference Optimization (NPO)**（Zhang et al., 2024）：对齐思想驱动的遗忘方法，通过似然比约束实现遗忘；本文将其纳入对比基线，发现其 Forget Score 有限（27.41）。
4. **Tofu / Relabeled Gradient Descent (RGD)**（Maini et al., 2024）：使用"我不知道"类无知回答重标记遗忘集；本文指出该方法严重破坏保留集，并提出姓名感知拒绝回答以解决该问题。
5. **DPO/RDPO**（Rafailov et al., 2024）：基于偏好优化的方法；本文将其改编为 Relabeled DPO 作为基线，RDPO Forget Score 较低（约 25），体现偏好优化对遗忘任务的适配性有限。
6. **Differential Privacy**（Dwork et al., 2006; McMahan et al., 2017）：训练阶段引入噪声保障隐私；本文强调 DP 通常降低准确率且增加训练时间，MU 是更务实的"事后去记忆"方案。

## 局限性与未来方向
1. **数据集规模受限**：RETURN 仅 2,492 条记录，受限于 GPT-4 API 成本；更大规模数据集扩展是未来方向。
2. **细粒度保护缺失**：当前方法仅支持"全有或全无"式的个体级遗忘，无法自主判断哪些具体信息应被保护而哪些良性信息应保留；未来可探索细粒度隐私保护。
3. **评估仅限人名实体**：当前框架聚焦于真实个体（celebrity-level）隐私，未覆盖虚构角色、地理位置、受版权保护的作品等内容；未来可扩展至实体级或概念级。
4. **仅评测单一基座模型**：实验仅在 LLaMA-3-8B-Instruct 上进行，未验证跨模型泛化性。
5. **未考虑 adversarial 场景**：当前为 black-box API 查询假设，未评估对抗性提取攻击下的防御强度。

## 研究启发与可借鉴点
1. **姓名感知的拒绝回答设计**：相比通用"我不知道"，将目标姓名嵌入拒绝语（如"I can't answer questions about [NAME]"）可为模型提供精确的语义边界信号，这一设计可直接迁移至其他实体级遗忘任务。
2. **对称式对比数据增强策略**：对遗忘集做"名字替换+负标记"、对保留集做"名字替换+原模型预测标记"的双向增强，是一种低成本、不依赖额外标注的分布扩展技巧，适用于小样本 MU 场景。
3. **正则化策略的选择权衡**：GD 正则化在遗忘/保留间更均衡，KLD 正则化在遗忘强度上更高，实践中可按需求折中选择；这一结论为后续工作提供了明确的策略指导。
4. **RETURN 基准的可复用性**：数据集与评测流程对任何 LLM 开放，可作为隐私保护 MU 方法的通用评测平台；可进一步扩展至非英语名人、低资源场景。
5. **遗忘/保留比例的规律发现**：增大遗忘集比例可改善保留集性能（因固定 budget 下保留集采样相对更多）；这一发现对后续资源受限的 MU 实验设计具有参考价值。

## 关键术语表
**Machine Unlearning (MU)**：机器学习遗忘，指在不重新训练模型的前提下，消除模型对特定数据的记忆或影响的技术。
**Right To Be Forgotten (RTBF)**："被遗忘权"，欧盟 GDPR 赋予个人要求数据处理者删除其个人数据的权利。
**Forget Score**：遗忘得分，衡量模型在遗忘集上准确率相对原始模型的下降比例，值越高表示遗忘效果越好。
**Retain Score**：保留得分，衡量遗忘后模型在保留集上的准确率相对于原始模型的比例，值越高表示保留效果越好。
**Name-Aware Refusal Answer**：姓名感知拒绝回答，一种包含目标人物姓名的定制化拒绝语模板，帮助模型精准识别需保护的目标个体。
**Contrastive Data Augmentation (CDA)**：对比数据增强，通过对名字替换生成对比样本以扩展遗忘集与保留集的数据分布。
**Gradient Ascent (GA)**：梯度上升，直接最大化遗忘集上的损失以实现遗忘的最简单 MU 方法。
**Relabeled Gradient Descent (RGD)**：重标记梯度下降，将遗忘集标签替换为"我不知道"等无知回答后进行梯度下降的遗忘方法。

## 可复现要素
- **数据集**：RETURN（2,492 个真实人物 + 20× QA 对），基于 PopQA + Wikipedia + GPT-4 生成；论文未明确说明是否公开，但提供了完整的数据构建流程。
- **代码/权重**：论文未提及代码和权重是否开源。
- **关键超参**：学习率 $1\text{e}{-5}$，batch size=32，5 epochs，$\beta=0.1$（NPO/RDPO），优化器 AdamW，2× NVIDIA A100-40GB，DeepSpeed ZeRO-3 Offload。
- **基座模型**：LLaMA-3-8B-Instruct。
- **评估工具**：NLI 模型用于答案正确性判定（entailment/neutral=正确，contradiction=错误）。
