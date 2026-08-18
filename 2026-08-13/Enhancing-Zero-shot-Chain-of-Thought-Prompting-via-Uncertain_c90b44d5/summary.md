---
title: "Enhancing-Zero-shot-Chain-of-Thought-Prompting-via-Uncertain"
source: https://aclanthology.org/2025.coling-main.137.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:30:56"
field: "大语言模型提示工程"
keywords: ["Chain-of-Thought", "Zero-shot Prompting", "Uncertainty Estimation", "Demonstration Selection", "LLM Reasoning"]
innovations: ["融合温度扰动、触发词变化和题目改写三种扰动估计LLM不确定性", "基于不确定性均值和标准差定义多层次选择策略", "发现不确定性-准确率负相关并用其自动选择最优策略"]
benchmarks: ["GSM8K", "StrategyQA", "Logical Fallacy", "EPR"]
---

# 论文速读：Enhancing-Zero-shot-Chain-of-Thought-Prompting-via-Uncertain

## 一句话总结
本文提出了ZEUS（Zero-shot Uncertainty-based Selection）方法，通过结合温度扰动、触发词变化和题目改写三种扰动策略估计LLM的不确定性，进而自动选择最优示范题目构建少样本演示，在四个推理基准上显著提升了零样本CoT效果。

## 研究问题与动机
- **Manual-CoT依赖人工标注**：手工构建推理链需要大量专家精力，难以扩展。
- **Zero-Shot-CoT缺乏高质量演示**：仅靠触发词（如"Let's think step by step"）生成的演示质量不稳定，性能低于人工构建的Manual-CoT。
- **Auto-CoT聚类选择不精确**：现有自动方法（如Auto-CoT）通过无标签问题聚类选代表，但无法准确区分"有帮助"与"无效"的问题，导致选择质量有限。
- **已有不确定性方法灵敏度不足**：单纯温度扰动方法虽校准良好，但区分问题价值的能力较弱，尤其在复杂推理任务上表现不稳定。

## 核心贡献（创新点）
- **提出ZEUS不确定性估计框架**：融合温度扰动、触发词变化和题目改写三种扰动，通过预测熵量化LLM对每个问题的不确定程度，显著优于单一温度扰动方法。
- **定义多层次不确定性选择策略**：基于不确定性的均值$\mu$和标准差$\sigma$定义7种选择策略（Trivial到Very Hard），揭示不同能力模型的最优策略差异。
- **发现不确定性-准确率负相关规律**：证明通过Temp-Perb估计的不确定性与策略准确率呈强负相关，据此提出ZEUS(LU)——选择最低不确定性策略即可接近最优表现，无需手动调参。
- **无需模型参数访问的零样本方法**：整个过程仅需黑盒调用LLM生成响应，不依赖梯度或内部状态。

## 方法详解
**三阶段架构：**

1. **不确定性估计（Stage 1）**：
   - 对每个无标签问题$q_j$，生成$n \times t \times v$个响应池，其中$n$为温度采样数，$t$为触发词数（5种：空字符串、"Let's think step by step"、"Let's think about this logically step by step"、"Before we dive into the answer"、"Before answering the question, let's understand the input"），$v$为改写版本数。
   - 每个触发词在temperature=1下生成2个响应，共10个；改写后用trigger phrase在temperature=0下生成5个响应。总计15个响应/问题。
   - 统计每个唯一答案$y_j^c$的出现频率作为置信度：
     $$p(y_j^c | q_j) = \frac{1}{n}\sum_{l=1}^{n} \mathbf{1}(y_j^c = a_j^l)$$
   - 使用预测熵度量不确定性：
     $$u_j = -\sum_{c=1}^{C} p(y_j^c | q_j) \cdot \log(p(y_j^c | q_j))$$

2. **不确定性导向选择（Stage 2）**：
   - 计算无标签集$Q$的均值$\mu$和标准差$\sigma$。
   - 定义7种策略（表1），按不确定性范围$[u_{min}, u_{max}]$筛选问题子集$Q_s$。

3. **演示构建（Stage 3）**：
   - 采用Auto-CoT的多样性保障机制：用Sentence Transformers编码候选问题，K-Means++聚类后选取距质心最近的问题，附Zero-Shot-CoT生成的推理链构成演示集$D$。

## 实验与结果
- **数据集**：GSM8K（算术推理）、StrategyQA（隐式多跳推理）、Logical Fallacy（谬误检测）、EPR（认识论推理）。除GSM8K外，其余均按7:3划分无标签集和测试集。
- **模型**：GPT-4o、Mistral-7B、Phi-3-mini、text-davinci-002（GPT3-XL）、text-davinci-003（GPT3.5）。
- **基线**：Zero-Shot、Few-Shot、Zero-Shot-CoT、Manual-CoT、Auto-CoT。
- **关键结果**（Table 3）：
  - **GSM8K**：ZEUS(LU)在GPT-4o上达95.8%，超越Auto-CoT(94.2)和Zero-Shot-CoT(94.8)；Phi-3上89.9% vs Auto-CoT(87.6)。
  - **Fallacy**：ZEUS在多数模型上全面超越基线，GPT-4o达98.0%，GPT3.5达79.4%。
  - **StrategyQA**：ZEUS(HA)在GPT-4o上达82.2%，略超Manual-CoT(81.1)。
  - **EPR**：ZEUS surpasses Zero-Shot-CoT和Auto-CoT在所有模型上的表现。
- **灵敏度分析**（Figure 5）：ZEUS的置信度-准确率线性回归斜率系数接近理想值1，而Temp-Perb在Fallacy和EPR上灵敏度极低。
- **策略选择**：ZEUS(LU)（基于最低不确定性选策略）与ZEUS(HA)（基于实际最高准确率选策略）性能几乎一致，证明不确定性估计可有效指导策略选择。

## 相关工作脉络
- **Manual-CoT (Wei et al., 2022)**：依赖人工标注的推理链演示；本文无需人工标注。
- **Zero-Shot-CoT (Kojima et al., 2022)**：仅用触发词生成推理链；本文在此基础上通过不确定性筛选提升演示质量。
- **Auto-CoT (Zhang et al., 2022)**：无监督聚类选择演示；本文用不确定性替代聚类，更精准区分问题价值。
- **Active Learning for CoT (Diao et al., 2023/2024; Bayer & Reuter, 2024)**：利用不确定性选择待标注问题；本文场景无标注数据，完全零样本。
- **Self-Consistency (Wang et al., 2022)**：多采样取多数投票；本文关注如何选择演示而非如何聚合答案。
- **Perturbation-based Uncertainty (Gao et al., 2024; Kuhn et al., 2023)**：单用温度或语义扰动估计不确定性；本文融合多维扰动，提升灵敏度。

## 局限性与未来方向
- **策略选择需枚举探索**：当前需逐一尝试7种策略找到最优，计算成本高；未来可用贪婪搜索自动化。
- **未考虑数据集属性**：未分析数据集多样性、大小等因素对不确定性估计的影响。
- **改写依赖外部模型**：问题改写需调用GPT-4o等外部模型，增加成本和延迟。
- **未验证跨语言泛化**：仅测试英文数据集，对多语言场景的有效性未知。
- **模型能力二分法可能过于简化**：将模型分为"advanced"和"simpler"两类，边界可能模糊。

## 研究启发与可借鉴点
- **多维度扰动融合**：单一扰动（如温度）灵敏度不足，结合触发词变化和改写能更全面捕获LLM的不确定性模式。
- **不确定性-准确率负相关规律**：可作为自动化策略选择的可靠信号，减少手动调参需求。
- **模型能力自适应策略**：高性能模型适合Hard挑战性问题，低性能模型适合Easy问题——提示我们在构建演示时应根据目标模型能力匹配难度。
- **无需模型参数的黑盒方法**：适用于闭源API模型场景，拓展了CoT优化方法的适用边界。
- **预测熵作为不确定性度量**：比准确率或 logits 更稳定，适合衡量生成式模型的认知状态。

## 关键术语表
**Zero-shot CoT**：无需人工标注演示，仅用触发词引导LLM生成中间推理链的提示方法。
**ZEUS (Zero-shot Uncertainty-based Selection)**：本文提出的方法，通过不确定性估计自动筛选高质量演示问题。
**Predictive Entropy (PE)**：基于答案分布的信息熵，用于量化LLM对某问题的不确定程度。
**Temp-Perb (Temperature Perturbation)**：仅通过温度调整估计不确定性的基线方法，校准良好但灵敏度不足。
**Selection Strategy**：根据不确定性阈值（基于$\mu$和$\sigma$）筛选演示问题的规则，共7种难度等级。
**Monte Carlo Sample**：通过多次扰动采样近似LLM响应分布，用于估计不确定性。
**Active Learning for Prompting**：利用不确定性选择最有价值样本进行人工标注或演示构建的思路。
**Self-Consistency**：对同一问题多次采样推理链并取多数投票答案的方法。

## 可复现要素
- **数据集**：GSM8K、StrategyQA、Logical Fallacy、EPR均为公开数据集。
- **代码/权重**：论文未提及代码开源情况（URL来源为ACL Anthology，通常有代码附件链接，但本文未明确声明）。
- **关键超参**：k=8（除StrategyQA用k=6）、temperature=1用于不确定性估计、temperature=0用于最终推理、5种触发词、15个响应/问题。
- **模型**：GPT-4o、Phi-3-mini-4k-instruct、Mistral-7B-Instruct-v0.2、text-davinci-002、text-davinci-003。
