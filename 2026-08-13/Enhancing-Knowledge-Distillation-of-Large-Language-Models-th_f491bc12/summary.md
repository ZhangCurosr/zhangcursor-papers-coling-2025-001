---
title: "Enhancing-Knowledge-Distillation-of-Large-Language-Models-th"
source: https://aclanthology.org/2025.coling-main.169.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:40:50"
field: "大语言模型压缩与知识蒸馏"
keywords: ["知识蒸馏", "大语言模型", "多模态分布", "排序损失", "模型压缩", "Ranking Loss"]
innovations: ["提出基于Spearman秩相关系数的词级排序损失，优化师生模型top-k预测顺序一致性", "实验验证多模态分布对齐对下游性能的关键作用，并提供CR/MOR评估指标", "排序损失与现有多种蒸馏目标（KL/RKL/JSD/TVD/AKL）高度兼容，稳定提升性能"]
benchmarks: ["GSM8K", "Dolly-15k", "Xsum", "SlimPajama"]
---

# 论文速读：Enhancing-Knowledge-Distillation-of-Large-Language-Models-th

## 一句话总结
本文针对大语言模型知识蒸馏中多模态分布学习效率低的问题，提出了**基于排序损失的知识蒸馏方法（RLKD）**，通过优化师生模型在top-k预测上的顺序一致性，显著提升学生模型对多模态分布的学习能力，并在预训练及多个下游任务中取得显著性能提升。

## 研究问题与动机
1. **多模态分布学习困难**：LLM的概率分布通常具有多模态特性，包含多个潜在正确预测，现有KD方法难以有效帮助学生模型学习这种复杂分布。
2. **现有目标函数存在缺陷**：KL散度存在模式平均问题（mode-averaging），student模型倾向于学习过于平滑的分布；RKL虽然关注峰值预测，但容易导致概率过度集中在特定区间，遗漏其他峰值。
3. **缺乏实证验证**：虽有理论分析表明不同目标函数的优劣，但未能在真实任务中验证这些目标是否真正增强了学生对多模态分布的学习能力。
4. **峰值预测质量与性能强相关**：实验证明，模型峰值预测一致性越高，其下游任务性能越好，仅关注top-1预测是不够的。

## 核心贡献（创新点）
1. **提出词级排序损失（Word-level Ranking Loss）**：基于Spearman秩相关系数（SRCC）度量师生模型top-k预测顺序的一致性，直接优化峰值预测的对齐。
2. **实验验证多模态分布对齐的重要性**：通过CR和MOR指标量化证明，峰值预测质量与模型性能强相关，且现有KD目标在学习多模态分布上存在明显不足。
3. **证明排序损失与现有目标高度兼容**：可无缝融合KL、RKL、JSD、TVD、AKL等主流目标，在预训练中稳定提升多模态分布对齐效果。
4. **在多个下游任务中取得显著提升**：在GSM8K、Dolly、Xsum等任务上，引入排序损失后准确率提升超20%，ROUGE-L提升超1.0分。

## 方法详解
### 核心设计：排序损失（Ranking Loss）
- **计算对象**：取师生模型各自top-k预测的并集，对这两个子集中的概率值计算排序一致性。
- **优化目标**：使用Spearman秩相关系数（SRCC）而非Pearson，因语言模型输出为离散非线性的logits，SRCC更关注顺序一致性而非实际数值相关性。
- **公式**：
$$
\mathcal{L}_{\mathrm{Ranking}} = 1 - \rho_{\mathrm{srcc}}(p, q) = 1 - \frac{\mathrm{Cov}(R_p, R_q)}{\sigma_{R_p} \cdot \sigma_{R_q}}
$$
其中 $p$ 和 $q$ 分别为师生模型在top-k并集上的概率子集，$R_p$、$R_q$ 为其对应的秩索引。

### 总损失函数
- **固定比例分配**：$\mathcal{L}_{\mathrm{total}} = 2 \cdot \mathcal{L}_{\mathrm{Ranking}} + \mathcal{L}_{\mathrm{logits}}$
- **动态比例分配（针对AKL）**：利用top-k交集重叠率（OR）作为学生模型对当前输入理解程度的指标，动态调整 $\mathcal{L}_{\mathrm{logits}}$ 权重：
$$
\mathcal{L}_{\mathrm{total}} = 2 \cdot \mathcal{L}_{\mathrm{Ranking}} + \frac{|p^k \cap q^k|}{k} \cdot \mathcal{L}_{\mathrm{logits}}
$$

### 实现细节
- 使用 **torchsort** 库实现可微排序算子（基于Blondel et al., 2020）。
- 仅增加约1%训练时间开销。

## 实验与结果
### 数据集与模型
- **教师模型**：Llama-2-7B（GSM8K使用gsm8k-rft-llama7b2-u13b）
- **学生模型**：TinyLlama-1.1B
- **预训练数据**：SlimPajama（5,000个切片）
- **下游任务**：GSM8K（数学推理）、Dolly（指令微调）、Xsum（文本摘要）

### 预训练任务结果
- 引入排序损失后，CR（一致性率）提升约**30%-95%**，MOR（平均重叠率）提升约**50%-120%**。
- Top-1准确率和困惑度未受负面影响，甚至略有提升。
- 与所有五种基线目标（KL/RKL/JSD/TVD/AKL）结合均稳定有效。

### 下游任务结果（GSM8K）
| 方法 | 正确数 | 提升 |
|------|--------|------|
| KL | 219 | — |
| KL+R | 267 | **+48 (+21.9%)** |
| TVD | 0 | — |
| TVD+R | 240 | **+240** |

### 下游任务结果（Dolly/Xsum）
- ROUGE-L提升普遍超过**1.0分（Dolly）**和**0.7分（Xsum）**。
- 即使仅使用排序损失（Rank-15），在GSM8K上也达到236正确数，超过多数基线。

## 相关工作脉络
1. **SeqKD**：基于beam search序列级蒸馏，依赖教师生成数据；本文聚焦词级软标签优化，与SeqKD正交。
2. **Gu et al. (2023) GKD**：使用RKL替代KL以关注峰值预测；本文进一步指出RKL仍会遗漏多模态峰值。
3. **Wen et al. (2023) f-DISTILL**：引入JSD/TVD等对称散度缓解模式问题；本文认为根本问题在于未显式优化峰值顺序。
4. **Wu et al. (2024b) AKL**：理论上证明KL与RKL在LLM蒸馏中优化目标一致，提出自适应组合；本文在其基础上添加排序损失进一步提升。
5. **Ko et al. (2024) DistillMN**：探索 streamlined distillation；本文方法可与其框架结合。

## 局限性与未来方向
1. **架构局限**：仅在Llama架构上验证，未来需扩展至其他模型族（如Baichuan、GLM等）。
2. **超参数敏感**：top-k范围（推荐5-15）和损失比例（1:2）需经验调优，动态调整策略在batch计算中可能增加负担。
3. **仅关注白盒蒸馏**：未涉及黑盒场景（如GPT-4类闭源模型），适用性受限。

## 研究启发与可借鉴点
1. **多模态分布评估指标可迁移**：CR和MOR指标可用于评估任意生成模型的对齐质量，可作为KD过程的中间监控信号。
2. **排序损失可泛化到其他生成任务**：在机器翻译、语音合成等同样存在多模态输出的任务中，可能具有类似价值。
3. **可结合课程学习策略**：动态分配损失的思路（基于OR指标）可扩展为自适应蒸馏课程。
4. **与团队方向结合机会**：若团队研究模型压缩或高效推理，可将RLKD与量化、剪枝等方法结合，探索联合优化路径。

## 关键术语表
**Knowledge Distillation (KD)**：通过软标签将大模型知识迁移到小模型的过程。
**Multi-Modal Distribution**：LLM预测概率分布中同时存在多个显著峰值的现象。
**Top-k Sampling**：从概率最高的k个token中采样的解码策略。
**Spearman Rank Correlation Coefficient (SRCC)**：衡量两个变量排序一致性的非参数量。
**Consistency Rate (CR)**：师生模型top-k预测完全一致（类别和顺序）的比例。
**Mean Overlap Rate (MOR)**：师生模型top-k预测共享类别比例的平均值。
**Adaptive Kullback-Leibler (AKL)**：动态组合KL和RKL的蒸馏目标。
**Differentiable Sorting**：使排序操作可微的技术，支持梯度下降优化。

## 可复现要素
- **数据集**：SlimPajama、GSM8K、Dolly-15k、Xsum，均公开可用。
- **代码/权重**：教师模型Llama-2-7B、学生模型TinyLlama-1.1B开源；ranking loss使用torchsort库实现。
- **关键超参**：学习率2e-5、batch size 64（下游任务）、序列长度2048、训练轮数2（下游）/20（预训练）、蒸馏温度1.0、top-k范围5-15、损失比例 $\mathcal{L}_{\mathrm{Ranking}}:\mathcal{L}_{\mathrm{logits}}=2:1$。
