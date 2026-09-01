---
title: "NYT-CONNECTIONS-A-Deceptively-Simple-Text-Classification-Tas"
source: https://aclanthology.org/2025.coling-main.134.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:57:18"
field: "大语言模型推理能力评估"
keywords: ["benchmark", "reasoning", "system-1 system-2", "language model evaluation", "chain-of-thought", "word classification", "cognitive testing"]
innovations: ["提出 NYT-CONNECTIONS 基准专门惩罚 System 1 直觉思维", "揭示 CoT/SC 在困难任务上收益递减的反直觉发现", "建立人类与 LLM 在 deliberate reasoning 任务上的量化性能差距"]
benchmarks: ["NYT-CONNECTIONS"]
---

# 论文速读：NYT-CONNECTIONS-A-Deceptively-Simple-Text-Classification-Tas

## 一句话总结
本文提出了 NYT-CONNECTIONS 基准，源自《纽约时报》的 Connections 字谜游戏，通过 358 个单词分类任务专门惩罚快速直觉的 System 1 思维，迫使模型进行系统性深思熟虑的推理；评估发现即使是 GPT-4、Claude 3.5 等顶尖大模型也落后人类近 30%，且 CoT 等提示技巧在难度上升时收益递减。

## 研究问题与动机
- **现有基准难以隔离认知能力**：当前 NLP 基准（如 MMLU、Big-Bench）常混合多种认知过程，无法独立评估单一推理能力。
- **Shortcut learning 现象普遍**：模型常利用统计规律或表面线索“走捷径”，而非真正理解任务语义。
- **缺乏对 System 2 思维的测评工具**：现有基准无意中奖励直觉型 System 1 思维，缺少对 deliberate reasoning（深思熟虑推理）的有效评估手段。
- **数据泄露风险日益严重**：随着训练语料膨胀，测试集污染现象加剧，需持续更新的基准缓解此问题。

## 核心贡献（创新点）
1. **提出 NYT-CONNECTIONS 基准**：构建 358 个源自真实 NYT Connections 游戏的单词分类谜题，专门设计以抑制 System 1 捷径。
2. **多维度评测框架**：设计 One Try、No Hints、Full Hints 三种配置，全面评估模型在不同约束下的推理表现。
3. **设计系统化的启发式基线**：提出基于嵌入聚类的简单启发式算法，模拟人类初始直觉判断，作为对照基准。
4. **揭示 LLM 推理能力的本质局限**：发现 CoT/SC 等先进提示技术在困难任务上反而不如简单 IO 提示，挑战了当前对大模型推理能力的认知。
5. **建立人类 vs 机器性能差距的量化证据**：证明即使最强 LLM 在 System 2 任务上也落后人类近 30%，填补该领域的评估空白。

## 方法详解
**任务形式**：给定 16 个单词，将其划分为 4 组，每组 4 个语义相关的词（如 "Skin Types"：Normal, Dry, Combination, Oily）。

**数据集构建**：
- 收集 2023.06.12–2024.06.03 期间完整的 358 个 NYT Connections 谜题
- 引入难度评级（1-5级），选取中位数难度（3.0）附近的 100 个谜题进行实验

**启发式基线设计**：
评分函数 $S = G - P$，其中：
- 组相似性 $G = 0.4 \cdot I + 0.3 \cdot s + 0.3 \cdot V$
  - $I = -K(E)$：k-means 聚类惯性（簇内平方和）
  - $s$：组内最小余弦相似度
  - $V = \frac{\text{mean}(P)}{1 + \text{var}(P)}$：相似度一致性
- 惩罚项 $P = \frac{1}{|R|} \sum_{r \in R} \cos(\mu_C, r)$，衡量候选组与剩余词的区分度
- 使用 beam search（宽度=10）搜索最优分组

**实验设置**：
- 模型：Claude 3.5 Sonnet、GPT-4、GPT-4o、Gemini 1.5 Pro、LLaMA 3 70B、LLaMA 3.1 405B
- 提示方法：IO（直接输入输出）、CoT（思维链）、CoT-SC（自一致性）
- 评估配置：One Try（单次尝试）、No Hints（最多4次尝试无提示）、Full Hints（含"差一个"提示）

## 实验与结果
**数据集**：100 个谜题（难度中位数 3.0），来源 NYT-CONNECTIONS 库（358 个）

**主要结果**（Table 2，IO 提示）：

| 模型 | One Try | No Hints | Full Hints |
|------|---------|----------|------------|
| GPT-4 | 4.0% | 35.5% | 32.5% |
| Claude 3.5 | 11.0% | 36.75% | 40.25% |
| GPT-4o | 8.0% | 45.0% | 33.75% |
| LLaMA 3.1 405B | 7.0% | 35.5% | 34.75% |
| Gemini 1.5 Pro | 5.0% | 30.5% | 31.5% |
| LLaMA 3 70B | 1.0% | 23.75% | 28.5% |
| Heuristic | 1.0% | 13.25% | 13.25% |
| **Humans** | **39.33%** | **56.0%** | **60.67%** |

**关键发现**：
- 最佳模型 Claude 3.5 (Full Hints: 40.25%) 仍落后人类 (60.67%) 约 **20.4 个百分点**
- One Try 配置下，GPT-4 仅达 4%，而人类达 39.33%
- CoT/SC 在困难谜题上反而表现更差，Simple IO 提示有时更优
- 启发式方法 (13.25%) 与较小 LLM (LLaMA 3 70B: 23.75%) 差距不大
- GPT-4o 在 Full Hints 下性能反降（33.75% vs 45%），暗示提示可能干扰推理

## 相关工作脉络
1. **Shortcut learning 研究**（Geirhos et al., 2020; Trichelair et al., 2019）：本文延续对模型利用表面线索问题的关注，但聚焦词义聚类任务。
2. **System 1/2 框架**（Hagendorff et al., 2023）：本文直接应用该心理学框架设计测评基准，区分直觉与深思熟虑推理。
3. **数据泄露应对**（Balloccu et al., 2024; Huang et al., 2024）：通过每日更新的谜题源缓解泄露，区别于静态基准。
4. **多步推理基准**（Suzgun et al., 2023; McKenzie et al., 2024）：本文同样挑战 System 1 启发式，但采用词义关联而非数学/代码任务。
5. **动态基准工作**（Sun & Emami, 2024; Li et al., 2024; Jain et al., 2024）：本文与之相似但聚焦 NLP 语义理解而非代码生成。

## 局限性与未来方向
- **嵌入模型规模限制**：启发式基线使用较小的 Multilingual-E5-Large-Instruct，更大模型可能改变结论
- **提示工程范围有限**：受成本限制，未测试 Tree of Thoughts 等复杂长上下文方法
- **人类样本较小**：仅 3 名评估者，可能无法代表更广泛人群的问题解决能力
- **数据集时间局限性**：尽管每月更新，但 2023-2024 年的谜题可能随语言/文化演变而过时
- **跨文化适用性**：谜题主要为英语西方受众设计，需开发多语言/文化适配版本

## 研究启发与可借鉴点
1. **游戏化基准设计思路**：将日常益智游戏（如 Connections）转化为科学评测工具，兼具趣味性与诊断价值，可迁移至其他认知能力评估。
2. **三阶段提示对比实验设计**：IO → CoT → CoT-SC 的递进式提示方案，系统性地检验不同推理策略的有效性，值得复用。
3. **启发式基线+LLM对比范式**：用简单规则基线揭示"大模型是否真的在推理"，而非仅报告绝对分数，为后续研究提供鉴别框架。
4. **难度动态调节机制**：通过 Difficulty 1-5 评级筛选实验样本，确保任务难度可控，可推广至其他基准构建。
5. **错误模式分析价值**：展示 GPT-4 在 Laundry 类别上的典型失败案例，揭示 CoT 的浅层思维局限，为提示工程改进提供诊断依据。

## 关键术语表
**System 1 Thinking**：快速、自动、直觉性的思维方式，依赖启发式和表面线索，无需 conscious effort。
**System 2 Thinking**：缓慢、审慎、深思熟虑的推理方式，需要 conscious effort 和 deliberate processing。
**Shortcut Learning**：模型利用数据中的统计规律或表面特征而非真正理解任务语义的现象。
**Chain-of-Thought (CoT)**：通过要求模型逐步展示推理过程来提升复杂任务表现的提示技术。
**Self-Consistency (SC)**：在 CoT 基础上采样多个推理路径并取多数投票，提升输出稳定性的方法。
**Inertia (I)**：k-means 聚类中簇内样本到质心的平方距离之和，衡量簇的紧密程度。
**Beam Search**：保留 top-k 候选解的搜索算法，在探索与利用之间取得平衡。
**One Away Hint**：提示用户某组仅差一个词即正确的 contextual hint，辅助解决问题。

## 可复现要素
- **数据集**：NYT-CONNECTIONS，358 个谜题，来源于纽约时报官方游戏（2023.06.12–2024.06.03），作者承诺每月更新
- **代码**：论文未明确提及开源仓库链接，但提供了详细的启发式算法公式与伪代码
- **权重**：使用 Multilingual-E5-Large-Instruct 嵌入模型（HuggingFace 公开）
- **关键超参**：
  - Beam search width = 10
  - k-means k = 1
  - 权重系数：G = 0.4·I + 0.3·s + 0.3·V
  - LLaMA temperature = 0.6，其余模型 temperature = 0.5
  - 实验样本：100 个中位数难度谜题
