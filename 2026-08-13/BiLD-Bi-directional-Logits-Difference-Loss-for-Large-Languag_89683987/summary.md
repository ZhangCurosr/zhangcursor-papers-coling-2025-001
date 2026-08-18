---
title: "BiLD-Bi-directional-Logits-Difference-Loss-for-Large-Languag"
source: https://aclanthology.org/2025.coling-main.78.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:01:10"
field: "大语言模型压缩与蒸馏"
keywords: ["Knowledge Distillation", "Large Language Models", "Logits Distillation", "Model Compression", "LLM Pruning"]
innovations: ["提出BiLD loss通过top-k截断和双向logits差值同时过滤长尾噪声和利用内部排序信息", "发现LLM logits比视觉模型更具长尾分布特征，揭示排序信息对生成策略的关键作用"]
benchmarks: ["SuperGLUE", "ARC", "HellaSwag", "PIQA", "WinoGrande"]
---

# 论文速读：BiLD-Bi-directional-Logits-Difference-Loss-for-Large-Languag

## 一句话总结
本文针对大语言模型（LLM）logits蒸馏中存在的长尾噪声和排序信息利用不足问题，提出了Bi-directional Logits Difference (BiLD) loss；该损失通过top-k截断过滤噪声、构造双向logits差值显式利用内部排序信息，在13个NLP数据集上以仅top-8 logits实现了对SFT及多种现有蒸馏方法的超越。

## 研究问题与动机
1. **LLM logits长尾噪声问题**：相比视觉模型，LLM的输出空间为离散token序列，其logits呈现更显著的长尾分布，尾部存在大量"噪声"干扰蒸馏性能。
2. **现有方法未充分利用logits内部排序信息**：top-k sampling和top-p sampling等LLM生成策略受logits内部排序影响，但现有logits蒸馏方法（如vanilla KL、RKL等）未能显式利用这一信息。
3. **现有LLM logits蒸馏方法的缺陷**：Reverse KL存在"mode-seeking"问题；基于optimal transport的方法（如Sinkhorn Distance）计算复杂，难以应用于数十亿参数的开源LLM。
4. **视觉蒸馏方法难以直接迁移到LLM**：DKD、NKD、NormKD等面向视觉模型的logits蒸馏方法未考虑LLM输出空间的独特性。

## 核心贡献（创新点）
1. **揭示了LLM logits的长尾分布特征与内部排序重要性**：通过kurtosis和top-k占比实验证明文本logits比图像logits更"尖锐"，且top-k排序信息对LLM生成策略有直接影响——与已有工作将logits视为概率分布整体处理的思路本质不同。
2. **提出Bi-directional Logits Difference (BiLD) loss**：通过top-k截断过滤长尾噪声并构造双向logits差值来利用内部排序信息，而非直接最小化logits之间的KL散度。
3. **证明了仅top-8 logits即可实现SOTA蒸馏性能**：BiLD仅使用teacher/student各top-8 logits就超越了使用完整logits的vanilla KL及五种其他蒸馏方法——区别于top-k KL仅做截断不利用排序信息的做法。
4. **提供了新指标overlap@k评估logits级别对齐效果**：通过top-1和top-8 overlap衡量学生模型在greedy search和开放场景下对教师主要行为的模仿程度——扩展了现有蒸馏评估维度。

## 方法详解
BiLD loss由两个对称分量组成：teacher-led logits difference (t-LD) loss和student-led logits difference (s-LD) loss。

**t-LD loss计算流程**：
1. 选择top-k teacher logits并按降序排列：$\mathbf{z}_{\text{led}}^t = [z_{i_1}^t, z_{i_2}^t, \cdots, z_{i_k}^t]$，其中$z_{i_1}^t \geq z_{i_2}^t \geq \cdots \geq z_{i_k}^t$
2. 按相同位置索引选取对应的student logits：$\mathbf{z}_{\text{cor}}^s = [z_{i_1}^s, z_{i_2}^s, \cdots, z_{i_k}^s]$
3. 计算内部成对差值构建logits differences：
   - $\mathbf{d}_{\text{led}}^t = [z_{i_m}^t - z_{i_n}^t \mid 1 \leq m < n \leq k]$
   - $\mathbf{d}_{\text{cor}}^s = [z_{i_m}^s - z_{i_n}^s \mid 1 \leq m < n \leq k]$
4. 对差值进行softmax归一化得到概率分布
5. 计算KL散度：$\mathcal{L}_{t\text{-LD}} = D_{\text{KL}}[\mathbf{p}_{\text{led}}^t \parallel \mathbf{p}_{\text{cor}}^s]$

**s-LD loss**以相同逻辑对称构造：选择top-k student logits，提取对应位置的teacher logits，计算差值和KL散度。

**BiLD loss**为两者之和：$\mathcal{L}_{\text{BiLD}} = \mathcal{L}_{t\text{-LD}} + \mathcal{L}_{s\text{-LD}}$

**排序信息利用原理**：由于$\mathbf{z}_{\text{led}}^t$已排序，其差值$\mathbf{d}_{\text{led}}^t$所有元素非负；若student对应差值出现负值（即排序不一致），KL散度会惩罚该偏差，从而迫使student logit排序与teacher对齐。

**关键超参**：$k=8$（经消融实验确定）、temperature $T=3$。

## 实验与结果
**数据集**：13个NLP数据集，包括SuperGLUE基准8个（BoolQ、CB、COPA、MultiRC、ReCoRD、RTE、WiC、WSC）和额外5个（Arc-C、Arc-E、HellaSwag、PIQA、WinoGrande）。

**基线方法**：SFT、vanilla KL、top-k KL（$k=1024$）、DKD、NKD、NormKD、RKL。

**模型配置**：Teacher使用BLOOM-7B和Qwen-4B；Student使用BLOOM-3B、BLOOM-1B、Qwen-1.8B、Qwen-0.5B。

**主要结果**（Average Accuracy）：
- **BLOOM-7B → BLOOM-3B**：BiLD以71.85%优于SFT (66.56%)、vanilla KL (71.21%)和top-k KL (70.57%)
- **BLOOM-7B → BLOOM-1B**：BiLD以68.92%超越vanilla KL (67.82%)约1.10%
- **Qwen-4B → Qwen-1.8B**：BiLD以75.07%超越vanilla KL (73.98%)约1.09%
- **Qwen-4B → Qwen-0.5B**：BiLD以70.68%超越vanilla KL (67.16%)约3.52%

**Logits级别对齐效果**（overlap@k指标）：
- 在所有4组蒸馏设置中，BiLD的overlap@8均最高（如Qwen-4B→0.5B达到68.58%），表明学生模型更好地模仿了教师的top-8主要行为。
- overlap@1保持竞争性水平，说明BiLD不损害greedy search场景下的精确预测能力。

**消融实验结论**：
- Temperature $T=3$时表现最佳，过低温度显著降低性能
- $k=8$在计算开销和性能间取得最佳平衡，$k \in \{1,2\}$时性能较差，$k>8$时overlap@8/32持续提升但计算成本急剧增加

## 相关工作脉络
1. **Vanilla KL蒸馏（Hinton et al., 2015）**：基础logits蒸馏方法，直接最小化teacher-student logits概率分布的KL散度；BiLD在此基础上引入top-k截断和差值构造，解决了LLM长尾噪声问题。
2. **Reverse KL (RKL)**：用于缓解mode-averaging问题，但易导致mode-seeking；BiLD通过双向差值机制避免了单一方向散度的缺陷。
3. **视觉logits蒸馏方法（DKD/NKD/NormKD）**：面向视觉模型设计，解耦target/non-target logits或使用动态temperature；BiLD针对LLM输出空间特殊性重新设计，未直接迁移这些方法。
4. **Sinkhorn Distance蒸馏（SinKD, Cui et al., 2024）**：计算复杂度高，难以应用于大规模LLM；BiLD以$O(k^2)$复杂度（$k=8$）实现更高效的distillation。
5. **Top-k KL蒸馏**：仅做logits截断过滤噪声，未利用内部排序信息；BiLD在此基础上增加了差值构造显式利用排序关系。
6. **White-box/Black-box蒸馏分类**：BiLD属于black-box蒸馏（仅需teacher logits），但可作为vanilla KL的替代品与white-box方法结合使用。

## 局限性与未来方向
1. **需要teacher logits访问权限**：无法直接使用GPT-4、Gemini等仅提供文本输出的闭源模型作为teacher。
2. **要求teacher-student共享词表**：vector space需要对齐，限制了跨架构蒸馏的应用。
3. **计算复杂度随k增大而急剧上升**：logits差值计算为$O(k^2)$，增加clipped logits数量会带来显著时间开销。
4. **长尾知识丢失问题**：直接截断长尾部分虽提升性能，但不可避免地损失了尾部包含的知识；如何更好地利用长尾分布中的知识是未来方向。
5. **仅在task-specific distillation场景验证**：未探索general-purpose蒸馏或chain-of-thought蒸馏场景的适用性。

## 研究启发与可借鉴点
1. **长尾噪声过滤思路可迁移**：top-k截断作为预处理步骤可应用于其他logits-level方法（如RKL、对比学习蒸馏），提升LLM蒸馏鲁棒性。
2. **排序信息利用的通用框架**：BiLD的差值构造机制可推广到其他序列生成任务（如机器翻译、文本摘要），帮助小模型学习教师的候选token偏好排序。
3. **overlap@k评估指标的价值**：该指标区分了"精确匹配能力"（overlap@1）和"关键行为模仿能力"（overlap@8/32），可作为后续工作的标准评估补充。
4. **与白盒方法结合的潜力**：BiLD作为black-box loss可直接替换现有白盒蒸馏框架中的vanilla KL项，无需修改架构即可提升性能。
5. **k值的自适应探索**：当前固定$k=8$，可研究根据任务难度或模型尺寸动态调整k值的策略，平衡性能与计算效率。

## 关键术语表
**Logits Distillation**：通过最小化teacher和student模型输出logits之间差异来进行知识蒸馏的方法，区别于基于hidden states的white-box蒸馏。
**Long-tail Distribution**：指logits中少数高值占据大部分概率质量、大量低值（噪声）分布于尾部的分布特性，LLM中比视觉模型更显著。
**Top-k Sampling**：LLM生成时仅从logits最高的k个token中进行采样的策略，用于控制生成多样性。
**Overlap@k**：本文提出的新评估指标，衡量teacher和student模型在同一位置上top-k logit对应token的重合比例。
**Mode-seeking vs Mode-averaging**：Reverse KL导致的两种极端行为，前者使学生过度集中在高概率区域，后者使分布过于平滑。
**Teacher-led / Student-led Logits Difference**：BiLD的两个对称分量，分别以teacher和student的top-k logits为基准构造差值进行蒸馏。

## 可复现要素
- **数据集**：SuperGLUE及额外5个NLP数据集（均为公开基准）
- **代码**：已开源，https://github.com/fpcsong/BiLD
- **模型权重**：使用开源模型BLOOM和Qwen1.5
- **关键超参**：temperature $T=3$，$k=8$，batch size=64，micro batch size=2，learning rate (SFT: 1e-5, distillation: 2e-5)，warmup steps=64，epochs (SFT: 3, distillation: 8)
- **硬件**：8× NVIDIA A100 GPU
- **训练技巧**：DeepSpeed、gradient checkpointing、BFLOAT16
