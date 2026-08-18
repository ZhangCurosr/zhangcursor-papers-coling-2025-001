---
title: "Monte-Carlo-Tree-Search-Based-Prompt-Autogeneration-for-Jail"
source: https://aclanthology.org/2025.coling-main.71.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:44:38"
field: "大语言模型安全与对抗攻击"
keywords: ["jailbreak attack", "MCTS", "prompt generation", "LLM safety", "black-box attack", "adversarial suffix"]
innovations: ["基于MCTS的提示自动生成框架MPA，通过树搜索策略提升越狱攻击效率", "设计结构化动作空间（10种策略）与先验置信概率P(s,a)引导搜索", "在黑盒设置下实现跨模型高ASR（97%-100%）与低ANA的平衡"]
benchmarks: ["AdvBench Subset", "MaliciousInstruct"]
---

# 论文速读：Monte-Carlo-Tree-Search-Based-Prompt-Autogeneration-for-Jailbreak-Attacks-against-LLMs

## 一句话总结
论文提出基于蒙特卡洛树搜索（MCTS）的提示自动生成方法（MPA），通过设计结构化动作空间和先验置信概率，在黑盒设置下高效、自动地生成对抗性越狱提示，在多个开源与闭源LLM上显著提升了攻击成功率与搜索效率。

## 研究问题与动机
- 现有白盒越狱攻击（如GCG、AutoDAN）依赖模型内部梯度或logits信息，跨模型迁移时有效性与效率显著下降；黑盒方法则存在随机性强、需要人工设计初始提示等问题。
- 当前黑盒LLM-based攻击（如PAIR、TAP）虽能生成语义性越狱提示，但缺乏系统性的搜索策略引导，导致平均尝试次数（ANA）较高、收敛慢。
- 越狱攻击空间庞大且非结构化，需设计有效的探索-利用平衡机制以提升搜索效率。
- 研究旨在为LLM安全评估提供更高效、更通用的攻击方法，帮助发现模型防御漏洞。

## 核心贡献（创新点）
- **提出MPA框架**：将越狱提示生成建模为MCTS搜索问题，通过四阶段迭代（动作选择、状态扩展、状态评估、反向传播）自动寻找对抗性提示。与现有LLM-based方法的本质区别在于引入树搜索策略而非简单迭代优化。
- **设计结构化动作空间**：提出10种攻击策略，分为角色扮演、形式变换、提示修改三类，覆盖语义重构、角色代入、语言转换等多种越狱路径。与随机搜索或单一策略方法的本质区别在于多策略协同探索。
- **引入先验置信概率P(s,a)**：基于目标模型拒绝响应的log概率（以"I"、"I'm"等前缀为target token）计算动作先验，引导搜索聚焦高成功概率动作。与无先验MCTS的本质上是通过可观测信号减少无效探索。
- **黑盒通用性验证**：在Llama-3-8B、gemma2-9b、Mistral-7B及闭源GPT-4o-mini上均取得高ASR（97%-100%），显著优于GCG、AutoDAN、PAIR、TAP等基线。与白盒方法的本质区别在于无需模型内部参数即可实现高效攻击。

## 方法详解
**整体框架**：MPA包含三个LLM——攻击LLM A（Vicuna-13b-v1.5）、目标LLM T（待攻击模型）、评估LLM E（GPT-4o-mini）。以有害问题q为根节点，通过MCTS迭代搜索对抗提示p*。

**动作空间设计**：
- **角色扮演类（5种）**：通用角色设定、问题解决任务、顾问咨询任务、情绪操控、问题合法化
- **形式变换类（3种）**：语言翻译、时态转换、科学实验设计
- **提示修改类（2种）**：重新措辞、添加上下文

**动作选择（PUCT算法）**：
$$a^* = \arg\max_{a \in A} \left[ c_p \cdot P(s_m, a) \sqrt{\frac{\sum_a N(s_m, a)}{1 + N(s_m, a)}} + Q(s_m, a) \right]$$
其中$Q(s,a)$为平均奖励，$N(s,a)$为访问次数，$c_p=1$为探索-利用平衡系数。

**先验置信概率设计**：
- 定义target token为模型拒绝回答时的固定前缀（如"I"、"I'm"）
- 通过目标模型T的log概率计算置信度：$\mathcal{C}(s_m, a^*) = \exp(\log P_T(t|p_{s_m}))$
- 先验概率：$P(s_m, a^*) = 1 - \mathcal{C}(s_m, a^*)$，反映动作绕过安全过滤的可能性

**状态评估**：
- 评估LLM E对目标模型响应打分（1-10分），10分为完全越狱成功
- 达到阈值则终止搜索

**反向传播**：
- 更新节点Q值和访问次数N值
- 将奖励值沿路径回传至根节点

## 实验与结果
**数据集**：
- AdvBench Subset：50个有害提示，覆盖32个类别
- MaliciousInstruct：100个有害指令，覆盖10个类别

**评估基线**：
- 白盒：GCG、AutoDAN
- 黑盒：PAIR、TAP、RS（随机搜索，含初始化版本）

**主要结果**（AdvBench Subset）：
| 方法 | Llama3-8B | gemma2-9b | Mistral3-7B | GPT-4o-mini |
|------|-----------|-----------|-------------|-------------|
| MPA | **100%** ASR / 15.64 ANA | **100%** ASR / 5.06 ANA | **100%** ASR / 3.12 ANA | 98% ASR / 20.40 ANA |
| PAIR | 74% / 43.60 | 84% / 31.32 | 100% / 6.10 | 96% / 21.58 |
| TAP | 4% / 97.84 | 80% / 36.66 | 100% / 8.12 | 64% / 54.50 |
| RS_init | 100% / 3.94 | 12% / 89.92 | 100% / 3.12 | 8% / 96.10 |

**主要结果**（MaliciousInstruct）：MPA在Llama3-8B、gemma2-9b、Mistral3-7B、GPT-4o-mini上分别达到97%、98%、99%、97% ASR，ANA分别为15.18、5.75、5.18、16.90。

**消融实验**：去除先验概率P(s,a)后，Llama3-8B在AdvBench上的ASR从100%降至58%，验证了先验机制对搜索效率的关键作用。

**最强结果**：MPA在AdvBench Subset上对Llama3-8B、gemma2-9b、Mistral3-7B均达到100% ASR，ANA显著低于PAIR（降低64%~91%）和TAP（降低97%~99%）。

## 相关工作脉络
- **GCG/AutoDAN**：白盒攻击，依赖梯度优化对抗后缀；MPA定位为黑盒通用方法，无需内部信息。
- **PAIR/TAP**：LLM-based黑盒攻击，通过迭代生成+评估优化提示；MPA引入树搜索策略，减少无效探索，提升收敛速度。
- **RS（随机搜索）**：利用log概率指导搜索；MPA通过结构化动作空间与先验概率，比纯随机搜索更高效、更具针对性。
- **RL-JACK**：基于深度强化学习的越狱攻击；MPA采用MCTS而非DRL，搜索过程更透明可控。
- **GCG++/AmpleGCG**：改进型白盒攻击；MPA与之差异在于完全黑盒设置，适用于API访问场景。
- **定位总结**：MPA填补了黑盒设置下高效、系统化越狱搜索的空白，兼顾攻击成功率与搜索效率。

## 局限性与未来方向
- **效率瓶颈**：方法性能受限于LLM推理速度，需结合加速推理技术提升实用性。
- **ASR未达完美**：部分模型（如GPT-4o-mini）未达100% ASR，需进一步优化动作空间或搜索策略。
- **未来方向**：在更大规模模型（如Llama-3-70B）和更强安全防御（如Constitutional AI）上测试泛化性；探索多轮交互与上下文学习的结合。

## 研究启发与可借鉴点
- **MCTS框架迁移**：可将树搜索策略应用于其他对抗性文本生成任务（如毒性检测绕过、隐私泄露攻击），提升搜索效率。
- **先验概率设计**：利用模型输出log概率作为探索引导信号，可在其他黑盒对抗攻击中复用，减少无效查询。
- **动作空间设计**：三类十种策略的结构化设计思路可推广至多领域提示工程，如角色扮演、形式变换、上下文增强等通用模式。
- **评估范式**：结合ASR与ANA双重指标评估攻击效率，可为后续安全基准测试提供标准化参考。
- **团队结合点**：若团队关注LLM安全评估，可借鉴MPA的MCTS框架开发自动化红队测试工具；若关注对抗样本生成，可参考其先验概率设计优化搜索策略。

## 关键术语表
- **Jailbreak Attack**：通过 crafted prompt 诱导LLM绕过安全约束、生成有害内容的攻击方法。
- **MCTS（Monte Carlo Tree Search）**：基于树结构的启发式搜索算法，通过探索-利用平衡优化决策过程。
- **ASR（Attack Success Rate）**：攻击成功样本占总样本的百分比，衡量攻击有效性。
- **ANA（Average Number of Attempts）**：平均尝试次数，衡量攻击搜索效率。
- **PUCT（Predictor Upper Confidence Tree）**：结合先验策略网络与Q值的树搜索选择策略，用于平衡探索与利用。
- **Prior Confidence Probability P(s,a)**：基于目标模型拒绝响应的log概率计算的先验分布，引导搜索方向。
- **Black-box Attack**：仅通过API查询获取模型输出，无法访问内部参数或梯度的攻击设置。
- **AdvBench Subset**：包含50个有害提示的评测数据集，用于系统评估越狱攻击效果。

## 可复现要素
- **代码开源**：https://github.com/KDEGroup/MPA
- **数据集**：AdvBench Subset（公开）、MaliciousInstruct（公开）
- **攻击模型**：Vicuna-13b-v1.5
- **目标模型**：Llama-3-8B-Instruct、gemma-2-9b-it、Mistral-7B-Instruct-v0.3、GPT-4o-mini
- **评估模型**：GPT-4o-mini
- **关键超参**：$c_p = 1$，最大搜索迭代次数 = 100，temperature = 0，生成长度 = 150 tokens
- **论文未提及**：具体训练数据、模型微调细节
