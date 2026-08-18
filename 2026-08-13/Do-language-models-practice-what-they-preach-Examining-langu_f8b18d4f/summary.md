---
title: "Do-language-models-practice-what-they-preach-Examining-langu"
source: https://aclanthology.org/2025.coling-main.80.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:30:10"
field: "NLP 与社会公平 / 语言意识形态评估"
keywords: ["language ideologies", "gendered language reform", "metalinguistic preferences", "political bias", "value alignment", "singular they", "role nouns"]
innovations: ["提出基于元语言偏好的 LLM 语言意识形态量化方法", "首次揭示积极元语言属性下 LLM 对单数 they 的保守派偏向", "证明 LLM 在不同元语言语境下使用改革语言的频率不一致"]
benchmarks: ["Papineau role noun set (N=12/52)", "Camilliere singular they stimuli (N=40)"]
---

# 论文速读：Do-language-models-practice-what-they-preach-Examining-language-ideologies-about-gendered-language-reform-encoded-in-LLMs

## 一句话总结
本文通过性别中立语言改革（角色名词与单数 they）的案例研究，系统检验 LLM 中编码的语言意识形态，发现：（1）在"正确/自然"等积极元语言提示下，LLM 的语言选择隐含保守派政治偏见；（2）LLM 在不同元语言语境下的用词选择存在内部不一致性。

## 研究问题与动机
- **核心问题**：LLM 在元语言语境（metalinguistic context）中的词汇选择是否反映特定政治立场的语言意识形态？其语言选择在不同元语言语境下是否保持一致？
- **现有方法不足**：既往研究多关注 LLM 的偏见输出（如方言歧视、性别排他），但缺乏对"元语言偏好"——即模型对什么是"正确/自然"语言的评价——的系统测量，而这些偏好会潜移默化地影响用户采纳语言改革的意愿。
- **社会影响关切**：LLM 若将单数 they（非二元代词）判定为"不正确"，会边缘化非二元群体；而改革语言的推广依赖于元语言陈述（metalinguistic statements）的传播，因此模型在此类语境中的表现具有重要社会后果。
- **价值对齐缺口**：当前值对齐（value alignment）工作主要关注显式价值观，但忽视了模型在看似"中立"的元语言判断中隐含的政治立场与行为不一致性。

## 核心贡献（创新点）
1. **提出基于元语言偏好的语言意识形态分析方法**：通过比较模型在不同元语言提示下的变体概率，量化其隐含的语言意识形态，区别于以往仅测量生成文本表层偏见的工作。
2. **揭示积极元语言属性的保守派偏向**：首次系统证明，被请求使用"correct/natural"等积极属性的模型，其在单数 they 上的选择与保守派提示高度吻合，而非进步派。
3. **发现元语言语境依赖性导致的不一致性**：证明模型在更明确的元语言语境下使用更多改革语言，但在直接语境下减少使用，形成用户预期与实际输出的落差。
4. **构建涵盖 52 个角色名词集 + 40 个单数代词句的受控刺激集**：为后续研究提供可复用的实验材料和度量框架。
5. **桥接社会语言学与 NLP 的立场分析（stancetaking）**：将语言意识形态研究与 NLP 立场分析结合，区分政治群体标签与政治立场（stances）的差异化影响。

## 方法详解
- **实验对象**：9 个 LLM——GPT-3（text-curie-001, 175B）、GPT-3.5（text-davinci-002/003, ~1.3B–175B）、Flan-T5（small 80M/large 780M/xl 3B）、Llama（llama-2-7B/llama-3-8B/llama-3.1-8B）。GPT 与 Llama 为自回归模型，其余含指令微调，text-davinci-003 另有 RLHF。
- **刺激材料**：角色名词域——52 个核心句模板 [NAME] is a [ROLE-NOUN]，每句含 3 个变体（中立/女性/男性）；代词域——40 个句子模板（主格/宾格/反身/所有格四种语法形式），来源 Camilliere et al. (2021)。
- **变量定义**：对每个提示项 i 和变体集 V，计算改革变体 $v_r$ 的相对概率：
  $$p(\text{reform}|i) = \frac{p(v_r|i)}{\sum_{v \in V} p(v|i)}$$
  其中 $p(v|i)$ 依模型架构不同：自回归模型取目标 token 概率乘积（含 Salazar et al. 2020 的全上下文修正）；Flan-T5 用哨兵 token 机制分别查询各变体。
- **实验 1（政治偏向）**：提示条件包括 positive-metaling（7 条，如"correct""natural"）、prog（2 条）、cons（1 条）、prog-stance（3 条）、cons-stance（3 条）。对每个句模板 t 计算 $\delta_t(\text{prog}, \text{meta})$ 和 $\delta_t(\text{cons}, \text{meta})$，经配对 t 检验判断模型的正向元语言行为更接近哪一方。
- **实验 2（内部一致性）**：操控"提问方式"（direct vs. indirect，含 adjective best/likely、verb refer/complete）与"前言条件"（null vs. choices vs. individual-declaration vs. ideology-declaration）。使用 beta 回归模型：
  ```
  p_reform ~ indirect + best + refer + choices + ind_dec + ideo_dec + (1|item) + (1|name)
  ```
- **统计显著性**：$p < 0.05$，Bonferroni 校正按模型数。

## 实验与结果
- **数据集**：角色名词 52 刺激 × 40 名字 = 2080 数据点（GPT 仅 12 刺激 × 40 名字 = 480）；单数代词 40 刺激 × 40 名字 = 1600 数据点。实验 2 共 32,000–41,600 数据点。
- **实验 1 结果**：
  - 角色名词：结果混合，部分模型显示无明确偏向（图 3a）；GPT 子集结果与全集合一致（附录 B）。
  - 单数代词：positive-metaling 行为与保守派提示显著相似（橙色连接线，图 3b），改革语言使用率极低。
  - 立场提示：几乎所有情况下 positive-metaling 更接近保守立场（图 3c,d），prog-stance 改革率高于 prog 标签本身，说明立场比群体标签更具诊断力。
- **实验 2 结果**：
  - GPT 与 Llama 模型：多数条件下更元语言的提示（best、choices、individual-declaration、ideology-declaration）显著提升改革语言使用率（Table 4a,b 绿色显著系数）；indirect 预测在某些条件下反而降低使用率。
  - 代词域中 refer 一致预测更低改革率（可能反映单数 they 接受度低于角色名词）。
  - Flan-T5 模型结果差异最大，且不与模型规模单调相关——80M 与 3B 表现不一。
  - 最强效应：davinci-003 在 individual-declaration 条件下对代词的 $\beta = 5.13$（Table 4b），显著高于其他条件。
- **关键结论**：LLM 在"正确/自然"语境下隐含保守派语言意识形态；不同元语言语境下使用改革语言的频率显著不同，呈现内部不一致。

## 相关工作脉络
- **Blodgett et al. (2020)** 批判性综述 NLP 中的"偏见"概念，本文在此基础上聚焦语言意识形态这一更细粒度维度。
- **Hu & Levy (2023)** 证明 LLM 在元语言语境下的偏好比一般语言使用更不准确；本文进一步指出元语言偏好并非"噪音版本"，而是传达有意义的社会政治信息。
- **Dentella et al. (2023)** 发现 LLM 在元语言问题上准确性低且响应不稳定；本文与其互补，强调不一致性本身即是一种社会危害来源。
- **Hossain et al. (2023) MISGENDERED** 评估 LLM 理解代词的能力；本文聚焦元语言陈述而非纯理解能力。
- **Behzad et al. (2023) ELQA** 构建元语言问答数据集；本文不关注问答准确率，而关注模型在元语言框架下选择的变体类型所反映的意识形态。
- **Watson et al. (2023)** 研究 BERT 编码的性别社会态度；本文扩展至 LLM 生成阶段，并引入政治立场维度。

## 局限性与未来方向
- 仅研究英语性别语言改革，结论未必推广至其他改革类型（称谓、neopronouns）或其他语言（含语法性别系统语言如瑞典语）。
- 刺激材料基于美国流行名字，种族/文化多样性不足。
- OpenAI Completions API 已弃用且移除 GPT-3.5 的 token 级概率访问，GPT 结果难以复现。
- 受控刺激不一定反映真实用户与 LLM 的交互场景；需补充自然主义用户研究。
- 仅测试 9 个模型，覆盖有限；模型选择可能影响结论的普遍性。
- 未来方向：扩展到自然场景（如文本修订任务）；研究其他语言改革；评估非营利组织/政治团体对模型语言选择与自身价值观的一致性。

## 研究启发与可借鉴点
- **元语言偏向检测框架可直接迁移**：将目标语言现象替换为其他社会相关语言变体（如非二元称谓、方言标记），即可检测 LLM 在各类语言改革中的意识形态偏向，适合用于模型社会影响评估。
- **区分群体标签与立场提示的实验设计**：本文发现立场提示（progressive-stance）比群体标签（progressive）更具诊断力，这一设计原则可推广至政治偏见评估的其他场景。
- **beta 回归 + 随机截距的重复测量范式**：将概率型因变量建模为 beta 回归、控制 item/name 随机效应，是处理变体选择概率数据的标准方法，可直接复用于其他类似实验。
- **结合社会语言学理论指导提示设计**：本文从 Zimman (2017)、Kroskrity (2004) 等文献中系统提取用于构建提示的形容词，而非凭直觉构造，这一方法路径值得借鉴以确保生态效度。
- **团队可结合方向**：在中文性别中立语言（如"TA"替代"他/她"）的研究中复用本方法，检测中文 LLM 在此类改革中的意识形态倾向与一致性。

## 关键术语表
- **语言意识形态（language ideologies）**：关于语言的评价性观念与信念（如什么是"正确的""自然的"），常隐含对社会群体的价值判断。
- **元语言陈述（metalinguistic statements）**：表达关于语言使用之价值判断的任何语句，是传播和采纳语言改革的关键机制。
- **元语言偏好（metalinguistic preferences）**：LLM 在元语言语境下对不同语言变体的相对概率选择，反映其隐含的语言意识形态。
- **立场（stance / stancetaking）**：说话者对某话题所采取的评价性位置，与社会身份群体（如政治派别）相关联。
- **性别中立改革语言（gender-neutral reform language）**：如 congressperson、singular they，旨在取代性别标记形式的包容性语言变体。
- **价值对齐（value alignment）**：使 LLM 的输出与人类价值观保持一致的训练与评估目标，本文主张需纳入语言意识形态维度。
- **beta 回归（beta regression）**：适用于因变量为连续概率值（0,1）的广义线性回归模型，本文用于分析实验 2 的多因素效应。
- **角色名词（role nouns）**：表示社会角色的名词，本文选取包含中性/女性/男性三变体的集合（如 congressperson/congresswoman/congressman）。

## 可复现要素
- **代码**：MIT 许可证开源，GitHub 地址见论文脚注 7（原文未给完整 URL）。
- **数据集**：角色名词刺激来自 Papineau et al. (2022)（MIT 许可证，GitHub：github.com/BranPap/gender_ideology）；代词刺激来自 Camilliere et al. (2021)，需向原作者索取。
- **模型**：Flan-T5（Apache 2.0，Hugging Face transformers）；Llama（Llama Community License）；GPT 模型通过 OpenAI API 访问（API 已弃用，不可复现）。
- **超参数**：论文未详列温度、top-p 等生成超参；统计检验使用 paired t 检验（Bonferroni 校正）与 beta 回归，$p < 0.05$。
- **硬件**：NVIDIA Titan Xp 与 Quadro RTX 6000 GPU，累计 46 GPU 小时。
