---
title: "The-PRECOM-SM-Corpus-Gambling-in-Spanish-Social-Media"
source: https://aclanthology.org/2025.coling-main.2.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:00:48"
field: "计算社会科学与健康NLP"
keywords: ["赌博沉迷检测", "西班牙语NLP", "社交媒体语料库", "行为健康", "早期风险预测", "文本分类"]
innovations: ["构建首个西班牙语赌博沉迷社交媒体语料库PRECOM-SM，覆盖五种赌博类型和多平台数据", "提出以交互频率(MF/LF)作为沉迷代理标签的分类框架，无需临床诊断即可开展检测研究", "系统比较Textflow手工特征、BoW-TF.IDF和RoBERTa深度学习编码三类方法在赌博文本上的性能差异"]
benchmarks: ["eRisk/CLEF 赌博沉迷检测评测"]
---

# 论文速读：The-PRECOM-SM-Corpus-Gambling-in-Spanish-Social-Media

## 一句话总结
本文构建了首个面向西班牙语社交媒体赌博沉迷检测的语料库 **PRECOM-SM Corpus**，涵盖赌博、交易/加密货币、游戏、开箱等五种赌博类型；同时提出以交互频率作为沉迷程度的替代标签，并给出了基于 BoW、深度编码和人工特征三类方法的基线实验。

## 研究问题与动机
1. **语言空白**：现有赌博沉迷检测研究几乎全部基于英语社交媒体（尤其是 eRisk/CLEF 评测），尚无针对西班牙语的已发表研究成果。
2. **青少年沉迷趋势严峻**：WHO 已将病理性赌博列为精神障碍；西班牙 2022 年调查显示 14–18 岁青少年中 20.1% 参与过赌博，其中 17.9% 存在潜在问题，线上赌博风险更高。
3. **缺乏公开语料支撑**：现有工作多依赖赌博平台交易日志或客服对话文本，而非真实的青少年社交媒体自然文本，难以用于早期风险自动检测。
4. **假设驱动**：基于已有文献（Savolainen et al., 2022; Vepsäläinen et al., 2024），作者在社区中的参与频率与赌博风险呈正相关，据此用交互频率作为分类标签的代理。

## 核心贡献（创新点）
1. **构建首个西班牙语赌博沉迷社交媒体语料库 PRECOM-SM**：覆盖 Telegram、Twitch、Reddit、Ludopatia.org 五类赌博主题，填补了西班牙语无公开语料的空白，与现有英语语料形成跨语言对比基础。
2. **提出交互频率（MF/LF）标签体系**：通过手动审核用户帖子时序规律将用户划分为高频（MF）和低频（LF）两类，无需临床诊断即可为下游检测任务提供可操作标签。
3. **提供三种特征的对比基线实验（Textflow 特征 / BoW-TF.IDF / DL-RoBERTa）**：在五个赌博子集上系统比较了不同特征工程路径与多种 ML 算法，为后续研究提供可复现的参照系。
4. **揭示不同类型赌博社区的语言学差异**：指出 Botting/Trading 社区高频用户使用更简单句式和强烈情绪，而 Loot Boxes/Gambling 社区低频用户反而写更长复杂句子，为细粒度建模提供线索。

## 方法详解
- **数据收集**：从四类平台抓取西班牙语文本（2017–2023，主要为 2021 年），按五种赌博类型分设子集：Betting（体育博彩）、Gambling（轮盘/老虎机等）、Trading & Crypto、Video games、Loot boxes。由西班牙语言学家审核确保内容为青少年用语。
- **预处理**：去除换行符、Tab/多余空格、Bot 消息、不可读内容（GIF/图片/sticker 标记为"Ilegible"）、高频重复弹幕；移除 URL 和 Twitch 命令；小写化；去除西班牙语 stop words；编码统一为 Unicode。
- **特征工程（三类实验）**：
  - **Exp 1（Textflow 特征）**：使用 Textflow 库提取 46 维特征，涵盖文本复杂度（AvgLenSent、Huerta 可读性、nRare 等）、词汇多样性（MTLD、HDD、MaasTTR 等）、情绪（RoBERTuito-emotion-analysis，六类 Ekman 情绪 + 中性）、反讽（NDF/UOIronity 等）和极性；对同一用户的消息取各特征均值作为该用户的特征向量。
  - **Exp 2（BoW）**：对用户消息拼接后计算 TF.IDF 向量。
  - **Exp 3（DL）**：使用 MarIA 项目的 Spanish RoBERTa 模型进行编码。
- **分类框架**：二分类任务（MF vs LF）；采用 Leave-one-out 交叉验证；评估指标为 Accuracy、Precision、Recall、F1（macro weighted）。
- **超参数**：主要使用 scikit-learn 默认配置；RF（20 棵树，seed=45）、MLP/LR/SGD（max_iter=1000）、KNN/SVM/DecisionTree 均为默认参数，未做调优。

## 实验与结果
- **语料规模**（Table 3）：Trading 最大（MF 608,821 条消息 / 573 用户；LF 44,499 条 / 573 用户），Betting 次之（246,829 条），Gambling（52,173 条），Loot boxes（876 条），Video games 最小（73 条）。
- **最佳结果**（Table 5）：
  - **Trading（BoW + RF）**：Accuracy 0.965，Precision 0.967，Recall 0.965，F1 0.966（全子集中最优）。
  - **Betting（BoW + RF）**：Accuracy 0.910，F1 0.910。
  - **Gambling（DL + MLP）**：Accuracy/Precision/Recall/F1 均为 0.921。
  - **Loot boxes（DL + LR）**：Accuracy 0.889，F1 0.887。
  - **Video games（DL + MLP）**：各项指标均为 1.000，但样本仅 6 用户，结论有限。
- **规律**：Telegram 平台（Trading、Betting）以 BoW 效果最佳，表明共同词汇模式较强；Twitch 平台（Gambling、Loot boxes）以 Deep Learning 优势明显；Random Forest 和 MLP 整体表现稳定。
- **统计学差异**（Table 4）：高频用户整体情绪表达更强（愤怒、恐惧、悲伤突出），反讽使用显著更高；不同赌博类型间词汇多样性与句子复杂度指标差异显著（p < 0.05）。

## 相关工作脉络
1. **eRisk/CLEF 系列评测（Parapar et al., 2021, 2023）**：当前英语赌博沉迷检测的主要评测基准，最佳方法为 SVM + Character 4-grams BoW（F1 0.721）；本文定位为其西班牙语扩展，填补语言空白。
2. **UNSL 团队（Loyola et al., 2021）**：eRisk 2021 冠军方案，Character n-gram + TF.IDF + SVM；本文 BoW 在 Trading 上达到更高 F1（0.966），但标签定义不同（交互频率 vs 临床诊断）。
3. **SINAI 团队（Mármol-Romero et al., 2022）**：eRisk 2022 在早期预警（ERDE）指标上表现突出；本文侧重于 corpus 构建而非早期预警机制设计。
4. **Hernández-Ruiz & Gutiérrez（2021）**：唯一涉及的西班牙语先验工作，分析西班牙体育博彩 Twitter 账户的极性，不涉及成瘾检测；本文在其基础上延伸至多平台和成瘾分类。
5. **Fino et al.（2021）**：COVID-19 期间对英文 Twitter 的赌博相关情感分析，使用 BTM 主题建模；本文聚焦文本特征 + 分类任务而非主题发现。
6. **Li et al.（2020）**：对 enTenTen13 和 Google Books 语料中 gaming/gambling 词汇历时分析；本文关注当代社交媒体实时语言模式。

## 局限性与未来方向
1. **标签非临床诊断**：MF/LF 分类基于交互频率的假设，虽与文献一致但未经专业心理评估验证，可能存在误标（非常活跃用户未必沉迷）。
2. **样本量不均衡**：Trading 数据量远超其他四类，Video games 仅 73 条消息，部分结果（如 1.000 准确率）可靠性存疑。
3. **未做超参数调优**：所有模型使用 scikit-learn 默认参数，可能存在性能提升空间。
4. **语言覆盖单一**：仅西班牙语，未考虑拉美变体（文中已提及发现 vos 等拉美用法，但未建模区分）。
5. **未来方向**：计划引入 Sentence Transformers 和 Longformer 以处理更长用户文本；探索半自动化标注（如识别"我已确诊成瘾"类表达）；补充临床诊断数据以提升标签可信度。

## 研究启发与可借鉴点
1. **交互频率作为代理标签**：在缺乏临床标注的情况下，用社区参与频率区分高风险/低风险用户是一种实用策略，可迁移至其他成瘾行为（游戏、社交媒体）的早期检测研究。
2. **多特征路径对比范式**：同时评测手工特征（Textflow）、传统 BoW 和深度学习编码三类方法，为语料库类论文提供了完整的 baseline 参照，适合在本团队研究中复现。
3. **跨平台数据整合思路**：单一平台信息有限，本文整合 Telegram/Twitch/Reddit/forum 四类来源并按赌博主题细分，对多源社交文本研究具有方法论参考价值。
4. **青少年语言特征建模**：文中明确关注青少年用语（anglicisms、neologisms、emoji 使用），提示在低资源语言的情感/行为检测中需将变体语言纳入特征体系。
5. **结合情绪与反讽信号**：Textflow 提取的情绪（愤怒/悲伤/快乐）和反讽指标在不同赌博类型中表现不同，可作为后续多任务学习或特征选择的起点。

## 关键术语表
**PRECOM-SM Corpus**：本文构建的西班牙语赌博沉迷社交媒体语料库，包含五种赌博类型、两个交互频率等级的文本数据。
**MF / LF（Most Frequent / Least Frequent）**：基于用户发帖频率和活跃度划分的高频/低频用户标签，作为沉迷程度的代理分类。
**Textflow**：Python 文本分析库，提供复杂度、词汇多样性、情绪、反讽、极性、NER 等 46 维特征提取能力。
**RoBERTuito**：在 5 亿条西班牙推文上预训练的 RoBERTa 模型，用于西班牙语社交媒体情绪识别。
**BoW（Bag of Words）**：将文本表示为词频/TF.IDF 向量的传统特征提取方法，本文在 Trading 和 Betting 上表现最佳。
**DL（Deep Learning encodings）**：使用预训练 RoBERTa 模型（MarIA 项目）提取的句子/文档级语义向量。
**eRisk / CLEF**：跨语言网络早期风险预测评测活动，是当前行为健康检测领域最重要的英语评测基准。
**Loot boxes**：免费游戏中可通过真实货币购买的随机虚拟道具盒，与问题赌博高度相关的游戏机制。

## 可复现要素
- **数据集**：PRECOM-SM Corpus，已在 Zenodo 公开（https://zenodo.org/records/8055604），包含各赌博类型的独立数据集及按日/周/月的用户交互统计文件。
- **代码/权重**：论文未提供官方代码仓库；模型权重方面使用了 RoBERTa（MarIA 项目）和 RoBERTuito（Pérez et al., 2021），均为开源模型；特征提取依赖 Textflow Python 库。
- **关键超参**：RF（20 棵树，seed=45）；MLP/LR/SGD（max_iter=1000）；SGD 早停阈值 1×10⁻⁴；其余算法（KNN、SVM、DecisionTree）使用 scikit-learn 默认参数；未进行超参调优。
- **评估协议**：Leave-one-out 交叉验证；指标为 Accuracy、Precision、Recall、F1（macro weighted）。
