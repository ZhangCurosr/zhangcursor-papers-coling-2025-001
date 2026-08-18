---
title: "Dying-or-Departing-Euphemism-Detection-for-Death-Discourse-i"
source: https://aclanthology.org/2025.coling-main.90.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:03:08"
field: "历史文本计算语言学与数字人文"
keywords: ["euphemism detection", "death discourse", "historical text analysis", "pre-trained language models", "Danish-Norwegian literature", "computational humanities", "GPT-4o prompting"]
innovations: ["提出'委婉vs非委婉用法'的语境敏感二分类框架，区别于传统委婉vs字面对立", "构建首个丹麦-挪威19世纪文学死亡委婉语标注数据集(799样本)，填补低资源历史文本研究空白", "发现领域适配PLM(DanskBERT F1=0.87)在历史委婉语检测上全面超越GPT-4o few-shot策略"]
benchmarks: ["MeMo Corpus (859 novels, 64M+ tokens)", "ScandEval", "自建799样本委婉语检测数据集"]
---

# 论文速读：Dying-or-Departing-Euphemism-Detection-for-Death-Discourse-i

## 一句话总结
本文构建了首个面向19世纪丹麦-挪威文学语料中死亡相关委婉语的标注数据集，并系统评估了预训练语言模型与GPT-4o在该任务上的表现，同时揭示了死亡委婉语使用随时间推移从固定表达向隐喻表达转变、以及世俗化进程中小说类型影响委婉语使用频率的文化历史趋势。

## 研究问题与动机
1. **核心问题**：如何在历史文学文本中自动检测并分类与死亡相关的委婉语？现有工作多聚焦于社交媒体或当代语言，缺乏针对北欧历史文学语料的系统性研究。
2. **方法论缺口**：以往研究多区分"委婉 vs. 字面"，本文明确提出区分"委婉 vs. 非委婉用法"（同一表达在不同语境下可分别为两种用法），与"字面 vs. 比喻"区分相互独立但又相关。
3. **文化史视角**：19世纪末丹麦-挪威海 secularization（世俗化）加速，社会对死亡的话语方式发生转变，计算语言学方法可量化这一语言演变过程。
4. **多语言/低资源挑战**：丹麦语和挪威语作为低资源语言，其历史文学文本上的委婉语检测研究尚属空白。

## 核心贡献（创新点）
1. **构建了首个丹麦-挪威历史文学死亡委婉语标注数据集**：从MeMo语料库中提取799段标注样本（50.3%委婉语 vs. 49.7%非委婉语），涵盖29个Potential Euphemistic Terms (PETs)；与已有工作相比，首次聚焦北欧19世纪文学语境下的死亡委婉语，填补了语种和历史时期的空白。
2. **提出了"委婉 vs. 非委婉用法"而非"委婉 vs. 字面"的二分类框架**：同一PET（如"go to sleep"类表达）在不同上下文中可分别标注为委婉或非委婉，比传统的委婉/字面对立更能捕捉语言使用的语境敏感性。
3. **系统评估了多个PLMs与GPT-4o在历史委婉语检测上的性能**：发现领域适配的历史模型MeMo-BERT-03验证集F1达0.93，而通用模型DanskBERT在测试集上以0.87超越所有基线（包括GPT-4o）；与已有GPT-4o研究（Firsich & Rios, 2024）相比，本文揭示了历史文本场景下微调PLMs仍优于zero-shot/few-shot大模型。
4. **通过计算分析方法揭示了死亡话语的文化历史演变趋势**：固定表达使用频率下降、隐喻性委婉语上升；历史题材小说中委婉语使用显著多于同时代当代小说，反映了19世纪世俗化对文学话语的影响。

## 方法详解
1. **语料库与PET收集**：使用MeMo语料库（19世纪末859部丹麦/挪威小说，6400万+词元），由领域专家筛选29个Potential Euphemistic Terms (PETs) 作为关键词（如`gå bort`、`sove ind`、`komme hjem`等），通过正则表达式提取包含PET的上下文片段，共获得11,280个文本片段。
2. **采样与标注策略**：对11,280个片段分层采样799个（高频PET各取47个随机样本，低频PET全量采集），由三位领域专家（教授、副教授、博士生）进行二分类标注（euphemism / non-euphemism）；测试集由三人共同标注并以多数投票决定最终标签。标注单元为5句窗口（目标句前后各2句）。
3. **模型评估方案**：数据划分为训练集(69%, 552例)、验证集(15%, 122例)、测试集(16%, 125例)。微调实验使用4个PLM（DanskBERT、DFM Large、MeMo-BERT-03、NB-BERT-base），batch size=32，10 epochs，AdamW优化器，学习率10⁻³，以F1为指标。Prompt实验使用GPT-4o（gpt-4o-2024-08-06），temperature=0，包含zero-shot、zero-shot with context、few-shot（2例/4例）五种提示变体。
4. **分类器辅助语料分析**：用测试集表现最佳的DanskBERT对全部11,280个未标注片段自动预测，识别出1,312个委婉语片段（11.6%），结合出版年份和小说类型（历史/当代）进行历时和文体分布分析。
5. **inter-annotator agreement**：测试集125个样本的Cohen's Kappa为0.86，表明标注者间达"substantial agreement"水平。

## 实验与结果
- **数据集**：799段标注样本（402委婉语/397非委婉语），基于MeMo语料库（859部小说，1,936,527个片段，64,227,927词）。
- **微调PLM结果**（F1-score）：
  - DanskBERT：**测试集0.87**（验证集0.92），为最佳整体表现，且未见PET在测试集出现于训练集时仍达0.79
  - MeMo-BERT-03：验证集0.93（最高）/ 测试集0.86，存在一定过拟合
  - DFM (Large)：验证集0.90 / 测试集0.85
  - NB-BERT-base：验证集0.89 / 测试集0.85
- **GPT-4o结果**（F1-score）：
  - Few-shot with targeted examples（4例）：验证集0.85 / 测试集**0.75**（最优prompt策略）
  - Few-shot with random examples：验证集0.79 / 测试集0.74
  - Zero-shot：验证集0.77 / 测试集0.72
  - Zero-shot with context：验证集0.62 / 测试集0.61
- **关键结论**：①微调PLMs（尤其是DanskBERT）全面超越GPT-4o的各类prompt策略；②Few-shot显著提升GPT-4o性能，targeted示例优于random；③DanskBERT展现了优秀泛化能力。
- **语料分析发现**：固定表达（如`omkomme`/perish）使用频率随时间下降，隐喻性表达（如`miste ham/hende`/lose him/her、`gå bort`/pass away）频率上升；历史题材小说中委婉语使用比例显著高于当代小说。

## 相关工作脉络
1. **Firsich & Rios (2024)**：利用GPT-4o进行跨语言委婉语检测（zero-shot/few-shot），在FigLang 2024任务中获胜；本文与其定位差异在于聚焦北欧历史文学文本，且发现微调PLM在特定历史语料上仍优于GPT-4o few-shot。
2. **Lee & Feldman (2024)**：Multilingual Euphemism Detection Shared Task的赛事报告，整合了多语言委婉语检测方法；本文扩展了这一方向至丹麦-挪威低资源历史文本。
3. **Vitiugin & Paakki (2024)**：提出集成上下文特征的跨语言委婉语检测模型，大幅超越传统方法；本文采用类似的微调PLM路线但聚焦单一语义域（死亡）。
4. **Hankins (2024)**：用LoRA微调PLM评估多语言委婉语检测；本文未用LoRA但覆盖了更广泛的北欧模型基线。
5. **Zhu et al. (2021)**：使用phrase mining和word embedding进行多词委婉语检测；本文采用人工筛选PET+regex提取的确定性方法，而非无监督发现。
6. **Devi & Saharia (2024)**：通过聚类识别未知领域的委婉语；本文方法依赖预定义PET列表，作者在Limitations中明确指出此限制。

## 局限性与未来方向
1. **PET列表非 Exhaustive**：仅依赖29个专家筛选的关键词，可能遗漏其他语境依赖型或不在列表中的委婉表达。
2. **缺乏与非委婉死亡表达的对照分析**：未将委婉语与直接死亡动词（如`dø`/to die）的使用频率进行系统对比，无法量化两者比例变化趋势。
3. **宗教语境下的定义模糊性**：在宗教世界观中，"回家到上帝那里"可能是字面信仰表达而非委婉语，标注和分类难以完全捕捉这种emetic视角差异。
4. **语种与时期局限**：数据集仅限于19世纪末丹麦-挪威小说，结论难以推广到其他语言、时期和文化语境。
5. **模型可进一步提升**：未探索unsupervised方法自动发现新委婉表达，也未进行更细致的error analysis。
6. **未来方向**：①与直接死亡表达（如`dø`的各种变体）的对照分析；②扩展至其他语言和文化遗产语境；③探索宗教话语与世俗化交叉的委婉语演变。

## 研究启发与可借鉴点
1. **"委婉 vs. 非委婉"的语境敏感标注框架**：同一lexical item在不同上下文可不同标注，这一思路可迁移至其他敏感话题（如疾病、贫困、性）的委婉语研究中，避免简单的一词一标签假设。
2. **领域适配历史PLM的显著优势**：MeMo-BERT-03和DanskBERT在历史文学文本上明显优于通用模型和GPT-4o few-shot，印证了"历史文本→领域继续预训练→下游微调"范式的价值，可推广至其他低资源历史语言。
3. **分类器辅助大规模语料文化历史分析**：用少量标注数据微调模型后，对11,280个未标注片段进行批量预测并历时/按文体分布分析，这种"小数据标注→大数据推断→文化计算分析" pipeline对数字人文研究具有示范意义。
4. **Few-shot示例的选择策略影响显著**：targeted示例（与待分类样本同类的示例）比random示例效果好，提示在low-resource场景下prompt工程需重视示例的代表性和针对性。
5. **跨学科协作的标注范式**：三位不同背景专家（语言学、数字人文、文学）协作标注，IAA达0.86，为NLP与人文社科学者的合作标注流程提供了可复现的方法论参考。

## 关键术语表
**Euphemism（委婉语）**：用温和或间接的表达替代令人不快或禁忌的直接说法的语言手段。
**Potential Euphemistic Term (PET)**：作者预先收集的、可能用作死亡委婉语的词或短语列表（共29个），作为文本检索的关键词。
**MeMo Corpus**：包含19世纪末859部丹麦和挪威小说的大型语料库，共计6400万+词元，是本文的数据基础。
**Secularization（世俗化）**：本文所指19世纪末北欧社会宗教影响减弱、世俗价值观兴起的文化历史进程，被认为是死亡话语转变的重要动因。
**Historical vs. Non-historical Novels**：历史小说（故事设定在过去时代）与当代/现实小说（故事设定在写作同时代），前者委婉语使用更频繁。
**Cohen's Kappa**：衡量标注者间一致性的统计量，本文测试集得分为0.86（substantial agreement）。
**Pre-trained Language Model (PLM)**：在大规模文本上预训练的神经网络语言模型，本文fine-tune用于委婉语检测任务。
**Zero-shot / Few-shot**：大模型prompt策略，前者不给示例直接预测，后者提供少量标注示例辅助分类。

## 可复现要素
- **数据集**：已公开，GitHub链接：https://github.com/mime-memo/Euphemism-of-Death
- **代码**：已公开（同上GitHub链接）
- **模型权重**：DanskBERT、MeMo-BERT-03、NB-BERT-base均有公开版本；DFM (Large) 来自 Danish Foundation Models 项目
- **关键超参**：batch size=32，10 epochs，AdamW优化器，学习率10⁻³，dropout=0.1（DFM），max position=512
- **GPT-4o模型版本**：gpt-4o-2024-08-06，temperature=0
- **评测指标**：F1-score
- **评估基准**：ScandEval（丹麦/挪威NLP基准），自身构建的799样本标注数据集
- **论文未提及**：LoRA适配参数、GPU硬件配置详情、具体训练时间
