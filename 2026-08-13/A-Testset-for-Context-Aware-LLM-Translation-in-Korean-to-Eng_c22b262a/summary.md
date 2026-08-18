---
title: "A-Testset-for-Context-Aware-LLM-Translation-in-Korean-to-Eng"
source: https://aclanthology.org/2025.coling-main.110.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:58:53"
---

# 论文速读：A-Testset-for-Context-Aware-LLM-Translation-in-Korean-to-Eng

## 一句话总结
本文构建了首个无数据泄露风险的韩英篇章级人工挑战测试集（600例，覆盖6种语篇语言现象），并通过对比Google Translate与GPT-4o在零样本与自研CAP（上下文感知提示）框架下的表现，证实了逐步CoT推理能显著提升复杂语篇翻译质量，同时揭示了现有自动评估指标在篇章级细粒度评测中的失效问题。

## 研究问题与动机
- **单句评测天花板已现**：高资源语言对的单句机器翻译人机差异已极小，传统sentence-level评测难以有效刻画LLM在需要跨句上下文推理时的真实潜力与局限。
- **既有评测数据集存在数据泄露**：广泛使用的语篇评测集（如Voita et al. 2019）早在2019年公开，极易被现代LLM训练数据污染，导致评测分数失真。
- **评测形式与语言覆盖不足**：过往工作多采用二选一选择题间接评估，且以英语/欧洲语言为主；缺乏面向非欧洲语言、直接生成译文输出、且按语篇现象细粒度划分的人工构建挑战集。

## 核心贡献（创新点）
1. **构建防泄露的韩英篇章挑战测试集**：手工撰写600个多句文本实例，按Ambiguity、Ellipsis、Literalism三大类划分六种子现象，每类100例；与基于WMT或网络爬虫挖掘的现有数据集本质不同，彻底规避了LLM训练数据泄露风险。
2. **提出CAP上下文感知提示框架**：将CoT思想适配为“检测→策略→候选生成→抽取”四步串行流水线，强制模型在翻译前显式推断跨句上下文与语用意图；与直接zero-shot翻译相比，本质区别在于将隐式语篇推理转化为显式的多步规划。
3. **建立现象专项人工评测体系并诊断自动指标盲区**：为六种现象分别制定3分制评分标准，并结合人机偏好对比研究；发现s-BLEU、COMET22、d-BLEU、BlonDe及GEMBA-MQM均无法稳定反映CAP相比零样本带来的细粒度提升，指出当前自动指标不适用于篇章级翻译的精细化评估。
4. **系统性定位SOTA LLM在语篇翻译上的残留短板**：实证表明即使GPT-4o零样本在多数现象上已超越GMT，但在Zero Anaphora等强跨句推理现象上仍存在明显缺陷，而CAP可带来显著增益（如Zero Anaphora上提升0.31分）。

## 方法详解
- **测试集设计**：源文由韩语母语者构造并标注目标现象位置，参考译文由精通英韩的双语高级译者手工完成。三类宏观现象下细分六项：Lexical Ambiguity（一词多义依赖语境消歧）、Zero Anaphora（韩语代词脱落需跨句恢复主语/宾语）、Slang/Idiom/Figurative Language/Implicature（均属Literalism范畴，需语用与常识推理）。
- **CAP四步流水线**：
  1. **Detection**：使用针对特定现象定制的Step-1 Prompt，要求模型识别源文现象并结合跨句语境推断实际发生的情境与说话意图。
  2. **Strategy**：基于检测结论制定翻译策略，仅输出推理过程与关键考量点，暂不生成正式译文。
  3. **Translation Candidates**：根据策略生成5个英文候选译文，要求模型逐项权衡并选出最优项，附理由。
  4. **Extraction**：清洗输出，仅保留最终选定的译文。
- **实验设置**：基线包括Google Translate（GMT）与GPT-4o Zero-Shot；CAP同样基于GPT-4o API，temperature固定为0。所有Prompt模板与现象专项检测提示详见附录B/D。
- **评估方式**：① 现象级3分制人工评分（4名双语评测员）；② 人机/机机偏好对比；③ 自动指标对比（s-BLEU, COMET22, d-BLEU, BlonDe, GEMBA-MQM）。

## 实验与结果
- **数据集**：600例韩英篇章测试集，平均每篇约2.93句，源文均长11.14词/句，译文均长10.71词/句（Appendix A）。
- **3分制人工评估（表2）**：CAP平均分2.72，Zero-Shot 2.59，GMT 2.35。CAP在Zero Anaphora（+0.31）、Idiom（+0.16）、Figurative Language（+0.18）、Implicature（+0.07）上显著优于零样本；Lexical Ambiguity与Slang两者持平。GMT在Slang上得分最低（2.06），反映传统NMT缺乏网络俚语常识。
- **偏好研究（表3）**：CAP对比人类翻译胜率39.67%（人类负率29.50%），Zero-Shot胜率32.00%（人类负率39.50%），GMT胜率26.83%。除Slang外，CAP在所有现象类别中均获最高偏好胜率。
- **自动指标（表4）**：s-BLEU普遍给予Zero-Shot更高分数，COMET22/d-BLEU/BlonDe趋势相近，GEMBA-MQM在部分现象上零样本与CAP差异微弱；自动指标整体呈现“零样本≥CAP”的反直觉分布，与人工评估结论背离，
