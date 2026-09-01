---
title: "TRANSMI-A-Framework-to-Create-Strong-Baselines-from-Multilin"
source: https://aclanthology.org/2025.coling-main.32.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:00:05"
field: "低资源机器翻译与跨语言NLP"
keywords: ["跨语言迁移", "转写", "词表扩展", "多语言预训练模型", "零训练适配"]
innovations: ["提出TRANSMI零训练框架，通过词表转写-合并-初始化三步直接构造转写数据基线模型", "设计Min/Max/Average三种处理转写歧义的词表合并模式", "证明改造模型在不损失原文处理能力的前提下显著提升转写数据表现"]
benchmarks: ["SR-B (Bible)", "SR-T (Tatoeba)", "Taxi1500", "SIB200", "WikiANN (NER)", "Universal Dependencies (POS)"]
---

# 论文速读：TRANSMI-A-Framework-to-Create-Strong-Baselines-from-Multilin

## 一句话总结
论文提出 TRANSMI（Transliterate-Merge-Initialize）框架，无需任何训练即可将现有 mPLM 改造为能同时处理原文与非拉丁脚本转写数据的强基线模型，在多个跨语言下游任务上对转写数据实现 3%~34% 的显著提升。

## 研究问题与动机
- **脚本壁垒制约跨语言迁移**：mPLM 的词汇表以非拉丁脚本（如汉字、阿拉伯文、西里尔文等）为主，导致转写为统一拉丁脚本后，tokenizer 产生无意义切分（如"今天"→"_jint", "ian"），无法对应语言层面单元。
- **已有方案计算开销大**：现有方法（Dhamecha et al., 2021; Moosa et al., 2023）要么从头训练新 mPLM，要么通过参数微调适配转写数据，均需大量计算预算。
- **缺乏零训练适配范式**：能否直接利用已有 mPLM 及其 tokenizer 知识，以"零训练"方式构建适配转写数据的强基线模型，尚无人系统研究。
- **单一脚本模型的局限性**：针对转写数据的专用模型通常仅处理一种脚本，难以兼顾原始脚本与非转写数据的处理能力。

## 核心贡献（创新点）
1. **提出 TRANSMI 零训练框架**：通过翻译词表→合并新词表→初始化新 embedding 三步，直接基于现有 mPLM 构建可处理转写数据的强基线，无需任何训练。
2. **设计三种处理转写歧义的 Merge 模式**：Min-Merge、Max-Merge、Average-Merge，分别保留低频子词、优先高频子词和平衡平均，以应对多原词共享同一转写的歧义问题。
3. **证明方法不损害原文处理能力**：实验表明 TRANSMI 改造后的模型在原始脚本（非转写）数据上性能几乎无损，同时在转写数据上全面提升。
4. **揭示不同脚本受益差异**：对音节/字母文字（西里尔、阿拉伯、天城文等）的转写数据提升显著；对汉字（语义信息损失）和拉丁文（本身接近）的提升较小，验证了方法的适用范围。

## 方法详解
**TRANSMI 三阶段流程：**

**阶段一：词表转写（Tokenizervocabulary Transliteration）**
- 使用确定性转写工具 Uroman 将 mPLM 原始词表 $V^{\text{orig}}$ 中每个子词 $w$ 转写为拉丁形式 $v = \text{Transli}(w)$。
- 每个三元组 $(v, w, s)$ 携带原始 log probability 分数 $s$，构成集合 $T = \{(v, w, s) \mid v = \text{Transli}(w) \wedge w \in V^{\text{orig}}\}$。
- 转写后可能出现歧义：不同原词可能转写为同一拉丁串（如"太阳"和"太陽"均转写为"taiyang"）。

**阶段二：词表合并（Merge New Vocabulary）**
- 对于 1:1 对应关系：直接添加转写词及其分数。
- 对于歧义情况（多个 $w$ 映射到同一 $v'$），定义集合 $U(v') = \{(v, w, s) \in T \mid v = v'\}$，采用三种模式：
  - **Min-Merge**：取 $s'_{\min}(v') = \min_{(v,w,s)\in U(v')} s$，保留低频子词优先级，对 tail 语言更友好。
  - **Max-Merge**：取 $s'_{\max}(v') = \max_{(v,w,s)\in U(v')} s$，保留高频子词优先级，对 head 语言更优，整体效果最佳。
  - **Average-Merge**：取 $s'_{\text{avg}}(v') = \frac{\sum_{(v,w,s)\in U(v')} s}{|U(v')|}$，介于两者之间。

**阶段三：Embedding 初始化（Subword Embedding Initialization）**
- 创建新嵌入矩阵 $E^{\text{add}}$，与原始 $E^{\text{orig}}$ 拼接。
- 1:1 对应子词：直接复制 $E^{\text{add}}_{v'} = E^{\text{orig}}_{w'}$。
- 歧义子词按合并模式对应初始化：
  - Min-Merge：$E^{\text{add}}_{v'} = E^{\text{orig}}_{w^*}$，其中 $w^*$ 是取 min score 对应的原词。
  - Max-Merge：$E^{\text{add}}_{v'} = E^{\text{orig}}_{w^*}$，其中 $w^*$ 是取 max score 对应的原词。
  - Average-Merge：$E^{\text{add}}_{v'} = \frac{\sum_{(v,w,s)\in U(v')} E^{\text{orig}}_w}{|U(v')|}$，对所有对应原词 embedding 取平均。
- 最终嵌入矩阵为 $E^{\text{orig}}$ 与 $E^{\text{add}}$ 的拼接，确保 tokenizer 索引与新嵌入索引一致。

## 实验与结果
- **模型**：XLM-R（100 语言）、Glot500（>500 语言，XLM-R 继续预训练）、FURINA（Glot500 经转写对比学习微调）。
- **转写工具**：Uroman（通用规则转写工具）。
- **任务**：句子检索（SR-B、SR-T）、文本分类（Taxi1500、SIB200）、序列标注（NER、POS）；全部采用英语中心 zero-shot 跨语言设置。
- **关键结果（转写数据，Table 3）**：
  - XLM-R Max-Merge：SR-B all 从 12.5→14.6（+2.1），SR-T all 从 32.9→43.0（**+10.1**），Taxi1500 all 从 17.4→22.1（+4.7），SIB200 all 从 47.4→59.5（**+12.1**），NER all 从 46.3→50.5（+4.2），POS all 从 51.3→60.2（**+8.9**）。
  - Glot500 Max-Merge：SR-B all 从 32.7→35.4（+2.7），SR-T all 从 43.0→53.7（**+10.7**），Taxi1500 all 从 40.2→46.0（+5.8），SIB200 all 从 57.9→70.4（**+12.5**），NER all 从 54.0→70.4（**+16.4**），POS all 从 60.2→68.8（**+8.6**）。
  - FURINA Max-Merge：提升幅度相对较小（因 FURINA 已在预训练阶段接触转写数据）。
- **非转写数据（Table 4）**：改造模型在非转写数据上性能与原始 mPLM 几乎持平，有极微小下降（因拉丁文字符处理受到影响）。
- **三模式对比（Table 5）**：三种模式差异很小（因 91%-92% 新词为 1:1 无歧义），但 Max-Merge 整体最优，Min-Merge 对 tail 语言略有利。
- **脚本受益分析（Figure 2）**：对 Cyrl、Arab、Deva 等拼音文字的转写数据提升最大；Hani（汉字）和 Latn（拉丁文）提升最小。

## 相关工作脉络
1. **Dhamecha et al. (2021); Moosa et al. (2023)**：将数据转写后从头训练 mPLM；本文无需重新训练，直接复用已有 mPLM。
2. **Muller et al. (2021); Purkayastha et al. (2023)**：通过参数高效微调适配转写数据；本文完全零训练，仅需词表和 embedding 操作。
3. **Liu et al. (2024b) FURINA / Xhelili et al. (2024)**：将转写作为辅助输入进行对比学习微调以改善跨脚本对齐；本文聚焦于零训练基线构造，可与上述方法结合形成更强的两阶段方案。
4. **Kajiura et al. (2023)**：用领域语料替换子词做领域适应，需训练新 embedding；本文直接复制/平均原始 embedding，无需训练。
5. **Pfeiffer et al. (2021)**：为 unseen language 训练新 tokenizer 并微调；本文避免重新训练，仅修改词表结构。
6. **J et al. (2024) RomanSetu**：通过罗马化解锁 LLM 跨语言能力；与本文思路类似但针对 LLM 而非 mPLM，且涉及更多训练。

## 局限性与未来方向
- **转写与原文性能仍有差距**：因转写歧义和 tokenization 变化导致性能下降，需进一步微调或 continue pretraining 来缩小差距。
- **仅支持 SentencePiece Unigram tokenizer**：论文指出可扩展至其他 tokenizer（如 WordPiece 用频率替代分数），但未验证。
- **Uroman 通用转写工具存在缺陷**：如中文会丢失声调导致大量歧义，未来可探索更细粒度的语言特定转写工具。
- **未探索与后续训练方法的组合**：作者建议在 TRANSMI 基线基础上进行 continue pretraining 或 fine-tuning，作为社区未来工作。

## 研究启发与可借鉴点
1. **零训练词表扩展范式**：通过"转写→合并→初始化"三步构建基线的方法，可迁移至其他脚本相关语言任务（如梵语→天城文转写），为低资源语言提供低成本起点。
2. **歧义处理的三模式设计**：Min/Max/Average 三种策略简单有效，为后续在 tokenization 权重决策场景提供了可复用的设计范式。
3. **与后续微调方法的组合潜力**：TRANSMI 可作为 fine-tuning/continue pretraining 的优质初始化起点，值得探索"零训练基线+轻量微调"的两阶段 pipeline。
4. **脚本-受益关系的量化分析**：通过按原始脚本分组评估方法效果，为跨语言 NLP 的研究设计提供了可借鉴的评估维度。
5. **嵌入初始化策略的推广**：直接复制/平均原始 embedding 的思路可适用于其他需要扩展词表的场景（如领域术语注入、新语言适配），无需从头训练 embedding。

## 关键术语表
- **mPLM (multilingual Pretrained Language Model)**：在多语言语料上预训练的跨语言语言模型，如 XLM-R、Glot500。
- **TRANSMI (Transliterate-Merge-Initialize)**：本文提出的三阶段框架，用于将现有 mPLM 改造为处理转写数据的基线模型。
- **Transliteration (转写)**：将一种书写系统的文本转换到另一种书写系统（如汉字转写为拉丁字母），不涉及语义翻译。
- **Head/Tail Languages**：按是否在 XLM-R 中被覆盖来划分，head 语言为高资源语言，tail 语言为低资源语言。
- **Unigram Tokenizer (SentencePiece)**：基于语言模型的概率子词切分算法，为每个子词保留 log probability 分数。
- **Zero-shot Crosslingual Transfer (零样本跨语言迁移)**：在英语上微调模型，直接应用于其他语言测试，无需目标语言监督数据。
- **Uroman**：一种通用的规则转写工具，可将任何脚本转换为拉丁字母形式。

## 可复现要素
- **数据集**：Bible (SR-B)、Tatoeba (SR-T)、Taxi1500、SIB200、WikiANN (NER)、Universal Dependencies (POS)；均为公开数据集。
- **代码开源**：https://github.com/cisnlp/TransMI
- **模型**：XLM-R、Glot500、FURINA（均为公开）
- **关键超参**：fine-tuning 学习率 1e-5（分类）/2e-5（序列标注），batch size=8，gradient accumulation=2/4，Adam optimizer，最大训练 epoch=40/10
- **Uroman**：作为转写工具，为规则驱动，无需训练
