---
title: "PrahokBART-A-Pre-trained-Sequence-to-Sequence-Model-for-Khme"
source: https://aclanthology.org/2025.coling-main.87.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:38:40"
field: "低资源语言自然语言生成"
keywords: ["预训练模型", "高棉语", "序列到序列", "低资源语言", "文本生成", "机器翻译"]
innovations: ["首个高棉语专用轻量级PS2S模型，参数量仅为mBART50的1/3", "在预处理中集成归一化和分词模块，保留功能空格作为独立token", "系统评估语言学模块对无词边界语言的NLG任务影响"]
benchmarks: ["ALT", "Lr-sum"]
---

# 论文速读：PrahokBART-A-Pre-trained-Sequence-to-Sequence-Model-for-Khme

## 一句话总结
本文提出了首个专为高棉语（Khmer）设计的轻量级预训练序列到序列模型 PrahokBART，通过引入归一化和分词等语言学模块解决高棉语特有的书写系统问题（如无词边界、编码歧义、功能空格），在机器翻译、文本摘要和标题生成三个任务上显著优于多语言基线模型 mBART50。

## 研究问题与动机
- **低资源语言的专用模型缺失**：尽管 mBART50、mT5 等多语言模型支持高棉语，但参数量庞大导致低资源语言表示不足，而针对高棉语等独特书写系统的专用预训练模型仍然空白。
- **高棉语特有语言挑战被忽视**：高棉语无标准词边界（word boundaries）、存在编码歧义（encoding ambiguities），且空格在文本中具有功能性作用（functional spaces），这些特性在多语言模型中被忽略。
- **子词分词可能破坏语义单元**：无分词直接使用 Unigram tokenizer 会导致语义无关的子词合并（如将不相关的词片段组合），影响模型学习能力。
- **功能空格的生成质量影响文本自然度**：高棉语文本中功能空格的使用对可读性和自然度至关重要，但现有模型对此处理能力不足。

## 核心贡献（创新点）
- **首个高棉语专用 PS2S 模型**：提出 PrahokBART，是第一个针对高棉语设计的小型预训练序列到序列模型，结合归一化和分词模块，与 mBART50 相比参数量仅为 1/3（big 版）或 1/10（base 版）。
- **语言学模块的显式集成**：在预处理阶段引入归一化（移除不可见字符、统一编码）和基于 khmer-nltk 的分词，分词模块同时保留功能空格作为独立 token，这是与多语言模型处理方式的本质区别。
- **系统性的实验评估与洞察**：在机器翻译、文本摘要、标题生成三个任务上全面评估，证明专用模型在效率和生成质量（BLEU、ChrF、Rouge-L、COMET）上均优于 mBART50，并提供各模块的消融分析。
- **功能空格生成质量的量化分析**：首次系统评估模型生成功能空格的能力，发现分词模块通过将空格作为独立 token 显著提升了空格生成的准确性。
- **公开的资源与基线**：开源代码和模型权重（GitHub 和 HuggingFace），并构建了可作为高棉语 NLG 基准的数据集组合。

## 方法详解
**数据收集与清洗**：
- 数据来源：高棉语来自 Common Crawl（mC4、WMT2020）和 Wikimedia（Wikipedia、Wikibooks）；英语来自 Common Crawl（mC4）。
- 高棉语与英语数据比例约为 1:5（0.7B vs 3.5B tokens，共约 4.2B tokens），以平衡数据稀缺性。
- 过滤规则：字符数 < 10 或 > 20、功能空格比例 > 30%、数字比例 > 20%、emoji 比例 > 10%、标点比例异常、混合脚本 > 20%、目标语言概率 < 50%。

**预处理流程**：
1. **归一化（Normalization）**：
   - `rm_inv`：移除 29 种不可见字符（如零宽空格 U+200B、方向控制字符等）。
   - `enc_norm`：按 Hosken et al. (2022) 的规则统一编码，消除等价 Unicode 序列的歧义。
2. **分词（Word Segmentation）**：使用 khmer-nltk 工具，将文本切分为词序列，并保留功能空格作为独立 token。
3. **子词分词（Subword Tokenization）**：基于分词后的文本训练 Unigram tokenizer，词汇表大小 32k。

**模型架构**：
- 基于 BART 架构，使用 YANMTT 工具训练。
- Base 版：6 层 encoder/decoder，8 个 attention heads，hidden dim 512，intermediate dim 2048，总参数 62M。
- Big 版：12 层 encoder/decoder，16 个 attention heads，hidden dim 1024，intermediate dim 4096，总参数 211M。
- 最大序列长度 1024，超长文本按句子边界截断。
- 预训练：掩码 35% 的词，Poisson 分布采样 span 长度（λ=3.5），dropout 0.1，label smoothing 0.1，Adam 优化器（max lr 0.001，warmup 16k steps），batch size 4096 tokens，40 张 V-100 GPU 训练约 16 epochs。

## 实验与结果
**数据集**：
- 机器翻译（MT）：Asian Language Treebank (ALT)，评估 en→km 和 km→en 方向。
- 文本摘要（TextSum）：Lr-sum 数据集中的高棉语部分。
- 标题生成（HeadGen）：Lr-sum 数据集（摘要→标题）。

**评估指标**：BLEU、ChrF（MT）、Rouge-L（摘要/标题）、COMET（MT 神经评估）。

**主要结果（Table 2-3）**：
| 模型 | en→km BLEU | km→en BLEU | TextSum Rouge-L | HeadGen Rouge-L | COMET (en→km) | COMET (km→en) |
|------|-----------|-----------|-----------------|-----------------|---------------|---------------|
| Random | 1.13 | 9.06 | 10.67 | 11.10 | 70.51 | 72.41 |
| mBART50 | 19.47 | 19.47 | 25.38 | 22.15 | 74.71 | 78.47 |
| PrahokBART_base | 22.53 | 24.27 | 19.67 | 20.42 | 76.28 | 79.36 |
| PrahokBART_big | **23.70**† | **24.81** | **26.23** | **22.92** | **77.69**† | **82.00**† |

**关键结论**：
- PrahokBART_big 在所有任务上显著优于 mBART50（p < 0.01），且参数量仅为 mBART50 的 1/3（211M vs 611M）。
- PrahokBART_base 参数量仅为 mBART50 的 1/10，性能仍优于或接近 mBART50。
- 相比抽取式基线（Lead-3、LexRank），PrahokBART 在 TextSum 上大幅提升（Rouge-L 26.23 vs 7.85/7.38）。
- 数据规模扩展实验（Table 10）显示，增加预训练数据（从 39M 到 4.2B tokens）持续提升性能。

## 相关工作脉络
- **mBART50 / mT5**：多语言 PS2S 模型，覆盖 50+ 语言，但低资源语言表示不足；本文专注于单一语言的深度优化。
- **语言特定 PS2S 模型**：如 Barthez（法语）、BARTpho（越南语）、IndicBART（印度语言组）；本文填补了高棉语专用 NLG 预训练模型的空白。
- **Khmer NLU 预训练**：Jiang et al. (2021) 提出 Khmer BERT，但仅针对 NLU 任务；本文首次探索高棉语 NLG 预训练。
- **Tokenizer 设计**：Gow-Smith et al. (2022) 提出将空格作为独立 token 改善复杂词处理；本文在高棉语中验证该方法的有效性并扩展到 PS2S 模型。
- **低资源语言 NLG 基准**：IndicNLG、IndoNLG、AfriNLG 等；本文指出高棉语缺乏正式 NLG 基准，所构建数据集可作为基准。
- **编码歧义处理**：Hosken et al. (2022) 分析高棉语编码结构；本文将其应用于预训练数据清洗流程。

## 局限性与未来方向
- **语言覆盖局限**：模型仅支持高棉语和英语，词汇表局限于这两种语言，不适用于其他语言对（如泰语-高棉语翻译）。
- **数据规模受限**：当前预训练数据约 4.2B tokens，作者认为增加更多数据（如 Common Crawl 快照、内部数据集）可进一步提升性能。
- **模型规模受限**：根据 scaling law，若数据翻倍至 8B tokens，模型参数需增至约 400M 以充分利用数据。
- **TextSum/HeadGen 挑战**：模型在这些任务上的表现仍远低于 Oracle，说明高棉语摘要和标题生成仍是开放难题。
- **未来方向**：探索词汇适配（vocabulary adaptation）技术以扩展到其他语言对；增加预训练数据量并按 scaling law 扩大模型规模。

## 研究启发与可借鉴点
- **语言学模块在 PS2S 中的价值**：对于无词边界语言，分词预处理和归一化对下游 NLG 任务有显著提升，可作为类似语言（如泰语、老挝语）模型设计的参考。
- **功能空格作为独立 token**：将功能空格保留为独立 token 而非合并到子词中，有效提升了文本自然度，该方法可迁移至其他依赖空格语用的语言。
- **数据平衡策略**：高棉语与英语 1:5 的比例设计，解决了低资源语言数据稀缺问题，同时利用英语辅助学习，策略可复用。
- **高效预训练实践**：使用 YANMTT 工具和简化超参数配置，在有限计算资源下实现高效训练，适合资源受限团队参考。
- **评估框架构建**：整合 ALT 和 Lr-sum 数据集形成高棉语 NLG 基准，为后续研究提供了可复用的评估管道。

## 关键术语表
**PS2S（Pre-trained Sequence-to-Sequence）**：预训练序列到序列模型，通过在大规模文本上使用去噪目标预训练，再在下游任务微调。
**Functional Spaces（功能空格）**：高棉语中用于分隔短语或句子的空格，无强制书写规范但对文本可读性至关重要。
**Encoding Ambiguities（编码歧义）**：同一单词可由不同 Unicode 序列表示，导致文本不规范和模型学习困难。
**Unigram Tokenizer**：基于概率的 subword 分词方法，通过迭代删除最低概率子词构建词汇表。
**Fertility**：衡量分词器将文本切分为 subword 数量的指标，反映分词粒度。
**COMET**：基于 XLM-RoBERTa 的神经机器翻译评估指标，衡量翻译质量而非表面匹配。
**Alt Dataset（Asian Language Treebank）**：包含东南亚语言平行语料的高质量 MT 数据集。
**Lr-sum**：面向低资源语言的文本摘要数据集，包含多语言版本。

## 可复现要素
- **数据集**：ALT（公开）、Lr-sum（公开）、Common Crawl 和 Wikimedia 数据（公开但需自行下载处理）。
- **代码与权重**：代码开源（https://github.com/hour/prahokbart），模型权重公开（https://huggingface.co/prajdabre/prahokbart）。
- **关键超参**：词汇表大小 32k，掩码比例 35%，Poisson λ=3.5，dropout 0.1，label smoothing 0.1，max lr 0.001，warmup 16k steps，batch size 4096 tokens，训练 16 epochs，40×V-100 GPU。
- **工具**：YANMTT（训练）、khmer-nltk（分词与句子分割）、SentencePiece（subword 训练）、SacreBLEU（评估）。
