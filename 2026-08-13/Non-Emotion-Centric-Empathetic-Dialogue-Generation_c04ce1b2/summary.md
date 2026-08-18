---
title: "Non-Emotion-Centric-Empathetic-Dialogue-Generation"
source: https://aclanthology.org/2025.coling-main.66.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:45:18"
field: "共情对话生成"
keywords: ["共情对话生成", "对比学习", "常识知识", "非情绪中心", "开放域对话"]
innovations: ["用对比学习替代显式情绪识别以规避级联错误", "双通道实体+社会常识注入框架", "四类负样本联合训练提升多样性与 informativeness"]
benchmarks: ["EMPATHETICDIALOGUES"]
---

# 论文速读：Non-Emotion-Centric-Empathetic-Dialogue-Generation

## 一句话总结
本文提出 NEC（Non-Emotion-Centric empathetic dialogue generation）框架，通过对比学习替代显式情绪识别，并结合上下文敏感的实体与社会常识知识，有效缓解细粒度情绪识别误差引发的级联错误，提升共情对话生成的质量与多样性。

## 研究问题与动机
1. **细粒度情绪识别性能差**：EMPATHETICDIALOGUES 包含 32 种情绪类别，现有方法准确率低于 50%（Gao et al., 2021 等），导致后续生成出现级联错误。
2. **情绪识别与对话历史冲突**：错误识别的情绪会污染对话历史中的信息，使模型难以生成既贴切又 informative 的回复（如高频输出 "I'm sorry to hear that"）。
3. **已有方法过度依赖情绪识别**：大多数先验工作（Multi-TRS、MoEL、MIME、EmpDG 等）以显式情绪识别为核心，一旦识别出错便影响整条生成链路。
4. **生成结果缺乏多样性与信息量**：高频通用回复频繁出现，导致对话连贯性与人机交互体验下降。

## 核心贡献（创新点）
1. **以对比学习替代显式情绪识别**：通过构建不同情绪的负样本约束模型输出符合正确情绪方向，从根本上规避情绪识别误差的级联传播，与以往基于识别→生成的两阶段方法本质不同。
2. **引入上下文敏感的实体常识（Entity Commonsense）**：利用 COMET 结合对话历史提取实体三元组，并通过 BERT 评分器筛选与回复最相关的 top-3 三元组，使模型更深入理解对话上下文中的实体情感含义。
3. **融合社会常识（Social Commonsense）注入解码器**：通过 MPNet 计算语义相似度选取与上下文最相关的社会常识（xEffect/xReact/xIntent/xNeed/xWant），在 Decoder 中通过多头注意力层实现知识融合，与 CEM、DCKS 等静态常识注入方式形成差异。
4. **设计四类对比负样本联合训练**：构建不同情绪样本（D）、自生成样本（y'）、批次内样本（B）、高频句样本（H）四种负样本，同时惩罚高频通用回复并引导情绪表达，显著提升多样性与 informativeness。
5. **提出 Sentence-level 多样性度量 Sent-Std**：弥补现有 n-gram 级别多样性指标的不足，从句子层面衡量生成回复的分布均匀性。

## 方法详解
**整体架构**：以 BART 为骨干编码器-解码器，结合 COMET 常识抽取与对比学习策略。

**实体常识集成（Section 3.2）**：
- 使用对话历史 $U$ 作为头，对每个实体 $ent_j$ 和 COMET 关系 $rel_i$（ObjectUse/AtLocation/MadeUpOf/HasProperty）构造输入 $IE_i = U \oplus P_c \oplus ent_j \oplus [rel_i]$，通过 COMET 抽取三元组集合 $K_{all}$。
- 用 BERT 评分器构造正/负样本对，从 $K_{all}$ 中选取 top-$m$ 个得分最高的三元组 $K_E$。
- Encoder 输入：$I_{enc} = U \oplus [SEP] \oplus P_r \oplus L(K_E)$，输出隐状态 $\mathbf{z_x} = BART\_Encoder(I_{enc})$。

**社会常识注入（Section 3.3）**：
- 以最后一句对话 $u_{n-1}$ 输入 COMET，获取五类社会常识关系（xEffect/xReact/xIntent/xNeed/xWant）的候选集合。
- 用 MPNet 计算上下文与候选知识的余弦相似度：$Score(ks_i) = \frac{\mathbf{h}_{ctx} \cdot \mathbf{h}_{ks,i}}{\|\mathbf{h}_{ctx}\| \cdot \|\mathbf{h}_{ks,i}\|}$，每类关系取最高分。
- Decoder 通过额外多头注意力层融合 $K_S$：$\mathbf{z_y} = BART\_Decoder(K_S, \mathbf{z_x}, Y)$。

**对比学习训练（Section 3.4）**：
- 总损失：$L_{total} = L_{NLL} + L_{CL}$
- NLL 为标准语言建模损失：$L_{NLL} = -\sum_{t=1}^{N} \log P(y_t | I_{enc}, K_S, y_{<t})$
- 对比损失：$L_{CL} = \sum_{(y^+, y^-) \in \mathcal{P}} \max\{0, D(\mathbf{z_x}, y^+, y^-) + \xi\}$，其中 $D = \cos(\mathbf{z_x}, \mathbf{z_{y^-}}) - \cos(\mathbf{z_x}, \mathbf{z_{y^+}})$，$\xi = \gamma \cdot (rank(y^-) - rank(y^+))$，$\gamma = 0.01$。
- 四类负样本：D（不同情绪样本）、$y'$（beam search 自生成）、B（同批次其他样本）、H（训练集高频句）。
- **推理时扩展 Beam Search**：最终得分 $S = \alpha S_{sim} + (1-\alpha) S_{lm}$，其中 $S_{sim} = \cos(\mathbf{z_x}, \mathbf{z_{\hat{y}}})$，$\alpha = 0.7$。

## 实验与结果
**数据集**：EMPATHETICDIALOGUES（25K 开放域共情对话），按 8:1:1 划分 train/valid/test。

**评估指标**：自动评测（PPL↓、BLEU-1~4↑、Distinct-1/2↑、Sent-Std↓）、人工评测（Coherence/Empathy/Informativeness/Continuity，1-5 分）、高频句占比（Table 3）。

**基线模型**：Transformer、Multi-TRS、MoEL、MIME、EmpDG、CEM、DCKS（SOTA）。

**主要结果**（Table 1）：
- NEC 在所有自动指标上全面优于 7 个基线。
- 相较 SOTA 模型 DCKS：BLEU-1 从 21.73 → 23.62（+8.7%），BLEU-4 从 4.09 → 4.78（+16.9%），Dist-2 从 9.61 → 13.91（+44.8%），PPL 从 16.08 降至 12.89（-19.8%），Sent-Std 从 36.89 降至 18.61（-49.6%）。
- 高频句占比：NEC 仅 20.32%，显著低于 DCKS（34.41%）和 MIME（54.58%）。
- 人工评测（Table 2）：NEC 在 Coh./Emp./Inf./Cont. 四个维度均最高（4.34/4.32/4.08/4.12），Fleiss kappa = 0.40。
- **消融实验（Table 4）**：移除任一模块均有性能下降；移除不同情绪负样本（w/o DiffEmo）后人工共情评分显著降低（Table 2），验证其对共情质量的必要性。

## 相关工作脉络
1. **Multi-TRS / MoEL / MIME / EmpDG**（情绪识别驱动方法）：均以显式情绪识别为核心，NEC 通过对比学习替代这一环节，规避级联错误。
2. **CEM**（Sabour et al., 2022）：利用 affective 和 cognitive 常识增强共情生成，但未解决情绪识别误差问题；NEC 在此基础上引入对比学习约束与更精细的实体常识筛选。
3. **DCKS**（Cai et al., 2023，当前 SOTA）：基于 BART 动态选择常识，但仍依赖情绪状态驱动选择；NEC 进一步通过对比负样本减少对情绪标签的依赖。
4. **ConceptNet / ATOMIC / COMET**：常识知识库的前置工作；NEC 将 COMET 用于实体常识与社会常识双通道抽取，并结合评分器/相似度筛选实现更精准的常识注入。
5. **Cont（An et al., 2022）**：对比神经文本生成方法；NEC 借鉴其对比损失设计但扩展了四类负样本并应用于共情对话场景。

## 局限性与未来方向
1. **训练效率低**：对比学习引入多类负样本导致训练时间显著增加（论文 self-limitation 明确提及）。
2. **常识抽取依赖外部模型**：COMET 的抽取质量直接影响常识知识的有效性，且需额外推理开销。
3. **仅在一个数据集上验证**：仅在 EMPATHETICDIALOGUES 上评估，泛化性待进一步验证。
4. **高频句仍有改进空间**：与人类回复相比，模型仍倾向使用"共情通用句+信息句"的机械模式（Error Analysis 提及）。
5. **未来方向**：提升训练效率、集成 BERTScore 做更精确的语义聚类分析。

## 研究启发与可借鉴点
1. **对比学习替代显式标签**：用对比学习规避下游任务中中间模块（如情绪识别）的误差传递，这一思路可迁移至其他需要多阶段推理的对话生成任务（如立场生成、情感支持对话）。
2. **双通道常识融合策略**：实体常识（细粒度、上下文敏感）+ 社会常识（粗粒度、关系导向）的组合方式，可作为知识增强的通用范式应用于其他 grounded dialogue 任务。
3. **Sent-Std 多样性度量**：句子级多样性指标弥补了 n-gram 指标的不足，可推广至开放域对话系统的多样化评测体系构建。
4. **四类负样本构造思路**：不同情绪/自生成/批次/高频句的负样本设计逻辑，可适配到其他需要兼顾语义丰富性与表达多样性的文本生成任务。
5. **解码时融合相似度与生成概率**：推理阶段 $\alpha S_{sim} + (1-\alpha) S_{lm}$ 的加权策略，是一种轻量级的知识引导解码方式，可在不增加训练负担的前提下提升生成质量。

## 关键术语表
**Empathetic Dialogue Generation**：共情对话生成，指系统根据对话历史理解说话人的情感状态并生成具有安慰、同情、理解等共情性质的回复。
**Contrastive Learning**：对比学习，通过拉近正样本对、推远负样本对来学习语义表示的技术，本文用于替代显式情绪识别。
**COMET**：Atomic 2020 知识库的常用接口工具，可从文本中提取实体常识（如 ObjectUse/AtLocation）和社会常识（如 xReact/xIntent）。
**Sent-Std**：本文提出的句子级多样性指标，衡量生成句子频率分布的标准差，值越小表示分布越均匀、多样性越高。
**Cascading Errors**：级联错误，指上游模块（如情绪识别）的错误传递给下游模块（如响应生成），导致误差累积放大的现象。
**Entity Commonsense vs. Social Commonsense**：实体常识关注具体事物的属性与用途，社会常识关注事件中人物的反应、意图与需求。
**Beam Search with Similarity Augmentation**：在标准 Beam Search 基础上引入余弦相似度得分加权，平衡语言模型概率与上下文契合度。

## 可复现要素
- **数据集**：EMPATHETICDIALOGUES，公开可获取（https://github.com/facebookresearch/empathetic-dialogues）
- **代码**：论文声明代码将在 https://github.com/huangfu170/NEC-empchat 开源
- **骨干模型**：Base BART（HuggingFace 版本）
- **常识工具**：COMET 2020 版本
- **关键超参**：学习率 0.00001、batch size 训练=8/测试=16、$\alpha = 0.7$、$\gamma = 0.01$、最大解码步数 30、test 阶段 top-3 实体常识三元组
- **硬件**：NVIDIA A100-PCIE (40GB)
