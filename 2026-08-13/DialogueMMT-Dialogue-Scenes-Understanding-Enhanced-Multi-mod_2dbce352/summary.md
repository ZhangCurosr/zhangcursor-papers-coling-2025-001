---
title: "DialogueMMT-Dialogue-Scenes-Understanding-Enhanced-Multi-mod"
source: https://aclanthology.org/2025.coling-main.170.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:02:53"
field: "多模态情感分析与对话理解"
keywords: ["对话情绪识别", "多模态大模型", "多任务指令微调", "链式思维", "视觉语言模型"]
innovations: ["首次将多模态大型视觉-语言模型引入ERC任务并进行跨数据集统一微调", "提出多任务指令微调框架（ASD+FER+DDP辅助）增强对话场景理解", "引入链式思维策略（先极性后情感）提升细粒度情感分类性能"]
benchmarks: ["MELD", "EmoryNLP", "IEMOCAP"]
---

# 论文速读：DialogueMMT: Dialogue Scenes Understanding Enhanced Multi-modal Multi-task Tuning for Emotion Recognition in Conversations

## 一句话总结
本文提出DialogueMMT框架，首次将多模态大型视觉-语言模型引入对话情绪识别（ERC）任务，通过统一多数据集多任务指令微调与链式思维策略，同时捕捉视觉场景与文本对话依赖，在MELD、EmoryNLP、IEMOCAP三个基准数据集上均取得SOTA结果。

## 研究问题与动机
1. **多模态视觉场景复杂性与对话语境依赖难以同时处理**：现有ERC方法要么仅利用纯文本建模时序/ utterance关系，忽略视觉模态；要么仅改进融合方法使用面部信息，忽略对话上下文依赖。
2. **跨数据集泛化能力未验证**：现有SOTA ERC模型在各单一数据集上分别训练和测试，缺乏跨域统一训练的验证。
3. **辅助任务协同不足**：已有研究多仅引入单一辅助任务（如说话人识别或Emotion Shift Detection），未同时利用多种对话相关辅助任务的协同增益。

## 核心贡献（创新点）
1. **首个多模态LLM驱动的ERC框架**：首次探索多模态大型语言模型用于ERC任务，并使用统一ERC数据集进行跨域联合微调。与已有ERC方法本质区别在于利用Video-LLM的通用理解能力而非专用架构。
2. **多模态多任务指令微调**：联合主任务ERC与三个辅助任务（ASD主动说话人检测、FER面部表情识别、DDP对话话语解析）进行统一指令微调，与单辅助任务方法相比能更全面地增强对话场景理解。
3. **链式思维策略（CoT）**：在ERC任务中先预测情感极性（3类）再细化到具体情感类别，降低分类难度。与直接分类方法本质区别在于分步推理提升了难区分情感的识别率。
4. **统一多数据集训练验证跨域能力**：将MELD、EmoryNLP、IEMOCAP三个数据集合并训练并语义对齐标签，首次系统性验证多模态LLM在跨域ERC中的泛化性能。

## 方法详解
- **视觉编码器**：采用预训练CLIP ViT-L/14（336分辨率）提取视频帧特征，输入固定8帧，每张图resize至336×336。
- **STC连接器**：引入预训练的Spatial-Temporal Convolution连接器，包含两个空间交互模块和一个时空聚合模块，通过3D下采样操作压缩时空token，并在上下采样前后插入强卷积块补充信息损失（公式1：f_V = STC(H_V), H_V = V(X_V)）。
- **投影层**：可训练投影矩阵W将视觉特征f_V映射为与语言嵌入同维度的视觉query向量Z_V（公式2：Z_V = W·f_V）。
- **主干模型**：以Mistral-7B-Instruct为LLM骨干，通过FlashAttention + BFloat16加速训练。
- **多任务指令微调**：
  - **主任务ERC**：限制历史对话轮数为m（最终选取m=20），为每轮对话添加位置索引<u_i>，引入CoT策略（先极性后情感）。
  - **辅助任务ASD**：基于VideoLLaMA2生成instruction数据，识别当前说话人的性别、年龄、外观、穿着、动作等信息。
  - **辅助任务FER**：结合额外面部表情库增强对面部情绪的感知。
  - **辅助任务DDP**：采用图指令微调，将16种话语依赖类型转化为code-like格式（entity_list + triple_list），增强文本层面对话结构理解。
- **训练配置**：LoRA微调（rank=128, α=256），AdamW优化器，初始学习率2e-5，全局batch size=16，2×48G Nvidia A40 GPU训练，最大序列长度2048。

## 实验与结果
- **数据集**：MELD（TV《老友记》，7/3类标签，含视频）、EmoryNLP（《老友记》，7/3类标签，仅文本）、IEMOCAP（演员表演对话，6/3类标签，含视频），详细统计见Table 1。
- **评估指标**：准确率（Acc）、加权F1（w-F1）、宏平均F1（macro-F1）及每类F1。
- **主要结果（Table 2）**：
  - MELD：Acc 71.19%，w-F1 70.66%（vs SACL-LSTM 67.51%/66.45%，提升+3.68%/+4.21%）
  - EmoryNLP：Acc 45.02%，w-F1 40.36%
  - IEMOCAP：Acc 72.58%，w-F1 72.71%
  - 三数据集平均：Acc 62.93%，w-F1 61.24%（vs SACL-LSTM平均59.60%/58.44%，提升+3.33%/+2.80%）
- **细粒度结果（Table 3）**：MELD上Fear提升7.45%、Anger提升9.86%、Disgust提升6.69%；EmoryNLP上Sad提升8.07%；IEMOCAP上Joyful提升7.19%。
- **消融实验（Table 7）**：移除CoT导致性能下降最大（Avg w-F1从61.24%降至58.18%），移除DDP次之（降至60.10%），移除STC对MELD影响显著（70.66%→69.39%）。
- **跨域评估（Table 8）**：仅在MELD上微调的DialogueMMT_Single在IEMOCAP上w-F1下降46.06%，验证了统一多数据集微调的必要性。

## 相关工作脉络
1. **VideoLLaMA2（Cheng et al., 2024）**：DialogueMMT的STC连接器与视觉-语言对齐方案源自此工作，但本文将其引入ERC这一全新下游任务。
2. **MultiEMO（Shi & Huang, 2023）**：多模态注意力相关性感知融合框架，但仅聚焦融合策略，未利用大型语言模型的通用推理能力。
3. **FacialMMT（Zheng et al., 2023）**：引入帧级FER辅助任务的单辅助 multitask方法，本文扩展为多任务协同（ASD+FER+DDP）。
4. **DualGATs（Zhang et al., 2023a）**：纯文本图注意力模型建模话语结构与speaker-aware context，本文在其基础上增加视觉模态与LLM推理能力。
5. **MiniGPT-4 / Video-LLaVA（Liu et al., 2023; Lin et al., 2023）**：早期多模态LLM工作，本文采用更先进的VideoLLaMA2架构并在ERC任务中验证多任务指令微调的有效性。
6. **UniMSE（Hu et al., 2022c）**：将音频和视觉信号注入T5的 multimodal统一框架，但依赖特定架构设计，泛化性不及LLM-based方法。

## 局限性与未来方向
1. **未利用音频模态**：仅融合视觉与文本，忽略了语音语调等音频情绪线索。
2. **辅助任务精度瓶颈**：辅助任务（尤其是DDP话语解析和视觉情感检测）的准确性直接影响主任务，当前教师模型生成的辅助数据存在误差传递风险。
3. **类别不均衡问题**：统一数据集导致部分细粒度情感类别（如Peaceful、Fear）样本占比小，易被误分为Neutral，数据分布优化有待改进。
4. **未来可探索**：引入音频模态、改进辅助任务标注质量、针对长对话上下文压缩策略优化。

## 研究启发与可借鉴点
1. **CoT策略在细粒度情感分类中的迁移价值**：先预测极性再细化情感的两步推理可复用于其他多分类NLP任务（如立场分析、意图识别），值得在本团队相关任务中验证。
2. **多模态LLM + 多任务指令微调的范式**：将多个对话相关辅助任务（ASD、FER、DDP）统一纳入指令微调框架，为多模态对话理解提供了可复用的训练范式。
3. **STC连接器在视频-语言对齐中的有效性**：相较于简单线性投影，STC连接器保留了时空顺序信息，该设计可迁移至其他视频理解任务（如视频问答、动作识别）。
4. **统一多数据集联合训练提升跨域泛化**：在ERC、视觉问答、对话生成等不同任务中均可探索跨数据集统一微调策略，避免单一数据集过拟合。
5. **图话语结构的code-like格式化**：将依存图转化为结构化文本表示的方法可复用于其他需要理解对话结构的任务（如对话续写、话轮转换预测）。

## 关键术语表
- **Emotion Recognition in Conversations (ERC)**：对话情绪识别，在多轮对话上下文中识别每个话语的情绪类别。
- **Active Speaker Detection (ASD)**：主动说话人检测，从多人场景中识别当前正在发言的人物。
- **Facial Expression Recognition (FER)**：面部表情识别，从人脸图像中识别情绪表情类别。
- **Dialogue Discourse Parsing (DDP)**：对话话语解析，分析话语之间的依赖关系并构建对话结构图。
- **Chain-of-Thought (CoT)**：链式思维，引导模型分步推理（先极性后情感）以提升复杂分类性能的策略。
- **Spatial-Temporal Convolution (STC) Connector**：时空卷积连接器，保留视频时空顺序的模态对齐模块。
- **Low-Rank Adaptation (LoRA)**：低秩适配，通过在LLM中注入低秩矩阵实现参数高效微调的技术。
- **Unified ERC Dataset**：统一ERC数据集，将多个ERC数据集合并并语义对齐标签后的联合训练集。

## 可复现要素
- **数据集**：MELD、EmoryNLP、IEMOCAP（均为公开数据集）；辅助任务使用STAC、Molweni、AffectNet（公开）。
- **代码**：已开源，地址 https://github.com/he2720/DialogueMMT
- **权重**：骨干模型Mistral-7B-Instruct（开源），CLIP ViT-L/14（开源），STC连接器基于VideoLLaMA2（开源）。
- **关键超参**：LoRA rank=128, α=256；学习率=2e-5；输入帧数=8；最大对话轮数m=20；最大序列长度=2048；batch size=16；训练设备=2×48G A40 GPU。
