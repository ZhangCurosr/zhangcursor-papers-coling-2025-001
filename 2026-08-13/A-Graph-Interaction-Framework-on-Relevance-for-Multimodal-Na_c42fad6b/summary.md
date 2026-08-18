---
title: "A-Graph-Interaction-Framework-on-Relevance-for-Multimodal-Na"
source: https://aclanthology.org/2025.coling-main.82.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:58:27"
field: "多模态自然语言处理"
keywords: ["多模态命名实体识别", "多图像MNER", "人类偏好强化学习", "图Transformer", "相关性判别", "跨模态交互"]
innovations: ["首次将RLHF引入多模态NER的相关性判断，替代不可靠的CLIP相似度方法", "构建基于相关性的异构图并用图Transformer实现多图间信息交互", "每层独立Prompt设计实现细粒度跨模态信息交互"]
benchmarks: ["MNER-MI", "MNER-MI-Plus"]
---

# 论文速读：A-Graph-Interaction-Framework-on-Relevance-for-Multimodal-Na

## 一句话总结
本文针对多图文场景下的多模态命名实体识别（MNER）任务，提出一种基于**人类偏好强化学习**的图交互框架（GIFR），通过判别器量化图文相关性并构建异构图，结合图Transformer实现图文信息的有效交互，在 MNER-MI 与 MNER-MI-Plus 数据集上均取得 SOTA。

## 研究问题与动机
1. **多图像引入噪声问题**：社交媒体帖子中包含大量与命名实体无关的图像，直接引入会干扰模型判断，且随图像数量增加噪声愈发严重。
2. **现有相似度过滤方法失效**：基于 CLIP 的图文相似度计算在实际中不可靠（如图1b所示，相关性图像的 CLIP 相似度仅 0.18~0.19），且 CLIP 训练目标不包含命名实体、对非标准语言泛化性差。
3. **人类直觉难以建模**：人类可凭直觉判断图像与实体是否相关，但这种复杂抽象的能力难以用传统规则或相似度量捕获。
4. **多模态交互机制不足**：现有方法（门控机制、GCN）在多图像场景下无法充分建模图像间的结构关系，且 GCN 存在过平滑与过压缩问题。

## 核心贡献（创新点）
1. **首次将人类偏好强化学习引入 MNER 领域的图文相关性判断**：与以往直接计算 CLIP 相似度的方法本质不同，该方法通过人类标注的排序偏好训练判别器，赋予每个图像任务导向的相关性分数，而非通用语义相似度。
2. **基于相关性分数的异构图建模**：将图像与文本显式建模为异构图，以图像与空白图像的相关性差值作为边权重，区别于以往仅用门控或 GCN 隐式融合的方法。
3. **引入图 Transformer 替代传统图神经网络**：利用 Transformer 架构避免 GCN 的过平滑/过压缩问题，并将图结构信息（节点度）融入位置编码与注意力计算，实现更充分的多图像间信息交互。
4. **每层独立 Prompt 设计实现细粒度跨模态交互**：BERT 的每个 Transformer 层配有独立的视觉 Prompt，使不同层可与不同视觉信息交互，区别于单一层级注入视觉信息的已有方案。

## 方法详解
GIFR 由三个核心模块组成：

### 4.1 基于相关性的图像判别器（Relevance-based Image Discriminator）
- 使用 **CLIP-vit-base-patch32** 分别编码文本与图像，提取 `[EOS]` 与 `[CLS]` 的最终层表示 $T_e \in \mathbb{R}^{d_t}$ 和 $V_e \in \mathbb{R}^{d_v}$。
- 投影至同一维度后拼接，经 MLP 输出标量奖励 $r$。
- 对图像对 $(I_A, I_B)$（$I_A$ 为人类排序更靠前、相关性更高者），采用排序损失：
$$L_D = -\frac{1}{|D|} \sum_{(I_A, I_B) \in D} \log(\sigma(r(I_A) - r(I_B)))$$
- 测试时将图像按相关性排序，插入空白图像作为阈值：空白图像之前的图像视为相关，之后的视为无关。

### 4.2 异构图构建
- 将同一图文对内的图像与文本设为节点，构建异构图。
- 图像 $I_i$ 与文本节点之间的边权重由相关性差值经 sigmoid 计算：
$$R_i = \sigma(r(I_i) - r(I_{blank}))$$

### 4.3 模态内交互（Intra-modal Interaction，图 Transformer）
- 使用 **ViT-base-patch16** 提取图像表征 $V_i$。
- 位置编码：利用节点度 $deg(V_i)$ 映射为可学习向量 $z_{deg(V_i)}$ 叠加至节点特征：
$$h_{V_i} = V_i + z_{deg(V_i)}$$
- 注意力计算中融入边权重 $R$（元素乘）：
$$U = \text{softmax}\left(\frac{Q_{H_V} K_{H_V}^T}{\sqrt{d_v}}\right) \odot R \; V_{H_V}$$

### 4.4 模态间交互（Inter-modal Interaction）
- 使用 **BERT-base** 编码文本，每个 Transformer 层 $l$ 配有独立的视觉 Prompt：
$$P^l = W_p^l U^T, \quad 1 \leq l \leq L$$
- 对 Prompt 做线性变换生成补充 Key/Value：
$$K_P^l = W_k^l P^l, \quad V_P^l = W_v^l P^l$$
- 第 $l$ 层的自注意力在 Key/Value 中拼接视觉信息：
$$H^l = \text{softmax}\left(\frac{(Q^l)^T [K_P^l, K^l]}{\sqrt{d_t}}\right)[V_P^l, V^l]^T$$
- 最终输出经 CRF 解码，损失为负对数似然：
$$L_N = -\frac{1}{|D_{model}|} \sum_{k=1}^{N} \log P(y_k | H_k^L)$$

## 实验与结果
- **数据集**：MNER-MI（训练6,856/验证860/测试860，共24,021张图像）与 MNER-MI-Plus（训练10,229/验证1,583/测试1,583，含单图样本，共28,840张图像）。
- **评估指标**：Precision、Recall、F1。
- **基线**：包括 BiLSTM-CRF、BERT（文本单模态）、GVATT、UMT、UMGF、VisualPT-MoE、MAF 等多模态模型，以及 LLM（GPT-4、MiniGPT-4）。
- **主要结果**：

| 模型 | MNER-MI F1 | MNER-MI-Plus F1 |
|---|---|---|
| VisualPT-MoE-MI | 76.62 | 82.70 |
| TPM-MI | 77.32 | 83.42 |
| **GIFR（本文）** | **78.10** | **83.97** |

- **最强结果**：GIFR 在 MNER-MI 上 F1 达 **78.10**，较次优基线 TPM-MI 提升 **+0.78 点**；在 MNER-MI-Plus 上 F1 达 **83.97**，较次优基线 TPM-MI 提升 **+0.55 点**。
- **消融实验**：移除判别器（w/o D）导致 MNER-MI 下降 0.60、MNER-MI-Plus 下降 0.52；移除图结构（w/o G）下降最显著（MNER-MI 降 1.87 点），验证了各模块的有效性。

## 相关工作脉络
1. **Xu et al. (2022b) MAF**：使用对比学习对齐图文表示，依赖 CLIP 计算相似度过滤无关图像；本文指出 CLIP 对命名实体相关性的判断不可靠，转而用人类偏好强化学习建模相关性。
2. **Zhang et al. (2021) UMGF**：采用视觉定位（visual grounding）关联文本与图像区域；本文关注整图与文本的相关性判断，粒度不同，且进一步建模多图间交互。
3. **Zhao et al. (2022)**：用 GCN 建模图文关系图；本文认为 GCN 存在过平滑/过压缩问题，改用图 Transformer 以保留更多结构信息。
4. **Huang et al. (2024) TPM-MI**：将多图建模为帧并使用 Prompt 交互；但未显式过滤无关图像，本文通过相关性判别器主动识别并加权处理。
5. **Liu et al. (2020) RLHF**：人类反馈强化学习用于文本摘要；本文首次将该范式迁移至多模态 NER 领域的相关性判断任务。
6. **Lample et al. (2016) HBiLSTM-CRF**：文本单模态 NER 基线；本文表明引入多图像后 BERT 单模态基线（F1=71.22/78.26）远优于传统 BLSTM 方法。

## 局限性与未来方向
- **粒度不足**：当前相关性判断聚焦于整张图像与文本的全局相关性，未考虑图像内部无关区域的细粒度过滤（论文自述 Limitations 部分）。
- **依赖人类标注**：判别器训练需人工对图像按相关性排序，数据收集成本较高。
- **未来可探索**：引入区域级（region-level）相关性判断以实现更细粒度的视觉信息筛选；探索自动化排序标签生成以减少对人标注的依赖。

## 研究启发与可借鉴点
1. **RLHF 用于模态过滤**：将人类偏好强化学习迁移至多模态 NER 的相关性判断，为其他多模态任务（如视觉问答、图像描述生成）中的模态选择问题提供了新思路。
2. **图 Transformer 在多模态图建模中的应用**：利用节点度映射位置编码、将边权重融入注意力计算的设计，可有效替代 GCN 解决过平滑问题，适用于任意多图多文本交互场景。
3. **每层独立 Prompt 的跨模态交互设计**：BERT 多层独立注入视觉 Prompt 的机制，使不同抽象层级的文本表征可与不同粒度的视觉信息交互，该设计可复用于其他多模态序列标注任务。
4. **空白图像作为相关性阈值**：引入空白图像作为相关性排序的中间参考点，是一种简洁有效的二值划分策略，可推广至其他需要图像筛选的任务。
5. **与团队方向的结合机会**：本工作对"相关性判断+图交互"的框架可作为后续研究的基础模块，尤其适用于多视图/多源数据的实体识别、事件抽取等任务。

## 关键术语表
- **MNER（Multimodal Named Entity Recognition）**：结合文本与图像模态的命名实体识别任务，利用图像补充文本语义歧义。
- **RLHF（Reinforcement Learning from Human Feedback）**：通过人类偏好信号（奖励）训练模型的强化学习方法，本文用于学习图文相关性判断。
- **GIFR（Graph Interaction Framework on Relevance）**：本文提出的图交互框架，核心思想是通过相关性判别构建异构图并进行图Transformer交互。
- **Heterogeneous Graph（异构图）**：包含多种类型节点的图，本文中文本节点与图像节点共存，边权重由相关性决定。
- **Graph Transformer**：将标准 Transformer 扩展至图结构数据的方法，本文采用节点度位置编码与边权重注意力改进版。
- **Visual Prompt**：将视觉表征投影后作为补充输入注入 BERT 各层，实现跨模态信息交互。
- **MNER-MI / MNER-MI-Plus**：包含多图像帖子和单图像帖子的多模态 NER 基准数据集。
- **BIO 标注体系**：NER 常用序列标注格式，B（Begin）、I（Inside）、O（Outside）标识实体边界。

## 可复现要素
- **数据集**：MNER-MI 与 MNER-MI-Plus（Huang et al., 2024 发布），论文未明确声明开源链接，需向原作者获取。
- **代码/权重**：论文未提及开源代码或预训练权重。
- **关键超参**：优化器 AdamW，学习率范围 [1e-5, 8e-5]，batch size [8, 32]，训练轮数 [10, 25]；图像预处理尺寸为 224×224，划分为 7×7 个 32×32 区域。
- **基础模型**：CLIP-vit-base-patch32（判别器）、BERT-base（文本编码）、ViT-base-patch16（图像编码）。
