---
title: "Read-Before-Grounding-Scene-Knowledge-Visual-Grounding-via-M"
source: https://aclanthology.org/2025.coling-main.76.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:59:03"
field: "视觉语言定位"
keywords: ["Visual Grounding", "Scene Knowledge Visual Grounding", "SK-VG", "Large Language Model", "Zero-shot", "Multi-modal", "Text Compression"]
innovations: ["提出ReadVG零样本即插即用方法，通过LLM多步解析将长场景知识转化为简洁视觉描述", "两步解析架构（Reading Module + Visual Descriptor Generation）有效缓解长文本对多模态模型的负面影响", "系统验证了文本长度对MLLM性能的显著影响，并提出可迁移的缓解策略"]
benchmarks: ["SK-VG"]
---

# 论文速读：Read-Before-Grounding-Scene-Knowledge-Visual-Grounding-via-M

## 一句话总结
论文提出 ReadVG，一种零样本、即插即用的视觉定位方法，通过利用大语言模型（LLM）的语言理解能力，将冗长的场景知识文本逐步解析为目标类别、名称及简洁的视觉描述，从而显著提升多模态模型在 SK-VG（场景知识视觉定位）任务上的目标定位精度。

## 研究问题与动机
1. **现有 VG 数据集表达过于简单**：RefCOCO、RefCOCO+ 等数据集的查询语句直观简单，缺乏丰富的属性和空间关系信息，难以考察模型对复杂语义关系的理解能力。
2. **SK-VG 任务挑战严峻**：场景知识（K）包含大量实体关系、背景信息等冗长文本，查询语句侧重于实体间关系而非直接视觉特征，现有模型难以处理长文本与复杂语义关联。
3. **大 MLMM 在 VG 任务上表现不佳**：尽管 MLLM（如 GPT-4o、Shikra 等）具备强大的语言理解和推理能力，但因预训练数据缺乏接地（grounding）特定信息，其在 SK-VG 任务上几乎全部失效，且长文本对模型性能有显著负面影响。
4. **小参数端到端模型能力受限**：KeViLi 等专门面向 SK-VG 的端到端小模型在处理复杂场景文本时性能有限，过度依赖大规模训练数据。

## 核心贡献（创新点）
1. **提出 ReadVG 零样本即插即用框架**：无需对 grounding 模型进行额外训练或微调，即可应用于超长文本的 SK-VG 任务，与已有全监督方法（如 KeViLi FT）形成鲜明对比。
2. **两步解析算法（Reading Module + Visual Descriptor Generation）**：先通过 LLM 从场景知识中提取目标类别和名称，再基于名称生成简洁（≤8词）的视觉描述，与直接拼接原文本的方式本质不同，显著降低了长文本对 grounding 模型的干扰。
3. **系统验证了文本长度对多模态模型的负面影响**：通过对比 Q、Q+K 和 Ours 三种输入设置，量化了长场景知识导致的性能下降（Q+K 比 Q 下降最高达 43.3%），并证明本方法可将零样本性能差距缩小至仅 6.1%（相对于 KeViLi FT）。
4. **揭示了非 LLM-based 小模型经本方法后可超越大型 LLM-based 模型的潜力**：ONE-PEACE（非 LLM-based）在使用 ReadVG 后性能超过所有 LLM-based 方法，说明文本编码器大小并非视觉定位的决定性因素。

## 方法详解
方法整体分为两个模块，形成端到端的零样本流水线：

**（1）Reading Module（阅读模块）**
- 输入：场景知识 K + 查询 Q
- 目标：提取目标的类别 C（person 或 item）和名称 N
- 对于 person：以 Q 和 C 作为问题、K 作为上下文，通过 LLM 在场景知识中搜索匹配的实体名称（类似阅读理解）
- 对于 item：直接提取物品名称
- 形式化表达：
  - C = LLM(Q, K)
  - N = LLM(Q, K, C)，若 C = person；N = item name，若 C = item

**（2）Visual Descriptor Generation Module（视觉描述器生成模块）**
- 输入：目标类别 C、名称 N、场景知识 K、查询 Q
- 目标：生成简洁、信息密集的视觉描述 V
- 生成规范：聚焦目标自身外观特征（发色、性别、服装、配饰等）；避免提及名字、吸引力、眼色、体型；描述不超过 8 词且无标点
- 形式化表达：V = LLM(N, Q, K)

**最终流程**：将生成的视觉描述 V 与图像 I 输入 grounding 模型，预测目标边界框 B。

## 实验与结果
**数据集**：SK-VG 测试集，共 6,598 条样本，每图含 4-6 个物体，分为 Easy（45.89%）、Medium（27.71%）、Hard（26.40%）三个难度等级。

**评估基线**：
- 非 LLM-based：KeViLi（FT/PT/Zero-shot）、OFA、ONE-PEACE、UNINEXT、GroundVLP
- LLM-based：Shikra-7B、InternVL2-2B、GroundingGPT

**评估指标**：IoU ≥ 50% 的平均准确率（Acc_avg），按难度分层报告（Acc_e、Acc_m、Acc_h）。

**主要结果**：
- **KeViLi**：FT Acc_avg = 28.43；零样本 Q = 23.64，Q+K = 16.13；**ReadVG 达 26.69，差距仅 6.1%**（远超 Q+K 的 43.3% 下降）
- **OFA**：Q = 35.34，Q+K = 18.80；ReadVG = 40.12，Hard 级提升高达 **79%**
- **ONE-PEACE**：Q = 38.51，Q+K = 17.04；ReadVG = 44.44，Hard 级提升高达 **70.5%**
- **GroundingGPT**：Q = 40.22，Q+K = 32.83；ReadVG = 40.57
- **整体趋势**：Q+K 在所有模型上均出现大幅性能下滑（最长文本干扰），本方法在所有模型上均稳定提升

**消融实验**：移除 Reading Module 后几乎所有模型性能下降，证明两阶段解析的必要性；在 GroundingGPT 的 Easy 级上，无 Reading 的变体略优，说明简化描述可能损失部分细粒度信息（见 Figure 3）。

**跨 LLM 分析**：Qwen-turbo 略优于 GLM4-flash，但后者以更小参数和更快速度达到相近效果，证明方法的可迁移性。

## 相关工作脉络
1. **传统 VG 数据集（RefCOCO/RefCOCO+/RefCOCOg）**：以简单指代表达为主，缺乏复杂场景关系建模，本文指出其不足以评估深层次图文联合推理能力。
2. **KeViLi（Chen et al., 2023c）**：首个针对 SK-VG 任务的端到端模型，但依赖大量训练数据，零样本性能较差；本文方法无需微调即可逼近其 fine-tune 水平。
3. **MLLM-based 接地方法（Shikra、InternVL2、GroundingGPT）**：具备强语言理解但接地能力弱；本文通过 LLM 预处理弥补了这一缺陷，而非重新训练 grounding 模型。
4. **Set-of-Mark 等 Prompt-based 接地方法（Yang et al., 2023）**：通过在图像上添加标记辅助定位；本文从文本侧入手，通过多步解析压缩信息密度，路径不同但目标一致。
5. **长文本对 MLMM 的影响（Pope et al., 2023）**：本文验证了文本长度对多模态模型性能的负面影响，与前述工作结论一致，并提出了缓解策略。

## 局限性与未来方向
1. **依赖 LLM 文本理解能力的局限性**：面对非常规或高度复杂的语言表达时，Reading Module 可能产生不确定性错误。
2. **密集场景下实体混淆风险**：当图像中包含大量相似实体时，仅靠名称可能无法精确定位，存在性能瓶颈。
3. **缺乏领域微调**：未针对 SK-VG 特有的长文本场景知识进行 LLM 微调，可能限制对特定领域表达的适配能力。
4. **数据源单一**：实验数据主要来自电影场景，可能无法完全代表其他视觉定位场景（如真实世界导航、机器人操作等）。
5. **未来方向**：可探索 LLM 领域微调、结合 Set-of-Mark 等视觉提示技术、扩展至更多场景类型的泛化能力验证。

## 研究启发与可借鉴点
1. **"LLM 作为预处理模块"的设计范式**：将 LLM 强大的语言理解能力嵌入到视觉任务前，通过文本压缩和结构化提取减轻下游模型负担，该思路可迁移至其他需要处理长上下文的多模态任务（如 VQA、图像描述生成）。
2. **多步解析（Step-by-Step Parsing）策略**：将复杂推理任务拆解为"识别目标→生成描述→定位"多个子步骤，每步聚焦单一目标，避免了端到端模型在长文本上的注意力分散问题。
3. **零样本与微调性能的对比实验设计**：通过对比 Q、Q+K、Ours 三种输入设置，清晰揭示了文本长度对模型的影响机制，该实验设计可直接复用于其他视觉语言任务的诊断分析。
4. **跨模型普适性验证**：在非 LLM-based 和 LLM-based 两类模型上均验证有效性，证明了方法的不依赖特定架构，适合团队后续在不同 backbone 上快速集成测试。
5. **生成规范的工程细节值得借鉴**：对视觉描述的长度限制（≤8 词）、内容约束（避免名字、眼色等）等 prompt 工程技巧，可直接迁移到其他需要生成紧凑视觉描述的 task。

## 关键术语表
**Visual Grounding (VG)**：根据文本查询在图像中定位目标对象的视觉-语言基础任务。

**Scene Knowledge Visual Grounding (SK-VG)**：结合场景知识（实体关系、背景信息）的视觉定位任务，查询侧重实体间关系而非直接视觉特征。

**ReadVG**：本文提出的零样本即插即用方法，利用 LLM 进行多步解析辅助视觉定位。

**Reading Module**：ReadVG 的第一步，从场景知识中识别查询目标所属类别及具体名称。

**Visual Descriptor Generation Module**：ReadVG 的第二步，基于目标名称生成简洁的视觉描述（≤8词）。

**KeViLi**：面向 SK-VG 的端到端专用模型，fine-tune 后达到 28.43% Acc_avg，但零样本性能较差。

**GroundingGPT**：LLM-based 多模态接地模型，在 ReadVG 辅助下 Hard 级 Acc 达到 51.44%。

**OFA / ONE-PEACE**：非 LLM-based 多模态模型，经 ReadVG 处理后性能显著提升，Hard 级分别达 48.51% 和 53.44%。
