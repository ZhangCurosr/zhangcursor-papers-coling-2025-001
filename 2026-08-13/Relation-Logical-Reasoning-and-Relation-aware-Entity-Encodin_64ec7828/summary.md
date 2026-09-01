---
title: "Relation-Logical-Reasoning-and-Relation-aware-Entity-Encodin"
source: https://aclanthology.org/2025.coling-main.88.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:39:00"
field: "时序知识图谱推理"
keywords: ["Temporal Knowledge Graph", "Relation Logical Reasoning", "Entity Encoding", "Knowledge Graph Completion", "Link Prediction"]
innovations: ["提出关系逻辑推理模块，从实体对间关系推理路径学习不同关系在时序层面的潜在逻辑关联", "设计关系感知实体编码模块，通过注意力机制学习时间特定的实体嵌入", "采用乘积融合关系级与实体级贡献分，结合幂函数时间衰减实现外推预测"]
benchmarks: ["ICEWS14", "ICEWS18", "ICEWS0515", "WIKI", "YAGO"]
---

# 论文速读：Relation-Logical-Reasoning-and-Relation-aware-Entity-Encodin

## 一句话总结
本文提出 RLEE 模型，通过**关系逻辑推理模块**从实体对间的关系推理路径中学习不同关系在时序层面的潜在逻辑关联，以及**关系感知实体编码模块**利用注意力机制学习时间特定的实体嵌入，从而显著提升时序知识图谱外推链接预测的性能。

## 研究问题与动机
1. **忽视时序层面关系的逻辑关联性**：现有图结构 TKGR 方法（CyGNet, CENET, HGLS, EvoExplore 等）关注的是单一实体在固定关系下的变化，忽视了相同实体对在**不同时间戳**下多条关系之间存在的隐含逻辑关联。
2. **忽视实体对关系的依赖性**：已有方法为实体学习**静态通用**表示，但同一实体在不同关系下应扮演不同角色（如 Obama 在 "criticize" 和 "endorse" 关系下的语义应不同），实体表示应随查询关系动态变化。
3. 如何在低维向量空间中高效编码时序信息，同时挖掘关系的逻辑关联与实体-关系的动态交互，是 TKGR 的两大核心挑战。

## 核心贡献（创新点）
1. **关系逻辑推理模块**：通过实体对间的关系推理路径挖掘关系间潜在的时序逻辑关联，并以关系嵌入空间距离隐式表达逻辑强度——与 TECHS、DaeMon 等依赖显式规则/路径搜索的方法本质不同，无需硬约束路径枚举。
2. **关系感知实体编码模块**：引入关系感知注意力机制，为实体子图邻居按与查询关系的逻辑相关性分配不同权重，学习查询关系特定的实体嵌入——与 RE-GCN、CEN 等静态实体聚合方法相比，有效避免了无关子图信息的干扰。
3. **双模块乘积融合预测**：将关系级贡献分与实体结构级贡献分相乘整合，并在时间维度引入幂函数衰减加权历史时段——相比简单加法融合，显著提升了推理精度。
4. 在五个主流数据集（ICEWS14/18/0515, WIKI, YAGO）上全面超越 SOTA 基线，并在关系推理迁移实验中验证了跨数据集的泛化能力。

## 方法详解
模型采用双层表示架构：**实体层**与**关系层**，分步推理：

**① 时间编码（周期+非周期）**
- 分别用余弦函数和 tanh 函数建模事件的周期性与非周期性：
  - $\mathbf{v}_{\triangle t}^p = \cos(\beta_t \triangle t + \phi_c)$
  - $\mathbf{v}_{\triangle t}^{np} = \tanh(\gamma_t \triangle t + \phi_t)$
  - $\mathbf{T}_{\triangle t} = \mathbf{v}_{\triangle t}^p + \mathbf{v}_{\triangle t}^{np}$

**② 关系逻辑推理（Relation Layer）**
- 关系时间增强：$\mathbf{r}_{j, t_q} = \mathbf{r}_{j, t_\tau} + \mathbf{T}_{\triangle t}$
- 关系推理路径 $pa(r_j^{t_\tau}, r_q^{t_q})$ 的置信度由点积计算：$con = \mathbf{r}_{j, t_q} \cdot \mathbf{r}_{q, t_q}$
- 关系级贡献分：对所有历史路径置信度求和 $score_r^{t_\tau} = \sum_{r_j \in R_{s \to o_i}^{t_\tau}} con(pa(...))$

**③ 关系感知实体编码（Entity Layer）**
- 基于 ω 层 RGCN 聚合邻居，注意权重由查询关系与邻居关系的余弦相似度经 softmax 计算：$att_{r_q, r_j}^{t_\tau} = \frac{\exp(\cos(\mathbf{r}_{j,t_\tau}, \mathbf{r}_{q,t_\tau}))}{\sum \exp(\cos(...))}$
- 激活函数为 RReLU，最终加时间编码得动态实体嵌入
- DistMult 三元组打分：$score_e^{t_\tau} = \sigma(\langle \mathbf{e}_{s,r_q}^{t_\tau}, \mathbf{r}_{q,t_\tau}, \mathbf{e}_{o_i,r_q}^{t_\tau} \rangle)$

**④ 融合与预测**
- 乘积融合：$score_{o_i}^{t_\tau} = score_r^{t_\tau} \cdot score_e^{t_\tau}$
- 时间衰减加权：$W_d(t_q, t_\tau) = (t_q - t_\tau)^{-\gamma}$
- 最终得分：$score(o_i|s,r_q,t_q) = \sum_{\tau=q-len}^{q-1} W_d \cdot score_{o_i}^{t_\tau}$
- 取 argmax 输出预测实体

**⑤ 训练损失**
- SoftMarginLoss：$L = \sum \log(1 + \exp(-y \cdot score))$，正样本 y=1，负样本 y=-1

## 实验与结果
**数据集**：ICEWS14、ICEWS18、ICEWS0515、WIKI、YAGO（5 个主流公开 TKG 数据集，含 6869~23033 实体、10~256 关系）

**评估指标**：MRR、Hits@1/3/10（time-aware filtering）

**主要结果**（相对当前 SOTA 的绝对/相对提升）：
| 数据集 | MRR | H@1 | 相对 MRR 提升 |
|--------|-----|-----|--------------|
| ICEWS14 | **52.63%** | **39.53%** | +12.65%（vs DLGR 46.72%） |
| ICEWS0515 | **56.84%** | **44.37%** | +7.55%（vs DLGR 未报告但优于 HiSMatch 52.85%） |
| YAGO | **92.43%** | **91.02%** | +0.37%（vs TiPNN 92.06%） |
| WIKI | **85.53%** | **81.65%** | +2.49%（vs TiPNN 83.04%） |

**消融实验结论**：
- 去除任一模块（w/o R 或 w/o E）MRR 下降约 2~5 个百分点
- 乘积融合（RLEE）显著优于加法融合（RLEE-Add，如 ICEWS14: 52.63 vs 48.81）
- 去除关系注意力权重下降约 2%
- 去除时间编码下降约 5%

**迁移实验**：从 YAGOs 学习的关系逻辑迁移到 YAGO 上，LIMP > 90%，证明关系推理模块具有跨数据集泛化能力。

## 相关工作脉络
1. **DLGR (Xiao et al., 2024)**：采用双解耦表示学习+自监督，与 RLEE 均聚焦于关系-实体分离建模，但 DLGR 缺乏显式的关系逻辑推理路径。
2. **TECHS (Lin et al., 2023)**：基于可解释外推的时间逻辑图网络，通过路径搜索提取逻辑规则；RLEE 不依赖显式路径枚举，直接在嵌入空间隐式学习逻辑关联。
3. **DaeMon (Dong et al., 2023)**：利用时间线关系自适应捕捉时序路径；与 RLEE 类似关注关系演进，但 DaeMon 侧重于时序路径的自适应提取，不涉及关系间逻辑推理。
4. **HiSMatch (Li et al., 2022)** & **EvoExplore (Zhang et al., 2022)**：基于历史结构匹配与局部全局演化建模实体；与 RLEE 的关键区别在于二者均未建模关系间的逻辑关联。
5. **CyGNet (Zhu et al., 2021)**：基于序列复制生成网络统计实体频率；完全忽略关系的逻辑时序关联。
6. **RPC (Liang et al., 2023)**：利用关系相关性与周期事件，与 RLEE 在时间编码思想上有部分相似，但 RPC 未设计关系逻辑推理模块。

## 局限性与未来方向
1. **稀疏关系场景受限**：实验发现 ICEWS18 数据集上模型表现明显下降，原因是该数据集实体众多但关系路径置信度低，关系逻辑关联难以有效学习。
2. **仅适用于同关系集合的跨数据集迁移**：迁移实验在 YAGO/YAGOs 上进行，但未测试关系集合完全不同的数据集间的迁移效果。
3. **时间编码依赖周期性假设**：周期性和非周期性编码分别用 cos 和 tanh 建模，对于复杂多周期混合事件可能需要更灵活的时间建模。
4. **路径搜索范围受限于历史子图半径**：推理路径仅基于直接相连的关系，更长的路径或间接推理未被考虑。

## 研究启发与可借鉴点
1. **关系嵌入空间的逻辑关联建模思路**：将关系间逻辑关联转化为嵌入空间中的距离度量（点积），是一种简洁且可微的有效范式，可迁移至静态 KG 的关系推理任务。
2. **关系感知注意力用于实体编码**：为不同关系赋予不同权重的实体编码策略，值得在多关系 KBQA、推荐系统等领域尝试。
3. **乘法融合 vs 加法融合的实验设计**：消融实验通过对比乘积与加法融合验证了模块间的强相关性，这种对比实验设计值得借鉴。
4. **跨数据集关系逻辑迁移评估**：LIMP（Logical Inference Migration Performance）指标的提出为关系推理模块的泛化能力提供了新的评估范式。
5. **周期/非周期双分支时间编码**：分别用 cos 和 tanh 建模两类时间模式，设计思想可用于其他时序建模任务。

## 关键术语表
**Temporal Knowledge Graph (TKG)**：将时间信息纳入传统知识图谱，以四元组 (主体, 关系, 客体, 时间戳) 表示事实的动态知识图谱。
**Relation Logical Reasoning**：通过挖掘实体对间关系推理路径，学习不同关系在时序层面的潜在逻辑关联。
**Relation-aware Entity Encoding**：在实体嵌入学习中引入关系感知注意力机制，使实体表示随查询关系动态调整。
**Relation Inference Path**：连接实体对在历史时间戳 $t_\tau$ 和查询时间戳 $t_q$ 上的两条关系构成的路径 $(r_j^{t_\tau}, r_q^{t_q})$。
**DistMult**：一种三元组打分函数，通过张量分解（三线性点积）衡量三元组的合理性，计算高效且可解释性强。
**Time-aware Filtering**：在评估时仅过滤与查询 (s,r) 在同一时间戳出现的错误实体，而非所有历史出现的实体，更接近真实场景。
**LIMP (Logical Inference Migration Performance)**：衡量关系逻辑推理模块跨数据集迁移能力的指标，即跨数据集推理得分与直接推理得分的比值。
**RGCN (Relational GCN)**：面向多关系图的图卷积网络，对不同关系类型使用不同的变换矩阵进行消息聚合。

## 可复现要素
- **数据集**：ICEWS14、ICEWS18、ICEWS0515、WIKI、YAGO，均为公开数据集
- **代码开源情况**：论文未明确说明开源，部分基线（CyGNet、CENET、DaeMon、EvoExplore、HiSMatch、RE-GCN、TiTer、xERTE）使用开源代码复现；部分基线（DLGR、TiPNN、RPC、TECHS、CluSTeR）因未开源而采用论文报告结果
- **关键超参**：embedding 维度 200、学习率 0.001、RGCN 层数 ω=2、dropout=0.2、历史窗口 len=10、时间衰减指数 γ=0.8、训练 100 epochs、早停 patience=10
- **硬件**：单卡 Tesla T4 GPU 16GB
- **训练损失**：SoftMarginLoss
