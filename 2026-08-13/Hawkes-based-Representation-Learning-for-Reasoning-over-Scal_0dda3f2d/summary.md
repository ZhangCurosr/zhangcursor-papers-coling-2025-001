---
title: "Hawkes-based-Representation-Learning-for-Reasoning-over-Scal"
source: https://aclanthology.org/2025.coling-main.198.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:33:27"
field: "时序知识图谱推理"
keywords: ["Temporal Knowledge Graph", "Hawkes Process", "Graph Convolutional Network", "Community Detection", "Scale-free Network", "Reasoning", "FiLM Hypernetwork"]
innovations: ["将Hawkes指数衰减核嵌入RGCN消息传递以建模事件影响的时间衰减", "用FiLM超网络实现查询感知的条件解码以缓解长尾分布偏见", "基于扩展Louvain社区检测的图卷积嵌入初始化增强社区结构利用"]
benchmarks: ["ICEWS14", "ICEWS18", "WIKI", "YAGO"]
---

# 论文速读：Hawkes-based-Representation-Learning-for-Reasoning-over-Scal

## 一句话总结
本文提出 **HERLN（Hawkes process-based Evolutional Representation Learning Network）**，通过联合建模时序知识图谱（TKG）的社区结构、无标度（长尾分布）特性和事件影响时间衰减性，同时在实体预测和关系预测任务上超越所有现有基线。

## 研究问题与动机
1. **社区结构被忽视**：真实网络（如社交网络）具有显著社区结构，现有TKG推理模型未显式利用该结构性信息。
2. **事件影响衰减未建模**：事件的影响力随时间衰减是真实世界普遍现象，但既有模型多假设时间序列等间隔（如RE-NET、RE-GCN），未考虑连续时间下的渐弱效应。
3. **长尾分布导致偏见**：TKG中实体和关系呈无标度分布，高频实体/关系在训练中占据优势，推理时模型偏向常见演化模式，低频事件被压制。
4. **Hawkes过程与图结构的割裂**：已有Hawkes方法（如GHNN、GHT）未充分融合图拓扑信息，而图神经网络方法未引入时间衰减机制。

## 核心贡献（创新点）
1. **首次联合建模TKG的社区结构、无标度特性和时间衰减性**——与以往仅关注单一特性的模型（如RE-GCN忽略时间衰减、GHNN忽略社区结构）形成本质区别。
2. **设计基于Hawkes过程的关系图卷积网络（HRGCN）**——将指数衰减核函数作为消息传递的时序权重，替代传统RGCN的等距假设，使历史信息按时间距离自适应衰减。
3. **提出查询感知的条件解码器（FiLM超网络）**——通过查询向量动态生成ConvTransE的参数偏移和缩放因子（α、β），使解码器能针对稀有演化模式自适应调整，而非全局共享固定参数。
4. **社区检测驱动的嵌入初始化模块**——扩展Louvain算法至多关系图，基于社区子图上的GCN生成初始嵌入，强化社区内实体表示相似性，辅助推理。

## 方法详解
HERLN由三个模块串联构成：

**模块一：嵌入初始化（Embedding Initializing Module）**
- 在输入TKG上运行扩展的多关系Louvain社区检测算法，计算每个关系的模块度 $Q_r$（公式1），迭代合并社区直至 $\Delta Q$ 不再变化（公式2）。
- 在社区子图上运行GCN（公式3）：$h_i = \sigma\left(\sum_{j\in\mathcal{N}_i}\frac{1}{|\mathcal{N}_i|}Wh_j^{init}\delta(c_i,c_j) + W_0h_i^{init}\right)$，其中 $\delta(c_i,c_j)$ 为社区指示函数，输出初始嵌入矩阵 $H_C$。

**模块二：演化编码（Evolution Encoding Module）**
- **HRGCN（公式7）**：核心更新方程为 $h_o^t = \sigma\left(W_1 h_o + \sum_{s,r,t'\in\mathcal{F}_o^t}\frac{1}{|\mathcal{F}_o^t|}W_r(h_s+h_r)\widetilde{\kappa}(t-t')\right)$，其中 $\widetilde{\kappa}(t-t') = \frac{\exp(-\delta(t-t'))}{\sum_{t''}\exp(-\delta(t-t''))}$ 是对归一化的指数衰减核，$\delta$ 为可学习参数。
- **门控融合（公式8-10）**：用全局图嵌入 $h_g$ 生成门控权重 $\gamma$，平衡演化信息与原始信息：$H^t = \gamma * H_T^t + (1-\gamma) H_C$。

**模块三：条件解码（Conditional Decoding Module）**
- 通过FiLM超网络（公式12-14）根据查询 $(s_q, r_q, t_q)$ 的嵌入生成缩放因子 $\alpha^{(s_q,r_q,t_q)}$ 和偏移因子 $\beta^{(s_q,r_q,t_q)}$，对ConvTransE的原始参数 $\theta$ 进行变换：$\theta^q = (\alpha+1)\odot\theta + \beta$。
- 使用变换后的参数 $\theta^q$ 计算候选实体条件强度：$\lambda_o = \text{ConvTransE}([h_{s_q}^{t_q}, h_{r_q}]; h_{o_q}^{t_q}; \theta^q)$（公式15），取最高分实体作为预测结果。

**学习目标（公式16）**：实体预测视为多分类任务，使用交叉熵损失 $\mathcal{L}_e = -\sum_{t_q}\sum_{\mathcal{F}_{t_q}}\sum_{k=1}^K y_k \log p(o_k|s_q,r_q,t_q)$。

## 实验与结果
- **数据集**：ICEWS18、ICEWS14（时间跨度24小时）、WIKI、YAGO（时间跨度1年），共4个标准TKG基准。
- **基线模型**：10个模型，包括RE-NET、RE-GCN、EvoKG、CENET、HGAT、TiPNN（embedding-based）、TG-Tucker、TLogic（path-based）、GHNN、GHT（Hawkes-based）。
- **评估指标**：MRR、Hits@1/3/10。
- **实体预测最强结果**：
  - ICEWS18：MRR 31.33（+1.01 vs TiPNN）、Hits@1 21.93
  - ICEWS14：MRR 43.94（+2.74 vs RE-GCN）、Hits@1 34.62
  - WIKI：MRR 79.10（+5.11 vs TiPNN）、Hits@1 74.92
  - YAGO：MRR 84.47（+4.17 vs TiPNN）、Hits@1 80.31
- **关系预测最强结果（MRR）**：ICEWS18 51.47、ICEWS14 50.55、WIKI 98.50、YAGO 98.54，均大幅领先（最高提升约10% MRR）。
- **关键发现**：HERLN在长时序跨度（WIKI/YAGO，1年）上提升更显著，而在短时序（ICEWS，24小时）上提升有限，印证了时间衰减建模的价值随时间跨度增大而增强；消融实验证实四个模块各自贡献为正（见原表3）。

## 相关工作脉络
1. **RE-GCN / RE-NET / EvoKG**：使用RNN建模时序演化，隐含等间隔假设，无法捕捉非均匀时间戳下的事件影响衰减；HERLN用HRGCN+指数核替代RNN，直接建模连续时间衰减。
2. **GHNN（Han et al., 2020）**：首次将Hawkes过程引入TKG推理，但未利用图结构信息，仅做点过程建模；HERLN将其与RGCN消息传递深度结合，实现结构+时序联合编码。
3. **GHT（Sun et al., 2022）**：用Temporal Transformer捕获长短期信息，但同样未考虑事件影响的逐时衰减特性；HERLN的 $\widetilde{\kappa}$ 函数显式编码衰减机制。
4. **CENET / HiSMatch**：通过历史词汇生成或背景图匹配建模演化，但未利用社区结构信息；HERLN在社区子图上的GCN初始化为后续演化编码提供更具语义凝聚性的起点。
5. **xERTE / HiSMatch**：编码时间戳并与实体嵌入拼接，仅学到"不同时间间隔有不同影响"的离散差异，而非渐进衰减的连续函数；HERLN用指数核给出平滑的衰减曲线。
6. **TITer / DREAM**：基于强化学习的路径搜索方法，聚焦局部子图，长程推理效果受限；HERLN的全局图消息传递更适合长期演化模式的捕捉。

## 局限性与未来方向
1. **未见实体问题**：测试集中存在仅在测试期出现的实体（unseen entities），模型无法为其分配足够信息以生成有效嵌入，这是TKG外推任务的普遍难题。
2. **短时序场景增益有限**：ICEWS系列（24小时窗口）上提升幅度小于WIKI/YAGO，说明在极短时域内时间衰减信号的边际效益较弱，如何在小时间窗上进一步提升尚待探索。
3. **社区检测的稳定性**：Louvain算法在大规模动态图上的效率与稳定性有待进一步优化，多关系扩展可能引入额外的计算开销。
4. **未来方向（作者自述）**：开发针对未见实体的嵌入学习方法，提升模型在新实体出现时的泛化能力。

## 研究启发与可借鉴点
1. **Hawkes核函数作为图消息传递的时序权重**：将 $\exp(-\delta(t-t'))$ 归一化后嵌入GNN消息聚合（公式7）的设计简洁且可微，可迁移至任意图演化建模任务（如动态图链接预测、事件推荐），替代RNN/LSTM的离散时序处理。
2. **FiLM超网络实现查询感知的参数自适应**：用查询向量动态调制解码器参数的思路（公式12-14），可直接复用于知识图谱补全、关系预测等需要区分不同查询模式的场景，避免长尾分布下的 biased prediction。
3. **社区结构辅助的嵌入初始化策略**：先检测社区再用社区子图GCN生成初始嵌入的方法，可有效提升社区密集型图的表征质量；可推广至社交网络分析、合作网络研究等场景。
4. **长时序 vs 短时序的分析范式**：论文通过对不同时间跨度数据集上的表现差异来论证模块价值（Section 5.2.1），这种跨数据集异质性分析可为模型设计提供更有力的证据支撑。
5. **组合式创新机会**：将HERLN的HRGCN与社区初始化模块结合，或可将FiLM解码器与对比学习（如CENET的思路）融合，探索"社区感知+时间衰减+查询自适应"三位一体的新推理框架。

## 关键术语表
**Temporal Knowledge Graph (TKG)**：包含时间戳的动态多关系知识图谱，事实表示为四元组 (主体, 关系, 客体, 时间)。
**Hawkes Process**：一种自激点过程，通过条件强度函数建模事件的历史依赖性，近期事件对当前事件发生率的影响更大，随时间指数衰减。
**Scale-free Network**：无标度网络，节点度服从幂律分布，少数节点（枢纽）连接极多，大多数节点连接稀少，呈现长尾特性。
**Community Structure**：图网络中节点倾向于与同组节点形成密集连接的结构特征，组内连接密度远高于组间。
**FiLM (Feature-wise Linear Modulation)**：一种通过查询向量线性调制特征层参数的机制，可实现条件化的特征变换。
**Hyper Network**：生成其他网络参数的网络，在此用于根据查询动态生成解码器的专属参数。
**Extrapolation Reasoning**：时序知识图谱推理的外推设定，预测未来时间点（$t > t_K$）缺失的事实。
**ConvTransE**：将TransE距离度量与一维卷积结合的解码函数，用于计算候选实体的得分。

## 可复现要素
- **数据集**：ICEWS14、ICEWS18、WIKI、YAGO（均为公开基准数据集）。
- **代码**：已开源，地址 https://github.com/WisdomMLlab/HERLN。
- **关键超参**：实体嵌入维度 $d_e=200$、关系嵌入维度 $d_r=200$、HRGCN层数=2、dropout=0.2、优化器Adam、学习率=0.001、ConvTransE解码器50个 $2\times3$ 卷积核；$\delta$ 为可学习衰减参数（论文未明确给出初始值，需参考代码）。
- **环境**：PyTorch + DGL，Nvidia GeForce Titan GPU。
