---
title: "FedMKT-Federated-Mutual-Knowledge-Transfer-for-Large-and-Sma"
source: https://aclanthology.org/2025.coling-main.17.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:31:54"
field: "联邦大语言模型"
keywords: ["Federated Learning", "Large Language Models", "Small Language Models", "Knowledge Distillation", "Model Heterogeneity", "Token Alignment", "Parameter-Efficient Fine-Tuning"]
innovations: ["提出双向相互知识转移框架实现LLM与SLM同步增强", "设计基于MinED的token对齐解决异构tokenizer不匹配", "引入DualMinCE选择性知识融合避免有害知识干扰"]
benchmarks: ["RTE", "WIC", "BoolQ", "CQA", "ARC-E", "ARC-C", "S-NI", "DialogSum"]
---

# 论文速读：FedMKT-Federated-Mutual-Knowledge-Transfer-for-Large-and-Sma

## 一句话总结
论文提出 FedMKT，一个面向大语言模型（LLM）与小语言模型（SLM）的参数高效联邦相互知识转移框架，通过最小编辑距离（MinED）进行双向 token 对齐与选择性知识融合，实现服务器 LLM 与客户端 SLM 的同步性能增强。

## 研究问题与动机
- 现有联邦 LLM 研究主要聚焦于客户端间协同微调同质 LLM，或仅从服务器 LLM 单向向下游 SLM 转移知识，缺乏服务端与客户端模型间的**双向互促**机制。
- LLM 与 SLM 之间存在显著的**模型异构性**，尤其是 tokenizer 不匹配导致输出 logit 分布难以直接对齐，阻碍了有效的知识传递。
- 在异构环境中，部分客户端的知识可能对服务器或其他客户端产生负面影响，需**选择性**提取有益知识以避免有害干扰。
- 隐私保护与通信效率的权衡：共享完整模型参数开销巨大，而仅共享公开数据集上的 loss-logit 对可在保护隐私的同时降低通信负担。

## 核心贡献（创新点）
- **联邦相互知识转移框架**：首次实现服务器 LLM 与客户端 SLM 之间的双向知识流动，同时提升双方性能；与已有工作（如 LLM2SLM）的本质区别在于双方模型均参与更新。
- **双向 Token 对齐机制**：基于 MinED 构建词汇映射表并利用动态规划对齐异构 tokenizer 的输出；与直接拼接 logits 的方法相比，可精确处理 1-to-1、1-to-N 等多种对齐形态。
- **DualMinCE 选择性知识转移**：通过比较 loss 值从多方 logits 中筛选对目标模型最有利的知识；与平均聚合策略不同，该机制能主动过滤有害或无关知识。
- **参数高效联邦训练设计**：冻结基础模型参数，仅更新服务器与客户端的 LoRA 适配器，大幅降低计算与通信开销；与全参数微调方案相比，成本仅为 0.12%（以 OPT-1.3B 为例）。

## 方法详解
- **问题设定**：1 个服务器持有 LLM $f_\psi$，K 个客户端各持有一个 SLM $g_{\phi_k}$；客户端拥有私有数据 $\mathcal{D}_k$，服务器与客户端共享公开数据集 $\mathcal{D}_p$。
- **参数高效微调**：客户端在本地数据 $\mathcal{D}_k$ 上训练 LoRA 适配器 $\theta_k$（冻结 $\phi_k$），服务器在公开数据 $\mathcal{D}_p$ 上同时优化本地任务损失与来自客户端的知识蒸馏损失（冻结 $\psi$，更新 $\omega$）。
- **双向 Token 对齐（MinED）**：对于任意 LLM token，在 SLM 词汇表中寻找 MinED 最小的 token 作为映射目标（若有多个则取字典序最小）；利用动态规划确定句子级别的最优匹配路径，并将 LLM 的 Top-K logit 映射至 SLM 的对应位置。
- **Selective Mutual Knowledge Transfer（DualMinCE）**：
  - 客户端上传基于 $\mathcal{D}_p$ 计算的损失-logit 对集合 ${\cal S}_k = \{l_k^i, p_k^i\}$。
  - 服务器对每个样本 $x^i$ 选择 loss 最小的客户端 logit $p_{k^*}^i$；若其损失 $l_{k^*}^i$ 小于服务器本地 LLM 的损失 $l_{\text{local}}^i$，则将其加入选择性知识集 $\tilde{\cal S}_0$。
  - 服务器利用 $\tilde{\cal S}_0$ 计算知识蒸馏损失 $\mathcal{L}_{\text{KD}}^f$，并与本地任务损失 $\mathcal{L}_{\text{FT}}^f$ 加权合并：$\mathcal{L}_2 = \lambda \mathcal{L}_{\text{FT}}^f + (1-\lambda) \mathcal{L}_{\text{KD}}^f$（论文中 $\lambda=0.9$）。
  - 服务器将更新后的知识集 ${\cal S}_0$ 下发给客户端；客户端执行反向 token 对齐（LLM→SLM）并重复类似的选择与蒸馏过程，更新客户端 SLM 的适配器 $\theta_k$。

## 实验与结果
- **数据集与任务**：6 个 QA 数据集（RTE、WIC、BoolQ、CQA、ARC-E、ARC-C）与 2 个指令跟随数据集（S-NI、DialogSum）；评价指标为 Accuracy（QA）与 Rouge-L（指令跟随）。
- **模型配置**：服务器使用 LLaMa2-7B；客户端在异构设置下分别部署 GPT-2-xlarge (1.5B)、OPT-1.3B、Bloom-1.1B、LLaMa2-1.3B。
- **基线方法**：Centralized（集中式微调）、Zero-Shot、Standalone（独立微调）、FedAvg（仅同构）、LLM2SLM（仅一对一单向转移）。
- **异构设置结果**：
  - 客户端 SLM：FedMKT 在所有 4 个 SLM 上均超越 Zero-Shot 与 Standalone。以 RTE 任务为例，GPT-2-xlarge 相对 Zero-Shot 提升 34%、相对 Standalone 提升 7%。
  - 服务器 LLM：RTE 任务上达到 82.3，约为 Centralized（85.9）的 96%，显著优于 Zero-Shot（63.2）。
- **同构设置结果**：
  - S1（LLaMa2-7B vs 4×LLaMa2-1.3B）：服务器 CQA 达到 68.8（Centralized 69.5）。
  - S2（LLaMa2-7B vs 4×OPT-1.3B）：服务器 CQA 达到 71.3，**超过** Centralized（69.5），为最强结果之一。
  - 客户端：FedMKT 在 CQA 任务上相对 FedAvg 提升 4%（LLaMa2-1.3B）与 5%（OPT-1.3B）。
- **一对一设置结果**：服务器性能与 Centralized 持平或略优；客户端 SLM 超越 Standalone 与 LLM2SLM。
- **主要结论**：FedMKT 在不同设置下均能同时增强 LLM 与 SLM，且在部分任务上超越集中式微调。

## 相关工作脉络
- **FedMD / FedET**：基于知识蒸馏的异构联邦学习，仅聚合客户端 logits 构建全局 logits，未涉及 LLM 与 SLM 之间的双向互促。
- **Deep Mutual Learning / PFML / FedLoRA**：互学习框架，通常在每个客户端内部构建大小模型对，而非跨服务器‑客户端架构。
- **FedAvg**：标准联邦平均，要求所有客户端模型架构相同，无法处理异构场景。
- **LLM2SLM**：单向知识转移（LLM → SLM），服务器 LLM 不参与更新，与本文的双向 mutual transfer 形成鲜明对比。
- **FedPETuning / Federated Adapter Tuning / FATE-LLM**：将 PEFT 方法引入联邦 LLM 训练，但聚焦于客户端间协同微调同质 LLM，未考虑服务端大模型与客户端小模型的互补增强。

## 局限性与未来方向
- **隐私保护缺乏理论保证**：仅共享公开数据集上的 loss-logit 对比共享梯度更安全，但理论上仍可能存在信息泄露风险，有待进一步研究。
- **未考虑恶意客户端（spammer）**：有害客户端可能破坏服务器 LLM 与其他客户端 SLM 的性能，检测 spammers 是重要的未来方向。
- **计算与存储约束**：受限于实验算力，未探索更大规模的 LLM，后续需在效用与效率之间进一步优化权衡。

## 研究启发与可借鉴点
- **双向 token 对齐机制**可迁移至其他异构模型联邦学习任务（如视觉‑语言多模态模型），解决 tokenizer/embedding 不匹配的通用难题。
- **DualMinCE 选择性知识融合策略**适用于任何需要筛选有益知识的联邦蒸馏场景，避免平均聚合带来的负迁移。
- **LoRA 与联邦学习结合的低成本验证路径**：冻结主体参数仅更新适配器，便于资源受限团队快速复现与扩展。
- **异构/同构/一对一三重视角实验设计**为后续研究提供了全面的基准对照范式，可直接复用于新方法的性能评估。

## 关键术语表
- **FedMKT**：联邦相互知识转移框架，实现服务器 LLM 与客户端 SLM 的双向性能增强。
- **MinED（Minimum Edit Distance）**：最小编辑距离，用于构建异构模型词汇表之间的最优映射关系。
- **DualMinCE**：双向最小交叉熵选择策略，从多方 logits 中挑选对目标模型最有利的知识进行蒸馏。
- **PEFT（Parameter-Efficient Fine-Tuning）**：参数高效微调，如 LoRA，冻结主体参数仅更新少量适配器以降低计算与通信开销。
- **Knowledge Distillation**：知识蒸馏，通过软标签（logits）将大模型知识转移至小模型。
- **Heterogeneous Federated Learning**：异构联邦学习，允许不同架构或规模的模型参与同一联邦训练任务。

## 可复现要素
- **数据集**：RTE、WIC、BoolQ、CQA、ARC-E、ARC-C、S-NI、DialogSum，均从 HuggingFace 下载并公开。
- **代码/权重**：代码已开源至 FATE 项目（https://github.com/FederatedAI/FATE-LLM/tree/main/python/fate_llm/algo/fedmkt）。
- **关键超参**：LoRA r=8, lora_alpha=16, dropout=0.05；batch_size=4；AdamW beta1=0.9, beta2=0.95；warmup_ratio=0.008；weight_decay=0.1；max_grad_norm=1.0；λ=0.9；本地轮数 E=1，服务器本地轮数 R=1；学习率因模型/任务而异（3e-5 或 3e-4）。
