---
title: "CodeJudge-Eval-Can-Large-Language-Models-be-Good-Judges-in-C"
source: https://aclanthology.org/2025.coling-main.7.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:02:01"
field: "代码理解与评测基准"
keywords: ["code understanding", "LLM evaluation", "benchmark", "code judging", "large language models", "APPS"]
innovations: ["以判题视角构建细粒度代码理解基准 CJ-Eval，区分 CE/WA/RE/TLE/AC 九类错误组合", "提出 Easy/Medium/Hard 三级标签体系以覆盖从粗粒度到细粒度的评测需求", "揭示 LLM 代码生成能力与其代码评审能力之间的显著鸿沟"]
benchmarks: ["CJ-Eval", "APPS", "HumanEval", "MBPP"]
---

# 论文速读：CodeJudge-Eval-Can-Large-Language-Models-be-Good-Judges-in-C

## 一句话总结
本文提出了 **CodeJudge-Eval (CJ-Eval)**，一个以"代码评审"而非"代码生成"视角评估 LLM 代码理解能力的新基准，涵盖从编译错误到时间超限的五类细粒度判错任务；12 个主流模型在最具挑战性的 Hard 设置上最高仅达 macro F1 ≈ 50，表明当前 SOTA 模型对代码语义的理解仍存在显著缺口。

## 研究问题与动机
- **现有评测过度依赖"代码生成 + 单元测试通过"范式**：HumanEval、APPS 等 benchmark 主要检验模型能否从描述生成正确代码并跑过预置测试用例，但该路径易受数据泄漏（data leakage）与答案记忆的影响，未必反映真正的代码理解。
- **通过单一/少量测试用例无法覆盖全部边界情况**：Liu et al. (2023) 与 Dong et al. (2024) 指出预置 test cases 难以穷举所有输入与边缘场景，导致评测可靠性受限。
- **"会生成"不等于"会判断"**：教育理论（Care et al., 2012）认为，能准确评判他人解法正确性的人更可能真正理解题目；但直觉上，LLM 能写出 AC 代码的能力与识别他人代码中 CE/WA/RE/TLE 等多类错误的能力并不等价（Figure 1、8）。
- **对抗记忆攻击**：LLM 可记住远超英文维基百科的知识（Allen-Zhu & Li, 2024），而本题设定下每道题对应多份候选代码，规模远大于题目本身，令单纯记忆所有解法难度大增（相比 LiveCodeBench、LiveBench 通过引入新题缓解记忆的策略）。

## 核心贡献（创新点）
1. **提出 CJ-Eval 基准**，以"代码评审"角度衡量 LLM 代码理解能力，区别于传统语言→代码生成评测，首次将 LLM 置于 judge 角色并系统评估其区分五类代码错误（CE/WA/RE/TLE/AC）的能力。
2. **设计三层难度的细粒度标签体系**（Easy 3-label / Medium 6-label / Hard 9-label），使评测既能覆盖粗粒度的 AC vs 非 AC 二分，也能考察模型识别复合错误类型（如 WA+RE、WA+TLE、RE+TLE 等）的精细认知。
3. **基于 5,000 道 APPS 题 + 16 种 LLM 生成候选代码 + 自建细粒度执行器** 的完整数据流水线，经"测试用例数量阈值"和"每个 verdict 仅保留一道"双重过滤后得到 1,860 道精炼评测样本。
4. **揭示生成与评审能力之间存在显著鸿沟**：实验表明在若干任务上能通过预置测试的模型，并不能据此准确判定其他候选代码的错误类别；这为代码能力评测提供了独立于生成正确性的新维度。
5. **系统对比 12 个主流 LLM（含商业闭源与开源通用/代码专用模型）**，发现即便 GPT-4o / Claude-3.5-Sonnet 在 Easy 设置上 macro F1 也仅约 50，Open-Source 小模型（≤7B）多低于随机基线，明确了当前前沿模型在代码理解上的天花板。

## 方法详解
- **基准形式化**：每道题 $P_i = (S_i, \mathcal{T}_i, \mathcal{C}_i, Y_i, [V_1^i, \ldots, V_{m_i}^i])$，其中 $S_i$ 为题目文本，$\mathcal{T}_i$ 为输入-输出测试用例集合，$\mathcal{C}_i$ 为由 16 个 LLM 生成的候选代码集合，$V_k^i$ 是代码 $c_k^i$ 在 $\mathcal{T}_i$ 上逐用例执行的判决列表，$Y_i = [y_1^i, \ldots, y_{m_i}^i]$ 是由 $V_k^i$ 聚合得到的单标签。
- **细粒度判决系统**：基于 Codeforces / LeetCode 风格重新实现本地评测器，区分五种 verdict——CE（编译失败）、RE（运行时异常）、TLE（单次超时 2s）、WA（输出错误）、AC（完全正确）；原版 APPS 判优系统仅区分对/错，不区分错误类型。
- **标签定义（Hard 九类）**：A=全 AC；B=全 CE；C=仅含 WA；D=仅含 RE；E=同时含 WA 与 RE；F=仅含 TLE；G=同时含 WA 与 TLE；H=同时含 RE 与 TLE；I=三者皆有（WA+RE+TLE）。共 $2^5=32$ 种组合中剔除不可能（如 CE 与其他错并存）后得到 9 类。
- **Easy / Medium 压缩**：Medium 将 E–I 合并为"Not A or B 且至少含两类错误"（6 类）；Easy 将 C–I 全部合并为"Not A or B"（3 类）。
- **数据筛选**：① 按测试用例数量下界过滤（intro 20 / interview 80 / competition 40）得 457 题；② 每题内每种 verdict 仅随机保留一份代码，最终得到 1,860 个样本，平均每道题 4.1 份候选代码。
- **Prompt 模板**：固定 four-place 模板（Figure 9），Placeholder A 插入不同难度下的选项说明；Placeholder B 在 few-shot / CoT 设置下填入示例；C/D 填待评题目与代码；输出要求仅一个字母。
- **Few-shot / CoT**：One-shot 给出一个样例；One-shot CoT 附加逐步推理过程（Figure 11）；图 5、9 进一步扩展到 2-shot / 3-shot。

## 实验与结果
- **数据集**：APPS test set 中 457 题、1,860 个候选代码；按 difficulty 分布为 intro 133 / interview 178 / competition 146。
- **评测模型**：12 个（GPT-4o、Claude-3.5-Sonnet、Gemini-1.5-Pro、GPT-3.5-Turbo + 8 个开源通用/代码模型）；temperature=0。
- **度量**：Accuracy 与 Macro F1（重点指标，缓解 Easy 设置下 C 类占 ~81% 的类别不平衡）。
- **最强表现**（zero-shot，Table 2）：
  - Easy：GPT-4o Acc=84.30%，Macro F1=38.16；Claude-3.5-Sonnet Macro F1=**50.83**（三类设置中最高单值）。
  - Medium：Claude-3.5-Sonnet Macro F1=27.02；GPT-4o=20.67。
  - Hard：Claude-3.5-Sonnet Macro F1=19.05；GPT-4o=13.61。
- **关键对比**：
  - 开源模型整体低于随机基线（Macro F1 常 <10），如 DeepseekCoder-Instruct 6.7B Hard 仅 1.97。
  - 虽部分开源代码模型在 HumanEval/MBPP 上可与 GPT-3.5 比肩，但在 CJ-Eval 上远落后，说明评测视角转变揭示了新的能力 gap。
- **Scaling 规律**（Figure 4）：Qwen2-72B 接近 GPT-4o 水平，但 Llama-3.1 70B→405B 提升边际。
- **Few-shot 增益**（Figure 5、Table 9）：1-shot 显著提升（GPT-4o Easy F1 +14.36；Claude-3.5 约 +4.85），但 shots 继续增加反而下降，作者推测长 prompt 损害推理。
- **CoT 效果**（Figure 6、Table 9）：在 Hard 上带来显著增益（Claude-3.5 1-shot CoT Hard F1=27.91 vs 19.05）；Easy 上则出现小幅下降，归因于任务过于简单时 CoT 引入额外干扰。
- **混淆矩阵**（Table 6，GPT-4o Hard 1-shot CoT）：对 AC 识别较准（对角线 153/185），主要错误集中于将 B（CE）误判为 C（仅 WA），或在 C/D/E/F/G/H/I 之间相互混淆。

## 相关工作脉络
1. **HumanEval / MBPP / EvalPlus**：语言→代码生成基准，依赖预置测试用例，本文通过"判题而非生成"绕过这些局限。
2. **APPS (Hendrycks et al., 2021)**：本文选用其 test set 作为原始题目来源，但将其从生成题转换为多选项判决题；同时替换了原 APPS 的粗粒度判题器为细粒度版本。
3. **LiveCodeBench / LiveBench**：通过不断添加新题以缓解数据泄漏与记忆问题；本文从另一方向（每题多份候选代码、增加判断任务复杂度）抵抗记忆。
4. **MT-Bench / AlpacaEval**：在通用领域已验证 LLM-as-a-Judge 可行性，本文将该范式迁移到代码理解领域，首次以硬标签（CE/WA/RE/TLE）做分类评测。
5. **ICE-Score (Zhuo, 2024)**：同样关注 LLM 对生成代码的评估有用性，但侧重生成后的自评；本文强调跨模型、跨错误类型的多维分类判断。
6. **Gu et al. (2024a) / CodeHalu (Tian et al., 2024)**：研究 LLM 对自身错误生成的理解及代码幻觉；本文进一步以外部生成的、多样化错误候选代码作为评测对象，构建独立基准。
7. **CodeScope (Yan et al., 2023a) / CruxEval (Gu et al., 2024b)**：后者侧重代码推理与执行，但仍是"生成+执行"范式；本文引入细粒度错误类型分类，更贴近实际代码评审场景。

## 局限性与未来方向
- **并非替代现有生成基准**：不评估代码生成能力，未来应与语言→代码评测结合形成综合体系。
- **未能完全消除故意记忆风险**：虽然多候选代码提高了记忆门槛，但仍存在被刻意训练以记住所有候选解的可能性。
- **领域泛化受限**：基准设计针对编程域，迁移至更通用的代码理解或非代码场景面临适配困难。
- **类别严重不平衡**（Easy 设置）：C 类占比 ~81%，促使作者采用 Macro F1；未来可通过采样或加权策略进一步平衡。
- **测试用例数量过滤阈值主观**：intro / interview / competition 的阈值分别为 20 / 80 / 40，可能对边缘题目精度产生影响。

## 研究启发与可借鉴点
1. **"判题"视角作为生成评测的强补充**：在团队已有的代码生成 benchmark 旁，可并行引入类似的多候选判错题，构建"生成 + 理解"双通道评测。
2. **细粒度错误标签体系（CE/WA/RE/TLE 的组合展开）**：可用于训练/评测阶段构造"错误感知"数据集，例如以 CE→RE→TLE→WA 为序构造 curriculum。
3. **Few-shot 与 CoT 在此任务上的"倒 U 型"效应**：提示工程需警惕 shot 数或推理链长度带来的反作用，对简单设置可省 CoT。
4. **开源 vs 闭源鸿沟的新证据**：在 HumanEval 上相近的模型在 CJ-Eval 上差距巨大，提示团队在选型时不能仅依赖生成 pass@k。
5. **数据筛选策略（按 verdict 去重 + 按用例数阈值）**：可作为构造同类多候选评测数据集的通用 pipeline 参考。

## 关键术语表
- **CodeJudge-Eval (CJ-Eval)**：本文提出的代码理解评测基准，让 LLM 作为 judge 判断候选代码的真实执行结果所属的细粒度类别。
- **Verdict**：单次测试用例执行所返回的具体结果（AC/CE/WA/RE/TLE）。
- **Label**：由一份代码在所有测试用例上的 verdict 序列聚合而成的最终分类标签（A–I 九类）。
- **Hard / Medium / Easy setting**：三档评测难度，分别对应 9 / 6 / 3 类标签。
- **Macro F1**：各类别 F1 的算术平均，用于缓解类别不平衡下的评估偏差。
- **LLM-as-a-Judge**：将大语言模型作为自动评判器的范式，本文首次系统性迁移到代码错误分类任务。
- **Chain-of-Thought (CoT)**：在 prompt 中提供分步推理示例，引导模型显式演绎判决过程。
- **APPS test set**：5,000 道来自 Codeforces / LeetCode / Kattis 的编程题，涵盖入门/面试/竞赛三级难度，本文以其作为原始题目池。

## 可复现要素
- **数据集**：CJ-Eval 已开源，链接 https://github.com/CodeLLM-Research/CodeJudge-Eval（原文 Abstract 声明）；APPS test set 为公开数据集。
- **代码**：基准构建流程与筛选算法（Algorithm 1）在附录中给出；具体执行器实现参见仓库。
- **关键超参**：evaluation temperature=0.0；CE/WA/RE/TLE 判定依据 Codeforces/LeetCode 风格；单次测试超时阈值 2 秒；筛选阈值 intro≤20 / interview≤80 / competition≤40 的测试用例数；每题每种 verdict 随机保留一份。
- **模型权重**：开源模型均来自 HuggingFace 公开发布（见 Appendix C）；商业模型通过 API 评测。
