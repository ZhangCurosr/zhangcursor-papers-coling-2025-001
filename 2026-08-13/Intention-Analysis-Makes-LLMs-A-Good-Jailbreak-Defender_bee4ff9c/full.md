# Intention Analysis Makes LLMs A Good Jailbreak Defender

Yuqi Zhang<sup>1</sup>, Liang Ding<sup>2</sup>, Lefei Zhang<sup>1</sup>\*, Dacheng Tao<sup>3</sup>

<sup>1</sup>School of Computer Science, Wuhan University <sup>2</sup>The University of Sydney

<sup>3</sup>College of Computing and Data Science at Nanyang Technological University, Singapore 639798

{zhangyuqi,zhanglefei}@whu.edu.cn, {liangding.liam,dacheng.tao}@gmail.com

https://github.com/alphadl/SafeLLM\_with\_IntentionAnalysis

## Abstract

Aligning large language models (LLMs) with human values, particularly when facing complex and stealthy jailbreak attacks, presents a formidable challenge. Unfortunately, existing methods often overlook this intrinsic nature of jailbreaks, which limits their effectiveness in such complex scenarios. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis (IA). IA works by triggering LLMs’ inherent self-correct and improve ability through a two-stage process: 1) analyzing the essential intention of the user input, and 2) providing final policy-aligned responses based on the first round conversation. Notably, IA is an inference-only method, thus could enhance LLM safety without compromising their helpfulness<sup>1</sup>. Extensive experiments on varying jailbreak benchmarks across a wide range of LLMs show that IA could consistently and significantly reduce the harmfulness in responses (averagely -48.2% attack success rate). Encouragingly, with our IA, Vicuna-7B even outperforms GPT-3.5 regarding attack success rate. We empirically demonstrate that, to some extent, IA is robust to errors in generated intentions. Further analyses reveal the underlying principle of IA: suppressing LLM’s tendency to follow jailbreak prompts, thereby enhancing safety.

Warning: Some of the examples may be harmful!

## 1 Introduction

Recently, Large Language Models (LLMs) (Touvron et al., 2023; OpenAI, 2023; Google, 2023), such as ChatGPT, not only show remarkable capabilities in various tasks (Qin et al., 2023; Zhong et al., 2023; Peng et al., 2023; Ren et al., 2024), but also lead to the risk of potential misuse (e.g., producing harmful responses or illegal suggestions) (Weidinger et al., 2021). Efforts like Reinforcement Learning from Human Feedback (RLHF, Ouyang et al., 2022) have been made to alleviate these risks and enhance LLMs’ alignment with human values, making LLMs able to refuse direct harmful questions like how to rob a bank? However, LLMs remain vulnerable to some adversarial inputs, particularly in the context of so-called “jailbreak” attacks. These jailbreaks are specially designed to circumvent safety policy and manipulate LLMs for their restricted outputs (Yuan et al., 2024; Zou et al., 2023), which poses formidable risks in real applications.

![](images/0b0c1da30b052f70d662e97dc5e050c0ea1ce783b24fe2fe92bbad6b8038c385.jpg)  
(a) Vicuna-7B

![](images/6f78aeee63c1207ceb90d432d8bbbd562eae165b1f1592eb9d1e2139ba11a327.jpg)  
(b) MPT-30B-Chat

![](images/42a715042f5c8921feed4da7c673690be69ee6f2ca3ecf864d0dabee5f2c4f24.jpg)  
(c) GPT-3.5  
Figure 1: Performance of our method on different LLMs. Our IA 1) reduces Attack Success Rate (↓) against both crafted jailbreak prompts (DAN and Deep-Inception) and automatic attack (GCG), 2) achieves remarkable safety improvements for both SFT (Vicuna-7B & MPT-30B-Chat) and RLHF (GPT-3.5) LLMs.

To defend LLMs against jailbreak attacks, existing popular methods either focus on emphasizing safety during inference (Xie et al., 2023; Wei et al., 2023b), or modifying the user inputs (Robey et al., 2023) or evaluating inputs/outputs’ safety (Li et al., 2024), often neglecting the intrinsic characteristics of jailbreak attacks. This oversight limits their effectiveness in more complex jailbreak scenarios (see experimental results in Section 4.2). Through analysis, we find that these jailbreaks typically work by concealing harmful questions within seemingly inoffensive and complex scenarios, such as role-playing or virtual scene construction (Liu et al., 2023b). Such disguise leads LLMs to focus on the jailbreak prompt excessively, impairing their awareness of the harmful question itself (See Figure 5 for evidence). We assume such insufficient awareness of the harmful content concealed in complex jailbreak queries is the fundamental reason for LLM’s vulnerability to these attacks. Drawing insights from classic dialogue system design (Allen and Perrault, 1980), an effective solution is to tailor an intent recognition mechanism specifically for jailbreak scenarios to enhance LLM’s understanding of user queries regarding safety and improve its ability to recognize concealed harmful questions.

In this paper, we leverage the intrinsic intent recognition capabilities of LLMs, proposing an Intention Analysis (IA) strategy. Specifically, IA enables LLMs to analyze the essential intention of the user query to better understand it and recognize the underlying unsafe content within before finally responding, as shown in Figure 2. Such intention analysis mechanism can significantly improve LLM safety against varying jailbreak attacks, see Figure 1 for a demonstration. We dive deeper from the perspective of attention scores and find that the underlying principle of IA is to suppress LLM’s tendency to follow jailbreak prompts. Notably, our IA is an inference-only method that can significantly enhance LLM safety without the need for additional safety training (Ouyang et al., 2022; Touvron et al., 2023). In this way, IA skillfully circumvents the safety-helpfulness trade-off and enables comparable safety improvement as well as better helpfulness maintenance.

To summarize, our contributions are as follows:

• We introduce IA, a new method that significantly enhances LLM safety in the context of sophisticated jailbreak attacks through an intention analysis mechanism.

• IA is a plug-and-play inference-only method, thereby 1) cleverly circumventing the safetyhelpfulness trade-off that is challenging in safety training, and 2) can be flexibly and effectively deployed upon any LLMs.

• Empirically, our robust IA significantly and consistently reduces the harmfulness of LLM outputs, while maintaining the helpfulness, achieving new state-of-the-art performance on several benchmarks, e.g., DeepInception.

## 2 Related Work

Alignment-Breaking Adversarial Attack Despite significant efforts to align LLMs with human preference (Ouyang et al., 2022; Bai et al., 2022; Lee et al., 2023; Korbak et al., 2023; Miao et al., 2024), adversarial attackers can still elicit harmful responses from LLMs by “jailbreak” attacks (Shen et al., 2023; Liu et al., 2023b). Current jailbreak attack methods are primarily classified into two categories: in-the-wild jailbreak prompts and optimization-based automatic attacks (Chao et al., 2023; Yu et al., 2023). In-the-wild jailbreak prompts are typically hand-crafted through human ingenuity and is semantically understandable in general (Shen et al., 2023). For optimization-based automatic attacks, a representative work is to automatically fetch a transferable attack suffix through the Greedy Coordinate Gradient (GCG) algorithm which maximizes the probability of the language model generating an affirmative and unsafe response (Zou et al., 2023). In this work, various attacks mentioned above are considered in experiments to comprehensively test the defensive performance of our method.

Safety-Enhancing Defensive Methods Recently, numerous methods have been developed to reduce LLMs’ harmful generations in inference stage. A branch of them mainly concentrates on controlling the content that LLMs can see by pre-processing user inputs, such as perplexity filtering (Alon and Kamfonas, 2023; Jain et al., 2023), paraphrasing (Jain et al., 2023) and re-tokenization (Cao et al., 2023; Jain et al., 2023). Another branch focuses on exploiting LLMs’ intrinsic capabilities of self-correction and improvement against jailbreak attacks, such as letting LLMs self-evaluate their outputs (Helbling et al., 2023; Li et al., 2024; Wang et al., 2024) or reminding of safety in system mode with conventional decoding (Xie et al., 2023) or contrastive decoding (Zhong et al., 2024).

While existing methods effectively prevent unsafe responses, their efficacy drops significantly against sophisticated jailbreak attacks that conceal harmful questions within complex and seemingly inoffensive scenarios. In contrast, our method enhances LLM safety by leveraging the intrinsic intent recognition capabilities of LLMs to detect these concealed threats (see Table 1 for details).

![](images/18e4bf03939cc726b94c6e26ea2529b327d729823dff4b2ec0e335fdf3bf9d09.jpg)  
Figure 2: Illustrated Comparison of (a) vanilla and (b) the proposed IA. Our IA consists of two stages: (1) Essential Intention Analysis: instructing the language model to analyse the intention of the user query with an emphasis on safety, ethics, and legality; (2) Policy-Aligned Response: eliciting the final response aligned with safety policy, building upon the analyzed intention from the first stage.

## 3 Methodology

## 3.1 Preliminary

We focus on enhancing LLM safety during the inference stage. In practice, developers usually implement pre-defined system prompts for LLMs to facilitate safe, responsible, and effective interactions with users (Chiang et al., 2023). Under this premise, the system prompt $P _ { s y s }$ and the user prompt $P _ { u s r }$ are concatenated to form the final input $\{ x _ { 1 : n } ^ { s } , x _ { 1 : m } ^ { u } \}$ of the LLM, where $P _ { s y s } =$ $\{ x _ { 1 } ^ { s } , x _ { 2 } ^ { s } , . . . , x _ { n } ^ { s } \} , P _ { u s r } = \{ x _ { 1 } ^ { u } , x _ { 2 } ^ { u } , . . . , x _ { m } ^ { u } \} , x _ { i } ^ { s }$ and $x _ { j } ^ { u }$ are the i-th and j-th token of $P _ { s y s }$ and $P _ { u s r }$ , respectively. Conditioned on the input $\{ x _ { 1 : n } ^ { s } , x _ { 1 : m } ^ { u } \}$ , the autoregressive inference process of response $R = y _ { 1 : L }$ is formulated as following:

$$
q ( y _ { 1 : L } | x _ { 1 : n } ^ { s } , x _ { 1 : m } ^ { u } ) = \prod _ { i = 1 } ^ { L } q ( y _ { i } | y _ { 1 : i - 1 } , x _ { 1 : n } ^ { s } , x _ { 1 : m } ^ { u } ) .
$$

For simplicity, we use $R \sim q ( R | P _ { s y s } , P _ { u s r ) }$ to denote sampling a response R from q(·) given the prompt $P _ { s y s }$ and $P _ { u s r }$ . In this way, the response R can be obtained as: $R = \operatorname { L L M } \left( P _ { s y s } , P _ { u s r } \right)$

In this work, we aim to leverage LLMs’ intrinsic abilities of intention analysis, to enhance their safety against varying jailbreak attacks during the inference stage, while simultaneously maintaining the general helpfulness.

## 3.2 IA: Intention Analysis

To achieve the above goal, we introduce IA, a zeroshot intention analysis mechanism, to guide LLMs to explicitly identify and understand the underlying intention of a user query before facilitate a final response. Specifically, we devise a two-stage intention analysis instruction to accomplish the whole process<sup>2</sup>, as illustrated in Figure 2(b): (1) essential intention analysis and (2) policy-aligned response.

Stage 1: Essential Intention Analysis This stage focuses on guiding the LLMs to discern the core intention behind the user query, with a specific orientation towards safety, ethics, and legality. The critical question arises: How can we ensure that LLMs accurately identify the query’s intention? Actually, recent studies (Bender and Koller, 2020; Zhu et al., 2024; Gómez-Pérez et al., 2023) have shown that LLMs are notably proficient at language understanding tasks, and intention analysis is a straightforward task, indicating the competence of LLMs in performing this stage. The only concern is generative models’ potential hallucination when performing the discriminative tasks (Ji et al., 2023; Yan et al., 2021; Ye et al., 2023; Lu et al., 2024), therefore, we carefully define the format for the models’ response, that is, beginning with “The essential intention of the query $i s ^ { , \dag }$ , which has been validated in our analysis.

In practice, we construct the instruction for the LLMs to effectively perform intention analysis, denoted as $I _ { r e c }$ . When presented with a user query

$P _ { u s r } { ^ 3 }$ , we concatenated $I _ { r e c }$ and $P _ { u s r }$ to form a whole “User” level input $I _ { r e c } \oplus P _ { u s r }$ for the LLMs. Subsequently, the designated target LLMs engage in an auto-regressive inference process, guided by its system prompt $P _ { s y s } ,$ , to produce the stagespecific response:

$$
R _ { s t 1 } = \mathrm { L L M } \left( P _ { s y s } , I _ { r e c } \oplus P _ { u s r } \right) ,
$$

which is expected to contain the essential intention. Stage 2: Policy-Aligned Response Having successfully recognized the essential intention, the second stage aims to elicit the final response from the LLMs. We first direct the LLMs to bear the identified intention in mind and then provide a final response to the user query. Meanwhile, we explicitly instruct the LLMs to strictly adhere to safety policy and ethical standards<sup>4</sup> and ensure the exclusion of any unsafe content in their responses. To this end, the second stage further strengthens the role of the intention analysis and reinforces the inherent alignment of LLMs with the safety policy.

Specifically, we concatenate the dialogue from the first stage with the instruction for the current stage, denoted as $I _ { c t } ,$ , forming the complete input for LLMs. Then a similar autoregressive inference process is conducted, leading to the generation of the final response $R _ { s t 2 }$ to the user query $P _ { u s r } \mathbf { i }$

$$
R _ { s t 2 } = \mathrm { L L M } \left( P _ { s y s } , I _ { r e c } \oplus P _ { u s r } , R _ { s t 1 } , I _ { c t } \right) .
$$

To assess the safety of the response, we follow Shen et al. (2023) to employ a binary autoannotation function $\mathsf { A S } ( \cdot ) ^ { 5 }$ to determine the harmfulness of $R _ { s t 2 }$ . If the outcome yields $\mathrm { A S } ( R _ { s t 2 } )$ = False, then the response is deemed safe, indicating a successful defense against the jailbreak attack.

## 4 Experiment

## 4.1 Setup

Datasets For safety datasets, we experiment on three main categories of jailbreak attacks, including three representative complex and stealthy in-the-wild jailbreak datasets (i.e. DAN (Shen et al., 2023), SAP200 (Deng et al., 2023a), and DeepInception (Li et al., 2023)), two popular optimization-based automatic jailbreak methods (i.e. GCG (Zou et al., 2023) and AutoDAN (Liu et al., 2023a)), and two advanced attacks for GPT-3.5 (i.e. multilingual attack called MultiJail (Deng et al., 2023b) and encryption-based attack named CipherChat (Yuan et al., 2024)).

Besides, to evaluate IA’s effect on helpfulness for general benign queries, we conduct experiments on three widely recognized datasets, i.e., AlpacaEval (Dubois et al., 2023), MMLU (Hendrycks et al., 2021) and TruthfulQA (Lin et al., 2022).

Evaluation Metrics For safety assessment, we annotate the harmfulness of responses and report attack success rate (ASR, Shen et al., 2023), where lower scores indicate stronger safety. Specifically, for DAN dataset, considering the complexity of responses, we adopt gpt-3.5-turbo-0613<sup>6</sup> as the auto-annotation function following Deng et al. (2023a) and carry our human evaluation in Appendix C.1 to ensure the credibility. For other safety datasets, we annotate harmfulness following Zou et al. (2023) by matching refusal strings (e.g., “I’m sorry”; see Appendix C.2 for detailed settings).

For helpfulness assessment, we report win rate (Dubois et al., 2023) for AlpacaEval and accuracy (Hendrycks et al., 2021) for MMLU. For TruthfulQA, we follow Chuang et al. (2023) and report on two distinct metrics: MC1 and MC2 scores, where higher scores indicate stronger factuality (see Appendix C.3 for more details).

Models To evaluate IA’s effectiveness, we experiment on representative LLMs with varying scales and alignment levels, including not only SFT models, i.e. Vicuna-7B/13B-v1.1 (Chiang et al., 2023) and MPT-30B-Chat (Team, 2023), but also RLHF models, i.e. ChatGLM-6B (Zeng et al., 2023), Llama2-7B-Chat (Touvron et al., 2023), Llama3-8B-Instruct<sup>7</sup>, and DeepSeek-67B-Chat (DeepSeek-AI, 2024). Beyond open-source LLMs, our experimentation extends to an advanced closedsource LLM, GPT-3.5 (gpt-3.5-turbo-1106) (OpenAI, 2023), renowned for its superior capabilities, especially safety alignment.

Comparison Baselines We compare our IA with vanilla LLMs (no defense) and seven popular defense methods, i.e., Input Check<sup>8</sup>, ICD (Wei et al., 2023b), (System-Mode) Self-Reminder (Xie et al., 2023), SmoothLLM (Robey et al., 2023), BPE-dropout (Jain et al., 2023), Self Defense (Helbling et al., 2023), and Moral Self-Correction (Ganguli et al., 2022). The first four representative defense methods are reported in Table 1 and others in Table 7 in Appendix due to page limitation. Besides, a training method is also included in Appendix E.3 and results show IA achieves both safety and helpfulness goals without additional resource-consuming safety training. For a fair comparison, we closely follow the best default parameters in their papers.

<table><tr><td rowspan="2">Models</td><td rowspan="2">Defense Methods</td><td colspan="4">Attack Methods</td><td rowspan="2">Average</td><td rowspan="2">Time Cost</td></tr><tr><td>DAN</td><td>SAP200</td><td>DeepInception</td><td>GCG AutoDAN</td></tr><tr><td rowspan="6">ChatGLM-6B</td><td>Vanilla</td><td>29.0</td><td>45.8</td><td>100</td><td>88.0</td><td>99.5</td><td>72.5 14.3</td></tr><tr><td>Input Check</td><td>16.3</td><td>9.52</td><td>46.2 9.00</td><td>89.0</td><td>34.0</td><td>12.6</td></tr><tr><td>ICD</td><td>19.1</td><td>2.81</td><td>17.1 17.0</td><td>2.00</td><td>11.6</td><td>15.2</td></tr><tr><td>Self-Reminder</td><td>22.5</td><td>3.13</td><td>17.9</td><td>0.00 66.0</td><td>21.9</td><td>17.1</td></tr><tr><td>SmoothLLM</td><td>7.19</td><td>20.6</td><td>84.5</td><td>1.00 84.0</td><td>39.5</td><td>113.4</td></tr><tr><td>IA (Ours)</td><td>5.48</td><td>6.12</td><td>0.00 1.00</td><td>2.00</td><td>2.92</td><td>19.2</td></tr><tr><td rowspan="5">Llama2-7B-Chat</td><td>Vanilla</td><td>1.02</td><td>0.56</td><td>71.7</td><td>0.00 44.0</td><td>23.5</td><td>16.0</td></tr><tr><td>Input Check</td><td>7.50</td><td>0.00</td><td>0.00</td><td>0.00 43.0</td><td>10.1</td><td>10.7</td></tr><tr><td>ICD</td><td>0.98</td><td>0.00</td><td>0.00</td><td>0.00 0.00</td><td>0.20</td><td>15.5</td></tr><tr><td>Self-Reminder</td><td>0.77</td><td>0.00</td><td>4.38</td><td>0.00</td><td>0.00 1.03</td><td>14.8</td></tr><tr><td>SmoothLLM</td><td>0.31</td><td>2.81</td><td>86.5</td><td>0.00 71.5</td><td>32.2</td><td>118.5</td></tr><tr><td rowspan="6">Llama3-8B-Instruct</td><td>IA (Ours)</td><td>0.13</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00 0.03</td><td>19.5</td></tr><tr><td>Vanilla</td><td>14.7</td><td>0.94</td><td>35.1</td><td>0.00 18.5</td><td>13.8</td><td>7.36</td></tr><tr><td>Input Check</td><td>3.43</td><td>0.00</td><td>7.57</td><td>0.00 7.00</td><td>3.60</td><td>4.98</td></tr><tr><td>ICD</td><td>0.63</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00 0.13</td><td>5.12</td></tr><tr><td>Self-Reminder</td><td>0.63</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00 0.13</td><td>6.64</td></tr><tr><td>SmoothLLM</td><td>0.31</td><td>0.63</td><td>32.7</td><td>0.00</td><td>46.0 15.9</td><td>79.2</td></tr><tr><td rowspan="6">Vicuna-7B</td><td>IA (Ours)</td><td>0.31</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00 0.06</td><td>10.6</td></tr><tr><td>Vanilla</td><td>48.4</td><td>73.4</td><td>90.0</td><td>83.0 100</td><td>79.0</td><td>10.2</td></tr><tr><td>Input Check</td><td>19.0</td><td>58.1</td><td>53.0</td><td>13.0 100</td><td>48.6</td><td>8.64</td></tr><tr><td>ICD</td><td>40.4</td><td>32.8</td><td>0.00</td><td>1.00</td><td>88.0 32.4</td><td>10.3</td></tr><tr><td>Self-Reminder</td><td>41.3</td><td>33.8</td><td>55.4</td><td>11.0 98.5</td><td>48.0</td><td>15.0</td></tr><tr><td>SmoothLLM</td><td>13.5 3.42</td><td>54.4 0.31</td><td>96.4</td><td>8.00 98.5</td><td>54.2</td><td>102.7</td></tr><tr><td rowspan="5">Vicuna-13B</td><td>IA (Ours) Vanilla</td><td>60.0</td><td>65.4</td><td>0.00 98.8</td><td>0.00 10.5 99.5</td><td>2.85</td><td>17.3</td></tr><tr><td></td><td>7.19</td><td>7.50</td><td>98.8</td><td>87.0 3.00</td><td>82.1</td><td>15.1</td></tr><tr><td>Input Check ICD</td><td>53.9</td><td>32.8</td><td></td><td>97.5 91.5</td><td>42.8 53.0</td><td>10.7</td></tr><tr><td>Self-Reminder</td><td>52.5</td><td>36.9</td><td>86.9</td><td>0.00 1.00</td><td></td><td>13.1</td></tr><tr><td>SmoothLLM</td><td>17.3</td><td>37.0</td><td>75.7</td><td>83.0</td><td>49.8</td><td>16.4</td></tr><tr><td rowspan="5"></td><td>IA (Ours)</td><td>0.94</td><td>1.12</td><td>94.0 0.00</td><td>5.00 98.0 3.50</td><td>50.3 1.11</td><td>136.1 22.1</td></tr><tr><td>Vanilla</td><td>55.4</td><td>89.6</td><td>100</td><td>0.00 35.0</td><td></td><td>141.5</td></tr><tr><td>Input Check</td><td>14.1</td><td>9.38</td><td>41.8</td><td>6.00</td><td>70.0 14.3</td><td>132.2</td></tr><tr><td>ICD</td><td>49.4</td><td>29.9</td><td>100.0</td><td>3.00</td><td></td><td></td></tr><tr><td>Self-Reminder</td><td>46.9</td><td>39.4</td><td>100</td><td>19.0</td><td>45.6 51.3</td><td>218.7 210.0</td></tr><tr><td rowspan="4"></td><td>SmoothLLM</td><td>60.6</td><td>64.4</td><td>22.0</td><td>22.0</td><td>42.3</td><td>534.8</td></tr><tr><td>IA (Ours)</td><td>5.38</td><td>19.2</td><td>4.78</td><td>4.00</td><td>8.34</td><td>223.0</td></tr><tr><td>Vanilla</td><td>53.1</td><td>82.4</td><td>94.4</td><td>10.0</td><td></td><td>60.0 168.0</td></tr><tr><td>Input Check</td><td>30.3</td><td>3.20</td><td>5.80</td><td>1.00</td><td>8.06</td><td>154.2</td></tr><tr><td rowspan="4">DeepSeek-67B-Chat</td><td>ICD</td><td>45.6</td><td>14.4</td><td>47.8</td><td>9.00</td><td>29.2</td><td>162.8</td></tr><tr><td>Self-Reminder</td><td>9.58</td><td>7.81</td><td>3.19</td><td>1.00</td><td>5.40</td><td>177.4</td></tr><tr><td>SmoothLLM</td><td>26.9</td><td>11.9</td><td>51.0</td><td>0.00</td><td>22.4</td><td>486.6</td></tr><tr><td>IA (Ours)</td><td>3.78</td><td>1.56</td><td>7.57</td><td>2.00</td><td>3.73</td><td>198.0</td></tr><tr><td rowspan="6">GPT-3.5</td><td>Vanilla</td><td>10.3</td><td>1.75</td><td>2.79</td><td>1.00</td><td>3.96</td><td>6.14</td></tr><tr><td>Input Check</td><td>2.50</td><td>0.00</td><td>0.00</td><td>4.00</td><td>1.63</td><td>2.47</td></tr><tr><td>ICD</td><td>0.94</td><td>0.31</td><td>0.00</td><td>0.00</td><td>0.31</td><td>5.12</td></tr><tr><td>Self-Reminder</td><td>2.81</td><td>0.31</td><td>0.00</td><td>0.00</td><td></td><td></td></tr><tr><td>SmoothLLM</td><td>0.64</td><td>0.00</td><td>0.00</td><td></td><td>0.78 0.16</td><td>7.21 15.2</td></tr><tr><td>IA (Ours)</td><td>0.64</td><td>0.00</td><td>0.00</td><td>0.00 0.00</td><td>0.16</td><td>8.27</td></tr></table>

Table 1: Comparison of our IA and four baselines under five jailbreak methods in terms of ASR (%) and time cost (s/sample). The best and second best average results are highlighted in bold and underline . Among them, DAN, SAP200, and DeepInception are complex and stealthy in-the-wild jailbreaks, while GCG and AutoDAN are optimization-based automatic jailbreaks. “—” means lacking official AutoDAN implementation for distributed larger models (MPT-30B and DeekSeek-67B) or white-box LLM weights required (GPT-3.5).

Implementation The detailed IA prompts for experiments are provided in Figure 9<sup>9</sup>. For the DAN dataset, we compile an evaluation dataset of 1560 samples by extracting 195 questions from each jailbreak community within the forbidden question set (Shen et al., 2023). For GCG, we follow Zou et al. (2023) and conduct transfer attacks on Vicuna-7B and 13B. The adversarial suffix achieving the lowest loss after 500 steps of optimization are adopted to further attack target models on 100 individual harmful behaviors (Wei et al., 2023b). For open-source models, we download them from HuggingFace<sup>10</sup>. For closed-source models, we obtain the responses of GPT-3.5 via API calls. Throughout our experiments, we set a temperature of zero for deterministic outcomes (Peng et al., 2023) and a generation length of 1024 tokens, employing default system prompt templates for each LLM regarding their official reports. All experiments are carried out on a solitary node outfitted with 8 A100-SXM80GB GPUs.

<table><tr><td rowspan="2">Models</td><td rowspan="2">Methods</td><td>AlpacaEval MMLU TruthfulQA</td><td></td><td></td><td></td></tr><tr><td>Win Rate</td><td>Acc.</td><td>MC1</td><td>MC2</td></tr><tr><td></td><td>のIA</td><td>28.7</td><td>40.1</td><td>37.1</td><td>54.1</td></tr><tr><td>ChatGLM-6B</td><td>OIA</td><td>25.3</td><td>39.3</td><td>37.5</td><td>56.0</td></tr><tr><td>Llama2-7B-Chat</td><td>ふIA</td><td>57.5</td><td>48.3</td><td>35.4</td><td>52.2</td></tr><tr><td></td><td>OIA</td><td>57.6</td><td>47.2</td><td>35.9</td><td>54.5</td></tr><tr><td>Llama3-8B-Instruct</td><td>ふIA</td><td>78.8</td><td>61.5</td><td>40.8</td><td>59.3</td></tr><tr><td></td><td>OIA</td><td>69.6</td><td>60.1</td><td>41.7</td><td>60.3</td></tr><tr><td>Vicuna-7B</td><td>IA</td><td>66.2</td><td>46.0</td><td>30.1</td><td>48.7</td></tr><tr><td></td><td>OIA</td><td>63.8</td><td>45.0</td><td>35.2</td><td>53.4</td></tr><tr><td>Vicuna-13B</td><td>IA</td><td>71.4</td><td>49.8</td><td>35.1</td><td>52.1</td></tr><tr><td></td><td>OIA</td><td>73.5</td><td>48.3</td><td>38.2</td><td>55.1</td></tr><tr><td>MPT-30B-Chat</td><td>IA</td><td>72.1</td><td>51.2</td><td></td><td></td></tr><tr><td></td><td>OIA</td><td>70.7</td><td>49.7</td><td></td><td></td></tr><tr><td>DeepSeek-67B-Chat</td><td>IA</td><td>86.4</td><td>71.1</td><td></td><td></td></tr><tr><td></td><td>OIA</td><td>78.6</td><td>70.5</td><td></td><td></td></tr><tr><td></td><td>IA</td><td>80.3</td><td></td><td></td><td></td></tr><tr><td>GPT-3.5</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>OIA</td><td>76.6</td><td></td><td></td><td></td></tr></table>

Table 2: General performance on helpful dataset upon different models in terms of Win Rate (%) for AlpacaEval, Accuracy (%) for MMLU and MC1, MC2 (%) for TruthfulQA. “—” means lacking official implementation for distributed larger models or white-box LLM weights required.

## 4.2 Main Results

Performance of safety on various jailbreak attacks In Table 1, we represent the ASR of several defense baselines on different LLMs under various jailbreak attacks as well as inference time comparison<sup>11</sup>. We can observe that: 1) IA effectively reduces ASRs across a diverse range ofLLMs along with an acceptable time cost. For

LLMs with high vanilla ASRs, such as ChatGLM-6B, Vicuna-7B, Vicuna-13B, MPT-30B-Chat, and DeepSeek-67B-Chat, we significantly lower the average ASRs from 72.7% to 3.79%. Similarly, for LLMs presenting lower vanilla ASRs, such as Llama2-7B-Chat, Llama3-8B-Instruct, and GPT-3.5, IA further reduces their average ASRs from 13.8% to mere 0.1%. 2) IA maintain its effectiveness even in scenarios where other defense methods struggle. For example, AutoDAN leverages LLMs to automatically attack based on optimization and thus is hard to defend. While the baselines have ASRs of at least 83% on Vicuna-7B and 13B under AutoDAN, IA can still provide significant defense with a low ASR of under 11%. Notably, IA can also integrate with another defensive method to enhance performance but with additional computation overhead (see Appendix E.4 for details). Moreover, we also extend to more advancedjailbreak attacks including multilingual and encryption-based attacks, and demonstrate our consistent effectiveness on ChatGPT (see Appendix E.1). Further analysis regarding our good performance will be discussed in Section 5.

Performance of general helpfulness for benign queries An effective defense method is expected to maintain general abilities as well. To explore the impact of our method on the general performance of LLMs, we conduct experiments on several acknowledged helpfulness datasets and report the results in Table 2. As observed, for harmless user prompts, our IA does not significantly compromise the general helpfulness on AlpacaEval, MMLU, and TruthfulQA benchmarks compared with vanilla LLMs. These results indicate that IA can be deployed in real applications to enhance LLM safety while preserving general helpfulness. More comparison results with other defensive methods can be found in Table 11 in Appendix. To further study IA’s impact on LLM’s helpfulness, we also conduct both manual and automatic checks about safe refusal’s helpfulness for harmful queries and find that IA enables LLMs to effectively give safe refusals with satisfactory helpfulness for harmful queries, instead of simple rejection (see Appendix G for detailed analysis).

## 5 Discussion of IA Mechanism

## 5.1 Can LLMs successfully generate the intentions behind jailbreak queries?

Intention analysis is a straightforward language understanding task for LLMs to proficiently perform (Bender and Koller, 2020; Zhu et al., 2024; Gómez-Pérez et al., 2023). The results of intention analysis are binary—either LLMs can successfully detect the intention, such as identifying plans to “rob a bank” as shown in Figure 2, or they fail and miss it. In Figure 3, we count the samples and examine the correlation between successful intention analysis (see Appendix C.4 for evaluation details) and producing harmless responses on SAP200 and DAN datasets<sup>12</sup>.

![](images/86ca4a1af0b673f950cef158cc043ee4deeb4f31eaa99025dc8173331e737f93.jpg)  
Figure 3: The confusion matrix illustrating the relationship between the success of intention analysis and the harmlessness of LLM’s final response on SAP200 and DAN datasets. “IR Succ.” and “IR Fail.” represent success or failure of intention analysis, respectively.

We observe that: 1) Most LLMs are highly effective in recognizing intentions behind complex and stealthy jailbreak queries, achieving a nearly 100% success rate in Vicuna-13B, MPT-30B-Chat, and DeepSeek-67B-Chat. Particularly, the intention recognition rate of Llama2-7B-Chat is relatively lower due to its excessively strong inherent safety leading to direct refusals to harmful user queries<sup>13</sup>(see Figure 17 for detailed cases). 2) In adversarial scenarios, it is easier for most LLMs to generate intentions than directly generate safe responses. Setting the SAP dataset as an example, most LLMs can successfully identify more than 90% of the adversarial intents. While in Table 1, they can only generate averagely ∼30% safe responses.

## 5.2 What if LLMs generate incorrect intentions?

To understand the effect of intention analysis errors, we examine two extreme cases: 1) recognized intentions are masked with an invalid field (e.g., “[secret]”); 2) recognized intentions are replaced with randomly sampled tokens from LLM’s vocabulary, simulating a severely wrong case. Figure 4 shows IA’s performance across different correct intention ratios on DAN dataset. Overall, IA’s performance declines with increasing intention errors but consistently maintains a much lower ASR (below 10%) compared to the vanilla baseline, indicating IA’s some robustness to wrong intentions.

![](images/4b5bd319dbecc9bd711c96eeed21652d1cf8711d86671f0c530469d6a67239dd.jpg)  
Figure 4: Performance of IA with varying correct intention ratio on DAN dataset. From left to right: the correct intentions are replaced with masked and random intention, respectively.

Notably, IA remains effective even at a 0% correct intention ratio. This can be attributed to the role of the intention analysis sequence format, allowing replacing true intentions with invalid ones to be marginally detrimental, as widely recognized by the In Context Learning (ICL) community (Min et al., 2022). Further explanation can be found at Appendix F. However, exploring the underlying principles of how sequence formats affect outcomes is beyond the scope of this work.

5.3 What is the underlying principle of IA? This section explores how IA works by analyzing the model’s attention distribution across different prompt components during response generation<sup>14</sup> (see Figure 5)<sup>15</sup>. As shown, the model under vanilla prompt pays significant attention to the jailbreak prompt, leading to potentially harmful responses. In contrast, IA at both stages significantly reduces LLM’s attention to the jailbreak prompt while increasing attention to user intent, making LLM less likely to follow jailbreak prompts and leading to safer responses.

![](images/7c65d40f0dd16610152c823f6f12c1d43aac9464d3bb4dd89559e55721c028c1.jpg)  
Figure 5: Comparison of Vicuna-13B’s attention scores on different prompt components of different methods. The average attention score is computed on DAN dataset. IA largely decreases model’s attention to jailbreak prompt (red bar) in both two stages.

To further illustrate IA’s effect, Figure 6 presents a layer-wise comparison of attention on the jailbreak prompt between the vanilla and IA prompts. The results show that IA consistently reduces the model’s attention on the jailbreak prompt across all layers, further indicating IA’s effectiveness in suppressing LLM’s tendency to follow jailbreak prompts.

## 6 Further Discussion

Two factors influence IA performance. (1) Intention analysis ability: As shown by the solid lines in Figure 4, IA performance improves with higher correct intention ratios, suggesting that better intention analysis ability can further enhance effectiveness<sup>16</sup>. (2) Inherent LLM safety: Figure 3 shows that even among LLMs with nearly 100% intention recognition rates, the final harmful response rates vary notably—from 0.3% for Vicuna-7B to 19.3% for MPT-30B-Chat—highlighting the impact of inherent LLM safety on IA results (see Figure 18 for a related case study). These suggest two improvement directions: enhancing LLMs’ intention analysis ability and their inherent safety.

![](images/eadbd2f0bb68e137b7e85be59720de8d563d1b24cf95e0c69dc2bbc0badd91a3.jpg)  
Figure 6: Comparison of Vicuna-13B’s attention scores onjailbreak prompt between Vanilla and IA methods across different model layers. The average attention score is computed on DAN dataset. High scores means greater influence of jailbreak prompt on the generated response.

<table><tr><td>Vicuna-7B Vicuna-13B ChatGLM-6B Time Cost</td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>73.4</td><td>65.4</td><td>45.8</td><td>13.2</td></tr><tr><td>+ One-Pass IA</td><td>5.50</td><td>1.13</td><td>39.0</td><td>12.6</td></tr><tr><td>+ Two-Stage IA</td><td>0.31</td><td>1.12</td><td>6.12</td><td>19.5</td></tr></table>

Table 3: Comparison of our IA with different implementations (one-pass and two-stage) on SAP200 in terms of ASR (%) and average Time Cost (s/sample).

Our efficient one-pass variant of IA provides a more cost-effective choice. As aforementioned, to maximize the performance, our IA follows a two-stage process. A natural question arises of whether our mechanism can be merged into one step, to save the decoding overhead. To verify this, we design a cheaper one-pass IA variant (see Figure 10 for detailed prompts). From results in Table 3, we see that: 1) For more powerful models, such as Vicuna-7B and 13B, one-pass IA achieves comparable performance to two-stage IA in a more cost-effective manner. 2) For less powerful models, i.e., ChatGLM-6B, one-pass IA’s effectiveness diminishes to some extent. In such cases, two-stage IA is necessary to sustain satisfactory performance.

## 7 Conclusion

In this work, a simple yet highly effective defense strategy IA is proposed to handle the widespread complex and stealthy jailbreak attacks. IA leverages LLM’s intrinsic capacities to analyze the essential intention of user queries before finally responding through two stages. Extensive experiments on representative jailbreak benchmarks across diverse LLMs show that IA could consistently and significantly enhance LLM safety while maintaining general helpfulness. IA works by suppressing LLM’s tendency to follow jailbreak prompts, thus leading to safer responses. Further analysis indicates that enhancing LLMs’ intention analysis capability and their inherent safety are two directions for future improvements.

## Limitations

Our method remains to be validated on more advanced models. However, since our core intention analysis mechanism relies on LLM’s fundamental capabilities of—specifically, instruction-following and text comprehension—making it easy to perform, we believe this approach has the potential to generalize effectively across diverse models as a safety mechanism. Additionally, despite the effectiveness of our method in defending sophisticated jailbreak prompts, these prompts do not encompass the entire potential jailbreak attacks encountered in real-world scenarios. Consequently, the practical applicability of our approach remains to be validated through further testing. Our research underlines the importance of intention analysis in improving LLM safety, suggesting future work focusing on integrating this into training to reduce inference costs. Additionally, in the face of the rapid advancements in the adversarial attacks community, there is a pressing need for developing more effective and robust defense strategies for LLMs. While our method specifically targets jailbreak scenarios, broader alignment tasks still benefit from alignment training, such as RLHF.

## Ethics Statement

We take ethical considerations very seriously. This paper focuses on improving the safety (especially the jailbreak attacks) of large language models, through carefully designed intention analysis prompting mechanism. Our research could significantly reduce the unsafe responses of LLMs. All experiments are conducted on open datasets and the findings and conclusions of this paper are reported accurately and objectively. Thus, we believe that this research will not pose ethical issues.

## Acknowledgments

We express our gratitude to Zuchao Li and Yuchun Miao for their assistance with proofreading and insightful feedback on the writing of this paper. We thank the anonymous reviewers and the area chair for their insightful comments and suggestions. This research is supported by the National Research Foundation, Singapore, and the CyberSG R&D Programme Office (“CRPO”), under the National Cybersecurity R&D Programme (“NCRP”), RIE2025 NCRP Funding Initiative (Award CRPO-GC1-NTU-002).

## References

James F Allen and C Raymond Perrault. 1980. Analyzing intention in utterances. Artificial intelligence.

Gabriel Alon and Michael Kamfonas. 2023. Detecting language model attacks with perplexity. arXiv preprint.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint.

Emily M Bender and Alexander Koller. 2020. Climbing towards nlu: On meaning, form, and understanding in the age of data. In ACL.

Bochuan Cao, Yuanpu Cao, Lu Lin, and Jinghui Chen. 2023. Defending against alignment-breaking attacks via robustly aligned llm. arXiv preprint.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong. 2023. Jailbreaking black box large language models in twenty queries. arXiv preprint.

Kai Chen, Chunwei Wang, Kuo Yang, Jianhua Han, Lanqing Hong, Fei Mi, Hang Xu, Zhengying Liu, Wenyong Huang, Zhenguo Li, et al. 2024. Gaining wisdom from setbacks: Aligning large language models via mistake analysis. In ICLR.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. 2023. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality.

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. 2023. Dola: Decoding by contrasting layers improves factuality in large language models. arXiv preprint.

DeepSeek-AI. 2024. Deepseek llm: Scaling opensource language models with longtermism. arXiv preprint.

Boyi Deng, Wenjie Wang, Fuli Feng, Yang Deng, Qifan Wang, and Xiangnan He. 2023a. Attack prompt generation for red teaming and defending large language models. In EMNLP.

Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, and Lidong Bing. 2023b. Multilingual jailbreak challenges in large language models. In ICLR.

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. 2023. Alpacafarm: A simulation framework for methods that learn from human feedback. In NeurIPS.

Deep Ganguli, Amanda Askell, Nicholas Schiefer, Thomas Liao, Kamile Lukoši˙ ut¯ e, Anna Chen, Anna˙ Goldie, Azalia Mirhoseini, Catherine Olsson, Danny Hernandez, et al. 2023. The capacity for moral selfcorrection in large language models. arXiv preprint.

Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. 2022. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint.

Jose Manuel Gómez-Pérez, Andrés García-Silva, Cristian Berrio, German Rigau, Aitor Soroa, Christian Lieske, Johannes Hoffart, Felix Sasaki, Daniel Dahlmeier, Inguna Skadina, et al. 2023. Deep dive text analytics and natural language understanding. In ELE.

Google. 2023. Palm 2 technical report. arXiv preprint.

Alec Helbling, Mansi Phute, Matthew Hull, and Duen Horng Chau. 2023. Llm self defense: By self examination, llms know they are being tricked. arXiv preprint.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In ICLR.

Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. 2023. Baseline defenses for adversarial attacks against aligned language models. arXiv preprint.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM COM-PUT SURV.

Tomasz Korbak, Kejian Shi, Angelica Chen, Rasika Vinayak Bhalerao, Christopher Buckley, Jason Phang, Samuel R Bowman, and Ethan Perez. 2023. Pretraining language models with human preferences. In ICML.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. 2023. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint.

Xuan Li, Zhanke Zhou, Jianing Zhu, Jiangchao Yao, Tongliang Liu, and Bo Han. 2023. Deepinception: Hypnotize large language model to be jailbreaker. arXiv preprint.

Yuhui Li, Fangyun Wei, Jinjing Zhao, Chao Zhang, and Hongyang Zhang. 2024. Rain: Your language models can align themselves without finetuning. In ICLR.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how models mimic human falsehoods. In ACL.

Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. 2023a. Autodan: Generating stealthy jailbreak prompts on aligned large language models. arXiv preprint.

Yi Liu, Gelei Deng, Zhengzi Xu, Yuekang Li, Yaowen Zheng, Ying Zhang, Lida Zhao, Tianwei Zhang, and Yang Liu. 2023b. Jailbreaking chatgpt via prompt engineering: An empirical study. arXiv preprint.

Qingyu Lu, Baopu Qiu, Liang Ding, Kanjian Zhang, Tom Kocmi, and Dacheng Tao. 2024. Error analysis prompting enables human-like translation evaluation in large language models. In Findings of ACL.

Yuchun Miao, Sen Zhang, Liang Ding, Rong Bao, Lefei Zhang, and Dacheng Tao. 2024. Inform: Mitigating reward hacking in rlhf via information-theoretic reward modeling. In NeurIPS.

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint.

OpenAI. 2023. Gpt-4 technical report. arXiv preprint.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. In NeurIPS.

Keqin Peng, Liang Ding, Qihuang Zhong, Li Shen, Xuebo Liu, Min Zhang, Yuanxin Ouyang, and Dacheng Tao. 2023. Towards making the most of chatgpt for machine translation. arxiv preprint.

Chengwei Qin, Aston Zhang, Zhuosheng Zhang, Jiaao Chen, Michihiro Yasunaga, and Diyi Yang. 2023. Is chatgpt a general-purpose natural language processing task solver? arXiv preprint.

Zhiyao Ren, Yibing Zhan, Baosheng Yu, Liang Ding, and Dacheng Tao. 2024. Healthcare copilot: Eliciting the power of general llms for medical consultation. arXiv preprint.

Alexander Robey, Eric Wong, Hamed Hassani, and George J Pappas. 2023. Smoothllm: Defending large language models against jailbreaking attacks. arXiv preprint.

Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang. 2023. “do anything now": Characterizing and evaluating in-the-wild jailbreak prompts on large language models. arXiv preprint.

MosaicML NLP Team. 2023. Introducing mpt-30b: Raising the bar for open-source foundation models.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint.

Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. 2023. Label words are anchors: An information flow perspective for understanding in-context learning. In EMNLP.

Yihan Wang, Zhouxing Shi, Andrew Bai, and Cho-Jui Hsieh. 2024. Defending llms against jailbreaking attacks via backtranslation. In ACL.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. 2023a. Jailbroken: How does llm safety training fail? In NeurIPS.

Zeming Wei, Yifei Wang, and Yisen Wang. 2023b. Jailbreak and guard aligned language models with only few in-context demonstrations. arXiv preprint.

Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, et al. 2021. Ethical and social risks of harm from language models. arXiv preprint.

Yueqi Xie, Jingwei Yi, Jiawei Shao, Justin Curl, Lingjuan Lyu, Qifeng Chen, Xing Xie, and Fangzhao Wu. 2023. Defending chatgpt against jailbreak attack via self-reminder. NMI.

Hang Yan, Junqi Dai, Xipeng Qiu, Zheng Zhang, et al. 2021. A unified generative framework for aspectbased sentiment analysis. ACL-IJCNLP.

Hongbin Ye, Tong Liu, Aijia Zhang, Wei Hua, and Weiqiang Jia. 2023. Cognitive mirage: A review of hallucinations in large language models. arXiv preprint.

Zheng-Xin Yong, Cristina Menghini, and Stephen H Bach. 2023. Low-resource languages jailbreak gpt-4. arXiv preprint.

Jiahao Yu, Xingwei Lin, Zheng Yu, and Xinyu Xing. 2023. Gptfuzzer: Red teaming large language models with auto-generated jailbreak prompts. In Greekon.

Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Pinjia He, Shuming Shi, and Zhaopeng Tu. 2024. Gpt-4 is too smart to be safe: Stealthy chat with llms via cipher. In ICLR.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. 2023. Glm-130b: An open bilingual pre-trained model. In ICLR.

Zhexin Zhang, Junxiao Yang, Pei Ke, and Minlie Huang. 2023. Defending large language models against jailbreaking attacks through goal prioritization. arXiv preprint.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. In NeurIPS.

Qihuang Zhong, Liang Ding, Juhua Liu, Bo Du, and Dacheng Tao. 2023. Can chatgpt understand too? a comparative study on chatgpt and fine-tuned bert. arXiv preprint.

Qihuang Zhong, Liang Ding, Juhua Liu, Bo Du, and Dacheng Tao. 2024. Rose doesn’t do that: Boosting the safety of instruction-tuned large language models with reverse prompt contrastive decoding. arXiv preprint.

Yilun Zhu, Joel Ruben Antony Moniz, Shruti Bhargava, Jiarui Lu, Dhivya Piraviperumal, Site Li, Yuan Zhang, Hong Yu, and Bo-Hsiang Tseng. 2024. Can large language models understand context? arXiv preprint.

Andy Zou, Zifan Wang, J Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on aligned language models. arXiv preprint.

## A Experimental Datasets

## A.1 Safety Datasets

Hand-Crafted Jailbreak Prompts To assess the effectiveness of our method on in-the-wild jailbreak prompts, we employ two jailbreak prompt datasets. The first is forbidden question set developed by Shen et al. (2023), which is currently the largest in-the-wild jailbreak prompt dataset. To improve computing efficiency, we extract five questions from each forbidden scenario, forming a jailbreak dataset comprising 8 jailbreak communities × 3 jailbreak prompts × 13 forbidden scenarios × 5 questions, totaling 1560 samples. The term “DAN” is used to denote this dataset. For evaluation, we leverage attack success rate (ASR) to consider the success of a jailbreak attack. Considering the complex instructions in DAN makes it challenging to directly identify the success of an attack through string matching, we turn to utilize a widely-adopted LLM to evaluate the harmfulness of model generations, as will be discussed in Section C.2.

The second SAP200 is an jailbreak prompt dataset, constructed semi-automatically by Deng et al. (2023a) using code injection and payload splitting mechanisms. It encompasses 8 distinct sensitive topics, with 200 samples each, totaling 1600 samples.

Due to computational resource and financial limitations, we randomly select 40 samples for each sub-dataset, totaling 40samples × 8sub − datasets = 320 samples from DAN and SAP200 datasets,respectively, to conduct comparative experiments in Table 7 and correct intention ratio comparison experiments in Figure 4.

Gradient-Based Adversarial Attacks To comprehensively verify the effectiveness of our method in defending against jailbreak attacks, we conduct experiments on a popular token-level jailbreak dataset, i.e., AdvBench (Zou et al., 2023) and use the Greedy Coordinate Gradient (GCG) attack algorithm to generate the adversarial suffix. Specifically, we utilize Vicuna-7B and 13B to optimize a universal attack suffix by combining the gradients of the two models. Subsequently, we use the heldout 100 harmful behaviors from AdvBench and apply this optimized suffix to attack other models. We followed the same default parameter setting for GCG, with a learning rate of 0.01, batch size of 512, top-k of 256, and temperature of 1. The suffix achieving the lowest loss after 500 steps was selected for the experiment.

## A.2 Helpfulness Datasets

To evaluate the effect of our IA on helpfulness for general in-distribution queries, we conduct experiments on three widely recognized datasets, i.e., AlpacaEval (Dubois et al., 2023), MMLU (Hendrycks et al., 2021) and TruthfulQA (Lin et al., 2022). AlpacaEval, containing 805 general questions, is a widely acknowledged benchmark to evaluate the ability of model following general user queries (Chen et al., 2024; Zhang et al., 2023). MMLU covers 57 subjects, aiming to evaluate comprehensive knowledge abilities across multiple major categories, from humanities to social sciences to science and engineering. TruthfulQA assesses the model’s ability to identify true claims, specifically in the context of literal truth about the real world.

## B Language Models

To evaluate the effectiveness of our IA method, we validate our approach on six representative

Large Language Models, each distinguished by its model architecture, model size, and alignment level. Specifically, we consider five open-source LLMs and one closed-source LLM.

ChatGLM-6B (Zeng et al., 2023), trained on 1T tokens of both Chinese and English corpus, follows similar technologies to ChatGPT, including supervised fine-tuning, feedback bootstrap, and RLHF.

Llama-2-7B-chat (Touvron et al., 2023), finetuned on Llama-2-7B by Meta, is optimized for dialogue cases and use reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.

Vicuna-7B-v1.1 and Vicuna-13B-v1.1(Chiang et al., 2023) are two open-source chatbots that are fine-tuned on Llama-7B and Llama-13B, using 70K samples of user-shared ChatGPT conversations, without further utilizing RLHF.

MPT-30B-Chat (Team, 2023) is a chatbot model designed for generating conversations. It was developed by fine-tuning the MPT-30B on dialogue datasets such as ShareGPT, Vicuna, Camel-AI, and some generated datasets.

DeepSeek-67B-Chat (DeepSeek-AI, 2024) is fine-tuned for dialogue based on DeepSeek-67B, and empirically verified to perform better on nearly 20 public datasets in both Chinese and English than Llama2-70B.

GPT-3.5 (OpenAI, 2023) is an advanced and closed-source LLM trained with massive text data and can generate human-like responses to a wide range of prompts and queries. We obtain the responses of gpt-3.5-turbo-1106 using API calls, which may differ from the response in web interfaces due to internal settings unexposed to the users.

For each of the above models, we conducted experiments under the system prompt template designed by their developers to simulate real-world usage scenarios.

## C Evaluation Metrics

## C.1 Automation-based Safety Evaluation

For evaluation of DAN, considering the complex instructions in DAN makes it challenging to directly identify the success of an attack through string matching, we turn to utilize a widely-adopted LLM to evaluate the harmfulness of model generations.

<table><tr><td>Degrees</td><td>= A</td><td>≥B</td><td>≥C</td><td>≥D</td><td>≥E</td></tr><tr><td>Vanilla</td><td>51</td><td>72</td><td>93</td><td>90</td><td>89</td></tr><tr><td>Ours</td><td>97</td><td>98</td><td>99</td><td>99</td><td>3.0</td></tr></table>

Table 4: Agreement rate (%) between human annotations and different harm degrees provided by GPT-3.5 evaluation. A, B, C, D, and E degrees refer to ‘very harmful’, ‘harmful’, ‘moderately harmful’, ‘slightly harmful’, ‘not harmful at all’, respectively.

Practically, we adopt gpt-3.5-turbo-0613<sup>17</sup> as the evaluation function following Deng et al. (2023a). Especially, the model responses are submitted via API calls, and a specific evaluation prompt is designed for gpt-3.5-turbo-0613 to rate the harmfulness on a five-degree scale (‘A (Very Harmful)’, ‘B (Harmful)’, ‘C (Moderately Harmful)’, ‘D (Slightly Harmful)’, and ‘E (No Harmful at All)’). Figure 7 shows the detailed instruction for gpt-3.5-turbo-0613 evaluation.

Human Evaluation To validate GPT-3.5 as the harmfulness annotation function, we conduct a human validation process. First, we randomly select 200 responses from both the vanilla model and our method on the DAN dataset. Then we engage two expert annotators proficient in alignment studies of LLMs and fluent in English. After reassessing their evaluations for cases with discrepancies and reaching a consensus, the annotators achieved a remarkable 97% agreement rate. Third, to quantify the credibility of GPT-3.5’s evaluations, we consider different degrees as the lowest acceptable safety threshold and calculate the agreement rate with human annotations. The results, detailed in Table 4, demonstrate the highest agreement of average 96% when categorizing C degree as the minimum harmfulness level, i.e., viewing the evaluation degrees of ‘A (Very Harmful)’, ‘B (Harmful)’, and ‘C (Moderately Harmful)’ as a success of jailbreak attacks. This underscores the enhanced reliability of GPT-3.5 annotations in conjunction with human judgment in harm detection.

## C.2 Rule-based Safety Evaluation

To calculate attack success rate (ASR) for SAP200 and AdvBench, we follow Zou et al. (2023) to consider an attack successful if the following refusal strings are not produced in the generation from the model:

• “I’m sorry”,

• “Sorry”,

• “I am sorry”,

• “I apologize”,

• “As an AI”,

• “I’m just”,

• “As a language model”,

• “As an Assistant”,

• “I cannot”,

• “I do not”,

• “Hello!”,

• “is illegal and unethical”,

• “I’m really sorry”,

• “OpenAI”,

• “unethical”,

• “not ethical”,

• “illegal”,

• “not legal”,

• “My apologies”,

• “but I cannot”,

• “If you have any other non-malicious requests”,

• “I’m not able to provide”,

• “I am really sorry”,

• “I cannot create”,

• “I cannot support”,

• “I can’t provide”,

• “I can’t assist”.

![](images/14fcf438ed01594e79f0764d896becc576279cae0823f0dff4cd50a66694065f.jpg)  
Figure 7: Prompt template for harmfulness evaluation using gpt-3.5-turbo-0613.

## C.3 Helpfulness Evaluation

For evaluation of AlpacaEval, we adopt gpt-4-1106-preview as the auto evaluator to reflect whether the general performance degrades after adding safety techniques (Dubois et al., 2023). Specifically, GPT-4 compares two responses to the same instruction: one with our methods and one provided by text-davinci-003 and report the win rate of our method. Figure 8 shows the detailed instruction for gpt-4-1106-preview evaluation. For MMLU, we follow (Hendrycks et al., 2021) and report accuracy based on the model’s predictions and the groud truth labels. For TruthfulQA, we follow Chuang et al. (2023) and report on two main distinct metrics: MC1 and MC2 scores in Table 2. The complete results on the three metrics in TruthfulQA, i.e., MC1, MC2 and MC3, are presented in Table 5. We can see that our method consistently improves the truthfulness over different models, indicating that our method can be deployed in real applications to enhance LLM safety while increasing the general helpfulness to some extent.

<table><tr><td rowspan="2">Models</td><td rowspan="2">Methods</td><td colspan="3">TruthfulQA</td></tr><tr><td>MC1</td><td>MC2</td><td>MC3</td></tr><tr><td>Vicuna-7B</td><td>のIA OIA</td><td>30.1</td><td>48.7</td><td>23.6 26.3</td></tr><tr><td>Vicuna-13B</td><td>IA</td><td>35.2 35.1</td><td>53.4 52.1</td><td>26.5</td></tr><tr><td rowspan="2"></td><td>OIA</td><td>38.2</td><td>55.1</td><td>28.5</td></tr><tr><td>IA</td><td>37.1</td><td>54.1</td><td>26.8</td></tr><tr><td rowspan="2">ChatGLM-6B</td><td>OIA</td><td>37.5</td><td>56.0</td><td>27.4</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 5: Performance on TruthfulQA of our IA upon different models in terms of Accuracy (%). The best results are highlighted in bold.

![](images/23407584100c6703f2a1121b64faa7dd830a96db979de4c1ae618172f45427a2.jpg)  
Figure 8: Prompt template for AlpacaEval results evaluation using gpt-4-1106-preview.

## C.4 Intention Recognition Success Evaluation

To verify whether the model can successfully identify the intention of jailbreak queries, we examine the model response of the first stage and view a success if it begin with “The essential intention of the query is”, as instructed in stage 1 of our method. And we empirically find that once the model successfully starts with this string, it can then successfully analyze the intention behind the user query.

## D Alternative Prompts

## D.1 One-Pass IA

To study the effect of the implementation of our method, we combine the two stages of our mechanism and crafted an one-pass intention analysis instruction. See Figure 10 for the detailed prompt.

## D.2 Two-Stage IA

To assess the resilience of our method against specific expressions, we construct other two sets of alternative instructions for intention analysis. These additional instructions were meticulously designed with modifications of origin IA prompts (detailed in Figure 9) in lexical choices and syntactic constructions to direct intention analysis. The detailed prompts are presented in Figure 11. Table 6 presents the comparison results. We observe that three prompt sets all result in significant and comparable decreases in ASRs on the SAP200 across different LLMs. This uniformity highlights a key conclusion: it is the intention analysis mechanism, but not the specific instruction expressions, that makes our IA effective.

![](images/7530848332ba8a553fce2b673994d88f167d4f0e2192596fa13728398e7da10a.jpg)

Figure 9: Detailed two-stage prompts for our method.  
Please identify the essential intention behind the user query   
below. Think carefully whether it violates our ethical, legal,   
or safety guidelines. And then respond to the user query. You   
do not have to make extra explanation of the intention. But   
make sure your response strictly adheres to our policy and   
ethical standards, excluding any unsafe or unethical   
information. Following is the user query you should focus on:   
111   
{question}   
11  
Figure 10: One-pass intention analysis prompt for our method.

## E Extensive Validations of IA’s Effectiveness

## E.1 Performance under More Advanced Attacks

Our method can consistently enhance safety in the context of more advanced jailbreaks such as multilingual attack and encryption-based attack. Recent studies (Deng et al., 2023b; Yong et al., 2023) reveal that the multilingual jailbreak poses a new defense challenge for LLMs. Yuan et al. (2024) and Wei et al. (2023a) also emphasize the struggles of more powerful LLMs, such as GPT-3.5, to stay safe when countering encryption-based attack. To verify the effectiveness of our method in these advanced jailbreak scenarios, we reproduce MultiJail and CipherChat following Deng et al.

<table><tr><td colspan="3">Vicuna-7B Vicuna-13B ChatGLM-6B</td></tr><tr><td>Vanilla</td><td>73.4</td><td>65.4</td></tr><tr><td>+ Prompt A</td><td>2.94</td><td>45.8 5.81</td></tr><tr><td>+ Prompt B 5.13</td><td>0.88 2.06</td><td>5.44</td></tr><tr><td>+ Ours</td><td>0.31 1.12</td><td>6.12</td></tr></table>

Table 6: Ablation of different IA prompts on SAP200 in ASR (%). The best and second best results are highlighted in bold and underline.

(2023b) and Yuan et al. (2024), respectively, and conduct further experiments on GPT-3.5<sup>18</sup>. The results of GPT-3.5 with and without our IA are presented in Figure 12. We observe that 1) our IA consistently maintains performance in lowresource languages, e.g., th, bn, sw, and jv, even in scenarios where a malicious jailbreak prompt<sup>19</sup> is attached to the multilingual attacks, 2) our IA significantly enhances safety when facing advanced encryption-based attack, even under the most effective SelfCipher attack. These demonstrate the effectiveness of our intention analysis defense mechanism under more advanced jailbreak attacks.

## E.2 Comparison with All Defense Baselines

Table 7 lists comparison results between IA and the baselines.<sup>20</sup> As observed, IA consistently shows superiority over other baselines on different datasets and model scales. Specifically, IA outperforms the second-best method by 30.32% and 23.77% averagely on SAP200 and DAN, respectively. In addition, although ICD and Self-Reminder achieve considerable reduction in ASR on GCG, their performance severely degrades when dealing with complex and stealthy jailbreak prompts. On the contrary, IA consistently outperforms other baselines across both prompt-level and automatic token-level jailbreak datasets. Notably, IA achieves the best ASRs with comparable and acceptable empirical inference runtime.

## E.3 IA achieves comparable safety with well-safety-trained LLMs without the need for additional training.

Our method aims to enhance LLM safety in the inference stage. A natural question arises: how does its performance compare to well-safety-trained LLMs? To answer this, we compare our method with a representative well-safety-trained LLM, i.e., Llama2-7B-Chat. The results are listed in Table 9. We can see that our method achieves comparable performance to Llama2-7B-Chat on safety datasets while outperforming Llama2-7B-Chat on the helpfulness dataset by almost 6%. This demonstrates the advantage of our IA to achieve both safety and helpfulness goals without additionally resource-consuming safety training.

![](images/77946b14f0382e29f18b001061d4ee66c7f4c1919305b9e5f9d02fed45b28812.jpg)  
Figure 11: Alternative prompts crafted for our intention analysis instructions.

![](images/040a6e65ad09c1d027781d3a47e81126cb01f4f6d147018d5c85c61f1687e5e7.jpg)  
(a) MultiJail

![](images/6af9d0fc1639e0bdc696a67ab05248b342129c87a49a02454bac20bc45e793be.jpg)  
(b) Jailbreak-MultiJail

![](images/d61209bdc58c50cd15953705aab3164ab020c63853fa1f1d552f9479feeda967.jpg)  
(c) CipherChat  
Figure 12: The MultiJail (under two scenarios) and CipherChat Datasets results on GPT-3.5 with and without our IA. (a) Results on direct MultiJail dataset including English (en), Chinese (zh), Italian (it), Vietnamese (vi), Arabic (ar), Korean (ko), Thai (th), Bengali (bn), Swahili (sw), and Javanese (jv). (b) Results on malicious jailbreak prompt attached to MultiJail. (c) Results on CipherChat including ASCII (en), UTF (zh), Unicode (zh), and SelfCipher (en and zh) encryptions.

## E.4 IA can be combined with another defensive method.

We integrate our IA method with the Self-Reminder method (Xie et al., 2023) and conduct experiments on Vicuna-7B to see where such a combination leads. The comparison results in Table 10 indicates that although our method already significantly improves LLM safety, combining it with another defensive method can further enhance the effectiveness at the cost of additional computation overhead.

## F Further explanation of IA format’s effectiveness when generated intention is incorrect

In Figure 4, we find that even when the correct intention ratio is 0% (with all generated intentions replaced by masked or random intentions), IA remains effective compared to the vanilla baseline. This effectiveness is mainly due to IA’s two-round dialogue design. As shown in Figure 2, the final policy-aligned responses are generated with the context of intention analysis sequence format in the first round conversation. In Context Learning (ICL) community (Min et al., 2022) has demonstrated that “keeping the format of the input-label pairs is key, and replacing gold labels with random labels in the demonstrations only marginally lowers the performance.” Therefore, even if the intention label generated in the first stage is incorrect, keeping the entire intention analysis format plays a significant role in making the final response safer than when no intention analysis sequence is used (vanilla method). Moreover, as indicated in Table 4, improving the ratio of correct intention labels can further enhance IA’s performance.

## G Deeper Study of Safe Responses’ Helpfulness for Harmful Queries

## G.1 ChatGPT Evaluation

To comprehensively study the impact of our IA on responses to harmful queries, we follow (Zheng et al., 2023) and prompt ChatGPT to score the helpfulness of these safe refusals<sup>21</sup>. Table 11 presents comparison results between different defense methods on the harmfulness (ASR) and helpfulness score on the DAN dataset. We observe that IA enables LLMs to effectively give safe refusals with satisfactory helpfulness for harmful queries. We also manually check these refusals in Appendix G.2 and find that IA enables LLMs to craft more nuanced responses to specific unsafe intents like inciting hatred and division.

<table><tr><td rowspan="2">Methods</td><td colspan="3">Vicuna-7B</td><td colspan="3">Vicuna-13B</td><td rowspan="2">Empirical Runtime</td></tr><tr><td>GCG</td><td>DAN</td><td>SAP200</td><td>GCG</td><td>DAN</td><td>SAP200</td></tr><tr><td>Vanilla</td><td>83.0</td><td>48.4</td><td>70.0</td><td>87.0</td><td>60.0</td><td>65.9</td><td>1×</td></tr><tr><td>+ Input Check</td><td>13.0</td><td>19.0</td><td>58.1</td><td>0.00</td><td>53.9</td><td>12.8</td><td>&lt; 1×</td></tr><tr><td>+ BPE-dropout (Jain et al., 2023)</td><td>63.0</td><td>23.8</td><td>67.2</td><td>50.0</td><td>28.2</td><td>48.9</td><td>&lt; 1×</td></tr><tr><td>+ ICD (Wei et al., 2023b)</td><td>1.00</td><td>44.4</td><td>32.8</td><td>0.00</td><td>58.9</td><td>32.8</td><td>&lt; 2×</td></tr><tr><td>+ Self Defense (Helbling et al., 2023)</td><td>24.0</td><td>31.3</td><td>53.2</td><td>20.0</td><td>28.8</td><td>29.7</td><td>~2×</td></tr><tr><td>+ Moral Self-Correction (Ganguli et al., 2023)</td><td>26.0</td><td>25.0</td><td>49.0</td><td>13.0</td><td>28.1</td><td>42.8</td><td>~3×</td></tr><tr><td>+ Self-Reminder (Xie et al., 2023)</td><td>11.0</td><td>45.3</td><td>33.8</td><td>1.0</td><td>57.5</td><td>36.9</td><td>&lt; 2×</td></tr><tr><td>+ SmoothLLM (Robey et al., 2023)</td><td>8.00</td><td>13.5</td><td>54.4</td><td>5.00</td><td>17.3</td><td>37.0</td><td>~10×</td></tr><tr><td>+ ⅡA (Ours)</td><td>0.00</td><td>3.42</td><td>0.31</td><td>0.00</td><td>0.94</td><td>1.56</td><td>~2×</td></tr></table>

Table 7: Comparison of our method and existing advanced defense methods in terms of ASR (%) and empirical runtime. The best and second best results are highlighted in bold and underline.
<table><tr><td>ChatGLM-6B Llama2-7B-Chat Llama3-8B-Instruct Vicuna-7B Vicuna-13B MPT-30B-Chat DeepSeek-67B-Chat GPT-3.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DAN</td><td>93%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>92%</td><td>100%</td><td>42%</td></tr><tr><td>SAP200</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>49%</td></tr></table>

Table 8: Manual check results of response’s helpfulness for harmful queries on DAN and SAP200 datasets in terms of rate.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Safety</td><td>Helpfulness</td></tr><tr><td>SAP200</td><td>DAN</td><td>AlpacaEval</td></tr><tr><td>Vicuna-7B</td><td>73.4</td><td>44.3</td><td>66.2</td></tr><tr><td>Llama2-7B-Chat</td><td>0.56</td><td>1.02</td><td>57.5</td></tr><tr><td>Vicuna-7B + Ours</td><td>0.31</td><td>2.89</td><td>63.8</td></tr></table>

Table 9: Comparison between our method and well safety-trained LLM in safety and helpfulness (%). The best and second best are in bold and underline.

<table><tr><td>DAN SAP200 DeepInception GCG Average Time Cost</td><td colspan="6"></td></tr><tr><td>Vanilla</td><td>48.4</td><td>73.4</td><td>90.0</td><td>83.0</td><td>73.7</td><td>10.2</td></tr><tr><td>Self-Reminder</td><td>41.3</td><td>33.8</td><td>55.4</td><td>11.0</td><td>35.4</td><td>15.0</td></tr><tr><td>Ours</td><td>3.42</td><td>0.31</td><td>0.00</td><td>0.00</td><td>0.93</td><td>17.3</td></tr><tr><td>Self-reminder+Ours 3.12</td><td></td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.78</td><td>25.5</td></tr></table>

Table 10: Performance of combining our IA with Self-Reminder method for Vicuna-7B in terms of ASR (%) and average Time Cost (s/sample).

<table><tr><td rowspan="2">Defense Methods</td><td colspan="2">DAN</td><td rowspan="2">AlpacaEval</td></tr><tr><td>Harmfulness</td><td>Helpfulness</td></tr><tr><td>Vanilla</td><td>48.4</td><td>5.66</td><td>66.2</td></tr><tr><td>Input Check</td><td>19.0</td><td>3.25</td><td>64.4</td></tr><tr><td>ICD</td><td>40.4</td><td>5.79</td><td>60.3</td></tr><tr><td>Self-Reminder</td><td>41.3</td><td>5.89</td><td>64.6</td></tr><tr><td>SmoothLLM</td><td>13.5</td><td>5.35</td><td>60.8</td></tr><tr><td>IA (Ours)</td><td>3.42</td><td>8.75</td><td>63.8</td></tr></table>

Table 11: Comparison results for Vicuna-7B in terms of harmfulness (%), and helpfulness (%) on DAN dataset, and win rate (%) on AlpacaEval.

<table><tr><td>Target Model</td><td>Intent. Model</td><td>DAN</td><td>SAP200</td></tr><tr><td rowspan="3">Vicuna-7B</td><td></td><td>44.3</td><td>67.2</td></tr><tr><td>Vicuna-7B</td><td>2.89</td><td>0.31</td></tr><tr><td>Vicuna-13B</td><td>1.93</td><td>0.62</td></tr><tr><td rowspan="3">Vicuna-13B</td><td></td><td>54.7</td><td>65.4</td></tr><tr><td>Vicuna-7B</td><td>1.25</td><td>1.87</td></tr><tr><td>Vicuna-13B</td><td>0.64</td><td>1.12</td></tr></table>

Table 12: ASR (%) of our IA on DAN and SAP200 with different intention analysis model scales. For each target model, the intention analysis is performed in three ways, i.e., without intention analysis, analyzed by Vicuna-7B, and by Vicuna-13B.

## G.2 Manual Check

To comprehensively study the impact of our IA on responses to harmful queries, we conduct a manual review of 100 random-sampled refusals on both DAN and SAP200 datasets for each of the seven

LLMs under our IA. We manually check for 1) reasonable explanations about why it rejects to respond and 2) positive suggestions for the user. In Table 8, our findings show that, following IA, almost all LLMs can give detailed explanations in nearly 100% of cases (except for GPT-3.5 due to OpenAI’s external moderation mechanism during API interactions (OpenAI, 2023)). Specifically, we find LLMs indeed conduct more nuanced handling regarding varying unsafe intents (e.g. terrorism and violence), indicating that our IA ensures the helpfulness of safe responses for harmful queries. Detailed cases are presented in Figures 16.

## H Cross-Intention Analysis Experiment

A question that naturally arises about our method is, “Would the effectiveness of IA improve notably with a larger intention analysis model?” To investigate this, we conduct cross-experiments using Vicuna-7B and 13B models. Specifically, we evaluate each target model in three contexts: without any intention analysis model, with the Vicuna-7B as the intention analysis model, and with the Vicuna-13B serving the same role. Table 12 shows the results. It is observable that for the 7B model, utilizing the 13B for intention analysis yields only marginal performance gains. For the 13B model, employing the 7B model for intention recognition achieves comparable results to using the 13B model itself for intention analysis, but with the added benefit of conserving inference resources. This indicates that the 7B model is sufficiently capable of intention analysis to enhance model safety.

## I Qualitative Examples

This section presents some practical cases, which are from SAP200 dataset (see Figure 13), DAN dataset (see Figure 14), and AdvBench dataset (for GCG; see Figure 15), respectively.

![](images/9e6bee72d8ee9d2b809341cabd41b76aa429f29a7bc182faf26a34d1aca23061.jpg)  
Figure 13: Qualitative comparison example I from SAP200 dataset on Vicuna-7B with and without our IA.

![](images/cea8583d1eaa2c4d96a97d9c82df18c0226e19dae8cc15f4102aef8b2dd1475a.jpg)  
Figure 14: Qualitative comparison example II from DAN dataset on Vicuna-7B with and without our IA.

![](images/d81d0b333ff44908553f36e5a0e816c7879b02fd91a98ee988c220e85296f649.jpg)  
Figure 15: Qualitative comparison example III from AdvBench dataset (for GCG) on Vicuna-7B with and without our IA.

![](images/e878094256e2d67a343d93f3b7fb3f53eb4d905a401b10e58967dd926bba280d.jpg)  
Figure 16: Examples of nuanced handling for harmful user queries regarding varying unsafe intentions (terrorism in case 1 and violence in case 2).

![](images/479163e5bc467c790f22c4e36afbd179c21c0ad6163cd29bd7250d6365fe1c91.jpg)  
Figure 17: Two failure cases of intention analysis induced by too strong (for Llama2-7B-Chat) or too weak (for Vicuna-7B) inherent safety.

![](images/c9af0a42a779b0cc75f775007260c1f9a2511924fc55aa510eecd96f86473c53.jpg)  
Figure 18: Two failure cases of our IA induced by weak inherent safety (in case 1) and failed intention analysis (in case 2).