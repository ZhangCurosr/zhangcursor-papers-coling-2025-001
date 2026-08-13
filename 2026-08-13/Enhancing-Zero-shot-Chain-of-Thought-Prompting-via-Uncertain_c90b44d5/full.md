# Enhancing Zero-shot Chain of Thought Prompting via Uncertainty-Guided Strategy Selection

Shanu Kumar Saish Mendke Karody Lubna Abdul Rahman

Santosh Kurasa Parag Agrawal Sandipan Dandapat

Microsoft Corporation, India

{shankum,saishmendke,lubnakarody,skurasa,paragag,sadandap}@microsoft.com

## Abstract

Chain-of-thought (CoT) prompting has significantly enhanced the capability of large language models (LLMs) by structuring their reasoning processes. However, existing methods face critical limitations: handcrafted demonstrations require extensive human expertise, while trigger phrases are prone to inaccuracies. In this paper, we propose the Zero-shot Uncertainty-based Selection (ZEUS) method, a novel approach that improves CoT prompting by utilizing uncertainty estimates to select effective demonstrations without needing access to model parameters. Unlike traditional methods, ZEUS offers high sensitivity in distinguishing between helpful and ineffective questions, ensuring more precise and reliable selection. Our extensive evaluation shows that ZEUS consistently outperforms existing CoT strategies across four challenging reasoning benchmarks, demonstrating its robustness and scalability.

## 1 Introduction

Large Language Models (LLMs) have achieved remarkable performance in a wide range of natural language processing tasks (Brown et al., 2020; Touvron et al., 2023; Thoppilan et al., 2022). However, they often struggle with tasks that require complex reasoning (Rae et al., 2021; Liang et al., 2022). The "chain-of-thought" (CoT) prompting technique (Wei et al., 2022; Feng et al., 2024) has been proposed to address this limitation by generating intermediate rationales (r) along with the final answer (a) for a given question (q). In this context, few-shot in-context examples, referred to as demonstrations $D = ( q _ { j } , r _ { j } , a _ { j } ) _ { j = 1 } ^ { k }$ , consist of k example questions $q _ { j }$ , manually crafted rationales $r _ { j }$ , and answers a<sub>j</sub>. This approach, known as Manual-CoT, relies on handcrafted rationales to guide the model.

Building on Manual-CoT, Zero-Shot-CoT (Kojima et al., 2022) presents a novel prompting method where LLMs generate rationales using a trigger phrase t (e.g., "Let’s think step by step") appended to the input question q, without requiring manually crafted demonstrations. While Zero-Shot-CoT is cost-effective, its performance often falls short compared to Manual-CoT due to the absence of effective demonstrations.

Crafting rationales manually is typically laborintensive and time-consuming, particularly for tasks demanding intricate reasoning. To mitigate this, Auto-CoT (Zhang et al., 2022) combines Manual-CoT and Zero-Shot-CoT, thereby reducing the performance gap while minimizing manual effort. Auto-CoT employs self-supervised learning on a set of unlabeled questions $Q = \{ q _ { j } \} _ { j = 1 } ^ { m }$ to generate rationales and answers. Demonstrations are created by clustering Q into k groups and selecting a representative question, rationale, and answer from each cluster. This clustering approach aims to maintain diversity in the demonstrations, which can help mitigate the impact of any errors in the generated rationales.

In this work, we seek to enhance the creation of demonstrations that improve LLM performance solely using unlabeled questions Q without any rationale and answer. The selection process of examples $q _ { j }$ in demonstrations D has been to significantly influence LLM performance (Wan et al., 2023), and generating consistent rationales (Wang et al., 2022) is crucial. Recent CoT prompting methods (Diao et al., 2024; Bayer and Reuter, 2024) have utilized Active Learning (AL) (Fu et al., 2013; Settles and Craven, 2008; Rotman and Reichart, 2022; Kumar et al., 2022) to identify examples for human annotation, showing that annotating the most uncertain examples yields the best performance. Drawing on these principles, we propose several selection strategies based on the uncertainty of unlabeled questions.

To estimate uncertainty, we adopt perturbationbased methods (Ribeiro et al., 2020; Kuhn et al., 2023; Gao et al., 2024; Tomani et al., 2024), which operate on the principle that incorrect predictions can be detected by resampling rationales through perturbations, such as temperature adjustments. If the LLM is confident in its prediction, perturbations are unlikely to affect the outcome. However, if the LLM’s prediction is uncertain, different perturbations can lead to varied responses. Our initial experiments reveal that while temperature-based perturbation estimates are well-calibrated, they lack sufficient sensitivity.<sup>1</sup> To address this, we propose a robust method for estimating uncertainty that exhibits near-ideal linearity with accuracy.

Our primary contributions are threefold: i) We present ZEUS,<sup>2</sup> a method for estimating LLM uncertainty that is both well-calibrated and sensitive. ii) We leverage these uncertainty estimates to guide the selection of most informative demonstrations and show that these strategies outperform existing prompting methods across four challenging reasoning tasks. iii) We demonstrate that the performance of ZEUS correlates strongly with few-shot uncertainty estimates on the unlabeled set, providing actionable recommendations for creating effective demonstrations.

## 2 Related Work

Chain-of-Thought (CoT) prompting has significantly influenced various advanced techniques designed to enhance reasoning capabilities. These include Tree of Thoughts (Yao et al., 2023), Role Play (Kong et al., 2024), and Collaborative Prompting (Zhu et al., 2023; Yin et al., 2023; Liang et al., 2023; Wang et al., 2023), each building on the CoT methodology to improve model performance in complex reasoning tasks. Concurrently, Active Learning (AL)-based methods have gained traction in few-shot prompting scenarios. Diao et al. (2023) enhance CoT prompting within an AL framework by actively selecting questions based on an uncertainty metric and manually constructing demonstrations. Shum et al. (2023) work with labeled questions devoid of rationales, generating rationales through pruning and using an AL-inspired variancereduced policy gradient strategy to select the most informative examples. Similarly, Bayer and Reuter (2024) apply uncertainty-based AL methods to identify the most valuable questions for annotation. Unlike these studies, our work addresses a more challenging scenario where neither humanannotated labels nor rationales are available.

Our method relies on accurate uncertainty estimation due to the lack of human supervision. Estimating uncertainty is a well-explored challenge, with methods ranging from Bayesian approaches and ensemble methods to more recent perturbationbased techniques (Hendrycks and Gimpel, 2016; Gal and Ghahramani, 2016; Lakshminarayanan et al., 2017; Guo et al., 2017; Van Amersfoort et al., 2020; Ovadia et al., 2019). Perturbationbased methods, which include techniques like temperature adjustments and question rephrasing, have shown promise in recent studies (Gao et al., 2024). These methods, while effective, may not be universally applicable to LLMs due to their generative nature (Vashurin et al., 2024). Other recent work has explored uncertainty estimation for specific tasks, such as hallucination detection in LLMs (Kuhn et al., 2023; Tomani et al., 2024). We extend these perturbation techniques by enhancing their sensitivity to capture finer distinctions between questions, thereby improving uncertainty estimation in LLMs.

## 3 ZEUS: Zero-shot Uncertainty-based Selection

We propose the ZEUS method, which aims to construct useful demonstrations containing a specific level of required uncertainty. It is comprised of three stages: (i) uncertainty estimation, (ii) uncertainty-based question selection, and (iii) demonstration construction. We have illustrated all the stages of ZEUS in Figure 1.

## 3.1 Uncertainty Estimation (Stage 1)

In the ZEUS method, uncertainty estimation is a critical step and is performed using perturbation. We exploit three distinct types of perturbations to estimate uncertainty for each unlabeled question in the set Q. These perturbations include temperature adjustments, trigger phrase variations, and question rephrasing.

Temperature Perturbation: This perturbation technique is based on the principle that a question can be answered in multiple ways, and these variations can be explored by adjusting the temperature parameter during decoding. Temperature perturbation helps in simulating different reasoning paths within the LLM. When the temperature is set to a higher value, the model’s outputs become more diverse, while a lower temperature typically results in more confident and consistent responses (Koehn, 2009). According to Wang et al. (2022), if an LLM is confident in its answer to a question $q _ { j } .$ , the responses generated at various temperatures should reach the same answer. In contrast, if the LLM is uncertain, different temperatures will yield a range of potentially inconsistent answers. To estimate uncertainty using this property, we generate n responses for a question $q _ { j }$ by using the highest temperature (=1). These responses $\{ r _ { j } ^ { l } \} _ { l = 1 } ^ { n }$ form the basis for our temperature-based uncertainty estimation.

![](images/42bff2df42ae2790c451e314e565de5e308558adba58188183dae86936888554.jpg)  
Figure 1: Overview of ZEUS: Uncertainty for a question $q _ { j }$ is calculated using a pool of answers generated using various prompts, including trigger phrases, non-zero temperature-based decoding, and rephrasing of $q _ { j }$ . Subsequently, questions with uncertainty within a certain range are selected and used for constructing demonstrations.

Trigger Phrase Perturbation: This factor leverages the sensitivity of LLM performance to trigger phrases. Kojima et al. (2022) demonstrated that appending different trigger phrases to a question can affect the LLM’s output. By introducing variations in trigger phrases, we can assess whether the LLM’s responses remain consistent. If the LLM provides the same answer across different trigger phrases, it suggests a high level of confidence in its response. Conversely, varying answers across trigger phrases indicate that the question $q _ { j }$ is challenging or that the LLM is uncertain. To apply this perturbation, we append a set of t different trigger phrases to the original question $q _ { j }$ and generate a corresponding set of responses $\{ r _ { j } ^ { l } \} _ { l = 1 } ^ { t }$

Rephrasing Perturbation: The third technique utilizes rephrasing of the input question to explore the impact on the LLM’s responses. We hypothesize that if the LLM is confident about its answer, rephrasing the question should not significantly alter the generated answer. On the other hand, if the LLM’s answer is influenced by specific biases or ambiguities in the original question, rephrasing may lead to a different response. To estimate uncertainty using rephrasing, we generate v rephrased versions of the question $q _ { j }$ and obtain the sets of responses $\{ r _ { j } ^ { l } \} _ { l = 1 } ^ { v }$

By integrating these three types of perturbations—temperature adjustment, trigger phrase variation, and question rephrasing, we generate a diverse set of responses for each question $q _ { j } .$ . Specifically, we produce a total of $n \times t \times v$ responses. This pool of answers reflects variations due to different decoding settings, trigger phrases, and question rephrasing, serving as Monte Carlo samples from the LLM’s likelihood distribution. From these responses, we identify $C \ ( \leq \ n )$ unique answers $y _ { j } ^ { 1 } , \dotsc , y _ { j } ^ { C }$ for the question $q _ { j }$ . The confidence score $p ( \check { y } _ { j } ^ { c } | { q } _ { j } )$ for each unique answer $y _ { j } ^ { c }$ is computed based on the consistency of responses across the different perturbations. This score quantifies the degree of certainty associated with each answer and serves as a basis for selecting informative demonstrations in subsequent stages of the ZEUS method. The confidence score $p ( \boldsymbol { y } _ { j } ^ { c } | \boldsymbol { q } _ { j } )$ for a unique answer $y _ { j } ^ { c }$ is defined as:

$$
p ( y _ { j } ^ { c } | q _ { j } ) = \frac { 1 } { n } \sum _ { l = 1 } ^ { n } 1 ( y _ { j } ^ { c } = a _ { j } ^ { l } )\tag{1}
$$

where 1(·) is the indicator function that evaluates to 1 if $y _ { j } ^ { c }$ matches $a _ { j } ^ { l }$ and 0 otherwise.

To represent the uncertainty of the LLM regarding the question $q _ { j }$ , we use predictive entropy (PE) (Kumar et al., 2022). PE is maximized when confidence scores are uniformly distributed across many unique answers and increases as the number of unique answers grows. It reaches zero when all answers are identical. The PE for the question $q _ { j }$ is computed as follows:

<table><tr><td>Strategy</td><td>Umin</td><td>Umax</td></tr><tr><td>Trivial</td><td>0</td><td> $\mu - \sigma$ </td></tr><tr><td>Very Easy</td><td>0</td><td> $\mu$ </td></tr><tr><td>Éasy</td><td>0</td><td> $\mu + \sigma$ </td></tr><tr><td>Moderate</td><td> $\mu - \sigma$ </td><td> $\mu$ </td></tr><tr><td>Challenging</td><td> $\mu - \sigma$ </td><td> $\mu + \sigma$ </td></tr><tr><td>Hard</td><td> $\mu - \sigma$ </td><td> $\infty$ </td></tr><tr><td>Very Hard</td><td> $\mu$ </td><td> $\infty$ </td></tr></table>

Table 1: Selection Strategies used in ZEUS with their minimum $\mu _ { \mathrm { m i n } }$ and maximum $\mu _ { \mathrm { m a x } }$ range.

$$
u _ { j } = - \sum _ { c = 1 } ^ { C } p ( y _ { j } ^ { c } | q _ { j } ) \cdot \log ( p ( y _ { j } ^ { c } | q _ { j } ) )\tag{2}
$$

where $u _ { j }$ measures the degree of uncertainty by quantifying the diversity of the answers.

## 3.2 Uncertainty-based Selection (Stage 2)

We define the LLM’s overall understanding of the task using the mean uncertainty $\mu$ and the standard deviation $\sigma$ of the uncertainty estimates from the unlabeled set $Q$ . A higher mean $\mu$ indicates a more challenging task for the LLM, while a higher standard deviation $\sigma$ reflects greater variability in question difficulty within $Q .$ These two parameters provide insight into the usefulness of a question for improving the LLM’s performance.

For instance, we hypothesize that when the mean uncertainty $\mu$ is low (indicating the LLM is performing well on the task), selecting questions with uncertainties lower than $\mu$ would not contribute valuable information. On the other hand, when the mean $\mu$ is high (suggesting the LLM struggles with the task), selecting questions with uncertainties significantly higher than $\mu$ may lead to less informative or erroneous rationales.

Based on these assumptions, we propose selecting a subset of questions $Q _ { s }$ that fall within a specific uncertainty range, as defined by the following condition:

$$
Q _ { s } \subset Q = \{ q _ { j } \mid u _ { \mathrm { m i n } } \leq u _ { j } < u _ { \mathrm { m a x } } \}\tag{3}
$$

Here, $u _ { \mathrm { m i n } }$ and $u _ { \mathrm { m a x } }$ represent the minimum and maximum uncertainty thresholds used to select questions. In the subsequent section, we will detail the specific ranges (cf. Table 1) based on $\mu$ and $\sigma$ for constructing demonstrations.

## 3.3 Demonstration Construction (Stage 3)

We adopt the demonstration construction methodology from Auto-CoT, which emphasizes diversity to mitigate the influence of incorrect rationales generated by the Zero-Shot-CoT method. The selected questions $Q _ { s }$ are first encoded into vector representations using Sentence Transformers (Reimers and Gurevych, 2019). These vectors are then clustered using k-Means++ (Arthur and Vassilvitskii, 2007), forming k distinct clusters. From each cluster, the question closest to the cluster centroid is selected. The associated rationale and answer, generated by the Zero-Shot-CoT method, are then combined to form the demonstration set D. During inference, a test question $q$ is appended to the constructed demonstration $D$ and passed to the LLM for final predictions.

## 4 Experimental Setup

Datasets: We evaluate our proposed method on four challenging reasoning datasets. GSM8K (Cobbe et al., 2021) comprises arithmetic reasoning problems. StrategyQA (Geva et al., 2021) is a question-answering benchmark requiring implicit multi-hop reasoning. Logical Fallacy (referred to as Fallacy) (Jin et al., 2022) involves reasoning about arguments and detecting formal and informal fallacies. Epistemic Reasoning (EPR) (Sileo and Lernould, 2023) is a natural language inference task that challenges LLMs to reason about human mental states. For a fair comparison, we split all datasets, except GSM8K, into two sets using stratified sampling: (i) an unlabeled set (70%) for demonstration creation, and (ii) a test set (30%) for zero-shot performance evaluation. GSM8K already contains train and test sets, so no further split was needed.

Implementation: We conduct experiments using five LLMs: GPT-4o (OpenAI, 2024), Mistral-7B-Instruct-v0.2 (Mistral) (Jiang et al., 2023), Phi-3-mini-4k-instruct (Phi3) (Abdin et al., 2024), text-davinci-002 (GPT3-XL), and text-davinci-003 (GPT3.5) (Brown et al., 2020). Note that this models are including both open-source (Phi3, Mistral) and proprietary models (GPT-4o, GPT3.5, GPT3- XL). To ensure consistency with prior work such as Auto-CoT, we use $k = 8$ demonstrations for all datasets, except for StrategyQA, where we use $k = 6 .$ . Additionally, during the evaluation of the LLMs, we set the temperature to 0 to ensure deterministic outputs, and report the average performance across three runs to maintain consistency in predictions.

![](images/c0884cc13db95277d13535e376ab2238051445f46a9b42fc7666181755d14629.jpg)  
Figure 2: Mean and standard deviation of uncertainty values as error graph -specific statistics across models.

![](images/78a4a59da29572b5cc10f11980a1ad4777173e8b32b931b06e30fc4cbbe6117e.jpg)  
Figure 3: Probability density function of uncertainty estimates of our method using GPT3.5 on GSM8K.

Uncertainty Estimation in ZEUS: Uncertainty in ZEUS is estimated using a combination of three perturbation methods: (1) non-zero temperature decoding, (2) trigger phrase variation, and (3) question rephrasing. We use five trigger phrases: " " (Empty), "Let’s think step by step." (SS), "Let’s think about this logically step by step." (LSS), "Before we dive into the answer," (BDA), and "Before answering the question, let’s understand the input." (BQU). For each question, we generate two rationale-answer pairs per trigger phrase at a temperature of 1, producing 10 rationale-answer pairs.

Each question is also rephrased using the instruction "Rephrase the below passage" with GPT4o.<sup>3</sup> We then generate five additional rationale-answer pairs using these rephrased questions with trigger phrases at a temperature of 0 to ensure precise responses. Thus, a total of 15 rationale-answer pairs are generated for each question to estimate uncertainty.

Selection Strategies in ZEUS: We define seven selection strategies based on the mean $\mu$ and standard deviation σ of uncertainty values across the unlabeled set, detailed in Table 1. These strategies include: Trivial, Very Easy, and Easy (selecting the lowest uncertainty demonstrations), Challenging, Hard, and Very Hard (focusing on high uncertainty values), and Moderate (selecting demonstrations from a range of uncertainty levels around µ).

Baselines: We compare ZEUS against five baseline methods: Zero-Shot, Few-Shot,<sup>4</sup> Zero-Shot-CoT (Kojima et al., 2022), Manual-CoT (Few-Shot-CoT) (Wei et al., 2022), and Auto-CoT.

## 5 Results & Qualitative Analysis

## 5.1 Uncertainty Distribution Analysis

In this subsection, we present an analysis of the mean (µ) and standard deviation (σ) of uncertainty estimates for different LLMs across various reasoning datasets. In Figure 3, we illustrate the distribution of uncertainty estimates for GPT-3.5 on the GSM8K dataset. We have provided the comprehensive plots of the distributions in the appendix (see Figures 7 –11). The mean µ and standard deviation σ of the uncertainty estimates using the unlabeled set Q has been shown through an error bar graph in Figure 2. Notably, LLMs such as GPT3-XL and Mistral show higher uncertainty in GSM8K, particularly with a larger deviation, whereas for tasks like StrategyQA and EPR, the uncertainty is generally more consistent across models, with GPT4o displaying the lowest variation. The trend highlights that model uncertainty is highly taskdependent, with complex reasoning tasks eliciting higher variability in predictions.

![](images/0dc8856ff76f479fd463180e7eb5e00f1970d65e3ef37f04773d12f19c989dc7.jpg)  
(a) GSM8K

![](images/9a6d24ce481c76640b2fece7c3147594d47354014b3268617863742b5a0d9074.jpg)  
(b) Fallacy

![](images/b057797ea5b769d6423f0272b42f9d86510319193a99980ddd22763e37b0b99a.jpg)  
(c) StrategyQA

![](images/90353413420018fcc21d54084a34eb53cb482c0008f74a651dd3240b8e830b99.jpg)  
(d) EPR

Figure 4: Normalized values of accuracy for various selection strategies using multiple LLMs.  
![](images/92c72cc102144c8d4136cf958c539a0082006fec062a19efb35de5bfa736fdbb.jpg)  
Figure 5: Sensitivity coefficient of confidence score wrt accuracy. Blue indicates ZEUS and Magenta for Temp-Perb. Solid for GPT3-XL and Dashed for GPT3.5. Coefficient using ZEUS is closest to ideal coefficient.

## 5.2 Sensitivity of Uncertainty Estimates

To assess the sensitivity of uncertainty estimates in distinguishing between helpful and redundant questions, we investigate the relationship between confidence scores and accuracy. This is done by fitting a linear regression (LR) model between the confidence score of the most common answer and its corresponding accuracy. In an ideally sensitive model, the slope coefficient of the LR would be one, indicating that a unit change in confidence directly corresponds to a unit change in accuracy. We compare our confidence scoring method against a temperature-based perturbation approach (Wan et al., 2023; Diao et al., 2023; Gao et al., 2024), referred to as Temp-Perb. This comparison is carried out using Zero-Shot-CoT prompting with 15 distinct temperature perturbations.

Figure 5 shows the slope coefficients for both ZEUS and Temp-Perb. Our results demonstrate that ZEUS consistently produces slope coefficients closer to the ideal sensitivity compared to Temp-Perb. Interestingly, Temp-Perb shows notably low sensitivity in the Logical Fallacy and EPR datasets, indicating a lack of reliability. In contrast, for GSM8K, Temp-Perb exhibits a coefficient exceeding 1, reflecting excessive sensitivity in this task.

## 5.3 Analysis of Selection Strategies

We present the normalized accuracy values for all selection strategies, including AutoCoT, in Figure 4. Our analysis reveals that AutoCoT was consistently outperformed by at least one other strategy across all LLMs and datasets. This indicates that leveraging uncertainty-based demonstration creation can more effectively identify valuable questions that enhance model performance. To provide a clearer perspective, Table 2 details the best and worst selection strategies for each model and dataset.

LLMs exhibit distinct performance patterns across varying levels of question difficulty, which allows us to categorize them into two broad groups: advanced models (GPT-4o, Phi3, GPT-3.5) and simpler models (Mistral, GPT-3 XL). This classification is based on observed performance trends rather than model size or architecture alone. Advanced models excel in handling Hard and Very Hard questions due to their superior reasoning capabilities, but they show limited gains when engaging with Trivial or Very Easy strategies, where their advanced abilities remain underutilized. On the other hand, simpler models perform better with Trivial and Easy strategies, as these align well with their baseline capabilities. However, they struggle considerably with Hard and Very Hard questions, where errors and uninformative outputs become more prevalent.

<table><tr><td></td><td colspan="2">GSM8K</td><td colspan="2">Fallacy</td><td colspan="2">StrategyQA</td><td colspan="2">EPR</td></tr><tr><td>Model</td><td>Best</td><td>Worst</td><td>Best</td><td>Worst</td><td>Best</td><td>Worst</td><td>Best</td><td>Worst</td></tr><tr><td>GPT40</td><td>Hard</td><td>Trivial</td><td>Moderate</td><td>Easy</td><td>Hard</td><td>Trivial</td><td>Very Hard</td><td>Trivial</td></tr><tr><td>Phi3</td><td>Challenging</td><td>Very Hard</td><td>Moderate</td><td>Trivial</td><td>Challenging</td><td>Trivial</td><td>Moderate</td><td>Trivial</td></tr><tr><td>Mistral</td><td>Easy</td><td>Hard</td><td>Challenging</td><td>Very Hard</td><td>Challenging</td><td>Very Hard</td><td>Easy</td><td>Moderate</td></tr><tr><td>GPT3.5</td><td>Challenging</td><td>Trivial</td><td>Very Hard</td><td>Trivial</td><td>Moderate</td><td>Trivial</td><td>Moderate</td><td>Very Hard</td></tr><tr><td>GPT3-XL</td><td>Trivial</td><td>Very Hard</td><td>Challenging</td><td>Trivial</td><td>Easy</td><td>Very Hard</td><td>Challenging</td><td>Very Hard</td></tr></table>

Table 2: Best and worst-performing strategies across tasks for each model, indicating that GPT4o requires harder strategies for optimal performance, while GPT3-XL shows improved results with easier strategies.

![](images/a3c6dec863675695713f6cc6220402ec53003ea644975169be8cc0f6b7e1d310.jpg)  
(a) GSM8K

![](images/ba4c4ca756d152bb6ee0c2c5705616f086fcad21cd64bd660cc3a69a94d997f4.jpg)  
(b) StrategyQA  
Figure 6: Accuracy vs Temp-Perb Uncertainty trend across all selection strategies for GPT4o.

To capture general trends, we analyzed performance across the best and worst strategies for each model. Our findings highlight that while Trivial and Very Easy strategies consistently yield the lowest performance for advanced models, simpler models face significant challenges with Hard and Very Hard strategies. Notably, our categorization focuses on overall performance trends rather than model size, which places models like GPT-4o and Phi3 in the same group.

Among the selection strategies, Trivial and Very Hard tend to yield poorer performance across most models. This suggests that extremes in task difficulty—whether too easy or too hard—are generally detrimental to model accuracy. The Hard strategy generally improves performance for GPT-4o, whereas the Challenging strategy appears to be optimal for Phi3, Mistral, and GPT-3.5. These findings align with the overall performance trends observed for these models.

However, performance variations still exist across tasks and models. For instance, the Mistral model’s performance declines with Moderate and harder strategies on the EPR task, while it improves with higher uncertainty estimates on other tasks. This indicates that selecting the optimal strategy can be complex and task-dependent. To address this, the next subsection will explore methods for determining the most effective selection strategy.

## 5.4 Choosing Optimal Selection Strategy

Upon constructing the demonstration for each strategy, we need to identify the optimal strategy for a given task and model. We calculate the average uncertainty on the unlabelled set Q while keeping the demonstration unchanged. The optimal strategy is the one with the lowest entropy, as this tends to strongly correlate with higher accuracy. Temp-Perb provides well-calibrated uncertainty estimates, although it lacks the sensitivity required to effectively differentiate between similar questions. Despite this limitation, its well calibration makes Temp-Perb suitable for selecting the bestperforming strategy based on uncertainty estimates.

<table><tr><td></td><td colspan="5">GSM8K</td><td colspan="5">Fallacy</td></tr><tr><td>Method</td><td>GPT40</td><td>Phi3</td><td>Mistral</td><td>GPT3.5</td><td>GPT3-XL</td><td>GPT40</td><td>Phi3</td><td>Mistral</td><td>GPT3.5</td><td>GPT3-XL</td></tr><tr><td>Zero-Shot</td><td>49.4</td><td>50.7</td><td>45.3</td><td>12.6</td><td>10.7</td><td>80.5</td><td>81.8</td><td>71.1</td><td>63.9</td><td>48.3</td></tr><tr><td>Few-Shot</td><td>84.0</td><td>50.7</td><td>45.3</td><td>16.5</td><td>14.4</td><td>92.5</td><td>90.4</td><td>62.9</td><td>76.9</td><td>79.8</td></tr><tr><td>Zero-Shot-CoT</td><td>94.8</td><td>85.9</td><td>51.8</td><td>60.4</td><td>44.7</td><td>84.8</td><td>87.5</td><td>67.1</td><td>67.7</td><td>61.7</td></tr><tr><td>Manual-CoT</td><td>89.3</td><td>81.9</td><td>42.4</td><td>56.4</td><td>43.9</td><td>90.1</td><td>90.1</td><td>64.3</td><td></td><td></td></tr><tr><td>Auto-CoT</td><td>94.2</td><td>87.6</td><td>47.2</td><td>58.5</td><td>44.6</td><td>97.0</td><td>85.6</td><td>74.4</td><td>76.9</td><td>66.7</td></tr><tr><td>ZEUS (LU)</td><td>95.8</td><td>89.9</td><td>57.3</td><td>62.9</td><td>51.9</td><td>98.0</td><td>94.0</td><td>78.5</td><td>79.4</td><td>76.4</td></tr><tr><td>ZEUS (HA)</td><td>95.8</td><td>89.9</td><td>57.6</td><td>62.9</td><td>51.9</td><td>98.0</td><td>94.0</td><td>78.5</td><td>79.4</td><td>76.4</td></tr><tr><td>StrategyQA</td><td colspan="5"></td><td colspan="5">EPR</td></tr><tr><td></td><td>GPT40</td><td>Phi3</td><td>Mistral</td><td>GPT3.5</td><td>GPT3-XL</td><td>GPT40</td><td>Phi3</td><td>Mistral</td><td>GPT3.5</td><td>GPT3-XL</td></tr><tr><td>Zero-Shot</td><td>65.2</td><td>56.6</td><td>59.8</td><td>54.4</td><td>16.6</td><td>61.2</td><td>72.2</td><td>45.2</td><td>60.0</td><td>61.5</td></tr><tr><td>Few-Shot</td><td>77.6</td><td>65.8</td><td>61.1</td><td>66.2</td><td>64.8</td><td>83.0</td><td>64.0</td><td>55.2</td><td>75.6</td><td>58.2</td></tr><tr><td>Zero-Shot-CoT</td><td>70.7</td><td>67.5</td><td>59.0</td><td>57.4</td><td>51.2</td><td>64.7</td><td>79.8</td><td>65.7</td><td>60.2</td><td>59.3</td></tr><tr><td>Manual-CoT</td><td>81.1</td><td>68.9</td><td>63.8</td><td>68.6</td><td>57.6</td><td>84.2</td><td>64.0</td><td>57.7</td><td></td><td></td></tr><tr><td>Auto-CoT</td><td>80.1</td><td>64.5</td><td>57.9</td><td>64.9</td><td>64.1</td><td>68.2</td><td>75.3</td><td>52.5</td><td>52.5</td><td>59.5</td></tr><tr><td>ZEUS (LU)</td><td>81.1</td><td>67.7</td><td>59.8</td><td>66.8</td><td>66.5</td><td>72.8</td><td>76.2</td><td>68.5</td><td>65.3</td><td>66.2</td></tr><tr><td>ZEUS (HA)</td><td>82.2</td><td>67.7</td><td>59.8</td><td>66.8</td><td>66.5</td><td>72.8</td><td>77.0</td><td>68.5</td><td>65.3</td><td>66.2</td></tr></table>

Table 3: Accuracy on various datasets. ZEUS (HA) chooses the best performing strategy for each dataset while ZEUS (LU) chooses the strategies having lowest Temp-Perb uncertainty estimates.

Therefore, we use Temp-Perb for uncertainty estimation to determine the optimal selection strategy for a given model and task.

In Figure 6, we illustrate the accuracy of various selection strategies for GPT-4o in relation to Temp-Perb based uncertainty estimates. The data indicates that the accuracy is inversely correlated with uncertainty across all four datasets. This inverse relationship allows us to identify the optimal selection strategy as the one associated with the lowest uncertainty. We have included similar analyses for other models in the appendix (cf. Figures 12 – 16).

## 5.5 Comparison with Baselines

The selection strategy with the lowest uncertainty is denoted as ZEUS (LU), while the strategy with the highest accuracy is represented by ZEUS (HA). Table 3 demonstrates that ZEUS (LU) and ZEUS (HA) yield nearly identical performance, underscoring the robustness of the Temp-Perb uncertainty estimates. In general, the optimally selected ZEUS(LU) either outperforms all baseline methods or comes in a close second to in a few cases across three datasets (GSM8K, Fallacy, and Strategy QA), with only a few exceptions. ZEUS methods consistently outperform all baseline strategies on the GSM8K and Fallacy datasets, with the exception of GPT-3 XL on the Fallacy dataset. For the StrategyQA dataset, Manual-CoT achieves the highest accuracy for most models, highlighting the effectiveness of human-crafted demonstrations. On the EPR dataset, ZEUS surpasses Zero-Shot, Zero-Shot-CoT, and Auto-CoT methods across most models. Overall, ZEUS methods either match or exceed the accuracy of these baseline strategies without requiring manual annotations.

## 6 Conclusion

This paper introduces the zero-shot uncertaintybased ZEUS method for evaluating and selecting optimal strategies based on uncertainty estimates. Our analysis reveals that ZEUS provides highly sensitive and reliable uncertainty estimates, outperforming temperature-based perturbation approaches (Temp-Perb) in distinguishing between helpful and redundant questions.

Our findings classify models into two groups based on their optimal strategies. Advanced models like GPT-4o, Phi3, and GPT3.5 perform best with Hard and Challenging example selection strategies, effectively leveraging their greater capabilities to tackle complex queries. In contrast, simpler models such as Mistral and GPT3-XL benefit more from Trivial and Easy strategies, where even low-uncertainty questions yield valuable information. By selecting the strategy with the lowest uncertainty estimates, ZEUS(LU) (recommended) achieves performance comparable to the best-performing strategies ZEUS(HA), without requiring manual annotations. Overall, ZEUS consistently matches or surpasses baseline accuracy, demonstrating its robustness and sensitivity in improving model performance.

## 7 Limitation

While our work demonstrates the effectiveness of the ZEUS method, there are several limitations and avenues for future research. First, the selection strategies in our current approach require exhaustive exploration to find the optimal strategy, which can be time-consuming and computationally expensive. This process could be automated by incorporating a greedy search algorithm based on uncertainty estimates, allowing for more efficient strategy selection. Another limitation is our reliance on uncertainty estimates from unlabeled questions, without examining the impact of dataset attributes like diversity or size. These factors could affect the estimates and lead to suboptimal strategy selection. Future work should explore these effects to improve robustness.

## References

Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. 2024. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219.

David Arthur and Sergei Vassilvitskii. 2007. Kmeans++ the advantages of careful seeding. In Proceedings ofthe eighteenth annual ACM-SIAM symposium on Discrete algorithms, pages 1027–1035.

Markus Bayer and Christian Reuter. 2024. Activellm: Large language model-based active learning for textual few-shot scenarios. arXiv preprint arXiv:2405.10808.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro

Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Shizhe Diao, Pengcheng Wang, Yong Lin, Rui Pan, Xiang Liu, and Tong Zhang. 2024. Active prompting with chain-of-thought for large language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1330–1350, Bangkok, Thailand. Association for Computational Linguistics.

Shizhe Diao, Pengcheng Wang, Yong Lin, and Tong Zhang. 2023. Active prompting with chain-ofthought for large language models. arXiv preprint arXiv:2302.12246.

Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. 2024. Towards revealing the mystery behind chain of thought: a theoretical perspective. Advances in Neural Information Processing Systems, 36.

Yifan Fu, Xingquan Zhu, and Bin Li. 2013. A survey on instance selection for active learning. Knowledge and information systems, 35(2):249–283.

Yarin Gal and Zoubin Ghahramani. 2016. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In Tnternational Conference on Machine Learning, pages 1050–1059.

Xiang Gao, Jiaxin Zhang, Lalla Mouatadid, and Kamalika Das. 2024. Spuq: Perturbation-based uncertainty quantification for large language models. In Proceedings ofthe 18th Conference ofthe European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2336–2346.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346– 361.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321– 1330. PMLR.

Dan Hendrycks and Kevin Gimpel. 2016. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In International Conference on Learning Representations.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Zhijing Jin, Abhinav Lalwani, Tejas Vaidhya, Xiaoyu Shen, Yiwen Ding, Zhiheng Lyu, Mrinmaya Sachan, Rada Mihalcea, and Bernhard Schoelkopf. 2022.

Logical fallacy detection. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, pages 7180–7198.

Philipp Koehn. 2009. Statistical machine translation. Cambridge University Press.

Takeshi Kojima, Shixiang (Shane) Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In Advances in Neural Information Processing Systems, volume 35, pages 22199–22213. Curran Associates, Inc.

Aobo Kong, Shiwan Zhao, Hao Chen, Qicheng Li, Yong Qin, Ruiqi Sun, Xin Zhou, Enzhi Wang, and Xiaohang Dong. 2024. Better zero-shot reasoning with role-play prompting. In Proceedings of the 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4099–4113, Mexico City, Mexico. Association for Computational Linguistics.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. arXiv preprint arXiv:2302.09664.

Shanu Kumar, Sandipan Dandapat, and Monojit Choudhury. 2022. ” diversity and uncertainty in moderation” are the key to data selection for multilingual few-shot transfer. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 1042–1055.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. 2017. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. 2022. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Zhaopeng Tu, and Shuming Shi. 2023. Encouraging divergent thinking in large language models through multi-agent debate. arXiv preprint arXiv:2305.19118.

OpenAI. 2024. Introducing gpt-4o. https://openai. com/index/hello-gpt-4o/. Accessed: 2024-09- 16.

Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. 2019. Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift. Advances in neural information processing systems, 32.

Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. 2021. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of nlp models with checklist. arXiv preprint arXiv:2005.04118.

Guy Rotman and Roi Reichart. 2022. Multi-task active learning for pre-trained transformer-based models. Transactions of the Association for Computational Linguistics, 10:1209–1228.

Burr Settles and Mark Craven. 2008. An analysis of active learning strategies for sequence labeling tasks. In Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing, pages 1070–1079.

Kashun Shum, Shizhe Diao, and Tong Zhang. 2023. Automatic prompt augmentation and selection with chain-of-thought from labeled data. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 12113–12139, Singapore. Association for Computational Linguistics.

Damien Sileo and Antoine Lernould. 2023. Mindgames: Targeting theory of mind in large language models with dynamic epistemic modal logic. arXiv preprint arXiv:2305.03353.

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. 2022. Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239.

Christian Tomani, Kamalika Chaudhuri, Ivan Evtimov, Daniel Cremers, and Mark Ibrahim. 2024. Uncertainty-based abstention in llms improves safety and reduces hallucinations. arXiv preprint arXiv:2404.10960.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Joost Van Amersfoort, Lewis Smith, Yee Whye Teh, and Yarin Gal. 2020. Uncertainty estimation using a

single deep deterministic neural network. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 9690–9700. PMLR.

Roman Vashurin, Ekaterina Fadeeva, Artem Vazhentsev, Akim Tsvigun, Daniil Vasilev, Rui Xing, Abdelrahman Boda Sadallah, Lyudmila Rvanova, Sergey Petrakov, Alexander Panchenko, et al. 2024. Benchmarking uncertainty quantification methods for large language models with lm-polygraph. arXiv preprint arXiv:2406.15627.

Xingchen Wan, Ruoxi Sun, Hanjun Dai, Sercan Arik, and Tomas Pfister. 2023. Better zero-shot reasoning with self-adaptive prompting. In Findings of the Association for Computational Linguistics: ACL 2023, pages 3493–3514, Toronto, Canada. Association for Computational Linguistics.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.

Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, Tao Ge, Furu Wei, and Heng Ji. 2023. Unleashing cognitive synergy in large language models: A task-solving agent through multi-persona selfcollaboration. arXiv preprint arXiv:2307.05300, 1(2):3.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601.

Zhangyue Yin, Qiushi Sun, Cheng Chang, Qipeng Guo, Junqi Dai, Xuanjing Huang, and Xipeng Qiu. 2023. Exchange-of-thought: Enhancing large language model capabilities through cross-model communication. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 15135–15153, Singapore. Association for Computational Linguistics.

Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2022. Automatic chain of thought prompting in large language models. In The Eleventh International Conference on Learning Representations.

Xinyu Zhu, Junjie Wang, Lin Zhang, Yuxiang Zhang, Yongfeng Huang, Ruyi Gan, Jiaxing Zhang, and Yujiu Yang. 2023. Solving math word problems via cooperative reasoning induced language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4471–4485, Toronto, Canada. Association for Computational Linguistics.

A Appendix

![](images/c8717103d7d8629137447448998daaa7ce2b2e25227f4223a6f6e93fbb63db66.jpg)  
(a) GSM8K

![](images/fbca0a924e1de72440a5e30d26135f0eb5b476e0d0774d046d27bf0cb4eb86e8.jpg)  
(b) Fallacy

![](images/f891cc254711898f244adbd8c77865c1ad30a613831ff87e2c998210e948562d.jpg)  
(c) EPR

![](images/889f892d9da8993d7bbc0e1f88e2a6d8b682f186dd476e7fd9e5cab7119a51ef.jpg)  
(d) StrategyQA  
Figure 7: Probability density function of uncertainty estimates of our method using GPT4o.

![](images/8f7856feeaaadd819655a08832f09e33e1b036ffa4474c652799e297b78a0359.jpg)  
(a) GSM8K

![](images/a04d6d34b082d08e800acc779a2750616fb28619ba1da949ce82617d85abc42a.jpg)  
(b) Fallacy

![](images/597811567de8ad992ad8cce43a0c36808c89d1f0dc78ef9c09f3b4899d543ebc.jpg)  
(c) EPR

![](images/24d60e4f68cfdc716451168df648aec10c659352d65835632fc4a24c4c72fdfb.jpg)  
(d) StrategyQA  
Figure 8: Probability density function of uncertainty estimates of our method using Phi3.

![](images/d36559e77d1dcf7bc6110a3d165e696ef17ae6a39130b9763281f29725c86161.jpg)  
(a) GSM8K

![](images/3c87d1753a45709a3a87db7bc02b37fd8a30eb02bf1772355f18b51a055c47ea.jpg)  
(b) Fallacy

![](images/cb7a1a8310a09f79f8c14b773bb70c653b0faf475266dfe3da1fc3beaf13baf4.jpg)  
(c) EPR

![](images/a5612568863b8a92c11afd6f3078defb6b8da9db3d7ad5ea3fc695b3fa8b0f26.jpg)  
(d) StrategyQA  
Figure 9: Probability density function of uncertainty estimates of our method using Mistral.

![](images/f73d2c61db668d5027cf7016d51d65fb0b3b1a3cad3093695fe7070b0ba107c9.jpg)  
(a) GSM8K

![](images/a3b1c8d8885497feb5f304d1a238142f5a8f796f5fc9c3efe3b217866e92d864.jpg)  
(b) Fallacy

![](images/5156b494672da6767de6fd2bdfefb88b21485d6235a9b642f1e3c338628a2899.jpg)  
(c) EPR

![](images/3df8eddcd59b3d16e76f85b72f2dfa94a34b40d2c925ea2ed0152e3770cd7c7d.jpg)  
(d) StrategyQA  
Figure 10: Probability density function of uncertainty estimates of our method using GPT3.5.

![](images/ded07ec9f839cbfae5542811eca5b613114831da3b2af7a84ca661c00f27adcd.jpg)  
(a) GSM8K

![](images/a4c63e396ebd12d339905de6fe303aed2dcd15cc70e663874827100e5ff79d7b.jpg)  
(b) Fallacy

![](images/c7e1aabf00e1b3c510fb872f7122c327212df6bed813242dfdecc18971449e45.jpg)  
(c) EPR

![](images/59404a43a69e648a50f8239c6cfcea313f58b3f8bd8e4bba4fe9302701633484.jpg)  
(d) StrategyQA  
Figure 11: Probability density function of uncertainty estimates of our method using GPT3-XL.

![](images/09be331c28f73a1736f447325d5bc0babaa2866e81c5a9eb0f23aa43de44cafd.jpg)  
(a) GSM8K

![](images/57c0a477ea1f863e458be341decd20ad900d66eb7c6d42a31b5332f6b422d5bf.jpg)  
(b) Fallacy

![](images/ef8fc9e9cf244ef7c5556d33b01226c8e1cd23104d199bc7eb585edf23cfb517.jpg)  
(c) StrategyQA

![](images/39bbf888e236402bb10a2aa756e887c7c8ebc6cdd23ef4f64c05fb0782c16dfd.jpg)  
(d) EPR  
Figure 12: Accuracy vs Temp-Perb Uncertainty trend across all selection strategies forGPT4o

![](images/b6b5722edf962ddc1c69be8e1744d2bbc3b355e221d582db8b24d72e5c44b945.jpg)  
(a) GSM8K

![](images/88705783532d754613968f7200dcbb7314cdc56d44dee6715b3b54c342bde7ec.jpg)  
(b) Fallacy

![](images/74ddf8ca4cb95e556861713bbba3239ce1e68153a1f47b49ab22a3dd54ce29af.jpg)  
(c) StrategyQA

![](images/5cc687f0953f82227e4ab55c68aa41e913f80e721e6c4bae06990d093e2d7ef0.jpg)  
(d) EPR  
Figure 13: Accuracy vs Temp-Perb Uncertainty trend across all selection strategies forPhi3

![](images/d6c9405f5a60b6b9121ff08f541f3fd06726f97934030fc75f09da3424427aeb.jpg)  
(a) GSM8K

![](images/b45a769812eeb9dd65b7d0af455a42587ba1f116986361b6e99b3cde8dedaf20.jpg)  
(b) Fallacy

![](images/e644d180b987ea6daa7d4e581f7be78eb60dcdaf46b5e846cdf6c5305db4f3cb.jpg)  
(c) StrategyQA

![](images/385fc7452506b368a90eeb01eabadaf854ed59da6463324219d852d4e86703ee.jpg)  
(d) EPR  
Figure 14: Accuracy vs Temp-Perb Uncertainty trend across all selection strategies forMistral

![](images/14c3c8473cddc2e40545c910623ad0aa634bd3bcbea2f933ea6ae8e210452574.jpg)  
(a) GSM8K

![](images/95ef6364d622a1274052fdfcd5960a29cc4a4f030577c88032ca6ec61a148296.jpg)  
(b) Fallacy

![](images/c3eaa936fcbd57a833426578c13422458177a76dd87a86597e65ffbf48b4da73.jpg)  
(c) StrategyQA

![](images/0f20445932e1e4bd8dcfefa4f464379fed51525c2d6fc591583d2b8ec5e5ff7a.jpg)  
(d) EPR  
Figure 15: Accuracy vs Temp-Perb Uncertainty trend across all selection strategies forGPT3.5

![](images/eb3e66f78d09322fd1a77a5cac4d1e60b0444f11d034dcc9398742b78e351d3c.jpg)  
(a) GSM8K

![](images/a84a2b5d3c9c4f0f22f761f6c6e4f9db647ee1fdf17105002f7553f0597f367e.jpg)  
(b) Fallacy

![](images/922d9bcc7d490f8ff7efd38fdedf23cf796c30608c2c4378802286f7816eb40e.jpg)  
(c) StrategyQA

![](images/2f7d401eea382e45cb1ca44525062392095185ba6c42a08f1d8303cfea3422f4.jpg)  
(d) EPR  
Figure 16: Accuracy vs Temp-Perb Uncertainty trend across all selection strategies forGPT3-XL

<table><tr><td rowspan="3">Dataset</td><td colspan="2">GPT3.5</td><td colspan="2">GPT3-XL</td><td colspan="2">GPT40</td><td colspan="2">Phi3</td><td colspan="2">Mistral</td></tr><tr><td>μ</td><td>σ</td><td>μ</td><td>σ</td><td>μ</td><td>σ</td><td>μ</td><td>σ</td><td>μ</td><td>σ</td></tr><tr><td>GSM8K</td><td>1.21</td><td>0.53</td><td>1.55</td><td>0.48</td><td>0.30</td><td>0.35</td><td>0.45</td><td>0.48</td><td>1.28</td><td>0.76</td></tr><tr><td>Fallacy</td><td>0.49</td><td>0.36</td><td>0.57</td><td>0.37</td><td>0.26</td><td>0.26</td><td>0.37</td><td>0.23</td><td>0.41</td><td>0.25</td></tr><tr><td>EPR</td><td>0.55</td><td>0.18</td><td>0.22</td><td>0.21</td><td>0.39</td><td>0.27</td><td>0.46</td><td>0.22</td><td>0.42</td><td>0.25</td></tr><tr><td>StrategyQA</td><td>0.43</td><td>0.35</td><td>0.83</td><td>0.22</td><td>0.32</td><td>0.31</td><td>0.39</td><td>0.29</td><td>0.28</td><td>0.30</td></tr></table>

Table 4: Mean and standard deviation of uncertainty values as error graph -specific statistics across models.