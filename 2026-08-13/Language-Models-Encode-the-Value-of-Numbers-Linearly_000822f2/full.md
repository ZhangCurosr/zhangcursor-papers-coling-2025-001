# Language Models Encode the Value of Numbers Linearly

Fangwei Zhu, Damai Dai, Zhifang Sui

School of Computer Science,

State Key Laboratory of Multimedia Information Processing, Peking University zhufangwei2022@stu.pku.edu.cn {daidamai, szf}@pku.edu.cn

## Abstract

Large language models (LLMs) have exhibited impressive competence in various tasks, but their internal mechanisms on mathematical problems are still under-explored. In this paper, we study a fundamental question: how language models encode the value of numbers, a basic element in math. To study the question, we construct a synthetic dataset comprising addition problems and utilize linear probes to read out input numbers from the hidden states. Experimental results support the existence of encoded number values in LLMs on different lay ers, and these values can be extracted via linear probes. Further experiments show that LLMs store their calculation results in a similar manner, and we can intervene the output via simple vector additions, proving the causal connection between encoded numbers and language model outputs. Our research provides evidence that LLMs encode the value of numbers linearly, offering insights for better exploring, designing, and utilizing numeric information in LLMs. The code and data are available at https: //github.com/solitaryzero/NumProbe.

## 1 Introduction

Large language models (LLMs) have demonstrated excellent ability in various scenarios like question answering (Zhao et al., 2023; Li et al., 2023b), instruction following (Brown et al., 2020; Ouyang et al., 2022; Taori et al., 2023), and code generation (Chen et al., 2021; Nijkamp et al., 2022; Li et al., 2023a). Solving mathematical problems is generally viewed to be more difficult (Yu et al., 2023), and language models even struggle to solve simple arithmetic problems (Dziri et al., 2024).

Numbers are fundamental elements in math. In order to accurately answer mathematical problems, LLMs should be able to precisely encode value of numbers in the input text. Currently, the way how LLMs process numbers is still not fully explored. While previous studies (Stolfo et al., 2023; Hanna et al., 2024) have explored the inner mechanisms of language models on mathematical problems, they focus on small numbers or modular arithmetic (Engels et al., 2024; Zhong et al., 2024), and how LLMs utilize numbers in a larger, unconstrained range remains largely unknown.

![](images/51332f36c259a06ae221b6a7a2628ec527755bf1f1e633048682e08e8d04c774.jpg)  
Figure 1: Encoded number values in the hidden state of language models. We find that both the value of input numbers (blue and green) and calculation results (red) can be read out from the hidden state of language models via linear probes.

In this paper, we explore the question whether and how LLMs encode the value of numbers through extracting numerical information from their internal representations. To be specific, we construct a synthetic dataset comprising simple addition questions, and train linear probes (Nanda et al., 2023; Gurnee and Tegmark, 2023) on the hidden states of LLMs to predict the number values provided in the input text. Experimental results on the dataset demonstrate that the value of input numbers can be probed from the hidden states of language models from early layers, as illustrated in Figure 1. Both input values and calculation results can be read out, and encoded values can be found at different token positions. These results support that language models do encode numerical information, and possibly in a linear manner.

To further verify the fact that the encoded number values are utilized by language models, we study the causal connection between numeric information and model outputs. To be specific, we discover that we can influence the calculation result of language models by performing interventions like activation patching or adding linear vectors.

The above discoveries may reveal future directions for utilizing the encoded numerical information, for example, specialized encoding systems and error mitigation modules.

To sum up, our contributions can be listed as: (1) We study the question of whether language models are able to encode the value of numbers in the input text and construct a synthetic dataset to analyze the language models. (2) We discover that language models encode the value of numbers linearly by utilizing linear probes to probe encoded number values in the hidden states of language models. (3) We further prove that language models utilize the encoded numerical information by revealing the causal connection between encoded number values and the final output of language models.

## 2 Probing Numbers in Language Models

## 2.1 The Goal of Probing

Given that there is a number x in the input text t, we assume that a language model $L M$ can encode the number in its hidden state $\mathbf { h } _ { i } \in \mathbb { R } ^ { d _ { m o d e l } }$ of a specific layer i, where $d _ { m o d e l }$ is the hidden dimension. We denote the mapping as:

$$
\mathbf { h } _ { i } = f _ { i } ( x , t - x )\tag{1}
$$

where $f _ { i }$ refers to the encoding process on layer i, and $t - x$ refers to the tokens in t excluding x.

If the mapping function f is a bijective function, there will exist an inverse function $f _ { i } ^ { - 1 }$ that reconstructs the original number $x$ from the hidden state $\mathbf { h } _ { i }$ . For each layer i, we aim to find a optimal predictor $\mathcal { P } _ { i } ^ { * }$ that imitates $f _ { i } ^ { - 1 }$ , whose prediction best fits the original number x:

$$
\mathcal { P } _ { i } ^ { * } = \underset { \mathcal { P } _ { i } } { \arg \operatorname* { m i n } } \left| x - \mathcal { P } _ { i } ( \mathbf { h } _ { i } ) \right|\tag{2}
$$

Considering the numerical stability, we probe the logarithmic value $\log _ { 2 } ( x )$ instead of the original number x in all our experiments.

We can assess the existence of encoded number values by observing how much the probing result $\mathcal { P } _ { i } ^ { * } ( ( \mathbf { h } _ { i } ) )$ resembles the original number x.

## 2.2 Dataset Construction

To investigate whether LLMs encode numbers, we construct a synthetic dataset containing different magnitudes of numbers. The dataset contains numbers ranging from 2 digits to 10 digits, with each digit corresponding to 1000 entries<sup>1</sup>. We split the dataset into training, validation, and test sets at a ratio of 80%/10%/10%.

To observe how LLMs encode and utilize numbers, we adopt addition problems as our prompt<sup>2</sup>. Let a and b be two randomly generated numbers, each question is formulated as:

Question: What is the sum of {a} and {b}? Answer: {a + b}

## 2.3 Probing Method

Obtaining Hidden States. We choose the LLaMA-2 model family (Touvron et al., 2023b) and Mistral-7B (Jiang et al., 2023) as base models to be investigated. We feed the question text in Section 2.2 into the models, and save the hidden states of all layers. For each layer, we obtain a set of hidden states (i.e. the residual stream) $\mathbf { H } \in \mathbb { R } ^ { n \times d _ { m o d e l } }$ at every token position, where n is the number of samples in the dataset.

Training Probes. Following previous work, we adopt the widely acknowledged linear probing technique to reconstruct numbers from the hidden states. To be specific, for each layer, given a set of hidden states H and their corresponding original numbers $\mathbf { X } = \{ x \}$ , we train a linear regressor $\mathcal { P }$ that yields best predictions $\mathbf { P } = \mathbf { H } \mathbf { W } + b ,$ where $\mathbf { W } \in \mathbb { R } ^ { d _ { m o d e l } }$ and b are the weights of .

In practice, directly performing linear regression could give erroneous results, as the value of numbers varies over a wide range. We do a logarithmic operation on input numbers X with a base of 2 to guarantee the numerical stability of probes.

We utilize Ridge regression, which adds L2 regularization to the vanilla linear regression model, to construct the probes:

$$
\mathbf { W } ^ { * } , b ^ { * } = \mathop { \arg \operatorname* { m i n } } _ { \mathbf { W } , b } | | \log _ { 2 } ( \mathbf { X } ) - \mathbf { H } \mathbf { W } - b | | _ { 2 } ^ { 2 } + \lambda | | \mathbf { W } | | _ { 2 } ^ { 2 }\tag{3}
$$

where $\mathbf { W } ^ { * } , b ^ { * }$ are the weights of regressors, and λ is a hyperparameter that controls regularization strength. In this way, we can predict logarithmic results $\mathbf { P } ^ { * } = \mathbf { H } \mathbf { W } ^ { * } + b ^ { * }$ based on the hidden states.

![](images/f52eccf4e939bd191e6f8f64a0b701f23b762d1afeb031e805ce34fcc34f8555.jpg)  
(a) ρ of probes on a.

![](images/9281e7da88d1cf3d29c72a83d03bf3467adf518634422a75fe8837a64f27d59e.jpg)  
(b) ρ of probes on $b .$

![](images/052cc8bdc35e67ad36e17c9db8c7c97ecb4c526dd3e7be73bac36853edcae5fe.jpg)  
(c) ρ of probes on o.

![](images/6524a07e93c234cbad3f17509b06b148d354e8d8076cf4291ebd60d91a95fd7b.jpg)  
(d) $R ^ { 2 }$ of probes on a.

![](images/a7a8f6dbd8cff8812416052f3f09371e85c145760e9b8abd109533ef0d2a708d.jpg)  
(e) $R ^ { 2 }$ of probes on $b .$

![](images/3a40b1280a6ee54a3c56b65a5b2cf5947c7d8aa6ac1b99219970a0b27ea5a68f.jpg)  
(f) $R ^ { 2 }$ of probes on o.  
Figure 2: Pearson coefficient $( \rho )$ and out-of-sample $R ^ { 2 }$ of probes on different layers. a and b refer to the two input numbers denoted in Section 2.2, and o refers to the prediction of language models respectively. High $\rho$ and $R ^ { 2 }$ indicate the existence of encoded number values in the hidden states.

## 2.4 Evaluation Metrics

We use two standard regression metrics on the probing task to evaluate the probes: $R ^ { 2 }$ which determines the proportion of variance in the dependent variable that can be explained by the independent variable, and the Pearson coefficient $\rho$ which measures the linear correlation between two variables.

As mathematical problems require a precise understanding of numbers, we introduce two additional metrics to examine how well can a language model encode numbers:

Approximate accuracy (AAcc) evaluates whether the predicted number is approximately the same as the original number, namely with an error margin of $< 1 \%$ . Higher AAcc indicates that the number encoding is more likely to be precise.

Mean square error (MSE) is the average squared difference between probe predictions and actual values. Smaller MSE means lower loss during the encoding process.

$$
\operatorname { A A c c } ( \mathbf { P } ^ { * } , \mathbf { X } ) = { \frac { | ( 2 ^ { \mathbf { P } ^ { * } } - \mathbf { X } ) < 0 . 0 1 X | } { | \mathbf { X } | } }\tag{4}
$$

$$
\operatorname { M S E } ( \mathbf { P } ^ { * } , \mathbf { X } ) = \arg ( ( \mathbf { P } ^ { * } - \log _ { 2 } \mathbf { X } ) ^ { 2 } )\tag{5}
$$

## 2.5 Experimental Setup

We use the original LLaMA-2-7B, LLaMA-2-13B, and Mistral-7B models without fine-tuning for all experiments. The outputs are obtained by performing greedy search with a max new token restriction of 30 during decoding. The regularization strength is set to λ <sub>=</sub> 0.1 for all probes.<sup>3</sup>.

In main experiments, we probe 3 distinct values at different positions: the first number a at the last digit of $a$ (for example, 3 for 123), the second number b at the last digit of b, and the prediction of language models o at the last token of the entire input text. We report the accuracy of o, i.e. the ratio of $o = a + b ,$ in Appendix D.

## 3 Do LLMs Encode Number Values?

## 3.1 The Existence of Encoded Number Values

LLMs do encode number values. We first inspect the overall Pearson coefficient $( \rho )$ and out-ofsample $R ^ { 2 }$ on all layers. High $\rho$ and $R ^ { 2 }$ indicate that LLMs are likely to be able to encode number values in their hidden states. As illustrated in Figure 2, the probes achieve surprisingly high $\rho$ and $R ^ { 2 }$ on all layers, proving that the hidden states of LLMs contain the encoded value of input numbers, and the encoding process starts from even the first layer. Meanwhile, notice that both $\rho$ and $R ^ { 2 }$ slightly drop on late layers, which may indicate that intermediate layers better encode number values.

![](images/bd3ef5deaa28a5bb32b1dc30ac21ccdeac7c6ffe12205a00742499e157a0abed.jpg)  
(a) AAcc of probes on a.

![](images/f78eb9227edc0de0762cd33c7492a6ff57e71a6481c26d0918af845f5e854e18.jpg)  
(b) AAcc of probes on b.

![](images/a58b5eba7b9d23db0b889c44ef5c577be625f9f76b84e7dcee357e6f688ec322.jpg)  
(c) AAcc of probes on o.

![](images/211dd557ece09d7ae2043e2ecefb05b97e1e06ad581c7613d4ecd022cd57612a.jpg)  
(d) MSE of probes on a.

![](images/33a11711635d51093857fa458e42727b1fe14f3007854f8d52d9d3501a55bf4b.jpg)  
(e) MSE of probes on b.

![](images/5896a4d5974d71d064cfb2bd03601a3596dee71e0d7410542972b411b975bda8.jpg)  
(f) MSE of probes on o.  
Figure 3: Approximate accuracy (AAcc) and mean square error (MSE) of probes on different layers. a and b refer to the two input numbers denoted in Section 2.2, and o refers to the prediction of language models respectively. High AAcc and low MSE indicate precise number encoding.

Linear probes cannot reconstruct the precise value. Aside from the existence of encoded number values, we are also interested in their precision, which is depicted by AAcc and MSE in Figure 3.

In contrast to high correlation coefficients, the AAcc is below 50% on all layers, which means that the linear probes have difficulty in precisely reconstructing the input numbers. The trends in AAcc and MSE are consistent with ρ and $R ^ { 2 }$ , indicating that LLaMA-2 models achieve the most precise number encoding in intermediate layers, but the encoding faces more error in deeper layers.

This phenomenon may indicate that language models use stronger non-linear encoding systems, which we will further explore in Section 3.4; Or it may be a hint that the number encoding in language models is not precise<sup>4</sup>.

## 3.2 Number Encoding Patterns are Different across Layers

To better analyze how language models encode numbers, we pick distinct layers in LLaMA-2-7B and observe how the pattern of probe predictions changes as the layer gets deeper. Layer 0 (i.e. the first transformer block after embedding layer), 10, and 30 are selected to represent early, intermediate, and late layers respectively. The trend of change on the first input number a is shown in Figure 4.

On early layers like layer 0, the predictions of probes are distorted to some extent: for original numbers with the same length, their corresponding predictions in the figure display a pattern of horizontal lines. This phenomenon indicates that early layers focus on the length of numbers, which corresponds to the number of input digit tokens.

As the layer gets deeper, probes on intermediate layers show the best performance. On layer 10, the predicted results are very close to the actual answers, yielding a near-perfect linear probe for original numbers. However, noise emerges in the prediction results again in late layers, with the form of uniformly distributed errors.

The trend of change leads us to a conjecture that language models first roughly estimate the value of a number with its token length, and then refine the estimation in subsequent layers. The process may not be lossless, which leads to errors in the final number encoding of language models.

## 3.3 Numeric Information Persist at Subsequent Positions

Another question is whether these encoded values are only stored at certain positions, or are they persist at subsequent positions. For input number values $^ { a , }$ b, we train probes at every individual token position to examine where these values exist. Figure 5 shows the mean square error of probes on the LLaMA-2-7B model.

![](images/aecc7601a18d29814355b15c74e6555e003733acc7487c9405e7e7c5f8f83560.jpg)  
(a) Layer 0

![](images/93530bcd1fbb5fa9fe24da70adc35ef042ba31b66fe76bccea745346efdce4fd.jpg)  
(b) Layer 10

![](images/8f4cd72c65ce7f4d02ad819d395912602c5e2a3ebc66d98914e1e08852b021fb.jpg)  
(c) Layer 30

Figure 4: How the pattern of probe predictions on the first input number a changes as the layer gets deeper. Probe predictions on different layers of LLaMA-2-7B show different patterns.  
![](images/a001d665ed54a35b3b28e57a992e7a66d32ff0ddc28c57460a670eae28ec0f22.jpg)  
(a) MSE of probes on a.

![](images/92d097103958d1664e1d031a177933972c10b4bfedd58308bb4bcf8cefd20fde.jpg)  
(b) MSE of of probes on b.  
Figure 5: The mean square error (MSE) of probes at different token positions on LLaMA-2-7B. <n1> represents the last token of the first input number a, and <n2> represents the last token of the second input number b, respectively. The rectangular pattern indicates that the value of an input number can be read out at any subsequent position.

The results demonstrate a clear rectangular pattern, indicating that the value of an input number can be read out at any subsequent position. In other words, the number values would persist at subsequent positions. It is also worth noticing that the probing accuracy on the last token is lower than other positions, which may be interpreted as language models do not continue to remember input numbers after computing the final outcome.

## 3.4 LLMs Encode Numbers Linearly

Previous work (Nanda et al., 2023; Gurnee and Tegmark, 2023) on probing neural networks propose the linear representation hypothesis: the presence of features of a neural network can be proved by training a linear projector which projects the activation vector to the feature space, and complex structures are unnecessary. To verify whether the numbers can be represented linearly, we follow the method of Gurnee and Tegmark (2023) which trains two-layer MLP probes and compares their performance with linear probes. The MLP probes have an intermediate hidden state of 256 dimensions and can be formulated as:

$$
\mathbf { P } = \mathbf { W } _ { \mathrm { 2 } } \mathbf { R e L U } ( \mathbf { W } _ { \mathrm { 1 } } \mathbf { H } + b _ { \mathrm { 1 } } ) + b _ { \mathrm { 2 } }\tag{6}
$$

where $\mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } , b _ { 1 }$ and $b _ { 2 }$ are trainable weights.

Figure 6 demonstrates the comparison between MLP probes and linear probes on mean square error. We find that nonlinear MLP probes do not show any clear advantage over linear probes, proving that the encoded number values can be represented linearly, or at least near-linearly.

## 4 Do LLMs Utilize Number Values?

The previous section has proved the existence of encoded number values in language models. However, an inherent issue is that the probed information is only correlational to the output of models, and no causal effects can be directly claimed (Belinkov, 2022).

![](images/b8eab719dbeb9dee2ceca968ab1aea6e279f9b296c342b693cbc36da835a569b.jpg)  
Figure 6: The comparison between linear probes and MLP probes on mean square error (MSE). The MLP probes do not show advantage over linear probes. More detailed experiments are reported in Appendix E.

In this section, we will try to verify the hypothesis that language models do use the encoded number values to get their calculation results by performing a set of intervention experiments. Given an input question $Q$ with an expected result of $^ { O , }$ we intervene in the internal activation of language models to make it believe in an altered question $Q ^ { \prime }$ and observe how the new result $o ^ { \prime }$ changes.

To ensure the effectiveness of the intervention, we conduct the experiments on Mistral-7B with 4- digit addition questions as input, where the model could correctly answer most of the questions.

![](images/a61079ef39df4e3517607333fdd5c265dce7749be92c69d5511023709b14cdc0.jpg)  
Figure 7: The effect of patching on different components. Early and mid refer to the non-number tokens before and after the first input number a, and last refers to the last token of the input text.

## 4.1 Patching Encoded Numbers

Firstly, we study the influence of number encoding at different positions by changing the activation of language models. We adopt the activation patching technique proposed by Stolfo et al. (2023) to quantify the importance of encoded number values hi at different layers i and different token positions.

To be specific, given an input addition problem consisting of input numbers a and $b ,$ we will conduct the following procedure:

1. Obtain the clean output of the language model $o = L M ( a , b )$

2. Replace a with another number $a ^ { \prime }$ to get a new output $o ^ { \prime } = L M ( a ^ { \prime } , b )$ , and record the hidden states $\mathbf { h } ^ { \prime }$ at certain position t during the forward pass;

3. Perform an additional forward pass with a and b as input numbers, where we substitute the hidden state $\mathbf { h } _ { i }$ of layer i with $\mathbf { h } _ { i } ^ { \prime } .$ . This would lead to an intervened result $o ^ { * }$

We set $a ^ { \prime } = 9 9 9 9$ in our experiments, and evaluate the effect of intervention as:

$$
E ( i , t ) = \frac { \left| o ^ { * } - o \right| } { \left| o ^ { \prime } - o \right| }\tag{7}
$$

which measures how much a specific layer i at position t affects the final result. Note that the metric is intended for qualitative rather than quantitative analysis.

Figure 7 demonstrates the effect of activation patching on different components, from which we can draw multiple observations:

<table><tr><td>Patching</td><td>Result</td><td>Explanation</td></tr><tr><td>None</td><td>6912</td><td> $5 6 7 8 + 1 2 3 4 = 6 9 1 2$ </td></tr><tr><td>Full</td><td>11233</td><td> $9 9 9 9 + 1 2 3 4 = 1 1 2 3 3$ </td></tr><tr><td> $5  9$ </td><td>10912</td><td> $9 6 7 8 + 1 2 3 4 = 1 0 9 1 2$ </td></tr><tr><td> $6  9$ </td><td>7212</td><td> $5 9 7 8 + 1 2 3 4 = 7 2 1 2$ </td></tr><tr><td> $7  9$ </td><td>6932</td><td> $5 6 9 8 + 1 2 3 4 = 6 9 3 2$ </td></tr><tr><td> $8  9$ </td><td>6913</td><td> $5 6 7 9 + 1 2 3 4 = 6 9 1 3$ </td></tr></table>

Table 1: Patching results on the question “Question: What is the sum of 5678 and $1 2 3 4 ^ { \overline { { { \cdot } } } \overline { { { \cdot } } } }$ by patching the activation on layer 8.

Each digit affects the result independently. The effect of patching on different number digits displays a clear pattern: the earlier a digit appears, the more patching it changes the final output value. While the latter digits encode the values of partial number sequences (See Appendix F for details), activation patching seems to only change the final result by the value of the digit itself. For example, although the activation at digit $\mathbf { \bar { \Psi } } ^ { 6 6 } 3 ^ { \mathbf { \vec { \prime } } } \mathbf { \bar { \Psi } }$ in “1234” encodes the value of 123, patching it equals changing the input number to "1294" rather than "9994", as demonstrated in Table 1. More detailed experiments are reported in Appendix H.

![](images/09d3c556f58ebe9320ba687d565eaf36305d6ab5d004f2111d22916c31a433c8.jpg)  
Figure 8: The success rate of performing a linear intervention on 6 consecutive layers. More detailed experiments are reported in Appendix I.

Language models concern only certain tokens during calculation. Despite our finding in Section 3.3 that encoded number values would persist in subsequent tokens, patching non-number tokens has almost zero effect on the final outcome. This pattern indicates that the encoded number values at most positions are simply “memorized” rather than “used” by the language model. An exception is the last token, where language models seem to store their calculation results.

Early and late layers play different roles. The effect of activation patching can be divided into two parts: on early layers before layer 14, patching the number tokens greatly influences the final outcome, while patching the last token is mostly ineffective; but in late layers after layer 20 it is just the opposite. We assume that early layers perform the task of processing the value of input number token sequences, while late layers use encoded values to calculate the final outcome, which is similar to the findings in Stolfo et al. (2023).

## 4.2 Linearly Intervening Encoded Numbers

To determine whether the encoded computational results causally affect the outcome of language models, we linearly intervene the hidden states and see whether the output changes as expected.

Method. Following the method of Nanda et al. (2023), for each intervened layer i, we add the number encoding direction vector $\mathbf { d } _ { i }$ to the residual stream hi:

$$
\mathbf { h } _ { i } ^ { ' } = \mathbf { h } _ { i } + \alpha \mathbf { d } _ { i }\tag{8}
$$

where $\alpha > 0$ is a scaling factor and the direction vector $\mathbf { d } _ { i }$ is obtained by normalizing the probe coefficients:

$$
\mathbf { d } _ { i } = \frac { \mathbf { W } _ { i } } { | \mathbf { W } _ { i } | }\tag{9}
$$

Considering that the probed number value is the projection of $\mathbf { h } _ { i }$ along the direction ${ \bf d } _ { i }$ , the effect of our intervention is to “push” the residual stream towards a larger encoded number. We set $\alpha = 2$ in our experiments, and intervened language models outputting a larger number $\smash { o ^ { \prime } > o }$ than the original prediction o is viewed as a success.

In the linear intervention experiment, we choose probes for language model predictions o at the last input token to obtain the direction vector $\mathbf { d } _ { i } ,$ and perform an intervention on every newly generated token. We use a test set of 1,000 entries and measure the efficacy of our intervention by observing the ratio of successful interventions.

We also use two alternative directions as baselines: normalized $\mathbf { h } _ { i }$ as null intervention, and a random unit vector I as random intervention.

Result and Findings. Figure 8 shows the success rate of intervening on 6 consecutive layers. Linear intervention achieves the highest success rate of 0.73 when intervening layer 14 to layer 19, outperforming the null intervention baseline by a large margin. This suggests that the linearly encoded number values are causal to model predictions.

It is also worth noticing that intervening on midlate layers is significantly more effective than on early layers and late layers. We hypothesize that this phenomenon is related to the findings of Stolfo et al. (2023): language models use mid-late layers to perform arithmetic computations, while the late layers are responsible for converting the computational result to output tokens.

## 5 Discussion and Future Directions

In previous sections, we find that LLMs know the value of numbers and utilize the encoded number values to perform calculations. However, the compression may not be lossless, and the calculation ability scales with model size. Moreover, the ability to understand and utilize numbers is positively correlated to mathematical competency. These findings reveal some future research directions that are potentially promising.

The exact way that LLMs encode numbers. While our experiments show that the original input number cannot be reconstructed from the hidden state via linear probes, there exists a possibility that the LLMs encode numbers in a way that is close to a linear projection but not identical, such as the floating-point system (Muller et al., 2018). Finding out the exact encoding, if possible, will give us a better insight into how LLMs function.

Specialized number encoding systems. The loss of encoded number values in LLMs will inevitably bring errors to subsequent computation, especially for large input numbers. Developing specialized encoding systems that could give precise presentations for numbers (Golkar et al., 2023) could eliminate errors at the root, thus helping LLMs better solve mathematical problems.

Mitigating computational errors with encoded numbers. By adding modules that directly utilize the encoded numbers in language models, the computational errors may be further reduced, especially on large-number calculations. We conduct a pioneer experiment in Appendix J to reveal the potential of controlling computational errors with probed numbers.

## 6 Related Work

Large Language Models on Mathematical Problems. Large language models (LLMs) like the GPT series (OpenAI, 2023), PaLM (Anil et al., 2023) and LLaMA (Touvron et al., 2023a,b) have demonstrated their impressive ability in various fields (Zhao et al., 2023; Li et al., 2023b; Taori et al., 2023; Chen et al., 2021; Nijkamp et al., 2022; Li et al., 2023a). On mathematical datasets like GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021), there have been methods like chain-of-thought reasoning (Wei et al., 2022) and self-consistency (Wang et al., 2022) to help LLMs better solve these questions. Specialized large language models like MetaMath (Yu et al., 2023) and Math-Shepherd (Wang et al., 2023) also show great competency.

Interpreting Internal Representations in Language Models. Prior research has unveiled that language models are able to store certain information in their hidden states, for example, passive voice (Shi et al., 2016) and sentence structure (Tenney et al., 2018). By adopting the probing technique (Alain and Bengio, 2016; Belinkov, 2022), complex representations have also been detected in language models: Li et al. (2022) shows that language models are capable of memorizing the state of an Othello game, and Nanda et al. (2023) further proves that the states can be linearly represented; Li et al. (2021) claims that language models are able to encode the properties and relations of entities; Gurnee and Tegmark (2023) reveals evidence that large language models build spatial and temporal representations about an entity from early layers.

Explaining Numbers and Arithmetic in Language Models. How language models process numbers has been studied by multiple researchers. Wallace et al. (2019) detects the existence of numeracy in static pre-trained word embeddings. Hanna et al. (2024) finds a critical circuit that performs greater-than comparisions in GPT-2. Stolfo et al. (2023) studies how language models process arithmetic information by intervening on specific modules of the model. Zhong et al. (2024); Engels et al. (2024) discover evidence that numbers on modular arithmetic may be circularly encoded.

## 7 Conclusion

In this paper, we study the question of whether and how large language models encode the value of numbers. If number values can be extracted from the internal representations of LLMs, we can assume that LLMs encode the value of numbers in their hidden states. We construct a dataset consisting of simple addition problems and introduce linear probes to investigate whether language models encode number values.

Experimental results prove that LLMs do encode the value of input numbers, and the representation could be linearly read out. The ability to linearly encode numbers is consistent across different model scales, and the encoding seems to be the most precise on intermediate layers. Further experiments show that LLMs utilize the encoded number values to perform arithmetic calculations, and the behavior of language models can be controlled via simple linear interventions, proving the causal connection between encoded numbers and model outputs.

Our work shows a glimpse of the internal mechanisms of how language models solve mathematical questions. Future works on the internal representations of numbers, for example, better probes and specialized number encoders, may enhance the mathematical competence of language models in an explainable way.

## Acknowledgements

We thank the anonymous reviewers for their insightful comments. This paper is supported by NSFC project 62476009. The contact author is Zhifang Sui.

## Limitations and Risks

While we explore the inner mechanisms of how language models understand numbers, the probes trained in our current method are only approximations of the encoded numbers rather than exact internal presentations. Directly performing calculations with probes would lead to undesired results. Meanwhile, our experiments are conducted on LLMs whose parameters are openly available, while other LLMs the ChatGPT or GPT-4 may exhibit different behaviors.

## References

Guillaume Alain and Yoshua Bengio. 2016. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. arXiv preprint arXiv:2305.10403.

Yonatan Belinkov. 2022. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Sean Welleck, Peter West, Chandra Bhagavatula, Ronan Le Bras, et al. 2024. Faith and fate: Limits of transformers on compositionality. Advances in Neural Information Processing Systems, 36.

Joshua Engels, Isaac Liao, Eric J Michaud, Wes Gurnee, and Max Tegmark. 2024. Not all language model features are linear. arXiv preprint arXiv:2405.14860.

Siavash Golkar, Mariel Pettee, Michael Eickenberg, Alberto Bietti, Miles Cranmer, Geraud Krawezik, Francois Lanusse, Michael McCabe, Ruben Ohana, Liam Parker, et al. 2023. xval: A continuous number encoding for large language models. In NeurIPS 2023 AIfor Science Workshop.

Wes Gurnee and Max Tegmark. 2023. Language models represent space and time. arXiv preprint arXiv:2310.02207.

Michael Hanna, Ollie Liu, and Alexandre Variengien. 2024. How does gpt-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model. Advances in Neural Information Processing Systems, 36.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. In Thirtyfifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Belinda Z Li, Maxwell Nye, and Jacob Andreas. 2021. Implicit representations of meaning in neural language models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1813–1827.

Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2022. Emergent world representations: Exploring a sequence model trained on a synthetic task. In The Eleventh International Conference on Learning Representations.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. 2023a. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161.

Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng Ding, Lidong Bing, Shafiq Joty, and Soujanya Poria. 2023b. Chain of knowledge: A framework for grounding large language models with structured knowledge bases. arXiv preprint arXiv:2305.13269.

Thomas McGrath, Matthew Rahtz, Janos Kramar, Vladimir Mikulik, and Shane Legg. 2023. The hydra effect: Emergent self-repair in language model computations. arXiv preprint arXiv:2307.15771.

Jean-Michel Muller, Nicolas Brisebarre, Florent De Dinechin, Claude-Pierre Jeannerod, Vincent Lefevre, Guillaume Melquiond, Nathalie Revol, Damien Stehlé, Serge Torres, et al. 2018. Handbook of floating-point arithmetic. Springer.

Neel Nanda, Andrew Lee, and Martin Wattenberg. 2023. Emergent linear representations in world models of self-supervised sequence models. In Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 16–30.

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2022. Codegen: An open large language model for code with multi-turn program synthesis. In The Eleventh International Conference on Learning Representations.

OpenAI. 2023. Gpt-4 technical report.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Xing Shi, Inkit Padhi, and Kevin Knight. 2016. Does string-based neural mt learn source syntax? In Proceedings of the 2016 conference on empirical methods in natural language processing, pages 1526– 1534.

Alessandro Stolfo, Yonatan Belinkov, and Mrinmaya Sachan. 2023. A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 7035–7052.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. 2023. Stanford alpaca: An instruction-following llama model.

Ian Tenney, Patrick Xia, Berlin Chen, Alex Wang, Adam Poliak, R Thomas McCoy, Najoung Kim, Benjamin Van Durme, Samuel R Bowman, Dipanjan Das, et al. 2018. What do you learn from context? probing for sentence structure in contextualized word representations. In International Conference on Learning Representations.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Eric Wallace, Yizhong Wang, Sujian Li, Sameer Singh, and Matt Gardner. 2019. Do nlp models know numbers? probing numeracy in embeddings. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5307–5315.

Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Y Wu, and Zhifang Sui. 2023. Math-shepherd: A label-free step-by-step verifier for llms in mathematical reasoning. arXiv preprint arXiv:2312.08935.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284.

Ruochen Zhao, Xingxuan Li, Shafiq Joty, Chengwei Qin, and Lidong Bing. 2023. Verify-and-edit: A knowledge-enhanced chain-of-thought framework. arXiv preprint arXiv:2305.03268.

Ziqian Zhong, Ziming Liu, Max Tegmark, and Jacob Andreas. 2024. The clock and the pizza: Two stories in mechanistic explanation of neural networks. Advances in Neural Information Processing Systems, 36.

## A Dataset Details

The dataset in Section 2.2 contains 9000 addition problems. For each number of digits between 2 and 10, 1000 problems are generated, and two numbers in the same problem share the same digit. For questions whose number has 4 or fewer digits, we list all possible combinations of numbers and randomly sample 1,000 of them to generate the questions. For questions whose number has 5 or more digits, we randomly sample both numbers to generate the 1000 questions.

## B Experiment Implementation

The experiments are conducted on 4 NVIDIA GTX 3090 GPUs. Acquiring the hidden states of LLMs on our synthetic dataset requires 1020 GPU hours<sup>˜</sup> per model.

We obtain the LLaMA-2 models and Mistral-7B model from the huggingface model hub, and implement the experiments with the huggingface transformers Python library. The probes are trained with the scikit-learn Python library. We use the TransformerLens library<sup>5</sup> for intervention experiments. We follow the terms of use of all models and use them only for research.

![](images/8bde9eb994a2d1971df86e34507a7c8f409f2946c870db137710140abcf78453.jpg)  
Figure 9: The overall accuracy of language model predictions on addition problems.

## C Experiments on Subtraction Problems

In the main paper, we only show the results of probing on addition problems. We also conduct experiments on subtraction problems with the form of:

Question: What is the result of {a} minus {b}?

where we assert $a > b$ to ensure the result being a positive number.

Figure 10 demonstrates the result of probing on subtraction problems. We can clearly observe that the trends of different metrics are similar to those on addition problems. In other words, the behaviour of language models on subtraction problems are similar to the behaviour on addition problems.

## D Overall Accuracy

Figure 9 shows the overall accuracy of different language models on addition problems. We can see that the accuracy of all models, especially LLaMA-2 models, faces a sharp decline at 6-digit problems, which may have a possible correlation with the partial number encoding accuracy demonstrated in Figure 12.

In the LLaMA-2 family, the 13B model does not show any advantage over the 7B model on probing metrics. In contrast, Mistral-7B displays better performance on all probing metrics, which is consistent with its outstanding math ability. The difference implies that the ability to encode numbers is consistent across different model scales, but varies between different model families. Meanwhile, the ability to understand numbers show a positive correlation with the math ability of LLMs.

## E Detailed Experiments on Linearity

Figure 11 shows the comparison between linear probes and MLP probes on $\rho , R ^ { 2 }$ and MSE. We can observe that MLP probes generally perform no better than linear probes.

## F Experimental Results on Partial Number Encoding

In large language models like LLaMA-2, large numbers are split into multiple tokens, where each token represents a certain digit of the original number. This raises a question: whether the encoding process will proceed from token to token, or will it only happen at the end of number token sequences?

To investigate the problem, we choose addition problems consisting of 8-digit numbers and probe the value of the partial number sequence at every token position. For example, given a number token sequence “12345678”, we will probe the value 12 at the position of token “2”, and probe the value 123 at the position of token “3”.

Figure 12 shows the probing accuracy of 3 models. It can be observed that the value of the partial number sequence can be read out at every token position. In other words, language models encode the number token sequence incrementally.

Meanwhile, the accuracy significantly declines as the token sequence gets longer, which means that language models face increasing difficulty in capturing the precise value as the number gets larger in scale. Notice that Mistral-7B suffers less from accuracy decay, we can assume that the ability to precisely encode long number token sequences is positively correlated to the mathematical ability of language models.

![](images/9379d0dfcb1dac09abaea3ac7569508775f0c21bb8feb5075263998d24c0c3fb.jpg)  
(a) ρ of probes on a.

![](images/a2bf32cda781f0a9544e871baf71a1293dde6617938aeed3f47b191f7f2fcf30.jpg)  
(b) $\rho$ of probes on b.

![](images/009c7fe5ec7f02312331229ebd4a0d1b94433a87986ccbd9ef2cad49f6b76d9c.jpg)  
(c) ρ of probes on o.

![](images/555f1238bc7cc6897390e06a390dace2c03280184512e36c9bf2d173fcb1c8a4.jpg)  
(d) $R ^ { 2 }$ of probes on a.

![](images/b114d0fec3307fe1e6240b3719a432d465a354f85b99bb67aeb2bb9e01954e10.jpg)

![](images/35e751dd8f5f1e8109bd0a2c9bde661969070630c5ee7de8a9ed81dfd4f206fc.jpg)  
(e) $R ^ { 2 }$ of probes on b.  
(f) $R ^ { 2 }$ of probes on o.

![](images/de38577cc800ff2f9dd49e3d894ba0b1dbeef5060edcc7b0848380d4d8dc92a3.jpg)  
(g) AAcc of probes on a.

![](images/4b664ac05787e972de700ae762d3f3b2c6f460edc133c616bf38d4a635f4499a.jpg)

![](images/b401d482e1654012d274b9c7f8097cd76e392cdc6d67f51112c15dd505dbb065.jpg)  
(h) AAcc of probes on b.  
(i) AAcc of probes on o.

![](images/5eb6a2d83735581a129c0136e166263eea6989b1380d7aa562e48a496a3974ec.jpg)  
(j) MSE of probes on a.

![](images/bb8cf7a86b8401b6f187c50b5af7bbbd0a295124aa304353b9aec0f4a775498f.jpg)  
(k) MSE of probes on $b .$

![](images/f04aeac97f0551e76d58f2139c5271720f5000024a82de90c8eee90357286aa1.jpg)  
(l) MSE of probes on o.

Figure 10: Pearson coefficient $( \rho ) ,$ , out-of-sample $R ^ { 2 }$ , approximate accuracy (AAcc), and mean square error (MSE) of probes on different layers for subtraction problems. a and b refer to the two input numbers denoted in Section 2.2, and o refers to the prediction of language models respectively. High $\rho$ and $R ^ { 2 }$ indicate the existence of encoded number values in the hidden states.

![](images/4d7388e3843968047ef7368d65c4ba8b786b5a98cf2fb83b8613182d298dae19.jpg)  
(a) Pearson coefficient

![](images/4cf718353ff165724cfc387da288d5f4e0363b00309d0dd4988234b9c065ab70.jpg)  
(b) Out of sample $R ^ { 2 }$

![](images/c4c1fd2a6739c5f3d043c5c9cab497830e93a7dc5061ca166b46d49320de56f8.jpg)  
(c) Mean square error

Figure 11: Comparison between linear probes and non-linear MLP probes. Pearson coefficient, out-of-sample $R ^ { 2 } .$ and AAcc of probes on the first input number a on different layers are shown in the figure.  
![](images/a1a0f1bf9536afe263caccfd938ce65b36bdf7349b577a6ef8d6d0b3c6fc96f0.jpg)  
(a) AAcc of LLaMA-2-7B.

![](images/f28c966cd04a301b3f907bdd0db2f95764ab1630b31d20ddba11b37cec061afb.jpg)  
(b) AAcc of LLaMA-2-13B.

![](images/e053164f47e4c7d46a4274783fd7aef4e430d1238df7936b18693661ebd58474.jpg)  
(c) AAcc of Mistral-7B.  
Figure 12: The approximate accuracy (AAcc) of probes on partial number sequence of 8-digit numbers. The y-axis represents the index of number tokens in the token sequence.  
the probing performance on c and $a , b .$

Figure 13 shows the Pearson coefficient, out-ofsample $R ^ { 2 }$ , and mean square error of probes on partial sequence of 8-digit numbers. These metrics remain stable as the length of number token sequence gets longer, indicating that language models do have the ability to incrementally encode number values, but there would be more error when the number gets larger in scale.

## G Probing With Control Tasks

There exists the risk that probes may learn to extract values that language models do not encode. In Figure 5, we can see that probing on the second input number b at positions before it appears would lead to extremely large mean square errors, which acts as a piece of preliminary evidence that the probe performance does not solely come from probe strength.

To quantify the influence of probe strength, we conduct an experiment that probes with control tasks. For each question, we generate a random number c that shares the same digit with a and b as the control signal. If the probing performance comes from the encoded number values rather than probe strength, there would be a clear gap between

Figure 14 shows the difference between probe performances. It can be observed that probing on input numbers constantly yields better performance than probing on random control signals, proving that language models do encode number values in their hidden states.

Meanwhile, probing b on positions before b shows performance similar to probing c, which corresponds to the fact that b is unknown to the model at these positions.

<table><tr><td>Patching</td><td>Result</td><td>Explanation</td></tr><tr><td>None</td><td>6912</td><td> $5 6 7 8 + 1 2 3 4 = 6 9 1 2$ </td></tr><tr><td>Full</td><td>11233</td><td> $9 9 9 9 + 1 2 3 4 = 1 1 2 3 3$ </td></tr><tr><td> $5  9$ </td><td>10912</td><td> $9 6 7 8 + 1 2 3 4 = 1 0 9 1 2$ </td></tr><tr><td> $6  9$ </td><td>7212</td><td> $5 9 7 8 + 1 2 3 4 = 7 2 1 2$ </td></tr><tr><td> $7  9$ </td><td>6932</td><td> $5 6 9 8 + 1 2 3 4 = 6 9 3 2$ </td></tr><tr><td> $8  9$ </td><td>6913</td><td> $5 6 7 9 + 1 2 3 4 = 6 9 1 3$ </td></tr></table>

Table 2: Patching results on the question “Question: What is the sum of 5678 and 1234 $\because$ by patching the activation on layer 8.

![](images/3109ef526c7a649b8f013ed184b86ed0427d84af1a35ed07fa321a7ed176336d.jpg)  
(a) ρ of LLaMA-2-7B.

![](images/e764cea5bd185a42eb8789b6db7b1563e59311e9f218d4fdcabf87a284e70340.jpg)  
(b) ρ of LLaMA-2-13B.

![](images/bb5914d977f4eca16a1d06bed4757d0923229a9ea033a8846dd3d392fc6f0689.jpg)  
(c) ρ of Mistral-7B.

![](images/57a42d492684046026faa0f2f139b07c028495b542dff026d710dc83cb959d6a.jpg)  
(d) $R ^ { 2 }$ of LLaMA-2-7B.

![](images/7d13a86f805eea79af1864e208d695beb1a57b70edad3e6208c42426331e6719.jpg)  
(e) $R ^ { 2 }$ of LLaMA-2-13B.

![](images/46c74da0234492a20fa51ca14b1d26119a699d33435b51c1f42990ad9cddc2d7.jpg)  
(f) $R ^ { 2 }$ of Mistral-7B.

![](images/708440131b121f9c2aef6452bf573ece203d72664f1a2829e0bc43da3235f3d9.jpg)  
(g) MSE of LLaMA-2-7B.

![](images/78de74157c7b604f226c8e6448c33b9898a05adbb144bd08286dd79530363e84.jpg)  
(h) MSE of LLaMA-2-13B.

![](images/f466db90058b2da1286c67d6b99a1a51f80cc0a88703f527a736df752dbd4c0d.jpg)  
(i) MSE of Mistral-7B.  
Figure 13: The Pearson coefficient (ρ), out-of-sample $R ^ { 2 } ,$ , and mean square error (MSE) of probes on partial number sequence of 8-digit numbers. The y-axis represents the index of number tokens in the token sequence.

## H Detailed Experiments on Activation Patching

Table 3 shows the results of patching on layer 8 of Mistral-7B on the question “Question: What is the sum of 5678 and 1234 ?”

We can clearly see that patching a digit will only influence the value of the digit itself, rather than the value of the partial token sequence: patching the last digit 8 in 5678 equals changing the number to 5679 rather than 9999, although the encoded value of 9999 can be found in the activation. We hypothesize that language models encode the number values from scratch at every new position, rather than using previous encoded values.

We also notice that patching the last number digit on early layers shows a higher effect than expected, but the reason why the last digit is more special is still unknown.

## I Detailed Experiments on Linear Intervention

## I.1 Success Rate

Figure 15 shows the success rate of intervening on 5 consecutive layers with a maximum success rate of 0.698, and Figure 16 shows the success rate of intervening on a series of layers starting from layer 14. It can be observed that a sufficient number of layers need to be intervened for language models to successfully change their predictions. Nanda et al. (2023) observed a similar phenomenon in OthelloGPT, and a related hypothesis is that language models demonstrate the Hydra effect (McGrath et al., 2023), where other layers would self-repair the intervention on certain layers.

## I.2 Output Patterns

We also observe that while intervening on early or late layers both lead to poor success rates, they display different patterns of output. Table 3 shows the result of intervening on different layers of Mistral-7B. It can be seen that performing a linear intervention on early layers would completely destroy the final outcome, while intervening on late layers will not change the result at all. We hypothesize that the number encoding in early layers has not fully developed yet, and intervening in it would lead to unexpected results; In late layers, the number encoding is simply remembered but not used, and the language models rely on other subspace to decode the final outcome.

![](images/b2d58237461991735db900b210fc5eb266521a59e1cafaf832129d4a70d259bf.jpg)  
(a) a versus c of LLaMA-2-7B.

![](images/d11a7c5e7e13b3a86ac6101af29a060e84f2d8994f6dc41ace4506cf4720e848.jpg)  
(b) a versus c of LLaMA-2-13B.

(c) a versus c of Mistral-7B.  
![](images/565070cedb5084f3ea9907e5df99ac569922bd88ffaa343c1256cac686b7d036.jpg)

![](images/58ae838352a6300edd517f51cdaf47fd87f0ab346e7ca38a042871d8b260ce01.jpg)  
(d) b versus c of LLaMA-2-7B.

![](images/3aeba8412d56af8f86813468281ab1b072abc7db0241b98a1fe2cc96784c394a.jpg)  
(e) b versus c of LLaMA-2-13B.

![](images/2077573f8aa7389c6de74c1fe869781796e888615e46ea0c48d4d542495dbd33.jpg)  
(f) b versus c of Mistral-7B.

Figure 14: The difference in mean square error (MSE) between probes on input numbers and control signals. A lighter color indicates a greater performance gap.  
![](images/6a13ad3f8a31ae8aad0a932d45f0e4f3ecdff9f634e694c133e354401b597da3.jpg)  
Figure 15: The success rate of performing a linear intervention on 5 consecutive layers.

![](images/e249210ff714ebf350843c12bbde3df0069df58c9887561ff1a9031d62fabb94.jpg)  
Figure 16: The success rate of performing a linear intervention on layers starting from layer 14.

## I.3 Additional Experiments

We have also tried to change the probed number from the original value o to a new value $o + o ^ { \prime } \mathrm { { i } }$

$$
\mathbf { h } _ { i } \mathbf { W } _ { i } + b _ { i } = o\tag{10}
$$

$$
\mathbf { d } _ { i } = o ^ { \prime } \frac { \mathbf { W } _ { i } } { | \mathbf { W } _ { i } | ^ { 2 } }\tag{11}
$$

$$
( \mathbf { h } _ { i } + \mathbf { d } _ { i } ) \mathbf { W } _ { i } + b _ { i } = o + o ^ { \prime }\tag{12}
$$

However, the intervention does not yield results as expected: the intervened model continues to predict o rather than $o + o ^ { \prime } .$

A possible hypothesis is that the probed number value is the projection of $\mathbf { h } _ { i }$ along the direction $\mathbf { W } _ { i } ,$ and simply adding vectors to $\mathbf { h } _ { i }$ would draw it away from its valid subspace. To maintain intervened $\mathbf { h } _ { i }$ in its valid subspace, it should be rotated along certain direction. The method of precisely changing the encoded number values in language models still remains to be explored.

<table><tr><td>Layer</td><td>Generation Result</td></tr><tr><td>0-5</td><td>Answer: gainedcnt I</td></tr><tr><td>14-19</td><td>IIIIIIIIICCC Answer: 12515</td></tr><tr><td>25-30</td><td>Answer: 6455</td></tr></table>

Table 3: Intervention results on the question “Question: What is the sum of 2936 and 3519 ?”. Running Mistral-7B without intervention would lead to the result of 6455.

![](images/b34fe2e7f647694dc9b060bec0eb97996bc383794a874f9371e5694162e85d40.jpg)  
Figure 17: The success rate of performing a linear intervention on 6 consecutive layers, with a negative $\alpha = - 2 . 0$

We also experimented on negative α values, which will "push" the residual stream towards a smaller encoded number. The results are demonstrated in Figure 17. We can see that the trend of success rate is similar to the trend in Figure 8, further proving that the value of calculation result can be linearly intervened.

## J Directly Calculate with Encoded Number Values

We are curious about whether the probed number values could help LLMs better perform calculations. Considering that adding the probed input numbers does not yield precise answers (Section 3.1), we evaluate the sum of probed numbers with two new metrics: logMSE and error margin.

$$
\mathrm { l o g M S E } ( \mathbf { S } , \mathbf { G } ) = \mathrm { a v g } ( ( \log _ { 2 } \mathbf { S } - \log _ { 2 } \mathbf { G } ) ^ { 2 } )
$$

$$
\operatorname* { m a r g i n } ( \mathbf { S } , \mathbf { G } ) = \operatorname* { m i n } ( \frac { \operatorname* { m a x } ( | \mathbf { S } - \mathbf { G } | } { \mathbf { G } } ) , 1 )\tag{13}
$$

(14)

where S and G represent predicted answers and golden answers respectively. Both metrics indicate how much the calculated results deviate from the golden answers.

In Figure 18, despite failing to generate accurate answers, all three models could keep their logMSE and error margin at a very low level by adding probed a and $b ,$ while directly accepting the output of language models would lead to results that deviate far away from the golden answers. We think that this reveals a possibility to control the computational error of language models within a reasonable range, and will not produce results that are far too unreasonable.

We also notice that for LLaMA-2 models, adding the probed number on late layers will result in a high error margin, which may be a result of the findings in Section 4.1: number encoding on late layers is not used by the model.

![](images/2066bdde1bcd30ec896b77ef334848246d44ea911593b702db1b22b75b8fb181.jpg)  
(a) logMSE

![](images/dc687823b27f8e6791acb8ac14d0156bb0b5116a48dc653d23e17fe91b24f6e7.jpg)  
(b) Error margin  
Figure 18: Comparison between the sum of probed $( a , b )$ and language model predictions. AB means the sum of probed $( a , b )$ and LM means language model predictions.