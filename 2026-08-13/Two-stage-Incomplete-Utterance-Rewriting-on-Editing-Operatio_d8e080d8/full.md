# Two-stage Incomplete Utterance Rewriting on Editing Operation

Zhiyu Cao, Peifeng Li\*, Qiaoming Zhu, Yaxin Fan

School of Computer Science and Technology, Soochow University, Suzhou, China zycao18@stu.suda.edu.cn, yxfansuda@stu.suda.edu.cn {pfli, qmzhu}@suda.edu.cn

## Abstract

Previous work on Incomplete Utterance Rewriting (IUR) has primarily focused on generating rewritten utterances based solely on dialogue context, ignoring the widespread phenomenon of coreference and ellipsis in dialogues. To ad dress this issue, we propose a novel framework called TEO (Two-stage approach on Editing Operation) for IUR, in which the first stage generates editing operations and the second stage rewrites incomplete utterances utilizing the generated editing operations and the dialogue context. Furthermore, an adversarial perturbation strategy is proposed to mitigate cascading errors and exposure bias caused by the inconsistency between training and inference in the second stage. Experimental results on three IUR datasets show that our TEO outperforms the SOTA models significantly.

## 1 Introduction

Dialogue understanding (e.g., dialogue generation, dialogue sentiment analysis, and intent recognition) often suffers from ellipsis and coreference, because people often omit certain information or use pronoun in utterances for the sake of convenience in real-world conversations. Incomplete Utterance Rewriting (IUR) is to rewrite incomplete utterances more specific and direct, which is beneficial for many downstream dialogue understanding tasks, such as conversational dense retrieval (Qian and Dou, 2022) and dialogue summarization (Fang et al., 2022). Actually, IUR can be specifically categorized into coreference and ellipsis resolution. As shown in Table 1, the incomplete utterance u<sub>3</sub> uses the pronoun “he” to represent “Ben Affleck” and omits “as Batman”. The rewritten utterance $u _ { 3 } ^ { \prime } \ { } ^ { \ast } I t$ is Ben Affleck who acted as Batman” is more informative and direct than the original incomplete utterance “It is he who acted”.

<table><tr><td>Speaker(turn)</td><td>Utterance</td></tr><tr><td>Speaker1(u1)</td><td>I think Batman is very handsome.</td></tr><tr><td> $\mathbf { S p e a k e r _ { 2 } ( u _ { 2 } ) }$ </td><td>The poster looks a bit like Ben Affleck.</td></tr><tr><td>Speaker1(u3)</td><td>It is he who acted. (Incomplete utterance)</td></tr><tr><td> $\mathbf { S p e a k e r _ { 1 } ( u _ { 3 } ^ { \prime } ) }$ </td><td>It is Ben Affleck who acted as Batman.</td></tr></table>

Table 1: An example of IUR, where red and blue color indicate coreference and ellipsis, respectively. The first two utterances $\mathbf { u } _ { 1 }$ and $\mathbf { u } _ { 2 }$ are dialogue history, the third u is an incomplete utterance to be rewritten, and the fourth $\mathbf { u } _ { 3 } ^ { \prime }$ is the rewritten utterance.

Although previous methods have achieved great success, they only regarded IUR as a generation or sequence labeling task, which did not explicitly consider the two different operations of replacement (for coreference) and insertion (for ellipsis) (Hao et al., 2021; Chen, 2023; Li et al., 2023b; Peng et al., 2024). Coreference resolution typically only involves the replacement of a single span (most of them are entities, e.g., “Ben Affleck”), while ellipsis resolution might insert multiple discontinuous spans (e.g., “as” and “Batman”) from dialogue history at the same position in an incomplete utterance. Therefore, previous work treated coreference and ellipsis as a whole, which failed to account for the fundamental distinction between coreference and ellipsis. This resulted in the generation of utterances containing erroneous tokens or structures.

To address the aforementioned issues, we draw inspiration from the process of human article editing. When editing an article, humans typically read the entire text first, identify each area that requires insertion, deletion or replacement, and then reread the entire article to assess the reasonableness of each change. This prompts the question of whether it is possible to explicitly introduce editing operations into incomplete utterance rewriting.

We instantiate this idea by adopting a similar pipeline framework with a generation model to simulate human article editing. Specifically, we propose a novel framework called TEO (Two-stage approach on Editing Operation). We first use the editing operations as the pivot in the rewriting process, and train a model to generate editing operations in the first stage. The second stage is responsible for generating rewritten utterances based on the dialogue context and the editing operations. Furthermore, to mitigate the exposure bias caused by inconsistency between training and inference in the second stage, we propose three perturbation methods of editing operations to improve robustness. The experimental results on three IUR datasets show that our TEO significantly outperforms the SOTA models. In summary, our two-stage generation framework TEO has three advantages:

• The editing operations are taken as the pivot, with both insertion and replacement operations explicitly considered in order to address the issues of coreference and ellipsis.

• The local editing operations are first generated, and then the global dialogue context information and the editing operations are used to generate the final rewritten utterances. This enables the TEO to capture more fine-grained information.

• An adversarial perturbation strategy is proposed for editing operations that can mitigate the occurrence of cascading errors and exposure bias caused by inconsistency between training and inference.

## 2 Related Work

Research on IUR can be mainly divided into two types: generation methods (Huang et al., 2021; Inoue et al., 2022; Li et al., 2023b) and sequence labeling methods (Liu et al., 2020; Jin et al., 2022; Si et al., 2022; Chen, 2023; Du et al., 2023; Li et al., 2023a; Peng et al., 2024).

Most previous studies did not explicitly consider coreference resolution and ellipsis resolution. The initial research on IUR predominantly employed generation methodologies. For example, Huang et al. (2021) first employed a tagger to predict the rewriting labels and then utilized an autoregressive with copy mechanism for generating the rewritten utterances. Inoue et al. (2022) jointly trained two tasks: selecting key words and generating rewritten utterances.

Subsequent studies have indicated that the source and target utterances exhibit similar structural characteristics. Consequently, numerous subsequent studies have proposed sequence labelling methods. For example, Liu et al. (2020) regarded incomplete utterance rewriting as predicting wordlevel editing matrix. To address the issue of inserting multiple spans at one location, Chen (2023) directly selected spans from the context to form complete utterances. Du et al. (2023) incorporated sentence-level semantic relations between dialogue context and incomplete utterance. Li et al. (2023a) introduced the MLP architecture to mine the correlation between the contextual utterances and the rewritten utterance to obtain the editing matrix. Peng et al. (2024) paired spans and labeling the action types between spans.

Only a few works on IUR (Si et al., 2022; Li et al., 2023b) explicitly considered coreference and ellipsis resolution. Si et al. (2022) inserted markers into incomplete utterances to represent coreference and ellipsis through manually designed rules. However, these rules cannot cover all situations. Li et al. (2023b) first predicted the positions of insertion and replacement, and then filled these positions. However, they replaced the spans to be replaced with [MASK], resulting in the loss of span information before replacement.

## 3 Methodology

## 3.1 Task Definition

Let each piece of data be defined as $\{ H i s t , U _ { n } , Y \}$ where $\begin{array} { r c l } { H i s t } & { = } & { \left\{ U _ { 1 } , U _ { 2 } , . . . U _ { n - 1 } \right\} } \end{array}$ is the dialogue history including $n - 1$ utterances and $U _ { n }$ is the incomplete utterance that requires rewriting. The rewritten utterance of $U _ { n }$ is denoted as $Y _ { \pm }$ , where $Y$ keeps the semantics of $U _ { n }$ unchanged and complements the coreferential and omitted information in it. The goal of IUR is to learn a mapping $P ( T | H i s t , U _ { n } )$ satisfying $Y =$ arg max $P ( T | H i s t , U _ { n } )$   
T

## 3.2 Overview

We adopt a text generation approach to the IUR task. In contrast to previous research, which has treated IUR as a single task, we decompose it into two subtasks: editing operation generation and editing-aware rewritten utterance generation by using the editing operations $\mathcal { E }$ as the pivot. The objective of the overall training process is to maximize

$$
\begin{array} { r } { P ( Y | H i s t , U _ { n } ) = P ( \mathcal { E } | H i s t , U _ { n } ) P ( Y | H i s t , U _ { n } , \mathcal { E } ) , } \end{array}\tag{1}
$$

with the goal of maximizing $P ( \mathcal { E } | H i s t , U _ { n } )$ in the first stage and maximizing $P ( Y | H i s t , U _ { n } , { \mathcal { E } } )$ in the second stage, where $\mathcal { E }$ is the predicted editing operations in the first stage.

![](images/e810b9efb1ae8872ac289192e8048af486fd2d7cf852960bb230d2c5df4f45c1.jpg)  
Figure 1: An overview of our framework TEO, which includes two stages Editing Operation Generation and Editing-aware Rewritten Utterance Generation.

Our framework TEO is shown in Figure 1. In the first stage Editing Operation Generation, we generate the editing operations based on the dialogue history and the incomplete utterance that needs to be rewritten. Subsequently, in the second stage Editing-aware Rewritten Utterance Generation, we generate the rewritten utterance based on the editing operations and dialogue context.

## 3.3 Editing Operation Generation

In the process of text editing, humans primarily engage in the insertion, deletion, and replacement of tokens. In the first stage, the objective is to generate editing operations to guide IUR. As previously stated, IUR primarily aims to address the coreference and ellipsis issues in utterances. The coreference resolution is typically achieved through replacement operations, while the ellipsis resolution is typically achieved through insertion operations. Consequently, two editing operations have been defined for IUR: insertion and replacement.

To express the semantics of the editing operations in a more concise manner, it is not necessary to use natural language descriptions to represent insertion and replacement operations. Instead, three types of markers are employed for insertion and replacement. Specifically, we use $\mathbf { \hat { \Sigma } } ^ { \left\{ 6 \right\} } \left[ I \right] \mathbf { \Sigma } ^ { t k ^ { \prime } }$ to represent an insertion operation where $^ { 6 6 } [ I ] ^ { , , }$ refers to the insertion action and $^ { 6 6 } t k ^ { , 3 }$ refers to the inserted tokens. The replacement operation is indicated by the following sequence: $^ { * * } [ D ] t k _ { 1 } [ R ] t k _ { 2 } { } ^ { * }$ . Here, $^ { 6 6 } [ D ] ^ { , , }$ and $" [ R ] "$ refer to the deletion and replacement action, respectively, and the tokens $" t k _ { 1 } "$ are replaced by $\mathbf { \ w ^ { 6 6 } } t k _ { 2 } \mathbf { \ w ^ { , } }$

By doing so, we can obtain the structured editing operations and denote it as E, where each of the editing operations is ordered according to its position in the original utterance. As shown in Figure 1, by comparing the incomplete utterance “It is he who acted” with the rewritten utterance “It is Ben Affleck who acted as Batman”, we can obtain the set of editing operations as “[D] he [R] Ben Affleck [I] as Batman”.

To train a generation model to generate editing operations, we use the incomplete utterances and the rewritten utterances in the training set to create editing operations. Specifically, we first compare the incomplete utterance $U _ { n }$ and the rewritten utterance $Y$ to find their longest common subsequence, denoting as ${ \mathcal { L } } .$ For those tokens that appear in $U _ { n }$ but not in ${ \mathcal { L } } ,$ we mark them as deletion. For those tokens that appear in $Y$ but not in ${ \mathcal { L } } ,$ we mark them as insertion. Finally, we obtain a set of deletion and insertion operations. For each deletion and insertion operation with the same context, we label it as replacement, resulting in a set of replacement operations. By using the above methods, we can obtain the insertion operation set $\mathcal { T }$ and the replacement operation set R. For each incomplete utterance, our goal of editing operation generation is to generate the set of editing operations as follows,

$$
T ( \mathcal { Z } , \mathcal { R } ) = [ \mathrm { I } ] I _ { 1 } . . . [ \mathrm { I } ] I _ { i } . . . [ \mathrm { D } ] D _ { 1 } [ \mathrm { R } ] R _ { 1 } . . . [ \mathrm { D } ] D _ { j } [ \mathrm { R } ] R _ { j } ,\tag{2}
$$

where $I _ { i }$ represents the i-th span to be inserted, and

$D _ { j }$ and $R _ { j }$ represent the j-th span to be replaced and the span after replacement, respectively.

We use BART (Lewis et al., 2020) as our generation model and concatenate the dialogue history Hist and the incomplete utterance $U _ { n }$ as input. The output is the set of editing operations $\mathcal { E }$ of length l. The training objective of the editing operation generation is to minimize the negative log-likelihood loss function $L o s s _ { e d i t }$ as follows,

$$
\mathit { L o s s } _ { \mathit { e d i t } } = - \log \sum _ { i = 1 } ^ { l } P ( \mathcal { E } _ { i } | \mathcal { E } _ { < i } , H i s t , U _ { n } , \theta _ { 1 } ) ,\tag{3}
$$

where $\theta _ { 1 }$ is the parameters of the model in the first stage.

## 3.4 Editing-aware Rewritten Utterance Generation

Our editing process in the first stage can be seen as a process of proposing insertions and replacements. Due to the errors in the predicted editing operations of the first stage and the unknown positions for the insertion operations, we combine global dialogue information with the preliminary generated editing operations to generate the final rewritten utterances in the second stage. We hope that the model can review and refine the generated editing operations based on contextual information, in order to obtain correct rewritten utterances. Specifically, we train the model to generate the final rewritten utterances based on the dialogue history, the utterances to be rewritten, and the editing operations. Moreover, to construct editing-aware context data, we use the correct editing operations during the training stage and use the noisy editing operations generated by the first stage during the prediction stage.

The exposure bias caused by the inconsistency in the training and inference stages can affect the effectiveness of the model, leading to overfitting of correct editing operations. Here we adopt three editing operation perturbation methods to alleviate exposure bias. Specifically, we add three types of perturbations to the editing operations during training to improve the robustness of the model, namely random replacement, random deletion and random insertion.

The probability of perturbations is denoted by $p r o b _ { p }$ . For gold editing operations, the probability of randomly replacing the span of the operation is prob<sub>r</sub>, while the probability of randomly deleting it is 1 − prob<sub>r</sub>. For each editing operation e, we first sample two probability values prob<sub>1</sub> and prob<sub>2</sub> from a uniform distribution. If $p r o b _ { 1 } \leq p r o b _ { p } ,$ we perform random deletion or random replacement as follows: if $p r o b _ { 2 } \ \leq \ p r o b _ { r }$ , we randomly replace the text (corresponding span) with a random span sampled from dialogue (Line 6-9); otherwise, we delete this editing operation (i.e., do not insert into $E _ { p } )$ . Taking the editing operation “[D] he [R] Ben Affleck” as an example, we use “Batman” to replace “Ben $A f f e c k ^ { \prime \prime }$ and form a new editing operation “[D] he [R] Batman”. Then, we randomly sample a probability value prob<sub>3</sub> from a uniform distribution. if $p r o b _ { 3 } \le p r o b _ { p }$ , we perform a random insertion (Line 15-20), including an insertion or a replacement operation. Taking the Figure 1 as an example, we insert an insertion operation “[I] Batman” or a replacement operation “[D] He [R] Batman” to $E _ { p }$ , resulting in an adversarial sample.

Algorithm 1: Perturbation Strategy   
input :Editing operations T; Perturbation   
probability prob<sub>p</sub>; Span replacement   
probability prob<sub>r</sub>   
output :Perturbed editing operations $E _ { p }$   
1 $E _ { p }  \{ \} ;$   
2 foreach e in $\tau$ do   
3 Draw prob<sub>1</sub>, prob<sub>2</sub> from Uniform(0, 1);   
4 // Add noise to editing operations with   
a probability of prob<sub>p</sub>   
5 if prob<sub>1</sub> $< = p r o b _ { p }$ then   
6 if prob<sub>2</sub> $< = p r o b _ { r }$ then   
7 e.text ← A random span sampled   
from the dialogue history;   
8 $E _ { p }  E _ { p } \cup \{ e \}$   
9 end   
10 else   
11 $E _ { p } \gets E _ { p } \cup \{ e \}$   
12 end   
13 end   
14 Draw $p r o b _ { 3 }$ from Uniform(0, 1);   
15 if prob $< = p r o b _ { p }$ then   
16 origin\_span $ \mathbf { A }$ random span sampled from the   
incomplete utterance;   
17 new\_span ← A random span sampled from the   
dialogue history;   
18 candidates=[“[I] new\_span”, “[D] origin\_span   
[R] new\_span”];   
19 $E _ { p } \gets E _ { p } \cup$ {the operation randomly sampled   
from the candidates}   
20 end   
21 return $E _ { p }$

We also use BART as the generation model and concatenate the history $H i s t .$ , the incomplete utterance $U _ { n }$ and the editing operations E as input, i.e., {[CLS]Hist[SEP]U<sub>n</sub>[SEP]E[SEP]}, where [CLS] and [SEP] represent the special tokens in BART. The output is the rewritten utterance Y = $\{ y _ { 1 } , . . . y _ { i } , . . . y _ { m } \}$ where $y _ { i }$ is the i-th token. The training objective of the editing-aware rewritten utterance generation is to minimize the negative log-likelihood loss function $L o s s _ { u t t r }$ as follows,

<table><tr><td>Model</td><td>EM</td><td> ${ \bf B } _ { 4 }$ </td><td> $\mathrm { F _ { 1 } }$ </td></tr><tr><td>BARTbase (Lewis et al., 2020)</td><td>70.1</td><td>83.9</td><td>69.5</td></tr><tr><td>QUEEN (Si et al., 2022)</td><td>71.6</td><td>86.3</td><td>NA</td></tr><tr><td>SGT (Chen, 2023)</td><td>71.1</td><td>86.7</td><td>85.0</td></tr><tr><td>MIUR (Li et al., 2023a)</td><td>70.9</td><td>86.0</td><td>72.3</td></tr><tr><td>Locate-Fill (Li et al., 2023b)</td><td>75.0</td><td>87.3</td><td>84.2</td></tr><tr><td>XSS (Peng et al., 2024)</td><td>70.2</td><td>85.6</td><td>70.4</td></tr><tr><td>TEO-Stage1</td><td>52.7</td><td>75.7</td><td>62.5</td></tr><tr><td>TEO-RFIS</td><td>77.7</td><td>88.5</td><td>85.3</td></tr><tr><td>TEO-T5</td><td>76.4</td><td>87.6</td><td>84.5</td></tr><tr><td>TEO (Ours)</td><td>78.1</td><td>88.7</td><td>85.8</td></tr><tr><td>TEO-Gold</td><td>83.5</td><td>91.5</td><td>90.9</td></tr></table>

Table 2: Result comparison on English TASK.

$$
L o s s _ { u t t r } = - \log \sum _ { i = 1 } ^ { m } P ( y _ { i } | y _ { < i } , H i s t , U _ { n } , \mathcal { E } , \theta _ { 2 } ) ,\tag{4}
$$

where $\theta _ { 2 }$ is the parameters of the model in the second stage. We initialize the model using the weights of the model trained in the first stage.

## 4 Experimentation

## 4.1 Experimental Settings

Datasets We conduct experiments on three popular IUR datasets: the Chinese open-domain dialogue datasets REWRITE (Su et al., 2019) and RESTORATION-200K (abbreviated as RES200K) (Pan et al., 2019), and the English task-oriented dataset TASK (Quan et al., 2019). Specific statistics for the datasets are provided in Appendix A.

Metrics In order to evaluate our method, we employ four different automatic evaluation metrics, namely EM, B $\mathbf { \mathcal { L } U } _ { n }$ (abbreviated as $\mathbf { B } _ { n } )$ (Papineni et al., 2002), $\mathsf { R O U G E } _ { n } \left( \mathbb { R } _ { n } \right)$ (Lin, 2004), and Restoration ${ \mathrm { F } } { \mathrm { - s c o r e } } _ { n } \left( \mathrm { F } _ { n } \right)$ (Pan et al., 2019). These metrics can effectively reflect the quality of rewriting. Following previous work (Liu et al., 2020; Si et al., 2022; Li et al., 2023a), we also report different metrics on different datasets.

Baselines We compare our TEO with the following strong baselines: $\mathbf { B A R T _ { \mathrm { b a s e } } }$ (Lewis et al., 2020), QUEEN (Si et al., 2022), RAU (Zhang et al., 2022), SGT (Chen, 2023), MIUR (Li et al., 2023a), Locate-Fill (Li et al., 2023b), MGIIF (Du et al., 2023) and XSS (Peng et al., 2024).

The implementation and details of metrics and baselines are provided in Appendix B, C and D.

## 4.2 Experimental Results

We evaluate our TEO and the baselines on the three datasets and the results are shown in Tables 2, 3 and $^ { 4 , }$ respectively, where “NA” refers to the metrics that the original paper did not report. The results on three datasets show that our TEO outperforms the baselines on almost all metrics. In terms of the most rigorous metric, EM, our TEO achieves an improvement of 3.1, 2.6 and 0.6 on the TASK, REWRITE and RES200K datasets, respectively, in comparison with the previous SOTA models. This result demonstrates that the robustness of our TEO allows it to avoid various minor boundary errors. Furthermore, the improvements observed on the English TASK and Chinese REWRITE and RES200K datasets demonstrate the applicability of our TEO to diverse languages and domains.

Furthermore, it is observed that on the three datasets, our TEO exhibits superior performance compared to the baselines in $\mathrm { F } _ { n }$ , while it is almost on par with them in $\mathrm { B L E U } _ { n }$ . This can be attributed to the fact that $\mathrm { F } _ { n }$ emphasizes the ability of the model to identify omitted or referential information in the incomplete utterance, whereas $\mathrm { B L E U } _ { n }$ considers all n-gram information present in the utterance including the span that is already present in the original incomplete utterance. The higher performance in $\mathrm { F } _ { n }$ indicates that our TEO can better capture contextual information of conversations and subsequently address the issues of coreference and ellipsis based on the context. Moreover, we find that our TEO achieves the improvements of 0.8, 2.2 and 2.6 in $\mathrm { F _ { 1 } , F _ { 2 } } .$ and $\mathrm { F _ { 3 } }$ scores, respectively, in comparison with the previous SOTA model MIUR on the RES200K dataset. In essence, the $\mathrm { F _ { 1 } }$ score is indicative of the accuracy of token restoration, irrespective of the insertion position within the utterance. However, the positioning of newly added tokens will influence the values of $\mathrm { F _ { 2 } }$ and $\mathrm { F _ { 3 } }$ . The enhanced performance observed in $\mathrm { F _ { 2 } }$ and $\mathrm { F _ { 3 } }$ also suggests that our TEO is capable of not only generating missing tokens but also inserting them at the appropriate locations.

To ensure a fair comparison, we maintained consistent experimental settings with previous related studies and utilized $\mathbf { B A R T _ { \mathrm { b a s e } } }$ as the backbone. Additionally, we conducted experiments on three datasets using $\mathrm { T } 5 _ { \mathrm { b a s e } }$ as the backbone and the results of these experiments are presented in Tables 2, 3 and 4, as TEO-T5, respectively. It is worth noting that $\mathbf { B A R T _ { \mathrm { b a s e } } }$ outperformed $\mathrm { T } 5 _ { \mathrm { b a s e } }$ in most metrics, which may be attributed to the different pre-training objectives of the two models. During pre-training, BART introduces noise to the text and reconstructs the original text at the decoder. In contrast, T5 models various classification and generation tasks in a unified text-to-text format during pre-training. BART’s pre-training objective is similar to our IUR task because coreference and ellipsis in IUR can be viewed as a type of noise that our task aims to recover.

<table><tr><td>Model</td><td> $\mathrm { F _ { 1 } }$ </td><td> $\mathrm { F _ { 2 } }$ </td><td> $\mathrm { F _ { 3 } }$ </td><td> ${ \bf B } _ { 1 }$ </td><td> ${ \bf B } _ { 2 }$ </td><td> ${ \bf R } _ { 1 }$ </td><td> $\mathrm { R _ { 2 } }$ </td><td> $\mathrm { R } _ { L }$ </td><td>EM</td></tr><tr><td> $\mathbf { B A R T _ { b a s e } }$  (Lewis et al., 2020)</td><td>81.2</td><td>76.0</td><td>79.7</td><td>93.9</td><td>90.8</td><td>95.2</td><td>91.8</td><td>92.4</td><td>70.5</td></tr><tr><td>RAU (Zhang et al., 2022)</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>91.6</td><td>NA</td><td>90.6</td><td>93.9</td><td>68.4</td></tr><tr><td>QUEEN (Si et al., 2022)</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>92.1</td><td>NA</td><td>90.9</td><td>94.6</td><td>70.1</td></tr><tr><td>MIUR (Li et al., 2023a)</td><td>NA</td><td>82.2</td><td>NA</td><td>NA</td><td>91.2</td><td>NA</td><td>90.7</td><td>93.7</td><td>67.7</td></tr><tr><td>Locate-Fill (Li et al., 2023b)</td><td>89.9</td><td>83.9</td><td>79.4</td><td>93.8</td><td>91.8</td><td>95.9</td><td>91.6</td><td>94.0</td><td>70.9</td></tr><tr><td>XSS (Peng et al., 2024)</td><td>89.8</td><td>82.0</td><td>76.1</td><td>92.4</td><td>91.0</td><td>95.8</td><td>90.7</td><td>93.7</td><td>66.7</td></tr><tr><td>TEO-Stage1</td><td>85.6</td><td>63.0</td><td>28.3</td><td>90.9</td><td>82.6</td><td>94.0</td><td>75.9</td><td>75.9</td><td>34.0</td></tr><tr><td>TEO-RFIS</td><td>90.0</td><td>83.9</td><td>80.9</td><td>93.3</td><td>91.2</td><td>95.9</td><td>91.6</td><td>94.2</td><td>72.1</td></tr><tr><td>TEO-T5</td><td>91.2</td><td>85.9</td><td>81.1</td><td>94.1</td><td>92.2</td><td>96.2</td><td>92.0</td><td>94.1</td><td>72.6</td></tr><tr><td>TEO (Ours)</td><td>91.0</td><td>85.4</td><td>82.1</td><td>94.4</td><td>92.5</td><td>96.3</td><td>92.3</td><td>94.7</td><td>73.5</td></tr><tr><td>TEO-Gold</td><td>98.1</td><td>94.3</td><td>91.2</td><td>98.4</td><td>97.3</td><td>99.1</td><td>96.7</td><td>97.5</td><td>86.3</td></tr></table>

Table 3: Result comparison on Chinese REWRITE.
<table><tr><td>Model</td><td> $\mathrm { P _ { 1 } }$ </td><td> ${ \bf R } _ { 1 }$ </td><td> $\mathrm { F _ { 1 } }$ </td><td> $\mathrm { P _ { 2 } }$ </td><td> ${ \bf R } _ { 2 }$ </td><td> $\mathrm { F _ { 2 } }$ </td><td> $\mathrm { P _ { 3 } }$ </td><td> ${ \mathrm { R } } _ { 3 }$ </td><td> $\mathrm { F _ { 3 } }$ </td><td> ${ \bf B } _ { 1 }$ </td><td> ${ \bf B } _ { 2 }$ </td><td> $\mathrm { R } _ { 1 }$ </td><td> $\mathrm { R _ { 2 } }$ </td><td>EM</td></tr><tr><td>BART (Lewis et al., 2020)  $\mathbf { b a s e }$ </td><td>70.9</td><td>55.8</td><td>62.4</td><td>60.8</td><td>47.4</td><td>53.3</td><td>54.0</td><td>41.8</td><td>47.1</td><td>90.5</td><td>87.9</td><td>91.8</td><td>85.5</td><td>52.9</td></tr><tr><td>RAU (Zhang et al., 2022)</td><td>75.0</td><td>65.5</td><td>69.9</td><td>61.2</td><td>54.3</td><td>57.5</td><td>52.5</td><td>47.0</td><td>49.6</td><td>92.4</td><td>89.6</td><td>92.8</td><td>86.0</td><td>NA</td></tr><tr><td>QUEEN (Si et al., 2022)</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>92.4</td><td>89.8</td><td>92.5</td><td>86.3</td><td>53.5</td></tr><tr><td>MGIIF (Du et al., 2023)</td><td>NA</td><td>NA</td><td>70.8</td><td>NA</td><td>NA</td><td>58.5</td><td>NA</td><td>NA</td><td>50.5</td><td>93.1</td><td>90.4</td><td>93.2</td><td>86.6</td><td>NA</td></tr><tr><td>MIUR (Li et al., 2023a)</td><td>76.4</td><td>63.7</td><td>69.5</td><td>62.7</td><td>52.7</td><td>57.3</td><td>54.3</td><td>45.9</td><td>49.7</td><td>93.0</td><td>90.1</td><td>92.6</td><td>85.7</td><td>51.0</td></tr><tr><td>Locate-Fill (Li et al., 2023b)</td><td>73.1</td><td>61.9</td><td>67.0</td><td>62.6</td><td>52.4</td><td>57.0</td><td>55.4</td><td>46.0</td><td>50.2</td><td>92.5</td><td>89.9</td><td>92.5</td><td>86.3</td><td>53.6</td></tr><tr><td>XSS (Peng et al., 2024)</td><td>NA</td><td>NA</td><td>70.9</td><td>NA</td><td>NA</td><td>57.0</td><td>NA</td><td>NA</td><td>47.9</td><td>92.5</td><td>89.7</td><td>92.7</td><td>85.9</td><td>50.1</td></tr><tr><td>TEO-Stage1</td><td>73.8</td><td>65.0</td><td>69.1</td><td>42.7</td><td>39.5</td><td>41.0</td><td>24.6</td><td>23.0</td><td>23.7</td><td>92.9</td><td>86.8</td><td>92.6</td><td>78.6</td><td>42.5</td></tr><tr><td>TEO-RFIS</td><td>74.2</td><td>66.6</td><td>70.2</td><td>62.7</td><td>55.9</td><td>59.1</td><td>55.0</td><td>48.9</td><td>51.8</td><td>92.5</td><td>89.7</td><td>92.7</td><td>86.2</td><td>53.3</td></tr><tr><td>TEO-T5</td><td>76.7</td><td>65.6</td><td>70.7</td><td>61.4</td><td>55.1</td><td>58.1</td><td>53.8</td><td>48.3</td><td>50.9</td><td>91.3</td><td>88.7</td><td>93.6</td><td>87.9</td><td>53.6</td></tr><tr><td>TEO (Ours)</td><td>74.4</td><td>66.7</td><td>70.3</td><td>63.4</td><td>56.1</td><td>59.5</td><td>55.9</td><td>49.1</td><td>52.3</td><td>92.8</td><td>90.1</td><td>92.8</td><td>86.5</td><td>54.2</td></tr><tr><td>TEO-Gold</td><td>93.5</td><td>94.3</td><td>93.9</td><td>83.9</td><td>84.1</td><td>84.0</td><td>76.8</td><td>76.9</td><td>76.9</td><td>97.6</td><td>95.6</td><td>98.0</td><td>93.6</td><td>80.3</td></tr></table>

Table 4: Result comparison on Chinese RES200K.

## 4.3 Analysis on Editing Operation Generation

As shown in Table 5, we calculate the EM metric of the editing operations generated in the first stage, i.e., editing operation generation. It can be observed that the EM metric is also relatively high. For example, in the TASK and REWRITE datasets, the EM metrics of the first stage are 73.1 and 75.2, respectively, while the metrics for the second stage are 78.1 and 73.5, respectively. It can be observed that the EM metric is higher in the second stage of the TASK dataset in comparison with the first stage. This further corroborates the hypothesis that even if erroneous editing operations are generated in the first stage, a portion of them can be rectified

in the second stage.

To further validate the effectiveness of the first stage, we use the correct editing operations during inference in the second stage, shown in Tables 2, 3 and 4, as TEO-Gold, respectively. The results show a significant improvement in all metrics for all three datasets after using the correct editing operations. This indicates that the performance of editing operation generation is positively correlated with that of the second stage. Simultaneously, the performance of directly using $\mathrm { B A R T _ { b a s e } , }$ i.e., removing the first stage, is much lower than our TEO, which also proves the effectiveness of the first stage and illustrates that the editing operations are the pivot for the IUR task.

## 4.4 Analysis on Editing-aware Rewritten Utterance Generation

To verify the effectiveness of the second stage, i.e., editing-aware rewritten utterance generation, it is not necessary to proceed to the second stage after completing the first stage. Instead, the rewritten utterance can be generated directly according to the following rules: For those replacement operations, we directly parse the editing operations generated in the first stage. For insertion operations, since it is not possible to determine the exact positions at which the insertion should occur, a random selection of positions is made. The results are shown in Tables 2, 3 and 4, as TEO-Stage1 respectively. It can be observed that the outcomes of the first stage are considerably inferior to those of the twostage method TEO. This evidence corroborates the efficacy of the second stage and the beneficial interaction between the first and second stages.

<table><tr><td>Dataset</td><td>EM</td><td>E2C(%)</td><td>C2E(%)</td></tr><tr><td>TASK</td><td>73.1</td><td>9.71</td><td>20.56</td></tr><tr><td>REWRITE</td><td>75.2</td><td>12.53</td><td>3.41</td></tr><tr><td>RES200K</td><td>59.8</td><td>4.82</td><td>29.66</td></tr></table>

Table 5: The EM, E2C and C2E of the first stage.

In addition, we define two metrics, error to correct rate (E2C) and correct to error rate (C2E),

$$
E 2 C = \frac { \# e r r _ { - } c o r } { \# e r } , C 2 E = \frac { \# c o r \_ e r r } { \# c o r } ,\tag{5}
$$

where #err\_cor refers to the number of samples that were predicted incorrectly in the first stage but correctly in the second stage, #er refers to the number of samples that were predicted incorrectly in the first stage, #cor\_err refers to the number of samples that were predicted correctly in the first stage but incorrectly in the second stage, and #cor refers to the number of samples that were predicted correctly in the first stage. The first metric measures the proportion of samples that were incorrectly predicted in the first stage but correctly predicted in the second stage. The second metric measures the proportion of samples that were correctly predicted in the first stage but incorrectly predicted in the second stage. With these two metrics, we can quantitatively analyze the correlation between the two stages.

As shown in Table 5, we find that E2C is relatively higher than C2E on the REWRITE dataset. This indicates that a higher proportion of incorrect editing operations in the first stage are corrected in the second stage, thereby proving that the combination of global dialogue context information with local editing operation information is effective. Through our observations, we find that there are far more cases of ellipsis than the cases of coreference in RES200K and TASK. This implies that there are more instances of insertion. In the first stage, we only obtain the inserted tokens but do not know its positions. Therefore, even if we correctly predict the inserted tokens in the first stage, there are still many cases of incorrect insertion positions in the second stage. Consequently, the value of C2E is higher in RES200K and TASK.

<table><tr><td rowspan=1 colspan=1>Context:Speaker1: Don&#x27;t you have any musical instruments that youwant to learn? I think the piano and guitar sound great.Speaker2: Piano.Speaker1: The sound of this musical instrument sounds verypleasant, wow. (Incomplete utterance)</td></tr><tr><td rowspan=1 colspan=1>Reference: The sound of the piano sounds very pleasant,WOW.</td></tr><tr><td rowspan=1 colspan=1>Correct editing operation:[D] this musical instrument [R] the piano</td></tr><tr><td rowspan=1 colspan=1>Predicted editing operation in the first stage:[D] this [R] the piano X</td></tr><tr><td rowspan=1 colspan=1>Predicted rewritten utterance in the second stage:The sound of the piano sounds very pleasant, wow. √</td></tr></table>

Table 6: A boundary error in the first stage is corrected by the second stage.  
![](images/247fd18569331cb45253979a5ddda7b795ba071e67ac62f45128248d6941fec7.jpg)  
Figure 2: The correspondence between the positive and negative examples in the first stage and the second stage, where “wrong” and “right” on the vertical axis respectively represent incorrect and correct editing operations in the first stage, while “wrong” and “right” on the horizontal axis respectively represent incorrect and correct rewritten utterances in the second stage.

Since the first stage can solve the issue of coreference, the replacement operations can be used to rewrite utterances. Instead of that, we feed both the replacement and insertion operations to the second stage. To conduct insightful analysis, we perform predictive replacement operations after the first stage and only handle insertion operations in the second stage. The results are shown in Tables 2, 3, 4 as TEO-RFIS, respectively. This approach yields inferior results across all metrics compared to simultaneously handling replacement and insertion operations in the second stage. Taking table 6 as example, the first stage generates an incorrect replacement operation, predicting the replacement of “this musical instrument” with “this”. If the replacement is executed directly, an incorrect output is produced. However, the second stage is able to correct this error and produce the correct utterance.

Furthermore, we also investigated the results predicted in the second stage for the samples corresponding to correct or incorrect editing operations predicted in the first stage. The results are shown in Figure 2. The majority of samples with accurate editing operation predictions in the first stage were correctly identified in the second stage, while some of the incorrectly predicted samples were mitigated in the second stage. We find that 8.0%, 5.1% and 1.4% of samples are corrected by the second stage on TASK, REWRITE, and RES200K respectively. However, these samples did not have correct editing operations in the first stage. This suggests that our second stage editing-aware rewritten utterance generation is capable of correcting the errors produced in the first stage.

![](images/15f0b5940f1c1d1dae40b326f561eff2244ce52e4b343d45749d87fc2e80ef69.jpg)  
Figure 3: The trend chart of EM metric changing with the variation of adversarial perturbation probability on REWRITE.

It is worth noting that the edit operations predicted in the first stage can be used in the training of the second stage and our experimental results show the values of $\mathrm { F _ { 1 } }$ and EM were 89.7 and 72.3, respectively, which are inferior to our TEO (91.2 and 73.5). This is because it would result in a fixed distribution of erroneous editing operations in the training data of the second stage. Nevertheless, the utilization of our proposed adversarial perturbation strategy enables the dynamic adjustment of noise within training samples, thereby enhancing the model’s robustness.

## 4.5 Impact of Adversarial Perturbation

As mentioned in Section 3.4, the introduction of adversarial perturbations is employed to mitigate the impact of exposure bias, which arises due to inconsistency between the training and prediction stages. To assess the efficacy of adversarial perturbations, experiments were conducted with varying perturbation probabilities, and the experimental results are presented in Figure 3. As the probability of perturbation increases, the EM value initially increases and then decreases, reaching its peak when the probability is 0.6. This may be due to the inconsistency between the training and inference data distributions when the perturbation probability is low. When the perturbation probability is too high, TEO fails to capture knowledge in editing operations. However, we observe that even when the perturbation probability is 1, TEO can still achieve good results. This also indicates that in situations that are close to zero-shot, our TEO can still perform well.

![](images/768e7ee918489f73057564ed7c55b6b6f74545eb41609214c355cb44affed559.jpg)  
Figure 4: Performance comparison of our method with $\mathrm { B A R T _ { b a s e } , }$ Locate-Fill and XSS on human evaluation.

## 4.6 Human Evaluation

Incomplete utterance rewriting often results in nonunique rewriting outcomes, making it challenging to fully assess the performance of the model based on automatic evaluation metrics alone. To address this, we conducted a human evaluation to compare our proposed method with $\mathbf { B A R T _ { b a s e } } ,$ , Locate-Fill and XSS. A total of fifty data points were randomly sampled from the test set and distributed to three raters. Each rater selected a better result from the outputs generated by two methods.

The results are presented in Figure 4, it can be seen that our method outperforms $\mathbf { B A R T _ { b a s e } } ,$ Locate-Fill and XSS, indicating consistency with the results observed in our automatic evaluation. However, it was observed that our method has a smaller advantage when evaluated manually against $\mathbf { B A R T _ { b a s e } } .$ , compared to Locate-Fill and XSS. After analyzing the rewritten utterances produced by these four methods, it was discovered that while $\mathbf { B A R T _ { \mathrm { b a s e } } }$ demonstrated an average ability to rewrite, it was capable of generating more fluent utterances, which ultimately led to superior human evaluation results.

## 4.7 Comparison with ChatGPT

At present, the majority of LLMs lack the capacity to process coreference and ellipsis resolution, which are integral to IUR. We conducted preliminary experiments utilising the in-context learning method with GPT-4 (the version used here is gpt-4-1106-preview) as a case study and the prompt used in GPT-4 is provided in Appendix E.

<table><tr><td>Model</td><td> $\mathrm { F _ { 1 } }$ </td><td> $\mathrm { F _ { 2 } }$ </td><td> ${ \bf B } _ { 1 }$ </td><td> ${ \bf B } _ { 2 }$ </td><td> ${ \bf R } _ { 1 }$ </td><td> $\mathrm { R _ { 2 } }$ </td><td> $\mathrm { R } _ { L }$ </td><td>EM</td></tr><tr><td>GPT-4</td><td>79.1</td><td>66.7</td><td>85.3</td><td>80.4</td><td>87.7</td><td>77.6</td><td>82.1</td><td>35.7</td></tr><tr><td>Ours</td><td>91.0</td><td>85.4</td><td>94.4</td><td>92.5</td><td>96.3</td><td>92.3</td><td>94.7</td><td>73.5</td></tr></table>

Table 7: Performance comparison between TEO and GPT-4 on REWRITE.

We provided five samples as demonstrations to GPT-4 and the experimental results on REWRITE are shown in Table 7. Even though the number of parameters in GPT-4 is much higher than our model, our model still achieves better performance than it. In addition, we observed that some utterances generated by GPT-4 do not match incomplete utterances in terms of semantics, which also indicates that the understanding of global conversational semantics by LLMs needs to be improved.

## 4.8 Case Study

Our TEO is capable of accommodating both instances where the rewritten utterance contains tokens that are not present in the dialogue history, as well as instances where multiple discontinuous spans must be inserted at the same position in the current utterance. We compare our TEO with three baselines, i.e., $\mathbf { B A R T _ { \mathrm { b a s e } } }$ , RUN, and HCT, and the results on the Chinese REWRITE dataset are shown in Table 8. We need to insert four spans <sub>“</sub>天龙八部里<sub>”</sub> <sub>(in</sub> <sub>Demi-Gods</sub> <sub>and</sub> <sub>Semi-Devils),</sub> <sub>“</sub>段誉<sub>” (Duan Yu), “</sub>的<sub>” (of), and “</sub>武功最高<sub>” (the</sub> highest martial arts skill) at the end of the current utterance. Although $\mathbf { B A R T _ { \mathrm { b a s e } } }$ generates a more fluent rewritten utterance, it does not fit into the context of the conversation. RUN selects all correct spans but fails to insert them in the correct order within the utterance (Although the English translation is coherent, there is an word order error in Chinese sentences). HCT has boundary errors for some spans; for example, instead of inserting “<sup>段誉</sup>” (Duan Yu), it inserts “<sup>是段誉</sup>” (is Duan Yu). Additionally, there is also an issue with span insertion order. Only our TEO outputs correct and complete utterance, which benefits from the complementary effects of the two-stage mechanism.

## 4.9 Error Analysis

To analyse the errors in our TEO model, we compiled the experimental results and found that the majority of errors arise from the insertion operation rather than the replacement operation. For instance, in REWRITE, insertion errors account for 79.3%, while replacement errors account for only 20.7%. This outcome is mainly due to the uncertainty of the insertion position. Secondly, in contrast to REWRITE, RES200K and TASK contain utterances that do not require rewriting. This can lead to errors when editing these utterances. For example, in RES200K, 38.7% of utterances do not require rewriting and have an EM score of 77.5, with almost all errors resulting from incorrect replacement operations. Finally, the EM scores for the replacement operation are considerably lower for RES200K (0) and TASK (28.6) than for REWRITE (83.0). This is primarily due to the limited number of replacement operations in these two datasets (RES200K: 0.15%; TASK: 12.42%).

<table><tr><td rowspan=1 colspan=1>Context:A: 天龙八部里谁的武功最高(Who has the highest mar-tial arts skill in &quot;Demi-Gods and Semi-Devils&quot;?)B: 是段誉(It&#x27;s Duan Yu.)A: 为什么(Why?) (Incomplete utterance)</td></tr><tr><td rowspan=1 colspan=1>Reference: 为什么天龙八部里段誉的 武功最高(Whyis Duan Yu&#x27;s martial arts the highest in ‘Demi-Gods and Semi-Devils&quot;?)</td></tr><tr><td rowspan=1 colspan=1> $\mathbf { B A R T } _ { \mathrm { b a s e } } .$ 为什么天龙八部里谁的武功最高(Why iswhose martial arts the highest in &quot;Demi-Gods and Semi-Devils&quot;?) X</td></tr><tr><td rowspan=1 colspan=1>RUN: 为什么天龙八部里的武功最高段誉(Why is themartial arts in “Demi-Gods and Semi-Devils&quot; the highest,Duan Yu?) X</td></tr><tr><td rowspan=1 colspan=1>HCT: 为什么天龙八部里谁武功最高是段誉(Why is DuanYu whose martial arts is the highest in “Demi-Gods and Semi-Devils&quot;?) X</td></tr><tr><td rowspan=1 colspan=1>TEO (Ours): 为什么天龙八部里段誉的武功最高(Why isDuan Yu&#x27;s martial arts the highest in “Demi-Gods and Semi-Devils&quot;?) √</td></tr></table>

Table 8: A case of our TEO and three baselines $\mathrm { B A R T _ { b a s e } , }$ RUN and HCT on REWRITE.

## 5 Conclusion

We propose a two-stage IUR framework by taking the editing operations as the pivot, in which the first stage generates editing operations for IUR and the second stage rewrites incomplete utterances utilizing the generated editing operations and the dialogue context. Moreover, an adversarial perturbation strategy is proposed to enhance model robustness. The experimental results on three IUR datasets show that our TEO outperforms the SOTA models significantly. Our future work will focus on how to introduce LLMs to assist IUR.

## Limitations

Although this paper may contribute to incomplete utterance rewriting and some downstream dialogue tasks, it still suffers from two shortcomings, which are our future work. First, we only use one representation of the editing operation in our model. We believe that better templates can help the model better understand and improve the effectiveness of dialogue rewriting. Second, we only used the editing operations generated in the first stage to assist in rewriting the dialogue utterances in the second stage, but did not attempt to use the dialogue rewriting of the second stage to facilitate the first stage. Therefore, how to promote the complementary interaction between the two stages is our future research.

## Acknowledgements

The authors would like to thank the three anonymous reviewers for their comments on this paper. This research was supported by the National Natural Science Foundation of China (Nos. 62376181 and 62276177), and Project Funded by the Priority Academic Program Development of Jiangsu Higher Education Institutions.

## References

Yunshan Chen. 2023. Incomplete utterance rewriting as sequential greedy tagging. In Findings of the Association for Computational Linguistics: ACL 2023, pages 7265–7276.

Haowei Du, Dinghao Zhang, Chen Li, Yang Li, and Dongyan Zhao. 2023. Multi-granularity information interaction framework for incomplete utterance rewriting. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 2576– 2581.

Yue Fang, Hainan Zhang, Hongshen Chen, Zhuoye Ding, Bo Long, Yanyan Lan, and Yanquan Zhou. 2022. From spoken dialogue to formal summary: An utterance rewriting for dialogue summarization. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3859–3869.

Jie Hao, Linfeng Song, Liwei Wang, Kun Xu, Zhaopeng Tu, and Dong Yu. 2021. RAST: Domain-robust dialogue rewriting as sequence tagging. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4913–4924.

Mengzuo Huang, Feng Li, Wuhe Zou, and Weidong Zhang. 2021. SARG: A novel semi autoregressive

generator for multi-turn incomplete utterance restoration. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications ofArtificial Intelligence, IAAI 2021, pages 13055–13063.

Shumpei Inoue, Tsungwei Liu, Son Nguyen, and Minh-Tien Nguyen. 2022. Enhance incomplete utterance restoration by joint learning token extraction and text generation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3149–3158.

Lisa Jin, Linfeng Song, Lifeng Jin, Dong Yu, and Daniel Gildea. 2022. Hierarchical context tagging for utterance rewriting. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, pages 10849–10857.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 7871–7880.

Jiang Li, Xiangdong Su, Xinlan Ma, and Guanglai Gao. 2023a. How well apply simple MLP to incomplete utterance rewriting? In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 1567– 1576.

Zitong Li, Jiawei Li, Haifeng Tang, Kenny Zhu, and Ruolan Yang. 2023b. Incomplete utterance rewriting by a two-phase locate-and-fill regime. In Findings of the Associationfor Computational Linguistics: ACL 2023, pages 2731–2745.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81.

Qian Liu, Bei Chen, Jian-Guang Lou, Bin Zhou, and Dongmei Zhang. 2020. Incomplete utterance rewriting as semantic segmentation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 2846–2857.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA.

Zhufeng Pan, Kun Bai, Yan Wang, Lianqiang Zhou, and Xiaojiang Liu. 2019. Improving open-domain dialogue systems via multi-turn incomplete utterance restoration. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 1824–1833.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318.

Letian Peng, Zuchao Li, and Hai Zhao. 2024. Fast and accurate incomplete utterance rewriting. IEEE/ACM Transactions on Audio, Speech, and Language Processing, pages 1–11.

Hongjin Qian and Zhicheng Dou. 2022. Explicit query rewriting for conversational dense retrieval. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 4725– 4737.

Jun Quan, Deyi Xiong, Bonnie Webber, and Changjian Hu. 2019. GECOR: An end-to-end generative ellipsis and co-reference resolution model for taskoriented dialogue. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 4547– 4557.

Shuzheng Si, Shuang Zeng, and Baobao Chang. 2022. Mining clues from incomplete utterance: A queryenhanced network for incomplete utterance rewriting. In Proceedings ofthe 2022 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4839–4847.

Hui Su, Xiaoyu Shen, Rongzhi Zhang, Fei Sun, Pengwei Hu, Cheng Niu, and Jie Zhou. 2019. Improving multi-turn dialogue modelling with utterance ReWriter. In Proceedings ofthe 57th Annual Meeting of the Association for Computational Linguistics, pages 22–31.

Yong Zhang, Zhitao Li, Jianzong Wang, Ning Cheng, and Jing Xiao. 2022. Self-attention for incomplete utterance rewriting. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2022, Virtual and Singapore, pages 8047–8051.

## A Dataset Statistics

The specific information of the three datasets is shown in Table 9.

## B Implementation

The pre-trained $\mathrm { B A R T _ { b a s e } }$ is employed as the backbone model, with all experiments conducted using the open-source library PyTorch. To reduce the latency during inference, a greedy decoding strategy is employed. Both the first stage and the second stage are fine-tuned for 30 epochs. The AdamW optimiser (Loshchilov and Hutter, 2019) is used, with a learning rate of 5e-5. In the second stage, the probabilities of random deletion and random replacement are both set to 0.5. Given that the length of the edit operation is considerably shorter than that of the rewritten utterance, the decoding time of the first stage is negligible in comparison to that of the second stage.

<table><tr><td>Category</td><td>REWRITE</td><td>RES200K</td><td>TASK</td></tr><tr><td>Language</td><td>Chinese</td><td>Chinese</td><td>English</td></tr><tr><td>Train</td><td>18K</td><td>194K</td><td>2.2K</td></tr><tr><td>Dev</td><td>2K</td><td>5K</td><td>0.5K</td></tr><tr><td>Test</td><td>2K</td><td>5K</td><td>0.5K</td></tr><tr><td>#Avg. Cont</td><td>17.7</td><td>25.5</td><td>52.6</td></tr><tr><td>#Avg. Curr</td><td>6.5</td><td>8.6</td><td>9.4</td></tr><tr><td>#Avg. Rewr</td><td>10.5</td><td>12.4</td><td>11.3</td></tr><tr><td>#Insertion</td><td>14070</td><td>136339</td><td>1572</td></tr><tr><td>#Replacement</td><td>7853</td><td>203</td><td>223</td></tr></table>

Table 9: Statistics of different datasets, where “Cont”, “Curr” and “Rewr” are the abbreviations for the context, current, and rewritten utterance, respectively.

## C Details of Metrics

EM refers to exact matching accuracy, which is a strict metric, representing the ratio of correctly predicted samples to the total number of samples. The BLEU metric evaluates accuracy by calculating the matching degree of n-grams. BLEU<sub>1</sub> is a metric that measures the accuracy at the word level, while higher-order BLEU can be used to assess the fluency of utterances. Additionally, we employ $\mathrm { R O U G E } _ { n }$ to measure recall in IUR. ROUGE evaluates recall by counting n-gram co-occurrences. $\mathrm { F } _ { n }$ (Pan et al., 2019) is utilized to identify the words that have been added to the utterance for rewriting. The n-gram restoration precision, recall and F-score are calculated as follows,

$$
\begin{array} { l } { p _ { n } = \frac { \displaystyle \big | \left\{ \mathrm { \ r e s t o r e d ~ n - g r a m s ~ } \right\} \cap \left\{ \mathrm { \ n - g r a m s ~ i n ~ r e f ~ } \right\} \big | } { \displaystyle \big | \left\{ \mathrm { \ r e s t o r e d ~ n - g r a m s ~ } \right\} \big | } } \\ { r _ { n } = \frac { \displaystyle \big | \left\{ \mathrm { \ r e s t o r e d ~ n - g r a m s ~ } \right\} \cap \left\{ \mathrm { \ n - g r a m s ~ i n ~ r e f ~ } \right\} \big | } { \displaystyle \big | \left\{ \mathrm { \ n - g r a m s ~ i n ~ r e f ~ } \right\} \big | } } \\ { f _ { n } = 2 \cdot \frac { p _ { n } \cdot r _ { n } } { p _ { n } + r _ { n } } } \end{array}
$$

where “restored n-grams” denotes n-grams in model-generated utterances that contain at least one restored word, and “n-grams in ref” denotes n-grams in reference utterances that contain at least one restored word.

## D Details of Baselines

We introduce eight strong baselines to verify the effectiveness of our proposed model TEO as follows.

(1) $\mathbf { B A R T _ { \mathrm { { b a s e } } } } \mathbf { . }$ it generated rewritten utterance using dialog history and incomplete utterance as input;

(2) QUEEN (Si et al., 2022): it proposed a query template that was concatenated with utterance as input;

(3) RAU (Zhang et al., 2022): it extracted relations between tokens from a self-attention matrix;

(4) SGT (Chen, 2023): it first identified fragments and their relative order, and then generated the target utterance;

(5) MIUR (Li et al., 2023a): it mined latent semantic information through a layer of MLP and predicted token types through a joint feature matrix;

(6) Locate-Fill (Li et al., 2023b): it proposed a two-phase incomplete utterance rewriting method that first predicted empty slots and then filled them;

(7) MGIIF (Li et al., 2023b): it proposed a multitask information interaction framework for incomplete utterance rewriting;

(8) XSS (Peng et al., 2024): it is an incomplete utterance rewriting model based on span pairing.

## E Prompts Used in GPT-4 Evaluation

The prompt used in the GPT-4 evaluation is as follows.

## Prompt used in GPT-4 assessment

The goal of dialogue rewriting is to resolve coreference and ellipsis, that is, to complete the coreferential and omitted information in the dialogue without changing its original semantics. Please rewrite the final utterance in the following dialogue.

Examples: {Examples}

Input: {Input}