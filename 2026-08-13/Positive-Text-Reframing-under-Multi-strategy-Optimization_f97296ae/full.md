# Positive Text Reframing under Multi-strategy Optimization

Shutong Jia<sup>1,2</sup>, Biwei Cao<sup>1</sup>, Qingqing Gao<sup>1</sup>, Jiuxin Cao<sup>1</sup>\*, Bo Liu<sup>1</sup> <sup>1</sup>Southeast University, <sup>2</sup>State Grid Tianjin Power Dongli Power Supply Branch {shutong\_jia,caobiwei,qingqing\_gao,jx.cao,bliu}@seu.edu.cn

## Abstract

Differing from sentiment transfer, positive reframing seeks to substitute negative perspectives with positive expressions while preserving the original meaning. With the emergence of pre-trained language models (PLMs), it is possible to achieve acceptable results by fine-tuning PLMs. Nevertheless, generating fluent, diverse and task-constrained reframing text remains a significant challenge. To tackle this issue, a multi-strategy optimization framework (MSOF) is proposed in this paper. Starting from the objective of positive reframing, we first design positive sentiment reward and content preservation reward to encourage the model to transform the negative expressions of the original text while ensuring the integrity and consistency of the semantics. Then, different decoding optimization approaches are introduced to improve the quality of text generation. Finally, based on the modeling formula of positive reframing, we propose a multi-dimensional re-ranking method that further selects candidate sentences from three dimensions: strategy consistency, text similarity and fluency. Extensive experiments on two Seq2Seq PLMs, BART and T5, demonstrate our framework achieves significant improvements on unconstrained and controlled positive reframing tasks.

## 1 Introduction

The concept of style transfer initially emerges within the domain of computer vision (CV) with the objective of accomplishing image style transfer (Gatys et al., 2016). Inspired by this, Hu et al. (2017) proposed text style transfer (TST), whose main purpose is to automatically control the text style and preserve the style-independent content. In recent years, there has been an increasing focus on TST, which has gradually evolved into a significant subfield within the domain of natural language generation. Many corresponding task variants also have been proposed, such as text form transfer (Briakou et al., 2021), topic transfer (Huang et al., 2020), text simplification (Cao et al., 2020), and sentiment transfer (Mueller et al., 2017), etc.

![](images/68ca09b5aa140e351ed77087c5da8303c9641146a1787e9a3f7022a7b14ef624.jpg)  
Figure 1: The difference between sentiment transfer and positive reframing.

Among them, sentiment transfer primarily focuses on reversing the sentiment polarity of the original text. However, it relies on the straightforward replacement of opinion words, such as substituting negative opinion words with their positive counterparts of the opposite meaning. On the one hand, it retains the content irrelevant to style to some extent, such as the invariance of described object entities. On the other hand, it also inherently alters the meaning of the original text (Liao et al., 2018; Li et al., 2018). To this end, Ziems et al. (2022) proposed positive reframing. In contrast to sentiment transfer, positive reframing adopts principles from psychology to reframe negative text by introducing a complementary positive viewpoint while simultaneously maintaining the underlying meaning conveyed in the original text. A toy example of their difference can be seen in Figure 1.

More specifically, positive reframing encompasses various tasks, including unconstrained positive reframing, controlled positive reframing, and derivative tasks such as reframe strategy classification. The unconstrained positive reframing task focuses on generating reframed text without explicit guidance of the corresponding reframe strategy. In contrast, the controlled positive reframing task involves reframing text based on the given strategy. And the reframe strategy classification task entails determining the specific strategy employed in reframing text. Ziems et al. (2022) gives six positive reframing strategies, namely growth mindset, impermanence, neutralization, optimism, self-affirmation and thankfulness.

However, most of the existing methods only finetune PLMs on the corresponding dataset, ignoring the consistency requirement between the model training objective and the target of positive reframing, and also failing to fully utilize the known condition of the reframing strategy under the controlled setting, making it difficult to ensure that the generated text meets the task requirements. Therefore, this paper proposes a multi-strategy optimization framework (MSOF) for positive reframing and our contributions are as follows:

• Firstly, from the target of positive reframing, we design and implement the positive sentiment reward and content preservation reward to optimize the sequence-level training objective, and then apply various decoding improvement approaches to alleviate text degeneration and elevate the quality and diversity of the generated text.

• Secondly, we propose a multi-dimensional reranking approach based on the modeling formula of positive reframing, which comprehensively evaluates the quality of the candidate text based on strategy consistency, text similarity and fluency.

• Extensive experimental results demonstrate that our proposed multi-strategy optimization framework achieves significant improvement on both unconstrained and controlled positive reframing task. And we would release our code to encourage future research<sup>1</sup>.

## 2 Related Work

Early research on text style transfer mostly relied on artificial design features such as syntax (Carroll et al., 1999) and phrase (Quirk et al., 2004) modeling, etc. Similar to other tasks in NLP, the advent of deep learning has resulted in the growing application of neural network models to TST. For example, Jhamtani et al. (2017) investigated the utilization of the Seq2Seq model for transforming modern English into Shakespearean-style English. Wang et al. (2019) applied GPT-2 to accomplish the formal-informal transfer. Sancheti et al. (2020) extended the work of Jhamtani et al. (2017) by incorporating a reinforcement learning framework. Lai et al. (2021) further applied this framework to PLMs. Above studies are mainly based on parallel corpora. Although satisfactory results can be achieved, the cost of constructing parallel corpora is expensive. Therefore, semi-supervised learning and unsupervised learning are widely used in TST. The main methods include data augmentation or text retrieval (Zhang et al., 2020; Jin et al., 2019), adversarial learning (Hu et al., 2017; Fu et al., 2018), back-translation (Prabhumoye et al., 2018; Wei et al., 2023), and reinforcement learning (Luo et al., 2019; Gong et al., 2019).

Specific to sentiment transfer, the early goal is to extract sentiment words that describe the corresponding entities, and then replace them with expressions of the opposite sentiment attribute. The representative one is the “Delete, Retrieve, Generate” strategy (Li et al., 2018). Furthermore, Sudhakar et al. (2019) applied the transformer architecture to the above strategy. To better distinguish content and style, Kim and Sohn (2020) divided the model into sentence reconstruction module and style module to complete their respective task. Han et al. (2023) introduced the adaptive clustering and contrastive learning modules to better explore sentence transmission patterns to main and utilize the latent transfer patterns.

Although sentiment transfer preserves attributeindependent content, the intrinsic meaning of the original text expression is also changed. To this end, Ziems et al. (2022) introduced positive reframing, aiming to preserve the original meaning by substituting negative viewpoints with complementary positive expressions, and constructed the corresponding parallel dataset. For unconstrained positive reframing, Xu et al. (2023) decoupled the sentiment and style of the text to complete the positive reframing. Then, Sheng et al. (2023) further decomposed positive reframing into paraphrase generation and sentiment transfer and constructed corresponding pseudo datasets to fuse generation capabilities through multi-task learning, but also led to the inability to apply their method under the controlled setting.

## 3 Methodology

## 3.1 Problem Definition

Let $( x , y , \psi _ { x } )$ be a triple in the positive reframing task, where $x = \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } \}$ is the original text with negative sentiment, and $y = \{ y _ { 1 } , y _ { 2 } , \dots , y _ { m } \}$ is the target sentence with complementary positive expressions corresponding to x, m and n represent the sentence length. $\psi _ { x } \subseteq \{ \mathrm { G r o w t h } $ Mindset, Impermanence, Neutralizing, Optimism, Self-affirmation, Thankfulness} is the positive reframing strategy used to reframe the negative text x, which can use multiple strategies simultaneously. This paper researches the following three tasks.

![](images/7ff63067597fb0658c3ce0773ce5c88173c47fa49e2c5ef90881a1b14bc1508e.jpg)  
Figure 2: The overall architecture of MSOF. We respectively use BART and T5 as the basic model for positive reframing. The positive sentiment reward and content preservation reward are applied to optimize the model training process. Then, we adopt various decoding improvement approaches (e.g. beam search, random sampling) during the decoding stage to improve the quality of text generation. Finally, multi-dimensional re-ranking is used to comprehensively evaluate candidate sentences and select the candidate with the highest score as the final output.

The target of unconstrained positive reframing is to generate the target sentence y from the original text x without any reframe strategy guidance. This task can be modeled as follows:

$$
p ( \boldsymbol { y } | \boldsymbol { x } ) = \prod _ { t = 1 } ^ { m } p ( y _ { t } | \boldsymbol { x } , y _ { < t } )\tag{1}
$$

where $y _ { < t }$ represents what has been generated before time t.

Regarding reframe strategy classification, its requirement is to predict the reframing strategy ψ<sub>x</sub> used to reframe the original sentence x.

For controlled positive reframing, the primary objective is to generate the target sentence y from the original text x under given strategy $\psi _ { x }$ , This problem can be modeled as the following formula.

$$
p ( \boldsymbol { y } | \boldsymbol { x } , \psi _ { x } ) = \prod _ { t = 1 } ^ { m } p ( \boldsymbol { y } _ { t } | \boldsymbol { x } , \psi _ { x } , \boldsymbol { y } _ { < t } )\tag{2}
$$

## 3.2 Framework

As shown in Figure 2, our proposed framework mainly consists of four modules, namely sequence-

to-sequence, reinforcement training, decoding improvement and multi-dimensional re-ranking.

## 3.2.1 Sequence-to-sequence

Consistent with Ziems et al. (2022), we also use T5 (Raffel et al., 2019) and BART (Lewis et al., 2020) as the basic text generation model, which are both mainly composed of two components, namely encoder and decoder.

Encoder This part is to encode original sentence x and reframe strategy $\psi _ { x }$ into hidden vector H. We use T5 and BART as the basic generation model, and the encoder part is as follows:

$$
H = { \mathrm { E n c o d e r } } ( [ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } ] , \psi _ { x } )\tag{3}
$$

where $H \in \mathbb { R } ^ { l \times d } ,$ , l is the length of sequence, and d is the hidden dimension.

Decoder The output $y _ { t }$ of the decoder part takes the hidden vector output of the encoder and the output $y _ { < t }$ of the decoder before time t as input, the equation is as follows.

$$
y _ { t } = \operatorname { D e c o d e r } ( H ; y _ { < t } )\tag{4}
$$

## 3.2.2 Reinforcement Training

As shown in Figure 3, based on the objective of positive reframing, the generated text should transform the negative sentiment of the original text and keep the semantics unchanged. Therefore, we design and implement positive sentiment reward and content preservation reward to optimize the overall training process.

![](images/e774a5865236bd3116eef95e5221a709d31d121ee482cfdb82e5283bd8e8e180.jpg)  
Figure 3: The reinforcement training procedure of the Seq2Seq-based model.

Positive sentiment reward We first design the positive sentiment reward loss based on binary cross entropy (BCE). Specifically, we fine-tune the binary sentiment classifier RoBERTa (Liu et al., 2019) and utilize it to determine the sentiment change degree of the generated sentence relative to the original text. The positive sentiment reward loss function is formulated as follows:

$$
p ( s _ { t } | y ^ { \prime } , x ) = \mathrm { S i g m o i d } ( \mathrm { R o B E R T a } ( y ^ { \prime } , x ) )\tag{5}
$$

$$
L _ { c l s } = - \mathrm { l o g } ( p ( s _ { t } | y ^ { \prime } , x ) )\tag{6}
$$

where $s _ { t }$ represents the target style, and $y ^ { \prime }$ is the generated sentence.

Content preservation reward Inspired by Lai et al. (2021), we use BLEU score as the reward for content preservation and leverage SCST (Self-Critic Sequence Training) approach (Rennie et al., 2017) as the optimization method. The corresponding loss function is as follows:

$$
\begin{array} { r } { L _ { c o n t } = \displaystyle \sum _ { i } l o g ( p ( y _ { i } ^ { s } | y _ { 1 : i - 1 } ^ { s } , x ) ) ( b l e u ( y ^ { \prime } , y ) } \\ { - b l e u ( y ^ { s } , y ) ) } \end{array}\tag{7}
$$

where $y ^ { s }$ is sampled from the distribution of model outputs at each time step, and $y ^ { \prime }$ is the greedy generation from the model.

The overall loss is a weighted sum of the positive sentiment reward loss $L _ { c l s } .$ , content preservation reward loss $L _ { c o n t }$ , and language modeling loss $L _ { l m }$

$$
L _ { l m } = \sum _ { i } l o g ( p ( y _ { i } | y _ { 1 : i - 1 } , x ) )\tag{8}
$$

$$
L _ { f i n a l } = \alpha L _ { c l s } + \beta L _ { c o n t } + \gamma L _ { l m }\tag{9}
$$

## 3.2.3 Decoding Improvement

Although T5 and BART have demonstrated their superiority in the field of NLG, the sentences generated by default greedy search often result in text degeneration (i.e., empty or repeated sequences) during the decoding stage (Fan et al., 2018; Holtzman et al., 2019). Therefore, in this paper, various decoding improvement ways such as Beam search (Wiseman and Rush, 2016), Top-k sampling (Fan et al., 2018), Top-p sampling (Holtzman et al., 2019) and Typical sampling (Meister et al., 2023) are applied to the decoding stage of the Seq2Seq model to improve the quality of text generation. And Eq. 4 is changed as follows.

$$
y _ { t } = \operatorname { P o s t - P r o c e s s i n g } ( \operatorname { D e c o d e r } ( H ; y _ { < t } ) )\tag{10}
$$

## 3.2.4 Multi-dimensional Re-ranking

According to Bayes Rule, we can decompose Eq. 2 into the product of three probabilities:

$$
p ( y | x , \psi _ { x } ) = p ( \psi _ { x } | y , x ) \times p ( x | y ) \times p ( y )\tag{11}
$$

The first term $p ( \psi _ { x } | y , x )$ can be seen as the consistency of original-to-generative sentence transformation with given reframe strategy<sup>2</sup>. The second term $p ( x | y )$ represents the textual similarity. And the last term $p ( y )$ can be regarded as the overall fluency of the output.

Strategy consistency For this term, we propose Strategy-BERT to evaluate the consistency between text reframing and the given strategy, which draws on the idea of "breaking the whole into pieces" and prompt learning to transform the multi-label problem into multiple binary classification tasks, i.e. training the corresponding model for each reframing strategy. For one thing, this approach enables each model to concentrate on its specific aspect and thus not affect each other. For another thing, it facilitates context semantic enhancement by constructing an auxiliary sentence that incorporates supplementary task prompt to effectively mine the implicit task-specific knowledge contained in PLMs and alleviate the task awareness challenge.

![](images/66c8153f3ad259a3313786f5d64feec73bfb666595390c8cf2a2f8244b2154a4.jpg)  
Figure 4: The overall procedure of reframe strategy classification.

As shown in Figure 4, the original dataset is firstly divided according to the different strategies used in reframing, that is, if the strategy ψ is used in the original-reframed text transfer, this sentence pair will be regarded as a positive sample of corresponding strategy dataset, otherwise, it will be a negative sample. The dataset division results are shown in Table 1.

For different reframe strategies, this paper uses the following way to construct auxiliary question:

"Is the strategy + strategy type + used in the conversion $\mathrm { f r o m + o r i g i n a l + t o + r e f r a m e + ? ^ { \prime } }$ where the artificially added tokens are marked in red, and the reframe strategy, original sentence and reframed sentence are marked in blue. In this way, context semantic enhancement can be achieved by constructing auxiliary question.

Then, we fine-tune BERT on above dataset and propose Strategy-BERT specific to each reframe strategy, which is used to evaluate the strategy consistency score of candidate sentences. For each candidate sentence, we invoke the corresponding evaluation model to calculate its consistency score on the strategies used in positive reframing.

Textual similarity We still use BLEU to calculate this term because it can ensure that the generated text preserves style-independent content (Sancheti et al., 2020).

Fluency Recent works suggest that the probability of output generated from PLM is an appropriate automatic and referenceless measure of fluency (Suzgun et al., 2022; Ramirez et al., 2023). Therefore, we use $\mathrm { G P T } { - } 2 _ { \mathrm { l a r g e } }$ (Radford et al., 2019) to calculate the overall fluency of each candidate.

Finally, we take the product of scores from the above three items as the final score of the candidate sentence and choose the one with the highest score as the final output.

## 4 Experiment

## 4.1 Dataset

Reframe strategy classification To verify the effectiveness of Strategy-BERT, we conduct experiments on reframe strategy classification task. Since this paper converts the multi-label classification problem into multiple binary classification tasks, the dataset is also divided accordingly, and the division results are presented in Table 1.

Positive reframing For unconstrained positive reframing and controlled positive reframing, we adopt the dataset provided by Ziems et al. (2022) . and the specific statistics are given in Table 2.

## 4.2 Evaluating Metrics

Regarding classification task, following Ziems et al.   
(2022), we use F1 score as the evaluation metric.

<table><tr><td rowspan="2">Label</td><td colspan="2">Train</td><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><td>POS</td><td>NEG</td><td>POS</td><td>NEG</td><td>POS</td><td>NEG</td></tr><tr><td>Growth</td><td>1683</td><td>4996</td><td>216</td><td>619</td><td>221</td><td>614</td></tr><tr><td>Impermanence</td><td>1296</td><td>5383</td><td>172</td><td>663</td><td>157</td><td>678</td></tr><tr><td>Neutralizing</td><td>2410</td><td>4269</td><td>303</td><td>532</td><td>302</td><td>533</td></tr><tr><td>Optimism</td><td>3295</td><td>3383</td><td>373</td><td>462</td><td>400</td><td>435</td></tr><tr><td>Self-affirmation</td><td>673</td><td>6006</td><td>92</td><td>743</td><td>76</td><td>759</td></tr><tr><td>Thankfulness</td><td>882</td><td>5797</td><td>94</td><td>741</td><td>109</td><td>726</td></tr></table>

Table 1: The statistics of the reframe strategy classification dataset.

<table><tr><td>Label</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>Growth</td><td>1683</td><td>216</td><td>221</td></tr><tr><td>Impermanence</td><td>1296</td><td>172</td><td>157</td></tr><tr><td>Neutralizing</td><td>2410</td><td>303</td><td>302</td></tr><tr><td>Optimism</td><td>3295</td><td>373</td><td>400</td></tr><tr><td>Self-affirmation</td><td>673</td><td>92</td><td>76</td></tr><tr><td>Thankfulness</td><td>882</td><td>94</td><td>109</td></tr></table>

Table 2: The statistics of the positive reframing dataset (unconstrained & controlled).

For generation task, the following nine automatic metrics are used: (1) Content preservation-related metric, namely ROUGE-1 (R-1), ROUGE-2 (R-2), ROUGE-L (R-L) (Lin, 2004), BLEU (Papineni et al., 2002) and BERTScore (BScore) (Zhang et al., 2019). (2) ∆TextBlob (∆TB) (Loria, 2018) is used to report the average change in sentiment. (3) RTQE (Reframing Text Quality Evaluation) is proposed to evaluate the degree of positive text reframing (i.e. style strength), we fine-tune $\mathrm { R o B E R T a _ { l a r g e } }$ (Liu et al., 2019) to evaluate reframing degree and we regard the probability from the model prediction as the degree of positive reframing between the original and generated sentence; on the human reference it has the F1 score of 95.98% and accuracy of 97.41%. For more details refer to Appendix A (4) Perplexity (PPL) is an indicator of text fluency, and we use $\mathrm { G P T } { - } 2 _ { \mathrm { l a r g e } }$ as the evaluation model.

Finally, following Ziems et al. (2022), we randomly selected 50 samples from each generated file and assigned them to 3 well-educated raters with relevant professional backgrounds to score Meaning Preservation (Meaning), Positivity and Fluency of reframed sentences on a scale of 1 to 5. Since the main research of this paper falls on controlled positive reframing task, we only conducted human evaluation on this task.

## 4.3 Implementation Details

Reframe strategy classification $\mathrm { B E R T _ { b a s e } }$ (Devlin et al., 2019) and $\mathrm { R o B E R T a _ { b a s e } }$ (Liu et al.,

2019) are used as the backbone model in this task respectively. The maximum text embedding length is set to 110. AdamW is used as the optimizer, and the batch size is 16. In addition, all models in this paper are implemented through Hugging-Face (Wolf et al., 2020) and PyTorch (Paszke et al., 2019) on TITAN Xp GPU.

Positive reframing Following Ziems et al. (2022), we use T5 (Raffel et al., 2019) and BART (Lewis et al., 2020) with 6 layers in each of the encoder and decoder, and the hidden size of 768. The value of the learning rate is from 3e-5 to 3e-4, the batch size processed by each device is 6, and the text maximum input length is 80. α, β, γ are respectively set to 1, 0.2, 1, which are the choices obtained from multiple experiments. And the approach of obtaining the candidate sentence set can be seen in Appendix B.

## 4.4 Main Results

## 4.4.1 Reframe Strategy Classification

For this task, this paper selects the Multi-label-BERT and Multi-label-RoBERTa proposed by Ziems et al. (2022) as baselines to compare with the Strategy-BERT and Strategy-RoBERTa proposed in this paper. For fairness, we directly adopt the results reported by Ziems et al. (2022). Since they only report F1 score of their models, we only use it as the evaluation metric in this task. The detailed performance of our proposed models on other metrics can be found in Table 12 in Appendix D.1.

<table><tr><td>Label</td><td>Multi-label- BERT</td><td>Multi-label- RoBERTa</td><td>Strategy- BERT</td><td>Strategy- RoBERTa</td></tr><tr><td>Thankfulness</td><td>0.71</td><td>0.69</td><td>0.73</td><td>0.72</td></tr><tr><td>Neutralizing</td><td>0.59</td><td>0.61</td><td>0.61</td><td>0.61</td></tr><tr><td>Optimism</td><td>0.71</td><td>0.71</td><td>0.71</td><td>0.73</td></tr><tr><td>Impermanence</td><td>0.55</td><td>0.55</td><td>0.57</td><td>0.57</td></tr><tr><td>Growth</td><td>0.63</td><td>0.63</td><td>0.67</td><td>0.69</td></tr><tr><td>Self-affirmation</td><td>0.43</td><td>0.44</td><td>0.48</td><td>0.46</td></tr></table>

Table 3: The experimental results of reframe strategy classification on F1 score. And the best results in each label are in bold.

It can be seen from Table 3 that our models are able to outperform baselines on all labels, significantly on the Growth (Growth Mindset) label, the two models proposed in this paper have increased by 4 points and 6 points respectively. Furthermore, in terms of the Self-affirmation label, Strategy-BERT demonstrates a noteworthy improvement of 5 points compared to the corresponding baseline. Additionally, our method consistently achieves approximately 1 point of improvement on other labels, further affirming the effectiveness and superiority of our approach. Since the performance of Strategy-BERT and Strategy-RoBERTa are similar, we only use Strategy-BERT as the evaluation model to measure the strategy consistency of each candidate.

<table><tr><td>Label</td><td>Strategy-BERT w/o auxiliary</td><td>Strategy- BERT</td><td>Strategy-RoBERTa w/o auxiliary</td><td>Strategy- RoBERTa</td></tr><tr><td>Thankfulness</td><td>0.71</td><td>0.73</td><td>0.69</td><td>0.72</td></tr><tr><td>Neutralizing</td><td>0.59</td><td>0.61</td><td>0.60</td><td>0.61</td></tr><tr><td>Optimism</td><td>0.71</td><td>0.71</td><td>0.71</td><td>0.73</td></tr><tr><td>Impermanence</td><td>0.55</td><td>0.57</td><td>0.55</td><td>0.57</td></tr><tr><td>Growth</td><td>0.61</td><td>0.67</td><td>0.65</td><td>0.69</td></tr><tr><td>Self-affirmation</td><td>0.44</td><td>0.48</td><td>0.44</td><td>0.46</td></tr></table>

Table 4: The experimental results of different input ways on F1 score. The best results in each label are in bold and w/o auxiliary means without using auxiliary sentence.

In addition, the performance of the input approach of directly connecting the original and generated sentence is also tested to demonstrate the effectiveness of the contextual semantic enhancement strategy (i.e., the construction of auxiliary question) used in this paper. And the experimental results are given in Table 4. As can be seen, the F1 score on each label is greatly reduced without context enhancement strategy, but our models still achieve comparable performance with the multilabel classification models which once again shows the effectiveness of our method.

## 4.4.2 Unconstrained Positive Reframing

As shown in Table 5, our proposed framework MSOF achieves significant improvements compared to the baselines. When combining positive sentiment reward and content preservation reward only during the training process, i.e. MSOF<sub>Greedy</sub>, already outperforms the baselines on almost all metrics, especially ROUGE, BScore, RTQE, and PPL. When incorporating decoding optimization and multi-dimensional re-ranking, the performance of the model will be further improved. From the perspective of the model, the T5-based models achieve the best results on metrics such as ∆TB, RTQE and PPL, while the BART-based models reach SOTA on content preservation-related metrics such as ROUGE, BLEU, and BScore. This may be because BART prioritizes semantic preservation rather than sentiment change when reframing the negative text. Among different decoding methods, both beam search and random sampling-based methods are superior to greedy search. Specifically, Top-k sampling has the best overall performance, achieving the best or sub-optimal results on almost all metrics. Top-p sampling performs slightly lower than Top-k sampling. Compared to the above two decoding methods, beam search and Typical sampling are not satisfactory but still superior to the baseline method. Ultimately, regardless of whether T5 or BART is used as the basic generation model, $\mathrm { M S O F _ { T o p - k } }$ achieves the best results among all variant models, basically achieving at least 7% improvement on each metric compared to baselines, which strongly indicates the effectiveness of our proposed framework.

<table><tr><td colspan="2">Model</td><td>R-1</td><td>R-2</td><td>R-L</td><td>BLEU</td><td>BScore</td><td>∆TB</td><td>RTQE</td><td>PPL</td></tr><tr><td rowspan="10">T5</td><td>Vallina Fine-tune (Ziems et al., 2022)</td><td>27.4</td><td>9.8</td><td>23.8</td><td>8.7</td><td>88.7</td><td>0.38</td><td>84.8</td><td>42.7</td></tr><tr><td>FDSC (Xu et al., 2023)</td><td>30.4</td><td>10.9</td><td>25.2</td><td>8.1</td><td>88.8</td><td>0.39</td><td>93.1</td><td>30.0</td></tr><tr><td>PG2ST (Sheng et al., 2023)</td><td>31.1</td><td>11.2</td><td>25.5</td><td>8.9</td><td>88.7</td><td>0.35</td><td>85.4</td><td>41.0</td></tr><tr><td>ST2PG (Sheng et al., 2023)</td><td>30.8</td><td>11.3</td><td>25.5</td><td>8.8</td><td>88.7</td><td>0.33</td><td>84.6</td><td>43.2</td></tr><tr><td> $\mathbf { M S O F _ { G r e e d y } }$ </td><td>32.9</td><td>13.0</td><td>26.0</td><td>8.8</td><td>89.1</td><td>0.37</td><td>86.2</td><td>36.8</td></tr><tr><td> $\mathbf { M S O F _ { B e a m } }$ </td><td>34.1</td><td>14.0</td><td>27.1</td><td>9.7</td><td>89.2</td><td>0.37</td><td>89.0</td><td>35.4</td></tr><tr><td> $\mathbf { M S O F _ { T o p - k } }$ </td><td>34.8</td><td>14.7</td><td>27.7</td><td>10.1</td><td>89.5</td><td>0.44</td><td>93.5</td><td>22.3</td></tr><tr><td> $\mathbf { M S O F _ { T o p - p } }$ </td><td>34.4</td><td>14.6</td><td>27.6</td><td>10.1</td><td>89.4</td><td>0.43</td><td>93.5</td><td>22.2</td></tr><tr><td> $\mathbf { M S O F _ { T y p i c a l } }$ </td><td>32.9</td><td>13.5</td><td>26.2</td><td>9.1</td><td>89.3</td><td>0.39</td><td>94.5</td><td>22.6</td></tr><tr><td rowspan="7"></td><td>Vallina Fine-tune (Ziems et al., 2022)</td><td>27.7</td><td>10.8</td><td>24.3</td><td>10.3</td><td>89.3</td><td>0.23</td><td>63.8</td><td>86.0</td></tr><tr><td>FDSC (Xu et al., 2023)</td><td>32.7</td><td>13.4</td><td>27.0</td><td>10.4</td><td>88.5</td><td>0.21</td><td>60.1</td><td>77.5</td></tr><tr><td>PG2ST (Sheng et al., 2023)</td><td>32.6</td><td>13.5</td><td>26.9</td><td>10.3</td><td>88.4</td><td>0.19</td><td>60.9</td><td>86.2</td></tr><tr><td>ST2PG (Sheng et al., 2023)</td><td>32.9</td><td>13.6</td><td>27.1</td><td>10.9</td><td>88.4</td><td>0.20</td><td>61.5</td><td>78.9</td></tr><tr><td> $\mathbf { M S O F _ { G r e e d y } }$ </td><td>32.3</td><td>13.2</td><td>26.9</td><td>10.4</td><td>89.4</td><td>0.24</td><td>80.1</td><td>47.0</td></tr><tr><td> $\mathbf { M S O F _ { B e a m } }$ </td><td>34.2</td><td>14.2</td><td>28.1</td><td>10.9</td><td>89.5</td><td>0.24 0.31</td><td>87.3 87.3</td><td>33.6</td></tr><tr><td> $\mathbf { M S O F _ { T o p - k } }$ </td><td>34.8</td><td>14.9</td><td>29.3</td><td>12.0</td><td>89.9</td><td></td><td></td><td>25.8</td></tr><tr><td> $\mathbf { M S O F _ { T o p - p } }$ </td><td> $\mathbf { M S O F _ { T y p i c a l } }$ </td><td>34.8 32.5</td><td>14.9 12.8</td><td>29.2 26.9</td><td>12.0 10.4</td><td>89.8 89.5</td><td>0.30 0.30</td><td>87.2 88.5</td><td>27.3 29.6</td></tr></table>

Table 5: The experimental results of unconstrained positive reframing. The best in-category performance is bolded and the best overall performance is highlighted. And except for PPL, all other metrics are better when they are higher.
<table><tr><td colspan="2">Model</td><td>R-1</td><td>R-2</td><td>R-L</td><td>BLEU</td><td>BScore</td><td>∆TB</td><td>RTQE</td><td>PPL</td></tr><tr><td rowspan="7">T5</td><td>Vallina Fine-tune (Ziems et al., 2022)</td><td>27.7</td><td>10.0</td><td>23.9</td><td>8.8</td><td>88.8</td><td>0.36</td><td>86.2</td><td>62.1</td></tr><tr><td> $\mathbf { M S O F _ { G r e e d y } }$ </td><td>33.6</td><td>13.6</td><td>26.7</td><td>8.8</td><td>89.2</td><td>0.37</td><td>94.6</td><td>34.6</td></tr><tr><td> $\mathbf { M S O F _ { B e a m } }$ </td><td>34.6</td><td>14.4</td><td>27.5</td><td>9.5</td><td>89.3</td><td>0.36</td><td>96.2</td><td>34.5</td></tr><tr><td> $\mathbf { M S O F _ { T o p - k } }$ </td><td>34.8</td><td>15.0</td><td>28.0</td><td>9.9</td><td>89.5</td><td>0.43</td><td>97.7</td><td>23.1</td></tr><tr><td> $\mathbf { M S O F _ { T o p - p } }$ </td><td>34.1</td><td>14.2</td><td>27.6</td><td>9.3</td><td>89.5</td><td>0.42</td><td>96.6</td><td>23.0</td></tr><tr><td> $\mathbf { M S O F _ { T y p i c a l } }$ </td><td>33.2</td><td>13.4</td><td>26.5</td><td>8.6</td><td>89.3</td><td>0.42</td><td>97.0</td><td>23.8</td></tr><tr><td>Vallina Fine-tune (Ziems et al., 2022)</td><td>28.8</td><td>10.9</td><td>25.1</td><td>10.1</td><td>89.6</td><td>0.27</td><td>69.5</td><td>89.1</td></tr><tr><td rowspan="5">BART</td><td> $\mathbf { M S O F _ { G r e e d y } }$ </td><td>33.0</td><td>13.3</td><td>27.2</td><td>10.0</td><td>89.6</td><td>0.31</td><td>89.1</td><td>44.4</td></tr><tr><td> $\mathbf { M S O F _ { B e a m } }$ </td><td>34.6</td><td>14.2</td><td>28.2</td><td>10.5</td><td>89.7</td><td>0.34</td><td>94.8</td><td>31.8</td></tr><tr><td> $\mathbf { M S O F _ { T o p - k } }$ </td><td>34.8</td><td>14.7</td><td>29.0</td><td>11.4</td><td>90.1</td><td>0.36</td><td>94.0</td><td>29.4</td></tr><tr><td> $\mathbf { M S O F _ { T o p - p } }$ </td><td>34.6</td><td>14.4</td><td>28.8</td><td>11.3</td><td>90.0</td><td>0.36</td><td>94.0</td><td>30.8</td></tr><tr><td> $\mathbf { M S O F _ { T y p i c a l } }$ </td><td>33.2</td><td>13.2</td><td>27.5</td><td>10.1</td><td>89.8</td><td>0.36</td><td>94.0</td><td>29.8</td></tr></table>

Table 6: The experimental results of controlled positive reframing.

## 4.4.3 Controlled Positive Reframing

Since only Ziems et al. (2022) have studied controlled positive reframing, we use T5 and BART (Ziems et al., 2022) that are fine-tuned on the corresponding dataset as baselines for comparison. The primary experimental results are given in Table

6. It can be concluded that the performance of models under constraints is generally better than unconstrained, which provides support for the reframe strategy to play a role in assisting model inference to a certain extent. Consistent with the experimental results under the unconstrained setting, $\mathrm { M S O F _ { T o p - k } }$ still achieves the best results among all variant models. Compared with the baselines, $\mathrm { M S O F _ { T o p - k } }$ achieves an average improvement of 5 points on ROUGE, 1 point in BLEU, more than 10 points on both RTQE and PPL, and an improvement of about 20% on ∆TB. Moreover, it can be found that although Typical sampling does not perform as well as other decoding approaches on content preservation-related metrics such as ROUGE, BLEU, and BScore, it still achieves impressive results on ∆TB, RTQE and PPL, suggesting that its corresponding output is consistent with task requirements to some extend, even though there is less overlap with human reference.

<table><tr><td></td><td>Model</td><td>R-1</td><td>R-2</td><td>R-L</td><td>BLEU</td><td>BScore</td><td>∆TB</td><td>RTQE</td><td>PPL</td></tr><tr><td rowspan="4">T5</td><td> $\mathbf { M S O F _ { T o p - k } }$ </td><td>34.8</td><td>15.0</td><td>28.0</td><td>9.9</td><td>89.5</td><td>0.43</td><td>97.7</td><td>23.1</td></tr><tr><td>w.o Cls</td><td>34.5</td><td>14.5</td><td>27.5</td><td>9.4</td><td>89.4</td><td>0.41</td><td>96.7</td><td>25.3</td></tr><tr><td>w.o Cont</td><td>35.0</td><td>14.8</td><td>27.7</td><td>9.6</td><td>89.6</td><td>0.37</td><td>95.7</td><td>24.2</td></tr><tr><td>w.o Re-ranking</td><td>32.1</td><td>12.0</td><td>25.2</td><td>7.6</td><td>89.1</td><td>0.43</td><td>96.1</td><td>28.3</td></tr><tr><td rowspan="4">BART</td><td> $\mathbf { M S O F _ { T o p - k } }$ </td><td>34.8</td><td>14.7</td><td>29.0</td><td>11.4</td><td>90.1</td><td>0.36</td><td>94.0</td><td>29.4</td></tr><tr><td>w.o Cls</td><td>33.6</td><td>13.7</td><td>28.2</td><td>10.8</td><td>90.0</td><td>0.35</td><td>86.9</td><td>31.3</td></tr><tr><td>w.o Cont</td><td>33.1</td><td>13.7</td><td>27.5</td><td>10.9</td><td>89.7</td><td>0.38</td><td>86.2</td><td>34.6</td></tr><tr><td>w.o Re-ranking</td><td>31.9</td><td>11.9</td><td>26.2</td><td>9.4</td><td>89.6</td><td>0.35</td><td>92.9</td><td>38.8</td></tr></table>

Table 7: The ablation experimental results of MSOF under controlled setting. w.o Cls means without positive sentiment reward, w.o Cont represents without content preservation reward, w.o Re-ranking represents not using multi-dimensional re-ranking.

## 4.4.4 Ablation Experiment

In addition, from the ablation experimental results shown in Table 7, we can conclude that applying content preservation reward helps the model perform well on ROUGE, BLEU and BScore, but hinders the model from transferring text style. When using positive sentiment reward, although the model performs well on ∆TB and RTQE, it is not satisfactory in terms of content preservation. However, when the two are combined, the model can achieve a better balance between sentiment change and content preservation, exhibiting a more comprehensive performance. Furthermore, it can be observed that the multi-dimensional re-ranking significantly improves the model’s performance on multiple metrics. This demonstrates that it can effectively select the sentence from the candidate that better meets the requirements of positive reframing. Based on the above experimental results and analysis, the validity and rationality of each component of MSOF can be effectively illustrated. For more ablation experiments, please refer to Tables 13 and 14 in Appendix D.2 and Tables 15, 16 and 17 in Appendix D.3.

## 4.4.5 Human Evaluation

Finally, we adopt human evaluation to manually judge the quality of the reframed text. As can be seen from Table 8, our method is more applicable to T5, but for BART, its performance on Positivity is not satisfactory, which can also be reflected by ∆TB and RTQE. Combining the relevant experimental results in Table 6, we speculate this is because the BART-based models prioritize content preservation over sentiment change. In general, consistent with the results and conclusion of automatic metrics, our method can effectively improve the model’s performance, where the T5-based models perform better on Positivity and have a slightly higher score on Fluency, while BART-based models are better on Meaning.

<table><tr><td>Model</td><td>Meaning</td><td>Positivity</td><td>Fluency</td></tr><tr><td>T5 (Ziems et al., 2022)</td><td>4.13</td><td>3.89</td><td>4.07</td></tr><tr><td> $\mathrm { M S O F _ { T o p - k } }$ </td><td>4.38</td><td>4.22</td><td>4.58</td></tr><tr><td>BART (Ziems et al., 2022)</td><td>4.23</td><td>4.07</td><td>4.27</td></tr><tr><td> $\mathrm { M S O F _ { T o p - k } }$ </td><td>4.42</td><td>4.10</td><td>4.54</td></tr></table>

Table 8: The human evaluation results of controlled positive reframing.

## 5 Conclusion

We propose an original multi-strategy optimization framework (MSOF), which consists of reinforcement training, decoding improvement, and multi-dimensional re-ranking, to enhance the performance of PLMs on positive reframing. By conducting extensive experiments on T5-based and BART-based models separately, our framework achieves significant improvements over the baselines on various metrics. Future work includes further cleaning and expansion of the existing dataset to improve the quality and alleviate the imbalanced distribution of different reframe strategy labels, then exploring how the thought of controlled text generation can be applied to this task, followed by trying different approaches of context enhancement, and finally exploring how to apply large language models (LLMs) to positive reframing.

## Limitations

Firstly, the multi-strategy optimization framework proposed in this paper introduces reinforced reward in the model training stage and the multidimensional re-ranking to select the candidate text generated by the model. Therefore, compared with the baselines, our proposed framework needs more memory space and time during training and prediction. Then, this paper finds that the dataset provided by Ziems et al. (2022) has certain noise and label imbalance issues that may hinder the training of the model and there are currently no corresponding datasets in other languages. Finally, we also suggest that if PLMs could be further trained in a rich psychological corpus, the performance would be improved more.

## Ethics Statement

Similar to sentiment transfer, positive reframing has two sides, that is, our method can also be used to generate negative text and cause possible harmful effects on society. However, we still make our code public and hope others will be aware of the possible risks. We welcome any discussion and suggestions to minimize such risks.

## Acknowledgement

This work is supported by National Natural Science Foundation of China under Grants No.62172089, No.62472092, No.62106045. Natural Science Foundation of Jiangsu Province, China under Grants No.BK20241751. Jiangsu Provincial Key Laboratory of Computer Networking Technology, China. Jiangsu Provincial Key Laboratory of Network and Information Security, China under Grants No.BM2003201, and Key Laboratory of Computer Network and Information Integration of Ministry of Education of China under Grants No.93K-9, Nanjing Purple Mountain Laboratories, China. Startup Research Fund of Southeast University under Grants No.RF1028623097. We thank the Big Data Computing Center of Southeast University for providing the facility support on the numerical calculations.

## References

Eleftheria Briakou, Di Lu, Ke Zhang, and Joel Tetreault. 2021. Olá, bonjour, salve! XFORMAL: A benchmark for multilingual formality style transfer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3199–3216, Online. Association for Computational Linguistics.

Yixin Cao, Ruihao Shui, Liangming Pan, Min-Yen Kan, Zhiyuan Liu, and Tat-Seng Chua. 2020. Expertise

style transfer: A new task towards better communication between experts and laymen. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1061–1071, Online. Association for Computational Linguistics.

John Carroll, Guido Minnen, Darren Pearce, Yvonne Canning, Siobhan Devlin, and John Tait. 1999. Simplifying text for language-impaired readers. In Ninth Conference ofthe European Chapter ofthe Association for Computational Linguistics, pages 269–270, Bergen, Norway. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898, Melbourne, Australia. Association for Computational Linguistics.

Zhenxin Fu, Xiaoye Tan, Nanyun Peng, Dongyan Zhao, and Rui Yan. 2018. Style transfer in text: Exploration and evaluation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32.

Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. 2016. Image style transfer using convolutional neural networks. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Hongyu Gong, Suma Bhat, Lingfei Wu, JinJun Xiong, and Wen-mei Hwu. 2019. Reinforcement learning based text style transfer without parallel training corpus. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3168– 3180, Minneapolis, Minnesota. Association for Computational Linguistics.

Jingxuan Han, Quan Wang, Licheng Zhang, Weidong Chen, Yan Song, and Zhendong Mao. 2023. Text style transfer with contrastive transfer pattern mining. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7914–7927, Toronto, Canada. Association for Computational Linguistics.

Ari Holtzman, Jan Buys, Maxwell Forbes, and Yejin Choi. 2019. The curious case of neural text degeneration. CoRR, abs/1904.09751.

Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, and Eric P. Xing. 2017. Toward controlled generation of text. CoRR, abs/1703.00955.

Yufang Huang, Wentao Zhu, Deyi Xiong, Yiye Zhang, Changjian Hu, and Feiyu Xu. 2020. Cycle-consistent adversarial autoencoders for unsupervised text style transfer. In Proceedings of the 28th International Conference on Computational Linguistics, pages 2213–2223, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Harsh Jhamtani, Varun Gangal, Eduard Hovy, and Eric Nyberg. 2017. Shakespearizing modern language using copy-enriched sequence to sequence models. In Proceedings ofthe Workshop on Stylistic Variation, pages 10–19, Copenhagen, Denmark. Association for Computational Linguistics.

Zhijing Jin, Di Jin, Jonas Mueller, Nicholas Matthews, and Enrico Santus. 2019. IMaT: Unsupervised text attribute transfer via iterative matching and translation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natu ral Language Processing (EMNLP-IJCNLP), pages 3097–3109, Hong Kong, China. Association for Computational Linguistics.

Heejin Kim and Kyung-Ah Sohn. 2020. How positive are you: Text style transfer using adaptive style embedding. In Proceedings of the 28th International Conference on Computational Linguistics, pages 2115–2125, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Huiyuan Lai, Antonio Toral, and Malvina Nissim. 2021. Thank you BART! rewarding pre-trained models improves formality style transfer. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 484–494, Online. Association for Computational Linguistics.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 7871–7880, Online. Association for Computational Linguistics.

Juncen Li, Robin Jia, He He, and Percy Liang. 2018. Delete, retrieve, generate: a simple approach to sentiment and style transfer. In Proceedings ofthe 2018 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1865–1874, New Orleans, Louisiana. Association for Computational Linguistics.

Yi Liao, Lidong Bing, Piji Li, Shuming Shi, Wai Lam, and Tong Zhang. 2018. QuaSE: Sequence editing under quantifiable guidance. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3855–3864, Brussels, Belgium. Association for Computational Linguistics.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized BERT pretraining approach. CoRR, abs/1907.11692.

Steven Loria. 2018. textblob documentation. Release 0.15, 2:269.

Fuli Luo, Peng Li, Jie Zhou, Pengcheng Yang, Baobao Chang, Zhifang Sui, and Xu Sun. 2019. A dual reinforcement learning framework for unsupervised text style transfer. CoRR, abs/1905.10060.

Clara Meister, Tiago Pimentel, Gian Wiher, and Ryan Cotterell. 2023. Locally typical sampling. Transactions of the Association for Computational Linguistics, 11:102–121.

Jonas Mueller, David Gifford, and Tommi Jaakkola. 2017. Sequence to better sequence: continuous revision of combinatorial structures. In International Conference on Machine Learning, pages 2536–2544. PMLR.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

Shrimai Prabhumoye, Yulia Tsvetkov, Ruslan Salakhutdinov, and Alan W Black. 2018. Style transfer through back-translation. In Proceedings ofthe 56th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 866–876, Melbourne, Australia. Association for Computational Linguistics.

Chris Quirk, Chris Brockett, and William Dolan. 2004. Monolingual machine translation for paraphrase generation. In Proceedings of EMNLP 2014, pages 142–149, Barcelona, Spain. Association for Computational Linguistics.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners.

Colin Raffel, Noam Shazeer, Adam RoBERTs, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2019. Exploring the

limits of transfer learning with a unified text-to-text transformer. CoRR, abs/1910.10683.

Angela Ramirez, Kartik Agarwal, Juraj Juraska, Utkarsh Garg, and Marilyn Walker. 2023. Controllable generation of dialogue acts for dialogue systems via few-shot response generation and ranking. In Proceedings ofthe 24th Annual Meeting ofthe Special Interest Group on Discourse and Dialogue, pages 355–369, Prague, Czechia. Association for Computational Linguistics.

Steven J. Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, and Vaibhava Goel. 2017. Self-critical sequence training for image captioning. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Abhilasha Sancheti, Kundan Krishna, Balaji Vasan Srinivasan, and Anandhavelu Natarajan. 2020. Reinforced rewards framework for text style transfer. In Advances in Information Retrieval, pages 545–560, Cham. Springer International Publishing.

Ashish Sharma, Kevin Rushton, Inna Lin, David Wadden, Khendra Lucas, Adam Miner, Theresa Nguyen, and Tim Althoff. 2023. Cognitive reframing of negative thoughts through human-language model interaction. In ACL: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Xu Sheng, Fumiyo Fukumoto, Jiyi Li, Go Kentaro, and Yoshimi Suzuki. 2023. Learning disentangled meaning and style representations for positive text reframing. In Proceedings ofthe 16th International Natural Language Generation Conference, pages 424–430, Prague, Czechia. Association for Computational Linguistics.

Akhilesh Sudhakar, Bhargav Upadhyay, and Arjun Maheswaran. 2019. “transforming” delete, retrieve, generate approach for controlled text style transfer. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3269– 3279, Hong Kong, China. Association for Computational Linguistics.

Mirac Suzgun, Luke Melas-Kyriazi, and Dan Jurafsky. 2022. Prompt-and-rerank: A method for zeroshot and few-shot arbitrary textual style transfer with small language models. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 2195–2222, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Yunli Wang, Yu Wu, Lili Mou, Zhoujun Li, and Wenhan Chao. 2019. Harnessing pre-trained neural networks with rules for formality style transfer. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing

(EMNLP-IJCNLP), pages 3573–3578, Hong Kong, China. Association for Computational Linguistics.

Daimeng Wei, Zhanglin Wu, Hengchao Shang, Zongyao Li, Minghan Wang, Jiaxin Guo, Xiaoyu Chen, Zhengzhe Yu, and Hao Yang. 2023. Text style transfer back-translation. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7944– 7959, Toronto, Canada. Association for Computational Linguistics.

Sam Wiseman and Alexander M. Rush. 2016. Sequenceto-sequence learning as beam-search optimization. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1296–1306, Austin, Texas. Association for Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Sheng Xu, Yoshimi Suzuki, Jiyi Li, and Fumiyo Fukumoto. 2023. Decoupling style from contents for positive text reframing. In Neural Information Processing, pages 73–84, Singapore. Springer Nature Singapore.

Boliang Zhang, Ajay Nagesh, and Kevin Knight. 2020. Parallel corpus filtering via pre-trained language models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8545–8554, Online. Association for Computational Linguistics.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with BERT. CoRR, abs/1904.09675.

Caleb Ziems, William Held, Omar Shaikh, Jiaao Chen, Zhehao Zhang, and Diyi Yang. 2024. Can large language models transform computational social science? Computational Linguistics, 50(1):237–291.

Caleb Ziems, Minzhi Li, Anthony Zhang, and Diyi Yang. 2022. Inducing positive perspectives with text reframing. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3682–3700, Dublin, Ireland. Association for Computational Linguistics.

## A Reframing Text Quality Evaluation

## A.1 Problem Statement

The essence of existing TST metrics such as ROUGE and BLEU is to evaluate the similarity between the generated and reference sentence, so a simple copy can lead to a high score (Fan et al., 2018; Holtzman et al., 2019). And for an original sentence, there may be multiple corresponding reframed sentences, especially in the unconstrained case. Furthermore, existing metrics also cannot directly measure the degree of positive reframing. Therefore, this paper proposes a new metric RTQE (Reframing Text Quality Evaluation), which aims to evaluate the degree of positive reframing relationship between the generated and original text that can avoid the limitation of only compared with human reference given in the dataset.

## A.2 Evaluation Model

![](images/c2f17d7a852b41a1302139593ea217343ff58e1e561c3ce5bf0b5bfcd8b59407.jpg)  
Figure 5: The model for RTQE.

Taking the inspiration from Lai et al. (2021), the above problem is simplified into a binary classification task, i.e., judging whether there is a positive reframing relationship between two sentences. In practical evaluation, we regard the probability from the model prediction as the degree of positive reframing between the original and generated sentence. And the RTQE evaluation model established in this paper is shown in Figure 5. Given the original sentence x and the corresponding sentence y, we firstly concatenate them and input into the auto-encoding models such as BERT (Devlin et al., 2019) and RoBERTa (Liu et al., 2019) (without segment embedding). The encoder part is as follows:

$$
H ^ { e } = \operatorname { E n c o d e r } ( [ \operatorname { C L S } ] , x , [ \operatorname { S E P } ] , y , [ \operatorname { S E P } ] )\tag{12}
$$

where [CLS] and [SEP] are special tokens.

The feature vector can be refined through Llayer transformer and the representation of $H ^ { l }$ at the l-th layer $( l \in [ 1 , L ] )$ is calculated as below:

$$
H ^ { l } = \mathrm { T r a n s f o r m e r } _ { l } ( H ^ { l - 1 } ) , H ^ { 0 } = H ^ { e }\tag{13}
$$

We regard the hidden vector $H ^ { \mathrm { [ C L S ] } }$ corresponding to [CLS] at the last layer as the contextualized representation of the whole sequence. And the prediction is obtained through the following equation:

$$
\mathrm { O u t p u t } = \mathrm { S i g m o i d } ( W _ { o } H ^ { \mathrm { [ C L S ] } } + b _ { o } )\tag{14}
$$

where $W _ { o } \in \mathbb { R } ^ { \dim _ { H } \times | y | } .$ is the learnable parameter of the linear layer and $b _ { o }$ is the bias.

## A.3 Dataset

As we simplified the RTQE task as a binary classification question, which determines whether two sentences constitute the positive reframing relationship. Therefore, this paper reconstructs the positive reframing dataset (Ziems et al., 2022) in the following way: for each original sentence, we consider its corresponding reframing sentence as a positive sample, and we pair the original sentence with itself or randomly select other reframing sentences to create negative samples, aiming to enhance the learning depth and generalization ability of the model. The specific statistics are presented in Table 9.

<table><tr><td>Set</td><td>Positive</td><td>Negative</td></tr><tr><td>Train</td><td>6679</td><td>13358</td></tr><tr><td>Dev</td><td>835</td><td>1670</td></tr><tr><td>Test</td><td>835</td><td>1670</td></tr></table>

Table 9: The statistics of the RTQE dataset.

## A.4 Implementation Details

We use BERT (Devlin et al., 2019) and RoBERTa (Liu et al., 2019) as the backbone model respectively. For the base version, the model has 12 transformer encoder layers, and the hidden size is 768. For the large version, the model has 24 transformer encoder layers, and the hidden size is 1024. In this paper, the maximum text embedding length is set to 100 tokens, AdamW with an initial learning rate 1e-5 is used as the optimizer, and batch size is 32.

## A.5 Experiment Results

This paper mainly tests the performance of four models: $\mathrm { \Delta B E R T _ { b a s e } , B E R T _ { l a r g e } , R o B E R T a _ { b a s e } }$ and $\mathrm { R o B E R T a _ { l a r g e } } .$ . And the experimental results are shown in Table 10.

<table><tr><td>Model</td><td>P(%)</td><td>R(%)</td><td>F1(%)</td><td>Acc(%)</td><td>Ref(%)</td></tr><tr><td> $\mathrm { B E R T _ { b a s e } }$ </td><td>94.49</td><td>92.09</td><td>93.41</td><td>96.37</td><td>93.36</td></tr><tr><td> $\mathbf { B E R T _ { l a r g e } }$ </td><td>95.65</td><td>94.85</td><td>95.25</td><td>96.85</td><td>93.49</td></tr><tr><td> $\mathrm { R o B E R T a _ { b a s e } }$ </td><td>94.52</td><td>94.97</td><td>94.74</td><td>96.48</td><td>94.59</td></tr><tr><td> $\mathrm { R o B E R T a _ { l a r g e } }$ </td><td>96.16</td><td>96.05</td><td>96.11</td><td>97.41</td><td>95.98</td></tr></table>

Table 10: The experimental results of RTQE task. The column of Ref refers to the average degree of positive reframing relationship between the human reference and original text in the test set obtained by our models. The best results are in bold.

It can be seen from Table 10 that the performance of RoBERTa is generally better than BERT on all metrics, and the large version is better than the base, which shows that the more parameters and training corpus the model has, the better its performance will be. In the end, $\mathrm { R o B E R T a _ { l a r g e } }$ basically achieves the best results in all metrics and also reaches the F1 score of 95.98% and accuracy of 97.41% in the test of evaluating human reference, so finally this paper uses it as the evaluation model for RTQE.

Finally, we present the Pearson correlation between RTQE and manual evaluation in Table 11. It can be inferred that both the results of the T5-based models and BART-based models show a positive correlation with the three human evaluation metrics, particularly in terms of meaning preservation. This demonstrates that the introduction of the RTQE metric aligns with the task requirements, that is, positive reframing needs to prioritize maintaining the original meaning intact.

<table><tr><td></td><td>Meaning</td><td>Positivity</td><td>Fluency</td></tr><tr><td>T5-based models</td><td>0.78</td><td>0.22</td><td>0.91</td></tr><tr><td>BART-based models</td><td>0.85</td><td>0.62</td><td>0.43</td></tr></table>

Table 11: Pearson correlation between RTQE and human evaluation.

## B The Approach of Obtaining the Candidate Sentence

The approach of obtaining the candidate sentence set is as follows: when beam search is used, the number of candidate sentences with the same beam size can be returned directly, and beam size of 4, 5, and 6 are experimented in this paper; for Top-k sampling, the generated sentences of k = 30, 40, 50 and 60 are composed of candidate sentence set; for Topp sampling, the generated sentences of $p = 0 . 8 0 \mathrm { . }$ 0.85, 0.90 and 0.95 are selected to be composed the candidate sentence set; for Typical sampling, the sentences generated by $\tau = 0 . 2 0$ and 0.95 are selected according to the settings recommended by Meister et al. (2023) to form the candidate sentence set.

## C The Instruction for Human Evaluation

The specific instruction for human evaluation is as follows.

Give the original sentence with negative viewpoint and reframed sentence generated by our models. You need to score the Meaning Preservation (Meaning), Positivity and Fluency of the reframed sentence on a scale of 1 to 5.

Meaning: Indicate whether the reframed sentence preserves the original meaning.

1: Completely changed the original meaning.

3: Meaning related but with slight inconsistency or contradiction.

5: Faithful to the original meaning.

Choose 2 or 4 when you are hesitant.

Positivity: Indicate how positive the reframed sentence is.

1: As negative as the original sentence.

3: Neutral Sentiment, i.e. neither negative nor positive.

5: Very positive compared to the original sentence.

Choose 2 or 4 when you are hesitant.

Fluency: Indicate the fluency of the reframed sentence.

1: The reframed sentence does not make sense and it is unreadable.

3: The reframed sentence contains some minor grammatical errors, but does not affect reading.

5: The reframed sentence is human-like, without any grammatical errors.

Choose 2 or 4 when you are hesitant.

## D Additional Results

## D.1 Reframe Strategy Classification

We provide the detailed scores of our models on all classification evaluation metrics (i.e., accuracy, precision, recall, and F1 score) for others to compare and refer to, which can be found in Table 12.

## D.2 Unconstrained Positive Reframing

For this task, we provide additional ablation results of unconstrained positive reframing in Tables 13 and 14. It can be seen that when the positive sentiment reward is not used, the model’s score on metrics such as ∆TB and RTQE decrease. And when the content preservation reward is not used, the model’s performance on metrics such as ROUGE and BLEU may decline. In addition, it can be found that the improvement brought by multidimensional re-ranking is tremendous, significantly improving the model performance on multiple metrics, indicating that it can better select sentences that meet the requirements of positive reframing from the candidate text set. Based on the above experimental results and analysis, the effectiveness and rationality of each component of MSOF can be fully demonstrated.

## D.3 Controlled Positive Reframing

Similar to Appendix D.2, we provide more detailed ablation experimental results of controlled positive reframing in Tables 15 and 16. In addition to the conclusions already drawn in the unconstrained setting, it can be observed that beam search generates sentences with higher content preservation and achieves great results on ROUGE and BLEU. On the other hand, random sampling strategies, namely Top-k, Top-p, and Typical may yield lower scores on ROUGE and BLEU, but achieve better results on ∆TB, RTQE, and PPL, indicating that their generated text may not overlap much with human reference, but still aligns the task requirements and people’s daily usage habits better. This is also an important reason why we propose the RTQE metric, which can directly evaluate the degree of reframing of the model-generated text on the original text, thereby avoiding problems caused by unique human reference.

Additionally, we present the ablation results of multi-dimensional re-ranking under controlled setting in Table 17. It can be observed that when the strategy consistency evaluation is not used, the scores of $\mathrm { M S O F _ { T o p - k } }$ on RTQE and PPL will decrease significantly, but it has better performance on ROUGE and BLEU. When the text similarity evaluation is not used, the performance of $\mathrm { M S O F _ { T o p - k } }$ would significantly lower on content preservation-related metrics, but achieves best or sub-optimal results on ∆TB and RTQE. And when the fluency evaluation is not used, the model scores significantly lower on PPL, but still achieves suboptimal results on RTQE and content preservationrelated metrics. This paper suggests that the reason for the above phenomenon may be that the strategy consistency evaluation considers excessive content preservation as indicating incomplete reframing, and thus interacts with the text similarity evaluation. In addition, as can be seen from the results in the table, a decrease in text fluency (high PPL) is often accompanied by a decrease on ∆TB and RTQE. Therefore, there may be some positive correlation among them. Finally, although the overall framework does not achieve optimal results on all metrics, considering the performance of each variant model on each metric, choosing this way is the best trade-off.

## D.4 Positive Reframing under ICL

In this section, we test the experimental results of GPT-3.5 on the unconstrained positive reframing task and compare it with $\mathbf { M S O F } ^ { 3 }$ . Under zero-shot, we use Rephrase the above sentence to be more positive (Ziems et al., 2024) as the instruction. And under few-shot, following (Sharma et al., 2023; Ziems et al., 2022), we retrieve 5 representative exemplars with the closest semantic similarity from the training set as the context for each test case.

As shown in Table 18, MSOF still outperforms LLM regarding ROUGE, BLEU, BScore, PPL, and ∆TB, but LLM achieves higher RTQE scores. Based on the examples in Table 19, it can be seen that although the output of LLM reflects the concept of reframing, it tends to generate longer text and is more prone to hallucinations. This demonstrates the applicability of MSOF for positive reframing and the continued significance of studying small models in this task.

## D.5 Case Study

We provide the generated examples of unconstrained and controlled experiments in Tables 19 and 20. A comparative analysis reveals that our models generate more diverse and comprehensive outputs while effectively preserving the underlying meaning of the original text. Specifically, the outputs of the BART-based models are mostly similar, except for the sentences generated by Typical sampling. On the other hand, the T5-based models outperform the BART-based models and baselines by providing the benefits of weekends consistent with human reference. Additionally, although the text in the dataset may contain colloquialisms and even grammatical errors, our models can generate more formal sentences that avoid these issues. Therefore, we speculate that further cleaning and filtering of the data in the dataset can further improve the model’s performance. By comparing the results generated by the model in the unconstrained and controlled settings, it can be inferred that without reframe strategy, the reframing performance of the models will decrease, which proves that the reframing strategy plays an auxiliary role in helping the model generate results that better meet task requirements.

Finally, to further explore whether different reframe strategy will affect the generation results of the model, Table 21 shows the generation result of using different strategy to reframe the same negative text. It is obvious from the results that the model can generate reframing text with corresponding characteristics under the guidance of different reframe strategy, especially "Selfaffirmation", "Thankfulness" and "Growth Mindset". This proves that the model can learn some information from the reframe strategy and it also shows that the research on controlled positive reframing is valuable.

<table><tr><td rowspan="2">Label</td><td colspan="4">Strategy-BERT</td><td colspan="4">Strategy-RoBERTa</td></tr><tr><td>P(%)</td><td>R(%)</td><td>F1(%)</td><td>Acc(%)</td><td>P(%)</td><td>R(%)</td><td>F1(%)</td><td>Acc(%)</td></tr><tr><td>Thankfulness</td><td>77.55</td><td>69.72</td><td>73.43</td><td>93.41</td><td>76.84</td><td>66.97</td><td>71.57</td><td>93.05</td></tr><tr><td>Neutralizing</td><td>52.75</td><td>72.84</td><td>61.20</td><td>66.59</td><td>58.70</td><td>62.58</td><td>60.58</td><td>70.54</td></tr><tr><td>Optimism</td><td>61.04</td><td>85.00</td><td>71.06</td><td>66.83</td><td>63.57</td><td>84.50</td><td>72.69</td><td>69.58</td></tr><tr><td>Impermanence</td><td>56.10</td><td>58.60</td><td>57.32</td><td>83.59</td><td>49.76</td><td>65.61</td><td>56.59</td><td>81.08</td></tr><tr><td>Growth Mindset</td><td>58.70</td><td>77.82</td><td>66.92</td><td>79.64</td><td>65.04</td><td>72.40</td><td>68.52</td><td>82.40</td></tr><tr><td>Self-affirmation</td><td>50.72</td><td>46.05</td><td>48.28</td><td>91.02</td><td>47.22</td><td>44.74</td><td>45.94</td><td>90.42</td></tr></table>

Table 12: The detailed experimental results of reframe strategy classification. We provide detailed experimental results of our models on all classification metrics here for analysis and comparison. And the best results in each label are in bold.

<table><tr><td>Model</td><td>R-1</td><td>R-2</td><td>R-L</td><td>BLEU</td><td>BScore</td><td>∆TB</td><td>RTQE</td><td>PPL</td></tr><tr><td> $\mathbf { M S O F _ { G r e e d y } }$ </td><td>32.9</td><td>13.0</td><td>26.0</td><td>8.8</td><td>89.1</td><td>0.37</td><td>86.2</td><td>36.8</td></tr><tr><td>w.o Cls</td><td>32.3</td><td>12.9</td><td>25.8</td><td>8.8</td><td>89.1</td><td>0.37</td><td>86.1</td><td>39.6</td></tr><tr><td>w.o Cont</td><td>32.6</td><td>12.7</td><td>25.7</td><td>8.4</td><td>89.0</td><td>0.38</td><td>87.6</td><td>38.5</td></tr><tr><td> $\mathbf { M S O F _ { B e a m } }$ </td><td>34.1</td><td>14.0</td><td>27.1</td><td>9.7</td><td>89.2</td><td>0.37</td><td>89.0</td><td>35.4</td></tr><tr><td>w.o Cls</td><td>33.6</td><td>13.7</td><td>26.8</td><td>9.5</td><td>89.2</td><td>0.35</td><td>88.6</td><td>36.3</td></tr><tr><td>w.o Cont</td><td>33.6</td><td>13.6</td><td>26.7</td><td>9.3</td><td>89.1</td><td>0.36</td><td>90.2</td><td>40.0</td></tr><tr><td>w.o Re-ranking</td><td>33.1</td><td>13.2</td><td>26.3</td><td>9.1</td><td>89.1</td><td>0.36</td><td>84.3</td><td>39.5</td></tr><tr><td> $\mathrm { M S O F _ { T o p - k } }$ </td><td>34.8</td><td>14.7</td><td>27.7</td><td>10.1</td><td>89.5</td><td>0.44</td><td>93.5</td><td>22.3</td></tr><tr><td>w.o Cls</td><td>34.6</td><td>14.9</td><td>27.8</td><td>10.2</td><td>89.5</td><td>0.42</td><td>93.5</td><td>22.6</td></tr><tr><td>w.o Cont</td><td>34.0</td><td>14.5</td><td>27.4</td><td>9.6</td><td>89.4</td><td>0.39</td><td>94.1</td><td>23.6</td></tr><tr><td>w.o Re-rankig</td><td>31.9</td><td>11.7</td><td>25.1</td><td>7.7</td><td>89.1</td><td>0.42</td><td>92.7</td><td>27.0</td></tr><tr><td> $\mathrm { M S O F _ { T o p - p } }$ </td><td>34.4</td><td>14.6</td><td>27.6</td><td>10.1</td><td>89.4</td><td>0.43</td><td>93.5</td><td>22.2</td></tr><tr><td>w.o Cls</td><td>34.6</td><td>14.6</td><td>27.6</td><td>9.9</td><td>89.5</td><td>0.42</td><td>94.1</td><td>23.4</td></tr><tr><td>w.o Cont</td><td>34.2</td><td>14.6</td><td>27.6</td><td>9.7</td><td>89.4</td><td>0.37</td><td>93.6</td><td>21.5</td></tr><tr><td>w.o Re-ranking</td><td>31.9</td><td>12.2</td><td>25.3</td><td>8.2</td><td>89.1</td><td>0.41</td><td>90.7</td><td>28.0</td></tr><tr><td> $\mathrm { M S O F _ { T y p i c a l } }$ </td><td>32.9</td><td>13.5</td><td>26.2</td><td>9.1</td><td>89.3</td><td>0.39</td><td>94.5</td><td>22.6</td></tr><tr><td>w.o Cls</td><td>33.4</td><td>13.6</td><td>26.7</td><td>8.9</td><td>89.3</td><td>0.42</td><td>95.7</td><td>22.8</td></tr><tr><td>w.o Cont</td><td>32.2</td><td>12.9</td><td>25.8</td><td>8.3</td><td>89.2</td><td>0.38</td><td>95.3</td><td>23.0</td></tr><tr><td>w.o Re-ranking</td><td>31.7</td><td>12.0</td><td>25.3</td><td>8.0</td><td>89.1</td><td>0.40</td><td>92.2</td><td>31.7</td></tr><tr><td>Model</td><td>R-1</td><td>R-2</td><td>R-L BLEU</td><td>BScore</td><td></td><td>∆TB</td><td>RTQE PPL</td></tr><tr><td> $\mathbf { M S O F _ { G r e e d y } }$ </td><td>32.3</td><td>13.2</td><td>26.9 10.4</td><td>89.4</td><td>0.24</td><td>80.1</td><td>47.0</td></tr><tr><td>w.o Cls</td><td>32.9</td><td>13.3</td><td>27.2 10.1</td><td>89.3</td><td>0.20</td><td>75.9</td><td>53.7</td></tr><tr><td>w.o Cont</td><td>32.4</td><td>13.0 26.8</td><td>10.3</td><td>89.2</td><td>0.26</td><td>79.7</td><td>63.0</td></tr><tr><td>MSOFBeam</td><td>34.2</td><td>14.2</td><td>28.1</td><td>10.9 89.5</td><td>0.24</td><td>87.3</td><td>33.6</td></tr><tr><td>w.o Cls</td><td>34.1</td><td>14.2 27.9</td><td>10.6</td><td>89.5</td><td>0.22</td><td>85.9</td><td>35.0</td></tr><tr><td>w.o Cont</td><td>33.6</td><td>13.8 27.7</td><td>10.6</td><td>89.4</td><td>0.30</td><td>86.1</td><td>36.0</td></tr><tr><td>w.o Re-ranking</td><td>33.3</td><td>13.5 27.4</td><td>10.3</td><td>89.5</td><td>0.29</td><td>88.0</td><td>44.8</td></tr><tr><td> $\mathrm { M S O F _ { T o p \mathrm { - } } }$  -k</td><td>34.8</td><td>14.9</td><td>29.3 12.0</td><td>89.9</td><td>0.31</td><td>87.3</td><td>25.8</td></tr><tr><td>w.o Cls</td><td>34.9</td><td>15.1 29.1</td><td>12.2</td><td>89.8</td><td>0.31</td><td>85.6</td><td>30.2</td></tr><tr><td>w.o Cont</td><td>34.7</td><td>15.0 29.0</td><td>12.2</td><td>89.8</td><td>0.27</td><td>84.1</td><td>30.5</td></tr><tr><td>w.o Re-ranking</td><td>31.6</td><td>11.7 26.0</td><td>9.4</td><td>89.4</td><td>0.28</td><td>84.8</td><td>38.9</td></tr><tr><td>MSOFTop-P</td><td>34.8</td><td>14.9 29.2</td><td>12.0</td><td>89.8</td><td>0.30</td><td>87.2</td><td>27.3</td></tr><tr><td>w.o Cls</td><td>34.4</td><td>14.4 28.5</td><td>11.5</td><td>89.7</td><td>0.27</td><td>84.2</td><td>31.6</td></tr><tr><td>w.o Cont</td><td>34.8</td><td>14.8 29.0</td><td>11.8</td><td>89.8</td><td>0.28</td><td>86.1</td><td>31.5</td></tr><tr><td>w.o Re-ranking</td><td>31.4</td><td>11.9 25.9</td><td>9.3</td><td>89.4</td><td>0.29</td><td>85.6</td><td>37.3</td></tr><tr><td>MSOFTypical</td><td>32.5</td><td>12.8</td><td>26.9</td><td>10.4</td><td>89.5</td><td>0.30</td><td>88.5</td><td>29.6</td></tr><tr><td>w.o Cls</td><td>32.6</td><td>13.2</td><td>26.9</td><td>10.8</td><td>89.5</td><td>0.28</td><td>87.1</td><td>32.6</td></tr><tr><td>w.o Cont</td><td>33.0</td><td>13.2</td><td>27.3</td><td>10.7</td><td>89.5</td><td>0.34</td><td>92.8</td><td>32.6</td></tr><tr><td>w.o Re-ranking</td><td>31.5</td><td>11.9</td><td>25.8</td><td>9.2</td><td>89.2</td><td>0.25</td><td>82.4</td><td>41.3</td></tr></table>

Table 13: The detailed experimental results of unconstrained positive reframing (T5).

Table 14: The detailed experimental results of unconstrained positive reframing (BART).

<table><tr><td>Model</td><td>R-1</td><td>R-2</td><td>R-L</td><td>BLEU</td><td>BScore</td><td>∆TB</td><td>RTQE</td><td>PPL</td></tr><tr><td>MSOFGreedy</td><td>33.6</td><td>13.6</td><td>26.7</td><td>8.8</td><td>89.2</td><td>0.37</td><td>94.6</td><td>34.6</td></tr><tr><td>w.o Cls</td><td>33.5</td><td>13.4</td><td>26.6</td><td>8.9</td><td>89.1</td><td>0.35</td><td>91.2</td><td>38.4</td></tr><tr><td>w.o Cont</td><td>33.3</td><td>13.2</td><td>26.3</td><td>8.6</td><td>89.2</td><td>0.32</td><td>88.8</td><td>41.2</td></tr><tr><td>MSOFBeam</td><td>34.6</td><td>14.4</td><td>27.5</td><td>9.5</td><td>89.3</td><td>0.36</td><td>96.2</td><td>34.5</td></tr><tr><td>w.o Cls</td><td>33.7</td><td>13.7</td><td>26.5</td><td>8.9</td><td>89.2</td><td>0.30</td><td>94.3</td><td>39.7</td></tr><tr><td>w.o Cont</td><td>33.7</td><td>13.6</td><td>26.7</td><td>9.1</td><td>89.1</td><td>0.34</td><td>89.1</td><td>40.3</td></tr><tr><td>w.o Re-ranking</td><td>33.9</td><td>13.7</td><td>26.9</td><td>9.1</td><td>89.2</td><td>0.36</td><td>93.0</td><td>37.3</td></tr><tr><td> $\mathrm { M S O F _ { T o p - k } }$ </td><td>34.8</td><td>15.0</td><td>28.0</td><td>9.9</td><td>89.5</td><td>0.43</td><td>97.7</td><td>23.1</td></tr><tr><td>w.o Cls</td><td>34.5</td><td>14.5</td><td>27.5</td><td>9.4</td><td>89.4</td><td>0.41</td><td>96.7</td><td>25.3</td></tr><tr><td>w.o Cont</td><td>35.0</td><td>14.8</td><td>27.7</td><td>9.6</td><td>89.6</td><td>0.37</td><td>95.7</td><td>24.2</td></tr><tr><td>w.o Re-ranking</td><td>32.1</td><td>12.0</td><td>25.2</td><td>7.6</td><td>89.1</td><td>0.43</td><td>96.1</td><td>28.3</td></tr><tr><td> $\mathrm { M S O F _ { T o p - p } }$ </td><td>34.1</td><td>14.2</td><td>27.6</td><td>9.3</td><td>89.5</td><td>0.42</td><td>96.6</td><td>23.0</td></tr><tr><td>w.o Cls</td><td>34.6</td><td>14.5</td><td>27.5</td><td>9.5</td><td>89.5</td><td>0.41</td><td>95.7</td><td>25.7</td></tr><tr><td>w.o Cont</td><td>34.3</td><td>14.2</td><td>27.2</td><td>9.2</td><td>89.5</td><td>0.39</td><td>95.7</td><td>27.7</td></tr><tr><td>w.o Re-ranking</td><td>32.1</td><td>11.8</td><td>25.4</td><td>7.3</td><td>89.2</td><td>0.43</td><td>95.3</td><td>28.0</td></tr><tr><td> $\mathrm { M S O F _ { T y p i c a l } }$ </td><td>33.2</td><td>13.4</td><td>26.5</td><td>8.6</td><td>89.3</td><td>0.42</td><td>97.0</td><td>23.8</td></tr><tr><td>w.o Cls</td><td>33.2</td><td>13.2</td><td>26.3</td><td>8.5</td><td>89.3</td><td>0.41</td><td>96.9</td><td>26.2</td></tr><tr><td>w.o Cont</td><td>33.2</td><td>13.7</td><td>26.2</td><td>8.9</td><td>89.4</td><td>0.37</td><td>97.4</td><td>25.3</td></tr><tr><td>w.o Re-ranking</td><td>32.3</td><td>12.2</td><td>25.5</td><td>7.7</td><td>89.2</td><td>0.42</td><td>95.3</td><td>28.3</td></tr><tr><td> $\mathbf { M S O F _ { G r e e d y } }$ </td><td>33.0</td><td>13.3</td><td>27.2</td><td>10.0</td><td>89.6</td><td>0.31</td><td>89.1</td><td>44.4</td></tr><tr><td>w.o Cls</td><td>31.8</td><td>12.7</td><td>26.6</td><td>10.2</td><td>89.5</td><td>0.27</td><td>82.3</td><td>57.0</td></tr><tr><td>w.o Cont</td><td>33.3</td><td>13.2</td><td>27.0</td><td>9.8</td><td>89.5</td><td>0.29</td><td>88.6</td><td>47.4</td></tr><tr><td> $\mathbf { M S O F _ { B e a m } }$ </td><td>34.6</td><td>14.2</td><td>28.2</td><td>10.5</td><td>89.7</td><td>0.34</td><td>94.8</td><td>31.8</td></tr><tr><td>w.o Cls</td><td>33.8</td><td>14.2</td><td>27.9</td><td>10.8</td><td>89.6</td><td>0.30</td><td>90.7</td><td>36.4</td></tr><tr><td>w.o Cont</td><td>35.1</td><td>14.5</td><td>28.3</td><td>10.3</td><td>89.6</td><td>0.33</td><td>94.1</td><td>33.</td></tr><tr><td>w.o Re-rank</td><td>33.2</td><td>13.5</td><td>27.5</td><td>10.3</td><td>89.5</td><td>0.29</td><td>88.0</td><td>44.8</td></tr><tr><td> $\mathrm { M S O F _ { T o p - k } }$ </td><td>34.8</td><td>14.7</td><td>29.0</td><td>11.4</td><td>90.1</td><td>0.36</td><td>94.0</td><td>29.4</td></tr><tr><td>w.o Cls</td><td>33.6</td><td>13.7</td><td>28.2</td><td>10.8</td><td>90.0</td><td>0.35</td><td>86.9</td><td>31.3</td></tr><tr><td>w.o Cont</td><td>33.1</td><td>13.7</td><td>27.5</td><td>10.9</td><td>89.7</td><td>0.38</td><td>86.2</td><td>34.6</td></tr><tr><td>w.o Re-ranking</td><td>31.9</td><td>11.9</td><td>26.2</td><td>9.4</td><td>89.6</td><td>0.35</td><td>92.9</td><td>38.8</td></tr><tr><td>MSOFTop-p</td><td>34.6</td><td>14.4</td><td>28.8</td><td>11.3</td><td>90.0</td><td>0.36</td><td>94.0</td><td>30.8</td></tr><tr><td>w.o Cls</td><td>34.0</td><td>14.2</td><td>28.4</td><td>10.8</td><td>90.0</td><td>0.35</td><td>88.1</td><td>34.0</td></tr><tr><td>w.o Cont</td><td>33.5</td><td>14.0</td><td>27.9</td><td>11.2</td><td>89.8</td><td>0.39</td><td>87.8</td><td>33.6</td></tr><tr><td>w.o Re-ranking</td><td>31.4</td><td>11.9</td><td>26.2</td><td>8.9</td><td>89.6</td><td>0.33</td><td>86.9</td><td>47.1</td></tr><tr><td> $\mathrm { M S O F _ { T y p i c a l } }$ </td><td>33.2</td><td>13.2</td><td>27.5</td><td>10.1</td><td>89.8</td><td>0.36</td><td>94.0</td><td>29.8</td></tr><tr><td>w.o Cls</td><td>32.2</td><td>12.6</td><td>26.8</td><td>9.5</td><td>89.7</td><td>0.34</td><td>86.4</td><td>38.9</td></tr><tr><td>w.o Cont</td><td>32.1</td><td>12.6</td><td>26.4</td><td>10.0</td><td>89.5</td><td>0.36</td><td>90.3</td><td>37.6</td></tr><tr><td>w.o Re-ranking</td><td>31.0</td><td>11.6</td><td>25.7</td><td>8.7</td><td>89.5</td><td>0.34</td><td>86.5</td><td>45.6</td></tr></table>

Table 15: The detailed experimental results of contolled positive reframing (T5).

Table 16: The detailed experimental results of contolled positive reframing (BART).

<table><tr><td colspan="2">Model</td><td>R-1</td><td>R-2</td><td>R-L</td><td>BLEU</td><td>BScore</td><td>∆TB</td><td>RTQE</td><td>PPL</td></tr><tr><td rowspan="4">T5</td><td> $\mathrm { M S O F _ { T o p - k } }$ </td><td>34.8</td><td>15.0</td><td>28.0</td><td>9.9</td><td>89.5</td><td>0.43</td><td>97.7</td><td>23.1</td></tr><tr><td>w.o Strategy</td><td>35.6</td><td>15.8</td><td>28.8</td><td>10.7</td><td>89.5</td><td>0.41</td><td>95.0</td><td>30.0</td></tr><tr><td>w.o Similar</td><td>32.2</td><td>12.1</td><td>25.4</td><td>7.6</td><td>89.2</td><td>0.44</td><td>97.5</td><td>21.3</td></tr><tr><td>w.o Fluency</td><td>35.0</td><td>15.3</td><td>28.1</td><td>10.1</td><td>89.5</td><td>0.41</td><td>97.1</td><td>28.6</td></tr><tr><td rowspan="4">BART</td><td> $\overline { { { \bf M } { \bf S } { \bf O } { \sf F } _ { \mathrm { T o p - k } } } }$ </td><td>34.8</td><td>14.7</td><td>29.0</td><td>11.4</td><td>90.1</td><td>0.36</td><td>94.0</td><td>29.4</td></tr><tr><td>w.o Strategy</td><td>34.0</td><td>14.6</td><td>28.4</td><td>11.8</td><td>89.7</td><td>0.37</td><td>84.3</td><td>34.0</td></tr><tr><td>w.o Similar</td><td>29.6</td><td>10.6</td><td>24.4</td><td>8.3</td><td>89.3</td><td>0.41</td><td>85.8</td><td>32.3</td></tr><tr><td>w.o Fluency</td><td>33.9</td><td>14.3</td><td>28.2</td><td>11.6</td><td>89.7</td><td>0.35</td><td>86.2</td><td>46.9</td></tr></table>

Table 17: The ablation experimental results of multi-dimensional re-ranking. w.o Strategy means without strategy consistency evaluation, w.o Similarity represents without textual similarity evaluation and w.o Fluency represents not using fluency evaluation.

<table><tr><td>Setting</td><td>R-1</td><td>R-2</td><td>R-L</td><td>BLEU</td><td>BScore</td><td>∆TB</td><td>RTQE</td><td>PPL</td></tr><tr><td>Zero-shot</td><td>27.0</td><td>7.2</td><td>21.2</td><td>4.2</td><td>88.9</td><td>0.43</td><td>99.8</td><td>42.0</td></tr><tr><td>Five-shot</td><td>28.5</td><td>8.5</td><td>22.5</td><td>5.3</td><td>89.3</td><td>0.28</td><td>97.1</td><td>27.5</td></tr></table>

Table 18: The performance of GPT3.5 on positive reframing.

<table><tr><td>Original text</td><td>So glad that tomorrow is Friday. This has seriously been the longest week of my life</td></tr><tr><td>Reference</td><td>I&#x27;m glad the weekend is coming up,so I can rest</td></tr><tr><td>T5 (Ziems et al., 2022)</td><td>This week has been a long one, but I&#x27;m sure it will be over soon.</td></tr><tr><td>FDSC (Xu et al., 2023)</td><td>I&#x27;m so glad that tomorrow is Friday. This week has been a long one.</td></tr><tr><td>ST2PG (Sheng et al., 2023)</td><td>I&#x27;m glad that tomorrow is Friday. This has been the longest week of my life.</td></tr><tr><td>PG2ST (Sheng et al., 2023)</td><td>I&#x27;m glad that tomorrow is Friday. This has been the longest week of my life.</td></tr><tr><td> $\mathbf { M S O F _ { B e a m } }$ </td><td>I&#x27;m glad that tomorrow is Friday. It&#x27;s been a long week,but it&#x27;s going to be a good one</td></tr><tr><td> $\mathrm { M S O F _ { T o p - k } }$ </td><td>This week has been a long week, but I&#x27;m glad it&#x27;s Friday.I&#x27;ll be able to relax and enjoy the weekend</td></tr><tr><td> $\mathrm { M S O F _ { T o p - p } }$ </td><td>It&#x27;s been a long week, but it&#x27;s a good chance to get some rest</td></tr><tr><td> $\mathrm { M S O F _ { T y p i c a l } }$ </td><td>I&#x27;m glad that tomorrow is Friday. This week has been challenging, but I&#x27;m going to get through it.</td></tr><tr><td>BART (Ziems et al., 2022) FDSC (Xu et al., 2023)</td><td>I&#x27;m glad that tomorrow is Friday. This has been the longest week of my life, but I&#x27;m sure I&#x27;ll get through it.</td></tr><tr><td>ST2PG (Sheng et al., 2023)</td><td>So glad that tomorrow is Friday. This has been the longest week of my life. I&#x27;m tired, but I&#x27;m sure I can get through it.</td></tr><tr><td>PG2ST (Sheng et al., 2023)</td><td>I&#x27;m glad that tomorrow is Friday. This has been the longest week of my life, but I&#x27;m sure it will be over soon.</td></tr><tr><td> $\mathbf { M S O F _ { B e a m } }$ </td><td>I&#x27;m glad that tomorrow is Friday. This has been the longest week of my life, but I&#x27;m sure it will be over soon.</td></tr><tr><td> $\mathrm { M S O F _ { T o p - k } }$ </td><td>I&#x27;m glad that tomorrow is Friday. This week has been very challenging.</td></tr><tr><td> $\mathrm { M S O F _ { T o p - p } }$ </td><td>I&#x27;m glad that tomorrow is Friday. This week has been very challenging.</td></tr><tr><td> $\mathrm { M S O F _ { T y p i c a l } }$ </td><td>I&#x27;m glad that tomorrow is Friday. This week has been so long. I&#x27;m glad that tomorrow is Friday. This week has been challenging, but I&#x27;m going to get through it.</td></tr><tr><td>GPT-3.5 (zero-shot)</td><td></td></tr><tr><td>GPT-3.5 (five-shot)</td><td>I&#x27;m excited that tomorrow is finally Friday! This week has been full of new experiences and opportunities I can&#x27;t wait for Friday to finally arrive! This week has been a challenge, but I made it through. tired</td></tr></table>

Table 19: The reframing examples of unconstrained positive reframing. In order to better compare with the constrained settings. The pink text shows the positive perspective.

Table 20: The model comparison for reframing the same text and the reframing strategy is optimism. And we selected the same example as Table 19 to better compare the output of models under different settings.
<table><tr><td>Original text</td><td>So glad that tomorrow is Friday. This has seriously been the longest week of my life! tired</td></tr><tr><td>Reference</td><td>I&#x27;m glad the weekend is coming up,so I can rest.</td></tr><tr><td>T5 (Ziems et al., 2022)</td><td>So glad that tomorrow is Friday. This has seriously been the longest week of my life. I&#x27;m tired, but I know I&#x27;ll get through it.</td></tr><tr><td> $\mathbf { M S O F _ { B e a m } }$ </td><td>I&#x27;m so glad that tomorrow is Friday. This has been the longest week of my life,but I know that tomorrow will be a better day</td></tr><tr><td> $\mathrm { M S O F _ { T o p - k } }$ </td><td>I&#x27;m glad that tomorrow is Friday. This has been the longest week of my life, and I&#x27;ve had a lot of fun</td></tr><tr><td> $\mathrm { M S O F _ { T o p - p } }$ </td><td>I&#x27;m glad that tomorrow is Friday. This has been the longest week of my life,but I know it&#x27;s going to be a great day.</td></tr><tr><td> $\mathrm { M S O F _ { T y p i c a l } }$ </td><td>Tomorrow is Friday. This has been the longest week of my life, but I know I will make it to the end of the week. It will be great.</td></tr><tr><td>BART (Ziems et al., 2022)</td><td>I&#x27;m glad that tomorrow is Friday. This week has been long, but I&#x27;m looking forward to the weekend.</td></tr><tr><td> $\mathbf { M S O F _ { B e a m } }$ </td><td>I&#x27;m so glad that tomorrow is Friday. This has been the longest week of my life! I’m tiredbut I’m sure it will be good</td></tr><tr><td> $\mathrm { M S O F _ { T o p - k } }$   $\mathrm { M S O F _ { T o p - p } }$ </td><td>I&#x27;m really looking forward to Friday,so I can relax a bit</td></tr><tr><td></td><td>I&#x27;m glad that tomorrow is Friday.I&#x27;m going to feel so much better.</td></tr><tr><td> $\mathrm { M S O F _ { T y p i c a l } }$ </td><td>Even though Friday is the longest week in my life,I&#x27;m happy to have the chance to rest for a few days</td></tr></table>

<table><tr><td colspan="2">Original text</td><td>I hate that I stress my self out so much that I can&#x27;t fall asleep!</td></tr><tr><td rowspan="5"></td><td>Growth Mindset</td><td>I need to take better care of myself so that I can fall asleep in no time! I&#x27;m going to try to reduce my stress and improve my sleep.</td></tr><tr><td>Impermanence</td><td>I don&#x27;t like that I stress myself out so much that I can&#x27;t fall asleep, but I&#x27;m sure I&#x27;ll get better soon.</td></tr><tr><td>Neutralizing Optimism</td><td>I am stressed out so much that I can&#x27;t fall asleep, but I&#x27;m going to take a nap and sleep better so I can sleep better. I don&#x27;t like to stress myself out so much that I can&#x27;t fall asleep, but I&#x27;m sure I&#x27;ll fall asleep soon.</td></tr><tr><td>Self-affirmation</td><td>I don&#x27;t like that I stress my self out so much that I can&#x27;t fall asleep, but I&#x27;m a strong person, and I know I can do it.</td></tr><tr><td>Thankfulness</td><td>I&#x27;m glad I have a bed to sleep in after a long day of stressing myself out, I can&#x27;t sleep.</td></tr><tr><td rowspan="8"> $\mathbf { M S O F _ { T o p - k } ( B A R T ) }$ </td><td>Growth Mindset</td><td>I&#x27;m going to stop stressing out about things so that I can fall asleep.</td></tr><tr><td>Impermanence</td><td></td></tr><tr><td>Neutralizing</td><td>I&#x27;m going to take some time to myself to clear my head.</td></tr><tr><td>Optimism</td><td>Stress is part of life, and I can&#x27;t fall asleep, but I&#x27;m sure I&#x27;ll feel better soon.</td></tr><tr><td>Self-affirmation</td><td>I&#x27;m going to have to stay up all night tonight so that I can get some peace of mind.</td></tr><tr><td>Thankfulness</td><td>I am not able to sleep because of my stress. But I am a strong person, and I know I can get through this.</td></tr><tr><td></td><td>I&#x27;m thankful that I have a bed to sleep in when I&#x27;m stressed.</td></tr><tr><td></td><td></td></tr></table>

Table 21: A model comparison for reframing the same text using different reframe strategy.