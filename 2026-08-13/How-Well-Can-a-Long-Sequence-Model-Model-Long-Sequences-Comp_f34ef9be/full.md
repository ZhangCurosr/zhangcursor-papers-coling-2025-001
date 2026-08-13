# How Well Can a Long Sequence Model Model Long Sequences? Comparing Architectural Inductive Biases on Long-Context Abilities

Jerry Huang Mila - Quebec AI Institute & Université de Montréal jerry.huang@mila.quebec

## Abstract

Long sequences occur in abundance within real-world scenarios, hence properly modelling them opens numerous down-stream use-cases. Deep neural networks, however, have often struggled with these for a variety of reasons. Recent advances, both in system engineering as well as model design, have enabled the scaling up of model that are purported to support extended context length. In particular, the statespace and linear recurrent neural network families of models hypothetically can entend to infinite sequence lenth. However, is this too good to be true? We conduct a targeted evaluation, where we show that while such claims may have theoretical soundness under particular conditions, these may break down in practical settings where limitations exist. In particular, we observe that these new-age sequence models suffer similarily as attention-based models when it comes to long-contexts, highlighting the need to further study such paradigms and why they seemingly fail to behave as expected.

## 1 Introduction

Advances in AI system engineering (Dao et al., 2022; Dao, 2024; Rasley et al., 2020) and model design (Katharopoulos et al., 2020; Jiang et al., 2023; AI21, 2024) have opened language models to the broader public for a diverse set of purposes and use cases. However, Transformer-based architechtures (Vaswani et al., 2017) remain bounded in terms of their context windows, as they require fixed-length positional embedding representations (Press et al., 2022; Su et al., 2023; Peng et al., 2024) which cannot be modified a posteriori. With this glaring limitation, linear sequence models (Gu et al., 2022; Gu and Dao, 2024; Orvieto et al., 2023; Qin et al., 2023; Peng et al., 2023; De et al., 2024; Dao and Gu, 2024) have emerged as an alternative that present a seeming ability to extend to infinite-length contexts in theory while retaining all the original benefits of the Transformer related to training-based parallization.

However, despite the temptation to assert linear sequence models as superior, properly testing for information retention from long-context tasks remains callenging. Although some work has attempted to evaluate this ability through long contexts (Shaham et al., 2022; Pang et al., 2022; Dong et al., 2024; Bai et al., 2023; Li et al., 2023; Han et al., 2024), whether or not they truly require the use of long-contexts is uncertain and ascertaining long-context abilities from these tasks is difficult. This has prompted the use of more synthetic tasks (Hsieh et al., 2024), such as needle-ina-haystack (NIAH) (Kamradt, 2023) and passkey retreival (Mohtashami and Jaggi, 2023), to better control and evaluate the context sizes of models.

Nevertheless, an outstanding question remains whether or not long-context models can effectively model long contexts. While some works (Gu and Dao, 2024; Fu et al., 2023; Poli et al., 2023; Peng et al., 2024; Team, 2024) purport to be able to extrapolate towards sequences of long length (100k tokens+), further investigation has suggested differently. For example, Hsieh et al. (2024) claim modern LLMs significantly over-state true context windows on a number of synthetic tasks. Meanwhile Han et al. (2024) observe models to perform reasonably well on synthetic tasks, but struggle on real-world tasks, as do Li et al. (2023). Hence despite a consistent trend in models behaving underwhelmingly, it remains to be understood why this occurs. Yet one interesting question is whether or not linear sequence models are in fact more suited for these compared to Transformer-based ones, as has been claimed repeatedly.

To this end, we further analyze the behaviour of sequence models to observe how differently they behave compared to Transformer-based ones. We perform a more extensive study into each type of model, as well as a mixture of both, to better investigate how they perform in principle and how they change in behaviour when extending to longer and longer sequences. On both synthetic and realistic data, we conduct a thorough study and observe:

• All models, whether they use pure sequence layers, attention or a mix, struggle with extrapolating beyond their training context length.

• The abiliy to extrapolate can vary signficantly based on the format of the sequence even if the task remains constant. However models consistently struggle more with information placed in the middle of long contexts.

These results highlight that long sequence models suffer from significant limitations despite their theoretical soundness, highlighting a need to better understand this striking dissonance between expectation and observation and how to amend it for better long-context understanding and reasoning.

## 2 Related Work

Efficient Long-Context Models. Due to the computational bottleneck of attention (Bahdanau et al., 2015) relative to sequence length, significant modifications have been made to overcome this limitation of the Transformer (Child et al., 2019; Katharopoulos et al., 2020; Su et al., 2023) yet they remain theoretically bounded in terms of its context length. Alternatively, sequence models (Rumelhart et al., 1986; Jordan, 1986; Hochreiter and Schmidhuber, 1997; Cho et al., 2014) originally faced significant issues that limited their application but recent modifications (Gu et al., 2020, 2021) have led to the prominence of linear sequence models which are significantly more compute-effective than Transformer-based architechtures.

On the Limits of Long Sequence Models. Due to their more intuitive and interpretable architechture, long/linear sequence models remain easier to analyze when placed in comparision to Transformers. As such, their limitations also become easier to discover and analyze. Vardasbi et al. (2023) first show that SSMs struggle at sequenceto-sequence tasks due to to the use of a fixed-size hidden representation which compresses the entire prior context, making it difficult to extract information from the past, fact further substantiated by Jelassi et al. (2024). Park et al. (2024) additionally demonstrate that these models have difficulty with more complex in-context learning tasks, while

Merrill et al. (2024) show them to possess similar limitations in terms of representational power as Transformers (Merrill and Sabharwal, 2023). Waleffe et al. (2024) finally make a comparision between Mamba, Transformers as well as a hybrid and observe hybrid models to perform better on long-context tasks, while Mamba2 often trails behind Transformers. These observations thus beg a question: can long sequence models really model long sequences? Given the hints that long sequence models may not always be as they seem, a more formal investigation is necessary. We distinguish ourselves by conducting a more controlled but intricate study which aims to uncover why some of the prior results might occur, which we discuss in the work that follows.

## 3 Background

Attention and Long Sequences. Self-attention as used in Transformers is powerful but costly. When provided an embedded text representation as a sequence of tokens $\pmb { x } \in \mathbb { R } ^ { L \times d }$ , each Transformer layer in the network applies a function

$$
T _ { \ell } ( { \pmb x } ) = \mathrm { F F } _ { \ell } ( A _ { \ell } ( { \pmb x } ) + { \pmb x } ) + A _ { \ell } ( { \pmb x } )\tag{1}
$$

where $A _ { \ell }$ is the self-attention mechanism of the $\ell -$ th layer and $\operatorname { F F } _ { \ell }$ is the following feed-forward network<sup>1</sup>. Self-attention computes, for every position, a weighted average of the feature representations of all other positions with a weight proportional to a similarity score between the representations.

$$
\begin{array} { l } { { { \pmb Q } _ { \ell } = { \pmb x } { \pmb W } _ { \ell } ^ { Q } \quad { \pmb K } _ { \ell } = { \pmb x } { \pmb W } _ { \ell } ^ { K } \quad V _ { \ell } = { \pmb x } { \pmb W } _ { \ell } ^ { V } } } \\ { { \ A _ { \ell } ( { \pmb x } ) = { \pmb V } _ { \ell } ^ { \prime } = \mathrm { s o f t m a x } \big ( { \pmb Q } _ { \ell } { \pmb K } _ { \ell } ^ { T } / \sqrt { d } \big ) { \pmb V } _ { \ell } } } \end{array}\tag{2}
$$

As the softmax operation operates in $O ( L ^ { 2 } )$ time when applied naively, this limits the ability to process long-sequences.

Transformers to Sequence Models. The longsequence limitations of Transformers necessitates the need for alternatives in such settings, which have currently appeared under the form of stateof-the-art sequence models. An initial proposal borrowed from control theory, namely the notion of state-space models (SSMs). These model a dynamical system, traditionally mapping a 1-D continuous input signal $x ( t ) \in \mathbb { R }$ to an n-dimensional hidden state $h ( t ) \in \mathbb { R } ^ { n }$ that is projected back to a

1-D output $y ( t ) \in \mathbb { R }$ using:

$$
\left\{ \begin{array} { l l } { h ^ { \prime } ( t ) } & { = A h ( t ) + B x ( t ) } \\ { y ( t ) } & { = C h ( t ) + D x ( t ) } \end{array} \right.\tag{3}
$$

where A, B, C and D are all trainable parameters. Gu et al. (2021) use this paradigm to define a recurrent model to work on discrete signals, in which case the input can be regarded as discretized data sampled from a continuous signal with a step size $\Delta ,$ , for which the corresponding SSM is defined by:

$$
\begin{array} { l l } { { h _ { t } = \overline { { { A } } } h _ { t - 1 } + \overline { { { B } } } x _ { t } } } & { { y _ { t } = \overline { { { C } } } h _ { t } + \overline { { { D } } } x _ { t } } } \\ { { \overline { { { A } } } = \displaystyle \frac { \left( I + \Delta A / 2 \right) } { \left( I - \Delta A / 2 \right) } } } & { { \overline { { { B } } } = \displaystyle \frac { \Delta B } { \left( I - \Delta A / 2 \right) } } } \end{array}\tag{4}
$$

and $\overline { { C } } = C$ (They set $\overline { { \pmb { D } } } = 0$ due to being equivalent to a residual connection.) Thus the output y given an input x is

$$
\begin{array} { l } { { \overline { { { \cal K } } } = ( \overline { { { \cal C } } } \overline { { { \cal B } } } , \overline { { { \cal C } } } { \cal A } \overline { { { \cal B } } } , . . . , \overline { { { \cal C } } } { \cal A } ^ { L - 1 } \overline { { { \cal B } } } ) } } \\ { { \displaystyle y _ { t } = \sum _ { j = 0 } ^ { L - 1 } \overline { { { \cal C } { \cal A } } } ^ { j } \overline { { { \cal B } } } x _ { L - j } = \overline { { { \cal K } } } \ast x } } \end{array}\tag{5}
$$

where $\overline { { \kappa } }$ is the SSM kernel. As y can be computed in O(L log L) with a Fast Fourier Transform (Cormen et al., 2009), the entire output can be computed in tandem based on the input, given the matrices that parametrize the system. Gu et al. (2021) use this to overcome issues of parallelization and vanishing gradients (Bengio et al., 1994; Hochreiter et al., 2001; Pascanu et al., 2013) observed by prior recurrent models by

(1) Removing non-linearities in the recurrence, enabling the efficient pre-computation of $\overline { { \pmb { K } } }$

(2) Using a special matrix parameterization (Gu et al., 2020) for A to memorize the input and eliminate exponential gradient scaling.

This has sparked a new wave of recurrent models to compete with Transformers (Orvieto et al., 2023; Qin et al., 2023; De et al., 2024; Beck et al., 2024), with the added benefit of theoretically having longer context sizes that scale more efficiently.

## 4 Experiments and Results

Datasets. We conduct an initial evaluation using RULER (Hsieh et al., 2024), a set of synthetic benchmarks that test long-context information retention, before conductin a more fine-grained evaluation on a general needle-in-the-haystack task. We use this benchmark as for more granular control over the exact information that must be retained. Results are measured in terms of accuracy based on exact matching of predicted tokens.

Baselines. Our main objective is to compare how long-sequence models fare on long context tasks. To this end, we compare models with the same number of parameters that are evenly trained on the same data. Hence we first use Mamba2 (Dao and Gu, 2024) as well as a Transformer variant (Transformer++) as well as a hybrid Mamba2Attn, each with 2.7 billion parameters. We further add Sheared-LLaMA (Xia et al., 2024) and Recurrent-Gemma (Botev et al., 2024) baselines (with and without intruction-tuning) as same-sized baselines trained under different conditions. We finally add a 3 billion RWKV (Peng et al., 2023) variant as another sequence model baseline.

Results. We present initial results on the base set of RULER tasks (as defined by its original authors) in Table 1. Results presented are averaged across individual tasks within the benchmark, which are described in futher detail in Appendix B. However, we present two additional ablation studies. In the first, we use a single needle hidden within a large haystack; however, we modify its relative position within the context. The goal of this ablation, presented in Table 2 and 3, is to observe how the use of a unified hidden state rather than attention can affect the ability to retain information throughout a long sequence. The second (Table 4) further tests how this information retention may change when the content that is being memorized changes (e.g. numbers versus UUIDs within a haystack of repeated sentences or essays). In all tables, we abbreviate model names using titles noted in Appendix A.

<table><tr><td>Length</td><td>1K</td><td>2K</td><td>4K</td><td>8K</td><td>16K</td><td>Average</td></tr><tr><td>Mamba2</td><td>38.52</td><td>32.91</td><td>12.98</td><td>6.51</td><td>0.1</td><td>18.2</td></tr><tr><td>M2A</td><td>39.14</td><td>30.43</td><td>12.89</td><td>7.8</td><td>3.49</td><td>18.75</td></tr><tr><td>TPP</td><td>46.61</td><td>36.74</td><td>0.31</td><td>0.06</td><td>0.03</td><td>16.75</td></tr><tr><td>RG</td><td>78.82</td><td>71.72</td><td>22.45</td><td>11.21</td><td>6.29</td><td>38.1</td></tr><tr><td>SL</td><td>84.38</td><td>69.89</td><td>58.37</td><td>0.0</td><td>0.0</td><td>42.53</td></tr><tr><td>RWKV</td><td>68.09</td><td>55.27</td><td>37.47</td><td>23.73</td><td>13.81</td><td>39.67</td></tr><tr><td>RG-IT</td><td>85.64</td><td>79.45</td><td>44.33</td><td>24.19</td><td>14.18</td><td>49.56</td></tr><tr><td>SL-IT</td><td>86.22</td><td>77.54</td><td>74.25</td><td>0.0</td><td>0.0</td><td>47.6</td></tr></table>

Table 1: Results on RULER. Accuracy is aggregated across several tasks for each model and context length. Context length for which each model was trained is underlined. Best performing models are bolded.

<table><tr><td>Position</td><td>0</td><td>20</td><td>40</td><td>50</td><td>60</td><td>80</td><td>100</td><td>Avg</td></tr><tr><td>Mamba2</td><td>59.07</td><td>31.47</td><td>33.07</td><td>39.07</td><td>40.0</td><td>31.33</td><td>66.0</td><td>42.63</td></tr><tr><td>M2A</td><td>40.27</td><td>36.53</td><td>30.27</td><td>29.33</td><td>29.33</td><td>35.07</td><td>37.2</td><td>35.26</td></tr><tr><td>TPP</td><td>53.33</td><td>33.47</td><td>22.8</td><td>26.27</td><td>31.33</td><td>35.07</td><td>55.73</td><td>35.64</td></tr><tr><td>RG</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.47</td><td>99.92</td></tr><tr><td>SL</td><td>99.6</td><td>99.6</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.89</td></tr><tr><td>RWKV</td><td>82.4</td><td>100.0</td><td>100.0</td><td>80.27</td><td>100.0</td><td>100.0</td><td>100.0</td><td>94.67</td></tr><tr><td>RG-IT</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>SL-IT</td><td>98.27</td><td>99.6</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.73</td><td>99.66</td></tr></table>

Table 2: Results on needle-in-a-haystack task where the position of a single needle is at a fixed depth within the haystack. Context length is set to the maximum on which the models were trained.

<table><tr><td>Position</td><td>0</td><td>20</td><td>40</td><td>50</td><td>60</td><td>80</td><td>100</td><td>Avg</td></tr><tr><td>Mamba2</td><td>26.8</td><td>19.6</td><td>17.73</td><td>18.93</td><td>18.93</td><td>20.13</td><td>21.87</td><td>21.03</td></tr><tr><td>M2A</td><td>38.8</td><td>26.27</td><td>18.93</td><td>28.8</td><td>10.13</td><td>21.6</td><td>66.67</td><td>27.07</td></tr><tr><td>TPP</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>RG</td><td>0.0</td><td>0.0</td><td>0.0</td><td>99.87</td><td>100.0</td><td>100.0</td><td>96.27</td><td>56.59</td></tr><tr><td>SL</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>RWKV</td><td>33.47</td><td>99.6</td><td>100.0</td><td>36.53</td><td>100.0</td><td>100.0</td><td>100.0</td><td>81.37</td></tr><tr><td>RG-IT</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td><td>99.6</td><td>100.0</td><td>99.73</td><td>57.05</td></tr><tr><td>SL-IT</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

Table 3: Same results as above with context length set to twice the maximum training length.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Context Length</td><td colspan="3">Essay-Word-Num</td><td colspan="3">Essay-Word-UUID</td><td colspan="3">Repeat-Word-Num</td></tr><tr><td>0</td><td>50</td><td>100</td><td>0</td><td>50</td><td>100</td><td>0</td><td>50</td><td>100</td></tr><tr><td rowspan="3">Mamba2</td><td>1024</td><td>86.0</td><td>73.6</td><td>82.0</td><td>78.0</td><td>70.8</td><td>80.8</td><td>77.6</td><td>70.4</td><td>55.2</td></tr><tr><td>2048</td><td>45.6</td><td>20.8</td><td>65.2</td><td>49.6</td><td>20.4</td><td>66.0</td><td>82.0</td><td>76.0</td><td>66.8</td></tr><tr><td>4096</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>80.4</td><td>56.8</td><td>65.6</td></tr><tr><td rowspan="3">M2A</td><td>1024</td><td>37.2</td><td>28.0</td><td>48.0</td><td>39.2</td><td>26.8</td><td>48.0</td><td>47.2</td><td>44.4</td><td>70.0</td></tr><tr><td>2048</td><td>41.6</td><td>27.6</td><td>39.6</td><td>42.4</td><td>28.4</td><td>30.8</td><td>36.8</td><td>32.0</td><td>63.2</td></tr><tr><td>4096</td><td>29.2</td><td>25.6</td><td>59.2</td><td>27.6</td><td>28.0</td><td>58.0</td><td>59.6</td><td>32.8</td><td>82.8</td></tr><tr><td rowspan="3">TPP</td><td>1024</td><td>52.0</td><td>36.0</td><td>47.6</td><td>58.8</td><td>34.4</td><td>50.4</td><td>81.6</td><td>33.2</td><td>58.4</td></tr><tr><td>2048</td><td>51.6</td><td>29.6</td><td>62.4</td><td>44.8</td><td>36.0</td><td>55.6</td><td>63.6</td><td>13.2</td><td>49.2</td></tr><tr><td>4096</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

Table 4: Results on needle-in-a-haystack task where the position of a single needle is placed at the beginning, end or middle of the haystack while the types of each component varies. Context length is set to the maximum on which the models were trained.

## 5 Discussion

All models have limits. Our first observation is that regardless of the model, performance drops steeply upon testing with sequences that are longer than what the model was initially trained on. This is made clear in Table 1, where the performance decline is greatest once the evaluated sequences are longer than the training context (with the mild exception of RWKV which demonstrates approximately linear degredation as the sequences progressively double in length). However, an important observation is that linear sequence models do appear to extrapolate slightly better than pure-attention models, whose performance drop to near 0 performance upon the increase, as these models do show non-trivial accuracy even when evaluated on the longer sequences. This distinction is less clear when comparing between pure linear sequence models and hybrid models which alternate between sequence-model layers and attention layers, as there is no explicit pattern as to when one class will perform better on one length or another.

Being lost in the middle is a common event. Being lost in the middle, whereby models have difficulty recalling relevant information located positionally in the middle of long contexts (Liu et al., 2024), has been observed as a common limitation among attention-based models. In Table 2, this appears to be a common feature among all models we test, since all classes of models see increasing drops in performance as the information is more closely located at the center of the sequence. This suggests that despite their long-context modeling ability, recurrent models cannot effectively reason over their entire context window when prompted. However, when extending beyond the length of the training context (Table 3), there is less consistency in the pattern, but models generally remain more capable when information is close to either end of the sequence. Moreover, while Mamba models still appear lost-in-the-middle, other recurrent models such as RecurrentGemma and RWKV have no clear depth-to-performance trends, further bringing into question their general long-context modeling abilities and how they function.

Extrapolation can inconsistent. Furthermore, extrapolation can be inconsistent based on characteristics of the model as well as the data. In Table 4, we can first note that, depending on the data format of the haystack, key, and value to be retrieved, the performance of each model varies significantly even with the same task template, context length and needle position. Furthermore, extrapolation varies based on the model as these characteristics change. For example, pure sequence layers (Mamba2) appear to only extrapolate when the haystack is a repeated sequence and the retrived value is a number related to a key word. Upon changing the haystack to be essays, extrapolation craters, and the model fails. An equally trained hybrid model (M2A) can meanwhile always extrapolate to some degree, but performance on sequences up to the training context length appears to compare much worse. Pure attention (TPP) meanwhile performs favorably only when evaluating on the extact training context length under specific data formats, but otherwise underwhelms.

## 6 Conclusion

In this work, we conduct a comprehensive comparision between the long-sequence models and attention-based language models, showing that long-context abilities of such sequence models may hold from a theoretical perspective, they empirically still struggle in comparison to models that make no guarantees. This highlights the need to improve long sequence reasoning abilties not only for Transformer-based LLMs, but also SSMs and new classes of RNNs, which hopefully can serve as motivation to further analyze this topic.

## 7 Limitations

We limit ourself to a model size in which it is easy to compare models of various paradigms. As such, some perhaps more powerful models are not explored as the analysis between such models can become difficult due to multiple additional changing variables that can perhaps lead to incorrect or undersupported claims.

## 8 Ethical Concerns

This paper discusses how different types of language models behave on long-context data. It follows that mistakes in our methodology (both experimental and analytical) could lead to unsupported confidence or skepticism about LLMs. Though neither are unethical, unsupported confidence can be very dangerous. However, given that the overall claim is that LLMs should not be assumed to support context length that extend beyond what they have trained, regardless of their training data, we do not think this paper in itself could be misinterpreted for particularly dangerous outcomes.

As for model choices, we use publicly available models where the license agreements do not restrict what we can say about the model. This should give the reader confidence that our views are unbiased. This is unlike ChatGPT or GPT4, which include an unrestricted indemnity-clause in their license agreement, which could make us financially liable for damages.

## Acknowledgements

Jerry Huang is supported by a National Science and Engineering Research Council (NSERC) Canada

Graduate Scholarship, a Fonds de Recherche du Québec Nature et technologies (FRQNT) Training Scholarship and a Hydro-Québec Excellence Scholarship. The experiments were enabled in part by computational resources provided by Calcul Québec (calculquebec.ca). The author would like to thank Peng Lu and Qiuhao Zeng for useful discussions during the project.

## References

AI21. 2024. Introducing jamba: Ai21’s groundbreaking ssm-transformer model.

Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Ré. 2024. Zoology: Measuring and improving recall in efficient language models. In ICLR.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. In International Conference on Learning Representations.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. Preprint, arXiv:2308.14508.

Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. 2024. xlstm: Extended long shortterm memory. Preprint, arXiv:2405.04517.

Yoshua Bengio, Patrice Simard, and Paolo Frasconi. 1994. Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 5(2):157–166.

Aleksandar Botev, Soham De, Samuel L Smith, Anushan Fernando, George-Cristian Muraru, Ruba Haroun, Leonard Berrada, Razvan Pascanu, Pier Giuseppe Sessa, Robert Dadashi, Léonard Hussenot, Johan Ferret, Sertan Girgin, Olivier Bachem, Alek Andreev, Kathleen Kenealy, Thomas Mesnard, Cassidy Hardin, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Armand Joulin, Noah Fiedel, Evan Senter, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, David Budden, Arnaud Doucet, Sharad Vikram, Adam Paszke, Trevor Gale, Sebastian Borgeaud, Charlie Chen, Andy Brock, Antonia Paterson, Jenny Brennan, Meg Risdal, Raj Gundluru, Nesh Devanathan, Paul Mooney, Nilay Chauhan, Phil Culliton, Luiz GUStavo Martins, Elisa Bandy, David Huntsperger, Glenn Cameron, Arthur Zucker, Tris Warkentin, Ludovic Peran, Minh Giang, Zoubin Ghahramani, Clément Farabet, Koray Kavukcuoglu,

Demis Hassabis, Raia Hadsell, Yee Whye Teh, and Nando de Frietas. 2024. Recurrentgemma: Moving past transformers for efficient open language models. Preprint, arXiv:2404.07839.

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. 2019. Generating long sequences with sparse transformers. Preprint, arXiv:1904.10509.

Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder–decoder for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1724– 1734, Doha, Qatar. Association for Computational Linguistics.

Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. 2009. Introduction to Algorithms, Third Edition, 3rd edition. The MIT Press.

Tri Dao. 2024. Flashattention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations.

Tri Dao, Daniel Y Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems.

Tri Dao and Albert Gu. 2024. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In Forty-first International Conference on Machine Learning.

Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, Arnaud Doucet, David Budden, Yee Whye Teh, Razvan Pascanu, Nando De Freitas, and Caglar Gulcehre. 2024. Griffin: Mixing gated linear recurrences with local attention for efficient language models. Preprint, arXiv:2402.19427.

Zican Dong, Tianyi Tang, Junyi Li, Wayne Xin Zhao, and Ji-Rong Wen. 2024. BAMBOO: A comprehensive benchmark for evaluating long text modeling capacities of large language models. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 2086–2099, Torino, Italia. ELRA and ICCL.

Daniel Y Fu, Tri Dao, Khaled Kamal Saab, Armin W Thomas, Atri Rudra, and Christopher Ré. 2023. Hungry hungry hippos: Towards language modeling with state space models. In International Conference on Learning Representations.

Albert Gu and Tri Dao. 2024. Mamba: Linear-time sequence modeling with selective state spaces.

Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré. 2020. Hippo: Recurrent memory with optimal polynomial projections. In Advances in Neural Information Processing Systems, volume 33, pages 1474–1487. Curran Associates, Inc.

Albert Gu, Karan Goel, and Christopher Ré. 2022. Efficiently modeling long sequences with structured state spaces. In International Conference on Learning Representations.

Albert Gu, Isys Johnson, Karan Goel, Khaled Kamal Saab, Tri Dao, Atri Rudra, and Christopher Re. 2021. Combining recurrent, convolutional, and continuoustime models with linear state space layers. In Advances in Neural Information Processing Systems.

Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. 2024. LM-infinite: Simple on-the-fly length generalization for large language models.

Sepp. Hochreiter, Yoshua. Bengio, Paolo. Frasconi, and Jürgen Schmidhuber. 2001. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies. In S. C. Kremer and J. F. Kolen, editors, A Field Guide to Dynamical Recurrent Neural Networks. IEEE Press.

Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural Comput., 9(8):1735–1780.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. Ruler: What’s the real context size of your long-context language models? Preprint, arXiv:2404.06654.

Maor Ivgi, Uri Shaham, and Jonathan Berant. 2023. Efficient long-text understanding with short-text models. Transactions ofthe ACL, 11:284–299.

Samy Jelassi, David Brandfonbrener, Sham M. Kakade, and eran malach. 2024. Repeat after me: Transformers are better than state space models at copying. In Forty-first International Conference on Machine Learning.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Michael I. Jordan. 1986. Serial order: a parallel distributed processing approach. Technical report, University of California, San Diego: Institute for Cognitive Science.

Gregory Kamradt. 2023. Needle In A Haystack - pressure testing LLMs. Github.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020. Transformers are RNNs: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 5156–5165. PMLR.

Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. 2023. Loogle: Can long-context language models understand long contexts? Preprint, arXiv:2311.04939.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Associationfor Computational Linguistics, 12:157–173.

William Merrill, Jackson Petty, and Ashish Sabharwal. 2024. The illusion of state in state-space models. In Forty-first International Conference on Machine Learning.

William Merrill and Ashish Sabharwal. 2023. The parallelism tradeoff: Limitations of log-precision transformers. Transactions ofthe Associationfor Computational Linguistics, 11:531–545.

Amirkeivan Mohtashami and Martin Jaggi. 2023. Landmark attention: Random-access infinite context length for transformers. In Workshop on Efficient Systemsfor Foundation Models @ ICML2023.

Vincent Ng. 2010. Supervised noun phrase coreference research: The first fifteen years. In Proc. ofthe 48th Annual Meeting ofthe ACL.

Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. 2023. Resurrecting recurrent neural networks for long sequences. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 26670–26698. PMLR.

Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, and Samuel Bowman. 2022. QuALITY: Question answering with long input texts, yes! In Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5336–5358, Seattle, United States. Association for Computational Linguistics.

Jongho Park, Jaeseung Park, Zheyang Xiong, Nayoung Lee, Jaewoong Cho, Samet Oymak, Kangwook Lee, and Dimitris Papailiopoulos. 2024. Can mamba learn how to learn? a comparative study on in-context learning tasks. In Forty-first International Conference on Machine Learning.

Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. 2013. On the difficulty of training recurrent neural

networks. In Proceedings ofthe 30th International Conference on Machine Learning, volume 28 of Proceedings ofMachine Learning Research, pages 1310– 1318, Atlanta, Georgia, USA. PMLR.

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, Xingjian Du, Matteo Grella, Kranthi Gv, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartłomiej Koptyra, Hayden Lau, Jiaju Lin, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Johan Wind, Stanisław Wo´zniak, Zhenyuan Zhang, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. 2023. RWKV: Reinventing RNNs for the transformer era. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 14048–14077, Singapore. Association for Computational Linguistics.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2024. YaRN: Efficient context window extension of large language models. In International Conference on Learning Representations.

Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Ré. 2023. Hyena hierarchy: Towards larger convolutional language models. In Fortieth International Conference on Machine Learning.

Ofir Press, Noah Smith, and Mike Lewis. 2022. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations.

Zhen Qin, Songlin Yang, and Yiran Zhong. 2023. Hierarchically gated recurrent neural network for sequence modeling. In Thirty-seventh Conference on Neural Information Processing Systems.

Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know what you don’t know: Unanswerable questions for SQuAD. In Proc. of the 56th Annual Meeting of the ACL (Volume 2: Short Papers).

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’20, page 3505–3506, New York, NY, USA. Association for Computing Machinery.

David E. Rumelhart, James L. McClelland, and PDP Research Group. 1986. Parallel Distributed Processing, Volume 1: Explorations in the Microstructure of Cognition: Foundations. The MIT Press.

Uri Shaham, Elad Segal, Maor Ivgi, Avia Efrat, Ori Yoran, Adi Haviv, Ankit Gupta, Wenhan Xiong, Mor Geva, Jonathan Berant, and Omer Levy. 2022.

SCROLLS: Standardized CompaRison over long language sequences. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 12007–12021, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. 2023. Roformer: Enhanced transformer with rotary position embedding. Preprint, arXiv:2104.09864.

Qwen Team. 2024. Qwen2 technical report.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. Musique: Multihop questions via single-hop question composition. Transactions ofthe ACL, 10:539–554.

Ali Vardasbi, Telmo Pessoa Pires, Robin Schmidt, and Stephan Peitz. 2023. State spaces aren’t enough: Machine translation needs attention. In Proceedings of the 24th Annual Conference of the European Association for Machine Translation, pages 205–216, Tampere, Finland. European Association for Machine Translation.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert Gu, Ali Hatamizadeh, Sudhakar Singh, Deepak Narayanan, Garvit Kulshreshtha, Vartika Singh, Jared Casper, Jan Kautz, Mohammad Shoeybi, and Bryan Catanzaro. 2024. An empirical study of mambabased language models. Preprint, arXiv:2406.07887.

Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. 2024. Sheared LLaMA: Accelerating language model pre-training via structured pruning. In The Twelfth International Conference on Learning Representations.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In EMNLP.

## A Technical Implementation Details

## A.1 Models Used

Table 5 lists the public models we use for our experimentation.

## A.2 Computing Resources Used

All experiments were conduced using a single NVIDIA A100 80GB SXM GPU with 6 CPU worker cores. Experiments are run using PyTorch Version 2.2.0 and CUDA 11.8.

## B The RULER Benchmark

To conduct our study, we focus on the RULER benchmark (Hsieh et al., 2024), which comprises of tasks spanning across four categories: retrieval, multi-hop tracing, aggregation, and question answering. We use a publicly available repository<sup>2</sup> to generate evaluation examples based on specific input configurations (see Table 6 for example configurations) that define the length and complexity of each input. In RULER, the task complexity can be thought of as a function of the number of target output tokens and the signal-to-noise ratio in the context. For our experiments, we use the default set of tasks pre-defined by Hsieh et al. (2024).

## B.1 Retrieval: Needle-in-a-haystack (NIAH)

RULER includes multiple retrieval-based tasks, extending the vanilla NIAH test to evaluate models based to four NIAH tasks. The “needle” in each of these tasks is a key-value pair inserted into the “haystack” (long distractor texts). The query is located at the end of the sequence and serves as a cue for matching the keys in the context and subsequently retrieving the associated values.

• Single NIAH (S-NIAH): This comprises the standard/vanilla NIAH task where a single “needle” needs to be retrieved from the “haystack”. The query/key/value can take the form of words, numbers (7 digits), or UUIDs (32 digits). The “haystack” can be repeated noise sentences or Paul Graham essays (Kamradt, 2023).

• Multi-keys NIAH (MK-NIAH): Multiple “needles” are inserted into the “haystack”, and only one of them needs to be retrieved. The additional “needles” are hard distractors. The most challenging setting is a version where the “haystack” is filled with distractor needles.

• Multi-values NIAH (MV-NIAH): Multiple “needles” sharing the same key are inserted into the “haystack”. All values associated with the same key need to be retrieved.

• Multi-queries NIAH (MQ-NIAH): Multiple “needles” are inserted into the “haystack”. All “needles” with distinct keys need to be retrieved. This is the same multi-query associative recall task setup used by Arora et al. (2024). Together with MV-NIAH, these two tasks evaluate the retrieval capability without missing any critical information.

<table><tr><td>Model</td><td>Abbreviation</td><td>Public Model Name</td><td>HuggingFace Model</td></tr><tr><td>Mamba2</td><td>Mamba2</td><td>state-spaces/mamba2-2.7b</td><td>x</td></tr><tr><td>Mamba2Attention</td><td>M2A</td><td>state-spaces/mamba2attn-2.7b</td><td>x</td></tr><tr><td>Transformer++</td><td>TPP</td><td>state-spaces/transformerpp-2.7b</td><td>x</td></tr><tr><td>RWKV</td><td>RWKV</td><td>RWKV/rwkv-6-world-3b-v2.1</td><td>vvvvv</td></tr><tr><td>Sheared-LLaMA</td><td>SL</td><td>princeton-nlp/Sheared-LLaMA-2.7B</td><td></td></tr><tr><td>Sheared-LLaMA-ShareGPT</td><td>SL-IT</td><td>princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT</td><td></td></tr><tr><td>RecurrentGemma-2B</td><td>RG</td><td>google/recurrentgemma-2b</td><td></td></tr><tr><td>RecurrentGemma-2B-IT</td><td>RG-IT</td><td>google/recurrentgemma-2b-it</td><td></td></tr></table>

Table 5: Models used and public links to their weights.

<table><tr><td>Task</td><td>Configuration</td><td>Example</td></tr><tr><td>Single NIAH</td><td>type_key = word type_value = number</td><td>(essays) One of the special magic numbers for</td></tr><tr><td>(S-NIAH)</td><td>type_haystack = essay size_haystack ∝ context length</td><td>long-context is: 12345. What is the special magic number for</td></tr><tr><td></td><td></td><td>long-context mentioned in the provided text? Answer: 12345</td></tr><tr><td>Multi-keys NIAH</td><td>num keys = 2</td><td>(essays)</td></tr><tr><td>(MK-NIAH)</td><td>type_key = word type_value = number</td><td>One of the special magic numbers for long-context is: 12345.</td></tr><tr><td></td><td>type_haystack = essay</td><td>One of the special magic numbers for</td></tr><tr><td></td><td>size haystack ∝ context length</td><td>large-model is: 54321.</td></tr><tr><td></td><td></td><td>What is the special magic number for long-context mentioned in the provided</td></tr><tr><td></td><td></td><td>text? Answer: 12345</td></tr><tr><td>Multi-values NIAH</td><td>num_values = 2</td><td>(essays)</td></tr><tr><td>(MV-NIAH)</td><td>type_key = word type_value = number</td><td>One of the special magic numbers for long-context is: 12345.</td></tr><tr><td></td><td>type_haystack = essay size haystack ∝ context length</td><td>One of the special magic numbers for long-context is: 54321</td></tr><tr><td></td><td></td><td>What are all the special magic numbers</td></tr><tr><td></td><td></td><td>for long-context mentioned in the pro- vided text?</td></tr><tr><td>Multi-queries</td><td>num_queries = 2</td><td>Answer: 12345 54321 (essays)</td></tr><tr><td>NIAH (MQ-NIAH)</td><td>type_key = word</td><td>One of the special magic numbers for</td></tr><tr><td></td><td>type_value = number type_haystack = essay</td><td>long-context is: 12345. One of the special magic numbers for</td></tr><tr><td></td><td>size_haystack ∝ context length</td><td>large-model is: 54321.</td></tr><tr><td></td><td></td><td>What are all the special magic numbers for long-context and large-model men-</td></tr><tr><td></td><td></td><td>tioned in the provided text?</td></tr><tr><td>Variable</td><td>num_chains = 2</td><td>Answer: 12345 54321 (noises)</td></tr><tr><td>Tracking (VT)</td><td>num_hops = 2</td><td>VAR X1 = 12345 .....VAR Y1 = 54321</td></tr><tr><td></td><td>size_noises ∝ context length</td><td>VAR X2 = X1 VAR Y2 = Y1</td></tr><tr><td></td><td></td><td>VAR X3 = X2 VAR Y3 = Y2</td></tr><tr><td></td><td></td><td>Find all variables that are assigned the</td></tr><tr><td></td><td></td><td>value 12345.</td></tr><tr><td>Common Words</td><td></td><td>Answer: X1 X2 X3</td></tr><tr><td>Extraction</td><td></td><td>aaa bbb ccc aaa ddd eee ccc fff ggg hhh</td></tr><tr><td>(CWE)</td><td>freq_cw = 2, freq_ucw = 1</td><td>iiiiii</td></tr><tr><td></td><td>num_cw = 10</td><td>What are the 10 most common words in</td></tr><tr><td></td><td>num_ucw ∝ context length</td><td></td></tr><tr><td></td><td></td><td>the above list?</td></tr><tr><td></td><td></td><td>Answer: aaa ccc iii .....</td></tr><tr><td>Frequent Words</td><td></td><td></td></tr><tr><td></td><td>α = 2</td><td>aaa bbb ccc aaa ddd eee ccc fff gggaaa</td></tr><tr><td>Extraction</td><td></td><td></td></tr><tr><td></td><td>num_word ∝ context length</td><td>hhh aaa ccc iii iii</td></tr><tr><td>(FWE)</td><td></td><td></td></tr><tr><td></td><td></td><td>What are the 3 most frequently appeared</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>words in the above coded text?</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Answer: aaa ccc iii</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Question</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Document 1:</td></tr><tr><td></td><td>dataset = SQuAD</td><td>aaa</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Answering</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Document 2:</td></tr><tr><td></td><td></td><td>bbb</td></tr><tr><td></td><td>num document ∝ context length</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>(QA)</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

Table 6: Task examples with flexible configurations in RULER. Different colors highlight queries, keys, values, and distractors in each example. Examples are retrieved directly from Hsieh et al. (2024).

## B.2 Multi-hop Tracing: Variable Tracking (VT)

Variable tracking emulates a minimal coreference chain resolution (Ng, 2010) task. This task checks the behavior of tracking relevant co-occurrence patterns and drawing skipped connections within long input. Specifically, a variable $X _ { 1 }$ is initialized with a value V, followed by a linear chain of variable name binding statements (e.g., $X _ { 2 } = X _ { 1 } , X _ { 3 } =$ $X _ { 2 } , \ldots )$ , which are inserted at various positions of the input. The objective is to return all variable names pointing to V . The task complexity can be increased by adding more hops (i.e., the times of name binding) or more chains.

## B.3 Aggregation: Common Words (CWE) and Frequent Words Extraction (FWE)

In the common word extraction task (CWE), words are sampled from discrete uniform distributions, with the number of common words fixed while the number of uncommon words increases with the sequence length. In the frequent words extraction task (FWE), words are sampled from a Zeta distribution. A model needs to return the top-K frequent words in the context. In CWE, K is equivalent to the number of common words. In FWE, K is set to 3, as Hsieh et al. (2024) observe that increasing K leads to poor performance even at small context sizes for most models. The task complexity can be adjusted by varying the number of common words or the parameter of the Zeta distribution.

## B.4 Question Answering (QA)

The majority of existing QA datasets (Rajpurkar et al., 2018; Yang et al., 2018; Trivedi et al., 2022) are designed to answer questions based on short passages. These datasets can be extended to simulate long-context input by adding distracting information. In this task category, we insert the golden paragraphs (i.e., the paragraphs that contain answers) into paragraphs randomly sampled from the same dataset. This category is a real-world adaptation (Ivgi et al., 2023) of NIAH, where the question serves as the query, the golden paragraphs are the “needles”, and the distracting paragraphs form the “haystack”.

## C RULER Task Results

<table><tr><td>Length</td><td>1K</td><td>2K</td><td>4K</td><td>8K</td><td>16K</td><td>Average</td></tr><tr><td>Mamba2</td><td>66.8</td><td>71.6</td><td>60.0</td><td>62.4</td><td>0.0</td><td>52.16</td></tr><tr><td>M2A TPP</td><td>58.0 40.4</td><td>36.4 24.8</td><td>43.2 0.0</td><td>18.4 0.0</td><td>0.0 0.0</td><td>31.2 13.04</td></tr><tr><td>RG SL</td><td>100.0</td><td>100.0</td><td>52.0</td><td>24.8</td><td>10.0</td><td>57.36</td></tr><tr><td>RWKV</td><td>100.0 100.0</td><td>100.0 100.0</td><td>100.0 100.0</td><td>0.0 100.0</td><td>0.0 54.4</td><td>60.0 90.88</td></tr><tr><td>RG-IT</td><td>100.0</td><td>100.0</td><td></td><td></td><td></td><td></td></tr><tr><td>SL-IT</td><td>100.0</td><td>100.0</td><td>51.6 100.0</td><td>28.8 0.0</td><td>16.4 0.0</td><td>59.36 60.0</td></tr></table>

Table 7: Results on niah\_single\_1 task of RULER.

<table><tr><td>Length</td><td>1K</td><td>2K</td><td>4K</td><td>8K</td><td>16K</td><td>Average</td></tr><tr><td>Mamba2</td><td>62.4</td><td>60.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>24.56</td></tr><tr><td>M2A</td><td>33.2</td><td>34.8</td><td>9.6</td><td>4.8</td><td>0.0</td><td>16.48</td></tr><tr><td>TPP</td><td>50.8</td><td>48.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>19.76</td></tr><tr><td>RG SL</td><td>100.0 99.6</td><td>100.0 99.6</td><td>36.4 100.0</td><td>16.8 0.0</td><td>2.8 0.0</td><td>51.2 59.84</td></tr><tr><td>RWKV</td><td>100.0</td><td>100.0</td><td>53.6</td><td>30.4</td><td>9.6</td><td>58.72</td></tr><tr><td>RG-IT</td><td>100.0</td><td>100.0</td><td>55.2</td><td>24.4</td><td>12.8</td><td>58.48</td></tr><tr><td>SL-IT</td><td>100.0</td><td>100.0</td><td>100.0</td><td>0.0</td><td>0.0</td><td>60.0</td></tr></table>

Table 8: Results on niah\_single\_2 task of RULER.

<table><tr><td>Length</td><td>1K</td><td>2K</td><td>4K</td><td>8K</td><td>16K</td><td>Average</td></tr><tr><td>Mamba2</td><td>52.0</td><td>61.6</td><td>0.0</td><td>0.0</td><td>0.0</td><td>22.72</td></tr><tr><td>M2A</td><td>38.8</td><td>32.4</td><td>2.8</td><td>6.4</td><td>0.0</td><td>16.08</td></tr><tr><td>TPP</td><td>64.4</td><td>53.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>23.52</td></tr><tr><td>RG</td><td>100.0</td><td>100.0</td><td>39.2</td><td>16.8</td><td>8.4</td><td>52.88</td></tr><tr><td>SL</td><td>100.0</td><td>100.0</td><td>96.4</td><td>0.0</td><td>0.0</td><td>59.28</td></tr><tr><td>RWKV</td><td>99.2</td><td>96.4</td><td>15.2</td><td>19.6</td><td>4.4</td><td>46.96</td></tr><tr><td>RG-IT</td><td>100.0</td><td>100.0</td><td>53.6</td><td>24.0</td><td>13.6</td><td>58.24</td></tr><tr><td>SL-IT</td><td>100.0</td><td>99.6</td><td>99.6</td><td>0.0</td><td>0.0</td><td>59.84</td></tr></table>

Table 9: Results on niah\_single\_3 task of RULER.

<table><tr><td>Length</td><td>1K</td><td>2K</td><td>4K</td><td>8K</td><td>16K</td><td>Average</td></tr><tr><td>Mamba2</td><td>25.6</td><td>23.6</td><td>0.0</td><td>0.0</td><td>0.0</td><td>9.84</td></tr><tr><td>M2A</td><td>21.2</td><td>16.4</td><td>5.2</td><td>1.2</td><td>0.0</td><td>8.8</td></tr><tr><td>TPP</td><td>50.0</td><td>34.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>16.88</td></tr><tr><td>RG</td><td>98.8</td><td>98.8</td><td>23.2</td><td>15.6</td><td>4.4</td><td>48.16</td></tr><tr><td>SL</td><td>99.2</td><td>100.0</td><td>94.0</td><td>0.0</td><td>0.0</td><td>58.64</td></tr><tr><td>RWKV</td><td>81.6</td><td>64.0</td><td>30.4</td><td>18.0</td><td>11.2</td><td>41.04</td></tr><tr><td>RG-IT</td><td>99.2</td><td>100.0</td><td>36.8</td><td>17.6</td><td>11.2</td><td>52.96</td></tr><tr><td>SL-IT</td><td>99.6</td><td>99.2</td><td>98.0</td><td>0.0</td><td>0.0</td><td>59.36</td></tr></table>

Table 10: Results on niah\_multikey\_1 task of RULER.

<table><tr><td rowspan=1 colspan=1>Length</td><td rowspan=1 colspan=1>1K   2K   4K   8K  16K</td><td rowspan=1 colspan=1>Average</td></tr><tr><td rowspan=1 colspan=1>Mamba2</td><td rowspan=3 colspan=1>4.8   2.0  0.0  0.0  0.017.2  7.6  0.4  0.0  0.060.0 36.4 0.0  0.0  0.0</td><td rowspan=1 colspan=1>1.36</td></tr><tr><td rowspan=1 colspan=1>M2A</td><td rowspan=2 colspan=1>5.0419.28</td></tr><tr><td rowspan=1 colspan=1>TPP</td></tr><tr><td rowspan=1 colspan=1>RG</td><td rowspan=3 colspan=1>98.0 94.8 8.4  2.4  1.695.2  86.8 53.6 0.0  0.020.4  4.0  0.8  0.4  0.0</td><td rowspan=1 colspan=1>41.04</td></tr><tr><td rowspan=1 colspan=1>SL</td><td rowspan=2 colspan=1>47.125.12</td></tr><tr><td rowspan=1 colspan=1>RWKV</td></tr><tr><td rowspan=1 colspan=1>RG-IT</td><td rowspan=1 colspan=1>100.0 98.0 43.6 27.2 9.6</td><td rowspan=1 colspan=1>55.68</td></tr><tr><td rowspan=1 colspan=1>SL-IT</td><td rowspan=1 colspan=1>97.6 96.0 78.8 0.0  0.0</td><td rowspan=1 colspan=1>54.48</td></tr></table>

Table 11: Results on niah\_multikey\_2 task of RULER.

<table><tr><td>Length</td><td>1K</td><td>2K</td><td>4K</td><td>8K</td><td>16K</td><td>Average</td></tr><tr><td>Mamba2</td><td>14.4</td><td>2.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>3.36</td></tr><tr><td>M2A</td><td>17.6</td><td>12.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>6.0</td></tr><tr><td>TPP</td><td>61.2</td><td>56.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>23.52</td></tr><tr><td>RG</td><td>74.8</td><td>58.8</td><td>7.2</td><td>2.8</td><td>1.6</td><td>29.04</td></tr><tr><td>SL</td><td>96.4</td><td>46.4</td><td>38.8</td><td>0.0</td><td>0.0</td><td>36.32</td></tr><tr><td>RWKV</td><td>14.8</td><td>1.6</td><td>0.4</td><td>0.0</td><td>0.0</td><td>3.36</td></tr><tr><td>RG-IT</td><td>88.0</td><td>92.0</td><td>16.0</td><td>14.0</td><td>1.6</td><td>42.32</td></tr><tr><td>SL-IT</td><td>85.6</td><td>63.2</td><td>59.2</td><td>0.0</td><td>0.0</td><td>41.6</td></tr></table>

Table 12: Results on niah\_multikey\_3 task of RULER.

<table><tr><td>Length</td><td>1K</td><td>2K</td><td>4K</td><td>8K</td><td>16K Average</td></tr><tr><td>Mamba2</td><td>34.9</td><td>26.6</td><td>0.0</td><td>0.0 0.0</td><td>12.3</td></tr><tr><td>M2A</td><td>48.8</td><td>33.5</td><td>1.3</td><td>0.1 0.0</td><td>16.74</td></tr><tr><td>TPP</td><td>42.3</td><td>31.1</td><td>0.0</td><td>0.0 0.0</td><td>14.68</td></tr><tr><td>RG</td><td>97.4</td><td>95.1</td><td>14.7 3.3</td><td>3.0</td><td>42.7</td></tr><tr><td>SL</td><td>100.0</td><td>82.5</td><td>44.0 0.0</td><td>0.0</td><td>45.3</td></tr><tr><td>RWKV</td><td>96.5</td><td>87.0</td><td>57.2 10.8</td><td>5.2</td><td>51.34</td></tr><tr><td>RG-IT</td><td>96.7</td><td>87.6</td><td>41.8 22.0</td><td>11.3</td><td>51.88</td></tr><tr><td>SL-IT</td><td>100.0</td><td>87.5</td><td>77.2 0.0</td><td>0.0</td><td>52.94</td></tr></table>

Table 13: Results on niah\_multivalue task of RULER.

<table><tr><td rowspan=1 colspan=1>Length</td><td rowspan=1 colspan=2>1K   2K   4K   8K  16K</td><td rowspan=1 colspan=1>Average</td></tr><tr><td rowspan=1 colspan=1>Mamba2</td><td rowspan=2 colspan=2>39.1 39.2  0.0  0.0  0.054.4 37.5  1.6  0.0  0.044.4 34.8  0.0  0.0  0.0</td><td rowspan=1 colspan=1>15.66</td></tr><tr><td rowspan=1 colspan=1>M2ATPP</td><td rowspan=1 colspan=1>18.715.84</td></tr><tr><td rowspan=1 colspan=1>RG</td><td rowspan=2 colspan=2>99.5 99.7 4.7  2.8  2.845.6  0.0  0.0</td><td rowspan=1 colspan=1>41.9</td></tr><tr><td rowspan=1 colspan=1>SL</td><td rowspan=1 colspan=1>98.8 80.8 45</td><td rowspan=1 colspan=1>45.04</td></tr><tr><td rowspan=1 colspan=1>RWKV</td><td rowspan=1 colspan=2>94.3 80.7 38.4 9.3  2.4</td><td rowspan=1 colspan=1>45.02</td></tr><tr><td rowspan=1 colspan=1>RG-IT</td><td rowspan=1 colspan=2>97.8 97.9 48.5 21.1 11.4</td><td rowspan=1 colspan=1>55.34</td></tr><tr><td rowspan=1 colspan=1>SL-IT</td><td rowspan=1 colspan=2>98.4 94.7 85.9 0.0  0.0</td><td rowspan=1 colspan=1>55.8</td></tr></table>

Table 14: Results on niah\_multiquery task of RULER.

<table><tr><td>Length</td><td>1K</td><td>2K</td><td>4K</td><td>8K</td><td>16K</td><td>Average</td></tr><tr><td>Mamba2 M2A</td><td>69.12 78.24</td><td>36.64 56.88</td><td>35.2 9.6</td><td>20.72 1.76</td><td>0.0 0.56</td><td>32.34 29.41</td></tr><tr><td>TPP</td><td>40.88</td><td>21.12</td><td>0.0</td><td>0.0</td><td>0.0</td><td>12.4</td></tr><tr><td>RG SL RWKV</td><td>98.0 98.16 68.56</td><td>75.52 81.68 47.76</td><td>0.0 19.36 20.08</td><td>0.0 0.0 6.88</td><td>0.0 0.0 10.95</td><td>34.7 39.84 30.85</td></tr><tr><td>RG-IT SL-IT</td><td>84.24 93.68</td><td>79.36</td><td>50.4</td><td>31.76</td><td>19.92</td><td>53.14 42.58</td></tr></table>

Table 15: Results on vt task of RULER.

Table 18: Results on qa\_1 task of RULER.
<table><tr><td>Length</td><td>1K</td><td>2K</td><td>4K</td><td>8K 16K</td><td>Average</td></tr><tr><td>Mamba2 M2A</td><td>25.2 33.6</td><td>24.4 35.6</td><td>18.4 18.0</td><td>0.0 0.4 6.8 3.6</td><td>13.68 19.52</td></tr><tr><td>TPP RG</td><td>37.2 26.8</td><td>36.4 15.6</td><td>2.8 31.2</td><td>0.8 0.4 6.8 8.8</td><td>15.52 17.84</td></tr><tr><td>SL</td><td>41.6</td><td>37.2</td><td>37.2</td><td>0.0 0.0</td><td>23.2</td></tr><tr><td>RWKV</td><td>46.4</td><td>35.6</td><td>30.8</td><td>21.2 18.4</td><td>30.48</td></tr><tr><td>RG-IT</td><td>74.0</td><td></td><td></td><td></td><td></td></tr><tr><td>SL-IT</td><td>54.4</td><td>66.8 56.4</td><td>58.4 55.6</td><td>10.0 9.6 0.0 0.0</td><td>43.76 33.28</td></tr></table>

<table><tr><td>Length</td><td>1K</td><td>2K</td><td>4K</td><td>8K</td><td>16K</td><td>Average</td></tr><tr><td>Mamba2 M2A TPP</td><td>28.52 26.48 30.32</td><td>14.72 15.24 17.8</td><td>4.08 3.04 0.64</td><td>0.16 5.92 0.0</td><td>0.12 0.8 0.04</td><td>9.52 10.3 9.76</td></tr><tr><td>RG SL RWKV</td><td>48.6 71.2 57.08</td><td>21.32 25.32 3.24</td><td>42.88 55.24 45.0</td><td>17.24 0.0 14.84</td><td>4.24 0.04 1.92</td><td>26.86 30.36 24.42</td></tr><tr><td>RG-IT SL-IT</td><td>55.4 78.96</td><td>4.56 18.64</td><td>17.4 57.2</td><td>3.24 0.0</td><td>0.2 0.0</td><td>16.16 30.96</td></tr></table>

Table 16: Results on cwe task of RULER.

<table><tr><td>Length</td><td>1K</td><td>2K</td><td>4K</td><td>8K</td><td>16K</td><td>Average</td></tr><tr><td>Mamba2</td><td>57.87</td><td>44.67</td><td>40.67</td><td>0.53</td><td>0.0</td><td>28.75</td></tr><tr><td>M2A TPP</td><td>59.73 59.6</td><td>53.73 56.4</td><td>58.0 0.13</td><td>52.0 0.0</td><td>39.6 0.0</td><td>52.61 23.23</td></tr><tr><td>RG SL</td><td>56.0 72.0</td><td>53.87</td><td>7.6</td><td>15.6</td><td>17.33</td><td>30.08</td></tr><tr><td>RWKV</td><td>74.67</td><td>38.67 67.47</td><td>45.07 68.0</td><td>0.0 56.67</td><td>0.0 43.42</td><td>31.15 62.05</td></tr><tr><td>RG-IT SL-IT</td><td>80.8</td><td>67.87</td><td>69.73</td><td>64.8</td><td>50.67</td><td>66.77</td></tr></table>

Table 17: Results on fwe task of RULER.

<table><tr><td rowspan=1 colspan=1>Length</td><td rowspan=1 colspan=2>1K  2K   4K  8K  16K</td><td rowspan=1 colspan=1>Average</td></tr><tr><td rowspan=1 colspan=1>Mamba2</td><td rowspan=1 colspan=2>20.0 20.0 10.4 0.8  0.8</td><td rowspan=1 colspan=1>10.4</td></tr><tr><td rowspan=1 colspan=1>M2A</td><td rowspan=2 colspan=2>21.6 23.2 14.8 4.0  0.824.4 26.8 0.4  0.0  0.0</td><td rowspan=1 colspan=1>12.88</td></tr><tr><td rowspan=1 colspan=1>TPP</td><td rowspan=1 colspan=1>10.32</td></tr><tr><td rowspan=1 colspan=1>RG</td><td rowspan=1 colspan=2>26.8 18.8 24.4 20.8 16.8</td><td rowspan=1 colspan=1>21.52</td></tr><tr><td rowspan=1 colspan=1>SL</td><td rowspan=1 colspan=1>24.8 29.6 29.</td><td rowspan=1 colspan=1>6 0.0  0.0</td><td rowspan=1 colspan=1>16.8</td></tr><tr><td rowspan=1 colspan=1>RWKV</td><td rowspan=1 colspan=2>31.6 30.8 27.2 20.4 17.6</td><td rowspan=1 colspan=1>25.52</td></tr><tr><td rowspan=1 colspan=1>RG-IT</td><td rowspan=1 colspan=2>37.2 38.8 33.2 25.6 16.0</td><td rowspan=1 colspan=1>30.16</td></tr><tr><td rowspan=1 colspan=1>SL-IT</td><td rowspan=1 colspan=2>34.0 37.6 38.4 0.0  0.0</td><td rowspan=1 colspan=1>22.0</td></tr></table>

Table 19: Results on qa\_2 task of RULER.