# MLD-EA: Check and Complete Narrative Coherence by Introducing Emotions and Actions

Jinming Zhang Yunfei Long University of Essex jz22273@essex.ac.uk yl20051@essex.ac.uk

## Abstract

Narrative understanding and story generation are critical challenges in natural language processing (NLP), with much of the existing research focused on summarization and questionanswering tasks. While previous studies have explored predicting plot endings and generating extended narratives, they often neglect the logical coherence within stories, leaving a significant gap in the field. To address this issue, we introduce the Missing Logic Detector by Emotion and Action (MLD-EA) model, which leverages large language models (LLMs) to identify narrative gaps and generate coherent sentences that integrate seamlessly with the story’s emotional and logical flow. The experi mental results demonstrate that the MLD-EA model enhances narrative understanding and story generation, highlighting LLMs’ potential as effective logic checkers in story writing with logical coherence and emotional consistency. This work fills a gap in NLP research and advances border goals of creating more sophisticated and reliable story-generation systems.

## 1 Introduction

Narrative understanding and story generation have been a compelling challenge in Natural Language Processing (NLP) for a long. They evolved from early rule-based systems with limited creativity to sophisticated models that generate rich, engaging narratives (Mooney and DeJong, 1985; Fan et al., 2018). Introducing Transformer (Vaswani, 2017) models like BART (Lewis et al., 2020) and large language models (LLMs) like ChatGPT (OpenAI, 2022) revolutionized this task by utilizing advanced architectures to capture in-detailed dependencies.

Many previous studies have focused on tasks like summarizing (Awasthi et al., 2021; Jin et al., 2024), sentiment analysis (Lu et al., 2023; Zhao et al., 2025; Lu et al., 2025) and question-answering (QA) (Zhuang et al., 2024; Huang et al., 2024a). While previous story generation research often centered on predicting plot endings or crafting long narratives (Guan et al., 2020; Li et al., 2022). However, in general, story writing frequently needs to pay more attention to maintaining logical coherence (Oatley, 2002; Currie and Jureidini, 2004).

Not surprisingly, some recent works lead LLMs to maintain narrative coherence in different ways with effective results (Zhao et al., 2023; Wang et al., 2023). However, most of those works focus on continuously writing coherency stories by LLMs (Guan et al., 2021). There is still a gap in detecting the logical coherence in the narratives.

To address this gap, our approach focuses on the observable actions of characters rather than delving into their deeper motivations. This choice stems from the understanding that actual actions have a more immediate and direct impact on emotions, and conversely, emotions are often the driving force behind tangible actions (Zhu and Thagard, 2002; Döring, 2003). The James-Lange theory of emotion in psychology posits that physiological responses to a situation—such as a racing heart or clenched fists—occur first and then lead to the subjective experience of emotion (Cannon, 1927). This suggests that an observable action (like a person slamming a door) can directly trigger an emotional response (such as anger or frustration). Similarly, the cognitive-behavioral theory emphasizes that behaviors (actions) and emotions are closely linked, where a behavior change can directly influence emotional states, and vice versa (Maslow, 1943; Eisenberg, 2014; Leahy et al., 2022).

By prioritizing the direct interplay between observable actions and emotions, we aim to capture the essence of narrative logic in a way that reflects these well-documented psychological principles (Carver et al., 2000). This approach is supported by extensive psychological studies that emphasize the strong correlation between actions and emotional responses, such as how consistent patterns of behavior can shape long-term emotional states, as seen in theories of learned helplessness or social learning (Bandura, 1977).

In this study, we introduce the Missing Logic Detector by Emotion and Action (MLD-EA), a LLM-based model designed to identify gaps in narrative logic and generate missing plot elements that are coherent both logically and emotionally. By incorporating the relationship between actions and emotions, MLD-EA aims to enhance the logical structure of narratives. Experimental results demonstrate that our models can produce more believable and emotionally coherent stories by aligning narrative generation with these psychological insights. Our model improves narrative understanding and story generation, underscoring the potential of LLMs as story generators and powerful logic checkers in the creative process.

The main contributions of our work can be briefly summarized as follows: 1), We propose a novel task of narrative logic detection. 2), By grounding our model in cognitive-behavioral theories, we highlight how emotions directly interact with actions, leading to better narrative understanding and generation. 3), Experiments have shown that our MLD-EA model has achieved superior results in most aspects, including narrative logic checking with involved characters’ emotions and actions and missing plot completeness. Also, we demonstrate the importance of behavior and emotion in story logic detection and generation.

Leveraging this interaction between actions and emotions to assess and generate story logic more efficiently and accurately mirrors the natural causeand-effect relationships in human behavior.

## 2 Related Works

Several innovative approaches have been developed to enhance AI-generated narratives’ logical coherence, emotional depth in narrative understanding, and story generation within NLP. Paul and Frank (2021) framework introduces a recursive inference strategy that dynamically generates contextualized rules to guide narrative completion, focusing on maintaining coherence and logical flow throughout the story. Similarly, the CHAE model (Wang et al., 2022) offers fine-grained control over narrative elements, creating customized stories with specific characters, actions, and emotions, enhancing the personalization and richness of the narratives. Similarly, the COMMA (Xie et al., 2022) explores the relationships among motivations, emotions, and actions, providing a cognitive framework that deepens the understanding of narrative construction by modeling these interrelated factors. However, these traditional models often struggle to consistently integrate actions and emotions to maintain logical coherence throughout the entire narrative, leading to disjointed or emotionally inconsistent storylines when handling more complex plots (Kambhampati et al., 2024). Additionally, they may lack the flexibility to dynamically understand nuanced shifts in a character’s behavior or emotional progression.

![](images/4f412274a5395d1794d9b78d146e8fbd7f41f5836d2636e3cb2cc69412161b03.jpg)  
Figure 1: A task example. "Identify logical coherence with actions and emotions" is checking the logical coherence guided by the cognitive-behavioral theory.

Exploring LLMs, cognitive frameworks, and hybrid planning strategies has paved the way for more engaging and human-like stories. Alvarez (2023) used ChatGPT in interpreting narrative structures, which further extends the potential for generating stories based on predefined structures, offering new methods for narrative development. Notably, approaches such as iterative prompting-based planning for suspenseful story generation (Xie and Riedl, 2024), the combination of symbolic planning with neural models (Farrell and Ware, 2024), and the SWAG method (Patel et al., 2024), which utilizes action guidance in storytelling, have significantly improved the quality and engagement of AI-generated narratives. Additionally, comprehensive evaluations like "The Next Chapter" (Xie et al., 2023) and knowledge-enhanced pre-training models (Guan et al., 2020) have shown that LLMs can produce stories of high quality, sometimes approaching the level of human authors. LLMs often struggle to maintain consistent plots on generation, but they cannot check their generated stories by themselves (Huang et al., 2024b). In our approach, MLD-EA is able to find such logical loopholes by introducing the interaction between emotions and actions to keep stories coherent.

## 3 Problem Definition

The primary goal of MLD-EA is to identify whether the input story is logically completed, as Figure 1 shows. We divided the model into four main sub-tasks: 1), abstracting characters’ actions. 2), classifying their emotions for each sentence. 3), then locate the logical loopholes of the narrative in which the missing part should be inserted. 4), we complete the tale consistently by predicting the characters’ actions and emotions. Thereby preserving the narrative’s overall coherence and logical structure. The tasks are defined as follows:

For any input n sentences story $\begin{array} { r l } { ( S } & { { } = } \end{array}$ $s _ { 1 } , \cdots , s _ { n } )$ with m characters appeared in this story $( C = c _ { 1 } , \cdots , c _ { m } )$ , MLD-EA abstract characters’ actions a and classify their emotions e for each sentence, denoted as $\{ ( c , s )  ( a ( c , s ) , e ( c , s ) )$ $c \in C , s \in S \}$ , where $a ( c , s )$ represents the action of character c in sentence s and $e ( c , s )$ represents the emotion of character c in sentence s.

Sequently, given the story and characters’ actions and emotions, MLD-EA will use the provided information to review the story and find inconsistencies. Notably, our task is to find the logic gap in the inner story. We suppose the start and end of the story are always complete. The process involves identifying points where the characters’ actions or emotions exhibit abrupt changes that the preceding context cannot logically explain. After that, MLD-EA outputs the index k which the missing part should be inserted before it:

$$
k = \left\{ \begin{array} { l l } { 1 < k < n } & { i f t h e r e ~ i s ~ a } \\ & { m i s s i n g ~ s e n t e n c e } \\ { - 1 } & { o t h e r w i s e . } \end{array} \right.\tag{1}
$$

Formally, if MLD-EA identifies a logic gap before a specific place k in the story, it proceeds by predicting the most likely actions $\hat { a } ( c , s _ { k } )$ and emotions $\hat { e } ( c , s _ { k } )$ by using the sequence of preceding $( \{ a ( c , s _ { k - 1 } ) , e ( c , s _ { k - 1 } ) \} )$ and succeeding actions and emotions $( \{ a ( c , s _ { k } ) , e ( c , s _ { k } ) \} )$ ). Then MLD-EA estimates the most coherent missing sentence $s _ { k }$ according to $\hat { a } ( c , s _ { k } )$ and $\hat { e } ( c , s _ { k } )$ .

## 4 Methodology

In this section, we will provide a detailed methodology for each module within our MLD-EA model. The model architecture is shown in Figure 2.

## 4.1 Action Abstraction

The action abstraction module is designed to extract and abstract actions performed by characters in a given sentence, playing a crucial role in analyzing narrative structures and identifying logic gaps. The process begins with the model receiving a sentence $s ,$ a list of characters $C = \{ c _ { 1 } , c _ { 2 } , \dots , c _ { m } \}$ , and the story’s context S for reference.

Guided by prompt engineering (details in Appendix E), MLD-EA processes each sentence to identify and represent the actions performed by the characters as flowing:

For each character c in the characters list C, the model outputs an action in the following format: <c>Action(Target, Object)</c>, where c represents the character acting; Action denotes the action the character performs; Target is the target of the action (who or what the action is directed towards); Object specifies any object associated with the action (if applicable). If a character c does not perform any action in the sentence s, the model needs to output: <c>None</c>.

## 4.2 Emotion Classification

The emotion classification module in the MLD-EA categorizes characters’ emotions based on given sentences. This classification is based on eight basic emotion types from Plutchik’s model (Plutchik, 2001) —joy, trust, fear, surprise, sadness, disgust, anger, and anticipation—plus an additional "none" category for cases where no emotion is detected.

Before classifying emotions, the model first checks whether each character c in the list $C =$ $\{ c _ { 1 } , c _ { 2 } , \ldots , c _ { m } \}$ is affected by the events described in each sentence s. If the model determines that a character c is not affected, the emotion for that character is classified as none. In addition to the emotion classification, the model also outputs whether or not each character is affected by the sentence.

The model’s output for each character c includes the result of the $\mathbf { \lambda } ^ { \prime } a f f e c t e d ^ { \prime }$ and the emotion classification in <c>(Affected, $e ( c , s ) ) { < } / { \mathsf { c } } { > }$ , where

![](images/f3687b751b568a401c4d7c8f5449c443658e05234d8cbded549067836e717194.jpg)  
Figure 2: MLD-EA model overview. Each Input Story contains n sentences and m characters, which have a missing sentence $s _ { k }$ before index k. $e ( c , s )$ and $a ( c , s )$ denote the character’s emotion and action in the sentence, respectively; eˆ and aˆ denotes the predicted emotion and action.

Affected is a boolean value indicating whether the character c is affected by any event in the sentence s and $e ( c , s )$ represents the emotion associated with the character in sentence s, where $e \in$ {joy, trust, fear, surprise, sadness, disgust, anger, anticipation, none}.

## 4.3 Narrative Logic Checker By Characters’ Emotion and Action

The narrative logic checker component focuses on detecting potential gaps in the narrative by analyzing the relationship between characters’ actions and emotions. This process is grounded in the outputs from the previous modules: action abstraction and emotion classification. The prediction is based on detecting disruptions or inconsistencies in each character’s expected flow of actions and emotions.

Several key principles in behavior research (Cannon, 1927; Zhu and Thagard, 2002) guide this process: 1), emotions often drive actions. 2), actions can influence subsequent emotions. 3), and some actions directly reflect the character’s current emotional state, and vice versa.

MLD-EA then predicts the missing sentence index k, which is determined by evaluating the continuity and logical consistency of the sequences with the interaction of characters’ actions and emotions:

$$
\left( E , A \right) = \sum _ { s \in S , c \in C } \left[ e ( c , s ) , a ( c , s ) \right] ,\tag{2}
$$

$$
k = \mathrm { I n f } _ { I n d e x } \left[ ( S \oplus ( E , A ) ) , C \right] ,\tag{3}
$$

where Inf $\dot { \boldsymbol { \cdot } } _ { I n d e x }$ represent the model inference of missing sentence index prediction. A significant

deviation from expected values suggests a missing sentence, and k identifies the position where this sentence should be inserted.

## 4.4 Action/ Emotion prediction and sentence generation

Following the identification of the missing sentence index by analyzing characters’ actions and emotions, the next crucial step in the MLD-EA framework is to predict the actions and emotions of the missing sentence and subsequently generate the sentence. This process is essential to ensure the narrative remains coherent and logically consistent. The focus here is on the immediate context surrounding the predicted index. By examining the sequences of preceding actions and emotions and succeeding actions and emotions, the model estimates the most coherent actions $\hat { a } ( c , s _ { k } )$ and emotions $\hat { e } ( c , s _ { k } )$ for the missing sentence $s _ { k } \mathrm { . }$

$$
\begin{array} { r l r } { \ } & { { } } & { \left[ \hat { a } , \hat { e } \right] = \mathrm { I n f } _ { e a p } \left[ ( a ( c , s _ { k - 1 } ) , e ( c , s _ { k - 1 } ) ) , \right. } \\ { } & { { } } & { \left. ~ ( a ( c , s _ { k } ) , e ( c , s _ { k } ) ) \right] , } \end{array}\tag{4}
$$

where $\mathrm { I n f } _ { e a p }$ means the model inference of emotion and action prediction for the missing sentence. Once these predictions are made, the model generates a sentence to fill the identified gap:

$$
\begin{array} { r } { s _ { k } = \operatorname* { I n f } _ { g e n } ( S , C , k , ( \hat { a } , \hat { e } ) ) , } \end{array}\tag{5}
$$

where $\operatorname { I n f } _ { g e n }$ is a zero-shot inferring. This generated sentence encapsulates the character’s possible emotion and action, thereby maintaining the story’s coherence and flow and completing the narrative.

## 5 Experiment

## 5.1 Data

We use the Story Commonsense dataset for our task, which contains 4853 five-sentence stories with labeled emotions and motivation for characters (Rashkin et al., 2018). We only take the stories with labeled emotion because the labeled motivations are based on Maslow’s needs (Maslow, 1943) and Reiss’ motives (Reiss, 2004) theory, which are focused on the deeper motivation, not actual actions. By excluding motivations, which are more abstract and theoretical in nature, the analysis remains more grounded in observable narrative events, avoiding complexities that may not directly influence the characters’ visible actions. This also ensures that the model can better focus on the emotional states that drive the characters’ responses, making it easier to align predictions with surface-level events in the story. We then divided the data into 8:1:1 for training, validation, and testing.

To follow the task of emotion classification in section 4.2 and the task of narrative logic checker in section 4.3, we consolidate the characters’ emotions into a single tag by selecting the one with the highest confidence, as determined by three annotators in the original dataset. The details of choosing the missing sentence are in Appendix A.

## 5.2 Selected Baselines

We compare MLD-EA with the following baselines trained by different strategies and datasets:

Llama3-8B-Instruct (AI@Meta, 2024): Meta’s Llama3-8B-Instruct model is a cutting-edge LLM renowned for its exceptional ability to follow instructions meticulously. It is adept at crafting stories that are not only imaginative but also adhere to logical structures and factual integrity.

Gemma2-2B-it (Team, 2024): Gemma2-2B-it is a nimble and efficient model that packs a punch regarding text generation capabilities from Google. Despite its smaller size than some of its peers, it demonstrates remarkable skill in spinning engaging stories that captivate audiences.

Gemma2-9B-it (Team, 2024): Gemma2-9B-it is a larger version of Gemma2-2B-it. With a more vast dataset and bigger model size, it generates intricate and vivid stories rich in detail and depth.

We selected these particular models as baselines for several key reasons: 1) To the best of our knowledge, no prior research has focused on identifying logical gaps or inconsistencies at the sentence level within stories. This novel focus makes it difficult to directly compare our approach to existing studies. 2) While previous works on story generation have primarily relied on pre-trained models such as BERT and GPT-2 (Wang et al., 2022; Paul and Frank, 2021), our study specifically aims to evaluate the capabilities of newer LLMs. The baselines we selected models are all modern LLMs known for their advanced narrative understanding abilities. These models are particularly well-suited for complex tasks related to narrative. 3) We intentionally included models of different sizes and architectures to provide a comprehensive evaluation. This range allows us to compare varying complex models to understand how size and dataset diversity impact logical story generation.

## 5.3 Implement Setups

MLD-EA is built based on Llama3-8B-Instruct (AI@Meta, 2024) using the Huggingface’s libraries<sup>1</sup> (Wolf, 2019) and use Llama-Factory (Zheng et al., 2024) for supervised fine-tuning (Gunel et al., 2020). We use LoRA (Hu et al., 2021) to fine-tune our model. Please see Appendix B for hyper-parameters details and Appendix E for prompts technics we used and prompt templates.

We compute the micro-averaged result of all baselines by the same zero-shot (Wei et al., 2021), one-shot, and few-shot (Brown, 2020) prompts with original input labels from the dataset. All experiments run on two RTX 4090 24GB GPUs.

## 5.4 Evaluation Metrics

We use the following metrics to evaluate MLD-EA performance on the different sub-tasks:

(1) Both BLEU-1,2 (Papineni et al., 2002) and ROUGE-L (Lin, 2004) are used for evaluating the action abstraction task.

(2) We compute the micro-average Precision, Recall, and F1 score for each tag to show the accuracy of emotion classification.

(3) The micro-average Precision, Recall, and F1 score are also applied to evaluate the accuracy of the narrative logic checker on each candidate place.

(4) For final generation task based on predicted emotions and actions, we use BLEU-1,2,4, ROUGE-1,2,L and BERTScore<sup>2</sup> (Zhang et al.,

2019) to measure the similarity of candidate sentences and reference sentences. Furthermore, a Valence-Arousal-Dominance (VAD) model (Warriner et al., 2013) is used in psychology to describe and measure human emotions. These three dimensions are often used to provide a more comprehensive understanding of emotional states, as they capture different aspects of how emotions are experienced and expressed. We use a developed VAD model (Plisiecki and Sobieszek, 2024) to model the gap between candidate sentences and reference sentences. Also, Plisiecki and Sobieszek (2024) add Age of Acquisition (AoA) and Concreteness as important features in their VAD. AoA refers to the age at which a person learns a particular word or concept. Concreteness measures how tangible or perceptible through the senses a word is.

<table><tr><td></td><td>BLEU-1</td><td>BLEU-2</td><td>ROUGE-L</td></tr><tr><td>T2Act2T</td><td>40.94</td><td>34.82</td><td>53.67</td></tr></table>

Table 1: Result of Action Abstraction

<table><tr><td>Model</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>NPN (Rashkin et al., 2018)</td><td>24.33</td><td>40.10</td><td>30.29</td></tr><tr><td>Llama3-8B-Instruct</td><td>36.20</td><td>35.51</td><td>35.23</td></tr><tr><td>Ours</td><td>43.55</td><td>42.68</td><td>42.98</td></tr><tr><td>Ours-affected</td><td>48.51</td><td>50.33</td><td>49.03</td></tr></table>

Table 2: Result of Emotion Classification. The best performance is highlighted in bold, where ’Ours-affected means we consider the ’affected’ features during classification, and the affective features denote whether a character is influenced by any emotion.

## 6 Results and Analysis

## 6.1 Action Abstraction and Emotion Classification

The action abstraction has summarized the key concept from the original sentence as open-text, so we evaluate it by creating a simple process called ’Text to Action to Text (T2Act2T)’. T2Act2T takes the abstracted actions at first, and then it generates a new sentence only based on the abstracted actions. In the end, we compare the original sentence with the generated sentence to see how much information remained during the MLD-EA’s action abstraction module. Table 1 shows the result between original sentence and new sentence, which illustrates the degree of information kept by our method.

We give results for emotion classification in Table 2. Our model performs best compared to

Llama3-8B-Instruct baseline and a developed NPN model (Bosselut et al., 2017) in ROCStories dataset (Rashkin et al., 2018). After fine-tuning, the model archives a significant improvement in emotion classification. Also, when incorporating the ’affected’ feature to detect whether any emotion influences a character, our model attains an impressive F1 score of 88.51 on evaluating the accuracy of ’affected’, respectively. Our findings suggest that including features that account for emotional impact can dramatically improve classification performance, which has implications for various applications in natural language processing.

Furthermore, the partition relationship between the number of labels and their classified accuracy is in Figure 3. Classes with fewer instances show lower accuracy, indicating a need for better representation or enhanced feature engineering to improve performance across less frequent emotions.

![](images/2f32d3db79f3125baa8d374af054972207d2435801b107900df7046627928448.jpg)  
Figure 3: Emotion classification. This figure illustrates the relationship between the number of instances for each emotion class and the corresponding classification accuracy. Classes with more instances, such as ’joy, exhibit higher classification accuracy compared to less frequent classes like ’disgust’ and ’trust,’ reflecting the potential influence of data imbalance on performance.

## 6.2 Narrative Logic Checker

Table 3 presents the results of the narrative logic checker on predicting the index of missing part, which evaluates our model against various baselines both with and without incorporating actions and emotions<sup>3</sup>. MLD-EA model consistently outperforms all baselines across different sentence insertion points. Notably, including actions and emotions significantly improves the micro-averaged F1 scores for all baseline models. Specifically, when the story is complete (k = −1), there is a marked improvement in F1 scores for each baseline model, underscoring the critical role that action and emotion play in maintaining story logic.

<table><tr><td rowspan="2">Model</td><td colspan="3"> $k { = } { \bf - } { \bf 1 }$ </td><td colspan="3"> $\scriptstyle { \overline { { k = 2 } } }$ </td><td colspan="3"> $k { = } 3$ </td><td colspan="3"> $\overline { { k { = } 4 } }$ </td><td colspan="3"></td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td> $\mathbf { A v } \mathbf { g }$  R</td><td>F1</td></tr><tr><td>Without EA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llama3-8B-Instruct</td><td>0.00</td><td>0.00</td><td>0.00</td><td>17.90</td><td>64.43</td><td>24.80</td><td>89.72</td><td>29.47</td><td>44.29</td><td>14.04</td><td>57.82</td><td>19.47</td><td>30.41</td><td>37.93</td><td>22.14</td></tr><tr><td>Gemma2-2B-it</td><td>0.00</td><td>0.00</td><td>0.00</td><td>2.01</td><td>38.46</td><td>3.81</td><td>86.47</td><td>28.06</td><td>42.31</td><td>7.68</td><td>36.24</td><td>12.42</td><td>24.04</td><td>25.69</td><td>14.64</td></tr><tr><td>Gemma2-9B-it</td><td>22.44</td><td>21.73</td><td>21.79</td><td>23.94</td><td>83.03</td><td>29.37</td><td>81.45</td><td>34.93</td><td>48.28</td><td>31.58</td><td>66.97</td><td>41.23</td><td>39.85</td><td>51.66</td><td>35.17</td></tr><tr><td>With EA*</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llama3-8B-Instruct*</td><td>0.64</td><td>11.11</td><td>1.21</td><td>32.44</td><td>48.25</td><td>27.86</td><td>56.39</td><td>32.11</td><td>40.75</td><td>39.47</td><td>53.98</td><td>35.27</td><td>32.24</td><td>36.11</td><td>26.27</td></tr><tr><td>Gemma2-2B-it*</td><td>15.38</td><td>10.74</td><td>9.53</td><td>60.18</td><td>34.29</td><td>43.67</td><td>29.32</td><td>27.67</td><td>26.88</td><td>6.36</td><td>60.13</td><td>10.56</td><td>27.81</td><td>33.21</td><td>22.66</td></tr><tr><td>Gemma2-9B-it*</td><td>29.49</td><td>37.83</td><td>28.62</td><td>24.61</td><td>54.55</td><td>30.51</td><td>73.18</td><td>34.58</td><td>46.36</td><td>38.60</td><td>67.50</td><td>48.09</td><td>41.47</td><td>48.62</td><td>38.39</td></tr><tr><td>MLD-EA (Ours)</td><td>93.44</td><td>100.00</td><td>96.61</td><td>73.08</td><td>74.03</td><td>73.55</td><td>82.93</td><td>56.67</td><td>67.33</td><td>54.79</td><td>85.10</td><td>66.67</td><td>76.06</td><td>78.95</td><td>76.04</td></tr></table>

Table 3: Result of Narrative Logic Checker on predicting missing sentence position. The best performance on average is highlighted in bold. k=-1: The input story is completed; $k { = } 2 , 3 , 4 ;$ The missing one should be inserted before index[2,3,4], where story’s index starts at 1; Avg: the Micro-average score of all index’s F1 score; Without EA: prediction without involving emotions and actions. With EA and \*: prediction involving emotions and actions.
<table><tr><td rowspan="2">Model</td><td colspan="3">BLEU</td><td colspan="3">ROUGE</td><td colspan="3">BERTScore</td></tr><tr><td>-1</td><td>-2</td><td>-4</td><td>-1</td><td>-2</td><td>-L</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>Without EA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llama3-8B-Instruct</td><td>33.77</td><td>4.05</td><td>0.28</td><td>25.98</td><td>5.56</td><td>22.27</td><td>77.66</td><td>79.03</td><td>78.33</td></tr><tr><td>Gemma2-2B-it</td><td>30.31</td><td>2.88</td><td>0.15</td><td>23.36</td><td>3.92</td><td>20.19</td><td>76.91</td><td>78.36</td><td>77.67</td></tr><tr><td>Gemma2-9B-it</td><td>33.82</td><td>3.38</td><td>0.14</td><td>24.42</td><td>4.03</td><td>20.84</td><td>77.76</td><td>78.94</td><td>78.34</td></tr><tr><td>With EA*</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llama3-8B-Instruct*</td><td>36.29</td><td>5.83</td><td>0.54</td><td>28.18</td><td>7.18</td><td>23.99</td><td>77.59</td><td>78.98</td><td>78.27</td></tr><tr><td>Llama3-8B-Instruct干*</td><td>43.68</td><td>12.15</td><td>2.67</td><td>34.77</td><td>14.23</td><td>31.13</td><td>77.91</td><td>79.34</td><td>78.61</td></tr><tr><td>Gemma2-2B-it*</td><td>33.74</td><td>6.35</td><td>1.84</td><td>28.30</td><td>8.66</td><td>25.52</td><td>76.59</td><td>78.52</td><td>77.54</td></tr><tr><td> $\mathrm { G e m m a 2 { - } 2 B { - } i t } ^ { \mp * }$ </td><td>35.98</td><td>7.80</td><td>2.42</td><td>31.03</td><td>11.37</td><td>27.35</td><td>76.54</td><td>78.25</td><td>77.37</td></tr><tr><td>Gemma2  $- 9 \mathrm { B } { - } \mathrm { i t } ^ { * }$ </td><td>37.27</td><td>4.99</td><td>0.35</td><td>27.08</td><td>5.78</td><td>23.65</td><td>77.92</td><td>78.90</td><td>78.40</td></tr><tr><td> $\mathrm { G e m m a 2 } { \cdot } 9 \mathrm { B } { \cdot } \mathrm { i } \mathrm { t } ^ { \mp \ast }$ </td><td>40.14</td><td>7.83</td><td>0.91</td><td>30.56</td><td>9.01</td><td>26.82</td><td>77.87</td><td>79.21</td><td>78.53</td></tr><tr><td>Pre-training Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>COINS† (Paul and Frank, 2021)</td><td>22.82</td><td>10.52</td><td>-</td><td>-</td><td>-</td><td>19.4</td><td></td><td></td><td></td></tr><tr><td>CHAE† (Wang et al., 2022)</td><td>32.04</td><td>15.89</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>COG-BART† (Xie et al., 2022)</td><td>24.51</td><td>2.26</td><td>0.16</td><td>18.71</td><td>3.11</td><td>17.24</td><td></td><td></td><td></td></tr><tr><td>MLD-EA (Ours)</td><td>43.92</td><td>12.17</td><td>2.29</td><td>35.51</td><td>14.48</td><td>31.41</td><td>76.34</td><td>77.84</td><td>77.08</td></tr></table>

Table 4: Result of Missing Sentence Generation. The best performance is highlighted in bold. EA and \*: Emotions and Actions involved; ∓: Input with the action-emotion prediction of the missing sentence. †: The results are taken from the highest scores from their research output.

The superior performance of our MLD-EA model highlights its advanced capability to accurately predict the missing sentence in a narrative. This suggests that the model’s ability to consider emotional and action-related cues is essential for enhancing the logical coherence of stories. These findings emphasize the importance of incorporating nuanced narrative elements, such as emotions and actions, in developing more sophisticated and reliable models for story generation.

## 6.3 Sentence Generation

We also compare our model with the baselines for the Generation task, which considers the different situations. Also, we add the influence of actionemotion prediction on generation task. The results, as shown in Table 4, demonstrate that our MLD-EA model, particularly when incorporating predicted actions and emotions, achieves competitive performance across multiple metrics. Notably, our model with the action-emotion prediction achieves the highest scores in several key areas: BLEU-1, BLEU-2, and all ROUGE. Moreover, we notice that BLEU-4 rises dramatically after involving emotions and actions for the Gemma2-2B-it model. This means this method may be more suitable for small-size LLMs on generation tasks with consistency and coherency. We also compare this with previous studies in story plot generation, which are done by pre-training models. Obviously, the LLMsbased results achieve impressive improvement in generation tasks.

Incorporating actions and emotions into the generation process significantly enhances the model’s performance, as evidenced by the notable improvement in BLEU and ROUGE scores across all baselines. However, the difference in BERTScore is slight. Overall, the baselines involved in emotion and action while adding action-emotion prediction still outperform the fundamental baselines.

We also use VAD to measure the deviation from the original sentence with model generation. Table

5 concludes that both LLMs can make the generated sentence closer to an original sentence in emotional dimensions after introducing emotions and actions. This improvement indicates that incorporating emotional and action cues enhances the logical consistency of the narrative and ensures that the generated content aligns more closely with the emotional tone of the original text, making the output more authentic and contextually appropriate.

<table><tr><td>Model</td><td>V</td><td>A</td><td>D</td><td>MEAN</td><td>AoA</td><td>Con</td></tr><tr><td colspan="7">Without EA</td></tr><tr><td>Llama3-8B-Instruct</td><td>0.160</td><td>0.095</td><td>0.122</td><td>0.126</td><td>0.068</td><td>0.140</td></tr><tr><td>Gemma2-2B-it</td><td>0.176</td><td>0.092</td><td>0.129</td><td>0.133</td><td>0.069</td><td>0.160</td></tr><tr><td>Gemma2-9B-it</td><td>0.154</td><td>0.093</td><td>0.113</td><td>0.120</td><td>0.070</td><td>0.153</td></tr><tr><td colspan="7">With EA*</td></tr><tr><td>Llama3-8B-Instruct*</td><td>0.157</td><td>0.092</td><td>0.123</td><td>0.124</td><td>0.064</td><td>0.143</td></tr><tr><td>Gemma2-2B-it*</td><td>0.165</td><td>0.092</td><td>0.111</td><td>0.123</td><td>0.063</td><td>0.148</td></tr><tr><td>Gemma2-9B-it*</td><td>0.143</td><td>0.095</td><td>0.110</td><td>0.116</td><td>0.066</td><td>0.154</td></tr><tr><td>MLD-EA (Ours)</td><td>0.142</td><td>0.092</td><td>0.116</td><td>0.117</td><td>0.065</td><td>0.137</td></tr></table>

Table 5: VAD: deviation between original sentence and generated sentence. The closest result is highlighted in bold; V: Valence; A: Arousal; D: Dominance; MEAN: mean values of VAD; AoA: Age of Acquisition; Con: Concreteness; All values range from 0 to 1.

<table><tr><td>Model</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>MLD-EA</td><td>81.09</td><td>81.19</td><td>80.89</td></tr><tr><td>w/o ae</td><td>77.20</td><td>77.24</td><td>77.12↓2.97</td></tr><tr><td>w/o a</td><td>69.28</td><td>69.23</td><td>69.16↓10.93</td></tr><tr><td>w/o e</td><td>66.47</td><td>66.84</td><td>66.64↓13.45</td></tr></table>

Table 6: Ablation Study of Narrative Logic Checker on predicting missing sentence position with conditional training. w/o ae: without actions and emotions; w/o a: without actions, emotions only; w/o e: without emotions, actions only.

## 6.4 Ablation Study

MLD-EA’s primary task is to find the logic gap by providing characters’ emotions and actions. So, we focus on how actions and emotions affect the model’s performance in ablation studies. As shown in Table 6<sup>4</sup>, after we remove actions (w/o a) and emotions (w/o e), the accuracy of prediction drops out a lot, which decreases by more than 10 on micro-average F1 score. Surprisingly, we notice if we do not add actions and emotions as input (w/o ae), the prediction result is not bad. It may be because the default considering features is unrelated to characters’ behaviors; it may be more related to the inherent connection between sentences.

<table><tr><td rowspan=1 colspan=1>Input Story:</td><td rowspan=1 colspan=1>s0: &quot;Gary was looking for a new laptop.&quot;s1: &quot;Gary needed the laptop to be light, fast, andefficient.&quot;s2: &quot;After researching, Gary decided he would pur-chase a Mac Air.&quot;s3: &quot;Gary was not pleased with the operation of hisnew laptop at all.&quot;</td></tr><tr><td rowspan=1 colspan=1>Character:</td><td rowspan=1 colspan=1>Gary</td></tr><tr><td rowspan=1 colspan=1>Actions:</td><td rowspan=1 colspan=1>LookingFor(a new laptop) → Needed(laptop) → De-cided(Purchase a Mac Air) → NotPleased(laptop)</td></tr><tr><td rowspan=1 colspan=1>Emotion:</td><td rowspan=1 colspan=1>anticipation → anticipation → joy → anger</td></tr><tr><td rowspan=1 colspan=1>Missing Sen-tence Index:</td><td rowspan=1 colspan=1>Insert before sentence 3.</td></tr><tr><td rowspan=1 colspan=1>Missing Sen-tence:</td><td rowspan=1 colspan=1>&quot;After purchasing, Gary quickly realized he made thewrong decision.&quot;</td></tr><tr><td rowspan=1 colspan=1>Without EA</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Llama3-8B-Instruct</td><td rowspan=1 colspan=1>Insert before: 2Sentence: &quot;He had heard great reviews about theMac Air&#x27;s performance and durability.&#x27;</td></tr><tr><td rowspan=1 colspan=1>Gemma2-2B-it</td><td rowspan=1 colspan=1>Insert before: 2Sentence: &quot;Despite its sleek design and powerful pro-cessor, the Mac Air seemed to be plagued by frequentcrashes and slowdowns.&quot;</td></tr><tr><td rowspan=1 colspan=1>Gemma2-9B-it</td><td rowspan=1 colspan=1>Insert before: 3Sentence: &quot;Despite its sleek design and positive re-views, the Mac Air proved to be frustratingly slow.&quot;</td></tr><tr><td rowspan=1 colspan=1>With EA</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Llama3-8B-Instruct</td><td rowspan=1 colspan=1>Insert before: 3Sentence: &quot;Gary was frustrated with the slow perfor-mance and poor battery life of his new laptop.&#x27;</td></tr><tr><td rowspan=1 colspan=1>Gemma2-2B-it</td><td rowspan=1 colspan=1>Insert before: 2Sentence: &quot;After reading reviews and comparingspecs, Gary felt a surge of excitement at the prospectof finally getting a Mac Air.&quot;</td></tr><tr><td rowspan=1 colspan=1>Gemma2-9B-it</td><td rowspan=1 colspan=1>Insert before: 3Sentence: &quot;However, the Mac Air proved to be muchslower than he had anticipated.&quot;</td></tr></table>

Table 7: Example: different models with different rules on missing sentence index prediction and generation.

## 6.5 Case study

Table 7 shows the result of models in finding the missing one’s index and generating the sentence before and after the introduction of action and emotion. Without actions and emotions, most models incorrectly predicted the missing location, generating sentences that did not align with the emotional progression. For example, Llama3-8B-Instruct suggested inserting a sentence before s that did not logically lead to Gary’s later frustration.

When actions and emotions were included, model performance improved significantly. Both Llama3-8B-Instruct and Gemma2-9B-it accurately identified the correct index and generated sentences that better reflected the emotional shift from joy to anger, such as "Gary was frustrated with the slow performance and poor battery life of his new laptop.". The example of "However, the Mac Air proved to be much slower than he had anticipated." even reflects the previous emotion status, making the sentence more connective to the story’s consistent emotions and actions. This case study highlights the importance of action-emotion modeling in enhancing the accuracy and coherence of narrative generation, leading to more logically consistent and emotionally resonant outputs.

## 7 Conclusion

In this work, we introduced the MLD-EA model, a novel approach that leads LLMs to address gaps in narrative logic by integrating actions and emotions. MLD-EA extracts the actions and emotions of the characters in the input story and guides LLMs to find logical loopholes in the narrative by following the rules of interaction between actions and emotions. After getting the position where the missing part should be inserted, it combines the character behaviors and emotions in the context of the missing position to predict the possible character actions and emotions and complete the missing plot. The experimental results demonstrate that MLD-EA significantly improves narrative coherence and emotional alignment compared to existing models, highlighting its effectiveness in story logic detection and generation. By focusing on the interplay between actions and emotions, we have shown that maintaining logical consistency is crucial for producing believable and emotionally resonant narratives. This work advances the field of checking story logic and showcases the potential of LLMs as powerful tools for ensuring narrative cohesion.

## Limitations

First, the model has only been tested on short, fivesentence stories and has yet to be evaluated on longer, more complex narratives. This may limit its generalizability to extended storytelling contexts. Second, the model’s performance heavily relies on the quality of the original emotion labels and action abstractions. Any inaccuracies in these inputs could negatively affect the model’s ability to generate coherent and logically consistent narratives. Future work should address these limitations by testing the model on longer stories and improving the robustness of emotion and action extraction.

## Acknowledgments

This work is supported by the Alan Turning Institute/DSO grant: Improving multimodality misinformation detection with affective analysis. Yunfei Long, and Jinming Zhang acknowledge the financial support of the School of Computer Science and

Electrical Engineering, University of Essex.

## References

AI@Meta. 2024. Llama 3 model card.

Alberto Alvarez. 2023. Chatgpt as a narrative structure interpreter. In International Conference on Interactive Digital Storytelling, pages 113–121. Springer.

Ishitva Awasthi, Kuntal Gupta, Prabjot Singh Bhogal, Sahejpreet Singh Anand, and Piyush Kumar Soni. 2021. Natural language processing (nlp) based text summarization-a survey. In 2021 6th International Conference on Inventive Computation Technologies (ICICT), pages 1310–1317. IEEE.

Albert Bandura. 1977. Social learning theory. Prentice-Hall google schola, 2:101–123.

Antoine Bosselut, Omer Levy, Ari Holtzman, Corin Ennis, Dieter Fox, and Yejin Choi. 2017. Simulating action dynamics with neural process networks. arXiv preprint arXiv:1711.05313.

Tom B Brown. 2020. Language models are few-shot learners. arXiv preprint arXiv:2005.14165.

Walter B Cannon. 1927. The james-lange theory of emotions: A critical examination and an alternative theory. The American journal ofpsychology, 39(1/4):106– 124.

Charles S Carver, Steven K Sutton, and Michael F Scheier. 2000. Action, emotion, and personality: Emerging conceptual integration. Personality and social psychology bulletin, 26(6):741–751.

Gregory Currie and Jon Jureidini. 2004. Narrative and coherence. Mind & language, 19(4):409–427.

Sabine A Döring. 2003. Explaining action by emotion. The Philosophical Quarterly, 53(211):214–230.

Nancy Eisenberg. 2014. Altruistic emotion, cognition, and behavior (PLE: Emotion). Psychology Press.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898.

Rachelyn Farrell and Stephen G Ware. 2024. Planning stories neurally. Authorea Preprints.

Jian Guan, Fei Huang, Zhihao Zhao, Xiaoyan Zhu, and Minlie Huang. 2020. A knowledge-enhanced pretraining model for commonsense story generation. Transactions of the Association for Computational Linguistics, 8:93–108.

Jian Guan, Xiaoxi Mao, Changjie Fan, Zitao Liu, Wenbiao Ding, and Minlie Huang. 2021. Long text generation by modeling sentence-level and discourse-level

coherence. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6379–6393.

Beliz Gunel, Jingfei Du, Alexis Conneau, and Ves Stoyanov. 2020. Supervised contrastive learning for pretrained language model fine-tuning. arXiv preprint arXiv:2011.01403.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Guangming Huang, Yunfei Long, Cunjin Luo, Jiaxing Shen, and Xia Sun. 2024a. Prompting explicit and implicit knowledge for multi-hop question answering based on human reading process. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 13179–13189.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024b. Large language models cannot self-correct reasoning yet. In The Twelfth International Conference on Learning Representations.

Hanlei Jin, Yang Zhang, Dan Meng, Jun Wang, and Jinghua Tan. 2024. A comprehensive survey on process-oriented automatic text summarization with exploration of llm-based methods. arXiv preprint arXiv:2403.02901.

Subbarao Kambhampati, Karthik Valmeekam, Lin Guan, Kaya Stechly, Mudit Verma, Siddhant Bhambri, Lucas Saldyt, and Anil Murthy. 2024. Llms can’t plan, but can help planning in llm-modulo frameworks. arXiv preprint arXiv:2402.01817.

Robert L Leahy, David A Clark, and DJ Dozois. 2022. Cognitive-behavioral theories. Gabbard’s Textbook ofPsychotherapeutic Treatments, 151.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7871–7880.

Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. 2022. Diffusionlm improves controllable text generation. Advances in Neural Information Processing Systems, 35:4328– 4343.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Qiang Lu, Xia Sun, Yunfei Long, Zhizezhang Gao, Jun Feng, and Tao Sun. 2023. Sentiment analysis: Comprehensive reviews, recent advances, and open challenges. IEEE Transactions on Neural Networks and Learning Systems.

Qiang Lu, Xia Sun, Yunfei Long, Xiaodi Zhao, Wang Zou, Jun Feng, and Xuxin Wang. 2025. Multimodal dual perception fusion framework for multimodal affective analysis. Information Fusion, 115:102747.

AH Maslow. 1943. A theory of human motivation. Psychological Review google schola, 2:21–28.

Raymond J Mooney and Gerald DeJong. 1985. Learning schemata for natural language processing. In IJCAI, pages 681–687.

Keith Oatley. 2002. Emotions and the story worlds of fiction. Narrative impact: Social and cognitive foundations, 39:69.

OpenAI. 2022. Introducing chatgpt. https://openai. com/index/chatgpt/. Accessed: 2024-08-07.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting ofthe Associationfor Computational Linguistics, pages 311–318.

Zeeshan Patel, Karim El-Refai, Jonathan Pei, and Tianle Li. 2024. Swag: Storytelling with action guidance. arXiv preprint arXiv:2402.03483.

Debjit Paul and Anette Frank. 2021. Coins: Dynamically generating contextualized inference rules for narrative story completion. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5086–5099.

Hubert Plisiecki and Adam Sobieszek. 2024. Extrapolation of affective norms using transformer-based neural networks and its application to experimental stimuli selection. Behavior Research Methods, 56(5):4716–4731.

Robert Plutchik. 2001. The nature of emotions: Human emotions have deep evolutionary roots, a fact that may explain their complexity and provide tools for clinical practice. American scientist, 89(4):344–350.

Hannah Rashkin, Antoine Bosselut, Maarten Sap, Kevin Knight, and Yejin Choi. 2018. Modeling naive psychology of characters in simple commonsense stories. arXiv preprint arXiv:1805.06533.

Steven Reiss. 2004. Multifaceted nature of intrinsic motivation: The theory of 16 basic desires. Review ofgeneral psychology, 8(3):179–193.

Gemma Team. 2024. Gemma.

Ashish Vaswani. 2017. Attention is all you need. arXiv preprint arXiv:1706.03762.

Xinpeng Wang, Han Jiang, Zhihua Wei, and Shanlin Zhou. 2022. Chae: Fine-grained controllable story generation with characters, actions and emotions. In Proceedings of the 29th International Conference on Computational Linguistics, pages 6426–6435.

Yichen Wang, Kevin Yang, Xiaoming Liu, and Dan Klein. 2023. Improving pacing in long-form story planning. In The 2023 Conference on Empirical Methods in Natural Language Processing.

Amy Beth Warriner, Victor Kuperman, and Marc Brysbaert. 2013. Norms of valence, arousal, and dominance for 13,915 english lemmas. Behavior research methods, 45:1191–1207.

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652.

T Wolf. 2019. Huggingface’s transformers: State-ofthe-art natural language processing. arXiv preprint arXiv:1910.03771.

Kaige Xie and Mark Riedl. 2024. Creating suspenseful stories: Iterative planning with large language models. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2391–2407.

Yuqiang Xie, Yue Hu, Wei Peng, Guanqun Bi, and Luxi Xing. 2022. Comma: Modeling relationship among motivations, emotions and actions in language-based human activities. In Proceedings of the 29th International Conference on Computational Linguistics, pages 163–177.

Zhuohan Xie, Trevor Cohn, and Jey Han Lau. 2023. The next chapter: A study of large language models in storytelling. In Proceedings of the 16th International Natural Language Generation Conference, pages 323–351.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.

Qingqing Zhao, Yuhan Xia, Yunfei Long, Ge Xu, and Jia Wang. 2025. Leveraging sensory knowledge into text-to-text transfer transformer for enhanced emotion analysis. Information Processing & Management, 62(1):103876.

Zoie Zhao, Sophie Song, Bridget Duah, Jamie Macbeth, Scott Carter, Monica P Van, Nayeli Suseth Bravo, Matthew Klenk, Kate Sick, and Alexandre LS Filipowicz. 2023. More human than human: Llmgenerated narratives outperform human-llm interleaved narratives. In Proceedings of the 15th Conference on Creativity and Cognition, pages 368–370.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. arXiv preprint arXiv:2403.13372.

Jing Zhu and Paul Thagard. 2002. Emotion and action. Philosophical psychology, 15(1):19–36.

Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. 2024. Toolqa: A dataset for llm question answering with external tools. Advances in Neural Information Processing Systems, 36.

## A Chosen Missing Sentence

Given a sequence of emotions attributed to characters in a narrative, we determine where emotional changes are most pronounced. Specifically, we analyze the emotions expressed by each character at different steps, calculate the "distance" between emotions in sentences, and identify the step where the aggregate emotional change across all characters is the greatest. This is crucial for understanding key moments in emotional narratives, potentially highlighting climaxes or critical turning points.

For each character $^ { c , }$ at current sentence $s _ { i } ,$ we calculate the emotion change value $D ( e _ { s _ { i } } , e _ { s _ { j } } , c )$ for each sentence $s _ { j }$ :

$$
D ( e _ { s _ { i } } , e _ { s _ { j } } , c ) = \left\{ \begin{array} { l l } { \frac { d ( e _ { s _ { i } } ^ { c } , e _ { s _ { j } } ^ { c } ) } { | s _ { i } - s _ { j } | } } & { i f e _ { s _ { i } , c } } \\ & { a n d e _ { s _ { j } , c } } \\ { 0 } & { o t h e r w i s e } \end{array} \right.\tag{6}
$$

where $d ( x )$ represents the function to compute the distance between $e _ { s i } ^ { c }$ and $e _ { s _ { j } } ^ { c }$ . Then, identify the sentence $s _ { i _ { m a x } }$ where emotions change maximized:

$$
s _ { i _ { m a x } } = \arg \operatorname* { m a x } _ { s _ { i } } \sum _ { c = c _ { 1 } } ^ { c _ { m } } \sum _ { s _ { j } = s _ { 1 } } ^ { s _ { n } } D ( e _ { s _ { i } } , e _ { s _ { j } } , c ) ,\tag{7}
$$

where $i _ { \mathrm { m a x } }$ represents the index in the sequence where the emotions across all characters experience the greatest change. Then we remove this $s _ { i _ { m a x } }$ from the original story.

## B hyper-parameters Used in MLD-EA

Table 8 shows hyper-parameters of fine-tuning. The generation tasks’ hyper-parameters for all models are the same as shown in Table 9.

## C Details of Narrative Logic Checker result on predicting missing sentence index

Table 10 shows all results of the narrative logic checker running on baselines with different prompt techniques we used in the experiment.

<table><tr><td rowspan=1 colspan=1>Parameter name</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>lora_rank</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>lora_alpha</td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=1 colspan=1>lora_dropout</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1>lora_target</td><td rowspan=1 colspan=1>all</td></tr><tr><td rowspan=1 colspan=1>learing rate</td><td rowspan=1 colspan=1>2e − 5</td></tr><tr><td rowspan=1 colspan=1>epoches</td><td rowspan=1 colspan=1>3</td></tr></table>

Table 8: hyper-parameters of fine-tuning
<table><tr><td rowspan=1 colspan=1>Parameter name</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>torch_dtype</td><td rowspan=1 colspan=1>torch.float16</td></tr><tr><td rowspan=1 colspan=1>do_sample</td><td rowspan=1 colspan=1>True</td></tr><tr><td rowspan=1 colspan=1>temperature</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1>top_p</td><td rowspan=1 colspan=1>0.4</td></tr></table>

Table 9: hyper-parameters of generation

All results of missing sentence index prediction results when involved actions and emotions have increased on average. Especially before involving actions and emotions in inference, they are hard to recognize when the story is completed. However, after we add actions and emotions during inferring, the LLMs can recognize the completed story even with the zero-shot prompt (Gemma2-9B-it\*). These results illustrate that considering the interaction between actions and emotions can extraordinarily improve LLMs’ narrative logic checking.

## D Error Analysis: Generation results with correct index

One key area for error analysis involves evaluating how well the model predicts the correct index for the missing sentence. Misplacement of the generated sentence can disrupt the logical flow of the narrative. Table 11 shows the generation results when the input of the missing sentence index is correct. In this evaluation, we focused on how predicted action-emotion affects the generation quality of the missing part. So, we will only consider when the index is predicted correctly by the narrative logic checker in relation to the analysis results, which involve emotion and actions.

The results show the importance of when models predict the index of logical loopholes. The change of BLEU and ROUGE remains the same because they are all compared with the reference story. At the sentence level BERTScore measures, the F1 score increases dramatically if the generated sentence is filled in the right place. This highlights the model’s ability to produce more contextually appropriate and coherent content when the narrative gap is accurately identified. This underscores the importance of accurate index prediction in generating logically and emotionally consistent stories.

## E Prompt Engineering

We started at zero-shot for all the cases, then developed one-shot and few-shots after confirming the zero-shot prompt template. Also, we used Chainof-Thought as an assistant prompt strategy.

For emotion classification, we begin our approach by deploying a suite of meticulously designed prompts to leverage the MLD-EA’s capabilities in emotion classification, guiding the model to accurately discern and categorize the emotional spectrum associated with each character in a given sentence. After establishing a baseline performance using the inherent strengths of LLMs, we refine our MLD-EA model through a process inspired by the baseline. This refinement is achieved using supervised fine-tuning with a custom-tailored prompt that enhances the model’s ability to detect and classify emotions more precisely for individual characters. This targeted fine-tuning boosts the model’s proficiency, enhancing its analytical and emotional sentiment analysis capabilities.

There are some examples of prompt templates used for baselines on experiments. Table 12 shows the prompt template for action abstraction. We also present the prompt templates for both ’Without EA and ’With EA’ for the narrative logic checker and generation tasks. The prompt templates of ’Without EA’ mean the LLMs need to find the logic loopholes and complete the plot only along with the input story, which the zero-shot prompt template for the narrative logic checker is shown in Table 13 and the generation template is in Table 14. The prompt templates of ’With EA’ means the LLMs have to consider the characters’ emotions and actions during those tasks, which the zero-shot prompt template for the narrative logic checker is shown in Table 13 and the generation template is in Table 14. Also, Table 15 shows how we predict the actions and emotions for the missing part. Notably, All the results of ’Without EA’ are actually how LLMs face those tasks without any further information. Our study considers the interaction of actions and emotions to increase overall performances for LLMs on those tasks.

<table><tr><td rowspan="2">Actions and Emotions</td><td rowspan="2">Model</td><td colspan="3">k=-1</td><td colspan="3">k=2</td><td colspan="3">k=3</td><td colspan="3">k=4</td><td colspan="3">Avg</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td rowspan="9"></td><td>Llama3-8B- Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>zero-shot</td><td>0.00</td><td>0.00</td><td>0.00</td><td>10.73</td><td>57.14</td><td>18.08</td><td>95.49</td><td>27.97</td><td>43.27</td><td>0.00</td><td>0.00</td><td>0.00</td><td>26.56</td><td>21.28</td><td>15.34</td></tr><tr><td>one-shot</td><td>0.00</td><td>0.00</td><td>0.00</td><td>35.57</td><td>53.53</td><td>42.74</td><td>78.95</td><td>27.70</td><td>41.02</td><td>2.63</td><td>50.00</td><td>5.00</td><td>29.29</td><td>32.81</td><td>22.19</td></tr><tr><td>few-shot</td><td>0.00</td><td>0.00</td><td>0.00</td><td>12.75</td><td>73.08</td><td>21.71</td><td>90.98</td><td>32.27</td><td>47.64</td><td>36.18</td><td>67.90</td><td>47.21</td><td>34.98</td><td>43.31</td><td>29.14</td></tr><tr><td>Gemma2-2B-it</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>zero-shot</td><td>0.00</td><td>0.00</td><td>0.00</td><td>2.68</td><td>57.14</td><td>5.13</td><td>98.50</td><td>28.42</td><td>44.10</td><td>5.26</td><td>44.44</td><td>9.41</td><td>26.61</td><td>32.50</td><td>14.66</td></tr><tr><td>one-shot</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.34</td><td>15.38</td><td>2.47</td><td>76.69</td><td>26.98</td><td>39.92</td><td>0.00</td><td>0.00</td><td>0.00</td><td>19.51</td><td></td><td>10.60</td></tr><tr><td>few-shot</td><td>0.00</td><td>0.00</td><td>0.00</td><td>2.01</td><td>42.86</td><td>3.85</td><td>84.21</td><td>28.79</td><td>42.91</td><td>17.76</td><td>64.29</td><td>27.84</td><td>26.00</td><td>10.59 33.98</td><td>18.65</td></tr><tr><td>Gemma2-9B-it</td><td></td><td>0.00</td><td>0.00</td><td>3.36</td><td>1.00</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>zero-shot one-shot</td><td></td><td>0.00 32.69</td><td>24.29</td><td>27.87</td><td>55.03</td><td>6.49</td><td></td><td>95.49 29.74</td><td>45.36</td><td>23.68</td><td>67.92</td><td>35.12</td><td>30.62</td><td>49.42</td><td>21.74</td></tr><tr><td>few-shot</td><td></td><td>34.62</td><td>40.91</td><td>37.50</td><td>13.42 86.96</td><td>62.12</td><td>58.36 23.26</td><td>68.42 38.89 80.45</td><td>49.59</td><td></td><td>23.68 73.47</td><td>35.82</td><td></td><td>44.96 49.69</td><td>42.91</td></tr><tr><td>Llama3-8B-</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>36.15</td><td>49.88</td><td>47.37</td><td>59.50</td><td>52.75</td><td>43.96</td><td>55.88</td><td>40.85</td></tr><tr><td></td><td>Instruct*</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="8">With EA*</td><td>zero-shot</td><td>0.00</td><td>0.00</td><td>0.00</td><td>77.86</td><td>39.46</td><td>52.37</td><td>36.09</td><td>25.40</td><td>29.81</td><td>1.32</td><td>66.67</td><td>2.58</td><td>28.81</td><td>32.88</td><td>21.19</td></tr><tr><td>one-shot</td><td>0.00</td><td>0.00</td><td>0.00</td><td>14.09</td><td>43.75</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>27.44</td></tr><tr><td>few-shot</td><td>1.92</td><td>33.33</td><td>3.64</td><td>5.37</td><td>61.54</td><td>21.32 9.88</td><td>57.89</td><td>33.19</td><td>42.19</td><td>52.63</td><td>41.24</td><td>46.24</td><td>31.16</td><td>29.54</td><td></td></tr><tr><td>Gemma2-2B-it*</td><td></td><td></td><td></td><td></td><td></td><td></td><td>75.19</td><td>37.74</td><td>50.25</td><td>64.47</td><td>51.04</td><td>56.98</td><td>36.74</td><td>45.91</td><td>30.19</td></tr><tr><td>zero-shot</td><td>0.00</td><td>0.00</td><td>0.00</td><td>65.77</td><td>35.90</td><td>46.45</td><td>36.84</td><td>23.90</td><td>28.99</td><td></td><td>71.43</td><td></td><td>26.48</td><td>32.81</td><td>20.43</td></tr><tr><td>one-shot</td><td>38.46</td><td>11.17</td><td>17.32</td><td>46.98</td><td>28.57</td><td>35.53</td><td>12.03</td><td>27.59</td><td>16.75</td><td>3.29 0.66</td><td>50.00</td><td>6.29</td><td>24.53</td><td></td><td></td></tr><tr><td>few-shot</td><td>7.69</td><td>21.05</td><td>11.27</td><td>67.79</td><td>38.40</td><td>49.03</td><td>39.10</td><td>31.52</td><td>34.90</td><td>15.13</td><td>58.97</td><td>1.30</td><td>32.43</td><td>29.33</td><td>17.73 29.82</td></tr><tr><td>Gemma2-9B-it*</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>24.08</td><td></td><td>37.49</td><td></td></tr><tr><td></td><td></td><td>1.92</td><td>33.33</td><td>3.64</td><td></td><td></td><td>20.88</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>27.04</td></tr><tr><td></td><td>zero-shot</td><td></td><td></td><td></td><td>12.75</td><td>57.58</td><td></td><td>90.23</td><td>30.77</td><td>45.89</td><td>26.32</td><td>66.67</td><td>37.74</td><td>32.80</td><td>47.09</td><td></td></tr><tr><td></td><td>one-shot</td><td>38.46</td><td>44.44</td><td>41.24</td><td>48.32</td><td>51.80</td><td>50.00 20.65</td><td>61.65 67.67</td><td>37.10 35.86</td><td>46.33 46.88</td><td>36.18 53.29</td><td>70.51 65.32</td><td>47.83 58.70</td><td>46.16 45.45</td><td>50.96 47.79</td><td>46.35 41.80</td></tr><tr><td></td><td>few-shot</td><td>48.08</td><td>35.71</td><td>40.98</td><td>12.75</td><td>54.29</td></table>

Table 10: Details of Narrative Logic Checker result on predicting missing sentence index. -1: The input story is completed; 2,3,4: The missing sentence should be inserted before index[2,3,4], where story’s index starts at 1; Avg: the Micro-average score of all index’s F1 score; Without EA: prediction without involving emotions and actions. With EA and \*: prediction involving emotions and actions.

<table><tr><td rowspan="2">Model</td><td colspan="3">BLEU</td><td colspan="3">ROUGE</td><td colspan="3">BERTScore</td></tr><tr><td>-1</td><td>-2</td><td>-4</td><td>-1</td><td>-2</td><td>-L</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>Llama3-8B-Instruct</td><td>43.68</td><td>12.15</td><td>2.67</td><td>34.77</td><td>14.23</td><td>31.13</td><td>77.91</td><td>79.34</td><td>78.61</td></tr><tr><td>Llama3-8B-Instruct*</td><td>44.26</td><td>12.56</td><td>2.93</td><td>35.08</td><td>14.58</td><td>31.46</td><td>87.35</td><td>88.91</td><td>88.11</td></tr><tr><td>Gemma2-2B-it</td><td>35.98</td><td>7.80</td><td>2.42</td><td>31.03</td><td>11.37</td><td>27.35</td><td>76.54</td><td>78.25</td><td>77.37</td></tr><tr><td>Gemma2-2B-it*</td><td>36.42</td><td>6.96</td><td>1.55</td><td>30.34</td><td>10.15</td><td>27.08</td><td>80.81</td><td>81.95</td><td>81.04</td></tr><tr><td>Gemma2-9B-it</td><td>40.14</td><td>7.83</td><td>0.91</td><td>30.56</td><td>9.01</td><td>26.82</td><td>77.87</td><td>79.21</td><td>78.53</td></tr><tr><td>Gemma2-9B-it*</td><td>40.63</td><td>8.04</td><td>0.84</td><td>30.50</td><td>9.31</td><td>26.88</td><td>87.01</td><td>88.53</td><td>87.75</td></tr></table>

Table 11: Generation results with correct index. The model with \*: Input with the correct prediction of the missing sentence index. All results are based on the correct missing sentence index as input.

![](images/46222a6c7f59b796423d03879bb112ef1c848a662fc995cdd5d6db05e2aa3e6e.jpg)  
Table 12: Prompt template: Action Abstraction

![](images/e6b53a2aac077b5be2fb45e23cfd687e001f52eb4db4d1984ccd58d79fec45b3.jpg)  
Table 13: Prompt template: Narrative Logic Checker

<table><tr><td>Without EA:</td></tr><tr><td>Instruction: You are an AI assistant (Master in story writing) designed to help users analyze, evaluate, and complete stories by checking</td></tr><tr><td>their completeness and coherence. Generate a sentence to fill a gap in a narrative based on the surrounding context, ensuring the story remains coherent and</td></tr><tr><td>complete.</td></tr><tr><td>**Generate the Missing Sentence**:</td></tr><tr><td>-Create a sentence that naturally fits into the narrative at the specified index. -Ensure the new sentence connects logically with the sentences before and after it, maintaining a smooth and coherent flow.</td></tr><tr><td>–Match the style and tone of the existing story.</td></tr><tr><td>UserInput will provide a story with several sentences, and the index of missing one should be inserted before.</td></tr><tr><td>With EA but no prediction actions and emotions: Instruction:</td></tr><tr><td>You are an AI assistant (Master in story writing) designed to help users analyze, evaluate, and complete stories by checking</td></tr><tr><td>their completeness and coherence. Generate a sentence to fill a gap in a narrative based on the surrounding context, ensuring the story remains coherent and</td></tr><tr><td>complete.</td></tr><tr><td>**Generate the Missing Sentence**: -Create a sentence that naturally fits into the narrative at the specified index.</td></tr><tr><td>-Ensure the new sentence connects logically with the sentences before and after it, maintaining a smooth and coherent flow. —Match the style and tone of the existing story.</td></tr><tr><td>UserInput will provide a story with several sentences; characters&#x27; actions and emotions in sentences are shown after each</td></tr><tr><td>sentence. The Characters in story and the index of the missing sentence should be inserted before With EA and prediction actions and emotions:</td></tr><tr><td>You are an AI assistant (Master in story writing) designed to help users analyze, evaluate, and complete stories by checking</td></tr><tr><td>their completeness and coherence. Generate a sentence to fill a gap in a narrative based on the surrounding context, ensuring the story remains coherent and</td></tr><tr><td>complete.</td></tr><tr><td>**Generate the Missing Sentence**:</td></tr><tr><td>-Create a sentence that naturally fits into the narrative at the specified index. -Ensure the new sentence connects logically with the sentences before and after it, maintaining a smooth and coherent flow.</td></tr><tr><td>-Match the style and tone of the existing story.</td></tr><tr><td>-Consider whether the given actions and emotions are reasonable in this situation. Then, generate the sentence.</td></tr><tr><td>**Notes**:</td></tr><tr><td>1. The action form looks like this: Action(Target, ActionObject), where</td></tr><tr><td></td></tr><tr><td>(Action: The action performed by the character (i.e. Love, Loved, Loves, See, Saw, Attack, Attacks, Attacked, Move, Moves,</td></tr><tr><td>Moved, Move to, Come, Came, etc.).</td></tr><tr><td>Target: The target of the action (who or what the action is directed towards) (i.e. A Love B  $- > \mathsf { A }$  Love(B)).</td></tr><tr><td>ActionObject: The specific object related to the action (if any) (i.e. A give b an apply -&gt; A Give(b, an apply)).</td></tr><tr><td></td></tr><tr><td>0</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>2. The emotions are ONLY from Plutchik&#x27;s eight basic emotions (joy, trust, fear, surprise, sadness, disgust, anger, anticipation)</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>for the characters based on their likely emotional state based on the context and characters&#x27; actions. If &#x27;none&#x27; means the</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>characters do not have a discernible emotion or will not appear at this point.</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr></table>

Table 14: Prompt template: Sentence Generation

![](images/6410b8b895db218f40186933c3dc5d6a102a15e83afee4a98e59d4b2ac3e6111.jpg)  
Table 15: Prompt template: Actions and Emotions Prediction