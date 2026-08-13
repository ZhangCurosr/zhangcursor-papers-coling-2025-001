# XFORMPARSER: A Simple and Effective Multimodal Multilingual Semi-structured Form Parser

Xianfu Cheng<sup>1†</sup>, Hang Zhang<sup>1†</sup>, Jian Yang<sup>2</sup>, Xiang Li<sup>2</sup>, Weixiao Zhou<sup>1</sup>, Fei Liu<sup>3</sup>, Kui Wu<sup>1</sup>, Xiangyuan Guan<sup>1</sup>, Tao Sun<sup>1</sup>, Xianjie Wu<sup>1</sup>, Tongliang Li<sup>4\*</sup>, Zhoujun Li<sup>1,5\*</sup>,

<sup>1</sup>CCSE, Beihang University, <sup>2</sup>Beihang University, <sup>3</sup>Beijing Language and Culture University,

<sup>4</sup>Beijing Information Science and Technology University,

<sup>5</sup>Shenzhen Intelligent Strong Technology Co.,Ltd.

{buaacxf, zhbuaa0, jiaya, xlggg, wxzhou, gxy0615, buaast, wuxianjie, lizj}@buaa.edu.cn, 17861517116@163.com, wukui0099@gmail.com, tonyliangli@bistu.edu.cn

## Abstract

In the domain of Document AI, parsing semistructured image form is a crucial Key Information Extraction (KIE) task. The advent of pretrained multimodal models significantly empowers Document AI frameworks to extract key information from form documents in different formats such as PDF, Word, and images. Nonetheless, form parsing is still encumbered by notable challenges like subpar capabilities in multilingual parsing and diminished recall in industrial contexts in rich text and rich visuals. In this work, we introduce a simple but effective Multimodal and Multilingual semi-structured FORM PARSER (XFormParser), which anchored on a comprehensive Transformer-based pre-trained language model and innovatively amalgamates semantic entity recognition (SER) and relation extraction (RE) into a unified framework. Combined with Bi-LSTM, the performance of multilingual parsing is significantly improved. Furthermore, we develop InD-FormSFT, a pioneering supervised fine-tuning (SFT) industrial dataset that specifically addresses the parsing needs of forms in various industrial contexts. XFormParser has demonstrated its unparalleled effectiveness and robustness through rigorous testing on established benchmarks. Compared to existing state-ofthe-art (SOTA) models, XFormParser notably achieves up to 1.79% F1 score improvement on RE tasks in language-specific settings. It also exhibits exceptional cross-task performance improvements in multilingual and zero-shot settings.<sup>1</sup>

## 1 Introduction

Document AI is the technology of automatically reading, understanding, and analyzing business documents. It is widely applied in commercial, governmental, and educational sectors and is crucial to departmental efficiency and productivity. A key task of Document AI is to parse and extract form information from scanned documents. As shown in Figure 1, form parsing is essentially an entity relation mining task that connects the Named Entity Recognition (NER) task (Li et al., 2020) and KIE task (Cui et al., 2021; Yu et al., 2021; Hong et al., 2022). Due to the diversity of the layout and the format, poor quality of scanned document images, and complexity of template structures, representing and understanding the unstructured information in documents using generic rules becomes a highly challenging task.

![](images/6505d3f7d984fe57259f4801f5e60d3787f8fd266e8538c36d759e12ae8604be.jpg)  
(a) Text-centric form understanding on FUNSD  
(b) Image-centric layout analysis on PubLayNet  
Figure 1: An illustration of named entity recognition for unstructured forms.

Unlike traditional NER tasks which only deal with textual information, and traditional pattern recognition tasks (Medvet et al., 2011; Cheng et al., 2020), mainstream Document AI methods typically involve using deep neural networks to model elements in documents from the perspectives of computer vision (CV) (He et al., 2016; Ren et al., 2016), natural language processing (NLP) (Zhou et al., 2023; Zhang et al., 2024c), or multimodal fusion (Li et al., 2024). Apart from textual information, the position and layout of text blocks also play a crucial role in semantic interpretation.

Form parsing algorithms based on deep learning initially involved detecting and classifying specific regions of document images. Then, it utilized Optical Character Recognition (OCR) models (Li et al., 2023; Cheng et al., 2024) to extract text information. This process is called unstructured to semi-structured, and the resulting form is called semi-structured. Subsequently, different categories of form blocks and their corresponding text content were routed to dedicated information extraction modules, constructing a pipeline to process entirepage documents (Xu et al., 2020b; Wang et al., 2022). Current researchers believe that effective language models for form parsing must comprehend target entities and adapt to different document formats in contexts involving multiple modalities. Advanced Document AI models (Wang et al., 2020; Zhang et al., 2020; Li et al., 2021; Peng et al., 2022) are expected to automatically classify, extract, and structure information from business documents, minimizing manual intervention.

Although advanced models have made significant progress in the field of Semi-structured form parsing, as research and applications have become more widespread, most existing methods still face three limitations: 1) The accuracy of key information extraction from complex, multilingual form images remains insufficient; 2) Multimodal models dominated by visual modalities still lag behind text-dominated multimodal models in form parsing tasks in rich-text and long-text scenarios; 3) Multi-modal large language models (MLLM) such as GPT4o (Huang et al., 2023; Islam and Moushi, 2024) and LayoutLLM (Fujitake, 2024) are difficult to deploy to the end side and achieve fast and high-performance inference via CPUs or low-memory GPUs due to the excessive weight of model parameters. Therefore, in industrial applications, it is a crucial research direction to further mine the entity classification and the relations between entities in multimodal and multilingual forms based on simple and effective pre-trained models (PTM) (Zoph et al., 2020), and to study more effective fine-tuning paradigms in complex scenes.

To address these issues, we propose a simple but effective semi-structured form parser with multimodal and multilingual knowledge, named XForm-Parser. For input data from semi-structured forms, XFormParser utilizes the multilingual document understanding PTM LayoutXLM (Xu et al., 2022)

to generate vectors containing text, visual, and spatial positional information. Subsequently, these vectors are fed into the downstream joint network to complete two tasks: Semantic Entity Recognition (SER) and Relation Extraction (RE), to realize form parsing. The SER Task obtains text box classification through fully connected layers, and the RE task learns the categories of entity relations through a decoder based on Bi-LSTM (Sun et al., 2022) and Biaffine (Nguyen and Verspoor, 2019). In addition, we further build InDFormSFT, a Chinese and English multi-scenario form parsing SFT dataset for industrial applications, based on public benchmarks. Training the model on this dataset helps XFormParser learn semi-structured forms from the real world and achieve new SOTA performance.

The contributions of this paper are summarized as follows:

• We propose XFormParser, which integrates two tasks: SER and RE, along with the joint loss function and training method of soft labels warm-up in stages. XFormParser effectively enhances form parsing performance without additional inference resources and overhead.

• We construct InDFormSFT, a cross-scenario form parsing SFT dataset in both Chinese and English. It contains 562 form images collected from 8 major industrial application scenarios and corresponding annotation information in JSON format that is semi-automatically generated using tools such as GPT4o and rigorously verified by humans.

• Through experiments and analysis, we confirm that XFormParser achieves an F1 score of at most 1.79% over the SOTA model for RE tasks in Language-specific scenarios. XFormParser achieves significantly better results than SOTA for both dual-task in Multilanguage and Zero-shot scenarios. The ablation experiments on InDFormSFT demonstrate the effectiveness and robustness of XFormParser.

## 2 Related Work

Limited by the lack of training data and the complexity of the corpus, relatively few effective models for parsing multilingual forms have been proposed in the past few years. Most applications use a pipeline approach to process form information one modality by one, such as XLM-RoBERTa (Conneau et al., 2020), which is a multilingual version of RoBERTa (Liu et al., 2019), and InfoXLM (Chi et al., 2021), a multilingual pretrained model (Vaswani et al., 2017; OpenAI, 2023) that maximizes the mutual information between multilingual and multi-granularity texts, which are used as text information processors to analyze forms.

![](images/7271d0336a8c833e2ce53fb20f42ceee5b26d90669d0718406054c996985730f.jpg)  
Figure 2: Overall architecture of the proposed XFormParser. For Multimodal input, XFormParser utilizes layoutXLM to generate vectors containing text, visual, and spatial positional information. Subsequently, these vectors are fed into the downstream joint network to complete SER and RE tasks. The SER Task obtains text box classification through fully connected layers, and the RE task learns the categories of entity relations through a decoder based on Bi-LSTM and Biaffine.

The Visual-Language models (Wu et al., 2021; Radford et al., 2019; Zhang et al., 2024b) demonstrate the potential to effectively align gaps between textual and other modal features, which can be used to integrate form structure information and content information. It is further revealed that form information extraction can be achieved by multi-modal techniques combining CV (Cheng et al., 2022) and NLP (Yang et al., 2024, 2020, 2022, 2019; Chai et al., 2024; Zhang et al., 2024a).

The LiLT (Wang et al., 2022) model proposes a language-independent transformer focusing on text layout, introducing a new Bidirectional Attention Complement Mechanism to enhance cross-modal cooperation. It also proposes Key Point Localization and Cross-modal Alignment Identification tasks, combined with the widely used Masked Visual Language Model as pre-training objectives. LayoutLM (Xu et al., 2020a) introduced the PTM, which combined language models and Transformer, expanding BERT (Devlin et al., 2018) architecture by incorporating layout information to consider spatial relations between tokens and textual content connections, resulting in outstanding performance in form parsing tasks. Furthermore, LayoutLMv2 (Xu et al., 2020b) and LayoutLMv3 (Huang et al., 2022) enhance the integration of multimodal information in the pretraining stage. They also add pretraining strategies for text-image alignment and text-image matching, incorporating word token alignment targets.

Regarding the joint modeling of SER and RE, existing work primarily targets sequence labeling tasks, training jointly at the token granularity (Jiang et al., 2024; Ji et al., 2024; Wang et al., 2023; El Khbir et al., 2024; Liu et al., 2023). At present, there is a lack of available models or methods for joint pattern training with cell information.

GeoLayoutLM (Luo et al., 2023) improves the feature representation of text and layout by explicitly modeling geometric relationships and special pre-training tasks, which can improve the performance of form information extraction. However, the complex model structure makes the pre-training cost of related tasks high and too dependent on data. GOSE (Chen et al., 2023) first generates initial relation predictions for entity pairs extracted from document scan images. It then captures global structure knowledge from previous iterative predictions and incorporates it into entity representations. This "generate-capture-incorporate" loop is repeated multiple times, allowing entity representations and global structure knowledge to reinforce each other.

## 3 Method

## 3.1 Task Definition

The semi-structured form processed by the OCR model can be represented as a list of n semantic entities. Each entity consists of a set of words named $W o r d s _ { i } .$ , the coordinates of the bounding box named $B b o x _ { i } .$ and an image named $I m a g e _ { i }$ called i-th cell of the form and defined as $b _ { i } { \mathrm { : } }$

$$
b _ { i } = [ W o r d s _ { i } , B b o x _ { i } , I m a g e _ { i } ]\tag{1}
$$

The documents in our dataset are annotated with labels for each entity and relations between entities. We represent each comment document D as:

$$
D = [ B , L , R ]\tag{2}
$$

where $B { = } [ b _ { 1 } , \ldots , b _ { n } ]$ denote all the cells of the form; $L { = } [ l _ { 1 } , \ldots , l _ { n } ]$ is a predefined set of entity labels and l is the label of each entity; $R { = } [ ( b _ { 1 } , b _ { 2 } ) , \dots , ( b _ { j } , b _ { k } ) ]$ is the set of relations between entities, $( b _ { j } , b _ { k } )$ refers to the relation between j-th entity and k-th entity, as well as the link point from $b _ { k }$ to $b _ { j }$ . It is worth noting that one entity may have relations with multiple entities or may not have a relationship with some other entity.

Semantic Entity Recognition Task. The semantic entity recognition needs to classify all the cells and obtain a list of resulting labels $L ,$ the label of cells can be: Header entity (HEADER), QUES-TION entity (QUESTION), ANSWER entity (AN-SWER) and OTHER entity (OTHER). When the entity classification is completed, each token needs to be converted into a BIO label according to the entity category, to facilitate the alignment with experiments on open-source benchmark datasets.

Relation Extraction Task. Given the above definition and description of a form, for a form $D _ { : }$ , each cell $b _ { i }$ needs to find its corresponding sequence of cells in the whole form $B _ { b _ { i } } { = } [ b _ { j } , \ldots , b _ { k } ]$ , Finally, the normalized cell relation $ R _ { b _ { i } } { = } [ \{ ( b _ { i } , b _ { j } ) , \dots , ( b _ { i } , b _ { k } ) \} ]$ is obtained.

## 3.2 Overall Architecture

As shown in Figure 2, we built the XFormParser model based on the multimodal Transformer architecture. The model accepts information from three different modalities: text, position, and vision, encoded using text embedding, 2D position embedding, and image embedding layers, respectively. Concatenate the text and image embeddings and then add the position embeddings to obtain the input embeddings. The input embedding is encoded by layoutXLM, and the context representation is output through the dense layer. On the one hand, the representation vector is passed through the MLP layer to obtain the entity classification. On the other hand, the entity embedding vector is mined through the Bi-LSTM sequence relation and then entered into the $M L P _ { h e a d }$ or $M L P _ { t a i l }$ according to the type to obtain the entity expression. The Score is obtained by Biaffine to determine whether there is a relation between two entities.

## 3.3 PTM and Multimodal Input

layoutXLM adds two new embedding layers, 2-D Position Embedding and Image Embedding, based on BERT. The 2-D Position Embedding corresponds to text blocks with the content and position information in the document, both content and position information obtained by OCR and other technologies. By considering the top-left corner of the document page as the origin of the coordinate system, the representation of each text block in terms of horizontal coordinate, vertical coordinate, width, and height can be calculated, and the final 2-D Position Embedding is the sum of the representations of the four sub-layers. The image embedding divides the image into several blocks based on the bounding boxes of each word in the OCR results, and each block corresponds to each word. After normalizing embeddings of multiple modalities, they are input into layoutXLM.

Vectorization of Text Information. Tokenization, word embeddings, and sentence embeddings mainly realize the vectorization of text information. Among them, tokenization breaks down text into smaller units such as words or subwords. LayoutXLM uses RoBERTa Tokenizers to map these tokens into dense vectors in a continuous vector space where semantically similar words are positioned closely together. Sentence embeddings are also used, provide a vector representation for entire sentences, capturing the semantic meaning of the sentence as a whole.

Vectorization of Location Information. Twodimensional position embedding uses OCR and other techniques to obtain the content and position information of each text block in the document. The upper-left corner of the document page is regarded as the origin of constructing the coordinate system so that the embeddings of each text block corresponding to XY coordinates, width, and height can be calculated, and the final twodimensional position embedding is the sum of the four embeddings.

Vectorization of Image Feature. In this work, the image embedding is not simply obtained by uniform segmentation, but the text block is used to locate the bounding box of each word in the result, and the image is divided into several equal size patches, to ensure that each word has embedding vector of the image containing it.

## 3.4 Semantic Entity Recognizer

As shown in Figure 2, the PTM receives multimodal input $B { = } [ b _ { 1 } , \ldots , b _ { n } ]$ , obtains hidden states H, and processes the hidden representation H through a fully connected layer (Dense). The hidden representation $H _ { s e r }$ for the SER task is obtained, and then the classification result logits is obtained by a classification MLP. The formulas are expressed as follows:

$$
H = P L M ( B )\tag{3}
$$

$$
H _ { s e r } = D e n s e _ { s e r } ( H )\tag{4}
$$

$$
l o g i t s _ { s e r } = M L P _ { s e r } ( H _ { s e r } )\tag{5}
$$

Then $l o g i t s _ { s e r }$ uses argmax to get the predicted vectors $P { = } [ p _ { 1 } , \ldots , p _ { n } ] , p _ { i }$ corresponds to the label $l _ { i }$ of i-th entity, such as "QUESTION", "AN-SWER", etc.

## 3.5 Relation Extraction Decoder

For the RE task, we first get the PTM output of the i-th entity, denoted as $H _ { r e }$ . The model then uses a pooling operation (Reimers and Gurevych, 2019) on $H _ { r e }$ to obtain the embeddings of the entities. Then we concatenate the entity embeddings and the corresponding label embeddings $p _ { i }$ predicted by the SER. As the output of the encoder for each semantic entity, the formula is as follows:

$$
e _ { i } = p o o l i n g ( H _ { r e } ) \oplus p _ { i }\tag{6}
$$

Note that for two tasks to implicitly share H, $H _ { r e }$ needs to be obtained using a Dense layer identical to Eq. (4), that is, $H _ { r e } = D e n s e _ { r e } ( H )$ . In addition, mean-pooling is experimentally verified to be the most effective pooling operation applied in Eq. (6).

In the experiment, it was found that the distribution of entity expression and label embedding was not uniform, which would make it difficult for

MLP to learn the importance of both information. Therefore, $e _ { i }$ was input into the Bi-LSTM decoder to unify, and after the entity passed the decoder, according to the type, it entered the head entity decoder $M L P _ { h e a d }$ or the tail entity decoder $M L P _ { t a i l } { , }$ and finally produces the entity representation. It obtains the score through Biaffine to determine whether there is a relation between the two entities.

## 3.6 Traning Method

Training Loss. Since it is a joint model, the team must be trained to calculate the loss. The loss formula is as follows:

$$
L o s s _ { s e r } = \mathrm { C E } ( l o g i t s _ { s e r } , L _ { s e r } )\tag{7}
$$

$$
\begin{array} { r } { L o s s _ { r e } =  { \mathbf { C } }  { \mathbf { E } } ( S c o r e _ { B i a f f i n e } , L _ { r e } ) } \end{array}\tag{8}
$$

$$
L o s s = L o s s _ { s e r } + L o s s _ { r e }\tag{9}
$$

where CE is the cross-entropy function. $L o s s _ { s e r }$ is the loss for the SER task, $L o s s _ { r e }$ is the loss for the RE task, and the final training Loss is the sum of the two used as the learning target of the joint task.

Warm-up Soft Label. We propose an improved warm-up soft label (Huang et al., 2019) based on the warming mechanism, which can effectively improve the performance when applied to fine-tuning form parsing models. In the beginning stage of training, hard labels are used to supervise the model so that it can converge quickly. In the middle and later stages of training, soft labels are used to help model training. During the transition period between soft and hard labels, a warming mechanism is added, and the weight of soft tags is continuously increased. This is useful for tasks such as multi-label classification.

In actual experiments, it was found that the training effect was not good in the first half of model training. Although soft labels contain more information, they cannot guide the model in learning tasks in the early stage of training, so hard labels are still used in the early stage of model training. After the model has been trained to have preliminary capabilities, then use soft label training, and provide a transition for the conversion of hard label and soft label. For the construction method of the warm-up soft label and related training parameter Settings, please refer to Appendix A.

## 4 Experiments and Discussion

## 4.1 Experiment Settings

Datasets. FUNSD (Guillaume Jaume, 2019) is a scanned document dataset for form parsing. It is a subset of the RVL-CDIP (Harley et al., 2015) and consists of 149 training samples and 50 test samples with various layouts. XFUND (Xu et al., 2022) is a multilingual form parsing benchmark. It includes 1,393 fully annotated forms in 7 languages. Each language contains 199 forms, with 149 forms in the training set and 50 forms in the test set.

<table><tr><td>Setup</td><td>Multi-language model</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Weight ratio</td><td>0.1</td></tr><tr><td>Lr scheduler</td><td>LINEAR</td></tr><tr><td>vocab size</td><td>250002</td></tr><tr><td>Max Steps</td><td>512</td></tr><tr><td>Batch size</td><td>8</td></tr><tr><td>Initial learning rate</td><td>5e-5</td></tr><tr><td>Training epochs</td><td>100</td></tr><tr><td>Evaluation metric</td><td>ref. B.4</td></tr></table>

Table 1: Training configurations for XFormParser

XFUND and FUNSD datasets have two tasks: SER, where BIO labeling format is used for sequence labeling of each entity, and RE, where all possible pairs of given semantic entities are generated to gradually construct a candidate set of relations, identifying all entity pairs with existing associations.

In this paper, InDFormSFT is constructed based on the Chinese-English context and eight application scenarios, the training set contains 422 samples, and the validation set and test set contain 70 samples each. Compared with FUNSD and XFUND datasets, larger datasets can provide more samples for the model to learn and train, which helps to improve the generalization ability and performance of the model. Please refer to Appendix B for the construction process of InDFormSFT.

## 4.2 Language-specific Fine-tuning

Table 2 presents the results of specific language fine-tuning experiments on the XFUND and FUNSD public datasets. Each column in the table represents a different language (FUNSD, ZH, JA, ES, FR, IT, DE, PT) (Xu et al., 2022). The table compares different models, including XLM-RoBERTa<sub>BASE</sub>, InfoXLM<sub>BASE</sub>, LayoutXLM<sub>BASE</sub>, LayoutLMv3<sub>BASE</sub>, LiLT[InfoXLM]<sub>BASE</sub>, GeoLayoutLM, and XFormParser<sub>BASE</sub>. The numerical values in the table indicate the performance metrics of each model fine-tuned in specific languages. XFormParser demonstrates superior performance compared to other models in most languages. On the FUNSD dataset, XFormParser achieves a performance of 92.46%. XFormParser also exhibits strong performance on the XFUND datasets in different languages, with average performance metrics of 89.04% (SER) and 90.54% (RE).

## 4.3 Multi-language Fine-tuning

XFUND dataset inspired Multilingual fine-tuning Task, refers to multilingual task fine-tuning on XFUND dataset, all trained on 8 languages and tested on a specific language. As shown in Table 3, in the multilingual training task, XLM-RoBERTa<sub>BASE</sub>, InfoXL $\mathbf { M } _ { \mathrm { B A S E } }$ , LayoutXLM<sub>BASE</sub>, and XFormParser all show stronger ability than language-specific fine-tuning, obtaining higher F1 score. In the case of multiple languages, XFormParser<sub>BASE</sub> shows a more powerful extraction ability. XFormParser performs best in the multi-language fine-tuning experiment. The average F1 accuracy of SER task is 91.67%. For RE task, XFormParser also achieves the best performance in multi-language fine-tuning experiments, with an average F1 of 95.89%. These results underscore the efficacy of the XFormParser in handling multi-language tasks within the XFUND dataset, particularly in comparison to other baseline and advanced models. With more data, our model demonstrates stronger learning capabilities.

## 4.4 Zero-shot Fine-tuning

Previous experiments illustrate that our method achieves improvements using full training samples. We explored the transferability of our model structure and found that it will gain strong transferability on RE tasks. Thus, we compare with the previous SOTA model LiLT on few-shot settings. The experimental results in Table 4 indicate that the average performance of XFormParser still outperforms the SOTA model LiLT and GOSE. In the task of SER, XFormParser exhibited the highest F1 scores across all languages, achieving an average of 71.35%, with notable improvements in Chinese (72.00%) and Portuguese (73.46%). In the task of RE, the highest performance in the RE task was again seen with XFormParser, which achieved an impressive average F1 score of 81.18%, showing its robustness in relation extraction across different languages. the XFormParser consistently outperforms other models in both the SER and RE tasks across various languages, highlighting its superior cross-lingual transfer capabilities, and it can improve the generalization of the model.

<table><tr><td></td><td>Model</td><td>FUNSD</td><td>ZH</td><td>JA</td><td>ES</td><td>FR</td><td>IT</td><td>DE</td><td>PT</td><td>Avg.</td></tr><tr><td rowspan="6">SER</td><td>XLM-RoBERTaBASE InfoXLMBASE LayoutXLMBASE</td><td>66.70 68.52</td><td>87.74 88.68</td><td>77.61 78.65</td><td>61.05 62.30</td><td>67.43 70.15</td><td>66.87 67.51</td><td>68.14 70.63</td><td>68.18 70.08</td><td>70.47 72.07</td></tr><tr><td>LiLT[InfoXLM]BASE</td><td>79.40 84.15</td><td>89.24</td><td>79.21</td><td>75.50</td><td>79.02</td><td>80.82</td><td>82.22</td><td>79.03</td><td>80.56</td></tr><tr><td></td><td></td><td>89.38</td><td>79.64</td><td>79.11</td><td>79.53</td><td>83.76</td><td>82.31</td><td>82.20</td><td>82.51</td></tr><tr><td>GeoLayoutLM</td><td>92.86</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>XFormParser[LiLT]</td><td>91.42</td><td>91.89</td><td>82.25</td><td>87.18</td><td>87.64</td><td>89.42</td><td>87.05</td><td>87.86</td><td>88.09(+5.58)</td></tr><tr><td>XFormParser</td><td>92.46</td><td>93.14</td><td>82.59</td><td>87.77</td><td>88.69</td><td>90.51</td><td>88.48</td><td>88.68</td><td>89.04 (+6.53)</td></tr><tr><td rowspan="8">RE</td><td rowspan="8">XLM-RoBERTaBASE</td><td>26.59</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>47.69</td></tr><tr><td></td><td>51.05</td><td>58.00</td><td>52.95</td><td>49.65</td><td>53.05</td><td>50.41</td><td>39.82</td><td></td></tr><tr><td>InfoXLMBASE 29.20</td><td>52.14</td><td>60.00</td><td>55.16</td><td>49.13</td><td>52.81</td><td>52.62</td><td>41.70</td><td>49.10</td></tr><tr><td>LayoutXLMBASE</td><td>54.83</td><td>70.73 69.63</td><td>68.96</td><td>63.53</td><td>64.15</td><td>65.51</td><td>57.18</td><td>64.32</td></tr><tr><td>LiLT[InfoXLM]BASE</td><td>62.76</td><td>72.97 70.37</td><td>71.95</td><td>69.65</td><td>70.43</td><td>65.58</td><td>58.74</td><td>67.81</td></tr><tr><td>GeoLayoutLM</td><td>89.45</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GOSE[LiLT]</td><td>76.97</td><td>87.52</td><td>80.96</td><td>85.95</td><td>86.46</td><td>84.15</td><td>80.23</td><td>73.84</td></tr><tr><td>XFormParser[LiLT] XFormParser</td><td>90.02 91.24</td><td>92.00 93.42</td><td>91.32 92.19</td><td>90.21 90.82</td><td>91.01 91.55</td><td>91.37 92.48</td><td>91.11 92.36</td><td></td><td>88.48 89.12</td><td>82.01 90.82(+8.81) 91.65 5(+9.64)</td></tr></table>

Table 2: Language-specific fine-tuning F1 accuracy on FUNSD and XFUND (fine-tuning on X, testing on X).“SER” denotes the semantic entity recognition, and “RE” denotes the relation extraction.
<table><tr><td></td><td>Model</td><td>FUNSD</td><td>ZH</td><td>JA</td><td>ES</td><td>FR</td><td>IT</td><td>DE</td><td>PT</td><td>Avg.</td></tr><tr><td rowspan="4">SER</td><td>XLM-RoBERTaBASE InfoXLMBASE</td><td>66.33 65.38</td><td>88.3 87.41</td><td>77.86 78.55</td><td>62.23 59.79</td><td>70.35 70.57</td><td>68.14 68.26</td><td>71.46 70.55</td><td>67.26 67.96</td><td>71.49 71.06</td></tr><tr><td></td><td>79.24</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LayoutXLMBASE</td><td></td><td>89.73</td><td>79.64</td><td>77.98</td><td>81.73</td><td>82.1</td><td>83.22</td><td>82.41</td><td>82.01</td></tr><tr><td>LiLT[InfoXLM]BASE XFormParser</td><td>85.74 93.89</td><td>90.47 94.02</td><td>80.88 90.94</td><td>83.40</td><td>85.77</td><td>87.92</td><td>87.69</td><td>84.93</td><td>85.85</td></tr><tr><td rowspan="6"></td><td rowspan="6">XLM-RoBERTaBASE InfoXLMBASE</td><td></td><td></td><td></td><td>90.19</td><td>89.72</td><td>91.74</td><td>91.94</td><td>90.94</td><td>91.67(+5.82)</td></tr><tr><td>36.38</td><td>67.97</td><td>68.29</td><td>68.28</td><td>67.27</td><td>69.37</td><td>68.87</td><td>60.82</td><td>63.41</td></tr><tr><td>36.99</td><td>64.93</td><td>64.73</td><td>68.28</td><td>68.31</td><td>66.90</td><td>63.84</td><td>57.63</td><td>61.45</td></tr><tr><td>LayoutXLMBASE</td><td>66.71</td><td>82.41</td><td>81.42 81.04</td><td>82.21</td><td>83.10</td><td>78.54</td><td>70.44</td><td>78.23</td></tr><tr><td>LiLT[InfoXLM]BASE</td><td>74.07 97.00</td><td>84.71 83.45</td><td>83.35</td><td>84.66</td><td>84.58</td><td>78.78</td><td>76.43</td><td>81.25</td></tr><tr><td>XFormParser</td><td>95.49</td><td>94.53</td><td>95.67</td><td>96.76</td><td>97.3</td><td>95.49</td><td>95.06</td><td>95.89 (+14.64)</td></tr></table>

Table 3: Multi-language fine-tuning accuracy (F1) on the XFUND dataset (fine-tuning on 8 languages all, testing on X), where “SER” denotes the semantic entity recognition and “RE” denotes the relation extraction.
<table><tr><td></td><td>Model</td><td>FUNSD</td><td>ZH</td><td>JA</td><td>ES</td><td>FR</td><td>IT</td><td>DE</td><td>PT</td><td>Avg.</td></tr><tr><td>SER</td><td>XLM-RoBERTaBASE InfoXLMBASE LayoutXLMBASE LiLT[InfoXLM]BASE XFormParser</td><td>66.70 68.52 79.40 84.15 92.46</td><td>41.44 44.08 60.19 61.52 72.00</td><td>30.23 36.03 47.15 51.84</td><td>30.55 31.02 45.65 51.01</td><td>37.10 40.21 57.57 59.23</td><td>27.67 28.80 48.46 53.71</td><td>32.86 35.87 52.52 60.13</td><td>39.36 45.02 53.90 63.25</td><td>38.24 41.19 55.61 60.61 71.35(+10.74)</td></tr><tr><td>RE</td><td>XLM-RoBERTaBASE InfoXLMBASE LayoutXLMBASE LiLT[InfoXLM]BASE GOSE[LiLT] XFormParser</td><td>26.59 29.20 54.83 62.76 76.97 91.24</td><td>16.01 24.05 44.94 47.64 69.30 74.02</td><td>26.11 28.51 44.08 50.81 68.05 81.77</td><td>24.40 24.81 47.08 49.68 70.72</td><td>22.40 24.54 44.16 52.09 71.45</td><td>23.74 21.93 40.90 46.97 63.55</td><td>22.88 20.27 38.20 41.69 59.97</td><td>19.96 20.49 36.85 42.72 58.30</td><td>22.76 24.23 43.88 49.30 67.29</td></tr></table>

Table 4: Cross-lingual zero-shot transfer F1 accuracy on FUNSD and XFUND (fine-tuning on FUNSD, testing on XFUND).
<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="2">Task</td><td colspan="2">Component</td><td colspan="2">SER F1 Accuracy</td><td colspan="2">RE F1 Accuracy↑</td></tr><tr><td>SER</td><td>RE</td><td>Decoder</td><td>soft label</td><td>EN</td><td>ZH</td><td>EN</td><td>ZH</td></tr><tr><td></td><td>XFormParser</td><td>√</td><td>√</td><td>√</td><td>√</td><td>92.46</td><td>93.42</td><td>91.2</td><td>93.14</td></tr><tr><td>1</td><td>w/o Task RE</td><td>√</td><td>X</td><td>X</td><td>X</td><td>91.89</td><td>92.86</td><td></td><td></td></tr><tr><td>2</td><td>w/o Task SER</td><td>x</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>90.90</td><td>92.64</td></tr><tr><td>3</td><td>w/o Decoder</td><td>√</td><td>√</td><td>x</td><td>√</td><td>91.40</td><td>92.75</td><td>79.79</td><td>81.73</td></tr><tr><td>4</td><td>w/o soft label</td><td>√</td><td>√</td><td>√</td><td>x</td><td>91.19</td><td>92.19</td><td>90.75</td><td>91.20</td></tr></table>

Table 5: Ablation study of our model using LayoutXLM as the backbone, InDFormSFT as training dataset and test on the FUNSD and XFUND (ZH). The symbol EN denotes FUNSD and ZH means Chinese language.

## 4.5 Ablation Study

To better understand the working principle of the model and determine the extent to which the key components or strategies contribute to the model performance, we designed ablation experiments to verify the feasibility of the method proposed in this paper. In ablation experiments, a series of modifications or eliminations are made to the original model to see how these modifications affect the model’s performance. Ablation experiments are designed and implemented from three innovations of our method:

<table><tr><td>epoch start</td><td>epoch warm</td><td>RE F1</td></tr><tr><td>X</td><td>x</td><td>91.20</td></tr><tr><td>10</td><td>√</td><td>92.67</td></tr><tr><td>20</td><td>√</td><td>92.74</td></tr><tr><td>30</td><td>X</td><td>92.44</td></tr><tr><td>30</td><td>√</td><td>93.14</td></tr><tr><td>40</td><td>x</td><td>91.06</td></tr><tr><td>40</td><td>√</td><td>91.35</td></tr><tr><td>50</td><td>x</td><td></td></tr><tr><td></td><td></td><td>90.93</td></tr><tr><td>50</td><td>√</td><td>91.43</td></tr></table>

Table 6: For the ablation experiments of epoch start and epoch warm, epoch start refers to the epoch round when the soft label is started, epoch warm refers to whether to use the transition mechanism, and the total training rounds are 100. The first line refers to the method of warm-up soft label that is not applicable.

Effectiveness of Individual Components. We further investigate the effectiveness of different modules in our method. we compare our model with the following variants in Table 5.

1) w/o multi-task. In this variant, we try to remove the multi-task training method and only use SER or RE for training. This change causes a significant performance decay. The results shown in Table 5 suggest that training two tasks at the same time has the same effect on both SER and RE tasks. With improvement, two tasks that share PTM parameters will learn cross-information that improves this task.

2) w/o Decoder. In this variant, we remove the decoder. This change causes a significant performance decay. This suggests the injection of an additional decoder can guide the powerful decoding capabilities of entities and provide strong dependencies for relation classification.

3) w/o Warm-up soft label. In this variant, we remove the training method Warm-up soft label from XFormParser. This change means the model only uses hard labels due to the training. The results shown in Table 5 indicate that a Warm-up soft label can improve the effect of the model and prevent the model from overfitting. Experimental data Table 6 indicates that as epoch starts increases, model performance initially improves but then decreases. This trend suggests that the model learns sufficient data features by a certain stage, and further training with a Warm-up soft label does not enhance performance and may even lead to overfitting.

## 4.6 Visualization Display

We visualize the SER and RE of XFormParser on a text-intensive form image as shown in Figure 3(b), where the orange boxes are named entities and the arrows represent the matching relations between the entities. This figure confirms the effectiveness of XFormParser.

<table><tr><td rowspan=6 colspan=1></td><td rowspan=1 colspan=1>姓名</td><td rowspan=1 colspan=2>牟奇</td><td rowspan=1 colspan=1>性别</td><td rowspan=1 colspan=1>男、女</td><td rowspan=1 colspan=2>出生日期19□年8月28日</td></tr><tr><td rowspan=1 colspan=1>民族</td><td rowspan=1 colspan=2>汉 籍贯</td><td rowspan=1 colspan=1>河北省张家</td><td rowspan=1 colspan=1>身份证号</td><td rowspan=1 colspan=2>1417   1647</td></tr><tr><td rowspan=1 colspan=1>文化程度</td><td rowspan=1 colspan=2>本科</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>婚姻状况</td><td rowspan=1 colspan=2>未婚、已婚、离婚</td></tr><tr><td rowspan=1 colspan=1>宗教信仰</td><td rowspan=1 colspan=3>无</td><td rowspan=1 colspan=1>最低工资</td><td rowspan=1 colspan=2>4500</td></tr><tr><td rowspan=2 colspan=1>身体状况</td><td rowspan=1 colspan=1>身高</td><td rowspan=1 colspan=2>163cm</td><td rowspan=1 colspan=1>体重</td><td rowspan=1 colspan=2>54     kg</td></tr><tr><td rowspan=1 colspan=1>视力</td><td rowspan=1 colspan=2>左：1.4 右：1.4</td><td rowspan=1 colspan=1>血型</td><td rowspan=1 colspan=2>AB型</td></tr><tr><td rowspan=1 colspan=1>户口地址</td><td rowspan=1 colspan=1>河L家T</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>（邮编</td><td rowspan=1 colspan=1>：07）</td><td rowspan=1 colspan=1>电话</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>联系地址</td><td rowspan=1 colspan=5>河北省张家口         邮编：075□）</td><td rowspan=1 colspan=1>电话</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>电子邮件</td><td rowspan=1 colspan=5>@163.com</td><td rowspan=1 colspan=1>手机</td><td rowspan=1 colspan=1>1599205</td></tr></table>

(a)

![](images/45cbc8c3240fc844bf4238723694ee67b6f4ff552b4f7f594305039d785018d0.jpg)  
(b)  
Figure 3: Illustration of (a) The form image that is entered into the system; (b) Visualization of SER and RE results.

## 5 Conclusion

Aiming at the common problems in rich text form parsing tasks, this paper proposes a semi-structured form parser XFormParser based on multi-modal and multi-lingual knowledge. XFormParser integrates layoutXLM pre-trained backbone, semantic entity recognizer, and relation extraction decoder, and implements SER and RE tasks for semi-structured form parsing. At the same time, to enrich the experimental data in this field and improve the parsing ability of industrial application scenarios, this paper constructs a Chinese and English multi-scenario form parsing SFT dataset InDFormSFT. Four different Settings (such as Language-specific fine-tuning, Multi-language finetuning, Cross-lingual fine-tuning, and Zero-shot) are designed on two benchmark datasets and InD-FormSFT ), and the results show the effectiveness and superiority of XFormParser.

## 6 Limitations

The purpose of this work is to provide a simple, efficient, and easy-to-deploy semi-structured form parsing component for the end side (PC or mobile). Although we use a multi-modal approach and expand the training set to improve the performance of the model, while taking into account the multilanguage parsing scenario, this work still has the following limitations.

Diversity of Languages. InDFormSFT only includes two languages, Chinese and English, and lacks expansion of form knowledge for the other six languages in XFUND. After that, multi-language augmented data can be constructed by using the same data set construction process with the help of machine translation and layout design algorithms.

Model Diversity. The comparative experiments do not include the experimental results of the large model of multi-modal documents on relevant benchmarks, thus lacking the most powerful demonstration of the upper bound of the performance of the model for the current task. In addition, our work does not further improve the multilingual pre-trained model backbone and directly adopts the strongest and most easy-to-use model investigated as the backbone of XFormParser. These limitations need to be further studied and improved.

Completeness of Verification. There is a lack of validation of model compression and inference acceleration methods such as model distillation, model pruning, and model quantization.

## Acknowledgments

This work was partially supported by the National Natural Science Foundation of China (Grant Nos. 62276017, 62406033, U1636211, 61672081, 62272025), and the State Key Laboratory of Complex& Critical Software Environment (Grant No. SKLCCSE-2024ZX-18).

## References

Linzheng Chai, Jian Yang, Tao Sun, Hongcheng Guo, Jiaheng Liu, Bing Wang, Xinnian Liang, Jiaqi Bai, Tongliang Li, Qiyao Peng, and Zhoujun Li. 2024. xcot: Cross-lingual instruction tuning for crosslingual chain-of-thought reasoning. arXiv preprint arXiv:2401.07037, abs/2401.07037.

Xiangnan Chen, Juncheng Li, Duo Dong, Qian Xiao, Jun Lin, Xiaozhong Liu, and Siliang Tang. 2023.

Global structure knowledge-guided relation extraction method for visually-rich document. arXiv preprint arXiv:2305.13850.

Xianfu Cheng, Yanqing Yao, and Ao Liu. 2020. An improved privacy-preserving stochastic gradient descent algorithm. In International Conference on Machine Learningfor Cyber Security, pages 340–354. Springer.

XianFu Cheng, YanQing Yao, Liying Zhang, Ao Liu, and Zhoujun Li. 2022. An improved stochastic gradient descent algorithm based on rényi differential privacy. International Journal of Intelligent Systems, 37(12):10694–10714.

Xianfu Cheng, Weixiao Zhou, Xiang Li, Jian Yang, Hang Zhang, Tao Sun, Wei Zhang, Yuying Mai, Tongliang Li, Xiaoming Chen, et al. 2024. Sviptr: Fast and efficient scene text recognition with vision permutable extractor. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pages 365–373.

Zewen Chi, Li Dong, Furu Wei, Nan Yang, Saksham Singhal, Wenhui Wang, Xia Song, Xian-Ling Mao, He-Yan Huang, and Ming Zhou. 2021. Infoxlm: An information-theoretic framework for cross-lingual language model pre-training. In Proceedings ofthe 2021 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 3576–3588.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Édouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440– 8451.

Lei Cui, Yiheng Xu, Tengchao Lv, and Furu Wei. 2021. Document ai: Benchmarks, models and applications. arXiv preprint arXiv:2111.08609.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Niama El Khbir, Nadi Tomeh, and Thierry Charnois. 2024. Information extraction with differentiable beam search on graph rnns. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 9084–9096.

Masato Fujitake. 2024. Layoutllm: Large language model instruction tuning for visually rich document understanding. arXiv preprint arXiv:2403.14252.

Jean-Philippe Thiran Guillaume Jaume, Hazim Kemal Ekenel. 2019. Funsd: A dataset for form understanding in noisy scanned documents. In Accepted to ICDAR-OST.

Adam W Harley, Alex Ufkes, and Konstantinos G Derpanis. 2015. Evaluation of deep convolutional nets for document image classification and retrieval. In 2015 13th International Conference on Document Analysis and Recognition (ICDAR), pages 991–995. IEEE.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770– 778.

Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, and Sungrae Park. 2022. Bros: A pre-trained language model focusing on text and layout for better key information extraction from documents. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pages 10767– 10775.

Hanyao Huang, Ou Zheng, Dongdong Wang, Jiayi Yin, Zijin Wang, Shengxuan Ding, Heng Yin, Chuan Xu, Renjie Yang, Qian Zheng, et al. 2023. Chatgpt for shaping the future of dentistry: the potential of multimodal large language model. International Journal ofOral Science, 15(1):29.

Weipeng Huang, Xingyi Cheng, Taifeng Wang, and Wei Chu. 2019. Bert-based multi-head selection for joint entity-relation extraction. Preprint, arXiv:1908.05908.

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. Layoutlmv3: Pre-training for document ai with unified text and image masking. In Proceedings of the 30th ACM International Conference on Multimedia, pages 4083–4091.

Raisa Islam and Owana Marzia Moushi. 2024. Gpt-4o: The cutting-edge advancement in multimodal llm. Authorea Preprints.

Bin Ji, Shasha Li, Hao Xu, Jie Yu, Jun Ma, Huijun Liu, and Jing Yang. 2024. Span-based joint entity and relation extraction augmented with sequence tagging mechanism. Science China Information Sciences, 67(5):152105.

Shu Jiang, Zuchao Li, Hai Zhao, and Weiping Ding. 2024. Entity-relation extraction as full shallow semantic dependency parsing. IEEE/ACM Transactions on Audio, Speech, and Language Processing.

Chenliang Li, Bin Bi, Ming Yan, Wei Wang, Songfang Huang, Fei Huang, and Luo Si. 2021. Structurallm: Structural pre-training for form understanding. arXiv preprint arXiv:2105.11210.

Jing Li, Aixin Sun, Jianglei Han, and Chenliang Li. 2020. A survey on deep learning for named entity recognition. IEEE transactions on knowledge and data engineering, 34(1):50–70.

Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, and Furu Wei. 2023. Trocr: Transformer-based optical character recognition with pre-trained models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 13094–13102.

Xiang Li, Ming Lu, Ziming Guo, and Xiaoming Zhang. 2024. Adaptive token selection and fusion network for multimodal sentiment analysis. In International Conference on Multimedia Modeling, pages 228–241. Springer.

Peipei Liu, Hong Li, Zhiyu Wang, Yimo Ren, Jie Liu, Fei Lv, Hongsong Zhu, and Limin Sun. 2023. Centre: A paragraph-level chinese dataset for relation extraction among enterprises. In 2023 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Chuwei Luo, Changxu Cheng, Qi Zheng, and Cong Yao. 2023. Geolayoutlm: Geometric pre-training for visual information extraction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7092–7101.

Eric Medvet, Alberto Bartoli, and Giorgio Davanzo. 2011. A probabilistic approach to printed document understanding. International Journal on Document Analysis and Recognition (IJDAR), 14(4):335–347.

Dat Quoc Nguyen and Karin Verspoor. 2019. Endto-end neural relation extraction using deep biaffine attention. In European conference on information retrieval, pages 729–738. Springer.

OpenAI. 2023. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Qiming Peng, Yinxu Pan, Wenjin Wang, Bin Luo, Zhenyu Zhang, Zhengjie Huang, Teng Hu, Weichong Yin, Yongfeng Chen, Yin Zhang, et al. 2022. Ernielayout: Layout knowledge enhanced pre-training for visually-rich document understanding. arXiv preprint arXiv:2210.06155.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.

Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. 2016. Faster r-cnn: Towards real-time object detection with region proposal networks. IEEE transactions on pattern analysis and machine intelligence, 39(6):1137–1149.

Tao Sun, Dongsu Shen, Saiqin Long, Qingyong Deng, and Shiguo Wang. 2022. Neural distinguishers on tinyjambu-128 and gift-64. In International Conference on Neural Information Processing, pages 419– 431. Springer.

Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. NIPS.

Jiapeng Wang, Lianwen Jin, and Kai Ding. 2022. Lilt: A simple yet effective language-independent layout transformer for structured document understanding. arXiv preprint arXiv:2202.13669.

Youwei Wang, Ying Wang, Zhongchuan Sun, Yinghao Li, Shizhe Hu, and Yangdong Ye. 2023. Deep purified feature mining model for joint named entity recognition and relation extraction. Information Processing & Management, 60(6):103511.

Zilong Wang, Mingjie Zhan, Xuebo Liu, and Ding Liang. 2020. Docstruct: a multimodal method to extract hierarchy structure in document for general form understanding. arXiv preprint arXiv:2010.11685.

Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. 2021. Cvt: Introducing convolutions to vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22–31.

Y. Xu, M. Li, L. Cui, S. Huang, and M. Zhou. 2020a. Layoutlm: Pre-training of text and layout for document image understanding. In KDD ’20: The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

Y. Xu, Y. Xu, T. Lv, L. Cui, and L. Zhou. 2020b. Layoutlmv2: Multi-modal pre-training for visually-rich document understanding.

Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, and Furu Wei. 2022. XFUND: A benchmark dataset for multilingual visually rich form understanding. In Findings of the Association for Computational Linguistics: ACL 2022, pages 3214–3224, Dublin, Ireland. Association for Computational Linguistics.

Jian Yang, Hongcheng Guo, Yuwei Yin, Jiaqi Bai, Bing Wang, Jiaheng Liu, Xinnian Liang, Linzheng Chai, Liqun Yang, and Zhoujun Li. 2024. m3p: Towards multimodal multilingual translation with multimodal prompt. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, LREC/COLING 2024, 20-25 May, 2024, Torino, Italy, pages 10858–10871. ELRA and ICCL.

Jian Yang, Shuming Ma, Dongdong Zhang, Zhoujun Li, and Ming Zhou. 2020. Improving neural machine translation with soft template prediction. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online,

July 5-10, 2020, pages 5979–5989. Association for Computational Linguistics.

Jian Yang, Yuwei Yin, Shuming Ma, Dongdong Zhang, Zhoujun Li, and Furu Wei. 2022. Hlt-mt: High-resource language-specific training for multilingual neural machine translation. arXiv preprint arXiv:2207.04906.

Ze Yang, Wei Wu, Jian Yang, Can Xu, and Zhoujun Li. 2019. Low-resource response generation with template prior. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 1886– 1897. Association for Computational Linguistics.

Wenwen Yu, Ning Lu, Xianbiao Qi, Ping Gong, and Rong Xiao. 2021. Pick: processing key information extraction from documents using improved graph learning-convolutional networks. In 2020 25th International Conference on Pattern Recognition (ICPR), pages 4363–4370. IEEE.

Peng Zhang, Yunlu Xu, Zhanzhan Cheng, Shiliang Pu, Jing Lu, Liang Qiao, Yi Niu, and Fei Wu. 2020. Trie: end-to-end text reading and information extraction for document understanding. In Proceedings ofthe 28th ACM International Conference on Multimedia, pages 1413–1422.

Wei Zhang, Xianfu Cheng, Yi Zhang, Jian Yang, Hongcheng Guo, Zhoujun Li, Xiaolin Yin, Xiangyuan Guan, Xu Shi, Liangfan Zheng, and Bo Zhang. 2024a. Eclipse: Semantic entropy-lcs for cross-lingual industrial log parsing. Preprint, arXiv:2405.13548.

Wei Zhang, Hongcheng Guo, Anjie Le, Jian Yang, Jiaheng Liu, Zhoujun Li, Tieqiao Zheng, Shi Xu, Runqiang Zang, Liangfan Zheng, and Bo Zhang. 2024b. Lemur: Log parsing with entropy sampling and chainof-thought merging. Preprint, arXiv:2402.18205.

Wei Zhang, Hongcheng Guo, Jian Yang, Yi Zhang, Chaoran Yan, Zhoujin Tian, Hangyuan Ji, Zhoujun Li, Tongliang Li, Tieqiao Zheng, Chao Chen, Yi Liang, Xu Shi, Liangfan Zheng, and Bo Zhang. 2024c. mabc: multi-agent blockchain-inspired collaboration for root cause analysis in micro-services architecture. Preprint, arXiv:2404.12135.

Weixiao Zhou, Gengyao Li, Xianfu Cheng, Xinnian Liang, Junnan Zhu, Feifei Zhai, and Zhoujun Li. 2023. Multi-stage pre-training enhanced by chatgpt for multi-scenario multi-domain dialogue summarization. arXiv preprint arXiv:2310.10285.

Barret Zoph, Golnaz Ghiasi, Tsung-Yi Lin, Yin Cui, Hanxiao Liu, Ekin Dogus Cubuk, and Quoc Le. 2020. Rethinking pre-training and self-training. Advances in neural information processing systems, 33:3833– 3845.

## A Build the Warm-up Soft Label

Soft labeling, also known as soft targets or probabilistic labeling, is a technique in natural language processing where labels for training data are represented as probability distributions rather than as hard, binary labels. Soft labels provide more information about the uncertainty and distribution of classes, leading to better generalization. In traditional hard labels, each sample can only belong to one category, while using soft labels can represent situations where a sample may belong to multiple categories.

We used the improved warm-up soft label, which can effectively improve the performance when applied to fine-tuning form parsing models. In the beginning stage of training, hard labels are used to supervise the model so that it can converge quickly. In the middle and later stages of training, soft labels are used to help model training. During the transition period between soft and hard labels, a warming mechanism is added, and the weight of soft tags is continuously increased. This is useful for tasks such as multi-label classification. For the soft labels $l o g i t s _ { s e r }$ and its embedding are calculated. The specific formula is as follows,

$$
L E _ { s l } = \frac { s o f t m a x ( l o g i t s _ { s e r } ) \cdot L E _ { w e i g h t } } { N }\tag{10}
$$

where N is the number of tags, $L E _ { w e i g h t }$ is the weight of the word list that stores tag embeddings, $l o g i t s _ { s e r }$ first obtains the distribution probability through a layer of softmax, and $s o f t m a x ( l o g i t s _ { s e r } )$ Perform dot multiplication with $L E _ { w e i g h t }$ to get a weighted label embedding vector. Finally, the weighted label embedding vector is divided by the number of samples in the dataset N to obtain the average value of the label embedding vector $L E _ { s o f t l a b e l } , L E _ { s l }$ for short.

In actual experiments, it was found that the training effect was not good in the first half of model training. Although soft labels contain more information, they cannot guide the model in learning tasks in the early stage of training, so hard labels are still used in the early stage of model training. After the model has been trained to have preliminary capabilities, then use soft label training, and provide a transition for the conversion of hard label and soft label. In fact, the final label embedding calculation method is as follows:

$$
\alpha = \mathrm { m i n } ( 1 , ( e p - e p _ { s t a r t } / e p _ { w a r m } ) )\tag{11}
$$

![](images/00eab080ee6b5c2376fae38aee0c6162140fa362f4b769ddba718b30074021dc.jpg)  
Figure 4: It shows the process of data search. On the basis of the constructed data, the title of the document is extracted as the search term, and other form files are searched in the document search engine.

![](images/25f537065654a1b7b1af2a28166d1973bd8c64cac777c2c507d2d3b47317b8bb.jpg)

Figure 5: Firstly, the optical character recognition tool is used to process the file, and the text information and border information are obtained. The data structure of the form is constructed through the border and text, including the structure information and the text information of the form.  
![](images/6193ed0b76965bd6212f6af67fa53a9deaa779a65c743fe3674ce7bd7e5de1f9.jpg)  
Figure 6: It shows the construction of a form data auxiliary labeling tool. Firstly, the optical character recognition tool is used to process the file to obtain text and border information. Then the form data filtering tool is used to determine whether the cell can be labeled. Then the form information extraction model is used for auxiliary labeling, and finally, the data result is obtained by manual labeling.

$$
L E = \left\{ \begin{array} { l l } { L E _ { h l } , } & { e p \leq e p _ { s t a r t } } \\ { \alpha L E _ { s l } + \beta L E _ { h l } , } & { e p > e p _ { s t a r t } } \end{array} \right.\tag{12}
$$

where $\alpha$ is a parameter that decays sublinearly with training and $\beta { = } 1 { - } \alpha$ . where $e p$ represents the current training epoch, $e p _ { s t a r t }$ denotes the starting epoch, and $e p _ { w a r m }$ is a predefined value determining the warm-up duration. The parameter α acts as a scaling factor for the learning rate, ensuring a gradual transition from an initial learning rate to the desired rate, as the training progresses.

![](images/963d8747c3c595871be4bfd75d8dfb9a1a2013ec726442e70ea624d56364d619.jpg)  
Figure 7: Firstly, a PDF form is obtained, the text and border information are extracted through OCR, and the cell characteristics are calculated to determine whether it conforms to the form structure. If it does not meet the conditions, it is directly discarded. If it meets the conditions, the vacancy rate of the cell is calculated. The qualified forms are pre-predicted and labeled by the model, and the pre-labeled results are imported into the labeling system. Finally, the correct labeled data are obtained by manual verification for training and testing.

## B Approach to building InDFormSFT

The large labeled data set is the main support for the high performance of deep learning. Form datasets are a common type of datasets used to collect, store, and analyze user-submitted data. It is commonly used in various application areas such as market research, user surveys, online registration, order forms, etc. In recent years, there have been some academic data sets in the field of forms. This kind of data set contains two kinds of information, one is the image of the original document, and the other is the specific information of the document that has been annotated or parsed, usually including the image, the coordinates of the text box, the text content of the text box, the label of the text box and the relationship between the text box.

## B.1 Data Collection and Annotation

The basic data set is constructed through the Chinese and English data sets of FUNSD and XFUND, and then the Chinese form data on the Internet is collected. To avoid privacy and sensitive information issues of real-world documents, this paper collects documents publicly available on the Internet. Semi-structured forms were collected through Baidu Library, and the search keywords included: a comprehensive table of university teachers, a comprehensive table of senior safety engineers, a comprehensive table of professor-level senior engineers, etc. By downloading and collecting the files of the Shenzhen Stock Exchange, Shanghai Stock Exchange, and other financial platforms, the form files that meet the task definition are found.

Figures 4, 5, 6 show the data search process, the form data filtering, and the construction process of the auxiliary labeling tool, respectively. Finally, to facilitate the pre-processing and labeling of forms, a set of engineering developments of data screening and labeling process based on Chinese forms was completed, as shown in Figure 7 below.

## B.2 Instances of Semi-structured Data

The form data also aligns with the format of the XFUND dataset, with cell granularity, where each cell contains information such as absolute cell coordinates (box), cell text information (text), cell label information (label), cell ids (id), and linking between cells (linking).

The first row of the picture is shown in Figure 8(a), and the corresponding annotated data format is shown in Figure 8(b). In Figure 8(b), lang denotes the language of the text. In this case, its value is "zh", indicating that the text is Chinese. version indicates the version of the dataset, where the value is "0.1" and "split", which means the dataset is split into train, val, and test sets. "id" represents a unique identifier for the form data. documents contain a list of form cell information, and box represents the coordinates of the text’s bounding box on the image. The value is a list of four integers representing the top-left x-coordinate, the top-left y-coordinate, the bottom-right x-coordinate, and the bottom-right y-coordinate. text represents the text content of the cell. Here, the value is a string that contains some text. label denotes the label content of the cell, which includes the SINGLE entity (SINGLE), QUESTION entity (QUESTION), ANSWER entity (ANSWER), and continuous character entity (ANSWERNUM) mentioned above. id represents the unique identifier of the cell. linking represents a linking relationship between cells and is a list of entity pairs with two ids, the first for the question entity and the second for the answer or consecutive character entity. img contains the form image information, where fname represents the name of the image file, width represents the width of the image, and height represents the height of the image.

(a)
<table><tr><td>Dataset partition</td><td>Data volume</td><td>Question Entity</td><td>Answer Entity</td><td>Single Entity</td><td>Title Entity</td><td>Continuous char entity</td></tr><tr><td>Training set</td><td>422</td><td>6702</td><td>6825</td><td>422</td><td>370</td><td>194</td></tr><tr><td>Validation set</td><td>70</td><td>1375</td><td>1443</td><td>65</td><td>18</td><td>83</td></tr><tr><td>Testing set</td><td>70</td><td>1468</td><td>1644</td><td>78</td><td>58</td><td>18</td></tr></table>

Table 7: Analysis of the number of entity labels in InDFormSFT.

<table><tr><td>Dataset partition</td><td>One-to-one</td><td>One-to-two</td><td>One-to-three</td><td>One-to-many</td></tr><tr><td>Training set</td><td>11806</td><td>265</td><td>96</td><td>171</td></tr><tr><td>Validation set</td><td>2366</td><td>55</td><td>38</td><td>39</td></tr><tr><td>Testing set</td><td>2626</td><td>76</td><td>53</td><td>72</td></tr></table>

Table 8: Entity relationship analysis of InDFormSFT.

![](images/7e4745108d1dc346ee7eff59f660b56125a5a140b5d62b6f8f52c04ba284cd86.jpg)

![](images/a7db886ae1acf57fe0de36e044fffbc74f48a1cbb77d4ef9af9393f010d725f1.jpg)  
Figure 8: Illustration of (a) A form image in InD-FormSFT; (b) The annotation corresponding to the first row of the image.

## B.3 Analysis of InDFormSFT

As shown in Table 7, for the analysis of the entity content of the data set, according to the table content analysis, the entity labels of the data set of this paper have the following characteristics: the number of question entities and answer entities is large. In the training set, the number of question entities is 6702, the number of answer entities is 6825, and the number of question entities and answer entities is basically the same. This indicates that there are a large number of question-and-answer entities in the dataset that need to be identified and annotated. These entities may include person names, place names, organizations, etc., and we need our model to be able to accurately identify and label these entities. The number of single entities, title entities, and continuous character entities is relatively rare, and recognizing these sparse entities is a challenging task.

According to the content analysis of Table 8, the table shows the division of the data set and the corresponding relationship types. Entities are classified by relationship type, including one-to-one, one-to-two, one-to-three and one-to-many (greater than three). These relation types describe the number of entity-time correspondences in the form information extraction task. We can see that most relationships are concentrated in one-to-one relationships, and a few exist in one-to-two, one-to-three, and one-to-many (greater than three) relationships. By analyzing the distribution of the number of different correspondences, we can get the distribution of the number of samples of different relation types in the dataset.

## B.4 Evaluation Metrics

In this paper, the target tasks for the datasets used are divided into SER and RE. For the SER task, the model needs to determine the class of each Cell in the form: SINGLE, QUESTION, ANSWER, and ANSWERNUM. The evaluation metric is Cell Acc (CA). The accuracy of Cell Discrimination is determined by Correct Cell Discrimination (CCD) and Total Cell Count (TCC). The formula is as follows:

$$
C A = { \frac { C C D } { T C C } }\tag{13}
$$

For the cell relation linking task, we need to determine which combinations of cells in each table have a key-value relation. The F1-Score is a commonly used metric to evaluate the performance of classification models, which takes into account both Precision and Recall. The formula for calculating the F1 score is as follows:

$$
F 1 = 2 \times \frac { P r e c i s i o n \times R e c a l l } { P r e c i s i o n + R e c a l l }\tag{14}
$$

$$
P r e c i s i o n = \frac { T P } { T P + F P }\tag{15}
$$

$$
R e c a l l = { \frac { T P } { T P + F N } }\tag{16}
$$

In this task, TP represents the number of entity pairs correctly predicted by the model as having a relationship. Actually, having a relationship, FP represents the number of entity pairs predicted by the model as having a relationship but actually having no relationship. FN represents the number of entity pairs predicted by the model as having no relationship but actually having a relationship. The F1 value ranges from 0 to 1, with values closer to 1 indicating better performance of the model.