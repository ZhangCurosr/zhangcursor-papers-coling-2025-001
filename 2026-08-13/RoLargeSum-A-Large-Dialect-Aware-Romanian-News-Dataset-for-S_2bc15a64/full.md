# RoLargeSum: A Large Dialect-Aware Romanian News Dataset for Summary, Headline, and Keyword Generation

Andrei-Marius Avram<sup>1</sup>, Mircea Timpuriu<sup>1</sup>, Andreea Iuga<sup>1</sup>, Vlad-Cristian Matei<sup>1</sup>, Iulian-Marius Taiatu˘ <sup>1</sup>, Tudor Gain˘ a˘<sup>1</sup>, Dumitru-Clementin Cercel<sup>1</sup>\*, Florin Pop<sup>1,4</sup>, Mihaela-Claudia Cercel<sup>2,3</sup>

<sup>1</sup> National University of Science and Technology POLITEHNICA Bucharest, Romania <sup>2</sup> Paris 1 Panthéon-Sorbonne University, Paris, France <sup>3</sup> University of Bucharest, Bucharest, Romania

<sup>4</sup> National Institute for Research and Development in Informatics - ICI Bucharest, Romania

## Abstract

Using supervised automatic summarisation methods requires sufficient corpora that include pairs of documents and their summaries. Similarly to many tasks in natural language processing, most of the datasets available for summarization are in English, posing challenges for developing summarization models in other languages. Thus, in this work, we introduce RoLargeSum, a novel large-scale summarization dataset for the Romanian language crawled from various publicly available news websites from Romania and the Republic of Moldova that were thoroughly cleaned to ensure a highquality standard. RoLargeSum contains more than 615K news articles, together with their summaries, as well as their headlines, keywords, dialect, and other metadata that we found on the targeted websites. We further evaluated the performance of several BART variants and open-source large language models on RoLargeSum for benchmarking purposes. We manually evaluated the results of the bestperforming system to gain insight into the potential pitfalls of this data set and future development.

## 1 Introduction

Text summarization, an essential task in natural language processing (NLP), has seen significant progress thanks to the proliferation of deep learning and the models based on Transformer (Vaswani et al., 2017). In particular, models such as Bidirectional Auto-Regressive Transformers (BART) (Lewis et al., 2020), Text-to-Text Transfer Transformer (T5) (Raffel et al., 2020), and Generative Pre-trained Transformer (GPT) (Radford et al., 2019; Brown et al., 2020; Achiam et al., 2023), have established themselves as key tools in this field (González et al., 2022; Lam et al., 2023; Kwon et al., 2023). These models perform well in abstractive summarization, which involves generating new sentences that capture the essence of the original text rather than merely extracting and rephrasing existing sentences.

To keep track of progress in this field, many summarization datasets were proposed, such as CNN/ Daily Mail (Nallapati et al., 2016), Extreme Summarization (XSum) (Narayan et al., 2018a), New York Times (NY Times) (Hermann et al., 2015), BOOKSUM (Krysci´ nski et al.´ , 2022), CiteSum (Mao et al., 2022), and CCSUM (Jiang and Dreyer, 2024). However, most of the large summarization datasets were created for the English language, with few exceptions, such as OrangeSum (Eddine et al., 2021) for French and the Dataset for Automatic summarization of Catalan and Spanish newspaper Articles (DACSA) (Soriano et al., 2022) for Spanish. This limits the development and assessment of summarization models for other languages, impeding their performance and applicability in multilingual and diverse linguistic contexts.

Thus, we provide RoLargeSum, a dataset crawled from many news websites in Romania and the Republic of Moldova, which comprises 615,679 news articles and their summaries. In addition to the summary of each news article, we also attach to each sample of RoLargeSum, where present, the corresponding headline, keywords, dialect, and other metadata such as the domain and the author of the article. RoLargeSum contributes to the growing collection of Romanian NLP resources, complementing existing datasets for tasks such as named entity recognition (Dumitrescu and Avram, 2020; Avram et al., 2024), speech recognition (Avram et al., 2022), satire detection (Rogoz et al., 2021; Echim et al., 2023), offensive language detection (Hoefels et al., 2022; Matei et al., 2024; Nicola et al., 2024), lip reading (Jitaru et al., 2020; Manescu et al.˘ , 2023), multiword expression detection (Savary et al., 2018; Avram et al., 2023), and emotion detection (Ciobotaru et al., 2022).

We further evaluate the performance of several

BART models on RoLargeSum and further improve them by using dialect-based adversarial training (Ganin et al., 2016) and extending their input context with Unlimiformer (Bertsch et al., 2024). Our results show that, in general, this methodology increases the performance of the BART models, with the large version of multilingual BART (mBART) (Liu et al., 2020) achieving the highest overall results in most experiments. We also evaluated the capabilities of several large language models (LLMs) to produce summaries for the news articles found in RoLargeSum, and the evaluation results outlined that, in general, Romanian LLMs outperform their multilingual counterparts but still achieve robust performance. Finally, we performed a human evaluation of the best-performing model on RoLarge-Sum and discussed the resulting findings.

The main contributions of our work can be summarized as follows:

• We introduce RoLargeSum<sup>1</sup>, the largest dataset for Romanian summarization, which contains more than 615K news articles, together with their summaries and headlines, annotated by dialect. RoLargeSum is also the first Romanian dataset for keyword extraction.

• We created an in-depth analysis of RoLarge-Sum and compared it with the summarization datasets available in the literature.

• We propose strong baselines on RoLarge-Sum by employing various multilingual or Romanian LLMs. We also optimize several BART models and enhance their performance through dialect adversarial training and context extension with Unlimiformer.

## 2 Related Work

## 2.1 Text Summarization Datasets

Abstractive summarization is an important and challenging task in the field of NLP. It requires models with complex natural language understanding and generation abilities to perform well. Various domains and languages provide many datasets that can be used to train and evaluate machine learning models for abstractive summarization.

The CNN/Daily Mail dataset (Nallapati et al., 2016) is one of the most widely used summarization datasets, supporting both abstractive and extractive summarization for the English language. It contains 286,817 training samples, 13,368 validation samples, and 11,487 test samples, extracted from news articles written by journalists at CNN between 2007 and 2015 and the Daily Mail between 2010 and 2015.

The NY Times (Hermann et al., 2015) is another popular abstractive summarization dataset for the English language, which contains more than 1.8 million news articles published by New York Times journalists between 1987 and 2007. However, of those 1.8 million articles, only 650K have a corresponding summary written by library scientists. Also, the NY Times corpus contains more than 1.8M tags that were manually or semiautomatically annotated.

The task of generating a brief one-sentence summary using the question “What is the article about?” is known as extreme summarization (Pavel et al., 2024). XSum (Narayan et al., 2018a) addresses this task by providing 226,711 news articles (i.e., 204,045 samples for training, 11,332 samples for validation, and 11,334 samples for testing) with a one-sentence summary in the English language, which was collected from BBC between 2010 and 2017, covering a broad spectrum of domains.

With more than 2.8 million summarized news articles in Spanish and Catalan, DASCA (Soriano et al., 2022) is one of the largest and highest-quality summarization datasets available in the literature. The news was collected from 28 different news sources (i.e., 21 in Spanish and 7 in Catalan) with years of publications ranging from 2010 to 2020, resulting in 6 million news articles cleaned and filtered down using two thresholds.

To our knowledge, RoSummary (Niculescu et al., 2022) is the only summary dataset available for the Romanian language, containing 42,862 news articles gathered from one Romanian website. The news articles and their corresponding summaries were cleaned using various hand-made heuristics.

## 2.2 Text Summarization Methods

The research community has addressed the task of automating text summarization using abstractive, extractive, or hybrid methods. On the one hand, extractive approaches generate summaries by directly selecting sentences or words from input texts (Narayan et al., 2018b; Zhong et al., 2020; Xie et al., 2022; Zhang et al., 2023). Generally, these techniques involve a sequential binary classification task to identify the most important sentences from the documents. They employ various criteria for this purpose, including negative log-likelihood on chosen sentences or Recall Oriented Understudy for Gist Evaluation (ROUGE) (Lin, 2004) rewards within reinforcement learning frameworks (Yao et al., 2018; Dong et al., 2018).

On the other hand, abstractive summarization methods generate summaries by rephrasing the sentences that are included in the documents. Thanks to recent achievements in self-supervised learning, which involves pre-training neural networks on large amounts of texts and then fine-tuning them on a downstream task (Devlin et al., 2019; Touvron et al., 2023), text summarization research has increasingly shifted from extractive methods to abstractive ones (Wang et al., 2023; Karim et al., 2024), which usually rely mainly on the encoderdecoder framework (Sutskever et al., 2014). The most effective models in this category are FlanT5 (Chung et al., 2024), T5, and BART, which were pre-trained on large corpora such as Common Craw (Wenzek et al., 2020) and the Colossal Clean Crawled Corpus (Raffel et al., 2020).

Finally, there are hybrid approaches that combine extractive and abstractive techniques, either implemented separately or simultaneously within the models (Mendes et al., 2019; Zhang et al., 2020).

## 3 RoLargeSum Dataset

## 3.1 Dataset Construction

We first crawled many news websites from Romania and the Republic of Moldova that made their content publicly accessible to compile the RoLarge-Sum dataset. We did this by gathering the main article, its summary, its headline, its keywords, and other metadata. After that, we applied several filtering techniques to remove the artifacts from the corpus (see Appendix A for more details regarding the cleaning rules).

The resulting dataset contains 615,679 samples, which we split into 605,679 for training, 5,000 for validation, and 5,000 for testing. It should also be noted that not all websites we crawled simultaneously provided summaries, headlines, or keywords for their articles. Thus, only 529,800 of the 615,679 collected samples had summaries, 613,836 headlines, and 426,455 keywords. However, we ensured that the samples in the validation and testing sets contained these fields.

![](images/9e57f09d0fb25a09fffba7a308601a8bb50391288d97b12e3978373644c9d0c3.jpg)  
Figure 1: Boxplots depicting the numbers of words for the documents, summaries, headlines, and keywords in the RoLargeSum dataset.

## 3.2 Dataset Statistics

Figure 1 shows the RoLargeSum token length statistics. The average length of the gathered news articles is 337 tokens, with a minimum of 10 tokens and a maximum of 9,851 tokens; their summaries are 46 tokens long, with a minimum of 5 tokens and a maximum of 537 tokens; their headlines are 15 tokens long, with a minimum of 1 token and a maximum of 81 tokens; and lastly, each news article has an average of 8 keywords, with a minimum of 1 keyword and a maximum of 116 keywords.

We further classified the dataset by its dialect using the top-level domain (“.ro” - corresponding to newspapers from Romania or “.md” - corresponding to newspapers from the Republic of Moldova) of the websites from which the samples were collected. With 464,465 samples belonging to the Moldavian news articles and 151,214 samples belonging to the Romanian news articles, the dataset’s dialect distribution is highly imbalanced by a ratio of roughly 3:1. We also outline a dialect-based term frequency-inverse document frequency (TF-IDF) (Salton and Buckley, 1988) analysis in Appendix C.

## 3.3 Dataset Comparison

We further compare RoLargeSum with several popular summarization datasets available in the literature for English (i.e., CNN, DailyMail, NY Times, and XSum), Spanish (i.e., DACSA), Catalan (i.e., DACSA), French (i.e., OrangeSum), as well as Romanian (i.e., RoSummary). The results of the statistical analysis are shown in Table 1 (see Appendix

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Train/Val/Test</td><td rowspan=1 colspan=1>Avg. News Len.Words Sents.</td><td rowspan=1 colspan=1>Avg. Sum. Len.Words Sents.</td><td rowspan=1 colspan=1>Vocab. SizeNews Sums.</td></tr><tr><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>90.3K/1.2K/1.1K</td><td rowspan=1 colspan=1>760.50  33.98</td><td rowspan=1 colspan=1>45.70    3.58</td><td rowspan=1 colspan=1>34K   89K</td></tr><tr><td rowspan=1 colspan=1>DailyMail</td><td rowspan=1 colspan=1>197K/12.1K/10.4K</td><td rowspan=1 colspan=1>653.33  29.33</td><td rowspan=1 colspan=1>54.65   3.86</td><td rowspan=1 colspan=1>564K  180K</td></tr><tr><td rowspan=8 colspan=1>NY TimesXSumDACSA-SpanishDACSA-CatalanOrangeSum-SummaryOrangeSum-HeadlineRoSummary-SummaryRoSummary-Headline</td><td rowspan=4 colspan=1>590K/32.7K/32.7K204K/11.3K/11.3K1.8M/104K/104K636K/35.3K/35.3K</td><td rowspan=1 colspan=1>800.04  35.55</td><td rowspan=1 colspan=1>45.54   2.44</td><td rowspan=1 colspan=1>1.2M  293K</td></tr><tr><td rowspan=1 colspan=1>431.07  19.77</td><td rowspan=1 colspan=1>23.26   1.00</td><td rowspan=1 colspan=1>399K  81K</td></tr><tr><td rowspan=1 colspan=1>644.64  23.44</td><td rowspan=1 colspan=1>28.45   1.24</td><td rowspan=1 colspan=1>3.2M  516K</td></tr><tr><td rowspan=1 colspan=1>507.74  17.71</td><td rowspan=1 colspan=1>24.09    1.17</td><td rowspan=1 colspan=1>1.3M  223K</td></tr><tr><td rowspan=4 colspan=1>21.4K/1.5K/1.5K30.6K/1.5K/1.5K38.7K/2.0K/2.1K38.7K/2.0K/2.1K</td><td rowspan=1 colspan=1>350.01  12.06</td><td rowspan=1 colspan=1>32.12   1.43</td><td rowspan=1 colspan=1>420K   71K</td></tr><tr><td rowspan=2 colspan=1>315.31  10.87329.30   8.39</td><td rowspan=1 colspan=1>11.42   1.00</td><td rowspan=1 colspan=1>483K   43K</td></tr><tr><td rowspan=1 colspan=1>56.70   3.26</td><td rowspan=1 colspan=1>329K   98K</td></tr><tr><td rowspan=1 colspan=1>329.30   8.39</td><td rowspan=1 colspan=1>18.44    1.53</td><td rowspan=1 colspan=1>329K   49K</td></tr><tr><td rowspan=3 colspan=1>RoLargeSum-SummaryRoLargeSum-HeadlineRoLargeSum-Keywords</td><td rowspan=2 colspan=1>519K/5.0K/5.0K603K/5.0K/5.0K</td><td rowspan=1 colspan=1>336.88   9.96</td><td rowspan=1 colspan=1>46.13    1.57</td><td rowspan=1 colspan=1>1.3M  295K</td></tr><tr><td rowspan=1 colspan=1>337.11   9.82</td><td rowspan=1 colspan=1>15.87    1.24</td><td rowspan=1 colspan=1>1.4M  153K</td></tr><tr><td rowspan=1 colspan=1>416K/5.0K/5.0K</td><td rowspan=1 colspan=1>337.30   9.31</td><td rowspan=1 colspan=1>8.08    1.01</td><td rowspan=1 colspan=1>1.2M   83K</td></tr></table>

Table 1: Comparison of RoLargeSum with other summarization datasets available in the literature. We outline the train/validation/test distribution of the data, the average numbers of words and sentences in the news articles and their corresponding summaries, and the vocabulary size for both the news articles and their summaries. Part of the statistics were taken from (Eddine et al., 2021).

B for the statistics based on Romanian and Moldavian news articles). As can be observed, with 529K documents, RoLargeSum is the third largest dataset in this comparison, falling behind NY Times with 655K samples and DACSA with 2M samples for Spanish and 706K for Catalan.

On the other hand, the average number of words found in RoLargeSum’s documents is on the lower end, with approximately 337 words per document, surpassing only the headline generation subset of OrangeSum and the RoSummary dataset by a little margin. The average number of words in the summaries is the third highest, behind only Daily-Mail and RoSummary. Finally, with 1.3M words in the vocabulary, RoLargeSum has a vocabulary size comparable to other datasets of similar size.

## 3.4 Proposed Subtasks

We propose the following subtasks on RoLarge-Sum:

• RO+MD - Romanian and Moldavian summarization, as well as headline and keyword generation. The subtask generates the summary, headline, and keywords for the whole corpus, ignoring the dialect label.

• RO - Romanian intra-dialect summarization, as well as headline and keyword generation. The subtask generates the summary, headline, and keywords only for the news articles from Romania.

• MD - Moldavian intra-dialect summarization, as well as headline and keyword generation. The subtask generates the summary, headline, and keywords only for Moldavian news samples.

• RO→MD - cross-dialect summarization, together with headline and keyword generation. The subtask generates the summary, headline, and keywords for Moldavian news samples using a model trained on news articles from Romania.

• MD→RO - cross-dialect summarization, together with headline and keyword generation. The subtask generates the summary, headline, and keywords for news articles from Romania using a model trained on Moldavian news samples.

## 4 Baseline Models

## 4.1 Fine-tuned Models

Figure 2 shows our overall architecture, including an encoder-decoder system wrapped with Unlimiformer and adversarially trained to produce dialectindependent embeddings with the encoder. The architecture is used to fine-tune the BART variations in our experiments.

![](images/0e97c3121c3d7193ef047f8606567e3c85ca361dc0876b19b1f3e9a23d4c39c1.jpg)  
Figure 2: The proposed baseline architecture used to generate summaries, headlines, and keywords in Ro-LargeSum. We chunk a set of tokens and add them to each chunk in the neighbouring context. After receiving the chunks, the encoder generates the corresponding embeddings, which are then used to build a kNN index for the cross-attention mechanism of the decoder. Finally, we also adversarially train a feed-forward network to detect the dialect of the input text, which helps the encoder produce dialect-independent embeddings.

BART We establish several strong baselines on RoLargeSum by fine-tuning and evaluating the performance to generate summaries, headlines, and keywords for the base and large versions of BART (i.e., BART-base and BART-large, respectively) and the large multilingual version of BART (i.e., mBART-large) (Liu et al., 2020) on each of the dialect-based subtasks. An essential aspect of the fine-tuning process is that both BART-base and BART-large have been trained only in English text, with little to no Romanian. As recommended by Lewis et al. (2020) for machine translation, we develop a new Romanian vocabulary and randomly initialize the embedding layer included in both encoders and decoders to avoid this problem.

Unlimiformer Unlimiformer (Bertsch et al., 2024) is a versatile method that can be applied to any pre-trained encoder-decoder Transformer such as BART and T5. It does this by redesigning the cross-attention mechanism to a single k-Nearest Neighbor (kNN) index, using the attention dot-product scores, which are the distances provided by the kNN algorithm (Bertsch et al., 2024). We emphasize the ability of this algorithm to index inputs of unlimited length since each attention head in each decoder layer only retrieves the top k keys rather than all of the encoder’s keys (Bertsch et al., 2024).

We use Unlimiformer since, as depicted in Figure 1, the documents can include up to 10k tokens, which is significantly longer than the context length of BART. Thus, we perform two experiments for each BART variant, namely 1) we truncate all the documents in RoLargeSum that have more tokens than the supported maximum context length and fine-tune each BART variant without any modifications to the model, and 2) we truncate the documents during training and inject the Unlimiformer wrapper into the BART model during inference to process the full-length input sequences.

Adversarial Training We further adversarially train (Ganin et al., 2016) the BART variants to produce dialect-independent features in the encoder, which we employ only for the RO+MD subtask. To do that, we take the average of the embeddings produced by the encoder and add a feed-forward layer on top of the resulting vector, which is responsible for identifying whether the input news article is from Romania or the Republic of Moldova. Then, during backpropagation, we either 1) reverse the gradients before backpropagating them into the encoder using a gradient reversal layer (Ganin et al., 2016), or 2) directly change the sign of the loss function (Avram et al., 2024).

More specifically, we take the average of the embeddings $e _ { i }$ produced by the encoder for the input sequence and employ a linear layer with weights $\theta _ { D }$ and bias $b _ { D }$ to get the dialect label yˆ, as formulated in Equation 1.:

$$
\hat { y } = \sigma ( \theta _ { D } \bar { e } + b _ { D } )\tag{1}
$$

where $\sigma$ is the sigmoid activation function and e¯ is the average of the embeddings $e _ { i }$ . Then, we employ the binary cross-entropy loss as the objective function $\mathcal { L } _ { d } ,$ which can be formulated as:

$$
\mathcal { L } _ { D } = - \sum _ { i = 1 } ^ { | D | } ( y _ { i } \log ( \hat { y } _ { i } ) + ( 1 - y _ { i } ) \log ( 1 - \hat { y } _ { i } ) )\tag{2}
$$

where $| \mathcal D |$ is the number of documents in the dataset, and $y _ { i }$ is equal to 1 if the news article is from Romania and 0 otherwise.

We apply the adversarial training to this system by reversing the feed-forward gradient as follows:

$$
\theta _ { D } = \theta _ { D } - \alpha \frac { \partial \mathcal { L } _ { D } } { \partial \theta _ { D } }\tag{3}
$$

$$
\theta _ { d e c } = \theta _ { d e c } - \alpha \frac { \partial \mathcal { L } _ { d e c } } { \partial \theta _ { d e c } }\tag{4}
$$

$$
\theta _ { e n c } = \theta _ { e n c } - \alpha \bigg ( \frac { \partial \mathcal { L } _ { d e c } } { \partial \theta _ { d e c } } - \lambda _ { G R } \frac { \partial \mathcal { L } _ { D } } { \partial \theta _ { D } } \bigg )\tag{5}
$$

where α is the learning rate, $\theta _ { d e c }$ are the parameters of the decoder, $\mathcal { L } _ { d e c }$ is the loss function used by the decoder, $\theta _ { e n c }$ are the parameters of the encoder, and $\lambda _ { G R }$ is a scaling hyperparameter.

For the loss reversal adversarial training, we simply change the sign of the loss $\mathcal { L } _ { D }$ and multiply it by another scaling hyperparameter $\lambda _ { L R }$

## 4.2 Large Langauge Models

We also evaluate the performance of various Romanian and multilingual LLMs from the Llama, Mistral, and Gemma families. We provide more details in Appendix D. In addition, we provide more information on the prompt we used for summary generation in Appendix G.

## 5 Evaluation

We measure the performance of all models using the ROUGE score (Lin and Hovy, 2003) for which we provide more information in Appendix E. In addition, more details on implementing our methodology can be found in Appendix F.

## 5.1 Results for BART Fine-Tuning

The results of the BART models on all four subtasks proposed for RoLargeSum are depicted in Table 2. In general, the mBART-large model obtained the greatest scores. We believe that this is due to the pre-training data, this model being trained specifically on Romanian, among other languages, which naturally boosted its understanding of the input text, allowing it to outperform the other two evaluated BART models.

RO+MD The highest ROUGE scores for the RO+MD subtask in summarization were obtained by the mBART model that employed both the Unlimiformer wrapper and the loss reversal adversarial training method. It obtained an R-1 of 44.57, an R-2 of 24.03, and an R-L of 36.10. This methodology also obtained the highest scores for keyword generation, with an R-1 of 59.80, an R-2 of 34.01, and an R-L of 56.29. Still, it obtained only the highest R-1 for headline generation with 43.31, being outperformed by the mBART with the Unlimiformer and the gradient reversal algorithms on the R-2 and R-L scores.

RO The best scores for summary generation on the subset that used only the news articles from Romania were obtained by the mBART-large model with the Unlimiformer wrapper, which obtained an R-1 of 41.38, an R-2 of 20.88 and an R-L of 31.91. Alternatively, this methodology obtained the highest R-1 score for headline generation, with 39.09. However, it fell behind the BART-large variant with Unlimiformer in R-2, which obtained a score of 18.99, and in R-L compared to the plain variant of mBART-large, which achieved a score of 33.70. Finally, mBART-large also obtained the highest score on keyword generation by employing the Unlimiformer wrapper on R-1 with 51.12, R-L with 47.83, and without it on R-2 with 21.67.

MD The highest ROUGE scores for summarization on the RoLargeSum subset that contained only the news articles from the Republic of Moldova were also obtained by the mBART-large model with the Unlimiformer wrapper, obtaining an R-1 score of 42.48, an R-2 score of 22.85, and an R-L score of 33.61. This methodology also obtained the best score for headline generation, with an R-1 of 39.35, an R-2 of 19.86, and an R-L of 35.02. However, it was outperformed by the plain mBART-large in keyword generation on all three metrics, which achieved an R-1 of 72.97, an R-2 of 20.84, and an R-L of 72.76.

RO→MD The results are somewhat mixed on the subtask RO→MD, with the highest ROUGE scores oscillating between mBART-large with and without Unlimiformer. Thus, for summarization, the plain mBART-large obtained the highest scores R-1 and R-2 with 32.93 and 12.26, respectively, and by wrapping the model with Unlimiformer, its results increased to 24.73. For headline generation, the highest R-1 score was obtained by the plain mBART-large, and the highest R-2 and R-L scores were achieved by the mBART-large with Unlimiformer (i.e., 12.43 and 27.88, respectively). Finally, the Unlimiformer boosted the performance of mBART-large in R-1 with 40.34 and in R-2 with 16.10, but not in R-L, where the plain mBART-large performed better, obtaining a score of 40.72.

<table><tr><td rowspan="2">Model</td><td colspan="3">Summary</td><td colspan="3">Headline</td><td colspan="3">Keywords</td></tr><tr><td>R-1</td><td>R-2</td><td>R-L</td><td>R-1</td><td>R-2</td><td>R-L</td><td>R-1</td><td>R-2</td><td>R-L</td></tr><tr><td colspan="10">RO+MD</td></tr><tr><td>BART-base</td><td>37.56</td><td>18.62</td><td>28.68</td><td>35.32</td><td>18.22</td><td>30.40</td><td>51.27</td><td>15.10</td><td>48.86</td></tr><tr><td>/w Unlimiformer</td><td>37.15</td><td>18.95</td><td>28.59</td><td>34.97</td><td>19.20</td><td>29.26</td><td>52.65</td><td>15.29</td><td>48.97</td></tr><tr><td>/w Grad. Rev.</td><td>38.59</td><td>19.06</td><td>18.90</td><td>35.25</td><td>18.84</td><td>30.11</td><td>52.79</td><td>15.40</td><td>49.08</td></tr><tr><td>/w Loss Rev.</td><td>38.63</td><td>19.07</td><td>18.89</td><td>35.33</td><td>19.17</td><td>30.34</td><td>53.00</td><td>15.72</td><td>49.21</td></tr><tr><td>BART-large</td><td>39.18</td><td>19.92</td><td>29.92</td><td>37.08</td><td>19.58</td><td>31.65</td><td>56.28</td><td>29.40</td><td>54.79</td></tr><tr><td>/w Unlimiformer</td><td>39.19</td><td>18.61</td><td>29.95</td><td>37.73</td><td>21.99</td><td>33.75</td><td>55.41</td><td>29.10</td><td>54.18</td></tr><tr><td>/w Grad. Rev.</td><td>39.35</td><td>19.79</td><td>30.07</td><td>37.76</td><td>22.04</td><td>33.80</td><td>55.40</td><td>29.05</td><td>54.19</td></tr><tr><td>/w Loss Rev.</td><td>39.24</td><td>19.67</td><td>30.19</td><td>38.01</td><td>22.16</td><td>33.92</td><td>55.46</td><td>29.09</td><td>54.15</td></tr><tr><td>mBART-large</td><td>41.42</td><td>21.65</td><td>32.27</td><td>39.89</td><td>20.37</td><td>35.30</td><td>57.97</td><td>33.65</td><td>55.86</td></tr><tr><td>/w Unlimiformer</td><td>44.38</td><td>23.86</td><td>35.91</td><td>43.09</td><td>23.62</td><td>41.37</td><td>59.76</td><td>33.95</td><td>56.18</td></tr><tr><td>/w Grad. Rev.</td><td>44.49</td><td>23.98</td><td>36.02</td><td>43.25</td><td>23.70</td><td>41.47</td><td>59.72</td><td>33.90</td><td>56.15</td></tr><tr><td>/w Loss Rev.</td><td>44.57</td><td>24.03</td><td>36.10</td><td>43.31</td><td>23.64</td><td>41.12</td><td>59.80</td><td>34.01</td><td>56.29</td></tr><tr><td colspan="10">RO</td></tr><tr><td>BART-base</td><td>36.28</td><td>17.58</td><td>27.68</td><td>33.62</td><td>16.59</td><td>28.50</td><td>41.42</td><td>17.73</td><td>38.36</td></tr><tr><td>/w Unlimiformer</td><td>37.71</td><td>18.11</td><td>28.43</td><td>35.18</td><td>17.40</td><td>29.85</td><td>41.32</td><td>17.59</td><td>38.33</td></tr><tr><td>BART-large</td><td>38.14</td><td>18.57</td><td>28.42</td><td>36.28</td><td>18.61</td><td>30.50</td><td>42.54</td><td>18.33</td><td>38.73</td></tr><tr><td>/w Unlimiformer</td><td>39.19</td><td>18.21</td><td>29.95</td><td>36.73</td><td>18.99</td><td>32.75</td><td>42.18</td><td>18.06</td><td>37.98</td></tr><tr><td>mBART-large</td><td>40.34</td><td>20.58</td><td>31.31</td><td>38.49</td><td>18.84</td><td>33.70</td><td>50.52</td><td>21.67</td><td>47.42</td></tr><tr><td>/w Unlimiformer</td><td>41.38</td><td>20.88</td><td>31.91</td><td>39.09</td><td>18.62</td><td>32.37</td><td>51.12</td><td>21.56</td><td>47.83</td></tr><tr><td colspan="10">MD</td></tr><tr><td>BART-base</td><td>36.75</td><td>18.13</td><td>28.57</td><td>33.08</td><td>16.64</td><td>28.75</td><td>63.56</td><td>17.48</td><td>63.32</td></tr><tr><td>/w Unlimiformer</td><td>36.22</td><td>17.51</td><td>27.81</td><td>33.37</td><td>16.71</td><td>29.07</td><td>63.86</td><td>18.46</td><td>63.49</td></tr><tr><td>BART-large</td><td>38.46</td><td>19.38</td><td>29.60</td><td>35.59</td><td>17.75</td><td>29.68</td><td>63.88</td><td>17.59</td><td>63.64</td></tr><tr><td>/w Unlimiformer</td><td>39.09</td><td>19.52</td><td>29.82</td><td>36.08</td><td>19.36</td><td>31.28</td><td>64.62</td><td>18.02</td><td>64.36</td></tr><tr><td>mBART-large</td><td>42.00</td><td>22.52</td><td>33.11</td><td>38.94</td><td>19.49</td><td>34.49</td><td>72.97</td><td>20.84</td><td>72.76</td></tr><tr><td>/w Unlimiformer</td><td>42.48</td><td>22.85</td><td>33.61</td><td>39.35</td><td>19.86</td><td>35.02</td><td>72.35</td><td>20.59</td><td>72.28</td></tr><tr><td colspan="10">RO→MD</td></tr><tr><td>BART-base</td><td>28.56</td><td>9.89</td><td>20.19</td><td>25.94</td><td>10.17</td><td>21.07</td><td>37.84</td><td>13.49</td><td>35.06</td></tr><tr><td>/w Unlimiformer</td><td>29.25</td><td>10.76</td><td>20.77</td><td>26.86</td><td>10.96</td><td>21.91</td><td>37.47</td><td>13.25</td><td>34.89</td></tr><tr><td>BART-large</td><td>30.99</td><td>11.05</td><td>22.18</td><td>27.77</td><td>11.16</td><td>22.37</td><td>38.11</td><td>14.03</td><td>36.57</td></tr><tr><td>/w Unlimiformer</td><td>31.40</td><td>11.69</td><td>21.78</td><td>27.86</td><td>11.20</td><td>22.45</td><td>38.37</td><td>14.30</td><td>36.79</td></tr><tr><td>mBART-large</td><td>32.93</td><td>12.26</td><td>23.98</td><td>31.92</td><td>12.11</td><td>27.58</td><td>40.20</td><td>16.07</td><td>40.72</td></tr><tr><td>/w Unlimiformer</td><td>32.87</td><td>12.14</td><td>24.73</td><td>31.35</td><td>12.43</td><td>27.88</td><td>40.34</td><td>16.10</td><td>40.70</td></tr><tr><td colspan="10">MD→RO</td></tr><tr><td>BART-base</td><td>33.70</td><td>14.91</td><td>24.85</td><td>31.47</td><td>14.80</td><td>26.89</td><td>23.22</td><td>7.12</td><td>20.82</td></tr><tr><td>/w Unlimiformer</td><td>33.07</td><td>14.50</td><td>21.17</td><td>31.16</td><td>14.69</td><td>25.95</td><td>23.32</td><td>7.46</td><td>21.08</td></tr><tr><td>BART-large</td><td>35.46</td><td>16.37</td><td>26.26</td><td>34.57</td><td>17.23</td><td>29.33</td><td>24.38</td><td>7.46</td><td>21.88</td></tr><tr><td>/w Unlimiformer</td><td>36.35</td><td>17.58</td><td>27.14</td><td>35.23</td><td>18.19</td><td>30.32</td><td>23.68</td><td>7.04</td><td>21.16</td></tr><tr><td>mBART-large</td><td>37.19</td><td>17.11</td><td>28.38</td><td>36.76</td><td>16.88</td><td>32.09</td><td>28.10</td><td>8.81</td><td>27.20</td></tr><tr><td>/w Unlimiformer</td><td>38.13</td><td>16.86</td><td>28.28</td><td>37.16</td><td>16.97</td><td>31.86</td><td>28.21</td><td>8.87</td><td>27.08</td></tr></table>

Table 2: Evaluation results of the BART models for summary, headline, and keyword generation for the four subtasks proposed on RoLargeSum.

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>R-1   R-2   R-L</td></tr><tr><td rowspan=1 colspan=1>Llama-2-7B</td><td rowspan=1 colspan=1>22.23  10.04  16.83</td></tr><tr><td rowspan=1 colspan=1>Llama-2-7B-It</td><td rowspan=1 colspan=1>30.01 13.39 20.93</td></tr><tr><td rowspan=1 colspan=1>Llama-3-8B</td><td rowspan=1 colspan=1>25.81  12.23  18.79</td></tr><tr><td rowspan=1 colspan=1>Llama-3-8B-It</td><td rowspan=1 colspan=1>16.84  7.55  11.79</td></tr><tr><td rowspan=1 colspan=1>Llama-3.1-8B</td><td rowspan=1 colspan=1>27.07  12.71  19.39</td></tr><tr><td rowspan=1 colspan=1>Llama-3.1-8B-It</td><td rowspan=1 colspan=1>28.72  13.16 20.00</td></tr><tr><td rowspan=1 colspan=1>Mistral-7B-v0.1</td><td rowspan=1 colspan=1>7.45   4.14   6.09</td></tr><tr><td rowspan=1 colspan=1>Mistral-7B-It-v0.1</td><td rowspan=1 colspan=1>27.27  12.54  19.16</td></tr><tr><td rowspan=1 colspan=1>Mistral-7B-v0.2</td><td rowspan=1 colspan=1>9.61   5.14   7.78</td></tr><tr><td rowspan=1 colspan=1>Mistral-7B-It-v0.2</td><td rowspan=1 colspan=1>27.97  11.73  18.66</td></tr><tr><td rowspan=1 colspan=1>Mistral-7B-v0.3</td><td rowspan=2 colspan=1>9.63   5.13   7.8231.16 14.12  21.37</td></tr><tr><td rowspan=1 colspan=1>Mistral-7B-It-v0.3</td></tr><tr><td rowspan=1 colspan=1>Gemma-7B</td><td rowspan=1 colspan=1>19.12 09.85  14.49</td></tr><tr><td rowspan=1 colspan=1>Gemma-7B-It</td><td rowspan=1 colspan=1>03.60 01.41  02.85</td></tr><tr><td rowspan=1 colspan=1>Gemma-1.1-7B-It</td><td rowspan=2 colspan=1>04.88 02.17 04.0924.74 13.20 19.1205.08 03.55 04.76</td></tr><tr><td rowspan=1 colspan=1>Gemma2-9BGemma2-9B-It</td></tr><tr><td rowspan=2 colspan=1>RoLlama2-7BRoLlama2-7B-It</td><td rowspan=1 colspan=1>29.00 15.25 22.23</td></tr><tr><td rowspan=1 colspan=1>30.14  13.88 21.40</td></tr><tr><td rowspan=1 colspan=1>RoQLlama2-7B</td><td rowspan=1 colspan=1>25.34 11.37  18.88</td></tr><tr><td rowspan=3 colspan=1>RoLlama3-8B-ItRoMistral-7B-ItRoGemma-7B-It</td><td rowspan=1 colspan=1>29.39  13.01  20.10</td></tr><tr><td rowspan=1 colspan=1>29.77 13.68 20.63</td></tr><tr><td rowspan=1 colspan=1>26.66 12.14 18.61</td></tr></table>

Table 3: Evaluation results of multilingual and Romanian LLMs from the Llama, Mistral, and Gemma families.

MD→RO The results are again mixed for this subtask, similar to the RO→MD subtask. mBARTlarge obtained the top score R-1 (i.e., 38.13) for summarization by employing Unlimiformer, while the plain version obtained the highest scores R-2 and R-L. For headline generation, the Unlimiformer wrapper helped improve the results for the R-1 and R-2 scores, with mBART-large obtaining 37.16 and 16.97, respectively, but it did not help on the R-L score where the plain version achieved an R-L score of 32.09. The same pattern also applies to keyword generation, namely the highest R-1 and R-2 scores were obtained by mBART-large with Unlimiformer, and the highest R-L scores were obtained without this wrapper.

## 5.2 Results for LLMs

The results of the evaluation of LLMs for the RO+MD summarization task on RoLargeSum are presented in Table 3. The highest R-1 score was obtained by the Mistral-7B-It-v0.3 model with 31.16, while the highest R-2 and R-L scores were obtained by RoLlama2-7B with 15.25 and 22.23, respectively, underperforming compared to all fine-tuned BART models in this task.

<table><tr><td>Metric</td><td>Summary</td><td>Headline</td><td>Keywords</td></tr><tr><td>Coherence</td><td>4.3</td><td>4.8</td><td></td></tr><tr><td>Consistency</td><td>3.9</td><td>4.1</td><td></td></tr><tr><td>Coverage</td><td>3.4</td><td>4.3</td><td>4.4</td></tr><tr><td>Fluency</td><td>4.4</td><td>4.5</td><td></td></tr><tr><td>Overall</td><td>3.9</td><td>4.3</td><td>4.3</td></tr></table>

Table 4: Human evaluation of the best-performing model on RoLargeSum.

Additionally, we can see that in four of five instances, the Romanian variants of the model families performed better than their multilingual counterparts on all metrics: (1) RoGemma-7B-It outperformed all Gemma models, (2) RoLlama3-8B-it outperformed all Llama-3 models, (3) RoLlama2- 7B and RoQLlama2-7B outperformed Llama-2- 7B, (4) RoLlama2-7B-it outperformed Llama-2- 7B-It, with RoMistral-7B-It being surpassed only by Mistral-7B-It-v0.3.

## 5.3 Human Evaluation

We further manually analyze the ability of the bestperforming system, mBART-large + Unlimiformer + Loss Reversal, to generate summaries, headlines, and keywords on a subset of 100 documents extracted from the RoLargeSum’s test set by following (El-Shangiti et al., 2024; Barta et al., 2024). We accomplish this using five annotators who are required to assign a score ranging from 1 to 5 in the following categories to each generated summary, headline, and keywords (SHKs)<sup>2</sup>:

• Coherence: the generated SHKs are wellorganized and easy to read.

• Consistency: the generated SHKs are consistent with the information found in the news article.

• Coverage: the main ideas of the news article are covered by the generated SHKs.

• Fluency: the generated SHKs are written in good Romanian.

• Overall: how good the SHKs are overall.

The results are depicted in Table 4. We can observe that both the generated summaries and headlines are somewhat coherent and use the Romanian language correctly. However, they are less consistent with their respective news articles, and the summaries usually have low coverage of the full news articles. At the same time, keywords and headlines typically cover the full content. Finally, the headlines and keywords obtained a relatively high overall score, while, in general, the generated summaries were considered by the annotators less representative of the news articles.

## 6 Conclusion

This paper introduces RoLargeSum, a large dataset with more than 615K samples that can generate summaries, headlines, and keywords in the Romanian language. The data was retrieved from multiple news websites and cleaned to ensure high quality. We further distinguished between news articles written by authors in the Republic of Moldova and Romania, which allowed us to boost the performance of the evaluated BART models, and, together with the Unlimiformer wrapper, we managed to propose several strong baselines that other researchers can use to either build upon or compare to. Finally, we intend to incorporate both the Ro-LargeSum dataset and the results obtained by the evaluated models into the LiRo benchmark (Dumitrescu et al., 2021).

## 7 Limitations

The distribution of dialects in RoLargeSum is not uniform. This difference is mainly due to the varying number of newspapers in Romania and the Republic of Moldova. This unbalanced distribution highlights the importance of considering geographic and media-related factors when conducting data-driven research or analysis on this dataset.

## 8 Ethical Considerations

To align with ethical research practices, it is essential to clarify that the data collection process for the RoLargeSum dataset fully respects copyrights and intellectual property rights. The dataset consists of articles and their metadata from publicly accessible websites of major Romanian and Moldavian newspapers. Each news article in the dataset is adequately cited, including its source, with an associated URL for verification and direct reference. The data is used exclusively for academic and research purposes to advance the field of Romanian text summarization. The extraction and use of these data comply with all relevant ethical guidelines, ensuring the journalistic integrity of the original articles and their authors is preserved. Thus, the RoLargeSum dataset is intended to serve as a high-quality and diverse resource for research while upholding all ethical and legal standards.

## Acknowledgements

This work was supported by the National University of Science and Technology POLITEHNICA Bucharest through the PubArt program and a grant from the National Program for Research of the National Association of Technical Universities, GNAC ARUT 2023.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Andrei-Marius Avram, Andreea Iuga, George-Vlad Manolache, Vlad-Cristian Matei, Razvan-Gabriel Mi-˘ cliu¸s, Vlad-Andrei Muntean, Manuel-Petru Sorlescu, Drago-Andrei ¸Serban, Adrian-Dinu Urse, Vasile Pai¸s,˘ et al. 2024. Histnero: Historical named entity recognition for the romanian language. In International Conference on Document Analysis and Recognition, pages 126–144. Springer.

Andrei-Marius Avram, Verginica Barbu Mititelu, Vasile Pai˘ s, Dumitru-Clementin Cercel, and Stefan Trau˘ san-Matu. 2023. Multilingual multiword expression identification using lateral inhibition and domain adaptation. Mathematics, 11(11):2548.

Andrei-Marius Avram, Mihai-Virgil Nichita, Razvan-George Bartusica, and Mad˘ alin-Virgil Mihai. 2022.˘ Rosac: A speech corpus for transcribing romanian emergency calls. In 2022 14th International Conference on Communications (COMM), pages 1–5. IEEE.

Botond Barta, Dorina Lakatos, Attila Nagy, Milán Konor Nyist, and Judit Ács. 2024. From news to summaries: Building a hungarian corpus for extractive and abstractive summarization. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 7503–7509.

Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew Gormley. 2024. Unlimiformer: Long-range

transformers with unlimited length input. Advances in Neural Information Processing Systems, 36.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2024. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53.

Alexandra Ciobotaru, Mihai Vlad Constantinescu, Liviu P Dinu, and Stefan Dumitrescu. 2022. Red v2: enhancing red dataset for multi-label emotion detection. In Proceedings ofthe Thirteenth Language Resources and Evaluation Conference, pages 1392– 1399.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171– 4186.

George-Andrei Dima, Andrei-Marius Avram, Cristian-George Craciun, and Dumitru-Clementin Cercel. 2024. Roqllama: A lightweight romanian adapted language model. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 4531–4541.

Yue Dong, Yikang Shen, Eric Crawford, Herke van Hoof, and Jackie Chi Kit Cheung. 2018. Banditsum: Extractive summarization as a contextual bandit. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3739–3748.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Stefan Dumitrescu, Petru Rebeja, Beata Lorincz, Mihaela Gaman, Andrei Avram, Mihai Ilie, Andrei Pruteanu, Adriana Stan, Lorena Rosia, Cristina Iacobescu, Luciana Morogan, George Dima, Gabriel Marchidan, Traian Rebedea, Madalina Chitez, Dani Yogatama, Sebastian Ruder, Radu Tudor Ionescu, Razvan Pascanu, and Viorica Patraucean. 2021. Liro: Benchmark and leaderboard for romanian language tasks. In Proceedings ofthe Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1.

¸Stefan Daniel Dumitrescu and Andrei-Marius Avram. 2020. Introducing ronec-the romanian named entity corpus. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 4436– 4443.

Sebastian-Vasile Echim, Razvan-Alexandru Sm˘ adu,˘ Andrei-Marius Avram, Dumitru-Clementin Cercel, and Florin Pop. 2023. Adversarial capsule networks for romanian satire detection and sentiment analysis. In International Conference on Applications ofNatural Language to Information Systems, pages 428–442. Springer.

Moussa Kamal Eddine, Antoine Tixier, and Michalis Vazirgiannis. 2021. Barthez: a skilled pretrained french sequence-to-sequence model. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 9369–9390.

Ahmed El-Shangiti, Fakhraddin Alwajih, and Muhammad Abdul-Mageed. 2024. Arabic automatic story generation with large language models. In Proceedings of The Second Arabic Natural Language Processing Conference, pages 140–152.

Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario March, and Victor Lempitsky. 2016. Domainadversarial training of neural networks. Journal of machine learning research, 17(59):1–35.

José-Ángel González, Annie Louis, and Jackie Chi Kit Cheung. 2022. Source-summary entity aggregation in abstractive summarization. In Proceedings ofthe 29th International Conference on Computational Linguistics, pages 6019–6034.

Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. 2015. Teaching machines to read and comprehend. Advances in neural information processing systems, 28.

Diana Constantina Hoefels, Çagrı Çöltekin, and Irina Di-˘ ana Madroane. 2022. Coroseof-an annotated corpus˘ of romanian sexist and offensive tweets. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 2269–2281.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Xiang Jiang and Markus Dreyer. 2024. Ccsum: A largescale and high-quality dataset for abstractive news summarization. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7299–7329.

Andrei Cosmin Jitaru, ¸Seila Abdulamit, and Bogdan Ionescu. 2020. Lrro: a lip reading data set for the under-resourced romanian language. In Proceedings of the 11th ACM Multimedia Systems Conference, pages 267–272.

Abdelrhman A Karim, Mohammed Usama, Manar A Ibrahim, Youssef Hatem, Waleed Wael, Ahmad T Mazrua, Ghada K El-Monayer, Magy Elbanhawy, Khaled Foad, and Ibrahim F Moawad. 2024. Arabic abstractive summarization using the multilingual t5 model. In 2024 6th International Conference on Computing and Informatics (ICCI), pages 223–228. IEEE.

Wojciech Krysci´ nski, Nazneen Rajani, Divyansh Agar-´ wal, Caiming Xiong, and Dragomir Radev. 2022. Booksum: A collection of datasets for long-form narrative summarization. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, pages 6536–6558.

Jingun Kwon, Hidetaka Kamigaito, and Manabu Okumura. 2023. Abstractive document summarization with summary-length prediction. In Findings ofthe Association for Computational Linguistics: EACL 2023, pages 618–624.

Khang Lam, Thieu Doan, Khang Pham, and Jugal Kalita. 2023. Abstractive text summarization using the brio training paradigm. In Findings of the Association for Computational Linguistics: ACL 2023, pages 92–99.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 7871–7880.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Chin-Yew Lin and Eduard Hovy. 2003. Automatic evaluation of summaries using n-gram co-occurrence statistics. In Proceedings of the 2003 human language technology conference ofthe North American chapter ofthe associationfor computational linguistics, pages 150–157.

Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. 2020. Multilingual denoising pretraining for neural machine translation. Transactions of the Association for Computational Linguistics, 8:726–742.

Emilian-Claudiu Manescu, R˘ azvan-Alexandru Sm˘ adu,˘ Andrei-Marius Avram, Dumitru-Clementin Cercel, and Florin Pop. 2023. End-to-end lip reading in romanian with cross-lingual domain adaptation and lateral inhibition. In 2023 IEEE International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT), pages 287–293. IEEE.

Yuning Mao, Ming Zhong, and Jiawei Han. 2022. Citesum: Citation text-guided scientific extreme summarization and domain adaptation with limited supervision. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10922–10935.

Mihai Masala, Denis C Ilie-Ablachim, Alexandru Dima, Dragos Corlatescu, Miruna Zavelca, Ovio Olaru, Simina Terian-Dan, Andrei Terian-Dan, Marius Leordeanu, Horia Velicu, et al. 2024. " vorbe\c {s} ti rom\ˆ ane\c {s} te?" a recipe to train powerful romanian llms with english instructions. arXiv preprint arXiv:2406.18266.

Vlad-Cristian Matei, Iulian-Marius Taiatu, R˘ azvan-˘ Alexandru Smadu, and Dumitru-Clementin Cercel.˘ 2024. Enhancing romanian offensive language detection through knowledge distillation, multi-task learning, and data augmentation. In International Conference on Applications of Natural Language to Information Systems, pages 317–332. Springer.

Alfonso Mendes, Shashi Narayan, Sebastião Miranda, Zita Marinho, André FT Martins, and Shay B Cohen. 2019. Jointly extracting and compressing documents with summary state representations. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3955–3966.

Ramesh Nallapati, Bowen Zhou, Cicero dos Santos, Caglar Gulcehre, and Bing Xiang. 2016. Abstractive text summarization using sequence-to-sequence rnns and beyond. In Proceedings ofthe 20th SIGNLL Conference on Computational Natural Language Learning, pages 280–290.

Shashi Narayan, Shay B Cohen, and Mirella Lapata. 2018a. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1797–1807.

Shashi Narayan, Shay B Cohen, and Mirella Lapata. 2018b. Ranking sentences for extractive summarization with reinforcement learning. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1747–1759.

Elena-Beatrice Nicola, Dumitru-Clementin Cercel, and Florin Pop. 2024. Investigating the impact of semisupervised methods with data augmentation on offensive language detection in romanian language. Procedia Computer Science, 246:3447–3456.

Mihai Alexandru Niculescu, Stefan Ruseti, and Mihai Dascalu. 2022. Rosummary: Control tokens for romanian news summarization. Algorithms, 15(12):472.

Tikhonov Pavel, Anastasiya Ianina, and Valentin Malykh. 2024. Sumhis: Extractive summarization exploiting hidden structure. arXiv preprint arXiv:2406.08215.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67.

Ana-Cristina Rogoz, Gaman Mihaela, and Radu Tudor Ionescu. 2021. Saroco: Detecting satire in a novel romanian corpus of news articles. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 1073–1079.

Gerard Salton and Christopher Buckley. 1988. Termweighting approaches in automatic text retrieval. Information processing & management, 24(5):513– 523.

Agata Savary, Marie Candito, Verginica Barbu Mititelu, Eduard Bejcek, Fabienne Cap, Slavomírˇ Cé-<sup>ˇ</sup> plö, Silvio Ricardo Cordeiro, Gül¸sen Cebiroglu Ery-˘ igit, Voula Giouli, Maarten van Gompel, et al. 2018.˘ Parseme multilingual corpus of verbal multiword expressions. In Multiword expressions at length and in depth: Extended papersfrom the MWE 2017 workshop.

Encarnación Segarra Soriano, Vicent Ahuir, Lluís-F Hurtado, and José González. 2022. Dacsa: A largescale dataset for automatic summarization of catalan and spanish newspaper articles. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5931–5943.

Ilya Sutskever, Oriol Vinyals, and Quoc V Le. 2014. Sequence to sequence learning with neural networks. Advances in neural information processing systems, 27.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. 2024. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118.

Surendrabikram Thapa, Kritesh Rauniyar, Shuvam Shiwakoti, Sweta Poudel, Usman Naseem, and Mehwish Nasim. 2023. Nehate: Large-scale annotated data shedding light on hate speech in nepali local election discourse. In ECAI 2023, pages 2346–2353. IOS Press.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Mingye Wang, Pan Xie, Yao Du, and Xiaohui Hu. 2023. T5-based model for abstractive summarization: A semi-supervised learning approach with consistency loss functions. Applied Sciences, 13(12):7111.

Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Édouard Grave. 2020. Ccnet: Extracting high quality monolingual datasets from web crawl data. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 4003– 4012.

Qianqian Xie, Jennifer Amy Bishop, Prayag Tiwari, and Sophia Ananiadou. 2022. Pre-trained language models with domain knowledge for biomedical extractive summarization. Knowledge-Based Systems, 252:109460.

Kaichun Yao, Libo Zhang, Tiejian Luo, and Yanjun Wu. 2018. Deep reinforcement learning for extractive document summarization. Neurocomputing, 284:52– 62.

Haopeng Zhang, Xiao Liu, and Jiawei Zhang. 2023. Extractive summarization via chatgpt for faithful summary generation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 3270–3278.

Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Peter Liu. 2020. Pegasus: Pre-training with extracted gap-sentences for abstractive summarization. In International conference on machine learning, pages 11328–11339. PMLR.

Ming Zhong, Pengfei Liu, Yiran Chen, Danqing Wang, Xipeng Qiu, and Xuan-Jing Huang. 2020. Extractive summarization as text matching. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6197–6208.

## A Dataset Cleaning Rules

The most important rules that we used to clean the RoLargeSum dataset were:

• Removing news articles that had less than 100 characters.

• Removing samples for which the ratio between the news article and its corresponding summary was less than 1.2x.

• Removing news articles that contained more than 15% of their content written in the Cyrillic alphabet<sup>3</sup>.

• Removing crawling artifacts found in news articles such as “Articolul urmator” (eng. “Next˘ article”) or “Citeste si” (eng. “Read also”).

## B RoLargeSum Statistics

We outline the statistics of the RoLargeSum dataset for the news articles written by people in Romania and the Republic of Moldova in Table 5 and Table 6, respectively.

## C TF-IDF Analysis

We perform a TF-IDF analysis of the news articles found in RoLargeSum. The product of TF and ID is known as TF-IDF, where: (i) TF counts the frequency of each word that occurs in a particular document, and (ii) IDF calculates the frequency of the corresponding term in all documents (Salton and Buckley, 1988; Thapa et al., 2023). We note that the terms that are infrequent in the entire dataset but frequent in a small selection of texts are indicated by higher TF-IDF scores.

The results are summarized in Table 7. An important observation of this analysis is the domain of the extracted words. The words in the Romanian news articles with the highest TF-IDF score belong to Western company names (e.g., “Analytica”, “Kogan”, “Facebook”) or the technology domain (e.g., “aplicatie” — eng. “application”, “colectat” — eng. “collected”). Furthermore, most of the terms in the Moldavian news articles with the highest TF-IDF score are based on political names (e.g., “Voronin”, “Chetaru”), political office jobs (e.g., “Presedintelui” — eng. “President’s”, “Premierului” — eng. “Prime Minister’s”), or political parties (e.g., “PCRM” eng. “Party of Communists of the Republic of Moldova”, “PLDM” — eng. “Liberal Democratic Party of Moldova”). Thus, this analysis might indicate that the topic information of Romanian news is more externally focused, mostly on Western media, while Moldavian news is more internally focused on internal politics.

## D Large Language Models

We evaluate the performance of several LLMs that are pre-trained on Romanian or multilingual datasets and fall into one of three model families: Llama, Gemma, and Mistral.

Llama We perform experiments with LLama2- 7B (Touvron et al., 2023), LLama3-8B, and LLama3.1-8B (Dubey et al., 2024), using both the base and the instruction fine-tuned versions for all these variants.

Gemma We also test the performance of three Gemma LLMS (i.e., Gemma-7B, Gemma-7B-It, and Gemma-1.1.-7B-It) (Team et al., 2024), and two Gemma2 variants (i.e., Gemma2-9B and Gemma2-9B-It) (Team et al., 2024) on RoLarge-Sum.

Mistral We further evaluate the performance of the 7B Mistral versions (i.e., v0.1, v0.2, and v0.3) (Jiang et al., 2023), using both the base and the instruction fine-tuned variants.

Romanian LLMs Finally, we assess the performance of Romanian LLMs on RoLargeSum, which belong to the three families of LLMs that we were interested in as follows: RoLLama2-7B, RoLlama2-7B-It, RoLlama3-8B-It, RoMistral-7B-It, RoGemma-7B-It (Masala et al., 2024), and RoQLlama-7B (Dima et al., 2024).

## E Evaluation Metrics

Three versions of the Recall-Oriented Understudy for Gisting Evaluation (ROUGE) score (Lin and Hovy, 2003) were used to assess all of the models we used in our experiments: unigram-based ROUGE (R-1), bigram-based ROUGE (R-2), and longest common sequence ROUGE (R-L). Equation 6 provides the formula used to determine the R-1 and R-2 scores:

$$
\operatorname { R } - \operatorname { N } = { \frac { \displaystyle \sum _ { S \in { \mathcal { D } } } \sum _ { g \in S } { \mathrm { C o u n t } } _ { \operatorname { m a t c h } } ( g ) } { \displaystyle \sum _ { S \in { \mathcal { D } } } \sum _ { g \in S } { \mathrm { C o u n t } } ( g ) } }\tag{6}
$$

where D is the dataset containing the reference summaries, S is a summary in this dataset, g is either a unigram or bigram in S, and Count<sub>match</sub> calculates the matching grams between the reference summary S and the candidate summary, and Count simply counts the grams found in S.

On the other hand, R-L is computed as described in Equation 7:

$$
\mathrm { R - L } = { \frac { L C S ( C , S ) } { | S | } }\tag{7}
$$

<table><tr><td>Dataset</td><td>Train/Val/Test</td><td colspan="2">Avg. News Len. Words Sents.</td><td colspan="2">Avg. Sum. Len. Words Sents.</td><td colspan="2">Vocab. Size News Sums.</td></tr><tr><td>RoLargeSum-Summary</td><td>108K/2.5K/2.5K</td><td>387.46</td><td>12.57</td><td>49.56</td><td>1.48</td><td>496K</td><td>153K</td></tr><tr><td>RoLargeSum-Headline</td><td>104K/2.5K/2.5K</td><td>389.12</td><td>12.69</td><td>18.73</td><td>1.40</td><td>491K</td><td>81K</td></tr><tr><td>RoLargeSum-Keywords</td><td>108K/2.5K/2.5K</td><td>387.58</td><td>12.58</td><td>9.31</td><td>1.00</td><td>496K</td><td>42K</td></tr></table>

Table 5: RoLargeSum dataset statistics for the news articles from Romania.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Train/Val/Test</td><td colspan="2">Avg. News Len.</td><td colspan="2">Avg. Sum. Len.</td><td colspan="2">Vocab. Size</td></tr><tr><td>Words</td><td>Sents.</td><td>Words</td><td>Sents.</td><td>News</td><td>Sums.</td></tr><tr><td>RoLargeSum-Summary</td><td>411/2.5K/2.5K</td><td>314.78</td><td>9.08</td><td>46.80</td><td>1.63</td><td>897K</td><td>206K</td></tr><tr><td>RoLargeSum-Headline</td><td>499K/2.5K/2.5K</td><td>314.80</td><td>9.04</td><td>15.21</td><td>1.17</td><td>1.0M</td><td>118K</td></tr><tr><td>RoLargeSum-Keywords</td><td>308K/2.5K/2.5K</td><td>313.45</td><td>7.75</td><td>6.95</td><td>1.00</td><td>774K</td><td>47K</td></tr></table>

Table 6: RoLargeSum dataset statistics for the news articles from the Republic of Moldova.
<table><tr><td rowspan=1 colspan=1>Country Name</td><td rowspan=1 colspan=1>Words</td><td rowspan=1 colspan=1>Translation</td><td rowspan=1 colspan=1>TF-IDF Score</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Analytica</td><td rowspan=1 colspan=1>Analytica</td><td rowspan=1 colspan=1>0.447</td></tr><tr><td rowspan=9 colspan=1>Romania</td><td rowspan=1 colspan=1>Cambridge</td><td rowspan=1 colspan=1>Cambridge</td><td rowspan=1 colspan=1>0.386</td></tr><tr><td rowspan=1 colspan=1>Kogan</td><td rowspan=1 colspan=1>Kogan</td><td rowspan=1 colspan=1>0.301</td></tr><tr><td rowspan=1 colspan=1>Wylie</td><td rowspan=1 colspan=1>Wylie</td><td rowspan=1 colspan=1>0.191</td></tr><tr><td rowspan=1 colspan=1>Facebook</td><td rowspan=1 colspan=1>Moldavia&#x27;s</td><td rowspan=1 colspan=1>0.176</td></tr><tr><td rowspan=1 colspan=1>Alexandr</td><td rowspan=1 colspan=1>Alexandr</td><td rowspan=1 colspan=1>0.168</td></tr><tr><td rowspan=1 colspan=1>colectat</td><td rowspan=1 colspan=1>collected</td><td rowspan=1 colspan=1>0.132</td></tr><tr><td rowspan=1 colspan=1>Christopher</td><td rowspan=1 colspan=1>Christopher</td><td rowspan=1 colspan=1>0.128</td></tr><tr><td rowspan=1 colspan=1>aplicație</td><td rowspan=1 colspan=1>application</td><td rowspan=1 colspan=1>0.118</td></tr><tr><td rowspan=1 colspan=1>utilizatorilor</td><td rowspan=1 colspan=1>user&#x27;s</td><td rowspan=1 colspan=1>0.116</td></tr><tr><td rowspan=10 colspan=1>Moldova</td><td rowspan=1 colspan=1>Voronin</td><td rowspan=1 colspan=1>Voronin</td><td rowspan=1 colspan=1>0.231</td></tr><tr><td rowspan=1 colspan=1>Chetraru</td><td rowspan=1 colspan=1>Chetraru</td><td rowspan=1 colspan=1>0.219</td></tr><tr><td rowspan=1 colspan=1>Președintelui</td><td rowspan=1 colspan=1>President&#x27;s</td><td rowspan=1 colspan=1>0.209</td></tr><tr><td rowspan=1 colspan=1>propunerea</td><td rowspan=1 colspan=1>the proposal</td><td rowspan=1 colspan=1>0.201</td></tr><tr><td rowspan=1 colspan=1>Anticorupție</td><td rowspan=1 colspan=1>Anti-corruption</td><td rowspan=1 colspan=1>0.181</td></tr><tr><td rowspan=1 colspan=1>Național</td><td rowspan=1 colspan=1>National</td><td rowspan=1 colspan=1>0.153</td></tr><tr><td rowspan=1 colspan=1>PCRM</td><td rowspan=1 colspan=1>PCRM</td><td rowspan=1 colspan=1>0.146</td></tr><tr><td rowspan=1 colspan=1>Parlamentului</td><td rowspan=1 colspan=1>Parliament&#x27;s</td><td rowspan=1 colspan=1>0.144</td></tr><tr><td rowspan=1 colspan=1>Premierului</td><td rowspan=1 colspan=1>Prime Minister&#x27;s</td><td rowspan=1 colspan=1>0.142</td></tr><tr><td rowspan=1 colspan=1>PLDM</td><td rowspan=1 colspan=1>LDPM</td><td rowspan=1 colspan=1>0.142</td></tr></table>

Table 7: TF-IDF analysis of the top ten most common words in the RoLargeSum news articles in Romania and Moldova.

where LCS is the longest common sequence between the candidate summary C and the reference summary S, and |S| represents the word count in S.

## F Implementation Details

We performed a linear warm-up for the first 5k steps of the training process and fine-tuned the

BART models for ten epochs with a learning rate of 3e-5, a batch size of 16, and a gradient accumulation of 4 for a total batch size of 64. During inference, we decoded the output using a beam search with a beam width of 16. For adversarial training, we set $\lambda _ { G R } = 0 . 0 1$ for the gradient reversal method and $\lambda _ { L R } = 0 . 0 0 5$ for the loss reversal method.

For the summaries generation task using Romanian LLMs, we simply used HuggingingFace’s pipeline for text generation with the default parameters<sup>4</sup>.

## G Summarization Prompt

We evaluated the LLMs on the RoLargeSum dataset for Romanian summarization by inserting both the headline of the article and the text contained in the article using the following prompt:

Eu c i t e s c u r m a t o r u l p a r a g r a f s i   
i l s u m a r i z e z .   
T i t l u : { t i t l u }   
T e x t : { t e x t }   
S u m a r i z a r e :

where the fields in the curly brackets are filled with the corresponding headlines and documents in the test set of RoLargeSum.

The English version of the prompt is the following:

I r e a d t h e fo l l o w i n g p a r a g r a p h   
and summarize i t .   
H e a d l i n e : { h e a d l i n e }   
T e x t : { T e x t }   
Summary :

## H Generation Examples

In Tables 8, 9 and 10, we show several examples that were generated using the best-performing system (i.e., mBART-large with Unlimiformer and Loss Reversal) for summaries, headlines, and keywords, both in Romanian and translated into English. We can observe that the generated summaries, headlines, and keywords align with their respective reference text and the original news article.

First Example In the first example, the model does not recall in the summary the information on the dissolution of the federal agency that identified wildfire outbreaks in Siberia presented in the last part of the article. The generated headline fits well with the main subject of the article. The model was able to extract several relevant keywords, although they were not identical to the reference keywords.

Second Example In the second example, the topperforming system was able to extract the main takeaways of the article in the summary. It generated an almost identical headline to the reference headline and correctly identified all the reference keywords except the first one.

Third Example In the third example, the model does not incorporate the summary of the information presented in the last part of the article on the employment status on the Romanian coast. The candidate headline fits well with the main message of the article and is much shorter than the candidate headline. The extracted keywords are relevant to the article, but the model misses an important one about the Mamaia resort; however, it correctly identifies the Vama Veche resort. Also, it captures an important keyword (i.e., “salarii” — eng. “salaries”) that is absent from the reference keywords.

<table><tr><td>Document: Raportul ONU pe urgența schimbărilor climatice e confirmat de o informație de la NASA. Pentru prima dată în istorie, fumul de la incendiile de vegetație din Siberia a ajuns la Polul Nord. Nici nu e de mirare. În regiunea Yakutia ard 3,4 milioane de hectare de pădure, iarbă și arbuști. Într-un sat, 400 de locuitori au fost evacuați sâmbătă din calea flăcărilor. Acum, ei s-au întors dar au găsit totul în ruine. Au ars case multe. 34 de case, e un număr mare. Erau multe apartamente. Două case cu câte patru apartamente au ars pe partea asta. Două case cu câte două apartamente au ars în partea cealaltă. Au ars și case private", a spus Ilya Avakumov, locuitor din Byas-Kuel.Incendiile din Siberia sunt provocate de verile tot mai calde. La proporțiile dezastrului a contribuit și statul, care a desființat, în 2007, o agenție aviatică federală care avea ca scop identificarea focarelor cu ajutorul avioanelor. In plus, o lege permite autorităților locale să lase incendiile să ardă, dacă flăcările nu amenință satele, iar costurile de stingere sunt prea mari. Reference Summary: Fumul de la incendiile din Siberia a ajuns la Polul Nord, la 3.000 de km depărtare. Anul acesta, focul a distrus peste 14 milioane de hectare de vegetație în Siberia. O agenție care repera din</td></tr><tr><td>timp focarele a fost desființată în 2007. Candidate Summary: Pentru prima dată în istorie, incendiile de vegetație din Siberia au ajuns la Polul Nord. În regiunea Yakutia ard 3,4 milioane de hectare de pădure, iarbă și arbuști. Incendiile din Siberia sunt provocate de verile tot mai calde. Reference Headline: Fumul de la incendiile din Siberia a ajuns la Polul Nord. Focul a distrus deja 14</td></tr><tr><td>milioane de hectare de vegetație Candidate Headline: Pentru prima dată în istorie, fumul de la incendiile de vegetație din Siberia a ajuns la Polul Nord Reference Keywords: incendii siberia, polul nord, schimbri climatice</td></tr><tr><td>Candidate Keywords: siberia, incendii de vegetatie, polul nord Document: The UN report on the urgency of climate change is confirmed by information from NASA. For the first time in history, the smoke from wildfires in Siberia has reached the North Pole. It is no wonder, either. In the Yakutia region, 3.4 million hectares of forest, grass and shrubs are burning. In one village, 400 inhabitants were evacuated on Saturday from the path of the flames. Now, they returned but found everything in ruins. "Many houses burned down. Thirty-four houses, that's a big number. There were many apartments. Two houses with four apartments each burned on this side. Two houses with two apartments each burned on the other side. Private houses also burned," said Ilya Avakumov, a resident of Byas-Kuel. The fires in Siberia are caused by increasingly hot summers. The state also contributed to the scale of the disaster, which in 2007 abolished a federal aviation agency that aimed to identify outbreaks with the help of aeroplanes. In addition, a law allows local authorities to let the fires burn if the flames do not threaten</td></tr><tr><td>villages and the costs of extinguishing them are too high.</td></tr><tr><td>Reference Summary: Smoke from fires in Siberia reached the North Pole, 3,000 km away. This year, the fire destroyed more than 14 million hectares of vegetation in Siberia. An agency that used to spot outbreaks early was disbanded in 2007. Candidate Summary: For the first time in history, wildfires in Siberia have reached the North Pole. In the Yakutia region, 3.4 million hectares of forest, grass and shrubs are burning. The fires in Siberia are caused by increasingly hot summers Reference Headline: Smoke from fires in Siberia has reached the North Pole. The fire has already destroyed 14 million hectares of vegetation Candidate Headline: For the first time in history, smoke from wildfires in Siberia has reached the North</td></tr><tr><td>Document: Meteorologii anună că vremea se va încălzi în majoritatea regiunilor, dar va fi în general instabilă. În jumătatea de est a ării și în zonele montane vor fi perioade cu instabilitate atmosferică accentuată ce se va manifesta prin averse torențiale, descărcări electrice, intensificări ale vântului și izolat grindină și vijelii. În intervale scurte de timp sau prin acumulare, cantitățile de apă vor depăi pe arii restrânse 25... 50 1/mp. În restul teritoriului, cerul va fi variabil, iar astfel de fenomene se vor semnala pe spaii mici. Temperaturile maxime se vor încadra între 19 grade pe litoral și 28 de grade Celsius în Câmpia de Vest, iar cele minime între 10 și 17 grade Celsius. Izolat vor fi condiții de ceaă. Reference Summary: Temperaturile vor fi mai crescute, dar va continua să plouă în</td></tr><tr><td>majoritatea regiunilor. În sudul ării, vremea va fi instabilă. În Capitală, în a doua parte a zilei și la noapte vor fi averse. Candidate Summary: Meteorologii anună că vremea se va încălzi în majoritatea regiu- nilor, dar va fi în general instabilă. Temperaturile maxime se vor încadra între 19 grade pe litoral și 28 de grade Celsius în Câmpia de Vest. Temperaturile minime între 10 și 17 grade Celsius.</td></tr><tr><td>Reference Headline: Vremea se va încălzi în majoritatea regiunilor, dar va fi în general instabilă Candidate Headline: Vremea se încălzește în majoritatea regiunilor, dar va fi în general instabilă Reference Keywords: anm, meteo, prognoza meteo, vreme Candidate Keywords: meteo, prognoza meteo, vreme Document: Meteorologists say the weather will warm up in most regions, but it will be generally unsettled. In the eastern half of the country and the mountainous areas, periods</td></tr><tr><td>of increased atmospheric instability will manifest in which torrential downpours, electrical discharges, wind intensifications, isolated hail, and storms will manifest. In short periods or through accumulation, the amounts of water will exceed 25...50 l/m2 in limited areas. The sky will be variable in the rest of the territory, and such phenomena will be signalled in small spaces. The maximum temperatures will be between 19 degrees on the coast and 28 degrees Celsius in the Western Plain, and the minimum between 10 and 17 degrees Celsius. There will be isolated foggy conditions. Reference Summary: The temperatures will be higher, but it will continue to rain in most</td></tr><tr><td>regions. In the south of the country, the weather will be unstable. In the capital, showers will be provided during the second part of the day and at night. Candidate Summary: Meteorologists announce that the weather will warm up in most regions but will be generally unstable. Maximum temperatures will be between 19 degrees on the coast and 28 degrees Celsius in the Western Plains. Minimum temperatures between 10 and 17 degrees Celsius. Reference Headline: The weather will warm in most regions, but it will be generally unsettled Candidate Headline: The weather is warming up in most regions, but will be generally unsettled Reference Keywords: anm, weather, weather forecast, weather Candidate Keywords: weather, weather forecast, weather</td></tr></table>

Table 8: The first example of generated summaries, headlines and keywords, both in Romanian (top) and translated into English (bottom).  
Table 9: The second example of generated summaries, headlines and keywords, both in Romanian (top) and translated into English (bottom).

Document: Patronii de pe litoralul românesc n-au mai apelat anul acesta la forta de munca str˘ ain˘ a.˘ Angajatorii spun ca au primit zeci de solicit˘ ari de a angaja românii întor˘ si acasa din cauza pandemiei.˘ Administratorii din Vama Veche spun ca salariile angaja˘ tilor calificati se duc spre 2.000 de euro, iar urmatoarea categorie trece de 1.000 de euro. La Mamaia, salariile sunt îns˘ a altele. Un barmen câs˘ tiga˘ pe luna 3.500-4.000 de lei, iar un ajutor de barman sau osp˘ atar este pl˘ atit cu 1.500 -2.500 de lei.˘ Cristian Balan, administrator teras˘ a Mamaia: „Fa˘ ta de al˘ ti ani când aduceam din Republica, aduceam˘ din Bulgaria, gasim personal local. Avem de unde selecta fa˘ ta de al˘ ti ani unde mai pe vechi aveai pe birou 50 de CV-uri, acum pâna în pandemie ne descurcam cu 5-6, iar au ap˘ arut solicit˘ arile de munc˘ a.˘ „Cornel Lazar, administrator restaurant-taras˘ a Vama Veche:„Dac˘ a în al˘ ti ani înainte de deschidere cautam dispera˘ ti angajati, ajunsesem si la solutii de a aduce angajati din afara tarii, 2020 a venit cu un˘ surplus de forta de munc˘ a. E primul an în ultimii 10 în care primim oferte de munc˘ a de la lucr˘ atori˘ români.” Reference Summary: Patronii de pe litoralul românesc au zilnic cereri de a angaja români reveniti în tara. Salariile pe litoral încep de la 1.500-2.500 de lei, dar pot ajunge˘ si pâna la 2.000 de euro. „E˘ primul an în ultimii 10 în care primim oferte de munca de la lucr˘ atori români”, spune administratorul˘ unei terase. Candidate Summary: Angajatorii spun ca au primit zeci de solicit˘ ari de a angaja românii întor˘ si acasa˘ din cauza pandemiei. Administratorii din Vama Veche spun ca salariile angaja˘ tilor calificati se duc spre 2.000 de euro. La Mamaia, salariile sunt însa altele.˘ Reference Headline: Latura „buna” a pandemiei. În lipsa sezonierilor str˘ aini, patronii de pe litoralul˘ românesc angajeaza români˘ Candidate Headline: Patronii de pe litoralul românesc n-au mai apelat anul acesta la forta de munca˘ strain˘ a˘ Reference Keywords: mamaia, romani, sezonieri straini,vama veche Candidate Keywords: angajati, angajati straini, salarii, vama veche Document: Employers on the Romanian coast did not call on foreign labor this year. Employers say they have received dozens of requests to hire Romanians who have returned home due to the pandemic. The administrators of Vama Veche say that the salaries of qualified employees go up to 2,000 euros, and the next category exceeds 1,000 euros. In Mamaia, however, salaries are different. A bartender earns 3,500-4,000 lei per month, and a helper as a bartender or waiter is paid 1,500-2,500 lei. Cristian Balan, Mamaia terrace administrator: "Compared to other years when we brought from the Republic,˘ we brought from Bulgaria, we find local staff. We have something to choose from compared to other years where, in the old days, you had 50 CVs on your desk. Now, until the pandemic, we managed with 5-6, and the job requests appeared. "Cornel Lazar, restaurant-terrace administrator Vama Veche:˘ "If in other years before the opening we were desperately looking for employees, we had also reached solutions to bring in employees from outside the country, 2020 came with a labor surplus. It’s the first year in the last ten that we receive job offers from Romanian workers." Reference Summary: Employers on the Romanian coast have daily requests to hire Romanians who have returned to the country. Salaries on the coast start from 1,500-2,500 lei, but can reach up to 2,000 euros. "It’s the first year in the last ten that we receive job offers from Romanian workers," says the administrator of a terrace. Candidate Summary: Employers say that they have received dozens of requests to hire Romanians who have returned home due to the pandemic. The administrators of Vama Veche say that the salaries of qualified employees go up to 2,000 euros. In Mamaia, however, the salaries are different. Reference Headline: The "good" side of the pandemic. In the absence of foreign seasonal workers, employers on the Romanian coast hire Romanians Candidate Headline: Employers on the Romanian coast did not call on foreign labor this year Reference Keywords: mamaia, Romanians, seasonal foreigners,vama veche Candidate Keywords: employees, foreign employees, salaries, vama veche

Table 10: The third example of generated summaries, headlines, and keywords, both in Romanian (top) and translated into English (bottom).