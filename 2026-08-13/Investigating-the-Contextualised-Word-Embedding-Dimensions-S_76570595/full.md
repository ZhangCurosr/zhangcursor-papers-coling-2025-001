# Investigating the Contextualised Word Embedding Dimensions Specified for Contextual and Temporal Semantic Changes

Taichi Aida Tokyo Metropolitan University aida-taichi@ed.tmu.ac.jp

Danushka Bollegala University of Liverpool danushka@liverpool.ac.uk

## Abstract

The sense-aware contextualised word embeddings (SCWEs) encode semantic changes of words within the contextualised word embedding (CWE) spaces. Despite the superior performance of SCWEs in contextual/temporal semantic change detection (SCD) benchmarks, it remains unclear as to how the meaning changes are encoded in the embedding space. To study this, we compare pre-trained CWEs and their fine-tuned versions on contextual and temporal semantic change benchmarks under Principal Component Analysis (PCA) and Independent Component Analysis (ICA) transformations. Our experimental results reveal (a) although there exist a smaller number of axes that are specific to semantic changes of words in the pre-trained CWE space, this information gets distributed across all dimensions when fine-tuned, and (b) in contrast to prior work studying the geometry of CWEs, we find that PCA to better represent semantic changes than ICA within the top 10% of axes. These findings encourage the development of more efficient SCD methods with a small number of SCD-aware dimensions.<sup>1</sup>

## 1 Introduction

Meaning of a word is a dynamic phenomenon that is both contextual (i.e. depends on the context in which the word is used) (Pilehvar and Camacho-Collados, 2019) as well as temporal (i.e. the meaning of a word can change over time) (Tahmasebi et al., 2021). A large body of methods have been proposed to represent the meaning of a word in a given context (Devlin et al., 2019; Conneau et al., 2020; Zhou and Bollegala, 2021; Rachinskiy and Arefyev, 2021; Periti et al., 2024), or within a given time period (Hamilton et al., 2016; Rosenfeld and Erk, 2018; Aida et al., 2021; Rosin et al., 2022;

Aida and Bollegala, 2023b; Tang et al., 2023; Fedorova et al., 2024). In particular, SCWEs such as XL-LEXEME (Cassotti et al., 2023) obtained by fine-tuning masked language models (MLMs) such as XLM-RoBERTa (Conneau et al., 2020) on Word-in-Context (WiC) (Pilehvar and Camacho-Collados, 2019) have reported superior performance in SCD benchmarks (Cassotti et al., 2023; Aida and Bollegala, 2023a; Periti and Tahmasebi, 2024; Aida and Bollegala, 2024), implying that semantic changes can be accurately inferred from SCWEs.

Despite the empirical success, to the best of our knowledge, no prior work has investigated whether there are dedicated dimensions in the XL-LEXEME embedding space specified for the semantic changes of the words it represents. In this paper, we study this problem from two complementary directions. First, in §3, we investigate the embedding dimensions specific to the contextual semantic changes of words using WiC benchmarks (Pilehvar and Camacho-Collados, 2019; Raganato et al., 2020; Martelli et al., 2021; Liu et al., 2021) as the evaluation task. Second, in §4, we investigate the embedding dimensions specific to the temporal semantic changes of words on SemEval-2020 Task 1 (Schlechtweg et al., 2020) benchmark. In each setting, we compare pre-trained CWEs and the SCWEs obtained by fine-tuning on WiC using PCA and ICA, which have been used in prior work investigating dimensions in CWEs (Yamagiwa et al., 2023). Our investigations reveal several interesting novel insights that will be useful when developing accurate and efficient lowdimensional SCD methods as follows.

• PCA discovers contextual/temporal semantic change-aware axes within the top 10% of the transformed axes better than ICA.

• In pre-trained embeddings, we identify a small number of axes that are specified for contextual/temporal semantic changes, while such axes are uniformly distributed in the fine-tuned embeddings.

<table><tr><td>Type of Semantic Change</td><td colspan="2">Instances</td><td>Label</td></tr><tr><td rowspan="2">Contextual</td><td>.. . two points on a plane lies ... .. . the plane graph as the X-Y</td><td></td><td rowspan="2">True (Same meanings)</td></tr><tr><td>He lived on a worldly plane.</td><td>.. . the plane graph as the X-Y</td></tr><tr><td rowspan="3">Temporal</td><td>• .. . this is a horizontal plane, and ...</td><td>• .. . as the plane settled down at ...</td><td rowspan="3">True (Semantically Changed)</td></tr><tr><td>• ... because it is parallel with • ...558 combat planes and the ground plane . ..</td><td>4,000 tanks.</td></tr><tr><td></td><td>• .. . this is a horizontal plane, • The President&#x27;s plane landed at ...</td></tr></table>

Table 1: Examples of contextual/temporal semantic change tasks. In contextual semantic change tasks, models predict the meanings of a target word (e.g. plane) in each pair of sentences in the same time period. On the other hand, in temporal semantic change tasks, models predict the meaning of a target word (e.g. plane) from sets of sentences across different time periods.

• Semantic change aware dimensions report comparable or superior performance over using all dimensions in SCD benchmarks.

## 2 Task Description

In this section, we explain the two types of semantic changes of words considered in the paper: (a) contextual semantic changes and (b) temporal semantic changes.

Contextual Semantic Change Detection Task involves predicting whether the meaning of a word in a given pair of sentences are the same (Pilehvar and Camacho-Collados, 2019). For example, an ambiguous word can express different meanings in different contexts, which is considered under contextual semantic changes. Models are required to make a prediction for each pair of sentences.

Temporal Semantic Change Detection Task involves predicting the meanings of a word in given sets of sentences across different time periods (Schlechtweg et al., 2020). A word that was used in a different meaning in the past can be associated with novel meanings later on, which is considered as a temporal semantic change of that word. Models predict whether the meaning of the word has changed over time by comparing the given sets of sentences.

Models For the Contextual Semantic Change Detection Task, contextual word embeddings (Devlin et al., 2019; Conneau et al., 2020) are the primary choice, as they effectively capture word meanings based on sentence context. For the Temporal Semantic Change Detection Task, both static (Kim et al., 2014; Kulkarni et al., 2015; Hamilton et al., 2016; Yao et al., 2018; Aida et al., 2021) and contextual (Rosenfeld and Erk, 2018; Kutuzov and Giulianelli, 2020; Laicher et al., 2021; Aida and Bollegala, 2023b) embeddings can be applied. Notably, sense-aware contextual embeddings trained specifically for contextual semantic change tasks have achieved superior performance, demonstrating their broader applicability (Cassotti et al., 2023; Aida and Bollegala, 2024).

Both types of semantic changes are common and even the same word can undergo both types of semantic changes as shown in Table 1 for the word plane. The contextual semantic change task requires models to be sensitive to the context within just two given sentences, whereas the temporal semantic change task requires models to account for the semantic changes of words across two different time periods.

## 3 Contextual Semantic Changes

We first investigate the existence of axes specific to contextual semantic changes. Recall that XL-LEXEME is fine-tuned from XLM-RoBERTa on WiC datasets. Therefore, the emergence of any semantic change-aware axes due to fine-tuning can be investigated using contextual semantic change benchmarks. We use the test split of the English WiC (Pilehvar and Camacho-Collados, 2019), XL-WiC (Raganato et al., 2020), MCL-WiC (Martelli et al., 2021), and AM<sup>2</sup>iCo (Liu et al., 2021) datasets for evaluations.<sup>2</sup> Data statistics are in Appendix A.

![](images/d6709ff0e8eeefd504ce8016a929de1c2f0702fa6c27beff20f8c752c0993f5f.jpg)  
(a) Pre-trained CWE, Raw

![](images/f04cad43fb0be739898e8e1e666c2e0b1071e3c3cf11e9862fce11b22e485131.jpg)  
(b) Pre-trained CWE, PCA

![](images/006b21b991bb7c470b7f9c20beb158a68a93e97f4fada243a6ae44ac1ee13df3.jpg)  
(c) Pre-trained CWE, ICA

![](images/38166553229b73cc1c85ff5e9e449e9bb47b2ec41b73f16d672adb827d574b72.jpg)  
(d) Fine-tuned SCWE, Raw

![](images/cbec31fca10c20c1cda15a482a91184721abe88fa0db674474d63394e90e72f7.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/1c19e8fa3c5519e621a50c1c43ba2f42e7fad4bff2285da8d5099370ed63ee8b.jpg)  
(f) Fine-tuned SCWE, ICA  
Figure 1: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in the English WiC dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels. While the Raw dimensions display the information from the 0th to the 49th dimensions in the original order, the same observations are found in all dimensions.

RQ1: When do the contextual SCD-aware axes emerge? To investigate whether contextual semantic change-aware axes were already present in the pre-trained CWEs, or do they emerge during the fine-tuning step, for each sentence-pair in WiC datasets, we compute the difference between the two target word embeddings obtained from the pre-trained XLM-RoBERTa (CWEs) and the finetuned XL-LEXEME (SCWEs). To obtain the sets of target word embeddings, we follow Cassotti et al. (2023) by using a Sentence-BERT (Reimers and Gurevych, 2019) architecture. We conduct this analysis for the non-transformed original axes (indicated as Raw here onwards), as well as for the PCA/ICA-transformed axes in order to investigate whether such transformations can discover the axes specified for contextual semantic changes as proposed by Yamagiwa et al. (2023).<sup>3</sup> In this paper, PCA/ICA-transformed axes are sorted by the experimental variance ratio/skewness, and this process is consistently applied where PCA or ICA is used. If a particular axis is sensitive to contextual semantic changes, it will take similar values in the two target word embeddings, thus having a near-zero value in their subtraction.

To address RQ1, we visualised the difference vectors for sentence pairs where the target word takes the same meaning in the two sentences (True) vs. different meanings (False). This visualisation was performed by following steps: (a) we prepared Raw or PCA/ICA-transformed axes; (b) for each WiC instance, which contains two sentences and a label, we calculated the difference between pair of sentences; (c) we normalised each axis (min=0 and max=1) for visualisation purposes.

As shown in Figure 1, we see that the axes encoding contextual semantic changes are not obvious in the original CWEs after pre-training (Figure 1a), but materialise during the fine-tuning process (Figure 1d). Similar trends are observed with PCA-transformations (Figures 1e and 1b), whereas ICA shows contrasting results (Figures 1f and 1c). In contrast to prior recommendations for using ICA for analysing CWE spaces (Yamagiwa et al., 2023), we find ICA to be less sensitive to contextual semantic changes of words. Interestingly, similar results have been shown in other languages/datasets(Appendix B). 4

![](images/7bf3733eb21da513410841ab1380a57f301799a9332265a471bcdc906310c857.jpg)

(a) Pre-trained CWE (XLM-RoBERTa)  
![](images/0b421384bacacf56e980036999ac169ae43ce552a720caaa46e6f775369aee5f.jpg)  
(b) Fine-tuned SCWE (XL-LEXEME)  
Figure 2: The ROC curve on contextual semantic change task, the English WiC dataset. Raw indicates the performance of using full dimensions. PCA/ICA uses top-5/10/20/50/100% of axes.

RQ2: Can top-k PCA/ICA-transformed axes capture contextual semantic changes? Yamagiwa et al. (2023) discovered that ICAtransformed axes represent specific concepts and their linear combinations could represent more complex concepts (e.g. $c a r s + i t a l i a n = f e r r a r i )$ Based on this finding, we investigate whether a combination of top-k axes can collectively represent contextual semantic changes of words. Specifically, we select the top-k% of the axes to represent a target word embedding. We then compute the Euclidean distance between CWEs of the target word in each sentence for every test sentencepair in the WiC datasets. We predict the target word to have the same meaning in the two sentences, if the Euclidean distance is below a threshold value. We vary this threshold and report Area Under the Curve (AUC) of Receiver Operating Characteristic (ROC) curves, where higher AUC values are desirable. In Figure 2, we show results for top $k \in \{ 5 , 1 0 , 2 0 , 5 0 , 1 0 0 \}$ of the PCA/ICAtransformed axes and compare against the baseline that uses all of the Raw dimensions.

For the pre-trained CWEs (Figure 2a), we see that Raw reports slightly better AUC than PCA, but when fine-tuned (Figure 2b) PCA matches Raw even by using less than 10% of the axes. On the other hand, ICA reports lower AUC values than both Raw and PCA in both models. These results indicate that PCA is better suited for discovering axes specified for contextual semantic changes than ICA. We suspect that although ICA is able to retrieve concepts such as topics (Yamagiwa et al., 2023), it is less fluent when discovering task-specific axes that require the consideration of different types of information. In conclusion, (1) contextual semantic change-aware axes emerge during fine-tuning, and (2) they are discovered by PCA even within 10% of the principal components. Notably, in other languages/datasets, similar trends have been observed (Appendix B). These results suggest that contextual semantic changeaware dimensions can be observed within 10% of the PCA-transformed axes across different languages.

## 4 Temporal Semantic Changes

In contrast to contextual SCD, temporal SCD considers the problem of predicting whether a target word w represents different meanings in two text corpora $C _ { 1 }$ and $C _ { 2 } .$ , sampled at different points in time. For evaluations, we use the SemEval-2020 Task 1 dataset<sup>5</sup> (Schlechtweg et al., 2020), which contains a manually rated set of target words for their temporal semantic changes in English, German, Swedish, and Latin.<sup>6</sup>

RQ3: Can top-k PCA/ICA-transformed axes capture temporal semantic changes? Similar to Figure 2, we investigate whether PCA/ICA can discover axes specified for temporal semantic changes by considering the top-k% of axes for $k \in \{ 5 , 1 0 , 2 0 , 5 0 , 1 0 0 \}$ . We calculate the semantic change score of w as the average pairwise Euclidean distance over the two sets of sentences containing the target word w in $C _ { 1 }$ and $C _ { 2 }$ as conducted in previous work (Kutuzov and Giulianelli, 2020; Laicher et al., 2021; Cassotti et al., 2023). Finally, w is predicted to have its meaning changed between $C _ { 1 }$ and $C _ { 2 }$ , if its semantic change score exceeds a pre-defined threshold. We vary this threshold and plot ROC in Figure 3.

![](images/b46f9fb8219d8cefd8be16546797bc1afd28ea92111d14586209e715b21b639a.jpg)

(a) Pre-trained CWE (XLM-RoBERTa)  
![](images/8fcb63b0ad9a005d5a4c2265efdb374e7d8116803b31017a0e5d900e88902720.jpg)  
(b) Fine-tuned SCWE (XL-LEXEME)  
Figure 3: The ROC curve on temporal semantic change task, SemEval-2020 Task 1 (English). Raw indicates the performance of using full dimensions. PCA/ICA uses top-5/10/20/50/100% of axes.

In pre-trained CWEs, we can see that the use of the top 5% to 20% axes transformed by PCA is more effective in temporal semantic change detection than when all of the Raw dimensions are used (Figure 3a). On the other hand, in fine-tuned SCWEs, Figure 3b indicates that PCA-transformed axes achieve the same AUC scores as Raw, similar to the contextual semantic change (Figure 2b). Similar to the observation in contextual semantic change, ICA returns the lowest performance.

To further investigate whether the top PCA/ICA axes can explain the degree of temporal semantic change, we measure the Spearman correlation between the semantic change scores and human ratings available in the SemEval-2020 Task 1 following the standard evaluation protocol for this task (Rosin et al., 2022; Rosin and Radinsky, 2022; Aida and Bollegala, 2023b; Cassotti et al., 2023; Periti and Tahmasebi, 2024; Aida and Bollegala, 2024). As shown in Figure 4 for the pre-trained CWEs (Figure 4a), using only 10% of the axes, PCA outperforms Raw that uses all axes. Moreover, for the fine-tuned SCWEs (Figure 4b), using only 10% of the axes PCA achieves the same performance as Raw. However, ICA consistently underperforms in both pre-trained and fine-tuned settings. Importantly, we see similar trends in other languages (Appendix B). These results suggest that temporal semantic change-aware dimensions can also be observed within 10% of PCAtransformed axes across different languages.

![](images/68800c3f309e31afb017a6b7c8ae3f801d7bee13f353b39797202a9a8c195d2c.jpg)

(a) Pre-trained CWE (XLM-RoBERTa)  
![](images/3ddc2e5658905529d577c676080208a40b824c5e704008e0ace565851441f787.jpg)  
(b) Fine-tuned SCWE (XL-LEXEME)  
Figure 4: Spearman’s rank correlation on temporal semantic change task, SemEval-2020 Task 1 (English). Raw indicates the performance of using full dimensions. PCA/ICA cumulatively uses sorted axes.

## 5 Conclusion

We found that there exists a smaller number of axes that encode contextual and temporal semantic changes of words in MLMs, which are accurately discovered by PCA. These findings have several important practical implications. First, it shows that MLMs can be compressed into efficient and accurate lower-dimensional embeddings when used for SCD tasks. Second, it suggests the possibility of efficiently updating a pre-trained MLM to capture novel semantic associations of words since the MLM was first trained, by updating only a smaller number of dimensions.

## Limitations

In this paper, we limited experiments to XLM-RoBERTa based MLM models. These models are all fine-tuned on WiC datasets and have reported state-of-the-art (SoTA) performance in SCD benchmarks. We consider it would be important to further validate the findings reported in this paper using other embedding models and across multiple downstream applications.

## Ethical Considerations

In this paper, we focus on investigating the existence of dedicated dimensions capturing contextual/temporal semantic changes of words. For the best of our knowledge, no ethical issues have been reported for the WiC and SCD datasets we used in our experiments. On the other hand, we also used publicly available pre-trained/fine-tuned MLMs, some of which are known to encode and potentially amplify unfair social biases (Basta et al., 2019). Whether such social biases are influenced by the dimension selection methods we consider in the paper must be carefully evaluated before deploying any MLMs in downstream applications.

## Acknowledgements

Taichi Aida would like to acknowledge the support by JST, the establishment of university fellowships towards the creation of science technology innovation, Grant Number JPMJFS2139.

## References

Taichi Aida and Danushka Bollegala. 2023a. Swap and predict – predicting the semantic changes in words across corpora by context swapping. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7753–7772, Singapore. Association for Computational Linguistics.

Taichi Aida and Danushka Bollegala. 2023b. Unsupervised semantic variation prediction using the distribution of sibling embeddings. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 6868–6882, Toronto, Canada. Association for Computational Linguistics.

Taichi Aida and Danushka Bollegala. 2024. A semantic distance metric learning approach for lexical semantic change detection. In Findings of the Association for Computational Linguistics: ACL 2024, pages 7570–7584, Bangkok, Thailand. Association for Computational Linguistics.

Taichi Aida, Mamoru Komachi, Toshinobu Ogiso, Hiroya Takamura, and Daichi Mochihashi. 2021. A

comprehensive analysis of PMI-based models for measuring semantic differences. In Proceedings of the 35th Pacific Asia Conference on Language, Information and Computation, pages 21–31, Shanghai, China. Association for Computational Lingustics.

Christine Basta, Marta R. Costa-jussà, and Noe Casas. 2019. Evaluating the underlying gender bias in contextualized word embeddings. In Proceedings ofthe First Workshop on Gender Bias in Natural Language Processing, pages 33–39, Florence, Italy. Association for Computational Linguistics.

Pierluigi Cassotti, Lucia Siciliani, Marco DeGemmis, Giovanni Semeraro, and Pierpaolo Basile. 2023. XL-LEXEME: WiC pretrained model for cross-lingual LEXical sEMantic changE. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 1577–1585, Toronto, Canada. Association for Computational Linguistics.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Mariia Fedorova, Andrey Kutuzov, and Yves Scherrer. 2024. Definition generation for lexical semantic change detection. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 5712– 5724, Bangkok, Thailand. Association for Computational Linguistics.

William L. Hamilton, Jure Leskovec, and Dan Jurafsky. 2016. Diachronic word embeddings reveal statistical laws of semantic change. In Proceedings ofthe 54th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1489–1501, Berlin, Germany. Association for Computational Linguistics.

Yoon Kim, Yi-I Chiu, Kentaro Hanaki, Darshan Hegde, and Slav Petrov. 2014. Temporal analysis of language through neural language models. In Proceedings of the ACL 2014 Workshop on Language Technologies and Computational Social Science, pages 61–65, Baltimore, MD, USA. Association for Computational Linguistics.

Vivek Kulkarni, Rami Al-Rfou, Bryan Perozzi, and Steven Skiena. 2015. Statistically significant detection of linguistic change. In WWW 2015, pages 625– 635.

Andrey Kutuzov and Mario Giulianelli. 2020. UiO-UvA at SemEval-2020 task 1: Contextualised embeddings for lexical semantic change detection. In Proceedings ofthe Fourteenth Workshop on Semantic Evaluation, pages 126–134, Barcelona (online). International Committee for Computational Linguistics.

Severin Laicher, Sinan Kurtyigit, Dominik Schlechtweg, Jonas Kuhn, and Sabine Schulte im Walde. 2021. Explaining and improving BERT performance on lexical semantic change detection. In Proceedings of the 16th Conference ofthe European Chapter ofthe Association for Computational Linguistics: Student Research Workshop, pages 192–202, Online. Association for Computational Linguistics.

Qianchu Liu, Edoardo Maria Ponti, Diana McCarthy, Ivan Vulic, and Anna Korhonen. 2021.´ AM2iCo: Evaluating word meaning in context across lowresource languages with adversarial examples. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7151–7162, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Federico Martelli, Najla Kalach, Gabriele Tola, and Roberto Navigli. 2021. SemEval-2021 task 2: Multilingual and cross-lingual word-in-context disambiguation (MCL-WiC). In Proceedings ofthe 15th International Workshop on Semantic Evaluation (SemEval-2021), pages 24–36, Online. Association for Computational Linguistics.

Francesco Periti, David Alfter, and Nina Tahmasebi. 2024. Automatically generated definitions and their utility for modeling word meaning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 14008–14026, Miami, Florida, USA. Association for Computational Linguistics.

Francesco Periti and Nina Tahmasebi. 2024. A systematic comparison of contextualized word embeddings for lexical semantic change. In Proceedings of the 2024 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4262–4282, Mexico City, Mexico. Association for Computational Linguistics.

Mohammad Taher Pilehvar and Jose Camacho-Collados. 2019. WiC: the word-in-context dataset for evaluating context-sensitive meaning representations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1267–1273, Minneapolis, Minnesota. Association for Computational Linguistics.

Maxim Rachinskiy and Nikolay Arefyev. 2021. Gloss-Reader at SemEval-2021 task 2: Reading definitions improves contextualized word embeddings. In Proceedings ofthe 15th International Workshop on Semantic Evaluation (SemEval-2021), pages 756–762, Online. Association for Computational Linguistics.

Alessandro Raganato, Tommaso Pasini, Jose Camacho-Collados, and Mohammad Taher Pilehvar. 2020. XL-WiC: A multilingual benchmark for evaluating semantic contextualization. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7193–7206, Online. Association for Computational Linguistics.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Alex Rosenfeld and Katrin Erk. 2018. Deep neural models of semantic shift. In Proceedings ofthe 2018 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 474–484, New Orleans, Louisiana. Association for Computational Linguistics.

Guy D. Rosin, Ido Guy, and Kira Radinsky. 2022. Time masking for temporal language models. In Proceedings ofthe Fifteenth ACM International Conference on Web Search and Data Mining, WSDM ’22, pages 833–841, New York, NY, USA. Association for Computing Machinery.

Guy D. Rosin and Kira Radinsky. 2022. Temporal attention for language models. In Findings ofthe Association for Computational Linguistics: NAACL 2022, pages 1498–1508, Seattle, United States. Association for Computational Linguistics.

Dominik Schlechtweg, Barbara McGillivray, Simon Hengchen, Haim Dubossarsky, and Nina Tahmasebi. 2020. SemEval-2020 task 1: Unsupervised lexical semantic change detection. In Proceedings of the Fourteenth Workshop on Semantic Evaluation, pages 1–23, Barcelona (online). International Committee for Computational Linguistics.

Nina Tahmasebi, Lars Borina, and Adam Jatowtb. 2021. Survey of computational approaches to lexical semantic change detection. Computational approaches to semantic change, 6:1.

Xiaohang Tang, Yi Zhou, and Danushka Bollegala. 2023. Learning dynamic contextualised word embeddings via template-based temporal adaptation. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9352–9369, Stroudsburg, PA, USA. Association for Computational Linguistics.

<table><tr><td>Dataset</td><td>Language</td><td>#Train</td><td>#Dev</td><td>#Test</td></tr><tr><td>Monolingual</td><td></td><td></td><td></td><td></td></tr><tr><td>WiC</td><td>English German</td><td>5.4k 48k</td><td>6.4k</td><td>1.4k 1.1k</td></tr><tr><td>XL-WiC</td><td>French Italian Arabic</td><td>39k 1.1k</td><td>8.9k 8.6k 0.2k</td><td>22k 0.6k</td></tr><tr><td>MCL-WiC</td><td>English French Russian Chinese</td><td>4.0k 一</td><td>0.5k 0.5k 0.5k 0.5k 0.5k</td><td>0.5k 0.5k 0.5k 0.5k 0.5k</td></tr><tr><td>Cross-lingual</td><td></td><td></td><td></td><td>1.0k</td></tr><tr><td>AM²iCo</td><td>German Russian Japanese Chinese Arabic Korean Finnish Turkish Indonesian Basque</td><td>50k 28k 16k 13k 9.6k 7.0k 6.3k 3.9k 1.6k 1.0k</td><td>0.5k 0.5k 0.5k 0.5k 0.5k 0.5k 0.5k 0.5k 0.5k 0.5k</td><td>1.0k 1.0k 1.0k 1.0k 1.0k 1.0k 1.0k 1.0k 1.0k</td></tr></table>

Table 2: Statistics of the contextual SCD benchmarks used in the fine-tuning for XL-LEXEME. #Train, #Dev, and #Test show the number of instances. AM<sup>2</sup>iCo is a cross-lingual contextual SCD benchmark, where the second language in each pair is English.

Hiroaki Yamagiwa, Momose Oyama, and Hidetoshi Shimodaira. 2023. Discovering universal geometry in embeddings with ICA. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 4647–4675, Singapore. Association for Computational Linguistics.

Zijun Yao, Yifan Sun, Weicong Ding, Nikhil Rao, and Hui Xiong. 2018. Dynamic word embeddings for evolving semantic discovery. In WSDM 2018, page 673–681.

Yi Zhou and Danushka Bollegala. 2021. Learning sensespecific static embeddings using contextualised word embeddings as a proxy. In Proceedings ofthe 35th Pacific Asia Conference on Language, Information and Computation, pages 493–502, Shanghai, China. Association for Computational Lingustics.

## A Data Statistics

Full statistics of contextual and temporal SCD benchmarks are shown in Table 2 and Table 3.<sup>7</sup>

## B Full Results

In this section, we present the full results of contextual and temporal SCD tasks. For the contextual

<table><tr><td>Language</td><td>Time Period</td><td>#Targets</td><td>#Tokens</td></tr><tr><td>English</td><td>1810-1860 1960-2010</td><td>37</td><td>6.5M 6.7M</td></tr><tr><td>German</td><td>1800-1899 1946-1990</td><td>48</td><td>70.2M 72.3M</td></tr><tr><td>Swedish</td><td>1790-1830 1895-1903</td><td>31</td><td>71.0M 110.0M</td></tr><tr><td>Latin</td><td>B.C. 200–0 0-2000</td><td>40</td><td>1.7M 9.4M</td></tr></table>

Table 3: Statistics of the temporal SCD benchmark, SemEval-2020 Task 1. #Targets and #Tokens show the number of target words and tokens, respectively.

SCD, visualisations of instances in all datasets are as follows: XLWiC (Figure 5, Figure 6, and Figure 7), MCLWiC (Figures 8, 9, 10, 11, and 12), and AM<sup>2</sup>iCo (Figures 13, 14, 15, 16, 17, 18, 19, 20, 21, and 22). Similar to § 3, the contextual semantic change-aware axes emerged after the finetuning process. Moreover, full results related to the prediction task are as follows: XLWiC (Figure 23), MCLWiC (Figure 24 and Figure 25), AM<sup>2</sup>iCo (Figure 26, Figure 27, and Figure 28). As shown in §3, 10% PCA-transformed axes are able to obtain contextual semantic change-aware dimensions.

On the other hand, for the temporal SCD, results for other languages (German, Swedish, and Latin) are shown in Figure 29 and Figure 30. Similar to §4, temporal semantic change-aware dimensions are observed within 10% PCA-transformed axes. However, there are some difficulties in obtaining these dimensions by PCA-transformed axes with insufficient pretraining data (Swedish) (Conneau et al., 2020) or lack of supervision for fine-tuning (Latin) shown in Table 2. In those cases, the use of ICA-transformed axes proved to be effective. More detailed analysis and understanding of those axes for interpretability will be addressed in future work.

<sup>7</sup>WiC, XL-WiC, and MCL-WiC are licensed under the Creative Commons Attribution-NonCommercial 4.0 License, while AM<sup>2</sup>iCo and SemEval-2020 Task 1 are licensed under the Creative Commons Attribution 4.0 International License.

![](images/a0a69bd88fcbe825cc294554e3f42599d7cbc03464a1eca546a3cf51c1f7dd22.jpg)  
(a) Pre-trained CWE, Raw

![](images/8099d07c9968edf54cd659d249c792bbb044df371bdec553e84e99032429e1e9.jpg)  
(b) Pre-trained CWE, PCA

![](images/3c237a5074483057556784eb5015936a1e1693eb2d9f2499a4bccca7d0b5a837.jpg)  
(c) Pre-trained CWE, ICA

![](images/2560d01cca2dbea62773a796aa8aa8f22d9ffbc21d115a72db799080d0618d1f.jpg)  
(d) Fine-tuned SCWE, Raw

![](images/502bf3568c0503c3a4e8e46358ec71f8bbd648e0c72a5cc533df35d91a69f947.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/eb571e5955249fe213e3c3c91af5d82545792e19d456522fb662e5ace415c042.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 5: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in XLWiC (German) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/4ae59162c138665c9940fe7fea758dac81dc2543afc81ac5fc726fe5cd4d8435.jpg)  
(a) Pre-trained CWE, Raw

![](images/0913fc89168ba0b990c470bb930b04b265393a37ed08a010f0f777243a72a1d1.jpg)

![](images/9ccf2c5c637d139fa7b9ff269f830efe184ed9b31af726698f537cc659e499d1.jpg)

![](images/6035ba398d63adc4c837be0e049dd8135c7166ec3b1a3676985e40e9425ac1c8.jpg)  
(d) Fine-tuned SCWE, Raw

(b) Pre-trained CWE, PCA  
![](images/e11a2d4447096d76ec7492ef9034624e1c17260d2fb0c718d2a9add9ab025c3a.jpg)  
(e) Fine-tuned SCWE, PCA

(c) Pre-trained CWE, ICA  
![](images/69545133db949d682f158056616ed5093348510033c0b6fa45687d8ad71722a0.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 6: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in XLWiC (French) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/f1e5bcf9902e9677828709d1425620b8cfcdab287c49c57c6c3dd996ae04f4a1.jpg)  
(a) Pre-trained CWE, Raw

![](images/82b4e52f950691661c7daf45a40c9f76ae5a376b198070e1c65d7e6d8103cf2f.jpg)  
(b) Pre-trained CWE, PCA

![](images/2cc7863f3bb8d9112e2c05c9eacf408a00a2640b62317582a00488373181e9a3.jpg)  
(c) Pre-trained CWE, ICA

![](images/ac11d3b73ab391b09a50c44807991db6ab3f50c046ceab3da8b60a49866199c5.jpg)  
(d) Fine-tuned SCWE, Raw

![](images/1341731cdd3f0951d8cd58e2b600cac105d30678abd220c6f05f67c222684c65.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/b3abd6e2e64019dc5b46979871ff707ec852e4ab00125821e89660ea1d612f33.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 7: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in XLWiC (Italian) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/a0e16d96923f1e9e92408779103703eba549fc72f113d570d8603bfb554c961f.jpg)  
(a) Pre-trained CWE, Raw

![](images/e1c0de6ee23b20f7394ac28cebeb3836f407ded379593e237d93ee3e2dc65fde.jpg)

![](images/277c33c56ae8e95897b3d5c9d227c1a6e75ceb1a4312063de5c70e7f2bb1c5df.jpg)

![](images/f42394cafd9181b74df923d58d7e128a65f2aee8bac61553f56c166c6096775a.jpg)  
(d) Fine-tuned SCWE, Raw

(b) Pre-trained CWE, PCA  
![](images/5da79ef99a9ae44d105aaef0824d8ac21bd9202ab15d41da86297baef7d566bd.jpg)  
(e) Fine-tuned SCWE, PCA

(c) Pre-trained CWE, ICA  
![](images/c981377de7cfd7289e2ed15cab2a58df02b1041644eace0a0c319ca7f7d517db.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 8: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in MCLWiC (Arabic) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/fdd5d20f645e188a96a8482ece9e7d72d441e314d70db28832f8ec639e7fdb31.jpg)  
(a) Pre-trained CWE, Raw

![](images/9e1ff54b7db8679a6f9a262c667c205223b893e7eea97a299cb85ec97307b758.jpg)  
(b) Pre-trained CWE, PCA

![](images/d83b4a28d82ce1c4bd38b964ceae1cb9b7fe7e0d6826f575c6e0810c349961f7.jpg)  
(c) Pre-trained CWE, ICA

![](images/fa3c054ff596c778cd8ab850f9fb3f7f3223679327b06f81dda371a801836f5d.jpg)  
(d) Fine-tuned SCWE, Raw

![](images/8374c0b5c565c298519c75d1993fd67ac9b351a735a7c9b6d708093dfe9c512f.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/eba40bc6bfba776e87fcf1e44abda409fed8b6cfbd6c30d8ddefcf27e5ebc90c.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 9: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in MCLWiC (English) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/f0d30c99d6f42e53d11382d999e110d6a0c975f1429b942e22218a1fe896c5a8.jpg)  
(a) Pre-trained CWE, Raw

![](images/6b67f28d3b15d0a284fd87f2b022c4e39eb4f2a6f9fb2050f4432d0d4b512a86.jpg)

![](images/440890664dd34fa1777614ae779bf0cb2e0354aada32c8384d1c3a0ade665996.jpg)  
(b) Pre-trained CWE, PCA

![](images/beecc5691ee27682da0bf11864545a8a7c6684a70f00dc93913d98f7c44cb29f.jpg)  
(d) Fine-tuned SCWE, Raw

(c) Pre-trained CWE, ICA  
![](images/eb7c8d1c3f0000edcc0ed5af4f2c34b05c50038ab222393ba916355479ab9ec3.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/506fef328c03339b94e849fdaa044dbccda690e6eef8a55193d34b86f93ca0d9.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 10: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in MCLWiC (French) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/f9f455798429c82f4409b85601f80d1dfe9133f2eb0596b4925f0b22a679ce9f.jpg)  
(a) Pre-trained CWE, Raw

![](images/533f49bfed8f10c418cd97c05de7a75c852857fabb0ea7a3b0424d41c3543cf6.jpg)  
(b) Pre-trained CWE, PCA

![](images/1ad440cd0ca1c9c2c776f4c0dce36320c1bc45d916e569e9e84b992dd2752d04.jpg)  
(c) Pre-trained CWE, ICA

![](images/3c87a56a43b49d84bd70704bc74b6d1d664f6e424903d6ce29f7fa71eea73278.jpg)  
(d) Fine-tuned SCWE, Raw

![](images/5f1db9ffdb9ab2dad3d320e2c7b9d655ddfaf81adea78fd4b027e3eff5c72d27.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/d07f38eaae25064bf05f846db20c1f9cc298d41ed14765f7b781f19375900531.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 11: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in MCLWiC (Russian) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/2ac8fd178e3493dcec5f736232c36ff387eb9e694c94fc591d4fe950e2a11151.jpg)  
(a) Pre-trained CWE, Raw

![](images/a46b414a65de06915c16f8ec0b48505cb33529c684e761db4715f1d303d1419b.jpg)

![](images/1d3f5ef64cbbe85c36f69f118f960c90429774daa4e9755bba0fd19fed3ecfd4.jpg)  
(b) Pre-trained CWE, PCA

![](images/d0d464e634d712a5e5cdfee6b8b5451b6b01159842912436e9a1ddd5beef8368.jpg)  
(d) Fine-tuned SCWE, Raw

(c) Pre-trained CWE, ICA  
![](images/89b817ae4a411626e327927b2d9fcb25ae0e2c1230d5691ef2482285776cab3d.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/4be62f1d7487db477211d024c8dcde70de8ac274d9b83607e1788e5c79f7f19e.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 12: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in MCLWiC (Chinese) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/86f24292b32dac0638d9c70c6d0a87b2e1c090f767c5ba496e1c11f3623b49ab.jpg)  
(a) Pre-trained CWE, Raw

![](images/051c4f93c176098dffbff8a30715c5eb4531bb0b851fc189d63fa4302e0accfe.jpg)  
(b) Pre-trained CWE, PCA

![](images/b3c3e0d616e6def1ed0487b10d2843818ae0b6f5bfdfa58c989c80428e898600.jpg)  
(c) Pre-trained CWE, ICA

![](images/a0020478d5954f8d9491f6d5b850b71b4874001684529d66ef5ba95d5c129e1e.jpg)  
(d) Fine-tuned SCWE, Raw

![](images/29dc570d6a08e8e7527f0f715943970604caccdd2458f0b3c1122b71996bb309.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/9d5ff149e5a3814576247a1a725ccabdda832a59e3163d33d59534dcecc93fc6.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 13: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in AM<sup>2</sup>iCo (German) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/db08bbbe2f82acff3a467ff22f9899444806371cd103a0c0838cf21401aeae56.jpg)  
(a) Pre-trained CWE, Raw

![](images/7c395b1a5e4c287f51c1f3d8bd2f672ca2efebc5733bff234c1b082d4052098f.jpg)

![](images/dc76028223abce73878790f1b1f2607d6cf5092927337a3c383274c78ab0c907.jpg)  
(b) Pre-trained CWE, PCA

![](images/b282fc15138f659c7a4df2faa9fa7d015bd4a942160a8f6787666de637d0ea76.jpg)  
(d) Fine-tuned SCWE, Raw

![](images/ea31c35c6d7700f9abaf0336f142cacca50851f3c96362bf5cecc6b623c370b1.jpg)  
(e) Fine-tuned SCWE, PCA

(c) Pre-trained CWE, ICA  
![](images/6ee0ae84822fac1856b8a55b3ba5b9f46bb350508a4eb01e445c0fea4318be4f.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 14: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in AM<sup>2</sup>iCo (Russian) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/893ed3775864f8615672b69435bd2da91c74198040e10eae0c1a33e43457b712.jpg)  
(a) Pre-trained CWE, Raw

![](images/5a747a03fd81d0fc019dca8d84caaad83b25497e9a353b5509b3a3a79367a35e.jpg)  
(b) Pre-trained CWE, PCA

![](images/19d8ca988a1e27b71ac607c51441acfb70db20471941bd8b39f8211d7932c910.jpg)  
(c) Pre-trained CWE, ICA

![](images/b936ae5f3edf692f87dfc218774cd650e2e91058ddd198d013cba87e73b8fa61.jpg)  
(d) Fine-tuned SCWE, Raw

![](images/61da754bf498dc1a320715725769b217efaf3cede093093f264d5dccc990b003.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/0b308b1a40c62c2c197cdce28afc6756e94897b46fddba3c286ccf05a888e750.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 15: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in AM<sup>2</sup>iCo (Japanese) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/a7cda18f2f42a91fefda355b0a3a371908d9509dfdf519a96410a29db37e08b7.jpg)  
(a) Pre-trained CWE, Raw

![](images/7372d47f8ac4bbdc9f1b322735f97b4aa16b3688fa4db872b257fd0f9b54dedc.jpg)

![](images/0c4b1bd6ed0a83b1b2cf6f4473ae88d81f6860e1a3f68f1d4b2cabfce76b4c4a.jpg)  
(b) Pre-trained CWE, PCA

![](images/7ad05f807449302ab6c7e7c45eff1d2be097daaa5e88e750335c640e61068eac.jpg)  
(d) Fine-tuned SCWE, Raw

(c) Pre-trained CWE, ICA  
![](images/a43de5608afd63ec01d829013ed6ce4027998ff9f5e7cab3ad012677ff068c62.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/41ef807fee2ba55212dcb37a06b79d4ad905fc7c69e8eb528ff477c550f7b42a.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 16: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in AM<sup>2</sup>iCo (Chinese) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/1873be4e027ef917883506e7a12b68a85e6981832fe28c09261a8ebc11e3b582.jpg)  
(a) Pre-trained CWE, Raw

![](images/7579861a4bf35024b421a5601d2a8747965bda74a9c5c713e0129f83694f5b5e.jpg)  
(b) Pre-trained CWE, PCA

![](images/97399963b79a71db5f308db4be97493180cc60f431af78fae31e427208475d20.jpg)  
(c) Pre-trained CWE, ICA

![](images/e7c47b5a059caec5ab5517dd6a8a5bbc6e2baa21166ee446e92f574e0f618fa0.jpg)  
(d) Fine-tuned SCWE, Raw

![](images/a7a44f0660d69b008f8ffd9a8e10f02dec94e87861c189894ad9f8d4df144639.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/362bc23fa90b2ff9e141983555305344258e84f5c6f6cfeb976823380411e429.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 17: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in AM<sup>2</sup>iCo (Arabic) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/94ebe0d01490a4b4d482f72130d42c70df7c02b2009af19610452307dd25d5c4.jpg)  
(a) Pre-trained CWE, Raw

![](images/2894e9ca104bdc6691ccd19cc436deb8c50c913f16a788def61ba8d67669a32a.jpg)

![](images/ccb9552b31e9f5f2808fd187b141ceb164efa9fd7e85c719a24ac755dfe8e78e.jpg)

![](images/e00527ddf2c9d8480d950b4a8e38b14f3acefd29f24215f04494036bb40e1a43.jpg)  
(d) Fine-tuned SCWE, Raw

(b) Pre-trained CWE, PCA  
![](images/da44e5b12c136db60af6beb7256e133079dee72c964bce048b046ddcc936d246.jpg)  
(e) Fine-tuned SCWE, PCA

(c) Pre-trained CWE, ICA  
![](images/d66be3b6d9429a0598b30facf9ada3f19944d3e8b11a4198a77da682aa0b500e.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 18: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in AM<sup>2</sup>iCo (Korean) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/60329f0808edf3060d9dbf935c7255c348dbd1262c33df6e4870cf45b0cd36b4.jpg)  
(a) Pre-trained CWE, Raw

![](images/c934be05f802aa2f3d7ef983c4a7bb89b1aacbbcc4fb5f0148a81e531279a3c4.jpg)  
(b) Pre-trained CWE, PCA

![](images/35f6b86d48a8a8f742d8c2d57d4a3a0c6a8776007d3403116a8857836043f1ad.jpg)  
(c) Pre-trained CWE, ICA

![](images/76024dd5d985937f123856eb909ab2736c7a499aa67d52a90d7bfc050ae7215a.jpg)  
(d) Fine-tuned SCWE, Raw

![](images/d536045c48c87f7c64cdb4934faf03e1cec76780e29f00a49f2099b1b7d94262.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/51fc58fb4530e1aeacfd5ad5415d42811b44ae91a2408a1df5b707133eb48dca.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 19: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in AM<sup>2</sup>iCo (Finnish) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/41e4be14c00c4d244e7a68dac6c095ec9e719e0d6fc23ddaf915852af27fc5d4.jpg)  
(a) Pre-trained CWE, Raw

![](images/6ac03487cba06014b0b4dbd21e1792f1fbe705620fa15a73869fd77a38b5a37c.jpg)

![](images/f87fd0cf8b4b5ff2b01a1a14babc6f19e92c1091955574253a808f1ff0603897.jpg)

![](images/d82f7edfcc5b6b7b63776c4b696cd54be8f53fe6076262c5de3626ec08209686.jpg)  
(d) Fine-tuned SCWE, Raw

(b) Pre-trained CWE, PCA  
![](images/15bfcc6e9815958f4fed8f7b621c5ca1f8b6719571877d2c62a56c442bab112d.jpg)  
(e) Fine-tuned SCWE, PCA

(c) Pre-trained CWE, ICA  
![](images/5f203adca4af22ab1324b414150d31e0c36a51b8a949fae88ac80c933e6502e4.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 20: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in AM<sup>2</sup>iCo (Turkish) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/918560371f7c4900c9c52778f5bd98b7500fc157fd264dcbad24e554e3d638c6.jpg)  
(a) Pre-trained CWE, Raw

![](images/2b1cb1237c07440317b3df53ba47f1cf440cc72662de172464eb0c730b15b3ce.jpg)  
(b) Pre-trained CWE, PCA

![](images/84eece3fdb9fc88026288fd66401241abf89cf6289d5560134a8c480ee895697.jpg)  
(c) Pre-trained CWE, ICA

![](images/0d216b94b994aad7ea293196b7e165a8c9b4c86cf50ba965e1abcaacddda4c9e.jpg)  
(d) Fine-tuned SCWE, Raw

![](images/c524f8d19bb888b0f9b544470db7b17fdcf0feec6b617c9448d7f7d6a79931e3.jpg)  
(e) Fine-tuned SCWE, PCA

![](images/530ecb9106dbd3d62e1fc8606a4e72a7d1fc3595b471a2b50f9e4b7df13e94a1.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 21: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in AM<sup>2</sup>iCo (Indonesian) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/3812445502bc2c4e2cda059bdd5c6b354c4633882a46fb60c6c0984d8628a94a.jpg)  
(a) Pre-trained CWE, Raw

![](images/352adeb503b5cf8bf55752ebbf752d9139362c7c215a1b7732d0bb237b478aa2.jpg)

![](images/e491a0136922ff761d560a718a242ad188d768eb2aa29e5599f179c9742ea4e5.jpg)

![](images/b44665e8f4787e5ff458583dae1919ad1a8d203efc929d0cd29bf5c070861caa.jpg)  
(d) Fine-tuned SCWE, Raw

(b) Pre-trained CWE, PCA  
![](images/1815b5ec596c888349e8875f32867c5545c78a1b2481a96236f4e3228a78ba53.jpg)  
(e) Fine-tuned SCWE, PCA

(c) Pre-trained CWE, ICA  
![](images/2c8f7442a2cb9348132777202bba234b4091c83079d2dc82ee7056e500a0e898.jpg)  
(f) Fine-tuned SCWE, ICA

Figure 22: Visualisation of the top-50 dimensions of pre-trained CWEs (XLM-RoBERTa) and SCWEs (XL-LEXEME) for each instance in AM<sup>2</sup>iCo (Basque) dataset, where the difference of vectors is calculated for (a/d) Raw vectors, (b/e) PCA-transformed axes, and (c/f) ICA-transformed axes. In each figure, the upper/lower half uses instances for the True/False labels.

![](images/feebea8cb93430b4f72b9478eda539a979aa115d814e4bfa4c6a93827380d952.jpg)  
(a) Pre-trained CWE, De

![](images/458d633930546174ce7f1f052a8bc98790e3f8d0ad3b8f379c15bb7b96562d2f.jpg)  
(b) Fine-tuned SCWE, De

![](images/62793f6b9ad7e4613f876d3ed88b0af15e4e0d5169dc99f21d52ceaca583d1f6.jpg)  
(c) Pre-trained CWE, Fr

![](images/3fb6fbe48de39f295469690c704d61cc1bf3f6938d5e7365074464507999fa65.jpg)  
(d) Fine-tuned SCWE, Fr

![](images/4f4b873f68a71d546348ac332cef25eae46c6edd351d486773b0333d84107d14.jpg)  
(e) Pre-trained CWE, It

![](images/3f482fd07812b24323fbe93e17d851d12da2bac58aa59f0eead08f3ae3bdd77e.jpg)  
(f) Fine-tuned SCWE, It  
Figure 23: The ROC curve on the contextual SCD benchmark, XLWiC dataset (De: German, Fr: French, It: Italian). Raw indicates the performance of using full dimensions. PCA/ICA uses top-5/10/20/50/100% of axes.

![](images/018e7301ba0c1ecce2fdac6da92f403dec18bb78da40a5922e90d7381a3f1c60.jpg)  
(a) Pre-trained CWE, Ar

![](images/275f08bc8771b9da4d4a85a61dda20577a6ff59149663171c8e4db2409372d52.jpg)  
(b) Fine-tuned SCWE, Ar

![](images/7b8da1405f03e01896ad62067ccc0da30d4aeb83269c052c6a261ebbcac2af4c.jpg)  
(c) Pre-trained CWE, En

![](images/0c7c7de7b7464c74dbaf7c3be441b0a43aa05ec34ebca4dfddb31b55a29a79bd.jpg)  
(d) Fine-tuned SCWE, En

![](images/654b3b0bae9dbcbc3e02e1460a663ad18a4c8b69c24a8138f2f62a2372394c0b.jpg)  
(e) Pre-trained CWE, Fr

![](images/7b3811124452b3d4c1e3b170acae953aa97fe3732946f9cf1bfe9850d092f106.jpg)  
(f) Fine-tuned SCWE, Fr  
Figure 24: The ROC curve on the contextual SCD benchmark, MCLWiC dataset (Ar: Arabic, En: English, Fr: French). Raw indicates the performance of using full dimensions. PCA/ICA uses top-5/10/20/50/100% of axes.

![](images/65b6e2cb4d73c7352e248ed7cb59f9099a8242bda86a27cf04c87c49a970c26a.jpg)  
(a) Pre-trained CWE, Ru

![](images/520be2d0adff82599bf698fe384e65b874663181795ad2e0b7eb81eef5d50fcd.jpg)  
(b) Fine-tuned SCWE, Ru

![](images/7cde98b61138ba7fd23241c0883e3a540ed32c923cfaa824d04752fc59b7af75.jpg)  
(c) Pre-trained CWE, Zh

![](images/adca2a42b4863887989d8ef8ceae979dce5636633d820e3b32ab8b7ae87d9558.jpg)  
(d) Fine-tuned SCWE, Zh  
Figure 25: The ROC curve on the contextual SCD benchmark, MCLWiC dataset (Ru: Russian, Zh: Chinese). Raw indicates the performance of using full dimensions. PCA/ICA uses top-5/10/20/50/100% of axes.

![](images/ed8291dd6ad4570053f9cdc1e5f8a501d99cf31e80772dcf36040f1fef09ee88.jpg)  
(a) Pre-trained CWE, De

![](images/267d5ff14669022870743805a6163ac54c40bcdb3a5d9a5f18518c3a447ed2ed.jpg)  
(b) Fine-tuned SCWE, De

![](images/56b45f34bce5633ab0265d4ca263ae0013688ffd30231284a1ef67c1f26c1650.jpg)  
(c) Pre-trained CWE, Ru

![](images/347f44bb93a30a9de0982a31b31ecb9834d187e61a88f688512dc0ca935e672e.jpg)  
(d) Fine-tuned SCWE, Ru

![](images/aa9ead13fdef10e0762328a08f8e1805338c0b688d31ebcdcc8f042e008066d2.jpg)  
(e) Pre-trained CWE, Ja

![](images/e76866f830ee527d6c380d82c3cbaf58a32a13670fb73b032ddaeb3b44cced6c.jpg)  
(f) Fine-tuned SCWE, Ja

![](images/589128c566067da881aea381c05cc6fa3eb87dc4fc76a18ea55f860796fdada0.jpg)  
(g) Pre-trained CWE, Zh

![](images/9ae40dfdcfac98b404a2a6f9bcdb2fdcc4753f37c7fd1d342b75b8880954d1b3.jpg)  
(h) Fine-tuned SCWE, Zh  
Figure 26: The ROC curve on the contextual SCD benchmark, AM<sup>2</sup>iCo dataset (De: German, Ru: Russian, Ja: Japanese, Zh: Chinese). Raw indicates the performance of using full dimensions. PCA/ICA uses top-5/10/20/50/100% of axes.

![](images/1cfb0aab4955f46771c9aeb1f1b625348a36fb1c0d570992b74b7b0ca5d02354.jpg)  
(a) Pre-trained CWE, Ar

![](images/06d534f1d85ecbe2487a1f174b95d14886e51dae068974705a4e5520b09ad481.jpg)  
(b) Fine-tuned SCWE, Ar

![](images/290ea500ae9023d138c0f6bafbbad3c39268ba53a17bbafd1944d2c8369a6dbd.jpg)  
(c) Pre-trained CWE, Ko

![](images/79da5b09181453655addad835243de11942c695b452adee6df660a18eb8f8b83.jpg)  
(d) Fine-tuned SCWE, Ko

![](images/21b9c750db3cfdc0d2adfd0008561241decf6ae98b8ff4586daa0365f3108bb8.jpg)  
(e) Pre-trained CWE, Fi

![](images/efe9fdb2e41a4fd623945c4c1ea0d699d4a2dfacc6bee0b3c086f187e44dac2f.jpg)  
(f) Fine-tuned SCWE, Fi  
Figure 27: The ROC curve on the contextual SCD benchmark, $\mathbf { A M ^ { 2 } i C o }$ dataset (Ar: Arabic, Ko: Korean, Fi: Finnish). Raw indicates the performance of using full dimensions. PCA/ICA uses top-5/10/20/50/100% of axes.

![](images/4115d1ab54321c4dc86a1ab929a56c8f7dc357a0b327825657979bcecf7c207b.jpg)  
(a) Pre-trained CWE, Tr

![](images/33abcbd53be5619a1086e018c892e198aa1e9aa55bcccf7cf895db8a832bdd30.jpg)  
(b) Fine-tuned SCWE, Tr

![](images/6d466a4f95d989a61c80a06dce41ff47e20cf30e429269eed7574bb6a87996f4.jpg)  
(c) Pre-trained CWE, Id

![](images/e3f9e04b4b449614371d1a3e2c66e838e68d07699b8dc4a622b7471650acd254.jpg)  
(d) Fine-tuned SCWE, Id

![](images/b1376266ee4bf16e7d790dfda05d2ebbbc98ec60a55ba1dfb1a0985550f1e345.jpg)  
(e) Pre-trained CWE, Eu

![](images/88882d6f1c93a3c5153a25d5679524fd34b1d9f1804b0520e6f335858279db30.jpg)  
(f) Fine-tuned SCWE, Eu

Figure 28: The ROC curve on the contextual SCD benchmark, $\mathbf { A M ^ { 2 } i C o }$ dataset (Tr: Turkish, Id: Indonesian, Eu: Basque). Raw indicates the performance of using full dimensions. PCA/ICA uses top-5/10/20/50/100% of axes

![](images/89383749ef2d2e8dc700bc4e2cc7f2f2ff4c7984b8a01ea926abb21490e6f644.jpg)  
(a) Pre-trained CWE, De

![](images/25fafcb306eea7b6e886be6f5e086fde653b646f87c13bc564ae92208a87337f.jpg)  
(b) Fine-tuned SCWE, De

![](images/7de199a042af3f36f36eb0a0978d6e31e169ab61bafb0837d1a15746b023cab8.jpg)  
(c) Pre-trained CWE, Sv

![](images/d18badb862f5efb85c18fbbdc6a3078d0f86c8bb779ea280ce3dc1435b3f276d.jpg)  
(d) Fine-tuned SCWE, Sv

![](images/9e67051e0681871a9c4b34afe42b8e4b99469587e116377419902e9368db160c.jpg)  
(e) Pre-trained CWE, La

![](images/5cf850162f34c9d0d806cadd40d222ce112f049febfe6188b0eff9f84bf6578f.jpg)  
(f) Fine-tuned SCWE, La

Figure 29: The ROC curve on the temporal SCD benchmark, SemEval-2020 Task 1 (De: German, Sv: Swedish, La: Latin). Raw indicates the performance of using full dimensions. PCA/ICA uses top-5/10/20/50/100% of axes.

![](images/6a5a156fdadbd4b3f6c29cc708ffa387c20b176d4bc3b273c97e475ee24fe80b.jpg)  
(a) Pre-trained CWE, De

![](images/bb0f7daf042f9c94e0b41285dbef1617fc4977faba476ac1efe4d37c56f65633.jpg)  
(b) Fine-tuned SCWE, De

![](images/bedce4fcbdfd9db85c23799de5191ceff1fd626eb27256100d50c9c07d2c31b2.jpg)  
(c) Pre-trained CWE, Sv

![](images/cf8b38ff3496492347f7581573e0be3fb17cc8b0d064cb47b7110c6e92e67c5c.jpg)  
(d) Fine-tuned SCWE, Sv

![](images/9a63f23ecebdea8f57bf7a20f40ea6d8611c9cfb3bb6852a2700a3d47896a876.jpg)  
(e) Pre-trained CWE, La

![](images/f7923a706a9861480377bcd1e5f8051183380a5aa22bae523fbafe3e70a3ad45.jpg)  
(f) Fine-tuned SCWE, La  
Figure 30: Spearman’s rank correlation on the temporal SCD benchmark, SemEval-2020 Task 1 (De: German, Sv: Swedish, La: Latin). Raw indicates the performance of using full dimensions. PCA/ICA cumulatively uses sorted axes.