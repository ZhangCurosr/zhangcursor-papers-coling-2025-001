# Efficient Solutions For An Intriguing Failure of LLMs: Long Context Window Does Not Mean LLMs Can Analyze Long Sequences Flawlessly

Peyman Hosseini<sup>1</sup> Ignacio Castro<sup>1</sup> Iacopo Ghinassi<sup>1</sup> Matthew Purver<sup>1,2</sup>

<sup>1</sup>School of EECS, Queen Mary University of London, London, UK

<sup>2</sup>Department of Knowledge Technologies, Jožef Stefan Institute, Ljubljana, Slovenia {s.hosseini, i.castro, i.ghinassi, m.purver}@qmul.ac.uk

## Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities in comprehending and analyzing lengthy sequential inputs, owing to their extensive context windows that allow processing millions of tokens in a single forward pass. However, this paper uncovers a surprising limitation: LLMs fall short when handling long input sequences. We investigate this issue using three datasets and two tasks (sentiment analysis and news categorization) across various LLMs, including Claude 3, Gemini Pro, GPT 3.5 Turbo, Llama 3 Instruct, and Mistral Instruct models. To address this limitation, we propose and evaluate ad-hoc solutions that substantially enhance LLMs’ performance on long input sequences by up to 50%, while reducing API cost and latency by up to 93% and 50%, respectively.

## 1 Introduction

LLMs have demonstrated remarkable capabilities in natural language understanding and generation tasks. Leveraging extensive pretraining on massive text corpora, the new generation of LLMs can perform a wide range of language tasks with minimal task-specific fine-tuning. Additionally, these LLMs are equipped with behemothic context windows that enable them to analyze inputs spanning up to tens or hundreds of pages in one forward pass. In this paper, we study the performance of Claude 3 Haiku (Anthropic, 2024), GPT3.5-Turbo (OpenAI, 2022), Gemini-1.0-pro (Team et al., 2023), Llama 3 8b Instruct (FacebookResearch, 2024), and Mistral 7b Instruct (Jiang et al., 2023a). Some of these LLMs are equipped with context windows that can support up to 200,000 tokens in one forward pass.

Related Work. Prompting strategies have emerged as a promising avenue for improving LLM performance by providing concise and informative input (Liu et al., 2023; Brown et al.,

2020; Jiang et al., 2023b; Ge et al., 2024). These strategies involve extracting key information from the input text and presenting it to the LLM in a structured manner. However, despite being equipped with context windows capable in theory of supporting large amounts of text, the performance of LLMs often suffers on lengthy input sequences as the prompt length grows (Li et al., 2023, 2024).

On the other hand, many general summarization techniques are available for condensing lengthy texts into more manageable snippets. Extractive Summarization methods such as TextRank (Mihalcea and Tarau, 2004) are widely used to identify and extract the most significant sentences from a document for different purposes (Cachola et al., 2020; Feng et al., 2022; Balcerzak et al., 2014; Wang et al., 2020). Although not designed for prompt compression, these techniques might therefore be useful in this context, and have relatively low computational overheads; in this paper, we therefore investigate the use of real-time summarization pipelines and text truncation techniques to boost LLM performance by optimizing the input while reducing their load.

Motivation. There is a body of research dedicated to studying the limitations of LLMs on long sequences and proposing mitigations at both architecture-level (Beltagy et al., 2020; Bertsch et al., 2024) as well as prompt-level (Wei et al., 2022). These studies often involve defining and exploring overly complex problems such as those about extreme-label classification (Li et al., 2024) or “Needle In a Haystack" (Machlab and Battle, 2024). However, a systematic study of LLM capabilities and limitations on long-form analysis tasks such as news categorization or sentiment analysis of long reviews which require a common general understanding of the input context is still lacking. Furthermore, the emphasis on approaches involving prompt-tuning has diverted attention away from optimizing and streamlining the information fed to LLMs. This study serves to fill these gaps by showcasing the failure of LLMs on canonical NLP tasks when dealing with long sequences and to ignite a spark of interest in the research community to explore the untapped potential of optimizing and condensing the information fed to LLMs.

![](images/4e09ce64a99b38f8010ba2515a1ebbee70ddff6a52bc7ad077c7216249abc6ac.jpg)  
Figure 1: The summarization pipelines for summarising information. The diverse summarization approach builds on top of the purely extractive approach but gives higher priority to lexical diversity.

## Contribution. Our main contributions are:

1. We systematically study the performance of state-of-the-art (SotA) LLMs on sentiment analysis and news categorization, revealing their limitations in processing long-form text effectively.

2. We propose and evaluate ad-hoc solutions using extractive and diverse summarization as well as selective truncation to condense input text, which substantially improves LLM performance by up to 50%, reduces API costs by as much as 93% and significantly reduces latency.

3. We present comprehensive empirical and ablation studies examining the relationship between input length, summarization strategies, and model performance, providing insights into optimal summarization approaches for LLMs.

## 2 Methodology

First, we present our summarization pipelines; then our evaluation scenarios.

## 2.1 Summarization Methodology

We study two pipelines for extracting key information from the documents and providing input for prompting. Given our requirement for real-time operation and semantic fidelity, we investigate only extractive approaches here.

Pure Extractive Summarization: As shown in Fig. 1a, we use TextRank (Mihalcea and Tarau, 2004), a well-known unsupervised extractive summarization algorithm, to select the most important sentences. TextRank uses a graph-based ranking model to measure the similarity between sentences and their centrality within the graph. We then write the instruction to the LLM (i.e., categorize or rate) and append the extracted sentences as input.

Diverse Summarization: Builds on top of the previous approach. We discard the least relevant sentences using TextRank ranking. Then we use TF-IDF to represent the sentences as vectors and calculate the diversity scores based on the dissimilarity between sentences using cosine similarity. The top N sentences with the highest diversity scores are chosen as the input used in prompting (see §A.2.1). This extension aims to maximise the diversity of the information for the LLMs.

## 2.2 Prompting Scenarios

To investigate the performance of LLMs on sentiment analysis and news categorization tasks involving long input sequences, we employ 7 prompting strategies and evaluate their effectiveness on three datasets, which we introduce in the next section. These prompting strategies include:

1. Full Context: The entire lengthy review is provided as input for analysis (Motivation: the baseline approach for comparison with other methods).

2. Full Context + Summary: The N-sentence summary extracted using Fig. 1a pipeline is appended to the lengthy review. (Motivation: how does emphasizing a selected summary with repetition affect the performance?)

<table><tr><td>Dataset</td><td>Domain</td><td>#Classes</td><td>Language</td><td>Min. Thresh. (#sent.)</td><td>Avg. Inp. Len. (#tokens)</td></tr><tr><td>GameSpot</td><td>Game Reviews</td><td>10*</td><td>English</td><td>45</td><td>2120</td></tr><tr><td>BBC News</td><td>News Documents</td><td>5</td><td>English</td><td>35</td><td>1150</td></tr><tr><td>20 NewsGroup</td><td>News Documents</td><td>20</td><td>English</td><td>60</td><td>3450</td></tr></table>

Table 1: Datasets’ key information. “Min. Thresh”. indicates the minimum sentence count used to filter documents suitable for our study, ensuring only sufficiently long documents are included. “Avg. Inp. Len” shows the mean document length in the studied subset for each dataset. For GameSpot, review scores (originally 1-100) are rounded to the nearest multiple of 10, resulting in 10 classes.

3. First Sentences: We crop the initial N sentences from the text and provide it as input. (Motivation: how does choosing the ‘opening section of a lengthy review affect the performance?)

4. Last Sentences: We crop the ending N sentences from the text and provide it as input. (Motivation: how does choosing the ‘ending’ section of a lengthy review affect the performance?)

5. Summary: We provide the extracted N sentence summary (Fig. 1a) as input. (Motivation: how does choosing a summary affect performance?)

6. Diverse Summary: We provide the extracted N-sentence summary (Fig. 1b) as input. (Motivation: how does giving more priority to lexical diversity in the summary affect performance?)

7. Random Sampling: We randomly select N sentences from the document. (Motivation: how does randomly choosing a short snippet perform in comparison to providing the full context?)

## 3 Evaluation

## 3.1 Datasets

We now introduce the datasets we study (see § A for more details). For each dataset, as our interest is in LLM performance with long inputs, we use only the subsets of the data that exceed a minimum length. We report the average length of the studied subset (in number of tokens) in Tabs. 5 to 7. Additionally, Tab. 1 aggregates and reports key information about each dataset in a nutshell.

GameSpot Reviews (GameSpot, 2024): more than 12,000 long game reviews with a sentiment score assigned by the author ranging from 1 to 100.

20 Newsgroups (Lang, 1995): nearly 20,000 news documents belonging to 20 different topic categories. High-level topics include politics, religion, sports, and computers.

BBC News Archive (Greene and Cunningham, 2006): 2,225 BBC articles covering business, entertainment, politics, sport, and tech.

## 3.2 Experiments

We evaluated the performance of Claude 3 Haiku, Gemini-1.0-Pro, GPT-3.5 Turbo, Llama 3 8b Instruct, and Mistral 7b Instruct on the datasets and tasks detailed in §3.1, using the prompting scenarios discussed in §2.2. The summarized results are available in Tabs. 2 to 4. More detailed analysis for each LLM is available in Tabs. 5 to 7 in § A. Aside from major performance metrics (i.e., loss, accuracy, and F1), we also report the average latency for each scenario under ‘Avg. Lat.’ column and the average input length (in terms of tokens) under ‘Inp. Len.’ column in all tables.

Please also note that the parameter N (discussed in items 2-7 of § 2.2) in Tabs. 2 to 4 for all summarization and truncation scenarios is set to 7. We chose 7 sentences for the summary/truncation length using insights driven from our ablation study presented in Fig. 5. That is 7 sentences provide a sweet spot between short length and performance.

## 3.2.1 Sentiment Analysis

<table><tr><td>Scenario</td><td>MSE</td><td>MAE</td><td>Accuracy</td><td>Avg. Lat.</td><td>Inp. Len.</td></tr><tr><td>Full</td><td>272.8 (6)</td><td>11.7 (6)</td><td>36.0 (7)</td><td>1.27 (6)</td><td>2120 (6)</td></tr><tr><td>Full+Sum.</td><td>403.1 (7)</td><td>13.5 (7)</td><td>38.2 (6)</td><td>1.33 (7)</td><td>2450 (7)</td></tr><tr><td>First Sent.</td><td>169.1 (5)</td><td>9.9 (5)</td><td>41.2 (5)</td><td>0.82 (1)</td><td>320 (1)</td></tr><tr><td>Last Sent.</td><td>99.6 (1)</td><td>7.9 (1)</td><td>50.7 (2)</td><td>0.82 (1)</td><td>320 (1)</td></tr><tr><td>Sum.</td><td>124.2 (2)</td><td>8.8 (4)</td><td>48.2 (4)</td><td>0.82 (1)</td><td>320 (1)</td></tr><tr><td>Div. Sum.</td><td>133.7 (4)</td><td>8.6 (2)</td><td>54.0 (1)</td><td>0.82 (1)</td><td>320 (1)</td></tr><tr><td>Rand. Samp.</td><td>129.6 (3)</td><td>8.7 (3)</td><td>50.0 (3)</td><td>0.82 (1)</td><td>320 (1)</td></tr></table>

Table 2: Average performance of different LLMs for Sentiment Analysis on GameSpot over 5 runs.

As shown in Tab. 2, both Full and Full+Sum approaches failed to perform favourably in predicting the article scores. However, extracting a subset of the input text, and providing it in the prompt, even through randomly sampling sentences, yielded superior performance. The ‘Last Sent.’ scenario performs the best in both loss metrics while the ‘Div Sum.’ achieves the highest accuracy by a substantial margin, performing 50% better than when providing the LLM with Full Context. Detailed analysis of the performance for each LLM on the GameSpot dataset is available in Tab. 7.

## 3.2.2 News Categorization

We evaluated the performance of LLMs on two news categorization datasets. Tabs. 3 and 4 summarize these results across different prompting scenarios. Detailed analyses for each LLM for both experiments are available in Tabs. 5 and 6 in §A.

<table><tr><td>Scenario</td><td>Mac. F1</td><td>Acc.</td><td>Avg. Lat.</td><td>Inp. Len.</td></tr><tr><td>Full</td><td>0.27 (7)</td><td>35.2 (7)</td><td>1.58 (6)</td><td>3450 (6)</td></tr><tr><td>Full+Sum.</td><td>0.30 (3)</td><td>38.2 (5)</td><td>1.94 (7)</td><td>3700 (7)</td></tr><tr><td>First Sent.</td><td>0.30 (3)</td><td>39.1 (3)</td><td>0.79 (1)</td><td>240 (1)</td></tr><tr><td>Last Sent.</td><td>0.29 (5)</td><td>39.1 (3)</td><td>0.79 (1)</td><td>240 (1)</td></tr><tr><td>Sum.</td><td>0.31 (1)</td><td>39.4 (2)</td><td>0.79 (1)</td><td>240 (1)</td></tr><tr><td>Div. Sum.</td><td>0.31 (1)</td><td>39.5 (1)</td><td>0.79 (1)</td><td>240 (1)</td></tr><tr><td>Rand. Samp.</td><td>0.29 (5)</td><td>37.6 (6)</td><td>0.79 (1)</td><td>240 (1)</td></tr></table>

Table 3: Average performance of different LLMs for news categorization on 20 NewsGroup over 5 runs.

For the 20 NewsGroup dataset (see Tab. 3) both ‘Sum.’ and ‘Div. Sum.’ approaches achieve the highest F1 scores. With a 39.5% accuracy, ‘Div. Sum’ outperforms all other scenarios in this metric. Importantly, all approaches achieve better results than providing the full context to the LLM, showing the effectiveness of summarising the information provided to the LLM for this dataset.

<table><tr><td>Scenario</td><td>Mac. F1</td><td>Acc.</td><td>Avg. Lat.</td><td>Inp. Len.</td></tr><tr><td>Full</td><td>0.51 (6)</td><td>54.0 (6)</td><td>1.07 (6)</td><td>1150 (6)</td></tr><tr><td>Full+Sum.</td><td>0.50 (7)</td><td>53.7 (7)</td><td>1.72 (7)</td><td>1400 (7)</td></tr><tr><td>First Sent.</td><td>0.61 (1)</td><td>64.5 (1)</td><td>0.78 (1)</td><td>230 (1)</td></tr><tr><td>Last Sent.</td><td>0.53 (4)</td><td>57.9 (4)</td><td>0.78 (1)</td><td>230 (1)</td></tr><tr><td>Sum.</td><td>0.56 (3)</td><td>59.8 (3)</td><td>0.78 (1)</td><td>230 (1)</td></tr><tr><td>Div. Sum.</td><td>0.58 (2)</td><td>60.9 (2)</td><td>0.78 (1)</td><td>230 (1)</td></tr><tr><td>Rand. Samp.</td><td>0.53 (4)</td><td>57.6 (5)</td><td>0.78 (1)</td><td>230 (1)</td></tr></table>

Table 4: Average performance of different LLMs for news categorization task on BBC News over 5 runs.

As summarized in Tab. 4, we observe a similar trend for the BBC dataset: all approaches except ‘Full+Sum.’ outperform ‘Full’ scenario in all metrics. This emphasizes the importance of selectively summarising the information provided to LLMs.

## 3.2.3 Ablation Studies

To further investigate the performance of our models under different conditions, we conducted two ablation studies. First, we varied the truncation/- summary lengths from 3 to 15 sentences for all models and evaluated their performance on the GameSpot dataset (Figs. 2 and 5). Our solutions consistently outperformed the baseline across different lengths. As you can see from Fig. 5, providing the LLM with as little as 4-5 sentences truncated from the text (whether from its beginning sentences, ending sentences, or even randomly) yields generally better results compared to the LLMs processing the full document. This study aims to isolate the impact of length, revealing LLMs’ performance degradation beyond 7-8 sentences. The performance curves for truncation methods (First Sentences and Last Sentences) provide clear evidence of the relationship between input length and model performance across the 3-15 sentence range. We elaborate on these findings in §4.

In the second ablation, we studied the effect of temperature on the Mistral and Llama 3 Instruct models for news categorization on the 20 News-Group dataset (Fig. 4). Since we are using LLMs for classification and categorization purpose, we try temperatures below 0.1 as higher temperatures lead to higher randomness which is not ideal for classification purposes. We try 0.000, 0.025, 0.050, 0.075, and 0.100 temperatures to see how the performance of the models is affected when changing only this variable. We see that across all these temperatures, the summarization approaches demonstrate superior performance compared to the “Full Text” baseline. This experiment demonstrates that LLMs’ performance degradation with long sequences persists across different temperature settings, indicating that the observed limitations in handling long texts cannot be attributed to the models’ probabilistic sampling behavior.

## 4 Discussion

Cohesiveness-Performance Gain Study: We conducted a preliminary study analyzing the correlation between document cohesiveness and performance gains attained by truncation and summarization approaches for each corpus. Using Average Relative Proximity (ARP) score (Ghinassi et al., 2023) with average cosine similarity scoring over 2-sentence segments, we found that more cohesive corpora (lower ARP) had higher average performance gains from summarization (Fig. 3). Future studies could establish these findings further.

![](images/692a1f2abf6ef5593c6ba72f0410e692c61582074442d5974cae44db698deeb0.jpg)

![](images/df96f8b2ec90ec07268e6d7526b7e9d77113867fbf83bb34093abfbd19bd0532.jpg)

![](images/89b9e53751285e683af2e6fdf109511d2a7a2f0f94a2c0e1b4376e6704b662ad.jpg)

Figure 2: Ablation study on the length of the selected truncation/summary for different scenarios using Claude 3 Haiku over 5 runs with 85% Confidence Intervals. The results show the efficacy of approaches optimizing LLMs input. ‘Full’ context performs poorly on all metrics. Additionally, after the length of input exceeds 10 sentences, less meaningful improvement in the performance of all scenarios is observed. A similar trend is seen for all LLMs.  
![](images/dfb28b3cd95bc069cec57f40595010a6015bd1a1e24798eaccfd64d43f76c35a.jpg)  
Figure 3: Performance gain (% accuracy boost) vs. normalized ARP score for each summarization/truncation scenario compared to the ‘Full’ baseline. Lower ARP scores (more cohesive corpora) generally yield higher performance gains across scenarios.

Isolation of Length as a Contributing Factor: This study quantitatively assesses Large Language Models’ (LLMs) performance degradation with lengthy inputs, even for straightforward tasks, and proposes summarization as a mitigation strategy. Our findings demonstrate the efficacy of summarization in addressing length-related challenges. However, the performance improvements extend beyond length reduction alone. This is because summarization inherently identifies salient information crucial for tasks like Sentiment Analysis and News Categorization. Our ablation study (see Fig. 5) isolates the length factor by demonstrating that with simple truncation methods such as First Sentences, performance metrics plateau after 7- 8 sentences, corroborating our central argument about LLMs’ inherent difficulties in processing longer sequences.

Findings and Future Directions: This study’s findings highlight that despite long context windows, SotA LLMs still struggle to effectively process long text sequences, a critical limitation underexamined by prior research for common NLP tasks relying on contextual understanding. Our results highlight the need for more research into optimizing lengthy text inputs to enhance LLM performance. Additionally, we advocate for the development of specialized datasets where input length serves as the primary controlled variable while maintaining consistent content complexity. Such datasets would enable researchers to systematically analyze LLMs’ performance degradation with increasing sequence length, particularly for tasks like sentiment analysis and news categorization where contextual understanding is crucial.

## 5 Conclusions

This paper examined the performance of various LLMs (Claude 3 Haiku, Gemini 1 Pro, GPT 3.5 Turbo, Llama 3 Instruct, and Mistral Instruct) on lengthy inputs in core NLP tasks. The results, across three datasets and two tasks, consistently indicate longer inputs result in worse performance. Further ablation experiments on truncation/summarization length and model temperature disqualified the random sampling behaviour of the LLMs to be a confounding factor and showcased LLMs struggles in even handling moderately lengthy inputs. We proposed several ad-hoc solutions to substantially enhance LLMs’ performance (up to 50%) on long input sequences and reduce API cost (up to 93%) and average latency (up to 50%).

## Acknowledgements

This work was partially supported by UKRI via the Centre for Doctoral Training in Intelligent Games and Game Intelligence (IGGI, EP/S022325/1) and projects AP4L (EP/W032473/1), ARCIDUCA (EP/W001632/1) and AdSoLve (Responsible AI UK, EP/Y009800/1, project KP0016); and by the Slovenian Research Agency via research core funding for the programme Knowledge Technologies (P2-0103) and the project EMMA (L2-50070).

We also thank Mehran Hosseini and Zahraa Al-Sahili for their comments and discussions which helped us improve this manuscript.

## Limitations

Rapid change of model specifications. First, we examined a diverse set of SotA large language models, all of which are among the most commonly used LLMs. However, the rapidly evolving nature of this field means these findings may not be fully generalized to future LLMs with different architectures and training paradigms.

Tasks requiring general context understanding. Second, although we evaluated performance on two core NLP tasks (sentiment analysis and news categorization) across three datasets, further research is needed to determine if our conclusions hold for a wider range of natural language understanding tasks and domains. The datasets we used, while lengthy, may not fully capture the types of longform content that LLMs will need to process in other real-world applications. However, our studies here laid the foundation to show the limitations of LLMs when dealing with long sequences even in canonical NLP tasks as an underexplored problem. We believe our ad-hoc solutions are applicable to a wide variety of tasks that require a general understanding of the input sequence rather than a detailed understanding of the whole context.

Societal Impact and Ethical Considerations In terms of societal impact, we believe our findings can help enable the more effective and efficient application of LLMs to tasks involving longer documents, which has the potential to unlock significant value in domains like business analytics, legal contract review, and scientific literature mining. In addition, our ad-hoc solutions are a step forward for democratizing access to AI. Even though using LLM APIs is becoming more affordable, our approaches reduce the API cost for users by up to 93% and this further enables more accessibility across different sections of society. At the same time, the ability to extract key information from lengthy privacy policies, terms of service, and other consumer agreements could be misused in ways that fail to represent the full context. As LLMs achieve greater summarization capabilities, extra care will be needed to ensure these summaries are accurate, unbiased and not misused for deceptive purposes.

Overall, while our work provides important empirical insights into the limitations of current LLMs on long sequence tasks and highlights promising directions for overcoming these challenges, we see it as a motivating starting point rather than a conclusive result. We encourage the research community to further test and expand on our findings to drive the development of more capable and robust prompting techniques.

## References

Anthropic. 2024. Claude. https://www.anthropic. com. Version claude-3-haiku-20240307. Accessed: 2024-04-05 to 2024-06-11.

Bartomiej Balcerzak, Wojciech Jaworski, and Adam Wierzbicki. 2014. Application of textrank algorithm for credibility assessment. In 2014 IEEE/WIC/ACM International Joint Conferences on Web Intelligence (WI) and Intelligent Agent Technologies (IAT), volume 1, pages 451–454. IEEE.

Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew Gormley. 2024. Unlimiformer: Long-range transformers with unlimited length input. Advances in Neural Information Processing Systems, 36.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Isabel Cachola, Kyle Lo, Arman Cohan, and Daniel Weld. 2020. TLDR: Extreme summarization of scientific documents. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4766–4777, Online. Association for Computational Linguistics.

FacebookResearch. 2024. Llama3. https://github. com/meta-llama/llama3. Accessed: 2024-05-05 to 2024-06-11.

Xiachong Feng, Xiaocheng Feng, and Bing Qin. 2022. A survey on dialogue summarization: Recent advances and new frontiers. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, pages 5453–5460. International Joint Conferences on Artificial Intelligence Organization. Survey Track.

GameSpot. 2024. Gamespot reviews. Accessed: 2024- 02-05 to 2024-06-11.

Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2024. In-context autoencoder for context compression in a large language model. In 12th International Conference on Learning Representations, ICLR. OpenReview.net.

Iacopo Ghinassi, Lin Wang, Chris Newell, and Matthew Purver. 2023. Comparing neural sentence encoders for topic segmentation across domains: not your typical text similarity task. PeerJ Computer Science, 9:e1593.

Derek Greene and Pádraig Cunningham. 2006. Practical solutions to the problem of diagonal dominance in kernel document clustering. In Proc. 23rd International Conference on Machine learning (ICML’06), pages 377–384. ACM Press.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023a. Mistral 7b. arXiv preprint arXiv:2310.06825.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023b. LLMLingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13358–13376. Association for Computational Linguistics.

Ken Lang. 1995. Newsweeder: Learning to filter netnews. In Proceedings of the Twelfth International Conference on Machine Learning, pages 331–339.

Dacheng Li, Rulin Shao, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph Gonzalez, Ion Stoica, Xuezhe Ma, and Hao Zhang. 2023. How long can context length of open-source llms truly promise? In NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following.

Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. 2024. Long-context llms struggle with long in-context learning. arXiv preprint arXiv:2404.02060.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Comput. Surv., 55(9).

Daniel Machlab and Rick Battle. 2024. Llm incontext recall is prompt dependent. arXiv preprint arXiv:2404.08865.

Rada Mihalcea and Paul Tarau. 2004. TextRank: Bringing order into text. In Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing, pages 404–411. Association for Computational Linguistics.

OpenAI. 2022. Gpt-3.5-turbo. https://openai.com/ blog/chatgpt. Accessed: 2024-03-10 to 2024-06- 11.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Hei-Chia Wang, Wei-Ching Hsiao, and Sheng-Han Chang. 2020. Automatic paper writing based on a rnn and the textrank algorithm. Applied Soft Computing, 97:106767.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837.

## A Appendix

## A.1 Datasets and Main Experiments

We used three datasets in our evaluations. Here, we provide a more detailed explanation of each dataset and task.

## A.1.1 Sentiment Analysis

GameSpot. The GameSpot Reviews dataset contains over 12,000 lengthy video game reviews with author-assigned sentiment scores ranging from 1 to 100. Almost all the reviews in this dataset are quite lengthy and by using a minimum threshold of 45 sentences, there are still thousands of reviews available in this dataset.

In the sentiment analysis experiments, we asked each LLM to give a sentiment rating of 1 to 100 for each document. Most labels in the data were multiples of 10 (i.e., 10, 20, 30, . . . , 90, 100). However, sometimes the labels had other values like 95, or 85 as well. To this end and to cover even the corner cases, we asked the LLM to predict the label as an integer from 1 to 100. The calculation of MSE and MAE metrics is straightforward and according to the standard definition. For calculating the accuracy, we considered a prediction as accurate if it was within 5 scoring points of the label. For example, if the label has a value of 70, a predicted label between 65-75 range is considered an accurate prediction and any prediction outside this range is considered not accurate.

Regarding the temperature used in this study as well as other studies, we tried different temperature values for the LLMs but no significant change or decrease was observed by doing this and summarization/truncation methods always showed superior performance. The results reported in this experiment are the average over 5 runs and where applicable we have reported the 85% Confidence Interval as well. The temperature in the experiments reported here was set to 0.01.

## A.2 News Categorization

20 NewsGroup. The 20 Newsgroups dataset features nearly 20,000 documents across 20 topic categories like politics, religion, sports and computers. We focused on the subset longer than 60 sentences, averaging 3450 tokens per document when tokenized by the NLTK word tokenizer.

BBC News Archive. The BBC News Archive, consisting of 2,225 articles covering business, entertainment, politics, sports and tech. We focused our study on the subset with more than 35 sentences, averaging 1150 tokens per document when tokenized by the NLTK word tokenizer.

As we explain in the main body of the paper as well as the results shown in fig. 4, when trying different temperatures, there were no meaningful changes in the results and order of the approaches in terms of their performance.

<table><tr><td>Model</td><td>Scenario</td><td>Mac. F1</td><td>Acc.</td><td>Avg. Lat.</td><td>Inp. Len.</td></tr><tr><td rowspan="7">Cla ε Haiku</td><td>Full</td><td>0.54</td><td>67.8</td><td>1.24</td><td>3450</td></tr><tr><td>Full+Sum.</td><td>0.58</td><td>71.2</td><td>1.78</td><td>3700</td></tr><tr><td>First Sent.</td><td>0.50</td><td>66.4</td><td>0.72</td><td>240</td></tr><tr><td>Last Sent.</td><td>0.49</td><td>64.4</td><td>0.72</td><td>240</td></tr><tr><td>Sum.</td><td>0.56</td><td>69.6</td><td>0.72</td><td>240</td></tr><tr><td>Div. Sum.</td><td>0.52</td><td>65.5</td><td>0.72</td><td>240</td></tr><tr><td>Rand. Samp.</td><td>0.50</td><td>64.6</td><td>0.72</td><td>240</td></tr><tr><td rowspan="7">GP  bo0</td><td>Full</td><td>0.38</td><td>47.1</td><td>1.19</td><td>3450</td></tr><tr><td>Full+Sum.</td><td>0.41</td><td>53.2</td><td>1.95</td><td>3700</td></tr><tr><td>First Sent.</td><td>0.42</td><td>45.2</td><td>0.33</td><td>240</td></tr><tr><td>Last Sent.</td><td>0.40</td><td>42.8</td><td>0.33</td><td>240</td></tr><tr><td>Sum.</td><td>0.38</td><td>43.4</td><td>0.33</td><td>240</td></tr><tr><td>Div. Sum.</td><td>0.39</td><td>44.3</td><td>0.33</td><td>240</td></tr><tr><td>Rand. Samp.</td><td>0.41</td><td>43.1</td><td>0.33</td><td>240</td></tr><tr><td rowspan="7">Geani ro</td><td>Full</td><td>0.30</td><td>34.6</td><td>3.48</td><td>3450</td></tr><tr><td>Full+Sum.</td><td>0.36</td><td>35.2</td><td>3.88</td><td>3700</td></tr><tr><td>First Sent.</td><td>0.36</td><td>43.2</td><td>1.12</td><td>240</td></tr><tr><td>Last Sent.</td><td>0.45</td><td>46.4</td><td>1.12</td><td>240</td></tr><tr><td>Sum.</td><td>0.39</td><td>40.4</td><td>1.12</td><td>240</td></tr><tr><td>Div. Sum.</td><td>0.38</td><td>40.8</td><td>1.12</td><td>240</td></tr><tr><td>Rand. Samp.</td><td>0.36</td><td>40.2</td><td>1.12</td><td>240</td></tr><tr><td rowspan="7">Misitrab</td><td>Full</td><td>0.01</td><td>1.1</td><td>0.96</td><td>3450</td></tr><tr><td>Full+Sum.</td><td>0.01</td><td>2.3</td><td>0.99</td><td>3650</td></tr><tr><td>First Sent.</td><td>0.03</td><td>11.5</td><td>0.87</td><td>180</td></tr><tr><td>Last Sent.</td><td>0.04</td><td>12.2</td><td>0.87</td><td>180</td></tr><tr><td>Sum.</td><td>0.05</td><td>12.3</td><td>0.87</td><td>180</td></tr><tr><td>Div. Sum.</td><td>0.04</td><td>12.6</td><td>0.87</td><td>180</td></tr><tr><td>Rand. Samp.</td><td>0.05</td><td>12.2</td><td>0.87</td><td>180</td></tr><tr><td rowspan="7">8 118</td><td>Full</td><td>0.15</td><td>25.6</td><td>1.02</td><td>3450</td></tr><tr><td>Full+Sum.</td><td>0.16</td><td>29.3</td><td>1.11</td><td>3650</td></tr><tr><td>First Sent.</td><td>0.17</td><td>29.1</td><td>0.89</td><td>180</td></tr><tr><td>Last Sent.</td><td>0.17</td><td>29.8</td><td>0.89</td><td>180</td></tr><tr><td>Sum.</td><td>0.19</td><td>31.2</td><td>0.89</td><td>180</td></tr><tr><td>Div. Sum.</td><td>0.21</td><td>34.1</td><td>0.89</td><td>180</td></tr><tr><td>Rand. Samp.</td><td>0.16</td><td>27.8</td><td>0.89</td><td>180</td></tr></table>

Table 5: The performance of different LLMs for Categorization task on 20 NewsGroup dataset. N parameter for summarization and truncation is set to 7 for the Mistral and Llama models and 10 for the others.

The results reported in this paper in the tables for the news categorization task are averaged over 5 runs and we have reported 85% Confidence Interval when applicable. The temperature of the models was set to 0.0 for the results reported in the tables.

## A.2.1 Diverse Summarization

Here we provide more explanation about the diverse summarization approach and how the greencoloured component (in Fig. 1b) which is the diversity selector in our algorithm works.

<table><tr><td>Model</td><td>Scenario</td><td>Mac. F1</td><td>Acc.</td><td>Avg. Lat.</td><td>Inp. Len.</td></tr><tr><td rowspan="7">CIlue εE Haiku</td><td>Full</td><td>0.63</td><td>63.8</td><td>0.69</td><td>1150</td></tr><tr><td>Full+Sum.</td><td>0.67</td><td>67.1</td><td>1.45</td><td>1400</td></tr><tr><td>First Sent.</td><td>0.69</td><td>70.4</td><td>0.65</td><td>230</td></tr><tr><td>Last Sent.</td><td>0.56</td><td>56.9</td><td>0.65</td><td>230</td></tr><tr><td>Sum.</td><td>0.61</td><td>61.5</td><td>0.65</td><td>230</td></tr><tr><td>Div. Sum.</td><td>0.64</td><td>63.8</td><td>0.65</td><td>230</td></tr><tr><td>Rand. Samp.</td><td>0.60</td><td>61.2</td><td>0.65</td><td>230</td></tr><tr><td rowspan="7">G  b0</td><td>Full</td><td>0.75</td><td>83.8</td><td>0.54</td><td>1150</td></tr><tr><td>Full+Sum.</td><td>0.67</td><td>76.4</td><td>1.08</td><td>1400</td></tr><tr><td>First Sent.</td><td>0.80</td><td>86.3</td><td>0.49</td><td>230</td></tr><tr><td>Last Sent.</td><td>0.69</td><td>78.4</td><td>0.49</td><td>230</td></tr><tr><td>Sum.</td><td>0.72</td><td>79.9</td><td>0.49</td><td>230</td></tr><tr><td>Div. Sum.</td><td>0.69</td><td>78.7</td><td>0.49</td><td>230</td></tr><tr><td>Rand. Samp.</td><td>0.68</td><td>77.7</td><td>0.49</td><td>230</td></tr><tr><td rowspan="7">Geani ro</td><td>Full</td><td>0.65</td><td>61.2</td><td>2.26</td><td>1150</td></tr><tr><td>Full+Sum.</td><td>0.69</td><td>66.6</td><td>2.61</td><td>1400</td></tr><tr><td>First Sent.</td><td>0.63</td><td>59.7</td><td>1.04</td><td>230</td></tr><tr><td>Last Sent.</td><td>0.64</td><td>58.4</td><td>1.04</td><td>230</td></tr><tr><td>Sum.</td><td>0.61</td><td>57.7</td><td>1.04</td><td>230</td></tr><tr><td>Div. Sum.</td><td>0.61</td><td>56.8</td><td>1.04</td><td>230</td></tr><tr><td>Rand. Samp.</td><td>0.61</td><td>56.2</td><td>1.04</td><td>230</td></tr><tr><td rowspan="7">Misittrtb</td><td>Full</td><td>0.13</td><td>22.1</td><td>0.97</td><td>1150</td></tr><tr><td>Full+Sum.</td><td>0.03</td><td>11.7</td><td>1.67</td><td>1330</td></tr><tr><td>First Sent.</td><td>0.32</td><td>37.1</td><td>0.85</td><td>170</td></tr><tr><td>Last Sent.</td><td>0.23</td><td>32.1</td><td>0.85</td><td>170</td></tr><tr><td>Sum.</td><td>0.33</td><td>38.5</td><td>0.85</td><td>170</td></tr><tr><td>Div. Sum.</td><td>0.35</td><td>39.9</td><td>0.85</td><td>170</td></tr><tr><td>Rand. Samp.</td><td>0.28</td><td>33.5</td><td>0.85</td><td>170</td></tr><tr><td rowspan="7">80 11mm8</td><td>Full</td><td>0.38</td><td>39.2</td><td>0.89</td><td>1150</td></tr><tr><td>Full+Sum.</td><td>0.42</td><td>46.6</td><td>1.81</td><td>1330</td></tr><tr><td>First Sent.</td><td>0.62</td><td>68.8</td><td>0.85</td><td>170</td></tr><tr><td>Last Sent.</td><td>0.53</td><td>63.5</td><td>0.85</td><td>170</td></tr><tr><td>Sum.</td><td>0.54</td><td>61.2</td><td>0.85</td><td>170</td></tr><tr><td>Div. Sum.</td><td>0.59</td><td>65.3</td><td>0.85</td><td>170</td></tr><tr><td>Rand. Samp.</td><td>0.51</td><td>59.2</td><td>0.85</td><td>170</td></tr></table>

Table 6: The performance of different LLMs for Categorization task on BBC News dataset. N parameter for summarization and truncation is set to 7 for the Mistral and Llama models and 10 for the others.

To write equations describing what the green component is doing, we can focus on the main functions in this component and their inputs and outputs. Let’s denote the input text as $T ,$ the number of desired sentences as N, and the set of sentences in the text as ${ \cal S } = s _ { 1 } , s _ { 2 } , \ldots , s _ { M }$

1. Tokenize Sentences:

$$
S = { \mathrm { s e n t \_ t o k e n i z e } } ( T )\tag{1}
$$

2. Calculate sentence embeddings:

$$
E = { \mathrm { T f i d f V e c t o r i z e r } } ( S )\tag{2}
$$

where E is the TF-IDF matrix representing the embeddings of the sentences.

3. Calculate diversity scores:

$$
D = 1 - \cos \_ \sin ( E )\tag{3}
$$

$$
D _ { \mathrm { s u m } } = \sum _ { i = 1 } ^ { M } D _ { i }\tag{4}
$$

where $D$ is the dissimilarity matrix, and $D _ { \mathrm { s u m } }$ is the sum of dissimilarity scores for each sentence.

4. Select top N diverse sentences:

$$
S _ { \mathrm { t o p \mathrm { N } } } = \arg \operatorname* { m a x } _ { N } ( D _ { \mathrm { s u m } } )\tag{5}
$$

where $S _ { \mathrm { t o p s } }$ is the set of $N$ sentences with the highest diversity scores.

5. Generate the final summary by joining the sentences:

$$
\mathrm { s u m m a r y } = \mathrm { j o i n } ( S _ { \mathrm { t o p _ { N } } } )\tag{6}
$$

where the final summary is the concatenation of the selected top N diverse sentences.

The main steps denoted in the above equations can be summarized as follows:

1. Tokenize the input text $T$ into a set of sentences S.

2. Compute the TF-IDF embedding matrix E for the sentences.

3. Calculate the dissimilarity matrix D using cosine similarity and sum the dissimilarity scores for each sentence. In Eqn. (3), D represents the dissimilarity matrix, which is calculated as 1 minus the cosine similarity of the sentence embeddings E. This means D contains the pairwise dissimilarity scores between all sentences. In Eqn. (4), D\_sum is calculated by summing the dissimilarity scores for each sentence. This means $D _ { - }$ sum is a vector where each element represents the total dissimilarity score for a particular sentence compared to all other sentences.

4. Select the top N sentences $S _ { \mathrm { t o p \scriptscriptstyle N } }$ with the highest diversity scores.

5. Concatenate the selected sentences to form the final summary.

![](images/ff455eccb574beefcde2faff7e2fd8e1cc9e3df601c2162d2dd212b0114fc105.jpg)  
(a) Mistral 7b Instruct F1-Temperature Curve

![](images/02775e8f899a0fe5505b9cc61b746a32d4fb3d71afcb68f972f463a56c9adf96.jpg)  
(b) Mistral 7b Instruct Accuracy-Temperature Curve

![](images/86256e08376427171175e165ac6cda065fbb31e3a3d78fd86d60f3876dca81a7.jpg)  
(c) Llama3 8b Instruct F1-Temperature Curve

![](images/081e5451ce76156119115681322168370d58c60ceb06e4edbcca830177de7265.jpg)  
(d) Llama3 8b Instruct Accuracy-Temperature Curve

Figure 4: Ablation Study on temperature in the news categorization task on 20 newsgroup dataset. The results show the performance of the models does not experience much difference as we change the temperature from 0 to 0.1 aside from a slight increase in the confidence interval. These results are over 5 runs. The width of the 85% Confidence Interval for the ‘Random Sampling’ scenario is much bigger due to the randomness introduced by selecting the sentences.

<table><tr><td>Model</td><td>Scenario</td><td>MSE</td><td>MAE</td><td>Accuracy</td><td>Avg. Lat.</td><td>Inp. Len.</td></tr><tr><td rowspan="6">Cla ε Haku</td><td>Full</td><td>112.3 (6)</td><td>8.50 (6)</td><td>43.8 (7)</td><td>1.04</td><td>2120</td></tr><tr><td>Full+Sum.</td><td>80.2 (2)</td><td>7.17 (2)</td><td>50.6 (5)</td><td>1.05</td><td>2450</td></tr><tr><td>First Sent.</td><td>131.8 (7)</td><td>8.87 (7)</td><td>44.4 (6)</td><td>0.77</td><td>320</td></tr><tr><td>Last Sent.</td><td>77.0 (1)</td><td>6.58 (1)</td><td>60.2 (2)</td><td>0.77</td><td>320</td></tr><tr><td>Sum.</td><td>97.6 (3)</td><td>7.55 (4)</td><td>56.8 (4)</td><td>0.77</td><td>320</td></tr><tr><td>Div. Sum.</td><td>106.0 (5)</td><td>7.21 (3)</td><td>65.2 (1)</td><td>0.77</td><td>320</td></tr><tr><td rowspan="6">G  b0</td><td>Rand. Samp.</td><td>104.2 (4)</td><td>7.56 (5)</td><td>59.0 (3)</td><td>0.77</td><td>320</td></tr><tr><td>Full</td><td>134.9 (4)</td><td>9.76 (6)</td><td>40.4 (7)</td><td>0.59</td><td>2120</td></tr><tr><td>Full+Sum.</td><td>110.8 (1)</td><td>8.75 (1)</td><td>46.6 (5)</td><td>0.61</td><td>2450</td></tr><tr><td>First Sent.</td><td>177.3 (7)</td><td>10.81 (7)</td><td>41.0 (6)</td><td>0.47</td><td>320</td></tr><tr><td>Last Sent. Sum.</td><td>119.1 (3)</td><td>8.78 (2)</td><td>55.6 (2)</td><td>0.47</td><td>320</td></tr><tr><td>Div. Sum.</td><td>117.8 (2) 141.7 (6)</td><td>8.94 (3) 9.11 (4)</td><td>53.0 (3) 59.2 (1)</td><td>0.47</td><td>320 320</td></tr><tr><td rowspan="6">Geanıi ro</td><td>Rand. Samp.</td><td>135.3 (5)</td><td>9.27 (5)</td><td>51.8 (4)</td><td>0.47 0.47</td><td>320</td></tr><tr><td>Full</td><td>130.8 (6)</td><td>9.89 (7)</td><td>43.8 (7)</td><td>2.74</td><td>2120</td></tr><tr><td>Full+Sum.</td><td>117.6 (6)</td><td>9.14 (6)</td><td>53.4 (6)</td><td>2.96</td><td>2450</td></tr><tr><td>First Sent.</td><td>88.0 (4)</td><td>7.15 (4)</td><td>61.6 (5)</td><td>1.15</td><td>320</td></tr><tr><td>Last Sent.</td><td>83.3 (3)</td><td>7.14 (3)</td><td>63.8 (4)</td><td>1.08</td><td>320</td></tr><tr><td>Sum.</td><td>73.3 (1)</td><td>6.80 (1)</td><td>67.4 (2)</td><td>1.08</td><td>320</td></tr><tr><td></td><td>Div. Sum.</td><td>94.2 (5)</td><td>7.23 (5)</td><td>69.6 (1)</td><td>1.08</td><td>320</td></tr><tr><td rowspan="6">Misitrab</td><td>Rand. Samp.</td><td>78.8 (2)</td><td>7.05 (2)</td><td>66.2 (3)</td><td>1.08</td><td>320</td></tr><tr><td>Full</td><td>811.6 (6)</td><td>19.54 (6)</td><td>13.1 (6)</td><td>1.01</td><td>2120</td></tr><tr><td>Full+Sum. First Sent.</td><td>1532.1 (7) 236.4 (4)</td><td>30.68 (7)</td><td>8.9 (7)</td><td>1.04</td><td>2450</td></tr><tr><td>Last Sent.</td><td>93.5 (1)</td><td>10.81 (5) 7.96 (1)</td><td>29.0 (4) 38.0 (1)</td><td>0.91</td><td>320</td></tr><tr><td>Sum.</td><td>172.5 (4)</td><td>10.86 (4)</td><td>23.1 (5)</td><td>0.91 0.91</td><td>320 320</td></tr><tr><td>Div. Sum.</td><td>155.1 (2)</td><td>9.56 (2)</td><td>35.8 (2)</td><td>0.91</td><td>320</td></tr><tr><td></td><td>Rand. Samp.</td><td>164.1 (3)</td><td>9.57 (3)</td><td>34.6 (3)</td><td>0.91</td><td>320</td></tr><tr><td rowspan="6">1m8b</td><td>Full</td><td>174.7 (5)</td><td>10.62 (5)</td><td>39.1 (3)</td><td>0.95</td><td>2120</td></tr><tr><td>Full+Sum.</td><td>192.4 (6)</td><td>11.79 (7)</td><td>31.3 (6)</td><td>0.99</td><td>2450</td></tr><tr><td>First Sent.</td><td>212.1 (7)</td><td>11.76 (6)</td><td>30.1 (7)</td><td>0.90</td><td>320</td></tr><tr><td>Last Sent.</td><td>125.4 (1)</td><td>9.33 (1)</td><td>36.1 (5)</td><td>0.90</td><td>320</td></tr><tr><td>Sum.</td><td>159.7 (2)</td><td>10.06 (2)</td><td>40.5 (1)</td><td>0.90</td><td>320</td></tr><tr><td>Div. Sum.</td><td>171.6 (4)</td><td>10.15 (4)</td><td>40.0 (2)</td><td>0.90</td><td>320</td></tr><tr><td>Rand. Samp.</td><td></td><td>165.5 (3)</td><td>10.10 (3)</td><td>38.2 (4)</td><td>0.90</td><td>320</td></tr></table>

Table 7: The performance of different LLMs for Sentiment Analysis task on GameSpot dataset. N parameter for summarization and truncation is set to 7 for all the models.

![](images/f50551bfbab3f8bb9ad96fb74cb39a33ec92cabc1376da37d0503514704e75b6.jpg)  
(a) Claude3 Haiku Accuracy Curve

![](images/bfc499a0394b9bf450e01962f5d52aaba2c5b4234073869cd857eb896c03e753.jpg)  
(b) Claude3 Haiku MAE Curve

![](images/b64d36b2e658f6e7644fd383c382f28a3c4f0bb5708a9793ce719a615672adc7.jpg)  
(c) Claude3 Haiku MSE Curve

![](images/74a71a74d3d6663f2c058090c1152b107d5702462c8b52abe17cbb5438c65083.jpg)  
(d) Gemini Pro Accuracy Curve

![](images/c4a483d896b25bfa3435bfab0935c84c5226434463e4e87db2e27af327396d34.jpg)  
(e) Gemini Pro MAE Curve

![](images/64c0791bcbc0c6a59f169997cf7b51d92e3b7529bc19974d24ff4395c3014e5a.jpg)  
(f) Gemini Pro MSE Curve

![](images/8289f0f34a58536f5d775db4a65f6f8699fdcef1bfc877ec6cfe23d6cf84a45b.jpg)  
(g) GPT3.5 Turbo Accuracy Curve

![](images/d2a1287916a3ab12ae0bba61e49bbf5d5caca542e1a6e63dc17e86937c5e2952.jpg)  
(h) GPT3.5 Turbo MAE Curve

![](images/07b5924a14314c1c1db4db7e58b3b174c1b1345de5f1fc3bae49f3ab9cbc9dfa.jpg)  
(i) GPT3.5 Turbo MSE Curve

![](images/a817ca2d0f67a7ed3c52174c518fab4cbaa126daee6fe9b35e64e569e7ac8649.jpg)  
(j) Mistral 7b Instruct Accuracy Curve

![](images/29f55c6c925e9bd199f0c1225a4cdeeeb0dadea3c01a68328617320717ce33f0.jpg)  
(k) Mistral 7b Instruct MAE Curve

![](images/50d53ddf4ecf1db9b42600e642ae266d16a815a4e28b0cb6aa1f4c38e0e48b6b.jpg)  
(l) Mistral 7b Instruct MSE Curve

![](images/1ead41904c03547b9aaa7c0ba4a200648d542b0c6f5836ac1629328a158836f2.jpg)  
(m) Llama3 8b Instruct Accuracy Curve

![](images/652307980b61284ebc80a8734de62ac17b65d913151b9c3e7d144abbb76779ee.jpg)  
(n) Llama3 8b Instruct MAE Curve

![](images/504b7c9f580f85eb8dad2af0816b7446047e7f7a3937d50acd100f152a157cbc.jpg)  
(o) Llama3 8b Instruct MSE Curve  
Figure 5: LLMs performance over 5 runs on sentiment analysis on GameSpot dataset. The “Full Text” scenario is a horizontal line since it always contains the full text, not summary/truncated text and we have included it as a horizontal line here as baseline 1891