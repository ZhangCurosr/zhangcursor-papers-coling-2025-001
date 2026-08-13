# CMMaTH: A Chinese Multi-modal Math Skill Evaluation Benchmark for Foundation Models

Zhong-Zhi Li<sup>1,2</sup>\* , Ming-Liang Zhang<sup>1,2</sup>\*, Pei-jie Wang<sup>1,2</sup>, Jian Xu<sup>1,2</sup>, Rui-Song Zhang<sup>1,2</sup>, Fei Yin<sup>1,2</sup>, Zhi-Long Ji<sup>3</sup>, Jin-Feng Bai<sup>3</sup>, Zhen-Ru Pan<sup>3</sup>, Jiaxin Zhang<sup>4</sup>, Cheng-Lin Liu<sup>1,2†</sup>

School of Artifcial Intelligence, University of Chinese Academy of Sciences<sup>1</sup> MAIS, Institute of Automation of Chinese Academy of Sciences<sup>2</sup> Tomorrow Advancing Life<sup>3</sup>, University of Strathclyde <sup>4</sup>   
{lizhongzhi2022, zhangmingliang2018}@ia.ac.cn, {fyin, liucl}@nlpr.ia.ac.cn,   
{jizhilong, baijinfeng, panzhenru,}@tal.com, {jiaxinzhang625}@gmail.com

## Abstract

With the rapid advancements in multimodal large language models, evaluating their multimodal mathematical capabilities continues to receive wide attention. Although datasets such as MathVista have been introduced for evaluating mathematical capabilities in multimodal scenarios, there remains a lack of evaluation tools and datasets tailored for fine-grained assessment in Chinese K12 education. To systematically evaluate the ability of multimodal large models to solve Chinese multimodal mathematical problems, we propose a Chinese Multi-modal Math Skill Evaluation Benchmark (CMMaTH), containing 23,856 multimodal K12 math related questions, making it the largest Chinese multimodal mathematical problem benchmark to date. CM-MaTH includes questions ranging from elementary to high school levels, offering greater diversity in problem types, solution goals, visual elements, detailed knowledge points, and standard solution annotations. To facilitate stable, fast, and cost-free model evaluation, we have developed an open-source tool called GradeGPT, which is integrated with the CM-MaTH dataset. Our data and code are available at https://github.com/zzli2022/CMMaTH.

## 1 Introduction

Large language models (LLMs) excel in various language tasks, while Large Multimodal Models (LMMs) effectively handle visual-language problems. They drive advancements in natural language processing and computer vision, offering powerful solutions for complex tasks, demonstrating potential as versatile solvers for multimodal problems.

The systematic evaluation of LLM/LMMs’ performance across various mathematical reasoning scenarios has been an extensive research subject. GSM8K and MATH (Cobbe et al., 2021;

Hendrycks et al., 2021b) assessed the ability for multi-step mathematical reasoning by constructing a high-quality set of elementary school math word problems or various competition mathematics problems. By collecting a diverse set of mathematical problems containing both textual and visual components, Lu et al. (2023), Wang et al. (2024), Zhang et al. (2024c) systematically evaluated the ability of LMMs to perceive visual elements and solve corresponding multimodal problems. Shi et al. (2023a) constructed a multilingual mathematical reasoning dataset, MGSM, for evaluating the LLM reasoning ability in multilingual environments. However, in non-English multimodal contexts, especially in Chinese scenarios, there is still a lack of sufficiently detailed and diverse benchmarks for assessing mathematical abilities. To assess the capability of large language models in non-English contexts, Huang et al. (2023) and Zhang et al. (2024b) constructed multidisciplinary Chinese question answering datasets C-Eval and CMMMU to evaluate the knowledge and reasoning abilities of LMMs. However, C-Eval lacks evaluation in multimodal contexts, while CMMMU’s dataset has relatively low diversity, consisting of only 540 questions.

Meanwhile, existing datasets contain numerous problem scenarios that are not typically examined in real K12 education. There are inconsistencies between the assessment results and the actual abilities of students in K12 mathematics. Liu et al. (2024b) has introduced a benchmark for mathematics assessment, which includes questions with detailed annotations of textbook knowledge points. The goal is to address these discrepancies in real K12 educational settings. However, this benchmark is limited to text-only scenarios and does not include multimodal questions and knowledge points.

Existing math benchmarks for answer evaluation can be categorized into two types:rule-based Cobbe et al. (2021), Hendrycks et al. (2021b), He et al. (2024) and API-based methods Lu et al. (2023), Zhang et al. (2024c), Hendrycks et al. (2021a). API-based methods leverage the zero-shot in-context learning capabilities of LLM. However, these methods are associated with high costs and significant time consumption, often leading to unstable and inconsistent evaluation outcomes. Rulebased methods, aim to derive the correct option from model responses through the design of various extraction systems. Nonetheless, these methods face challenges in handling highly diverse contents of benchmarks. Also, it is difficult to maintain handcrafted rules for dynamically updated benchmarks. Current multimodal math benchmark evaluations often resort to multiple-choice or true/false question formats, using rules or API-based LLM to extract options for assessing answers.

![](images/9ea99e2baa7e4e43b0cd9928473b98e13d46d059e05e0404c364485c021cb7b9.jpg)

![](images/cfebd050054cd227b68c4aa4a6f586b8850bd328695914dafee886c3d4b0af4e.jpg)

Figure 1: The CMMaTH dataset focuses on the multi-modal mathematical ability assessment of Chinese scenes and has extremely strong diversity and large data scale. CMMaTH combines fine-grained multi-modal mathematical knowledge point annotation. The size of the hollow circle corresponds to the designed number of knowledge points in the dataset.
<table><tr><td>Dataset</td><td>Language Domain</td><td>Knowledge Annotation</td><td>Knowledge Domain</td><td>Knowledge Point Number</td><td>Size</td><td>Modality</td><td>Source</td><td>Answer</td></tr><tr><td>VQAv2(Goyal et al., 2017)</td><td>En</td><td></td><td>General</td><td></td><td>&gt; 1M</td><td>V+T</td><td>Annotated</td><td>Open/MC/TF</td></tr><tr><td>SEED(Li et al., 2023a)</td><td>En</td><td></td><td>General</td><td></td><td>19K</td><td>V+T</td><td>Annotated</td><td>MC</td></tr><tr><td>MMBench(Liu et al., 2023b)</td><td>En</td><td></td><td>General</td><td></td><td>3K</td><td>V+T</td><td>Repurposed</td><td>MC</td></tr><tr><td>MM-Vet(Yu et al., 2023)</td><td>En</td><td>××××</td><td>General</td><td></td><td>0.2K</td><td>V+T</td><td>Annotated</td><td>Open</td></tr><tr><td>ScienceQA(Lu et al., 2022)</td><td>En</td><td>x</td><td></td><td>Science</td><td>6K</td><td>V+T</td><td>Textbooks</td><td>MC</td></tr><tr><td>MMMU(Yue et al., 2023)</td><td>En</td><td>x</td><td>General</td><td></td><td>11.5K</td><td>V(30 Types)+OC</td><td>Textbooks</td><td>Open/MC</td></tr><tr><td>CMMMU(Zhang et al., 2024b)</td><td>ZH</td><td>X</td><td>General</td><td></td><td>&lt; 1K(Math Part)</td><td>V(5 Types)+OC</td><td>Internet</td><td>Open/MC</td></tr><tr><td>MathVista(Lu et al., 2023)</td><td>ZH/En</td><td>X</td><td>Math</td><td></td><td>1K/6K</td><td>V(5 Types)+OC</td><td>Synthesized</td><td>Open/MC/TF</td></tr><tr><td>OlympiadBench(He et al., 2024)</td><td>ZH/En</td><td>X</td><td>Math/Physics</td><td></td><td>6.5K(Math Part)</td><td>V(5 Types)</td><td>Internet</td><td>Open</td></tr><tr><td>MathVerse(Zhang et al., 2024e)</td><td>ZH/En</td><td></td><td>Math</td><td></td><td>2.6K/15K</td><td>V(3 Types)</td><td>Synthesized</td><td>MC</td></tr><tr><td>MATH-Vision(Wang et al., 2024)</td><td>En</td><td>X</td><td>Math</td><td></td><td>3K</td><td>V(16 Types)+IC</td><td>Synthesized</td><td>Open/MC</td></tr><tr><td>MM-MATH(Sun et al., 2024)</td><td>ZH/En</td><td></td><td>Math</td><td></td><td>5.9k</td><td>V(5 Types)</td><td>Internet</td><td>Open</td></tr><tr><td>MathBench(Liu et al., 2024b)</td><td>ZH/En</td><td></td><td>Math</td><td>60</td><td>3.7k</td><td>Text-only</td><td>Internet</td><td>Open</td></tr><tr><td>MathScape(Zhou et al., 2024a)</td><td>ZH</td><td></td><td>K12 Math</td><td>107</td><td>1.3K</td><td>V</td><td>Photo-based Paper</td><td>Open</td></tr><tr><td>CMMaTH</td><td>ZH</td><td></td><td>K12 Math</td><td>784</td><td>23K</td><td>V(14 Types), OC, IC</td><td>Internet&amp;Annotated</td><td>Open/MC</td></tr></table>

Table 1: Comparison with other multimodal benchmarks. V: visual input, OC: optical characters caption, T: Question Text, IC: Image Caption, Open: open questions, MC: multiple choice questions, TF: true or false questions.

Based on the above considerations, we propose a new multimodal mathematical benchmark CM-MaTH. Compared to previous benchmarks, our benchmark demonstrates greater diversity and increased depth of reasoning in the Chinese multimodal math context. It also includes finer-grained knowledge annotation to grasp different levels and types of K12 math knowledge. We provide an open-source lightweight answer comparator called GradeGPT, designed to compare the consistency between outputs from different LLM/LMMs and standard answers, thus avoiding expensive evaluation costs. Leveraging the CMMaTH dataset and GradeGPT tool, we evaluate mainstream opensource and commercial LMMs in Table 3, reporting comprehensive evaluation results along with various and extensive case analyses, and knowledge skill analyses. In summary, our paper makes the following contributions:

• We develop a high-quality multimodal mathematics benchmark specifically tailored for the Chinese language context, featuring detailed knowledge point annotations, extensive quantity, and diversity. It serves as a reference for evaluating the multimodal mathematical reasoning capabilities of foundational models within Chinese language contexts. We also provide an English version of this dataset.

• Compared to previous multimodal mathematical benchmarks, our dataset exhibits a great depth of reasoning and diversity. Our benchmark simulates more realistic educational Q&A scenarios, encompassing a wider variety of question types and answer formats. Additionally, we provide each question with detailed multimodal knowledge points to evaluate the mastery level of current large models. The CMMaTH dataset is dynamically maintained and will be periodically updated.

• We build an evaluation assistant named GradeGPT on the CMMaTH dataset, which anables for comparing the proximity of model responses to standard answers and assessing the correctness of results. GradeGPT features lightweight open-source characteristics, avoiding the instability and high costs associated with commercial models.

• We conduct a systematic evaluation of existing mainstream LLM/LMMs, quantitatively and qualitatively comparing with existing models.

## 2 Related Work

## 2.1 Assessment of mathematical abilities

To evaluate the performance of LLM/LMMs in mathematical reasoning and examine hallucinations during the reasoning process, numerous benchmarks (Liu et al., 2023b; Sun et al., 2024; Yue et al., 2023; Yu et al., 2023; Huang et al., 2024b; Bi et al., 2024c) have been proposed for evaluating the mathematical reasoning capabilities of large models. GSM8K (Cobbe et al., 2021) is the first and most widely used mathematical dataset used for large model math evaluation, consisting of 1k math word problem test samples and corresponding answers. The MATH (Hendrycks et al., 2021b) dataset, in comparison to GSM8K, presents a greater challenge in terms of reasoning difficulty. This dataset demands a more profound understanding and intuition in various mathematical domains such as Algebra, Number Theory, and Geometry. MathVista (Lu et al., 2023) is the first dataset used to evaluate the multimodal mathematical capabili ties of large models, but it has relatively simple rea soning depth. MATH-VISION (Wang et al., 2024) has richer visual elements and deeper reasoning difficulty. MathVerse (Zhang et al., 2024e) constructs several subsets of datasets to assess whether existing multimodal large models can truly understand mathematical abstract forms. MathBench (Liu et al., 2024b) attempts to assess the level of mastery of specific mathematical skills in existing large models, but this work focuses solely on the pure text domain and annotates a relatively coarse of knowledge points. MM-MATH (Sun et al., 2024) is similar to our work but primarily focuses on English scenarios. The CMMaTH Benchmark, in comparison to existing works on the evaluation of mathematical proficiency, places a greater emphasis on the analysis of mathematical abilities within the context of the Chinese language. The data distribution of the CMMaTH dataset more closely aligns with the actual distribution found in K12 educational settings, and it provides detailed annotations of mathematical knowledge points to facilitate the assessment of models’ mastery of knowledge and skills.

## 2.2 Large Model Evaluation Tool

Due to their strong generalization capabilities and extensive world knowledge, large language models have achieved outstanding results in tasks such as machine translation (Zhu et al., 2023), question answering (Kamalloo et al., 2023), dialogue (Duan et al., 2023), and so on by generating text. Evaluating the comprehensive abilities of large models, such as clarity, adherence to instructions, comprehensiveness, formality, and mathematical reasoning ability, has received widespread attention (Ke et al., 2023; Mei et al., 2024c,b; Zhou et al., 2024b). Currently, many works opt to use powerful commercial model APIs, such as GPT-4, to assist in evaluating the comprehensive abilities of large models. For instance, in the field of geometric problem solving(Ning et al., 2023; Li et al., 2023c; Zhang et al., 2024d) and multimodal reasoning, MathVista (Lu et al., 2023) and GeoEval (Zhang et al., 2024c) use GPT-4’s API to extract correct answers for evaluation. These methods face several challenges: they are costly and time-consuming, and they struggle to keep up with rapid model iterations. Besides, these methods face challenges in terms of consistency and reproducibility (Wang et al., 2023a; Ke et al., 2023).

Recent methods have proposed using metrics such as BERT score (Zhang et al., 2020) or MAUVE (Pillutla et al., 2021) for evaluation. However, the numerical indicators produced by these methods are difficult to interpret when it comes to the erroneous responses generated by LLM. PandaLM and CritiqueLLM (Wang et al., 2023c; Ke et al., 2023) are similar to our work. They proposed a fine-tuning method based on open-source LLMs, distilling the evaluation capabilities of GPT-3.5 into a series of smaller open-source models. However, they are focused on the automated evaluation of more general text generation tasks, while we are targeting the automated evaluation of responses from large models for multimodal mathe-

<table><tr><td>Statistic</td><td>Number</td></tr><tr><td>Total questions</td><td>23856</td></tr><tr><td>- Choice-mode questions - Free-form questions</td><td>18191 5665</td></tr><tr><td>- Questions in the testmini set Choice-mode questions</td><td>1371 18191(76.2%)</td></tr><tr><td>- Single-choice questions - Multiple-choice &amp; Composite questions</td><td>13706(57.4%) 4485(18.8%)</td></tr><tr><td>Knowledge Point Number</td><td>784</td></tr><tr><td>Visual Subjects</td><td>13</td></tr><tr><td>Maximum question length</td><td></td></tr><tr><td>Minimum question length</td><td>593 6</td></tr><tr><td>Average question length</td><td>75.1</td></tr><tr><td>Grade Distribution Elementary(1-6)</td><td></td></tr><tr><td>Junior(7-9)</td><td>800</td></tr><tr><td>Senior(10-12)</td><td>5082 17972</td></tr></table>

Table 2: Key statistics of CMMaTH. The unit of question length is words. For more information can refer to Appendix E on the definitions of "Question Difficulty Levels," "Visual Subjects," and "Knowledge Point Number" and "Composite questions"

## matics problems.

Unlike PandaLM (Wang et al., 2023c) trying to evaluate the relative conciseness, clarity, our evaluation model, GradeGPT, is a dataset-oriented answer comparator that can provide specific reasons based on the standard answer and a model’s response. We distilled the answer comparison capability of GPT-4 using the Cross-Lingual Judge-of-Chain method and enhanced GradeGPT’s answer discrimination ability.

## 3 CMMaTH Dataset

## 3.1 Overview of CMMaTH

We selected diverse multimodal mathematical problems from a vast pool of K12 educational questions, comprising 23,856 items across 14 visual themes and encompassing 784 types of knowledge points. More detailed statistical data can be found in Table 2 and Appendix E.

## 3.2 Collection Guidelines

We collected a large number of multimodal mathematics questions from open-source websites, which host a vast collection of K12 math problems. The quality and distribution of the data were guided by the following criteria during collection.

• Diverse Mathematical Visual Elements. We have collected solutions to multimodal mathematical problems that rely on understanding image content, especially those containing a large amount of Chinese visual content such as text and symbols. Table 7 shows some visual elements subject of CMMaTH.

• High relevance to the K12 math knowledge and skill. The collector, being knowledgeable in the field, must ensure that each multimodal question targets a specific K12 math concept during the collection process. The dataset primarily includes K12-level math questions, enabling the evaluation of large-scale multimodal models’ potential in mathematics education.

• High-quality images and answers. During the collection phase, we instruct collectors to disregard multimodal math questions with erroneous symbols or low-quality images (blurry images). Collectors are required to ensure that the collected questions are generally solvable.

## 3.3 Data Collections

Collection from Diverse Multimodal Math Sources CMMaTH’s is mainly based on “Jiaoyan Yun”<sup>1</sup> and “Zujuan”<sup>2</sup>. These two websites have collected a large number of real K12 education questions with test paper source annotations, grade levels and coarse-grained knowledge point annotation. We purchased the copyright for Jiaoyan Yun’s data and worked with the official website to ensure that it can be open sourced. After preliminary data collection, we compiled about 200,000 preliminary data for each grade. For more data cleaning details, refer to Appendix E.

Data Filtering We excluded all questions without images in the question stems, including those non-Chinese language questions, and those solvable without visual content. Due to OCR processing may result in inaccuracies, To ensure the quality of both images and text-based questions, we removed any images with a width or height less than 100 pixels and employed the GPT-4 API to evaluate data quality. For more details on data cleaning, refer to Appendix E.3.

Data Labeling We have adopted the current commercial knowledge graph for mathematics education, Jiaoyan Cloud, which has been validated by a large number of users and teachers. We first crawled the knowledge graph involved in "Jiaoyan Cloud", which contains 5531 knowledge points.

![](images/d46dee3238850b2a52a829d3b78d4b88e8223b66eafca76f30636a1a4943f35a.jpg)  
Figure 2: Part of the knowledge points involved in the CMMaTH dataset.

Then, we filtered out the multimodal knowledge points that were not involved, and obtained 784 knowledge points. Although the mathematical problems in "Zujuan" do not have knowledge point annotations that can be directly crawled and are not organized in the way of "Jiaoyan Cloud", we have conducted fine-grained annotation of knowledge points on the problem data from the "Zujuan" section. The math problem is first solved through GPT-4. The GPT-4 assisted annotation manual annotation details can be referred to Appendix E.4.

## 3.4 Comparison with Existing Benchmarks

The CMMaTH dataset is primarily used to evaluate multimodal math reasoning capabilities in K12 Chinese educational scenarios. We compared the current mainstream multimodal mathematical datasets and large model benchmarks in Table 1. Compared to existing multimodal benchmarks and multimodal reasoning benchmarks, the CMMaTH dataset has the following characteristics:

Tailored for real Chinese K12 Multimodal scenarios MathVista features a substantial number of problems that are associated with natural and synthetic images. However, these images do not accurately represent the genuine data distribution encountered in K12 mathematics educational settings. OlympiadBench is a bilingual multimodal benchmark at the Olympiad level, but it is too challenging and doesn’t align well with real K12 multimodal math scenarios. Additionally, the variety of multimodal visual elements is relatively limited. Compared to comprehensive datasets like MMMU and CMMMU, CMMaTH is specifically designed for Chinese subjects and exhibits significant diversity in multimodal Chinese math problems. Instead, We collect multimodal mathematical question data from real Chinese exam aggregation websites, specifically tailored to the Chinese K12 educational context. As illustrated in Figure 7, the questions in the CMMaTH dataset require comprehensive understanding multimodal elements, including Chinese text and math symbols, as well as mastery of K12 knowledge points.

High-quality Fine-grained Annotation and Evaluation Tool Every question in our dataset is meticulously annotated with standardized answers, solutions expressed in natural language, associated multimodal knowledge points, visual element categories, and K12 grade levels. This fine-grained annotation enables a more nuanced evaluation of multimodal mathematical proficiency within the K12 educational context. Despite the fact that numerous benchmarks, such as MathVista and GeoEval, rely on GPT-4 for answer extraction and validation, we introduce an open-source model named GradeGPT. GradeGPT stands out by providing a stable, costfree, and swift accuracy evaluation specifically tailored for the CMMaTH dataset. Also，this approach makes it easier to dynamically maintain our dataset.

Extreme Diversity Currently, high-quality Chinese multimodal mathematics datasets are scarce. MATH-VISION lacks Chinese content, MATH-VISTA has only a few Chinese samples, and CM-MMU includes just 540 math problems, which are not detailed enough. We have included about 23k fine-grained multimodal mathematics assessment samples, covering 14 K12 mathematics visual categories, making it the largest known multimodal Chinese dataset to date. The CMMaTH dataset also includes a diverse range of question types, featuring many multi-choice questions, free-form answer questions, and "Composition Questions," as illustrated in Table 2 and Figure 8, which are commonly found in real-world Chinese mathematics education.

![](images/5d844f9d20edc9e853f18d0f360028a92895d59bb0bd1fe337abd59e20d30395.jpg)  
Figure 3: Instruction Construction Pipeline of GradeGPT.

![](images/6b508e0cdf85a91ba80aa8aa132cc27cd385082eeaa2fd0c2ce57339f3fa011c.jpg)  
Figure 4: The results of mainstream multimodal large models and pure text large models on the CMMaTH dataset. Left: represents the performance evaluation of selected LMMs and LLMs across various Visual Subjects. Right: the performance assessment of these models on different educational grade-level questions.

## 4 GradeGPT

The CMMaTH dataset encompasses a large variety of problem-solving objectives, such as mathematical expressions, multiple-choice options, numerical outcomes, coordinate points, conclusion figures, and correctness assessments. Traditionally, in reasoning or evaluation contexts, problems have been formulated as multiple-choice or true/false questions to facilitate comparison and to simplify the extraction of results. Also, it is difficult to maintain dynamically updated benchmark. Employing API models for evaluation is prohibitively expensive, and the resulting evaluations are not consistently stable, which also hampers the iterative development of models on benchmarks, such as hyperparameter selection.

To provide a stable, free, fast, and easy-toupdate model response evaluation tool, we introduce GradeGPT, an answer comparison model tailored for the CMMaTH dataset. GradeGPT is designed to receive a question, its standard answers, and a model-generated response. It extracts key steps including results from Chinese output. Determine whether the result is consistent with the standard answer. Our GradeGPT is a streamlined, open-source model. When integrated with frameworks such as vLLM using the 14B model, it can swiftly compare a myriad of model-generated answers, accomplishing a remarkable judgment accuracy of 96.1% for assessing responses comparable with GPT-4 API.

## Prompt Format

In the prompt input of GradeGPT, there are "questions," "reference answers," and "model output answers." The model is required to provide an answer in the form of "<Yes>" or "<No>" indicating whether the model output answer is equivalent to the standard reference answer. We have designed an instruction format named Cross-Lingual-Judgeof-Chain(CL-JoC) for the purpose of determining answer consistency. CL-JoC first analyzes the model response and finds the key sentences that give the answer in the model response, understand key chinese sentences in English. Then analyze the standard answer, determine the type of the standard answer, and then determine whether the standard answer is included in the model response. More details can be found in Appendix G.

<table><tr><td>Model</td><td>Overall</td><td>Flow</td><td>Bar</td><td>Scatter</td><td>Line Plot Fan</td><td>LiDAR</td><td>Visual-Table</td><td>Three View</td><td>Folded Image</td><td>Analytic</td><td>Solid</td><td>Plane Venn</td><td>Abt-Analogy</td></tr><tr><td colspan="10">LLMs(Text-only, Zeroshot)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Baichuan-13B(Yang et al., 2023)</td><td>8.4 6.7</td><td></td><td>4.8</td><td>12.2 12.4</td><td>13.1</td><td>16.2</td><td>5.4</td><td>4.1</td><td>8.5</td><td>11.1</td><td>6.7</td><td>13.7</td><td>12.8</td><td>9.3</td></tr><tr><td>Qwen-14B(Bai et al., 2023)</td><td>13.7</td><td>15.5</td><td>7.3</td><td>14.3</td><td>13.6</td><td>10.8</td><td>11.4</td><td>12.8</td><td>14.8</td><td>15.9</td><td>12.7</td><td>17.8</td><td>20.4</td><td>19.3</td></tr><tr><td>LLama2-70B(Touvron et al., 2023)</td><td>4.5 4.7</td><td></td><td>2.5</td><td>4.4</td><td>7.4</td><td>8.1</td><td>3.4</td><td>5.4</td><td>5.1</td><td>5.3</td><td>4.1</td><td>5.3</td><td>5.9</td><td>4.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Math LLMs(Text Input, Zeroshot)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MetaMath-70B(Yu et al., 2024) 5.7 8.7</td><td>4.6</td><td></td><td>3.3</td><td>6.6</td><td>5.7</td><td>0.2</td><td>4.2</td><td>4.1</td><td>8.5</td><td>7.2</td><td>4.8</td><td>8.5</td><td>9.8</td><td>5.4</td></tr><tr><td>DeepSeek-Math-7B(Shao et al., 2024) InternLM2-Math-20B(Ying et al., 2024)</td><td>14.0</td><td>13.4</td><td>6.7</td><td>14.7</td><td>13.1 12.5</td><td>12.2</td><td>8.1</td><td>13.5</td><td>12.3</td><td>17.2</td><td>16.5</td><td>21.6</td><td>19.5</td><td>13.8</td></tr><tr><td>MAmmoTH2-8x7B(Yue et al., 2024)</td><td>6.2 16.1</td><td>4.9</td><td>3.4</td><td>6.6</td><td>9.5 5.7</td><td>1.0</td><td>4.0</td><td>3.9 14.4</td><td>8.8 12.8</td><td>8.5 18.0</td><td>5.3</td><td>9.1</td><td>9.7</td><td>6.0</td></tr><tr><td></td><td></td><td>14.1</td><td>8.0</td><td>15.3</td><td>13.4 13.2</td><td>12.7</td><td>9.4</td><td></td><td></td><td></td><td>17.3</td><td>21.6</td><td>20.3</td><td>14.5</td></tr><tr><td colspan="10">Open-source LMMs (Text + Image, Zeroshot) 6.2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-v1.5-7B(Liu et al., 2023a)</td><td>5.5</td><td>1.5</td><td>4.2 2.1</td><td>5.4</td><td>5.4</td><td>3.6</td><td>4.0</td><td>4.2</td><td>5.3</td><td>4.8</td><td>3.9</td><td>8.4</td><td>6.1</td><td>4.2</td></tr><tr><td>LLaVA-NEXT-8B(Liu et al., 2024a)</td><td>5.4 13.7</td><td>7.0 12.8</td><td>5.2</td><td>6.8 11.4 10.9</td><td>1.7</td><td>8.1</td><td>4.4</td><td>6.7</td><td>4.2</td><td>6.2</td><td>4.2</td><td>7.7</td><td>7.5 19.3</td><td>6.6</td></tr><tr><td>LLaVA-OneVision-7B(Li et al., 2024)</td><td>8.3</td><td>7.1</td><td>4.6</td><td>10.2 14.6</td><td>9.7 8.5</td><td>8.1</td><td>8.1</td><td>13.8 5.9</td><td>16.5</td><td>16.8</td><td>13.6</td><td>25.2</td><td>11.3</td><td>13.6</td></tr><tr><td>Yi-VL-34B(Young et al., 2024) CogVLM-18B-Chat(Wang et al., 2023b)</td><td>9.4</td><td>10.6</td><td>4.6</td><td>9.5 12.0</td><td>7.5</td><td>6.8 8.4</td><td>7.7</td><td>10.2</td><td>6.4 9.7</td><td>10.1 12.1</td><td>7.8</td><td>12.2</td><td>19.0</td><td>7.9 10.8</td></tr><tr><td>Qwen2-VL-7B(team, 2024)</td><td>10.0</td><td>10.3</td><td>3.5</td><td>10.4</td><td>13.6</td><td>1.4</td><td>8.1</td><td>10.2</td><td>11.0</td><td>10.6</td><td>9.2 6.8</td><td>10.2 20.0</td><td>13.0</td><td>11.8</td></tr><tr><td>CogAgent-18B-Chat(Hong et al., 2023)</td><td>10.6</td><td>12.2</td><td>5.2</td><td>10.8</td><td>8.0</td><td></td><td>7.5</td><td>11.2</td><td>10.2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>InternLM-XComposer2-VL(Dong et al., 2024)</td><td>3.4</td><td>3.3</td><td>5.3</td><td>3.2</td><td>11.3</td><td>9.5</td><td>8.8</td><td>4.0</td><td>0.5</td><td>13.2 0.4</td><td>10.5</td><td>11.8</td><td>19.9</td><td>12.2</td></tr><tr><td>InternVL2-8B(Chen et al., 2024b)</td><td>23.9</td><td>44.6</td><td>11.0</td><td>22.4</td><td>29.0</td><td>6.2 14.9</td><td>5.4 17.4</td><td>22.3</td><td>14.4</td><td></td><td>3.6 19.4</td><td>1.5 30.5</td><td>1.8 25.1</td><td>3.6 19.7</td></tr><tr><td colspan="10">22.6 Closed-source LMMs (Text + Image, Zeroshot)</td><td>25.0</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>21.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT4V(OpenAI, 2023)</td><td>27.0 35.7</td><td>39.3 58.9</td><td>12.5 21.1</td><td>30.2 47.1</td><td>22.9 31.2</td><td>38.6 32.4</td><td>16.9 27.4</td><td>18.3 24.7</td><td>20.0 20.3</td><td>37.5 37.3</td><td>15.8 29.5</td><td>21.5 42.2</td><td>58.0 53.8</td><td>29.9 31.5</td></tr><tr><td>Gemini-Pro(Anil et al., 2023) Claude-3.5(Anthropic, 2024)</td><td>37.4</td><td>63.4</td><td>20.9</td><td>56.6</td><td>50.6 44.3 60.2</td><td>35.1</td><td>31.7</td><td>30.8</td><td>21.6</td><td>37.6</td><td>29.1</td><td>37.7</td><td>59.6</td><td>38.8</td></tr><tr><td>GPT4o(OpenAI, 2024)</td><td>47.8</td><td>59.1</td><td>45.5</td><td>70.3</td><td>50.0</td><td>18.9</td><td>48.0</td><td>32.2</td><td>31.8</td><td>49.1</td><td>43.2</td><td>47.4</td><td>61.2</td><td>45.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>37.3</td><td>Heuristics baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">12.4</td><td></td><td></td><td></td><td></td><td></td><td>14.3</td></tr><tr><td>Random Guess Frequent Guess</td><td>14.4 15.1</td><td>13.3</td><td>7.9 7.6</td><td>15.4 16.0</td><td>13.5 13.8</td><td>12.8 13.5</td><td>7.9 9.2</td><td>13.2 15.2</td><td>12.3 13.8</td><td>17.2 17.8</td><td>16.4 17.3</td><td>21.8 22.6</td><td>19.6 20.1</td><td></td></tr><tr><td></td><td>14.6</td><td></td><td></td><td></td><td>14.5</td><td>Human Performance</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>14.3</td></tr><tr><td colspan="10"></td><td></td><td></td><td>51.6</td><td>72.1</td><td>89.1</td><td>83.1</td></tr><tr><td></td><td>80.1</td><td>73.7</td><td>78.9</td><td>96.2</td><td>95.1 57.4</td><td>91.7</td><td>83.5</td><td>69.2</td><td>63.2</td><td>67.5</td><td></td><td></td><td></td><td></td></tr></table>

Table 3: Comparison of model performances across various mathematical visual subjects. Visual subjects: Flow: Flow Chart, Bar: Bar Chart, Scatter: Scatter Chart, Line Plot: Line Curve and Plot, Fan: Fan Chart, LiDAR: LiDAR Chart, Visual-Table: Visual-Table Chart, Three View: Three View Graph, Folded Image: Folded Image Graph, Analytic: Analytic Geometry Problem, Solid: Solid Geometry Problem, Plane: Plane Geometry Problem, SolG: Venn: Set Venn Graph, Abt-Analogy: Abstract Analogy Graph. The first and second highest accuracy of LMMs are marked in red and blue, respectively.

## Instruction Construction

We first generate inference results on CMMaTH using multiple Multimodal LLMs and provide GPT-4 with a detailed few-shot prompt to synthesize answer judgments in the form of a Cross-Lingual Judge-of-Chain response. By employing GPT-4’s In-Context Learning, as shown in Figure 3, we have established a procedure for synthesizing instruction data and have produced approximately 56k crosslingual result judge instruction pairs. Through fine-tuning the model with these instructions, we obtained an expert model, GradeGPT, which possesses the capability to compare answers.

## 5 Experiments

We conducted a comprehensive series of experiments to evaluate various models on the CM-MaTH dataset. Specifically, we assessed multiple LLM/LMMs, including 15 open-source models and 4 API-based closed-source models. We also evaluated the performance variations of different models under conditions augmented with auxiliary information, such as OCR Caption result. Additionally, we investigated the effectiveness of cross-lingual reasoning techniques in enhancing the multimodal mathematical capabilities of LMMs in the Chinese multimodal context. Through systematic experimental design and data analysis, our objective was to elucidate the strengths and weaknesses of these models in handling complex Chinese multimodal contexts. Further details on experiments related to the dependency on visual elements within the CMMaTH dataset, as well as the implementation of contextual learning capabilities in LMMs and evaluation details can be found in Appendix 6.

We also conducted a detailed analysis and evaluation of GPT4o on a random miniset CMMaTH, categorizing errors into four types: Perceptual Errors, Reasoning errors, Calculation errors, and Reject Errors. The error type distribution of GPT-4o on CMMaTH is shown in Figure 6. A more detailed definition of hallucination types can be found in

![](images/3413be87e2eeeb9e7ac8b95da0c179b5fdd790a36a98528c0403179e79c7f038.jpg)  
Figure 5: The metrics of different LMM/LLM models about KSSR.

## Appendix F.

## 5.1 Main Experiments on LLM/LMMs

We evaluated the results of mainstream multimodal large models and mathematical expert models in Table 3. We analyzed the trend of existing large models in descending with problems and conditions, as well as the effectiveness of techniques such as Cross-Lingual Prompting in solving Chinese multimodal mathematical problems. The experimental results indicate that our data exhibits extremely strong diversity and relatively challenging reasoning depth. Figure 4 and Table 3 show models such as GPT-4o struggle to comprehend our multimodal content and reasoning questions effectively, resulting in significant performance gaps between open-source and proprietary models.

## 5.2 Knowledge Skill Analysis

Quantitative Analysis We have formulated a Knowledge Successful Solve Rate(KSSR) as a structural metric to gauge the proficiency level of multi-modal extensive models in mastering knowledge points. $N _ { k n }$ is the total number of knowledge points of CMMaTH. $A c c _ { k n _ { i } }$ is the proportion of correct answers to questions labeled as i’th knowledge point. I denotes an indicator function.

$$
K S S R @ \alpha = \frac { \sum _ { i = 1 } ^ { N _ { k n } } I ( A c c _ { k n _ { i } } > \alpha ) } { N _ { k n } }\tag{1}
$$

We contend that a knowledge point can be deemed comprehensively understood only when the accuracy rate of solving problems related to that knowledge point surpasses a predefined threshold, denoted as α. For our investigation, we have established α at the values of 0.1, 0.2, 0.3, and 0.6 to demarcate the levels of mastery. As shown in Figure 5, our experiment showed that when subjected to a more stringent KSSR metric standard, the most advanced models performed poorly.

Qualitative Analysis Based on the fine-grained knowledge point annotation, we conducted a detailed knowledge point skill level analysis of current LMMs. We ranked the knowledge points based on their respective problem-solving rates in Appendix B. Our analysis revealed the significant gap between commercial models and open-source models, as well as the distribution of the multimodal knowledge points mastered.

<table><tr><td>LMM</td><td>Overall-Acc</td></tr><tr><td>LLaVA-v1.5 InternLM-XComposer2-VL</td><td>5.5 3.4</td></tr><tr><td>Gemini-1.5-Pro LLaVA-v1.5 + En-CoT</td><td>37.2 9.4(+3.9)</td></tr><tr><td>InternLM-XComposer2-VL + En-CoT Gemini-1.5-Pro + En-CoT</td><td>16.9(+13.5) 41.1(+3.9)</td></tr><tr><td> $\operatorname { L L a V A - v } 1 . 5 + \operatorname { C L P }$ </td><td>12.7(+7.2)</td></tr><tr><td>InternLM-XComposer2-VL + CLP Gemini-1.5-Pro + CLP</td><td>17.1(+13.7) 43.8(+6.6)</td></tr></table>

Table 4: The performance of train-free CoT reasoning techniques on the CMMaTH dataset.  
![](images/d94efe415ebe2871890f9f652a73fd75d6035e69efe7484839858c9b5ed8240d.jpg)  
Figure 6: Distribution of Error Types in GPT-4o.

## 5.3 Experiments of Cross-language Reason Technology

We also attempted several multilingual Chain-of-Thought approaches such as En-CoT, CLP(Cross-Lingual Prompting) used by Qin et al. (2023) to observe whether multimodal mathematical problems could be enhanced through context learning techniques without training. The results indicate that multilingual CoT methods face challenges in solving, possibly due to the abundance of Chinese contextual text in the image content, which may necessitate the model to demonstrate excellent crosslingual OCR capabilities. We have included more details on the implementation of Cross-Lingual Prompting and En-CoT on the CMMaTH dataset in Table 4. The prompt implementation detail can refer to Appendix H.3.

## 6 Conclusions

We introduce CMMaTH, a detailed Chinese K12 multimodal reasoning benchmark with diverse question types, visual elements, and complex reason depth. The benchmark includes detailed knowledge points, standard solutions, and grade levels to measure the mastery of knowledge points in the K12 multimodal math skill. To evaluate large multimodal models quickly and affordably, we built GradeGPT, an open-source tool for assessing results. Extensive experimental results on CMMaTH manifest the limitations of current models.

## Limitation & Potential Impact

Our dataset CMMaTH, as a multimodal mathematics dataset aimed at the K12 education sector, can facilitate model evaluation and iteration of multimodal large models in this field, and may promote the research and development of educational artificial intelligence. GradeGPT is a result-oriented, relatively coarse reasoning response evaluator. How to construct a process evaluation model for finegrained assessment of the reasoning ability of large models remains an area to be explored in the future.

## Acknowledgement

This work has been supported by the National Key Research and Development Program Grant 2020AAA0109700, and the National Natural Science Foundation of China (NSFC) Grant U23B2029.

## References

Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Slav Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy P. Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul Ronald Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, and et al. 2023. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805.

Anthropic. 2024. Claude-3.5.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin,

Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen technical report. CoRR, abs/2309.16609.

Baolong Bi, Shenghua Liu, Lingrui Mei, Yiwei Wang, Pengliang Ji, and Xueqi Cheng. 2024a. Decoding by contrasting knowledge: Enhancing llms’ confidence on edited facts. arXiv preprint arXiv:2405.11613.

Baolong Bi, Shenghua Liu, Yiwei Wang, Lingrui Mei, Junfeng Fang, Hongcheng Gao, Shiyu Ni, and Xueqi Cheng. 2024b. Is factuality enhancement a free lunch for llms? better factuality can lead to worse contextfaithfulness. Authorea Preprints.

Baolong Bi, Shenghua Liu, Yiwei Wang, Lingrui Mei, Hongcheng Gao, Junfeng Fang, and Xueqi Cheng. 2024c. Struedit: Structured outputs enable the fast and accurate knowledge editing for large language models. arXiv preprint arXiv:2409.10132.

Baolong Bi, Shenghua Liu, Yiwei Wang, Lingrui Mei, Hongcheng Gao, Yilong Xu, and Xueqi Cheng. 2024d. Adaptive token biaser: Knowledge editing via biasing key entities. arXiv preprint arXiv:2406.12468.

Yi Chen, Jian Xu, Xu-Yao Zhang, Wen-Zhuo Liu, Yang-Yang Liu, and Cheng-Lin Liu. 2024a. Recoverable compression: A multimodal vision token recovery mechanism guided by text information. CoRR, abs/2409.01179.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. 2024b. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. CoRR, abs/2110.14168.

Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. 2024. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. Preprint, arXiv:2401.16420.

Haodong Duan, Jueqi Wei, Chonghua Wang, Hongwei Liu, Yixiao Fang, Songyang Zhang, Dahua Lin, and Kai Chen. 2023. Botchat: Evaluating llms’ capabilities of having multi-turn dialogues. CoRR, abs/2310.13650.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 6325–6334. IEEE Computer Society.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. CoRR, abs/2402.14008.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the MATH dataset. In Proceedings ofthe Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. 2023. Cogagent: A visual language model for GUI agents. CoRR, abs/2312.08914.

Yiming Huang, Xiao Liu, Yeyun Gong, Zhibin Gou, Yelong Shen, Nan Duan, and Weizhu Chen. 2024a. Key-point-driven data synthesis with its enhancement on mathematical reasoning. CoRR, abs/2403.02333.

Yiming Huang, Jianwen Luo, Yan Yu, Yitong Zhang, Fangyu Lei, Yifan Wei, Shizhu He, Lifu Huang, Xiao Liu, Jun Zhao, and Kang Liu. 2024b. DAcode: Agent data science code generation benchmark for large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 13487–13521, Miami, Florida, USA. Association for Computational Linguistics.

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. 2023. C-eval: A multi-level multi-discipline chinese evaluation suite

for foundation models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Mingyu Jin, Qinkai Yu, Jingyuan Huang, Qingcheng Zeng, Zhenting Wang, Wenyue Hua, Haiyan Zhao, Kai Mei, Yanda Meng, Kaize Ding, et al. 2024a. Exploring concept depth: How large language models acquire knowledge at different layers? arXiv preprint arXiv:2404.07066.

Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du. 2024b. The impact of reasoning step length on large language models. arXiv preprint arXiv:2401.04925.

Ehsan Kamalloo, Nouha Dziri, Charles L. A. Clarke, and Davood Rafiei. 2023. Evaluating open-domain question answering in the era of large language models. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 5591–5606. Association for Computational Linguistics.

Pei Ke, Bosi Wen, Zhuoer Feng, Xiao Liu, Xuanyu Lei, Jiale Cheng, Shengyuan Wang, Aohan Zeng, Yuxiao Dong, Hongning Wang, Jie Tang, and Minlie Huang. 2023. Critiquellm: Scaling llm-as-critic for effective and explainable evaluation of large language model generation. CoRR, abs/2311.18702.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024. Llava-onevision: Easy visual task transfer. Preprint, arXiv:2408.03326.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023a. Seed-bench: Benchmarking multimodal llms with generative comprehension. CoRR, abs/2307.16125.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023b. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 292–305. Association for Computational Linguistics.

Zhong-Zhi Li, Ming-Liang Zhang, Fei Yin, and Cheng-Lin Liu. 2023c. Lans: A layout-aware neural solver for plane geometry problem. arXiv preprint arXiv:2311.16476.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning. CoRR, abs/2310.03744.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024a. Llavanext: Improved reasoning, ocr, and world knowledge.

Hongwei Liu, Zilong Zheng, Yuxuan Qiao, Haodong Duan, Zhiwei Fei, Fengzhe Zhou, Wenwei Zhang, Songyang Zhang, Dahua Lin, and Kai Chen. 2024b. Mathbench: Evaluating the theory and application proficiency of llms with a hierarchical mathematics benchmark. arXiv preprint arXiv:2405.12209.

Wentao Liu, Qianjun Pan, Yi Zhang, Zhuo Liu, Ji Wu, Jie Zhou, Aimin Zhou, Qin Chen, Bo Jiang, and Liang He. 2024c. Cmm-math: A chinese multimodal math dataset to evaluate and enhance the mathematics reasoning of large multimodal models. Preprint, arXiv:2409.02834.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. 2023b. Mmbench: Is your multi-modal model an all-around player? CoRR, abs/2307.06281.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. 2023. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. CoRR, abs/2310.02255.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Lingrui Mei, Shenghua Liu, Yiwei Wang, Baolong Bi, and Xueqi Chen. 2024a. Slang: New concept comprehension of large language models. arXiv preprint arXiv:2401.12585.

Lingrui Mei, Shenghua Liu, Yiwei Wang, Baolong Bi, Jiayi Mao, and Xueqi Cheng. 2024b. " not aligned" is not" malicious": Being careful about hallucinations of large language models’ jailbreak. arXiv preprint arXiv:2406.11668.

Lingrui Mei, Shenghua Liu, Yiwei Wang, Baolong Bi, Ruibin Yuan, and Xueqi Cheng. 2024c. Hiddenguard: Fine-grained safe generation with specialized representation router. arXiv preprint arXiv:2410.02684.

Maizhen Ning, Qiu-Feng Wang, Kaizhu Huang, and Xiaowei Huang. 2023. A symbolic characters aware model for solving geometry problems. In Proceedings of the 31st ACM International Conference on Multimedia, pages 7767–7775.

OpenAI. 2023. Gpt-4v.

OpenAI. 2024. Hello gpt-4o.

Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaïd

Harchaoui. 2021. MAUVE: measuring the gap between neural text and human text using divergence frontiers. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 4816–4828.

Libo Qin, Qiguang Chen, Fuxuan Wei, Shijue Huang, and Wanxiang Che. 2023. Cross-lingual prompting: Improving zero-shot chain-of-thought reasoning across languages. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6- 10, 2023, pages 2695–2709. Association for Computational Linguistics.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. 2023a. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Yongxin Shi, Dezhi Peng, Wenhui Liao, Zening Lin, Xinhong Chen, Chongyu Liu, Yuyi Zhang, and Lianwen Jin. 2023b. Exploring ocr capabilities of gpt-4v(ision) : A quantitative and in-depth evaluation. Preprint, arXiv:2310.16809.

Kai Sun, Yushi Bai, Ji Qi, Lei Hou, and Juanzi Li. 2024. Mm-math: Advancing multimodal math evaluation with process evaluation and fine-grained classification. Preprint, arXiv:2404.05091.

Qwen team. 2024. Qwen2-vl.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas

Scialom. 2023. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. 2024. Measuring multimodal mathematical reasoning with math-vision dataset. Preprint, arXiv:2402.14804.

Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. 2023a. Large language models are not fair evaluators. CoRR, abs/2305.17926.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. 2023b. Cogvlm: Visual expert for pretrained language models. CoRR, abs/2311.03079.

Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, Wei Ye, Shikun Zhang, and Yue Zhang. 2023c. Pandalm: An automatic evaluation benchmark for LLM instruction tuning optimization. CoRR, abs/2306.05087.

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai, Guosheng Dong, Haizhou Zhao, Hang Xu, Haoze Sun, Hongda Zhang, Hui Liu, Jiaming Ji, Jian Xie, Juntao Dai, Kun Fang, Lei Su, Liang Song, Lifeng Liu, Liyun Ru, Luyao Ma, Mang Wang, Mickel Liu, MingAn Lin, Nuolan Nie, Peidong Guo, Ruiyang Sun, Tao Zhang, Tianpeng Li, Tianyu Li, Wei Cheng, Weipeng Chen, Xiangrong Zeng, Xiaochuan Wang, Xiaoxi Chen, Xin Men, Xin Yu, Xuehai Pan, Yanjun Shen, Yiding Wang, Yiyu Li, Youxin Jiang, Yuchen Gao, Yupeng Zhang, Zenan Zhou, and Zhiying Wu. 2023. Baichuan 2: Open large-scale language models. CoRR, abs/2309.10305.

Huaiyuan Ying, Shuo Zhang, Linyang Li, Zhejian Zhou, Yunfan Shao, Zhaoye Fei, Yichuan Ma, Jiawei Hong, Kuikun Liu, Ziyi Wang, Yudong Wang, Zijian Wu, Shuaibin Li, Fengzhe Zhou, Hongwei Liu, Songyang Zhang, Wenwei Zhang, Hang Yan, Xipeng Qiu, Jiayu Wang, Kai Chen, and Dahua Lin. 2024. Internlmmath: Open math large language models toward verifiable reasoning. CoRR, abs/2402.06332.

Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. 2024. Yi: Open foundation models by 01.ai. CoRR, abs/2403.04652.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo

Li, Adrian Weller, and Weiyang Liu. 2024. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. Open-Review.net.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2023. Mm-vet: Evaluating large multimodal models for integrated capabilities. CoRR, abs/2308.02490.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. CoRR, abs/2311.16502.

Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. 2024. Mammoth2: Scaling instructions from the web. CoRR, abs/2405.03548.

Duzhen Zhang, Yahan Yu, Chenxing Li, Jiahua Dong, Dan Su, Chenhui Chu, and Dong Yu. 2024a. Mmllms: Recent advances in multimodal large language models. arXiv preprint arXiv:2401.13601.

Ge Zhang, Xinrun Du, Bei Chen, Yiming Liang, Tongxu Luo, Tianyu Zheng, Kang Zhu, Yuyang Cheng, Chunpu Xu, Shuyue Guo, Haoran Zhang, Xingwei Qu, Junjie Wang, Ruibin Yuan, Yizhi Li, Zekun Wang, Yudong Liu, Yu-Hsuan Tsai, Fengji Zhang, Chenghua Lin, Wenhao Huang, Wenhu Chen, and Jie Fu. 2024b. CMMMU: A chinese massive multi-discipline multimodal understanding benchmark. CoRR, abs/2401.11944.

Jiaxin Zhang, Zhongzhi Li, Mingliang Zhang, Fei Yin, Chenglin Liu, and Yashar Moshfeghi. 2024c. Geoeval: Benchmark for evaluating llms and multimodal models on geometry problem-solving. CoRR, abs/2402.10104.

Ming-Liang Zhang, Zhong-Zhi Li, Fei Yin, Liang Lin, and Cheng-Lin Liu. 2024d. Fuse, reason and verify: Geometry problem solving with parsed clauses from diagram. arXiv preprint arXiv:2407.07327.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, and Hongsheng Li. 2024e. Mathverse: Does your multi-modal LLM truly see the diagrams in visual math problems? CoRR, abs/2403.14624.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with BERT. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Minxuan Zhou, Hao Liang, Tianpeng Li, Zhiyu Wu, Mingan Lin, Linzhuang Sun, Yaqi Zhou, Yan Zhang, Xiaoqin Huang, Yicong Chen, et al. 2024a. Mathscape: Evaluating mllms in multimodal math scenarios through a hierarchical benchmark. arXiv preprint arXiv:2408.07543.

Zihao Zhou, Shudong Liu, Maizhen Ning, Wei Liu, Jindong Wang, Derek F Wong, Xiaowei Huang, Qiufeng Wang, and Kaizhu Huang. 2024b. Is your model really a good math reasoner? evaluating mathematical reasoning with checklist. arXiv preprint arXiv:2407.08733.

Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Jiajun Chen, Lei Li, and Shujian Huang. 2023. Multilingual machine translation with large language models: Empirical results and analysis. CoRR, abs/2304.04675.

## A More Related Work

## A.1 Multimodal Large Model Evaluation

The multimodal large models face serious hallucination issues in perceiving objects and executing reason and inference (Zhang et al., 2024a; Mei et al., 2024a; Jin et al., 2024b,a; Shi et al., 2023b). How to evaluate and reduce inference hallucinations of MLLMs (Chen et al., 2024a; Bi et al., 2024a,b) has received widespread attention. To systematically evaluate the various capabilities of multimodal large models, diverse multimodal benchmarks are utilized for assessing the abilities of large models and aiding iterative development. POPE (Li et al., 2023b) is used to evaluate the accuracy of large models in identifying perceptual objects. MMMU and CMMMU (Yue et al., 2023; Zhang et al., 2024b) are comprehensive subject datasets design to assess the proficiency of large models in mastering massive multimodal multi-disciplinary knowledge. SEED-Bench designed 19,000 diverse multimodal questions spanning video and image modalities to evaluate the spatiotemporal capabilities of multimodal large models (Li et al., 2023a). MMVet (Yu et al., 2023) attempts to design datasets to evaluate the integrated capabilities of different multimodal large model systems in combining various Vision-Language skills.

## A.2 Concurrent Work Discussion

MathBench (Liu et al., 2024b) is the first attempt to introduce a fine-grained knowledge point system (Huang et al., 2024a; Bi et al., 2024d) for evaluating large models’ mathematical abilities. However, MathBench is a purely text-based mathematical benchmark and does not involve multimodal skills, such as understanding diagrams, and its annotation granularity is relatively coarse.

MathScape (Zhou et al., 2024a) and CMM-Math (Liu et al., 2024c) are concurrent works that were developed after ours, but MathScape only contains 1/10 of the data size and a coarser knowledge point system. MathScape also adopts a method that embeds problem text into images for evaluation, focusing more on end-to-end recognition and solving of problems. CMM-Math provides a dataset for training and evaluation, with around 5k test samples for assessment. In contrast, our CMMaTH dataset is larger in scale, with finer annotation granularity, and is specifically designed for evaluating multimodal mathematical capabilities in Chinese.

## B Analysis of K12 knowledge point mastery level of LMMs

In addition to KSSR, based on the fine-grained knowledge point annotation of CMMaTH, we also clustered several LMMs on the specific level of mastering K12 knowledge points. According to the accuracy of solving the questions corresponding to each knowledge point, for GPT-4o, Gemini-Pro and Claude-3.5, LLaVA-v1.5, we listed the top 20 knowledge points that these models have the best in Figure 16, Figure 17, Figure 18, Figure 19.

At the same time, we identified the K12 knowledge points that these models struggle with and visualized them in Figure 20. Our fine-grained annotation and analysis reveal the shortcomings of current LMMs in solving K12 education problems from the perspective of knowledge points.

Gemini-Pro appears to be optimized for largescale triangle-related geometry problems but lacks the balanced knowledge and skill level found in models like Claude-3.5 and GPT-4. Additionally, our analysis suggests that open-source models generally struggle with fundamental knowledge skills such as parallelism and positional relationships. We hope that our datasets and tools can promote the data collection and synthesis of existing LMMs from a perspective based on knowledge points.

## C Analysis About Visual/Auxiliary Information for LLMs Inference on CMMaTH

## C.1 The Impact of OCR Information

OCR information includes important information such as Chinese characters on the coordinate axes in mathematical abstract forms, recognized mathematical symbols, etc., and plays a key role in assisting the understanding of visual information. We also evaluated the ability of OCR information for LLM/LMMs to solve CMMaTH Chinese multimodal mathematics questions.

## C.2 The Impact of Visual Content on LMM

In real questions in the real world and academic datasets like MathVista, the text part of many questions contains descriptions of the visual part, which leads to many models that may be able to solve mathematical problems with text reasoning capabilities. In order to evaluate our dataset Regarding the degree of dependence on the visual part, we evaluated the problem-solving capabilities of relatively strong interface models, such as GPT-4o, Gemini-Pro, and Claude-3.5, when there is no visual input and only text input. Our results in Table 5 show that GPT-4o and Gemini-Pro and Claude-3.5 suffer huge performance degradation in the absence of visual input. This shows that in addition to understanding the text part, a large number of questions in our benchmark require a full understanding of the corresponding visual elements in order to solve the questions.

<table><tr><td>Model</td><td colspan="2">| Overall | | Flow</td><td>Bar</td><td>Scatter</td><td>Line Plot</td><td>Fan</td><td>LiDAR</td><td>Visual-Table</td><td>Three View</td><td>Folded Image</td><td>Analytic</td><td>Solid</td><td>Plane</td><td>Venn</td><td>Abt-Analogy</td></tr><tr><td colspan="10">LMMs(Text+Image+OCR Caption, Zeroshot)</td><td colspan="7"></td></tr><tr><td>LLaVA-v1.5-7B</td><td>5.5</td><td>1.5</td><td>4.2</td><td>5.4</td><td>6.2</td><td>5.4</td><td>3.6</td><td>4.0</td><td>4.2</td><td>5.3</td><td>4.8</td><td>3.9</td><td>8.4</td><td>6.1</td><td></td><td>4.2</td></tr><tr><td>Yi-VL-34B</td><td>8.3</td><td>7.1</td><td>4.6</td><td>10.2</td><td>14.6</td><td>8.5</td><td>6.8</td><td>7.7</td><td>5.9</td><td>6.4</td><td>10.1</td><td>7.8</td><td></td><td>12.2</td><td>11.3</td><td>7.9</td></tr><tr><td>Qwen2-VL-7B</td><td>13.7</td><td>15.5</td><td>7.3</td><td>14.3</td><td>16.9</td><td>13.6</td><td>10.8</td><td>11.4</td><td>12.8</td><td>14.8</td><td>15.9</td><td></td><td>12.7</td><td>17.8</td><td>20.4</td><td>19.3</td></tr><tr><td>LLaVA-v1.5-7B+OCR Caption</td><td>4.9</td><td>5.2</td><td>1.3</td><td>3.4</td><td>4.5</td><td>5.9</td><td>4.4</td><td>2.6</td><td>3.5</td><td>3.9</td><td>4.6</td><td>3.8</td><td></td><td>3.7</td><td>8.0 13.0</td><td>5.6</td></tr><tr><td>Yi-VL-34B + OCR Caption</td><td>10.2</td><td>8.3</td><td>5.6</td><td>11.6</td><td>14.8</td><td>9.8</td><td>7.4</td><td>9.2</td><td>5.9</td><td>6.8</td><td>10.8</td><td>8.4 14.1</td><td></td><td>13.2</td><td></td><td>9.4</td></tr><tr><td>Qwen2-VL-7B + OCR Caption</td><td>14.1</td><td>16.1</td><td>8.5</td><td>15.1</td><td>18.2</td><td>14.1</td><td>12.3</td><td>12.6</td><td>13.6</td><td>16.3</td><td>17.0</td><td></td><td></td><td>19.3</td><td>21.2</td><td>20.5</td></tr><tr><td>GPT4V</td><td>27.0</td><td>39.3</td><td>12.5</td><td>30.2</td><td>21.0</td><td>22.9</td><td>38.6</td><td>16.9</td><td>18.3</td><td>20.0</td><td></td><td>37.5</td><td>15.8</td><td>21.5</td><td>58.0</td><td>29.9</td></tr><tr><td>Gemini-Pro Claude-3.5</td><td>35.7</td><td>58.9</td><td>21.1</td><td>47.1</td><td>31.2</td><td>50.6</td><td>32.4</td><td>27.4</td><td>24.7</td><td>20.3 21.6</td><td>37.3</td><td>29.5</td><td></td><td>42.2</td><td>53.8</td><td>31.5 38.8</td></tr><tr><td>GPT40</td><td>37.4 47.8</td><td>63.4 59.1</td><td>20.9 45.5</td><td>56.6 70.3</td><td>44.3 37.3</td><td>60.2</td><td>35.1</td><td>31.7</td><td>30.8</td><td>31.8</td><td>37.6</td><td>29.1</td><td></td><td>37.7</td><td>59.6 61.2</td><td>45.6</td></tr><tr><td>Gemini-Pro + OCR Caption</td><td></td><td></td><td>50.0</td><td></td><td>21.0</td><td>50.0</td><td>18.9</td><td>48.0</td><td>32.2</td><td>19.5</td><td>49.1 36.7</td><td>43.2 35.2</td><td></td><td>47.4</td><td></td><td>33.8</td></tr><tr><td>GPT4o + OCR Caption</td><td>39.7</td><td>34.5</td><td></td><td>50.9</td><td>39.7</td><td>25.3</td><td>8.5</td><td>44.4</td><td>26.7</td><td></td><td>51.5</td><td></td><td></td><td>41.0</td><td>42.7</td><td></td></tr><tr><td>Claude-3.5 + OCR Caption</td><td>50.1 46.7</td><td>61.5 55.6</td><td>47.7 48.0</td><td>72.9 70.9</td><td>45.9</td><td>52.6 51.5</td><td>21.2 36.2</td><td>50.1 49.8</td><td>34.8 36.5</td><td>34.3 25.9</td><td>45.1</td><td>45.0 39.7</td><td></td><td>50.0</td><td>63.7</td><td>48.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>41.8</td><td>67.8</td><td>44.2</td></tr><tr><td colspan="10">LMMs(Text-only, Zeroshot)</td><td colspan="7"></td></tr><tr><td>GPT4o-w/o Visual Diagram Gemini-Pro-w/o Visual Diagram</td><td>17.9 14.8</td><td></td><td>45.0 7.0</td><td>39.2</td><td>17.4</td><td>45.8</td><td>19.5</td><td>18.6</td><td></td><td>17.7</td><td>10.1 1.4</td><td>28.4 18.2</td><td>19.8 4.9</td><td>19.1 18.8</td><td>43.2 31.0</td><td>17.5 10.0</td></tr><tr><td>Claude-3.5–w/o Visual Diagram</td><td>19.9</td><td>25.0</td><td>4.3</td><td>18.0</td><td>7.6</td><td>24.4</td><td>12.9</td><td>14.2</td><td>11.3 20.0</td><td></td><td>7.5</td><td>29.8</td><td>15.1</td><td>20.4</td><td>42.2</td><td>15.6</td></tr><tr><td></td><td></td><td>37.0</td><td>10.1</td><td>32.0</td><td>18.1</td><td>46.8</td><td>17.2</td><td>13.4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 5: Model performance of LLMs, Math LLMs, and various commercial API models on CMMaTH with auxiliary OCR caption input. The results of the advanced model removing visual information are also reported in the table. The definitions of different subjects are the same as in Table 3. The first and second highest accuracy of LMMs are marked in red and blue, respectively.

## C.3 Few-Shot Evaluation on CMMaTH Dataset

We evaluated the reasoning capabilities of several advanced API-based models under three few-shot conditions(3-shot) in Table 8. The results indicate that few-shot learning can slightly enhance the performance of current commercial-grade API models. However, there remains a significant gap in effectively addressing multimodal mathematical problems in real-world Chinese multimodal reason contexts compared to our human evaluation baseline.

## D Model Generation Details

## D.1 Model Weight Version

We have listed the parameter versions and the Hugging Face repository names of the open-source models and version for API-based model used in Table 16.

## D.2 Model Sampling Parameter

We have listed the corresponding hyperparameters used by the models in Table 15. For API models, we have indicated the corresponding release

<table><tr><td>Statistic</td><td>Number</td></tr><tr><td>Single choice&#x27;s</td><td></td></tr><tr><td>distribution of question options</td><td></td></tr><tr><td>- Proportion of answers A</td><td>2694(14.8%)</td></tr><tr><td>- Proportion of answers B</td><td>3903(21.4%)</td></tr><tr><td>- Proportion of answers C</td><td>3961(21.7%)</td></tr><tr><td>- Proportion of answers D</td><td>3148(17.5%)</td></tr><tr><td>Knowledge point number</td><td>784</td></tr><tr><td>Levels</td><td>5</td></tr><tr><td>Visual Subjects</td><td>13</td></tr><tr><td>Maximum question length</td><td>593</td></tr><tr><td>Minimum question length</td><td>6</td></tr><tr><td>Average question length</td><td>75.1</td></tr><tr><td>Grade Distribution Elementary(1-6)</td><td>800</td></tr><tr><td>Junior(7-9)</td><td>5082</td></tr><tr><td>Senior(10-12)</td><td>17972</td></tr></table>

Table 6: Key statistics of CMMaTH. The unit of question length is words.

versions. Models using vLLM for inference are annotated.

## E CMMaTH Dataset Details

## E.1 More Detailed Statistics

For readers’ reference and understanding of the characteristics of the dataset, we provide other statistical information on our CMMaTH dataset, statistics on options, number of knowledge points, question difficulty level, number of visual element types in Table 6. We have also provided representive examples in Figure 7 and new question type example "Composition Question" from the dataset in Figure 8.

![](images/7208892e240bfef802cdf087b375b2f10ae5fb3da15c0213d114a76f260f1059.jpg)  
Figure 7: Some samples related to "Folded Image Graph", "Flow chart", "Stem-and-Leaf Display" visual subject on the CMMaTH datase

## E.2 Data Source Detail

These two electronic data websites, "Jiaoyan Yun 3 and "Zujuan"<sup>4</sup>, mainly collect electronic exam paper data from primary, middle, and high schools. The electronic test paper data includes detailed information on the grade level of the exam, the exam date, the school administering the exam, and the exam scope. The Jiaoyan Cloud, containing more than 1.2 million Chinese math problems, forms a major part of our private database. Additionally, we have crawled a large number of multimodal math test questions from "Zujuan" totaling around 400k multimodal math questions. We did not crawl all the data, especially since crawling data from Jiaoyan Cloud requires obtaining the corresponding copyrights. Ultimately, the ratio of data in our private database from Jiaoyan Cloud to Zujuan is approximately 7:3.

To format all questions for use, we processed them by OCR engine like Mathpix<sup>5</sup> interface. Due to inherent errors in the OCR engine, we introduced manual checks to ensure the accuracy of parsing results and to verify whether the questions belong to multimodal math problems.

To more clearly elucidate our data collection process, we have depicted the overall pipeline of data collection in Figure 15.

## E.3 Quality Check Detail

Since we use GPT-4 as a quality check tool, we provide the prompt for GPT-4 quality check in Figure 9, GPT-4 assigned a confidence level to the data on a scale of 1 to 5. Data with a confidence level lower than 3 were filtered out. To ensure the high quality of the final data, we conducted sampling and manual verification. We performed three random samples, each consisting of 500 multimodal samples, to check the data quality and ensure the consistency of the knowledge points and data. When verifying whether problem is solvable, we use multiple closed-source interface models (GPT-4o, Gemini-Pro and Claude-3.5-Sounet) to solve each problem. For problems that cannot be solved, we perform manual checks to compare whether the reference analysis given is correct and whether the problem can be solved.

## E.4 Knowledge Point Assistant Labeling Detail

The data from Jiaoyan Yun already includes detailed knowledge point classifications. We have also provided fine-grained knowledge point annotations for the questions sourced from Zujuan. The GPT-4 prompt for knowledge point labeling includes a detailed documentation of knowledge point types. Considering GPT-4’s context length limitation and the large number of knowledge points, we assess whether a problem belongs to a specific category in batches. A GPT-4 Knowledge point classification prompt in Figure 10.

![](images/3e96359693a0f5be7e1c2bef9728b6da26914c890486236066c9d9df88c9a511.jpg)

Figure 8: Examples of Composition questions question types  
![](images/c79bfa4a6b9739a690a87e24f154158160ff2b85f0e053dea9253a2689240fe4.jpg)  
Figure 9: Prompt used for quality inspection with GPT-4.

We used GPT-4 to formalize the standard form of knowledge points as a classification problem. After meticulous prompt processing, GPT-4 achieved an accuracy rate of 92%(We validated the preliminary accuracy of knowledge point annotation using GPT on a relatively 500 test set with only a few hundred questions). But this still did not meet our requirements. Therefore, GPT-4 is used here solely to assist with annotation and speed up the process. Afterward, manual verification is performed to ensure the accuracy of the annotations.

## E.5 Visual Subjects Detail

We provide the Chinese and English explanations of the Visual Subjects involved in the topics in the dataset in Table 7.

The definition of "Visual Subject" was extracted and screened from the knowledge point names of "Jiaoyan Yun". The classification of "Visual Subject" is coarser compared to the Jiaoyan Yun system and is based solely on the types of images used in multimodal math problems. Jiaoyan Yun’s question knowledge point annotations involve the examination of specific "Visual Subjects." For example, for the "Visual-Table" subject, it involves Jiaoyan

![](images/be9e51fb58de78c2743a7654f9c1c69ce1c934ecec9b7e0dbfea1ebde56d5f83.jpg)  
Figure 10: Prompt form used to annotate knowledge points.

Yun’s knowledge point "calculation of median and mode based on statistical tables." For "Solid Geometry", it involves "volume calculation of solid geometric shapes".

## E.6 Knowledge Point Detail

The CMMaTH dataset mainly adopts the knowledge point system of "Jiaoyan Yun". "Jiaoyan Yun" is a relatively mature commercial knowledge graph widely used in the field of Chinese mathematics education. It has undergone long-term user validation in the mathematics education sector. Compared to the coarse-grained knowledge point classification structures used by MathBench and MM-Math, "Jiaoyan Yun" employs a very rich knowledge structure with a massive scale of knowledge points, which better meets the needs of real-world educational scenarios in the industry.

MathBench(Liu et al., 2024b) is another related work trying to provide knowledge point label in math evaluation benchmark. However, their knowledge system is build by "Subject Area" and "Topic", which has coarse-grained classification. We provide some comparisons of some knowledge points here and the one-to-many correspondence between them in Figure 12 and Figure 13.

We provided detailed annotations of knowledge points for our dataset and conducted preliminary clustering of these knowledge points in "Jiaoyan Yun". The distribution of knowledge points in different clusters is showed in Figure 11.

## E.7 Question Level Detail

The "Level" is a reference question difficulty provided by the Teaching and Research Cloud, which can be obtained through OCR tags during crawling. It is manually marked by teachers and corresponds to difficulty levels 1-5: "very simple", "simple", "moderate difficulty", "relatively difficult", "difficult".

## E.8 Characteristics Of Annotators

We utilized a standard team of eight people, who spent two weeks annotating the data. All annotators have a university undergraduate education and are well-versed in basic knowledge of the K12 education field. To ensure quality, each question was verified by at least two people.

## E.9 Heuristics baselines Detail

Similar to MathVista, we added two heuristic Baselines. These two heuristic strategies can only handle multiple-choice questions in the dataset. "Random Guess" selects one from the options with equal probability each time, and "Frequent guess" follows the options. The proportion in the dataset serves as a prior probability to sample an option. For the evaluation of human performance, we used a subset of the CMMaTH dataset, consisting of approximately 1,500 samples. The participants were high school students from three groups, each from a top high school. Every student in each group was required to answer the questions in the subset, and we reported the average accuracy of the answers for the three groups.

<table><tr><td>Image Type</td><td>#Num</td><td>Image Type</td><td>#Num</td><td>Image Type</td><td>#Num</td><td>Image Type</td><td>#Num</td></tr><tr><td>视觉表格 Visual-Table</td><td>1513</td><td>折叠展开图 Folded Image Graph</td><td>235</td><td>立体几何图 Solid Geometry</td><td>2054</td><td>解析几何图 Analatic Geometry</td><td>3060</td></tr><tr><td>流程图 Flow Chart</td><td>3120</td><td>条形图 Bar Chart</td><td>4924</td><td>散点图 Scatter Chart</td><td>517</td><td>平面几何图 Plane Chart</td><td>3834</td></tr><tr><td>折线图 Line Chart</td><td>846</td><td>饼状图 Fan Chart</td><td>175</td><td>雷达图 LiDAR Chart</td><td>73</td><td>抽象类比图 Abstract Analog Graph</td><td>440</td></tr><tr><td>三视图 Three View Graph</td><td>22</td><td>枝页图 Stem-and-Leaf display</td><td>23</td><td></td><td>其他 Other Image type</td><td></td><td>240</td></tr></table>

Table 7: Primary visual element types involved in the CMMaTH dataset.  
![](images/495cee8f165f1bb4c6a32957860419157ba2fb719e6e73fe1b7f83558e7f87a3.jpg)  
Figure 11: Cloud diagram of the knowledge points contained in the CMMaTH dataset.

## F Hallucinations Types Defination of Human Evaluation

In our study, we employed a detailed typology of hallucinations for human evaluation on the CMMaTH subset(about 500 examples).

## Perception Errors

Perception Error refers to the model’s erroneous interpretation and utilization of diagram content during reasoning. For example, incorrect OCR, misidentification of numerical relationships, geometric relationships, logical relationships, etc.

## Reasoning Errors

Reasoning Error are quite common during the solving process. For instance, the model may misinterpret symbols or use incorrect logic or

![](images/1d0657d0ecdf2f3e8169908a11d5aaaf37c47d5553bfb4dfb2ef7f1ae9d3c813.jpg)

Figure 12: Knowledge Point Annotation Comparisons in Topic Plane Circle.
<table><tr><td>Model</td><td>Overall</td><td>Flow</td><td>Bar</td><td>Scatter</td><td>Line Plot</td><td>Fan LiDAR</td><td>Visual-Table</td><td></td><td>Three View</td><td>Folded Image</td><td>Analytic</td><td>Solid</td><td>Plane</td><td>Venn</td><td>Abt-Analogy</td></tr><tr><td colspan="10">LMMs(Text+Image, Zeroshot)</td><td colspan="7"></td></tr><tr><td>GPT4V</td><td>27.0</td><td>39.3</td><td>12.5</td><td>30.2</td><td>21.0</td><td>22.9</td><td>38.6</td><td>16.9</td><td>18.3</td><td>20.0 20.3</td><td>37.5 37.3</td><td>15.8</td><td>21.5</td><td>58.0</td><td></td><td>29.9</td></tr><tr><td>Gemini-Pro Claude-3.5</td><td>35.7 37.4</td><td>58.9 63.4</td><td>21.1</td><td>47.1</td><td>31.2</td><td>50.6</td><td>32.4</td><td>27.4</td><td>24.7</td><td></td><td></td><td>37.6</td><td>29.5</td><td>42.2</td><td>53.8</td><td>31.5 38.8</td></tr><tr><td>GPT40</td><td>47.8</td><td>59.1</td><td>20.9 45.5</td><td>56.6 70.3</td><td>44.3</td><td>60.2</td><td>35.1 18.9</td><td>31.7 48.0</td><td>30.8</td><td>21.6</td><td>49.1</td><td></td><td>29.1 43.2</td><td>37.7 47.4</td><td>59.6 61.2</td><td>45.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>37.3</td><td>50.0</td><td></td><td></td><td>32.2</td><td>31.8</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">LMMs(Text+Image, Few shot)</td><td colspan="7"></td></tr><tr><td>Gemini-Pro(3-Shot) Claude-3.5(3-Shot)</td><td>39.3</td><td>34.9</td><td>49.7</td><td>50.5</td><td>20.8</td><td>24.9</td><td>9.0</td><td>44.1</td><td>26.6</td><td></td><td>19.7</td><td>36.5</td><td>34.6</td><td>41.0</td><td>42.5</td><td>34.0</td></tr><tr><td>GPT4o(3-Shot)</td><td>48.8 52.2</td><td>57.9 63.9</td><td>50.4</td><td>73.8</td><td>47.9</td><td>53.5 55.0</td><td>38.6</td><td>52.0</td><td>38.6 37.0</td><td>28.8 36.8</td><td>47.8</td><td></td><td>42.6</td><td>43.8</td><td>70.2</td><td>47.2 50.5</td></tr><tr><td></td><td></td><td></td><td>50.4</td><td>75.3</td><td>42.9</td><td></td><td>24.0</td><td>53.6</td><td></td><td></td><td></td><td>54.3</td><td>48.6</td><td>52.6</td><td>66.1</td><td></td></tr></table>

Table 8: Model performance of few-shot(3-shot) experiment on CMMaTH.

knowledge for inference. The frequency of Reasoning Errors reflects the model’s logical and mathematical reasoning capabilities.

## Calculation Errors

Calculation Error refers to the model performing incorrect mathematical operations, such as writing equations or solving equations incorrectly.

## Reject Errors

Reject Error refers to the model’s inability to solve a problem that is actually solvable. The frequency of such errors reflects the model’s ability to follow instructions.

## F.1 Case Study

We conducted a fine-grained manual evaluation of GPT-4V’s output on CMMaTH, with the results shown in Figure 21 to Figure 31.

## G GradeGPT details

## G.1 GradeGPT Prompt Detail

We have listed detailed Fewshot Examples using the GPT-4-generated GradeGPT model responses in Table 14. Through this table, you can observe the specific form of the Cross-Lingual-Judge-of-Chain that we have used.

## G.2 GradeGPT Performance Metric

GradeGPT’s performance evaluation metric is precision in comparison. We constructed a model that responds to a test set containing outputs from various large models (including both correct and incorrect model outputs). Each output is labeled as correct or incorrect based on its result. GradeGPT is tasked with assessing whether the model responses are correct or incorrect, and this performance evaluation metric is a binary classification metric.

![](images/9dc3177f4d5962e9b083d7ca0595698ceaa87c08bf623d995829240c8d4cf821.jpg)  
Figure 13: Knowledge Point Annotation Comparisons in Topic Solid Geometry.

## G.3 GradeGPT Training Details

We generated cross-lingual evaluation instruction pairs using the outputs from InternLM-XComposer, LLaVA-v1.5, CogAgent-18B and Yi-VL-34B. These outputs were produced using GPT-4 Fewshot. The generated evaluation instructions were filtered based on specific rules, retaining only those responses from GPT-4 that contained the fields: <Yes>/<No>. Ultimately, we constructed a cross-lingual format instruction set comprising 56k instruction pairs.

GradeGPT was trained on 8 H800, with the Qwen-14B-Chat version used as the base model. The model’s batch size was set to 16. The learning rate was set to 1e-4, and the gradient accumulation step was set to 16. It was trained for 10 epochs on a 40k bilingual Judge-of-Chain dataset. A detail example of instruction can refer to Figure 14.

## G.4 Futher More Ablation Study

We conducted experiments on a development set comprising outputs from a 0.5k model. The development set was sampled from a subset of 0.5k questions on CMMaTH. Each question was accompanied by answers provided by GPT-4V, GPT-4o, and middle school students. Each answer was manually annotated to indicate whether it was correct. We use 2 to measure the answer judgment capability of different LMMs, including Zershot LMMs and LLMs after Finetune.

Ablation On Instruction Format We conducted experiments on various instruction enhancement techniques used by GradeGPT and compared the results with GPT-4 in Table 9. The results suggest that after various instruction enhancements, the accuracy of GradeGPT in model response judgment on CMMaTH can be improved to 96.1%, significantly surpassing the accuracy of GPT-4. The proposed strategy can significantly enhance GradeGPT’s ability to judge results. It is only slightly weaker than the performance of GPT-4(Fewshot) executed with a large number of examples. Our GradeGPT, as an open-source parameter model of approximately 14B, can serve as a stable, low-cost, and efficient alternative to GPT-4.

The Baseline we compared, Qwen-7B/14B(4- Shot), GPT-4(4-Shot), Naive Outcome Finetune, Judge-of-Chain. In the Naive Outcome Finetune format of instructions, the model is required to output its results indicating whether they are correct in the form of "<Yes>"/"<No>". Judge-of-Chain also includes the understanding of results and natural language descriptions of model outputs, but does not include the part of extracting key Chinese outputs and translating them into English. Compared to having the language model directly predict the <Yes>/<No> judgment labels, directly using Chinese Judge-of-Chain to construct Judge-of-Chain improves the model’s performance in answer evaluation. However, they didn’t yield good results. The performance disparity may stem from an imbalance in the quality of Chinese and English components within some bilingual base models due to the training corpus.

Instruction Example:   
Role: System   
You are an answer evaluation assistant. Based on the following questions and reference answers, determine if the model's   
output is the same answer. When making the determination, please provide the analysis process. Note: for multiple-choice   
questions, it is only correct if the output includes all the corresponding options. Note: if the model response does not   
specify a particular option, and its correctness cannot be determined, the answer consistency should be: <No>.   
Role: User   
以下是某道题目, 答案和一条模型响应。请确定模型响应中是否给出了正确的参考答案.   
Question: 如图：若∆ABE≌∆ACD，∠A= 50<sup>∘</sup> ， ∠ C=20<sup>∘</sup>，则∠ ADC=?.   
Reference Answer: 110<sup>∘</sup>   
Model Response: 首先，我们需要了解这个问题的背景和目标。题目给出了三角形的三个角度，并要求我们找到   
另一个角度。我们可以通过三角形的内角和来解决这个问题。首先，我们计算三角形的内角和：∠A + ∠ B + ∠ C   
= 180<sup>∘</sup> 。 50<sup>∘</sup> + 20<sup>∘</sup> + ∠ ADC = 180<sup>∘</sup> 。接下来，我们需要找到另一个角度，即∠ ADC。我们可以通过将内角和减   
去已知角度来得到它: ∠ ADC = 180<sup>∘</sup> - ∠ A+ ∠ B+ ∠ C. ∠ ADC = 180<sup>∘</sup> - (50<sup>∘</sup> + 20<sup>∘</sup> )。 ∠ ADC = 180<sup>∘</sup> - 70<sup>∘</sup> , ∠ ADC   
= 110<sup>∘</sup> 。所以，三角形∆ABE与∆ACD的另一个角度是∠ ADC=110<sup>∘</sup> 。   
Judgment Result:   
Role: Assistant   
The relevant paragraph in the Model Response that provides the answer is   
所以，三角形∆ABE与∆ACD的另一个角度是∠ ADC=110<sup>∘</sup>   
Understand the question in Chinese. The Chinese model response means: Therefore, another angle of triangle ∆ ABE and   
∆ ACD is ∠ ADC= 110<sup>∘</sup>.   
Next, let’s analyze whether the correct answer has been given. The reference answer 110<sup>∘</sup> is a numerical answer, and the   
output given in the model response is 110<sup>∘</sup> . The consistency of the answers is: <Yes>.  
Figure 14: A instruction example to finetune GradeGPT.

Our findings show that fine-tuning with Cross-Lingual-Judge-of-Chain for detailed thought chain refinement significantly improves the performance of open-source models in outcome analysis tasks. Additionally, we discovered that using bilingual thought chains instead of Chinese-only thought chains for base model fine-tuning effectively enhances performance in outcome determination tasks. By using and synthesizing the instructions in the form of Cross-Lingual-Judge-of-Chain that we designed, we are able to efficiently distill the answer reviewing capabilities of GPT-4.

Ablation On Instruction Data Source The instruction data for Cross-Lingual Judge-of-Chain Prompts comes from outputs of various LLMs on CMMaTH. We conducted ablation experiments on the sources of instruction data in Table 10, which showed the impact of using different LLM models in constructing diverse and effective instruction

<table><tr><td>LLM</td><td>Accoutcome</td></tr><tr><td>Qwen-7B-Chat(4-Shot)</td><td>35.1</td></tr><tr><td>+Naive Outcome Finetune</td><td>51.5</td></tr><tr><td>+Judge-of-Chain</td><td>65.3</td></tr><tr><td>+Cross-Lingual-Judge-of-Chain</td><td>85.1</td></tr><tr><td>Qwen-14B-Chat(4-Shot)</td><td>43.7</td></tr><tr><td>GradeGPT(14B)</td><td>96.1</td></tr><tr><td>GPT-4(4-Shot)</td><td>97.2</td></tr></table>

Table 9: Ablation study on the instruction fine-tuning of GradeGPT commands
<table><tr><td>Model Response Source</td><td>AcCoutcome</td></tr><tr><td>LLaVA-v1.5 response</td><td>77.2</td></tr><tr><td>+InterLM-XComposer2-VL response</td><td>83.1</td></tr><tr><td>+CogAgent-Chat response</td><td>84.2</td></tr><tr><td>+Yi-VL-34B response</td><td>85.1</td></tr></table>

Table 10: Ablation study on the instruction fine-tuning of GradeGPT commands. All experiments were conducted using the GradeGPT (7B) version.

data.

$$
A c c _ { o u t c o m e } = \frac { I ( G r a d e G P T ( R _ { i } ) , O v e r c o m e \alpha _ { G T } ) } { N _ { r e s p o n s e } } \times 1 0 0\tag{2}
$$

## H Prompt Details

## H.1 Prompt For Step Response Generation

When evaluating hallucinations during the assessment process, we use a few-shot prompt format

![](images/bf22b588757d70c09c7bb6f7c119bb9b1e0396e9a9689328cb2b35183e2eb0c9.jpg)

Figure 15: Overall Data Collection Pipeline of CMMaTH.
<table><tr><td>Element</td><td>Prompt</td></tr><tr><td>system-prompt</td><td>你是一个数学题解题助手，你的输入是一道数学题，以及这道题目的图像，你的任 务是输出这道题目的求解思路以及答案.输出格式是一步一步的方式.</td></tr><tr><td>Example1</td><td>Question：小明有2000元，一个苹果2元，小明买了10个苹果。小明还有多少钱。 Model response: 首先计算小明买苹果的开销为2乘10等于20元. &lt;STEP1&gt;用2000元减 去所有的开销，2000-20=1980. &lt;STEP2&gt;所以最终的答案为1980元. &lt;STEP3&gt; Question：如图，在△ABC中，∠C=90°，∠B=30°，以A为圆心，任意长为半径画弧</td></tr><tr><td>Example2</td><td>分别交AB、AC于点M和N，再分别以M、N为圆心，大于MN/2的长为半径画弧，两 弧交于点P，连结AP并延长交BC于点D，则下列说法中正确的个数是（）①AD是 ∠BAC的平分线；②∠ADC=60°；③点D在AB的中垂线上；④BD=2CD. A.4B.3C.2 D.1 Model response：首先确定△ABC的几何性质。&lt;STEP1&gt;∠C=90°，∠B=30°， ∠A=120°。&lt;STEP2&gt;△ABC是一个等腰三角形。&lt;STEP3&gt;根据等腰三角形 的性质，AD是∠BAC的平分线。&lt;STEP4&gt;①AD是∠BAC的平分线，正确。 &lt;STEP5&gt;②∠ADC=60°，错误。&lt;STEP6&gt;③点D在AB的中垂线上，错误。 &lt;STEP7&gt;④BD=2CD，正确。&lt;STEP8&gt;因此，正确的个数为4。&lt;STEP9&gt;答案 为A.4。&lt;STEP10&gt;请你根据这个例子，解决下面的数学题。问题：在△ABC中， ∠C=90°，∠B=30°，以A为圆心，任意长为半径画弧分别交AB、AC于点M和N， 再分别以M、N为圆心，大于MN/2的长为半径画弧，两弧交于点P，连结AP并延 长交BC于点D，则下列说法中正确的个数是（）①AD是∠BAC的平分线；② ∠ADC=60°；③点D在AB的中垂线上；④BD=2CD. A.4 B.3 C.2 D.1 求解步骤： 首先确定△ABC的几何性质。∠C=90°，∠B=30°，∠A=120°。△ABC是一个等腰 三角形。根据等腰三角形的性质，AD是∠BAC的平分线。①AD是∠BAC的平分 线，正确。②∠ADC=60°，错误。③点D在AB的中垂线上，错误。④BD=2CD，正 确。因此，正确的个数为4。答案为A.4。</td></tr></table>

Table 11: Prompt for all model to generate step-by-step answer.

to elicit step-by-step outputs from the model as showed in Table 11.

## H.2 Prompt For GradeGPT

We also listed the prompts used by GradeGPT in Table 12.

## H.3 Prompt For Cross-Lingual Prompting and En-CoT

We have listed the specific prompts used for En-CoT in Table 13 and Cross-Lingual Prompt during actual execution in Table 14. Unlike the original

Cross-Lingual Prompt paper, for experimental simplicity, we only adopted a single-turn format. However, this suffices to illustrate the varying inferential capabilities across different languages in current LMMs.

<table><tr><td>Element</td><td>Prompt</td></tr><tr><td>system-prompt</td><td>You are an answer evaluation assistant. Based on the following questions and reference answers, determine if the model&#x27;s output is the same answer. When making the determination, please provide the analysis process in the following format. Note, for multiple-choice questions, it is only correct if the output includes all the corresponding options. (For example: If the reference answer is: A, C, and the model response is: C, the answer consistency is: No) Note, if the model response does not specify a particular option, and its correctness cannot be determined, the answer consistency should be: No.</td></tr><tr><td></td><td>Question:如图所示，AD是△ABC的中线，点E在AC上，BE交AD于点F. 当  $\begin{array} { r } { \frac { A F } { F D } = } \end{array}$  1时， AE AC 二 ModelResponse:首先，我们需要找到中线AD，然后找到点E在AC上的位置。找到 中线AD，可以通过勾股定理求解，  $A D ^ { 2 } = A D \times A C ^ { 2 }$  。论  ${ \stackrel {  } { \operatorname { x } } } A D = x , A C = y$  ，则有  $\dot { x ^ { 2 } } = x y ^ { 2 }$  。&lt;STEP1&gt; 解这个方程，我们得到  $y = { \sqrt { x } }$  。现在我们知道点E在AC上，所以E的横坐标为y， 纵坐标为x。 接下来，我们需要找到点F在BD上的位置。由于F点在BD上，我们可以设F点</td></tr></table>

Table 12: Detail prompt for GradeGPT and it’s instruction construction prompt by GPT4.
<table><tr><td>Element</td><td>Prompt</td></tr><tr><td></td><td>You are a math problem-solving assistant. Your input is a math problem and an image of the system-prompt problem. Your task is to provide the solution steps and the final answer. The output format should be step-by-step.</td></tr><tr><td>Example</td><td>Question: Xiaoming has 2000 yuan, an apple costs 2 yuan, Xiaoming bought 10 apples. How much money does Xiaoming have left? Model response: First, calculate the cost of the apples Xiaoming bought, which is 2 times 10 equals 20 yuan. &lt;STEP1&gt; Subtract the total cost from 2000 yuan, 2000-20=1980. &lt;STEP2&gt; So the final answer is 1980 yuan. &lt;STEP3&gt;</td></tr></table>

Table 13: Prompt for En-CoT in Our Cross-Lingual Experiment.
<table><tr><td>Element</td><td>Prompt</td></tr><tr><td></td><td>Please act as an expert in multi-lingual understanding in Chinese. After understanding, system-prompt you should act as an expert in arithmetic reasoning in English. Let&#x27;s resolve the task you understand above step-by-step! Finally, you should format your answer as &#x27;Answer: [num]’.</td></tr><tr><td>Example</td><td>Request:玛丽比琼小两岁，琼比杰莎大五岁。如果杰莎20岁，这三个女孩的年龄总 和是多少? Reason Result: Let&#x27;s understand the task in English step-by-step! 1. &quot;玛丽比琼小两岁&quot; means &quot;Mary is two years younger than Joan.&quot; 2. &quot;琼比杰莎大五岁&quot; means &quot;Joan is five years older than Jessie. &quot;3. &quot;如果杰莎 20 0  $\varPsi "$  means &quot;If Jessie is 20 years old. &quot; Then, Let&#x27;s solve the task step-by-step: 1. According to the first sentence, Mary is two years younger than Joan. Let&#x27;s suppose Joan&#x27;s age is J, then Mary&#x27;s age is J - 2. 2. According to the second</td></tr></table>

Table 14: Detail prompt for Cross-Lingual Prompt(CLP) Reason technology for our Experiment.

<table><tr><td>Model Name</td><td>Generation Parameters</td><td>Comments</td></tr><tr><td>MAmmoTH-70B</td><td>temperature=0.2, max_new_tokens=2048</td><td>vllm package</td></tr><tr><td>InternLM-Math</td><td>temperature=0.2, max_new_tokens=2048</td><td>vllm package</td></tr><tr><td>MetaMath-70B</td><td>temperature=0.2, max_new_tokens=2048</td><td>vllm package</td></tr><tr><td>DeepSeek-Math-7B</td><td>temperature=0.2, max_new_tokens=2048</td><td>vllm package</td></tr><tr><td>Llama-2-70B</td><td>do_sample=True, top_k=0.5, top_p=0.5, max_tokens=512</td><td>vllm package</td></tr><tr><td>Baichuan-13B</td><td>temperature=0.2, max_new_tokens=2048</td><td>vllm package</td></tr><tr><td>Qwen-14B</td><td>temperature=0.2, max_new_tokens=2048</td><td>vllm package</td></tr><tr><td>llava-7B-V1.5</td><td>temperature=0.2, max_new_tokens=2048</td><td>llava package</td></tr><tr><td>Yi-VL-34B</td><td>temperature=0.2, max_new_tokens=2048</td><td>Huggingface</td></tr><tr><td>LLaVA-NEXT-34B</td><td>temperature=0.2, max_new_tokens=2048</td><td>Huggingface</td></tr><tr><td>LLaVA-OneVision</td><td>temperature=0.2, max_new_tokens=2048</td><td>Huggingface</td></tr><tr><td>CogAgent-Chat</td><td>temperature=0.2, max_new_tokens=2048</td><td>Huggingface</td></tr><tr><td>Qwen2-VL</td><td>temperature=0.2, max_new_tokens=2048</td><td>Huggingface</td></tr><tr><td>InternVL2</td><td>temperature=0.2, max_new_tokens=2048</td><td>Huggingface</td></tr><tr><td>InterLM-XComposer2-VL</td><td>temperature=0.2, max_new_tokens=2048</td><td>Huggingface</td></tr><tr><td>CogVLM</td><td>temperature=0.2, max_new_tokens=2048</td><td>Huggingface</td></tr><tr><td>GPT-4</td><td>temperature=0.2, max_tokens=2048</td><td>version=&quot;gpt-4-1106-preview&quot;</td></tr><tr><td>Gemini</td><td>temperature=0.2, max_tokens=2048</td><td>version=&quot;gemini-1.5-Pro-2023-05-15&quot;</td></tr><tr><td>Claude</td><td>temperature=0.2, max_tokens=2048</td><td>version=&quot;claude-3.5-sonnet-2024-05-24&quot;</td></tr><tr><td>GPT-4V</td><td>temperature=0.2, max_tokens=2048</td><td>version=&quot;gpt-4-vision-2023-05-15&quot;</td></tr><tr><td>GPT-40</td><td>temperature=0.2, max_tokens=2048</td><td>version=&quot;gpt-4o-2024-05-14&quot;</td></tr></table>

Table 15: The hyperparameters for the models used in the evaluation are detailed. When the "comments" section includes the format model = "", it signifies that the model was loaded from the transformer package. The vLLM package indicates that models are implemented by the vLLM package, where more details can be found in https://github.com/vllm-project/vllm. For models other than OpenAI’s GPT, custom codes were utilized for evaluation unless specified otherwise in the comments.

<table><tr><td>Model</td><td>Name</td></tr><tr><td>Llama-2-70B Baichuan-13B Qwen-14B</td><td>meta-1lama/Llama-2-70b-hf baichuan-inc/Baichuan2-13B-Chat Qwen/Qwen-14B-Chat</td></tr><tr><td>InternLM-Math DeepSeek-Math-7B</td><td>internlm/internlm2-math-20b deepseek-ai/deepseek-math-7b-instruct</td></tr><tr><td>MetaMath-70B</td><td>meta-math/MetaMath-70B-V1.0</td></tr><tr><td>MAmmoTH2-8x7B</td><td>TIGER-Lab/MAmmoTH2-8x7B</td></tr><tr><td>Yi-VL-34B</td><td>01-ai/Yi-VL-34B</td></tr><tr><td>Qwen2-VL</td><td></td></tr><tr><td></td><td>Qwen/Qwen2-VL-7B-Instruct</td></tr><tr><td>LLaVA-v1.5</td><td>liuhaotian/llava-v1.5-13b</td></tr><tr><td>LLaVA-NEXT-8B</td><td>1lava-hf/llama3-llava-next-8b-hf</td></tr><tr><td>LLaVA-OneVision</td><td></td></tr><tr><td>InternVL2-8B</td><td>lmms-lab/llava-onevision-qwen2-7b-ov</td></tr><tr><td></td><td>OpenGVLab/InternVL2-8B</td></tr><tr><td>InterLM-XComposer2-VL</td><td>internlm/internlm-7b</td></tr><tr><td>CogVLM</td><td>THUDM/cogvlm-chat-hf</td></tr><tr><td>CogAgent</td><td>THUDM/cogagent-vqa-hf</td></tr><tr><td>Gemini</td><td></td></tr><tr><td>Claude</td><td>&lt;Gemini 1.5 Pro&gt;</td></tr><tr><td>GPT4o</td><td>&lt;Claude 3.5 Sonnet&gt; &lt;GPT4o-20240201&gt;</td></tr></table>

Table 16: LLMs used in our experiments and their corresponding names in Huggingface Hub.

Top 20 Knowledge Points by Correctness Rate in GPT-4o Model  
![](images/da8a9a702d45a49584bfa2d1f7d5e07e01ec7df267c2ce05e36a1ead9334d55a.jpg)  
Figure 16: Top 20 Knowledge Point Correctness of GPT-4o.

Top 20 Knowledge Points by Correctness Rate in Gemini Model  
![](images/270e262881a304031473a14ed412857f140b7d525b864fefdbd40f36dba17459.jpg)  
Figure 17: Top 20 Knowledge Point Correctness of Gemini-Pro.

Top 20 Knowledge Points by Correctness Rate in Claude 3.5 Model  
![](images/3d0ba32c532e426f7c1e388d513c90166499179e0010c55a02913027e1991565.jpg)  
Figure 18: Top 20 Knowledge Point Correctness of Claude-3.5.

Top 20 Knowledge Points by Correctness Rate in LLaVA-v1.5 Model  
![](images/28166ea7c7b81b9ae36383ac3b3aec194f22351146d541eb2cba5e4d13767dba.jpg)  
Figure 19: Top 20 Knowledge Point Correctness of LLaVA-v1.5.

![](images/aa2acf13f18239431033076a3693549242a1974df4509c1709481dce75a6ca39.jpg)

![](images/d5673953fa134808bf09c2be81a5200c0c864c4634cfd1cc5bdb66c2ef5b9a34.jpg)  
Figure 20: Poor Solving Performance Knowledge Point Cloud of Gemini-Pro, Claude-3.5, GPT-4o and LLaVA-v1.5.

![](images/22d4ba83ce5b7e7e170f927c5f7db5aae89b5a4b830e80afc746ed3721639eab.jpg)  
Figure 21: Case from GPT-4V/o. The red ones are marked as generated inference hallucinations.

![](images/7737363c59dea9606dfb5870c64ff4feab7a1c41d77ea281e9c020888d50fedc.jpg)  
Figure 22: Case from GPT-4V/o. The red ones are marked as generated inference hallucinations.

![](images/755bb8f0bef41866972fad5cee800a16a11b058161db63a09cc0e902ed772525.jpg)  
Figure 23: Case from GPT-4V/o. The red ones are marked as generated inference hallucinations.

![](images/70e46236d228b48055d3fbc68b5cec520389ceb0527976dc8f4f45499f19d641.jpg)  
Figure 24: Case from GPT-4V/o. The red ones are marked as generated inference hallucinations.

![](images/85d40172b21c0938b88795f6d05c8a3197c6d657b940da23058c8c1bae2abfba.jpg)  
Figure 25: Case from GPT-4V/o. The red ones are marked as generated inference hallucinations.

![](images/ba6ad3e1b6ba22338be9872d2d62c1d2a5ad5b5d7dc898f9a7eb1da3fd071c96.jpg)  
Figure 26: Case from GPT-4V/o. The red ones are marked as generated inference hallucinations.

![](images/f221892fcab3f05f4d7694bc3e5fcb486ce1ed6f80baa99dcd026538df11ad84.jpg)  
Figure 27: Case from GPT-4V/o. The red ones are marked as generated inference hallucinations.

![](images/aeff8127088d4753cd73351f2c1af19edb6704c5269c95b1f591576af8824ff2.jpg)  
Figure 28: Case from GPT-4V/o. The red ones are marked as generated inference hallucinations.

![](images/809f0dc49ba2449b43fcf8c5852c19087f981f266f2ba56015de2ff3e831a2be.jpg)  
Figure 29: Case from GPT-4V/o. The red ones are marked as generated inference hallucinations.

![](images/53f305e2aa771e12aed93837d6178c3aa2c21bd3c8a8240e34bcdb0c3848e62c.jpg)  
Figure 30: Case from GPT-4V/o. The red ones are marked as generated inference hallucinations.

![](images/daf16151d8507f0e044c815830225f6afbb47afa1df6832c9d89a23579ed8dff.jpg)  
Figure 31: Case from GPT-4V/o. The red ones are marked as generated inference hallucinations.