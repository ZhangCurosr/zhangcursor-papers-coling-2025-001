# A Career Interview Dialogue System using Large Language Model-based Dynamic Slot Generation

Ekai Hashimoto<sup>1</sup>, Mikio Nakano<sup>1</sup>\*, Takayoshi Sakurai<sup>1</sup>, Shun Shiramatsu<sup>1</sup>,

Toshitake Komazaki<sup>2</sup>, Shiho Tsuchiya<sup>3</sup>

<sup>1</sup>Nagoya Institute of Technology

<sup>2</sup>Tokyo Healthcare University

<sup>3</sup>Kitasato University Hospital

e.hashimoto.611@stn.nitech.ac.jp

## Abstract

This study aims to improve the efficiency and quality of career interviews conducted by nursing managers. To this end, we have been developing a slot-filling dialogue system that engages in pre-interviews to collect information on staff careers as a preparatory step before the actual interviews. Conventional slot-fillingbased interview dialogue systems have limitations in the flexibility of information collection because the dialogue progresses based on predefined slot sets. We therefore propose a method that leverages large language models (LLMs) to dynamically generate new slots according to the flow of the dialogue, achieving more natural conversations. Furthermore, we incorporate abduction into the slot generation process to enable more appropriate and effective slot generation. To validate the effectiveness of the proposed method, we conducted experiments using a user simulator. The results suggest that the proposed method using abduction is effective in enhancing both informationcollecting capabilities and the naturalness of the dialogue.

## 1 Introduction

In hospitals, nursing managers regularly conduct career interviews in order to support the careers of individual staff members. However, nursing managers are faced with the time burden of dealing with information that staff are confiding in them for the first time, and of eliciting problems that the staff have. Furthermore, interviews are often focused on resignation or career transitions, which imposes a psychological burden on both nursing managers and staff.

In this study, we aim to improve the efficiency and quality of career interviews and provide support to nursing managers through our dialogue system. We developed a dialogue system for preinterviews to collect information on staff careers in preparation for actual interviews and share this information as a report with nursing managers. Through the reports, nursing managers will be able to better understand staff concerns and expectations before conducting interviews and provide a more personalized experience.

As indicated by Lucas et al. (2014), people tend to disclose more information in interactions with systems compared to humans. Therefore, we can expect that staff may disclose information to the system that they find difficult to share with nursing managers. Additionally, staff will have the opportunity to reassess their careers through interactions with the system before the interview.

As the purpose of our system is to collect information from staff and share it with nursing managers, we designed it to understand the user’s speech through slot-filling and to ask questions to fill in the missing slots. For slot-filling and question generation, the system utilizes a large language model (LLM). Once 80% of the slots are filled or a predetermined number of dialogue turns have passed, the system ends the conversation and generates a report.

Conventional slot-filling-based interview dialogue systems operate using a predefined set of slots, which presents two problems. The first is the limited ability to collect information, as the system can only inquire about information related to the predefined slots. The second challenge is the unnaturalness of the dialogue, as it attempts to extract all slot information from every user.

To address these challenges, our method generates slots dynamically during the conversation using an LLM and asks questions based on the newly added slots. This is expected to enhance both the information-collecting ability of the dialogue system and the naturalness of the conversation. We also propose a method for generating more effective slots by incorporating an abductive process into the slot-generation prompts.

To validate the effectiveness of the proposed method, we conducted comparative experiments utilizing a user simulator to assess the system’s ability to collect information in individual staff situations and the naturalness of the dialogues.

## 2 Related Work

## 2.1 Slot-filling Dialogue Systems using LLMs

Slot-filling-based dialogue management, where knowledge acquired during the conversation is represented by a set of slots and utterances are generated based on that slot set, has long been utilized in dialogue systems (Bobrow et al., 1977). Recently, LLMs have increasingly been used for slotfilling (dialogue state tracking) in dialogue systems (Hudecek and Dusekˇ , 2023; Siddique et al., 2021; Coope et al., 2020) to leverage their advanced language understanding capabilities. In particular, models like GPT have achieved higher accuracy compared to previous LLM models (Sun et al., 2024; Feng et al., 2023; Heck et al., 2023).

## 2.2 Interview Dialogue Systems

Various interview dialogue systems have been developed (DeVault et al., 2014; Stent et al., 2006; Johnston et al., 2013; Kobori et al., 2016; Zeng et al., 2023; Pooja et al., 2020; Nagasawa et al., 2024; Ge et al., 2023). Particularly relevant to our study are the interview systems used in job interviews (Inoue et al., 2020; Su et al., 2018, 2019).

Inoue et al. (2020) developed a dialogue system that acts as an interviewer in job interviews to probe the candidate’s motivations. Their system asks a base question followed by two follow-up questions generated by the system. First, the system evaluates how well the candidate’s response has fulfilled the base question, and then based on the results, it presents the prepared follow-up questions.

If the candidate’s response to the follow-up question includes specific keywords, the system generates additional follow-up questions using those keywords. However, we believe that Inoue et al.’s method lacks the flexibility needed for application in our study. Job interview scenarios are relatively constrained, and predefined questions and scenarios can effectively probe. In contrast, career interviews require probing across a wide range of scenarios, including individual career goals, workplace experiences, and personal issues. Therefore, a static set of questions may be insufficient.

Our proposed method uses LLMs to generate new, highly relevant probing items (slots) and then comes up with follow-up questions to fill the empty slots.

## 2.3 Generating Slots using LLMs

Komada et al. (2024) proposed a method that utilizes dynamic slots to maintain dialogue coherence in tabletop role-playing games (TRPGs), where flexible goal-setting is required. Their method generates slots filled with information based on the dialogue history, aiming to preserve coherence within the TRPG scenario.

However, applying Komada et al.’s method to our use case presents the challenge of collecting user information. Their method focuses primarily on the role of recording the TRPG’s progress and does not aim to develop the scenario or collect new information. In contrast, career interviews require probing for information that matches the individual circumstances of each staff member. Therefore, the system must dynamically generate empty slots suitable for the context of the dialogue and explore them further.

Our proposed method collects user information by generating empty slots through abductive reasoning. We believe this will more effectively extract user information and improve the quality of the dialogue.

## 3 Proposed Methods

We adopted slot filling because it is useful for comprehensively gathering the information required to generate a report useful for nursing managers’ interviews.

As stated earlier, there are two challenges in adapting conventional slot-filling-based interview dialogue systems to career interviews. The first is the limited ability to collect information. Slotfilling dialogue systems ask questions to elicit specific information predetermined by a set of slots. If the slot sets are appropriately defined, effective information collection becomes possible, but it is unrealistic to prepare an optimal slot sets for every user.

The second challenge is the unnaturalness of the resulting dialogue. Slot-filling dialogue systems tend to ask nearly identical questions to all users based on predefined slot sets. If slots related to personal issues are included, even users who are not concerned with such issues may be asked these questions. This can result in an unnatural dialogue experience for the user.

To address these two challenges, the proposed method utilizes LLMs to generate slots that take the context of the conversation into account. Then, the system asks relevant questions based on the newly added slots. This approach is expected to improve both the information-collection ability of the dialogue system and the naturalness of the conversation.

## 3.1 Generating Slots Using LLMs

Slot generation is conducted through prompt engineering with an LLM. The prompts include basic instructions, previous slots, previous dialogue history, and the user’s recent utterance. The prompt for slot generation contains instructions on how to generate slots related to career topics as well as instructions to generate slots that are as useful as possible for report generation. The full content of all prompts is provided in the Appendix A.

## 3.2 Incorporating Abductive Process

During the initial stage of development, we observed that in the absence of clear rules for generation, the slots generated by the system tended to be divergent. As a result, there were cases where the generated slots lacked consistency, and in particular, slots unsuitable for career interviews were generated. To address this issue, we propose using an abductive approach as a rule for slot generation.

Takanashi (2024) confirmed that in dietary assessments aimed at ensuring optimal nutritional management, supervisors engage in abductive questioning when significant potential risks are reported by patrol members. Abduction is a reasoning process formalized as follows (Peirce, 1974):

## Formulation of Abduction

The surprising fact, C, is observed; But if A were true, C would be a matter of course, Hence, there is reason to suspect that A is true.

We believe Takanashi’s findings can be used to identify significant potential risks in a user’s career. For example, suppose there is a user who is considering resignation due to unhappiness with working night shifts. If their utterance indicates “an interest in management positions” the surprising fact C is “interest in management” and A could be “the desire for a management position with fewer night shifts”. A “dissatisfaction with night shifts”

slot can then be generated to explore A. By asking questions to fill this “dissatisfaction with night shifts” slot, it is expected that significant potential risks, such as resignation due to dissatisfaction with night shifts, can be identified.

In short, we aim to identify significant potential risks by allowing the LLM to engage in abductive reasoning in the prompt before generating slots. This approach is an example of a method where the model performs step-by-step reasoning to arrive at the final output for complex problems, which is known as Chain of Thought (CoT) (Wei et al., 2024). CoT methods have been shown to improve both the consistency of reasoning and the accuracy of outputs. The full content of all prompts is provided in the Appendix A.

## 4 Implementation

This section describes our career interview dialogue system that uses the slot generation method explained in the previous section. Our system works in Japanese.

The system begins by conducting a dialogue in the small talk phase, where it engages in casual conversation or discussions about concerns with the user. After this, the career interview phase starts. Once the interview phase is complete, the system generates a report based on the information collected from the user. The user can then review the generated report and select only the information they wish to share with the nursing manager. For dialogue management, we utilized the state-transition network-based dialogue management block of DialBB (Nakano and Komatani, 2024). The flowchart of the system is shown in Figure 1.

## 4.1 Self-assessment Questionnaire

In a university hospital where one of the authors works, nurses complete a self-assessment questionnaire before the interview. This sheet includes both open-ended and multiple-choice questions asking about the nurse’s preferences for the next year, training, and career development plans. The nursing managers then use this self-assessment questionnaire as the basis for the career interview. In our system, users also complete a self-assessment questionnaire before using the system, and the system uses that information during the dialogue.

The format of the self-assessment questionnaire is shown in Table 1.

![](images/063a3ea27dd7ad6cbf155c83c7dfbab350affd9460f90d36b35b25278a0f3f48.jpg)  
Figure 1: The process flow of our system. (The process for small talk is simplified.)

<table><tr><td>Question What are your plans for</td><td>Options - Nursing management</td></tr><tr><td>future career develop- ment? (Multiple choice &amp; Op- tional free-form descrip- tion)</td><td>- Generalist - Clinical nurse educator - Nurse department facult - Specialized nursing area</td></tr><tr><td>What kind of training would you like to attend? (Select one)</td><td>- In-hospital - Outside-hospital (Specific training name)</td></tr><tr><td>Your preferences for next year (Multiple choice &amp; Op- tional free-form descrip- tion)</td><td>- Continue - Transfer - Resignation - Further education</td></tr></table>

Table 1: Self-assessment questionnaire.
<table><tr><td>Slot name</td><td>Category</td></tr><tr><td>Career aspirations for next year</td><td>Career</td></tr><tr><td>Career development plan</td><td>Career, Plan</td></tr><tr><td>Future department preferences</td><td>Career, Preference</td></tr><tr><td>Career-related concerns</td><td>Career, Concerns</td></tr><tr><td>Training preferences</td><td>Training, Preference</td></tr><tr><td>Current job duties</td><td>Job</td></tr><tr><td>Job satisfaction</td><td>Job, Satisfaction</td></tr><tr><td>Job dissatisfaction</td><td>Job, Dissatisfaction</td></tr></table>

Table 2: Initial slot set.

## 4.2 Initial Slot Set

The system starts with eight slots, referred to as the initial slot set, as shown in Table 2. The initial slot set contains the minimum information required for career interviews as determined by the nursing managers at the university hospital where the experiment was conducted. Each slot has a category to provide a broad classification. Categorizing the slots allows the system to organize the user’s information by category, resulting in a report that is easier for nursing managers to understand.

## 4.3 Dialogue Management in the Interview Phase

The system includes a small talk phase before the career interview phase, where the user is encouraged to open up through casual conversation. Once career-related topics arise in the small talk, the system transitions to the interview phase. In each turn, the system performs slot-filling using an LLM based on the conversation up to that point, and the LLM generates new slots as needed. The system also generates questions using the LLM to fill the empty slots. The dialogue ends either when a predefined number of turns have passed or when the slot-filling rate exceeds 80%. After the dialogue ends, a report summarizing the slots and conversation history is generated by the LLM.

When using the LLM, the system includes information such as its persona, the content of the self-assessment questionnaire, the slots, and the dialogue history up to that point in the prompt. In all modules, the LLM model used is OpenAI’s GPT-4o (gpt-4o-2024-05-13) <sup>1</sup> with a temperature setting of 0.1.

## 4.4 Examples of Generated Slots and Questions Among Methods

Figure 2 shows the system’s next question based on the same dialogue history, the user’s recent utterance, and slots for each method. In the Baseline Method, the system asked a question related to the “future department preferences” slot, which was included in the initial slot set. In Proposed Method 1, after receiving the user’s utterance, two new slots were created: “interest in other occupations” and “occupations of interest”. The system then asked a question to fill the “interest in other occupations” slot. In Proposed Method 2, after receiving the user’s utterance, the system first conducted abduction. Afterward, three slots were generated to clarify the inferred matters, and the system asked questions to fill the “dissatisfaction with nursing career” slot.

![](images/44a93d928defa797802a234566afd7c53323e3383b2064623969d23a2e78fff0.jpg)

Figure 2: Comparison of generated slots and question among methods.
<table><tr><td>Method</td><td>Slots gen.</td><td>Abduction</td></tr><tr><td>Baseline</td><td>×</td><td>×</td></tr><tr><td>Proposed Method 1</td><td>√</td><td>X</td></tr><tr><td>Proposed Method 2</td><td>√</td><td>√</td></tr></table>

Table 3: Compared methods.

## 5 Evaluation

We evaluated and compared the following three methods: Baseline, which uses only the initially set slots without generating any new ones, Proposed Method 1, which generates slots but does not use abductive reasoning, and Proposed Method 2, which performs abductive reasoning before slot generation. The slot generation and use of abduction for each method are summarized in Table 3.

The experiment was conducted to verify whether the proposed system can naturally collect information from users through dialogue. Specifically, we implemented our system through dialogue with a user simulator based on an LLM and then analyzed the resulting dialogue examples.

Each method was evaluated based on two criteria: “ability to collect information” and “naturalness of dialogues”.

## 5.1 Evaluation Items

## 5.1.1 Ability of Collecting Information

The system’s effectiveness as a pre-interview information collection tool for career interviews can be determined by its ability to extract more information within a limited number of turns or time. Therefore, considering the persona settings of the user simulator, an average of three items to be extracted were set for each persona. We then examined whether these items were mentioned during the dialogue with the proposed system.

Ideally, the evaluation should focus on whether the proposed method can generate appropriate slots. However, determining the quality of the generated slots is challenging. Therefore, we adopted a method that directly evaluates whether the necessary information was ultimately obtained.

## 5.1.2 Naturalness of Dialogues

Even with a limited number of turns or time, the system needs to collect information from the user with a natural conversation. The content of the system’s questions is determined by the slots, which are closely related to the naturalness of the dialogue. Therefore, we evaluated how natural the dialogue conducted by each method was.

Our assessment of the naturalness of the dialogue was based on Relevance Theory. This theory evaluates the naturalness of dialogues in terms of processing effort and cognitive effects (Wilson and Sperber, 1995). To investigate the processing effort, we examined the unnaturalness of the system’s questioning intentions and the frequency of abrupt topic shifts. To investigate cognitive effects, we analyzed how much information the system’s questions were able to extract. To this end, dialogue examples generated during the experiment were randomly selected and evaluated manually using a 7-point Likert scale on the crowd-sourcing platform Lancers.<sup>2</sup>

To examine the processing effort of the system’s questions, Evaluation Items 1 and 2 in Table 4 were set for each dialogue example. A score of 7 was assigned when the system’s questions were considered very natural, and a score of 1 when they were considered very unnatural.

<table><tr><td colspan="2">Effort for processing questions:</td></tr><tr><td>1:</td><td>Did you feel that the system&#x27;s questions often had unclear intentions? (1: Strongly agree, 7: Strongly disagree)</td></tr><tr><td>2:</td><td>Did you feel that there were many abrupt topic shifts in the system&#x27;s questions? (1: Strongly agree, 7: Strongly disagree)</td></tr><tr><td>3:</td><td>Cognitive effect of questions: How effective do you think the system&#x27;s questions</td></tr><tr><td>4:</td><td>were in extracting information from the nurse? (1: Very ineffective, 7: Very effective) Do you think the system&#x27;s questions were able to extract detailed information from the nurse?</td></tr></table>

Table 4: Evaluation items used to assess the naturalness of dialogues through manual evaluation. (Originally in Japanese)

To examine the cognitive effect of the system’s questions, Evaluation Items 3 and 4 in Table 4 were set for each dialogue example. A score of 7 was assigned when the system’s questions were considered very effective at extracting information, and a score of 1 when they were considered ineffective.

## 5.2 Example Dialogues Used for Evaluation

In this experiment, it is desired to conduct subject experiments under various conditions. However, such experiments are psychologically burdensome for the participants. To confirm the proposed method’s effectiveness, we conducted a preliminary evaluation through simulations. Specifically, we generated dialogues using a user simulator in which each simulated user played a person of a nurse.

Each persona was assigned concerns and other settings, but this information was not provided to the dialogue system, which then interacted with the user simulator. The user simulator’s prompt included instructions to engage in dialogue with the AI and the persona settings. The LLM utilized for the user simulator was GPT-4o (gpt-4o-2024- 05-13).

A total of 16 personas representing nurses were prepared and self-assessment questionnaires were created for each. Each persona’s career history and future career plans were defined, followed by a review and evaluation by nursing managers. Two persona examples are shown in Appendix B.1 and C.1.

Through dialogues between the user simulator utilizing these 16 personas and the three methods, a total of 48 dialogue examples were generated. These dialogues lasted a minimum of 6 turns and a maximum of 15 turns, with an average of 9.2 turns, including the small talk phase.

<table><tr><td>Ave. number</td><td>Base</td><td>P1</td><td>P2</td></tr><tr><td>Overall ave. number collected items (upper limit = 3.1)</td><td>2.3</td><td>2.0</td><td>2.8</td></tr><tr><td>Job transfer/resignation ave. number collected items (upper limit = 3.4)</td><td>2.3</td><td>2.3</td><td>2.9</td></tr></table>

Table 5: Average number of collected items in each example dialogue for Overall and Job transfer/resignationrelated items. (Base: Baseline 1, P1: Proposed Method 1, P2: Proposed Method 2)

Examples of dialogue are shown in Appendices B.2, B.3, and B.4.

## 5.3 Results

## 5.3.1 Ability of Collecting Information

For the 16 personas, an average of 3.1 check items per persona was set, resulting in 50 check items in total. These items were objectively evaluated by determining whether they were mentioned during the dialogue with the proposed system.

The evaluation of each method’s informationcollecting ability is shown in Table 5. The method that collected the most check items was Proposed Method 2 using abductive reasoning. This result demonstrates the effectiveness of slot generation using abductive reasoning.

When limited to 27 check items related to eight personas concerned with job transfer or resignation, Proposed Method 2 collected 23 items (2.9 items on average), while the other two methods collected 18 items (2.3 items on average) each. Notably, Proposed Method 1 failed to collect check items from personas with low intent to resign.

## 5.3.2 Naturalness of Dialogues

To evaluate the naturalness of each method’s dialogue, we conducted the same experiment twice using the crowd-sourcing platform Lancers.In the first experiment, 40 workers rated the dialogues naturalness, resulting in an effect size of approximately 0.40, with a calculated valid sample size of about 90. We then collected additional 48 workers, bringing the total to 88 workers for the analysis. The dialogue examples used for evaluation were selected randomly from the 48 dialogue examples, with three dialogues (one per method, with the same persona) being evaluated. The differences in the evaluation results based on the dialogue management methods were analyzed using Friedman’s test and the Wilcoxon signed-rank sum test. The results are shown in Table 6.

After conducting Friedman’s test across the three methods, Wilcoxon signed-rank tests were performed for all pairs. Pairwise comparisons were also adjusted using the Bonferroni method for multiple comparisons.

As shown in Table 6, Proposed Method 2 achieved the highest average scores in all evaluation items. Additionally, it exhibited lower standard deviations for most evaluation items.

## 5.3.3 Performance of the Slot-generating Module

We calculated the average number of generated slots for each user’s response based on the generated dialogue examples. The results are shown in Table 7. These evaluations confirmed that the slotgeneration module’s performance, regarding the number of generated slots, meets a certain standard.

## 5.3.4 Performance of the Slot-filling Module

This evaluation is based on the assumption that the system’s language understanding performance, specifically the accuracy of slot filling, is relatively high. So, we randomly selected 66 user turns from the generated dialogue examples and calculated the F1 score. The F1 score of the slot-filling module is shown in Table 8. The performance of the slotfilling module was confirmed to meet a certain standard through these evaluations.

## 5.3.5 Performance of the User Simulator

If the user simulator makes unnatural utterances, the entire dialogue will become unnatural regardless of the system’s utterances, making this evaluation meaningless. Therefore, in addition to the experiment evaluating the naturalness of the dialogue, we added a phase to measure the performance of the user simulator. We also asked the crowd-workers who evaluated the naturalness of the example dialogues to asses the user utterances in them using a 7-point Likert scale. The questions and results utilized for this evaluation are shown in Table 9. From these results, the performance of the user simulator was confirmed to meet a certain standard.

## 6 Discussion

## 6.1 Ability of Collecting Information

As shown in Table 5, Proposed Method 2 collected the most check items. It was able to infer the reason A whenever the user made a statement suggesting they might leave their current job. To clarify A, the system generated a slot related to A and posed a corresponding question. This process enabled the system to collect more latent risks and related information (reasons and circumstances) from the users.

The reason Proposed Method 1 collected fewer items than the Baseline is likely due to its inability to control which items should be probed further. As shown in Table 5, although the score for personas with high intent to leave was the same as the Baseline, Proposed Method 1 collected five fewer items (0.3 items on average) overall. One common example where Proposed Method 1 failed to collect check items was when it focused too narrowly on a single issue (particularly current job duties). As a result of probing too deeply into a narrow topic, Proposed Method 1 was unable to collect a wider range of check items. The LLM determines which items to inquire about based on the prompt. When the conversation shifts to a potentially concerning topic, the system keeps asking about that topic. This is likely why Proposed Method 1 collected the fewest items.

## 6.2 Naturalness of Dialogues

## 6.2.1 Processing Effort of Questions

Although there was no statistically significant difference in Evaluation Item 1 in Table 4, Proposed Method 2 produced the best results. Proposed Method 2 dynamically generated questions based on abductive reasoning following the user’s responses, asking questions to test the hypothesis. Since the user may not fully understand the hypothesis or the reasoning process, this could have led to situations where the questions seemed unclear in intent. As a potential solution, presenting the reasoning process behind the hypotheses to the user might reduce the number of questions with unclear intentions.

In Evaluation Item 2, Proposed Method 2 showed statistically significant differences compared to the other methods and achieved relatively high results. As with the reasoning presented in Evaluation Item 1, Proposed Method 2 used an abductive reasoning approach for generating questions. This approach allowed for a natural flow of conversation and reduced the occurrence of sudden topic shifts. However, Proposed Method 2 did not achieve sufficiently high results. Proposed Method 2 generated different hypotheses in response to the user’s responses, which could lead to a variety of topics being explored. As a result, users may have encountered unexpected questions, leading evaluators to feel that the system was abruptly changing topics.

<table><tr><td rowspan="2">Question items</td><td rowspan="2"></td><td colspan="4">mean (S.D.)</td><td colspan="4">p-value</td></tr><tr><td>Base</td><td></td><td>P1</td><td>P2</td><td>Base vs. P1 vs. P2</td><td>Base vs. P1</td><td>Base vs. P2</td><td>P1 vs. P2</td></tr><tr><td>Processing</td><td>1</td><td>4.90 (1.45)</td><td></td><td>4.94 (1.46)</td><td>5.23 (1.29)</td><td>† 0.071</td><td>0.734</td><td>0.058</td><td>0.040</td></tr><tr><td>effort</td><td>2</td><td>4.45 (1.41)</td><td>4.33</td><td>(1.55)</td><td>5.06 (1.48)</td><td>**0.008</td><td>0.508</td><td>* 0.003</td><td>*p &lt; 0.001</td></tr><tr><td>Cognitive</td><td>3</td><td>4.65 (1.45)</td><td>4.58</td><td>(1.59)</td><td>5.01 (1.37)</td><td>† 0.087</td><td>0.641</td><td>0.080</td><td>0.010</td></tr><tr><td>effect</td><td>4</td><td>4.55 (1.39)</td><td>4.58</td><td>(1.37)</td><td>5.14 (1.31)</td><td>*0.019</td><td>0.866</td><td>*0.004</td><td>*0.004</td></tr></table>

† p<0.1, ∗ p<0.05, ∗∗ p<0.01: Friedman’s test

⋆ p<0.017: Wilcoxon signed-rank test with Bonferroni correction for multiple comparisons  
Table 6: Results of the evaluation of the naturalness of dialogues (n = 88). (Base: Baseline, P1: Proposed Method 1, P2: Proposed Method 2)
<table><tr><td>Methods</td><td>mean (S.D.) in a turn</td></tr><tr><td>P1</td><td>2.38 (1.80)</td></tr><tr><td>P2</td><td>3.78 (1.26)</td></tr></table>

Table 7: Average number of each method’s generated slots. (P1: Proposed Method 1, P2: Proposed Method 2)
<table><tr><td>Items</td><td>Score</td></tr><tr><td>Precision</td><td>0.825 (85/103)</td></tr><tr><td>Recall</td><td>0.842 (85/101)</td></tr><tr><td>F1</td><td>0.833</td></tr></table>

Table 8: Performance of the slot-filling module.
<table><tr><td>Questions</td><td>mean (S.D.)</td></tr><tr><td>Did the nurse&#x27;s statements align with the persona settings? (1: Strongly disagree, 7: Strongly agree)</td><td>5.04 (1.31)</td></tr><tr><td>Were the nurse&#x27;s statements natural? (1: Strongly disagree, 7: Strongly agree)</td><td>5.13 (1.36)</td></tr></table>

Table 9: Performance of the user simulator.

In Evaluation Item 2, the Baseline Method and Proposed Method 1 achieved lower results compared to Proposed Method 2. In dialogues created using Proposed Method 1 during this experiment, topic shifts occurred when the LLM determined that further exploration was not feasible. This was particularly evident when the slots for deepening the discussion were not appropriately generated, leading to abrupt topic changes. Compared to Proposed Method 2, the slots generated by Proposed

Method 1 for deeper exploration were less suitable, which likely contributed to its lower evaluation. In contrast, the Baseline Method switched topics whenever one of the initial slot set in the predefined set was completed. Due to the weak relationships between the slots in the initial set, evaluators likely perceived the topic shifts to new slots as abrupt.

## 6.2.2 Cognitive Effect of Questions

Although there was no statistically significant difference in Evaluation Item 3, Proposed Method 2 produced the best results. Proposed Method 2 dynamically generated slots and hypotheses using abductive reasoning based on user responses, guiding the dialogue through hypothesis testing. This strategy resulted in a more natural conversational flow and helped prevent topic drift, even when handling multiple slots. As a result, it is believed that the method was able to extract the necessary information more effectively.

On the other hand, Proposed Method 1 had the lowest score and the largest standard deviation. This can be attributed to its approach of generating “slots suitable for career counseling” to guide the dialogue. However, it often focused excessively on specific topics (e.g., current tasks) or exhibited weak associations between slots, leading to scattered questions. Consequently, it was evaluated as less effective in collecting information.

In Evaluation Item 4, Proposed Method 2 showed statistically significant differences compared to the other methods and achieved relatively high results. Statistically significant differences were observed between the Baseline Method and the proposed methods. Proposed Method 2, which employs abductive reasoning for probing, asked deeper and more detailed questions, improving the overall quality of questions throughout the dialogue.

## 7 Concluding Remarks

In this study, we developed a slot-filling dialogue system that generates slots to support the preparation for career interviews through the collection of user information.

We conducted an experiment utilizing the user simulator to verify the effectiveness of the proposed methods, evaluating its ability to collect information and the naturalness of the dialogue.

The results suggest that slot generation using abductive reasoning contributes to more effective information collection from users. However, we also observed that focusing on information collecting led to a loss of naturalness in the dialogue, with some questions lacking clear intent. We plan to conduct experiments with actual nurses and further improve the effectiveness of slot generation.

Since the experiments in this study were conducted with the user simulator, it remains unclear how the system will perform in real interactions with nurses.

## 8 Limitations

This study has several limitations. The most significant is the issue concerning the accuracy of slot generation. In our method, slot generation is performed using an LLM based on abductive reasoning, but there is a risk of generating inappropriate slots due to excessive inference or misinterpretation. LLMs can sometimes generate slots that do not fit the context or are based on non-existent information, leading to unnatural questions that disrupt the flow of the dialogue. Preventing such mis-generation and improving the accuracy of the generated slots will be a key challenge in future research.

Excessive slot generation can result in overly long prompts, potentially compromising the naturalness of speech and the consistency of the dialogue. The core issue lies in retaining too many slots, which can lead to inappropriate responses. To address this, first, we aim to filter out less relevant slots, thereby shortening the prompt. Second, we are considering a design where users can stop the dialogue halfway through if inappropriate responses are generated, reducing the psychological burden.

The second limitation is related to the question generation module. If this module creates questions based on inappropriate slots, it may ask ineffective questions, negatively impacting information collection and diminishing the naturalness of the dialogue. Therefore, further optimization of the question generation module is necessary.

The final limitation is the use of the user simulator for evaluation. Although the results were promising, our system has not yet been verified using interactions with actual nurses. Real users may exhibit unexpected statements or emotional reactions, which could alter the system’s performance and behavior. Thus, evaluating the system with real users is essential for future research.

Considering these limitations, future studies should focus on refining the slot generation algorithm, improving the question generation module, and conducting evaluations with real users.

## 9 Ethical Considerations

When this system is deployed, users will share their personal information. However, the dialogue history will not be shared with the nursing manager. Before sharing the report to the nursing manager, the users have the opportunity to review and edit its contents. Therefore, there are no privacy concerns.

Potential risks include the possibility of the system being used for training purposes and the risk of OpenAI or the system’s servers being hacked. However, this system does not require users to input their names. By linking users to their employee ID numbers, the risk of external identification can be kept low. Additionally, since users can edit the report content, if logs are not stored, privacy can be effectively maintained.

## Acknowledgment

This study is partially supported by JST CREST (JPMJCR20D1), NEDO (JPNP20006) and JSPS KAKENHI(24K03052, 22K12325).

## References

Daniel G. Bobrow, Ronald M. Kaplan, Martin Kay, Donald A. Norman, Henry Thompson, and Terry Winograd. 1977. GUS, a frame-driven dialog system. Artificial Intelligence, 8(2):155–173.

Samuel Coope, Tyler Farghly, Daniela Gerz, Ivan Vulic,´ and Matthew Henderson. 2020. Span-ConveRT: Fewshot span extraction for dialog with pretrained conversational representations. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 107–121, Online. Association for Computational Linguistics.

David DeVault, Ron Artstein, Grace Benn, Teresa Dey, Ed Fast, Alesia Gainer, Kallirroi Georgila, Jonathan Gratch, Arno Hartholt, Margot Lor-Lhommet, Gale Lucas, Stacy Marsella, Fabrizio Morbini, Angela Nazarian, Stefan Scherer, Giota Stratou, Apar Suri, David Traum, Rachel Wood, and Louis-Philippe Morency. 2014. Simsensei kiosk: A virtual human interviewer for healthcare decision support. In Proceedings of the 13th International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2014, volume 2, pages 1061–1068.

Yujie Feng, Zexin Lu, Bo Liu, Liming Zhan, and Xiao-Ming Wu. 2023. Towards LLM-driven dialogue state tracking. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 739–755, Singapore. Association for Computational Linguistics.

Yubin Ge, Ziang Xiao, Jana Diesner, Heng Ji, Karrie Karahalios, and Hari Sundaram. 2023. What should I ask: A knowledge-driven approach for follow-up questions generation in conversational surveys. In Proceedings of the 37th Pacific Asia Conference on Language, Information and Computation, pages 113– 124, Hong Kong, China. Association for Computational Linguistics.

Michael Heck, Nurul Lubis, Benjamin Ruppik, Renato Vukovic, Shutong Feng, Christian Geishauser, Hsienchin Lin, Carel van Niekerk, and Milica Gasic. 2023. ChatGPT for zero-shot dialogue state tracking: A solution or an opportunity? In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 936–950, Toronto, Canada. Association for Computational Linguistics.

Vojtech Hudeˇ cek and Ondrej Dusek. 2023.ˇ Are large language models all you need for task-oriented dialogue? In Proceedings of the 24th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 216–228, Prague, Czechia. Association for Computational Linguistics.

Koji Inoue, Kohei Hara, Divesh Lala, Kenta Yamamoto, Shizuka Nakamura, Katsuya Takanashi, and Tatsuya Kawahara. 2020. Job interviewer android with elaborate follow-up question generation. International Conference on Multimodal Interaction (ICMI), pages 324–332.

Michael Johnston, Patrick Ehlen, Frederick G. Conrad, Michael F. Schober, Christopher Antoun, Stefanie Fail, Andrew Hupp, Lucas Vickers, Huiying Yan, and Chan Zhang. 2013. Spoken dialog systems for automated survey interviewing. In Proceedings of the SIGDIAL 2013 Conference, pages 329–333, Metz, France. Association for Computational Linguistics.

Takahiro Kobori, Mikio Nakano, and Tomoaki Nakamura. 2016. Small talk improves user impressions of interview dialogue systems. In Proceedings of the 17th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 370–380, Los Angeles. Association for Computational Linguistics.

Keigo Komada, Kaori Abe, Shoji Moriya, and Jun Suzuki. 2024. Improving dialogue consistency using dynamic slots in dialogue tasks requiring flexible responses. Proceedings of the Annual Conference of the Japanese Society for Artificial Intelligence, JSAI2024(0):4Xin2102–4Xin2102. (in Japanese).

Gale M. Lucas, J. Gratch, Aisha King, and Louis-Philippe Morency. 2014. It’s only a computer: Virtual humans increase willingness to disclose. Computers in Human Behavior, 37:94–100.

Fuminori Nagasawa, Shogo Okada, Takuya Ishihara, and Katsumi Nitta. 2024. Adaptive interview strategy based on interviewees’ speaking willingness recognition for interview robots. IEEE Transactions on Affective Computing, 15(3):942–957.

Mikio Nakano and Kazunori Komatani. 2024. DialBB: A dialogue system development framework as an educational material. In Proceedings ofthe 25th Annual Meeting ofthe Special Interest Group on Discourse and Dialogue, pages 664–668, Kyoto, Japan. Association for Computational Linguistics.

Charles Sanders Peirce. 1974. Collected papers of charles sanders peirce, volume 5. Harvard University Press.

Rao S B Pooja, Manish Agnihotri, and Dinesh Babu Jayagopi. 2020. Automatic follow-up question generation for asynchronous interviews. In Proceedings of the Workshop on Intelligent Information Processing and Natural Language Generation, pages 10–20, Santiago de Compostela, Spain. Association for Computational Lingustics.

A.B. Siddique, Fuad Jamour, and Vagelis Hristidis. 2021. Linguistically-enriched and contextawarezero-shot slot filling. In Proceedings of the Web Conference 2021, WWW ’21, page 3279–3290, New York, NY, USA. Association for Computing Machinery.

Amanda Stent, Svetlana Stenchikova, and Matthew Marge. 2006. Dialog systems for surveys: the ratea-course system. In 2006 IEEE Spoken Language Technology Workshop, pages 210–213.

Ming-Hsiang Su, Chung-Hsien Wu, and Yi Chang. 2019. Follow-Up Question Generation Using Neural Tensor Network-Based Domain Ontology Population in an Interview Coaching System. In Proc. Interspeech 2019, pages 4185–4189.

Ming-Hsiang Su, Chung-Hsien Wu, Kun-Yi Huang, Qian-Bei Hong, and Huai-Hung Huang. 2018. Follow-up Question Generation Using Pattern-based Seq2seq with a Small Corpus for Interview Coaching. In Proc. Interspeech 2018, pages 1006–1010.

Guangzhi Sun, Shutong Feng, Dongcheng Jiang, Chao Zhang, Milica Gasic, and Phil Woodland. 2024. Speech-based slot filling using large language models. In Findings ofthe Associationfor Computational Linguistics ACL 2024, pages 6351–6362, Bangkok,

Thailand and virtual meeting. Association for Computational Linguistics.

Katsuya Takanashi. 2024. The structure of hypothesisforming interviews by supervisors: Inheritance of classical ai approaches. proceedings of the special interest group on language and speech understanding and dialogue processing. The Japanese Society for Artificial Intelligence, 100:148–153. (in Japanese).

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2024. Chain-of-thought prompting elicits reasoning in large language models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NeurIPS ’22, Red Hook, NY, USA. Curran Associates Inc.

Deirdre Wilson and Dan Sperber. 1995. Relevance: Communication and Cognition, 2nd edition. Black well Publishers, Oxford; Cambridge.

Jie Zeng, Yukiko Nakano, and Tatsuya Sakato. 2023. Question generation to elicit users’ food preferences by considering the semantic content. In Proceedings of the 24th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 190–196, Prague, Czechia. Association for Computational Linguistics.

## A Prompts Used in the System

Figure 3 shows the prompt for Proposed Method 1. Figure 4 and 5 show the prompt for Proposed Method 2. Figure 6 shows the prompt for slot filling. Figure 7 shows the prompt for generating questions. Figure 8 shows the prompt for the user simulator.

The texts enclosed in < and > in Figures 3, 5, 6, 7 and 8 represent placeholders. For instance, “<Dialogue History>” denotes the section of the prompt where the actual user-system interaction up to the current turn is inserted.

![](images/4e0f77f8c9d4a2745962d378dbb05811c1148758bc2ce01238070d1692681d82.jpg)  
Figure 3: Prompt used for Proposed Method 1. (Originally in Japanese)

![](images/05b6b8ba6a0f729696802cd5e9c80b8769d27749e31b9ba3edc0e5847a98c271.jpg)  
Figure 4: Prompt used for Proposed Method 2. (first half) (Originally in Japanese)

Slot Description:   
In the career consultation support system for nurses, slots are used to organize and systematically   
record various types of information and data related to career development. Each slot corresponds to a   
specific topic or question and reflects concrete aspects related to the nurse's career. These may   
include career goals, specialized skills, intentions for further education, interests in training   
programs, as well as non-career-related information.   
Things You Must Not Do:   
Do not generate slots that are identical to existing ones. Do not repeat the same abduction as before   
by referencing the abduction history. However, it is acceptable if the inferred matter differs.   
Slot Description:   
In the career consultation support system for nurses, slots are used to organize and systematically   
record various information and data related to career development. Each slot corresponds to a specific   
topic or question and reflects concrete aspects related to the nurse's career. It may also include   
non-career-related information.   
Instructions for Output:   
Always output in JSON format. Do not include previous slots in the output. Do not include the content   
in the "value" field for new slots. Ensure the "value" is always set to null.   
Output Example (multiple slots are acceptable) :   
{   
"Surprising Fact C": "", # Content of C. If none, set to null   
"Reason to Suspect A": "", # Content of A. If none, set to null   
"New Slot": { # A maximum of 5 new slots can be generated   
"XXX": { # Slot name   
"category": "xxx,zzz", # Slot category   
"value": null # Always set to null   
},   
"YYY": { # Slot name   
"category": "yyy, zzz", # Slot category   
"value": null # Always set to null   
}   
}   
}   
Dialogue History:   
< Dialogue History >   
Current Slots:   
< Current Slots >   
Abduction History:   
< Abduction History >  
Figure 5: Prompt used for Proposed Method 2. (second half) (Originally in Japanese)

![](images/b47415b863913b2e0f7a4f92d06dd685e35da7d3d37ec7673a0c153a2aecadc4.jpg)  
Figure 6: Prompt used for slot-filling. (Originally in Japanese)

![](images/aecfe9fc6e4ebc9f8d6abb72c1d0d5df6d16db9d0bc52c45a1120dd5f20317be.jpg)  
Figure 7: Prompt used for generating question. (Originally in Japanese)

![](images/29c6b6b7c1dfe60d4461f96e6df6b7a9c3e796f4734305d1b0f9e8287db3db44.jpg)  
Figure 8: Prompt used for the user simulator. (Originally in Japanese)

## B An Example of Persona (Aoi Endo) and Dialogues

## B.1 An Example of Persona Setting: Aoi Endo (Originally in Japanese)

Table 10 shows the persona setting of Aoi Endo.

## B.2 An Example Dialogue using Baseline Method with Aoi Endo (Originally in Japanese)

System: Have you been busy lately? Aoi Endo : Yes, I’m busy, but I feel it’s rewarding.

System: Despite being busy, what kind of support do you think would make it easier to work?

Aoi Endo : I’m dissatisfied with the lack of promotion opportunities. How can this be improved?

System: Regarding the lack of promotion opportunities, what specific support or systems do you think would help improve this situation?

Aoi Endo : It would be helpful to have training and support for obtaining qualifications for promotion.

System: Regarding the support for training and obtaining qualifications for promotion, what kind of content or format do you think would be ideal?

Aoi Endo : Online courses for obtaining qualifications and regular training sessions would be ideal.

System: For the online courses and regular training sessions, what specific content or themes do you think would be useful?

Aoi Endo : Leadership training for management positions and updates on the latest medical knowledge.

System: Regarding the leadership training for management positions and the latest updates on medical knowledge, what specific content or themes do you think would be useful?

Aoi Endo : For leadership training, effective team management and communication skills are important.

System: Can you tell me more specifically about the team management and communication skills you’d like to learn in the leadership training?

<table><tr><td rowspan=1 colspan=1>Attribute</td><td rowspan=1 colspan=1>Details</td></tr><tr><td rowspan=1 colspan=1>Name</td><td rowspan=1 colspan=1>Aoi Endo</td></tr><tr><td rowspan=1 colspan=1>Age</td><td rowspan=1 colspan=1>30 years old</td></tr><tr><td rowspan=1 colspan=1>Hometown</td><td rowspan=1 colspan=1>Niigata Prefecture</td></tr><tr><td rowspan=1 colspan=1>Gender</td><td rowspan=1 colspan=1>Female</td></tr><tr><td rowspan=1 colspan=1>Personality</td><td rowspan=1 colspan=1>Sincere, cooperative</td></tr><tr><td rowspan=1 colspan=1>Past Career</td><td rowspan=1 colspan=1>A nurse working at a uni-versity hospital, in her8th year as a nurse, andhas been working therethroughout her career.</td></tr><tr><td rowspan=1 colspan=1>CurrentCareer</td><td rowspan=1 colspan=1>Works    in    internalmedicine and servesas a deputy leader in theteam.</td></tr><tr><td rowspan=1 colspan=1>FutureAspirations</td><td rowspan=1 colspan=1>Aims to advance into anursing management posi-tion.</td></tr><tr><td rowspan=1 colspan=1>Thoughts</td><td rowspan=1 colspan=1>Satisfied with good rela-tionships at her currentworkplace but feels dis-satisfied with the lack ofpromotion opportunities.Wishes to continue work-ing at the same hospital.</td></tr><tr><td rowspan=1 colspan=1>Other Details</td><td rowspan=1 colspan=1>Her hobby is handicraftsShe has one child, andher parents live nearby andprovide support for child-care.</td></tr><tr><td rowspan=1 colspan=1>Check Items</td><td rowspan=1 colspan=1>Intentions toward nurs-ing management positions,Dissatisfaction with pro-motion, Desire for conti-nuity.</td></tr></table>

Table 10: Persona: Aoi Endo.

Aoi Endo : In terms of team management, effective task delegation and maintaining motivation are important. System: That’s all for today!

## B.3 An Example Dialogue using Proposed Method 1 with Aoi Endo (Originally in Japanese)

System: Have you been busy lately? Aoi Endo: Yes, I’m busy, but I feel it’s rewarding.

System: I see, it’s wonderful that you’re feeling rewarded! By the way, can you tell me about your current job responsibilities?

Aoi Endo: I’m supporting the team as a deputy leader in the internal medicine department.

System: As a deputy leader, can you tell me specifically what kind of tasks you’re responsible for?

Aoi Endo: I’m in charge of coordinating the team, training new staff, and creating care plans for patients.

System: I see, you’re handling a variety of roles. Can you tell me more about your specific career goals?

Aoi Endo: I want to move into a nursing management position.

System: So, you’re aiming for a nursing management position. Can you share your career development plan in more detail?

Aoi Endo: Since there are few promotion opportunities, I’m also considering other hospitals.

System: You’re interested in other hospitals, but what kind of hospitals or environments are you specifically considering?

Aoi Endo: I’m conflicted because the relationships at my current workplace are good.

System: It’s great that the relationships are good! Specifically, what aspects of the relationships do you feel are good?

Aoi Endo: My colleagues are cooperative, and it’s easy to exchange opinions.

System: You mentioned there are few promotion opportunities. What kind of support do you think would increase the chances of promotion?

Aoi Endo: Support for management training and obtaining qualifications would be helpful.

System: You mentioned that support for management training and obtaining qualifications would be helpful. Specifically, what qualifications would you like to obtain?

Aoi Endo: I would like to obtain a nursing management qualification and take leadership training.

System: What kind of support do you feel is necessary for obtaining the nursing management qualification?

Aoi Endo: I need time off for studying and financial assistance.

System: You need time off and financial assistance for obtaining qualifications. Regarding your interest in other hospitals, what kind of hospitals or environments are you specifically considering?

Aoi Endo: A place where I can receive support for my family would be ideal.

System: That’s all for today!

## B.4 An Example Dialogue using Proposed Method 2 with Aoi Endo (Originally in Japanese)

System: Have you been busy lately? Aoi Endo: Yes, I’m busy, but I feel it’s rewarding.

System: I see, it’s wonderful that you’re feeling rewarded! By the way, can you tell me about your current job responsibilities?

Aoi Endo: I’m supporting the team as a deputy leader in the internal medicine department.

System: As a deputy leader, can you tell me specifically what kind of tasks you’re responsible for?

Aoi Endo: I’m in charge of coordinating the team, training new staff, and creating care plans for patients.

System: I see, you’re handling a variety of roles. Regarding leadership skills, what specific skills do you want to improve?

Aoi Endo: I’d especially like to strengthen my communication skills as part of my leadership skills.

System: Can you tell me about your specific goals for the next year?

Aoi Endo: Next year, I want to take the next step towards a nursing management position.

System: As you’re aiming to step into a nursing management position, what preparations are you currently making?

Aoi Endo: I’m participating in management training and learning about leadership.

System: You’re participating in management training. Can you tell me more about the specific content of that training?

Aoi Endo: In the training, I’m learning skills related to management and problem-solving.

System: What kind of results or changes have you noticed since taking that training?

Aoi Endo: My management skills have improved, and it’s become easier to coordinate the team.

System: What kind of results or changes are you hoping to achieve through this training?

Aoi Endo: I’m aiming to improve the overall performance of the team and ensure smooth communication.

System: Regarding the improvement of overall team performance, can you tell me more specifically what improvements you’re aiming for?

Aoi Endo: I’d like to improve the quality of training for new staff and streamline patient care.

System: Regarding improving the quality of training for new staff, what specific improvements are you aiming for?

Aoi Endo: I want to enhance practical guidance so that new staff can quickly adapt to the workplace.

System: That’s all for today!

## C An Example of Persona (Aoi Takahashi) and Dialogues

## C.1 An Example of Persona Setting: Aoi Takahashi (Originally in Japanese)

Table 11 shows the persona setting of Aoi Takahashi.

## C.2 An Example Dialogue using Baseline Method with Aoi Takahashi (Originally in Japanese)

System: Have you been busy lately? Aoi: I’m getting used to work, but I’m worried about my son starting elementary school.

System: I see, so your son is starting elementary school. What are you

<table><tr><td rowspan=1 colspan=1>Attribute</td><td rowspan=1 colspan=1>Details</td></tr><tr><td rowspan=1 colspan=1>Name</td><td rowspan=1 colspan=1>Aoi Takahashi</td></tr><tr><td rowspan=1 colspan=1>Age</td><td rowspan=1 colspan=1>32 years old</td></tr><tr><td rowspan=1 colspan=1>Hometown</td><td rowspan=1 colspan=1>Osaka Prefecture</td></tr><tr><td rowspan=1 colspan=1>Gender</td><td rowspan=1 colspan=1>Female</td></tr><tr><td rowspan=1 colspan=1>Personality</td><td rowspan=1 colspan=1>Extroverted</td></tr><tr><td rowspan=1 colspan=1>Past Career</td><td rowspan=1 colspan=1>A nurse working at a uni-versity hospital. Has beena nurse for 10 years, withexperience in pediatricsfor five years.</td></tr><tr><td rowspan=1 colspan=1>CurrentCareer</td><td rowspan=1 colspan=1>Currently works in ortho-pedics and considers her-self a skilled nurse.</td></tr><tr><td rowspan=1 colspan=1>FutureAspirations</td><td rowspan=1 colspan=1>Aims to balance familyand work, and is consider-ing becoming a generalist.</td></tr><tr><td rowspan=1 colspan=1>Thoughts</td><td rowspan=1 colspan=1>Feels confident in her job.She is discussing work-life balance with her hus-band as her son will startelementary school nextyear.</td></tr><tr><td rowspan=1 colspan=1>Other Details</td><td rowspan=1 colspan=1>Her son, who will start el-ementary school next year,wants to attend a soccerschool after school. Shelives with her husband andson. They often take walksin the local park on week-ends. Her favorite TVshow is &quot;World HeritageSites.&quot;</td></tr><tr><td rowspan=1 colspan=1>Check Items</td><td rowspan=1 colspan=1>Intentions toward becom-ing a generalist, Concernsabout work-life balance.</td></tr></table>

Table 11: Persona: Aoi Takahashi.

worried about?

Aoi: I’m anxious about balancing work and raising my child.

System: I see, that must be tough. Specifically, what are you concerned about?

Aoi: I’m worried about how my son will spend his time after school and getting called in suddenly.

System: That is concerning. Do you have anyone to support you?

Aoi: My husband is supportive, but I still want to be the main one taking care of him.

System: I see, family support is important. What kind of support do you think would help with your son’s after-school time?

Aoi: My son wants to join a soccer school, and I’m worried about the pick-up and drop-off.

System: What kind of support do you think would help with the soccer school pick-up and drop-off?

Aoi: It would help if my husband and I could take turns with the pick-up and drop-off. What do you think?

System: That’s a great idea! It might be a good idea to discuss it with your husband. Is there anything else you need support with?

Aoi: I want support for when I’m suddenly called in.

System: What kind of support do you think would help for sudden call-ins?

Aoi: Flexible shift arrangements at work would help.

System: You mentioned flexible shifts at work would help, but what kind of shift adjustments do you think would be ideal?

Aoi: A flexible shift that allows me to respond to sudden call-ins would be ideal.

System: Specifically, what kind of shift adjustments would be ideal? For example, is there a particular time of day or day of the week when you’d like more flexibility?

Aoi: It would help if I had flexibility in the afternoons or evenings on weekdays. System: That’s it for today!

## C.3 An Example Dialogue using Proposed Method 1 with Aoi Takahashi (Originally in Japanese)

System: Have you been busy lately? Aoi: I’m getting used to work, but I’m worried about my son.

System: I see, that sounds tough. What are you worried about with your son, specifically?

Aoi: My son wants to go to soccer

school, and it’s hard to adjust my time.

System: That does sound difficult. Do you think you can adjust your shifts? Is there anything I can help with?

Aoi: Adjusting my shifts might be hard.   
I’ll try discussing it with my husband.

System: That sounds like a good idea. I hope you and your husband can work something out.

Aoi: Thanks, I’ll try that. I also need to keep up with work.

System: Work must be tough too. Could you tell me more about your job responsibilities?

Aoi: I handle general nursing duties in orthopedics.

System: Orthopedic nursing sounds tough. How satisfied are you with your job?

Aoi: I find my job fulfilling, but I get tired sometimes.

System: It must be exhausting at times. Do you have any particular department you’d like to be assigned to in the future? Aoi: I’m aiming to be a generalist, so I’m not particular about a specific department.

System: I see, you’re aiming to become a generalist. Is there any other support or concerns you have about your family? Aoi: For now, everything is fine, but I’ll consult you again if I need to.

System: Got it, feel free to ask anytime. By the way, could you tell me more specifically about your shift preferences? Aoi: Which is easier to adjust, early or late shifts?

System: Could you tell me specifically what kind of shift adjustments you’d like?

Aoi: Early shifts might be easier to adjust because I have to pick up and drop off my son.

System: That’s it for today!

## C.4 An Example Dialogue using Proposed Method 2 with Aoi Takahashi (Originally in Japanese)

System: Have you been busy lately? Aoi: I’m getting used to work, but things have been hectic with my son.

System: I see, that sounds tough. Is there anything I can help with regarding your son?

Aoi: Thanks, but we’re planning to manage it as a family.

System: Got it, don’t push yourself too hard. Is there anything work-related you’re struggling with?

Aoi: Balancing work and raising my child is difficult, and I’m not sure what to do.

System: I see, balancing both really is tough. Specifically, what part of it is difficult?

Aoi: My work shifts and my son’s soccer school schedule overlap.

System: If you have any shift preferences, could you let me know? For example, if you need specific days or times off?

Aoi: I would like to have afternoons off on weekends, so I can accompany my son to soccer.

System: I see, so you’d like to have weekends off in the afternoon. Can you tell me more about your hopes for family time?

Aoi: I want to prioritize family time, so I’d like to spend as much time as possible with them on weekends.

System: Got it, it’s wonderful to prioritize family time. By the way, could you tell me about your current job responsibilities?

Aoi: I work in orthopedics, assisting with surgeries and caring for patients.

System: Are you satisfied with your job in orthopedics, or is there something you’d like to change?

Aoi: I’m satisfied with my job, but I want to keep growing.

System: What specific goals do you have for growth?

Aoi: I’m aiming to be a generalist, so I want to gain broad knowledge and experience.

System: What areas or skills are you particularly interested in learning to gain broader experience?

Aoi: I want to learn more about emergency medicine and home healthcare.

System: That’s it for today!