# Multi-Graph Co-Training for Capturing User Intent in Session-based Recommendation

Zhe Yang<sup>1</sup>, Tiantian Liang<sup>1</sup>\*

<sup>1</sup>School of Computer Science and Technology, Soochow University, Suzhou, China yangzhe@suda.edu.cn, ttliang2023@stu.suda.edu.cn

## Abstract

Session-based recommendation focuses on predicting the next item a user will interact with based on sequences of anonymous user sessions. A significant challenge in this field is data sparsity due to the typically short-term interactions. Most existing methods rely heavily on users’ current interactions, overlooking the wealth of auxiliary information available. To address this, we propose a novel model, the Multi-Graph Co-Training model (MGCOT), which leverages not only the current session graph but also similar session graphs and a global item relation graph. This approach allows for a more comprehensive exploration of intrinsic relationships and better captures user intent from multiple views, enabling session representations to complement each other. Additionally, MGCOT employs multi-head attention mechanisms to effectively capture relevant session intent and uses contrastive learning to form accurate and robust session representations. Extensive experiments on three datasets demonstrate that MGCOT significantly enhances the performance of session-based recommendations, particularly on the Diginetica dataset, achieving improvements of up to 2. 00% in P @ 20 and 10. 70% in MRR @ 20. Resources have been made publicly available in our GitHub repository https://github.com/ liang-tian-tian/MGCOT.

## 1 Introduction

Session-based recommendation aims to discover user intent by learning from the sequence of items in the current session, ultimately recommending items of interest to the user. A session typically refers to a sequence of user interactions with multiple items within a period of time, such as consecutively clicking on several products on a shopping platform. Session-based recommendation is particularly effective in attracting and retaining anonymous users, especially those who prioritize privacy or are first-time users of the platform. This approach is crucial for e-commerce platforms and streaming services, such as Amazon or YouTube. However, the greatest challenge in session-based recommendation is severe data sparsity, as it primarily focuses on the user actions within the current session and fails to adequately capture the intrinsic relationships between items and the similar intents across different sessions.

Early session-based recommendation models leverage the Markov chain assumption (Rendle et al., 2010) to capture sequential patterns. With advances in neural networks, Recurrent Neural Networks (RNNs) (Hidasi et al., 2015; Li et al., 2017; Ren et al., 2019) are employed to extract item transition relationships using recurrent units or attention layers. Graph Neural Networks (GNNs) (Wu et al., 2019) convert session sequences into graph structures to capture higher-order item relationships. While GNNs outperform in capturing pairwise item transitions, they may have weaker long-term dependencies. Graph Attention Networks (GATs) (Wang et al., 2019, 2020) address this issue by incorporating attention mechanisms, but their high memory consumption limits their application to current session data, often neglecting global item correlations. Without attention mechanisms, recommendation precision may decline.

Self-Supervised Learning (SSL) (Liu et al., 2021; Xia et al., 2021b,a) provides effective solutions for data sparsity by constructing both global and local graphs to enhance session representations. However, these methods often fail to capture similar intents across different sessions, leading to incomplete information modeling.

More recently, MiasRec(Choi et al., 2024) generates multiple session representations centered around each item and dynamically selects the most relevant ones to capture user intent. This approach performs well in longer-session contexts, but its effectiveness diminishes in shorter-session scenarios.

To address these issues, we propose a multigraph co-training model with various attention mechanisms that captures user intent from different views and filters out irrelevant items. Our model includes tree views to obtain the session representation: the current view, which reflects item transitions within the current session; the local view, which captures relationships between similar sessions; and the global view, which encompasses item relationships across all sessions. Each view includes an encoding layer, implemented with either Gated Recurrent Unit (GRUs) or GNNs, along with attention mechanisms to generate session embeddings. Finally, contrastive learning is applied between the combination of current and local graphs and the global graph to capture more accurate session representations.

In summary, the main contributions of this paper are as follows:

• We introduce the construction of a frequencybased current item graph and employ shortest path algorithms in the global graph to further enhance the model’s capacity to comprehensively transform session data into graph representations.

• We introduce various attention mechanisms to effectively capture session information. These mechanisms allow the model to extract relevant data from diverse aspects of the session and emphasize critical information, thereby aligning more closely with the user’s intent.

• We propose a co-training approach between the combination of current and local graphs and the global graph using contrastive learning, enabling a more comprehensive and complementary understanding of user intent from different views.

• We conduct extensive experiments on three real-world datasets, demonstrating that MG-COT outperforms SOTA models. Specifically, MGCOT achieves a 5.02% increase in M@10 on Tmall, a 2.17% increase on RetailRocket, and a 10.53% increase on Diginetica.

## 2 Related Work

In this section, we introduce the related work of our model MGCOT, which includes GNN-based

methods and self-supervised learning.

## 2.1 GNN-based Methods

GNNs (Wu et al., 2020) have been widely used in capturing complex transition relationships and have demonstrated substantial effectiveness. Sessions can be well represented as graphs, and various studies have explored how GNNs can enhance session recommendation accuracy. The SR-GNN model (Wu et al., 2019) is the first to utilize the Gated Graph Neural Network (GGNN) for learning item embeddings by propagating information on the session graph. Qiu et al. (Qiu et al., 2019) propose the FGNN model, which leverages multi-head attention to aggregate information from an item’s neighborhood. GC-SAN (Xu et al., 2019) is an evolution of the SR-GNN, which applies a self-attention mechanism to model item co-occurrences. GCE-GNN (Wang et al., 2020) aggregates item information from both item-level and session-level through graph convolution and self-attention mechanism. MGIR (Han et al., 2022) models not only sequential and global co-occurrence relations but also incompatible relations within a graph. KMVG(Chen et al., 2023) utilizes multi-view graph neural networks and a knowledge graph to more accurately capture user intent. MSGAT (Qiao et al., 2023) introduces a bi-channel model with multiple sparse graph attention networks that takes into account the effects of session intent and noise items. In the GNN-based session recommendation models, multi-graph models have shown significant advantages over single-graph models. This has inspired using a multi-graph co-training model with attention mechanisms to capture session intent more comprehensively.

## 2.2 Self-supervised Learning

In recent years, SSL has proven to be effective for recommendation. $S ^ { 3 } .$ -Rec (Zhou et al., 2020) uses the mutual information maximization principle to learn the underlying relationships among items, attributes, and sequences. $S ^ { 2 }$ -DHCN (Xia et al., 2021b) employs a contrastive learning mechanism to enhance hyper-graph modelling through a different line of GCN models. COTREC (Xia et al., 2021a) proposes constructing session data into two views to capture the internal and external connectivity of sessions. CGL (Pan et al., 2022) integrates SSL with supervised learning to explore correlations across different sessions, thereby improving item representations. HGCMA(Chen et al., 2024)

employs random masking and contrastive learning to learn discriminative node representations in heterogeneous graphs. While SSL has demonstrated great performance in capturing user intent from various views, these methods overlook the potential benefits of combining similar session intents to achieve a more comprehensive and accurate understanding.

## 3 Methods

The model learns session representations from three views: current view, local view, and global view, as illustrated in Figure 1. In the current view, we adopt the SR-GNN approach(Wu et al., 2019), utilizing the Gated Graph Neural Network (GGNN) as the initial encoder. This method combines the strengths of Graph Neural Networks (GNNs) and Gated Recurrent Units (GRUs), enabling the model to effectively capture item relationships within a session and extract the session’s intent. In the local view, the model first generates session representations for the current batch. It then refines the session representation from the current view by integrating it with representations from the top-k similar sessions identified within the local context. The enhanced session representation is subsequently utilized for the primary recommendation task. In the global view, the model generates the current session representation by extracting item information from all sessions. We also incorporate various attention mechanisms to extract crucial information in these views.

To further improve the model’s ability to capture item relationships, a contrastive learning approach is employed. This approach compares the enhanced session representation, which integrates current session representation with similar session information from the local view, with the session representation generated from the global view. The method of enhancing the current session representation by integrating similar session information in the local view is inspired by (Qiao et al., 2023).

In this section, we focus on introducing our key ideas: graph construction from different views, various attention mechanisms, and contrastive learning.

## 3.1 Problem Definition

We represent the set of sessions in the dataset as ${ \cal S } = \{ s _ { 1 } , s _ { 2 } , \ldots , s _ { M } \}$ , where M denotes the total number of sessions. The set of all items is defined as $V ~ = ~ \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { N } \}$ , where N is the total number of items in the dataset. Each session $s _ { t }$ is generated by an anonymous user interacting with a set of items. The session at time t is denoted as $s _ { t } = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { L } \}$ , where L is the length of the current session. The objective of session-based recommendation is to capture the user intent based on the first L consecutive interactions and predict the L+1-th potential interaction item.

## 3.2 Graph Construction from Different Views

To fully leverage the available information, we explore relationships between items and sessions from three views: the current view, global view, and local view. These views focus on item relationships within the current session, across all sessions, and among batch sessions. To capture the intrinsic correlations, we first convert sessions into graphs. We propose the current frequency item graph, the global shortest-path item graph, and introduce the local session graph.

## 3.2.1 Current Frequency Item Graph

In session-based recommendation, the order in which users click on items reflects changes in their interests. Sequence information helps the model better understand the current context of the user, thereby improving recommendation accuracy. However, in traditional directed graph construction, different sessions composed of the same items might generate identical graph structures. For example, session sequences $s _ { 1 } ~ = ~ \{ v _ { 2 } , v _ { 4 } , v _ { 5 } , v _ { 5 } , v _ { 4 } , v _ { 4 } \}$ and $s _ { 2 } =$ $\{ v _ { 2 } , v _ { 4 } , v _ { 4 } , v _ { 5 } , v _ { 5 } , v _ { 4 } \}$ may result in indistinguishable graphs. This can lead to a loss of critical sequential information, negatively impacting recommendation results.

To preserve as much information from the original sessions as possible, we propose a method for constructing directed graphs based on the frequency of item occurrences within the current session. In this method, the in-degree frequency of an item is used as the edge weight in the directed graph structure. For instance, in the session sequence $s _ { 3 } = \{ v _ { 2 } , v _ { 4 } , v _ { 5 } , v _ { 8 } , v _ { 4 } \}$ , the edge from $v _ { 2 }$ to $v _ { 4 }$ has a weight of 1, while the edge from $v _ { 8 }$ to $v _ { 4 }$ has a weight of 2, as shown in Figure 2. By introducing frequency-based weights, this method effectively reduces information loss during the session graph construction process, ensuring that sessions like $s _ { 1 }$ and $s _ { 2 }$ , which differ in edge weights, generate distinct graph structures. This approach better preserves and allows the model to learn the comprehensive information within sessions.

![](images/37ab315461c41136cf48f633eea751329bbb5f70e142a99b0ae2ed6e5ca2182e.jpg)  
Figure 1: An overview of the proposed MGCOT framework.

![](images/8a2105af2906ad078e7170c6cc417d177a807fbf76fd83fecaea7dfeadb2cabf.jpg)  
Figure 2: Session Graph with Frequency Indegree

![](images/4b2013368514dabac41d1c869f8ddbbbf05dd123857ebe4609e1da613b3e5bcb.jpg)  
(a) Occurrence Weight Graph

![](images/adc3d0c0a9d1ab03d05e89bdf2ea0d44a46bc59c4f4d3397d3b66fbf30d39428.jpg)  
(b) Shortest-path Graph  
Figure 3: Global Shortest-path Item Graph

## 3.2.2 Global Shortest-path Item Graph

Most existing models based on GNNs perform poorly in capturing long-range dependencies because GNNs only aggregate information from neighboring nodes in each layer. By stacking multiple layers of GNNs, the model can gradually aggregate information from more distant neighbors, but too many layers may lead to overfitting.

To address the issue of capturing relationships between distant nodes, we construct the global item graph (as shown in Figure 3a) and then use Dijkstra’s algorithm to compute the shortest path between pairs of nodes (as shown in Figure 3b). First, we transform the weight of each edge to its inverse weight by subtracting the edge weight from the maximum weight of all edges to obtain the corresponding cost value $c _ { i j }$ . Then, for each node in the global item graph, we calculate the shortest path from that node to all other nodes based on the minimum total cost of all edges along the path. Finally, the calculated cost values are inverted back into weights, allowing the global graph based on the shortest paths to effectively capture the relationships between distant nodes. In the global graph, the minimum cost matrix $\hat { C }$ and the final edge weight $\hat { w } _ { i j }$ are defined as follows:

$$
d _ { i j } = m i n ( d _ { i j } , d _ { i k } + c _ { k j } )\tag{1}
$$

$$
w _ { i j } = m a x ( \hat { C } + 1 ) - \hat { c } _ { i j }\tag{2}
$$

Here, $d _ { i j }$ represents the current shortest distance from the start node i to node $j ,$ , and $d _ { i k } + c _ { k j }$ represents the total cost of the path from the start node i to node j via node k. The Equation 1 indicates that if the cost of reaching node $j$ through node k is less than the currently known shortest path, the shortest path value is updated accordingly. In Equation 2, C<sup>ˆ</sup> represents the matrix of minimum costs for all edges, and $\hat { c } _ { i j }$ denotes the minimum cost from node i to node j.

## 3.2.3 Local Session Graph

When constructing a local session graph, we follow the method described in (Qiao et al., 2023). Each session is treated as a node in the graph. The edges between nodes are determined by calculating the Jaccard similarity of the items shared between sessions. A higher Jaccard similarity indicates a greater overlap of items between the two sessions, resulting in a higher edge weight. This indicates the intent of the two sessions is more similar.

## 3.3 Attention Mechanisms

The attention mechanism can effectively capture important information related to the intent of the session. In this paper, we focus on two main types of attention mechanisms: the multi-head attention mechanism and the target attention mechanism. The multi-head attention mechanism is used to comprehensively capture significant information within the current session, while the target attention mechanism extracts information from a global view and learns information related to the target item by incorporating the context from the current view. Additionally, we incorporate the cross attention mechanism as described in (Qiao et al., 2023).

## 3.3.1 Multi-head Attention Mechanism

The multi-head attention mechanism uses multiple independent attention heads to compute attention scores in different subspaces simultaneously. This approach allows the model to focus on various aspects of the input sequence within the same layer. The self-attention mechanism is inherently global, enabling each position’s output vector to interact with every other position in the input sequence, effectively capturing long-range dependencies. In the current view, the multi-head self-attention mechanism effectively captures relevant information in the session.

The multi-head self-attention mechanism mainly consists of three parts. First, a feedforward neural network is employed to enhance the representation of the query vector $Q .$ , making it more flexible and general, thus distinguishing it from the key vector $K$ and the value vector $V .$ . Here, $H _ { t }$ denotes the initial embedding of the session in the current view.

$$
\hat { Q } = f ( H _ { t } W _ { Q } + b _ { Q } )\tag{3}
$$

where $W _ { Q } \in \mathbb { R } ^ { 2 d \times 2 d }$ is the weight matrix, $b _ { Q } \in$ $\mathbb { R } ^ { 2 d }$ is the bias vector, and $f ( \cdot )$ denotes the ReLU activation function.

Second, a sparse transformation is applied to generate attention weights, ensuring that the new item embeddings are more relevant to the target item embeddings. The attention weights are calculated as follows:

$$
\alpha _ { t } = \sigma ( W _ { \alpha _ { t } } h _ { t } + b _ { \alpha _ { t } } ) + 1\tag{4}
$$

where $W _ { \alpha _ { t } } \in \mathbb { R } ^ { 1 \times d _ { k } }$ is the weight matrix, $b _ { \alpha _ { t } } \in$ $\mathbb { R } ^ { d _ { k } }$ is the bias vector, $d _ { k }$ is the dimension for each attention head, and $\sigma$ denotes the sigmoid activation function. The vector $h _ { t }$ is a special item index added to the end of the input sequence to indicate the item to be predicted. This special item embedding helps the model capture the overall session pattern rather than focusing solely on individual item characteristics.

Finally, the score of the multi-head attention mechanism is computed as follows:

$$
H _ { c u r } ^ { k } = { \alpha _ { t } } . . e n t m a x ( \frac { \hat { Q } K ^ { T } } { \sqrt { d _ { k } } } ) V\tag{5}
$$

$$
H _ { c u r } = C o n c a t ( H _ { c u r } ^ { 1 } , H _ { c u r } ^ { 2 } , . . . , H _ { c u r } ^ { H _ { n } } )\tag{6}
$$

where $\hat { Q }$ is the mapped representation of the current session, $K$ and $V$ are the key and value representations of the current session, $d _ { k }$ is the dimension of each attention head, $H _ { c u r } ^ { k }$ is the output of the k-th attention head, and entmax is a sparse attention mechanism. $H _ { n }$ represents the number of attention heads.

Although the multi-head attention mechanism learns new representations for all items, it is primarily based on linear projections. Subsequently, a feedforward neural network is applied to learn more nonlinear features:

$$
\hat { H } _ { c u r } = D r o p o u t ( W _ { 2 } ( f ( W _ { 1 } H _ { c u r } + b _ { 1 } ) ) + b _ { 2 } ) + H _ { c u r }\tag{7}
$$

where $W _ { 1 } , W _ { 2 } \ \in \ \mathbb { R } ^ { 2 d \times 2 d }$ are weight matrices, $b _ { 1 } , b _ { 2 } \in \mathbb { R } ^ { 2 d }$ are bias vectors, and $f ( \cdot )$ represents the ReLU activation function. $\hat { H } _ { c u r }$ $\{ h _ { 1 ^ { \prime } } , h _ { 2 ^ { \prime } } , . . . , h _ { t ^ { \prime } } \}$ represents the output processed by the multi-head self-attention mechanism, where $h _ { t ^ { \prime } }$ is the learned target item embedding. The dropout layer is included to prevent overfitting, while residual connections and layer normalization are applied to mitigate instability during training.

## 3.3.2 Target Attention Mechanism

The target attention mechanism aims to learn the representation of the entire session based on the learned target embeddings and initial inputs. It adjusts weights to reduce noise in the current initial session representation $H _ { g }$ from the global view. First, the target attention weights are computed as follows:

$$
\alpha _ { s } = \sigma ( W _ { \alpha _ { s } } h _ { t ^ { \prime } } + b _ { \alpha _ { s } } ) + 1\tag{8}
$$

where $W _ { \alpha _ { s } } \in \mathbb { R } ^ { 1 \times 2 d }$ is the weight matrix, $b _ { \alpha _ { s } } \in$ $\mathbb { R } ^ { 2 d }$ is the bias vector, σ denotes the sigmoid activation function, and $h _ { t ^ { \prime } }$ is the representation of the

target item obtained through the multi-head attention mechanism in the current view. The attention weight $w _ { s }$ is calculated as follows:

$$
w _ { s } = \alpha _ { s } – e n t m a x ( W _ { 0 } f ( W _ { 1 } H _ { g } + W _ { 2 } h _ { t ^ { \prime } } + b _ { 0 } ) )\tag{9}
$$

where $W _ { 1 } , W _ { 2 } \ \in \ \mathbb { R } ^ { 2 d \times 2 d }$ are weight matrices, $W _ { 0 } \in \mathbb { R } ^ { 1 \times 2 d }$ is a weight matrix, $b _ { 0 } \in \mathbb { R } ^ { 2 d }$ is a bias vector, $f ( \cdot )$ is the ReLU activation function. Finally, the final session representation $\hat { H } _ { g }$ in the global view is computed as:

$$
\hat { H } _ { g } = \sum _ { k = 0 } ^ { n } w _ { s } h _ { g } ^ { k }\tag{10}
$$

where $h _ { g } ^ { k } \in H _ { g }$ denotes the representation of each item in initial session embeddings.

## 3.4 Contrastive Learning

The core idea of contrastive learning is to build better feature representations by learning the similarities between similar samples and the differences between dissimilar samples. Specifically, for each session, we use the session representation from the current and local views, denoted as ${ \hat { H } } _ { r } ^ { b } .$ , which integrates information from other similar sessions within the batch. This representation is then contrasted with the session representation obtained from the global view, denoted as $\hat { H } _ { g } ^ { b } .$ . During training, we treat the representations of the same session from different views within the same batch as positive samples, aiming to pull these positive samples closer together. Conversely, we treat the representations of other sessions within the same batch as negative samples, aiming to push them further away from the current session representation. The similarity scores for positive and negative samples are calculated as follows:

$$
\mathrm { S i m } _ { p } = \hat { H } _ { r } ^ { b } \cdot \hat { H } _ { g } ^ { b }\tag{11}
$$

$$
\mathrm { S i m } _ { n } = \hat { H } _ { r } ^ { b } \cdot \hat { H } _ { g _ { \mathrm { s h u f f l e d } } } ^ { b }\tag{12}
$$

where $\hat { H } _ { g _ { \mathrm { s h u f f l e d } } } ^ { b }$ represents the global view representations of other sessions in the batch, excluding the current session and randomly shuffled. The contrastive learning loss is computed as:

$$
\begin{array} { r } { L _ { \mathrm { c o n t r a s t i v e } } = - l o g ( \sigma ( \frac { \mathrm { S i m } _ { p } } { \tau } ) ) - l o g ( \sigma ( - \frac { \mathrm { S i m } _ { n } } { \tau } ) ) } \end{array}\tag{13}
$$

where $\tau$ is a temperature parameter used to scale the similarity scores to enhance the effectiveness of

Table 1: Dataset Statistics
<table><tr><td>Dataset</td><td>Train</td><td>Test</td><td>Items</td><td>Avg.Len.</td></tr><tr><td>Tmall</td><td>351,268</td><td>25,898</td><td>40,728</td><td>6.69</td></tr><tr><td>RetailRocket</td><td>433,643</td><td>15,132</td><td>36,968</td><td>5.43</td></tr><tr><td>Diginetica</td><td>719,470</td><td>60,858</td><td>43,097</td><td>5.12</td></tr></table>

contrastive learning. The main recommendation encoder uses the cross-entropy loss function, defined as:

$$
\begin{array} { r } { L _ { \operatorname* { m i n } } = - \sum _ { i = 1 } ^ { N } y _ { i } l o g ( \hat { y } _ { i } ) + ( 1 - y _ { i } ) l o g ( 1 - \hat { y } _ { i } ) } \end{array}\tag{14}
$$

where $\hat { y } _ { i }$ denotes the probability of item $v _ { i }$ being the next click in the current session, and $y _ { i }$ is a binary label that equals 1 if item $v _ { i }$ is the ground truth next click and 0 otherwise. The total loss function is defined as:

$$
L = L _ { \mathrm { m a i n } } + \beta L _ { \mathrm { c o n t r a s t i v e } }\tag{15}
$$

where $\beta$ is a hyperparameter used to control the extent of contrastive learning.

## 4 Experiments

## 4.1 Datasets

We evaluate our model using three real-world benchmark datasets: Tmall, RetailRocket and Diginetica. The details of these datasets are presented in Table 1. The Tmall dataset contains user shopping logs and is provided by the IJCAI-15 competition. The RetailRocket dataset, released by an e-commerce company on Kaggle, includes six months of user browsing activities. The Diginetica dataset consists of typical transaction data from the CIKM Cup 2016.

To ensure data quality and relevance, we preprocess the data as follows(Wu et al., 2019; Wang et al., 2020; Xia et al., 2021a): We exclude sessions with a length of 1 and remove items that appear fewer than 5 times. Datasets are split with the most recent data as the test set and the rest as the training set. We also enhance the data by segmenting each session and generating labels, where each sequence is paired with the next item as the label. This augmentation improves the model’s ability to learn sequential patterns.

## 4.2 Baselines

To ensure a fair comparison, we select representative models from various categories, including traditional methods such as FPMC (Rendle et al., 2010), RNN-based models like GRU4Rec (Hidasi et al., 2015) and NARM (Li et al., 2017), GNNbased models such as SR-GNN (Wu et al., 2019), GCE-GNN (Wang et al., 2020), HICN(Sun et al., 2024), Mssen(Zheng et al., 2024) and MGIR (Han et al., 2022), attention-based models like STAMP (Liu et al., 2018), MTAW (Ouyang et al., 2023) and MSGAT (Qiao et al., 2023), and contrastive learning methods such as S<sup>2</sup>-DHCN (Xia et al., 2021b), to compare with the MGCOT model.

## 4.3 Experiment Setting

Following previous work (Qiao et al., 2023), we set the batch size to 512, the embedding size to 100, and the $L _ { 2 }$ regularization to $1 0 ^ { - 5 }$ . We use the Adam optimizer with a learning rate of 0.001, which decays by a factor of 0.1 every three epochs. Our model uses a single layer of GCN. The number of similar sessions selected is 6 for Tmall, 3 for Diginetica, and 2 for RetailRocket. The scale factor β for the contrastive learning loss is set to 0.05 for Tmall, and 5 for both Diginetica and RetailRocket. The number of attention heads $H _ { n }$ in the multihead attention mechanism is set to 1 for Diginetica, and 2 for Tmall and RetailRocket.

## 4.4 Experiment Results

Table 2 presents the experimental results of the MG-COT model compared to baseline models across three datasets. The best results are highlighted in bold, and the second-best results are underlined.

Experimental results show that traditional machine learning methods like FPMC underperform compared to deep learning approaches. FPMC, which combines matrix factorization and Markov chains, fails to capture long-term dependencies. Among RNN-based models, NARM outperforms GRU4Rec by using attention mechanisms to identify key relationships within sessions. STAMP relies solely on self-attention focused on the last item, replacing RNN encoders with attention layers to better capture short-term interests. MTAW, which models user interests dynamically with an attention mechanism and an adaptive weight loss function, enhances recommendation personalization. Overall, these models demonstrate the effectiveness of attention mechanisms in session-based recommendations.

Recently, GNN-based models have surpassed RNNs by uncovering spatial relationships between items. SR-GNN employs gated GNNs and a selfattention mechanism to capture session embeddings, while GCE-GNN constructs global and local graphs for cross-session learning. S<sup>2</sup>-DHCN converts sessions into hypergraphs and line graphs using self-supervised learning, inspiring the application of contrastive learning in multi-graph models. MGIR improves session representations by modeling global item relationships, including negative, co-occurrence, and sequential links. HICN boosts performance by leveraging sequential hyperedges and inter-hyperedge modules. Mssen uses multi-collaborative self-supervised learning in hypergraph neural networks to capture high-order relationships and address data sparsity. MSGAT, with dual-channel GNNs and attention mechanisms, excels at modeling both intra- and inter-session information, highlighting the advantages of GNNs with integrated attention.

Compared to the best baseline models, our MG-COT model shows significant performance improvements. By leveraging graph neural networks, attention mechanisms, and contrastive learning, MGCOT effectively captures latent relationships between sessions and items from current, local, and global views.

## 4.5 Ablation Experiments

To investigate the contribution of each module in MGCOT, we conduct ablation experiments with the following variants: (1) -NeighborSessions, where the fusion of similar session information from the local view is removed; (2) -MultiAttention, where the multi-head attention mechanism on session embeddings from the current view is removed; and (3) -ContrastiveLearning, where contrastive learning between the session embedding generated from the global view and the main session embedding fused from the local view and current view is removed.

As shown in Table 3, removing the fusion of similar session information led to a significant drop in evaluation metrics, indicating that similar sessions are as important as similar items in capturing user intent. Furthermore, both the multi-head attention mechanism and contrastive learning improve model performance, demonstrating the importance of assigning different weights to items when capturing session intent and the benefit of understanding session intent from multiple views.

## 4.6 Hyperparameter Experiments

In this hyperparameter experiment, we analyze the sensitivity of the MGCOT model to different parameter settings across datasets.

Figure 4 shows that we select 2 attention heads for Tmall and RetailRocket, and 1 for Diginetica. In longer sessions, such as those in Tmall, users may experience interest drift, with multiple preference shifts emerging throughout the session. Multiple attention heads are more effective in capturing these varying interests. In shorter sessions, like those in Diginetica, a single attention head is sufficient to capture the main behavioral pattern.

Table 2: Performances of all comparison methods on three datasets
<table><tr><td>Dataset</td><td colspan="4">Tmall</td><td colspan="4">RetailRocket</td><td colspan="4">Diginetica</td></tr><tr><td>Method</td><td>P@10</td><td>M@10</td><td>P@20</td><td>M@20</td><td>P@10</td><td>M@10</td><td>P@20</td><td>M@20</td><td>P@10</td><td>M@10</td><td>P@20</td><td>M@20</td></tr><tr><td>FPMC(WWW&#x27;10)</td><td>13.10</td><td>7.12</td><td>16.06</td><td>7.32</td><td>25.99</td><td>13.38</td><td>32.37</td><td>13.82</td><td>15.43</td><td>6.20</td><td>26.53</td><td>6.95</td></tr><tr><td>GRU4Rec(ICLR&#x27;16)</td><td>9.47</td><td>5.78</td><td>10.93</td><td>5.89</td><td>38.35</td><td>23.27</td><td>44.01</td><td>23.67</td><td>17.93</td><td>7.33</td><td>29.45</td><td>8.33</td></tr><tr><td>NARM(CIKM&#x27;17)</td><td>19.17</td><td>10.42</td><td>23.30</td><td>10.70</td><td>42.07</td><td>24.88</td><td>50.22</td><td>24.59</td><td>35.44</td><td>15.13</td><td>49.70</td><td>16.17</td></tr><tr><td>STAMP(SIGKDD&#x27;18)</td><td>22.63</td><td>13.12</td><td>26.47</td><td>13.36</td><td>42.95</td><td>24.61</td><td>50.96</td><td>25.17</td><td>33.98</td><td>14.26</td><td>45.64</td><td>14.32</td></tr><tr><td>SR-GNN(AAAI&#x27;19)</td><td>23.41</td><td>13.45</td><td>27.57</td><td>13.72</td><td>43.21</td><td>26.07</td><td>50.32</td><td>26.57</td><td>36.86</td><td>15.52</td><td>50.73</td><td>17.59</td></tr><tr><td>GCE-GNN(SIGIR’20)</td><td>28.01</td><td>15.08</td><td>33.42</td><td>15.42</td><td>48.22</td><td>28.36</td><td>55.78</td><td>28.72</td><td>41.16</td><td>18.15</td><td>54.22</td><td>19.04</td></tr><tr><td>S2-DHCN(AAAI&#x27;21)</td><td>26.22</td><td>14.60</td><td>31.42</td><td>15.05</td><td>46.15</td><td>26.85</td><td>53.66</td><td>27.30</td><td>39.87</td><td>17.53</td><td>53.18</td><td>18.44</td></tr><tr><td>MGIR(SIGIR’22)</td><td>30.71</td><td>17.03</td><td>36.31</td><td>17.42</td><td>47.90</td><td>28.68</td><td>55.35</td><td>29.20</td><td>40.63</td><td>17.86</td><td>53.73</td><td>18.77</td></tr><tr><td>MTAW(SIGIR’23)</td><td>31.67</td><td>18.90</td><td>37.17</td><td>19.14</td><td>48.41</td><td>29.96</td><td>56.39</td><td>30.52</td><td></td><td></td><td></td><td></td></tr><tr><td>MSGAT(CIKM’23)</td><td>39.21</td><td>20.92</td><td>45.43</td><td>21.35</td><td>57.00</td><td>32.73</td><td>63.68</td><td>33.21</td><td>57.09</td><td>26.30</td><td>66.97</td><td>26.91</td></tr><tr><td>HICN(SDM&#x27;24)</td><td>31.31</td><td>18.90</td><td>35.48</td><td>19.17</td><td>49.74</td><td>29.81</td><td>57.85</td><td>30.37</td><td></td><td></td><td></td><td></td></tr><tr><td>Mssen(LREC-COLING’24)</td><td>33.53</td><td>18.98</td><td>38.51</td><td>19.60</td><td></td><td></td><td></td><td></td><td>42.33</td><td>19.88</td><td>55.17</td><td>19.64</td></tr><tr><td>MGCOT</td><td>41.28</td><td>21.97</td><td>47.80</td><td>22.40</td><td>57.57</td><td>33.44</td><td>63.78</td><td>33.89</td><td>58.04</td><td>29.07</td><td>68.31</td><td>29.79</td></tr><tr><td>Improv.(%)</td><td>5.28</td><td>5.02</td><td>5.22</td><td>4.92</td><td>1.00</td><td>2.17</td><td>0.16</td><td>2.05</td><td>1.66</td><td>10.53</td><td>2.00</td><td>10.70</td></tr></table>

Table 3: Ablation study of components in MGCOT.
<table><tr><td>Dataset</td><td colspan="2">Tmall</td><td colspan="2">RetailRocket</td><td colspan="2">Diginetica</td></tr><tr><td>Method</td><td>P@20</td><td>M@20</td><td>P@20</td><td>M@20</td><td>P@20</td><td>M@20</td></tr><tr><td>-NeighborSessions</td><td>28.69</td><td>14.73</td><td>54.06</td><td>29.63</td><td>52.07</td><td>18.50</td></tr><tr><td>-MultiAttention</td><td>46.85</td><td>21.52</td><td>62.89</td><td>33.25</td><td>67.87</td><td>29.47</td></tr><tr><td>-ContrastiveLearning</td><td>47.60</td><td>21.87</td><td>63.17</td><td>33.81</td><td>67.84</td><td>29.31</td></tr><tr><td>MGCOT</td><td>47.80</td><td>22.40</td><td>63.78</td><td>33.89</td><td>68.31</td><td>29.79</td></tr></table>

![](images/a369579996629b23c1e1a85d338cae804721ae5c713b7d2219067307540c125b.jpg)

![](images/cdf266c6ee5aa0b57cd0908b8f1070c576bb0d0c67544a77df641ae898117898.jpg)  
Figure 5: The weight of contrastive loss $\beta$

In Figure 5, the contrastive loss weight is set to 0.05 for Tmall and 5 for RetailRocket and Diginetica. In the longer sessions of Tmall, users may frequently compare or select similar items, attention mechanisms and similar session fusion more effectively capture user intent, making a lower contrastive loss weight beneficial. In the shorter sessions of RetailRocket and Diginetica, higher contrastive loss weights help generate more comprehensive session representations by capturing intent from different views.

Figure 4: The number of attention heads h  
![](images/00b3edcc135cfbc8fff1981dc928b3ee3d1294016877c832c981a04c3bd2d4e2.jpg)

![](images/88ecbbe846ec1503e92c58881770b12c536bbfd7cfc4108ed813dd8b4f81c6c0.jpg)

![](images/5e1e1c79e45d01248ca0a8ed548489d941f9182583c7356d8e65fed45c217ab6.jpg)  
Figure 6: Multi-head attention visualization

## 4.7 Further Experiments

In this section, we analyze the model’s attention distribution and results for short and long sessions.

As shown in Figure 6, the importance of items is represented by the depth of color, with darker colors indicating higher importance. Based solely on the existing session sequence, it is difficult to directly determine the relationship between session items and the target item. However, through the visualization of the multi-head attention mechanism, we observe that the attention weight distribution varies across different sessions, reflecting the varying contributions of items in capturing user intent. When the multi-head attention mechanism is removed, the evaluation metrics on the RetailRocket dataset show a significant decline, further highlighting the critical role of attention in capturing user intent. Therefore, beyond emphasizing the last item in the session, it is essential to dynamically learn and evaluate the influence of items at different positions on session intent.

![](images/4d1445d4b498f3255982fbf48e8aba945c0fa80531a1c65934143048a017a570.jpg)

![](images/1132d2c711648bc58d8ba32bb3c41697ff01b498201a4256915d7533b2e5fe72.jpg)  
Figure 7: P@20 results on short and long sessions.

We divide the Tmall and RetailRocket datasets into short and long sessions, with short sessions having 5 or fewer items and long sessions exceeding 5. We compare MGCOT with several representative baseline models, including SR-GNN, GCE-GNN, DHCN, and MSGAT. As shown in Figure 7, MGCOT consistently outperforms these baselines across all session lengths, demonstrating its effectiveness in real-world session-based recommendation tasks.

## 5 Conclusion

This paper introduces the MGCOT model, which builds multiple graphs to capture the session intent from current, local, and global views. By integrating attention mechanisms, MGCOT effectively captures important information, while incorporating contrastive learning to generate more comprehensive and complementary session representations. Extensive experiments on three datasets demonstrate that our MGCOT model outperforms current SOTA models, validating its effectiveness in session-based recommendation tasks.

## 6 Limitation

The MGCOT model has several limitations. First, the construction of multiple graphs increases the storage space requirements. Second, the complexity of building self-supervised contrastive learning models leads to limited transferability and bulky model structures.

## Acknowledgments

This work was supported by the NSFC (62376180, 62176175, 62302329), the major project of natural science research in Universities of Jiangsu Province (21KJA520004), Suzhou Science and Technology Development Program (SYG202328, SKY2023128) and the Project Funded by the Priority Academic Program Development of Jiangsu Higher Education Institutions.

## References

Qian Chen, Zhiqiang Guo, Jianjun Li, and Guohui Li. 2023. Knowledge-enhanced multi-view graph neural networks for session-based recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 352–361.

Xiaoru Chen, Yingxu Wang, Jinyuan Fang, Zaiqiao Meng, and Shangsong Liang. 2024. Heterogeneous graph contrastive learning with metapath-based augmentations. IEEE Transactions on Emerging Topics in Computational Intelligence, 8(1):1003–1014.

Minjin Choi, Hye-young Kim, Hyunsouk Cho, and Jongwuk Lee. 2024. Multi-intent-aware sessionbased recommendation. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2532–2536.

Qilong Han, Chi Zhang, Rui Chen, Riwei Lai, Hongtao Song, and Li Li. 2022. Multi-faceted global item relation learning for session-based recommendation. In Proceedings ofthe 45th international ACM SIGIR conference on research and development in information retrieval, pages 1705–1715.

Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939.

Jing Li, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Tao Lian, and Jun Ma. 2017. Neural attentive sessionbased recommendation. In Proceedings ofthe 2017 ACM on Conference on Information and Knowledge Management, pages 1419–1428.

Qiao Liu, Yifu Zeng, Refuoe Mokhosi, and Haibin Zhang. 2018. Stamp: short-term attention/memory priority model for session-based recommendation. In Proceedings ofthe 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1831–1839.

Xiao Liu, Fanjin Zhang, Zhenyu Hou, Li Mian, Zhaoyu Wang, Jing Zhang, and Jie Tang. 2021. Selfsupervised learning: Generative or contrastive. IEEE transactions on knowledge and data engineering, 35(1):857–876.

Kai Ouyang, Xianghong Xu, Miaoxin Chen, Zuotong Xie, Hai-Tao Zheng, Shuangyong Song, and Yu Zhao. 2023. Mining interest trends and adaptively assigning sample weight for session-based recommendation. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2174–2178.

Zhiqiang Pan, Fei Cai, Wanyu Chen, Chonghao Chen, and Honghui Chen. 2022. Collaborative graph learning for session-based recommendation. ACM Transactions on Information Systems (TOIS), 40(4):1–26.

Shutong Qiao, Wei Zhou, Junhao Wen, Hongyu Zhang, and Min Gao. 2023. Bi-channel multiple sparse graph attention networks for session-based recommendation. In Proceedings ofthe 32nd ACM International Conference on Information and Knowledge Management, pages 2075–2084.

Ruihong Qiu, Jingjing Li, Zi Huang, and Hongzhi Yin. 2019. Rethinking the item order in session-based recommendation with graph neural networks. In Proceedings of the 28th ACM international conference on information and knowledge management, pages 579–588.

Pengjie Ren, Zhumin Chen, Jing Li, Zhaochun Ren, Jun Ma, and Maarten de Rijke. 2019. Repeatnet: A repeat aware neural recommendation machine for session-based recommendation. Proceedings of the AAAI Conference on Artificial Intelligence, 33(1):4806–4813.

Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized markov chains for next-basket recommendation. In Proceedings ofthe 19th international conference on World wide web, pages 811–820.

Tianqi Sun, Hongrui Guo, Zihan Zhang, Hongzhi Liu, and Zhonghai Wu. 2024. Exploiting multifaceted nature of items and users for session-based recommendation. In Proceedings ofthe 2024 SIAM Inter national Conference on Data Mining (SDM), pages 580–588.

Meirui Wang, Pengjie Ren, Lei Mei, Zhumin Chen, Jun Ma, and Maarten De Rijke. 2019. A collaborative session-based recommendation approach with parallel memory modules. In Proceedings ofthe 42nd international ACM SIGIR conference on research and development in information retrieval, pages 345– 354.

Ziyang Wang, Wei Wei, Gao Cong, Xiao-Li Li, Xian-Ling Mao, and Minghui Qiu. 2020. Global context enhanced graph neural networks for session-based recommendation. In Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval, pages 169–178.

Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan. 2019. Session-based recommendation with graph neural networks. Proceedings of the AAAI Conference on Artificial Intelligence, 33(1):346–353.

Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S Yu Philip. 2020. A comprehensive survey on graph neural networks. IEEE transactions on neural networks and learning systems, 32(1):4–24.

Xin Xia, Hongzhi Yin, Junliang Yu, Yingxia Shao, and Lizhen Cui. 2021a. Self-supervised graph co-training for session-based recommendation. In Proceedings of the 30th ACM international conference on information & knowledge management, pages 2180–2190.

Xin Xia, Hongzhi Yin, Junliang Yu, Qinyong Wang, Lizhen Cui, and Xiangliang Zhang. 2021b. Selfsupervised hypergraph convolutional networks for session-based recommendation. Proceedings of the AAAI Conference on Artificial Intelligence, 35(5):4503–4511.

Chengfeng Xu, Pengpeng Zhao, Yanchi Liu, Victor S. Sheng, Jiajie Xu, Fuzhen Zhuang, Junhua Fang, and Xiaofang Zhou. 2019. Graph contextualized selfattention network for session-based recommendation. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19, pages 3940–3946.

Xiangping Zheng, Bo Wu, Alex X Zhang, and Wei Li. 2024. Hypergraph-based session modeling: A multi-collaborative self-supervised approach for enhanced recommender systems. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 8493–8504.

Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In Proceedings ofthe 29th ACM international conference on information & knowledge management, pages 1893–1902.