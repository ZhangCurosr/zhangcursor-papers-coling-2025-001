# Hawkes based Representation Learning for Reasoning over Scale-free Community-structured Temporal Knowledge Graphs

Yuwei Du, Xinyue Liu\*, Wenxin Liang, Linlin Zong, Xianchao Zhang School of Software, Dalian University of Technology, China {xyliu, wxliang, llzong, xczhang}@dlut.edu.cn duyuwei2023@mail.dlut.edu.cn

## Abstract

Temporal knowledge graph (TKG) reasoning has become a hot topic due to its great value in many practical tasks. The key to TKG reasoning is modeling the structural information and evolutional patterns of the TKGs. While great efforts have been devoted to TKG reasoning, the structural and evolutional characteristics of real-world networks have not been considered. In the aspect of structure, real-world networks usually exhibit clear community structure and scale-free (long-tailed distribution) properties. In the aspect of evolution, the im pact of an event decays with the time elapsing. In this paper, we propose a novel TKG reasoning model called Hawkes process-based Evolutional Representation Learning Network (HERLN), which learns structural information and evolutional patterns of a TKG simultaneously, considering the characteristics of realworld networks: community structure, scalefree and temporal decaying. First, we find communities in the input TKG to make the encoding get more similar intra-community embeddings. Second, we design a Hawkes processbased relational graph convolutional network to cope with the event impact-decaying phe nomenon. Third, we design a conditional decoding method to alleviate biases towards fre quent entities caused by long-tailed distribution. Experimental results show that HERLN achieves significant improvements over the state-of-the-art models.

## 1 Introduction

Temporal Knowledge Graph (TKG) is a dynamic multi-relational graph used to record evolutionary events and knowledge in the real world. TKGs indicate facts as quadruples (subject, relation, object, time) and are actually sequences of temporal subgraphs divided by the time (timestamp) dimension. Reasoning over TKGs aims to infer the missing quadruple facts, which has two settings: interpolation and extrapolation. Given a TKG with timestamps from $t _ { 0 }$ to $t _ { K }$ , interpolation aims at inferring missing facts that occur at time t, where $t _ { 0 } < t < t _ { K }$ . Oppositely, extrapolation attempts to predict facts that occur at time t with $t > t _ { K }$ . In this paper, we focus on the extrapolation setting, which has gained much attention in recent years due to its great practical value in event prediction, question answering, and other areas (Wang et al., 2019b; Lan and Jiang, 2020). There are mainly two extrapolation tasks: entity prediction and relation prediction. We aim to propose a unified model that can accomplish both the entity and relation prediction tasks.

The key to TKG reasoning is modeling the structural information and evolutional patterns of the TKGs. The prior extrapolation TKG reasoning models such as CyGNet (Zhu et al., 2021) and CENET (Yi et al., 2023), learn the evolutional patterns by generating the historical event vocabulary to predict repetitive events. Later models such as RE-GCN (Li et al., 2021), HisMatch (Li et al., 2022) and HGLS (Zhang et al., 2023b), employ a relational graph convolutional network (RGCN) (Schlichtkrull et al., 2018) to capture the structural information from historical snapshots and use a recurrent neural network (RNN) to model the evolutional patterns. Some recent works such as TITer (Sun et al., 2021) and DREAM (Zheng et al., 2023) introduce reinforcement learning on the TKG reasoning task.

Nevertheless, while TKGs are reflections of the real world, the structural and evolutional characteristics of real-world networks have not been considered in previous models. In the aspect of structure, real-world networks (e.g., social-networks) usually exhibit clear community structure and scale-free (long-tailed distribution) properties (Barabási and Albert, 1999). In the aspect of evolution, the impact of an event decays with the time elapsing (Hawkes,

1971). Taking these characteristics into consideration not only can improve the reasoning performance, but also better facilitate the down-stream tasks.

In this paper, we propose a novel TKG reasoning model called Hawkes process-based Evolutional Representation Learning Network (HERLN), which learns structural information and evolutional patterns of a TKG simultaneously, considering the characteristics from real-world networks: community structure, scale-free and temporal decaying. Specifically, our model consists of three modules: an embedding initializing module, an evolution encoding module and a conditional decoding module.

In the embedding initializing module, to exploit the community structure properties of TKGs, we first find communities in the input TKG, and apply a graph convolution network to get embeddings of events within each community. The embeddings are then used as inputs in the evolution encoding module, which make the evolution encoding module output more similar intra-community embeddings. In the evolution encoding module, to cope with the event impact-decaying phenomenon, we design a Hawkes process-based relational graph convolutional network (HRGCN). The graph convolutional network contracts the structural information of the TKG, while the Hawkes process assigns different weights to the timestamps such that the impacts of events decay over time. In the conditional decoding module, to alleviate biases towards frequent entities caused by long-tailed distribution, we construct a conditional decoder which consists of a hyper network and a query-specific decoder. The hyper network adjusts the parameters according to the query events and the decoder generates conditional intensity scores for the candidate entities based on the adjusted parameters.

The main contributions of this work are summarized as follows:

• We recognize that the TKGs possess the structural and evolutional characteristics inherited from real-world networks: community structure, scale-free and temporal decaying, but they have not been considered or well exploited in previous studies.

• We propose a Hawkes process-based evolutional representation learning network (HERLN), which consists of three modules: (1) An embedding initialize module, which extracts semantic information of community structure; (2) An evolution encoding module, which addresses the temporal decaying of event impact; (3) A conditional decode module, which alleviates the biases towards frequent entities caused by long-tailed distribution.

• Our proposed model HERLN can predict entities and relations at the same time. Experimental results on four benchmark TKG datasets show that HERLN achieves significant improvements over the state-of-the-art models.

## 2 Related work

## 2.1 Embedding-based methods

Embedding-based methods encode the whole or part of the TKGs to obtain the embeddings of entities and relations, and use the embeddings to evaluate the possibility of missing facts.

RE-NET (Jin et al., 2020) proposes an autoregressive architecture which uses a graph neural network (GNN) to capture local entity embeddings and a RNN to model interactions between entities over time. RE-GCN (Li et al., 2021) constructs a static graph to get the static attributes and presents a framework that can execute both entity and relation reasoning.

EvoKG (Park et al., 2022) uses an autoregressive architecture and captures the everchanging structural and temporal dynamics via recurrent event modeling. HiSMatch (Li et al., 2022) generates background graphs, entity-related graphs and relation-related graphs to jointly model the evolutional patterns. HGLS (Zhang et al., 2023b) models local snapshots or global graphs by using different GNNs and decodes them to get the predicting scores. CENET (Yi et al., 2023) combines historical and non-historical information and identifies highly related entities via contrastive learning. TARGAT (Xie et al., 2023) captures the interactions of multi-facts at different timestamps. DLGR (Xiao et al., 2024) learns the local and global perspective representations in a contrastive manner. DSTKG (Li et al., 2024) introduces two latent variables to capture the dynamic and static characteristics of entities in TKGs.

L<sup>2</sup>TKG (Zhang et al., 2023a) finds the missing relationships on the known KGs first and then reasons on the completed graph and the original graph jointly. RETIA (Liu et al., 2023) constructs a hyper-graph to connect different relations in a high-dimensional space. These models use RNNs to represent the temporal information. Thus they are based on an assumption that the temporal sequences are equidistant, which is inconsistent with many real-life event sequences (Sun et al., 2022).

## 2.2 Path-based methods

Path-based methods find several related paths of query facts and select the most relevant one as the answer. TITer (Sun et al., 2021) adopts reinforcement learning to sample actions from queryrelated trajectories based on a time-shaped reward function. xERTE (Han et al., 2021a) samples and prunes the query-related subgraph according to query-dependent attention scores. TANGO (Han et al., 2021b) explores the neural ordinary differential equation to build a continuous-time model. TLogic (Liu et al., 2022) automatically mines recurrent temporal logic rules by extracting temporal random walks. DREAM (Zheng et al., 2023) use generative adversarial networks to design an adaptive reward function. However, the path-based methods focus on the local structure graph of the query, ignore the potential connection of events, and do not perform well on long-term reasoning.

In addition, it is worth noting that methods such as xERTE (Han et al., 2021a) and HISMatch(Li et al., 2022) consider the impact of temporal information on prediction results. They encode timestamps and concat timestamps with entity embeddings. However, different from our work, they actually learns that entities with different time intervals have different impacts on results, rather than considering the gradual decay of event impacts.

## 2.3 Hawkes process-based methods

The Hawkes process (Hawkes, 1971) is a stochastic process that models sequential discrete events occurring in continuous time. There are several works that combine the Hawkes process and neural networks for TKG reasoning. Know-Evolve (Trivedi et al., 2017) introduces a temporal point process to model facts evolved in the continuous time domain. GHNN (Han et al., 2020) proposes a graph Hawkes process to capture the potential temporal dependence across different timestamps. However, Know-Evolve and GHNN do not use the graph structural information. GHT (Sun et al., 2022) uses a temporal Transformer to capture long-term and short-term information jointly. However, none of the previous work has considered problem of temporal decaying of events’ impacts. Our proposed module uses the Hawkes process to assign different weights to the timestamps during message passing, thus the information of the event impact-decaying is encoded and utilized.

## 3 Problem Formulation

A temporal knowledge graph (TKG) ${ \cal G } \mathrm { ~  ~ { ~ = ~ } ~ }$ $\{ \mathcal { E } , \mathcal { R } , \mathcal { T } , \mathcal { F } \}$ is a directed multi-relational graph, where E, R, T and $\mathcal { F }$ denote the sets of entities, relations, timestamps and facts, respectively. A node in $G$ represents an entity $i \in \mathcal { E }$ , and an edge $e _ { i j }$ represents the interaction between node i and node $j$ with relation $r \in \mathcal { R }$ at timestamp $t \in \mathcal { T } . \mathrm { A }$ fact in $G$ is a quadruple $\boldsymbol { q } = \left( s , r , o , t \right)$ that represents a real-world event consisting of the relation r between a subject entity s and an object entity o at timestamp t.

Given a TKG $G _ { [ t _ { 1 } : t _ { k } ] } = \{ \mathcal { E } , \mathcal { R } , \mathcal { T } , \mathcal { F } | \mathcal { T } =$ $[ t _ { 1 } , t _ { k } ] \}$ , the extrapolation reasoning task is to predict object $o _ { q }$ in a query like $\left( s _ { q } , r _ { q } , ? , t _ { q } \right)$ where $t _ { q } ~ > ~ t _ { k }$ , or predict relation $r _ { q }$ in a query like $( s _ { q } , ? , o _ { q } , t _ { q } )$ where $t _ { q } > t _ { k }$

## 4 Method

The proposed Hawkes process-based evolutional representation learning network (HERLN) model is shown in Fig. 1, it consists of three modules, an embedding initializing module, an evolution encoding module and a conditional decoding module. The embedding initializing module detects communities in the input TKG and embeds the interaction frequencies between entities within each community into the initialized entity embeddings. With the initialized entity embeddings as input, the evolution encoding module updates the entity embeddings by learning the structural information from historical graph. The conditional decoding module uses representations of query quadruple to adjust the parameters and generate scores for candidates.

## 4.1 Embedding Initializing Module

To initialize the embeddings of entities, we first identify the communities, and then extract semantic information in the communities to get the initialized entity embedding matrix $\mathrm { H } _ { C }$

## 4.1.1 Community Detection

The interactions between entities in real world exhibit distinct community structures. Exploiting of the community structural properties contributes to the improvement of reasoning performance. For example, an entity that cooperates with a country like America is more likely to be a government or an organization rather than the citizens of a country. This information could be used to reduce the scores of entities that are not consistent with the facts.

![](images/6b9da6f2da829765baf0899adff03a40413bd89ce5d6c89e6d649889e5b5f417.jpg)  
Figure 1: Overall framework of the proposed HERLN model. HERLN is consists of three modules: an embedding initializing module, an evolution encoding module and a conditional decoding module. First, the embedding initializing module extracts the community structural information in input TKGs to get the initialized embeddings. Then the evolution encoding module updates the embeddings with the candidate-related historical structure to learn the evolutional patterns of events. Finally, the conditional decoding module reasons according to the embeddings and gets scores of the candidate entities, then select the entity with the highest score as the results. The input TKG contains 6 nodes (marked as A to F), three different types of edges (indicated in red, blue and green respectively) and three timestamps (from $t _ { 1 } \ t _ { 0 } t _ { 3 } )$ . Additionally, the model receives an incomplete quadruple as query. The query given in the figure is $\left( A , r _ { 3 } , ? , t _ { 3 } \right)$

We use a community detection algorithm on the entire TKG, which divides the entities set into different communities according to interaction frequencies, and obtains a graph that only contains inner-community links. The algorithm assigns each entity i to its community $c _ { i } .$ , and there are a total of K communities in the TKG.

In TKGs, there are no inherent community labels. Therefore, we require an unsupervised and reliable method to detect possible communities in the TKGs. And since a TKG is a multi-relational graph, we extend the Louvain algorithm (Blondel et al., 2008) to handle with the multi-relational links. Specifically, we calculate modularity $Q _ { r }$ for different relation r by Eq. 1, which is an indicator that measures the quality of community detection.

$$
Q _ { r } = \sum _ { c } [ \frac { \sum _ { i n } } { 2 m } - ( \frac { \sum _ { } t o t } { 2 m } ) ^ { 2 } ] = \sum _ { c } [ e _ { c } - a _ { c } ^ { 2 } ]\tag{1}
$$

where $\sum i n$ is the sum of weights of innercommunity edges with relation $r ; \sum t o t$ is the sum of weights of all the edges with relation r and m is the total weight of edges on the whole graph.

During the optimization process, when a community is merged into another community, the algorithm will calculate the modularity of the new entire graph, compare it with the modularity before merging to get $\Delta Q$ as described in Eq. 2.

$$
\begin{array} { l } { { \displaystyle \Delta Q = \sum _ { r } \Delta Q _ { r } } } \\ { { \displaystyle ~ = \sum _ { r } \left( [ \frac { \sum { i n + k _ { i , i n } } } { 2 m } - ( \frac { \sum { t o t + k _ { i } } } { 2 m } ) ^ { 2 } ] \right. } } \\ { { \displaystyle ~ \left. - [ \frac { \sum { i n } } { 2 m } - ( \frac { \sum { t o t } } { 2 m } ) ^ { 2 } - ( \frac { k _ { i } } { 2 m } ) ^ { 2 } ] \right) } } \\ { { \displaystyle ~ = \frac { 1 } { 2 m } ( k _ { i , i n } - \frac { \sum _ { t o t } k _ { i } } { m } ) } } \end{array}\tag{2}
$$

where $k _ { i , i n }$ is the sum of the edges’ weights between node i and the new community in; $k _ { i }$ is the

sum of the edges’ weights between node i and all the nodes in the graph. The algorithm ends when $\Delta Q$ no longer changes.

## 4.1.2 Embedding Initialization

In order to import the information contained in the communities into the embeddings, we use a GCN to generate embeddings on the community subgraphs, which is formalized as:

$$
h _ { i } = \sigma \Big ( \sum _ { j \in \mathcal { N } _ { i } } \frac { 1 } { | \mathcal { N } _ { i } | } W h _ { j } ^ { i n i t } \delta ( c _ { i } , c _ { j } ) + W _ { 0 } h _ { i } ^ { i n i t } \Big )\tag{3}
$$

where $h _ { i } ^ { i n i t }$ and $h _ { \it j } ^ { i n i t }$ are randomly initialized embeddings of nodes $i , j ;$ W is the parameter of message passing between nodes; $W _ { 0 }$ is the parameter of self updating of a node; $\delta ( c _ { i } , c _ { j } )$ is an indicator which is set to 1 if i and j belong to a same community and 0 otherwise; $\sigma ( )$ is an activate function; ${ \mathcal { N } } _ { i }$ is the set of neighbors of node i.

## 4.2 Evolution Encoding

After getting the initialized embedding matrix, the next move is to encode the candidate-related historical structure to learn the evolutional patterns of events. In real-world events, the entity’s status changes over time. Another common real-world phenomenon is that the impact of a event decays over time, The entity embeddings should be able to cope with the variations. This module achieves these points by updating the entity embeddings with historical information.

## 4.2.1 The Hawkes Process on TKGs

The Hawkes process is a stochastic process that models a sequential discrete events that occur chronologically, which is typically modeled by a conditional intensity function. The intensity function $\lambda ( t )$ represents the probability that events happen at t, it is defined as follows:

$$
\lambda _ { ( s , r , o ) } ( t ) = \sum _ { ( s ^ { \prime } , r ^ { \prime } , o , t ^ { \prime } ) \in \mathcal { F } _ { o } } \gamma _ { s ^ { \prime } , r ^ { \prime } } ( t ^ { \prime } ) \kappa ( t - t ^ { \prime } )\tag{4}
$$

where $\mu _ { s , r , o } ( t )$ is the base intensity at time t; $\mathcal { F } _ { o }$ is the set of historical events related to node $o ; \gamma ( \dot { ) }$ represents the amount of excitement induced by the corresponding events on results and $\kappa ( )$ is a kernel function to model the effect of historical events on the current event.

To integrate the Hawkes Process into the TKG, we treat the Hawkes Process as an encoder-decoder network. For encoding, we use a function $E n ( )$ to get representations $h _ { s } , h _ { r }$ and $h _ { o }$ derived from their historical neighbors’ information, as shown in $\operatorname { E q }$ 5. For decoding, we transfer the representations into a certain intensity value with an appropriate decoding function $D e ( )$ , as shown in Eq. 6.

$$
h _ { s } , h _ { r } , h _ { o } = E n ( G )\tag{5}
$$

$$
\lambda _ { ( s , r , o ) } ( t ) = D e ( h _ { s } , h _ { r } , h _ { o } )\tag{6}
$$

We present detailed implementations of $E n ( )$ and $D e ( )$ in the following sections.

## 4.2.2 Encoding with Hawkes process-based RGCN

To get the historical representation of entities described in Eq.5, we design a Hawkes process-based relation graph convolutional network (HRGCN) as $E n ( )$ to pass message and update entity embeddings on the TKG.

As shown in Fig. 2, the improvement of HRGCN over traditional RGCN is that HRGCN can effectively learn the structural information from neighbors and assign weights of the messages which represents the temporal decaying of these messages, $\mathrm { i . e . }$

$$
h _ { o } ^ { t } = \sigma \Bigg ( W _ { 1 } h _ { o } + \sum _ { s , r , t ^ { \prime } \in \mathcal { F } _ { o } ^ { t } } \frac { 1 } { | \mathcal { F } _ { o } ^ { t } | } W _ { r } ( h _ { s } + h _ { r } ) \widetilde { \kappa } ( t - t ^ { \prime } ) \Bigg )\tag{7}
$$

where $h _ { s } , h _ { o }$ and $h _ { r }$ are input embeddings of nodes $s , o$ and edge r got from $\mathrm { H } _ { C }$ and R, respectively; $\mathcal { F } _ { o } ^ { t }$ represents historical neighbors of node o, which contains all ${ \tt q u a d r u p l e s } = ( s , r , o , t ^ { \prime } )$ where $t ^ { \prime } < t ;$ $W _ { r }$ is the parameter of message passing between s and $o ; W _ { 1 }$ is the parameter of self updating of node $\begin{array} { r } { o ; \widetilde { \kappa } ( t - t ^ { \prime } ) = \frac { \kappa ( t - t ^ { \prime } ) } { \sum _ { s , r , o , t ^ { \prime \prime } \in \mathcal { F } _ { o } ^ { t } } \kappa ( t - t ^ { \prime \prime } ) } } \end{array}$ is the temporal decaying function, where $\overset { \vartriangle } { \kappa } ( t - t ^ { \prime } ) = \exp ( - \delta ( t -$ $t ^ { \prime } ) )$ .

This module outputs the updated embedding matrix $\mathrm { H } _ { T } ^ { t }$ that contains historical information.

## 4.2.3 Gating Integration

The learned embeddings by HRGCN get the historical information of entities, while some useful original information directly from the input TKG may be overwritten. We use a control gate to balance the contribution of the two kinds of information.

Specifically, we employ a fully connected layer to generate a graph embedding $h _ { g }$ according to the embeddings $\mathrm { H } _ { T } ^ { t }$ from HRGCN. Then we use another fully connected layer to calculate the weight of $\mathrm { H } _ { T } ^ { t }$ according to graph embedding $h _ { g }$ . The final embedding matrix is a weighted sum of $\mathrm { H } _ { T } ^ { t }$ and the initialized embedding matrix $\mathrm { H } _ { C }$

![](images/e832fc1f36bf12fff24e693a92889f13bb8d37e0fdf2033bc1a62fe52001d41c.jpg)  
Figure 2: The update process of HRGCN. (a) is a historical structure of node A, (b) shows the update process of a traditional RGCN, which does not use timestamp information and (c) shows the update process of HRGCN, which takes the time interval as decaying weight to represent the event declines over time.

$$
h _ { g } = \sigma ( W _ { g r a p h } \mathrm { H } _ { T } ^ { t } + b _ { g r a p h } )\tag{8}
$$

$$
\gamma = \sigma ( W _ { g a t e } h _ { g } + b _ { g a t e } )\tag{9}
$$

$$
\mathrm { H } ^ { t } = \gamma * \mathrm { H } _ { T } ^ { t } + ( 1 - \gamma ) \mathrm { H } _ { c }\tag{10}
$$

where $h _ { g }$ is the generated graph embedding, $W _ { g r a p h } , W _ { g a t e } , b _ { g r a p h }$ and $b _ { g a t e }$ are learnable parameters.

The output of the encoding module is the balanced evolutional embedding matrix $\mathrm { H } ^ { t }$ that encodes the evolutional information on graph $G _ { [ t _ { 1 } , t ] } .$

## 4.3 Conditional Decoding

The last step of our model is to do reasoning according to the embedding matrix and get the conditional intensity scores of the candidate entities, then select the entity with the highest score as the result, as described in $\mathrm { E q } . 6$

## 4.3.1 Previous RGCN Decoding

Existing works (Li et al., 2021, 2022) use ConvTransE as the decoding function $D e ( )$ for traditional RGCN to calculate the certain intensity values $\lambda _ { o }$ of entity o by Eq. 11.

$$
\lambda _ { o } ^ { o r i } = \mathrm { C o n v T r a n s E } ( c o n c a t ( h _ { s _ { q } } ^ { t _ { q } } , h _ { r _ { q } } ) ; h _ { o _ { q } } ^ { t _ { q } } )\tag{11}
$$

Note that the parameters of ConvTransE are fixed over different queries, which leads the model to tend to reason using the few evolutional patterns of the most common events, causing bias against other evolutional patterns of the non-common events. We adjust ConvTransE to avoid the bias caused by long-tailed distribution in the following.

## 4.3.2 Avoiding Long-tailed Distribution Bias

We use feature linear modulation (FiLM) to construct a hyper-network, which adjusts parameters of decoder according to different queries, so that it can choose the appropriate query-specific evolutional pattern for reasoning.

Specifically, given a query quadruple = $( s _ { q } , r _ { q } , o _ { q } , t _ { q } )$ , the hyper-network generates a shifting factor $\alpha ^ { \prime } \dot { }$ and a scaling factor $\beta ^ { ( s _ { q } , r _ { q } , t _ { q } ) }$ according to the vector of the query quadruple to scale and shift the decoding parameters.

$$
\alpha ^ { ( s _ { q } , r _ { q } , t _ { q } ) } = \sigma ( ( h _ { s _ { q } } ^ { t _ { q } } | | h _ { r _ { q } } ) W _ { \alpha } + b _ { \alpha } )\tag{12}
$$

$$
\beta ^ { ( s _ { q } , r _ { q } , t _ { q } ) } = \sigma ( ( h _ { s _ { q } } ^ { t _ { q } } | | h _ { r _ { q } } ) W _ { \beta } + b _ { \beta } )\tag{13}
$$

$$
\theta ^ { q } = ( \alpha ^ { ( s _ { q } , r _ { q } , t _ { q } ) } + 1 ) \odot \theta + \beta ^ { ( s _ { q } , r _ { q } , t _ { q } ) }\tag{14}
$$

where $h _ { s } ^ { t _ { q } }$ and $h _ { r _ { q } }$ are embeddings of subject $s _ { q }$ at time $t _ { q }$ and relation $r _ { q }$ respectively; $W _ { \alpha } , W _ { \beta } , b _ { \alpha }$ and $b _ { \beta }$ are learnable parameters; θ is the original parameters from decoder and $\theta ^ { q }$ is query-specific parameters; ⊙ is Harmard product.

## 4.3.3 Adjusted Decoding

The adjusted decoder extracts the multidimensional features of the query quadruple through one-dimensional convolution and gets the conditional intensity scores of candidate entities via the inner product with the embedding matrix.

$$
\lambda _ { o } = \mathrm { C o n v T r a n s E } ( [ h _ { s _ { q } } ^ { t _ { q } } , h _ { r _ { q } } ] ; h _ { o _ { q } } ^ { t _ { q } } ; \theta ^ { q } )\tag{15}
$$

$\lambda _ { o }$ is the conditional intensity of candidate entity o; [·] means the concat function; $\theta ^ { q }$ is the adjusted parameters got from the feature transform unit. The decoder chooses the entity which has the highest score as the missing part of the query quadruple.

## 4.4 Learning Objective

The task to predict the missing entity of a given quadruple could be seen as a multi-classification task and each entity in the candidate set belongs to one class. The optimization objective of entity prediction task is to maximum the scores of the ground truth entities, which can be convert to a cross-entropy loss $\mathcal { L } _ { e }$

$$
\mathcal { L } _ { e } = - \sum _ { t _ { q } \in \mathcal { T } } \sum _ { \mathcal { F } _ { t _ { q } } } \sum _ { k = 1 } ^ { K } y _ { k } \log p ( o _ { k } | s _ { q } , r _ { q } , t _ { q } )\tag{16}
$$

where $\tau$ is the timestamp set; $\mathcal { F } _ { t _ { q } }$ is the quadruple set with timestamp $t _ { q } ;$ K is the number of entities; $y _ { k } = 1$ if entity $o _ { k }$ equals to ground truth $o _ { q } ,$ otherwise 0; $p ( o _ { k } | s _ { q } , r _ { q } , t _ { q } )$ is the probability of $o _ { k }$ , normalized by the scores of all the candidate entities in the candidate sets.

## 5 Experiments

## 5.1 Experimental Setup

## 5.1.1 Datasets

We use four benchmark datasets which are generally used in TKG reasoning task to evaluate the effectiveness of HERLN, ICEWS14 (Li et al., 2021), ICEWS18 (Li et al., 2021), WIKI (Jin et al., 2020) and YAGO (Jin et al., 2020). ICEWS is a database got from more than 100 data sources over more than 250 countries and regions. ICEWS14 and ICEWS18 datasets contain events occurred in 2014 and 2018 respectively. WIKI and YAGO are subsets of the Wikipedia history and YAGO3 respectively. We list the statistics of these datasets in Appendix A.

## 5.1.2 Baselines

We compare HERLN with 10 TKG reasoning models, which can be categorized into three classes. (1) Embedding-based models, RE-NET (Jin et al., 2020), REGCN (Li et al., 2021), EvoKG (Park et al., 2022), CENET (Yi et al., 2023), HGAT (Shao et al., 2023) and TiPNN (Dong et al., 2024); (2) Path-based models, TG-Tucker (Han et al., 2021b) and TLogic (Liu et al., 2022); (3) Hawkes processbased models, GHNN (Han et al., 2020) and GHT (Sun et al., 2022).

## 5.1.3 Evaluation Metrics

We report MRR, which is the mean of the reciprocal values of the actual missing entities’ ranks averaged by all the queries, and Hits@1/3/10, i.e., the proportion of correct test cases that are ranked within top 1/3/10.

## 5.1.4 Implementation Details

We implement our model in Pytorch (Paszke et al., 2019) and DGL Library (Wang et al., 2019a). The experiments are conducted on a Nvidia GeForce Titan GPU. To be consistent with the baselines, we set the embedding dimension of entities $d _ { e }$ and relations $d _ { r }$ to 200. The number of HRGCN layers is set to 2 and the dropout rate for each layer is set to 0.2. We set all weights of edges to 1 in the embedding initializing module. We use the same hyperparameter settings of ConvTransE given by Li et al. (2021), the decode unit has 50 convolutional kernels with a size of $2 \times 3$ for each kernel. Adam (Kingma and Ba, 2015) is adopted for parameter learning with the learning rate of 0.001 on all the datasets. We report the average experimental results on three random seeds. Our code is available at https://github.com/WisdomMLlab/HERLN.

## 5.2 Experimental Results

## 5.2.1 Entity Prediction

Table 1 shows the entity prediction results on the benchmark datasets. The best results are marked in bold and the second best ones are underlined. It can be seen that our proposed HERLN performs the best nearly on all the settings, an only exception is that it achieves the second best Hits@3 score on ICEWS14. GHNN and GHT combines Hawkes point process with neural networks. But they do not consider the temporal decaying of event impact, which limits their performance. RE-GCN uses RNN on static graphs and loses temporal information during long-term evolution. CENET measures the probability of different entities by constructing the historical vocabulary. It does not exploit the graph structural information. TiPNN focuses on query-aware temporal path to capture the feature and doesn’t take the potential relations between entities into account.

Our model can capture the historical structure and the event evolutional patterns at the same time, considering the community structure, long-tailed distribution, and temporal decaying characteristics, thus it makes significant improvements over the baselines on all the datasets.

<table><tr><td rowspan="2">Method</td><td colspan="4">ICEWS18</td><td colspan="4">ICEWS14</td><td colspan="4">WIKI</td><td colspan="4">YAGO</td></tr><tr><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td></tr><tr><td>GHNN</td><td>27.93</td><td>18.77</td><td>31.55</td><td>45.80</td><td>37.54</td><td>26.65</td><td>41.83</td><td>52.66</td><td>59.69</td><td>57.25</td><td>60.93</td><td>63.99</td><td>63.17</td><td>58.41</td><td>65.45</td><td>72.18</td></tr><tr><td>GHT</td><td>27.40</td><td>18.08</td><td>30.76</td><td>45.76</td><td>37.40</td><td>27.77</td><td>41.66</td><td>56.19</td><td>60.02</td><td>59.43</td><td>61.52</td><td>63.16</td><td>68.26</td><td>60.78</td><td>70.37</td><td>79.93</td></tr><tr><td>RE-NET</td><td>28.81</td><td>19.05</td><td>32.44</td><td>47.51</td><td>38.28</td><td>28.68</td><td>41.34</td><td>54.52</td><td>49.66</td><td>46.88</td><td>51.19</td><td>53.48</td><td>58.02</td><td>53.06</td><td>61.08</td><td>66.29</td></tr><tr><td>RE-GCN</td><td>30.55</td><td>20.00</td><td>34.73</td><td>51.46</td><td>41.50</td><td>30.86</td><td>46.60</td><td>62.47</td><td>51.53</td><td></td><td>58.29</td><td>69.53</td><td>63.07</td><td></td><td>71.17</td><td>82.07</td></tr><tr><td>TG-Tucker</td><td>28.68</td><td>19.35</td><td>32.17</td><td>47.04</td><td>26.25</td><td>17.30</td><td>29.07</td><td>44.18</td><td>50.43</td><td>48.52</td><td>51.47</td><td>53.58</td><td>57.83</td><td>53.05</td><td>60.78</td><td>65.85</td></tr><tr><td>TLogic</td><td>29.82</td><td>20.54</td><td>33.95</td><td>48.53</td><td>43.04</td><td>33.56</td><td>48.27</td><td>61.23</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EvoKG</td><td>27.20</td><td>17.61</td><td>31.14</td><td>45.82</td><td>35.78</td><td>26.82</td><td>39.75</td><td>52.90</td><td>67.68</td><td>55.02</td><td>79.48</td><td>84.03</td><td>68.87</td><td>54.47</td><td>81.22</td><td>89.81</td></tr><tr><td>CENET</td><td>27.75</td><td>18.92</td><td>32.08</td><td>46.19</td><td>38.24</td><td>28.82</td><td>42.14</td><td>56.82</td><td>62.67</td><td>59.18</td><td>65.31</td><td>67.90</td><td>65.15</td><td>56.96</td><td>68.13</td><td>70.35</td></tr><tr><td>HGAT</td><td>28.50</td><td>19.60</td><td>32.70</td><td>46.60</td><td>38.90</td><td>29.70</td><td>42.40</td><td>56.40</td><td>56.10</td><td>52.90</td><td>58.10</td><td>61.80</td><td>63.60</td><td>59.80</td><td>66.00</td><td>71.50</td></tr><tr><td>TiPNN</td><td>30.32</td><td>21.55</td><td>35.06</td><td>50.08</td><td>41.20</td><td>32.75</td><td>46.23</td><td>59.60</td><td>73.99</td><td>71.57</td><td>76.82</td><td>80.67</td><td>80.30</td><td>78.85</td><td>82.10</td><td>89.04</td></tr><tr><td>OurModel</td><td>31.33</td><td>21.93</td><td>35.59</td><td>52.01</td><td>43.94</td><td>34.62</td><td>49.48</td><td>63.44</td><td>79.10</td><td>74.92</td><td>82.47</td><td>85.31</td><td>84.47</td><td>80.31</td><td>87.56</td><td>91.06</td></tr></table>

Table 1: Entity prediction results. The best results are marked in bold and the second best ones are underlined.

And among these four datasets, our method exhibits significant improvements compared to the baseline on the WIKI and YAGO dataset, while the enhancement on the ICEWS dataset is not as pronounced. This discrepancy is due to the fact that the WIKI and YAGO data both have a temporal span of one year, whereas ICEWS has a span of 24 hours. This aligns with real-world phenomena because the impact of events diminishes more noticeably over a longer time span, and the decay of short-term events is limited. Consequently, HERLN demonstrates substantial improvement in TKGs with longer time spans and distinct community structures.

## 5.2.2 Relation Prediction

The results of the relation prediction task in terms of the MRR metric are shown in Table 2. In the relation prediction task experiment, we do not include TG and RE-NET as baseline because they conduct only entity prediction. It can be seen from Table 2 that our proposed HERLN outperforms all the baselines and receives a boost of up to 10% in the MRR metric. The reasons of the performances of both our model and the baselines are similar to those for the entity prediction task.

<table><tr><td>Method</td><td>ICEWS18</td><td>ICEWS14s</td><td>WIKI</td><td>YAGO</td></tr><tr><td>ConvE</td><td>37.73</td><td>38.80</td><td>78.23</td><td>91.33</td></tr><tr><td>ConvTransE</td><td>38.00</td><td>38.40</td><td>86.64</td><td>90.98</td></tr><tr><td>R-GCRN</td><td>37.14</td><td>38.04</td><td>88.88</td><td>90.18</td></tr><tr><td>RE-GCN</td><td>40.53</td><td>41.06</td><td>97.92</td><td>97.74</td></tr><tr><td>EvoKG</td><td>41.11</td><td>42.47</td><td>90.63</td><td>90.26</td></tr><tr><td>CENET</td><td>38.24</td><td>40.50</td><td>87.51</td><td>91.44</td></tr><tr><td>OurModel</td><td>51.47</td><td>50.55</td><td>98.50</td><td>98.54</td></tr></table>

Table 2: Relation prediction results by MRR metric.

## 5.2.3 Ablation Study

We conduct ablation experiments on the ICEWS14 dataset. (1) OurModel without (w.o) ConvTransE: we replace the ConvTransE in the decoder unit with a simple MLP layer. (2) OurModel without (w.o) FiLM: Instead of FiLM, we directly use ConvTransE. (3) OurModel without (w.o) HRGCN: we replace HRGCN with a RGCN to aggregate snapshots. (4) OurModel without (w.o) Community: we omit the embedding initialize module.

As shown in Table 3, replacing any component of our model degrades the performance, which demonstrates that each component of the model has a positive gain on the result. The variant (1) has an almost 6% drop in MRR, suggesting that an appropriate decoder can learn event evolution patterns effectively. The variant (2) shows 2% decreasing in Hits@1, indicating that the hypernetwork helps the model distinguish different event evolution patterns. The variant (3) has 7.89%, 6.89%, 9.31%, 11.90% drops in MRR, Hits@1, Hits@3, respectively, emphasizing the importance of incorporating the temporal information of events through HRGCN. The variant (4) has 3% decreasing in MRR, confirming the helpfulness of extracting community structures.

<table><tr><td>Method</td><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td></tr><tr><td>w.o ConvTransE</td><td>37.98</td><td>29.59</td><td>41.72</td><td>54.19</td></tr><tr><td>w.o FiLM</td><td>41.48</td><td>31.39</td><td>46.47</td><td>60.14</td></tr><tr><td>w.o HRGCN</td><td>36.05</td><td>27.73</td><td>40.17</td><td>51.54</td></tr><tr><td>w.o Community</td><td>41.05</td><td>31.86</td><td>45.96</td><td>58.25</td></tr><tr><td>OurModel</td><td>43.94</td><td>34.62</td><td>49.48</td><td>63.44</td></tr></table>

Table 3: Effect of main components

## 6 Conclusions

In this paper, we propose a Hawkes-based evolutional representation learning network (HERLN) to tackle the TKG reasoning tasks. We exploit the structural and evolutional characteristics of TKGs inherited from real-world networks to learn structural information and evolutional patterns. We initialize the embeddings with a community detection algorithm and a graph convolution network to make use of community structure. We design a Hawkes process-based relational graph convolutional network to tackle the temporal decaying phenomenon. We construct a conditional decoder to alleviate the biases caused by scale-free property (long-tailed distribution). Experimental results show the superiority of our proposed model.

## 7 Limitations

Note that there are many entities appear only in the test set (named unseen entities), which will continue to appear as the size of TKG increases. Our model could not get sufficient information to assign proper embbedings for these entities. For further work, we plan to find a method to deal with the unseen entities.

## Acknowledgments

This work was supported by National Natural Science Foundation of China (No. 62476040) and the Fundamental Research Funds for the Central Universities.

## References

Albert-László Barabási and Réka Albert. 1999. Emergence of scaling in random networks. Science, 286:509–512.

Vincent D. Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. 2008. Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment, 2008:P10008.

Hao Dong, Pengyang Wang, Meng Xiao, Zhiyuan Ning, Pengfei Wang, and Yuanchun Zhou. 2024. Temporal inductive path neural network for temporal knowledge graph reasoning. Artificial Intelligence, 329:104085.

Zhen Han, Peng Chen, Yunpu Ma, and Volker Tresp. 2021a. Explainable subgraph reasoning for forecasting on temporal knowledge graphs. In ICLR 2021, 9th International Conference on Learning Representations, Virtual Event, Austria. OpenReview.net.

Zhen Han, Zifeng Ding, Yunpu Ma, Yujia Gu, and Volker Tresp. 2021b. Learning neural ordinary equations for forecasting future links on temporal knowledge graphs. In EMNLP 2021, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 8352–8364, Virtual Event / Punta Cana, Dominican Republic. Association for Computational Linguistics.

Zhen Han, Yunpu Ma, Yuyi Wang, Stephan Günnemann, and Volker Tresp. 2020. Graph hawkes neural network for forecasting on temporal knowledge graphs.

In AKBC 2020, Conference on Automated Knowledge Base Construction, Virtual. OpenReview.net.

Alan G. Hawkes. 1971. Spectra of some self-exciting and mutually exciting point processes. Biometrika, 58:83–90.

Woojeong Jin, Meng Qu, Xisen Jin, and Xiang Ren. 2020. Recurrent event network: Autoregressive structure inference over temporal knowledge graphs. In EMNLP 2020, Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing, pages 6669–6683, Online. Association for Computational Linguistics.

Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In ICLR 2015, 3rd International Conference on Learning Representations, San Diego, CA. OpenReview.net.

Yunshi Lan and Jing Jiang. 2020. Query graph generation for answering multi-hop complex questions from knowledge bases. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, ACL 2020, pages 969–974, Online. Association for Computational Linguistics.

Pengfei Li, Guangyou Zhou, Zhiwen Xie, Penghui Xie, and Jimmy Xiangji Huang. 2024. Learning dynamic and static representations for extrapolation-based temporal knowledge graph reasoning. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32:4741–4754.

Zixuan Li, Zhongni Hou, Saiping Guan, Xiaolong Jin, Weihua Peng, Long Bai, Yajuan Lyu, Wei Li, Jiafeng Guo, and Xueqi Cheng. 2022. Hismatch: Historical structure matching based temporal knowledge graph reasoning. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, pages 7328– 7338, Abu Dhabi, United Arab Emirate. Association for Computational Linguistics.

Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang, and Xueqi Cheng. 2021. Temporal knowledge graph reasoning based on evolutional representation learning. In SIGIR ’21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 408–417, Virtual Event, Canada. ACM.

Kangzheng Liu, Feng Zhao, Guandong Xu, Xianzhi Wang, and Hai Jin. 2023. RETIA: relation-entity twin-interact aggregation for temporal knowledge graph extrapolation. In 39th IEEE International Conference on Data Engineering, ICDE 2023, pages 1761–1774, Anaheim, CA, USA. IEEE.

Yushan Liu, Yunpu Ma, Marcel Hildebrandt, Mitchell Joblin, and Volker Tresp. 2022. Tlogic: Temporal logical rules for explainable link forecasting on temporal knowledge graphs. In AAAI 2022, Thirty-Sixth AAAI Conference on Artificial Intelligence, pages 4120–4127, Virtual Event. AAAI Press.

Namyong Park, Fuchen Liu, Purvanshi Mehta, Dana Cristofor, Christos Faloutsos, and Yuxiao Dong. 2022. Evokg: Jointly modeling event time and network structure for reasoning over temporal knowledge graphs. In WSDM ’22: The Fifteenth ACM International Conference on Web Search and Data Mining, pages 794–803, Virtual Event / Tempe, AZ, USA. ACM.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 8024–8035.

M. Schlichtkrull, Thomas Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. 2018. Modeling relational data with graph convolutional networks. In ESWC 2018, The Semantic Web - 15th International Conference, volume 10843 of Lecture Notes in Computer Science, pages 593–607, Heraklion, Crete, Greece. Springer.

Pengpeng Shao, Jiayi He, Guanjun Li, Dawei Zhang, and Jianhua Tao. 2023. Hierarchical graph attention network for temporal knowledge graph reasoning. Neurocomputing, 550:126390.

Haohai Sun, Shangyi Geng, Jialun Zhong, Han Hu, and Kun He. 2022. Graph hawkes transformer for extrapolated reasoning on temporal knowledge graphs. In EMNLP 2022: Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 7481–7493, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Haohai Sun, Jialu Zhong, Yunpu Ma, Zhen Han, and Kun He. 2021. Timetraveler: Reinforcement learning for temporal knowledge graph forecasting. In EMNLP 2021, Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 8306–8319, Virtual Event / Punta Cana, Dominican Republic. Association for Computational Linguistics.

Rakshit S. Trivedi, Hanjun Dai, Yichen Wang, and Le Song. 2017. Know-evolve: Deep temporal reasoning for dynamic knowledge graphs. In ICML 2017, Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, pages 3462–3471, Sydney, NSW, Australia. PMLR.

Minjie Wang, Lingfan Yu, Da Zheng, Quan Gan, Yujie Gai, Zihao Ye, Mufei Li, Jinjing Zhou, Qi Huang, Chao Ma, Ziyue Huang, Qipeng Guo, Haotong Zhang, Haibin Lin, J. Zhao, Jinyang Li, Alex Smola,

and Zheng Zhang. 2019a. Deep graph library: towards efficient and scalable deep learning on graphs. ICLR Workshop on Representation Learning on Graphs and Manifolds.

Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, and Tat-Seng Chua. 2019b. KGAT: knowledge graph attention network for recommendation. In Proceedings ofthe 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, pages 950–958, Anchorage, Alaska, USA. ACM.

Yao Xiao, Guangyou Zhou, Zhiwen Xie, Jin Liu, and Jimmy Xiangji Huang. 2024. Learning dual disentangled representation with self-supervision for temporal knowledge graph reasoning. Inf. Process. Manage., 61(3).

Zhiwen Xie, Runjie Zhu, Jin Liu, Guangyou Zhou, and Jimmy Xiangji Huang. 2023. Targat: A time-aware relational graph attention model for temporal knowledge graph embedding. IEEE/ACM Trans. Audio, Speech and Lang. Proc., 31:2246–2258.

Xu Yi, Ou Junjie, Xu Hui, and Fu Luoyi. 2023. Temporal knowledge graph reasoning with historical contrastive learning. In AAAI, pages 4765–4773, Washington, DC, USA. AAAI Press.

Mengqi Zhang, Yuwei Xia, Qiang Liu, Shu Wu, and Liang Wang. 2023a. Learning latent relations for temporal knowledge graph reasoning. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2023, pages 12617–12631, Toronto, Canada. Association for Computational Linguistics.

Mengqi Zhang, Yuwei Xia, Qiang Liu, Shu Wu, and Liang Wang. 2023b. Learning long- and short-term representations for temporal knowledge graph reasoning. In WWW 2023, Proceedings of the ACM Web Conference 2023, pages 2412–2422, Austin, TX, USA. ACM.

Shangfei Zheng, Hongzhi Yin, Tong Chen, Quoc Viet Hung Nguyen, Wei Chen, and Lei Zhao. 2023. DREAM: adaptive reinforcement learning based on attention mechanism for temporal knowledge graph reasoning. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, pages 1578–1588, Taipei, Taiwan. ACM.

Cunchao Zhu, Muhao Chen, Changjun Fan, Guangquan Cheng, and Yan Zhan. 2021. Learning from history: Modeling temporal knowledge graphs with sequential copy-generation networks. In AAAI 2021,Thirty-Fifth AAAI Conference on Artificial Intelligence, pages 4732–4740, Virtual Event. AAAI Press.

## Appendix

## A Statistics of the Datasets

The statistics of the datasets used in our experiment are summarized in Table 4. We divide

ICEWS14, ICEWS18, WIKI and YAGO into training, validation and test sets with a proportion of 80%, 10% and 10% in the chronological order, i.e., $t _ { t r a i n } < t _ { v a l i d } < t _ { t e s t }$

## B Case Study

In Table 5, we present three cases obtained from the ICEWS14s test set, each of which validates the effectiveness of one contribution within HERLN. In all three cases, HERLN achieves the highest score and successfully completes the predictions. In the first case, two related events share the same subject entity and relation, with a different object entity. HERLN leverages the community structure of Criminal (Venezuela) and Citizen (Venezuela) to complete the reasoning. In the second case, multiple events occurre between three entities, but the influence of earlier events on the reasoning outcome 10 months later is evidently smaller than that of more recent events. The Hawkes process helps our model to distinguish the useful information and make the correct choice. In the last case, Demand settling of dispute is a rare relation type, occurring less than 10 times in the dataset. The model is prone to be influenced by more common evolution patterns during reasoning. HERLN still successfully learns this evolution pattern by constructing a hyper network, completing the reasoning.

## C Time Cost Analysis

To evaluate the efficiency of our model, we compare HERLN with several TKG reasoning methods including RE-NET, RE-GCN and EvoKG. We select these baseline methods for the time analysis because these methods are similar to ours, being embedding-based methods. Comparing the time efficiency on these methods can better illustrate the efficiency improvement of our approach. CENET is an exception as it does not provide official code. Although we conduct the experiments, the computational time is significantly longer than other methods (exceeding 1 day on ICEWS14s). Therefore, we exclude CENET from the results. As Fig. 3 shows, our model runs faster than RE-NET and has a similar time consumption with RE-GCN. On the one hand, RE-NET uses multiple RNN structures to fit multi-level conditional probability distributions of events while ours relies on a Hawkes process for message passing and aggregation on one single graph. On the other hand, we use queryspecific version of ConvTransE as the decoder. The structure of ConvTransE allows it to predict multiple events at the same time, this high parallelism saves much time.

![](images/f40657bc39bbacb56c843506a94a4d77a47f4a0ef6e93efe882e47d61f53288b.jpg)  
Figure 3: Time cost analysis on ICEWS14s and YAGO.

<table><tr><td>Dataset</td><td>ε</td><td>R</td><td>F</td><td>T</td><td> $\mathcal { F } _ { t r a i n }$ </td><td> $\mathcal { F } _ { v a l i d }$ </td><td> $\mathcal { F } _ { t e s t }$ </td><td>Time interval</td></tr><tr><td>ICEWS18</td><td>23033</td><td>256</td><td>468558</td><td>304</td><td>373019</td><td>45996</td><td>49546</td><td>24 hours</td></tr><tr><td>ICEWS14</td><td>7128</td><td>230</td><td>90730</td><td>365</td><td>74846</td><td>8515</td><td>7372</td><td>24 hours</td></tr><tr><td>WIKI</td><td>12554</td><td>24</td><td>669934</td><td>231</td><td>539287</td><td>67539</td><td>63111</td><td>1 year</td></tr><tr><td>YAGO</td><td>10623</td><td>10</td><td>201089</td><td>187</td><td>161541</td><td>19524</td><td>20027</td><td>1 year</td></tr></table>

Table 4: Statistics of the datasets

<table><tr><td>Query</td><td>Answer</td><td>History</td></tr><tr><td>Criminal (Venezuela), fight with small arms and light weapons, ?, 2014/11/20</td><td>Citizen (Venezuela)</td><td>- Criminal (Venezuela), Use unconventional violence, Citizen (Venezuela), 2014/3/17 - Criminal (Venezuela), Use unconventional violence, Businessperson (Germany), 2014/6/19</td></tr><tr><td>Jason C. Hu, Yield, ?, 2014/11/29</td><td>Lin Chia-lung</td><td>- Jason C. Hu, Praise or endorse, Lin Chia-lung, 2014/11/25 - Jason C. Hu, consult, Ma Ying Jeou,2014/1/16</td></tr><tr><td>John Kerry, Demand settling of dispute, ?, 2014/11/12</td><td>Iran</td><td>- John Kerry , Make statement,Iran,2014/11/9</td></tr></table>

Table 5: Case Study