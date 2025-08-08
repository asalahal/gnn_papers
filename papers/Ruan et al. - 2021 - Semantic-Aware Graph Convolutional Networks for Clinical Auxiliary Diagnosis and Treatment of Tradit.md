Received November 27, 2020, accepted December 29, 2020, date of publication January 4, 2021, date of current version January 14, 2021.


_Digital Object Identifier 10.1109/ACCESS.2020.3048932_

# Semantic-Aware Graph Convolutional Networks for Clinical Auxiliary Diagnosis and Treatment of Traditional Chinese Medicine


CHUNYANG RUAN 1, YINGPEI WU 2, YUN YANG 3,
AND GUANGSHENG LUO 1, (Member, IEEE)
1 Department of Data Science and Big Data Technology, School of Economics and Finance, Shanghai International Studies University, Shanghai 200083, China
2 School of Software Engineering, Fudan University, Shanghai 200433, China
3 Department of Oncology, Longhua Hospital Shanghai University of Traditional Chinese Medicine, Shanghai 200032, China


Corresponding author: Guangsheng Luo (luoguangsheng03@126.com)


This work was supported in part by the National Science Foundation of China under Grant 61672161, and in part
by the Youth Foundation of Shanghai International Studies University under Grant 2020114096.


**ABSTRACT** Traditional Chinese Medicine (TCM) clinical informatization focuses on serving user-oriented
health knowledge and facilitating online diagnosis. Regularities are hidden in clinical knowledge play a
significant role in the improvement of the TCM informatization service. However, many regularities can
hardly be discovered because of specific data-challenges in TCM prescriptions at present. Therefore, in this
article, we propose an end-to-end model, called Semantic-aware Graph Convolutional Networks (SaGCN)
model, to learn the latent regularities in three steps: (1) We first construct a heterogeneous graph based
on prescriptions; (2) We stack Semantic-aware graph convolution to learn effective low-dimensional representations of nodes by meta-graphs and self-attention; (3) With the learned representations, we can detect
regularities accurately by clustering and linked prediction. To the best of our knowledge, this is the first
study to use metagraph and graph convolutional networks for modeling TCM clinical data and diagnosis
prediction. Experimental results on three real datasets demonstrate SaGCN outperforms the state-of-the-art
models for clinical auxiliary diagnosis and treatment.


**INDEX TERMS** Tranditional Chinese medicine, clinical knowledge discovery, metagraph, graph
convolutional networks.



**I. INTRODUCTION**

Research on Traditional Chinese Medicine (TCM) clinical informatization has changed from providing literature
resources to serving clinical auxiliary diagnosis and treatment. Promoting the research of TCM informatization and
facilitating online diagnosis along with the development of
data science are essential tasks of the clinical informatization

of TCM. Prescriptions are important data accumulated over
a long period of time in the clinical diagnosis and treatment
of TCM. They contain considerable TCM knowledge and are
the data basis of the TCM clinical informatization. In TCM,
a clinical prescription is a group of herbs, symptoms, diseases, and other clinical entities, recording a personalized
medical process for each patient. In clinical prescriptions,
the essence of regularities is multiple relations among different data entities, such as symptoms, herbs, and diseases.


The associate editor coordinating the review of this manuscript and


approving it for publication was Giovanni Dimauro .



Discovering regularities plays a significant role in improving
TCM clinical diagnosis and treatment and the development
of novel prescriptions [1].
Previous works proposed many machine learning-based
methods that could discover regularities in TCM clinical
prescriptions. They discover the latent relations among the
herbs, symptoms, syndromes, and improve diagnosis to some
extent [1]–[4]. However, the above methods failed to comprehensively explain how regularities are generated using multiple relations among different TCM entities or less consider
domain knowledge of TCM well.
Actually, we should solve the below challenges to address
shortcomings of the prior methods and support diagnosis decision-making. **(1)Random structure of data.** The
real-world TCM clinical prescriptions are usually represented
by natural languages and in free-text formats. To use entities
in clinical prescription, we should first model them well from
the text. **(2)Poor organization of data.** The prescriptions
have their own way of organizing various clinical entities



VOLUME 9, 2021 This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/ 8797


(e.g., herbs, symptoms, syndromes, and diseases), which are
often put in a disordered way. For example, a herb in the front
of the prescription may be correlated with the very last herb
instead of its surroundings. Implementing feature engineering
in a large number of semantic free-text prescriptions is a
challenging and costly task. **(3)The sparsity of data.** The
personalized views of TCM clinicians also influence datasets.
For example, since TCM clinicians’ prescribing habits often
converge to their individual experiences, some herbs or symptoms would not be recorded in prescriptions. This may result
in much difference and sparsity in recorded prescriptions of
clinical data.

To tackle all the aforementioned challenges, we propose a
graph convolutional networks based graph embedding model
Semantic-aware Graph Convolutional Networks (SaGCN).
In particular, we first construct a large graph from TCM
clinical prescriptions, in which herbs, symptoms, syndromes,
and diseases are defined as nodes. We then turn the massive
free-text prescriptions analysis problem into a large graph
analysis problem. To learn multiple relations in prescriptions,
we define the TCM graph as a heterogeneous information
network (HIN). HIN with various types of nodes and links
has the superior ability in modeling heterogeneous data and
learning the different semantics among nodes [5], and offers
the advantage of straightforward handling of missing values.
So, the TCM graph can be profoundly beneficial to better
express the rich information of clinical entities. However,
to analyze HIN is a complex problem. An effective approach
to solve this problem is to utilize Graph Representation
Learning (GRL) that uses deep learning and nonlinear dimensionality to encode graph structure into low-dimensional
embeddings [6]. One of the main problems of GRL in HIN
is semantic search while the central problem of GRL in
the TCM graphs is how to incorporate TCM knowledge
into the embedding model. Meta-graphs can capture various semantics among nodes on the graph [7]. We propose
a meta-graph and attention mechanism-based approach to
solve this problem. Next, we incorporate the semantics of the
TCM graph into a graph convolutional networks (GCN) to
obtain the node embedding of the TCM graph. After that,
we optimize the overall model by using backpropagation
and employ traditional machine learning algorithms to complete analysis for TCM clinical diagnoses supporting. We
evaluate five state-of-the-art approaches and our proposed
model _SaGCN_ on three real-world TCM medical datasets

for prediction and diagnostic tasks. The results show that our
proposed model _SaGCN_ outperforms other compared graph
convolution-based models. Using 9000+ clinical lung tumor
prescriptions, we also conduct two case studies, prescriptions
prediction and disease prediction, to qualitatively reveal the
robust power of _SaGCN_ in capturing relation in TCM clinical
data and reflect the clinical diagnostic patterns in TCM.
To summarize, our main contributions are as follows:

  - To the best of our knowledge, this is the first attempt to
take advantage of HIN and GCN with self-attention for
clinical auxiliary diagnosis and treatment task.



C. Ruan _et al._ : SaGCNs for Clinical Auxiliary Diagnosis and Treatment of TCM


  - We jointly model the clinical entities (herbs, symptoms,
syndromes, and diseases) from clinical prescriptions as
a large graph to provide effective and safe diagnosis
prediction.

  - We propose _SaGCN_, an accurate and robust learning
model based on meta-graph and _semantic-aware convo-_
_lution_ -based GCN for TCM clinical prescriptions, which
captures the multi-semantics and learns heterogeneous
node embedding tailored for TCM diagnosis prediction
tasks.


  - We compare several state-of-the-art models on real TCM
data qualitatively and quantitatively to demonstrate the
effectiveness and robustness of _SaGCN_ .


**II. RELATED WORK**

_A. TCM DIAGNOSIS PREDICTION_

Minning over medical, health, or clinical data is considered the most challenging domain for data mining [8]. With
the rapid development of machine learning, a large amount
of work has been focused on finding out various kinds of
hidden knowledge relations such as symptom and symptom, symptom and syndrome, and syndrome and disease for
improving the quality of clinical diagnosis and healthcare via
text mining [9]. Chen _et al._ [2] presented a HIN-based soft
clustering approach to discover the categories of formulas.
Li _et al._ [3] utilized seq2seq model with coverage mechanism to generate TCM prescription. Yao _et al._ [1] developed
a novel topic model to detect the relation between herbs
and symptoms and characterized the generative process of
prescriptions. Although these models are effective in TCM
exploration, they are limited to the traditional data mining
methods or the characteristics of data. Compared with all
the aforementioned NLP-based predictive methods, the proposed framework _SaGCN_ has the following advantages:
(1) It leverages the powerful data representation advantages
to overcome the drawback of TCM clinical prescriptions;
(2) It captures the semantics of TCM clinical prescriptions
without loss of generality and simplicity, which takes a good
predictive performance.


_B. HIN AND METAGRAPH_

HIN has attracted much attention in the past decade because
of its capability of representing the rich type information,
as well as the accompanying wide applications such as personalized recommendation [10], clustering [11], and outlier
detection [12]. Exploring semantics is the foundation step
of all HIN-based tasks [13]. Although meta paths have been
shown to be useful in different applications, they can only
express simple relations between source and target entities

[14]. Previous works [10], [11], [15], [16] focus on using
meta-path [17] to preserve the semantics in HIN. Recently
many works have been adopting meta-graph to preserve
the semantics in HIN, which measures semantics better
than meta-path. For example, Huang _et al._ [14] proposed
meta structure, a directed acyclic graph of entity types with
edge types connecting in between, to measure the proximity



8798 VOLUME 9, 2021


C. Ruan _et al._ : SaGCNs for Clinical Auxiliary Diagnosis and Treatment of TCM


between entities. Fan _et al._ [18] presented a meta-graph based
embedding model to depict the relatedness over files. Inspired
by these works, we utilize meta-graph to incorporate more
rich semantics into our GCN model.


_C. GRAPH CONVOLUTIONAL NETWORKS_

GCN is an extension of convolutional neural network for

processing the graph data, which has received growing attentions recently. GCN has been successfully used in many tasks

[19]–[23], such as neural recommendation [19], event detection [20], machine translation [21] and healthcare [22], [23].
Focusing on healthcare, most existing works aim to learn
relation in biomedical networks for prediction tasks, such
as medicine interaction prediction. In these tasks, the highlevel graph representations help the final predictions. GCN
has revolutionized the field of graph representation learning
through effectively learned node embeddings, and achieved
state-of-the-art results for many tasks [24]. Fout _et al._ [23]
stacked multiple layers of convolution and learned effective
latent representations that integrate information across the
graph to predict protein interface. Sankar _et al._ [25] presented
a novel spatial convolution operation to capture the key properties of local connectivity and translation invariance, using
high order connection patterns and attention mechanism.
Inspired by previous works [23], [25], we employ
heterogeneous graph representation learning based on
meta-graph and GCN alongside with the joint learning framework to learn TCM clinical entities representations and their
relations.


**III. METHODOLOGY**

_A. PROBLEM FORMULATION_


_Definition 1 (TCM Clinical Prescriptions and Diagnosis_
_Prediction):_ In TCM clinical settings, TCM doctors first
record symptoms that they observe in their patients. Furthermore, they aim to determine the patient’s syndrome according to the patient’s symptoms. The doctors then prescribe
herbs combination based on the patient’s disease profile.
These herbs, symptoms, syndromes, and diseases are captured and described as information entities such as ‘‘red

ginseng’’ and ‘‘apricot kernel’’ in clinical prescriptions. Each
full TCM clinical prescription of each patient can be represented as a word set of multivariate observations: _R_ [(] _[k]_ [)] =
_h_ [(] _[k]_ [)] _, s_ [(] _[k]_ [)] _,_ ˆ _s_ [(] _[k]_ [)] _, d_ [(] _[k]_ [)] _, k_ ∈ 1 _,_ 2 _,_ - · · _, K_ where _h_ [(] _[k]_ [)] is the herbs
combination, _s_ [(] _[k]_ [)] is the group of symptom, ˆ _s_ [(] _[k]_ [)] is a syndrome,
_d_ [(] _[k]_ [)] is a disease and _K_ is the total number of prescriptions. Prescribing herbs based on symptoms and discovering
new herbs for disease(i.e. drug repositioning) are valuable
diagnosis prediction. These prediction can be formulated as:

_f_ _f_
_s_ −→ _d_, and _s_ −→ _h_ where _f_ is a mapping function.
_Definition 2 (TCM Graph):_ TCM prescriptions are modeled as a heterogeneous graph _G_ = ( _V_ _, E_ ) where _V_ and _E_
refer to the set of nodes and links respectively. Each node _v_ ∈
_V_ is mapped to a specifc clinical entity _O_ (e.g. herb, symptom,
syndrome or disease) by an entity type mapping function
_φ_ : _V_ �→ _O_ . And, each link _e_ = ( _v_ _i_ _, v_ _j_ ) ∈ _E_ is mapped to type



**FIGURE 1.** Structures of TCM graph. Different icons and line styles display
different types of TCM nodes and links, respectively, which are closely
correlated with one another.


_L_ by link type mapping function _ϕ_ : _E_ �→ _L_ where the two
nodes _v_ _i_ _, v_ _j_ belong to two different types. Given TCM graph,
its schema _T_ _G_ is a directed graph defined over entity types _O_
and link types _L_, i.e. _T_ _G_ = ( _O, L_ ). The schema expresses all
allowable link types between entity types [14]. Fig. 1 shows
the abstract schema of the network illustrating node types and
basic links.

_Definition 3 (TCM Metagraph):_ A TCM metagraph is
defined as _g_ = ( _V_ [´] _,_ _E_ [´] _, v_ _s_ _, v_ _t_ ) defined on the given TCM graph
schema _T_ _G_ = ( _O, L_ ). _g_ has only a single source node _v_ _s_ and
single target node _v_ _t_ . _V_ [´] is a set of nodes and _E_ [´] is a set of
links satisfying two constraints: (1) ∀ _v_ _i_ ∈ _V_ [´] _,_ ∃ _φ_ ( _v_ _i_ ) ∈ _O_ ;
(2) ∀( _v_ _i_ _, v_ _j_ ) ∈ _E_ [´], ∃ _ϕ_ ( _φ_ ( _v_ _i_ ) _, φ_ ( _v_ _j_ )) ∈ _L_ . Fig. 2(a) shows the
abstract schema of the network illustrating node types and
basic links.

_Definition 4 (Instance of TCM Metagraph):_ Given TCM
graph and metgraph, an instance of metagraph with target is a
subgraph ofa mapping for(1) ∀ _v_ ∈ _V_ ´ _G_, ∃, denoted by _gφ_ _v_ (, _v ψ_ ) =: _V_ [´] _ψ_ _g_ �→ _g_ ( _v_ _v_ =); (2) _V_ (´ satisfying two constraints: _V_ [´] _g_ ∀ _,_ _E_ [´] _v_ _g_ _, u_ ) such that there exists ∈ _V_ ´ _,_ ( _v, u_ ) ∈ (∈ _/_
) _E_ [´] _,_ ∃(∄)( _ψ_ ( _v_ ) _, ψ_ ( _u_ )) ∈ (∈ _/_ ) _E_ [´] .


_B. SEMANTIC-AWARE_

TCM clinical prescriptions contain rich TCM domain knowledge. For example, the jun (emperor) herbs treat the main
cause or primary symptoms of a disease, and the zuo (assistant) herbs are used to improve the effects of jun and chen,
and to counteract the toxic or side effects of these herbs [1].
In Fig. 2(b), a disease _D_ 1 connects two herbs _H_ 1 and _H_ 4 via
the same meta-graph _Ms_ 3 . _H_ 1 and _H_ 4 may play a different role
for the disease. How to distinguish the semantics of two nodes
in this meta-graph comprehensively? In this article, we model
this issue as a weight measure. We employ Point-wise Mutual
Information (PMI) and degree to develop a structure-aware
approach.
To utilize the global characteristics of the relation between
two nodes, we use PMI with a constant size sliding window
on all documents in the prescriptions to gather co-occurrence
statistics of nodes. The weight of the link between node _i_ and
node _j_ can be defined as follows:


_ρ_ _ij_ = log _[♯][Count]_ [ ∗] _[Count]_ [(] _[i][,]_ _[j]_ [)] (1)

_Count_ ( _i_ ) ∗ _Count_ ( _j_ )



VOLUME 9, 2021 8799


C. Ruan _et al._ : SaGCNs for Clinical Auxiliary Diagnosis and Treatment of TCM


**FIGURE 2.** (a) Sample metagraph in TCM graph. Node Types: Herb (H), Symptom (S), Syndrome ( _S_ **[ˆ]** ) and Disease (D);
(b) Example subgraph with instances of _Ms_ 3 for target disease D.



where _Count_ ( _i_ ) and _Count_ ( _j_ ) are the number of sliding
windows in all prescriptions that contain entity _i_ and _j_ respectively, _Count_ ( _i, j_ ) is the number of sliding windows that contain both entity _i_ and entity _j_, and _♯Count_ is the total number
of sliding windows in all prescriptions. A high _ρ_ value implies
a high semantic correlation of node in the TCM graph, while
a low _ρ_ value indicates little or no semantic correlation in the
graph. Connecting to many nodes, however, a node (e.g., zuo
herb) just plays a minor role in the TCM graph. The degree
of nodes can well reflect the structures of graph [26]. So,
to solve this problem, we define a degree based approach to
optimize (1) as follows:


~~_ρ_~~ _ij_
~~_ρ_~~ _ij_ = ~~�~~ | _kV_ =|1 ~~_[ρ]_~~ _[ik]_ ~~�~~ | _kV_ =|1 ~~_[ρ]_~~ _[jk]_ (2)


where | _V_ | is number of nodes in TCM graph, ~~_ρ_~~ _ik_ and ~~_ρ_~~ _jk_
are the the values of PMI between node _k_ and _i_, _j_ respectively. Next, we explore the semantic characteristics of a
meta-graph’s instance. Different walk paths of a meta-graph
encode different semantics for TCM knowledge. In Fig. 2(b),
a disease _S_ 2 connects two herbs _H_ 1 and _H_ 2 via the same
meta-graph _Ms_ 1, the _H_ 1 also connects with _S_ 1 so that it may
be a zuo (assistant) herb. So, the two herb nodes _H_ 1 and _H_ 2
have different relevance with symptom node _S_ 2 . Formally,
because the attention mechanism aims to focus on the most

pertinent information, and if an instance of meta-graph _m_ _s_
with source nodes _v_ _s_ and target node _v_ _t_ is given, we define the
self-attention-based approach to preserve different semantics
between walk paths in _m_ _s_ as follows:



_a_ _l_ = _softmax_ ( _s_ [(] _l_ _[k]_ [)] [(] _[v]_ _[s]_ _[,][ v]_ _[t]_ [)][ ·] _[ w]_ _[l]_ [ ·] _L_ [1]



_L_

_s_ [(] _[k]_ [)]

� _i_ [(] _[v]_ _[s]_ _[,][ v]_ _[t]_ [))] _[,]_


_i_ =1



_s_ [(] _l_ _[k]_ [)] [(] _[v]_ _[s]_ _[,][ v]_ _[t]_ [)][ =] _[ s]_ [(] _[v]_ _[s]_ _[,][ v]_ _[t]_ [|][[1][ : |] _[l]_ [|][])][ =] _[ ρ]_ _[sm]_ × _s_ ( _v_ _m_ _, v_ _t_ |[2 : | _l_ |])

_♯ρ_ _s_



(3)


where _w_ _l_ is a parameter mapping between the context semantics of all walk path _L_ and each semantics _s_ [(] _l_ _[k]_ [)] [(] _[v]_ _[s]_ _[,][ v]_ _[t]_ [) in] _[ m]_ _[s]_ [,]
which is learned as part of the training process. | _l_ | is a length
of a walk path of _l_, _L_ is number of walk path in _m_ _s_ . _ρ_ _sm_ is
the weight of link between node _v_ _s_ and node _v_ _m_, and _♯ρ_ _s_
is the sum of weights among _v_ _s_ and its neighboring nodes.



The value of _ρ_ _sm_ _/♯ρ_ _s_ is the transition probability between
node _v_ _s_ and node _v_ _m_ .


_C. SEMANTIC MATRIX_
A semantic matrix **S** [(] _[k]_ [)] is a similarity matrix to encode the
relevance of nodes in each unique semantic role over all
instances of _Ms_ _k_ in the graph _G_ . **S** [(] _ij_ _[k]_ [)] is the transition probability between souce node _v_ _i_ and target node _v_ _j_ in an instance
of the _Ms_ _k_ . Formally, **s** [(] _ij_ _[k]_ [)] can be formulated as an iterative
function:


_s_ [(] _ij_ _[k]_ [)] = _s_ [(] _[k]_ [)] ( _v_ _i_ _, v_ _j_ | _g_ _v_ _i_ → _v_ _j_ ) (4)


where _g_ is an instance of _Ms_ _k_ . _s_ [(] _[k]_ [)] ( _v_ _i_ _, v_ _j_ | _g_ _v_ _i_ → _v_ _j_ ) can be computed by the aggregation of all value of _s_ [(] _l_ _[k]_ [)] [(] _[v]_ _[s]_ _[,][ v]_ _[t]_ [) in (][3][). For]
complex meta-graphs, computing **S** [(] _ij_ _[k]_ [)] is very complicated
because of the various ways to pass through the meta-graph

[27]. For _Ms_ 1 in Fig. 2(a), there is only one path to pass
through _Ms_ 1, and the transition probability between source
node and target node can be calculated by Eq.3. For _Ms_ 3
in Fig. 2(a), however, there are two ways to pass through the
meta-graph, which are _H_ − _S_ 1 − _D_ and _H_ − _S_ 2 − _D_ . Note
that _S_ 1 and _S_ 2 represent the different entity type symptoms in
the TCM graph. In _Ms_ 3, the path _H_ − _S_ 1 − _D_ means that herb
can cure a symptom of a disease, so that herb and disease
have some similarities. Similarly, in the path _H_ − _S_ 2 − _D_,
herb and disease have some similarities as well. Therefore,
we should define the logic of similarity and semantics when
there are multiple ways passing from the source node _H_ to the
target node _D_ in the meta-graph. Inspired by [27], we propose
a approach to obtain matrix **S** [(] _[k]_ [)] . Algorithm(1) depicts the
example of the similarity-based semantic matrix operations
for _Ms_ 3 where ⊙ is the Hadamard product, and the elements
of matrix _T_ is the transition probability between source node
and target node in the current path, which is calculated by
Eq.4, and **W** is the weight matrix consisting of the elements
_a_ _l_ in Eq.3. Note that this algorithm is not limited to _Ms_ 3 .
Any meta-graph with complex paths can be computed by
Hadamard product and multiplication on the corresponding
matrixes. We then can get different semantics between source
node and target node by computing semantic matrix for all



8800 VOLUME 9, 2021


C. Ruan _et al._ : SaGCNs for Clinical Auxiliary Diagnosis and Treatment of TCM


**Algorithm 1** Metagraph Based Computing Semantic for
_Ms_ 3


**Input** : TCM graph _G_ = ( _V_ _, E_ )
**Output** : **T** _HD_
**T** _HD_ 1 matrix
computation: **T** _HD_ 1 ← **T** _HS_ 1 × **T** _S_ 1 _D_ × **W** 1 ;
**T** _HD_ 2 matrix
computation: **T** _HD_ 2 ← **T** _HS_ 2 × **T** _S_ 2 _D_ × **W** 2 ;
**T** _HD_ matrix computation: **T** _HD_ ← **T** _HD_ 1 × **T** _HD_ 2 ;


meta-graphs denoted by { _Ms_ 1, _Ms_ 2, _Ms_ 3, · · ·, _Ms_ _k_ } in TCM
graph.


_D. SEMANTIC-BASED CONVOLUTION_

In this section, we consider learning representation for a
specific node type via spatial convolution operation, which
preserves the spatial locality and precise semantics.
Specifically, we provide a meta-graph _Ms_ with target node
_v_ _i_ ∈ _V_ as input. Convolution, intrinsically, is an aggregation
operation between local inputs and filters [28]. In this article,
filters should be in a position to aggregate local inputs with
diverse topological structures and semantics. A semantic filter for _Ms_ is defined by a weight matrix _W_ [(] _[t]_ [)] for target node
_v_ _i_ and a weight matrix _W_ [(] _[N]_ [)] for the sources of target node
in _Ms_, and each weight in _W_ [(] _[N]_ [)] differentiates the semantic
roles of source nodes in the receptive field. Our objective is
to design a semantic convolution kernel that can be applied to
heterogeneous graphs with spatial locality and rich semantics.
To summarize, we would like to learn a mapping function at
each node in the graph, which has the form:


_y_ _i_ = _σ_ _W_ ( _x_ _i_ _,_ { _x_ _s_ 1 _, x_ _s_ 2 _,_       - · · _, x_ _s_ _k_ }) (5)


where { _x_ _s_ 1 _, x_ _s_ 2 _,_ - · · _, x_ _s_ _k_ } are the source nodes of node _v_ _i_ that
defines the receptive field of the convolution; _σ_ is a non-linear
activation function, and W is the filter as learned parameters of the function. For the target node _v_ _i_, its spatial locality and semantics are captured by relation structure-aware.
Accordingly, we define semantic convolution as follows:



importance of different semantics for each node dynamically.



_z_ _i_ =



_K_
� _a_ ( _k,i_ ) - _y_ [(] _i_ _[k]_ [)] (7)


_k_ =1



where _z_ _i_ is defined as the weighted summation of every
semantic vectors _y_ [(] _i_ _[k]_ [)] for node _v_ _i_, _k_ = 1 _,_ 2 _,_ - · · _, K_, corresponding to the semantic index for each node. For each
semantic vector _y_ [(] _i_ _[k]_ [)] of node _v_ _i_, we compute a positive weight
_a_ ( _k,i_ ) which can be interpreted as the probability that _y_ [(] _i_ _[k]_ [)] is
assigned by node _v_ _i_ . We define the weight of semantics _k_ for
the node _v_ _i_ using a softmax function as follows:


exp( _λ_ ( _k,i_ ) )
_a_ ( _k,i_ ) = _K_ _,_
~~�~~ _k_ =1 [exp(] _[λ]_ [(] _[k][,][i]_ [)] [)]


⊤
_λ_ ( _k,i_ ) = _y_ [(] _i_ _[k]_ [)]       - _H_       - _y_ ( _s,i_ ) _,_



_y_ ( _s,i_ ) = [1]

_K_



_K_
� _y_ [(] _i_ _[k]_ [)] (8)


_k_ =1



where _y_ ( _s,i_ ) is the average of different semantic vectors, which
can capture the global context of the semantic information. _H_
is a vector mapping between the global context embedding
_y_ ( _s,i_ ) and each semantic vector _y_ [(] _i_ _[k]_ [)] [, which is learned as part]
of the training process. By introducing an attentive vector
_y_ [(] _i_ _[k]_ [)] [, we compute the relevance of each semantic vector to]
the node _v_ _i_ . If _y_ [(] _i_ _[k]_ [)] and _y_ ( _s,i_ ) have a large dot product, this
node believes that semantic _k_ is an informative semantics. For

example, the weight of semantic k for this node will be largely
based on the definition.
Once we obtain the weighted node vector representation
_z_ _i_, an objective function is used to learn a low-dimensional
node embedding. We use negative samples based hinge loss
to minimize the reconstruction error:



_J_ ( _θ_ ) =



_J_
� max(0 _,_ 1 − _z_ _i_ _x_ _i_ + _z_ _i_ _n_ _j_ ) (9)

_j_ =1



1
_y_ _i_ = _σ_ ( _W_ [(] _[t]_ [)] _x_ _i_ +
| _N_ _k_ |



� _W_ [(] _[N]_ [)] _S_ _ij_ _x_ _j_ + _b_ ) (6)

_v_ _j_ ∈ _N_ _k_



where _n_ _j_ is the negative node, which has no connection
with the target node in any meta-graph. _J_ is the times of
node sampling, and _θ_ is the set of parameters to be solved.
In addition, we adopt the asynchronous stochastic gradient
descent (ASGD) and the backpropagation algorithm to optimize the objective function. We also use dropout to prevent over-fitting. A multi-semantic convolutional layer is
illustrated in Fig. 3.


_F. DIAGNOSIS PREDICTION_

We obtain node embedding (e.g., herb, symptom, and disease) using model _SaGCN_ . These node embeddings can be
further extended to diagnosis tasks according to specific data
and problems. For example, given a set of herbs, the incompatible herb pair can be summarized in a latent embedding
space by similarity calculation:



where _N_ _k_ is the set of source nodes of node _v_ _i_, and _b_ is is a

vector of biases.


_E. COMBINING MULTI-SEMANTIC_

Given multiple semantics with certain characteristic metagraphs, a good proximity measure must account for different
semantics. Since different semantics can vary in their importance for the graph representation, we face the challenge of
appropriately weighting the extracted proximity at the end of
each semantics for effective feature propagation. To tackle
this challenge, we leverage attention mechanism to help learn
stable and robust node embedding of graph. Inspired by the
recent progress of the self-attention for machine translation

[29], we propose a semantic-attention model to weight the



_f_ ( _h_ 1 _, h_ 2) =



~~�~~
~~�~~
�
�



_d_
�



�( _h_ 1 _i_ − _h_ 2 _i_ ) [2] (10)


_i_ =1



VOLUME 9, 2021 8801


**FIGURE 3.** An overview of the proposed SaGCN model.


where _h_ 1 _, h_ 2 are the different herbs, and _d_ is the dimensionality of herb in embedding space. To optimize the aforementioned model, we set the goal is to minimize _J_ as a
function of _θ_ . In addition, we use the cross-entropy between
the ground truth visit _f_ and the predicted visit _f_ to calculate
the loss for each prescription from all the herb-pair as follows:


_L_ ( _h_ 1 _, h_ 2 _,_ - · · _, hm_ ; _f_ 1 _, f_ 2 _,_ - · · _, f_ _m_ )



C. Ruan _et al._ : SaGCNs for Clinical Auxiliary Diagnosis and Treatment of TCM


To validate the predictive performance of the proposed
_SaGCN_, we compare it with the following state-of-the-art
approaches:

  - **ASPEm** [15] is a node embedding learning for HIN,
which observes multiple aspects existing in HIN and
extends the skip-gram model to obtain the graph
representation.

  - **GWCN** [31] is a recently proposed spectral GCN, which
leverages wavelet transform to implement efficient
convolution on graph data.

  - **MotifCNN** [25] is a novel spatial convolution operation to model the key properties of heterogeneous local
connectivity and translation invariance, using high-order
connection patterns.

  - **HeteroMed** [5] is capable of capturing informative relation for the diagnosis goal and uses the best relation sampling strategy for learning clinical event representations
for EHR data.


  - **TM2P** [1] is a novel prescription topic model incorporated TCM knowledge to discover regularities on the
herbs composition and corresponding symptoms.


_B. SIMILAR DISCOVERY AND VISUALIZATION_

We conduct a series of comparative experiments of node
clustering to simulate the clinical analysis and detect the
combination of herbs curing a syndrome. Getting the node
embedding, we select a set of syndrome ( _S_ [ˆ] ) nodes, which are
assigned more labels from a TCM doctor selected set and we
use their representations as feature vectors to learn and test
a clustering algorithm Density-Based Spatial Clustering of
Applications with Noise (DBSCAN). We use _accuracy_ and
_normalized mutual information (NMI)_ as metrics for evaluation. _NMI_ is often employed to determine the gap between
the results of division and the true partition.
In fact, this clustering step can select groups without label
according to specific data. For clinical diagnostic prediction



1

= −
_m_ − 1



_m_ −1
�( _f_ _i_ log( _f_ _i_ ) + (1 − _f_ _i_ ) log(1 − _f_ _i_ )) (11)


_i_ =1



**IV. EXPERIMENTS**

In this section, we evaluate _SaGCN_ on three real-world TCM
datasets. First, it is evaluated under node clustering and link
prediction tasks. Then its TCM diagnosis prediction performance is compared to various state-of-the-art TCM predictive
models. Finally, it is qualitatively evaluated through a case
study.


_A. DATASET AND CONFIGURATIONS_

We use three TCM datasets in our evaluation.


  - **TCMRel [30]:** This is a candidate relation graph
composed of four node types: herbs (H), formula (F),
syndrome ( _S_ [ˆ] ) and disease (D), connected by five link
types: F-D, F- _S_ [ˆ], H-D, H- _S_ [ˆ], D- _S_ [ˆ] . We use a subset with
H, D, _S_ [ˆ], and their correlations.

  - **CMD** [1] **:** We use chinese medical information to create

a graph with four node types: herbs (H), symptom (S),
syndrome ( _S_ [ˆ] ), disease (D) linked by six fundamental
types: H-D, H-S, H- _S_ [ˆ], D-S, D- _S_ [ˆ], S- _S_ [ˆ] .

  - **CLLT:** We construct the clinical graph with three node
types: herb (H), symptom (S), disease D) and their
correlations from 7,000 clinical prescriptions for lung

tumor.


1 http://cmekg.pcl.ac.cn/



8802 VOLUME 9, 2021


C. Ruan _et al._ : SaGCNs for Clinical Auxiliary Diagnosis and Treatment of TCM


**TABLE 1.** Performance Evaluation of Similar Discovery.


task, we obtain groups as candidate sets in order to reduce the
number of node relation to be predicted.
Table 1 shows the performance of all the approaches on all
the three real-world TCM datasets. We can observe that our

proposed approache _SaGCN_ achieves the best performance
compared with all the baselines in terms of the values of
all the measures. On all datasets, the overall performance
of traditional topic model based on approache TM2P is
worse than that of the deep learning based approaches, and
SaGCN obtains the highest score among all baselines with
respect to _accuracy_ and _NMI_ . For deep learning baselines,
ASPEm, MotifCNN and HeteroMed achieved higher performance than GWCN on datasets TCMRel and CMD because

GWCN can not capture rich semantic proximity in large
graph with various nodes. But on dataset CLLT with less node
types, GWCN performanced better compared with ASPEm
and HeteroMed because its spectral convolution can capture
the structure of graph effectively. As for the representation
learning model for heterogeneous graph, the performance of
MotifCNN is close to that of the winner _NMI_ on all datasets,
while ASPEm and HeteroMed perform similarly. To intuitively illustrate the significance of node clustering for clinical diagnostic prediction, we give an illustrative visualization
of the herbs clustering. We use clustering to predicte relation
among herbs based on learned node embeding. We randomly
selected herbs, and use k-means to obtain the herb-clusters.
We used _FVIC_ [32] to test results. In addition, we asked two
TCM doctors to verify the clusters, they confirmed that the
herbs in the same cluster had the same function or belonged
to the frequently occurring herb-pairs, which basically conformed to the rule of clinical medication. Fig 4 shows the
visualization results. We can see that our model SaGCN

distinguishes different herbs much better than another node
embedding models, which also shows the power of SaGCN
on the task of unsupervised learning.


_C. RELEVENT SYMPTOM-HERB RELATION DISCOVERY_

In this article, we model the relation discovery problem
as a link prediction that aims to rank node pairs in terms



of their relevancy, which may lead to a potential linkage
between them. Using the symptom and herb distributions
of different syndromes of diseases from candidate sets,
we can further derive symptom-herb relation. Specifically,
given a TCM graph, we first generate a subgraph by selecting a herb-symptom class and randomly remove a certain
fraction (30% in our experiments) of links of the selected
class as missing links. Since the logistic regression converges faster, we use it to predict the missing edges between
herb–symptoms pairs as testing instances. We then use _Mean_
_Average Precision (MAP)_ as metric for evaluation.
From the main results on link prediction presented
in Table 2, we have observed consistent with the clustering
tasks that all the graph embedding methods perform better performance than the topic model based TM2P without
considering type information. For graph embedding methods, SaGCN outperforms all other models. MotifCNN and
GWCN achieve better results than ASPEm and HeteroMed.

GWCN has superior performance than ASPEm and HeteroMed, which preserves the important structural information
in the graph with fewer node types. Overall, we can see that
SaGCN outperforms all the other methods in the task of link
prediction.
In addition, We show the herbs with the highest relation
with two test symptoms respectively in Table 3. Each rank
is the conditional probability of herb given a test symptom.
We can see that our model not only discovered herbs for
highly frequent symptoms like ‘‘bitter figwort’’ for symptom
‘‘sore throat’’ but also found important infrequent herbs for
this symptoms like ‘‘oroxylum indicum’’. In fact, ‘‘oroxylum
indicum’’ can relieve asthma which has been verified in TCM
literature ‘‘Compendium of Materia Medica’’.


_D. CASE STUDY FOR DIAGNOSIS PREDICTION_

We choose clinical prescriptions from test datasets based on
the consideration of demonstrating the model effect on more
challenging cases: there are complex herbs composition and
corresponding symptoms. To examine the potential of this
direction, we use 75% prescriptions as training data and the



VOLUME 9, 2021 8803


C. Ruan _et al._ : SaGCNs for Clinical Auxiliary Diagnosis and Treatment of TCM


**FIGURE 4.** Comparison of herb groups visualization using unsupervised node embeddings on TCMRel.



**TABLE 2.** Performance Evaluation of herb-symptom link prediction.


remaining 25% as test data. We compare the results of disease and treatment prediction to the state-of-the-art predictive
models. We employ three standard metrics commonly used to
evaluate this prediction task: _F_ 1 score.
We compute the group of symptoms given a disease via soft
clustering. We then use the top N symptoms with the largest
similarity as the recommended symptoms. Fig. 5(a) gives
the performance of each model. Our model SaGCN achieves



**TABLE 3.** The example of Top-5 herbs given a symptom.


a significant increase generally. Numerically, our model
SaGCN achieves macro- _F_ 1 and micro-of _F_ 1 of respectively,
which is much higher than that of HeteroMed and TM2P.
Because they require the number of contexts, HeteroMed and
TM2P are limited to the missing value. Our model SaGCN
is not hampered by this restriction. In addition to supporting
treatment, our model can also recommend prescriptions as
references for doctors. We obtain a strong relation among
herbs and symptoms for a specific disease using link prediction. From Fig. 5(b), we can see that our model SaGCN
achieves a higher macro- _F_ 1 than HeteroMed and TM2P. This
suggests that SaGCN can recover the herbs that the doctor
actually prescribed to a disease, while also predict many
herbs that were not prescribed before. HeteroMed achieves
a relatively low _F_ 1, and TM2P performances much worse.



8804 VOLUME 9, 2021


C. Ruan _et al._ : SaGCNs for Clinical Auxiliary Diagnosis and Treatment of TCM


**FIGURE 5.** Comparison among our model SaGCN, HeteroMed and TM2P in disease and treatment
prediction.


**TABLE 4.** An example of the comparison between a diagnosis prediction from our model MsGCN and that from TCM doctor.



To examine this in detail, we compare the results predicted
by using our model SaGCN with the herbs prescribed by the
doctors diagnosing the disease, and the comparing results are
shown in Table 4. Given test symptoms, we can see that of
the eighteen prescription herbs prescribed by doctors, and
our model prescribed fifteen identical prescription herbs. Our
model also recommended seven herbs not prescribed by the
doctor. A doctor verified that these herbs are all known to
be associated with spleen deficiency. For example, ‘‘caulis
bambusae in taeniam’’ can replace ‘‘aster tataricus’’ to cure
the symptom ‘‘cough’’. At the bottom of the table, we give
the results of the quantitative validation for match degree
between the herbs given by the doctor and that generated by
SaGCN.


_E. COMPUTATIONAL EFFICIENCY_

We report running times on an Inter(R) Core(TM) i7-7700HQ
CPU @2.80GHz with 8 cores and 64GB memory.


1) STABILITY
We compare the convergence rates of different graph representation models by depicting the validation set loss in Fig 6.



Overall, SaGCN achieves faster convergence and lower error
in comparison to other models on all datasets. We think the
reason is that the SaGCN rationally combines more semantics
from all types of entities, which helps it to achieve steady
performance.


2) SCALABILITY
In order to illustrate its scalability, we apply SaGCN to
learn node representation on TCM datasets. We compute
the average runtime with increasing sizes from 100 to
1,000,000 nodes and average degree of 10. In Fig 6(a) we
empirically observe that SaGCN scales linearly with increase
in number of nodes generating representations for one million
nodes in less than three hours. In order to speed up training
the deep model, we use GCN with negative sampling. The
sampling procedure comprises of preprocessing for computing transition probabilities for our semantic. The optimization
phase is made efficient using negative sampling. In addition, we recorded the average runtime of each heterogeneous
graph representation model along with the increasing nodes
on TCMRel. Fig 6(b) shows that SaGCN achieved lower
average runtime in comparison to other heterogeneous graph



VOLUME 9, 2021 8805


C. Ruan _et al._ : SaGCNs for Clinical Auxiliary Diagnosis and Treatment of TCM


**FIGURE 6.** Comparison of validation loss w.r.t. epochs for all graph representation model.


**FIGURE 7.** (a) Scalability test of MsGCN on TCM graph with an average degree of 10 and
(b) scalability comparison of heterogeneous graph representation model.



representation models since it leverages relevant semantics
and negative sampling simultaneously.


**V. CONCLUSION**

In this article, we study an approach to discover regularities in
prescriptions, and we propose a model of clinical entities of
prescriptions as a heterogeneous TCM graph to address shortcomings of previous methods pursuing the same goals. Using
meta-graph and self-attention, our proposed model _SaGCN_ is
capable of capturing semantics for HIN. _SaGCN_ effectively
fuses semantics from multiple meta-graphs to learn clinical entities embedding through novel GCN. Experimental
results show that _SaGCN_ can achieve significantly better
performance in diagnosis task and prove the effectiveness
and robustness of _SaGCN_ . The model is helpful for clinical
research and practice. Future work includes incorporating
more diverse types of clinical information such as herbal
dosage, and more domain knowledge such as syndrome category as prior knowledge into our model. Evaluating herb roles
inferred by our model is another interesting task we are going
to investigate.


**REFERENCES**


[1] L. Yao, Y. Zhang, B. Wei, W. Zhang, and Z. Jin, ‘‘A topic modeling
approach for traditional Chinese medicine prescriptions,’’ _IEEE Trans._
_Knowl. Data Eng._, vol. 30, no. 6, pp. 1007–1021, Jun. 2018.




[2] X. Chen, C. Ruan, Y. Zhang, and H. Chen, ‘‘Heterogeneous information network based clustering for categorizations of traditional Chinese
medicine formula,’’ in _Proc. IEEE Int. Conf. Bioinf. Biomed. (BIBM)_,
Dec. 2018, pp. 839–846.

[3] W. Li, Z. Yang, and X. Sun, ‘‘Exploration on generating traditional chinese
medicine prescription from symptoms with an end-to-end method,’’ 2018,
_arXiv:1801.09030_ . [Online]. Available: http://arxiv.org/abs/1801.09030

[4] K. Yang, R. Zhang, L. He, Y. Li, W. Liu, C. Yu, Y. Zhang, X. Li, Y. Liu,
and W. Xu, ‘‘Multistage analysis method for detection of effective herb
prescription from clinical data,’’ _Frontiers Med._, vol. 2, no. 7, pp. 1–12,
2017.

[5] A. Hosseini, T. Chen, W. Wu, Y. Sun, and M. Sarrafzadeh, ‘‘HeteroMed:
Heterogeneous information network for medical diagnosis,’’ in _Proc. 27th_
_ACM Int. Conf. Inf. Knowl. Manage._, Oct. 2018, pp. 763–772.

[6] W. L. Hamilton, R. Ying, and J. Leskovec, ‘‘Representation learning on
graphs: Methods and applications,’’ _IEEE Data Eng. Bull._, vol. 40, no. 3,
pp. 52–74, 2017.

[7] Y. Fang, W. Lin, V. W. Zheng, M. Wu, K. C.-C. Chang, and X.-L. Li,
‘‘Semantic proximity search on graphs with metagraph-based learning,’’
in _Proc. IEEE 32nd Int. Conf. Data Eng. (ICDE)_, May 2016, pp. 277–288.

[8] J. F. Roddick, P. Fule, and W. J. Graco, ‘‘Exploratory medical knowledge
discovery: Experiences and issues,’’ _ACM SIGKDD Explor. Newslett._,
vol. 5, no. 1, pp. 94–99, Jul. 2003.

[9] C. Zhao, G.-Z. Li, C. Wang, and J. Niu, ‘‘Advances in patient classification for traditional Chinese medicine: A machine learning perspective,’’
_Evidence-Based Complementary Alternative Med._, vol. 2015, pp. 1–18,
Apr. 2015.

[10] Z. Jiang, H. Liu, B. Fu, Z. Wu, and T. Zhang, ‘‘Recommendation in
heterogeneous information networks based on generalized random walk
model and Bayesian personalized ranking,’’ in _Proc. 11th ACM Int. Conf._
_Web Search Data Mining_, Feb. 2018, pp. 288–296.



8806 VOLUME 9, 2021


C. Ruan _et al._ : SaGCNs for Clinical Auxiliary Diagnosis and Treatment of TCM


[11] X. Li, Y. Wu, M. Ester, B. Kao, X. Wang, and Y. Zheng, ‘‘Semi-supervised
clustering in attributed heterogeneous information networks,’’ in _Proc._
_26th Int. Conf. World Wide Web_, Apr. 2017, pp. 1621–1629.

[12] S. Hou, Y. Ye, Y. Song, and M. Abdulhayoglu, ‘‘HinDroid: An intelligent
Android malware detection system based on structured heterogeneous
information network,’’ in _Proc. 23rd ACM SIGKDD Int. Conf. Knowl._
_Discovery Data Mining_, Aug. 2017, pp. 1507–1515.

[13] C. Shi and S. Y. Philip, _Heterogeneous Information Network Analysis and_
_Applications_ . Cham, Switzerland: Springer, 2017.

[14] Z. Huang, Y. Zheng, R. Cheng, Y. Sun, N. Mamoulis, and X. Li, ‘‘Meta
structure: Computing relevance in large heterogeneous information networks,’’ in _Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data_
_Mining_, Aug. 2016, pp. 1595–1604.

[15] Y. Shi, H. Gui, Q. Zhu, L. Kaplan, and J. Han, ‘‘AspEm: Embedding
learning by aspects in heterogeneous information networks,’’ in _Proc._
_SIAM Int. Conf. Data Mining_, 2018, pp. 144–152.

[16] J. Yu, M. Gao, J. Li, H. Yin, and H. Liu, ‘‘Adaptive implicit friends
identification over heterogeneous network for social recommendation,’’ in
_Proc. 27th ACM Int. Conf. Inf. Knowl. Manage._, Oct. 2018, pp. 357–366.

[17] C. Shi, X. Kong, P. S. Yu, S. Xie, and B. Wu, ‘‘Relevance search in
heterogeneous networks,’’ in _Proc. 15th Int. Conf. Extending Database_
_Technol. (EDBT)_, 2012, pp. 180–191.

[18] Y. Fan, S. Hou, Y. Zhang, Y. Ye, and M. Abdulhayoglu, ‘‘Gotcha–sly
malware!: Scorpion a metagraph2vec based malware detection system,’’
in _Proc. 24th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining_,
Jul. 2018, pp. 253–262.

[19] R. Ying, R. He, K. Chen, P. Eksombatchai, W. L. Hamilton, and
J. Leskovec, ‘‘Graph convolutional neural networks for Web-scale recommender systems,’’ in _Proc. 24th ACM SIGKDD Int. Conf. Knowl. Discov-_
_ery Data Mining_, Jul. 2018, pp. 974–983.

[20] T. H. Nguyen and R. Grishman, ‘‘Graph convolutional networks with
argument-aware pooling for event detection,’’ in _Proc. AAAI_, vol. 18, 2018,
pp. 5900–5907.

[21] J. Bastings, I. Titov, W. Aziz, D. Marcheggiani, and K. Sima’an,
‘‘Graph convolutional encoders for syntax-aware neural machine translation,’’ 2017, _arXiv:1704.04675_ . [Online]. Available: https://arxiv.org/abs/
1704.04675

[22] M. Zitnik, M. Agrawal, and J. Leskovec, ‘‘Modeling polypharmacy
side effects with graph convolutional networks,’’ _Bioinformatics_, vol. 34,
no. 13, pp. i457–i466, Jul. 2018.

[23] A. Fout, J. Byrd, B. Shariat, and A. Ben-Hur, ‘‘Protein interface prediction
using graph convolutional networks,’’ in _Proc. Adv. Neural Inf. Process._
_Syst._, 2017, pp. 6530–6539.

[24] Z. Ying, J. You, C. Morris, X. Ren, W. Hamilton, and J. Leskovec, ‘‘Hierarchical graph representation learning with differentiable pooling,’’ in _Proc._
_Adv. Neural Inf. Process. Syst._, 2018, pp. 4800–4810.

[25] A. Sankar, X. Zhang, and K. C.-C. Chang, ‘‘Motif-based convolutional
neural network on graphs,’’ 2017, _arXiv:1711.05697_ . [Online]. Available:
http://arxiv.org/abs/1711.05697

[26] P. Kazienko, ‘‘Social network analysis: Selected methods and applications,’’ in _Proc. DATESO_, 2012, p. 151.

[27] H. Zhao, Q. Yao, J. Li, Y. Song, and D. L. Lee, ‘‘Meta-graph based recommendation fusion over heterogeneous information networks,’’ in _Proc._
_23rd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining_, Aug. 2017,
pp. 635–644.

[28] J. Chang, J. Gu, L. Wang, G. Meng, S. Xiang, and C. Pan, ‘‘Structureaware convolutional neural networks,’’ in _Proc. Adv. Neural Inf. Process._
_Syst._, 2018, pp. 11–20.

[29] M.-T. Luong, H. Pham, and C. D. Manning, ‘‘Effective approaches to
attention-based neural machine translation,’’ 2015, _arXiv:1508.04025_ .

[Online]. Available: https://arxiv.org/abs/1508.04025

[30] H. Wan, M.-F. Moens, W. Luyten, X. Zhou, Q. Mei, L. Liu, and J. Tang,
‘‘Extracting relations from traditional Chinese medicine literature via heterogeneous entity networks,’’ _J. Amer. Med. Inform. Assoc._, vol. 23, no. 2,
pp. 356–365, Mar. 2016.




[31] B. Xu, H. Shen, Q. Cao, Y. Qiu, and X. Cheng, ‘‘Graph wavelet neural
network,’’ in _Proc. ICLR_, 2019, pp. 1–13.

[32] C. Ruan, Y. Wang, Y. Zhang, J. Ma, H. Chen, U. Aickelin, S. Zhu, and
T. Zhang, ‘‘THCluster: Herb supplements categorization for precision
traditional Chinese medicine,’’ in _Proc. IEEE Int. Conf. Bioinf. Biomed._
_(BIBM)_, Nov. 2017, pp. 417–424.


CHUNYANG RUAN received the Ph.D. degree
from the School of Computer Science, Fudan University, Shanghai, China. He is currently a Lecturer with the Department of Data Science and
Big Data Technology, School of Economics and
Finance, Shanghai International Studies University, Shanghai. His main research interests include
graph embedding, graph neural networks, heterogeneous information networks, data analysis, and
knowledge discovery.


YINGPEI WU is currently pursuing the Ph.D.
degree with the School of Computer Science,
Fudan University, Shanghai, China. His current
research interests include heterogeneous information networks, graph neural networks, representation learning, and traditional Chinese medicine
informatization.


YUN YANG received the B.S. degree in traditional Chinese medicine and the M.S. degree in
internal medicine of traditional Chinese medicine

from the Shanghai University of Traditional Chinese Medicine, Shanghai, China, in 2008 and
2012, respectively. She is currently pursuing the
Ph.D. degree with Shanghai University of traditional Chinese Medicine of Internal medicine of

traditional Chinese Medicine. Her research inter
ests include the basic and clinical research of

traditional Chinese medicine treatment of tumor and the combination of

traditional Chinese medicine and artificial intelligence.


GUANGSHENG LUO (Member, IEEE) received
the B.S. degree in computer science from the
Huazhong University of Technology, Wuhan,
China, in 2005, and the M.S. degree in computer
science from Fudan University, Shanghai, China,
in 2015. He is currently a Lecturer with the Department of Data Science and Big Data Technology,
School of Economics and Finance, Shanghai International Studies University. His research interests include deep learning, smart financial, and
automatic speech recognition.



VOLUME 9, 2021 8807


