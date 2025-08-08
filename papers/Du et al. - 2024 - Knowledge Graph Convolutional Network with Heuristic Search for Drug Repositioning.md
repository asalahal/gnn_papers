pubs.acs.org/jcim Article

## **Knowledge Graph Convolutional Network with Heuristic Search for** **Drug Repositioning**

#### Xiang Du, Xinliang Sun, and Min Li*

### ACCESS Metrics & More Article Recommendations * sı Supporting Information


# ■ [INTRODUCTION] the first truly high-throughput sequencing platform in the mid
Drug repositioning (also called drug repurposing, reprofiling, 2000s, heralded a 50,000-fold drop in the cost of human
or retasking) is a strategy for identifying new uses for approved genome sequencing. The amount of sequencing data is
or investigational drugs that are outside the scope of the increasing exponentially. [4] In transcriptomics, the Human
original medical indication. [1] Drug repositioning provides a safe Protein Atlas [5] and the Genotype-Tissue Expression (GTEx) [6]
and efficient way to facilitate the market entry of potential project have generated atlases of gene expression of tissues, the
drugs, as repositioned drugs have already completed early trials brain, diseases, blood, and cells and quantified expression over
and have proven to be sufficiently safe in preclinical models common tissues, respectively. Given the diverse and complex
and humans. Therefore, drug repositioning requires less time biological data sources, a knowledge graph (KG), a semantic
and investment than developing new drugs for specific network comprising entities and their relations in the real
indications. For example, some repositioned drugs, such as world, [7] serves as an efficient means to integrate and store this
remdesivir, ritonavir, and tocilizumab, have provided rapid information. [8] Formally, a KG is a type of labeled multigraph [9]
response to the global coronavirus disease (COVID-19) that represents entities (also known as nodes or vertices) and
pandemic, [2] indicating that drug repositioning is a promising their relations (also known as edges, facts, or links). Two
strategy to fight against diseases.
With the continuous advancement of biomedical research
and the gradual accumulation of drug and pathological
information, [3] the scale of biological data is constantly Accepted: May 21, 2024
increasing, including various omics data such as genomics, Published: June 5, 2024
transcriptomics, proteomics, and metabolomics and related
information such as diseases, drugs, nutrition, etc. For instance,
in genomics, next-generation sequencing (NGS), the release of



the first truly high-throughput sequencing platform in the mid2000s, heralded a 50,000-fold drop in the cost of human
genome sequencing. The amount of sequencing data is
increasing exponentially. [4] In transcriptomics, the Human
Protein Atlas [5] and the Genotype-Tissue Expression (GTEx) [6]

project have generated atlases of gene expression of tissues, the
brain, diseases, blood, and cells and quantified expression over
common tissues, respectively. Given the diverse and complex
biological data sources, a knowledge graph (KG), a semantic
network comprising entities and their relations in the real
world, [7] serves as an efficient means to integrate and store this
information. [8] Formally, a KG is a type of labeled multigraph [9]

that represents entities (also known as nodes or vertices) and
their relations (also known as edges, facts, or links). Two





© 2024 American Chemical Society



**4928**



[https://doi.org/10.1021/acs.jcim.4c00737](https://doi.org/10.1021/acs.jcim.4c00737?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 4928−4937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


̅ ̅



entities connected by a relation are also known as the head and
tail or source node and target node.
Knowledge graph embedding (KGE) techniques aim to
discover missing links between entities in a knowledge graph.
KGE methods can be roughly classified into two main
categories: distance-based methods and semantic-matchingbased methods. Distance-based methods use translational
operations to measure the distance between the head and
the tail after applying the relation vector. The most
representative method in this category is TransE, [10] which
assumes that relations are translations from head entities to tail
entities. However, TransE cannot handle complex relations
such as one-to-many, many-to-one, or many-to-many. To
overcome this limitation, several variants of TransE have been
proposed, such as TransH, [11] TransR, [12] and TransD, [13] which
use different strategies to map entities and relations to different
spaces. Another distance-based method is RotatE, [14] which is
inspired by Euler’s identity and defines relations as rotations
from heads to tails in the complex plane. Semantic matchingbased methods use scoring functions to measure the semantic
similarity between the head, the relation, and the tail. [15]

RESCAL [16] is a seminal method in this category, which uses a
bilinear scoring function to solve a three-way matrix
factorization problem. However, RESCAL has a high computational cost and a large number of parameters, as it uses a dense
matrix to represent each relation. DistMult [17] reduces the
number of parameters by constraining the relation matrices to
be diagonal. ComplEx [18] extends DistMult to the complex
domain, which allows the same node to have different
representations depending on its position. Specifically, if a
node is represented asrepresented as _h_ ̅ when it is a tail, where _h_ when it is a head, then it is _h_ ̅ is the complex
conjugate of _h_ . This enables ComplEx to model antisymmetric
relations more effectively. QuatE [19] further extends ComplEx to
the quaternion domain and defines relations as rotations of
head entities. Then it computes the quaternion inner product
between the rotated head and the tail as the score. The goal of
semantic-matching-based methods is to maximize the scores of
positive triples and minimize the scores of negative triples. The
details and comparisons of these methods are provided in the
overview in ref 20. Despite the performance of the KGE
methods discussed above being promising, they often
encounter challenges in extracting high-order nonlinear
features, resulting in insufficient utilization of topological
structure information.
Graph neural networks (GNNs) can effectively extract the
structure and feature information on graph data. GNN [21] is
proposed as the first graph neural network, which is based on
information diffusion and relaxation mechanisms. Using the
spectrum of the graph Laplacian, Bruna et al. [22] first proposed
convolutional networks for graphs. In follow-up works, GCN [23]

constructed a simple graph convolution via a localized firstorder approximation. LightGCN [24] simplified the design of
GCN by including only the most essential component in
GCN�neighborhood aggregation. Inspired by the attention
mechanisms, GAT [25] defined the graph attention networks,
which learn the weight values between a node and its
neighbors. BridgeDPI [26] first constructs virtual nodes to bridge
the gap between drugs and proteins and then uses a Graph
Neural Network to capture the network-level information

−
among diverse drugs and proteins for predicting drug protein
interactions. CGraphDTA [27] designed a fusion protocol based
on multiscale convolutional neural networks and graph neural



networks, which can leverage both target sequence and

−
structure for predicting drug target binding affinity. GraphscoreDTA, [28] taking a combination of graph neural networks,
bitransport information mechanisms, and physics-based
distance terms, effectively captures mutual information

− −
between protein ligand pairs for predicting protein ligand
binding affinity. Guo et al. [29] designed a variational autoencoder
based on a gated mechanism and graph convolutions to extract

−
multilevel dependency information for predicting disease
miRNA associations. PSGCN [30] is a GCN-based method that
leverages the graph structure to capture the contextual

−
information on drug disease pairs for drug repositioning.
DRGCL [31] treats known drug−disease associations as a
topological graph and improves drug repositioning performance by constructing semantic graphs and applying graph
contrastive learning. AdaDR [32] is an adaptive GCN method for
drug repositioning that integrates both node features and
topological structures and models interactive information
between them with an adaptive graph convolution operation.
However, most of the GNN-based methods discussed above
are primarily designed for bipartite graphs, which do not
require consideration of diverse types of entities and relations,
thus limiting their performance on biomedical knowledge
graphs.
In recent years, with the development of GNNs, many KGbased methods have adopted GNNs as a basic component. For
example, in a recommendation system, KGCN [33] focuses on
the user preference for the relations in the KG and learns item
representations by aggregating and integrating neighborhood
information with bias. KGAT [34] designs a relational attention


̅ ̅

mechanism that calculates attention scores according to the
distance between head and tail entities in the relation space.
KACL [35] employs an attention mechanism and contrastive
learning to reduce recommendation-irrelevant information in
knowledge graphs and alleviate interaction domination issues.
In drug−drug interaction prediction, KGNN [36] introduced a

−
framework that predicts drug drug interactions by capturing
drugs and their potential neighborhoods via mining their
associated relations in a KG. In drug repositioning, DREAMwalk, [37] a method for drug repositioning leveraging biomedical
knowledge graphs, employs semantic information-guided
random walks to generate sequences of drug and disease
nodes, addressing challenges such as inadequate representation
due to gene dominance and the limited number of drug and
disease entities. DRKF [38] first extracts relations related to
Parkinson’s disease from medical literature to construct a
medical literature knowledge graph and then employs TransE
and DistMult methods to predict drug repositioning
candidates. Zhang et al. [39] proposed a neural network-based
literature-based discovery approach, which uses SemRep,
filtering rules, and an accuracy classifier developed on a
BERT variant to construct triples from PubMed and other
COVID-19-focused research literature, and then uses TransE,
RotatE, DistMult, and ComplEx methods to predict drug
repositioning candidates. However, in drug repositioning, more
research focuses on how to extract effective information from
the literature to construct knowledge graphs, and the models
used for prediction have limitations mentioned previously.
To overcome the mentioned limitations, we proposed a
knowledge graph convolutional network with a heuristic search
(KGCNH) for drug repositioning. We summarize the
contributions of this work as follows:


**4929** [https://doi.org/10.1021/acs.jcim.4c00737](https://doi.org/10.1021/acs.jcim.4c00737?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 4928−4937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


Figure 1. Overall architecture of KGCNH. KGCNH mainly consists of three modules: relation-aware attention mechanism, Gumbel-Softmaxbased heuristic search, and feature-based graph augmentation.




  - We introduce KGCNH, an novel end-to-end framework
for drug repositioning that utilizes biomedical knowledge graphs. It employs a relation-aware attention
mechanism to learn the importance of different
neighbors of entities under various relations. KGCNH
selectively aggregates neighborhood information based
on learned attention scores, enhancing the influence of
relevant neighbors while mitigating interference from
irrelevant ones. Furthermore, to augment KGCNH’s
performance and robustness, we introduce feature-based
enhanced views.

  - We introduce randomness into our model by using the
Gumbel-Softmax-based heuristic search module, which
enables the model to explore the optimal embeddings of
drugs and diseases.

  - We conduct extensive experiments to evaluate the
performance of our proposed method. The results
show that KGCNH outperforms competitive methods.
Moreover, the case studies present the potential of
KGCNH for real-world application.

# ■ [MATERIALS AND METHODS]

In this section, we describe the problem formulation, the data
sets, and the KGCNH framework. As Figure 1 depicts, our
framework comprises three components: (i) a relation-aware
attention mechanism, (ii) a Gumbel-Softmax-based heuristic
search, and (iii) a feature-based graph augmentation. The main
notations utilized throughout this article are presented in Table
1.
**Problem Formulation.** We begin by presenting the drug−
disease association graph and knowledge graph and then

−
describe the KG-based drug disease association prediction
task.

−
_Drug_ _Disease Association Graph._ We construct a bipartite
drug−disease association graph, _G_ d [= {] _U V_,, }, where _U_
represents drug nodes set in which | _U_ | is the number of drugs,
_V_ represents disease nodes set in which | _V_ | is the number of
diseases, and = {(, ) _u v u_ | _U v_, _V_ } is the edge set
representing observed associations between drugs and diseases.
The adjacency matrix of the graph is _A_ ∈ {0,1} [|] _[U]_ [|×|] _[V]_ [|] . In the



Table 1. Notations and Explanations


notations explanations


_G_ d drug-disease association graph
_U_, _V_, set of drugs, diseases, and edges
_A_ adjacency matrix of drug−disease association graph
| _U_ |, | _V_ | number of drugs and diseases
_G_ k knowledge graph
_E_, _R_ set of entities and relations

| _E_ | number of entities

| _R_ | number of relation types
_y_ ( _u_, _v_ ) score of drug _u_ being associated with disease _v_
e, r embeddings of entities and relations
_e_ _h_, _r_ (< _h_, _t_     - ), _e_ _t_ entity _h_, relation type between _h_ and _t_, entity _t_
e _h_, r _r_ (< _h_, _t_ >), e _t_ embedding of entity _h_, relation between _h_ and _t_ and
entity _t_
e [0] initial entity embeddings
e embeddings of drugs and diseases for prediction
_N_ ( _h_ ) set of neighbor entities of _h_
_α_ _ht_ attention score of entity _t_ to _h_
_w_ _ht_ relation weight of entity _t_ to _h_


matrix, for each entry _A_ _uv_ in _A_, if _A_ _uv_ = 1, it means that drug _u_ is
associated with disease _v_ . Note that if _A_ _uv_ = 0, it does not
necessarily mean there is no association between _u_ and _v_, as it
may be an unobserved potential association.
_Knowledge Graph._ We also have a knowledge graph _G_ k =
{ _E_, _R_ }, which is comprised of entity−relation−entity triples,
where _E_ represents set of entities in which | _E_ | is the number of
entities and _R_ represents the set of relations in which | _R_ | is the
number of relation types. For any knowledge graph triple ( _h_, _r_,
_t_ ), it indicates that there is a relation from head entity _h_ to tail
entity _t_, where _h_, _t_ ∈ _E_ and _r_ ∈ _R_ . For example, the triple
(methotrexate, treats, muscle cancer) states the fact that
methotrexate can be used to treat muscle cancer.
_KG-Based Drug Repositioning._ Given a drug−disease
association graph _G_ d and knowledge graph _G_ k, we defined a
score function _y_ ( _u_, _v_ ) = _f_ ( _u_, _v_ | _θ_, _G_ d, _G_ k ), where _y_ _uv_ is the score
of drug _u_ being associated with disease _v_ . The aim of the
optimization is generally to score an observed association of


**4930** [https://doi.org/10.1021/acs.jcim.4c00737](https://doi.org/10.1021/acs.jcim.4c00737?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 4928−4937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


**̂**



−
drug disease higher than an unobserved one. A higher score
implies a higher likelihood of the drug treating the disease.
**Data Sets.** To evaluate the performance of our proposed
model, we conducted experiments using the Hetionet data
set [40] and the DrugBank data set from the DRKG. [41] Table 2
displays the overall statistical results of the data sets, and more
[detailed statistics are available in the Supporting Information](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.4c00737/suppl_file/ci4c00737_si_001.pdf)
S1.


Table 2. Statistics of Data Sets


Hetionet DrugBank


no. drugs 1538 9708

no. diseases 136 1182

no. associations 1145 4968

no. entities 45158 19911

no. relations 15 6

no. triples 2249052 1419822


Hetionet contains 11 types of entities and 24 relation types
from 29 publicly available data sources. In order to reduce the
computational complexity, we performed the following
preprocessing on the Hetionet data set. We merge relations
of the same name. For instance, the relation “upregulates”
between compounds and genes, diseases and genes, and
anatomy and genes is treated as the same relation by us. As a
result, the number of relations decreased from 24 to 16. To
balance the data, we changed the “palliates” relation of 390
triples to “treats”. Thus, the preprocessed data contain 15
relations and 1145 associations between drugs and diseases.
Ioannidis et al. constructed a biomedical knowledge graph
based on the DrugBank database (version 5.1.5), which
contains 6 relation types, 1419822 triples, 4968 drug−disease
associations, and 19911 entities.
**Relation-Aware Attention Mechanism.** Inspired by the
interpretability and effectiveness of GAT, many previous
methods adopt GAT as the base architecture. However, the
original graph attention mechanism ignores the influence of
different types of relations on the attention scores. Therefore,
we designed a relation-aware knowledge attention mechanism
to calculate the attention score by incorporating the relation
embedding. Formally, the attention score _ht_ of the tail
entity _t_ to the head entity _h_ under the relation type _r_ (⟨ _h_, _t_ ⟩) is
computed as follows:


_T_
_ht_ = Leaky ReLU( _a W_ [( **e** _h_ **r** _r_ ( _h t_, ) ) **e** _t_ ]) (1)


**̂**



embedding initialization on model training and performance,
we drew inspiration from the annealing algorithm, utilized the
attention scores as heuristic information and introduced
randomness through Gumbel-Softmax. [42] Specifically, the
weight _w_ _ht_ is calculated as follows:


**̂**



exp((log( _ht_ ) + _g_ )/ )


**̂**



= _ht_ + _g_ _ht_

exp((log( _ht_ ) + _g_ _ht_ )/ ) + exp((log(1 _ht_ ) + _g_ _ht_


**̂**



_ht_ _g_

_w_ =


**̂**



_ht_ = _ht_ _ht_
exp((log( _ht_ ) + _g_ )/ ) + exp((log(1 _ht_ ) + _g_ )/ )


**̂**



+ + +


**̂**



_g_ _ht_ _g_


**̂**



(3)


where _g_ _ht_ and _g_ ′ _ht_ are independent and identically distributed
samples drawn from standard Gumbel distribution and _τ_ is the
temperature. The standard Gumbel distribution can be
sampled using inverse transform sampling by drawing a _ϵ_ ∈
Uniform(0, 1) and computing _g_ = − log(−log( _ϵ_ )).
Gumbel-Softmax has two properties. One is that for low
temperature ( _τ_ ≤ 1) the expected value of Gumbel-Softmax
random variables is close to the expected value of categorical
variables, i.e., the expected value of _w_ _ht_, ( _w_ _ht_ ) _ht_ [. As the]
temperature increases, the expected value of Gumbel-Softmax
random variables converges to a uniform distribution, i.e.,

( _w_ _ht_ ) 0.5 . The other property is that when temperature _τ_
goes to 0, samples from Gumbel-Softmax distributions move
toward binary, i.e., _w_ _ht_ is either close to 0 or close to 1; as
temperature _τ_ increases without bound, samples converge
toward a uniform distribution, i.e., _w_ _ht_ is close to 0.5.
Considering the properties of the Gumbel-Softmax function,
we employed a higher temperature coefficient during the initial
stages of model training. As training progresses, the temperature gradually decreases, which is akin to the annealing
algorithm. At higher temperatures, the sampled value _w_ _ht_ is
more uniformly distributed, resulting in a smaller impact of the
attention score _α_ _ht_ on _w_ _ht_ . As the temperature decreases, the
expected value of _w_ _ht_ approaches the attention score _α_ _ht_ . By
incorporating randomness with the Gumbel-Softmax distribution, the model can explore more optimal embeddings of drugs
and diseases.
For each entity _h_, the representation of its neighbors through
the linear combination is computed as follows:


**̂**



**e** _N_ ( ) _h_ = _w_ _ht_ **t** **e**
_t_ _N_


**̂**



(4)


**̂**



( ) _h_


**̂**



=


_t_ _N_ ( ) _h_


**̂**



_h_


**̂**



**̂**



exp( _ht_ )


**̂**



(2)


**̂**



Next, we used Aggregator inspired by LightGCN, which
discards the feature transformation and nonlinear activation to
update the representation of entity _h_ . More formally, the
representation of entity _h_ at the _l_ th layer is updated as


_l_ _l_ 1 _l_ 1
**e** _h_ = **e** _h_ + **e** _N_ ( ) _h_ (5)


where e _lh_ −1 and e _lN_ − ( 1 _h_ ) are the representations of _h_ and its
neighbors at the ( _l_ − 1)th graph propagation layer. For _l_ = 1,
the representations of entities e [0] are randomly initialized
embeddings. At the _l_ th graph propagation layer, which is
denoted as the final layer of graph propagation, the
representations of entities are given in _e_ _[l]_ . Subsequently, we
extract the representations of drugs and diseases from **̂** _e_ _[l]_,
denoted as e.
**Feature-Based Graph Augmentation.** To enhance
model performance and mitigate overfitting, we employed
feature-based graph augmentation to enrich the representation
of drugs and diseases by creating new features from existing
ones. Specifically, we randomly initialized the embeddings of
entities e [0], and then applied aggregation to obtain the


**4931** [https://doi.org/10.1021/acs.jcim.4c00737](https://doi.org/10.1021/acs.jcim.4c00737?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 4928−4937



**̂**



**̂**



| |× _E_ _d_ | |× _R_ _d_
where **e** and **r** are embeddings of entities
and relations, respectively, which are randomly initialized and
trainable, _d_ is the dimension of embeddings, e _h_ and e _t_ denote
the embeddings of head entity _h_ and tail entity _t_, r _r_ (⟨ _h_, _t_ ⟩)
represents the embedding of relation _r_ (⟨ _h_, _t_ ⟩), ⊙ denotes the
Hadamard product, _N_ ( _h_ ) denotes the set of neighbor entities of ∥ denotes the concatenation operation, _h_, and _a_ and _W_ are **̂**
trainable parameters.
**Gumbel-Softmax-Based Heuristic Search.** As previously mentioned, the embeddings of entities and relations are
randomly initialized, resulting in random initial attention
scores. Directly utilizing these attention scores may negatively
impact the model’s performance. To mitigate the effects of


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article

**̂**

**̂**

**̂**

**̂** **̃**


**̂**

**̂**

**̂**

**̂** **̃**


**̂**

**̂**

**̂**

**̂** **̃**



Figure 2. A toy example of generating an augmented view of a graph by using attribute shuffling.


**̂**

**̂**

**̂**

**̂** **̃**



**̂**
embeddings of drugs and diseases e _v_ 1 . We subsequently
introduced perturbation to e [0] using attribute shuffling to
construct the feature-based augmentation view. This augmented view, with the same topological structure as the
original graph but with permuted order of entity embeddings,
was then fed into the parameter-shared graph neural networks,
resulting in the augmented embeddings of drugs and diseases **̂**
e _v_ 2 . As shown in Figure 2, this is a toy example demonstrating
the generation of an augmented view of a graph through
attribute shuffling. **̂**
and **Model Training.** e _v_ **̂** 2 to obtain the final representation We simply concatenated embeddings e **̃** of drugs and e _v_ 1
diseases for predicting drug−disease associations. We

−
employed the inner product as a score function, and drug
disease association score _y_ ( _u_, _v_ ) is computed as follows:


_T_
_y u v_ (, ) = **e e** _u_ _v_ (6)


The BPR loss [43] is a commonly employed method for
optimizing models to capture pairwise associations. It is
defined as follows:


+ +
_L_ bpr (, _u v_, _v_ ) = log ( (, _y u v_ ) _y u v_ (, )) (7)


where ( _u_, _v_ [+] ) represents the observed association in the drug−
disease association graph, ( _u_, _v_ [−] ) denotes an unobserved
association with _A_ _uv_ − = 0, and _σ_ denotes the sigmoid function.

We utilize the Adam algorithm to optimize the trainable
parameters.

# ■ [RESULTS AND DISCUSSIONS]


**Evaluation Metrics.** We conducted 10-fold cross-valida
−
tion to evaluate our approach. We treat all of the known drug
disease associations as positive samples and randomly divided
them into ten equally sized subsets. In each fold, we used nine
subsets as the positive training set and the remaining subset as
the positive testing set. Then, we constructed negative samples
for both the positive training and the test sets. Using the
training set as an example, in each training iteration, for every
positive sample, _n_ negative samples are generated by randomly
pairing drugs and diseases that are not associated. Importantly,
these negative samples share the same drug as the positive
sample but differ from each other. Through the merging of
positive and negative samples, a training set is generated,
maintaining a positive to negative sample ratio of 1: _n_, where _n_
is a hyperparameter. The ratio of positive to negative samples
in the training set is specified in the hyperparameters’ setting
section. The test set had a balanced ratio of 1:1. We employed



**̂** the Area Under the Receiver Operating Characteristic curve

(AUROC) and the Area Under the Precision-Recall curve
(AUPRC) as evaluation metrics to assess the model performance, given their common utilization in drug repositioning
prediction tasks. To ensure reliable performance estimation,
we iteratively conducted the 10-fold cross-validation procedure

**̂** ten times and subsequently reported the average results. **Baseline Methods.** To evaluate the performance of our

proposed model, we compared KGCNH with the seven
methods listed below.


**̂**
**̂** **̃** - RotatE [14] treats each relation as a rotation from the head

to the tail entity in the complex vector space.

   - ComplEx [18] extends DistMult to the complex domain,
which allows the same node to have different
representations depending on its position. This enables
ComplEx to model antisymmetric relations more
effectively.

   - QuatE [19] further extends ComplEx to the quaternion
domain and defines each relation as a quaternion
rotation from the head to the tail in the quaternion

space.

   - GCN [23] is a model that applies neural networks to graph
data. GCN constructed a simple graph convolution via a
localized first-order approximation.

   - LightGCN [24] simplifies the design of GCN by including

−
only the most essential component in GCN neighborhood aggregation.

   - GAT [25] is inspired by the attention mechanisms and
defines the graph attention networks, which learn the
weight values between a node and its neighbors.

   - KGNN [36] introduces a framework that predicts DDIs by
capturing drugs and their potential neighborhoods via
mining their associated relations.

**Hyperparameter Setting.** In the KGCNH model, the
embedding dimension is set to 64 with a dropout rate of 0.2.
For the KGCNH model, the learning rates on the Hetionet
and DrugBank data sets are 0.002 and 0.0045, respectively,
with optimal positive-to-negative sample ratios of 1:10 and
1:50. The results of KGCNH performance under various
[positive-to-negative sample ratios are provided in Supporting](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.4c00737/suppl_file/ci4c00737_si_001.pdf)
[Information S2. Detailed hyperparameters for the baseline](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.4c00737/suppl_file/ci4c00737_si_001.pdf)
[models are provided in Supporting Information S3.](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.4c00737/suppl_file/ci4c00737_si_001.pdf)
**Performance of KGCNH in the Cross-Validation.** For a
fair comparison, we reported the average results of 10 times of
10-fold cross-validation on two data sets to demonstrate result
stability. As shown in Table 3, we emphasize the best results in


**4932** [https://doi.org/10.1021/acs.jcim.4c00737](https://doi.org/10.1021/acs.jcim.4c00737?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 4928−4937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article



Table 3. Performance Comparison of 10 Times 10-Fold
Cross-Validation between Our Method and Baselines over
Hetionet and DrugBank Data Sets


ComplEX 0.8073 0.7954 0.8320 0.8800

RotatE 0.7627 0.7279 0.7867 0.8311

QutatE 0.7334 0.7353 0.8106 0.8592

GCN 0.8770 0.8671 0.8371 0.8350

LightGCN 0.9094 0.9046 0.8158 0.8526

GAT 0.9058 0.9035 0.8537 0.8799

KGNN 0.7680 0.7487 0.8035 0.8105

KGCNH 0.9367 0.9383 0.8834 0.9091


bold. Notably, KGCNH consistently attained the best AUROC
and AUPRC values over both data sets. Specifically, KGCNH
achieves AUROC values of 0.9367 and 0.8834 on Hetionet
and DrugBank, respectively, which are improvements of 3.00%
and 3.47% compared to the second best methods LightGCN
and GAT. For the AUPRC values, KGCNH outperforms all
other methods, resulting in an average improvement of 3.73%
over LightGCN on HetioNet and about 3.30% over ComplEx
on DrugBank. The results of each 10-fold cross-validation are
highly consistent, proving that our model has convincing
performance and robustness(Figure 3). Additionally, to
validate KGCNH’s performance across diverse data sets, we
assessed its efficacy on the KEGG data set curated by Bang et
al. KGCNH achieved AUROC and AUPRC values of 0.93 and
[0.945, respectively. Further details are provided in Supporting](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.4c00737/suppl_file/ci4c00737_si_001.pdf)
[Information S4.](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.4c00737/suppl_file/ci4c00737_si_001.pdf)
**Test of KGCNH on Sparse Data.** In this section, we
investigate the model’s performance under sparse data
conditions. We randomly removed a fraction of data from
Hetionet at a ratio of 5%, 10%, 15%, and 20% to simulate the
scenario of incomplete data and evaluated the model by 10fold cross-validation. As shown in Figure 4, the observed
results show that more known triples are positively correlated



with better prediction results, and further demonstrates that

−
the number of triples is an important factor for the drug
disease association prediction. Additionally, our proposed
model consistently achieves high performance, outperforming
baseline models even when varying proportions of triplets are
removed.
**Ablation Study.** To assess the impact of key components
on the model’s performance, we devised two model variants:
KGCNH-w/o-GS, which removes the Gumbel-Softmax-based
heuristic search module, and KGCNH-w/o-Aug, which
removes the augmentation view module. We then employed
10-fold cross-validation to evaluate their performance. As
shown in Table 4, we highlight the best results in bold. Our
findings are as follows: (i) The complete version of KGCNH
consistently outperforms other variants, underscoring the
contribution of each component. (ii) The Gumbel-Softmaxbased heuristic search module is essential, facilitating the
model in finding superior embeddings and resulting in a
notable performance enhancement.
**Case Study.** To evaluate the practical use of KGCNH, we
conducted detailed case studies on the computationally
predicted candidate diseases for the two drugs, namely, lithium

−
and quetiapine. Specifically, we used all the known drug
disease associations in Hetionet as the training set and
considered the missing drug−disease associations as candidate
pairs. We subsequently ranked all the candidate diseases by the

−
computed drug disease association scores and verified the top

−
5 potential drug disease associations for each drug in the
CTD, [44] DrugCentral [45] databases, and PubMed.
_Lithium._ Lithium, an element in the alkali metals family, has
the atomic symbol Li, atomic number 3, and atomic weight

[6.938; 6.997]. Table 5 shows that KGCNH predicted five
candidate diseases for lithium, all of which are confirmed by
authoritative public databases (100% hit rate). For example,
bipolar disorder is a top predicted candidate disease, and
lithium has been proven as a first line treatment for it. [46] Figure
5 in DrugMechDB illustrates the path that represents the
mechanism of action from lithium to bipolar disorder.



Figure 3. Performance comparison of 10 times 10-fold cross-validation between our method and baselines over Hetionet and DrugBank data sets.
(a) Area under the receiver operating characteristic curves (AUROC) of prediction results. (b) Area under the precision-recall curves (AUPRC) of
prediction results.


**4933** [https://doi.org/10.1021/acs.jcim.4c00737](https://doi.org/10.1021/acs.jcim.4c00737?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 4928−4937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article

[̈] ̧



Figure 4. KGCNH performance for different sparsity ratios on Hetionet.


[̈] ̧



Table 4. Performance of KGCNH and Various Variants on
Hetionet and DrugBank Data Sets

[̈] ̧



KGCNH-w/o-GS 0.9226 0.9289 0.8699 0.9002

KGCNH-w/o-Aug 0.9339 0.9339 0.8816 0.9052

KGCNH 0.9367 0.9383 0.8833 0.909 [̈] ̧


Table 5. Top 5 Predicted Diseases Potentially Treatable by
Lithium


rank disease MESH evidence


1 Bipolar Disorder D001714 46
2 Tourette Syndrome D005879 47
3 Depressive Disorder D003866 48

4 Autistic Disorder D001321 49

5 Alzheimer’s Disease D000544 50


Tourette syndrome is another predicted candidate disease.
Erickson et al. [47] found that when the Li [+] blood levels had
stabilized at 0.8 to 0.9 mEq/L the major tics and involuntary
sounds cleared dramatically and the patients experienced no
side effects and were followed for several months without
recurrence of the original symptoms. For Depressive Disorder
also known as unipolar depression, Coppen [48] provided
substantial evidence that lithium treatment decreases morbidity, suggesting that systematic, long-term lithium treatment of
unipolar depression significantly reduced suicide rates. Autistic
disorder is a developmental disability that affects how people
interact with others and behave. Wang et al. [49] concluded that
environment-related lithium exposure protected against neurobehavioral deficits in the rat valproic acid model of autism,
implying that it may be a potential drug for the treatment of
autism. Alzheimer’s disease is a type of dementia that affects
memory, thinking, and behavior. Toledo and Inestrosa [50]

concluded that lithium and rosiglitazone, possibly by the
activation of the Wnt signaling pathway, reduced various
Alzheimer’s disease neuropathological markers and may be
considered as potential therapeutic agents against the disease.



_Quetiapine._ Similar to lithium, we also focused on analyzing
the top 5 disease candidates for quetiapine predicted by
KGCNH. Table 6 shows that these five potential diseases are
verified by reliable evidence with a 100% hit rate. Depressive
Disorder is a top predicted candidate disease, and Baune et
al. [51] found an independent influence of quetiapine on
improved depression, motor activity, and sleep. Matur and
U [̈] coķ [52] described a case of a young man who had Tourette
disorder and obsessive-compulsive disorder for 13 years and
developed mania on clomipramine. The mania resolved and
the symptoms of both disorders improved after he received
600 mg of quetiapine daily. Conduct disorder is a mental
disorder characterized by a repetitive and persistent pattern of
behavior that violates the basic rights of others or major ageappropriate societal norms or rules. Barzman et al. [53] found that
quetiapine can be used to treat impulsiveness and aggression in
adolescents with bipolar and disruptive behavior disorder.
Parkinson’s disease is a progressive, degenerative neurologic
disease. Quetiapine is an atypical antipsychotic with sedative
properties frequently used to treat hallucinations and psychosis
in Parkinson’s disease as reported by Juri et al. [54] Manic
Disorder is a type of anxiety disorder characterized by
unexpected panic attacks that last minutes or, rarely, hours.
Takahashi et al. [55] reported three patients who suffered from
panic attacks, and their symptoms improved significantly after
quetiapine.
In summary, these case studies demonstrate the promising
ability of KGCNH for discovering potential diseases of specific
drugs. We expect that the candidate diseases predicted by
KGCNH can provide a meaningful reference for clinicians in
practical applications.
**Discussion.** Compared with GCN, LightGCN, and GAT,
the KGCNH model variant without the Gumbel-Softmaxbased heuristic search module achieved superior performance,
mainly due to its relation-aware mechanism, which effectively
handles diverse entities and relationships, enhancing the
influence of relevant neighbors while mitigating interference
from irrelevant ones. Compared with ComplEx, RotatE, and
QuatE, KGCNH demonstrates superior performance, indicat

**4934** [https://doi.org/10.1021/acs.jcim.4c00737](https://doi.org/10.1021/acs.jcim.4c00737?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 4928−4937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


Figure 5. DrugMechDB-based mechanism of action path from lithium to a bipolar disorder.



Table 6. Top 5 Predicted Diseases Potentially Treatable by
Quetiapine


rank disease MESH evidence


1 Depressive Disorder D003866 51
2 Tourette Syndrome D005879 52

3 Conduct Disorder D019955 53

4 Parkinson’s Disease D010300 54, 56

5 Panic Disorder D016584 55


ing its ability to leverage neighborhood information more
comprehensively. Furthermore, the ablation experiment results
demonstrate that our designed Gumbel-Softmax-based heuristic search module can effectively introduce randomness,
expanding the model’s search space and finding better
representations of drugs and diseases. However, this module
still has limitations. First, it cannot guarantee finding the global
optimum. Second, it relies on parameter settings, such as initial
temperature and temperature drop rate.

# ■ [CONCLUSION]

In this article, we proposed a novel knowledge graph
convolutional network with a heuristic search for drug
repositioning, KGCNH. KGCNH incorporates relation
features into the attention score computation, enabling it to
effectively capture the importance between entities. To
mitigate the influence of random initial embedding on model
performance, we employed a heuristic search strategy based on
Gumbel-Softmax, which can effectively expand the model’s
search space and enable it to learn better representations of
drugs and diseases. We conducted extensive experiments to
evaluate the effectiveness of KGCNH in drug repositioning
tasks and compare it with several competing methods. The
results demonstrate that KGCNH significantly outperforms the



baselines. Furthermore, case studies suggest the ability of
KGCNH to predict potential drug−disease associations for
specific drugs.
Despite the superior performance of KGCNH, there is still
room for improvement. First, the model relies on transductive
learning, requiring retraining whenever new drugs or diseases
are introduced to the knowledge graph. In future work, we plan
to incorporate more contextual information about drugs and
diseases and integrate inductive learning techniques. Second,
our current model training involves the entire knowledge
graph, necessitating substantial computational resources. With
the default parameters that we have configured, employing an
NVIDIA A800 GPU, the training duration is approximately 1 h
for the Hetionet data set and 3.5 h for the DrugBank data set.
In future work, one can develop a sampler capable of effectively
filtering out irrelevant triples within the knowledge graph.

# ■ [ASSOCIATED CONTENT]


**Data Availability Statement**
The source code of KGCNH and the data underlying this
[article are available in our github repository at https://github.](https://github.com/xiang-Du/KGCNH)
[com/xiang-Du/KGCNH. The original data are available free of](https://github.com/xiang-Du/KGCNH)
[charge. Hetionet: https://github.com/hetio/hetionet; Drug-](https://github.com/hetio/hetionet)
[Bank (DRKG): https://github.com/gnn4dr/DRKG.](https://github.com/gnn4dr/DRKG)
 - **sı** **Supporting Information**
The Supporting Information is available free of charge at
[https://pubs.acs.org/doi/10.1021/acs.jcim.4c00737.](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00737?goto=supporting-info)


Performance of KGCNH under the various ratios of
positive to negative samples, statistics of data sets, the
hyperparameter settings for the baseline models, and
[performance of KGCNH on the KEGG data set (PDF)](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.4c00737/suppl_file/ci4c00737_si_001.pdf)


**4935** [https://doi.org/10.1021/acs.jcim.4c00737](https://doi.org/10.1021/acs.jcim.4c00737?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 4928−4937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


[́]

̀


̀


# ■ [AUTHOR INFORMATION]

**[Corresponding Author](https://pubs.acs.org/action/doSearch?field1=Contrib&text1="Min+Li"&field2=AllField&text2=&publication=&accessType=allContent&Earliest=&ref=pdf)**

Min Li − _School of Computer Science and Engineering, Central_
_South University, Changsha, Hunan 410083, China;_

[orcid.org/0000-0002-0188-1394; Email: limin@](https://orcid.org/0000-0002-0188-1394)
[mail.csu.edu.cn](mailto:limin@mail.csu.edu.cn)


**[Authors](https://pubs.acs.org/action/doSearch?field1=Contrib&text1="Xiang+Du"&field2=AllField&text2=&publication=&accessType=allContent&Earliest=&ref=pdf)**

Xiang Du − _School of Computer Science and Engineering,_
_Central South University, Changsha, Hunan 410083, China;_
_School of Information Engineering, Jiangxi University of_
_Science and Technology, Ganzhou, Jiangxi 341000, China_
Xinliang Sun − _School of Computer Science and Engineering,_
_Central South University, Changsha, Hunan 410083, China_


Complete contact information is available at:
[https://pubs.acs.org/10.1021/acs.jcim.4c00737](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00737?ref=pdf)


**Funding**
This work was supported by the National Natural Science
Foundation of China [62225209] and the science and
technology innovation program of Hunan Province [́]

[2021RC4008].

**Notes**
The authors declare no competing financial interest.

# ■ [ACKNOWLEDGMENTS]

We wish to thank the High Performance Computing Center of
Central South University for supporting this work.

# ■ [REFERENCES]


[(1) Ashburn, T. T.; Thor, K. B. Drug repositioning: identifying and](https://doi.org/10.1038/nrd1468)
[developing new uses for existing drugs.](https://doi.org/10.1038/nrd1468) _Nat. Rev. Drug Discovery_ 2004,
_3_, 673−683.
[(2) Zhou, Y.; Wang, F.; Tang, J.; Nussinov, R.; Cheng, F. Artificial](https://doi.org/10.1016/S2589-7500(20)30192-8)
[intelligence in COVID-19 drug repurposing.](https://doi.org/10.1016/S2589-7500(20)30192-8) _Lancet Digital Health_
2020, _2_, e667−e676.
(3) Yang, Z.; Liu, J.; Zhu, X.; Yang, F.; Zhang, Q.; Shah, H. A.
[FragDPI: a novel drug-protein interaction prediction model based on](https://doi.org/10.1007/s11704-022-2163-9)
[fragment understanding and unified coding.](https://doi.org/10.1007/s11704-022-2163-9) _Front. Comput. Sci._ 2023,
_17_, 175903.
[(4) Goodwin, S.; McPherson, J. D.; McCombie, W. R. Coming of](https://doi.org/10.1038/nrg.2016.49)
[age: ten years of next-generation sequencing technologies.](https://doi.org/10.1038/nrg.2016.49) _Nat. Rev._
_Genet._ 2016, _17_, 333−351.
[(5) Pontén, F.; Jirström, K.; Uhlen, M. The Human Protein Atlas�a](https://doi.org/10.1002/path.2440)
[tool for pathology.(6) The GTEx Consortium; Ardlie, K. G.; Deluca, D. S.; Segre](https://doi.org/10.1002/path.2440) _J. Pathol._ 2008, _216_, 387−393., A.̀ ̀
V.; Sullivan, T. J.; Young, T. R.; Gelfand, E. T.; Trowbridge, C. A.;
[Maller, J. B.; Tukiainen, T.; et al. The Genotype-Tissue Expression](https://doi.org/10.1126/science.1262110)
[(GTEx) pilot analysis: multitissue gene regulation in humans.](https://doi.org/10.1126/science.1262110) _Science_
2015, _348_, 648−660.
[(7) Wu, X.; Duan, J.; Pan, Y.; Li, M. Medical knowledge graph: Data](https://doi.org/10.26599/BDMA.2022.9020021)
[sources, construction, reasoning, and applications.](https://doi.org/10.26599/BDMA.2022.9020021) _Big Data Min. Anal._
2023, _6_, 201−217.
[(8) MacLean, F. Knowledge graphs and their applications in drug](https://doi.org/10.1080/17460441.2021.1910673)
[discovery.](https://doi.org/10.1080/17460441.2021.1910673) _Expert Opin. Drug Discovery_ 2021, _16_, 1057−1069.
(9) Rossi, A.; Barbosa, D.; Firmani, D.; Matinata, A.; Merialdo, P.
[Knowledge graph embedding for link prediction: A comparative](https://doi.org/10.1145/3424672)
[analysis.](https://doi.org/10.1145/3424672) _ACM Trans. Knowledge Knowl. Discovery Data_ 2021, _15_, 1−
49.
(10) Bordes, A.; Usunier, N.; Garcia-Duran, A.; Weston, J.;
Yakhnenko, O. Translating embeddings for modeling multi-relational
data. _Adv Neural Inf. Process. Syst._ 2013, _26_, 2787.
(11) Wang, Z.; Zhang, J.; Feng, J.; Chen, Z. Knowledge graph
embedding by translating on hyperplanes. _Proceedings of the AAAI_
_conference on artificial intelligence_ ; AAAI Press: Washington, DC, 2014.



(12) Lin, Y.; Liu, Z.; Sun, M.; Liu, Y.; Zhu, X. Learning entity and
relation embeddings for knowledge graph completion. _Proceedings of_
_the AAAI conference on artificial intelligence_ ; AAAI Press: Washington,
DC, 2015.
(13) Ji, G.; He, S.; Xu, L.; Liu, K.; Zhao, J. Knowledge graph
embedding via dynamic mapping matrix. _Proceedings of the 53rd_
_annual meeting of the association for computational linguistics and the 7th_
_international joint conference on natural language processing_ ; Association
for Computer Linguistics, 2015; Vol _1_ (Long papers), pp 687−696.
(14) Sun, Z.; Deng, Z.; Nie, J.; Tang, J. RotatE: Knowledge Graph
Embedding by Relational Rotation in Complex Space. _7th Interna-_
_tional Conference on Learning Representations, ICLR 2019_ ; 2019.
[(15) Choudhary, S.; Luthra, T.; Mittal, A.; Singh, R. A survey of](https://doi.org/10.48550/arXiv.2107.07842)
[knowledge graph embedding and their applications.](https://doi.org/10.48550/arXiv.2107.07842) _arXiv_ 2021,
No. arXiv:2107.07842.
[(16) Nickel, M.; Tresp, V.; Kriegel, H.-P. A three-way model for](https://doi.org/10.5555/3104482.3104584)
[collective learning on multi-relational data.](https://doi.org/10.5555/3104482.3104584) _ICML'11 Proceedings of the_
_28th International Conference on Machine Learning_ 2011, 809−816.
(17) Yang, B.; Yih, W.; He, X.; Gao, J.; Deng, L. Embedding Entities
and Relations for Learning and Inference in Knowledge Bases. _3rd_
_International Conference on Learning Representations, ICLR 2015, San_
_Diego, CA, USA, May 7_ − _9, 2015, Conference Track Proceedings_ ; 2015.
(18) Trouillon, T.; Welbl, J.; Riedel, S.; Gaussier, E [́] .; Bouchard, G.
Complex embeddings for simple link prediction. _International_
_conference on machine learning_ ; Association for Computing Machinery,
2016; pp 2071−2080.
(19) Zhang, S.; Tay, Y.; Yao, L.; Liu, Q. Quaternion knowledge
graph embeddings. _Adv Neural Inf. Process. Syst._ 2019, _32_, 2735.
[(20) Ge, X.; Wang, Y.-C.; Wang, B.; Kuo, C.-C. J. Knowledge Graph](https://doi.org/10.48550/arXiv.2309.12501)
[Embedding: An Overview.](https://doi.org/10.48550/arXiv.2309.12501) _arXiv_ 2023, No. arXiv:2309.12501.
(21) Scarselli, F.; Gori, M.; Tsoi, A. C.; Hagenbuchner, M.;
[Monfardini, G. The graph neural network model.](https://doi.org/10.1109/TNN.2008.2005605) _IEEE Trans. Neural_
_Networks_ 2009, _20_, 61−80.
(22) Bruna, J.; Zaremba, W.; Szlam, A.; LeCun, Y. Spectral
Networks and Locally Connected Networks on Graphs. _2nd_
_International Conference on Learning Representations, ICLR 2014,_
_Banff, AB, Canada, April 14_ − _16, 2014, Conference Track Proceedings_ ;
2014.
(23) Kipf, T. N.; Welling, M. Semi-Supervised Classification with
Graph Convolutional Networks. _5th International Conference on_

−
_Learning Representations, ICLR 2017, Toulon, France, April 24_ _26,_
_2017, Conference Track Proceedings_ ; 2017.
(24) He, X.; Deng, K.; Wang, X.; Li, Y.; Zhang, Y.; Wang, M.
Lightgcn: Simplifying and powering graph convolution network for
recommendation. _Proceedings of the 43rd International ACM SIGIR_
_conference on research and development in Information Retrieval_ ;

̀ Association for Computing Machinery, 2020; pp 639(25) Velickovic, P.; Cucurull, G.; Casanova, A.; Romero, A.; Lio−648., P.;̀

Bengio, Y. Graph Attention Networks. _6th International Conference on_
_Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30_

_- May 3, 2018, Conference Track Proceedings_ ; 2018.
[(26) Wu, Y.; Gao, M.; Zeng, M.; Zhang, J.; Li, M. BridgeDPI: a](https://doi.org/10.1093/bioinformatics/btac155)
[novel graph neural network for predicting drug−protein interactions.](https://doi.org/10.1093/bioinformatics/btac155)
_Bioinformatics_ 2022, _38_, 2571−2578.
[(27) Wang, K.; Li, M. Fusion-Based Deep Learning Architecture for](https://doi.org/10.1109/JBHI.2023.3315073)
[Detecting Drug-Target Binding Affinity Using Target and Drug](https://doi.org/10.1109/JBHI.2023.3315073)
[Sequence and Structure.](https://doi.org/10.1109/JBHI.2023.3315073) _IEEE J. Biomed. Health Inf_ 2023, _27_, 6112.
[(28) Wang, K.; Zhou, R.; Tang, J.; Li, M. GraphscoreDTA:](https://doi.org/10.1093/bioinformatics/btad340)

−
[optimized graph neural network for protein](https://doi.org/10.1093/bioinformatics/btad340) ligand binding affinity
[prediction.](https://doi.org/10.1093/bioinformatics/btad340) _Bioinformatics_ 2023, _39_, btad340.
[(29) Guo, Y.; Zhou, D.; Ruan, X.; Cao, J. Variational gated](https://doi.org/10.1016/j.neunet.2023.05.052)
[autoencoder-based feature extraction model for inferring disease-](https://doi.org/10.1016/j.neunet.2023.05.052)
[miRNA associations based on multiview features.](https://doi.org/10.1016/j.neunet.2023.05.052) _Neural Networks_

2023, _165_, 491−505.
[(30) Sun, X.; Wang, B.; Zhang, J.; Li, M. Partner-Specific Drug](https://doi.org/10.1109/JBHI.2022.3194891)
[Repositioning Approach Based on Graph Convolutional Network.](https://doi.org/10.1109/JBHI.2022.3194891)
_IEEE J. Biomed. Health Inf._ 2022, _26_, 5757−5765.


**4936** [https://doi.org/10.1021/acs.jcim.4c00737](https://doi.org/10.1021/acs.jcim.4c00737?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 4928−4937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


[̈] ̧



[(31) Jia, X.; Sun, X.; Wang, K.; Li, M. DRGCL: Drug Repositioning](https://doi.org/10.1109/JBHI.2024.3372527)
[via Semantic-enriched Graph Contrastive Learning.](https://doi.org/10.1109/JBHI.2024.3372527) _IEEE J. Biomed._
_Health Inf._ 2024, 3372527.
[(32) Sun, X.; Jia, X.; Lu, Z.; Tang, J.; Li, M. Drug repositioning with](https://doi.org/10.1093/bioinformatics/btad748)
[adaptive graph convolutional networks.](https://doi.org/10.1093/bioinformatics/btad748) _Bioinformatics_ 2024, _40_,
btad748.
(33) Wang, H.; Zhao, M.; Xie, X.; Li, W.; Guo, M. Knowledge graph [̈] ̧
convolutional networks for recommender systems. _The world wide web_

−
_conference_ ; Association for Computing Machinery, 2019; pp 3307
3313.
(34) Wang, X.; He, X.; Cao, Y.; Liu, M.; Chua, T.-S. Kgat:
Knowledge graph attention network for recommendation. _Proceedings_
_of the 25th ACM SIGKDD international conference on knowledge_
_discovery & data mining_ ; Association for Computing Machinery, 2019;
pp 950−958.
(35) Wang, H.; Xu, Y.; Yang, C.; Shi, C.; Li, X.; Guo, N.; Liu, Z.
Knowledge-adaptive contrastive learning for recommendation.
_Proceedings of the sixteenth ACM international conference on web search_
_and data mining_ ; Association for Computing Machinery, 2023; pp
535−543.
[(36) Lin, X.; Quan, Z.; Wang, Z.-J.; Ma, T.; Zeng, X. KGNN:](https://doi.org/10.24963/ijcai.2020/380)
[Knowledge Graph Neural Network for Drug-Drug Interaction](https://doi.org/10.24963/ijcai.2020/380)
[Prediction.](https://doi.org/10.24963/ijcai.2020/380) _Proceedings of the Twenty-Ninth International Joint_
_Conference on Artificial Intelligence_ 2020, 2739−2745.
[(37) Bang, D.; Lim, S.; Lee, S.; Kim, S. Biomedical knowledge graph](https://doi.org/10.1038/s41467-023-39301-y)
[learning for drug repurposing by extending guilt-by-association to](https://doi.org/10.1038/s41467-023-39301-y)
[multiple layers.](https://doi.org/10.1038/s41467-023-39301-y) _Nat. Commun._ 2023, _14_, 3570.
[(38) Zhang, X.; Che, C. Drug repurposing for parkinson’s disease by](https://doi.org/10.3390/fi13010014)
[integrating knowledge graph completion model and knowledge fusion](https://doi.org/10.3390/fi13010014)
[of medical literature.](https://doi.org/10.3390/fi13010014) _Future Internet_ 2021, _13_, 14.
(39) Zhang, R.; Hristovski, D.; Schutte, D.; Kastrin, A.; Fiszman, M.;
[Kilicoglu, H. Drug repurposing for COVID-19 via knowledge graph](https://doi.org/10.1016/j.jbi.2021.103696)
[completion.](https://doi.org/10.1016/j.jbi.2021.103696) _J. Biomed. Inf._ 2021, _115_, 103696.
(40) Himmelstein, D. S.; Lizee, A.; Hessler, C.; Brueggeman, L.;
Chen, S. L.; Hadley, D.; Green, A.; Khankhanian, P.; Baranzini, S. E.
[Systematic integration of biomedical knowledge prioritizes drugs for](https://doi.org/10.7554/eLife.26726)
[repurposing.](https://doi.org/10.7554/eLife.26726) _Elife_ 2017, _6_, No. e26726.
(41) Ioannidis, V. N.; Song, X.; Manchanda, S.; Li, M.; Pan, X.;
Zheng, D.; Ning, X.; Zeng, X.; Karypis, G. _DRKG - Drug Repurposing_
_Knowledge Graph for Covid-19_ [. https://github.com/gnn4dr/DRKG/,](https://github.com/gnn4dr/DRKG/)
2020.
(42) Jang, E.; Gu, S.; Poole, B. Categorical Reparameterization with
Gumbel-Softmax. _5th International Conference on Learning Representa-_

−
_tions, ICLR 2017, Toulon, France, April 24_ _26, 2017, Conference Track_
_Proceedings_ ; 2017.
(43) Rendle, S.; Freudenthaler, C.; Gantner, Z.; Schmidt-Thieme, L.
[BPR: Bayesian personalized ranking from implicit feedback.](https://doi.org/10.48550/arXiv.1205.2618) _arXiv_
2012, No. arXiv:1205.2618.
(44) Davis, A. P.; Grondin, C. J.; Johnson, R. J.; Sciaky, D.; Wiegers,
[J.; Wiegers, T. C.; Mattingly, C. J. Comparative toxicogenomics](https://doi.org/10.1093/nar/gkaa891)
[database (CTD): update 2021.](https://doi.org/10.1093/nar/gkaa891) _Nucleic Acids Res._ 2021, _49_, D1138−
D1143.
(45) Ursu, O.; Holmes, J.; Knockel, J.; Bologa, C. G.; Yang, J. J.;
[Mathias, S. L.; Nelson, S. J.; Oprea, T. I. DrugCentral: online drug](https://doi.org/10.1093/nar/gkw993)
[compendium.](https://doi.org/10.1093/nar/gkw993) _Nucleic Acids Res._ 2017, D932.
(46) Simhandl, C.; Mersch, J. Lithium and bipolar disorder−a
renaissance? _Neuropsychiatrie: Klinik, Diagnostik, Therapie und_
_Rehabilitation: Organ der Gesellschaft Osterreichischer Nervenarzte und_
_Psychiater_ 2007, _21_, 121−130.
[(47) Erickson, H.; Goggin, J.; Messiha, F. Comparison of lithium](https://doi.org/10.1007/978-1-4684-2511-6_11)
[and haloperidol therapy in Gilles de la Tourette syndrome.](https://doi.org/10.1007/978-1-4684-2511-6_11) _Adv. Exp._
_Med. Biol._ 1977, _90_, 197−205.
(48) Coppen, A. Lithium in unipolar depression and the prevention
of suicide. _J. Clin. Psychiatry_ 2000, _61_, 52−56.
[(49) Wang, J.; Xu, C.; Liu, C.; Zhou, Q.; Chao, G.; Jin, Y. Effects of](https://doi.org/10.1016/j.cbi.2022.110314)
[different doses of lithium on the central nervous system in the rat](https://doi.org/10.1016/j.cbi.2022.110314)
[valproic acid model of autism.](https://doi.org/10.1016/j.cbi.2022.110314) _Chem.-Biol. Interact._ 2023, _370_, 110314.
[(50) Toledo, E.; Inestrosa, N. Activation of Wnt signaling by lithium](https://doi.org/10.1038/mp.2009.72)
[and rosiglitazone reduced spatial memory impairment and neuro-](https://doi.org/10.1038/mp.2009.72)



[degeneration in brains of an APPswe/PSEN1ΔE9 mouse model of](https://doi.org/10.1038/mp.2009.72)
[Alzheimer’s disease.](https://doi.org/10.1038/mp.2009.72) _Mol. Psychiatry_ 2010, _15_, 272−285.
[(51) Baune, B. T.; Caliskan, S.; Todder, D. Effects of adjunctive](https://doi.org/10.1002/hup.817)
[antidepressant therapy with quetiapine on clinical outcome, quality of](https://doi.org/10.1002/hup.817)
[sleep and daytime motor activity in patients with treatment-resistant](https://doi.org/10.1002/hup.817)
[depression.](https://doi.org/10.1002/hup.817) _Hum. Psychopharmacol._ 2007, _22_, 1−9.
(52) Matur, Z.; U [̈] çok, A. Quetiapine treatment in a patient with
Tourette’s syndrome, obsessive-compulsive disorder and druginduced mania. _Isr. J. Psychiatry Relat. Sci._ 2003, _40_, 150.
(53) Barzman, D. H.; DelBello, M. P.; Adler, C. M.; Stanford, K. E.;
[Strakowski, S. M. The efficacy and tolerability of quetiapine versus](https://doi.org/10.1089/cap.2006.16.665)
[divalproex for the treatment of impulsivity and reactive aggression in](https://doi.org/10.1089/cap.2006.16.665)
[adolescents with co-occurring bipolar disorder and disruptive](https://doi.org/10.1089/cap.2006.16.665)
[behavior disorder (s).](https://doi.org/10.1089/cap.2006.16.665) _J. Child Adolescent Psychopharmacol._ 2006, _16_,
665−670.
(54) Juri, C.; Chaná, P.; Tapia, J.; Kunstmann, C.; Parrao, T.
[Quetiapine for insomnia in Parkinson disease: results from an open-](https://doi.org/10.1097/01.wnf.0000174932.82134.e2)
[label trial.](https://doi.org/10.1097/01.wnf.0000174932.82134.e2) _Clin. Neuropharmacol._ 2005, _28_, 185−187.
(55) Takahashi, H.; Sugita, T.; Yoshida, K.; Higuchi, H.; Shimizu, T.
[Effect of quetiapine in the treatment of panic attacks in patients with](https://doi.org/10.1176/jnp.16.1.113)
[schizophrenia: 3 case reports.](https://doi.org/10.1176/jnp.16.1.113) _J. Neuropsychiatry Clin. Neurosci._ 2004,
_16_, 113−115.
(56) Cannas, A.; Solla, P.; Floris, G.; Tacconi, P.; Loi, D.; Marcia, E.;
[Marrosu, M. G. Hypersexual behaviour, frotteurism and delusional](https://doi.org/10.1016/j.pnpbp.2006.05.012)
[jealousy in a young parkinsonian patient during dopaminergic therapy](https://doi.org/10.1016/j.pnpbp.2006.05.012)
[with pergolide: A rare case of iatrogenic paraphilia.](https://doi.org/10.1016/j.pnpbp.2006.05.012) _Prog. Neuro-_
_psychopharmacol Biol. Psychiatry_ 2006, _30_, 1539−1541.

[̈] ̧



**4937** [https://doi.org/10.1021/acs.jcim.4c00737](https://doi.org/10.1021/acs.jcim.4c00737?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 4928−4937


