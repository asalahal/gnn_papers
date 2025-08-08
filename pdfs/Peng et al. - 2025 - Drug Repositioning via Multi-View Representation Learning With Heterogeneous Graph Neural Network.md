1668 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 3, MARCH 2025

## Drug Repositioning via Multi-View Representation Learning With Heterogeneous Graph Neural Network


Li Peng, Cheng Yang, Jiahuai Yang, Yuan Tu, Qingchun Yu, Zejun Li, Min Chen, and Wei Liang



_**Abstract**_ **—Exploring simple and efficient computational**
**methods for drug repositioning has emerged as a popular**
**and compelling topic in the realm of comprehensive drug**
**development. The crux of this technology lies in identifying**
**potential drug-disease associations, which can effectively**
**mitigate the burdens caused by the exorbitant costs and**
**lengthy periods of conventional drugs development. How-**
**ever, existing computational drug repositioning methods**
**continue to encounter challenges in accurately predicting**
**associations between drugs and diseases. In this paper,**
**we propose a Multi-view Representation Learning method**
**(MRLHGNN) with Heterogeneous Graph Neural Network for**
**drug repositioning. This method is based on a collection of**
**data from multiple biological entities associated with drugs**
**or diseases. It consists of a view-specific feature aggrega-**
**tion module with meta-paths and auto multi-view fusion en-**
**coder. To better utilize local structural and semantic infor-**
**mation from specific views in heterogeneous graph, MRL-**
**HGNN employs a feature aggregation model with variable-**
**length meta-paths to expand the local receptive field. Ad-**
**ditionally, it utilizes a transformer based semantic aggre-**
**gation module to aggregate semantic features across dif-**
**ferent view-specific graphs. Finally, potential drug-disease**
**associations are obtained through a multi-view fusion de-**
**coder with an attention mechanism. Cross-validation exper-**
**iments demonstrate the effectiveness and interpretability**
**of the MRLHGNN in comparison to nine state-of-the-art**
**approaches. Case studies further reveal that MRLHGNN**
**can serve as a powerful tool for drug repositioning.**


Received 11 July 2023; revised 7 October 2023, 28 November 2023,
7 February 2024, and 22 July 2024; accepted 23 July 2024. Date of
publication 29 July 2024; date of current version 7 March 2025. This
work was supported in part by the National Natural Science Foundation
of China under Grant 62172158, in part by the National Natural Science
Foundation of Hunan Province under Grant 2023JJ30264 and Grant
2024JJ7115, and in part by the Scientific Research Project of Hunan
Education Department under Grant 22A0350. _(Corresponding authors:_
_Li Peng; Wei Liang.)_

Li Peng and Wei Liang are with the School of Computer Science and
Engineering, Hunan University of Science and Technology, Xiangtan
411201, China, and also with the Hunan Key Laboratory for Service
computing and Novel Software Technology Xiangtan 411201, China
[(e-mail: plpeng@hnu.edu.cn; wliang@hnust.edu.cn).](mailto:plpeng@hnu.edu.cn)

Cheng Yang, Jiahuai Yang, Yuan Tu, and Qingchun Yu are with the
School of Computer Science and Engineering, Hunan University of Sci[ence and Technology, Xiangtan 411201, China (e-mail: yangchengyjs@](mailto:yangchengyjs@163.com)
[163.com; jason1325070309@163.com; ty09060418@163.com; yqc@](mailto:yangchengyjs@163.com)
[hnust.edu.cn).](mailto:yqc@hnust.edu.cn)

Zejun Li and Min Chen are with the School of Computer Science and
Engineering, Hunan Institute of Technology, Hengyang 421002, China
[(e-mail: lzjfox@hnit.edu.cn; chenmin@hnit.edu.cn).](mailto:lzjfox@hnit.edu.cn)

Digital Object Identifier 10.1109/JBHI.2024.3434439



_**Index Terms**_ **—Drug, disease, drug repositioning, multi-**
**view representation learning, heterogeneous graph neural**
**network, meta-path.**


I. I NTRODUCTION


HE development of symptomatic drugs for an emerging
# T disease is a high-risk, extremely expensive and time
consuming process with a low success rate [1]. Drug repositioning can apply existing drugs or compounds to new therapeutic
targets, and also provide an efficient screening method for drug
combination strategies to treat diseases [2], [3]. This approach
is particularly effective in treating complex diseases because
it increases the likelihood of identifying effective therapeutic
drug combinations [4], [5]. In addition, the method provides
a reliable candidate for the design of targeted drugs, thereby
shortening the drug development cycle. Revealing potential
drug-disease associations is a crucial step in the process of
drug repositioning, which aims to identify new indications for
existing drugs. It reduces the cost of unnecessary biological
experiments and enhances the success rate of drug development.
With the explosive growth of medical data and the advancement
of deep learning techniques, various data-driven computational
models have been proposed specifically for predicting drug
small molecule-related markers [6], [7], [8], [9], disease-related
markers [10], [11], [12]. These methodologies have significant
implications for drug repositioning, and some can even be
directly applied. The primary objective of drug repositioning
approaches is to generate predictions for unknown drug-disease
pairs or reconstruct drug-disease association matrix based on
representations of drug and disease feature. Existing predictive
approaches for drug repositioning can be broadly classified into
two categories: classical machine learning methods and deep
learning techniques.

Traditional approaches primarily focus on matrix completion and network propagation. Zhang et al. [13] used known
associations to construct similarity measure graphs for diseases
and drugs, and then derived potential drug-disease association
scores using label propagation. Yang et al. [14] constructed
an heterogeneous association matrix based on known association and similarities, and subsequently utilized the bounded
kernel norm approach (BNNR) to derive association scores
of potential drug-disease associations. Afterwards, yang et



2168-2194 © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.

See https://www.ieee.org/publications/rights/index.html for more information.


Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:29:01 UTC from IEEE Xplore. Restrictions apply.


PENG et al.: DRUG REPOSITIONING VIA MULTI-VIEW REPRESENTATION LEARNING WITH HETEROGENEOUS GRAPH NEURAL NETWORK 1669


al. [15] proposed HGIMC, a heterogeneous graph inference
method based on matrix complementation (HGIMC), to predict
new drug indications using heterogeneous association information. Ji et al. [16] developed DTINet, a method for predicting new drug-target interactions through label propagation
and induction matrix completion based on network similarity
information.



Currently, several new deep learning-based heterogeneous
graph neural network approaches are being proposed to model
drug-disease associations, offering high flexibility and scalability. Their performance results are significantly more competitive
than traditional approaches. For instance, LAGCN [17] is Layer
Attention Graph Convolutional Network that utilizes layer attention mechanisms to learn representations of drug or disease
nodes in a heterogeneous drug-disease network. DRHGCN [18]
is an information fusion-based graph convolutional network designedwithinter-andintra-domainembeddingsforlearningrepresentations of drug or disease nodes. DRWBNCF [19] is a neural collaborative filtering approach for drug repositioning based
on neighbourhood interactions. The three aforementioned methods focus solely on predicting missing values in the drug-disease
bipartite association network, overlooking related biological
entities and their rich semantic context. REDDA [20] merges
three attention mechanisms to learn drug/disease representations
from the node embedding block, topological subnet embedding
block, graph attention block and layer attention block of a heterogeneous graph convolutional network in a sequential process.
However, REDDA integrates enhanced biological node data but
limits its graph convolution to first-order neighbors, neglecting
multi-order semantic insights and local structure of heterogeneous graph. Based on heterogeneous graph networks, Gu et
al. [21] proposed a multi-instance learning method (MilGNet)
for drug repositioning. MilGNet learns node feature representations at the meta-path level using a pseudo-meta-path instance
generator and bidirectional translational embedding projector.
MilGNet primarily focuses on meta-path level feature aggregation, neglecting the view level. And, its meta-path instance
enumeration significantly increases computational memory usage as the graph’s entity count grows. Additionally, multi-view
representation learning [22], [23], [24], which aggregates node
features through meta-paths of drugs’ view-specific graphs and
diseases’ view-specific graphs, and has not yet been examined
in the field of drug repositioning.

To address these issues, we proposed a Multi-view Representation Learning with Heterogeneous Graph Neural Network, called MRLHGNN, for drug repositioning. MRLHGNN
consists of two main modules that are innovative in the drug
repositioning: a module for multi-view representation learning,
which explores the intrinsic heterogeneity between nodes, and
an auto multi-view fusion decoder for predicting drug-disease
associations. After conducting a series of comprehensive experiments, it has been conclusively proven that MRLHGNN
outperforms9state-of-the-artmethodsonthebenchmarkdataset
and has potential for discovering new drug-disease associations.
In summary, the aims and contributions of this work can be
summarized as follows:



_Definition 3 (Metapath-based Graph):_ Given a metapath
_P_ _v_ : _O_ 1 _→_ _O_ 2 _→_ _. . . →_ _O_ _N_ and a target node _v_ with edge
type _Q_ _l−_ 1, the view-specific graph _GP_ _v_ is a directed graph
evoked from metapath-based neighbourhoods and intermediate
nodes following the metapath along with _v_ itself. An illustrative
example is shown in Fig. 1(c).



Fig. 1. An illustrative example of a heterogeneous graph and some
key concepts. (a) The local structure of node _R_ 1 in the heterogeneous
graph. (b) Two metapaths in heterogeneous graph. (c) Three types of
metapath-based local structures of drug node _R_ 1 .


r To enhance the characterization of drug and disease nodes

in the heterogeneous graph, we incorporated other biological entities (such as proteins and pathways) that are
associated with a particular drug or disease, resulting in a
more locally informative heterogeneous graph.
r This work proposed a view-specific feature aggregation

module, including metapath-level node feature aggregation with attention-based and view-level node feature fu
sion with transformer-based, to leverage the inherent heterogeneity information of local structures in the specific
view.

r We employed an automatic multi-view fusion encoder to

generate the final drug-disease association matrix, facilitating more accurate drug repositioning. Moreover, case
studies have affirmed the efficacy of the MRLHGNN
approach.


II. P ROBLEM D EFINITION


_Definition 1 (Heterogeneous Graph):_ A heterogeneous graph
_G_ = ( _V, E, T_ _v_ _, T_ _e_ ) is made up of a vertex set _v_ _i_ _∈V_ and an
edge set _e_ _i,j_ _∈E_, along with node type mapping function
_M_ _v_ ( _v_ _i_ ) : _V →T_ _v_ _i_ and edge type mapping function _M_ _e_ ( _e_ _i,j_ ) :
_E →T_ _e_ _i,j_ . _O_ and _Q_ denote the predefined sets of object types
and edge types, respectively, where _|O|_ + _|Q| >_ 2. An example
of this is given in Fig. 1(a).

_Definition 2 (Metapath):_ Consider _O_ _i_ _∈O_ and _Q_ _i_ _∈Q_ denote a node type and an edge type, respectively. A metapath
starting at node type _O_ 1 and ending at node type _O_ _l_ can be

_Q_ 1 _Q_ 2 _Q_ _l−_ 2 _Q_ _l−_ 1

expressed as _P_ ≜ _O_ 1 _→_ _O_ 2 _→· · ·_ _−→_ _O_ _l−_ 1 _−→_ _O_ _l_, which

describes a composite relation _Q_ = _Q_ 1 _◦_ _Q_ 2 _◦· · · ◦_ _Q_ _l−_ 1 between node types _Q_ 1 and _Q_ _l−_ 1, where _◦_ denotes the composition
operator over relations. Examples are given in Fig. 1(b).



_Q_ 2
_→· · ·_



_Q_ _l−_ 2
_−→_ _O_ _l−_ 1



expressed as _P_ ≜ _O_ 1



_Q_ 1
_→_ _O_ 2



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:29:01 UTC from IEEE Xplore. Restrictions apply.


1670 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 3, MARCH 2025


Fig. 2. The architecture of MRLHGNN.



III. M ATERIALS AND M ETHODS


In this section, the experimental benchmark dataset, the construction of heterogeneous graph, and the theory of the MRLHGNN model are introduced. The architecture of MRLHGNN

is shown in Fig. 2.


_A. Data Information_


We integrated the Fdataset [25], Cdataset [26] and additional
data downloaded from KEGG [27] and CTD [28] as our drugdisease association data, which contained 894 drugs (r), 454
diseases (d) and 2704 identified associations (d-r, r-d). The matrix _Y_ ( _r_ _i_ _, d_ _j_ ) _∈_ R _[N]_ _[r]_ _[×][N]_ _[d]_ represents drug-disease associations
matrix, with _N_ _r_ being 894, _N_ _d_ being 454. We collected 454
disease- or 894 drug-related biologic relationship data from
DrugBank [29], CTD [28], KEGG [27], STRING [30] and
UniProt [31], obtaining 18878 proteins, 314 pathways, 1048575
protein-protein associations (p-p) [30], 1669 pathway-pathway
associations (a-a) [27], 4397 drug-protein associations (r-p, pr) [29], and 19530 disease-pathway associations (d-a, a-d) [27].

For drug-drug associations (r-r), we calculated binary molecular fingerprints [32] of simplified molecular input line-entry
system (SMILES) that represent the structure of each drug.
Then, we used the Tanimoto method to calculate the drug-drug
pair similarity matrix _S_ _r_ . To simplify the construction of heterogeneous graphs, we converted the drug-drug similarity to a
binarized value. Specifically, in drug similarity matrix _S_ _r_, we
filtered the _topk_ similarity values for each drug to convert to 1
and the others to convert to 0. After the conversion, the _S_ _r_ _[′]_ [is]

procured as the ultimate drug-drug associations.

To calculate the disease-disease associations (d-d), we used
the Medical Subject Headings (MeSH) identifiers to obtain the
disease semantic similarity matrix _S_ _d_ through directed acyclic



TABLE I

T HE D ETAILS OF B ENCHMARK D ATASET


graph (DAG) [33]. Like the drug similarity conversion process,
we also used the _topk_ process to convert the similarities _S_ _d_ to
disease-disease associations _S_ _d_ _[′]_ [. The baseline dataset used in this]

work is shown in Table I. Furthermore, we expanded the relevant
data from the publicly available dataset used in the studies by
Yu et al. [17] and Guo et al. [34]. This creates new Dataset-B
and Dataset-C, which are used to test the sensitivity of model
performance to the associated data. The details of Dataset-B and
Dataset-C are shown in Table II.


_B. Construction of Drug-Disease Heterogeneous Graph_


Thedrugrepositioning,utilizinggraphneuralnetworkscanbe
defined as a link prediction task. It aims to predict the probability
of whether a potential association exists between given drug
( _r_ ) and a disease ( _d_ ) node. Hence, assume a heterogeneous
graph _G_ = ( _V, E_ ) where _V_ = ( _V_ _r_ _, V_ _d_ _, V_ _p_ _, V_ _a_ ) is the node set



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:29:01 UTC from IEEE Xplore. Restrictions apply.


PENG et al.: DRUG REPOSITIONING VIA MULTI-VIEW REPRESENTATION LEARNING WITH HETEROGENEOUS GRAPH NEURAL NETWORK 1671


TABLE II
T HE D ETAILS OF D ATASET -B AND D ATASET -C


Fig. 3. Information aggregation in the loacl structure.



and _E_ = ( _E_ ( _r−r_ ) _, E_ ( _d−d_ ) _, E_ ( _p−p_ ) _, E_ ( _a−a_ ) _, E_ ( _r−d,d−r_ ) _, E_ ( _r−p,p−r_ ) _,_
_E_ ( _d−a,a−d_ ) ) is the edge set. the associations matrix _Y_ of heterogeneous graph _G_ can be defined as:



To effectively process the feature information of drug nodes,
we develop a meta-path based feature-weighted aggregator,
similar to disease node. This approach offers the advantage
of aggregating node features guided by view-specific graph.
Furthermore, incorporating variable-length meta-path instances
into a view-specific graph can enhance the receptive field of local
node feature aggregation. A detailed example is shown in Fig. 3.
And, drug- and diseas-specific view graph of the MRLHGNN
model is depicted in Table III. The aggregation process can be
expressed as:



_S_ _[′]_




_[T]_ 0 _Y_ _[T]_

( _r,d_ ) (



( _[T]_ _a,d_ ) _S_ _d_ _[′]_



⎤

⎥⎥⎦ _._ (1)



_Y_ =



⎡

⎢⎢⎣



_S_ _r_ _[′]_ _Y_ ( _r,p_ ) 0 _Y_ ( _r,d_ )

_Y_ ( _[T]_ _r,p_ ) _Y_ ( _p,p_ ) 0 0

0 0 _Y_ ( _a,a_ ) _Y_ ( _a,d_ )
_Y_ _[T]_ 0 _Y_ _[T]_ _S_ _[′]_



⎫
⎬

⎭ _[,]_



_d_



⎞⎠ : _GP_ _r_ _∈_ Φ _r_



_u_ _i_ =



⎧
⎨



_i_

⎨

⎩ _[z]_ _[GP]_ _[r]_



_i_ =



1

⎛

⎝ _|S_ _[GP]_ _[r]_ _|_



�

_p_ _r_ ( _i,j_ ) _∈S_ _[GP][r]_



_w_ _j_ _x_ _j_



Then, the heterogeneous graph _G_ is constructed using Deep
Graph Library [35]. Additionally, we utilized the calculated similarities _S_ _r_ and _S_ _d_ to represent the chemical structural features
of the drug and the semantic features of the disease. the node
feature matrix _X ∈_ R [(] _[N]_ _[r]_ [+] _[N]_ _[d]_ [)] _[×]_ [(] _[N]_ _[r]_ [+] _[N]_ _[d]_ [)] of drug and disease in
graph _G_ can be initialized and represented as:



_S_ _r_ 0
0 _S_ _d_ �



_X_ =



�



_._ (2)



(3)


where _u_ _i_ _∈_ _U_ _r_ and _U_ _r_ = _{X_ _GP_ _r_ : _GP_ _r_ _∈_ Φ _r_ _}_ is a list of different semantic feature matrices generated for the set Φ _r_ of
all given view-specific _GP_ _r_ of drug. _S_ _[GP]_ _[r]_ is the set of all
metapath instances which correspond to view-specific graph
_GP_ _r_, _|S_ _[GP]_ _[r]_ _|_ denotes the number of meta-path instances in _GP_ _r_,
and _p_ _r_ ( _i, j_ ) is one metapath instance containing target node
_i_ and source node _j_ . _w_ _j_ is the weight of attention given to
feature _x_ _j_ _∈_ _X_ [0 : _N_ _r_ ] _∈_ R _[N]_ _[r]_ _[×]_ [(] _[N]_ _[r]_ [+] _[N]_ _[d]_ [)] of neighbouring nodes
on the meta-path. In executing the same feature aggregation
process, the weighted feature aggregator also generated a list
_U_ _d_ = _{X_ _GP_ _d_ : _GP_ _d_ _∈_ Φ _d_ _}_ which contains different semantic
feature matrices that correspond to different view-specific graph
_GP_ _d_ of disease, respectively. The process of node feature aggregation at the meta-path level in view-specific graph can
be clearly observed in Fig. 2. The step of aggregating node
features on metapaths enumerates all metapath-based neighbors
for each metapath in view-specific graph, which imposes a
high computational expense on the model when the number of
metapath instances grows exponentially with the length of the
metapath. Therefore, inspired by Sun et al. [36], we proposed a
new approach using adjacency matrix multiplication to further
simplify (3). The simplification proceeds as follows:


_X_ _GP_ _r_ = Y [ˆ] ( _r,e_ 1 ) Y [ˆ] ( _e_ 1 _,e_ 2 ) _· · ·_ Y [ˆ] ( _e_ _l−_ 1 _,r_ ) ( _w ⊙_ _X_ _r_ ) _,_ (4)


where _P_ _r_ = _re_ 1 _e_ 2 _· · · e_ _l−_ 1 _r_ is a _l_ -hop metapath, and Y [ˆ] ( _r,e_ 1 ) is
row-normalized form of adjacency matrix Y ( _r,e_ 1 ) between node
type _r_ and _e_ 1 in a view-specific _GP_ _r_ . _w ∈_ R _[N]_ _[r]_ _[×]_ [1] denotes the
feature aggregation weight vector of the drug neighbour nodes



The details of the constitutive heterogeneous graph in MRLHGNN are described in Supplementary Material.


_C. Aggregation of Metapath-Level Features in_
_View-Specific Graph_


Tocapturetheheterogeneityinformationbetweennodesinthe
local structure, we treat each semantic (represented by the metapath) _P_ _v_ as a view and perform view-specific metapath-level
message aggregation to obtain the node feature embedding for
each view. Specifically, according to the semantic information of
heterogeneity between nodes, the original heterogeneous graph
is decomposed into multiple view-specific graph [23]. This
operation has the advantage of decoupling multiple semantics.
Moreover, compared to neighborhoods and meta-path instances
based on meta-paths, the view-specific graph formed by drug or
disease nodes through intermediary neighboring nodes retains
a more comprehensive view-centric local structure under each
meta-path in the heterogeneous semantics. As a consequence,
under semantic _P_ _v_, the view-specific graph _GP_ _v_ preserves the
original and complete multi-hop structure in the heterogeneous
graph. An example is shown in Fig. 1(c).



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:29:01 UTC from IEEE Xplore. Restrictions apply.


1672 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 3, MARCH 2025


TABLE III
D ESCRIPTION OF V IEW -S PECIFIC G RAPH IN MRLHGNN



on the metapath _P_ _r_ . _⊙_ denotes the element-wise product of
vectors and _X_ _r_ = _X_ [0 : _N_ _r_ ] _∈_ R _[N]_ _[r]_ _[×]_ [(] _[N]_ _[r]_ [+] _[N]_ _[d]_ [)] is the raw feature
matrix of of all drug nodes. With the same simplified feature
aggregation method, we can obtained disease node feature embeddings matrix _X_ _GP_ _d_ at the metapath level within the disease
view-specific graph.

Feature embedding vectors with varying dimensions can pose
challenges when aggregating view-level information in subsequent modules. To solve this problem, we employ view-specific
featuretransformationstoconvertheterogeneousfeatureembedding vectors into features with consistent dimensionality. Taking
the example of the conversion of features for a specific-view of
drug, the specific process of conversion is as follows:


_H_ _GP_ _[′]_ _r_ [=] _[ MLP]_ _[GP]_ _r_ [(] _[X]_ _[GP]_ _r_ [)] _[,]_ (5)



_v_ _GP_ _ir_ = _W_ _V_ _h_ _[′]_ _GP_ _r_ _[i]_ _[,]_ (8)



_r_



�



exp



_q_ _GP_ _ir_ _∗_ _k_ _[T]_



_GP_ _r_ _[j]_



_α_ ( _GP_ _ir_ _,GP_ _rj_ ) [=]



~~�~~ _GP_ _r_ _[t]_ _∈_ Φ _r_ [exp]



�



~~�~~ _,_ (9)



~~�~~



_q_ _GP_ _ir_ _∗_ _k_ _GP_ _[T]_ _r_ _[t]_



_q_ _GP_ _ir_ _∗_ _k_ _[T]_



_α_ ( _GP_ _ir_ _,GP_ _rj_



_j_

_r_ ) _[v]_ _[GP]_ _r_ _[j]_



_h_ _GP_ _ir_ = _ϕ_



�



_r_ _[j]_ [+] _[ h]_ _[′]_



_GP_ _r_ _[i]_ _[,]_ (10)



_GP_ _r_ _[j]_ _[∈]_ [Φ] _r_



_d_ [2] _[, . . .,][ GP]_ _d_ _[M]_



_d_ [1] _[,][ GP]_ _d_ [2]



where _MLP_ _GP_ _r_ () is a multi-layer perception block for specificview _GP_ _r_ of drug, including a normalization layer, a nonlinear
layer, and a dropout layer between two consecutive linear layers.
Similar to the feature conversion process of drug view-specific,
we also obtained feature transformation matrix _H_ _[′]_

_GP_ _d_ [of disease]
for each view-specific _GP_ _d_ .


_D. Transformer-Based Semantic Feature Fusion of_
_View-Specific_


Inthissection,ouraimistofusedifferentrepresentationsfrom
multiple view-specific graph. To produce embedding vectors
with efficient representation learning capacity, we proposed
a transformer-based [37] semantic fusion module for further
exploring the cross-relationships between feature vectors after
meta-path aggregation in different view-specific. Specifically,
with the different view list Φ _r_ = _{GP_ _r_ [1] _[,][ GP]_ _r_ [2] _[, . . .,][ GP]_ _r_ _[M]_ _[}]_ [ of]

for each drug node,drug and converted semantic vectors _M_ represents the number of view-specific _{h_ _[′]_ _GP_ _r_ [1] _[, h]_ _GP_ _[′]_ _r_ [2] _[, . . ., h]_ _GP_ _[′]_ _r_ _[M]_ _[}]_
graph for both drug and disease nodes. The transformer-based
semantic feature fusion module for view-specific of drug learns
the mutual attention for each pair of semantic vectors. For each
semantic vector _h_ _[′]_

_GP_ _r_ _[i]_ [, it maps the vector into a query vector] _[ q]_ _[GP]_ _r_ _[i]_ [,]
a key vector _k_ _GP_ _ir_, and a value vector _v_ _GP_ _ir_ . The mutual attention
weight _α_ ( _GP_ _ir_ _,GP_ _rj_ [)] [ is the dot product result of the query vector]



where _W_ _Q_, _W_ _K_, _W_ _V_ and _ϕ_ are trainable parameters shared
by all view-specific of drug. Then, we obtained a list _H_ _r_ =
_{H_ _GP_ _r_ 1 _, H_ _GP_ _r_ 2 _, . . ., H_ _GP_ _rM_ _[}]_ [ of feature fusion matrices for each]
view-specific graph of drug. For the different view list Φ _d_ =
_{GP_ _d_ [1] _[,][ GP]_ _d_ [2] _[, . . .,][ GP]_ _d_ _[M]_ _[}]_ [, with the same transformer-based se-]

mantic feature fusion operation, we also obtained a list _H_ _d_ =
_{H_ _GP_ 1 _[, H]_ _[GP]_ [2] _[, . . ., H]_ _[GP]_ _[M]_ _[}]_ [ of feature matrices, which contain]



_{H_ _GP_ _d_ 1 _[, H]_ _[GP]_ _d_ [2] _[, . . ., H]_ _[GP]_ _d_ _[M]_ _[}]_ [ of feature matrices, which contain]

matrices corresponding to different view-specific graph _GP_ _d_ in
the set Φ _d_, respectively.



1

_d_ _[, H]_ _[GP]_ _d_ [2]



_d_ [2] _[, . . ., H]_ _[GP]_ _d_ _[M]_



_r_ [2] _[, . . .,][ GP]_ _r_ _[M]_

_[′]_ _GP_ _r_ [2] _[, . . ., h]_ _[′]_



_E. Auto Multi-View Fusion Decoder for Predicting_
_Drug-Disease Associations_


Inspired by yang et al. [38], to decode the semantic embedding
vectors of drug and disease fused on different views to achieve
thepredictionmission, this sectionusedanautomaticmulti-view
fusion decoder to predict drug-disease associations. According
to Table V, we performed one-to-one semantic decoding view
of the semantic matrices of view-specific in the set _H_ _r_ and _H_ _d_,
respectively,andthenintegratedthedecodedpreliminaryprediction score matrices using the attention mechanism to obtain the
final drug-disease matrix. Furthermore, we also perform similar
decoding of the initial drug node features _H_ 0 _[r]_ [=] _[ X]_ [[0 :] _[ N]_ _[r]_ []][ and]

the initial disease node features _H_ 0 _[d]_ [=] _[ X]_ [[] _[N]_ _[r]_ [: (] _[N]_ _[r]_ [+] _[ N]_ _[d]_ [)]][ in]

the heterogeneous graph to obtain the initial view of the automatic decoder. The above process can be formulated as:



_r_ [1] _[,][ GP]_ _r_ [2]

_[′]_

_GP_ _r_ [1] _[, h]_ _[′]_



1

_d_ [)] _[Y]_ _[ ′]_ (



� _T_ [�]



+ _β_ ( _GP_ _r_ 1 _,GP_ _d_ 1



_Y_ ( _[′]_ _r,d_ ) [=] _[ η]_



�



_d_ [1] [) +] _[ · · ·]_



_H_ 0 _[r]_ _[W]_ [0]



_H_ _[r]_



�



_H_ 0 _[d]_



_H_ _[d]_



( _[GP]_ _r_ [1] _[,][GP]_ _d_ [1]



_d_ _[M]_ [)] _[Y]_ _[ ′]_ (



_d_ _[M]_ [)]



+ _β_ ( _GP_ _rM_ _[,][GP]_ _d_ _[M]_



( _[GP]_ _r_ _[M]_ _[,][GP]_ _d_ _[M]_



weight _α_ ( _GP_ _ir_ _,GP_ _rj_ [)] [ is the dot product result of the query vector]

_q_ _GP_ _ir_ and the key vector _k_ _GP_ _rj_ [after a softmax normalization. The]



_q_ _GP_ _ir_ and the key vector _k_ _GP_ _rj_ [after a softmax normalization. The]

output vector _h_ _GP_ _ir_ of current semantic _GP_ _r_ _[i]_ [is the weighted sum]



� _T_ [�]



�



_H_ _[d]_



0



_M_
�


_n_ =1



_β_ ( _GP_ _rn_ _[,][GP]_ _d_ _[n]_ [)]



_β_ ( _GP_ _rn_ _[,][GP]_ _d_ _[n]_



�



�



_H_ 0 _[r]_ _[W]_ [0]



_H_ _[r]_



_H_ _GP_ _r_ _[n]_ _[W]_ _[n]_



�



_H_ _GP_ _[n]_



+



output vector _h_ _GP_ _ir_ of current semantic _GP_ _r_ [is the weighted sum]

of all value vectors _v_ _[GP]_ _r_ _[j]_ plus a residual connection. The process

of semantic fusion can is formulated as:



= _η_


_×_



_d_



� _T_ [�]



_,_ (11)



_q_ _GP_ _ir_ = _W_ _Q_ _h_ _[′]_ _GP_ _r_ _[i]_ _[,]_ (6)



where, _β_ ( _GP_ _rn_ _[,][GP]_ _d_ _[n]_



_k_ _GP_ _ir_ = _W_ _K_ _h_ _[′]_ _GP_ _r_ _[i]_ _[,]_ (7)



where, _β_ ( _GP_ _rn_ _[,][GP]_ _d_ _[n]_ [)] [ is the attention coefficient for each prelimi-]

nary score matrix _Y_ _[′]_ _[n]_ _[n]_ [ and] _[ W]_ _[n]_ [ is the parameter matrix]



_d_ _[n]_ [)] [ and] _[ W]_ _[n]_ [ is the parameter matrix]



( _GP_ _r_ _[n]_ _[,][GP]_ _d_ _[n]_



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:29:01 UTC from IEEE Xplore. Restrictions apply.


PENG et al.: DRUG REPOSITIONING VIA MULTI-VIEW REPRESENTATION LEARNING WITH HETEROGENEOUS GRAPH NEURAL NETWORK 1673



within each decoder. _Y_ ( _[′]_ _r,d_ ) [is the final drug-disease prediction]

matrix.


_F. Optimization_


We use a weighted cross-entropy loss function to balance
the effect of an unbalanced dataset in order to optimize the
parameters of the prediction model and ensure that MRLHGNN
focus on confirmed drug-disease associations. For _N_ _r_ drugs
and _N_ _d_ diseases in the heterogeneous network _G_, with confirmed/unconfirmed drug-disease associations labeled _S_ [+] and
_S_ _[−]_, respectively, the loss function of the MRLHGNN model
can be expressed as:


_Loss_ =



�

( _r,d_ ) _∈S_ [+]



log _Y_ ( _[′]_ _r,d_ ) [+]



1
_−_ _N_ _r_ + _N_ _d_



⎛

⎝ _γ_



⎛



( _r,d_ ) _∈S_ _[−]_



�



�



1 _−_ log _Y_ _[′]_



( _r,d_ )



�



_B. Baseline Methods_


To evaluate the model performances of our proposed MRLHGNN, we compared it to nine baseline methods: (i) traditional
machine learning methods including: NTSIM [13], BNNR [14],
and HGIMC [15]; (ii) deep learning methods including: NIMCGCN [39], LAGCN [17], DRHGCN [18], DRWBNCF [19],
REDDA [20] and MilGNet [21].


_C. Comparison With Other Methods_


Similar to [40] and [41], we employed five-fold crossvalidation (5-CV) to evaluate the predictive performance of the
methods. Specifically, we divided all positive samples (validated) and all negative samples (unvalidated) into five equal
parts, respectively. Four equal parts of positive and negative samples are treated as the training set, and one equal part of positive
and negative samples are considered as the test set. In particular,
the final predictive evaluation results of the model are obtained
by averaging the performance metrics value from 10 times
5-CV. The performance comparison are reported in Table IV,
where MRLHGNN demonstrates a competitive advantage, apart
from the _Specificity_ and _Precision_ metrics. The results in
Table IV indicated that our proposed MRLHGNN outperforms
the other nine baseline methods in terms of metrics including
_AUC_, _AUPR_, _F_ 1_ _score_, and _Recall_ . It shows relative improvements of 4.51%, 20.87%, 22.9%, and 32.5% respectively
compared to suboptimal methods. For _Accuracy_, _Specificit_ y
and _Precision_ metrics, our method’s performance remains
comparable. Among the compared deep learning methods, there
is a 13.47% improvement in _Precision_ metric compared to
the suboptimal method (DRHGCN). Benchmarking comparison
resultsshowthataddinginformationonproteins,sideeffectsinto
known drug-disease associations data while using a multi-view
based mechanism can significantly improve the comprehensive
prediction performance.

We also tested our model on two another datasets (DatasetB and Dataset-C) to demonstrate the reliability of MRLHGNN on public datasets. The performance results of MRLHGNN and nine baseline methods are presented in Tables VI
and VII, respectively. In Dataset-B, the MRLHGNN achieved
the best performance among the compared baseline methods for all evaluation metrics except the _AUPR_ (secondbest performance advantage). Specifically, compared to the
suboptimal methods, the MRLHGNN demonstrated improvements of 2.04% in _AUC_, 0.13% in _F_ 1_ _score_, 0.21% in
_Accuracy_, 4.42% in _Recall_, 6.05% in _Specificity_, and 1.18%
in _Precision_ . In Dataset-C, the MRLHGNN achieved comparatively favorable performance. Apart from obtaining the thirdbest performance advantage in terms of _AUPR_ and _F_ 1_ _score_,
itoutperformedthebaselinemethodsintheremainingevaluation
metrics,achievingeitherthebestorsecond-bestperformanceadvantage. In these two additional datasets, statistical comparisons
of evaluation metrics between the methods further highlight the
relatively reliable predictive performance of the MRLHGNN.

To evaluate MRLHGNN’s sensitivity to known drugdisease associations, we randomly removed a portion of



⎞



⎠ _,_



(12)


where, _γ_ = _|_ _[|]_ _S_ _[S]_ [+] _[−]_ _[|]_ _|_ [is the balance weight,] _[ |][S]_ [+] _[|]_ [ and] _[ |][S]_ _[−]_ _[|]_ [ are the]

number of confirmed/unconfirmed drug-disease associations in
the training set.


IV. E XPERIMENTS AND R ESULTS


_A. Evaluation Metrics and Parameters Setting_


To evaluate the overall performance of MRLHGNN, we
use evaluation metrics such as Area Under the Receiver

Operating Characteristic Curve ( _AUC_ ), the Area Under
the Precision-Recall curve ( _AUPR_ ), _F_ 1_ _score_, _Accuracy_,
_Recall_, _specificity_, and _Precision_ . Among these metrics,
_AUC_, _AUPR_ and _F_ 1_ _score_ are selected as core metrics to
further describe the performance benefits of MRLHGNN. Our
proposed MRLHGNN model uses the Adam optimizer to optimize the neural network parameters.

In the MRLHGNN model, the values of all hyperparameters
are referred to the practice of previous researchers and finally
determined by grid search, where the learning rate _r_ is 0.005, the
node feature dimension size _k_ in the network layer is 128, the
dropout rate _dr_ is 0.4 and the setting of similarity threshold
_topk_ affecting the number of associations of disease-disease
and drug-drug is set to 15. The details of the hyperparameters
in MRLHGNN are described in Supplementary Material. In
addition, all methods are compared under the same evaluation
settings, which include the datasets we used and the similarity
calculations for drug and disease. For the baseline model with
publicly available code, we refer to the best hyper-parameters
reported in its original paper to run the code.

All experiments are conducted on a Windows Pro PC with
a GeForce RTX 3090 GPU, 32 GB of RAM (Random Access
Memory) and Intel(R)Core(TM)i7-13700K CPU @ 5.40 GHz.
All algorithms are implemented in PyTorch and compiled using
[Python 3.8.1. We have released the code on https://github.com/](https://github.com/biohnuster/MRLHGNN)
[biohnuster/MRLHGNN.](https://github.com/biohnuster/MRLHGNN)



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:29:01 UTC from IEEE Xplore. Restrictions apply.


1674 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 3, MARCH 2025


TABLE IV
P ERFORMANCE OF T EN M ETHODS IN T ERMS OF _AUC_, _AUPR_, _F_ 1_ _score_, _Accuracy_, _Recall_, _Specificity_, AND _Precision_ U NDER 5-CV ON

B ENCHMARK D ATASET


TABLE V

T HE D ETAILS OF A UTO M ULTI -V IEW F USION D ECODER


Fig. 4. The distribution of attention coefficient in the MRLHGNN model
for decoding from each view in the automatic decoder.



known drug-disease associations in our benchmark dataset by
_{_ 0% _,_ 10% _,_ 20% _,_ 30% _,_ 40% _,_ 50% _}_ . This ensures model stability
in sparse networks without relying on unknown associations or
external data. The 5-CV results, including _AUC_, _AUPR_, and
_F_ 1_ _score_, are presented in Table VIII. The Table VIII reveals
that MRLHGNN’s predictive performance remains consistent
even with random removal of known drug-disease associations ( _AUC >_ 0 _._ 94, _AUPR >_ 0 _._ 59, _F_ 1_ _Score >_ 0 _._ 66). This
highlights the robustness of MRLHGNN against variations in
drug-disease associations, underscoring its potential for drug
repositioning within sparse biological networks.


_D. Attention Analysis of Views in Automatic Fusion_
_Decoder_


To explore the degree of attention for decoding features
aggregated from each view-specific graph, we observed the
distribution of attention coefficients in the automatic multi-view
fusion decoder when training is stable. As shown in Fig. 4,
among the sub-views that compose the automatic multi-view
fusion decoder, view1 (4 with an attention coefficient greater
than 1/9) and view7 (3 with an attention coefficient greater than
1/9) receive the most attention under each 5-CV compared to
the other sub-views. More specifically, views _GP_ _d_ [1] [and] _[ GP]_ _d_ [7] [of]

diseases and views _GP_ [1] [and] _[ GP]_ [7] [of drugs contributed more]



to the MRLHGNN, and in particular, view-specific graph _GP_ _d_ [1]

consisting of the diseases’ meta-path _D →_ _D_ and view-specific
graph _GP_ _r_ [1] [consisting of the drugs’ meta-path] _[ R][ →]_ _[R]_ [ produced]

the greatest impact.


_E. Ablation Study_


To assess the significance and validity of view-specific graphs
and sub-module design in MRLHGNN, we conducted relevant
ablation experiments by proposing and evaluating four model
variants.

r MRLHGNN without view-specific graphs of drugs and



diseases (w/o _GP_ _[i]_



_r_ _[i]_ [and] _[ GP]_ _d_ _[i]_



diseases (w/o _GP_ _r_ [and] _[ GP]_ _d_ [,] _[ i][ ∈]_ _[M]_ [) : To investigate the]

effects of view-specific design on the predictive performance of the model, we removed the construction of
view-specific graphs for drugs and diseases one by one.
r MRLHGNN without multi-layer feature projection (w/o



_d_ [1] [and] _[ GP]_ _d_ [7]



_r_ [1] [and] _[ GP]_ _r_ [7]



_r_ [of drugs contributed more]



MLP): Since the multilayer feature projection module is
what maintains the feature dimensions of the drug and
disease consistent, we designed this ablation experiment
using a single-layer feedforward neural network with an
output feature dimension of 128 for replacement.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:29:01 UTC from IEEE Xplore. Restrictions apply.


PENG et al.: DRUG REPOSITIONING VIA MULTI-VIEW REPRESENTATION LEARNING WITH HETEROGENEOUS GRAPH NEURAL NETWORK 1675


TABLE VI
P ERFORMANCE OF T EN M ETHODS IN T ERMS OF _AUC_, _AUPR_, _F_ 1_ _score_, _Accuracy_, _Recall_, _Specificity_, AND _Precision_ U NDER 5-CV ON

D ATASET -B


TABLE VII
P ERFORMANCE OF T EN M ETHODS IN T ERMS OF _AUC_, _AUPR_, _F_ 1_ _score_, _Accuracy_, _Recall_, _Specificity_, AND _Precision_ U NDER 5-CV ON

D ATASET -C


TABLE VIII

T HE I MPACT OF R EMOVING K NOWN D RUG -D ISEASE A SSOCIATIONS ON THE

P REDICTIVE P ERFORMANCE OF MRLHGNN



r MRLHGNN without Transformer (w/o Transformer): In

the MRLHGNN model, the view-pecific level node features are not aggregated using the transformer-based submodule.

r MRLHGNN without attention of auto muti-view fusion


decoder (w/o AD_attention): We removed the view feature
decoding attention mechanism from the auto multi-view
fusion decoder.

As shown in Fig. 5, each sub-module contributes to the final
performance. The Transformer-based view-specific semantic
feature fusion module is the most important one. When the MLP
module is removed, the model is only affected secondarily by
the Transformer-based sub-module. In contrast, removing the
attentionmechanisminthedecoderconverselyachievedtheleast



Fig. 5. Results of each ablation method on _AUC_, _AUPR_ and
_F_ 1_ _score_ metrics in 5-CV.


impact on the model. Surprisingly, the ablation experimental
design showed the flattest effect on _AUC_ metrics, but is more
significant on _AUPR_ and _F_ 1_ _score_ metrics. According to this
ablation study, we can draw the conclusion that a successful drug
repositioning prediction model should consider not only feature
projection mapping relationships, but also effective feature aggregation mechanisms.

As depicted in Table IX, the MRLHGNN exhibits more pronounced enhancements in _AUPR_ and _F_ 1_ _score_ metrics as the
number of view-specific graphs for drugs and diseases increases.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:29:01 UTC from IEEE Xplore. Restrictions apply.


1676 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 3, MARCH 2025


TABLE IX
P ERFORMANCE OF A BLATION M ETHODS IN T ERMS OF _AUC_, _AUPR_, _F_ 1_ _score_, _Accuracy_, _Recall_, _Specificity_, AND _Precision_ U NDER 5-CV ON

B ENCHMARK D ATASET


Fig. 6. MDA visualization of training disease features in MRLHGNN.



Thisclearlydemonstratesthatcombiningmoremeta-pathneighbor information into a heterogeneous graph can improve the predictive performance of the model. Furthermore, view-specific
graphs consisting of first- and second-hop meta-path neighbor
information for drugs and diseases exert a greater influence on
the performance of the model. Specifically, without considering
the information of second-hop meta-path neighbors and only
constructing view-specific graphs from the information of firsthop or higher meta-path neighbors, the _AUPR_ and _F_ 1_ _score_
metrics will reduce by 21.55% and 18.96%, respectively.


_F. Visualization_


To further demonstrate the efficacy of MRLHGNN, we conducted a series of visualization experiments on view-specific



graph layers and transformer feature fusion layers. As shown in
Fig. 6, we employed MDA [42] to project the node embeddings
onto a two-dimensional feature space of diseases or drugs.
Taking diseases as an example, we extracted feature embeddings
from different network layers within the disease-specific view
of the MRLHGNN model. The visualization experiments reveal
that as the number training epochs increases, the color patterns in
the feature visualization maps of MRLHGNN exhibit a tendency
to cluster and gradual change, while maintaining a continuous
and uniform shape. Specifically, in View2’s feature visualization, the internal manifold structure of the feature embeddings
becomes increasingly distinct with more training epochs, indicating stronger feature learning capability of the model. The
comprehensibility arises from the fact that deeper layers of the



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:29:01 UTC from IEEE Xplore. Restrictions apply.


PENG et al.: DRUG REPOSITIONING VIA MULTI-VIEW REPRESENTATION LEARNING WITH HETEROGENEOUS GRAPH NEURAL NETWORK 1677


Fig. 7. Visualizing feature learning of the transformer mechanism in disease view-specific graph using the MDA.



TABLE X
T OP 15 C ANDIDATE D RUGS R ELATED TO N ON -S MALL C ELL L UNG C ANCER

P REDICTED BY MRLHGNN


view extract higher-level features in order to better accomplish
the prediction task.

Additionally, we analyzed the feature learning of the disease
view-specific graph using transformer fusion in the model, again
using MDA for visualization. As shown in Fig. 7, we observed
that with more training epochs, the colors and shapes in the
MDA visualization become increasingly regular. This indicates
that the network features, after being fused by the transformer
mechanism, are orderly distributed in the manifold space. The
continuity of colors and shapes in the MDA visualizations suggests that the transformer mechanism maintains the geometric
relationships of the feature space well.


V. C ASE S TUDIES

To evaluate the actual predictive capability of MRLHGNN,
we conducted case studies on non-small cell lung cancer
(NSCLC, Mesh ID: D002289) with high morbidity and piroxicam drug (DrugBank ID: DB00554). Before conducting the
case study, we considered all known drug-disease associations
in the dataset as the training set and the unknown drug-disease
associations as the candidate set. Once the model is trained and

stable, MRLHGNN obtains the predicted probability of diseases
interacting with all drug candidates or drugs interacting with all



TABLE XI
T OP 15 C ANDIDATE D ISEASES R ELATED TO P IROXICAM P REDICTED BY

MRLHGNN


diseases candidates. We then rank the drugs’/diseases’ candidates based on their predicted probability, where higher-ranked
drugs/disease are most likely to treat the diseases or find new
indications for drugs.

The results of the top 15 drug candidates predicted by MRLHGNN to have a potential association with NSCLC are shown
in Table X. We choose three drugs in Table X to describe
them in detail. Azathioprine is a thiopurine, a prodrug that is
converted to 6-TG. An outcome of its complex metabolism is the
incorporatedof6-TGintoDNAbythereplicationprocess.Based
on the results of a related experiment, Lazarev et al. [43] are
enabled to treat patients with NSCLC suffering from ulcerative
colitis with azathioprine. Low-dose naltrexone (LDN) can be
beneficial as an adjuvant for patients with NSCLC. Miskoff
et al. [44] proposed a unique mechanism that allows LDN to
enhance the degree of the immune system’s ability to affect
lung cancer tumors at the cellular level, resulting in longer
survival cycles for patients. Trifluoperazine is primarily used to
treat schizophrenia. Jeong et al. [45] discovered that a synthetic
analogue of Trifluoperazine exhibited a strong activity against
lung cancer in clinical treatment.

The top15 disease candidates for piroxicam are shown in
Table XI. We also selected three disease candidates for detailed



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:29:01 UTC from IEEE Xplore. Restrictions apply.


1678 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 3, MARCH 2025



description. Piroxicam is a non-steroidal anti-inflammatory
drug, Thabet et al. [46] determined that piroxicam inhibits
inflammation-driven breast neoplasms from RAS and PAR-4
signaling. Besides, Mazzilli et al. [47] found that topical 0.8%
piroxicam and 50+ sunscreen filters reduced the location of
lesions in actinic keratoses for hypertensive patients. In addition,
through actual cases, Trujillo et al. [48] found that patients with
dermatitis taking piroxicam should avoid using it with other
medications.


Overall, 11 of the top 15 drugs’/diseases’ candidates for
prediction have been proven in the literature, which can further illustrate the reliable performance of MRLHGNN for drug
repositioning.


VI. C ONCLUSION


Drug repositioning can effectively improve the efficiency of
drug development and disease treatment. It is interdependent
with drug combination therapy in modern medicine and provides a new way for disease treatment [49], [50]. In this paper,
we propose a drug repositioning method (MRLHGNN) with
a transformer-based view-specific graph-level feature aggregation mechanism and a multi-view automatic fusion decoder.

In heterogeneous bioinformatics networks, we first construct
view-specific graphs based on meta-paths. The definition of
meta-path types is based on a pre-exploration of the structure and
domain expertise of bio-heterogeneous networks, making our
model interpretable. Then, we obtain the node-structure feature
vector representation at meta-path level and the low-dimensional
semantic vector representation of graph-level features for predicting drug-disease associations using view-specific graphweighted aggregation mechanism and transformer mechanism.
The transformer mechanism captures the interdependencies between view-specific graphs. To some extent, the multi-view
automatic fusion decoder reflects the impact of encoder views
on the prediction performance of drug-disease associations.
To validate the effectiveness of MRLHGNN, we compared it
with 9 state-of-the-art prediction methods for drug repositioning
on a benchmark dataset. The results indicate that MRLHGNN

exhibits significant performance enhancement and competitive.
Moreover, case studies have illustrated that MRLHGNN can be
accepted as a reliable tool for drug repositioning.

Although MRLHGNN achieves SOTA performance in the
comparison method, there are still some problems that deserve
deeper study, such as the imbalance between the number of positive and negative samples, and the selection strategy of reliable
negative samples. In the future, we will continue to address
these shortcomings through a comprehensive exploration of
supervised contrast learning [51] and hypergraph learning [52].


R EFERENCES


[1] H. S. Chan, H. Shan, T. Dahoun, H. Vogel, and S. Yuan, “Advancing drug

discovery via artificial intelligence,” _Trends Pharmacological Sci._, vol. 40,
no. 8, pp. 592–604, 2019.

[2] H. Liu, Y. Zhao, L. Zhang, and X. Chen, “Anti-cancer drug response

prediction using neighbor-based collaborative filtering with global effect
removal,” _Mol. Ther.-Nucleic Acids_, vol. 13, pp. 303–311, 2018.




[3] N.-N. Guan, Y. Zhao, C.-C. Wang, J.-Q. Li, X. Chen, and X. Piao, “Anti
cancer drug response prediction in cell lines using weighted graph regularized matrix factorization,” _Mol. Ther.-Nucleic Acids_, vol. 17, pp. 164–174,
2019.

[4] T.-H. Li, C.-C. Wang, L. Zhang, and X. Chen, “SNRMPACDC: Computa
tional model focused on siamese network and random matrix projection for
anticancer synergistic drug combination prediction,” _Brief. Bioinf._, vol. 24,
no. 1, 2023, Art. no. bbac503.

[5] X. Chen and L. Huang, “Computational model for disease research,” _Brief._

_Bioinf._, vol. 24, no. 1, 2023, Art. no. bbac615.

[6] X. Chen, C. Zhou, C.-C. Wang, and Y. Zhao, “Predicting potential small

molecule–miRNA associations based on bounded nuclear norm regularization,” _Brief. Bioinf._, vol. 22, no. 6, 2021, Art. no. bbab328.

[7] L. Peng, Y. Tu, L. Huang, Y. Li, X. Fu, and X. Chen, “DAESTB: Infer
ring associations of small molecule–miRNA via a scalable tree boosting
model based on deep autoencoder,” _Brief. Bioinf._, vol. 23, no. 6, 2022,
Art. no. bbac478.

[8] H. Wang, F. Huang, Z. Xiong, and W. Zhang, “A heterogeneous network
based method with attentive meta-path extraction for predicting drug–
target interactions,” _Brief. Bioinf._, vol. 23, no. 4, 2022, Art. no. bbac184.

[9] L. Zhang, C.-C. Wang, and X. Chen, “Predicting drug–target binding affin
ity through molecule representation block based on multi-head attention
and skip connection,” _Brief. Bioinf._, vol. 23, no. 6, 2022, Art. no. bbac468.

[10] W. Lan et al., “IGNSCDA: Predicting CircRNA-disease associations

based on improved graph convolutional network and negative sampling,”
_IEEE/ACM Trans. Comput. Biol. Bioinf._, vol. 19, no. 6, pp. 3530–3538,
Nov./Dec. 2021.

[11] L. Wang, L. Wong, Z.-H. You, D.-S. Huang, X.-R. Su, and B.-W. Zhao,

“NSECDA: Natural semantic enhancement for circRNA-disease association prediction,” _IEEE J. Biomed. Health Inform._, vol. 26, no. 10,
pp. 5075–5084, Oct. 2022.

[12] W. Liu, T. Tang, X. Lu, X. Fu, Y. Yang, and L. Peng, “MPCLCDA:

Predicting circRNA–disease associations by using automatically selected meta-path and contrastive learning,” _Brief. Bioinf._, vol. 24, 2023,
Art. no. bbad227.

[13] W. Zhang et al., “Predicting drug-disease associations based on the known

association bipartite network,” in _2017 IEEE Int. Conf. Bioinf. Biomed._,
2017, pp. 503–509.

[14] M. Yang, H. Luo, Y. Li, and J. Wang, “Drug repositioning based on

bounded nuclear norm regularization,” _Bioinformatics_, vol. 35, no. 14,
pp. i455–i463, 2019.

[15] M. Yang, L. Huang, Y. Xu, C. Lu, and J. Wang, “Heterogeneous graph

inference with matrix completion for computational drug repositioning,”
_Bioinformatics_, vol. 36, no. 22/23, pp. 5456–5464, 2020.

[16] X. Ji, J. M. Freudenberg, and P. Agarwal, “Integrating biological

networks for drug target prediction and prioritization,” in _Computa-_
_tional Methods for Drug Repurposing._ Berlin, Germany: Springer, 2019,
pp. 203–218.

[17] Z. Yu, F. Huang, X. Zhao, W. Xiao, and W. Zhang, “Predicting drug–

disease associations through layer attention graph convolutional network,”
_Brief. Bioinf._, vol. 22, no. 4, 2021, Art. no. bbaa243.

[18] L. Cai et al., “Drug repositioning based on the heterogeneous information

fusion graph convolutional network,” _Brief. Bioinf._, vol. 22, no. 6, 2021,
Art. no. bbab319.

[19] Y. Meng, C. Lu, M. Jin, J. Xu, X. Zeng, and J. Yang, “A weighted

bilinear neural collaborative filtering approach for drug repositioning,”
_Brief. Bioinf._, vol. 23, no. 2, 2022, Art. no. bbab581.

[20] Y. Gu, S. Zheng, Q. Yin, R. Jiang, and J. Li, “REDDA: Integrating

multiple biological relations to heterogeneous graph neural network for
drug-disease association prediction,” _Comput. Biol. Med._, vol. 150, 2022,
Art. no. 106127.

[21] Y. Gu, S. Zheng, B. Zhang, H. Kang, and J. Li, “Milgnet: A multi-instance

learning-based heterogeneous graph network for drug repositioning,” in
_2022 IEEE Int. Conf. Bioinf. Biomed._, 2022, pp. 430–437.

[22] Y. Li, M. Yang, and Z. Zhang, “A survey of multi-view representation

learning,” _IEEE Trans. Knowl. Data Eng._, vol. 31, no. 10, pp. 1863–1883,
Oct. 2019.

[23] Z. Shao, Y. Xu, W. Wei, F. Wang, Z. Zhang, and F. Zhu, “Heteroge
neous graph neural network with multi-view representation learning,”
_IEEE Trans. Knowl. Data Eng._, vol. 35, no. 11, pp. 11476–11488,
Nov. 2023.

[24] H. Fu, F. Huang, X. Liu, Y. Qiu, and W. Zhang, “MVGCN: Data inte
gration through multi-view graph convolutional network for predicting
links in biomedical bipartite networks,” _Bioinformatics_, vol. 38, no. 2,
pp. 426–434, 2022.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:29:01 UTC from IEEE Xplore. Restrictions apply.


PENG et al.: DRUG REPOSITIONING VIA MULTI-VIEW REPRESENTATION LEARNING WITH HETEROGENEOUS GRAPH NEURAL NETWORK 1679




[25] A. Gottlieb, G. Y. Stein, E. Ruppin, and R. Sharan, “PREDICT: A method

for inferring novel drug indications with application to personalized
medicine,” _Mol. Syst. Biol._, vol. 7, no. 1, 2011, Art. no. 496.

[26] H. Luo et al., “Drug repositioning based on comprehensive similarity

measures and bi-random walk algorithm,” _Bioinformatics_, vol. 32, no. 17,
pp. 2664–2671, 2016.

[27] M. Kanehisa and S. Goto, “KEGG: Kyoto encyclopedia of genes and

genomes,” _Nucleic Acids Res._, vol. 28, no. 1, pp. 27–30, 2000.

[28] A. P. Davis et al., “Comparative toxicogenomics database (CTD): Update

2021,” _Nucleic Acids Res._, vol. 49, no. D1, pp. D1138–D1143, 2021.

[29] D. S. Wishart et al., “DrugBank: A knowledgebase for drugs, drug actions

anddrugtargets,” _NucleicAcidsRes._,vol.36,no.suppl_1,pp.D901–D906,
2008.

[30] C. v. Mering, M. Huynen, D. Jaeggi, S. Schmidt, P. Bork, and B. Snel,

“STRING: A database of predicted functional associations between proteins,” _Nucleic Acids Res._, vol. 31, no. 1, pp. 258–261, 2003.

[31] U. Consortium et al., “UniProt: The universal protein knowledgebase,”

_Nucleic Acids Res._, vol. 46, no. 5, pp. D115–119, 2018.

[32] D. Rogers and M. Hahn, “Extended-connectivity fingerprints,” _J. Chem._

_Inf. Model._, vol. 50, no. 5, pp. 742–754, 2010.

[33] J. Z. Wang, Z. Du, R. Payattakool, P. S. Yu, and C.-F. Chen, “A new method

to measure the semantic similarity of go terms,” _Bioinformatics_, vol. 23,
no. 10, pp. 1274–1281, 2007.

[34] Z.-H. Guo, Z.-H. You, D.-S. Huang, H.-C. Yi, Z.-H. Chen, and Y.-B.

Wang, “A learning based framework for diverse biomolecule relationship
prediction in molecular association network,” _Commun. Biol._, vol. 3, no. 1,
2020, Art. no. 118.

[35] M. Y. Wang, “Deep graph library: Towards efficient and scalable deep

learning on graphs,” in _Proc. ICLR Workshop Representation Learn._
_Graphs Manifolds_ [, 2019, pp. 1–7. [Online]. Available: https://par.nsf.gov/](https://par.nsf.gov/biblio/10311680)
[biblio/10311680](https://par.nsf.gov/biblio/10311680)

[36] Y. Sun, D. Zhu, H. Du, and Z. Tian, “MHNF: Multi-hop heterogeneous

neighborhood information fusion graph representation learning,” _IEEE_
_Trans. Knowl. Data Eng._, vol. 35, no. 7, pp. 7192–7205, Jul. 2023.

[37] A. Vaswani et al., “Attention is all you need,” in _Proc. Adv. Neural Inf._

_Process. Syst._, 2017, pp. 1–15, vol. 30.

[38] B. Yang, W.-t. Yih, X. He, J. Gao, and L. Deng, “Embedding entities and

relations for learning and inference in knowledge bases,” in _Proc. ICLR_
_Workshop Representation Learn._, 2014, pp. 1–12.

[39] J. Li, S. Zhang, T. Liu, C. Ning, Z. Zhang, and W. Zhou, “Neural inductive

matrix completion with graph convolutional networks for miRNA-disease
association prediction,” _Bioinformatics_, vol. 36, no. 8, pp. 2538–2546,
2020.




[40] L. Peng, C. Yang, L. Huang, X. Chen, X. Fu, and W. Liu, “RNMFLP:

Predicting circRNA–disease associations based on robust nonnegative
matrix factorization and label propagation,” _Brief. Bioinf._, vol. 23, no. 5,
[2022, Art. no. bbac155, doi: 10.1093/bib/bbac155.](https://dx.doi.org/10.1093/bib/bbac155)

[41] L. Peng, C. Yang, Y. Chen, and W. Liu, “Predicting CircRNA-disease

associations via feature convolution learning with heterogeneous graph
attention network,” _IEEE J. Biomed. Health Informat._, vol. 27, no. 6,
pp. 3072–3082, Jun. 2023.

[42] M. T. Islam et al., “Revealing hidden patterns in deep neural network

feature space continuum via manifold learning,” _Nature Commun._, vol. 14,
no. 1, 2023, Art. no. 8506.

[43] I. Lazarev, N. Sion-Vardy, and S. Ariad, “EML4-ALK-positive non-small

cell lung cancer in a patient treated with azathioprine for ulcerative colitis,”
_Tumori J._, vol. 98, no. 4, pp. e98–e101, 2012.

[44] J. A. Miskoff and M. Chaudhri, “Low dose naltrexone and lung cancer: A

case report and discussion,” _Cureus_, vol. 10, no. 7, 2018, Art. no. e2924.

[45] J. Y. Jeong et al., “Trifluoperazine and its analog suppressed the tumori
genicity of non-small cell lung cancer cell; applicability of antipsychotic
drugs to lung cancer treatment,” _Biomedicines_, vol. 10, no. 5, 2022,
Art. no. 1046.

[46] N. A. Thabet, N. El-Guendy, M. M. Mohamed, and S. A. Shouman,

“Suppression of macrophages-induced inflammation via targeting ras and
par-4 signaling in breast cancer cell lines,” _Toxicol. Appl. Pharmacol._,
vol. 385, 2019, Art. no. 114773.

[47] S. Mazzilli et al., “Effects of topical 0.8% piroxicam and 50 sunscreen

filters on actinic keratosis in hypertensive patients treated with or without
photosensitizing diuretic drugs: An observational cohort study,” _Clin.,_
_Cosmetic Investigational Dermatol._, vol. 11, pp. 485–490, 2018.

[48] M. Trujillo et al., “Piroxicam-induced photodermatitis. Cross-reactivity

among oxicams. A case report,” _Allergologia et immunopathologia_,
vol. 29, no. 4, pp. 133–136, 2001.

[49] X. Chen, B. Ren, M. Chen, Q. Wang, L. Zhang, and G. Yan, “NLLSS: Pre
dicting synergistic drug combinations based on semi-supervised learning,”
_PLoS Comput. Biol._, vol. 12, no. 7, 2016, Art. no. e1004975.

[50] X. Chen et al., “Drug–target interaction prediction: Databases, web servers

and computational models,” _Brief. Bioinf._, vol. 17, no. 4, pp. 696–712,
2016.

[51] Z. Xiong et al., “Multi-relational contrastive learning graph neural network

for drug-drug interaction event prediction,” in _Proc. AAAI Conf. Artif._
_Intell._, 2023, vol. 37, pp. 5339–5347.

[52] Q. Ning et al., “AMHMDA: Attention aware multi-view similarity net
works and hypergraph learning for miRNA–disease associations identification,” _Brief. Bioinf._, vol. 24, no. 2, 2023, Art. no. bbad094.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:29:01 UTC from IEEE Xplore. Restrictions apply.


