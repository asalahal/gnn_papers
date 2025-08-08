[Expert Systems With Applications 252 (2024) 124289](https://doi.org/10.1016/j.eswa.2024.124289)


[Contents lists available at ScienceDirect](https://www.elsevier.com/locate/eswa)

# Expert Systems With Applications


[journal homepage: www.elsevier.com/locate/eswa](https://www.elsevier.com/locate/eswa)

## Drug–target interaction prediction based on improved heterogeneous graph representation learning and feature projection classification


Donghua Yu [a] [,] [b], Huawen Liu [a] [,] [b], Shuang Yao [c] [,] [d] [,][∗]


a _Department of Computer Science and Engineering, Shaoxing University, Shaoxing 312000, China_
b _Institute of Artificial Intelligence, Shaoxing University, Shaoxing 312000, China_
c _College of Economics and Management, China Jiliang University, Hangzhou 310000, China_
d _Institute of Digitalization and Data Intelligence, China Jiliang University, Hangzhou 310000, China_



A R T I C L E I N F O


_Keywords:_
Drug discovery and design
Drug–target interaction
Heterogeneous network
Feature projection
Molecular docking


**1. Introduction**



A B S T R A C T


Drug–target interaction (DTI) identification is a complex process that is time-consuming, costly and frequently
inefficient, with a low success rate, especially with wet-experimental methods. The prediction of DTI by
calculational methods is an effective way to solve this problem. However, most existing methods regard
drug–target pairs with unknown interaction relationship as negative samples, that the false negative samples
will affect the AUC and AUPR evaluation, leading to performance misjudgment. Therefore, in this paper, a
new DTI prediction method, DTI-HAN, is proposed to overcome the shortcoming and further improve the
predictive performance. In the drug and target feature representation learning stage, this method constructs
a drug–target heterogeneous network based on similarity and interaction relationship, and establishes metapath, node-level and semantic-level bi-attention mechanism. To avoid introducing false negative examples, an
improved loss function based only on known edges is proposed. In the DTI prediction stage, feature projection
and fuzzy theory are introduced, and membership distribution function is estimated only depended on positive
samples. Compared with the DTI prediction methods, DTI-GAT, DTI-GCN and DTI-GraphSAGE, as well as BLM,
DTHybrid, SCMLKNN and FPSC-DTI, the experimental results on Enzyme, GPCR, Ion Channel and Nuclear
Receptor 4 datasets showed that the DTI-HAN method can greatly improve the AUC and AUPR values on
at least 3 datasets to each method. Furthermore, the novel top 100-pair of DTIs prediction were verified by
KEGG, DrugBank, ChEMBL and SuperTarget databases, and obtained 21, 48, 61 and 28 validation records,
respectively. The top-5 remaining unverified DTIs were performed molecular docking with AutoDock Vina,
and the results showed that these drugs and targets have good binding properties. The code and data are
[available at https://github.com/Yu123456/DTI-HAN.](https://github.com/Yu123456/DTI-HAN)



Drug–target interaction (DTI) refers to a drug that controls, prevents, cures, and diagnoses diseases by reacting with a target and
eliciting some form of positive biological response, such as modifying
target function and/or activity (Yu, Liu, Zhao, Liu, & Guo, 2020).
The DTI prediction is an important link and research field in drug
discovery and design, and plays a crucial role in drug development,
such as virtual drug screening, new use of old drugs, drug toxicity
and side effects (Paul et al., 2010). However, the DTIs identification
by wet experimental methods is expensive, long-term, high-risk, low in
success rate and efficiency, and the prediction of DTI by computational
methods is an effective way to solve this problem, showing strong
vitality and advantages.



With the accumulation of data on drugs, targets, and drug–target
interactions and the in-depth research on theories and practices such
as machine learning, data mining, and network pharmacology, it is
possible to use machine learning methods to predict drug–target interaction (Bagherian et al., 2021; Liu et al., 2021b; Sydow et al., 2019).
The calculation methods of DTI prediction are mainly classified into
3 categories: ligand-based methods (Hopkins, Keserü, Leeson, Rees, &
Reynolds, 2014), docking-based methods (Santana et al., 2012) and
chemical genomics-based methods (Bredel & Jacoby, 2004). Ligandbased methods compare and analyze the similarity between candidate compounds and target ligands, such as QSAR (Du, Wang, &
Li, 2022), but currently there are only limited known target ligands
and their application scope is limited. Docking simulation methods



∗ Corresponding author at: College of Economics and Management, China Jiliang University, Hangzhou 310000, China.
_E-mail addresses:_ [donghuayu163@163.com (D. Yu), liu@usx.edu.cn (H. Liu), alloniam@163.com (S. Yao).](mailto:donghuayu163@163.com)


[https://doi.org/10.1016/j.eswa.2024.124289](https://doi.org/10.1016/j.eswa.2024.124289)
Received 26 March 2024; Received in revised form 1 May 2024; Accepted 20 May 2024

Available online 27 May 2024
0957-4174/© 2024 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.


_D. Yu et al._



_Expert Systems With Applications 252 (2024) 124289_



**Fig. 1.** DTI-HAN: DTI prediction framework incorporating heterogeneous graph attention network, feature project and fuzzy theory. (a) heterogeneous network and meta-path, (b)
node-level attention, (c) semantic-level attention, (d) feature projection and fuzzy theory.



mainly use dynamic simulation, such as Dock (Allen et al., 2015)
and AutoDock (Morris et al., 2009), but these methods are limited
to protein receptors (targets) whose 3D structure is known. However, the 3D structures of most targets have not yet resolved, so
their application scope is limited. Compared with the above methods,
chemical genomics-based methods require fewer restrictions, which
systematically screen the small molecule library of each target family
to develop new drugs, using only the genome and protein sequences
rather than 3D structures, or knowing only a small number of target
ligands, or even predicting unknown targets. In chemical genomicsbased methods, predicting DTI based on deep learning or/and graph
network has shown a good performance (Dehghan, Razzaghi, Abbasi,
& Gharaghani, 2023; Meng et al., 2024; Wen et al., 2017), and some of
these methods can be extended to synergy drug research. For examples,
AMDGT (Liu et al., 2024b) utilized dual graph transformer models to
learn representations of drugs and diseases from homogeneous and
heterogeneous networks, respectively. CFSSynergy (Rafiei et al., 2024)
explored the drug synergy problem between drug pairs and target
(cell lines), using a transformer framework to extract drug features
and concatenating the two drugs. These algorithms employed in deep
learning models for DTI prediction have grown progressively diverse
and intricate. The deep learning-based representation techniques have
directly contributed to the diversity in the input part of various drug
and target representation methods. Compared to traditional machine
learning methods, these deep learning-based models entail relatively
high resource consumption. For more work on predicting drug–target
interactions using deep learning, see the literature review (Shi, Yang,
Xie, Yin, & Zhang, 2024). The data about drugs, targets and verified
DTIs are converted into features and labels for training the prediction
models. These features in turn are used to predict new drug or target, as
well as new DTIs between them. When we fight against Covid-19, chemical genomics-based methods play a very important role in predicting
potential drug candidates against SARS-COV2 (Sivangi, Amilpur, &
Dasari, 2023). Ahmed et al. summarized the machine learning and deep
learning based drug repurposing studies for Covid-19 (Ahmed et al.,
2022).
However, most existing methods regard drug-target pairs with unknown interaction relationship as negative samples (Elbasani et al.,
2021; Li, Cai, Xu, & Ji, 2023; Liu et al., 2021b; Peng et al., 2021;
Tian et al., 2022; Yamanishi, Kotera, Kanehisa, & Goto, 2010). The
false negative samples, i.e. true samples labeled as negative samples,
are fed into the model for training. This erroneous prior information
can affect the model’s judgment of real sample labels and even lead
to misjudgments. In this paper, we proposed a new method (DTI-HAN)



based on heterogeneous graph attention mechanism, feature projection
and fuzzy theory to address this issue and improve prediction accuracy.
The main contributions are as follows:


1. A new framework for predicting drug-target interaction called
DTI-HAN has been proposed, see Fig. 1. In this framework, on
the one hand, only very simple input data is needed to construct heterogeneous networks, namely drug–target interaction
relationship, drug-drug similarity, and target–target similarity;
On the other hand, it integrates various very effective techniques
to avoid using false negative samples as much as possible and
improve prediction accuracy.
2. In the representation learning stage of drug and target feature,
the DTI-HAN introduces the most effective techniques currently
available, such as meta-path, node-level attention, semanticlevel attention, and simultaneously improves the loss function
which is only based on drug-drug, target–target similarities,
avoiding the introduction of false negative samples.
3. In the prediction stage of drug–target interaction, feature projection and fuzzy theory are introduced, and the membership distribution function is estimated only depending on positive samples to effectively prevent the negative impact of false negative
samples.
4. Experimental results on E, GPCR, IC and NR showed that the proposed DTI-HAN method has improved on AUC and AUPR, and
its overall performance is better than that of DTI-GAT, DTI-GCN
and DTI-GraphSAGE, as well as BLM, DTHybrid, SCMLKNN and
FPSC-DTI. The novel top 100-pair of DTIs prediction were verified by KEGG, DrugBank, ChEMBL and SuperTarget databases,
and obtained 21, 48, 61 and 28 validation records, respectively.
The top-5 remaining unverified DTIs were performed molecular
docking with AutoDock Vina, and the docking results showed
that these drugs and targets have good binding properties.


**2. Related works**


_2.1. Heterogeneous graph and meta-path_


Recently, network representation learning has attracted much attention to learn rich information from heterogeneous data and achieved
success in predicting DTIs (Djeddi, Hermi, Ben Yahia, & Diallo, 2023;
Su, Hu, Yi, You, & Hu, 2022).
In a heterogeneous graph, two nodes of the same type can be
connected by different semantic paths, called meta-path (Sun, Han,
Yan, Yu, & Wu, 2011). A meta-path _𝛷_ is defined as a path in the form of



2


_D. Yu et al._


**Fig. 2.** Multiple Node Types and Multiple Meta-paths Types in Heterogeneous Graph.


_𝑅_ 1 _𝑅_ 2 _𝑅_ _𝑙_
_𝐴_ 1 → _𝐴_ 2 → _..._ → _𝐴_ _𝑙_ +1 (abbreviated as _𝐴_ 1 _𝐴_ 2 _...𝐴_ _𝑙_ +1 ). Through the metapath, the neighbor set of the node can be found. Since heterogeneous
graph has multiple meta-paths and different types of nodes and edges,
it is unreasonable to consider each meta-path equally, which will
weaken the specific semantic information provided by some useful
meta-paths (Wang et al., 2021). As shown in Fig. 2, a heterogeneous
graph is composed of multiple types of nodes (Drug, Gene, Disease) and
edges (drug to disease, disease to gene). Two diseases can be linked
through multiple meta-paths, such as Disease–Drug–Disease(DiDrDi)
and Disease–Gene–Disease(DiGDi). Different meta-paths reveal different semantics. Meta-path is widely used in disease-related heterogeneous networks, such as HGATLDA (Zhao, Zhao, & Yin, 2022),
MHGNN (Li et al., 2023), MHTAN-DTI (Zhang, Wang, Wang, Meng, &
Cui, 2023).
Given a meta-path _𝛷_, there exists neighbor set of each node. Define
a set _𝑁_ _𝑖_ _[𝛷]_ [, which means the set of neighbors connected to node] _[ 𝑖]_
through the meta-path _𝛷_, as well as contained itself. Taking Fig. 2(d) as an example, given the meta-path Disease–Drug–Disease(DiDrDi),
the node _𝐷𝑖_ 1 neighbors based on DiDrDi includes _𝐷𝑖_ 1, _𝐷𝑖_ 2 and _𝐷𝑖_ 3 .
Similarly, the neighbors of _𝐷𝑖_ 1 based on meta-path Disease–Gene–
Disease(DiGDi) includes _𝐷𝑖_ 1 and _𝐷𝑖_ 2 . In heterogeneous graphs, metapath based neighbors can reveal diverse structure information and rich
semantic information (Liu, Chen, Lan, Lu & Zhang, 2024a; Su et al.,
2024; Zhang et al., 2023).


_2.2. Graph-based prediction of drug–target interaction_


With the popularity of graph neural network in various fields (Li,
Cao, Tanveer, Pandey & Wang, 2019; Li, Cao, Zhong & Li, 2019),
scholars began to apply it to DTI prediction. Thafar et al. (2020)
proposed a calculation method DTIGEM to predict DTIs. It combined
similarity-based and feature-based techniques, using graph embedding
and graph mining. Parvizi, Azuaje, Theodoratou, and Luz (2020) used
node2vec embedding method to extract drug and protein latent features
from drug and protein related networks and then to predict DTI. Liu,
Pliakos, Vens and Tsoumakas (2021a) proposed a neighbor-based DTI
prediction method WkNNIR which cannot only estimate the interaction
of any new drug and/or new target without any retraining, but also
recover the unproven interaction, namely the current false-negative

DTIs.

Then it gradually transitioned to the DTI heterogeneous network.
Heterogeneous network is a powerful tool for semantic modeling of
complex data with various vertices and edges (Wang et al., 2021).
It is popular to use heterogeneous network to represent the features
of drugs and targets and the different relationships between them,
e.g. GCRNN (Elbasani et al., 2021), EmbedDTI (Jin, Lu, Shi, & Yang,
2021), NGDTP (Xuan, Chen, Zhang, et al., 2020), DTI-HeNE (Yue & He,
2021), NeoDTI (Wan, Hong, Xiao, Jiang, & Zeng, 2019), EEG-DTI (Peng
et al., 2021), MHADTI (Tian et al., 2022), MHTAN-DTI (Zhang et al.,
2023), SSLDTI (Liu, Chen et al., 2024a), AMGDTI (Su et al., 2024) etc.



_Expert Systems With Applications 252 (2024) 124289_


Among them, GCRNN used a graph neural network to represent the
compound, and a convolutional layer extended with a bidirectional
recurrent neural network framework to vectorize the protein sequence.
EmbedDTI leveraged language modeling for pretraining the target feature embeddings and fed them into a convolutional neural network
model for further representation learning, and for drugs, it built two
levels of graphs to represent compound structural information and
employed a graph convolutional network with an attention module to
learn the embedding vectors. DTI-HeNE was used to deal with the DTI
of bipartite graph. NeoDTI integrated the neighbor information through
a large number of information transfer and aggregation operations,
and EEG-DTI learnt the latent feature representation of nodes through
an end-to-end framework based on heterogeneous graph convolution
network. SSLDTI utilized a GCN to extract features from heterogeneous graphs and generated embedding vectors of drugs and targets.
MHTAN-DTI applied meta-path instance-level transformer and attention network for DTIs prediction. AMGDTI automatically aggregated
semantic information from a heterogeneous network by training an
adaptive meta-graph.
The introduction of attention mechanism in heterogeneous networks
can also be beneficial of predicting performance improvement, such
as DTI-HETA (Shao et al., 2022), MHTAN-DTI (Zhang et al., 2023).
In addition to introducing attention mechanisms to obtain better drug
and target nodes embeddings, there are also studies introducing contrastive learning methods to extract more discriminative features of
drugs and targets to improve DTI prediction. For example, CCL-DTI (Dehghan, Abbasi, Razzaghi, Banadkuki, & Gharaghani, 2024) explored
four contrastive loss functions and demonstrated excellent performance
in DTI prediction. However, on the one hand, graph neural network
representation learning is easily over-smoothed and only captures loworder features; on the other hand, the introduction of too much extra
information would degrade the representation learning performance
of the target source data with the scarcity of shared data between
multi-source data. Zeng et al. (2020) introduced 15 heterogeneous
data sources at the same time. However, the conclusion showed that
the predicting performance is not improved as expected. Zong et al.
(2021) research showed that in order to achieve better prediction
performance, data sources need to be reasonably selected, not the
more data sources, the better performance. Different from graph neural
network, hypergraph neural network can capture higher-order features
and has been successfully applied to DTI prediction (Feng, You, Zhang,
Ji, & Gao, 2019), e.g. HGDD (Pang et al., 2021),HHDTI (Ruan et al.,
2021). However, the hyperedge structure is limited to known drugs and
targets, and cannot be directly extended to the prediction of new drugs
(or compounds) and new targets. In addition, HGDD and HHDTI are
also inadequate in providing higher-order characteristic information for
drug–target pairs with only unique interactions.


**3. Method**


_3.1. Construction of DTI heterogeneous network_


Drug _𝑑_ _𝑖_ and target _𝑡_ _𝑗_ are used as nodes in the network, drug-drug
similarity _𝑆_ _𝑑_ and target–target similarity _𝑆_ _𝑡_ are used as weights to
establish edges, and drug-target heterogeneous network is established
with the known DTIs relationship as edges and the weight set as 1.
In order to capture the topological information of DTI heterogeneous network, it is necessary to consider the appropriate information
aggregation mode. Therefore, in this paper, meta-paths consisting of
edges and nodes are established on the heterogeneous network, and
information aggregation is realized through these meta-paths.
In DTI heterogeneous network, according to different types of nodes
and edges, the following meta-paths are established:


_𝑟_ _𝑟_
_𝛷_ 1 ∶ Drug → Drug → Drug


_𝑟_ _𝑟_
_𝛷_ 2 ∶ Target → Target → Target



3


_D. Yu et al._


_𝑟_ _𝑟_
_𝛷_ 3 ∶ Drug → Target → Drug


_𝑟_ _𝑟_
_𝛷_ 4 ∶ Target → Drug → Target


where _𝑟_ denotes the edge of network.


_3.2. Node-level attention_


Since nodes are heterogeneous, different types of nodes belong to
different feature spaces corresponding to different feature dimensions.
In order to obtain the features of drug and target nodes, the typespecific transformation matrix _𝑀_ _𝜙_ _𝑖_ is designed to project drug and
target into the same feature space:


_ℎ_ _𝑖_ ′ = _𝑀_ _𝜙_ _𝑖_ ⋅ _ℎ_ _𝑖_ _,_ (1)


where _ℎ_ _𝑖_ and _ℎ_ _𝑖_ ′ are the original and projected feature of node _𝑖_, respectively. Node-level attention mechanism is established for different
types of nodes on the same meta-path. Given a node pair ( _𝑖, 𝑗_ ) which are
connected via meta-path _𝛷_, the asymmetric importance _𝑒_ _[𝛷]_ _𝑖𝑗_ [of meta-path]
based node pair ( _𝑖, 𝑗_ ) can be formulated as follows:

_𝑒_ _[𝛷]_ _𝑖𝑗_ [=] _[ 𝜎]_ ( _𝛼_ _𝛷_ _[𝑇]_ [⋅] [ _ℎ_ [′] _𝑖_ [∥] _[ℎ]_ [′] _𝑗_ ]) _,_ (2)


where _𝜎_ (∙) is the activation function, _𝛼_ _𝛷_ is the node-level attention
vector for meta-path _𝛷_, ∥ represents vector concatenation. Given metapath _𝛷_, _𝛼_ _𝛷_ is shared by all drug pairs or target pairs based on the
meta-path. The importance _𝑒_ _[𝛷]_ _𝑖𝑗_ [are normalized via softmax function to]
serve as the weight coefficient _𝛼_ _𝑖𝑗_ _[𝛷]_ [:]

_𝛼_ _𝑖𝑗_ _[𝛷]_ [= softmax] ( _𝑒_ _[𝛷]_ _𝑖𝑗_ ) = exp ( _𝑒_ _[𝛷]_ _𝑖𝑗_ ) _,_ (3)

~~∑~~ _𝑙_ ∈ _𝑁_ _𝑖_ _[𝛷]_ [exp] ~~[ (]~~ _[𝑒]_ _𝑖𝑙_ _[𝛷]_ ~~)~~


The embedding feature _𝑍_ _𝑖_ _[𝛷]_ [of node] _[ 𝑖]_ [for meta-path] _[ 𝛷]_ [can be aggregated]
by the following formula:



_Expert Systems With Applications 252 (2024) 124289_


_3.4. Improved loss function_


As emphasized in Section 2.2, most GNN-based methods ultimately
used cross entropy loss function, as shown in Eq. (8).



_𝐿_ = −
∑

( _𝑖,𝑗_ )∈ _𝑌_ [+] ∪ _𝑌_ [−]




[ _𝑦_ _𝑖𝑗_ ln _𝑦_ [′] _𝑖𝑗_ [+ (1 −] _[𝑦]_ _[𝑖𝑗]_ [) ln(1 −] _[𝑦]_ [′] _𝑖𝑗_ [)] ]



(8)



where _𝑌_ [+] and _𝑌_ [−] denote positive and negative training samples, respectively. ( _𝑖, 𝑗_ ) denotes a given pair of drug _𝑑_ _𝑖_ and target _𝑡_ _𝑗_ . _𝑦_ _𝑖𝑗_ and
_𝑦_ [′] _𝑖𝑗_ [denote the ground-truth label and predicted score, respectively. In]
the study of predicting drug–target interactions, _𝑌_ [−] is defined as negative sample set that actually contains drug–target pairs that have not
been validated by wet-experiment. However, there is real interaction
relationship between these unverified drug–target pairs that have not
yet been validated by biological experiments. This scenario is the core
foundation of drug repositioning research, and only these unknown
drug–target pairs with interaction relationship can find new targets
for known drugs. However, for this very reason, these true positives
are treated as negatives, i.e. false negatives, when used to train models, which has a negative impact on the prediction of drug–target

interactions.


To avoid the above situation, and considering that the topological
properties of the heterogeneous network and the relationship between
nodes should be maintained after nodes are embedded, the improved
loss function is proposed as follows:



_𝐿_ = ∑

_𝑟_ _𝑖𝑗_ ∈ _𝐸_



_𝑇_ 2
( _𝑠_ ( _𝑟_ _𝑖𝑗_ ) − _𝑍_ _𝑖_ [⋅] _[𝑍]_ _[𝑗]_ ) _,_ (9)



⎛
_𝑍_ _𝑖_ _[𝛷]_ [=] _[ 𝜎]_ ⎜⎜⎝



where _𝑠_ (∙) denotes the weight of heterogeneous network’s known edges,
namely drug-drug similarity and target–target similarity. _𝑟_ _𝑖𝑗_ represents
the edge from the source node _𝑖_ to the target node _𝑗_ . _𝑍_ is the embedding
vector of the drug or target. Because Eq. (9) only introduces known
drug-drug and target–target similarities without applying positive and
negative samples, it can effectively prevent the negative impact of false
negative samples.


_3.5. DTI predicting based on feature projection and Fuzzy theory_


The DTI prediction based on feature projection was proposed by Luo
et al. (2017) and Yu et al. (2020). A brief description of this method
is given below.

Let _𝑑_ _𝑖_ ∈ _𝑆𝑒𝑡_ _𝑑𝑟𝑢𝑔_ and _𝑡_ _𝑗_ ∈ _𝑆𝑒𝑡_ _𝑡𝑎𝑟𝑔𝑒𝑡_ be the drug and target embedding
feature vectors, respectively. Let define a feature projection matrix _𝑅_
and the projection relationship from the drug space to the target space
as follows:

_𝑑_ _𝑖_ _[𝑇]_ _[𝑅𝑡]_ _[𝑗]_ [=] _[ 𝐴]_ _[𝑖𝑗]_ [=] { 1 _,_ if _𝑑_ 0 _𝑖_ _,_ otherwiseinteracts with _𝑡_ _𝑗_ (10)


Where _𝐴_ _𝑖𝑗_ represents the interaction relationship between drug _𝑑_ _𝑖_ and
target _𝑡_ _𝑗_ .

For the DTI prediction problem, the following optimization problem
is formulated:



∑



⎞

_𝛼_ _𝑖𝑗_ _[𝛷]_ [⋅] _[ℎ]_ [′] _𝑗_ ⎟⎟⎠ _,_ (4)



_𝑗_ ∈ _𝑁_ _𝑖_ _[𝛷]_



Given a meta-path set [{] _𝛷_ 0 _, 𝛷_ 1 _,_ … _, 𝛷_ _𝑃_ }, after feeding node features into
node-level attention, we can obtain _𝑃_ groups of semantic-specific node
embeddings, denoted as { _𝑍_ _𝛷_ 0 _, 𝑍_ _𝛷_ 1 _,_ … _, 𝑍_ _𝛷_ _𝑃_ }.


_3.3. Semantic-level attention_


The attention mechanism of semantic-level is considered for differ
ent meta-paths on the same node, and _𝑃_ groups of semantic-specific
node embeddings learned from node-level attention are taken as input.
The importance of meta-path _𝛷_ _𝑖_ can be calculated as follows:



1
_𝜔_ _𝛷_ _𝑖_ = | _𝑉_ |



∑ _𝑞_ _[𝑇]_ ⋅ tanh ( _𝑊_ ⋅ _𝑍_ _𝑗_ _[𝛷]_ _[𝑖]_ + _𝑏_ ) _,_ (5)

_𝑗_ ∈ _𝑉_



where _𝑊_ is the weight matrix, _𝑏_ is the bias vector, _𝑞_ is the semanticlevel attention vector. _𝑉_ is the node set. Similarly, the weight coefficient _𝛽_ _𝛷_ _𝑖_ of meta-path _𝛷_ _𝑖_ is normalized via softmax function:



2



_,_ (11)
)



min
_𝑅_ _[𝐽]_ [= min] _𝑅_



2

_𝐴_ − _𝑋𝑅𝑌_ _𝑇_
(‖‖‖ ‖‖‖ _𝐹_



2

_𝐹_ [+] _[ 𝜆]_ [‖] _[𝑅]_ [‖] _𝐹_ [2]



_𝛽_ _𝛷_ _𝑖_ = exp ( _𝜔_ _𝛷_ _𝑖_



)



_,_ (6)
~~)~~



_𝑇_ _𝑇_
where _𝑋_ = [[] _𝑑_ 1 _,_ … _, 𝑑_ _𝑚_ ] and _𝑌_ = [ _𝑡_ 1 _,_ … _, 𝑡_ _𝑛_ ] are the drug and target
feature matrixes, respectively; _𝜆_ is a regularization parameter and ‖⋅‖ _𝐹_
represents the Frobenius norm. The Eq. (11) can be solved and obtained
optimal solution _𝑅_ [∗] . Then, the projection values _𝐴_ [∗] from the drug space
into the target space can be computed:


_𝐴_ [∗] = _𝑋𝑅_ [∗] _𝑌_ _[𝑇]_ _,_ (12)


Then, the fuzzy membership degree function _𝑓_ [(] _𝑥, 𝜇_ _𝑘_ _, 𝜎_ _𝑘_ ) is designed as follow:


− [(] _[𝑥]_ [−] _[𝜇][𝑘]_ [)] [2]
_𝑓_ [(] _𝑥, 𝜇_ _𝑘_ _, 𝜎_ _𝑘_ ) = _𝑒_ _𝜎𝑘_ [2] _,_ (13)



_𝑃_
~~∑~~ _𝑙_ =1 [exp] ~~(~~ _𝜔_ _𝛷_ _𝑙_



which can be interpreted as the contribution of the meta-path _𝛷_ _𝑖_ for
representation learning of drug or target. Finally, the embedding vector
_𝑍_ _𝑖_ for the node _𝑖_ is integrated by these semantic-specific embeddings
as follows:



_𝑍_ _𝑖_ =



_𝑃_
∑ _𝛽_ _𝛷_ _𝑗_ ⋅ _𝑍_ _𝑖𝛷_ _𝑗_ _,_ (7)

_𝑗_ =1



where _𝑃_ denotes the number of meta-paths.



4


_D. Yu et al._


**Table 1**

Statistic information of datasets.



_Expert Systems With Applications 252 (2024) 124289_


**Fig. 3.** DTI Cross Validation Process.


**Table 2**

Meta-path Information in Drug–target Heterogeneous Network.



Dataset Drugs Targets Known DTIs Unknown DTIs


E 445 664 2926 (1%) 292554 (99%)

GPCR 223 95 635 (3%) 20550 (97%)

IC 210 204 1476 (3.45%) 41364 (96.55%)

NR 54 26 90 (6.41%) 1314 (93.59%)


The parameters _𝜇_ _𝑘_, _𝜎_ _𝑘_ can be estimated by Eqs. (14) and (15).



Dataset Relation Number Number Number meta-path
(A–B) of A of B of A–B


_𝑑𝑟𝑢𝑔_ − _𝑑𝑟𝑢𝑔_ 445 445 198 025 DDD

E _𝑑𝑟𝑢𝑔_ − _𝑡𝑎𝑟𝑔𝑒𝑡_ 445 664 2926 DTD, TDT

_𝑡𝑎𝑟𝑔𝑒𝑡_ − _𝑡𝑎𝑟𝑔𝑒𝑡_ 664 664 440 896 TTT


_𝑑𝑟𝑢𝑔_ − _𝑑𝑟𝑢𝑔_ 223 223 49 729 DDD

GPCR _𝑑𝑟𝑢𝑔_ − _𝑡𝑎𝑟𝑔𝑒𝑡_ 223 95 635 DTD, TDT

_𝑡𝑎𝑟𝑔𝑒𝑡_ − _𝑡𝑎𝑟𝑔𝑒𝑡_ 95 95 9025 TTT


_𝑑𝑟𝑢𝑔_ − _𝑑𝑟𝑢𝑔_ 210 210 44 100 DDD

IC _𝑑𝑟𝑢𝑔_ − _𝑡𝑎𝑟𝑔𝑒𝑡_ 210 204 1476 DTD, TDT

_𝑡𝑎𝑟𝑔𝑒𝑡_ − _𝑡𝑎𝑟𝑔𝑒𝑡_ 204 204 41 616 TTT


_𝑑𝑟𝑢𝑔_ − _𝑑𝑟𝑢𝑔_ 54 54 2916 DDD

NR _𝑑𝑟𝑢𝑔_ − _𝑡𝑎𝑟𝑔𝑒𝑡_ 54 26 90 DTD, TDT

_𝑡𝑎𝑟𝑔𝑒𝑡_ − _𝑡𝑎𝑟𝑔𝑒𝑡_ 26 26 676 TTT


is shown in Fig. 3, and the red is the validation dataset. Note that in
cross-validation, each row and column is guaranteed to contain at least
one 1, i.e., no all zero rows or columns. The data is divided into training
set and test set. In the verification data of training set, if the element
is 1, it is reset to 0. When the element is 0, it is retained.
The FPRM model needs to pre-define the regularization parameter
_𝜆_, and the goal is to achieve a trade-off between complexity and
performance. If the _𝜆_ is too high, although the model will tend to be
simplified, it may cause data under-fitting and cannot obtain enough
information from the training data to make effective predictions. If the
_𝜆_ is too low, the model will tend to be complicated, it may cause data
over-fitting, resulting in a decline in the generalization performance of
the model and poor prediction performance for new data.
In order to evaluate the prediction performance of the model reasonably and fairly, AUC and AUPR are used as evaluation indexes
in this paper (Schrynemackers, Küffner, & Geurts, 2013). The ROC
curve directly shows the true case rate and false positive case rate
of the learner in the sample. If comparison need be made, the more
reasonable criterion is to compare the area under the ROC curve,
namely AUC. However, there are more negative samples than positive
samples, and AUC is unbalanced and insensitive to the categories with
large proportion, so the PR curve that severely punishes high-ranking
false positive predictions is also used. Similarly, if the PR curves of the
two learners cross, a more reasonable criterion is to compare the area
under the PR curve, that is, AUPR.


_4.3. Experimental results and analysis_


The meta-path information was shown in the Table 2. The _𝑑𝑟𝑢𝑔_ −
_𝑑𝑟𝑢𝑔_ represents the drug-drug compound similarity calculated based
on SIMCOMP algorithm (called SIMCOMP similarity). The _𝑡𝑎𝑟𝑔𝑒𝑡_ −
− _𝑡𝑎𝑟𝑔𝑒𝑡_ represents the target–target sequence similarity based on Smith
−
Waterman score (called Smith-Waterman similarity). The _𝑑𝑟𝑢𝑔_ _𝑡𝑎𝑟𝑔𝑒𝑡_
indicates whether there is an interaction relationship between the drug



_𝜇_ _𝑘_ =


_𝜎_ _𝑘_ [2] [=]



∑ _𝑖,𝑗_ _[𝜒]_ [(] _[𝐴]_ _𝑖𝑗_ [==] _[ 𝑙]_ _𝑘_ ) ⋅ _𝐴_ ∗ _𝑖𝑗_
_,_ (14)
~~∑~~ _𝑖,𝑗_ _[𝜒]_ ~~[(]~~ _[𝐴]_ _𝑖𝑗_ [==] _[ 𝑙]_ _𝑘_ ~~)~~


2
∑ _𝑖,𝑗_ _[𝜒]_ [(] _[𝐴]_ _𝑖𝑗_ [==] _[ 𝑙]_ _𝑘_ ) ⋅ ( _𝐴_ [∗] _𝑖𝑗_ [−] _[𝜇]_ _[𝑘]_ )

_,_ (15)
~~∑~~ _𝑖,𝑗_ _[𝜒]_ ~~[(]~~ _[𝐴]_ _𝑖𝑗_ [==] _[ 𝑙]_ _𝑘_ ~~)~~



where _𝑙_ _𝑘_ = 0 _,_ 1 represents non-interaction and interaction relationships,
respectively, and _𝜒_ (⋅) is an indicator function. In order to avoid the
influence of false negative samples, only the confidence score of positive samples is used to estimate the membership distribution function.
Therefore, set _𝑘_ = 2 and obtain the confidence score _𝑆_ _𝑖𝑗_ between _𝑑_ _𝑖_ and
_𝑡_ as follows:
_𝑗_


_𝑆_ _𝑖𝑗_ = _𝑓_ [(] _𝐴_ _𝑖𝑗_ ∗ _, 𝜇_ 2 _, 𝜎_ 2 ) _,_ (16)


**4. Results and analysis**


_4.1. Datasets_


In this paper, the 4 benchmark datasets, E, IC, GPCR and NR,
proposed by Yamanishi, Araki, Gutteridge, Honda, and Kanehisa (2008)
are used to evaluate DTI-HAN prediction performance. A detailed
description of them is shown in Table 1, where brackets indicate
proportions. It is worth noting that the number of positive samples is
much smaller than the number of negative samples, and there are a
large number of unknown DTIs that have not been verified by wetexperiment. Therefore, the positive and negative samples of these 4
datasets are seriously unbalanced and there are a large number of false
negative samples.


_4.2. Experimental settings and evaluation indexes_


On the basis of obtaining embedding features of drugs and targets
through GAT (Veličković, Cucurull, Casanova, Romero, Liò, & Bengio,
2018), GCN (Defferrard, Bresson, & Vandergheynst, 2016), GraphSAGE (Hamilton, Ying, & Leskovec, 2017) representation learning, they
are input into FPRM model to predict DTI. Meanwhile, we also carried
out a comparative analysis using traditional methods,BLM (Bleakley
& Yamanishi, 2009), DTHybrid (Alaimo, Pulvirenti, Giugno, & Ferro,
2013), SCMLKNN (Shi, Yiu, Li, Leung, & Chin, 2015), FPSC-DTI (Yu
et al., 2020). The model performance is compared by the 10-fold
cross-validation with evaluation indexes. The cross-validation process



5


_D. Yu et al._



_Expert Systems With Applications 252 (2024) 124289_



**Fig. 4.** AUC values of each model on different datasets.


**Fig. 5.** AUPR values of each model on different datasets.



and the target. If there exist interaction, the value is 1; otherwise, the
value is 0.

The histogram 4 given the AUC values of DTI-HAN, DTI-GAT,
DTI-GCN, DTI-GraphSAGE on 4 datasets, respectively. The red bar
represented the results of the proposed method DTI-HAN. The results
in the figure showed that the predicting performance of DTI-HAN is
better than that of the DTI-GAT (blue bar) in all 4 datasets. Compared
with DTI-GCN (purple bar), DTI-HAN was superior to it in E, GPCR
and IC. The DTI-HAN was slightly inferior to DTI-GCN in NR with
slight differences, but the AUC value of DTI-HAN was also more than
0.95. It was very close to the optimal value of 1. Compared with
the DTI-GraphSAGE (gray bar), its predicting performance in E, IC
and NR was inferior to that of the DTI-HAN, but in GPCR, it was
slightly superior to that of the DTI-HAN. However, The AUC of DTIHAN on the GPCR also exceeded 0.9, indicating excellent predictive
performance. In general, compared with the three comparison methods,
the predicting performance of DTI-HAN was better in at least 3 datasets.
GPCR and NR, which are inferior to the comparison method by slight
differences, were inferior to only one of them, indicating that the
prediction performance of DTI-HAN is relatively stable in each dataset.
Due to the majority of negative samples, it is a more reasonable to
consider AUPR metric. The Fig. 5 showed the AUPR values of DTI-HAN,



DTI-GAT, DTI-GCN and DTI-GraphSAGE on 4 datasets, respectively.
The red bar represented the results of the proposed method DTI-HAN.
Unlike the AUC values that are generally close to 1, the AUPR values
are generally small, all less than 0.7, except for the results of DTI-HAN
in the E dataset. Specifically, compared with DTI-GCN (purple bar),
the DTI-HAN was superior to DTI-GCN in E, GPCR and IC datasets,
and the performance improved significantly. In NR dataset, DTI-HAN
was slightly inferior to DTI-GCN with a small difference. However, the
AUPR value of DTI-HAN was also close to 0.6, indicating good performance. Compared with the DTI-GraphSAGE (gray bar), its predicting
performance on all 4 datasets was worse than that of DTI-HAN, and
the performance decreases significantly on E, GPCR and IC datasets.
On the whole, the evaluation results of AUPR were consistent with
that of AUC, but the performance improvement of DTI-HAN was much
larger than that of AUC. In this case where the negative samples are in
the majority, the evaluation of AUPR would be more convincing, and
on this indicator, the performance improvement of DTI-HAN is even

greater.

In addition to graph representation learning algorithms, Figs. 6 and
7 showed that the DTI-HAN method is also competitive with traditional
methods BLM, DTHybrid, SCMLKNN and FPSC-DTI. Specifically, in



6


_D. Yu et al._



_Expert Systems With Applications 252 (2024) 124289_



**Fig. 6.** AUC values of each model on different datasets.


**Fig. 7.** AUPR values of each model on different datasets.



terms of AUC values, the DTI-HAN method outperformed the comparison method on at least 3 datasets, and the AUC values of the
DTI-HAN method were all higher than 0.95. From the comparison of
the four methods, the FPSC-DTI method was the closest to the DTIHAN method proposed in this paper, but the latter still surpassed the
former on 3 datasets. Similarly, the DTI-HAN method was competitive
in terms of AUPR values. However, it should be acknowledged that on
the IC dataset, the DTI-HAN method with 0.62 was inferior to the BLM,
DTHybrid, SCMLKNN methods, and slightly better than the FPSC-DTI
method.

Comprehensively considering the evaluation of AUC and AUPR, the
proposed method DTI-HAN achieved excellent performance on all 4
datasets, and the predicting performance was more stable, guaranteeing
the top two in each dataset. Therefore, the DTI-HAN represented better
prediction performance and stronger adaptability for DTI prediction.


_4.4. Novel top 100-pair of DTIs prediction_


Further analysis the proposed model performance for predicting
novel DTIs, we performed the trained optimal parameter model to
predict all negative samples (non-interaction pairs) and ranked them
according to the confidence score. The 100 top-ranking non-interaction
pairs were taken for each dataset and verified whether have been



recorded as an interaction pair in KEGG, DrugBank, ChEMBL and
SuperTarget. Then, the top-100 DTIs network was constructed and
three types of interaction relationships (edges) were given in Fig. 8.
The known interactions were represented as black solid edges. The
predicting DTIs were represented as red edges, where the solid represent these DTIs verified and the dash represent others. For example,
in the E dataset show as Fig. 8-(a), the edge between drug D00449
and target hsa5743 was black solid line, and the pair was known
DTI. The red solid line between drug D00449 and target hsa5742
showed that this pair of DTI has been predicted by proposed model
and verified in databases. This result showed that this prediction has
been proved to be correct by biological experiments. The red dash
line, between drug D00786 and target hsa5743, indicated that the
interaction relationship exists predicted by the proposed model, but
further biological experiments are needed to verify it. In the proposed
model, targets with less known interactions could also be repositioned
to more drugs. For example, the target hsa5742 successfully repositions
to 4 drugs (D00449, D00300, D00448, D00414) in Fig. 8-(a). For the
known targets with many interactions, a large number of drugs also
have been found, such as hsa367, hsa2099 and hsa5241 in Fig. 8-(d).
Among them, the target hsa2099 has known 14 drugs, and through
the proposed model, 5 drugs have been correctly repositioned. Similar
results also appear in other datasets. In summary, among the top 100pair of predicted DTIs, the verified results showed that the proposed



7


_D. Yu et al._



_Expert Systems With Applications 252 (2024) 124289_



**Fig. 8.** Top-100 DTIs network. Light blue rectangles indicate drugs, and yellow ellipses indicate targets. Black solid edges represent known interactions. Red edges show the
predicted top 100 DTIs, where sold edges represent that they are verified in databases and dashed edges represent that they are unverified. The size of the drug (target) node is
proportional to the number of targets that the drug has (the number of drugs targeting the target).



model provides excellent prediction performance. The IC dataset had
the largest amount of verified DTIs, reaching 61 pairs, followed by
GPCR, NR and E datasets with 48, 28 and 21 pairs, respectively.


**5. Discussion**


In this section, we will further discuss the predictive performance
of the DTI-HAN in conjunction with molecular docking. In the top-100
DTIs, except for those verified by 4 databases, top-5 of remaining ones
were selected to further analyze. If the molecular docking results show
that the complex (drug and target) has good binding properties, then
the prediction results of the proposed model have high confidence level.
Therefore, this paper used AutoDock Vina (Eberhardt, Santos-Martins,
Tillack, & Forli, 2021) to analyze the top-5 remaining DTIs.
If the binding affinity value _<_ 0, it indicates that the drug and
the target have the possibility of binding, and the smaller the value,
the greater the possibility of binding. Table 3 represented the binding
affinity results between drug and target of the top-5 remaining DTIs.
It could be seen that the binding affinity values of all DTIs are _<_ −4,
which indicates that the potential interaction predicted by the proposed
model has good binding properties. To a certain extent, these conclusions support the accuracy of the prediction results of the proposed
model.

Further, Fig. 9 showed the conformation diagram of docking results,
as well as Figs. 10 and 11. The left half of the sub-figure is the overall
docking conformation, and the right half is the partial enlargement of
the docking pocket, where the yellow one is the drug, and the blue
one is the residue that interacts with the drug. The dotted and solid
lines of different colors give the interaction relationships and types
in detail, such as hydrophobic, hydrogen bond, _𝜋_ −stacking, _𝜋_ −cation,
salt bridge and halogen bond. From Fig. 9, one or more interactions
occurred in the docking results of each drug and target complex. These
multi-type and multi-group interaction relationships make the binding



**Table 3**

Top-5 of remaining DTIs docking results based on AutoDock Vina (Affinity unit:
kcal/mol)


Drug ID Target ID Affinity Drug ID Target ID Affinity


E GPCR


D03778 hsa1586 −7.369 D01358 hsa3269 −8.350

D03781 hsa1589 −8.308 D00493 hsa1128 −8.711

D00691 hsa5152 −6.637 D00503 hsa1128 −8.502

D00043 hsa5624 −5.360 D01020 hsa154 −6.444

D00131 hsa8854 −5.143 D00503 hsa150 −8.626


IC NR


D00761 hsa1137 −6.801 D00690 hsa2099 −4.047

D00547 hsa8001 −5.343 D00182 hsa2099 −9.485

D00332 hsa6334 – D01132 hsa5468 −8.263

D00617 hsa774 −7.259 D00506 hsa2099 −7.180

D00294 hsa3776 −4.943 D00348 hsa6257 −9.020


complex more stable. In other words, the proposed model predicts the
conclusion that the drug–target pair has interaction relationship, which
is also supported by the molecular docking result.


**6. Conclusion**


In this paper, the problem of predicting DTI in drug discovery and
design is studied, and the issue of false negative samples and further
improving prediction accuracy has been given attention. Therefore,
a heterogeneous network based on drug-drug similarity, target–target
similarity, and drug–target interaction relationship is established. The
heterogeneous graph neural network representation learning method
of the node-level and semantic-level bi-attention mechanism is used to

obtain the embedding latent features of the drugs and the targets, and
finally predict the drug–target interaction. To address the issue of false
negative samples, DTI-HAN has made improvements in two aspects:



8


_D. Yu et al._



_Expert Systems With Applications 252 (2024) 124289_



**Fig. 9.** Docking results of drug and target complex.


9


_D. Yu et al._



_Expert Systems With Applications 252 (2024) 124289_



**Fig. 10.** Docking results of drug and target complex.



first, in the embedding representation stage of drug and target nodes,
the improved loss function is proposed based on only drug-drug and
target–target similarity networks; Second, in the process of estimating
the membership function parameters, only the confidence score of
the positive samples is used. Compared with DTI-GAT, DTI-GCN, DTIGraphSAGE, BLM, DTHybrid, SCMLKNN and FPSC-DTI, the prediction
performance improvement of the proposed DTI-HAN is verified on E,
GPCR, IC, NR 4 datasets. The novel top 100-pair prediction of DTIs
are analyzed by using DTI network and molecular docking, which also
shows that the new interaction prediction results with high confidence
are obtained by the proposed method.
In future research, there are still some points worth exploring in the
DTI-HAN method. For examples, the current DTI-HAN method cannot
process the original features of drugs and targets, but directly inputs
drug-drug and target–target similarities. The current DTI-HAN method
is a relatively independent process in the drugs and targets embedding
representation and prediction stages, whose disadvantage is that it
requires two independent stages for parameter optimization, making
it difficult to obtain a globally optimal prediction model.


**CRediT authorship contribution statement**


**Donghua Yu:** Conceptualization, Methodology, Investigation, Software, Writing – original draft, Writing – review & editing, Funding



acquisition. **Huawen Liu:** Methodology, Writing – review & editing,
Funding acquisition. **Shuang Yao:** Investigation, Writing – original
draft, Writing – review & editing, Funding acquisition.


**Declaration of competing interest**


The authors declare that they have no known competing financial interests or personal relationships that could have appeared to
influence the work reported in this paper.


**Data availability**


I have shared the link of my code and data in the manuscript.


**Acknowledgments**


This work is supported by the National Natural Science Foundation
of China (No. 62002227), the Fundamental Research Funds for the
Provincial Universities of Zhejiang (No. 2021YW57), Major Humanities
and Social Sciences Research Projects in Zhejiang higher education
institutions (No. 2023QN120), and the Humanities and Social Sciences
Youth Project of the Ministry of Education (NO.22YJC630187).



10


_D. Yu et al._


**References**



_Expert Systems With Applications 252 (2024) 124289_



**Fig. 11.** Docking results of drug and target complex.



[Ahmed, F., Soomro, A. M., Salih, A. R. C., Samantasinghar, A., Asif, A., Kang, I. S.,](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb1)
[et al. (2022). A comprehensive review of artificial intelligence and network based](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb1)
[approaches to drug repurposing in Covid-19.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb1) _Biomedicine & Pharmacotherapy_, _153_,

[Article 113350.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb1)


[Alaimo, S., Pulvirenti, A., Giugno, R., & Ferro, A. (2013). Drug-target interaction](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb2)
[prediction through domain-tuned network-based inference.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb2) _Bioinformatics_, _29_ (16),

[2004–2008.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb2)


[Allen, W. J., Balius, T. E., Mukherjee, S., Brozell, S. R., Moustakas, D. T., Lang, P. T.,](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb3)
[et al. (2015). DOCK6: Impact of new features and current docking performance.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb3)
_[Journal of Computational Chemistry](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb3)_, _36_ (15), 1132–1156.


[Bagherian, M., Sabeti, E., Wang, K., Sartor, M. A., Nikolovska-Coleska, Z., & Najarian, K.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb4)
[(2021). Machine learning approaches and databases for prediction of drug–target](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb4)
interaction: a survey paper. _[Briefings in Bioinformatics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb4)_, _22_ (1), 247–269.


[Bleakley, K., & Yamanishi, Y. (2009). Supervised prediction of drug-target interactions](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb5)
using bipartite local models. _[Bioinformatics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb5)_, _25_ (18), 2397–2403.


[Bredel, M., & Jacoby, E. (2004). Chemogenomics: an emerging strategy for rapid target](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb6)
and drug discovery. _[Nature Reviews Genetics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb6)_, _5_ (4), 262–275.


[Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional neural networks](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb7)
[on graphs with fast localized spectral filtering.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb7) _Advances in Neural Information_
_[Processing Systems](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb7)_, _29_ .


[Dehghan, A., Abbasi, K., Razzaghi, P., Banadkuki, H., & Gharaghani, S. (2024). CCL-](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb8)
[DTI: contributing the contrastive loss in drug–target interaction prediction.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb8) _BMC_
_[Bioinformatics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb8)_, _25_, 48.


[Dehghan, A., Razzaghi, P., Abbasi, K., & Gharaghani, S. (2023). TripletMultiDTI:](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb9)
[multimodal representation learning in drug-target interaction prediction with triplet](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb9)
loss function. _[Expert Systems with Applications](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb9)_, _232_, Article 120754.


[Djeddi, W. E., Hermi, K., Ben Yahia, S., & Diallo, G. (2023). Advancing drug–](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb10)
[target interaction prediction: a comprehensive graph-based approach integrating](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb10)
[knowledge graph embedding and ProtBert pretraining.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb10) _BMC Bioinformatics_, _24_, 488.



[Du, Z., Wang, D., & Li, Y. (2022). Comprehensive evaluation and comparison of](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb11)
[machine learning methods in QSAR modeling of antioxidant tripeptides.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb11) _ACS_
_Omega_, _7_ [(29), 25760–25771.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb11)
[Eberhardt, J., Santos-Martins, D., Tillack, A. F., & Forli, S. (2021). AutoDock Vina](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb12)
[1.2.0: new docking methods, expanded force field, and python bindings.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb12) _Journal of_
_[Chemical Information and Modeling](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb12)_, _61_ (8), 3891–3898.
[Elbasani, E., Njimbouom, S. N., Oh, T.-J., Kim, E.-H., Lee, H., & Kim, J.-D. (2021).](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb13)
[GCRNN: graph convolutional recurrent neural network for compound–protein](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb13)
interaction prediction. _[BMC Bioinformatics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb13)_, _22_ (5), 1–14.
[Feng, Y., You, H., Zhang, Z., Ji, R., & Gao, Y. (2019). Hypergraph neural networks. In](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb14)
B. Williams, Y. Chen, & J. Neville (Eds.), _[Vol. 33](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb14)_, _Proceedings of the AAAI conference_
_[on artificial intelligence](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb14)_ (pp. 3558–3565).
[Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb15)
large graphs. _[Advances in Neural Information Processing Systems](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb15)_, _30_ .
[Hopkins, A. L., Keserü, G. M., Leeson, P. D., Rees, D. C., & Reynolds, C. H. (2014). The](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb16)
[role of ligand efficiency metrics in drug discovery.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb16) _Nature Reviews Drug Discovery_,
_13_ [(2), 105–121.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb16)
[Jin, Y., Lu, J., Shi, R., & Yang, Y. (2021). EmbedDTI: Enhancing the molecular](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb17)
[representations via sequence embedding and graph convolutional network for the](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb17)
[prediction of drug-target interaction.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb17) _Biomolecules_, _11_ (12), 1783.
[Li, M., Cai, X., Xu, S., & Ji, H. (2023). Metapath-aggregated heterogeneous graph neural](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb18)
[network for drug–target interaction prediction.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb18) _Briefings in Bioinformatics_, _24_ (1),

[bbac578.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb18)
[Li, Q., Cao, Z., Tanveer, M., Pandey, H. M., & Wang, C. (2019). A semantic collaboration](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb19)
[method based on uniform knowledge graph.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb19) _IEEE Internet of Things Journal_, _7_ (5),

[4473–4484.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb19)
[Li, Q., Cao, Z., Zhong, J., & Li, Q. (2019). Graph representation learning with encoding](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb20)
edges. _[Neurocomputing](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb20)_, _361_, 29–39.
[Liu, Z., Chen, Q., Lan, W., Lu, H., & Zhang, S. (2024a). SSLDTI: A novel method](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb21)
[for drug-target interaction prediction based on self-supervised learning.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb21) _Artificial_
_Intelligence in Medicine_, _149_, Article 102778.
[Liu, J., Guan, S., Zou, Q., Wu, H., Tiwari, P., & Ding, Y. (2024b). AMDGT: Atten-](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb22)
[tion aware multi-modal fusion using a dual graph transformer for drug–disease](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb22)
associations prediction. _[Knowledge-Based Systems](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb22)_, _284_, Article 111329.



11


_D. Yu et al._


[Liu, B., Pliakos, K., Vens, C., & Tsoumakas, G. (2021a). Drug-target interaction](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb23)

[prediction via an ensemble of weighted nearest neighbors with interaction recovery.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb23)
_[Applied Intelligence: The International Journal of Artificial Intelligence, Neural Networks,](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb23)_
_[and Complex Problem-Solving Technologies](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb23)_, 3705–3727.
[Liu, L., Yao, S., Ding, Z., Guo, M., Yu, D., & Hu, K. (2021b). Drug-target interaction](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb24)

[prediction based on Gaussian interaction profile and information entropy. In](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb24)
[Y. Wei, M. Li, P. Skums, & Z. Cai (Eds.),](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb24) _Vol. 13064_, _International symposium on_
_[bioinformatics research and applications](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb24)_ (pp. 388–399). Cham: Springer.
[Luo, Y., Zhao, X., Zhou, J., Yang, J., Zhang, Y., Kuang, W., et al. (2017). A network](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb25)

[integration approach for drug-target interaction prediction and computational drug](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb25)
[repositioning from heterogeneous information.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb25) _Nature Communications_, _8_ (1), 1–13.
[Meng, Y., Wang, Y., Xu, J., Lu, C., Tang, X., Peng, T., et al. (2024). Drug repositioning](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb26)

[based on weighted local information augmented graph neural network.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb26) _Briefings in_
_Bioinformatics_, _[25](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb26)_ (1), bbad431.
[Morris, G. M., Huey, R., Lindstrom, W., Sanner, M. F., Belew, R. K., Goodsell, D. S.,](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb27)

[et al. (2009). AutoDock4 and AutoDockTools4: Automated docking with selective](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb27)
receptor flexibility. _[Journal of Computational Chemistry](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb27)_, _30_ (16), 2785–2791.
[Pang, S., Zhang, K., Wang, S., Zhang, Y., He, S., Wu, W., et al. (2021). HGDD: A drug-](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb28)

[disease high-order association information extraction method for drug repurposing](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb28)
[via hypergraph. In Y. Wei, M. Li, P. Skums, Z. Cai (Eds.),](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb28) _International symposium_
_[on bioinformatics research and applications](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb28)_ (pp. 424–435).
[Parvizi, P., Azuaje, F., Theodoratou, E., & Luz, S. (2020). A network-based embedding](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb29)

[method for drug-target interaction prediction. In E. Sacristan (Ed.),](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb29) _2020 42nd_
_[annual international conference of the IEEE engineering in medicine & biology society](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb29)_
_(EMBC)_ [(pp. 5304–5307). Montreal, QC, Canada.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb29)
[Paul, S. M., Mytelka, D. S., Dunwiddie, C. T., Persinger, C. C., Munos, B. H., Lindborg, S.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb30)

[R., et al. (2010). How to improve R&D productivity: the pharmaceutical industry’s](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb30)
grand challenge. _[Nature Reviews Drug Discovery](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb30)_, _9_ (3), 203–214.
[Peng, J., Wang, Y., Guan, J., Li, J., Han, R., Hao, J., et al. (2021). An end-to-](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb31)

[end heterogeneous graph representation learning-based framework for drug–target](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb31)
interaction prediction. _[Briefings in Bioinformatics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb31)_, _22_ (5), bbaa430.
[Rafiei, F., Zeraati, H., Abbasi, K., Razzaghi, P., Ghasemi, J. B., Parsaeian, M., et](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb32)

[al. (2024). CFSSynergy: combining feature-based and similarity-based methods](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb32)
for drug synergy prediction. _[Journal of Chemical Information and Modeling](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb32)_, _64_ (7),

[2577–2585.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb32)

[Ruan, D., Ji, S., Yan, C., Zhu, J., Zhao, X., Yang, Y., et al. (2021). Exploring complex](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb33)

[and heterogeneous correlations on hypergraph for the prediction of drug-target](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb33)
interactions. _Patterns_, _[2](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb33)_ (12), Article 100390.
[Santana Azevedo, L., Pretto Moraes, F., Morrone Xavier, M., Ozorio Pantoja, E.,](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb34)

[Villavicencio, B., Aline Finck, J., et al. (2012). Recent progress of molecular docking](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb34)
[simulations applied to development of drugs.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb34) _Current Bioinformatics_, _7_ (4), 352–365.
[Schrynemackers, M., Küffner, R., & Geurts, P. (2013). On protocols and measures for the](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb35)

[validation of supervised methods for the inference of biological networks.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb35) _Frontiers_
_[in Genetics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb35)_, _4_, 262.
[Shao, K., Zhang, Y., Wen, Y., Zhang, Z., He, S., & Bo, X. (2022). DTI-HETA: prediction of](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb36)

[drug–target interactions based on GCN and GAT on heterogeneous graph.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb36) _Briefings_
_in Bioinformatics_, _[23](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb36)_ (3), bbac109.
[Shi, W., Yang, H., Xie, L., Yin, X.-X., & Zhang, Y. (2024). A review of machine learning-](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb37)

[based methods for predicting drug–target interactions.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb37) _Health Information Science_
_[and Systems](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb37)_, _12_, 30.
[Shi, J.-Y., Yiu, S.-M., Li, Y., Leung, H. C., & Chin, F. Y. (2015). Predicting drug-target](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb38)

[interaction for new drugs using enhanced similarity measures and super-target](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb38)
clustering. _[Methods](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb38)_, _83_, 98–104.
[Sivangi, K. B., Amilpur, S., & Dasari, C. M. (2023). ReGen-DTI: A novel generative](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb39)

[drug target interaction model for predicting potential drug candidates against](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb39)
SARS-COV2. _[Computational Biology and Chemistry](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb39)_, _106_, Article 107927.
[Su, Y., Hu, Z., Wang, F., Bin, Y., Zheng, C., Li, H., et al. (2024). AMGDTI: drug–target](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb40)

[interaction prediction based on adaptive meta-graph learning in heterogeneous](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb40)
network. _[Briefings in Bioinformatics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb40)_, _25_ (1), bbad474.



_Expert Systems With Applications 252 (2024) 124289_


[Su, X., Hu, P., Yi, H., You, Z., & Hu, L. (2022). Predicting drug-target interactions](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb41)

[over heterogeneous information network.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb41) _IEEE Journal of Biomedical and Health_
_Informatics_, _[27](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb41)_ (1), 562–572.
[Sun, Y., Han, J., Yan, X., Yu, P. S., & Wu, T. (2011). PathSim: meta path-based top-k](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb42)

[similarity search in heterogeneous information networks.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb42) _Proceedings of the VLDB_

_Endowment_, _4_ [(11), 992–1003.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb42)

[Sydow, D., Burggraaff, L., Szengel, A., van Vlijmen, H. W., IJzerman, A. P., van](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb43)

[Westen, G. J., et al. (2019). Advances and challenges in computational target](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb43)
prediction. _[Journal of Chemical Information and Modeling](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb43)_, _59_ (5), 1728–1742.
[Thafar, M. A., Albaradie, S., Olayan, R. S., Ashoor, H., Essack, M., & Bajic, V. B. (2020).](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb44)

[Computational drug-target interaction prediction based on graph embedding and](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb44)
[graph mining. In T. Akutsu, & W.-K. Sung (Eds.),](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb44) _Proceedings of the 2020 10th_
_[international conference on bioscience, biochemistry and bioinformatics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb44)_ (pp. 14–21).

[Kyoto Japan.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb44)
[Tian, Z., Peng, X., Fang, H., Zhang, W., Dai, Q., & Ye, Y. (2022). MHADTI: predicting](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb45)

[drug–target interactions via multiview heterogeneous information network em-](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb45)
[bedding with hierarchical attention mechanisms.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb45) _Briefings in Bioinformatics_, _23_ (6),

[bbac434.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb45)

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).

[Graph attention networks. arXiv:1710.10903, [cs, stat].](http://arxiv.org/abs/1710.10903)
[Wan, F., Hong, L., Xiao, A., Jiang, T., & Zeng, J. (2019). NeoDTI: neural integration](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb47)

[of neighbor information from a heterogeneous network for discovering new](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb47)
[drug–target interactions. In J. Wren (Ed.),](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb47) _Bioinformatics_, _35_ (1), 104–111.
Wang, X., Ji, H., Shi, C., Wang, B., Cui, P., Yu, P., et al. (2021). Heterogeneous graph


[attention network. arXiv:1903.07293, [cs].](http://arxiv.org/abs/1903.07293)

[Wen, M., Zhang, Z., Niu, S., Sha, H., Yang, R., Yun, Y., et al. (2017). Deep-learning-](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb49)

[based drug-target interaction prediction.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb49) _Journal of Proteome Research_, _16_ (4),

[1401–1409.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb49)

[Xuan, P., Chen, B., Zhang, T., et al. (2020). Prediction of drug-target interactions based](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb50)

[on network representation learning and ensemble learning.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb50) _IEEE/ACM Transactions_
_[on Computational Biology and Bioinformatics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb50)_ .
[Yamanishi, Y., Araki, M., Gutteridge, A., Honda, W., & Kanehisa, M. (2008). Prediction](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb51)

[of drug-target interaction networks from the integration of chemical and genomic](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb51)
spaces. _[Bioinformatics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb51)_, _24_ (13), i232–i240.
[Yamanishi, Y., Kotera, M., Kanehisa, M., & Goto, S. (2010). Drug-target interaction](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb52)

[prediction from chemical, genomic and pharmacological data in an integrated](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb52)
framework. _[Bioinformatics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb52)_, _26_ (12), i246–i254.
[Yu, D., Liu, G., Zhao, N., Liu, X., & Guo, M. (2020). FPSC-DTI: drug–target interaction](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb53)

[prediction based on feature projection fuzzy classification and super cluster fusion.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb53)

_Molecular Omics_, _[16](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb53)_ (6), 583–591.

[Yue, Y., & He, S. (2021). DTI-HeNE: a novel method for drug-target interaction](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb54)

[prediction based on heterogeneous network embedding.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb54) _BMC Bioinformatics_, _22_ (1),

[418.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb54)

[Zeng, X., Zhu, S., Lu, W., Liu, Z., Huang, J., Zhou, Y., et al. (2020). Target identification](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb55)

[among known drugs by deep learning from heterogeneous networks.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb55) _Chemical_

_Science_, _11_ [(7), 1775–1797.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb55)

[Zhang, R., Wang, Z., Wang, X., Meng, Z., & Cui, W. (2023). MHTAN-DTI: Metapath-](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb56)

[based hierarchical transformer and attention network for drug–target interaction](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb56)
prediction. _[Briefings in Bioinformatics](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb56)_, _24_ (2), bbad079.
[Zhao, X., Zhao, X., & Yin, M. (2022). Heterogeneous graph attention network based](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb57)

[on meta-paths for lncrna–disease association prediction.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb57) _Briefings in Bioinformatics_,

_23_ [(1), bbab407.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb57)

[Zong, N., Wong, R. S. N., Yu, Y., Wen, A., Huang, M., & Li, N. (2021). Drug-](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb58)

[target prediction utilizing heterogeneous bio-linked network embeddings.](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb58) _Briefings_
_in Bioinformatics_, _[22](http://refhub.elsevier.com/S0957-4174(24)01155-2/sb58)_ (1), 568–580.



12


