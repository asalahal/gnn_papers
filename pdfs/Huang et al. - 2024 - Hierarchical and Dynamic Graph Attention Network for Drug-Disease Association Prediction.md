2416 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 28, NO. 4, APRIL 2024

## Hierarchical and Dynamic Graph Attention Network for Drug-Disease Association Prediction


Shuhan Huang, Minhui Wang, Xiao Zheng, Jiajia Chen, and Chang Tang _, Senior Member, IEEE_



_**Abstract**_ **—In the realm of biomedicine, the prediction**
**of associations between drugs and diseases holds signif-**
**icant importance. Yet, conventional wet lab experiments**
**often fall short of meeting the stringent demands for pre-**
**diction accuracy and efficiency. Many prior studies have**
**predominantly focused on drug and disease similarities**
**to predict drug-disease associations, but overlooking the**
**crucial interactions between drugs and diseases that are**
**essential for enhancing prediction accuracy. Hence, in**
**this paper, a resilient and effective model named Hier-**
**archical and Dynamic Graph Attention Network (HDGAT)**
**has been proposed to predict drug-disease associations.**
**Firstly, it establishes a heterogeneous graph by leveraging**
**the interplay of drug and disease similarities and associa-**
**tions. Subsequently, it harnesses the capabilities of graph**
**convolutional networks and bidirectional long short-term**
**memory networks (Bi-LSTM) to aggregate node-level infor-**
**mation within the heterogeneous graph comprehensively.**
**Furthermore, it incorporates a hierarchical attention mech-**
**anism between convolutional layers and a dynamic at-**
**tention mechanism between nodes to learn embeddings**
**for drugs and diseases. The hierarchical attention mecha-**
**nism assigns varying weights to embeddings learned from**
**different convolutional layers, and the dynamic attention**
**mechanism efficiently prioritizes inter-node information by**
**allocating each node with varying rankings of attention co-**
**efficients for neighbour nodes. Moreover, it employs resid-**
**ual connections to alleviate the over-smoothing issue in**
**graph convolution operations. The latent drug-disease as-**
**sociations are quantified through the fusion of these em-**
**beddings ultimately. By conducting 5-fold cross-validation,**


Manuscript received 5 October 2023; revised 19 December 2023, 24
January 2024, and 28 January 2024; accepted 2 February 2024. Date
of publication 6 February 2024; date of current version 5 April 2024.
This work was supported by the National Natural Science Foundation of
China under Grant 62076228. _(Corresponding authors: Jiajia Chen and_
_Chang Tang.)_

Shuhan Huang and Chang Tang are with the School of Computer
Science, China University of Geosciences, Wuhan 430074, China
[(e-mail: huangshuhan@cug.edu.cn; tangchang@cug.edu.cn).](mailto:huangshuhan@cug.edu.cn)

Minhui Wang is with the Department of Pharmacy, Lianshui People’s
Hospital of Kangda College Affiliated to Medical University, Huai’an
[223300, China (e-mail: minhwang@163.com).](mailto:minhwang@163.com)

Xiao Zheng is with the School of Computer, National University
[of Defense Technology, Changsha 410073, China (e-mail: zhengx-](mailto:zhengxiao@nudt.edu.cn)
[iao@nudt.edu.cn).](mailto:zhengxiao@nudt.edu.cn)

Jiajia Chen is with the Department of Pharmacy, The Affiliated Huai’an
Hospital of Xuzhou Medical University, Xuzhou 221006, China, and also
with The Second People’s Hospital of Huai’an, Huai’an 223002, China
[(e-mail: jjiachen@outlook.com).](mailto:jjiachen@outlook.com)

Digital Object Identifier 10.1109/JBHI.2024.3363080



**HDGAT’s performance surpasses the performance of exist-**
**ing state-of-the-art models across various evaluation met-**
**rics, which substantiates the exceptional efficacy of HDGAT**
**in predicting drug-disease associations.**


_**Index**_ _**Terms**_ **—Drug-disease** **association** **prediction,**
**graph attention network, hierarchical attention mechanism,**
**bidirectional long short-term memory networks, residual**
**connections.**


I. I NTRODUCTION


HE analysis of drug-disease association prediction holds
# T paramount importance in the domain of biomedicine [1],

[2]. Its outcomes frequently wield substantial influence over
the decisions made by medical researchers during clinical trials, ultimately shaping the timely and effective alleviation and resolution of human diseases through pertinent
drugs.

In the realm of drug-disease association prediction, conventional wet lab experiments are marred by inefficiency, time
consumption, and often yield suboptimal accuracy. In response
to the stringent accuracy requirements in medical research, a
surge of researchers are now delving into the formulation of
efficient computational techniques to surmount the challenge of
low accuracy in experimental findings.

The prevailing algorithms for forecasting drug-disease associations can be broadly classified into four categories: 1. Literature
extraction-based methods; 2. Similarity-based methods; 3. Network fusion-based methods; 4. Deep learning-based methods.

Literature extraction-based methods commonly leverage natural language processing techniques to extract drug-disease
associations from extensive biomedical literature. The gleaned
information is subsequently harnessed to construct pertinent
feature representations and train models using suitable algorithms for forecasting drug-disease associations. For instance,
Karaa et al. [3] introduced a method employing natural language
processing and UMLS(Unified Medical Language System) ontology, coupled with a support vector machine classifier, to extract semantic relationships between drugs and diseases, leading
to substantial enhancements in performance. Besides, Wang
et al. [4] employed pattern matching and network embedding
algorithms to autonomously extract extensive and precise drugdisease pairs from medical literature, providing vital support
for drug repurposing endeavors. Although literature extractionbased methods can leverage the wealth of medical literature
to extract drug-disease associations with commendable interpretability,theirpredictiveefficacyisconsiderablyinfluencedby



2168-2194 © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.

See https://www.ieee.org/publications/rights/index.html for more information.


Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:33:11 UTC from IEEE Xplore. Restrictions apply.


HUANG et al.: HIERARCHICAL AND DYNAMIC GRAPH ATTENTION NETWORK FOR DRUG-DISEASE ASSOCIATION PREDICTION 2417



the literature’s quality and encounters various challenges during
the information extraction process.

Similarity-based methods commonly establish a bipartite
graph encompassing drugs and diseases. These methods subsequently employ similarity measurement techniques to evaluate
the likeness between drugs and diseases, in conjunction with
the structural attributes of the bipartite graph. Afterwards, they
prognosticate latent drug-disease associations via graph algorithms operating on the similarity-based graph. For instance,
Zhang et al. [5] introduced a novel computational approach
that exclusively utilizes established drug-disease associations
to foresee unobserved ones. This method employs linear neighbour similarities to compute the similarities between drugs and
diseases, and subsequently employs a label propagation process
to predict latent drug-disease associations within the similaritybased graph. Similarly, Di et al. [6] devised a novel approach
for amalgamating drug-disease associations, drug and chemical
data, drug target domain information, and target annotation
data to facilitate drug repositioning. They introduced interaction
profiles of drugs and diseases within a network, treating them as
label information for training models to predict new candidates.
Similarity-based methods adeptly account for the network’s
structural characteristics connecting drugs and diseases, thereby
enhancingthecomprehensionoftheirassociations.Nonetheless,
in cases where drug-disease associations are sparse, these methods may encounter prediction challenges and struggle to unearth
potential associations.

Network fusion-based methods commence by amalgamating
data regarding drug and disease associations and generating
a heterogeneous graph. Subsequently, network diffusion algorithms are employed to extract features from diffused nodes,
and suitable machine learning models are utilized for training, culminating in performance evaluation. In recent studies,
many researchers have employed neural networks based on
heterogeneous graphs[7], [8], [9], [10], [11] for tasks such as
Drug-Drug Interaction (DDI) and Drug-Target Interaction (DTI)
predictions. For instance, Tanvir et al. [8] adopted HAN-DDI,
a heterogeneous graph attention network to predict drug-drug
interactions in an end-to-end mode. Peng et al. [11] developed the (EEG)-DTI model, which constructs a heterogeneous
graph with different nodes representing drug-protein interactions, drug-disease interactions, and more. The model adopts
GCNs to aggregate node information and make DTI predictions in an end-to-end mode. Despite the capability of network
fusion-based methods to comprehensively consider the network
structural attributes between drugs and diseases and capture their
associations, these methods might overlook latent associations
due to their heavy reliance on network propagation, potentially
yielding incomplete prediction outcomes.

Deep learning methods are frequently employed across diverse practical tasks owing to their efficiency and high accuracy,
including pattern recognition and object detection [12]. Similarly, these methods find application in drug-disease association
prediction. Within this domain, deep learning methods generally
encompass data preprocessing and feature extraction, followed
by the selection of suitable deep learning models for trainingsuch as CNNs, RNNs, GCNs, and more. Ultimately, the model
undergoes evaluation, and hyperparameters are optimized. For
example, Liu et al. [13] proposed a technique leveraging a deep
neural network based on a heterogeneous drug-disease network
to predict novel drug-disease associations, which involves constructing drug-drug and disease-disease similarity networks, and



then integrating established drug-disease associations to extract
topological features fromtheheterogeneous networkfor training
the DNN model. Meanwhile, Xuan et al. [14] proposed a creative
method for predicting drug and disease association by utilizing graph convolutional and fully-connected autoencoders with
attention mechanisms to integrate drug-disease associations,
disease similarities, multiple drug similarities, and drug node
attributes. Although deep learning methods excel in numerous
tasks, their interpretability often remains limited.

In this study, we present a novel hierarchical and dynamic
graph attention network for the prediction of drug-disease associations. Firstly, we construct a heterogeneous graph utilizing
drug-drug similarities, disease-disease similarities, and drugdisease associations. Secondly, we pioneer the incorporation
of graph convolutional neural networks and bidirectional long
short-term memory networks to enhance the aggregation of
both node-specific and structural information. Thirdly, a hierarchical attention mechanism [15] and residual connections are
innovatively integrated across diverse network levels, while the
introductionofdynamicattentionmechanismswithineachlayer.
The feature embeddings of drug and disease are finally obtained
through these processes. Ultimately, the model’s performance
is assessed through the evaluation of undetected drug-disease
associations, facilitated by the fusion of embeddings. Significantly, HDGAT surpasses other baseline models, achieving an
impressive area under the precision-recall curve of 0.2665 and
a remarkable prediction accuracy of 0.9614 on main dataset.

Fig. 1 shows the whole workflow and details of HDGAT
model. In the future, drug-disease association prediction will
be widely applied to optimize clinical trials, and its results can
scientifically guide doctors in devising more suitable treatment
plans for patients. Moreover, in future research, employing
large language models to extract drug-disease associations from
literature data may also be a promising approach, which utilizes
increased computational power to capture potential drug-disease
associations in literature, offering a straightforward and efficient approach. Similarly, integrating further exploration of
heterogeneous associations between drugs and diseases will be
a future research direction, aiming to delve more profoundly
into the fundamental and intricate connections between drugs
and diseases. In the context of HDGAT, its application extends
to drug repositioning, disease mechanism analysis, and related
domains, therebyenhancingtheefficiencyinidentifyingsuitable
drugs for the treatment of diseases.


II. T HE P ROPOSED M ODEL


_A. Graph Preprocessing_


To capture the intricate interplay between drugs and diseases
more effectively, we opted to create a heterogeneous graph
encompassing drug-drug similarities, disease-disease similarities, and drug-disease associations. This heterogeneous graph
encapsulates nodes and edges that exhibit a multitude of connections, thus enabling the propagation of node information and
the acquisition of embeddings via graph convolution operations.
As a result, it aids in amplifying the efficacy of the model. Thus,
the construction of the heterogeneous graph emerges as a pivotal
facet in the holistic advancement of the model [16], [17].

_1) Calculations of Drug-Drug Similarities:_ To begin, we establish and elucidate the drug-drug similarity data. Due to the
diverse features present in drugs, we employ distinct sets of



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:33:11 UTC from IEEE Xplore. Restrictions apply.


2418 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 28, NO. 4, APRIL 2024


Fig. 1. Whole workflow and details of HDGAT: It contains three components: 1. Heterogeneous Network Construction: It incorporates drugdrug similarities, disease-disease similarities and drug-disease associations to construct a heterogeneous network; 2. Encoder for Embedding
Learning: The heterogeneous network, as input graph, is processed by Encoder to learn drug and disease embeddings; 3. Adjacency Matrix
Reconstruction: The Decoder reconstruct the adjacency matrix containing drug and disease associations based on drug embeddings and disease
embeddings.



drug features to represent different categories of drugs. Utilizing
existing data on diverse drug features, we analyze all possible
features that a specific drug may have. If the drug possesses a
particular feature, we set the corresponding value in the vector
to 1; otherwise, it is set to 0. This approach enables us to obtain
the binary feature vector of a drug. Besides, distinct binary
feature vectors serve to distinguish various drugs, aiding in
the representation of diverse drug entities. The collection of
different drug features forms the feature matrix for that drug.
Subsequently, various methods of similarity calculation can be
employed to quantify the drug-drug similarity based on this
feature matrix. Prominent techniques encompass the Jaccard
index [18], cosine similarity [19], Euclidean distance [20], Manhattan distance [21], and other comparable approaches. Then we
use a two-dimensional matrix, denoted as _M_, to represent the
matrix of drug-drug similarities, where each row and column
correspond to different types of drugs. The value _M_ _ij_ _[r]_ [represents]

the similarity between drug _r_ _i_ and drug _r_ _j_ .

The computation approach for evaluating drug-drug similarities using the Jaccard index is outlined as follows:



The calculation method for cosine similarity in assessing
drug-drug similarities is as follows:


_x_ _i_ _· x_ _j_

**M** **[r]** **ij** [=] (2)

_∥x_ _i_ _∥· ∥x_ _j_ _∥_


where _x_ _i_ and _x_ _j_ represent drug feature vectors, _∥x_ _i_ _∥_ and _∥x_ _j_ _∥_
denote the L2-norm of _x_ _i_ and _x_ _j_, respectively.

The calculation method for Euclidean distance in assessing
drug-drug similarities is as follows:



**M** **[r]** **ij** [=]



_n_
� ~~�~~ _k_ =1 [(] _[x]_ _[i,k]_ _[ −]_ _[x]_ _[j,k]_ [)] [2] (3)



where _x_ _i,k_ and _x_ _j,k_ represent the values of the _k_ _[th]_ dimension in
the feature vectors _x_ _i_ and _x_ _j_, respectively.

The calculation method for Manhattan Distance in assessing
drug-drug similarities is as follows:



**M** **[r]** **ij** [=]



_n_
�


_k_ =1



_|x_ _i,k_ _−_ _x_ _j,k_ _|_ (4)




**[r]** **ij** [=] _[|][x]_ _[i]_ _[ ∩]_ _[x]_ _[j]_ _[|]_



**M** **[r]**



(1)
_|x_ _i_ _∪_ _x_ _j_ _|_



where _x_ _i,k_ and _x_ _j,k_ represent the values of the _k_ _[th]_ dimension in
the feature vectors _x_ _i_ and _x_ _j_ respectively.

Comparing the advantages and disadvantages of the four
methods mentioned above, we have found that the evaluation
method of Jaccard index is not influenced by feature scales.
Unlike several other methods that employ distance measures,
Jaccard index disregards the numerical values of elements. This
insensitivity to numerical differences is advantageous when
dealing with potential variations between different drugs and
diseases since the Jaccard index focuses exclusively emphasizes



where _|x_ _i_ _∩_ _x_ _j_ _|_ signifies the count of situations where elements
in drug feature vector _x_ _i_ and corresponding elements in feature
vector _x_ _j_ are both equal to 1, while _|x_ _i_ _∪_ _x_ _j_ _|_ denotes the
count of situations where elements in drug feature vector _x_ _i_
or corresponding elements in feature vector _x_ _j_ are equal to 1.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:33:11 UTC from IEEE Xplore. Restrictions apply.


HUANG et al.: HIERARCHICAL AND DYNAMIC GRAPH ATTENTION NETWORK FOR DRUG-DISEASE ASSOCIATION PREDICTION 2419


the presence or absence of elements. As the Jaccard index
method is more suitable in this task, we adopt it to calculate
drug-drug similarities.



_2) Calculations of Disease-Disease Similarities:_ From the
description in [22], we can infer that the MeSH (Medical Subject
Headings) database can be utilized for naming diseases. Additionally, disease-disease similarities can be represented using
directed acyclic graphs (DAGs). Within these DAGs, a disease A
can be depicted in the form: _DAG_ _A_ = ( _S, D_ _A_ _, E_ _A_ ), where _D_ _A_
represents the set of nodes including node A and all its ancestor
nodes,and _E_ _A_ isthecollectionofdirectlinksbetweenparentand
child nodes. Based on the aforementioned DAG structure, we
define _W_ _A_ ( _d_ ) as the semantic contribution of a specific disease
_d_ to disease A within _DAG_ _A_, and its calculation method is as
follows:


_W_ _A_ ( _d_ )



Fig. 2. Module of Encoder: It employs a hierarchical and dynamic
graph attention network, coupled with Bi-LSTM mechanisms and residual connections, to process input data, capturing latent interactions between nodes and the structural information of a heterogeneous network.


similarity matrices, **S** **[˜]** **[M]** and **S** **[˜]** **[A]**, as follows:



=



1 _,_ if _d_ = _A_ (5)
�max _{_ Δ _· W_ _A_ ( _d_ _[′]_ ) _| d_ _[′]_ _∈_ children of _d},_ if _d ̸_ = _A_



_ij_ _[M]_ [(] **[D]** **[M]** [)] _[−]_ **2** **[1]**




_[A]_ **2**

_ij_ [(] **[D]** **[A]** [)] _[−]_ **[1]**



**2** (9)



where Δ is a parameter representing the semantic contribution
value of the edge link between disease _d_ and its child disease _d_ _[′]_,
ranging from 0 to 1. Based on previous experimental results, we
set it to 0.5 here. Utilizing (5), we define the semantic value of
disease A as _DV_ ( _A_ ):



**˜S** **[M]** = ( **D** **M** ) _[−]_ **2** **[1]**


**˜S** **[A]** = ( **D** **A** ) _[−]_ **2** **[1]**



_j_ _[S]_ _ij_ _[M]_



**2** _S_ _[M]_



**2** _S_ _[A]_



**2** (10)



_j_ _[S]_ _ij_ _[A]_



where **D** **M** = diag( [�]



_ij_ _[M]_ [)][,] **[ D]** **[A]** [ =][ diag][(][�]



_DV_ ( _A_ ) =



�

_d∈D_ _A_



_W_ _A_ ( _d_ ) (6)



where **D** **M** = diag( [�] _j_ _[S]_ _ij_ _[M]_ [)][,] **[ D]** **[A]** [ =][ diag][(][�] _j_ _[S]_ _ij_ [)][. Based on]

these works and previous learning experiences [23], we construct the drug-disease heterogeneous graph:



�



**˜S** **M** **R**
� **R** **[T]** **˜S** **[A]**



(11)



When measuring the semantic similarities between two diseases,
_A_ _i_ and _A_ _j_, we take their positional relationship within the DAG
into consideration. We hypothesize that having a greater number
of common ancestors in the DAG tends to imply higher semantic
similarities. We define the semantic similarities value between
_A_ _i_ and _A_ _j_ as _S_ _ij_ _[A]_ [. Therefore, we derive the following calculation]

formula:



**H** =



Finally, in order to control the contribution of drug and disease
similarities during the subsequent graph convolution propagation, we introduce a penalty factor _λ_ . As a result, we obtain the
ultimate input graph:



�



**G** =



_λ ·_ **S** **[˜]** **[M]** **R**

� **R** **[T]** _λ ·_ **S** **[˜]** **[A]**



(12)



_S_ _ij_ _[A]_ [=]



�



_d∈D_ _Ai_ _∩D_ _Aj_ [(] _[W]_ _[A]_ _i_ [(] _[d]_ [) +] _[ W]_ _[A]_ _j_ [(] _[d]_ [))]

(7)
_DV_ ( _A_ _i_ ) + _DV_ ( _A_ _j_ )



_d∈D_ _Ai_ _∩D_ _Aj_ [(] _[W]_ _[A]_ _i_ [(] _[d]_ [) +] _[ W]_ _[A]_ _j_ [(] _[d]_ [))]



Equation (7) illustratesthesemanticrelationshipamongdiseases
_A_ _i_, _A_ _j_ and their ancestor diseases.

_3) Drug-Disease_ _Associations:_ Drug-disease association
data is a two-dimensional matrix denoted as **R**, with dimensions
_N ∗_ _M_, where _N_ represents the number of drug types and _M_
represents the number of disease types. The element _R_ _ij_ in the
matrix corresponds to the value at the intersection of the _i_ _[th]_
row and _j_ _[th]_ column [13]. This value signifies the association
between the _i_ _[th]_ drug type and the _j_ _[th]_ disease type. The values
are as follows:



_B. Encoder_


Fig. 2 illustrates the encoder’s processing procedure. It introduces a hierarchical mechanism among embeddings acquired
from each layer and a dynamic attention mechanism among
nodes. Furthermore, a Bi-LSTM module and a residual connection module are utilized to integrate node and structural information within the heterogeneous network. Detailed descriptions
of these modules are provided below.

_1) Graph Convolutional Network:_ In the prediction of associations between drugs and diseases, graph convolutional
networks excel in learning low-dimensional node information.
They efficiently aggregate information from neighbouring nodes
aroundthecentralnodeandpropagateit,therebycapturinggraph
structuralfeatureseffectively,whichconfersanadvantageinlink
prediction tasks.

Our input graph is a sparse two-dimensional matrix with
a shape of ( _N_ + _M_ )*( _N_ + _M_ ), where _N_ and _M_ represent the
numbers of drug and disease, respectively. Due to its sparsity,
we employ graph convolutional networks, which is suitable for
sparse graphs, to aggregate neighbour node information and
the topological structure of heterogeneous graphs. The propagation mechanism for each layer in the graph convolutional



**R** **ij** =



1 _,_ if drugs _i_ is associated with disease _j_
(8)
�0 _,_ if drugs _i_ is not associated with disease _j_



_4) Construction of Heterogeneous Graph:_ The drug-disease
heterogeneous graph is constructed based on theses three parts
mentioned above.

The pairwise similarities among the N types of drug are
represented using the similarity matrix **S** **[M]** _∈_ R _[N]_ _[∗][N]_, while
the pairwise similarities among the M types of disease are
represented using the similarity matrix **S** **[A]** _∈_ R _[M]_ _[∗][M]_ . Here,
_S_ _ij_ _[M]_ [and] _[ S]_ _ij_ _[A]_ [denote the elements at the indices][ (] _[i, j]_ [)][ of these]

two matrices respectively. Furthermore, we normalize the two



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:33:11 UTC from IEEE Xplore. Restrictions apply.


2420 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 28, NO. 4, APRIL 2024



network [24] is computed as follows:




**[1]**

**2** **H** [(] **[l]** [)] **W** [(] **[l]** [)] [�]



(13)



**H** [(] **[l]** [+] **[1]** [)] = _σ_



�



**D** _[−]_ **2** **[1]**




**[1]**

**2** **GD** _[−]_ **2** **[1]**



resilience. Drawing upon the dynamic attention mechanism elucidated earlier, the weights attributed to diverse neighbour nodes
are computed. Following this, the central node assimilates information from its neighbours contingent on these varied weights.
Subsequent to a comprehensive normalization operation, the
node’s output feature vector is generated by amalgamating the
node’s associated features with attention coefficients and subjecting them to an activation function, as explicated in (17):



where **H** [(] **[l]** [+] **[1]** [)] and **H** [(] **[l]** [)] represent the node embeddings of the
( _l_ + 1) _[th]_ and _l_ _[th]_ layers respectively, D is the degree matrix of
graph **G**, defined as **D** = diag( [�] _j_ _[G]_ _[ij]_ [)] _[,]_ **[ W]** [(] **[l]** [)] [is the learnable]

weight matrix for layer l, and we initialize **W** [(] **[l]** [)] using Xavier
initialization. _σ_ ( _·_ ) denotes a nonlinear activation function.

During the application of graph convolutional network operations on a heterogeneous graph, information dissemination
occurs among nodes throughout the graph. This process empowers central nodes to effectively collect insights from a diverse array of neighbouring nodes, leading to the refinement
of their individual information. Furthermore, the utilization of
the exponential linear unit as the activation function within the
graph convolutional layer contributes to the augmentation of the
model’s capacity for generalization.

_2) Dynamic Self-Attention Mechanism in Convolutional Lay-_
_ers:_ In traditional convolutional networks, the process of gathering information from neighbouring nodes usually involves
assigning equal weights to all these neighbours. Nevertheless, in
actual heterogeneous graphs characterizing drug-disease associations, the associations of different neighbour nodes to a central
node evolves across layers. Thus, the assignment of distinct
weights to diverse neighbour nodes becomes indispensable. To
address this, we integrate a self-attention mechanism [25] into
the graph convolutional layer, where the fundamental procedure
for determining attention weights is outlined as follows:



_α_ _ij_ **W** _h_ _j_



_h_ _[′]_ _i_ [=] _[ σ]_



⎛



⎝ _j_ [�] _∈N_ _i_



⎞

(17)
⎠



_j∈N_ _i_



�



_e_ _ij_ = LeakyReLU



� **a** **[T]** _·_ [ **W** _h_ _i_ _||_ **W** _h_ _j_ ]



(14)



By integrating a dynamic attention mechanism into the graph
convolutional layer, the model becomes adept at dynamically
refining the central node’s information in accordance with the
varying significance of its neighbouring nodes. This augmentation not only fortifies the model’s ability to generalize, but also
effectively enhances the robustness when dealing with noisy
data.

_3) Bi-LSTM Module:_ The Bi-LSTM module [27], [28] is
composed of two LSTM modules designed for processing input
data. In the context of input data stemming from a heterogeneous
graph of drug-disease associations, the first LSTM within the
Bi-LSTM module is harnessed to manage the time-ordered
data of the initial round. Concurrently, the second LSTM is
responsible for processing the time-ordered data of the second
round in a reversed sequence. At last,the outcome of the module
is obtained through a multilayer perceptron. This dual-round
LSTM processing strategy effectively captures long-range dependencies between drug and disease nodes embedded within
the heterogeneous graph, thus enhancing prediction accuracy.
Moreover, recognizing the potential existence of unobserved
values within the heterogeneous graph, the Bi-LSTM module
exhibits an elevated capacity to handle missing values during
the processing of time-ordered data. This adaptive feature also
serves to mitigate the impact of data incompleteness. Compared
to the regular LSTM module, Bi-LSTM incorporates a process
of acquiring node information from the reverse sequence in its
structure. Through bidirectional processing of sequential data,
it comprehensively captures contextual semantic information,
thereby unveiling potential associations between drug and disease nodes. The distinct operational sequence of the Bi-LSTM
is illustrated in Fig. 3.

_4) Residual Connection Module:_ Theutilizationofaresidual
connection module [29] has proven effective in addressing challenges related to gradient vanishing and network degradation. In
the context of this task, as the depth of graph convolutional networksincreases,thereisasusceptibilitytoencounteringgradient
vanishing issues during the propagation of node information
through convolutional layers. To counteract this, we incorporate
a residual connection module, which involves adding the initial
input data to the output obtained from deeper convolutional
layers. This integration serves to alleviate the predicaments
associated with gradient vanishing. The computational process
canbesuccinctlydescribedas follows: _x_ _l_ +1 = _F_ ( _x_ ) + _x_, where
_x_ _l_ +1 denotes the output derived from the deeper convolutional
layer, _x_ signifies the original input, and _F_ ( _x_ ) represents the
output subsequent to undergoing regular convolutional layer
transformations.

The incorporation of residual connections can effectively
mitigate the concern of over-smoothing [30] that often arises



exp ( _e_ _ij_ )
_α_ _ij_ = softmax _j_ ( _e_ _ij_ ) = ~~�~~ _k∈N_ _i_ [exp (] _[e]_ _[ik]_ [)] (15)



In (14), _e_ _ij_ represents the significance score of neighbour node _j_
to node _i_ ’s features. Meanwhile, _h_ _i_ and _h_ _j_ represent sets of node
representations serving as inputs to a given layer. The _||_ denotes
a concatenation operation. Both **a** and **W** are learnable pa
_·_
rameters, initialized using Xavier initialization. _LeakyReLU_ ( )
represents a non-linear activation function. In (15), _α_ _ij_ denotes
the attention coefficient, _N_ _i_ denotes the set of neighbouring
nodes for node _i_, and _softmax_ _j_ ( _·_ ) is the normalization function
applied.

In the attention calculation approach discussed above, attention coefficients exhibit a uniform ordering across all nodes
within the graph and remain uninfluenced by the specific node.
However, in the context of our heterogeneous graph depicting
drug-disease associations, in order to more effectively capture
the attributes of nodes and edges belonging to various types
within this heterogeneous structure, we adopt a dynamic attention strategy [26]. To be precise, we enhance the mechanism by
refining the formulation outlined in (16) as follows:


_e_ _ij_ = **a** **[T]** LeakyReLU ( **W** [ _h_ _i_ _||h_ _j_ ]) (16)


Within this approach, unique key nodes are selected for different query nodes, thus yielding distinct neighbour nodes with
different scores. This technique excels in effectively prioritizing
relative information, surpassing the capabilities of static attention mechanisms, where the ranking of attention scores remains
unconditioned on the query node, and showcasing enhanced



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:33:11 UTC from IEEE Xplore. Restrictions apply.


HUANG et al.: HIERARCHICAL AND DYNAMIC GRAPH ATTENTION NETWORK FOR DRUG-DISEASE ASSOCIATION PREDICTION 2421



Fig. 3. Workflow of Bi-LSTM: The first LSTM Layer processes input
sequences in a forward sequence and then the second LSTM layer
processes the time-ordered data in a reversed sequence. Through a
MLP, a new sequence aggregating more information of the context is
obtained.


during the execution of standard graph convolution operations.
Moreover, it can also accelerate our model’s convergence speed.

_5) Hierarchical Attention Mechanism:_ In graph convolutional layers, each layer learns distinct node embeddings that
encapsulate the structural characteristics of the drug-disease
heterogeneous graph across multiple dimensions. Nevertheless,
the embeddings across different layers might lack continuity,
and the contributions of output embeddings from various layers
to the final output embedding can vary. Building upon earlier
studies [23], [31], we introduce a hierarchical attention mechanism to address this challenge. This mechanism assigns diverse
weights to node embeddings originating from different layers,
ultimately computing the resultant output node embedding. The
ultimate representations of drug and disease node embeddings
are formulated as follows:



Fig. 4. Module of Decoder: It is responsible for reconstructing the
adjacency matrix containing drug-disease associations. This is achieved
using a bilinear decoder that relies on drug embeddings and disease
embeddings.


task. The mathematical formulation for the bilinear decoder is

as follows:



**S** _[′]_ = _sigmoid_ ( _E_ _M_ **W** _[′]_ _E_ _A_



_T_ ) (19)



�



**E** **M**

**E** **A**



�



where **S** _[′]_ is the matrix of predicted association scores between
drugs and diseases and its element _s_ _[′]_ _ij_ [represents the predicted]

association score between drug _i_ and disease _j_ . **W** _[′]_ is a trainable
weight parameter, initialized using Xavier initialization [32],
and _sigmoid_ ( _·_ ) represents a nonlinear activation function. The
process of decoder is illustrated in Fig. 4.


_D. Model Optimization_


Based on previous learning experiences [23], [33], for a
dataset consisting of _N_ drugs and _M_ diseases, we select drugdisease association pairs as positive instances and all other
combinations as negative instances. We denote drug positive
instances and negative instances as _x_ [+] and _x_ _[−]_ respectively. As
the observed number of associations is significantly smaller than
the number of unobserved associations, we opt for weighted
cross-entropy [33] as the loss function, computed as follows:



=



�



_α_ _[l]_ **H** [(] **[l]** [)] (18)



where **E** **M** represents the final drug node embedding, **E** **A** represents the final disease node embedding, _α_ _[l]_ stands for the
learnable weight of the _l_ _[th]_ layer, and **H** [(] **[l]** [)] denotes the node
embeddings of the _l_ _[th]_ layer.

Hierarchical attention mechanism comprehensively consolidates graph structural information embeddings acquired from
diverse layers by attributing distinct weights to embeddings
of each layer. This facilitates the acquisition of an augmented
hierarchy of contextual node information, thereby amplifying
the model’s capacity for generalization and predictive precision.


_C. Decoder_


Once the encoder has generated the drug and disease node
embeddings, the decoder is instrumental in reconstructing the
adjacency matrix that captures the drug-disease associations.
This begins with the extraction of separate node embeddings for
drugs and diseases from the obtained node embeddings. Following this, a bilinear decoder is employed for the reconstruction



1
Loss = _−_
_N ×M_



⎛



_λ×_
⎝



1 _−_ _s_ _[′]_

_ij_



�



⎞



�



⎠



�



�



log _s_ _[′]_ _ij_ [+]



log



( _i,j_ ) _∈x_ [+]



( _i,j_ ) _∈x_ _[−]_



(20)
where _λ_ = _[|][x]_ _[−]_ _[|]_

_|x_ [+] _|_ [,] _[ |][x]_ [+] _[|]_ [ and] _[ |][x]_ _[−]_ _[|]_ [ denote the quantities of] _[ x]_ [+] [ and]
_x_ _[−]_ respectively, ( _i, j_ ) represents a drug-disease pair with drug
_M_ _i_ and disease _A_ _j_ . The parameter _λ_ acts as a weight factor to
mitigate the impact of data imbalance.

Followingthis,weapplytheAdamoptimizer [34] tominimize
the loss function. The Adam optimizer iteratively updates the
neural network parameters to minimize the loss. Furthermore,
to address overfitting concerns, we incorporate node dropout and
regulardropoutwithinthegraphconvolutionallayers.Moreover,
we adopt a cyclic learning rate [35] approach, which cyclically
adjusts the learning rate magnitude to enhance the convergence
speed of the model.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:33:11 UTC from IEEE Xplore. Restrictions apply.


2422 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 28, NO. 4, APRIL 2024


TABLE I

T WO D ATASET



_E. Model Interpretability_


In HDGAT, the information update for a specific node within
a heterogeneous graph is influenced by the information emanating from neighboring nodes within the same graph. Concurrently, neighboring nodes have the capacity to convey higherdimensional node information to the central node, facilitating
improved aggregation of information across the entire heterogeneous graph. Following the application of GCNs in HDGAT, the
input node feature vectors undergo a process that maps nodes
to low-dimensional vectors. The dynamic attention mechanism
embedded in GCN dynamically assigns varying weights to
the neighboring nodes of the central node. Additionally, the
hierarchical attention mechanism allocates distinct weights to
the embeddings of low-dimensional feature vectors learned by
differentlayers.Theseprocessesenhancetheprecisionoffeature
fusion from other relevant nodes, consequently augmenting the
accuracy of the model’s predictive decisions.


_F. Model Scalability_


Tovalidatethescalabilityofthemodel,twodatasetsofdistinct
scales, which contain 269 types of drugs and 598 types of disease,weredeliberatelychosen.Thesmallerdatasetencompasses
6244 known drug-disease associations, while the larger dataset
incorporates 18416 known drug-disease associations. After handling the heterogeneous graph, the dimensions of the input
graph are 867*867. This selection facilitated the observation of
the model’s predictive performance in scenarios characterized
by a higher number of nodes and edges within the heterogeneous graph. The experimental results illustrate that HDGAT
consistently delivers outstanding performance across datasets
of diverse scales. This observation attests to the commendable
scalability of HDGAT, affirming its capacity to undergo stable
training and prediction on larger-scale datasets.


III. E XPERIMENTAL R ESULTS AND D ISCUSSION


_A. Dataset_


The dataset employed in this study is sourced from prior
learning endeavours [23], [36], originating from the Comparative Toxicogenomics Database (CTD) [37]. This dataset encompasses 18,416 drug-disease associations, involving 269 distinct
drugs and 598 different diseases. Pertinent details pertaining
to the drugs, including targets, enzymes, pathways, drug-drug
interactions, and substructures, are meticulously extracted from
the DrugBank database [38]. Furthermore, disease similarities
are assessed based on their MeSH (Medical Subject Headings).
The data in the table represents the number of drug and disease
categories, and the number of different features of drugs. Different features of drugs can effectively distinguish the functions
and relationships among different drugs. Specifically, different
categories of drug substructures signify diverse structural units
or functional groups within drug molecules, exerting notable



TABLE II
E XPERIMENTAL S ETTING


influence on drug-drug interactions. Hence, we calculate drugdrug similarities based on their different features. In order to
validate the generality and robustness of our model, we utilized
a distinct therapeutic dataset comprising 6244 annotated therapeutic drug-disease associations from CTD for performance
comparison. For a comprehensive overview of the main dataset
and therapeutic dataset, refer to Table I.


_B. Experimental Parameter Configuration_


Before conducting our experiments, it was necessary to set the
parameters of the model, including the embedding dimension _k_,
the number of convolutional layers _L_, the learning rate _lr_, the
training epochs _n_, node dropout _dp_ _n_, regular dropout _dp_, penalty
factor _α_, and so on.

In the process of configuring experimental parameters, we
conducted experiments to fine-tune parameters within specified
ranges,suchas _lr ∈_ {0.001,0.005,0.05,0.1}, _dp_, _dp_ _n_ _∈_ {0.1,0.2,
0.3, 0.4, 0.5, 0.6, 0.7}, _L ∈_ {2, 3, 4, 5}, and so on. The learning
rate governs the extent of updating network weights, where an
excessively large learning rate may result in the model failing to
converge, while a too small learning rate may lead to slow model
convergence. The dropout rate serves as a strategy to mitigate
model overfitting, with an appropriately chosen dropout rate
enhancing the model’s generalization performance. However, an
excessively large dropout rate may cause significant information
loss, diminishing the model’s learning capabilities. The number
of graph convolutional layers in the model represents a critical
hyperparameter, where a too small value may yield insufficient
learningcapacity,whileanexcessivelylargevaluecaninducethe
oversmoothing phenomenon, characterized by node representations learned by deep graph convolution becoming increasingly
homogeneous. After multiple rounds of experimentation to finetune these parameters, we established the initial values for the
hyperparameters, as shown in Table II.

Upon model training, a 5-fold cross-validation approach is
employed to assess its performance. This technique, widely
adopted in statistical evaluation, involves the random partitioning of the dataset into five exclusive subsets. Across five rounds,
four subsets function as training data while the remaining subset
is allocated as validation data. During each round, the model is
trained on the training data and evaluated on the validation data.
After these five rounds are completed, the performance metrics
garnered from all iterations are averaged, culminating in a conclusive evaluation of the model’s performance. The application
of the 5-fold cross-validation technique effectively gauges the
model’s capacity to generalize to novel datasets, concurrently
illuminating potential issues of overfitting or underfitting.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:33:11 UTC from IEEE Xplore. Restrictions apply.


HUANG et al.: HIERARCHICAL AND DYNAMIC GRAPH ATTENTION NETWORK FOR DRUG-DISEASE ASSOCIATION PREDICTION 2423


Fig. 6. Model performance of different layer’s embedding.



Fig. 5. Model performance with different modules: HDGAT-S1:Replace
Bi-LSTM with LSTM on HDGAT; HDGAT-S2:Remove LSTM on HDGATS1; HDGAT-S3:Remove residual connection on HDGAT-S2; HDGATS4:Replace dynamic attention with static attention on HDGAT-S3;
HDGAT-S5:Remove static attention on HDGAT-S4.


_C. Ablation Experiment_


To gain a comprehensive understanding of the distinct contributions made by various components of the model to the overall
efficacy of drug-disease association prediction, we conducted
ablation experiments to discern the impact of different combinations of modules within the HDGAT framework. Ablation experiments constitute a pivotal aspect of our experimental approach,
as they serve to validate the effectiveness of specific modules
in enhancing performance for the designated task, ultimately
guiding refinements aimed at achieving heightened predictive

accuracy.

In this pursuit, we formulated five distinct sets of module
configurations in conjunction with our original HDGAT model
for rigorous experimentation. In the initial set, we substituted
Bi-LSTM with LSTM within the framework of HDGAT, denoted as HDGAT-S1. In the subsequent set, we excluded LSTM
from HDGAT-S1, resulting in HDGAT-S2. In the third set, we
eliminated residual connections from HDGAT-S2, identified
as HDGAT-S3. In the fourth set, we converted the dynamic
attention mechanism into a static attention mechanism based
on HDGAT-S3, labeled as HDGAT-S4. Finally, in the fifth set,
we omitted the static attention mechanism from HDGAT-S4,
designated as HDGAT-S5. Through rigorous experimentation
involving HDGAT and these three model variants, we garnered
outcomes delineated in Fig. 5, offering crucial insights into the
impact of diverse module combinations on the overall perfor
mance.


_D. Impact of Different Layer Embeddings_


From the results of our experimental analysis, it is evident
that HDGAT-S1 exhibits a notable reduction in both accuracy
and precision in comparison to the original HDGAT model. This
finding underscores the beneficial influence of the dual-round
LSTM processing strategy on the accuracy of drug-disease association prediction. HDGAT-S2 showcases a decrease in AUPR,
Precision when contrasted with HDGAT-S1, which manifests
that LSTM can capture the latent associations between drugs
and diseases. Notably, HDGAT-S3 displays a decrease across all



metrics. This shift indicates that the inclusion of the residual connection module has a favorable effect on predicting positive instances. Furthermore, comparing HDGAT-S4 and HDGAT-S5,
we observed a significant decline in all metrics for HDGAT-S5
except for Accuracy and Specificity. From this, it can be inferred
that assigning different weights to nodes can effectively enhance
node information aggregation capability.

In summation, the outcomes collectively underscore that
HDGAT emerges as the superior performer among the four
models under consideration. This observation corroborates that
the amalgamation of modules integrated into HDGAT distinctly
contributes to the model’s overall performance enhancement.
Besides, ablation experiments serve to elucidate the contributions of various modules to HDGAT.

To investigate the impact of different layer embeddings within
the graph convolutional networks on the overall predictive performance of the model, we conducted a series of experiments
specifically designed to assess the predictive capabilities of
embeddings derived from various layers. As outlined in our previous experimental configurations, we have employed a neural
network with a total of 3 layers. The rationale behind selecting
this architecture is rooted in the notion that a 3-layer neural networkdeliverscommendablepredictiveperformance.Employing
an excessive number of layers could potentially trigger the problem of gradient vanishing during training, whereas utilizing too
few layers might fail to capture the intricate high-dimensional
associations prevalent within the heterogeneous graph.

Hence, we proceeded to leverage the node embeddings acquired from each of the first, second, and third layers of the
network as standalone entities for prediction purposes. These
single-layer models are denoted as HDGAT-L1, HDGAT-L2,
HDGAT-L3, respectively. Through rigorous evaluation encompassing 5-fold cross-validation, we subjected these models to
meticulous scrutiny and juxtaposed their performances against
the original model. The comprehensive findings stemming from
these experiments are meticulously presented in Table III and
visually depicted in Fig. 6.

The experimental outcomes distinctly reveal that the predictive prowess exhibited by the HDGAT-L1 and HDGAT-L2
models markedly surpasses that of the HDGAT-L3 model.
This compelling evidence strongly suggests that, in contrast to
the embeddings originating from the third layer, the embeddings stemming from the first and second layers encompass
a wealth of pertinent information concerning graph structure
and inter-node relationships. This discernible pattern strongly
implies that the HDGAT framework could potentially encounter certain constraints when it comes to effectively capturing intricate high-dimensional associations prevalent among
nodes.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:33:11 UTC from IEEE Xplore. Restrictions apply.


2424 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 28, NO. 4, APRIL 2024


TABLE III

M ODEL P ERFORMANCE OF D IFFERENT L AYER ’ S E MBEDDING


TABLE IV

M ODEL P ERFORMANCE W ITH D IFFERENT A GGREGATE M ETHODS


TABLE V

P ERFORMANCE OF D IFFERENT M ETHODS ON M AIN D ATASET AND T HERAPEUTIC D ATASET



_E. Impact of Hierarchical Attention Mechanism_


The data presented in Table III undeniably highlights
HDGAT’s superior performance relative to the other three models, conclusively illustrating that embeddings derived from distinct layers exert disparate influences on the final predictive efficacy of the model. Hence, it becomes imperative to amalgamate
embeddings from various dimensions to effectively amalgamate
the acquired structural attributes and association insights. To
address this objective, we adopted three distinct methods for embedding aggregation: 1. HDGAT; 2. HDGAT-AVG; 3. HDGATCONCAT. In the HDGAT approach, we harnessed the hierarchical attention mechanism for embedding amalgamation. In
HDGAT-AVG, we aggregated embeddings through summation
followed by averaging. Conversely, HDGAT-CONCAT involved
a direct concatenation of node embeddings. The comprehensive results of these experiments are meticulously detailed in
Table IV.

The outcomes presented in Table IV illuminate HDGAT’s
supremacy over the other two methods concerning predictive
prowess. This observation underscores the inherent benefits of
employing the hierarchical attention mechanism to manage the
divergent contributions originating from embeddings of distinct
layers. The experimental results illustrated in Fig. 6 further
corroborate this assertion, revealing that node embeddings from
lower layers encompass a more substantial repository of graph
structural attributes and node association insights, while those
from higher layers contain comparatively diminished information. Thus, the hierarchical attention mechanism judiciously
assigns greater weights to lower-layer embeddings and correspondingly lighter weights to their higher-layer counterparts,
manifestly striving to attain optimal predictive performance for
the model.



_F. Comparisons With Other Models_


In this section, we compare the HDGAT model with two
baseline models and three existing excellent models for drugdisease association prediction. We selected NIMCGCN [39]
and LAGCN [23] as our baseline model due to their shared
foundation as GCN-based models. Based on the experimental
parameter configuration above and the reported data from prior
studies [23], we conducted comparative experiments on main
dataset. Besides, we also used therapeutic dataset to validate
our model’s generality. The results are shown in Table V.

_TL-HGBI [5]:_ It introduces an algorithm that iteratively updates to calculate the length of drug-disease pairs in a heterogeneous graph network integrating drug and disease information,
and predicts drug-disease associations.

_DRRS [40]:_ It first constructs a heterogeneous graph interaction network by combining drug-drug, disease-disease, and
drug-disease networks. Then, it utilizes a rapid Singular Value
Thresholding (SVT) algorithm based on recommendation systems to predict unknown drug-disease associations.

_DeepDR [39]:_ It learns higher-order features of drugs from
the heterogeneous graph network using a multi-modal deep
autoencoder. It encodes and decodes these learned features along
with known drug-disease pairs using a variational autoencoder
to infer drug-disease associations.

_NIMCGCN [41]:_ It initially employs Graph Convolutional
Networks (GCN) to learn latent feature representations of miRNAs and diseases from a similarity network. Then, it utilizes
a novel neural inductive matrix completion approach with the
learned features to generate the association matrix.

_LAGCN [23]:_ It employs GCN to learn drug-disease node
embeddings in a heterogeneous graph network, containing drug
and disease similarities and associations, and applies a layer



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:33:11 UTC from IEEE Xplore. Restrictions apply.


HUANG et al.: HIERARCHICAL AND DYNAMIC GRAPH ATTENTION NETWORK FOR DRUG-DISEASE ASSOCIATION PREDICTION 2425


attention mechanism to these embeddings for predicting drugdisease associations.



In the aforementioned experiments, the training data for
drug-disease associations in the two datasets exhibit disparate
distributions, encompassing variations in both the types of drug
associations and the sizes of the datasets. Consequently, the
experiments provide a robust observation of HDGAT’s generalization capability across diverse data distributions. The experimental results in Table V demonstrate that HDGAT consistently
achieves high accuracy across multiple datasets, outperforming
both baseline models and existing state-of-the-art models in
terms of overall performance. These results affirm the robust
generalization capacity of HDGAT. Notably, in terms of AUPR
and Precision, HDGAT exhibits significant improvement,which
indicates that our model is capable of maintaining a relatively
high recall while preserving a high precision. Compared to
the baseline models, the introduction of residual connections
helps overcome certain limitations inherent in NIMCGCN and
LAGCN, particularly alleviating the over-smoothing issue encountered during the training of multi-layer GCNs. Besides,
HDGAT employs a dynamic attention mechanism that allows for
more precise assignment of weights to nodes, thereby accurately
aggregating information from different nodes. This substantiates
the enhanced robustness exhibited by HDGAT when contrasted
with other models.

During the process of model training, the stacking of an excessive number of layers in GCNs results in the persistence of issues
such as gradient vanishing and oversmoothing. This occurrence
leads to a gradual reduction in the learning capacity of the model.
Consequently, the challenges posed by the oversmoothing problem and the gradient vanishing issue introduced by GCN persist
as significant obstacles for HDGAT in its development.


_G. Case Studies_


In this paper, we utilize HDGAT to predict new drug-disease
associations based on existing known drug-disease associations.
Previously, we evaluate the model’s predictive performance
solely based on the predicted association scores. However, assessing predictive accuracy solely based on prediction scores
might not provide a comprehensive measure of accuracy. Therefore, we compare the predicted drug-disease associations from
our model with the existing medical data and associations
present in databases to assess its practical applicability and
reliability.

In this experiment, we employ HDGAT to predict drugdisease associations. The top 10 drug-disease association pairs
with the highest prediction scores are calculated by HDGAT.
To explore the association scores of these 10 drugs with the 10
diseases, we generated a heatmap as depicted in Fig. 7.

Based on the predicted associations from the model, we try
to seek relevant medical evidence to determine whether there
is a connection between them. For instance, trandolapril is an
angiotensin-converting enzyme (ACE) inhibitor that is widely
used for the treatment of patients with Hypertension [42]. ACE
inhibitors help control hypertension by reducing the formation
of angiotensin II, and they can also reduce the cardiac load
and improve cardiac function. Maprotiline is an antidepressant,
however, the use of such antidepressants often comes with side
effects, including Tremor [43].

Furthermore, we conducted specific experiments on certain
drugs and diseases to explore the associations in more detail. We



Fig. 7. Association scores between the top 10 pairs of drug-disease
associations predicted by HDGAT.


TABLE VI

T OP 10 R ELEVANT D ISEASES A SSOCIATED W ITH S PECIFIC D RUG

P REDICTED BY HDGAT


TABLE VII
T OP 10 R ELEVANT D RUGS A SSOCIATED W ITH S PECIFIC D ISEASE

P REDICTED BY HDGAT


investigated the top ten diseases most associated with the drug
clobazam and the top ten drugs most associated with the disease
Adenocarcinoma of lung. The results of these experiments are
presented in Tables VI and VII.

For the drug-disease associations identified, we also validate them by consulting relevant literature. For example,
clobazam [44] has been confirmed as an effective antiepileptic
drug and is widely used for both pediatric and adult epilepsy
patients in various countries. Epilepsy is a neurological disorder
that can lead to various types of seizures, and clobazam can reduce the frequency of these seizures. Similarly, in the context of
advanced treatments for Adenocarcinoma of lung, Doxorubicin
and docetaxel are frequently utilized [45], [46] as chemotherapy
drugs. They are commonly used in cancer treatment, including
Adenocarcinoma of lung.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:33:11 UTC from IEEE Xplore. Restrictions apply.


2426 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 28, NO. 4, APRIL 2024



However, in the practical process of model prediction,
HDGAT falls short of achieving flawless and precise predictions
for all drug-disease associations. The intricacies of the relationships between drugs and diseases, with some associations being
direct and others indirect, present formidable challenges to the
model’s predictive capabilities. Moreover, the quality of medical
literature and the timeliness of medical research play pivotal
roles in influencing the accurate discernment of drug-disease
predictions.Consequently,encounteringfalsepositivesandfalse
negatives during the prediction process is at times inevitable.
In response to such instances, we opt to amplify the scale of
the dataset and broaden the scope of model predictions, with
the goal of fortifying the model’s robustness through more
extensive training data. Furthermore, validation using more
authoritative and official medical literature serves as an effective
means to corroborate the accuracy of the prediction results.

These case studies provide compelling support for the conclusions drawn from our model’s predictions, demonstrating
that the HDGAT model is capable of effectively predicting
drug-disease association information.


IV. C ONCLUSION


In this paper, we introduce HDGAT, a novel model designed
for the prediction of drug-disease associations. The methodology comprises three primary stages:

1. The initial phase involves the construction of a hetero
geneous graph by amalgamating networks representing
drug-drug similarities, disease-disease similarities, and
drug-disease associations.
2. Subsequently, a graph convolutional network equipped

with hierarchical and dynamic attention mechanisms, in
conjunction with a Bi-LSTM module, is employed to
acquire node embeddings and capture the structural information within the heterogeneous network.
3. In addition, residual connections are integrated into

HDGAT. Themodel’s performanceis assessedusingarigorous five-fold cross-validation approach, supplemented
by case studies to validate predictive outcomes.
HDGAT excels in capturing the intricate network structure of
the heterogeneous graph and relevant associations between drug
and disease over long distance. The incorporation of residual
connections mitigates over-smoothing issues associated with
graphconvolutionswhileenhancingbothconvergencespeedand
generalization capabilities. Besides, dynamic attention mechanisms assist in dynamically refining the central node’s information to more effectively capture the characteristics of nodes and
edges within a heterogeneous graph. The experimental results
robustly demonstrate HDGAT’s superiority across numerous
evaluation metrics in the context of drug-disease association prediction tasks when compared to existing state-of-the-art models.

However, compared with homogeneous graphs, HDGAT exhibits certain limitations in aggregating information from different nodes and edges in heterogeneous graphs, where connected nodes often exhibit similar properties. Besides, while
residual connection modules prove effective in mitigating the
oversmoothing phenomenon introduced by GCNs, the challenge
of gradient vanishing persists, particularly when stacking an
excessive number of layers in graph neural networks.



In future research, enhancing GCN-based models by integrating additional considerations for heterogeneous nodes
and edges, based on the foundation of the HDGAT framework, emerges as a promising avenue for improvement. Furthermore, investigating node information aggregation through
meta-paths [7] has the potential to mitigate oversmoothing phenomenon,makingitanotherviabledirectionforextendingfuture
models.


V. M ATERIALS


1. The drug-disease associations database: http://ctdbase.


org.
[2. The drug features database: https://www.drugbank.ca/.](https://www.drugbank.ca/)
[3. The disease MeSH database: https://meshb.nlm.nih.gov/.](https://meshb.nlm.nih.gov/)
4. All the materials and codes in this paper are available at

[https://github.com/37918273918/HDGAT.](https://github.com/37918273918/HDGAT)


R EFERENCES


[1] L. Liu et al., “Multi-view contrastive learning hypergraph neural network

for drug-microbe-disease association prediction,” in _Proc. 32nd Int. Joint_
_Conf. Artif. Intell._, vol. 2023, pp. 4829–4837.

[2] Z. Chu et al., “Hierarchical graph representation learning for the prediction

of drug-target binding affinity,” _Inf. Sci._, vol. 613, pp. 507–523, 2022.

[3] W. Ben Abdessalem Karaa, E. H. Alkhammash, and A. Bchir, “Drug dis
ease relation extraction from biomedical literature using nlp and machine
learning,” _Mobile Inf. Syst._, vol. 2021, pp. 1–10, 2021.

[4] P. Wang, T. Hao, J. Yan, and L. Jin, “Large-scale extraction of drug–disease

pairs from the medical literature,” _J. Assoc. Inf. Sci. Technol._, vol. 68,
no. 11, pp. 2649–2661, 2017.

[5] W. Zhang et al., “Predicting drug-disease associations based on the known

association bipartite network,” in _Proc. IEEE Int. Conf. Bioinf. Biomed._,
2017, pp. 503–509.

[6] Y.-Z. Di, P. Chen, and C.-H. Zheng, “Similarity-based integrated method

for predicting drug-disease interactions,” in _Proc. Intell. Comput. Theories_
_Appl.: 14th Int. Conf._, 2018, pp. 395–400.

[7] F. Tanvir, M. I. K. Islam, and E. Akbas, “Predicting drug-drug interactions

using meta-path based similarities,” in _Proc. IEEE Conf. Comput. Intell._
_Bioinf. Comput. Biol._, 2021, pp. 1–8.

[8] F.Tanvir,K.M.Saifuddin,M.IfteKhairulIslam,andE.Akbas,“Predicting

drug-drug interactions using heterogeneous graph attention networks,” in
_Proc. 14th ACM Int. Conf. Bioinf., Comput. Biol., Health Inf._, 2023, pp. 1–
6.

[9] F. Tanvir, K. M. Saifuddin, T. Hossain, A. Bagavathi, and E. Akbas,

“HeTriNet: Heterogeneous graph triplet attention network for drug-targetdisease interaction,” 2023, _arXiv:2312.00189_ .

[10] K. M. Saifuddin, B. Bumgardner, F. Tanvir, and E. Akbas, “HyGNN:

Drug-drug interaction prediction via hypergraph neural network,” in _Proc._
_IEEE 39th Int. Conf. Data Eng._, 2023, pp. 1503–1516.

[11] J.Pengetal.,“Anend-to-endheterogeneousgraphrepresentationlearning
based framework for drug–target interaction prediction,” _Brief. Bioinf._,
vol. 22, no. 5, 2021, Art. no. bbaa430.

[12] C. Tang et al., “DeFusionNET: Defocus blur detection via recurrently

fusing and refining discriminative multi-scale deep features,” _IEEE Trans._
_Pattern Anal. Mach. Intell._, vol. 44, no. 2, pp. 955–968, Feb. 2022.

[13] H. Liu, W. Zhang, Y. Song, L. Deng, and S. Zhou, “HNet-DNN: Infer
ring new drug–disease associations with deep neural network based on
heterogeneous network features,” _J. Chem. Inf. Model._, vol. 60, no. 4,
pp. 2367–2376, 2020.

[14] P. Xuan, L. Gao, N. Sheng, T. Zhang, and T. Nakaguchi, “Graph con
volutional autoencoder and fully-connected autoencoder with attention
mechanism based method for predicting drug-disease associations,” _IEEE_
_J. Biomed. Health Inform._, vol. 25, no. 5, pp. 1793–1804, May 2021.

[15] A. Vaswani et al., “Attention is all you need,” in _Proc. Adv. Neural Inf._

_Process. Syst._, 2017.

[16] C. Tang et al., “Spatial and spectral structure preserved self-representation

for unsupervised hyperspectral band selection,” _IEEE Trans. Geosci. Re-_
_mote Sens._, vol. 61, 2023, Art. no. 5531413.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:33:11 UTC from IEEE Xplore. Restrictions apply.


HUANG et al.: HIERARCHICAL AND DYNAMIC GRAPH ATTENTION NETWORK FOR DRUG-DISEASE ASSOCIATION PREDICTION 2427




[17] C. Tang, X. Zheng, W. Zhang, X. Liu, X. Zhu, and E. Zhu, “Unsupervised

feature selection via multiple graph fusion and feature weight learning,”
_Sci. China Inf. Sci._, vol. 66, no. 5, pp. 1–17, 2023.

[18] X. Zeng et al., “Measure clinical drug–drug similarity using electronic

medical records,” _Int. J. Med. Inf._, vol. 124, pp. 97–103, 2019.

[19] C. Yan, G. Duan, Y. Zhang, F.-X. Wu, Y. Pan, and J. Wang, “Pre
dicting drug-drug interactions based on integrated similarity and semisupervised learning,” _IEEE/ACM Trans. Comput. Biol. Bioinf._, vol. 19,
no. 1, pp. 168–179, Jan./Feb., 2022.

[20] X. Yang, G. Yang, and J. Chu, “The neural metric factorization for com
putational drug repositioning,” _IEEE/ACM Trans. Comput. Biol. Bioinf._,
vol. 20, no. 1, pp. 731–741, Jan./Feb. 2023.

[21] J. Yu, J. Amores, N. Sebe, P. Radeva, and Q. Tian, “Distance learning for

similarity estimation,” in _IEEE Trans. Pattern Anal. Mach. Intell._, vol. 30,
no. 3, pp. 451–462, Mar., 2008.

[22] D. Wang, J. Wang, M. Lu, F. Song, and Q. Cui, “Inferring the human mi
croRNA functional similarity and functional network based on micrornaassociated diseases,” _Bioinformatics_, vol. 26, no. 13, pp. 1644–1650,
2010.

[23] Z. Yu, F. Huang, X. Zhao, W. Xiao, and W. Zhang, “Predicting drug–

disease associations through layer attention graph convolutional network,”
_Brief. Bioinf._, vol. 22, no. 4, 2021, Art. no. bbaa243.

[24] T. N. Kipf and M. Welling, “Semi-supervised classification with graph

convolutional networks,” in _Proc. Int. Conf. Learn. Representations_, 2016.

[25] P. Veliˇckovi´c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio,

“Graph attention networks,” in _Proc. Int. Conf. Learn. Representations_,
2018.

[26] S. Brody, U. Alon, and E. Yahav, “How attentive are graph attention

networks,” in _Proc. Int. Conf. Learn. Representations_, 2021.

[27] S. Siami-Namini, N. Tavakoli, and A. S. Namin, “The performance of

LSTM and BiLSTM in forecasting time series,” in _Proc. IEEE Int. Conf._
_Big Data_, 2019, pp. 3285–3292.

[28] C. Zhang, D. Song, C. Huang, A. Swami, and N. V. Chawla, “Hetero
geneous graph neural network,” in _Proc. 25th ACM SIGKDD Int. Conf._
_Knowl. Discov. Data Mining_, 2019, pp. 793–803.

[29] K. He, X. Zhang, S. Ren, and J. Sun, “Identity mappings in deep residual

networks,” in _Proc. 14th Eur. Conf. Comput. Vis._, 2016, pp. 630–645.

[30] Q. Li, Z. Han, and X.-M. Wu, “Deeper insights into graph convolutional

networks for semi-supervised learning,” in _Proc. AAAI Conf. Artif. Intell._,
2018, pp. 3538–3545.

[31] S. Zhang, X. Xu, Y. Pang, and J. Han, “Multi-layer attention based CNN for

target-dependent sentiment classification,” _Neural Process. Lett._, vol. 51,
pp. 2089–2103, 2020.




[32] J. Sirignano and K. Spiliopoulos, “Scaling limit of neural networks with

the xavier initialization and convergence to a global minimum,” 2019,
_arXiv:1907.04108_ .

[33] Y. S. Aurelio, G. M. De Almeida, C. L. de Castro, and A. P. Braga, “Learn
ing from imbalanced data sets with weighted cross-entropy function,”
_Neural Process. Lett._, vol. 50, pp. 1937–1949, 2019.

[34] D. P. Kingma, “Adam: A method for stochastic optimization,” in _Proc. Int._

_Conf. Learn. Representations_, 2014.

[35] J. Li and X. Yang, “A cyclical learning rate method in deep learning

training,” in _Proc. IEEE Int. Conf. Comput., Inf. Telecommun. Syst_, 2020,
pp. 1–5.

[36] W. Zhang et al., “Predicting drug-disease associations by using similarity

constrained matrix factorization,” _BMC Bioinf._, vol. 19, pp. 1–12, 2018.

[37] A. P. Davis et al., “The comparative toxicogenomics database: Up
date2017,” _Nucleic Acids Res._, vol. 45, no. D1, pp. D972–D978, 2017.

[38] V. Law et al., “DrugBank 4.0: Shedding new light on drug metabolism,”

_Nucleic Acids Res._, vol. 42, no. D1, pp. D1091–D1097, 2014.

[39] X. Zeng, S. Zhu, X. Liu, Y. Zhou, R. Nussinov, and F. Cheng, “deepDR:

A network-based deep learning approach to in silico drug repositioning,”
_Bioinformatics_, vol. 35, no. 24, pp. 5191–5198, 2019.

[40] H. Luo, M. Li, S. Wang, Q. Liu, Y. Li, and J. Wang, “Computational

drug repositioning using low-rank matrix approximation and randomized
algorithms,” _Bioinformatics_, vol. 34, no. 11, pp. 1904–1912, 2018.

[41] J. Li, S. Zhang, T. Liu, C. Ning, Z. Zhang, and W. Zhou, “Neural inductive

matrix completion with graph convolutional networks for mirna-disease
association prediction,” _Bioinformatics_, vol. 36, no. 8, pp. 2538–2546,
2020.

[42] L. N. C. Duc and H. R. Brunner, “Trandolapril in hypertension: Overview

of a new angiotensin-converting enzyme inhibitor,” _Amer. J. Cardiol._,
vol. 70, no. 12, pp. D27–D34, 1992.

[43] J. Bouchard et al., “Citalopram versus maprotiline: A controlled, clinical

multicentre trial in depressed patients,” _Acta Psychiatrica Scandinavica_,
vol. 76, no. 5, pp. 583–592, 1987.

[44] Y.-t. Ng and S. D. Collins, “Clobazam,” _Neurotherapeutics_, vol. 4, no. 1,

pp. 138–144, 2007.

[45] W. M. Jordan et al., “Treatment of advanced adenocarcinoma of the lung

with ftorafur, doxorubicin, cyclophosphamide, and cisplatin (FACP) and
intensive IV hyperalimentation,” _Cancer Treat. Rep._, vol. 65, no. 3–4,
pp. 197–205, 1981.

[46] H. Yasuda et al., “Nitroglycerin treatment may enhance chemosensitivity

to docetaxel and carboplatin in patients with lung adenocarcinoma,” _Clin._
_Cancer Res._, vol. 12, no. 22, pp. 6748–6757, 2006.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:33:11 UTC from IEEE Xplore. Restrictions apply.


