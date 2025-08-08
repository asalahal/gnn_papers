IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 20, NO. 6, NOVEMBER/DECEMBER 2023 3353

# Microbe-Disease Association Prediction Using RGCN Through Microbe-Drug-Disease Network


Yueyue Wang, Xiujuan Lei, and Yi Pan



_**Abstract**_ **—Accumulating evidence has shown that microbes play**
**significant roles in human health and diseases. Therefore, identify-**
**ing microbe-disease associations is conducive to disease prevention.**
**In this article, a predictive method called TNRGCN is designed**
**for microbe-disease associations based on Microbe-Drug-Disease**
**Network and Relation Graph Convolutional Network (RGCN).**
**First, considering that indirect links between microbes and diseases**
**will be increased by introducing drug related associations, we**
**construct a Microbe-Drug-Disease tripartite network through data**
**processing from four databases including Human Microbe-Disease**
**Association Database (HMDAD), Disbiome Database, Microbe-**
**Drug Association Database (MDAD) and Comparative Toxicoge-**
**nomics Database (CTD). Second, we construct similarity networks**
**for microbes, diseases and drugs via microbe function similar-**
**ity, disease semantic similarity and Gaussian interaction profile**
**kernel similarity, respectively. Based on the similarity networks,**
**Principal Component Analysis (PCA) is utilized to extract main**
**features of nodes. These features will be input into the RGCN as**
**initial features. Finally, based on the tripartite network and initial**
**features, we design two-layer RGCN to predict microbe-disease**
**associations. Experimental results indicate that TNRGCN achieves**
**best performance in cross validation compared with other methods.**
**Meanwhile,casestudiesforType2diabetes(T2D),Bipolardisorder**
**and Autism demonstrate the favorable effectiveness of TNRGCN**
**in association prediction.**


_**Index**_ _**Terms**_ **—Autism,** **bipolar** **disorder,** **microbe-disease**
**associations,** **microbe-drug-disease** **network,** **relation** **graph**
**convolutional network, type 2 diabetes.**


I. I NTRODUCTION


Microbe communities are tiny organisms mainly including
eukaryotes, archaea, bacteria and viruses, which are regarded as
a special “organ” of human beings [1]. Thousands of microbes
reside in different parts of human organs and tissues such as
skin, oral cavity and gastrointestinal tract. In general, microbes
are harmless to human health, and even have a beneficial side.
Forexample,avarietyoforalmicrobiomesresidinginthehuman
oral cavity interact with each other, protecting the human body
against harmful external stimulation [2]. Researchers found


Manuscript received 26 February 2022; revised 11 October 2022; accepted
16 February 2023. Date of publication 22 February 2023; date of current version
26 December 2023. We thank the financial support from National Natural
Science Foundation of China under Grants 62272288, 61972451, 61902230
and U22A2041. _(Corresponding author: Xiujuan Lei.)_

Yueyue Wang and Xiujuan Lei are with the School of Computer Science,
[Shaanxi Normal University, Xi’an, Shaanxi 710119, China (e-mail: yueyue-](mailto:yueyuewang@snnu.edu.cn)
[wang@snnu.edu.cn; xjlei@snnu.edu.cn).](mailto:yueyuewang@snnu.edu.cn)

Yi Pan is with the Faculty of Computer Science and Control Engineering,
Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences,
[Shenzhen, Guangdong 518055, China (e-mail: yipan@gsu.edu).](mailto:yipan@gsu.edu)

Digital Object Identifier 10.1109/TCBB.2023.3247035



that intestinal flora affects brain development such as cognitive
function and basic behavior patterns, and its disorder will have
a negative impact on psychological health [3].

Most microbes residing in the human body maintain homeostasis, providing ecosystem services [4]. But these balances
can be easily changed by the external environment and their
own environment such as diet, genotype and colonization history [5]. For example, diet regulates and supports the intestinal microbiota. Different types, qualities and sources of food
affect the composition and function of intestinal microbes as
well as host-microbiome interactions [6]. These balances are
closely related to human diseases. For instance, in patients
with inflammatory bowel disease (IBD), microbial diversity is
generally reduced. On average, IBD patients carry 75% of the
microbial genes of healthy people [7]. Another study found
that the composition of intestinal microbial group is related to
the symbol of atherosclerosis and arterial hardness [8]. Thus,
knowing microbe-disease associations is beneficial to disease
diagnosis and treatment.

Recently, two databases for relationships with microbes and
diseases have been established: Human Microbe-Disease As
sociation Database (HMDAD) and Disbiome. HMDAD contains 483 microbe-disease associations collected manually from
61 literatures [9]. Disbiome is a constantly updated database,
including confirmed microbe-disease associations and experimental verification records from various literatures and database

[10].

Based on confirmed microbe-disease associations from these
databases, many predictive methods have been designed to
explore more unknown associations. In 2017, Chen et al. first integrated confirmed associations and Gaussian interaction profile
(GIP) similarities, to constract a microbe-disease heterogeneous
network. Then, they predicted microbe-disease associations by
integrating step length and the number of paths between two
nodes in the heterogeneous network [11]. Inspired by this
method, Li et al. reconstructed a microbe-disease heterogeneous
network by integrating confirmed associations and normalized
GIP kernel similarity. Then they combining bidirectional recommendations and KATZ method on the heterogeneous network

[12]. Jiang et al. constructed a knowledge graph centered on microbes and diseases by collecting data from multiple databases.
And then, they predicted potential associations by utilizing
graph neural network method to learn nodes’ representation of
knowledge graph [13]. Peng et al. predicted potential microbedisease associations by using Kronecker sum operations and
eigenvalue transformation on similarity networks [14]. Hua et al.



1545-5963 © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.

See https://www.ieee.org/publications/rights/index.html for more information.


Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:28:25 UTC from IEEE Xplore. Restrictions apply.


3354 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 20, NO. 6, NOVEMBER/DECEMBER 2023



utilized Graph Augmentation Convolutional Network and attention mechanism to learn node features. And then they combined
matrix completion to predict potential associations [15]. Li et al.
constructed a three-layer back-propagation neural network with
a new activation function based on hyperbolic tangent function.
In order to improve operating efficiency, they used microbe
GIP kernel similarity to weight the initial connection value

[16]. Wang et al. designed a predicted method called MSLINE.
This method learned the multi-level domain information in

microbe-diseasenetworkbycombininganembeddingalgorithm
LINE and random walk [17]. Liu et al. explored microbe-disease
associations by utilizing a multi-component graph attention network and a fully connected network [18]. Peng et al. aggregated
multi-view features including linear feature and nolinear feature.
This method takes into account the complementarity of different
features [19]. Chen et al. constructed a microbe-drug-disease
heterogeneous network, and then used the multi-head attention
mechanism to aggregate different meta-paths to learn the nodes’
features [20].

These methods achieved good predictive performance, but
currently used database contain few microbe-disease associations. For example, HMDAD contains only 39 diseases, and
using this database alone will result in too few diseases that
can be predicted. Moreover, many different databases including
microbes, diseases and drugs have been constructed, such as
Microbe-Drug Association Database (MDAD) and Comparative
Toxicogenomics Database (CTD). these databases contain various types of nodes, which can directly or indirectly increase the
relationship between microbes and diseases, and are conducive
to prediction.

Graph Convolution Network (GCN) aggregates neighbors’
information through convolutional operation to extracts node
features in a network [21]. It has been widely used and has
shown good performance in association prediction. In the field
of bioinformatics, GCN has been applied to predict circRNAdisease association [22], metabolite-disease association [23]
and drug–drug association [24]. However, in the process of
aggregation, it treats all neighbor nodes as the same type, and
does not selectively aggregate according to the type of neighbor
nodes. Considering this reason, Relation Graph Convolutional
Network (RGCN) [25] considers neighbor node types and the
connection direction with the current node when convolution.

Therefore, it can be applied to heterogeneous networks with
different types of nodes and edges.

In this article, we predict microbe-disease association based
on Microbe-Drug-Disease Tripartite Network and RGCN
(called TNRGCN). First, we construct a microbe-drug-disease
tripartite network by screening microbe-disease associations,
microbe-drug associations and disease-drug associations from
HMDAD, Disbiome, MDAD and CTD. Second, we use Principal Component Analysis (PCA) to extract main features of nodes
in similarity networks, and input them into RGCN as initial
features. These similarity networks include microbe function
similarity, disease semantic similarity and GIP similarity. Finally, based on the microbe-drug-disease tripartite network and
initial features, we utilize two-layer RGCN to predict potential
associations. Compared with other methods, TNRGCN achieves



TABLE I

D ETAILS OF THE F ILTERED D ATA


best performance. Case studies for Type 2 diabetes (T2D), Bipolar disorder and Autism also demonstrate the good performance
of TNRGCN. The flowchart of TNRGCN is shown in Fig. 1.

Our main contribution is as follows. First, we integrate HMDAD and Disbiome, including more microbe-disease associations. Second, we combine different associations among microbes, diseases and drugs, which can enrich link information in
the network. Third, RGCN is utilized to learn node features in
microbe-drug-disease tripartite network, considering different
types of nodes and edges.


II. M ATERIAL AND M ETHODS


_A. Material_


The data for this article is from four databases. The microbe
disease associations are collected from HMDAD and Disbiome.

HMDAD contains 483 confirmed microbe-disease associations
between 292 microbes and 39 diseases. By removing duplicate
associations, we obtained 450 records among them. Meanwhile,
we obtain all records from Disbiome as of December, 2020, including 1585 microbes, 353 diseases and 8695 microbe-disease
associations between them. After removing the duplicate associations, we finally obtain 1416 microbes, 243 diseases and
their corresponding 7052 association records. The microbe-drug
associations are collected from MDAD, which includes 180
microbes, 1388 drugs and their corresponding 5055 associations. The disease-drug associations are collected from CTD,
including 7119363 association relation records among 12791
drugs and 7098 diseases.


_B. Methods_


_1) Data Processing:_ First, we integrate all microbe-disease
associations in HMDAD and Disbiome and remove the duplicate
records. Considering that Disbiome contains 17 types disease
records, including Disease or Syndrome, Organism Function,
Individual Behavior and so on. According to the UMLS CUI in
DisGeNET [26], we compared disease ID with UMLS CUI, and
screened out three types of diseases: Disease or Syndrome, Mental or Behavioral Dysfunction and Neoplastic Process. Finally,
we obtain 254 diseases and 7258 microbe-disease associations

related to them, involving 1519 microbes. Second, for 1519
microbes associated with diseases, we screen out 3783 microbedrug associations from MDAD related to them, involving 1181
drugs. Third, for 1181 drugs and 254 diseases, we obtain 4552
disease-drug associations from CTD. The flowchart of data
processing is as Fig. 2. The specific details of the filtered data
are shown in Table I.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:28:25 UTC from IEEE Xplore. Restrictions apply.


WANG et al.: MICROBE-DISEASE ASSOCIATION PREDICTION USING RGCN THROUGH MICROBE-DRUG-DISEASE NETWORK 3355

_̸_


_̸_



Fig. 1. Flowchart of TNRGCN.


_̸_



_̸_



_̸_



_2) Tripartite Network Construction:_ After data processing,
we construct three adjacency matrices representing microbedisease association, microbe-drug association and disease-drug
association, respectively. _A_ _dm_ represents microbe-disease associations. If disease _i_ has association with microbe _j_, we set
_A_ _dm_ ( _i, j_ ) = 1, otherwise, _A_ _dm_ ( _i, j_ ) = 0. Similarly, _A_ _mu_ represents microbe-drug associations. If microbe _i_ has association
with drug _j_, _A_ _mu_ ( _i, j_ ) is set to 1, otherwise, _A_ _mu_ ( _i, j_ ) is set to
0. _A_ _du_ represents disease-drug associations. If disease _i_ has association with drug _j_, _A_ _du_ ( _i, j_ ) = 1, otherwise, _A_ _du_ ( _i, j_ ) = 0.

Since drugs are related to both microbes and diseases, we
build a microbe-drug-disease tripartite network _P_ to indirectly
increase microbe-disease associations by introducing drugs.

_3) Feature Initialization:_ During the learning process, we
first initialize the features of the nodes. By calculating the
similarity of microbes, diseases and drugs separately, and using
PCA to obtain the main features of the similarity as the initial
features.


1) Microbe similarity calculating
HMDAD and Disbiome contain the organs where microbes
live and the effects of microbes on them. In a previous article, authors calculated microbial function similarity based on their host
organs in the human body [27]. However, they only considered _̸_
the location of colonization, and did not consider the regulatory



_̸_



role of different microbes in the same organ. On this basis, we
calculate microbe function similarity based on the assumption
that microbes share stronger function similarities if they have
the same effects on the same organ. Specifically, if microbe _i_
and _j_ live in a same organ and have same regulation (increase or
decrease), we add 1 to _M_ _F_ ( _i, j_ ). After calculating the influence
of all microbes on the resident organs, we normalize microbe
function similarity according to (1):


_M_ _F_ ( _i, j_ ) = _[M]_ _[F]_ [(] _[i][,]_ _[j]_ [)] _[ −]_ [min][(] _[M]_ _[F]_ [)] (1)

max( _M_ _F_ ) _−_ min( _M_ _F_ )


where max( _M_ _F_ ) and min( _M_ _F_ ) arethemaximumandminimum
of matrix _M_ _F_, respectively.

2) Disease similarity calculating
The disease semantic similarity is calculated according to
Mesh database [28]. In Mesh, each disease is represented as
a Directed Acyclic Graph (DAG), including a disease and its
dependencies among all its ancestors. The contribution of every
element in a DAG according to (2) [29]:


_̸_



_D_ _con_ ( _d_ )=

_̸_



1 _if d_ = _D_
�max _{_ Δ _×D_ _con_ ( _d_ _[′]_ ) _|d_ _[′]_ _∈_ _childrenofd} ifd_ = _̸_ _D_

(2)



_̸_


Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:28:25 UTC from IEEE Xplore. Restrictions apply.


3356 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 20, NO. 6, NOVEMBER/DECEMBER 2023


Fig. 2. Flowchart of data processing.



where △ is a semantic contribution decay factor. The previous
article usually set △ to 0.5 [29].

The contribution to a certain disease is the sum of all the

contributions of all elements in its DAG:



(3)
_t∈V_ _d_ _[D]_ _[con]_ [(] _[t]_ [)]



_D_ _tc_ ( _d_ ) =



�



two similarities, we obtain the drug similarity network _G_ _u_,
which is shown as (7):


_G_ _u_ = ( _G_ _u_ 1 + _G_ _u_ 2 ) _/_ 2 (7)


PCA uses linear transformation to achieve dimensionality
reduction. The main idea of PCA is to map _n_ -dimensional
features to _k_ -dimensions. By retaining the first _k_ coordinate
axes containing most of the variance, and ignoring the feature
dimensions containing almost zero variance, the dimensionality
reduction processing of data features is realized. Due to different
dimensions of similarity between microbes, diseases and drugs,
we utilize PCA to reduce the dimensionality of each node and
feed them to RGCN as initial features. In this article, we set the

dimension of initial features to 128.


_4) Predicting_ _Microbe-Disease_ _Associations_ _Based_ _on_
_RGCN:_ GCN learns features by aggregating neighbors’ information through weighted summation of nodes in each layer.
The hidden layer _l_ -th representation of each node _i_ in GCN is
calculated as (8) [21]:


_H_ [(] _[l]_ [+1)] = _σ_ ( _L_ [�] _sym_ _W_ [(] _[l]_ [)] _H_ [(] _[l]_ [)] ) (8)


in (8), _H_ [(] _[l]_ [)] is the hidden features of nodes _l_ -th layer. _W_ [(] _[l]_ [)] is
the weight influence factor of nodes in the _l_ -th layer. In order
to prevent overfitting, the parameters in the method learning
process are limited by regularization terms. A commonly used
regularization term is the symmetric Laplace matrix _L_ [�] _sym_,
which is shown as (9),



where _V_ _d_ contains disease _d_ and all its ancestors.

For disease _i_ and _j_, the semantic similarity of them is calculated by the contribution of them. The formula is shown as (4):



_D_ _s_ ( _d_ _i_ _, d_ _j_ ) =



� _t∈V_ ( _d_ _i_ ) _∩V_ ( _d_ _j_ ) _[D]_ [(] _[i]_ [)] _con_ [(] _[t]_ [) +] _[ D]_ [(] _[j]_ [)] _con_ [(] _[t]_ [)] (4)

_D_ _tc_ ( _i_ ) + _D_ _tc_ ( _j_ )



3) Drug similarity calculating
We assume that when two drugs hare more of the same
neighbor nodes, their functions are more similar. Thus, we
calculate the drug GIP kernel similarity _G_ _u_ 1 according to the
disease-drug association network. The GIP similarity of drug _i_
and _j_ are calculate as (5) and (6) [30]:


_G_ _u_ 1 ( _i, j_ ) = exp( _−γ_ _u_ 1 _∥A_ _du_ ( _u_ ( _i_ )) _−_ _A_ _du_ ( _u_ ( _j_ )) _∥_ [2] ) (5)



_u_ _[′]_ 1 _[/]_ [1]



_γ_ u1 = _γ_ _[′]_



_N_ _u_



_N_ _u_
�


_i_ =1



_∥A_ _du_ ( _u_ ( _i_ )) _∥_ [2] (6)



where _A_ _du_ ( _u_ ( _i_ )) is the ith column of matrix _A_ _du_ . _γ_ _u_ 1 is a
normalized kernel bandwidth parameter affected by parameter
_γ_ _u_ _[′]_ 1 [. According to previous study, we set] _[ γ]_ _u_ _[′]_ 1 [to 1][ [30]][.] _[ N]_ _[u]_ [=]

1181, is the total number of drugs.

Similarily, the drug GIP kernel similarity _G_ _u_ 2 is calculated
based on microbe-drug association network. By combining the



�
_L_ _sym_ = � _D_ _[−]_ [1] 2




[1]

2 _A_ [�] _D_ [�] _[−]_ [1] 2



2 (9)



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:28:25 UTC from IEEE Xplore. Restrictions apply.


WANG et al.: MICROBE-DISEASE ASSOCIATION PREDICTION USING RGCN THROUGH MICROBE-DRUG-DISEASE NETWORK 3357


Fig. 3. Different convolution operations for GCN and RGCN in microbe-drug-disease tripartite network.



where _A_ [�] = _A_ + _I_ _N_, contains the adjacency information matrix
of the node and its own connection information. _D_ [�] is the degree
matrix of _A_ [�] . _σ_ is an activation function that passes information
from one layer to the next layer.

Since the loop of the node itself is not included in the microbedrug-disease tripartite network, the formula of GCN applied to
the network is shown in (10):



operation of it is calculated as (12) [25]:



_h_ [(] _i_ _[l]_ [+1)] = _σ_



��



_r∈R_



0 [(] _[l]_ [)] _[h]_ [(] _i_ _[l]_ [)]



_i_



�




[(] _j_ _[l]_ [)] + _W_ 0 [(] _[l]_ [)]



�



_j∈N_ _[r]_



_i_



_C_ _i,r_ _W_ _r_ [(] _[l]_ [)]



_r_ [(] _[l]_ [)] _[h]_ [(] _[l]_ [)]



_H_ [(] _[l]_ [+1)] = _σ_ ( _D_ _[−]_ [1] 2




[1]

2 _AD_ _[−]_ [1] 2



2 _W_ [(] _[l]_ [)] _H_ [(] _[l]_ [)] ) (10)



where _D_ is the degree matrix of _A_ .

Specifically, convolution process in _l_ -th layer is as follows:



�



_h_ [(] _i_ _[l]_ [+1)] = _σ_



��



_j∈N_ _i_ _[C]_ _[ij]_ _[W]_ [ (] _[l]_ [)] _[h]_ _j_ [(] _[l]_ [)]



(11)



(12)
where _h_ [(] _i_ _[l]_ [)] is the hidden features of node _i_ in the _l_ -th layer. _r ∈_ _R_

represents the type of edges. _N_ _i_ _[r]_ [includes all neighbors of node] _[ i]_

under relation _r_ . _C_ _i,r_ = 1 _/_ ( _N_ _i_ _[r]_ [)][, is a regularization term.] _[ W]_ _r_ [ (] _[l]_ [)]

is the weight corresponding to the relation _r_ in _l_ -th layer. _σ_ is an
activation function. In this article, we use ReLU as the activation

function.


RGCN considers both edge types and edge orientations, so
in microbe-drug-disease tripartite network, we consider 6 types
of edges including two directions: “microbe-influence-disease
(in/out)”,“microbe-relate-drug(in/out)”and“drug-treat-disease
(in/out)”. In the process of aggregation, because the microbedrug-disease tripartite network has no self-loop edges, the convolutional operation only accumulates all the features from the
neighbor nodes. The different convolution operations for GCN
andRGCNinmicrobe-drug-diseasetripartitenetworkareshown
in Fig. 3.

In the course of the experiment, we use two layers of RGCN
( _l_ = 2), to learn node features in the microbe-drug-disease
tripartite network. After that, we get the prediction score by
calculating the node features of each microbe and the disease
with dot product.

We use Adam Optimizer [31] to train the models by optimizing the cross entropy loss function. The formula of cross-entropy



where _N_ _i_ includes all neighbors of node _i_, _h_ [(] _j_ _[l]_ [)] is the hidden



_−_ [1]



_−_ [1]



features of node _j_ in _l_ -th layer. _C_ _ij_ = _N_ _i_ 2 _N_ _j_ 2, is the product

of the square root of the node degree.



features of node _j_ in _l_ -th layer. _C_ _ij_ = _N_



_i_ 2 _N_



GCN regards all neighbor nodes as the same type when aggregating node information. All neighbor nodes in layer _l_ share
one weight _W_ [(] _[l]_ [)] . On the contrary, RGCN considers different
types of nodes and the connection direction with the current
node. different nodes and connection directions share different

weights, and only edges of the same association type can use
the same weight. Specifically, for a node _i_, the convolutional



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:28:25 UTC from IEEE Xplore. Restrictions apply.


3358 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 20, NO. 6, NOVEMBER/DECEMBER 2023



TABLE II

M ETHOD P ERFORMANCE OF D IFFERENT D IMENSIONS


loss is shown as (13):



_Loss_ =



�

( _d_ _i_ _,m_ _j_ ) _∈E_



_−label_ ( _d_ _i_ _, m_ _j_ ) log( _p_ ( _d_ _i_ _, m_ _j_ ))



_−_ (1 _−_ _label_ ( _d_ _i_ _, m_ _j_ )) log(1 _−_ _p_ ( _d_ _i_ _, m_ _j_ )) (13)


where _E_ contains all edges where microbes and diseases are
connected. _label_ ( _d_ _i_ _, m_ _j_ ) is the real label of the connection
between disease _i_ and microbe _j_ . _p_ ( _d_ _i_ _, m_ _j_ ) is the predicted score
between disease _i_ and microbe _j_ .


III. E XPERIMENTS AND R ESULTS


In order to evaluate the prediction performance of TNRGCN,
we implement 5-fold cross validation. We randomly devide all
microbe-disease associations into five groups. Each group is
treated as test samples, while others are training samples. In
this validation, every confirmed microbe-disease association is
regarded as positive sample. In order to balance the positive and
negative samples, we randomly select unconfirmed microbedisease associations equal to positive samples as negative samples. In order to avoid the result deviation caused by different
sample segmentation, we run the cross validation for 10 times
andaveragethescores.Accordingtopredictedscores,weplotreceiver operating characteristic (ROC) curve and precision-recall
curve, and the performance of the method is evaluated by the
area under the receiver operating characteristic (AUC) and the
area of precision recall curve (AUPR).


_A. Parameter Selection_


We consider the feature dimensions of two layers to select the
best parameter combination for this method. We take different
dimensions between 32 and 128 to analyze the parameters.
According to Table II, the dimensions of feature are set to 128
and 64 in the first layer and the second layer, respectively.


_B. Model Analysis_


In order to evaluate the influence of different processes in
this method, we compare TNRGCN with its different variants,
which are as follows:


TNRGCNRF: it initializes features randomly.
BNRGCN: it utilizes two-layer RGCN to predict microbedisease associations on microbe-disease bipartite network, without drugs.

ONRGCN: it utilizes one-layer RGCN to predict microbedisease associations on microbe-drug-disease tripartite network.



TABLE III

P REDICTION P ERFORMANCE OF TNRGCN AND ITS V ARIANTS


TABLE IV

D ETAILS OF HMDAD AND D ISBIOME


TNGCN: it utilizes two-layer GCNtopredict microbe-disease
associations on microbe-drug-disease tripartite network.

As shown in Table III, the predictive performance is higher
when it uses similarities as initial features and adds drug nodes.
Compared with one-layer RGCN, two-layer RGCN can achieve
better performance. This is because only the first-order neighbor
information of the node is aggregated by using one-layer RGCN.
Through two layers RGCN, nodes can learn more neighbor information. Thus, we can see that the indirect connection between
nodes has an important impact on the prediction performance.

Besides, we compare the impact of using RGCN and GCN on
prediction performance. From the results, it can be seen that the
prediction results of RGCN are significantly higher than that of
GCN. The low prediction performance of GCN may be due to
the fact that the method does not consider the type of edges, and
the connection between microbe-drug and disease-drugs leads
to biased prediction weights. Instead, RGCN considers different
types of edge weights, and is more suitable for networks with
multiple types of nodes in reality.

In order to evaluate the adaptive performance of TNRGCN,
we used five-fold cross-validation on HMDAD and Disbiome,
respectively. The datasets and results are shown in Table IV.

According to Table IV, we can see that TNRGCN is equally
suitable fo prediction on small datasets. Its AUC value on
HMDAD reaches 0.9332. We believe that this is because the

HMDAD is small, so it has relatively more associations and
better prediction results.


_C. Comparison With Other Methods_


We compare TNRGCN with nine methods under 5-fold cross
validation. These methods include BRWMDA [32], BDSILP

[29], BiRWHMDA [33], KATZHMDA [11], NTSHMDA [34],
KATZBNRA [12], NCPHMDA [35], PBHMDA [36], and NCPLP [37]. According to Fig. 4 and Table V, the AUC and AUPR
values of TNRGCN are both the highest.

By comparison, we can see that most baseline methods perform poorly. We believe that this is due to the sparseness of
the huge biological network. Most methods pass information
through matrix operations, resulting in loss of information.
RGCN only considers edges and transmits information through



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:28:25 UTC from IEEE Xplore. Restrictions apply.


WANG et al.: MICROBE-DISEASE ASSOCIATION PREDICTION USING RGCN THROUGH MICROBE-DRUG-DISEASE NETWORK 3359


Fig. 6. Distribution of AUC value for all diseases in 5-fold cross validation.


Fig. 4. ROC curve of ten methods in 5-fold cross validation.


TABLE VI

P AIRWISE T-H YPOTHESIS T ESTING B ETWEEN TNRGCN AND O THER M ETHODS

TABLE V

AUPR P REDICTED BY T EN M ETHODS IN 5-F OLD C ROSS V ALIDATION


TABLE VII

T OP -10 M ICROBES A SSOCIATED W ITH T2D


Fig.5. Thenumberofcorrectlyassociationspredictedbytenmethodsin5-fold
cross validation.

IV. C ASE S TUDIES



edges, which effectively avoids the situation of data loss caused
by matrix operations.

Then, we compare the prediction scores of these methods
in the 1000 microbe-disease pairs with the highest scores. As
shown in Fig. 5, the number od confirmed microbe-diseases
associations predicted by is more among ten methods.

We also calculate the AUC value for each individual disease.

Fig. 6 showsthatAUCsformostdiseasespredictedbyTNRGCN
are above 0.8, with the higher mean and median value compared
with others. As shown in Table VI, we performed t-hypothesis
tests on different methods to test the degree of difference in
the mean values of TNRGCN and other methods. The _p_ -values
are all less than 0.05, indicating that TNRGCN is significantly
different from other methods.



We implement case studies for T2D, Bipolar disorder and
Autism to evaluate the predictive performance of TNRGCN. After sorting scores of unconfirmed microbe-disease associations
in descending order, we select top 10 microbes corresponding
to each disease.


T2D is a chronic disease due to impaired insulin secretion,
which is accompanied by a series of health problems such
as kidney failure, cardiovascular disease and weakness [38].
In this article, we select T2D for case study. As shown in
Table VII, VIII out of top 10 potential microbes are confirmed.
For example, Ruminococcus can be regarded as taxonomic
biomarkers for elderly diabetic patients [39]. Compared with
nondiabetic individuals, the percentages of Porphyromonas and
Prevotella melaninogenica are lower, while Desulfovibrio is
enriched in T2D patients [40], [41], [42]. In T2D patients, the



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:28:25 UTC from IEEE Xplore. Restrictions apply.


3360 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 20, NO. 6, NOVEMBER/DECEMBER 2023



TABLE VIII

T OP -10 M ICROBES A SSOCIATED W ITH B IPOLAR D ISORDER


proportion of Rothia increases dramatically in oral microbiota

[43], the level of Gram-negative Enterobacteriaceae is enhanced
in mesenteric adipose [44]. Moreover, in the intestines of T2D
patients, the abundance of Anaerotruncus is lower, which is
increasing slightly during treatment [45]. What’s more, in the
analysis of T2D-associated gut microbiota, Barnesiella is found
to be associated with the incidence of Mongolian T2D [46].

Bipolar disorder includes bipolar I disorder and bipolar II
disorder, which is often characterized by manic episode or
depression and usually occurs in adolescence [47]. At present,
the pathogenesis of bipolar disorder is still unclear. In the case
study of bipolar disorder, 5 of the top 10 potential microbes are
confirmed, which is shown in Table VIII. For instance, through
comparing the differences of intestinal microbiota with healthy
participants, the number of Streptococcus and Bifidobacterium
increase significantly in in the bipolar disorder participants at
genus level [48]. By examining bacterial counts in fecal samples
from patients with bipolar disorder, studies found that there is
a negative correlation between lactobacillus counts and sleep,
which is beneficial to the improvement of sleep quality [49].
Research also found that the abundance of Enterobacteriaceae is

increased in bipolar disorder’s patients [50]. Although Prevotella
and Blautia are not proved to be directly related to bipolar
disorder, researches have found that they were all associated
with mental or behavioral dysfunction such as major depressive
disorder and schizophrenia [51], [52].

Autism is a generalized developmental disorder that combines
cognitive function, language function, and interpersonal social
communication with special pathologies, resulting in significant
difficulties in adapting to social life. Studies have demonstrated
a strong positive relationship between autism severity and gastrointestinal dysfunction severity [53]. The combination of diet
and probiotics to modulate the composition of gut microbiota
is beneficial for the treatment of children with autism [54],

[55]. Thus, we do a case study on autism. As shown in Table IX, VIII of the top 10 potential microbes predicted by
TNRGCN are confirmed. For example, in studies of microbiome
profiles autistic patients and controls, the relative abundance
of Dialister, Parabacteroides in autistic patients are lower than
neurotypical controls, while Klebsiella is significantly higher

[56], [57]. A study for fecal microbiota of children with autism
and healthy children found that at the genus level, the number of



TABLE IX

T OP -10 M ICROBES A SSOCIATED W ITH A UTISM


Haemophilus bacteria is reduced in children with autism compared with controls [58]. In addition, through high-throughput
sequencing of oral samples, the bacterial diversity observed
in children with autism was lower compared to controls. The
abundance of Actinomyces in the patient’s saliva was reduced

[59].


V. C ONCLUSION


Knowing associations between microbes and diseases is beneficial to disease diagnosis and treatment. In this article, we
propose a method for microbe-disease association prediction
called TNRGCN based on Tripartite Network and RGCN. First,
considering that HMDAD contains relatively few microbes, diseases and associations, we integrate HMDAD and Disbiome to
obtain more related information. More associations can improve
the performance of predictive method. Second, we introduce
the drug information from MDAD and CTD to increase the
indirect associations in the microbe-disease network, thus building a microbe-drug-disease tripartite network. Third, we utilize
RGCN on the microbe-drug-disease tripartite network to predict
potential microbe-disease associations, which can be applied
to heterogeneous networks containing different types of nodes
and edges. TNRGCN has a good performance in 5-fold cross
validation. Experiments of case studies further demonstrate the
predictive performance of TNRGCN.

However, the number of microbe-drug associations and
disease-drug associations involved is relatively small although
we have added drug related associations in data process. With
the establishment of more databases, we will integrate more
confirmed associations in future work. In addition, with the
in-depthstudyofmulti-omicssuchasgenomics [60],proteomics

[61] and transcriptomics [62], the combination of multi-omics
data may further excavate more biological information, which is
conducive to the prediction, diagnosis and treatment of diseases

[63].


R EFERENCES


[1] H. M. P. Consortium, “A framework for human microbiome research,”

_Nature,_ vol. 486, no. 7402, pp. 215–221, Jun. 14, 2012.

[2] L. Gao et al., “Oral microbiomes: More and more importance in oral cavity

and whole body,” _Protein Cell,_ vol. 9, no. 5, pp. 488–500, May 2018.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:28:25 UTC from IEEE Xplore. Restrictions apply.


WANG et al.: MICROBE-DISEASE ASSOCIATION PREDICTION USING RGCN THROUGH MICROBE-DRUG-DISEASE NETWORK 3361




[3] T. G. Dinan et al., “Collective unconscious: How gut microbes shape

human behavior,” _J. Psychiatr. Res.,_ vol. 63, pp. 1–9, Apr. 2015.

[4] V. B. Young, “The role of the microbiome in human health and disease:

An introduction for clinicians,” _Brit. Med. J.,_ vol. 356, no. j831, pp. 1–14,
Mar. 2017.

[5] L. Dethlefsen, M. McFall-Ngai, and D. A. Relman, “An ecological and

evolutionary perspective on human-microbe mutualism and disease,” _Na-_
_ture,_ vol. 449, no. 7164, pp. 811–818, Oct. 2007.

[6] K. Makki et al., “The Impact of dietary fiber on gut microbiota in host

health and disease,” _Cell Host Microbe,_ vol. 23, no. 6, pp. 705–715, Jun.
13, 2018.

[7] J. D. Forbes, G. Van Domselaar, and C. N. Bernstein, “The gut microbiota

in immune-mediated inflammatory diseases,” _Front. Microbiol.,_ vol. 7,
no. 1081, pp. 1–18, Jul. 11, 2016.

[8] D. Kashtanova et al., “Gut microbiota and vascular biomarkers in patients

without clinical cardiovascular diseases,” _Artery Res.,_ vol. 18, pp. 41–48,
Jun. 2017.

[9] W. Ma et al., “An analysis of human microbe-disease associations,” _Brief._

_Bioinf.,_ vol. 18, no. 1, pp. 85–97, Jan. 2017.

[10] Y.Janssensetal.,“Disbiomedatabase:Linkingthemicrobiometodisease,”

_BMC Microbiol.,_ vol. 18, no. 50, pp. 1–6, Jun. 4, 2018.

[11] X. Chen et al., “A novel approach based on KATZ measure to predict

associations of human microbiota with non-infectious diseases,” _Bioinfor-_
_matics,_ vol. 33, no. 5, pp. 733–739, Mar. 1, 2017.

[12] H. Li et al., “A novel human microbe-disease association prediction

method based on the bidirectional weighted network,” _Front. Microbiol.,_
vol. 10, no. 676, pp. 1–13, Apr. 9, 2019.

[13] C. Jiang, M. Tang, J. Shuting, W. Huang, and X. Liu, “KGNMDA: A

knowledge graph neural network method for predicting microbe-disease
associations,” _IEEE/ACM Trans. Comput. Biol. Bioinf._, early access, Jun.
[20, 2022, doi: 10.1109/TCBB.2022.3184362.](https://dx.doi.org/10.1109/TCBB.2022.3184362)

[14] L. Peng et al., “Prioritizing human microbe-disease associations utilizing

a node-information-based link propagation method,” _IEEE Access,_ vol. 8,
pp. 31341–31349, 2020.

[15] M. F. Hua et al., “MVGCNMDA: Multi-view graph augmentation con
volutional network for uncovering disease-related microbes,” _Interdiscipl._
_Sci.-Comput. Life Sci.,_ vol. 14, no. 3, pp. 669–682, Sep. 2022.

[16] H. Li et al., “Identifying microbe-disease association based on a novel

back-propagation neural network model,” _IEEE-ACM Trans. Comput._
_Biol. Bioinf.,_ vol. 18, no. 6, pp. 2502–2513, Nov. 1, 2021.

[17] Y. Y. Wang, X. J. Lei, C. Lu, and Y. Pan, “Predicting microbe-disease

association based on multiple similarities and LINE algorithm,” _IEEE-_
_ACM Trans. Comput. Biol. Bioinf.,_ vol. 19, no. 4, pp. 2399–2408,
Jul./Aug. 2022.

[18] L. Dayun, L. Junyi, L. Yi, H. Qihua, and L. Deng, “MGATMDA: Pre
dicting microbe-disease associations via multi-component graph attention network,” _IEEE/ACM Trans. Comput. Biol. Bioinf._, vol. 19, no. 6,
[pp. 3578–3585, Sep. 2021, doi: 10.1109/TCBB.2021.3116318.](https://dx.doi.org/10.1109/TCBB.2021.3116318)

[19] W. Peng, M. Liu, W. Dai, T. Chen, Y. Fu, and Y. Pan, “Multi
view feature aggregation for predicting microbe-disease association,”
_IEEE/ACM Trans. Comput. Biol. Bioinf._, early access, Dec. 06, 2021,
[doi: 10.1109/TCBB.2021.3132611.](https://dx.doi.org/10.1109/TCBB.2021.3132611)

[20] Y. L. Chen and X. J. Lei, “Metapath aggregated graph neural network and

tripartite heterogeneous networks for microbe-disease prediction,” _Front._
_Microbiol._, vol. 13, May 2022, Art. no. 919380.

[21] T. N. Kip F and M. Welling, “Semi-supervised classification with graph

convolutional networks,” 2016, _arXiv:1609.02907_ .

[22] T. B. Mudiyanselage et al., “Predicting CircRNA disease associations

using novel node classification and link prediction models on graph
convolutional networks,” _Methods_, vol. 198, pp. 32–44, Feb. 2022.

[23] X. J. Lei, J. J. Tie, and Y. Pan, “Inferring metabolite-disease association

using graph convolutional networks,” _IEEE-ACM Trans. Comput. Biol._
_Bioinf._, vol. 19, no. 2, pp. 688–698, Mar./Apr. 2022.

[24] F. Wang et al., “Predicting drug-drug interactions by graph convolutional

network with multi-kernel,” _Brief. Bioinf._, vol. 23, no. 1, Jan. 17, 2022,
Art. no. bbab511.

[25] M. Schlichtkrull et al., “Modeling relational data with graph convolutional

networks,” in _Proc. Eur. Semantic Web Conf._, 2018, pp. 593–607.

[26] J. Pinero et al., “DisGeNET: A discovery platform for the dynamical

exploration of human diseases and their genes,” _Database_, vol. 2015, 2015,
Art. no. bav028.

[27] C. Yan, G. Duan, F.-X. Wu, Y. Pan, and J. Wang, “MCHMDA:Predicting

microbe-disease associations based on similarities and low-rank matrix
completion,” _IEEE-ACM Trans. Comput. Biol. Bioinf._, vol. 18, no. 2,
pp. 611–620, Mar./Apr. 2021.




[28] I. K. Dhammi and S. Kumar, “Medical subject headings (MeSH) terms,”

_Indian J. Orthopaedics_, vol. 48, no. 5, pp. 443–444, Sep./Oct. 2014.

[29] W. Zhang, W. T. Yang, X. T. Lu, F. Huang, and F. Luo, “The bi-direction

similarityintegrationmethodforpredictingmicrobe-diseaseassociations,”
_IEEE Access_, vol. 6, pp. 38052–38061, 2018.

[30] T. van Laarhoven, S. B. Nabuurs, and E. Marchiori, “Gaussian interaction

profile kernels for predicting drug-target interaction,” _Bioinformatics_,
vol. 27, no. 21, pp. 3036–3043, Nov. 1, 2011.

[31] D. Kingma and J. J. C. S. Ba, “Adam: A method for stochastic optimiza
tion,” Dec. 22, 2014, _arXiv:1412.6980_ .

[32] C. Yan, G. H. Duan, F. X. Wu, Y. Pan, and J. Wang, “BRWMDA: Predicting

microbe-disease associations based on similarities and bi-random walk on
disease and microbe networks,” _IEEE-ACM Trans. Comput. Biol. Bioinf._,
vol. 17, no. 5, pp. 1595–1604, Sep. 1, 2020.

[33] S. Zou, J. P. Zhang, and Z. P. Zhang, “A novel approach for predicting

microbe-disease associations by bi-random walk on the heterogeneous
network,” _Plos One_, vol. 12, no. 9, Sep. 7, 2017, Art. no. e0184394.

[34] J. W. Luo and Y. H. Long, “NTSHMDA: Prediction of human microbe
disease association based on random walk by integrating network topological similarity,” _IEEE-ACM Trans. Comput. Biol. Bioinf._, vol. 17, no. 4,
pp. 1341–1351, Jul./Aug. 2020.

[35] W.Z.Bao,Z.C.Jiang,andD.S.Huang,“Novelhumanmicrobe-diseaseas
sociation prediction using network consistency projection,” _BMC Bioinf._,
vol. 18, no. 543, pp. 173–181, Dec. 28, 2017.

[36] Z. A. Huang et al., “PBHMDA: Path-based human microbe-disease asso
ciation prediction,” _Front. Microbiol._, vol. 8, Feb. 22, 2017, Art. no. 233.

[37] M. M. Yin, J. X. Liu, Y. L. Gao, X.-Z. Kong, and C.-H. Zheng, “NCPLP:

A novel approach for predicting microbe-associated diseases with network consistency projection and label propagation,” _IEEE Trans. Cybern._,
vol. 52, no. 6, pp. 5079–5087, Jun. 2022.

[38] H. C. Gerstein et al., “Effects of intensive glucose lowering in type 2

diabetes,” _New England J. Med._, vol. 358, no. 24, pp. 2545–2559, Jun.
12, 2008.

[39] A. O. Afolayan et al., “Insights into the gut microbiota of Nigerian elderly

with type 2 diabetes and non-diabetic elderly persons,” _Heliyon_, vol. 6,
no. 5, May 2020, Art. no. e03971.

[40] R. C. V. Casarin et al., “Subgingival biodiversity in subjects with un
controlled type-2 diabetes and chronic periodontitis,” _J. Periodontal Res._,
vol. 48, no. 1, pp. 30–36, Feb. 2013.

[41] Y. K. Liu et al., “A salivary microbiome-based auxiliary diagnostic model

for type 2 diabetes mellitus,” _Arch. Oral Biol._, vol. 126, Jun. 2021,
Art. no. 105118.

[42] J. J. Qin et al., “A metagenome-wide association study of gut microbiota

in type 2 diabetes,” _Nature_, vol. 490, pp. 55–60, Oct. 4, 2012.

[43] R. Anbalagan et al., “Next generation sequencing of oral microbiota in

Type 2 diabetes mellitus prior to and after neem stick usage and correlation
with serum monocyte chemoattractant-1,” _Diabetes Res. Clin. Pract._,
vol. 130, pp. 204–210, Aug. 2017.

[44] F. F. Anhe et al., “Type 2 diabetes influences bacterial tissue compartmen
talisation in human obesity,” _Nature Metab._, vol. 2, no. 3, pp. 233–242,
Mar. 2020.

[45] M. Zhang et al., “The gut microbiome can be used to predict the

gastrointestinal response and efficacy of lung cancer patients undergoing chemotherapy,” _Ann. Palliat. Med._, vol. 9, no. 6, pp. 4211–4227,
Nov. 2020.

[46] S. C. Li et al., “Comparative analysis of type 2 diabetes-associated gut

microbiota between Han and Mongolian people,” _J. Microbiol._, vol. 59,
no. 7, pp. 693–701, Jul. 2021.

[47] R. S. McIntyre et al., “Bipolar disorders,” _Lancet_, vol. 396, no. 10265,

pp. 1841–1856, Dec. 2020.

[48] H. Rong et al., “Similarly in depression, nuances of gut microbiota:

Evidences from a shotgun metagenomics sequencing study on major depressive disorder versus bipolar disorder with current major
depressive episode patients,” _J. Psychiatr. Res._, vol. 113, pp. 90–99,
Jun. 2019.

[49] E. Aizawa et al., “Bifidobacterium and lactobacillus counts in the gut

microbiota of patients with bipolar disorder and healthy controls,” _Front._
_Psychiatry_, vol. 9, no. 730, pp. 1–8, Jan. 2019.

[50] T. T. Huang et al., “Current understanding of gut microbiota in mood

disorders: An update of human studies,” _Front. Genet._, vol. 10, no. 98,
pp. 1–12, Feb. 2019.

[51] McGuinness A. J. et al., “A systematic review of gut microbiota compo
sition in observational studies of major depressive disorder, bipolar disorder and schizophrenia,” _Mol. Psychiatry_, vol. 27, no. 4, pp. 1920–1935,
Apr. 2022.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:28:25 UTC from IEEE Xplore. Restrictions apply.


3362 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 20, NO. 6, NOVEMBER/DECEMBER 2023




[52] K. Yamaoka, N. Uotsu, and E. Hoshino, “Relationship between psychoso
cial stress-induced prefrontal cortex activity and gut microbiota in healthy
Participants-A functional near-infrared spectroscopy study,” _Neurobiol._
_Stress_, vol. 20, no. 100479, pp. 1–14, Sep. 2022.

[53] A. Tomova et al., “Gastrointestinal microbiota in children with autism in

Slovakia,” _Physiol. Behav._, vol. 138, pp. 179–187, Jan. 2015.

[54] R. Grimaldi et al., “A prebiotic intervention study in children with autism

spectrum disorders (ASDs),” _Microbiome_, vol. 6, no. 1, Aug. 2, 2018,
Art. no. 133.

[55] Y.-Q. Li et al., “Effect of probiotics combined with applied behavior

analysis in the treatment of children with autism spectrum disorder: A
prospective randomized controlled trial,” _Randomized Controlled Trial_,
vol. 23, no. 11, pp. 1103–1110, Nov. 2021.

[56] F. Ye et al., “Comparison of gut microbiota in autism spectrum disorders

and neurotypical boys in China: A case-control study,” _Synthetic Syst._
_Biotechnol._, vol. 6, no. 2, pp. 120–126, Jun. 2021.

[57] M. Y. Xu et al., “Association between gut microbiota and autism spec
trum disorder: A systematic review and meta-analysis,” _Front. Psychiatry_,
vol. 10, no. 473, pp. 1–11, Jul. 17, 2019.

[58] R. Zou et al., “Changes in the gut microbiota of children with autism

spectrum disorder,” _Autism Res._, vol. 13, no. 9, pp. 1614–1625, Sep. 2020.

[59] Y. A. Qiao et al., “Alterations of oral microbiota distinguish children

with autism spectrum disorders from healthy controls,” _Sci. Rep.s_, vol. 8,
no. 1597, pp. 1–12, Jan. 25, 2018.

[60] P. Mobadersany et al., “Predicting cancer outcomes from histology and

genomics using convolutional networks,” _Proc. Nat. Acad. Sci. USA_,
vol. 115, no. 13, pp. E2970–E2979, Mar. 2018.

[61] M. Zeng et al., “A deep learning framework for identifying essential pro
teins by integrating multiple types of biological information,” _IEEE-ACM_
_Trans. Comput. Biol. Bioinf._, vol. 18, no. 1, pp. 296–305, Jan. 2021.

[62] X.J.Leietal.,“Acomprehensivesurveyoncomputationalmethodsofnon
coding RNA and disease association prediction,” _Brief. Bioinf._, vol. 22,
no. 4, Jul. 2021, Art. no. bbaa350.

[63] Y. Pan, X. J. Lei, and Y. C. Zhang, “Association predictions of genomics,

proteinomics, transcriptomics, microbiome, metabolomics, pathomics, radiomics, drug, symptoms, environment factor, and disease networks: A
comprehensive approach,” _Med. Res. Rev._, vol. 42, no. 1, pp. 441–461,
Jan. 2022.



**Yueyue Wang** received the BS degree from the
School of Computer Science, Shaanxi Normal University, Xi’an, China, in 2019, where she is currently
working toward the MS degree. Her current research
interests include bioinformatics, data mining, and
deep learning.


**Xiujuan Lei** received the MS and PhD degrees from
NorthwesternPolytechnicalUniversity,Xi’an,China,
in 2001 and 2005, respectively. She is currently a professor with the School of Computer Science, Shaanxi
Normal University, Xi’an. Her research interests include bioinformatics, swarm intelligent optimization,
data mining, and deep learning.


**Yi Pan** received the BEng and MEng degrees in computer engineering from Tsinghua University, China,
in 1982 and 1984, respectively, and the PhD degree in
computer science from the University of Pittsburgh,
USA, in 1991. He is currently a professor with the
Faculty of Computer Science and Control Engineering, Shenzhen Institute of Advanced Technology,
Chinese Academy of Sciences. He has served as
chairofComputerScienceDepartment,GeorgiaState
University during 2005-2020. His current research
interests mainly include bioinformatics and health
informatics using Big Data analytics, cloud computing, and machine learning
technologies.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:28:25 UTC from IEEE Xplore. Restrictions apply.


