Chinese Journal of Electronics

vol. 33, no. 1, pp. 231–244, January 2024
https://doi.org/10.23919/cje.2022.00.384


**RESEARCH ARTICLE**

# **Drug-Target Interactions Prediction Based on** **Signed Heterogeneous Graph Neural Networks**


Ming CHEN [1], Yajian JIANG [1], Xiujuan LEI [2], Yi PAN [3], Chunyan JI [4], and Wei JIANG [1]


1. _College of Information Science and Engineering_, _Hunan Normal University_, _Changsha 410081_, _China_
2. _School of Computer Science_, _Shaanxi Normal University_, _Xi’an 710119_, _China_
3. _Faculty of Computer Science and Control Engineering_, _Shenzhen Institute of Advanced Technology_,
_Chinese Academy of Sciences_, _Shenzhen 518055_, _China_
4. _Computer Science Department_, _BNU-HKBU United International College_, _Zhuhai 519087_, _China_


Corresponding author: Yi PAN, Email: yi.pan@siat.ac.cn
Manuscript Received November 10, 2022; Accepted March 21, 2023
Copyright © 2024 Chinese Institute of Electronics


**Abstract —** Drug-target interactions (DTIs) prediction plays an important role in the process of drug discovery.
Most computational methods treat it as a binary prediction problem, determining whether there are connections between drugs and targets while ignoring relational types information. Considering the positive or negative effects of
DTIs will facilitate the study on comprehensive mechanisms of multiple drugs on a common target, in this work, we
model DTIs on signed heterogeneous networks, through categorizing interaction patterns of DTIs and additionally extracting interactions within drug pairs and target protein pairs. We propose signed heterogeneous graph neural networks (SHGNNs), further put forward an end-to-end framework for signed DTIs prediction, called SHGNN-DTI,
which not only adapts to signed bipartite networks, but also could naturally incorporate auxiliary information from
drug-drug interactions (DDIs) and protein-protein interactions (PPIs). For the framework, we solve the message passing and aggregation problem on signed DTI networks, and consider different training modes on the whole networks
consisting of DTIs, DDIs and PPIs. Experiments are conducted on two datasets extracted from DrugBank and related databases, under different settings of initial inputs, embedding dimensions and training modes. The prediction results show excellent performance in terms of metric indicators, and the feasibility is further verified by the case study
with two drugs on breast cancer.
**Keywords —** Drug-target interactions, Signed heterogeneous network, Link sign prediction, Graph neural networks.

**Citation —** Ming CHEN, Yajian JIANG, Xiujuan LEI, _et al_ ., “Drug-Target Interactions Prediction Based on
Signed Heterogeneous Graph Neural Networks,” _Chinese Journal of Electronics_, vol. 33, no. 1, pp. 231–244, 2024.
doi: 10.23919/cje.2022.00.384.



**I. Introduction**


The prediction of drug-target interactions (DTIs) is
of great significance to the fields of drug design and drug
development. However, traditional biological experiments
are time-consuming and cost-effective, so it has prompted more people to pay their attention to the use of computers to assist in predicting DTIs [1]–[6].

At present, there are many machine learning methods for DTIs prediction. Traditional approaches, including network/graph embedding models, matrix factorization or feature-based methods, either focus on extracting
associations between drugs and proteins or depend on in

Associate Editor: Prof. Hongkai XIONG, Shanghai Jiao Tong University.



formation from node attributes. Compared with traditional methods, as a deep learning branch for irregular
data, graph neural networks (GNNs) have shown excellent performance in mining biomedical networks [7]–[11].
GNNs can make better use of both node characteristics
and network structures [7], and their powerful computing platforms [12] provide convenient model training.
Currently, GNNs have been the popular methods in
DTIs prediction [13], [14].

Most of the existing DTIs prediction methods ignore the specific interaction patterns between drugs and
targets, however, enriching a drug-target network with
information of functional nature like the sign of the in

232 Chinese Journal of Electronics, vol. 33, no. 1



teractions allows to explore a series of network properties of key importance in the context of computational
drug combinatorics [15], [16]. For example, by attaching
signs to the mechanisms of action, we are able to quantify the amount of synergism (i.e., when coherence prevails over incoherence) in a drug pair, and to classify all
drug pairs accordingly.

In this study, we construct signed networks of drugs
and targets according to pharmacological DTIs. Additionally, we extract information from interactions between
drugs and their associated target proteins. But, how to
deal with DTIs prediction on signed heterogeneous networks is still an open issue. To solve the problem, we
dedicate signed heterogeneous graph neural networks
(SHGNNs), further put forward a method for signed
DTIs prediction, called SHGNN-DTI.


**1. Related work**


There are a large amount of DTIs prediction studies.
Their methods are roughly divided into two categories:
traditional approaches and deep learning methods. In
terms of the prediction form, existing studies either explore whether the drug can interact with the target or
exploit more informative details of DTIs.

Many machine learning directions have been applied
to DTIs prediction and most of them either focus on
node features or links between nodes. The former calcu
lates node representation vectors of drugs and targets respectively and then trains a discriminator to predict DTIs.
Typical node embedding methods include graph/network
based embedding [17] etc. And classifiers such as support vector machine [18] and random forest [19], can act
as a discriminator. However, traditional node representation based approaches need two-stage training process
and cannot fully extract deep and complex associations
between drugs and targets. Another direction is linkbased ways including matrix factorization (MF) [20] and
random walk (RW) [21]. MF uses the product of two or
more low-rank matrices to approximate the association
matrix, while RW [21] is widely used in methods based
on graph theories. But, they commonly ignore explicit
and natural integration of node features and graph structures. Other directions, such as hybrid methods [22] and
ensemble learning, also have potentials in DTIs prediction. However, the lack of computing platform support
hampers the development of traditional methods on
large-scale data.

Recently, GNNs, as a branch of deep learning dedicated for irregular data, show excellent performance in
graph mining research [7]–[11], [23]. Compared with traditional models, GNNs not only make better use of node
characteristics and network structures [7], but also inherit end-to-end learning frameworks from deep learning; In
addition, there are powerful computing platforms developed [12]. Until now, many kinds of GNNs have been explored to tackle with the heterogeneous graphs [9], [24].
Spectral convolution based GNNs [25], attention based



GNNs [13], meta-path based GNNs [14] have shown their
applications on DTIs networks.

However, most current studies take DTIs prediction
as a binary classification problem [1], [22] which lays a
good foundation for the initial stage of drug development. In order to further accelerate the process of drug
discovery, it is far from enough to explore whether the
drug can interact with the target [17]. To overcome the
disadvantages of binary classification, some researchers
turn to exploit more detailed information of DTIs. When
browsing DrugBank [26], we find more than 35 mechanisms of action modes of DTIs and most of them can be

reasonably classified as positive or negative on targets. It
is reported that signed DTIs facilitate the study of the
comprehensive mechanism of drug combinatorics [15], [16],

[27], which is the main motivation of this paper.

There are some signed GNNs extended from unsigned and raw models, but it is still open whether they
are applicable to signed DTIs networks. One extension is
from spectral GNNs, which explores frequency analysis
with signed graph Laplacians [28], [29] and commonly
adapts to homogeneous graphs. Another direction is to
combine spatial GNNs with a notable social theory on
signed graphs such as balance theory.

Typical models include signed graph convolutional
network (SGCN) [30] and signed bipartite graph neural
network (SBGNN) [31]. To our best knowledge, existing
signed GNNs focus on either signed graphs with the same
entities [30] or simple bipartite graphs [31] separately.
Hence, they are unable to handle the complexities
brought by both drug-drug interactions (DDIs) and protein-protein interactions (PPIs) information. In addition,
it is still open whether the most commonly used balance
theory is applicable to signed DTIs networks. Actually,
unsatisfactory results are observed in our initial attempt
to directly apply some of these models on signed DTIs
bipartite networks.


**2. Our contributions**


In this study, we first model DTIs on signed heterogeneous networks, and then put forward a signed GNN
framework for DTIs prediction by designing information
propagation process and considering different training
settings. Our contributions are concluded as follows.

  - In terms of interaction modes between drugs and
targets, we model DTIs prediction on signed heterogeneous networks. DTIs are categorized and expressed as
signed links [15], [16], [27], according to pharmacological
drug-target interactions in DrugBank [26] database, and
hence form a signed bipartite network. In addition, we
construct a two-layer signed heterogeneous network, by
extracting DDIs, drug chemical features from PubChem

[32], and PPIs from String [33].

  - We propose SHGNN-DTI, to realize the end-toend DTIs sign prediction, which takes drug-target embedding pair as the input and jointly train a DTIs discriminator. To deal with message passing and aggrega

Drug-Target Interactions Prediction Based on Signed Heterogeneous Graph Neural Networks 233



tion on signed DTIs networks, we dedicate signed aggregation on bipartite graphs. Furthermore, we propose a
three-module framework to handle two additional un
signed graphs, and train it with considerations of either
cooperative or independent mode and whether sharing
weights. It is noted that our SHGNN-DTI not only adapts
to signed bipartite networks, but also could naturally incorporate auxiliary information from DDIs and PPIs.

  - Comprehensive experiments on two DTIs networks are conducted to verify the validity of our prediction model. In terms of several performance metrics, its
performance greatly exceeds the baselines. The effect of
initial features of nodes, training modes and embedding
dimensions are also discussed. Ablation study further illustrates the role of three modules. In addition, we provide the case study on two drugs for breast cancer, and
find that seven new DTIs out of their Top-10 links have
support from other literature, which verifies the feasibility of SHGNNs-based DTIs prediction.


**II. Problem Descriptions**


In this section, we model DTIs on signed heterogeneous networks. Firstly, the DTIs is modeled on a signed
bipartite network, and then extended onto a two-level
signed heterogeneous network.

When browsing DrugBank [26], we see many possible mechanisms of drug-target actions. For example, a
drug activates or inhibits a target acting as an agonist or
antagonist, as a potentiator or blocker, as an inducer or
suppressor, and so on. There are different types of targets such as proteins, macro molecules, nucleic acids, and
small molecules, etc. Drug-target actions can be roughly
divided into positive or negative relations [15] and naturally represented as signed links. As shown in Table 1,
the action modes of drugs and targets are represented as
signed links [15]. Activator, agonist and other positive
types are classified into positive effects and represented
by the label +1, whereas types with negative effects like
inhibitor and antagonist are labeled with −1. For example, Pilocarpine is an activator of Muscarinic acetylcholine receptor, and the link is represented by +1. The
drug Bivalirudin, as an inhibitor of Prothrombin, inhibits thrombin action by binding the catalytic site of
thrombin to the external site of anion binding, and the



link is indicated by −1.

It is noted that some modes are labeled with 0, e.g.,
“modulators”, “binder”, “cleavage”, because they are impossible to classify with a sign, and hence cannot be included in our analysis. In this case, we do not construct
an edge between the drug and the target.

In this study, we further extract information from
interactions between drugs and their associated target
proteins, to model a two-level signed heterogeneous network. For all the drugs, we also crawl their features and
interaction information from DrugBank [26]. Meanwhile,
we search the known human protein interaction data in
String [33] database to obtain the interaction information among targets. For DDIs and PPIs, we use 1 to indicate the known interaction and 0 for the others.


Figure 1 shows the DTIs prediction on signed networks. The problem is to determine the sign of drug-target
action, i.e., ? _∈{_ + _, −}_ . Given a set of drugs _D_ =( _D_ 1 _, D_ 2 _,_

_. . ., D_ _n_ ), a set of targets _T_ =( _T_ 1 _, T_ 2 _, . . .,T_ _m_ ) and their

signed edges _E_ DT = _{e_ _ij_ _, i_ =1 _,_ 2 _, . . ., n, j_ =1 _,,_ 2 _, . . ., m}_,
we predict new DTIs on a signed bipartite network

_G_ DT = ( _D, T, E_ DT ) (shown in Figure 1(a)) according to

Table 1. Additionally, with the help of unsigned DDIs
network _G_ _D_ = ( _D,_ _**A**_ _D_ _, E_ _D_ ) and PPIs network _G_ _T_ =

( _T,_ _**A**_ _T_ _, E_ _T_ ), the problem becomes DTIs prediction on

the two-level signed heterogeneous network as shown in
Figure 1(b).



action, i.e., ? _∈{_ + _, −}_ . Given a set of drugs _D_ =( _D_ 1 _, D_ 2 _,_

_. . ., D_ _n_ ), a set of targets _T_ =( _T_ 1 _, T_ 2 _, . . .,T_ _m_ ) and their

signed edges _E_ DT = _{e_ _ij_ _, i_ =1 _,_ 2 _, . . ., n, j_ =1 _,,_ 2 _, . . ., m}_,



_G_ DT = ( _D, T, E_ DT )



network _G_ _D_ = ( _D,_ _**A**_ _D_ _, E_ _D_ ) and PPIs network _G_ _T_ =

( _T,_ _**A**_ _T_ _, E_ _T_ ), the problem becomes DTIs prediction on

















(a) A signed bipartite network (b) A two-level signed heterogeneous network


**Figure 1** DTIs prediction on signed networks. The problem is to determine the link sign between a drug and a target for (a) a signed
bipartite network and (b) a two-level signed heterogeneous net
work.


**III. SHGNNs-Based DTIs Prediction**

**Methods**


The positive and negative edges represent two polarization relations between drugs and targets, but coexistence of negativity and heterogeneity brings challenges to
the extension of the existing GNNs. As we know, current popular GNNs-based approaches mainly work on
unsigned graphs, which will aggregate the neighbor information in the same way for both positive and negative
edges. Even though some signed GNNs have been developed, the graph was assumed to have the same kind of
entities or be simple bipartite graphs, and thus, they are
not directly applicable to handle the complexities that
are brought by drug pairs and target pairs.








|Table 1 Drug modes|of action and edge signs|
|---|---|
|Edge sign|Action modes in DrugBank|
|Positive (+)|Agonist; partial agonist; activator; stimulator;<br>inducer; positive allosteric modulator;<br>potentiator; positive modulator|
|Negative (−)|Inhibitor; inhibitory allosteric modulator;<br>inhibitor competitive; antagonist; partial<br>antagonist; negative modulator; inverse<br>agonist; blocker; suppressor; desensitize the<br>target; neutralizer; reducer|
|Not classifiable (0)|Otherwise|


234 Chinese Journal of Electronics, vol. 33, no. 1



In this section, we propose SHGNNs for DTIs prediction on drug-target networks. SHGNN-DTI works following the framework described in Figure 2. Firstly, it
employs SHGNN to obtain embedding results of drugs
and targets. Secondly, a drug-target pair, i.e., ( _**z**_ _D_ _i_ _,_ _**z**_ _T_ _j_ ),
which is concatenated from above embeddings, is taken
as an input of a DTIs discriminator to predict the DTI



sign. Our SHGNNs and the DTIs discriminator are jointly trained with the loss function described in the last
sub-section. It is noted that, the framework still works
on signed bipartite networks via ignoring Module2 and
Module3 directly.

To enhance the readability and understandability,
Table 2 provides key symbols used in SHGNN-DTI.







































**Figure 2** DTIs prediction based on SHGNNs. It employs a SHGNN model to obtain embedding results of drugs and targets, and then discriminates the sign of concatenated embeddings of a drug-target pair. The SHGNN and the discriminator are jointly trained with a loss
function. When SHGNN works on only the DTIs subnetwork, Module2 and Module3 are missed.


|Table 2 Key symbols in SHGNNs|Col2|
|---|---|
|Symbols|Descriptions|
|**_h_**(0)<br>_Di_ **_h_**(0)<br>_Tj_<br>,|_Di_<br>_Tj_<br>Initial features of drug<br> and target|
|**_h_**DTI(_l_)<br>_Di_<br>**_h_**DTI(_l_)<br>_Tj_<br>,|_l_<br>Inputs of Module1 in -th layer|
|**_h_**DDI(_l_)<br>_Di_<br>**_h_**PPI(_l_)<br>_Tj_<br>,|_l_<br>Inputs of Module2 and Module3 in -th layer|
|**_z_**_Di_**_ z_**_Tj_<br>,|_Di_<br>_Tj_<br>Final embedding of<br> and|
|**_h_**_P_ (_l_)<br>_Di ,_**_ h_**_N_(_l_)<br>_Di_<br>_,_**_ h_**_D_(_l_)<br>_Di_|_Di_<br>_l_<br>Hidden representations of<br> in -th layer|
|**_h_**_P_ (_l_)<br>_Tj_<br>_,_**_ h_**_N_(_l_)<br>_Tj_<br>_,_**_ h_**_T_ (_l_)<br>_Tj_|_Tj_<br>_l_<br>Hidden representations of<br> in -th layer|
|**_Θ_**_P_ (_l_)<br>_D_<br>_,_**_ Θ_**_N_(_l_)<br>_D_|_l_<br>Learnable parameter matrices of drugs in -th layer on the DTIs subnetwork|
|**_Θ_**_P_ (_l_)<br>_T_<br>_,_**_ Θ_**_N_(_l_)<br>_T_|_l_<br>Learnable parameter matrices of targets in -th layer on the DTIs subnetwork|
|**_Θ_**_D_(_l_)<br>_D_<br>**_Θ_**_T_ (_l_)<br>_T_<br>,|_l_<br>Learnable parameter matrices in -th layer on the DDIs and PPIs subnetworks|
|_C|The cooperative mode|
|_I|The independent mode|
|_S|**_Θ_**<br>The same kind of nodes share<br> in different modules|
|CS|Chemical structures of drugs|
|AD|The link vectors of drugs|
|AP|The link vectors of targets|


Drug-Target Interactions Prediction Based on Signed Heterogeneous Graph Neural Networks 235



**1. The proposed SHGNNs**


In the design of signed GNNs, there are two key
problems. The first one is how to propagate messages on
DTIs networks and aggregate information for both drug
and target nodes. As shown in Figure 3, the drug (target) has signed relations. Inspired by SGCN [30] that is
customized on signed graphs with homogeneous entities,



we solve the first problem and then propose SHGNN
model to study node embedding results on signed bipartite network. The second problem is how to extend
SHGNN to additionally integrate DDIs and PPIs, and
solve the training problem. Here, we propose three-module SHGNNs and put forward an end-to-end learning
framework that adapts to both signed bipartite networks and two-level heterogeneous networks.









_**h**_ _D_ DTI _k_















_**h**_ _T_ DTI _k_











_**h**_ _T_ DTI _p_ _p_ _T_ _j_ _**h**_ _D_ DTI _p_



(a) Information propagation on signed bipartite network (b) Information propagation on signed heterogeneous network


**Figure 3** Information propagation on DTIs networks. (a) Signed bipartite network; (b) Signed heterogeneous network.



1) SHGNN on signed bipartite networks
We dedicate message passing and aggregation process based on signed relations following the framework of
the simple SGCN variant [30].

SGCN is the first GNN model dedicated on signed
graphs. Its raw version, which is built on balanced theory and one-level homogeneous networks, is not directly
applicable to our two-level DTIs signed networks. Here,
we borrow the aggregation in its variant. The -th SGCN _l_
layer is calculated in terms of signed relations [30] among
the same kind of entities:



_**h**_ _[P]_ _D_ _i_ [=] _T_ _k_ ∑ _∈N_ _Di_ [+] _|N_ _**h**_ _′_ _TD_ [+] _ki_ _[|]_ [ +] _**[ h]**_ _′_ _D_ _i_ _[,]_ _**[ h]**_ _D_ _[N]_ _i_ [=] _T_ _k_ ∑ _∈N_ _Di_ _[−]_

_**h**_ _[P]_ _T_ _j_ [=] _D_ _k_ ∑ _∈N_ _Tj_ [+] _|N_ _**h**_ _′_ _DT_ [+] _kj_ _[|]_ [ +] _**[ h]**_ _′_ _T_ _j_ _[,]_ _**[ h]**_ _T_ _[N]_ _j_ [=] _D_ _k_ ∑ _∈N_ _Tj_ _[−]_



bors, respectively. For DTIs bipartite networks, Figure 3
(a) illustrates information propagation, where each node
has heterogeneous neighbors. We denote the positively
(negatively)-linked target neighbor set of _D_ _i_ by _N_ _D_ [+] _i_ ( _[N]_ _D_ _[ −]_ _i_ ).
Analogously, we define _N_ _T_ [+] _j_, _[N]_ _T_ _[ −]_ _j_ for targets. After information propagation, nodes’ contents are updated as follows:



_**h**_ _′_ _T_ _k_ _′_
_|N_ _D_ _[−]_ _i_ _[|]_ [ +] _**[ h]**_ _D_ _i_


_**h**_ _′_ _D_ _k_ _′_
_|N_ _T_ _[−]_ _j_ _[|]_ [ +] _**[ h]**_ _T_ _j_







 _k∈N_ [∑]



_k∈N_ _i_ _[−]_



_**h**_ _[P]_ _i_ [ (] _[l]_ [)] ≜ _σ_


_**h**_ _[N]_ _i_ [(] _[l]_ [)] ≜ _σ_



 _**Θ**_ _[P]_ [ (] _[l]_ [)]



 _**Θ**_ _[N]_ [(] _[l]_ [)]





 _k_ [∑] _∈N_ _i_



_k∈N_ _i_ [+]



_**h**_ [(] _k_ _[l][−]_ [1)]
_|N_ _i_ [+] _[|]_ _[,]_ _**[ h]**_ _i_ [(] _[l][−]_ [1)]

















_**h**_ [DTI] _D_ _i_ [(] _[l][−]_ [1)] and _**h**_ [DTI] _T_ _j_ [(] _[l][−]_ [1)] denote the inputs to the - _l_



_**h**_ _k_ [(] _[l][−]_ [1)]
_i_
_|N_ _i_ _[−]_ _[|]_ _[,]_ _**[ h]**_ [(] _[l][−]_ [1)]













_**h**_ _[P]_ _i_ [ (] _[l]_ [)] and _**h**_ _i_ _[N]_ [(] _[l]_ [)] are hidden states of node, follow- _i_

ing positive and negative links, respectively. _N_ _i_ [+] ( _N_ _i_ _[−]_







_l_



where _**h**_ _i_ and _**h**_ _i_ are hidden states of node, follow- _i_
ing positive and negative links, respectively. _N_ _i_ [+] ( _N_ _i_ _[−]_ )
denotes the set of positive (negative) neighbors of _v_ _i_ .

_**Θ**_ _[P]_ [ (] _[l]_ [)] and _**Θ**_ _[N]_ [(] _[l]_ [)] are learnable parameter matrices for lin
ear transformation, and each _**Θ**_ matrix includes two
parts corresponding to neighbors and the node itself, respectively. _σ_ is an activation function which is set as

tanh . The concatenation of hidden states is taken as the

input for the next layer, i.e., _**h**_ [(] _i_ _[l]_ [)] ≜ _**h**_ _[P]_ _i_ [ (] _[l]_ [)] _∥_ _**h**_ _[N]_ _i_ [(] _[l]_ [)] .



Let _**h**_ _D_ _i_ and _**h**_ _T_ _j_ denote the inputs to the - _l_
th layer. With the consideration of linear transformation
and activation, the -layer embedding process on signed _l_
bipartite networks is







∑


_T_ _k_ _∈N_





_v_ _i_
_**Θ**_ _[P]_ [ (] _[l]_ [)] and _**Θ**_ _[N]_ [(] _[l]_ [)]



_**h**_ _[P]_ _D_ [ (] _i_ _[l]_ [)] ≜ _σ_





 _**Θ**_ _D_ _[P]_ [ (] _[l]_ [)]







_**h**_ _T_ [DTI] _k_ [(] _[l][−]_ [1)]



_**h**_ _T_ [DTI] _|N_ _k_ _D_ [(][+] _[l]_ _i_ _[−]_ _[|]_ [1)] _,_ _**h**_ [DTI] _D_ _i_ [(] _[l][−]_ [1)]



_**Θ**_



_T_ _k_ _∈N_ _Di_ [+]







∑


_T_ _k_ _∈N_



























_T_ _k_ _∈N_ _Di_ _[−]_





 _**Θ**_ _D_ _[N]_ [(] _[l]_ [)]







_σ_

tanh



_**h**_ _[N]_ _D_ _i_ [(] _[l]_ [)] ≜ _σ_



_**h**_ [DTI] _T_ _k_ [(] _[l][−]_ [1)]



_**h**_ [DTI] _T_ _|N_ _k_ _D_ [(] _[−][l]_ _i_ _[−]_ _[|]_ [1)] _,_ _**h**_ [DTI] _D_ _i_ [(] _[l][−]_ [1)]



_**h**_ [(] _i_ _[l]_ [)] ≜ _**h**_ _[P]_ _i_ [ (] _[l]_ [)] _∥_ _**h**_ _[N]_ _i_ [(] _[l]_ [)]







Inspired by SGCN [30], we first solve the message
aggregation and node update problem on signed DTIs bipartite networks. Let _**h**_ _′_ _k_ be the output of linear transformation on _v_ _k_ . The underly message passing and aggregation in simple SGCN work as follows:











∑


_D_ _k_ _∈N_





_D_ _k_ _∈N_ _Tj_ [+]



(1)


(2)



_**h**_ _[P]_ _T_ _j_ [ (] _[l]_ [)] ≜ _σ_





 _**Θ**_ _T_ _[P]_ [ (] _[l]_ [)]







_**h**_ _D_ [DTI] _k_ [(] _[l][−]_ [1)]



_**h**_ _D_ [DTI] _|N_ _k_ _T_ [(][+] _[l]_ _j_ _[−]_ _[|]_ [1)] _,_ _**h**_ [DTI] _T_ _j_ [(] _[l][−]_ [1)]







∑


_D_ _k_ _∈N_





_D_ _k_ _∈N_ _Tj_ _[−]_



























 _**Θ**_ _T_ _[N]_ [(] _[l]_ [)]







_**h**_ [DTI] _D_ _k_ [(] _[l][−]_ [1)]



_′_
_**h**_ _k_ _′_
_i_
_|N_ _i_ _[−]_ _[|]_ [ +] _**[ h]**_







_**h**_ _[N]_ _T_ _j_ [(] _[l]_ [)] ≜ _σ_



_**h**_ _[P]_ _i_ [=] ∑

_k∈N_ _i_ [+]



_′_
_**h**_ _k_ _′_
_i_ _[,]_ _**[ h]**_ _[N]_ _i_ [=] ∑
_|N_ _i_ [+] _[|]_ [ +] _**[ h]**_

_k∈N_ _i_ _[−]_



_**h**_ [DTI] _D_ _|N_ _k_ _T_ [(] _[−][l]_ _j_ _[−]_ _[|]_ [1)] _,_ _**h**_ [DTI] _T_ _j_ [(] _[l][−]_ [1)]



They are responsible for positive and negative neigh


Here, _l ∈{_ 1 _,_ 2 _, . . ., L}_, _i ∈{_ 1 _,_ 2 _, . . ., n}_, _j ∈{_ 1 _,_ 2 _, . . ., m}_ .


236 Chinese Journal of Electronics, vol. 33, no. 1



_**Θ**_ _[NN]_ [(] _[l]_ [)] ≜ _{_ _**Θ**_ _D_ _[P]_ [ (] _[l]_ [)] _,_ _**Θ**_ _D_ _[N]_ [(] _[l]_ [)] _,_ _**Θ**_ _T_ _[P]_ [ (] _[l]_ [)] _,_ _**Θ**_ _T_ _[N]_ [(] _[l]_ [)] _}_ are learnable pa
rameter matrices on the -th layer. _l_

We employ their concatenation as the input for the
next layer, i.e.,


_**h**_ [DTI] _D_ _i_ [(] _[l]_ [)] ≜ _**h**_ _[P]_ _D_ [ (] _i_ _[l]_ [)] _∥_ _**h**_ _[N]_ _D_ _i_ [(] _[l]_ [)] _[,]_ _**[ h]**_ _T_ [DTI] _j_ [(] _[l]_ [)] ≜ _**h**_ _T_ _[P]_ _j_ [ (] _[l]_ [)] _∥_ _**h**_ _[N]_ _T_ _j_ [(] _[l]_ [)] (3)


A -layer SHGNN generate final embedding results by _L_


_**z**_ _D_ _i_ ≜ _**h**_ _[P]_ _D_ [ (] _i_ _[L]_ [)] _∥_ _**h**_ _[N]_ _D_ _i_ [(] _[L]_ [)] _,_ _**z**_ _T_ _j_ ≜ _**h**_ _T_ _[P]_ _j_ [ (] _[L]_ [)] _∥_ _**h**_ _[N]_ _T_ _j_ [(] _[L]_ [)]


Algorithm 1 outlines the process of SHGNNs on
signed bipartite DTIs networks.


Algorithm 1 SHGNN on signed bipartite networks


Input : A signed bipartite network _G_ _DT_ = ( _D, T, E_ _DT_ ) ;

_**h**_ [(0)] _D_ _i_ [(] _[i]_ [ = 1] _[,]_ [ 2] _[,][ · · ·][, n]_ [)] _[,]_ _**[ h]**_ _T_ [(0)] _j_ [(] _[j]_ [ = 1] _[,]_ [ 2] _[,][ · · ·][, m]_ [)] ; the number of

layers . _L_


Output : Low-dimension representations _**z**_ _D_ _i_ and _**z**_ _T_ _j_ .
Initialization:


_**h**_ [DTI] _D_ _i_ [(0)] ≜ _**h**_ [(0)] _D_ _i_, _[i]_ [ = 1] _[,]_ [ 2] _[, . . ., n]_ ;


_**h**_ [DTI] _T_ _j_ [(0)] ≜ _**h**_ [(0)] _T_ _j_, _j_ = 1 _,_ 2 _, . . ., m_ ;

for _l ∈{_ 1 _,_ 2 _, . . ., L}_ do


Calculate _**h**_ _[P]_ _D_ [ (] _i_ _[l]_ [)] and _**h**_ _[N]_ _D_ _i_ [(] _[l]_ [)] using equation (1);


Calculate _**h**_ _[P]_ _T_ _j_ [ (] _[l]_ [)] and _**h**_ _[N]_ _T_ _j_ [(] _[l]_ [)] using equation (2);

Update _**h**_ [DTI] _D_ _i_ [(] _[l]_ [)] and _**h**_ [DTI] _T_ _j_ [(] _[l]_ [)] by eqution (3);

Return _**z**_ _D_ _i_ ≜ _**h**_ [DTI] _D_ _i_ [(] _[L]_ [)], _**z**_ _T_ _j_ ≜ _**h**_ [DTI] _T_ _j_ [(] _[L]_ [)] .


2) SHGNNs with additional DDIs and PPIs subnetworks


Since DDIs or PPIs represent very different semantics from heterogeneous DTIs, it needs to find out how to
extend SHGNN defined on _G_ _DT_ to cover _G_ _D_ and _G_ _T_,
and solve the training problem. Here, we divide all relations into three subnetworks, propagate the information
in Figure 3(b) via applying the signed GNN layer to

_G_ _DT_ and introducing unsigned GNN layers for _G_ _D_ and
_G_ _T_ . Then, we try several training modes for the three
module GNN framework.







∑

_T_ _k_ _∈N_















 (5)




_**h**_ _[T]_ _T_ _j_ [ (] _[l]_ [)] ≜ _σ_





 _**Θ**_ _T_ _[T]_ [ (] _[l]_ [)]



_T_ _k_ _∈N_ _Tj_ [PPI]



_**h**_ [PPI] _T_ _k_ [(] _[l][−]_ [1)]
_|N_ _T_ [PPI] _j_ _[|]_ _[,]_ _**[ h]**_ _T_ [PPI] _j_ [(] _[l][−]_ [1)]



Here, _l ∈{_ 1 _,_ 2 _, . . ., L}_, _i ∈{_ 1 _,_ 2 _, . . ., n}_, _j ∈{_ 1 _,_ 2 _, . . ., m}_ .

_**Θ**_ _[NN]_ [(] _[l]_ [)] ≜ _{_ _**Θ**_ _D_ _[P]_ [ (] _[l]_ [)] _,_ _**Θ**_ _D_ _[N]_ [(] _[l]_ [)] _,_ _**Θ**_ _D_ _[D]_ [(] _[l]_ [)] _,_ _**Θ**_ _T_ _[P]_ [ (] _[l]_ [)] _,_ _**Θ**_ _T_ _[N]_ [(] _[l]_ [)] _,_ _**Θ**_ _T_ _[T]_ [ (] _[l]_ [)] _}_ are

learnable parameter matrices on the -th layer. _l_



Here, _l ∈{_ 1 _,_ 2 _, . . ., L}_, _i ∈{_ 1 _,_ 2 _, . . ., n}_, _j ∈{_ 1 _,_ 2 _, . . ., m}_ .

_**Θ**_ _[NN]_ [(] _[l]_ [)] ≜ _{_ _**Θ**_ _D_ _[P]_ [ (] _[l]_ [)] _,_ _**Θ**_ _D_ _[N]_ [(] _[l]_ [)] _,_ _**Θ**_ _D_ _[D]_ [(] _[l]_ [)] _,_ _**Θ**_ _T_ _[P]_ [ (] _[l]_ [)] _,_ _**Θ**_ _T_ _[N]_ [(] _[l]_ [)] _,_ _**Θ**_ _T_ _[T]_ [ (] _[l]_ [)] _}_ are

learnable parameter matrices on the -th layer. _l_

After layers, we concatenate results from all mod- _L_
ules to get the final node representation:


_**z**_ _D_ [DTI] _i_ [≜] _**[h]**_ _D_ _[P]_ [ (] _i_ _[L]_ [)] _∥_ _**h**_ _[N]_ _D_ _i_ [(] _[L]_ [)] _,_ _**z**_ _D_ [DDI] _i_ [≜] _**[h]**_ _D_ _[D]_ _i_ [(] _[L]_ [)] _,_ _**z**_ _D_ _i_ ≜ _**z**_ _D_ [DTI] _i_ _[∥]_ _**[z]**_ _D_ [DDI] _i_
_**z**_ _T_ [DTI] _j_ [≜] _**[h]**_ _T_ _[P]_ _j_ [ (] _[L]_ [)] _∥_ _**h**_ _[N]_ _T_ _j_ [(] _[L]_ [)] _,_ _**z**_ _T_ [PPI] _j_ [≜] _**[h]**_ _T_ _[T]_ _j_ [ (] _[L]_ [)] _,_ _**z**_ _T_ _j_ ≜ _**z**_ _T_ [DTI] _j_ _∥_ _**z**_ _T_ [PPI] _j_

The initial features of nodes, i.e., _**h**_ [DTI] _D_ _i_ [(0)] _,_ _**h**_ [DTI] _T_ _j_ [(0)],

_**h**_ [DDI] _D_ _i_ [(0)] _,_ _**h**_ [PPI] _T_ _j_ [(0)], will be further discussed in our experi


The initial features of nodes, i.e., _**h**_ _,_ _**h**_,



_G_ _DT_ to cover _G_ _D_ and _G_ _T_



_G_ _DT_ and introducing unsigned GNN layers for _G_ _D_
_G_ _T_ . Then, we try several training modes for the three


As shown in Figure 2, SHGNNs obtain embedding
results not only from the DTIs network via Module1 defined with above subsection, but also from the DDIs network and PPIs. For a drug node _D_ _i_, let _**h**_ [DDI] _D_ _i_ [(] _[l][−]_ [1)] denotes the input to the -layer of Module2 and _l_ _N_ _D_ [DDI] _i_ denotes its neighbor nodes within the DDIs network. Similarly, we define _**h**_ [PPI] _T_ _j_ [(] _[l][−]_ [1)] and _N_ _T_ [PPI] _j_ for the target _T_ _j_ on
the PPIs network. Aggregation in Module2 and Module3

are



_**h**_ _D_ _i_ _,_ _**h**_ _T_ _j_, will be further discussed in our experi
ments later.


To train the three-module SHGNN, it needs to consider whether they interact with each other. One consideration is how to deal with outputs on the previous layer and feed them into the current layer. Another consideration is whether the same kind of nodes share learn
able parameters in linear transformation. Here, we propose to train a SHGNN model in either cooperative or
independent mode, and combine them with sharing
weights.

i) The cooperative mode. In this case, three modules are trained cooperatively by setting the inputs with
the same values for one kind of nodes:


_**h**_ [DTI] _D_ _i_ [(] _[l]_ [)] = _**h**_ [DDI] _D_ _i_ [(] _[l]_ [)] _←_ _**h**_ _[P]_ _D_ [ (] _i_ _[l]_ [)] _∥_ _**h**_ _[N]_ _D_ _i_ [(] _[l]_ [)] _∥_ _**h**_ _[D]_ _D_ _i_ [(] _[l]_ [)] (6)


_**h**_ [DTI] _T_ _j_ [(] _[l]_ [)] = _**h**_ [PPI] _T_ _j_ [(] _[l]_ [)] _←_ _**h**_ _[P]_ _T_ _j_ [ (] _[l]_ [)] _∥_ _**h**_ _[N]_ _T_ _j_ [(] _[l]_ [)] _∥_ _**h**_ _[T]_ _T_ _j_ [ (] _[l]_ [)] (7)


As a result, in each layer Module 1 interacts with
Module 2 and Module 3.

ii) The independent mode. In this case, three modules are trained independently within sub-networks. The
input of each module is updated as its last embedding result as follows:


_**h**_ [DTI] _D_ _i_ [(] _[l]_ [)] _←_ _**h**_ _[P]_ _D_ [ (] _i_ _[l]_ [)] _∥_ _**h**_ _[N]_ _D_ _i_ [(] _[l]_ [)] _[,]_ _**[ h]**_ _D_ [DDI] _i_ [(] _[l]_ [)] _←_ _**h**_ _[D]_ _D_ _i_ [(] _[l]_ [)] (8)


_**h**_ [DTI] _T_ _j_ [(] _[l]_ [)] _←_ _**h**_ _[P]_ _T_ _j_ [ (] _[l]_ [)] _∥_ _**h**_ _[N]_ _T_ _j_ [(] _[l]_ [)] _,_ _**h**_ [PPI] _T_ _j_ [(] _[l]_ [)] _←_ _**h**_ _[T]_ _T_ _j_ [ (] _[l]_ [)] (9)



_D_ _i_, let _**h**_ [DDI] _D_ _i_ [(] _[l][−]_ [1)]
_l_ _N_ _D_ [DDI] _i_



iii) Sharing weights. For both modes, we further
consider whether the same kind of nodes share one parameter matrix, i.e. _**Θ**_ _D_ _[P]_ [ (] _i_ _[l]_ [)] [=] _**[Θ]**_ _D_ _[N]_ _i_ [(] _[l]_ [)] = _**Θ**_ _D_ _[D]_ _i_ [(] _[l]_ [)], _**Θ**_ _T_ _[P]_ _j_ [ (] _[l]_ [)] = _**Θ**_ _T_ _[N]_ _j_ [(] _[l]_ [)] =



_**h**_ [PPI] _T_ _j_ [(] _[l][−]_ [1)] and _N_ _T_ [PPI] _j_ for the target _T_ _j_



_**Θ**_ _T_ _[T]_ _j_ [ (] _[l]_ [)]



_**Θ**_ _T_ _j_ . To keep the same embedding dimension, the inde
pendent mode replaces concatenation “ ” by addition _∥_
“+” in equation (8)–(9), i.e.,



_∥_







∑

_D_ _k_ _∈N_















 (4)




_**h**_ _[D]_ _D_ _i_ [(] _[l]_ [)] ≜ _σ_





 _**Θ**_ _D_ _[D]_ [(] _[l]_ [)]



_**h**_ [DTI] _D_ _i_ [(] _[l]_ [)] _←_ _**h**_ _[P]_ _D_ [ (] _i_ _[l]_ [)] + _**h**_ _[N]_ _D_ _i_ [(] _[l]_ [)] _[,]_ _**[ h]**_ _T_ [DTI] _j_ [(] _[l]_ [)] _←_ _**h**_ _[P]_ _T_ _j_ [ (] _[l]_ [)] + _**h**_ _[N]_ _T_ _j_ [(] _[l]_ [)]


Algorithm 2 outlines the embedding generation on
two-level DTIs networks. In Figure 4, we further show 2


_D_ _k_ _∈N_ _Di_ [DDI]



_**h**_ [DDI] _D_ _k_ [(] _[l][−]_ [1)]
_|N_ _D_ [DDI] _i_ _[|]_ _[,]_ _**[ h]**_ [DDI] _D_ _i_ [(] _[l][−]_ [1)]


Drug-Target Interactions Prediction Based on Signed Heterogeneous Graph Neural Networks 237



layer SHGNNs under either cooperative or independent

mode.


Algorithm 2 SHGNNs on signed heterogeneous networks


Input : A signed bipartite network _G_ _DT_ = ( _D, T, E_ _DT_ ) ;
a DDIs network _G_ _D_ = ( _D,_ _**A**_ _D_ _, E_ _D_ ) ; a PPIs network

_G_ _T_ = ( _T,_ _**A**_ _T_ _, E_ _T_ ) ; _**h**_ [(0)] _D_ _i_ [(] _[i]_ [ = 1] _[,]_ [ 2] _[, . . ., n]_ [)] _[,]_ _**[ h]**_ _T_ [(0)] _j_ [(] _[j]_ [ = 1] _[,]_ [ 2] _[, . . .,]_

_m_ ); the number of layers . _L_

Output : Low-dimension representations { _**z**_ _D_ _i_ and _**z**_ _T_ _j_ .}
Initialization:


_**h**_ [DTI] _D_ _i_ [(0)] = _h_ [DDI] _D_ _i_ [(0)] ≜ _**h**_ [(0)] _D_ _i_, _[i]_ [ = 1] _[,]_ [ 2] _[, . . ., n]_ ;


_**h**_ [DTI] _T_ _j_ [(0)] = _**h**_ [PPI] _T_ _j_ [(0)] ≜ _**h**_ [(0)] _T_ _j_, _j_ = 1 _,_ 2 _, . . ., m_ ;

for _l ∈{_ 1 _,_ 2 _, . . ., L}_ do


Calculate _**h**_ _[P]_ _D_ [ (] _i_ _[l]_ [)] and _**h**_ _[N]_ _D_ _i_ [(] _[l]_ [)] using equation (1);


Calculate _**h**_ _[P]_ _T_ _j_ [ (] _[l]_ [)] and _**h**_ _[N]_ _T_ _j_ [(] _[l]_ [)] using equation (2);

Calculate _**h**_ _[D]_ _D_ _i_ [(] _[l]_ [)] using equation (4);


Calculate _**h**_ _[T]_ _T_ [ (] _j_ _[l]_ [)] using equation (5);

Update _**h**_ [DTI] _D_ _i_ [(] _[l]_ [)] and _**h**_ [DDI] _D_ _i_ [(] _[l]_ [)] by equation (6) or equation (8);


Update _**h**_ [DTI] _T_ _j_ [(] _[l]_ [)] and _**h**_ [PPI] _T_ _j_ [(] _[l]_ [)] by equation (7) or equation (9);

Return _**z**_ _D_ _i_ ≜ _**h**_ [DTI] _D_ _i_ [(] _[L]_ [)] _∥_ _**h**_ [DDI] _D_ _i_ [(] _[L]_ [)] _,_ _**z**_ _T_ _j_ ≜ _**h**_ [DTI] _T_ _j_ [(] _[L]_ [)] _∥_ _**h**_ _[P P I]_ _T_ _j_ [(] _[L]_ [)] .


**2. The loss function**


Taking the final drug-target pair ( _**z**_ _D_ _i_ _,_ _**z**_ _T_ _j_ ) as an input, we employ a multilayer perception (MLP) to further extract characteristics of a drug-target pair, and
then utilize a softmax regression classifier to discriminate
the DTI type.

Here, we jointly train SHGNNs and the DTIs discriminator. Let _**Θ**_ _[NN]_ = _{_ _**Θ**_ _[NN]_ [(1)] _, . . .,_ _**Θ**_ _[NN]_ [(] _[L]_ [)] _}_ include
weight matrix parameters of _L_ layers SHGNNs and the



weight parameters _**Θ**_ _[MLP]_ for MLP, and _**Θ**_ _[R]_ = _{_ _**Θ**_ +1 _[R]_ _[,]_

_**Θ**_ _−_ _[R]_ 1 _[}]_ denotes regression coefficients, where _**Θ**_ +1 _[R]_ ( _**[Θ]**_ _−_ _[R]_ 1 ) is

the positive (negative) edge type coefficient. The loss
function is defined as follows:


_L_ ( _**Θ**_ _[NN]_ _,_ _**Θ**_ _[R]_ _,_ _**Θ**_ [MLP] )



where _e_ _ij_ _∈_ _S_ represents the type of edge between drug

_D_ _i_ and target _T_ _j_, _S ∈{_ +1 _, −_ 1 _}_ and _ω_ _e_ _ij_ denotes the

weight associated with link type _e_ _ij_ . I( _·_ ) returns 1 if a
given prediction is true, and 0 otherwise.


**IV. Experimental Settings and Analyses**


In this section, we conduct experiments on two
datasets, one of which is newly extracted in this study
from DrugBank [26] and related databases. First, our
SHGNN shows excellent performance on signed bipartite
DTIs networks compared with classic baselines. And
then, its performance is further verified on a two-level
DTIs signed network, and analyzed under different
modes and settings. Finally, a case study is provided to
verify our SHGNN-based DTI prediction.


**1. Datasets**


Two datasets are collected and their data statistics

are shown in Table 3, where signed DTIs are labeled according to Table 1. Torres _et al_ . [15] provided a signed
DTIs network from early version of DrugBank. Here, we
also extracted signed DTIs from a recent version of
DrugBank [26], and additional information from related
databases.


Dataset1 contains a signed bipartite network,
which has 1178 drug nodes, 578 target nodes, and 2599
signed DTIs.



_−ω_ _e_ _ij_ ∑
_ij_ _c∈S_



~~∑~~ exp( _**Θ**_ _q_ _[R]_ ~~[~~ MLP( _**z**_ _D_ _i_ _||_ _**z**_ _T_ _j_ ) ~~]~~

_q∈S_



=
∑



∑ I( _e_ _ij_ = _c_ ) log exp( _**Θ**_ _c_ _[R]_ [MLP( _**z**_ _D_ _i_ _||_ _**z**_ _T_ _j_ )])

_c∈S_ ~~∑~~ exp( _**Θ**_ _q_ _[R]_ ~~[~~ MLP( _**z**_ _D_ _i_ _||_ _**z**_ _T_ _j_



_**Θ**_ _[NN]_ = _{_ _**Θ**_ _[NN]_ [(1)] _, . . .,_ _**Θ**_ _[NN]_ [(] _[L]_ [)] _}_



_L_





(b) The framework of SHGNN_I











_**z**_ _Di_


_**z**_ _Tj_



(a) The framework of SHGNN_C























_**z**_ _Di_



_**z**_ _Tj_



































(c) A 2-layer SHGNN_C
(d) A 2-layer SHGNN_I


**Figure 4** Two modes of SHGNNs and its 2-layer illustrative examples. (a) and (b) are the framework under cooperative mode and independent mode respectively. (c) and (d) are their 2-layer illustrative examples, in which the dashed boxes show their differences. “&” in (d) denotes “ ” or “ _∥_ + ”.


238 Chinese Journal of Electronics, vol. 33, no. 1









|Table 3 Desc|criptions of da|atasets|Col4|Col5|Col6|
|---|---|---|---|---|---|
||Drugs<br>(features)|Target<br>proteins|DTIs|DDIs|PPIs|
|Dataset1|1,178 (–)|578|+:1,093<br>−:1,506|–|–|
|Dataset2|846 (881)|685|+:909<br>−:1,859|169,162|5,820|


Dataset2 contains a DTIs signed bipartite network,
and additional DDIs and PPIs within these drugs and
target proteins. It is processed as follows. According to
statistics on DrugBank 5.1.7 [26], in commonly used
drugs, the number of small molecule drugs can account
for a high probability. And hence, we first collect approved small molecule drugs, and their target proteins.
Also, we extract an unsigned DDIs network from DrugBank. Furthermore, we search the interaction information among target proteins from String 11.5 [33] database
to build an unsigned PPIs network. The chemical structure information of the drug is additionally obtained
from PubChem [32] database. Each drug is represented
by a 881-dimensional binary vector, where 1 value indicates that the drug has a specific chemical structure segment. After discarding drugs not in PubChem [32]
database and targets not in String 11.5 [33], we finally
keep 846 drugs and 685 targets, 2768 signed DTIs, 169162
DDIs and 5820 PPIs.


**2. Training settings and baselines**


Settings of SHGNNs Here, we list SHGNN variants, hyper-parameters settings, and initial features of
nodes.


  - SHGNN variants. Given datasets with only DTIs
networks, SHGNN in Algorithm 1 is employed. With additional DDIs and PPIs, SHGNNs in Algorithm 2 are
trained under the cooperative mode or independent
mode, as well as whether weight matrices are shared.
Here, we use “_C” and “_I” to distinguish two training
modes, and let “_S” denote sharing weights.

  - Initial features of nodes. For convenience, we use
“CS”, “AD” and “AP” to represent cases of initial features. As summarized in Table 2, they correspond to
chemical structures of drugs, the link vectors of drugs
(i.e., one row in adjacency matrix _**A**_ _D_ ), and the link vectors of targets (i.e., one row in adjacency matrix _**A**_ _T_ ), respectively.

  - Hyper-parameter settings. As shown in Table 4,
SHGNNs employ _L_ = 2 convolutional layers and an
Adam optimizer with a learning rate of 0.005. For simplicity, let drugs and targets have the same embedding
dimensions _d_ [out] after all linear transformation, i.e.,

_**h**_ _[P]_ _D_ [ (] _i_ _[l]_ [)] _[,]_ _**[ h]**_ _D_ _[N]_ _i_ [(] _[l]_ [)] _[,]_ _**[ h]**_ _D_ _[D]_ _i_ [(] _[l]_ [)] _[,]_ _**[ h]**_ _T_ _[P]_ _j_ [ (] _[l]_ [)] _[,]_ _**[ h]**_ _T_ _[N]_ _j_ [(] _[l]_ [)] _,_ _**h**_ _T_ _[T]_ _j_ [ (] _[l]_ [)] _∈_ R _[d]_ [out] . Here, we set

_d_ [out] 8 16 32 64 128 256 . We use different numbers



|Table 4 Hyper-parameters of SHGNN|Ns and GNNs baselines|
|---|---|
|Hyper-parameter|Value|
|Optimizer|Adam|
|Learning rate|0.005|
|Num of convolutional layers|2|
|_d_out<br>Embedding dimension|{8, 16, 32, 64, 128, 256}|
|Num of Epochs|2000|


of iterations to train the model and find that 2000 epochs
are enough to achieve good results.

Baselines In this study, we take 5 models as baselines. The first kind is from traditional models. Here, we
choose signed bipartite random walk (SBRW) [34]. With
the help of the extended balanced theory, it carries out a
random walk on signed bipartite networks. The second
kind of baseline is from deep learning frameworks. We
consider two most popular GNNs on unsigned graphs
such as GCN [35] and GraphSAGE [36]. SBGNN [31] is a
typical GNN model based on the extend balanced theory,
and adapts to signed bipartite networks. We apply it to
tackle with signed drug-target relations. The popular
heterogeneous graph attention network (HAN) [37] is also tested on our datasets. It is a heterogeneous GNN
model based on meta paths and hierarchical attention
mechanisms.


  - Code sources and configurations. Since SBRW
and SBGNN are dedicated on signed bipartite networks,
we run them only on DTIs subnetworks. Their official
source codes are employed [*1*2] . Since raw GCN and GraphSAGE adapt to unsigned networks, here we apply their
PyG code versions [*3] into the underly unsigned graphs of
signed DTIs networks and set the mean aggregator in
GraphSAGE. When applying HAN to signed relations,
we define different meta paths for positive and negative
DTIs. Its DGL code version [*4] with 9-head attentions is

employed here. For fair comparisons, all baselines are
trained with the same loss function in this study.

  - Hyper-parameter settings. All GNNs keep the
same hyper-parameter settings as those of SHGNNs, as
shown in Table 4. For SBRW, there are parameters including _ω_, _δ_ p and _δ_ _n_ . _ω_ is a bias parameter for random
walkers. The thresholds _δ_ p and _δ_ _n_ are used to define elements of adjacency matrices. Here, we varied _ω ∈{_ 1 _,_ 2 _,_ 3 _,_

4 _,_ 5 _}_, _δ_ p _∈{_ 0 _,_ 25 _,_ 50 _,_ 75 _,_ 100 _}_ and _δ_ _n_ _∈{_ 0 _, −_ 25 _, −_ 50 _, −_ 75 _,_
_−_ 100 _}_, and find that SBRW achieves optimal performance

when _ω_ = 2 _δ_, = 50, _δ_ _n_ = _−_ 100 .



**3. Comparisons**


We employ the area under the receiver operating
characteristic curve (AUC), accuracy (ACC) and two Fscore indicators to evaluate experimental results. A higher value of these metrics indicates better performance.



_ω_, _δ_ p and _δ_ _n_ . _ω_



_**A**_ _D_



_δ_ p and _δ_ _n_



_**A**_ _T_



ments of adjacency matrices. Here, we varied _ω ∈{_ 1 _,_ 2 _,_ 3 _,_

4 _,_ 5 _}_, _δ_ p _∈{_ 0 _,_ 25 _,_ 50 _,_ 75 _,_ 100 _}_ and _δ_ _n_ _∈{_ 0 _, −_ 25 _, −_ 50 _, −_ 75 _,_
_−_ 100 _}_, and find that SBRW achieves optimal performance



_L_ = 2



_ω_ = 2 _δ_, p = 50, _δ_ _n_ = _−_ 100



_d_ [out]



_**h**_ _[P]_ _D_ [ (] _i_ _[l]_ [)] _[,]_ _**[ h]**_ _D_ _[N]_ _i_ [(] _[l]_ [)] _[,]_ _**[ h]**_ _D_ _[D]_ _i_ [(] _[l]_ [)] _[,]_ _**[ h]**_ _T_ _[P]_ _j_ [ (] _[l]_ [)] _[,]_ _**[ h]**_ _T_ _[N]_ _j_ [(] _[l]_ [)] _,_ _**h**_ _T_ _[T]_ _j_ [ (] _[l]_ [)] _∈_ R _[d]_ [out]



_d_ [out] _∈{_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _,_ 256 _}_



*1



https://github.com/DSE-MSU/signed-bipartite-networks
*2



https://github.com/huangjunjie-cs/SBGNN
*3



https://github.com/pyg-team/pytorch_geometric
*4



https://github.com/dmlc/dgl/tree/master/examples/pytorch/han


Drug-Target Interactions Prediction Based on Signed Heterogeneous Graph Neural Networks 239



Each experiment run is 5-fold cross-validation (CV) and
all the results are the statistical values across 5 runs. In

CV experiments, we randomly divide all DTIs into 5
equal-size groups. 4-group DTIs serve as the training set
and the rest one is used to test the model. All DDIs and
(or) PPIs data are employed in training SHGNNs if the



responding module is integrated. In Tables 5–7, we show
the overview results of all methods in terms of best val
ues over all settings. And Figure 5 shows detailed performance with different embedding dimensions. We report
the average and the standard deviation (std) of these
metrics across 5 runs.




|Table 5 Comparis|sons of methods on the signed bipartite networks from dataset|Col3|Col4|Col5|t1 and dataset2|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||Dataset1|Dataset1|Dataset1|Dataset1|Dataset2|Dataset2|Dataset2|Dataset2|
|Methods|ACC|Macro-F1|Micro-F1|AUC|ACC|Macro-F1|Micro-F1|AUC|
|SBRW|_±_0_._013<br>0.775|_±_0_._012<br>0.757|_±_0_._013<br>0.775|_±_0_._015<br>0.777|_±_0_._018<br>0.756|_±_0_._018<br>0.738|_±_0_._018<br>0.756|_±_0_._012<br>0.824|
|SBGNN|_±_0_._006<br>0.867|_±_0_._006<br>0.865|_±_0_._006<br>0.867|_±_0_._017<br>0.928|_±_0_._015<br>0.865|_±_0_._015<br>0.849|_±_0_._015<br>0.865|_±_0_._019<br>0.917|
|GCN|_±_0_._002<br>0.872|_±_0_._008<br>0.868|_±_0_._002<br>0.872|_±_0_._003<br>0.945|_±_0_._012<br>0.869|_±_0_._013<br>0.867|_±_0_._012<br>0.869|_±_0_._006<br>0.916|
|GraphSAGE|_±_0_._008<br>0.876|_±_0_._008<br>0.870|_±_0_._008<br>0.876|_±_0_._003<br>0.943|_±_0_._007<br>0.885|_±_0_._007<br>0.873|_±_0_._007<br>0.885|_±_0_._004<br>0.922|
|HAN|_±_0_._004<br>0.878|_±_0_._003<br>0.875|_±_0_._004<br>0.878|_±_0_._003<br>0.947|_±_0_._003<br>0.868|_±_0_._003<br>0.851|_±_0_._003<br>0.868|_±_0_._002<br>0.929|
|SHGNN|_±_0_._003<br>0.885|_±_0_._003<br>0.882|_±_0_._003<br>0.885|_±_0_._006<br>0.950|_±_0_._003<br>0.889|_±_0_._006<br>0.874|_±_0_._003<br>0.889|_±_0_._004<br>0.935|
|Note: The highest score is in bold.|Note: The highest score is in bold.|Note: The highest score is in bold.|Note: The highest score is in bold.|Note: The highest score is in bold.|Note: The highest score is in bold.|Note: The highest score is in bold.|Note: The highest score is in bold.|Note: The highest score is in bold.|


|Table 6 Comparisons on the t|two-level signed heterogeneo|ous network from the datase|et2|Col5|
|---|---|---|---|---|
|Methods|ACC|Macro-F1|Micro-F1|AUC|
|GCN|_±_0_._003<br>0.892|_±_0_._003<br>0.879|_±_0_._003<br>0.892|_±_0_._007<br>0.921|
|GraphSAGE|_±_0_._003<br>0.889|_±_0_._002<br>0.878|_±_0_._003<br>0.889|_±_0_._002<br>0.924|
|HAN|_±_0_._003<br>0.874|_±_0_._003<br>0.858|_±_0_._003<br>0.874|_±_0_._004<br>0.933|
|SHGNN_I|~~_±_~~0_._005<br>0.896|~~_±_~~0_._010<br>0.884|~~_±_~~0_._005<br>0.896|~~_±_~~0_._010<br>0.935|
|SHGNN_C|_±_0_._004<br>0.904|_±_0_._006<br>0.889|_±_0_._004<br>0.904|_±_0_._006<br>0.938|
|Note: The best is marked in bold and the second best is in underline.|Note: The best is marked in bold and the second best is in underline.|Note: The best is marked in bold and the second best is in underline.|Note: The best is marked in bold and the second best is in underline.|Note: The best is marked in bold and the second best is in underline.|








|Table 7 The effect of differe|ent initial features of d|drug and target nodes o|on SHGNNs-based pred|diction methods|Col6|
|---|---|---|---|---|---|
|Methods|Feature|ACC|Macro-F1|Micro-F1|AUC|
|SHGNN_I (_S)|CS&AP|0.893 (0.893)|0.880 (0.877)|0.893 (0.893)|0.927 (0.928)|
|SHGNN_I (_S)|_∥_<br>CS AD&AP|0.892 (0.892)|0.884 (0.880)|0.892 (0.892)|0.933 (0.934)|
|SHGNN_I (_S)|AD&AP|0.896 (0.894)|0.884 (0.885)|0.896 (0.894)|0.935 (0.934)|
|SHGNN_C (_S)|CS&AP|0.897 (0.891)|0.884 (0.879)|0.897 (0.891)|0.931 (0.930)|
|SHGNN_C (_S)|_∥_<br>CS AD&AP|0.899 (0.896)|0.887 (0.883)|0.899 (0.896)|0.935 (0.933)|
|SHGNN_C (_S)|AD&AP|0.904 (0.903)|0.889 (0.887)|0.904 (0.903)|0.938 (0.936)|
|Note: The bold indicates the best within the group.<br> CS&AP: initial inputs include the chemical structures of drugs and the link vectors of targets.<br> AD&AP: initial inputs include link vectors of drugs and targets.<br>_∥_<br> CS AD&AP: initial inputs include the concatenation of chemical structures and link vectors of drugs, and link vectors of targets.|Note: The bold indicates the best within the group.<br> CS&AP: initial inputs include the chemical structures of drugs and the link vectors of targets.<br> AD&AP: initial inputs include link vectors of drugs and targets.<br>_∥_<br> CS AD&AP: initial inputs include the concatenation of chemical structures and link vectors of drugs, and link vectors of targets.|Note: The bold indicates the best within the group.<br> CS&AP: initial inputs include the chemical structures of drugs and the link vectors of targets.<br> AD&AP: initial inputs include link vectors of drugs and targets.<br>_∥_<br> CS AD&AP: initial inputs include the concatenation of chemical structures and link vectors of drugs, and link vectors of targets.|Note: The bold indicates the best within the group.<br> CS&AP: initial inputs include the chemical structures of drugs and the link vectors of targets.<br> AD&AP: initial inputs include link vectors of drugs and targets.<br>_∥_<br> CS AD&AP: initial inputs include the concatenation of chemical structures and link vectors of drugs, and link vectors of targets.|Note: The bold indicates the best within the group.<br> CS&AP: initial inputs include the chemical structures of drugs and the link vectors of targets.<br> AD&AP: initial inputs include link vectors of drugs and targets.<br>_∥_<br> CS AD&AP: initial inputs include the concatenation of chemical structures and link vectors of drugs, and link vectors of targets.|Note: The bold indicates the best within the group.<br> CS&AP: initial inputs include the chemical structures of drugs and the link vectors of targets.<br> AD&AP: initial inputs include link vectors of drugs and targets.<br>_∥_<br> CS AD&AP: initial inputs include the concatenation of chemical structures and link vectors of drugs, and link vectors of targets.|



The purpose of our experiments is to obtain the following observations: 1) the validity of the SHGNNs-DTI
framework, not only on the signed bipartite network but
also on the two-level heterogeneous network; 2) the effect of model settings and training modes on perfor
mance of SHGNNs.


From Table 5, we can find the performance of six
methods on signed bipartite networks as shown in Figure 1
(a) from dataset1 and dataset2. Here, since attributes of
drugs and targets are missing in dataset1, we employ the
link vectors between drugs and targets as the initial features of nodes. Compared with SBRW and SBGNN, the



performance of the SHGNN greatly exceeds those of the

baseline methods. It hints that the extended balance the
ory, which is the underly basis of the two baselines, is
not applicable between drugs and targets. Compared
with GCN and GraphSAGE, SHGNN follows signs to aggregate messages from different kinds of neighbors, and
hence it makes better use of signed graph structures. In
addition, it is observed that, the meta-path based HAN,
which also considers the link signs, is still inferior to our
SHGNN model in all metrics values. In summary,
SHGNN-DTI shows its significant better performance on
predicting signed DTIs on bipartite networks.


240 Chinese Journal of Electronics, vol. 33, no. 1



(a)


(b)


(c)


(d)



ACC


0.90

0.88


0.86

0.84


0.82
8 16 32 64



0.90


0.88


0.86


0.84


0.82
8 16 32 64



128 256



Macro-F1

0.90


0.88


0.86


0.84


0.82
8 16 32 64



128 256



AUC

0.94

0.93

0.92


0.91

0.90


0.89
8 16 32 64



128 256



Embedding dimensions


ACC



Embedding dimensions

SHGNN_C SHGNN_C_S


Macro-F1



Embedding dimensions


AUC



0.90


0.88


0.86


0.84


0.82
8 16 32 64



128 256



0.90


0.88


0.86


0.84


0.82
8 16 32 64



128 256



0.94

0.93

0.92


0.91

0.90


0.89
8 16 32 64



128 256



Embedding dimensions


ACC



Embedding dimensions

SHGNN_I SHGNN_I_S


Macro-F1



Embedding dimensions


AUC



0.90


0.88


0.86


0.84


0.82
8 16 32 64



0.94

0.93

0.92


0.91

0.90


0.89
8 16 32 64



128 256



128 256



0.90


0.88


0.86


0.84


0.82
8 16 32 64



128 256



Embedding dimensions


ACC



Embedding dimensions

SHGNN_C SHGNN_I


Macro-F1



Embedding dimensions


AUC



128 256



0.90


0.88


0.86


0.84


0.82
8 16 32 64



128 256



0.94

0.93

0.92


0.91

0.90


0.89
8 16 32 64



128 256



Embedding dimensions



Embedding dimensions

SHGNN_C_S SHGNN_I_S



Embedding dimensions



**Figure 5** Comparison of three-module SHGNNs with different embedding dimensions on dataset2. (a) Comparison of SHGNN_C and its
variant with sharing weights; (b) Comparison of SHGNN_I and its variant with sharing weights; (c) Comparison of SHGNN_C and
SHGNN_I; (d) Comparison of SHGNN_C_S and SHGNN_I_S.



In Table 6, we can find the performance of SHGNNs
and baselines on the two-level signed heterogeneous network (as shown in Figure 1(b)) from the dataset2. Firstly, both SHGNN_C and SHGNN_I are significantly superior to baselines such as GraphSAGE and HAN. Secondly, SHGNN_C performs better than SHGNN_I, and
the observation will be further analyzed later. In addition, compared with the results of dataset2 shown in
Table 5, SHGNNs and baselines promote almost all metric values. As a result, it illustrates benefits from DDIs
and PPIs networks.


In Table 7, we show the effect of initial nodes features on SHGNN_C, SHGNN_I and their versions with
sharing weights. Firstly, the model performance of
SHGNN-DTI maintains a high level for all initial features. Secondly, it is observed that the best values of
SHGNN generally occur in consideration of the link vector from _**A**_ _D_ as the initial features of drugs, and the link



vector from _**A**_ _T_ as the initial features of targets. This
may be caused by the inconsistency between the chemical structure features and the DTIs network.


In Figure 5, we further show the performance of
SHGNN_C, SHGNN_I and their versions with sharing
weights, in different embedding dimensions. Their MicroF1 values are not illustrated, but they also have similar
trends. Here, we employ the initial feature setting that
can achieve the best performance (i.e. AD&AP). Firstly,
versions with sharing weights are more stable, and their
model performance maintains a high level for any embedding dimensions with the maximum difference between
indicators not more than 1 percentage point. Secondly,
SHGNN_C is better than SHGNN_I, which hints that
when SHGNNs in cooperatively training mode, modules
can interact with each other to capture the hidden information within signed heterogeneous networks.

Observations about the SHGNN-DTI can be con

Drug-Target Interactions Prediction Based on Signed Heterogeneous Graph Neural Networks 241



cluded as follows: 1) it is an effective way to predict
signed DTIs; 2) DDIs and PPIs could provide benefits for
promoting performance of SHGNNs; 3) the cooperative
mode is better than the independent mode; 4) SHGNNs
with sharing weights are more robust in terms of embedding dimensions.


**4. Ablation study**


In this section, we vary the configurations of
SHGNNs to verify the role of modules. For the purpose
of illustration, we conduct experiments on subnetworks
extracted from dataset2. We consider several versions of

SHGNNs as follows.


  - SHGNN: the SHGNN model works on three mod
ules.

  - SHGNN (Wo_DDI): the SHGNN model works
without the DDI network.

  - SHGNN (Wo_PPI): the SHGNN model works
without the PPI network.

  - SHGNN (Wo_DTI): the SHGNN model works on
DDIs and PPIs networks, i.e., without the DTI network.

  - SHGNN (O_DTI): the SHGNN model works only on the signed bipartite DTI network, i.e., without
DDIs and PPIs.


The last one runs Algorithm 1, while others run Algorithm 2 under the cooperative mode without sharing
weights.

Figure 6 shows their best values of 5 runs of 5-fold
CV experiments. Their Micro-F1 values are not illustrated, but they also have similar trends. When SHGNN
works with all modules, it achieves the best performance,
i.e., ACC: 0.904, Macro-F1: 0.889, Micro-F1: 0.904 and
AUC: 0.938. Besides, we obtain some observations.












|Col1|Col2|
|---|---|
|||











|0.94<br>0.93<br>0.92<br>0.91<br>0.90<br>0.89<br>0.88|SHGNN (Wo_DTI) 0.9350.9370.9360.938<br>SHGNN (O_DTT)<br>SHGNN (Wo_PPI)<br>SHGNN (Wo_DDI) 0.917<br>SHGNN<br>0.9010.9030.904<br>0.889 0.8870.889<br>0.886<br>0.874|
|---|---|
|~~0.85~~<br>~~0.86~~<br>~~0.84~~<br>ACC<br>~~0.856~~<br>~~0.842~~<br>AUC<br>Macro-F1<br>~~0.87~~<br> <br>**Figure 6**Ablation study results of the variants on dataset2.<br> <br>i) Compared above results with those of SHG<br>(Wo_DDI), SHGNN (Wo_PPI) and SHGNN (Wo_D|~~0.85~~<br>~~0.86~~<br>~~0.84~~<br>ACC<br>~~0.856~~<br>~~0.842~~<br>AUC<br>Macro-F1<br>~~0.87~~<br> <br>**Figure 6**Ablation study results of the variants on dataset2.<br> <br>i) Compared above results with those of SHG<br>(Wo_DDI), SHGNN (Wo_PPI) and SHGNN (Wo_D|


we find that missing any module will lead to performance degradation. When the DTI module is removed,
scores of SHGNN decrease to ACC: 0.856, Macro-F1: 0.842,
Micro-F1: 0.856 and AUC: 0.917. The largest gap between SHGNN (Wo_DTI) and SHGNN indicates that
information within the DTI network is the most important in our model.

ii) The auxiliary role of DDI and PPI modules is il


lustrated. The performance of SHGNN (O_DTI), i.e.,
ACC 0.889, Macro-F1 0.874, Micro-F1 0.889 and AUC
0.935, is promoted when DDIs and (or) PPI networks are
(is) integrated. It illustrates that they can provide additional information.


**5. Case study**


Here, we choose Goserelin and Epirubicin to verify
our model’s performance, as they are common and important drugs for the treatment of breast cancer that is
the main cancer diagnosed in women. According to the
announcement of the World Health Organization in December 2020, breast cancer has surpassed lung cancer as
the most common cancer in the world [38], [39]. Goserelin is a synthetic analog of luteinizing hormone-releasing
hormone, which can be used to treat breast cancer by reducing secretion of gonadotropins from the pituitary.
Epirubicin is an anthracycline topoisomerase II inhibitor
used as an adjuvant to treating axillary node metastases
in patients who have undergone surgical resection of primary breast cancer.

However, we find in DrugBank [26] that the DTIs
related to these two drugs are very few. Goserelin only
contains two positive DTIs, and Epirubicin only contains two negative DTIs. Goserelin has “agonist ” relations with Gonadotropin-releasing hormone receptor and
Lutropin-choriogonadotropic hormone receptor. Meanwhile, Epirubicin has “antagonist ” relation with Chromodomain-helicase-DNA-binding protein 1 (CHD1), and
the “inhibitor” action mode on DNA topoisomerase 2-alpha. In this case study, we aim at verifying the validity
of the model on discovering potential DTIs, via predicting all signed relations between both drugs and their interactions with all targets.

5 runs of 5-fold cross-validation experiments are conducted on SHGNN_C, to predict known DTIs for
Goserelin and Epirubicin. The best model settings in
above section is employed, i.e., without sharing weights,
setting AD&AP for the initial features of nodes and the
number of dimensions equal to 32. Table 8 shows DTIs
within their top-10 ranks in terms of “softmax” values.

Top-10 DTIs of Goserelin Apoptosis regulator Bcl2 is the predicted target protein with the largest score in
the DTIs predicted to be positive. As reported in the literature [40], Goserelin results in increased expression of
Bcl-2 protein. Goserelin is shown in DrugBank [26] as an
agonist of Gonadotropin-releasing hormone receptor and
Lutropin-choriogonadotropic hormone receptor. Two target proteins of DTIs predicted to be negative, Plasma
kallikrein and Transmembrane protein serine 2, have also been confirmed by literature [41]. Among the DTIs
with the top-10 prediction scores, no relevant evidence
was found for the DTIs between Goserelin and Synaptic
vesicle glycoprotein 2A, Androgen receptor, Somatostatin receptor, Sterol o-acyltransferase 1 and Osteocalcin, leaving for further study.

Top-10 DTIs of Epirubicin The probably positive


242 Chinese Journal of Electronics, vol. 33, no. 1












|Table 8 Top|p-10 DTIs predicted by SHGNN for Goserelin and Epiru|ubicin|Col4|
|---|---|---|---|
|Drug|Target|Sign|Evidence|
|Goserelin|Apoptosis regulator Bcl-2|+|Goserelin results in increased expression of Bcl-2 protein [40]|
|Goserelin|Synaptic vesicle glycoprotein 2A|+|Null|
|Goserelin|Androgen receptor|−|Null|
|Goserelin|Gonadotropin-releasing hormone receptor|+|Agonist in DrugBank|
|Goserelin|Lutropin-choriogonadotropic hormone receptor|+|Agonist in DrugBank|
|Goserelin|Plasma kallikrein|−|Goserelin inhibited cell growth and Plasma kallikrein protein<br>secretion in LNCaP and C4-2 cells [41]|
|Goserelin|Somatostatin receptor|+|Null|
|Goserelin|Transmembrane protein serine 2|−|When treated with the combination of Goserelin and<br>Bicalutamide, Transmembrane protein serine 2 was strongly<br>inhibited in benign glands and moderately inhibited in malignant<br>glands [41]|
|Goserelin|Sterol o-acyltransferase 1|−|Null|
|Goserelin|Osteocalcin|−|Null|
|Epirubicin|Caspase-3|+|Epirubicin results in increased activity of Caspase-3 protein [42]|
|Epirubicin|ATP-binding cassette sub-family G member 1|+|Epirubicin analog results in increased expression of ATP-binding<br>cassette sub-family G member 1 mRNA [43]|
|Epirubicin|Fatty acid-binding protein|+|Null|
|Epirubicin|Estrogen receptor alpha|−|Epirubicin binds to and results in decreased activity of Estrogen<br>receptor alpha protein [44]|
|Epirubicin|Serine/threonine-protein kinase mTOR|+|Null|
|Epirubicin|Estrogen receptor beta|−|Epirubicin binds to and results in decreased activity of Estrogen<br>receptor beta protein [44]|
|Epirubicin|Nucleolar and coiled-body phosphoprotein 1|−|Null|
|Epirubicin|Synaptic vesicle glycoprotein 2A|−|Null|
|Epirubicin|DNA topoisomerase 2-alpha|−|Inhibitor in DrugBank|
|Epirubicin|Estrogen sulfotransferase|−|Null|



DTI related to Caspase-3 is supported in literature [42],
where Epirubicin was demonstrated to result in increased
activity of Caspase-3 protein. The link sign between
Epirubicin and ATP-binding cassette sub-family G member 1 is also predicted to be positive, which agrees with
that Epirubicin analog results in increased expression of
ATP-binding cassette sub-family G member 1 mRNA [43].
Within the negative DTIs, Epirubicin is predicted to interact with Estrogen receptor alpha (ER ) and Estrogen _α_
receptor beta (ER ). Estrogen antagonists and drugs _β_
that reduce estrogen biosynthesis have become highly
successful therapeutic agents for breast cancer patients,
the effects of estrogen are largely mediated by ER and _α_
ER [ _β_ 45]. A previous study [44] has shown that Epirubicin binds to and results in decreased activity of ER _α_
and ER . DNA topoisomerase 2-alpha as an inhibitor in _β_
DrugBank [26], is predicted to be negatively related to
Epirubicin. However, unconfirmed DTIs are still present
in the predicted results of Epirubicin, including Fatty
acid-binding protein, Serine/threonine-protein kinase
mTOR, Nucleolar and coiled-body phosphoprotein 1,
Synaptic vesicle glycoprotein 2A, and Estrogen sulfotransferase.


In addition, Synaptic vesicle glycoprotein 2A is
present in the top 10 DTIs for both Goserelin and Epiru


_α_

_β_



_α_
_β_ 45]. A previous study [44



_α_


and ER . DNA topoisomerase 2-alpha as an inhibitor in _β_



bicin, which are unproven DTIs in the relevant databases or literatures. In descriptions of DrugBank [26], Levetiracetam acts as an agonist of Synaptic vesicle glycoprotein 2A to treat various types of seizures caused by
epilepsy. This means that Goserelin and Epirubicin may
also be used as adjuvant treatment for such diseases.

In summary, half of 20 signed DTIs have evidence in
DrugBank [26] or have support in related literature. Although the known record between Epirubicin and CHD1
is not included within the Top-10 ranks, its score is still
very high and it belongs to top-50 DTIs of Epirubicin.
Especially, 7 records are out of DrugBank [26], which
further verifies the model performance in predicting new
signed DTIs.


**V. Conclusions**


DTIs prediction is a potential way to discover the
types of relations between drugs and target proteins,
which is of great significance for pharmaceutical
medicine. Existing computational methods can screen potential DTIs from a large number of drug pairs at low
cost, but they are mostly unable to predict specific types
of DTIs, such as positive and negative DTIs. In this paper, the DTIs prediction problem is regarded as the sign
prediction problem on signed heterogeneous networks,


Drug-Target Interactions Prediction Based on Signed Heterogeneous Graph Neural Networks 243



and an end-to-end prediction method based on the signed
heterogeneous graph neural networks (SHGNNs) is proposed to predict the link signs between drugs and targets. When designing SHGNNs, we dedicate message
passing and aggregation on signed bipartite networks,
and additionally incorporate DDIs and PPIs information,
further try several training modes. The performance of
the SHGNNs-based prediction method greatly exceeds
those of the baseline methods. We test the method with

different settings, including its working modes, embedding dimensions and initial features. In the case study,
two drugs for breast cancer are chosen for DTIs prediction and the results show feasibility of our method.

In future research, we will extend our method to
cold-start DTIs prediction problems with unknown drugs
and targets and consider multi-modal node attributes to
further improve the prediction performance.


**Acknowledgement**


This work was supported by the Shenzhen Science
and Technology Program (Grant No. KQTD20200820113
106007), the National Natural Science Foundation of
China (Grant No. U22A2041, 61972451, and 62272288),
the Guangdong Provincial Key Laboratory of Interdisciplinary Research and Application for Data Science, BNUHKBU United International College (Grant No. 2022B12
12010006), the Scientific Research Fund of Hunan Provincial Education Department of China (Grant No. 22B0097),
and the Changsha Natural Science Foundation of China
(Grant No. kq2202248).


**References**



sults to computational models,” _Briefings in Bioinformatics_,
vol. 21, no. 1, pp. 47–61, 2020.

G. Z. Zhang, M. L. Li, H. Deng, _et al_ ., “SGNNMD: Signed
graph neural network for predicting deregulation types of
miRNA-disease associations,” _Briefings in Bioinformatics_,
vol. 23, no. 1, article no. bbab464, 2022.

L. Guo, X. J. Lei, M. Chen, _et al_ ., “MSResG: Using GAE
and residual GCN to predict drug–drug interactions based on
multi-source drug features,” _Interdisciplinary Sciences: Com-_
_putational Life Sciences_, vol. 15, no. 2, pp. 171–188, 2023.

Y. C. Zhang, X. J. Lei, Y. Pan, _et al_ ., “Drug repositioning
with GraphSAGE and clustering constraints based on drug
and disease networks,” _Frontiers in Pharmacology_, vol. 13,
article no. 872785, 2022.

G. Zhao, Q. G. Wang, F. Yao, _et al_ ., “Survey on large-scale
graph neural network systems,” _Journal of Software_, vol. 33,
no. 1, pp. 150–170, 2022. (in Chinese)

L. Zhang, C. C. Wang, and X. Chen, “Predicting drug-target
binding affinity through molecule representation block based
on multi-head attention and skip connection,” _Briefings in_
_Bioinformatics_, vol. 23, no. 6, article no. bbac468, 2022.

H. Z. Wang, F. Huang, Z. K. Xiong, _et al_ ., “A heterogeneous
network-based method with attentive meta-path extraction
for predicting drug-target interactions,” _Briefings in Bioin-_
_formatics_, vol. 23, no. 4, article no. bbac184, 2022.

N. B. Torres and C. Altafini, “Drug combinatorics and side
effect estimation on the signed human drug-target network,”
_BMC Systems Biology_, vol. 10, no. 1, article no. 74, 2016.

B. F. Hu, H. Wang, and Z. M. Yu, “Drug side-effect prediction via random walk on the signed heterogeneous drug network,” _Molecules_, vol. 24, no. 20, article no. 3668, 2019.

Y. F. Shang, L. Gao, Q. Zou, _et al_ ., “Prediction of drug-target interactions based on multi-layer network representation
learning,” _Neurocomputing_, vol. 434, pp. 80–89, 2021.

Y. X. Gong, B. Liao, P. Wang, _et al_ ., “DrugHybrid_BS: Using hybrid feature combined with bagging-SVM to predict
potentially druggable proteins,” _Frontiers in Pharmacology_,
vol. 12, article no. 771808, 2021.

Y. Y. Chu, A. C. Kaushik, X. G. Wang, _et al_ ., “DTI-CDF: A
cascade deep forest model towards the prediction of drug-target interactions based on hybrid features,” _Briefings in_
_Bioinformatics_, vol. 22, no. 1, pp. 451–462, 2021.

Y. J. Ding, J. J. Tang, F. Guo, _et al_ ., “Identification of drugtarget interactions via multiple kernel-based triple collaborative matrix factorization,” _Briefings in Bioinformatics_, vol.
23, no. 2, article no. bbab582, 2022.

X. Chen, M. X. Liu, and G. Y. Yan, “Drug-target interaction prediction by random walk on the heterogeneous network,” _Molecular BioSystems_, vol. 8, no. 7, pp. 1970–1978,
2012.

Y. C. Zhang, X. J. Lei, Z. Q. Fang, _et al_ ., “CircRNA-disease
associations prediction based on metapath2vec++ and matrix factorization,” _Big Data Mining and Analytics_, vol. 3,
no. 4, pp. 280–291, 2020.

X. Liu and M. Yang, “Research on conversational machine
reading comprehension based on dynamic graph neural network,” _Journal of Integration Technology_, vol. 11, no. 2, pp.
67–78, 2022. (in Chinese)

Y. Y. Wang, X. J. Lei, Y. Pan, “Predicting microbe-disease
association based on heterogeneous network and global graph
feature learning,” _Chinese Journal of Electronics_, vol. 31, no.
2, pp. 345–353, 2022.

H. T. Fu, F. Huang, X. Liu, _et al_ ., “MVGCN: Data integration through multi-view graph convolutional network for predicting links in biomedical bipartite networks,” _Bioinformat-_
_ics_, vol. 38, no. 2, pp. 426–434, 2022.

D. S. Wishart, Y. D. Feunang, A. C. Guo, _et al_ ., “DrugBank
5.0: A major update to the drugbank database for 2018,” _Nu-_
_cleic Acids Research_, vol. 46, no. D1, pp. D1074–D1082,
2018.


T. Lee and Y. Yoon, “Drug repositioning using drug-disease
vectors based on an integrated network,” _BMC Bioinformat-_
_ics_, vol. 19, no. 1, article no. 446, 2018.




[1]


[2]


[3]


[4]


[5]


[6]


[7]



X. Q. Ru, X. C. Ye, T. Sakurai, _et al_ ., “Current status and
future prospects of drug-target interaction prediction,” _Brief-_
_ings in Functional Genomics_, vol. 20, no. 5, pp. 312–322,
2021.

X. Chen, C. C. Yan, X. T. Zhang, _et al_ ., “Drug-target interaction prediction: Databases, web servers and computational
models,” _Briefings in Bioinformatics_, vol. 17, no. 4, pp.
696–712, 2016.

C. C. Wang, Y. Zhao, and X. Chen, “Drug-pathway association prediction: From experimental results to computational
models,” _Briefings in Bioinformatics_, vol. 22, no. 3, article
no. bbaa061, 2021.

W. Zhang, W. R. Lin, D. Zhang, _et al_ ., “Recent advances in
the machine learning-based drug-target interaction prediction,” _Current Drug Metabolism_, vol. 20, no. 3, pp. 194–202,
2019.

Y. Pan, X. J. Lei, and Y. C. Zhang, “Association predictions
of genomics, proteinomics, transcriptomics, microbiome,
metabolomics, pathomics, radiomics, drug, symptoms, environment factor, and disease networks: A comprehensive approach,” _Medicinal Research Reviews_, vol. 42, no. 1, pp.
441–461, 2022.

X. H. Wu, J. W. Duan, Y. Pan, _et al_ ., “Medical knowledge
graph: Data sources, construction, reasoning, and applications,” _Big Data Mining and Analytics_, vol. 6, no. 2, pp.
201–217, 2023.

R. Li, X. Yuan, M. Radfar, _et al_ ., “Graph signal processing,
graph neural network and graph learning on biological data:
A systematic review,” _IEEE Reviews in Biomedical Engi-_
_neering_, vol. 16, pp. 109–135, 2023.




[9]


[10]


[11]


[12]


[13]


[14]


[15]


[16]


[17]


[18]


[19]


[20]


[21]


[22]


[23]


[24]


[25]


[26]


[27]




[8] X. Chen, N. N. Guan, Y. Z. Sun, _et al_ ., “MicroRNA-small
molecule association identification: From experimental re

244 Chinese Journal of Electronics, vol. 33, no. 1




[28]


[29]


[30]


[31]


[32]


[33]


[34]


[35]


[36]


[37]


[38]


[39]


[40]


[41]


[42]


[43]


[44]



M. Chen, Y. Pan, and C. Y. Ji, “Predicting drug drug interactions by signed graph filtering-based convolutional networks,” in _Proceedings of the 17th International Symposium_
_on Bioinformatics Research and Applications_, Shenzhen,
China, pp. 375–387, 2021.

M. Chen, W. Jiang, Y. Pan, _et al_ ., “SGFNNs: Signed graph
filtering-based neural networks for predicting drug–drug interactions,” _Journal of Computational Biology_, vol. 29, no.
10, pp. 1104–1116, 2022.

T. Derr, Y. Ma, and J. L. Tang, “Signed graph convolutional
networks,” in _Proceedings of_ _2018 IEEE International Con-_
_ference on Data Mining_, Singapore, pp. 929–934, 2018.

J. J. Huang, H. W. Shen, Q. Cao, _et al_ ., “Signed bipartite
graph neural networks,” in _Proceedings of the 30th ACM In-_
_ternational Conference on Information & Knowledge Man-_
_agement_, Queensland, Australia, pp. 740–749, 2021.

S. Kim, J. Chen, T. J. Cheng, _et al_ ., “PubChem in 2021: New
data content and improved web interfaces,” _Nucleic Acids_
_Research_, vol. 49, no. D1, pp. D1388–D1395, 2021.

D. Szklarczyk, A. L. Gable, K. C. Nastou, _et al_ ., “The string
database in 2021: Customizable protein–protein networks,
and functional characterization of user-uploaded gene/measurement sets,” _Nucleic Acids Research_, vol. 49, no. D1, pp.
D605–D612, 2021.

T. Derr, C. Johnson, Y. Chang, _et al_ ., “Balance in signed bipartite networks,” in _Proceedings of the 28th ACM Interna-_
_tional Conference on Information and Knowledge Manage-_
_ment_, Beijing, China, pp. 1221–1230, 2019.

T. N. Kipf and M. Welling, “Semi-supervised classification
with graph convolutional networks,” in _Proceedings of the_
_5th International Conference on Learning Representations_,
Toulon, France, 2017.

W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in _Proceedings of the_
_31st International Conference on Neural Information Pro-_
_cessing Systems_, Long Beach, CA, USA, pp. 1025–1035, 2017.

X. Wang, H. Y. Ji, C. Shi, _et al_ ., “Heterogeneous graph attention network,” in _Proceedings of_ _the World Wide Web_
_Conference_, San Francisco, CA, USA, pp. 2022–2032, 2019.

A. T. Jacobs, D. M. Castaneda-Cruz, M. M. Rose, _et al_ .,
“Targeted therapy for breast cancer: An overview of drug
classes and outcomes,” _Biochemical Pharmacology_, vol. 204,
article no. 115209, 2022.

Y. S. Lu, K. S. Lee, T. Y. Chao, _et al_ ., “A phase IB study of
alpelisib or buparlisib combined with tamoxifen plus goserelin in premenopausal women with HR-positive HER2-negative advanced breast cancer,” _Clinical Cancer Research_, vol.
27, no. 2, pp. 408–417, 2021.

Y. B. Baytur, K. Ozbilgin, S. Cilaker, _et al_ ., “A comparative
study of the effect of raloxifene and gosereline on uterine
leiomyoma volume changes and estrogen receptor, progesterone receptor, bcl-2 and p53 expression immunohistochemically in premenopausal women,” _European Journal of Obstet-_
_rics_ & _Gynecology and Reproductive Biology_, vol. 135, no. 1,
pp. 94–103, 2007.

E. A. Mostaghel, P. S. Nelson, P. Lange, _et al_ ., “Targeted androgen pathway suppression in localized prostate cancer: A
pilot study,” _Journal of Clinical Oncology_, vol. 32, no. 3, pp.
229–237, 2014.

Y. L. Lo and W. Wang, “Formononetin potentiates epirubicin-induced apoptosis via ROS production in HeLa cells _in_
_vitro_,” _Chemico-Biological Interactions_, vol. 205, no. 3, pp.
188–197, 2013.

Y. L. Lo and W. C. Tu, “Co-encapsulation of chrysophsin-1
and epirubicin in PEGylated liposomes circumvents multidrug resistance in HeLa cells,” _Chemico-Biological Interac-_
_tions_, vol. 242, pp. 13–23, 2015.

F. Fan, R. Hu, A. Munzli, _et al_ ., “Utilization of human nuclear receptors as an early counter screen for off-target activity:
A case study with a compendium of 615 known drugs,” _Toxi-_
_cological Sciences_, vol. 145, no. 2, pp. 283–295, 2015.



trogen receptor expression in human cancer,” _Experimental_
_Hematology_ & _Oncology_, vol. 7, article no. 24, 2018.


**Ming CHEN** is currently a Lecturer in the
College of Information Science and Engineering at Hunan Normal University, Changsha,
China. She received the M.S. degree in 2007
from Hunan Normal University, and the
Ph.D. degree in 2012 from Wuhan University.
Her current research interests mainly include
graph signal processing and deep learning.
(Email: chenming@hunnu.edu.cn)


**Yajian JIANG** is currently an M.S. student in
the College of Information Science and Engineering, Hunan Normal University. His current
research interests include bioinformatics, data
mining, and deep learning.
(Email: j_yj2020@hunnu.edu.cn)


**Xiujuan LEI** is currently a Professor in the
School of Computer Science at Shaanxi Normal University, Xi’an, China. She received the
M.S. and Ph.D. degrees from Northwestern
Polytechnical University, Xi’an, China, in 2001
and 2005, respectively. Her current research
interests mainly include intelligent computing
and bioinformatics. (Email: xjlei@snnu.edu.cn)


**Yi PAN** is currently a Professor of the Faculty of Computer Science and Control Engineering, Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences. He received the B.E. and M.E. degrees in computer engineering from Tsinghua University,
China, in 1982 and 1984, respectively, and the
Ph.D. degree in computer science from the
University of Pittsburgh, USA, in 1991. His

current research interests include bioinformatics and health in
formatics using big data analytics, cloud computing, and machine learning technologies. (Email: yi.pan@siat.ac.cn)


**Chunyan JI** is currently an Assistant Professor in Department of Computer Science of
BNU-HKBU United International College. She
received the M.S. and Ph.D. degrees in computer science from Georgia State University.
Her main research areas include deep learning,
bioinformatics and sound event detection.
(Email: chunyanji@uic.edu.cn)


**Wei JIANG** is currently an M.S. student in
the College of Information Science and Engineering, Hunan Normal University. His current
research interests include graph signal processing and deep learning.
(Email: jw2020@smail.hunnu.edu.cn)




[45] H. Hua, H. Y. Zhang, Q. B. Kong, _et al_ ., “Mechanisms for es

