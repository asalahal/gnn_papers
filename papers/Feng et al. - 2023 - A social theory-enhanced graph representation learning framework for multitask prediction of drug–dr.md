_Briefings in Bioinformatics_, 2023, **24(1)**, 1–11


**https://doi.org/10.1093/bib/bbac602**

**Problem Solving Protocol**

# **A social theory-enhanced graph representation learning** **framework for multitask prediction of drug–drug** **interactions**


Yue-Hua Feng, Shao-Wu Zhang, Yi-Yang Feng, Qing-Qing Zhang, Ming-Hui Shi and Jian-Yu Shi


Corresponding authors. Shao-Wu Zhang, E-mail: zhangsw@nwpu.edu.cn; Jian-Yu Shi, E-mail: jianyushi@nwpu.edu.cn


Abstract


Current machine learning-based methods have achieved inspiring predictions in the scenarios of mono-type and multi-type drug–
drug interactions (DDIs), but they all ignore enhancive and depressive pharmacological changes triggered by DDIs. In addition,
these pharmacological changes are asymmetric since the roles of two drugs in an interaction are different. More importantly, these
pharmacological changes imply significant topological patterns among DDIs. To address the above issues, we first leverage Balance
theory and Status theory in social networks to reveal the topological patterns among directed pharmacological DDIs, which are modeled
as a signed and directed network. Then, we design a novel graph representation learning model named SGRL-DDI (social theoryenhanced graph representation learning for DDI) to realize the multitask prediction of DDIs. SGRL-DDI model can capture the task-joint
information by integrating relation graph convolutional networks with Balance and Status patterns. Moreover, we utilize task-specific
deep neural networks to perform two tasks, including the prediction of enhancive/depressive DDIs and the prediction of directed DDIs.
Based on DDI entries collected from DrugBank, the superiority of our model is demonstrated by the comparison with other state-ofthe-art methods. Furthermore, the ablation study verifies that Balance and Status patterns help characterize directed pharmacological
DDIs, and that the joint of two tasks provides better DDI representations than individual tasks. Last, we demonstrate the practical
effectiveness of our model by a version-dependent test, where 88.47 and 81.38% DDI out of newly added entries provided by the latest
release of DrugBank are validated in two predicting tasks respectively.


Keywords: drug–drug interaction, graph representation learning, Balance theory, Status theory, multitask learning



Introduction


Polypharmacy—also termed as multidrug treatment—is becoming a promising strategy for treating complex diseases (e.g. diabetes and cancer) in recent years [1]. Nevertheless, the joint use
of two or more drugs trigger pharmacological changes (named
drug–drug interactions) may result in unexpected effects (e.g. side
effects, adverse reactions and even serious toxicity) [2]. Therefore,
it is crucial to identify drug–drug interactions (DDIs) before making safe polypharmacy. However, this task is still both expensive
and time-consuming _in vitro_ and _in vivo_ due to the vast space of
drug pairs. Over the past decade, the build-up of experimentally
determined DDI entries enables computational methods, especially machine learning-based methods, to identify the potential
DDIs [3].

With the advantages of both high efficiency and low costs,
various machine learning methods have been proved as promising
methods to provide preliminary screening of DDIs for further
experimental validation. Generally, they train models by using the
approved DDIs to infer the potential DDIs among massive unlabeled drug pairs. The training involves diverse drug properties,
such as chemical structure [4–8], targets [4, 5, 8, 9], anatomical
taxonomy [6, 9, 10] and phenotypic observation [7–10]. Earlier
methods focus on the mono-type DDI prediction, which just infers
whether a drug interacts with another or not. These methods are
usually implemented by classical classifiers (KNN [6], SVM [6],
logistic regression [4, 10], decision tree [11] and naïve Bayes [11]),
network propagation of the reasoning over drug–drug network



structure [10, 12], label propagation [13], random walk [5] and
probabilistic soft logic [9, 11] or matrix factorization [7, 8, 14].

Traditional machine learning methods focus on handextracted features and model-driven classifier design, whereas
the feature extraction process relies heavily on domain knowledge, so traditional machine learning methods are limited in
their ability to deal with raw data. Deep learning has achieved
initial success in DDI multi-type prediction due to its powerful
ability of automatic feature extraction [15–24]. In common, the
methods [15–17] first treat rows in a drug similar matrix as
corresponding drug feature vectors, and set the concatenation
of two feature vectors as the feature vector to represent a pair of
drugs, and then train a multilayer DNN with both feature vectors
and types of DDIs as the classifier to predict multi-type DDIs. In
addition, the methods [18, 21–24] construct graph neural network
framework for multi-type DDI prediction. Asada _et al_ . [25] utilizes
the DDI description of DrugBank and molecular structure for DDIs
prediction. GoGNN [22], MR-GNN [23] and SSI-DDI [18] extract
drug features from molecular structure graphs. After extracting
drug features, GoGNN [22] uses specific relationship matrix to
predict multi-type drug interactions and MR-GNN [23] uses a
2-layer MLP to predict multi-types DDIs. Although SSI-DDI [18]
contributes little to the improvement of prediction performance,
it makes an effort to DDIs interpretability. Moreover, MUFFIN [24]
and MIRACLE [23] construct a graph neural network framework
by combining DDI network with its molecular structure for multitype DDI prediction, in which MIRACLE uses the contrastive



**Received:** September 3, 2022. **Revised:** November 30, 2022. **Accepted:** December 6, 2022
© The Author(s) 2023. Published by Oxford University Press. All rights reserved. For Permissions, please email: journals.permissions@oup.com


2 | _Feng_ et al.


learning of mutual information maximization to learn drug
features.


Although these methods have achieved inspiring results, they
ignore enhancive and depressive pharmacological changes triggered by DDIs. In addition, they neglect the different pharmacological roles of two drugs in an interaction. More importantly,
such properties imply significant patterns (e.g. balance and status
triads) among DDIs [26]. To address the above issues, we first leverages Balance theory and Status theory from social networks to
characterize pharmacological patterns of DDI, and organize DDI
entries into a signed and directed network that reflects the relational semantics between drugs [27, 28], where edges with positive
sign and negative sign indicate enhancive and depressive pharmacological changes, source nodes and target nodes are influencing
drugs and affected drugs, respectively. And then, we design a
novel multitask graph representation learning model to capture
the task-joint information by integrating relation graph convolutional networks with Balance and Status patterns to predict the
enhancive/depressive DDIs and the directed DDIs. Through the
Balance theory and Status theory, we can reveal pharmacological
interaction patterns in the DDI network, which can help understand the underlying interaction mechanism between drugs.


Materials and methods

**Datasets**


We built the signed and directed DDI network by collecting DDI
entries from DrugBank (31 March 2021; [29]) in the following steps.
First, we downloaded the completed XML-formatted database
(including the comprehensive profiles of 11 440 drugs), and parsed
all approved small-molecule drugs (i.e. 1935 drugs) and their
589 827 DDIs entries.


Then, we organized them into a signed and directed DDI network where nodes are drugs and edges are their interactions, and
labeled the sign ‘+’ for enhancive DDIs and the sign ‘−’ for depressive DDIs according to the keywords ‘increase’ and ‘decrease’ in
interaction statements, respectively. Moreover, we set the influencing drugs as source nodes and the affected drugs as target
nodes according to their semantic roles in interaction statements.
For example, the interaction statement ‘The therapeutic efficacy
of Benmoxin can be increased when used in combination with

Pregabalin’ implies that the enhancive pharmacological change of
Benmoxin (the affected drug) has been induced by Pregabalin (the
influencing drug). Thus, the interaction between Pregabalin and
Benmoxin is labeled as ‘+’ and a directed edge from Pregabalin to
Benmoxin. Similarly, the interaction statement ‘The absorption of
Drug Rosuvastatin can be decreased when combined with Drug
Sodium bicarbonate’ indicates a degressive interaction between

−
Rosuvastatin and Sodium bicarbonate (denoted sign ‘ ’) and a
directed edge from Sodium bicarbonate to Rosuvastatin.

There are some issues that need to be stated. The first is

about extraction of interaction signs. First, It is much better if the
interaction signs are determined based on beneficial or adverse
effect of interaction statement semantics in DrugBank, but some
DDI statements could clearly tell whether the DDIs are beneficial
or adverse in terms of PD properties (e.g. the therapeutic efficacy
or adverse effect) and some could not be clearly determined
the benefit by their statements if they involve PK properties
(e.g. absorption, metabolism, serum concentration and especially
drug activity). For instance, the statements ‘Sodium bicarbonate
can cause a decrease in the absorption of Rosuvastatin’, and
‘Octreotide may increase the bradycardic activities of Ceritinib’.



We listed 10 interaction statement patterns of DDI-induced activ[ities as examples in Table S2 in Supplementary Data available](https://academic.oup.com/bib/article-lookup/doi/10.1093/bib/bbac602#supplementary-data)
online. Therefore, we use the keywords of ‘increase/decrease’ in
interaction statements to determine the signs of DDIs, which
only reflect their pharmacological enhancive/depressive changes.
The second issue is about extraction of interaction directions. We

determined influencing/affected roles between all drugs by their
interaction sentence patterns in DrugBank. Furthermore, after
checking through the 589 827 collected DDI entries, we found
no bi-directional interactions between drugs in DrugBank. And
we also verified consistent results in other databases, including
drugs.com and PubChem. Thus, there is no drug pair with mutual
interaction effects in DrugBank. Finally, we obtained a monodirected DDI network.


The statistical properties (i.e. the number of drugs, the number
of interactions, the number of enhancive DDIs, the number of
depressive DDIs, average degree and max degrees) of the signed
and directed DDI network built in this paper are summarized in
Table 1.


Moreover, we also extracted The MACCS fingerprints of
drug chemical structures and input them into the Task-joint
embedding module of SGRL-DDI along with the DDI network. In
details, after extracting the drug chemical structures represented
by simplified molecular input line entry system (SMEILS) strings
from the XML file of DrugBank, we encoded them by MACCSkeys
(Molecular ACCess System keys) Fingerprints [30]. Thus, each
drug is represented into a 166-dimensional binary vector, in
which the elements indicate the occurrence of a set of predefined
substructures.


**Problem formulation**


Suppose _n_ drugs _V_ = { _v_ _i_ } and _m_ interactions _L_ = { _l_ _ij_ } among these
drugs. The traditional DDI prediction and the multitask prediction
are the following different scenarios.


(i) The task of traditional DDI prediction learns a function
mapping _F_ : _V_ × _V_ →{0, 1} to deduce potential interactions
among unlabeled pairs of drugs in _V_ (Figure 1A).
(ii) The multitask prediction includes two subtasks, the task
of predicting enhancive and depressive pharmacological
changes and the task of predicting directed interactions
between drugs (Figure 1B). It learns two functions mapping
_F_ _sign_ : _V_ × _V_ → { _r_ +, _r_ − } and _F_ _direction_ : _V_ × _V_ → { _r_ _in_, _r_ _out_ },
where _r_ + _and r_ − denote the enhancive and depressive changes
triggered by an DDI (i.e. _v_ _i_ − −−− _[v]_ _[j]_ [, or] _[ v]_ _[i]_ [ +] −−− _[v]_ _[j]_ [), and] _[ r]_ _[in]_ [ and] _[ r]_ _[out]_

denote the directed relations between two interacting drugs
(i.e. _v_ _i_ → _v_ _j_, or _v_ _i_ → _v_ _j_ _)_, respectively.


**Balance theory and Status theory**


Balance theory and Status theory are classical sociological theories that play an essential role in analyzing and modeling signed
and directed graphs. According to Balance theory, the positive
sign of an edge represents trust, like and approval relationship,
whereas the negative sign indicates hostility, distrust, hate and
disapproval relationship between two nodes in social networks

[31]. In addition, the directed edge between nodes also reflects different relations and semantics [27, 32]. For example, in a directed
social network, nodes having many incoming edges and few outgoing edges are termed as celebrities, and in contrast, nodes
having few incoming edges and many outgoing edges are termed
as followers [33]. Analogous to drugs, celebrities are influencing


_A social theory-enhanced graph representation_ | 3


**Table 1.** Statistical properties of the signed and directed DDIs network


**# Drug** **# Interaction** **# Enhancive DDIs** **# Depressive DDIs** **Average degree** **Max degree**


**In_degree** **Out_degree**


1935 589 827 295 495 294 332 305 1274 1047


# Denotes the number.


Figure 1. Two DDI prediction scenarios. ( **A** ) Traditional DDI prediction. ( **B** ) Multitask prediction (i.e. pharmacological changes and directed relation).


Figure 2. Four types of triads in a signed network. ( **A** ) and ( **B** ) Strongly balanced. ( **C** ) Weakly balanced. ( **D** ) Unbalanced.



drugs, whereas followers are affected drugs in interactions. Status theory extends the application of balance theory to signed
directed network. It supposes directed relationship labeled by a
positive sign ‘+’ or negative sign ‘−’ means target node has a
higher or lower status than source node [34].

Therefore, in the signed and directed DDIs network, we use
the positive or negative sign to denote enhancive or depressive
pharmacological change and the direction to indicate the asymmetrical relation between two interacting drugs of a DDI. We used
a signed and directed network to represent the comprehensive
DDIs, and employed these social theories to extract topology feature of the DDIs network. In this section, we will briefly introduce
these two theories.


Balance theory is widely used to determine how balanced a
signed network is [35]. As shown in Figure 2A and B, triangle patterns (i.e. triads) represent the relations between three nodes in a
signed graph, where an even number of negative edges are taken
as the strong balanced [36]. For a toy example, Figure 2A and B
illustrate the strong balanced triads. Figure 2C is a weak balanced
triad according to the weak Balance theory [37],whereas Figure 2D
shows an unbalanced triads. Specifically, strong balanced triads
exemplify the principle that a person is also my friend if he/she
is my friend’s friend, and a person is my enemy if he/she is my
friend’s enemy. If the number of strong triads is significantly more
than the number of other triads, the signed network is called
balanced.


The Status theory is an extension of the Balance theory in the
case of networks with both signs and directions. It supposes that if



there is a positive link from node _v_ _i_ to node v _j_, then v _j_ is considered
to have a higher social status than _v_ _i_ . Also, if there is a negative
link from _v_ _i_ to _v_ _j_, then _v_ _i_ is considered to have a higher social
status than _v_ _j_ . In other words, we can formulate the Status theory


+ −
as that if _v_ _i_ → _v_ _j_, then _S_ _v_ _i_ _< S_ _v_ _j_ _,_ and if _v_ _i_ → _v_ _j_ _,_ then _S_ _v_ _i_ _> S_ _v_ _j_, where
_S_ _v_ _i_ denotes the status score of _v_ _i_ . In social networks, the status
may indicate relative prestige or reputation. There are totally 12
triads in a signed and directed network. For short, we illustrated
four cases in Figure 3, where the triads in Figure 3A and B satisfy
the Status theory, whereas the triads in Figure 3C and D do not.
Similarly, if the number of status triads is significantly more than
the number of other triads, the signed and directed network meets
the Status theory. In the context of the DDI network, we also
verified it meets both the Balance theory and Status theory in
Section ‘Balance and Status triad statistics’, and these patterns
can improve the representation of DDI.


**SGRL-DDI model**


We propose a multitask learning framework SGRL-DDI to deal
with two tasks, including the prediction of enhancive/depressive
DDIs and the prediction of directed DDIs. SGRL-DDI mainly consists of two modules (Figure 4). The first module is Task-joint
embedding, which is composed of a two-layer embedding and an
extra enhancer based on social theory. Each embedding layer is
constructed by a multi-relation GNNs [38, 39] and a two-layer
MLP (multilayer perception) to represent drugs in DDIs network
in two tasks simultaneously, whereas the status-based enhancer


4 | _Feng_ et al.


Figure 3. Four types of triads in a signed and directed network. ( **A** ) and ( **B** ) satisfy the Status theory. ( **C** ) and ( **D** ) do not satisfy the Status theory.


Figure 4. The overall framework of SGRL-DDI. ( **A** ) Task-joint embedding module. The two-layer multi-relation GNNs enhanced by Balance and Status
theories are built to encode drugs in the DDIs network into feature vectors Z, which captures topological properties of the network combining with
both information of signs and directions. ( **B** ) Task-specific prediction module. Concatenating two drug latent features to fed into two dense DNNs for
implementing two prediction tasks. ( **C** ) Illustration of an embedding layer in task-joint embedding module. A central node (i.e. green node) aggregates

+ − +
both the features of its first-order neighbor nodes (i.e. orange) and that of its own separately from different relations (i.e. _v_ _i_ → _v_ _j_ _,v_ _i_ → _v_ _j_ _,v_ _i_ → _v_ _j_ and

_v_ _i_ → − _v_ _j_ ) to update its features _z_ _i_ . Then, all the updated features are concatenated to feed into a two-layer DNN for generating the final embedding _z_ _i_ of
node _v_ _i_ .



enforces the drug task-joint embedding to follow the Balance
theory and the Status theory (Figure 4A). The second module is
a task-specific prediction (Figure 4B), in which the concatenation
vectors of two drug latent features are fed into two dense DNNs
to achieve two tasks of predicting enhancive/depressive DDIs
and predicting the directed DDIs. In SGRL-DDI model framework,
the task-joint features of drugs characterize the complementary
information of the two tasks, whereas the task-specific dense
DNNs capture their exclusive information.


_Task-joint embedding module_


We use a graph _G(_ _V_, _E_ _)_ to represent the signed and directed
DDIs network, where _V_ = { _v_ 1, _v_ 2, _. . ._, _v_ _n_ } is the set of drug nodes,
_E_ = { _(v_ _i_, _r_, _v_ _j_ _)_ _[n]_ _i_, _j_ =1 [,] _[ r]_ [ ∈] _[R]_ [}][ is the triple set of comprehensive interac-]
tions between drug nodes (Figure 4A), and _R_ = { _r_ → + [,] _[ r]_ → [ −] [,] _[ r]_ → [ +] [,] _[ r]_ → [ −] [}][ is]


+ −
the interaction type set with signs and directions ( _v_ _i_ → _v_ _j_ _, v_ _i_ → _v_ _j_ _,_



+ −
_v_ _i_ → _v_ _j_ _,_ and _v_ _i_ → _v_ _j_ ). For example, the triplet _(v_ _i_, _r_ → + [,] _[ v]_ _[j]_ _[)]_ [ denotes]
the interaction from drug _v_ _i_ to drug _v_ _j_ with sign ‘+’. Thus, except
for indicating interaction occurrence in the binary DDI network,
these edges contain signs and directions between drug nodes.

To propagate and aggregate information on such signed and
directed DDIs network, we designed a task-joint embedding module, in which each layer contains four parallel GATs and a two+
layer MLP. These GATs account for four relation triples (i.e. _v_ _i_ → _v_ _j_ _,_


− + −
_v_ _i_ → _v_ _j_ _, v_ _i_ → _v_ _j_ and _v_ _i_ → _v_ _j_ ) for propagating and aggregating
information for each drug from its different relational neighbors,
respectively. The MACCSkeys fingerprints of drug chemical structures are taken as original node features _h_ _i_ _[(]_ [0] _[)]_ in the DDI network.
Formally, the general propagation rule is defined as:



_h_ _i_ _[(]_ _[k]_ [+][1] _[)]_ = _σ_
��



_j_ ∈ _N_ _[r]_ _i_ _α_ _ij_ _[r]_ **[W]** _[r]_ _[(]_ [k] _[)][h]_ _[(]_ _j_ _[k][)]_ + **W** _[r][(][k][)]_ _h_ _[(]_ _i_ _[k][)]_ _[)]_ � (1)


where _N_ _[r]_ _i_ [denotes the set of] _[ v]_ _[i]_ [’s neighbors in relation] _[ r][(][ r]_ [ ∈] _[R][)]_ [,] _[ h]_ _i_ _[(][k][)]_
is the feature vectors from the layer _k_, **W** _r_ _[r][(][k][)]_ is the trainable weight
matrix of relation _r_ and _σ_ is a nonlinear element-wise activation
function (i.e. ReLU). Moreover, _α_ _ij_ _[r]_ [is the weight value between drug]
_v_ _i_ and drug _v_ _j_ in relation _r_ accumulated by attention mechanism
defined as:



_A social theory-enhanced graph representation_ | 5


The _q(v_ _i_, _v_ _j_ _)_ is the modified reduction of status-scores of two drugs
defined as follows:



max � _s_ _d_ _i_ − _s_ _d_ _j_, 0.5�, _if d_ _i_ → − _d_ _j_

+ (7)
min � _s_ _d_ _i_ − _s_ _d_ _j_, −0.5�, _if d_ _i_ → _d_ _j_



_q_ � _v_ _i_, _v_ _j_ � =



⎧
⎨

⎩



_α_ _ij_ _[r]_ [=][ softmax] � _e_ _ij_ � = ~~�~~ _k_ ∈ exp _N_ _[r]_ _i_ [exp] � _e_ _ij_ _[ (]_ � _[e]_ _[ik]_ _[)]_ (2)



where 0.5 is a threshold for separating the nodes with high/low
status-scores, and the status score of drug _v_ _i_ is computed by
_s_ _d_ _i_ = sigmoid _(_ **W** - _z_ _i_ + _b)_ .

The third loss function _L_ triangle measures how well node triads
_�_ _(i_, _j_, _k)_ meet the Balance theory.



_e_ _ij_ = LeakyReLU �−→ **a** T _r_ � **W** _[r]_ att _[h]_ _i_ _[(]_ [1] _[)]_ ∥ **W** _[r]_ att _[h]_ _j_ _[(]_ [1] _[)]_ �� (3)


where LeakyReLU is nonlinearity activation function (with negative input slope _α_ = 0.2), " ∥ " is the concatenation operation,
**W** _[r]_ att [∈] [R] _[F]_ [×] _[F]_ [′] [ is the weight matrix parameter,][ −→] **a** _r_ ∈ R [2] _[F]_ [′] is a shared
attention weight parameter vector denoted by a single-layer feedforward neural network, and _T_ denotes transposition operator. For
each relation, we obtained the embedding feature vector _h_ _[r]_ _i_ _[(][k]_ [+][1] _[)]_ for
drug _v_ _i_ .

In addition, the two-layer MLP accounts for the integration of
four triple representations to generate the integrated embedding
vector _h_ _i_ _[(][k]_ [+][1] _[)]_ of drug _v_ _i_ (Figure 4C). It works as an adhesive to
extract task-joint embeddings.


_h_ _i_ _[(]_ _[k]_ [+][1] _[)]_ = **W** _m_ _[(]_ _[k]_ 2 [+][1] _[)]_ � **W** _m_ _[(]_ _[k]_ 1 [+][1] _[)]_ � _h_ _ri_ 1 _(_ _k_ +1 _)_ ∥ _h_ _ri_ 2 _(_ _k_ +1 _)_ ∥ _h_ _ri_ 3 _(_ _k_ +1 _)_ ∥ _h_ _ri_ 4 _(_ _k_ +1 _)_ ��

(4)
where **W** _m_ _[(][k]_ 2 [+][1] _[)]_ and **W** _m_ _[(][k]_ 1 [+][1] _[)]_ are the trainable weight matrix in the
( _k_ + 1)-th layer of the MLP.

Finally, the task-joint embedding module adopts two sequential layers to enhance the nonlinear representation of signed and
directed interactions (denoted as _Z_ _i_ ∈ R [1][×][H] [2] in Figure 4).


_Social theory-enhanced loss functions in task-joint_
_embedding_


Corresponding to the components in the task-joint embedding
module, we considered three losses. The first one is traditional
loss function _L_ _network_ (i.e. cross-entropy loss) for reconstructing
the DDIs network, which measures the difference between the



_�(_ _i_, _j_, _k_ _)_ ∈T _[L]_ _[�(]_ _[i]_ [,] _[j]_ [,] _[k]_ _[)]_
(8)



_L_ triangle = − log _J_ triangle = −�



_�(_ _i_, _j_, _k_ _)_ ∈T [log] � _J_ _�(_ _i_, _j_, _k_ _)_ � = �



_J_ triangle = � _�(_ _i_, _j_, _k_ _)_ ∈T _[J]_ _[�(]_ _[i]_ [,] _[j]_ [,] _[k]_ _[)]_ (9)



J _�(_ _i_, _j_, _k_ _)_ = _f_ � _p_ � _v_ _i_, _v_ _j_ �� ∗ _f_ � _p_ � _v_ _i_, _v_ _k_ �� ∗ _f_ � _p_ � _v_ _k_, _v_ _j_ �� (10)


L _�(_ _i_, _j_, _k_ _)_ = _L_ _ij_ + _L_ _ik_ + _L_ _kj_, and _L_ _ij_ = _y_ _ij_ log � _p_ � _v_ _i_, _v_ _j_ ��


+ �1 − _y_ _ij_ � _(_ 1 − log � _p_ � _v_ _i_, _v_ _j_ �� (11)



_f_ � _p_ � _v_ _i_, _v_ _j_ �� =



1 − _p_ � _v_ _i_, _v_ _j_ �, _if_ d i → +− d j (12)
� _p_ � _v_ _i_, _v_ _j_ �, _if_ d i → d j



original network _G_ and the reconstructed network



∼
_G_ .



_L_ _network_ � _v_ _i_, _v_ _j_ � =−� _v_ _i_, _v_ _j_ ∈ _E_ _[y]_ _[ij]_ [ log] � _p_ � _v_ _i_, _v_ _j_ ��+�1 − _y_ _ij_ ��1 − log � _p_ � _v_ _i_, _v_ _j_ ���

(5)
where y ij is the true sign label of edge _(v_ _i_, _v_ _j_ _)_ for the pair of
drugs _v_ _i_ and _v_ _j_, _E_ is the set of edges between drug nodes representing DDI, _p(v_ _i_, _v_ _j_ _)_ = _σ(_ z _i_ - z _j_ [T] _)_ is the predicting probability computed by the inner product of feature vectors of two
drugs that have a link in DDI network, _σ_ is the sigmoid function, and the feature vectors Z is generated by the task-joint
embedding.

Since it is expected that the reconstructed network follows the
balance theory and state theory, the other two losses account
for direction patterns and sign patterns among in interaction
triads. That is, the second loss function _L_ _direction_ (i.e. square
loss) measures the difference between the modified reduction
_q(v_ _i_, _v_ _j_ _)_ of status-scores of two drugs and their true reduction
_(s_ _d_ _i_ − _s_ _d_ _j_ _)_ [40].



where p _(v_ _i_, _v_ _j_ _)_ is the predicting probability of edge _(v_ _i_, _v_ _j_ _)_ computed
by _σ(z_ _i_ - _z_ _j_ [T] _)_, _J_ _�_ _(i_, _j_, _k)_ is the balance score of a triad and _J_ triangle is the
balance score of the whole network.


Finally, the total loss function of the task-joint embedding
module can be formulated as:


_L_ _T_ = _L_ network + _L_ direction + _L_ triangle (13)


_Task-specific prediction module_


Once drug feature vectors containing signs and directions
of DDIs are generated by the task-joint embedding module, we concatenated pairwise embedding feature vectors to
form the task-joint feature vectors _h(v_ _i_, _v_ _j_ _)_ = [ _h_ _i_, _h_ _j_ ] of drug
pairs.

After obtaining feature vectors of drug pairs, two DNN-based
predictors are separately trained for two prediction tasks. In the
first prediction task, the positive samples are the edges with the
sign ‘+’ and the negative samples are the ones with the sign ‘−’.
In the second prediction task, the positive samples are all edges
in the DDIs network, whereas the negative samples are the same
drug pairs as the positive samples but with opposite directions. For
example, if an interaction is _A_ → _B_ labeled as a positive sample,
then its directional inverse _A_ → _B_ is labeled as a negative sample.
Each of the two prediction tasks employs in common a binary
cross-entropy loss function _L_ _P_ _(p_, _q)_ .



_L_ _P_ � _p_, _q_ � = −�



v i, v j ∈ _E_ _[p]_ � _v_ _i_, _v_ _j_ � log � _q_ � _v_ _i_, _v_ _j_ ��



+ �1 − _p_ � _v_ _i_, _v_ _j_ �� _(_ 1 − log � _q_ � _v_ _i_, _v_ _j_ �� (14)


where _p(v_ _i_, _v_ _j_ _)_ is the true label of the interaction _(v_ _i_, _v_ _j_ _)_, _q(v_ _i_, _v_ _j_ _)_ is
the predicting probability of interaction.



_L_ direction � _v_ _i_, _v_ _j_ � = �



_v_ _i_, _v_ _j_ ∈ _E_



2
� _q_ � _v_ _i_, _v_ _j_ � − � _s_ _d_ _i_ − _s_ _d_ _j_ �� (6)


6 | _Feng_ et al.


Figure 5. Eight different types of triangles in the signed and directed DDIs network.



**Assessment metrics**


We randomly partition the dataset into a training set (contains
75% samples), a validation set (contains 5% samples) and a testing
set (contains 20% samples). The training set is used to train the
model, the validation set is used to tune the parameters of the
model, whereas the testing set is used to assess how well the
trained model is. The procedure is repeated 10 times, and
the average performance is adopted to evaluate the prediction
performance of model.

The Accuracy (ACC), Precision, Recall, F1-score, AUC (area
under the receiver operating characteristic curve) and AUPR
(area under the precision-recall curve) are used to assess the
performance of SGRL-DDI. Receiver operating characteristic curve
reveals the relationship between true-positive rate (precision) and
false-positive rate based on various thresholds. Precision-recall
curve reveals the relationship between precision (true-positive
rate) and recall based on various thresholds. These metrics are
defined as follows:


_TP_ + _TN_
Accuracy = (15)
_TP_ + _FP_ + _TN_ + _FN_


_TP_
Precision = (16)
_TP_ + _FP_


_TP_
Recall = (17)
_TP_ + _FN_



_F_ 1 = [2][ ×] _[ Precision]_ [ ×] _[ Recall]_ (18)

_Precision_ + _Recall_



two tasks provides better DDI representations than individual
tasks. Eventually, we tested our SGRL-DDI with newly added
entries in latest release of DrugBank both for two predicting tasks,
respectively.


**Balance and Status triad statistics**


Balance theory and Status theory are two core sociological theories that play an essential role in modeling signed and directed
networks. The Balance theory reflects the relationship of triangular structures (triads), whereas the Status theory models the
directed relationship between two nodes. In the context of the
signed and directed DDI network, there are 26 475 975 triads
in total, which are grouped into eight types shown in Figure 5.
According to Balance theory and Status theory, we counted how
many triads are of strong balance patterns, weak balance patterns
and how many triads meet status patterns (Table 2). The proportional distribution of different patterns is shown in Figure 6.

As observed, only in the case of strong balance, ∼23.6% of triads
satisfy Balance theory, and 67.8% satisfy Status theory (Figure 6A).
If the weak balance is considered as a balance, ∼82.3% of triads
can be consistent with both theories (Figure 6B). Meanwhile, only
a tiny number of triangles (0.028%) meet neither the Balance
theory nor the Status theory. Therefore, the signed and directed
DDI network is similar to the social networks, which meets both
the Balance theory and the Status theory. Integrating these two
theories with GNNs can represent signed and directed DDIs better.


**Performance of SGRL-DDI in sign prediction**


In order to validate the performance of SGRL-DDI in sign interaction prediction, we compared SGRL-DDI with other three baseline
methods of GCN [42], SNEA [43] and SGCN [44]. The GCN [42] is
one of the outstanding graph representation learning methods
for binary networks. It maps drug nodes of the network into a
latent space to obtain their latent feature vectors for capturing
the topological relationship from its neighborhood drugs. Both
SNEA [43] and SGCN [44] consider that negative sign links not only
have different semantic meanings compared with positive sign
links, but also their principles are inherently different. Therefore,
these two methods extended the graph representation learning
of unsigned networks to that of signed ones. First, they employed
Balance theory to aggregate and propagate information for each
node from its neighboring nodes that are connected by different
sign types of links. Then they used a logic regression as the binary
classifier to train and classify the positive signs and negative
signs for links. Differently, SNEA [43] leveraged GATs with a selfattention mechanism, whereas SGCN [44] utilized standard GCNs



where _TP_, _FP_, _TN_ and _FN_ refer to the numbers of true-positive
samples, false-positive samples, true negative samples and false
negative samples, respectively. AUC value depends on the average
ranks of all true DDIs, whereas AUPR punishes the incorrect
predictions of top ranking DDIs more than AUC when the number
of negative samples is much larger than the number of positive
samples [41].


Results and discussion


In this section, we first revealed the topological patterns and
analyzed the signed and directed DDIs network to verify whether
it meets the Balance theory and Status theory. Then, we compared
our model with the state-of-the-art methods that are usually
applied in social networks for both the sign prediction and the
direction prediction. After that, we performed ablation studies
to verify whether Balance and Status patterns help characterize
the signed and directed pharmacological DDIs, and the joint of


_A social theory-enhanced graph representation_ | 7


**Table 2.** Statistics on Balance and Status theories in the signed and directed DDIs network


**Triad label** **#** **Triad** **Proportion** **Strong balance** **Weak balance** **Status**


T1 2 266 017 0.086 √ √ √

T2 825 613 0.031 × × √

T3 1 588 429 0.060 × × √

T4 747 020 0.028 × × ×

T5 2 296 230 0.087 √ √ √

T6 1 690 123 0.064 √ √ √

T7 1 518 931 0.057 √ √ ×
T8 15 543 612 0.587 × √ √


# Denotes the number of each type of triad. Proportion denotes the proportion of this triad in all triad. The three remaining columns indicate whether each
triad type satisfies the strong Balance, the weak Balance and the Status, respectively. √ Denotes yes. [×] Denotes no.


Figure 6. Proportional distribution of triangles satisfying Balance theory and/or Status theory. ( **A** ) Statistics without considering weak balance. ( **B** )
Statistics with considering weak balance as a kind of balance.



to obtain the latent representation vectors for each node in signed
networks.


In the process of sign prediction, enhancive DDIs were considered as positive samples (signed with ‘+’), and depressive DDIs

−
were labeled as negative samples (signed with ‘ ’). All these
samples were split into a training set, a validating set and a testing
set. Since GCN is only suitable for unsigned networks, we only took
above strategy for validating and testing samples, but made no
change on the positive and negative samples of the training set.

In order to learn an optimal model of sign and direction prediction, we tuned the parameters in all the methods. GCN [42],
SNEA [43] and SGCN [44] were implemented with their published
source codes, and the parameters have been tuned to optimal
values. All methods adopt a two-layer GNN, where the dimension
of embeddings is 64. In addition, the head number of SNEA is

set to 1.


In our SGRL-DDI, the hyper-parameters of the task-joint drug
embedding module include learning rate, epochs, batch size, as
well as neuron numbers in hidden layers. They are tuned by performing a grid search to the minimum value of the loss functions.
The hyper-parameters of the task-specific prediction module are
also selected by a grid search to obtain the best prediction results.
The optimal values of the hyper-parameters in SGRL-DDI are
[listed in Table S1, Supplementary Data available online.](https://academic.oup.com/bib/article-lookup/doi/10.1093/bib/bbac602#supplementary-data)

The comparison results in Table 3. The raw GCN treats all edges
as positive samples and non-DDIs as negative samples. Obviously,
it is incapable of discriminating enhancive and depressive interactions since all enhancive and depressive interactions are positive



samples in the training process. A very low precision (0.5015)
and an extremely high recall (0.9977) jointly reveal this issue and
results in AUC = 0.3947 lower than the random guess.

SNEA and SGCN achieve significantly better prediction than
GCN due to their characterization of different signs. In contrast,
our SGRL-DDI achieves the best performance in predicting
enhancive/depressive interactions with the significant improvements of 4.5–6.7%, 5.23–10.8%, 6.14–13.4%, 6.46–9.2%, 7.94–15.3%
and 3.25–10.2% against SNEA and SGCN in terms of AUC, AUPR,
F1-score, Accuracy, Precision and Recall, respectively. In short,
the results demonstrate the superiority of SGRL-DDI in sign
prediction.


**Performance of SGRL-DDI in direction prediction**


In order to evaluate the performance of SGRL-DDI in the directed
interaction prediction, we compared it with other four baseline
methods of GCN [42], GGCN-s/t [45], GGCN [45] and DGGAN [46].
These three methods for directed link prediction generally learn
two embedding vectors for each node, including a source vector
for its outgoing links and a target vector for its incoming links.
They are summarized as follows.

GGCN-s/t [45] measures the likelihood of a link from node _i_ to
node _j_ by **A** [ˆ] _ij_ = _σ(z_ _[(]_ _i_ _[s][)]_ _[z]_ _[(]_ _j_ _[t][)]_ [T] _)_ and the likelihood of a link from node _j_ to

node _i_ by **A** [ˆ] _ji_ = _σ(z_ _j_ _[(][s][)]_ _[z]_ _[(]_ _i_ _[t][)]_ [T] _)_, where _z_ _i_ _[(][s][)]_ is the source vector and _z_ _[(]_ _i_ _[t][)]_ is
the target vectors. The bigger likelihood determines the direction
of the edge _(v_ _i_, _v_ _j_ _)_ . DGGAN [46] uses adversarial mechanisms
to deploy a discriminator and two generators that jointly learn


8 | _Feng_ et al.


**Table 3.** Results of SGRL-DDI and other three methods for sign prediction


**Methods** **AUC** **AUPR** **F1** **Accuracy** **Precision** **Recall**


GCN 0.3947 0.4358 0.6675 0.5034 0.5015 0.9977

SNEA 0.8845 0.8435 0.7656 0.8030 0.7272 0.8088

SGCN 0.9069 0.8985 0.8346 0.8304 0.8008 0.8785

SGRL-DDI 0.9515 0.9511 0.8960 0.8950 0.8802 0.9110


**Table 4.** Results of SGRL-DDI and other four methods for direction prediction


**Methods** **AUC** **AUPR** **F1** **Accuracy** **Precision** **Recall**


GCN 0.5000 0.5000 0.5000 0.5000 0.5000 0.6670

GGCN-s/t 0.7235 0.7332 0.6942 0.6142 0.5750 0.8757

GGCN 0.6452 0.6460 0.6668 0.5042 0.5024 0.9921

DGGAN 0.6742 0.6760 0.8213 0.8011 0.7497 0.9054

SGRL-DDI 0.9234 0.9243 0.8396 0.8385 0.8157 0.8743


Figure 7. Ablation comparison in the signed interaction prediction ( **A** ) and in the directed interaction prediction ( **B** ).



each node’s source and target vectors. For a given node, the two
generators are trained to generate its fake target neighbor nodes
and source neighbor nodes from the same underlying distribution,
whereas the discriminator aims to distinguish whether a neighbor
node is real or fake. GGCN [45] computes the acceleration value
of each node embedding to indicate the likelihood that node _i_
is connected to node _j_ in the directed graph, and integrates the
acceleration values of nodes and node embeddings to build up an
asymmetric graph decoding scheme.

In the directed interaction prediction, all DDIs in the network
are considered the positive samples. For each positive sample
_(v_ _i_, _v_ _j_ _)_ (i.e. a directed edge from node _v_ _i_ to node _v_ _j_ ), we form its
reverse edge _(v_ _j_, _v_ _i_ _)_ as the corresponding negative sample. All
these positive and negative samples are randomly split into a
training set, a validating set and a testing set. As the GCN is only
suitable for unsigned and undirected networks, we only adopted
the above strategy for validating and testing samples. GGCN-s/t,
GGCN and DGGAN were implemented with their published source
codes and the parameters have been tuned to optimal values.

The comparison results (Table 4) of the directed interaction
prediction show that the standard GCN has the worst predicting
results since it ignores the directions when reconstructing the
adjacency matrix from node embeddings. Indeed, due to the symmetric decoder of the inner product **A** [ˆ] _ij_ = _σ(z_ [T] _i_ _[z]_ _[j]_ _[)]_ [ =] _[ σ(][z]_ [T] _j_ _[z]_ _[i]_ _[)]_ [ = ˆ] **[A]** _[ji]_ [,]
it obtains the same probability between an edge _(v_ _i_ → _v_ _j_ _)_ and the
reverse edge _(v_ _i_ → _v_ _j_ _)_ . As a consequence, as shown in Table 4,
standard GCN is incapable of predicting the directed link in the



directed networks, where relationships are not always reciprocal.
GGCN-s/t, GGCN and DGGAN achieve significantly better prediction results than GCN, due to their characterization of directed
edges. In contrast, our SGRL-DDI achieves the best performance
with the significant improvements of 20–27.8%, 19.1–27.8, 1.82–
17.3%, 3.74–33.4% and 6.6–31.3% against GGCN-s/t, GGCN and
DGGAN methods in terms of AUC, AUPR, F1-score, Accuracy and
Precision, respectively. In short, the results demonstrate the superiority of SGRL-DDI in direction prediction.


**Ablation study**


In this section, we investigated how the Balance theory and Status
theory used in the task-joint embedding module improve the
performance of SGRL-DDI. We made three ablated versions of
SGRL-DDI. The first version of SGRL-DDI (denoted as SGRL-DDIw/oBS) was implemented by removing both Balance-enhanced
and Status-enhanced parts from SGRL-DDI. The second one
(denoted as SGRL-DDI-w/oB) was implemented by removing the
Balance-enhanced part. The third one (denoted as SGRL-DDIw/oS) was implemented by removing the Status-enhanced part.
SGRL-DDI and its three variants were evaluated both in the signed
interaction prediction (Figure 7A) and the directed interaction
prediction (Figure 7B).

The results in Figure 7 reveal the following crucial points. (i)
SGRL-DDI with both the balance-enhanced part and the statusenhanced part achieves the best prediction, whereas SGRL-DDIw/oBS without social theory-based enhancer parts gives the worst


_A social theory-enhanced graph representation_ | 9


**Table 5.** Results of SGRL-DDI in the signed and directed interaction prediction for newly added DDIs entries


**Two tasks** **AUC** **AUPR** **F1** **Accuracy** **Precision** **Recall**


Signed DDI 0.9407 0.9423 0.8847 0.8847 0.8722 0.9023
Directed DDI 0.8973 0.8979 0.8233 0.8138 0.7899 0.8765



prediction results. (ii) Both SGRL-DDI-w/oB and SGRL-DDI-w/oS
achieve better prediction results than SGRL-DDI-w/oBS, indicating
that Balance-enhanced and Status-enhanced parts can improve
the prediction performance. In addition, we also see that the
results of SGRL-DDI-w/oS is better than that of SGRL-DDI-w/oB

for the directed prediction, showing that Balance-enhanced part
contributes more than Status-enhanced part for directed DDI
prediction. Therefore, the social theory-based enhancer in SGRLDDI plays a crucial role by capturing topological patterns in the
signed and directed DDI network.


**Validation of SGRL-DDI tested by newly added**
**DDIs entries**


Except for cross-validation,we further made a version-independent
test to validate the ability of our SGRL-DDI for predicting the new
DDIs with signed and directed interactions. We extracted newly
added interaction entries (i.e. 61 740 DDIs) from the latest release
of DrugBank (version 5.1.8, 20 September 2021), and took these
new DDIs as the independent testing set. The dataset collected
from the previous release of DrugBank (31 March 2021) was taken
as the training set (Section ‘Datasets’) to train our model. The
performance of SGRL-DDI is investigated in the tasks of the signed
and the directed interaction predictions, respectively.

The results in Table 5 show that SGRL-DDI is adequate for both
signed interaction prediction and directed interaction prediction
of newly added DDI entries.

To investigate the potential biases of identifying enhancive/depressive DDIs and affected/influencing drugs, we utilized confusion matrices to measure the prediction for newly added DDIs
[entries in two scenarios (Tables S3 and S4, see Supplementary](https://academic.oup.com/bib/article-lookup/doi/10.1093/bib/bbac602#supplementary-data)
Data available online). Specifically, we obtained the prediction
with Sensitivity (TPR) = 0.9023, Specificity (TNR) = 0.8699 in the
scenario of signed DDIs while obtaining the prediction with Sensitivity (TPR) = 0.8765 and Specificity (TNR) = 0.7453 in the scenario
of directed DDIs. Obviously, most DDI entries can be identified
in terms of pharmacological changes and influencing directions.
Thus, there is no significant bias.


Conclusions


The interactions between drugs are comprehensive because
they trigger enhancive/depressive and directed pharmacological
effects. Analogous to social associations between persons, a
set of DDI entries can be represented as a signed and directed
network, where the enhancive/depressive changes are labeled
as the signs of DDI edges, and pharmacological directions are
the directions of DDI edges. The underlying topology of the
signed and directed DDI network is often ignored in existing DDI
works, whereas the topological patterns hidden in the signed and
directed DDI network can help reveal the underlying mechanism
of DDIs. Thus, here we first leverage the Balance theory and
Status theory to uncover DDI interaction patterns, and then
design a novel multitask graph representation learning model
framework to predict the enhancive/depressive and asymmetric
pharmacological effects of DDIs (i.e. the signed and directed DDIs



prediction). SGRL-DDI can capture the task-joint information
by enhancing relation GNNs with Balance and Status patterns.
Moreover, SGRL-DDI utilizes task-specific DNNs to perform two
prediction tasks of sign and direction of DDIs. Experimental
results show that our SGRL-DDI is superior to other state-ofthe-art methods for DDI multitask prediction. Furthermore,
the ablation study verifies how each component in SGRL-DDI
contributes to the prediction performance. The prediction results
of enhancive, depressive and pharmacological effects of new DDIs
demonstrate the effectiveness of SGRL-DDI in real DDI prediction
scenario.


**Key Points**


  - We first leverage Balance theory and Status theory to
reveal the topological patterns among directed pharmacological DDIs, which are modeled as a signed and
directed network.

  - We design a novel graph representation learning model
named SGRL-DDI to realize the multitask prediction of
DDIs, including the prediction of enhancive/depressive
DDIs and the prediction of directed DDIs.

  - SGRL-DDI can capture the task-joint information by
integrating relation graph convolutional networks with
Balance and Status patterns.


Supplementary Data


[Supplementary data are available online at http://bib.oxfordjournals.](https://academic.oup.com/bib/article-lookup/doi/10.1093/bib/bbac602#supplementary-data)
[org/.](http://bib.oxfordjournals.org/)


Data availability


The datasets generated and analyzed during the current study
and the code of SGRL-DDI are openly available at the website of
[https://github.com/NWPU-903PR/ SGRL-DDI.](https://github.com/NWPU-903PR/)


Acknowledgments


We acknowledge anonymous reviewers for the valuable comments on the original manuscript.


Funding


This work has been supported by the National Natural Science
Foundation of China (grant numbers, 62173271, 61873202 and
61872297) and Shaanxi Provincial Key R&D Program, China (grant
number 2020KW-063).


Ethics approval and consent to participate


No ethics approval was required for the study.


10 | _Feng_ et al.


Consent for publication


Not applicable.


References


1. K. I. Cheng F, Barabási AL. Network-based prediction of drug
combinations. _Nat Commun_ 2019; **10** (1):1197.
2. S. R. Niu J, Mager DE. Pharmacodynamic drug-drug interactions.
_Clin Pharmacol Ther_ 2019; **105** (6):1395–406.
3. Sun M, Zhao S, Gilvary C, _et al._ Graph convolutional networks for
computational drug development and discovery. _Brief Bioinform_
2020; **21** (3):919–35.
4. Takeda T, Hao M, Cheng T, _et al._ Predicting drug-drug interactions through drug structural similarities and interaction networks incorporating pharmacokinetics and pharmacodynamics
knowledge. _J Chem_ 2017; **9** :16.
5. Zhang W, Chen Y, Liu F, _et al._ Predicting potential drug-drug
interactions by integrating chemical, biological, phenotypic and
network data. _BMC Bioinform_ 2017; **18** (1):18.
6. Andrej K, Polonca F, Brane LE, _et al._ Predicting potential drugdrug interactions on topological and semantic similarity features using statistical learning. _Plos One_ 2018; **13** (5):e0196865.
7. Yu H, Mao KT, Shi JY, _et al._ Predicting and understanding comprehensive drug-drug interactions via semi-nonnegative matrix
factorization. _BMC Syst Biol_ 2018; **12** (Suppl 1):14.
8. Wen Z, _et al._ SFLLN: a sparse feature learning ensemble method
with linear neighborhood regularization for predicting drug–
drug interactions. _J Inf Sci_ 2019; **497** :189–201.
9. Sridhar D, Fakhraei S. A probabilistic approach for collective
similarity-based drug-drug interaction prediction. _Bioinformatics_
2016; **32** (20):3175–82.
10. Gottlieb A, Stein GY, Oron Y, _et al._ INDI: a computational framework for inferring drug interactions and their associated recommendations. _Mol Syst Biol_ 2012; **8** (1):8–592.
11. Cheng F, Zhao Z. Machine learning-based prediction of drugdrug interactions by integrating drug phenotypic, therapeutic, chemical, and genomic properties. _J Am Med Inform Assoc_
2014; **21** (e2):e278–86.
12. Feng YH, Zhang SW, Shi JY. DPDDI: a deep predictor for drugdrug interactions. _BMC Bioinform_ 2020; **21** (419).
13. Zhang P, Wang F, Hu J, _et al._ Label propagation prediction of
drug-drug interactions based on clinical side effects. _Sci Rep_
2015; **5** (1):12339 2015/07/21.
14. Rohani N, Eslahchi C, Katanforoush A. ISCMF: integrated
similarity-constrained matrix factorization for drug–drug interaction prediction. _Netw Model Anal Health Inform Bioinform_
2020; **9** (1):1–8.
15. Ryu JY, Kim HU, Lee SY. Deep learning improves prediction of
drug-drug and drug-food interactions. _Proc Natl Acad Sci U S A_
2018; **115** (18):E4304–11.
16. P. C. Lee G, Ahn J. Novel deep learning model for more accurate prediction of drug-drug interaction effects. _BMC Bioinform_

2019; **20** :415.

17. Deng Y, Xu X, Qiu Y, _et al._ A multimodal deep learning framework for predicting drug-drug interaction events. _Bioinformatics_
2020; **36** (15):4316–22.
18. Nyamabo AK, Yu H, Shi JY. SSI-DDI: substructure-substructure
interactions for drug-drug interaction prediction. _Brief Bioinform_
2021; **22** (6):bbab133.
19. Zitnik M, Agrawal M, Leskovec J. Modeling polypharmacy
side effects with graph convolutional networks. _Bioinformatics_
2018; **34** (13):i457–66.



20. Yu Y, Huang K, Zhang C, _et al._ SumGNN: multi-typed drug interaction prediction via efficient knowledge graph summarization.
_Bioinformatics_ 2021; **37** (18):2988–95.
21. H. Wang, _et al._, GoGNN: graph of graphs neural network for predicting structured entity interactions, _Twenty-Ninth International_
_Joint Conference on Artificial Intelligence and Seventeenth Pacific Rim_
_International Conference on Artificial Intelligence_ IJCAI-PRICAI-20,
Yokohama Japan, 2020.
22. N. Xu, Wang, P., Chen, L., Tao, J., & Zhao, J. Mr-gnn: multiresolution and dual graph neural network for predicting structured entity interactions, _Proceedings of the 28th International_
_Joint Conference on Artificial Intelligence_, vol. IJCAI, Macau China,

2019.

23. Y. Wang, Min, Y., X Chen, & Wu, J. (2021). Multi-view graph
contrastive representation learning for drug-drug interaction
prediction, _WWW ’21: The Web Conference 2021_, Association for
Computing Machinery, Ljubljana, Slovenia, 2021.
24. Chen Y, _et al._ MUFFIN: multi-scale feature fusion for drug–drug
interaction prediction. _Bioinformatics_ 2021; **37** (17):2651–8.
25. Asada M, Miwa M, Sasaki Y. Using drug descriptions and molecular structures for drug–drug interaction extraction from literature. _Bioinformatics_ 2020; **37** (12):1739–46.
26. Shi J-Y, Mao K-T, Yu H, _et al._ Detecting drug communities
and predicting comprehensive drug–drug interactions via balance regularized semi-nonnegative matrix factorization. _J Chem_
2019; **11** (1):28.
27. Y. Chen, T. Qian, H. Liu, and K. Sun, Bridge: enhanced signed
directed network embedding, _CIKM 2018 - Proceedings of the 27th_
_ACM International Conference on Information and Knowledge Manage-_
_ment_, Association for Computing Machinery, 2018.
28. V. S. Jonker DM, Van der Graaf PH, Voskuyl RA, _et_ _al._
Towards a mechanism-based analysis of pharmacodynamic
drug-drug interactions in vivo. _Pharmacol Ther_ 2005; **106** (1):

1–18.

29. Wishart DS, _et al._ DrugBank 5.0: a major update to the DrugBank
database for 2018. _Nucleic Acids Res_ 2017; **46** (D1):D1074–82.
30. Rogers D HM. Extended-connectivity fingerprints. _J Chem Inf_
_Model_ 2010; **50** (5):742–54.
31. S. Kumar, F. Spezzano, V. Subrahmanian, and C. Faloutsos, Edge
weight prediction in weighted signed networks, _IEEE 16th Inter-_
_national Conference on Data Mining (ICDM)_, Spain, 2016.
32. Wen YM, Huang L, Wang CD, _et al._ Direction recovery in undirected social networks based on community structure and popularit. _Information Sciences_ 2018; **473** :31–43.
33. J. Kim, H. Park, J. E. Lee, and U. Kang, SIDE: representation learning in signed directed networks, _WWW ’21: The Web Conference_
_2021, Association for Computing Machinery_, Ljubljana, Slovenia,

2021.

34. T. Jie, T. Lou, and J. M. Kleinberg, Inferring social ties across
heterogenous networks, in _Proceedings of the Fifth International_
_Conference on Web Search and Web Data Mining, WSDM 2012_, Seattle, WA, USA, 8–12, 2012, **2012** .

35. Tang J, Chang Y, Aggarwal C, _et al._ A Survey of Signed Network
Mining in Social Media. _ACM Comput Surv_ 2015; **49** (3):37.
36. Q. V. Dang and C. L. Ignat, Link-sign prediction in dynamic signed
directed networks, in _2018 IEEE 4th International Conference on_
_Collaboration and Internet Computing (CIC)_, 2018.
37. Davis JAJSN. Clustering and structural balance in graphs. _Human_
_Relations_, **20** (2):181–187.
38. Schlichtkrull M, Kipf TN, Bloem P, _et al. Modeling Relational Data_
_with Graph Convolutional Networks_ . The Semantic Web, ESWC
2018. Lecture Notes in Computer Science, vol 10843. Springer,

Cham.


39. Velikovi P, Cucurull G, Casanova A, _et al._ Graph Attention Networks. _Conference Track 6th International Conference on Learning_
_Representations, ICLR_, Vancouver Canada, 2018.
40. Huang J, Shen H, Hou L, _et al._ SDGNN: Learning Node Representation for Signed Directed Networks. _Association for the_
_Advancement of Artificial Intelligence, AAAI_, on-line, 2021.
41. Yan XY, Zhang SW. Identifying drug-target interactions
with decision templates. _Curr_ _Protein_ _Pept_ _Sci_ 2018; **19** (5):

498–506.

42. Kipf TN, Welling M. Semi-supervised classification with graph

convolutional networks. _arXiv:160902907_ 2016.



_A social theory-enhanced graph representation_ | 11


43. Y. Li, Y. Tian, J. Zhang, and Y. Chang, Learning signed network
embedding via graph attention, _Proceedings of the AAAI Conference_
_on Artificial Intelligence_, vol. **34**, no. 4, pp. 4772-4779, 2020.
44. Derr T, Ma Y, Tang J. Signed Graph Convolutional Network.
_2018 IEEE International Conference on Data Mining, ICDM_, Singapore,

2018.

45. G. Salha, S. Limnios, R. Hennequin, V. A. Tran, and M. Vazirgiannis, Gravity-Inspired Graph Autoencoders for Directed Link
Prediction, in _the 28th ACM International Conference_, 2019.
46. Zhu S, Li J, Peng H, _et al._ Adversarial directed graph embedding

2020.


