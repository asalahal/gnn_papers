Applied Intelligence (2021) 51:8316–8333

https://doi.org/10.1007/s10489-021-02296-4

# **Modeling polypharmacy effects with heterogeneous signed graph** **convolutional networks**


**Taoran Liu** **[1]** **· Jiancong Cui** **[1]** **· Hui Zhuang** **[1]** **· Hong Wang** **[1,2]**


Accepted: 1 March 2021 / Published online: 1 April 2021
© The Author(s), under exclusive licence to Springer Science+Business Media, LLC, part of Springer Nature 2021


**Abstract**

Pharmaceutical drug combinations can effectively treat various medical conditions. However, some combinations can cause
serious adverse drug reactions (ADR). Therefore, predicting ADRs is an essential and challenging task. Some existing
studies rely on single-modal information, such as drug-drug interaction or drug-drug similarity, to predict ADRs. However,
those approaches ignore relationships among multi-source information. Other studies predict ADRs using integrated multimodal drug information; however, such studies generally describe these relations as heterogeneous unsigned networks
rather than signed ones. In fact, multi-modal relations of drugs can be classified as positive or negative. If these two
types of relations are depicted simultaneously, semantic correlation of drugs in the real world can be predicted effectively.
Therefore, in this study, we propose an innovative heterogeneous signed network model called SC-DDIS, to learn drug
representations. SC-DDIS integrates multi-modal features, such as drug-drug interactions, drug-protein interactions, drugchemical interactions, and other heterogeneous information, into drug embedding. Drug embedding means using feature
vectors to express drugs. Then, the SC-DDIS model is also used for ADR prediction tasks. First, we fuse heterogeneous drug
relations, positive/negative, to obtain a drug-drug interaction signed network (DDISN). Then, inspired by social network,
we extend structural balance theory and apply it to DDISN. Using extended structural balance theory, we constrain sign
propagation in DDISN. We learn final embedding of drugs by training a graph spectral convolutional neural network. Finally,
we train a decoding matrix to decode the drug embedding to predict ADRs. Experimental results demonstrate effectiveness
of the proposed model compared to several conventional multi-relational prediction approaches and the state-of-the-art deep
learning-based Decagon model.


**Keywords** Polypharmacy effects · Adverse drug reaction · DDI prediction · Graph convolutional neural network ·
Signed network · Structural balance theory



**1 Introduction**


Pharmaceutical drugs can treat and relieve symptoms and
prevent diseases; however, a single drug often shows
limited efficacy, poor safety, and developed drug resistance

[14]. Therefore, drug combination therapy has become a
very effective way to treat diseases [1, 9]. Many people,
particularly the elderly, take multiple drugs simultaneously.
However, drug combinations may cause serious adverse
drug reactions (ADR) due to drug-drug interactions (DDI).


� Hong Wang
[111052@sdnu.edu.cn](mailto: 111052@sdnu.edu.cn)


Extended author information available on the last page of the article.



For example, in the United States, 100000 people die of
ADRs each year [7]. At the same time, it is difficult to
identify the ADRs manually [21]. Therefore, predicting
previously unknown ADRs using heterogeneous drug
information is a significant and challenging problem.
Traditional ADR detection methods rely on a large
number of clinical trials. However, clinical trials are
expensive and time-consuming, and sometimes are not
possible due to the drug combinations’ complexity.
Currently, machine learning methods can be applied to
ADR detection. Specifically, such methods represent drugs
with their chemical structure and biological information and
then use machine learning approaches to predict ADRs.
Although these methods have been relatively successful [10,
20, 21, 24], many unsolved problems remain. First, such
methods only focus on a single type of direct relationship
between drugs and ignore the implicit, indirect contact


Modeling polypharmacy effects with heterogeneous signed graph convolutional networks 8317



between drugs, such as drug-chemical interactions, drugprotein interactions (DPI), and drug-disease interactions.
Therefore, the characterization of the potential real semantic
relationship between drugs is biased. Second, although
some studies consider the multi-source heterogeneous
relations between drugs [10, 21], the integration methods for
these multi-source heterogeneous relations do not take into
account the potential semantic information. This problem is
even more serious when there are both positive and negative
relations between drugs. Third, compared to the positive
effects, the negative relations, such as ADRs between drugs,
are relatively less frequent; in other words, the drug-drug
negative interaction network is very sparse, which has a
significantly adverse impact on ADR prediction.
To address these issues, inspired by the social network
analysis method [11, 19, 27], we define the multisource interactions between drugs using a heterogeneous
signed network that describes the direct relationships and
various indirect semantic relationships between drugs. In
addition, the heterogeneous signed network depicts the
positive and negative effects of drugs, which conforms
to the semantic information about real-world effects. A

signed network [27], also known as a polar network,
refers to a network where edges are weighted positively
and negatively according to their represented semantic
relations. Signed networks have been widely used in
various research areas, such as social, trust, and traffic

control networks [18, 26, 27] and such studies have
provided some promising results. However, few studies
have investigated ADR-oriented heterogeneous signed
networks to the best of our knowledge, and even fewer
have considered the spectral convolution on heterogeneous
signed networks. Therefore, we propose an innovative
method of spectral convolution on drug-drug interactions
signed network (SC-DDIS). Specifically, the SC-DDIS
method defines different types of semantic information
about drugs, such as DDIs, DPIs, and drug-chemical
interactions, with positive and negative drug interactions
and integrates them. The proposed SC-DDIS predicts drugdrug interaction relationships more accurately and also has
excellent interpretability. The primary contributions of the
proposed method are summarized as follows.


1) The proposed SC-DDIS is the first ADRs-oriented
spectral convolution model on drug-drug interaction
heterogeneous signed networks. It provides a more
comprehensive description of drug relations because it
simultaneously defines and depicts different types of
semantic information about drugs. The SC-DDIS considers the real-world semantic information, enriches
the information about drug characteristics, enhances



the interpretability of the model, and alleviates the
adverse effect of network sparsity on the model.
2) The proposed SC-DDIS extends the structure balance
theory of the signed network and applies it to learn
higher-order sign propagation features and further
restricts the sign propagation process in the DDISN.
These higher-order sign propagation features capture
latent semantic relations implied in the SC-DDIS
network, which improves the performance of the
proposed SC-DDIS model. The accessibility matrix

[28] can be formed by sign propagation. To vividly
express the matrix after sign propagation, we will
use the sign propagation matrix to refer to the
accessibility matrix. We will explain in detail what a
sign propagation matrix is in section 3.1.4.
3) We use a variety of methods to optimize the SCDDIS model. First, we use the proportion of positive
and negative samples to allocate different weights to
samples to handle the problem of data imbalance in the
SC-DDIS and further reduce the loss value. Second,
we train a decoding matrix in the SC-DDIS model
that can filter out noise, enlarge the critical features,
improve the prediction rate of drug side effects, and
further improve the model’s prediction performance
and robustness.

4) Extensive experiments were conducted to verify the
effectiveness of the proposed SC-DDIS model. The
experimental results demonstrated that the SC-DDIS
outperformed a state-of-the-art deep learning method

[21] and several other well-known multi-relational link
prediction approaches [22, 23, 25, 31].


**2 Related work**


Previous ADR prediction studies can be classified as traditional and deep learning-based methods. Conventionally,
pharmacological, topological, or semantic similarity based
on statistical learning is calculated to predict ADRs [4, 15].
Aurel proposed predictive pharmacy interaction networks
to predict unknown ADRs using the network structure
formed by known DDIs, and various drug classification
characteristics [2]. Huang found that integrating the proteinprotein interaction (PPI) networks and the drug structures
can improve ADR prediction performance [12]. Zhang proposed a matrix perturbation method based on DDI networks
combined with similar drug characteristics to predict ADRs

[29]. Park proposed the random walk with restart algorithm to simulate signal propagation on protein networks
to predict ADRs [24]. Zheng measured and established a
framework for drug similarity integration from various per

8318 T. Liu et al.



spectives and proposed a method to select highly reliable
negative samples to predict ADRs [30]. These methods do
not rely on deep learning methods with strong learning representation ability, which we call traditional methods for
predicting ADR.
With the development of deep learning, Kipf investigated
the convolution operation on graphs according to the
convolutional neural networks on images. He proposed
a graph convolutional neural network (GCN) algorithm
based on spectral convolution [17]. The algorithm uses the
Laplace feature spectrum to maintain the network structure
and the relationship between the Laplace matrix and the
Fourier transform, and performs a convolution operation
based on the Laplace feature spectrum. The GCN algorithm
has achieved excellent results in graph link prediction task
studies; therefore, many studies that employ a deep learning
method to predict the side effects of drugs primarily use
the graph convolution technique. Jure proposed a method to
predict ADR by using GCN for DDI, PPI and DPI end-toend learning [21]. In addition, many algorithms are based
on the GCN algorithm to learn the embedded representation
of the drug network, and finally predict ADRs [5, 29]. Liu
proposed a structural network embedding method based
on multi-mode deep automatic coding to predict ADRs

[20]. The previously described studies demonstrate that
the GCN has achieved excellent results in the embedded

representation of drugs. Therefore, this paper also uses the
GCN algorithm based on spectral convolution to obtain the
embedded expression of drugs. Differing from the described
related studies, we use a new method to integrate the drug
heterogeneous network to form the DDISN. Then, a higherorder sign propagation network is obtained by constraining
the sign propagation of DDISN with the extended structure
balance theory, which can increase the amount of available
information and compensate for the impact of some data
imbalances. On this basis, we use the GCN algorithm to
extract the information again to obtain the final drug feature
expression.
Although the methods mentioned above have achieved
some success, they ignore the rich implicit semantic
information between drugs. Therefore, the learned drug
node representation is inaccurate and difficult to interpret.
Thus, in this paper, a heterogeneous signed network
analysis method base on spectral convolution is proposed
to predict ADRs in consideration of the multi-source
semantic information between drugs, so as to improve
interpretability of the model and enhance DDI prediction
accuracy. In addition, we employ a variety of techniques
to improve the robustness and effectiveness of the model.
Specifically, we weight the loss value to solve the problem
of imbalance between positive and negative examples in
ADRs prediction. We also filter noise data using a training
decoding matrix.



**3 Methodology**


In this section, we introduce the model architecture and
related definitions, and use simple examples to explain the
definitions.


**3.1 SC-DDIS model**


**3.1.1 Background**


In this section, we will briefly introduce some concepts in
section 3.1.2 to understand the whole model’s architecture.

For more specific and detailed definitions of terms used
in this paper, we put them in section 3.1.4 for a detailed
explanation. DDI networks describe the side effects of drugs
and drugs taken together. Biological drug profiles refer
to drug subchemical, drug-protein interactions and other
information. M-order sign propagation matrix refers to the
matrix after the drug sign network has propagated m times,
expressed by the formula as _Matrix_ _s_ _[m]_ [.] _[ Matrix]_ _[s]_ [ is a matrix]
representation of sign networks. GCN is a deep learning
method applied to graph, which can effectively learn the
representation of nodes in graph.


**3.1.2 Model architecture**


As shown in Fig. 1, the proposed model is divided into
five sub-modules. For convenience, we assume that there
are _N_ drugs, and _M_ side effects. In Part A, different side
effects are extracted to construct the DDI networks. That is

to say, the number of DDI networks is _M_ . In Part B, we
fuse DDI networks and biological drug profiles to obtain
the _M_ signed networks according to the different semantic
relations. In Part C, we constrain the sign propagation
according to the enhanced structure balance theory and
obtain an _m_ -order sign propagation matrix as the drug
feature representation. In Part D, the initial DDI networks
and _m_ -order sign propagation matrix are used as input of
our SC-DDIS model. Note that the input dimensions of
a particular side effect prediction task are _N_ × _N_ and
_N_ × _N_ is related to _a_ and _b_ in Fig. 1, respectively. There
are _M_ side effects; thus, _M_ times inputs are required to
calculate each side effect prediction task’s score. Then,
the embedding representation of drugs is obtained and fed
into the GCN spectral convolution component. In Part E,
the embedded description of drugs is input into a fullyconnected neural network to obtain the reconstructed DDI

networks of different side effects to predict ADRs.


**3.1.3 Graph convolution network (GCN)**


Here, we briefly introduce GCN. Thomas Kpif proposed
GCN in the paper Semi-supervised classification with


Modeling polypharmacy effects with heterogeneous signed graph convolutional networks 8319


**Fig. 1** Proposed side effect
prediction model. The SC-DDIS
model proposed in this paper
consists of five parts: A-E



graph convolutional networks in 2017 [17]. It provides
a new idea for graph structure data processing and
applies the convolutional neural network commonly used
for images in deep learning to graph data. The core of
the GCN is based on the characteristic decomposition
of the Laplace matrix, which is a semi-positive definite
symmetric matrix with many good properties, e.g., a matrix
comprising eigenvectors is orthogonal and the eigenvalue is
nonnegative. The standard Laplace matrix [6] of the network



is given in (1).


_L_ = _D_ − _A_ (1)


Here, _D_ is the degree matrix (diagonal matrix) of the
vertex, the elements on the diagonal are the degrees of each
vertex in turn, and _A_ is the adjacency matrix of the graph.
The Laplacian matrix is a positive semi-definite symmetric
matrix, which can be spectrally decomposed, and has a


8320 T. Liu et al.



particular form after decomposition as follows.


_L_ = _UΛU_ [−][1] (2)


Here, _U_ = _(_ **u** **1** _,_ **u** **2** _, . . .,_ **u** **n** _)_ is the eigenvector matrix,
and **u** **i** is the column vector. _Λ_ = _diag (λ_ 1 _, λ_ 2 _, . . ., λ_ _n_ _)_
is a diagonal matrix of n eigenvalues. According to the
properties of the Laplace matrix, we know that _U_ is an
orthogonal matrix; thus, _UU_ _[T]_ = _IUU_ _[T]_ = _E_, i.e., _U_ [−][1] =
_U_ _[T]_ . Taking _U_ as the basis of Fourier transform, the rules of
Fourier forward transform and inverse transform are defined

as follows.


_F_ ˆ = _U_ _[T]_ _F_ (3)


_F_ = _U_ _F_ [ˆ] (4)
Here, _F_ _i_ is the vector representation of Drug _i_ nodes in
the DDIN. According to the positive and inverse Fourier
transform, we change the DDIN from the spatial domain
to the spectral domain via positive Fourier transform
( _U_ _[T]_ _F_ ). Similarly, we change the convolution kernel to the
spectral domain using positive Fourier transform ( _U_ _[T]_ _X_ ).
The inverse transformation of the product of the two Fourier
transforms is obtained by multiplying the product of the two
Fourier transforms by _U_ . The final convolution is expressed
as follows.


_(F_ ∗ _X)_ _G_ = _U((U_ _[T]_ _X)_ ⊙ _(U_ _[T]_ _F))_ (5)


Here, _X_ is the convolution kernel matrix. Equation (5) is
a general expression of graph convolution. This paper will
use a classical graph convolution neural network structure
proposed by kipf [17], such as (6).



_H_ _[(l]_ [+][1] _[)]_ = _σ(D_ [˜] [−] [1] 2




[1] ˜

2 _A_ ˜ _D_ [−] 2 [1]



2 _H_ _[(l)]_ _W_ _[(l)]_ _)_ (6)



DDNE, or ADR, is some undesirable effect caused by the
use of a drug. Note that most are natural pharmacological
actions. DDNE relation between drugs can be described by
a matrix denoted _A_ _NE_, where _A_ _NE_ _(i, j)_, i.e., an element
of _A_ _NE_, is 0 when the side effects between Drug _i_ and
Drug _j_ are unknown. If side effects occur between Drug _i_
and Drug _j_, then _A_ _NE_ _(i, j)_ = 1. At the same time, because
DDNE is mutual, _A_ _NE_ _(j, i)_ = 1. In other words, _A_ _NE_
should be a symmetric matrix.


2) Drug-drug positive effect (DDPE)


Let _S_ _ij_ denote a normalized similarity between Drug _i_ and
Drug _j_, _μ_ ∈ [0 _,_ 1 _)_ is the threshold value. If _S_ _ij_ _> μ_, there
is a positive effect between Drug _i_ and Drug _j_ . If _S_ _ij_ ≤ _μ_,
the positive effect between Drug _i_ and Drug _j_ is unknown.
That is to say, DDPE between two drugs depends on their
similarity on biological drug profile. The DDPE is that it is
not easy to cause adverse reactions between drugs.
The similarity may be different according to their
different relations. At the same time, Jaccard [13] similarity
is used to evaluate the similarity in two drugs in a specific
relationship. For example, describing the DPIs between
Drug _i_ and Drug _j_ is based on the Jaccard similarity of a
target protein of a drug. In addition, describing the drugchemical structure interactions between Drug _i_ and Drug _j_ is
based on the Jaccard similarity of the chemical structure of
the drug. Finally, the similarity of the two drugs is obtained
by weighting, according to (7).


| _Γ_ _rel_ _(i)_ ∩ _Γ_ _rel_ _(j_ _)_ |

_S(i, j)_ = � _rel_ _[α]_ _[rel]_ | _Γ_ _rel_ _(i)_ ∪ _Γ_ _rel_ _(j)_ | (7)


Here, _rel_ ∈{protein _,_ chemical _, . . ._ } is the set of drug
relationships, where [�] _rel_ _[α]_ _[rel]_ [ =][ 1.] _[ Γ]_ _[rel]_ _[ (i)]_ [ is the set of]
rel contacts for Drug _i_, _Γ_ _rel_ _(j)_ is the set of rel contacts for
Drug _j_, and _S (i, j)_ is the total similarity between Drug _i_ and
Drug _j_ .
We take the DPIs as an example to illustrate the positive
effects of drugs. Here, we set parameters _rel_ ∈{protein}
and _α_ _protein_ = 1, as shown in (8).


_S(i, j)_ = [|] _[Γ]_ _[p][rotein]_ _[(i)]_ [ ∩] _[Γ]_ _[p][rotein]_ _[(j)]_ [|] (8)

| _Γ_ _protein_ _(i)_ ∪ _Γ_ _protein_ _(j)_ |


Here, _Γ_ _protein_ _(i)_ represents the target protein collection
of Drug _i_,and _Γ_ _protein_ _(j)_ represents the target protein
collection of Drug _j_ .
As shown in Fig. 2, if the target protein similarity of a
drug pair is very high, the probability of side effects between
that pair is very low (and vice versa). Therefore, when the
similarity of the two drugs’ target proteins is very high,
the probability of side effects is negligible. Here, if the
similarity of the two drug’s target proteins exceeds the value
of _μ_, there is a positive effect between the drugs, i.e., not
easy to have side effects. In the experimental section of this



Here, _A_ [˜] = _A_ + _I_ _N_ is the adjacency matrix of an undirected
_D_ graph with self-connections.˜ _ii_ = [�] _j_ _A_ ˜ _ij_ and _W_ _[(l)]_ is the trainable weight matrix of _I_ _N_ is the identity matrix.
the _l_ _[th]_ layer. _σ(_ - _)_ denotes an activation funtion, such as
the ReLU(·) = max(0, ·). _H_ _[l]_ ∈ R _[N]_ [×] _[D]_ is the expression
matrix activated at the _l_ _[th]_ layer. In a graph, _N_ can represent
the number of nodes in the graph, and _D_ means to use
d-dimensional data to represent a node in the graph.


**3.1.4 Definitions**


Here, we define the drug-drug negative effect, the drug-drug
positive effect, and the drug-drug interaction network. We
then describethe DDI signed network according to semantic
information about real-world effects. Then, we explain the
extended structure balance theory and sign propagation
matrix, which can be applied to DDISN. Finally, the
convolution operation on the drug-drug interaction network
of side effects is defined.


1) Drug-drug negative effect (DDNE)


Modeling polypharmacy effects with heterogeneous signed graph convolutional networks 8321


**Fig. 2** Relationship between
target protein-based DDPE and
DDNE. It shows the relation

between the target protein
similarity of any drug pair and
its ADR probability. When the
target protein similarity is 0, the
probability of side effect of any
drug pair is between 66-72%.
When the target protein
similarity increases to 60%, the
probability of side effect
decreases to 20-23%. As the

target protein similarity
continues to rise to 100%, the
probability of side effect of any
drug pair is lower than 11%. To
prove the validity of this
relationship, we conduct four
group of experiments, all drug
combinations, randomly
selected drug combinations,
drug combinations that cause
diarrhea, and drug combinations
that cause nausea



article, we validate the effectiveness of defining the positive
effects of drugs by the similarity of target proteins. Here,
_μ_ is the hyperparameter of the proposed SC-DDIS model,
which is obtained experimentally.


3) Drug-drug interactions network (DDIN)


The DDIN is formalized as _G_ _DDIN_ = _(V, E, A_ _DDIN_ _)_ .
Here, _V_ is the set of nodes (nodes represent drugs), _E_ is
the set of relationships between nodes, and _A_ _DDIN_ is the
adjacency matrix of network _G_ _DDIN_ . In addition, _e (i, j)_ ∈
{0 _,_ 1} represents the interaction between nodes _i_ ∈ _V_ and
_j_ ∈ _V_ . If there is a DDNE between Drug _i_ and Drug _j_, then
_e (i, j)_ = 1. If the DDNE between Drug _i_ and Drug _j_ is
unknown, then _e (i, j)_ = 0. Without loss of generality, we
assume _e (i, j)_ = _e (j, i)_ ; therefore, element _A_ _DDIN_ _(i, j)_
of the adjacency matrix _A_ _DDIN_ of network _G_ _DDIN_ can be
expressed as follows.


1 _,_ there is DDNE between _i_ and _j_
_A_ _DDIN_ _(i, j)_ =
� 0 _,_ DDNE between _i_ and _j_ is unknown

(9)


4) DDI signed network(DDISN)


The DDISN is formalized as _G_ _DDISN_ = _(V, E, A_ _DDISN_ _,_
_A_ _DDIN_ _, S, μ)_, where _V_ is a collection of nodes (nodes
represent drugs), _E_ is the set of relationships between nodes,
and _A_ _DDISN_ is the adjacency matrix of network _G_ _DDISN_ . In
addition, _A_ _DDIN_ is the adjacency matrix of the DDIN, _S_ is
the drug similarity matrix, _μ_ ∈ [0 _,_ 1 _)_ is the drug similarity
threshold, and _e (i, j)_ ∈ _E_ represents the interactions
between Drug _i_ ∈ _V_ and Drug _j_ ∈ _V_ . Here, if _S_ _ij_ _> μ_ and
_A_ _DDIN_ _(i, j)_ = 0, there is a positive effect between Drug _i_



and Drug _j_, and _S_ _ij_ ≤ _μ_ and _A_ _DDIN_ _(i, j)_ = 0 suggests
that the effect between Drug _i_ and Drug _j_ is unknown. If
_A_ _DDIN_ _(i, j)_ = 1, there are side effects between Drug _i_ and
Drug _j_ . Without loss of generality, we assume _e (i, j)_ =
_e (j, i)_ such that element _A_ _DDISN(i,j)_ of adjacency matrix
_A_ _DDISN_ of network _G_ _DDISN_ can be expressed as follows.



5) Extended structural balance theory


The structural balance theory was put forward by Heider
in the 1940s [3]. Some basic rules defined in this theory have
been widely used in link signed prediction tasks in signed
networks, which has become the basic theory of signed
networks. The theory points out that “friends of friends are
more likely to be my friends” and “enemies of friends are
more likely to be my enemies” are consistent with social
psychology. Based on Heider’s structural balance theory, we
define the extended structural balance theory as follows. If
Drug _A_ and Drug _B_ have a positive effect and Drug _B_ and
Drug _C_ have a positive effect, then Drug _A_ and Drug _C_ have
a positive effect. In addition, if Drug _A_ and Drug _B_ have
side effects and Drug _B_ and Drug _C_ have positive effects,
then Drug _A_ and Drug _C_ have side effects. In other words, if
Drug _A_ and Drug _B_ are very similar, and Drug _B_ and Drug _C_
are also very similar, then Drug _A_ and Drug _C_ are likely very
similar, i.e., there is likely no side effect between Drug _A_
and Drug _C_ . In addition, if Drug _A_ and Drug _B_ have a side
effect, and Drug _B_ and Drug _C_ are very similar, then Drug _A_



1 _,_ there is DDPE between _i_ and _j_
0 _,_ DDPE between _i_ and _j_ is unknown
−1 _,_ there is DDNE between _i_ and _j_

(10)



_A_ _DDISN_ _(i, j)_ =



⎧
⎨

⎩


8322 T. Liu et al.



and Drug _C_ are likely to have the same side effect. As shown
in the drug balance triangle in Fig. 3, a third virtual edge can
be predicted given two real edges.


6) Accessibility Matrix


Accessibility matrix refers to the degree that can be
achieved after a certain length of the path between a directed
graph’s nodes. In the scenario described in this article, let’s
give a new name to the Accessibility Matrix, called Sign
Propagation Matrix (SPM). The SPM is a matrix obtained
from _A_ _DDISN_ using a propagation operation according to the
extended structural balance theory. The SPM is initialized
first.


_SPM_ 0 = _I_ (11a)


_SPM_ 1 = _A_ _DDISN_ (11b)


Here, _I_ ∈ R _[N]_ [×] _[N]_ is an identity matrix. The propagation
mode of the SPM is defined as follows.


_SPM_ _m_ = Sign _(SPM_ _m_ −1 × _SPM_ 1 _)_



According to the extended structure balance theory, we can
perform signed propagation in the DDISN and obtain highorder drug features. For example, the drug balance triangle
in Fig. 3 can predict the third virtual edge when two real
sides are known. With the above theory, the signs in the
DDISN are propagated to obtain SPM _m_ . Through SPM _m_,
we can mine potential DDIs to obtain the higher order
of DDI information, which plays an essential role in the
convolution neural network.


7) Convolution on the DDIN of side effect _r_


We will apply the classic graph convolutional neural
network structure to the prediction of drug side effects. Then
the formula can be expressed by some concepts defined in
this article.



˜ − [1] 2
_H_ _[(l)]_ ← _σ_ _D_ _r_
�




[1] 2 _A_ ˜ _r_ ˜ _D_ _r_ − [1] 2



2 _H_ _(l_ −1 _)_ _W_ _(lr_ −1 _)_



(15)
�



_n_ −1
�
� _k_ =0



�



_with m_ ≥ 2


(12)



= Sign




[ _SPM_ _m_ −1 _(i, k)_ × _SPM_ 1 _(k, j)_ ]

_k_ =0



˜
Here, _A_ _r_ = _A_ _rDDIN_ + _I_ _r_ is the adjacency matrix of
the DDIN, _G_ _rDDIN_ = _(V, E, A_ _rDDIN_ _)_ with added self_D_ connections, and˜ _r_ _(i, i)_ = [�] _j_ _IA_ _r_ is the N-order identity matrix. Here,˜ _r_ _(i, j)_ . _W_ _r_ _[(l)]_ is the trainable weight
matrix of the _l_ _[th]_ layer. _σ(_ - _)_ denotes an activation funtion.
_H_ _[l]_ ∈ R _[N]_ [×] _[D]_ is the drug expression matrix activated at the
_l_ _[th]_ layer. In a DDISN, _N_ can represent the number of drug
nodes in the graph, and _D_ means to use d-dimensional data
to represent a drug node in the graph.


**3.1.5 Simple application**


Here, we use an actual drug heterogeneity network to
apply the above theory to facilitate a comprehensive
understanding of the theory. The drug heterogeneity
network is shown in Fig. 4.
Figure 4 helps us understand the above theory. First,
we only focus on the solid red lines, from which we can
extract the known DDIN, which can be represented by the
following adjacency matrix.



Sign( _x_ ) in (13) is defined as a sign function. Based on
the extended structure balance theory, the propagation of
the sign in the DDISN is constrained. The sign propagation
method is defined by formal language to obtain _m_ -order
sign propagation matrix (SPM _m_ ) in the DDISN.



Sign _(x)_ =



⎧ 1 _,_ if _x >_ 0

0 _,_ if _x_ = 0 (13)

⎨

⎩ −1 _,_ if _x <_ 0



The element of SPM _m_ is defined as follows.



_SPM_ _m_ _(i, j)_ =



⎧ 1 _,_ positive effect between _i_ and _j_

0 _,_ unknown effect between _i_ and _j_

⎨

⎩ −1 _,_ side effect between _i_ and _j_


(14)



⎞

⎟⎟⎠ (16)



_A_ _DDIN_ =



⎛

⎜⎜⎝



0 0 0 0

0 0 1 0

0 1 0 0

0 0 0 0



**Fig. 3** Drug balance triangle based on extended structure balance
theory. Drug _A_, Drug _B_, and Drug _C_ represent drugs. + indicates a
positive effect between drugs, and − indicates a side effect between
drugs. The solid line indicates the known relationship, and the dotted
line indicates the relationship predicted according to the solid line



Here, _μ_ = 0.5 _, rel_ = {protein}, and _α_ _protein_ = 1. It is
easy to calculate _S_ _AB_ = 23 _[> μ]_ [ and] _[ S]_ _[CD]_ [ =] _[ S]_ _[AA]_ [ =]
_S_ _BB_ = _S_ _CC_ = _S_ _DD_ = 1 _> μ_ by observing the drugprotein network in Fig. 4. In other words, there is a positive
effect between Drug _A_ and Drug _B_, and Drug _C_ and Drug _D_ .
Similarly, the green dotted lines in Fig. 5 indicate a positive
effect. In addition, the similarity between each drug and
itself is 1, which has a positive effect. Note that this is
an obvious conclusion; thus, it is not identified in Fig. 5
According to the known DDIN, the adjacency matrix is used


Modeling polypharmacy effects with heterogeneous signed graph convolutional networks 8323


The SPM is initialized according to the above definition, as
shown in (18).



⎞

⎟⎟⎠ (18)



_SPM_ 1 = _A_ _DDISN_ =



⎛

⎜⎜⎝



1 1 0 0

1 1 −1 0

0 −1 1 1

0 0 1 1



Using the signed propagation, the second-order SPM is
obtained as follows.


_SPM_ 2 = Sign _(SPM_ 1 × _SPM_ 1 _)_



⎞ (19)

⎟⎟⎠



=



⎛

⎜⎜⎝



1 1 − **1** 0

1 1 −1 − **1**

− **1** −1 1 1

0 − **1** 1 1



**Fig. 4** Drug initial heterogeneity network. The figure is a heterogeneous DDI network with one kind of side effect


to represent the DDISN as follows.



Through the second-order SPM, we can easily find that
two more pairs of side effects than the initial DDISN, i.e.,
Drug _A_ and Drug _C_, Drug _B_ and Drug _D_ in Fig. 5 (red dotted
lines). We find many potential reasonable relationships in
SPM 2 . Then, SPM 2 can be used as the initial expression
of drug characteristics, which achieved good results in our
experiment.


**3.2 Process and algorithm**


Here, we will elaborate on the process and algorithm of the
proposed SC-DDIS.


**3.2.1 Process**


The process is described as follows.


**Step 1.** A DDIN with specific side effect _r_ is extracted
from the heterogeneous drug network, and
adjacency matrix _A_ _rDDIN_ ∈ R _[N]_ [×] _[N]_ is used to
represent the DDIN, _G_ _rDDIN_ = _(V, E, A_ _rDDIN_ _)_ .
**Step 2.** Drug similarity threshold hyperparameter _μ_ is
set, and the DDISN is constructed based on
semantic information. Here, adjacency matrix
_A_ _rDDISN_ ∈ R _[N]_ [×] _[N]_ is used to represent the DDISN
_G_ _rDDISN_ = _(V, E, A_ _rDDISN_ _, A_ _rDDIN_ _, S, μ)_ .
**Step 3.** The extended structural balance theory is used as
a constraint condition for sign propagation, and
the sign in the DDISN is propagated to obtain
_SP M_ _m_ as a higher-order expression of the drug
feature.

**Step 4.** _A_ _rDDIN_ is used as the initial adjacency matrix,
and _SP M_ _m_ is taken as the initial vector representation of the drug. Here, we use Xavier [8] to
initialize the convolution kernel. _Z_ _r_ of the drug
feature is extracted using a graph convolution
neural network.

**Step 5.** A decoding matrix of the drug side effect _r_ is
trained, and the decoding matrix _X_ _r_ ∈ R _[N]_ [×] _[N]_ is



⎞

⎟⎟⎠ (17)



_A_ _DDISN_ =



⎛

⎜⎜⎝



1 1 0 0

1 1 −1 0

0 −1 1 1

0 0 1 1



**Fig. 5** Drug final DDI network. The solid line represents the known
DDIN, and the dotted line represents the predicted DDIN based on the
above definition


8324 T. Liu et al.


used to predict the side effect _r_ between drugs.
The decoding process is expressed as follows.


_A_ ˆ _r_ = _σ(Z_ _r_ _X_ _r_ _Z_ _r_ _[T]_ _[)]_ [ with] _[ Z]_ _[r]_ [ =][ SC-DDIS] _[(SPM]_ _[m]_ _[, A]_ _[rDDIN]_ _[)]_


(20)


Here, _A_ [ˆ] _r_ is the reconstructed adjacency matrix.
ˆ _ij_
_A_ _r_ is the probability of side effects between
Drug _i_ and Drug _j_, and _σ_ is the sigmoid activation
1
function, i.e., _σ (x)_ = 1+ _e_ ~~[−]~~ ~~_[x]_~~ [. By training the]
decoding matrix of side effect _r_, the noise data
can be filtered out, and the vital drug feature
information can be retained, thereby enhancing
the effectiveness and robustness of the model.

**Step 6.** The loss function value is calculated. Note that

the number of side effects due to DDIs is less

than the total number of drug combinations. In
other words, the dataset of drug combinations that
produce side effects is imbalanced, i.e., there are
fewer positive samples. In other words, there are
fewer edges with ADR. To solve this problem,
we use a weighted cross-entropy loss function
to calculate the loss value. The loss of positive
samples is increased according to the proportion
of positive and negative examples. As a result, the
overall cost is adjusted to minimize the impact
of the dataset’s imbalance on the proposed SC
DDIS.


ˆ
_Loss_ _r_ = � − _βA_ _[ij]_ _rDDIN_ _[log]_ � _A_ _[ij]_ _r_ � − �1 − _A_ _[ij]_ _rDDIN_ � _log_ �1 − _A_ [ˆ] _[ij]_ _r_ �


(21)



Here, _β_ = | _Sa_ − | _/_ | _Sa_ + | is the ratio of negative samples
to positive samples. | _Sa_ − | is the number of negative
samples, i.e., the number of edges with unknown effects in
the DDIN. | _Sa_ + | is the number of positive samples, i.e., the
number of edges known to generate DDNE in the DDIN.
Finally, the Adam algorithm [16] is used to optimize and
complete the model’s training task. The pseudocode for the
proposed SC-DDIS’s algorithm is presented in Algorithm 1.



Here, lines 1 to 17 construct the DDISN based on semantic information by fusing a multi-source heterogeneous
network of drugs and setting drug similarity threshold _μ_ .
Lines 18 to 22 perform signed propagation on the DDISN


Modeling polypharmacy effects with heterogeneous signed graph convolutional networks 8325



based on the extended structural balance theory to obtain
the _SPM_ _m_ . Lines 23 to 26 initialize the SC-DDIS model
parameters, and lines 27 to 35 perform _Ep_ rounds of iterative training on the DDIN using the proposed SC-DDIS
model. Lines 28 to 31 obtain the low-dimensional vector

representation of the drug, and line 32 passes the decoding
matrix to decode the drug code to obtain a reconstructed
DDIN. Finally, lines 33 to 34 calculate the loss value of the
proposed SC-DDIS model and update the parameter matrix.


**4 Experiment**


**4.1 Datasets**


In this study, we used the dataset in the Decagon model
[(http://snap.stanford.edu/decagon). This benchmark dataset](http://snap.stanford.edu/decagon)
includes 645 drug nodes, 7,795 protein nodes, 4,576,785
DDIs, and 18,690 DPIs. The dimension of DDIs is 645 ×
645 × 964 (964 ADR events), and the dimension of DPIs
is 645 × 7 _,_ 795 (18,690 interactions). If ADR occurs, the
corresponding element in the DDIs dataset is marked as 1.
Here, we focused on 964 common side effects, and each side
effect occurs in at least 500 drug combinations; among these
side effects, such as anaemia, diversity breathing, nausea
and so on. Drug data statistics are shown in Table 1.
To test the generalization of the model, we introduce
a new data set. The dataset can be obtained from the

[https://github.com/ltrbless/Dataset/tree/master website. In](https://github.com/ltrbless/Dataset/tree/master)
this dataset, 548 drug nodes and 97168 DDIs were obtained
from TWOSIDES, and 881 chemical substructure types
were obtained from PubChem. The type of side effect is not
specified in this data set. If there are side effects between
two drugs, the corresponding element in the DDIs dataset is
marked as 1.


**4.2 Experimental settings**


We used TensorFlow 1.13.1 to implement the proposed
SC-DDIS, and we used the Adam adaptive learning rate
optimizer to train the model. Note that effective parameter
values were determined experimentally. We then set the
drug similarity threshold _μ_ = 0.8, parameter _rel_ ∈


**Table 1** Drug data statistics



{protein}, and the proportion of target protein similarity
weight _α_ _protein_ = 1. In addition, the learning rate was set to
0.01, we used a second-order SPM, and iteratively trained
200 times for each side effect. Here, we used a two-layer
deep neural network with 32 and 16 dimensions, and the
tanh function was used as the activation function between

the first hidden layer and second hidden layer to extract the
features of the SPM.

For the second new dataset, we will define the positive
effect in terms of the similarity of chemical substructures.
We set the drug similarity threshold _μ_ = 0.8, parameter
_rel_ ∈{chemical}, and the proportion of target protein
similarity weight _α_ _chemical_ = 1. Other settings are the same
as above.


**4.3 Baseline**


We also used the latest Decagon model based on deep
learning and other classic multi-relational link prediction
approaches to evaluate and compare the proposed SC-DDIS.


1) Decagon model [21]: This graph convolution neural
network is developed to predict multi-relation links
in heterogeneous networks. In this model, end-to-end
learning of DDIs, DPI, and protein-protein interactions
are performed via graph convolution to obtain a drug
feature expression. Then, ADRs are predicted.
2) RESCAL [22]: This is a relational learning method
based on the factorization of a three-way tensor, which
obtains predictions by decomposing the matrix of drug
side effect _r_ .

3) DEDICOM [23]: This technique uses tensor decomposition to provide useful potential information from
DDIs for the prediction of ADRs. It is decomposed into
_X_ _r_ = _AU_ _r_ _T U_ _r_ _A_ _[T]_ by the DDIN _X_ of known drug side
effects r. The probability of side effects from Drug _i_ and
Drug _j_ is calculated as _p_ _ij_ = _a_ _i_ _U_ _r_ _T U_ _r_ _a_ _j_ .
4) DeepWalk [25, 31]: This technique is based on a
biased random walk, and a low-dimensional feature
representation of drug nodes is obtained by learning
the neighborhood nodes of the drug-drug interaction
networks. Then, the representations of each drug pair
are spliced to represent the characteristics of each pair



Data type Data Dimension


Node Drug node 645

Node Protein node 7795

Interaction DDIs 645 × 645 (964 ADR events and 4,576,785 interactions)

Interaction Target protein 645 × 7795 (18,690 interactions)


8326 T. Liu et al.



of drugs. Finally, an independent logistic regression
classifier is trained for each side effect to predict the

ADRs.

5) Concatenated drug features [21]: In this method,
principal component analysis is used to reduce the
dimension of DPI network, which is used as the
expression of the initial drug characteristics. Then,
the features of each pair of drugs are spliced as the
characteristics of each pair of drugs. Finally, a gradient
boosting tree classifier is employed to predict the
ADRs of each drug pair.


**4.4 Evaluation**


We used the 10-fold cross-validation method to compare
these algorithms. For a particular side effect, we randomly
took 10% of the drug pairs with and without side effects
as the test set and removed the known drug interaction
relationship from the dataset. The other 90% of drug pairs
was used as a training set to train the model. The final
evaluation score for a specific side effect was taken as
the average of all results repeated 10 times under different
random partitions of the dataset. Finally, the average rating
of 964 side effects was used as the final evaluation score.


**4.5 Metrics**


We use three commonly used metrics to evaluate the
performance of the model: AUROC (area under the receiver
operating characteristic curve), AUPRC (area under the
precision-recall curve), and AP@K (average precision at
K). The three evaluation criteria and related concepts are
defined as follows.

Firstly, for a binary classification problem, the result of
classification is positive (P) or negative (N). There are four
scenarios in the forecast.


1) True Positive (TP): the prediction value is P, and the
actual value is also P.

2) False Positive (FP): the prediction value is P, but the
actual value is N.

3) True Negative (TN): the prediction value is N, and the
actual value is also N.

4) False Negative (FN): the prediction value is N, but the
actual value is P.


True positive rate (TPR) is the probability of positive
samples among all positive samples, that is, model’s
sensitivity to positive samples. False positive rate (FPR) is
the probability of positive samples in all negative samples,
that is, model’s sensitivity to negative samples. TPR and



FPR are defined as follows.


_T P_
_T P R_ = (22)
_T P_ + _FN_


_FP_
_FP R_ = (23)
_FP_ + _T N_

Taking FPR as the x-axis and TPR as the y-axis,
the Receiver Operating Characteristic Curve (ROC) can
be obtained by setting different classification thresholds.
AUROC is the area under the ROC curve. The larger the
value of AUROC, the better the discrimination performance
of the model. When the distribution of positive and negative
samples in the test set changes, the ROC curve can keep
stable, making the value of AUROC stable, so the AUROC
index is robust.

Secondly, The concept of Precision is the proportion
of correctly classified samples to the total samples. The
definition of Recall is the same as that of TPR as the

probability of positive samples. The definition of Precision
and Recall is shown as follows.


_T P_
_P recision_ = (24)
_T P_ + _FP_


_T P_
_Recall_ = (25)
_T P_ + _FN_

With Recall as the x-axis and precision as the y-axis, the
precision-recall curve (PRC) can be obtained by setting
different classification thresholds. AUPRC is the area under

PRC. It is noted that the AUPRC value can reflect the

quality of the classifier more effectively than that of the
AUROC, as the AUPRC can reflect the actual performance
of the classification when the proportion of positive and
negative samples is quite different.
Thirdly, the definition of average precision at K (AP @
K) is shown as follows.


_K_
_AP_ @ _K_ = � _i_ =1 _[P recision(i)]_ (26)

_min(L, K)_


_P recision(i)_ is the precision before position i in the test
set’s sorted prediction result. _L_ is the total number of drug
combinations with ADRs in the test set.


**5 Results**


**5.1 Prediction results**


As discussed previously, proposed SC-DDIS model is
compared to the latest Decagon model and other classic
multi-relational link prediction methods. The results are
given in Table 2. Experimental results show that SCDDIS performs better than other prediction methods under


Modeling polypharmacy effects with heterogeneous signed graph convolutional networks 8327


**Table 2** Average AUROC, AUPRC, and AP@50 in the prediction of 964 side effects


Approach AUROC AUPRC AP@50


Concatenated drug features 0.793 0.764 0.712

DeepWalk neural embeddings 0.761 0.737 0.658

DEDICOM tensor factorization 0.705 0.637 0.567

RESCAL tensor factorization 0.639 0.613 0.476

Decagon 0.872 0.832 0.803

SC-DDIS 0.947 0.930 0.895



the three scoring methods and confirm the SC-DDIS
correctness and effectiveness.

We further compare the performance of our SC-DDIS
model with that of the Decagon model in detail, as the
Decagon model performs the best in our baseline models.
Concretely, we compare their best and worst performance in
predicting 10 sorts of representative ADRs, respectively. As
there is a massive proportion gap between the positive and
negative samples in the dataset, the AUPRC value is used to
evaluate the prediction results, as shown in Table 3.
Tables 2 and 3 show that the proposed SC-DDIS
outperformed the compared models in all aspects, which
verifies its effectiveness. For the second new dataset, the
results are given in Table 4. Experimental results show that
SC-DDIS performs better than other prediction methods
and confirms the SC-DDIS correctness and effectiveness.

To better reflect the difference in AP score, we set _K_ to
2000. Due to the lack of protein-protein interactions in
the new data set, the decagon method’s results cannot be
calculated. However, this does not affect the effectiveness
of the proposed method. This is because we have proved
that the SC-DDIS method outperforms the Decagon method
in data sets hundreds of times richer than the new data set.

The main purpose of introducing a second new dataset is
to verify the effectiveness of using other drugs’ biological
profiles to calculate the drug-drug positive effect and verify



our sign propagation’s effectiveness and test the SC-DDIS
model’s generalization.
Table 4 shows that the SC-DDIS model proposed in
this paper is superior to the comparison model in all
aspects, which verifies its generalization, the effectiveness
of using other drugs’ biological profiles to calculate the
drug-drug positive effect, and the effectiveness of our sign
propagation.


**5.2 Parameters analysis**


In this section, we mainly investigate two parameters, the
order of sign propagation m and the threshold _μ_ . We select
three side effects with fewer edges among the 964 types
of side effects. These three side effects have relatively few
edges; thus, they contain less information. As a result, they
better reflect the performance of the proposed SC-DDIS
model. Data for the three side effects are shown in Table 5.

We first test the effect of m to our model. Here, the
experimental parameter settings were the same as above
except for m. When _m_ = _(_ 0 _,_ 1 _, . . .,_ 9 _)_, we calculate the
change in AUROC and AUPRC scores for these three side
effects, as shown in Fig. 6 and 7.
When _m_ = 0, the SPM is the identity matrix, and,
when _m_ = 1, the SPM is a DDISN. It can be seen from
the experimental results that when _m_ = 1, that is, after



**Table 3** Best/Worst AUPRC scores for Decagon model and SC-DDIS model


Best AUPRC Decagon SC-DDIS Worst AUPRC Decagon SC-DDIS


Mumps 0.964 0.996 Bleeding 0.679 0.827

Carbuncle 0.949 0.994 Increased body temperature 0.680 0.839

Coccydynia 0.943 0.996 Emesis 0.693 0.856

Tympanic membrane perfor 0.941 0.988 Renal disorder 0.694 0.825

Dyshidrosis 0.938 0.997 Leucopenia 0.695 0.829

Spondylosis 0.929 0.980 Diarrhea 0.705 0.860

Schizoaffective disorder 0.919 0.993 Icterus 0.707 0.847

Breast dysplasia 0.918 0.978 Nausea 0.711 0.860

Ganglion 0.909 0.997 Itch 0.712 0.842

Uterine polyp 0.908 0.990 Anaemia 0.712 0.926


8328 T. Liu et al.


**Table 4** AUROC and AUPRC, and AP@2000 scores


Scenario AUROC AUPRC AP@2000


Concatenated drug features 0.845 0.851 0.829

DeepWalk neural embeddings 0.913 0.918 0.984

DEDICOM tensor factorization 0.775 0.799 0.824

RESCAL tensor factorization 0.776 0.800 0.827

SC-DDIS _(m_ =0 _)_ 0.863 0.896 0.981

SC-DDIS _(m_ =1 _)_ 0.905 0.923 0.990

SC-DDIS _(m_ =2 _)_ 0.928 0.935 0.990


**Table 5** Number of edges in three sorts of ADRs data


Side effect name The number of edges The number of edge when _e(i, j)_ = _e(j, i)_


Avascular necrosis 498 964

Carcinoma of the cervix 656 900

Phlebitis superficial 583 929


**Fig. 6** AUROC scores of three
side effects


Modeling polypharmacy effects with heterogeneous signed graph convolutional networks 8329


**Fig. 7** AUPRC scores of three
side effects



adding a positive effect between drugs to form a DDISN,
the product is significantly improved. The empirical results
show that considering the semantic information in the
real world, medicines’ characteristic information can be
enriched, the side effect relationship network’s sparseness
can be reduced, and the prediction performance can be
improved (Table 6).
When _m_ = 2, the SPM is the first-order sign propagation
based on the structural balance theory in the DDISN. As
shown in Figs. 6 and 7, improvement to the score is obvious
and stable for _m_ = 2. According to the experimental results,
we can find that after the first-order sign propagation,
the effect has been significantly improved. The experience
result also proves once again that the proposed SC-DDIS
extends the structure balance theory of the signed network
and applies it to learn higher-order sign propagation features
is correct.

However, when the order is greater than 2, non-critical
information may be introduced, thus making the the model’s
performance in-stable. Therefore, _m_ was generally set
to 2 to enhance the effectiveness and robustness of the


**Table 6** AUROC and AUPRC scores of three side effects



model. Note that parameter _m_ can be controlled flexibly
for different side effects to achieve the best model effect.

For example, the optimal prediction result for the avascular
necrosis side effect is obtained when _m_ = 7.

Then, we test the impact of _μ_ on the model’s
performance. We used the above three side effects to test
the impact varying _μ_ change on the model’s performance.
Here, the parameter settings were the same as that of the
original trial except for _μ_ . When _μ_ = _(_ 0.05 _,_ 0.1 _, . . .,_ 0.99 _)_,
we obtained changes to the AUROC and AUPRC scores for
these three side effects.

From Figs. 8 and 9, we see that both scores are
satisfactory enough when the value of _μ >_ 0.8. It is
practically significant, as the larger the _μ_ value,the higher
the similarity of two drugs. Here, we consider that the two
drugs have a positive effect when _S_ _ij_ _>_ 0.8 Therefore, a
DDISN with larger _μ_ is frequently more reliable. However,
if the _μ_ value is close to 1, less information about the
positive effects between drugs is obtained thus making
model performance decrease. This is why we set _μ_ = 0.8 in
our experiments. For the second new dataset, we calculate



Scenario Avascular necrosis Carcinoma of the cervix Phlebitis superficial


AUROC AUPRC AUROC AUPRC AUROC AUPRC


m = 0 0.956 0.974 0.962 0.975 0.955 0.969

m = 1 0.969 0.976 0.979 0.981 0.968 0.970

m = 2 0.985 0.989 0.983 0.985 0.984 0.984


8330 T. Liu et al.


**Fig. 8** AUROC scores of three
side effects



the AUROC and AUPRC scores by changing the value of _m_
to verify the effectiveness of the sign propagation matrix, as
shown below (Figs. 10 and 11).
When m = 1, that is, the positive interaction between
drugs is considered. After adding the positive action
between drugs, the drug sign network will be formed.
There were 17,146 pairs of positive reactions between drugs
and 97,168 pairs of side effects in the drug sign network.


**Fig. 9** AUPRC scores of three
side effects



The first-order drug sign network is used as the initial
feature expression of drugs. According to the experimental
results, it can be found that the performance of the model
has been dramatically improved. Once again proved the
generalization and effectiveness of our method. The reason
for the performance improvement is the introduction of
semantic information between drugs in the real world. That
is, there are positive effects and side effects of drugs. When


Modeling polypharmacy effects with heterogeneous signed graph convolutional networks 8331


**Fig. 10** AUROC scores



m = 2, based on extends the structure balance theory of the
signed network, it is applied to learn the high-order symbol
propagation characteristics. According to the experimental
results, it is proved again that we can obtain high-order
semantic information according to the extent of the signed
network’s structural balance theory, which can significantly
improve the effect.


**Fig. 11** AUPRC scores



**6 Conclusions**


In this paper, we proposed the use of semantic information
to integrate drug heterogeneous networks. Then, we applied
an extended structure balance theory and sign propagation
technology to improve the performance of predicting
adverse drug reactions. A heterogeneous information


8332 T. Liu et al.



network embedding method (i.e., the proposed SC-DDIS)
based on spectral convolution was designed to fuse
multi-source drug information. We developed a general
fusion method based on Jaccard similarity to obtain sign
network integrated information for different heterogeneous
information. Then, the sign network’s learning is performed
via the extended structure balance theory and sign
propagation to obtain more abundant information in
drug networks, which significantly improves the model’s
prediction performance. A large number of experiments
demonstrate that model parameters _μ_ = 0.8, and _m_ = 2
yield excellent prediction performance in most cases. We
further improve model robustness by training a decoding
matrix to predict ADR, and a weighted cross-entropy loss
function is used to calculate the loss value to reduce the

negative impact of sample imbalance on the proposed
SC-DDIS. We performed many ADR prediction task
experiments and compared the results to those obtained
by Decagon model and other classic multi-relational link
prediction methods, and the results proved the effectiveness
of SC-DDIS. We also analyzed the essential parameters
of the proposed SC-DDIS model and factors that affect
parameter selection.
In this study, we only considered the relationship
between drugs and other information. Thus, in future, we
plan to investigate how to add protein-protein interactions
while ensuring rationality. To further solve the critical
problems in ADR prediction, we also plan to consider how
to make full use of semantic information to improve ADR
prediction results’ interpretability.


**Acknowledgements** This work was supported by the National Natural
Science Foundation of China under Grant 61672329 and 81273704, in
part by the Project of the Shandong Provincial Project of Education
Scientific Plan (No.SDYY18058).


**References**


1. A community computational challenge to predict the activity of
pairs of compounds. Nature Biotechnology 32(12), 1213–1222
2. Cami A, Manzi S, Arnold A, Reis BY (2013) Pharmacointeraction
network models predict unknown drug-drug interactions. PLOS
[ONE 8(4):1–9. https://doi.org/10.1371/journal.pone.0061468](https://doi.org/10.1371/journal.pone.0061468)
3. Cartwright D, Harary F (1977) Structural balance: A generalization of heider’s theory 1. Soc Netw 63(5):9–25
4. C¸ elebi R, Mostafapour V, Yasar E, G¨um¨us O, Dikenelli O
(2015) Prediction of drug-drug interactions using pharmacological
similarities of drugs. In: 2015 26th International workshop on
database and expert systems applications (DEXA), pp 14–17
5. Chen X, Liu X, Wu J (2019) Drug-drug interaction prediction
with graph representation learning. In: 2019 IEEE International
conference on bioinformatics and biomedicine (BIBM), pp 354–
361



6. D’Informatique D, Ese N, Esent P, Au E, Gers F, Hersch P,
Esident P, Frasconi P (2001) Long short-term memory in recurrent
neural networks. Epfl 9(8):1735–1780
7. Giacomini KM, Krauss RM, Roden DM, Eichelbaum M, Hayden
MR, Nakamura Y (2007) When good drugs go bad. Nature
446(7139):975–977
8. Glorot X, Bengio Y (2010) Understanding the difficulty of
training deep feedforward neural networks. In: Proceedings of the
thirteenth international conference on artificial intelligence and
statistics, pp 249–256
9. Han K, Jeng EE, Hess GT, Morgens DW, Li A, Bassik MC (2017)
Synergistic drug combinations for cancer identified in a crispr
screen for pairwise genetic interactions. Nature Biotechnology
10. Hu B, Wang H, Wang L, Yuan W (2018) Adverse Drug Reaction
Predictions Using Stacking Deep Heterogeneous Information Network Embedding Approach. Molecules. 23(12):3193.
[https://doi.org/10.3390/molecules23123193.](https://doi.org/10.3390/molecules23123193) [https://www.mdpi.](https://www.mdpi.com/1420-3049/23/12/3193)
[com/1420-3049/23/12/3193](https://www.mdpi.com/1420-3049/23/12/3193)

11. Hu B, Wang H, Yu X, Yuan W, He T (2017) Sparse network
embedding for community detection and sign prediction in signed
social networks. Journal of Ambient Intelligence & Human[ized Computing 10 (1)1–12. https://doi.org/10.1007/s12652-017-](https://doi.org/10.1007/s12652-017-0630-1)
[0630-1](https://doi.org/10.1007/s12652-017-0630-1)

12. Huang LC, Wu X, Chen JY (2013) Predicting adverse drug
reaction profiles by integrating protein interaction networks with
drug structures. Proteomics 13(2):313–324
13. Jaccard P (1912) The distribution of flora in the alpine zone. N
Phytol 11(2):37–50
14. Jia J, Zhu F, Ma X, Cao ZW, Li YX, Chen YZ (2009) Mechanisms
of drug combinations: interaction and network perspectives. Nat
Rev Drug Discov 8(6):516–516
15. Kastrin A, Ferk P, Leskoˇsek B (2018) Predicting potential
drug-drug interactions on topological and semantic similarity
features using statistical learning. PLOS ONE 13(5):1–23.
[https://doi.org/10.1371/journal.pone.0196865](https://doi.org/10.1371/journal.pone.0196865)
16. Kingma DP, Ba J (2014) Adam: A method for stochastic
optimization. Computer Science
17. Kipf TN, Welling M (2016) Semi-supervised classification with
graph convolutional networks
18. Liu H, Liu B, Zhang H, Li L, Qin X, Zhang G (2018) Crowd
evacuation simulation approach based on navigation knowledge
and two-layer control mechanism. Inform Sci 436–437:247–267
19. Liu R, Wang H, Yu X (2018) Shared-nearest-neighbor-based
clustering by fast search and find of density peaks. Inform Sci
450:200–226

20. Liu S, Huang Z, Qiu Y, Chen YP, Zhang W (2019) Structural
network embedding using multi-modal deep auto-encoders for
predicting drug-drug interactions. In: 2019 IEEE International
conference on bioinformatics and biomedicine (BIBM), pp 445–
450

21. Marinka Z, Monica A, Jure L (2018) Modeling polypharmacy
side effects with graph convolutional networks. Bioinformatics
34(13):i457–i466
22. Nickel M, Tresp V, Kriegel HP (2011) A three-way model for
collective learning on multi-relational data. In: Proceedings of the
28th international conference on machine learning, ICML 2011,
Bellevue, Washington, USA, June 28 - July 2, 2011
23. Papalexakis EE, Faloutsos C, Sidiropoulos ND (2016) Tensors for
Data Mining and Data Fusion: Models, Applications, and Scalable
Algorithms. Association for Computing Machinery, New York,
[NY, USA 8(2) 44. 2157–6904. https://doi.org/10.1145/2915921](https://doi.org/10.1145/2915921)


Modeling polypharmacy effects with heterogeneous signed graph convolutional networks 8333



24. Park K, Kim D, Ha S, Lee D (2015) Predicting pharmacodynamic drug-drug interactions through signaling propagation
interference on protein-protein interaction networks. Plos One
10(10):e0140816
25. Perozzi B, Al-Rfou R, Skiena S (2014) Deepwalk: Online
learning of social representations. In: Proceedings of the
20th ACM SIGKDD international conference on knowl
edge discovery and data mining, KDD ’14, 701–710, Association for Computing Machinery, New York, NY, USA.
[https://doi.org/10.1145/2623330.2623732](https://doi.org/10.1145/2623330.2623732)
26. Qin X, Liu H, Zhang H, Liu B (2018) A collective motion
model based on two-layer relationship mechanism for bi-direction
pedestrian flow simulation. Simul Modell PractTheory 84:268–
285

27. Tang J, Chang Y, Aggarwal C, Liu H (2015) A survey of signed
network mining in social media. ACM Computing Surveys
28. Wang XX, Li JB (2005) Method of computing accessibility matrix


**Affiliations**


**Taoran Liu** **[1]** **· Jiancong Cui** **[1]** **· Hui Zhuang** **[1]** **· Hong Wang** **[1,2]**


Taoran Liu

[ltrbless@163.com](mailto: ltrbless@163.com)


Jiancong Cui
[201711010104@sdnu.edu.cn](mailto: 201711010104@sdnu.edu.cn)


Hui Zhuang
sdnu [zh@163.com](mailto: sdnu_zh@163.com)


1 School of Information Science and Engineering, Shandong
Normal University, Jinan, 250358, China
2 Shandong Provincial Key Laboratory for Distributed Computer
Software Novel Technology, Shandong Normal University, Jinan,
250358, China



from adjacency matrix. Journal of Jilin Institute of Chemical
Technology
29. Zhang W, Chen Y, Liu F, Luo F, Tian G, Li X (2017) Predicting potential drug-drug interactions by integrating chemical,
biological, phenotypic and network data. Bmc Bioinformatics
18(1):18
30. Zheng Y, Peng H, Ghosh S, Lan C, Li J Inverse similarity and reliable negative samples for drug side-effect
prediction. BMC Bioinformatics 19(13) 1471–2105.
[https://doi.org/10.1186/s12859-018-2563-x](https://doi.org/10.1186/s12859-018-2563-x)
31. Zong N, Hyeoneui K, Victoria N, Olivier H (2017) Deep
mining heterogeneous networks of biomedical linked data to
predict novel drugtarget associations (15)15. Bioinformatics
[33(15):1367–4803. https://doi.org/10.1093/bioinformatics/btx160](https://doi.org/10.1093/bioinformatics/btx160)


**Publisher’s note** Springer Nature remains neutral with regard to
jurisdictional claims in published maps and institutional affiliations.


