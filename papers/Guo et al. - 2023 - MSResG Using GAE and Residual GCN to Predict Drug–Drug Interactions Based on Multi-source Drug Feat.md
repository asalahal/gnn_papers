Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188
https://doi.org/10.1007/s12539-023-00550-6




# **MSResG: Using GAE and Residual GCN to Predict Drug–Drug** **Interactions Based on Multi‑source Drug Features**

**Lin Guo** **[1]** **· Xiujuan Lei** **[1]** **[· Ming Chen](http://orcid.org/0000-0002-9901-1732)** **[2]** **· Yi Pan** **[3]**


Received: 15 September 2022 / Revised: 5 January 2023 / Accepted: 7 January 2023 / Published online: 17 January 2023
© International Association of Scientists in the Interdisciplinary Areas 2023


**Abstract**

Drug–drug interaction refers to taking the two drugs may produce certain reaction which may be a threat to patients’ health,
or enhance the efficacy helpful for medical work. Therefore, it is necessary to study and predict it. In fact, traditional experimental methods can be used for drug–drug interaction prediction, but they are time-consuming and costly, so we prefer
to use more accurate and convenient calculation methods to predict the unknown drug–drug interaction. In this paper, we
proposed a deep learning framework called MSResG that considers multi-sources features of drugs and combines them
with Graph Auto-Encoder to predicting. Firstly, the model obtains four feature representations of drugs from the database,
namely, chemical substructure, target, pathway and enzyme, and then calculates the Jaccard similarity of the drugs. To balance different drug features, we perform similarity integration by finding the mean value. Then we will be comprehensive
similarity network combined with drug interaction network, and encodes and decodes it using the graph auto-encoder based
on residual graph convolution network. Encoding is to learn the potential feature vectors of drugs, which contain similar
information and interaction information. Decoding is to reconstruct the network to predict unknown drug-drug interaction.
The experimental results show that our model has advanced performance and is superior to other existing advanced methods.
Case study also shows that MSResG has practical significance.


Extended author information available on the last page of the article

## Vol.:(0123456789) 1 3


172 Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188


**Graphical Abstract**


**Keywords** Drug–drug interactions · Graph auto-encoder · Heterogeneous network · Residual graph convolutional network

## 1 3


Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188 173



**Fig. 1** Examples of drug-drug interactions


**1 Introduction**


Nowadays, combined drug therapy is gradually becoming
a very effective method to treat cancer, cardiovascular and
cerebrovascular diseases, lung diseases and other diseases,
because they can improve the therapeutic effect of drugs and
reduce the toxicity of drugs [1]. However, combined drug
therapy may lead to drug–drug interaction (DDI). DDI refers
to the compound effect produced by the patient taking two
or more drugs at the same time or within a certain period
2]. Such
of time. One drug may change the effect of others [
changes include enhancement or weakening of physiological
1, the metabolism of vandetanib
effects. As shown in Fig.
(DrugBank [3] ID: DB05294) can be decreased when combined with Citalopram (DrugBank ID: DB00215). When
combined with Omeprazole (DrugBank ID: DB00338),
the serum concentration of fosphenytoin (DrugBank ID:
DB01320) will increase. When using combined drug therapy, we are happy to see the effect of enhanced efficacy,
but the occurrence of side effects may even endanger the
patients’ life. Therefore, doctors should pay attention to the
different features of various drugs when taking combined
drugs for patients, so as to strengthen the efficacy as much
as possible, reduce adverse reactions of drugs, and eliminate



life-threatening situations [4]. In effect, we can detect DDI
through traditional experimental methods, but they are timeconsuming and costly, so we prefer to use more accurate and
convenient computational methods to predict DDI.
At present, the mainstream calculation methods for
DDI prediction are mainly based on the traditional classi5]. These two
fier method and the deep learning method [
methods establish models by using the known DDI and
different drug features in the existing drug databases, and
then predict the drugs with unknown interaction relations.

[6, 7] Both of them reduce the experimental time and cost
vastly. For large sample data, it far outperforms traditional
laboratory methods.
The methods based on traditional classifiers usually measure the similarity with the help of different features of drugs.
Similarity measurement methods commonly used include
Jaccard similarity [8], Cosine similarity [9] and so on. We
used the similarity measurement to construct drug features,
the classifier to predict the probability of potential DDI. The
common classifiers include random forest, k-nearest neighbor, logistic regression, support vector machine, adaptive
enhancement and so on. Cheng et al. [10] calculated the
drug similarity with the help of four different features, and
then applied the five models of naive Bayes, decision tree,
support vector machine, logistic regression and k-nearest
neighbor to predict the unknown DDI.
With the continuous development of computer technology, the DDI prediction method based on deep learning has gradually become the mainstream method. This
method predicts unknown DDI by using deep learning to
obtain the potential features of drugs. Several common
deep learning technologies in this field include Convolutional Neural [11] (CNN), Recursive Neural Networks [12]
(RNN), Graph Neural Networks [13, 14] (GNN) and other
neural networks, as well as various variants of these neural
networks. Ryu et al. [15] proposed a computing framework
based on deep learning called deepDDI. This framework
uses the Simplified Molecular Input Line Entry System
(SMILES) sequence of drugs to calculate the Structural
Similarity Profile (SSP) to obtain the drug feature vectors,
and concatenates the feature vectors of drug pairs. Then it
uses the Principal Component Analysis (PCA) to reduce
the dimension and input it into the deep neural network
(DNN) for DDI prediction. Feng et al. [16] proposed a
deep learning model of DPDDI which first uses a two-layer
graph convolutional network (GCN) to learn the potential
feature representation of drugs by topological information
from DDI networks, and concatenate the feature vectors of
drug pairs, and then inputs the feature vectors representing drug pairs into the DNN to predict DDI. Lin et al. [17]
proposed a deep learning framework based on knowledge
map called KGNN. The author extracted DDI from the
database and constructed a knowledge map. Subsequently,

## 1 3


174 Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188


**Table 1**
Specific information of Dataset Drug DDI Drug pair Sparsity Feature Data Source Number
DB1 and DB2


DS1 548 97,168 300,304 67.64 Target DrugBank 780

Enzyme DrugBank 129

Pathway KEGG 253

Chemical substructure PubChem 881

DS2 707 68,794 499,849 86.24 Chemical substructure PubChem 881



the GCN was used to learn the representation of drugs and
their topological neighborhood and perform feature aggregation and to predict potential drug interactions finally.
On the basis of KGNN, Han et al. [18] proposed the SmileGNN using SMILES sequence to extract the structural
features of drugs. The topological features of drugs are
extracted from the knowledge graph (KG), and then the
structure and topological features of drugs are aggregated
to predict DDI.
In this article, we proposed a new model called MSResG.
Firstly, this model obtained four feature representations of
drugs, namely, chemical substructure, target, pathway and
enzyme, and then calculated the Jaccard similarity measure
of drugs. In order to comprehensively consider the contribution of drug features, we integrated the similarity by averaging to obtain the comprehensive similarity network of drugs.
Secondly, we combined the drug comprehensive similarity
network with the DDI network, and used the Residual Graph
Convolutional Network (ResGCN) [19] encoder to learn the
potential embedding vector of the drug on the combined
network. Finally, the decoder was used to reconstruct the
network to obtain the predicted DDI. Experiments show
that the performance of MSResG is better than the existing
methods for predicting DDI. In general, this is due to the
following advantages of our model:


1. MSResG effectively integrates multi-source drug features, including targets, chemical substructures, enzymes
and pathways, comprehensively considers the impact of
different information contained in different features on
model performance.
2. In order to extract more drug information, we built a
heterogeneous network containing both DDI information and similarity information. In the past, most models
only used DDI information networks as feature extraction sources [16, 20].
3. MSResG compared the model performance of norm
GCN, PlainGCN and ResGCN when they were used as
encoders by ablation experiments. The ablation experiments verified that the residual network can improve
the convolutional operation of graphs with more than 3

## 1 3



layers. So we use the Graph Auto-Encoder [21] (GAE)
based on ResGCN to predict DDI.


**2 Methods**


**2.1 Datasets**


In this article, we selected two data sets from previous studies, namely DS1 and DS2 which are from [22] and [23].
Lin et al. [24], Schwarz et al. [25] and Feng et al. [16] also

used these two datasets or one of them in their own articles

and introduce them. Table 1
shows the specific information
of them. DS1 contains 548 drugs, 97,168 DDIs, and 8 features of drugs: chemical substructure, side effect, enzyme,
transporter, target, etc. Among them, we selected four features: chemical substructure, target, pathway and enzyme
for experiments. DS2 contains 707 drugs, 34,412 DDIs, and
one drug feature that is chemical substructure. We can obtain
drug pathway information from Kyoto Encyclopedia of
Genes and Genomes (KEGG) [26], drug target and enzyme
information from DrugBank, and drug chemical substructure
information from PubChem [27].
We can find that the number of drugs in DS2 is more
than that in DS1, but the DDI is less than that in DS1. The
sparsity of DS2 reaches 86.24%, which is more than 67.64%
in DS1. These are two datasets with large differences in sparsity. We will explore the changes of sparsity to our model
performance in Sect. 3.6, and verify the impact of heterogeneous network sparsity.
DrugBank database is an authoritative database in the
field of drug information. Nowadays, DrugBank database
has been widely used in the fields of drug retrieval, drug
metabolism prediction, drug-target interaction prediction,
drug reposition [28, 29] etc. PubChem database, an open
chemical database of US National Institutes of health, contains biological activity data of organic small molecules,
where we can obtain much information regarding chemical
structure, identifier, chemical and physical properties, biological activity, toxicity data and so on. KEGG database was
established by Kanehisa Laboratory of bioinformatics center


Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188 175


**Fig. 2** In the process of multi-source heterogeneous network construction module



of Kyoto University, Japan. It has multiple sub databases
including genome, biochemical reactions, biochemical substances, diseases and drugs, and pathway information. These
databases having made great contributions to the research of
bioinformatics [30].


**2.2 Workflow**


To predict unknown DDI in combination with different features of drugs. We propose a GAE model called MSResG
mainly consisting of two parts, the multi-source heteroge
neous network construction module and the GAE module.

As shown in Fig. 2, the multi-source heterogeneous
network construction module firstly selects the chemical
substructure, target, pathway and enzyme to calculates the
Jaccard similarity matrix, and then obtains the comprehensive similarity matrix of the drug by means of averaging.
Then we introduce the DDI network and fuse them, the
edges of the fused network contain interaction relations
and multi-source similarity relations, so the fused network
is a multi-source heterogeneous network.



As shown in Fig. 3, in the GAE module, we encode
and decode the multi-source heterogeneous network, use
the ResGCN as the encoder to obtain the potential feature
vector of the drugs, and use the decoder to reconstruct the
network to obtain the predicted DDI.


**2.3 Multi‑source heterogeneous network**
**construction**


For this article, the calculation of drug similarity net
work takes into account the multi-source characteristics
of drugs, namely the similarity matrix of four different
characteristics of drug: chemical substructure, target,
pathway and enzyme. Chemical substructure belongs to
the chemical features of drugs, and target, enzyme and
pathway belong to the biological features of drugs. The
chemical and biological features of a drug can reflect the
nature and attributes of the drug itself. The features of a
drug correspond to a group of descriptors, so each drug
can be expressed as a binary vector. The vector element

## 1 3


176 Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188


where _d_ _i_ and _d_ _j_ represent the feature vectors of drug _i_
and drug _j_, respectively, from which four drug similarity
matrixes obtained. They are target similarity matrix _S_ _[t]_,
chemical substructure similarity matrix _S_ _[s]_, enzyme similarity matrix _S_ _[e]_, and pathway similarity matrix _S_ _[p]_ .
In addition to the Jaccard similarity measurement method,

we can also consider other measurement methods. We can

also use Cosine similarity [31] calculation method to calculate drug similarity. Cosine similarity calculation formula
is defined as:



**Fig. 3** The GAE module process based on ResGCN


can be 1 or 0 indicates the presence of the element at the
corresponding position of the descriptor, while 0 indicates
the absence. Take drug targets as an example. In DS1,
we have 548 drugs and 780 corresponding drug targets.
Therefore, drugs can be expressed as a 780-dimensional
binary vector. The element value of 1 means the presence
of a target and 0 indicates the absence. Drug DB01197
contains five targets, namely P12821, P08253, P14780,
P09960 and P46663. For DB01197, there are five positions
where the element is 1 and the remaining is 0. Other drug
features were calculated according to the same method.
Actually, since the vectors of these features has high
dimensions while most of the them are 0. Therefore, we
use Jaccard similarity to measure the drug similarity and
construct the drug similarity matrix. The Jaccard similarity

is calculated as follows:



(1)



Jaccard [(] _d_ _i_, _d_ _j_ ) =



| _d_ _i_ ∩ _d_ _j_ |

| _d_ _i_ ∪ _d_ _j_ |



_d_ _i_ ⋅ _d_ _j_

Cosine( _d_ _i_, _d_ _j_ ) = ‖‖ _d_ _i_ ‖‖ × [‖] ‖‖ _d_ _j_ [‖] ‖‖ (2)


where || _d_ _i_ ||and || _d_ _j_ || denote the L2-norm of _d_ _i_ and _d_ _j_ _._
In this paper, we use the Jaccard similarity to calculate
the drug similarity and consider the Cosine similarity. The
two similarity methods are compared in the chapter "3.1
Selection of Similarity Calculation Methods and Parameter

Discussion".

After obtaining the four similarity matrixes of drugs, we
need to conduct similarity fusion and comprehensively consider the contribution of each drug feature to the model. The
comprehensive similarity matrix of drugs can be obtained
by finding the mean value, it is defined as:


_[⊕]_ _[S]_ _[s]_ _[⊕]_ _[S]_ _[e]_ _[⊕]_ _[S]_ _[p]_
_S_ = _[S]_ _[t]_ (3)

4


where ⊕ represents an element addition operation.
The acquisition of DDI matrix is relatively simple. We
express DDI matrix as _I_ _[n]_ [ × ] _[n]_ . n is the number of drugs. The

matrix element of 1 indicates that there is an interaction

relationship between drug _i_ and drug _j_, and 0 indicates no
interaction relationship.
Combining DDI network and drug similarity network, we
can construct heterogeneous networks defined by adjacency
matrix as follows:


_A_ = [[] _S I_ []] ∈ ℝ _[N]_ _[d]_ [×][2] _[N]_ _[d]_ (4)


where _S_ is the comprehensive similarity network and _I_ is

the DDI network.


**2.4 Encoder**


We use the ResGCN in this article as the encoder to

encode the heterogeneous network, so as to mine the
deeper potential features of drugs.
Thomas N. Kipf [32] proposed the GCN to process
the data with graph as input. Each convolutional layer of
the GCN can process the information of the first-order
neighborhood, that is, obtain the vertex information of



=



| _d_ _i_ ∩ _d_ _j_ |

| _d_ _i_ | + | _d_ _j_ | − | _d_ _i_ ∩ _d_ _j_ |


## 1 3


Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188 177



the first-order neighborhood, and then realize the information aggregation of the multi-order neighborhood by
superimposing convolutional layers to obtain the embedded expression of all vertices.
GCN develops rapidly in the research of the real field

[33–37], but it is easy to eliminate the gradient problem
and usually limited to relatively shallow models. The number of common GCN layers in the actual research is mostly
1 to 3. Therefore, ResGCN is considered as the encoder

in the GAE. Residual connection [38] can alleviate the
problem of gradient disappearance. Based on GCN, ResGCN encodes drug heterogeneous networks and extracts
high-order neighborhood information of drugs.
In this article, we define the propagation rules of each
layer of GCN is as follows:



ResGCN learns the required underlying mapping _H_ by
fitting the residual mapping _F_ . After _G_ _l_ is transformed by
residual mapping _F_, vertex addition is performed to obtain
_G_ _l_ + 1 . Residual mapping _F_ takes a graph as input and
outputs a representation of residual graph _F_ ( _G_ _l_ _, W_ _l_ ) at the
next layer, can be written as _Gres l_ + _1_, _G_ _l_ + 1 is defined as:



_G_ _l_ +1 = _H_ [(] _G_ _l_, _W_ _l_



)



= _F_ [(] _G_ _l_, _W_ _l_




[(] _G_ _l_, _W_ _l_ ) + _G_ _l_

= _G_ _[res]_ [+] _[ G]_



(9)




_[res]_

_l_ +1 [+] _[ G]_ _[l]_



(



_̃_
_D_ [−] [1] 2




[1]

2 _H_ [(] _[l]_ [)] _W_ [(] _[l]_ [)] [)]




[1]

2 _A_ _[̃]_ _D_ _[̃]_ [−] 2 [1]



_̃_

_H_ [(] _[l]_ [+][1][)] = _𝜎_ _D_ [−] 2 _A_ _[̃]_ _D_ _[̃]_ [−] 2 _H_ [(] _[l]_ [)] _W_ [(] _[l]_ [)] (5)



where _W_ _l_ is the set of learnable parameters of layer _l_ .
In short, each ResGCN module accepts the output of the
previous layer and the residual connection as input. In this
article, we use 5 layers ResGCN. We also consider the sum
of the outputs of all residual functions. The expression of
the characteristics of any deep _L_ is as follows:



where, _H_ [(] _[l]_ [ + 1)] represents the feature vector of the node in
layer _l_ + 1, that is the output of layer _l_ + 1, _H_ [(] _[l]_ [)] represents
the feature vector of the node in layer _l_, that is, the output
of layer _l_, _W_ [(] _[l]_ [)] means the trainable weight matrix of layer
_l_, and σ means the activation function, in this model, we

select the ReLU as the activation function. And in the formula, _A_ _[̃]_ = _A_ + _I_ _N_, _I_ _N_ is the identity matrix, the diagonal element of _I_ _N_ is 1 and the other elements are 0, _D_ _[̃]_ indicating the
diagonal node degree matrix of _A_ _[̃]_ .
More specifically, the specific propagation method of
each node is as follows:



_L_ −1
∑



{



_F_ [(] _G_ _l_, _W_ _l_

_F_ [(] _G_ _l_, _W_ _l_



) + _W_ _s_ _G_ _l_, different dimensions



_F_ ( _G_ _i_, _W_ _i_ ) + _G_ 0



_G_ _L_ = _F_ ( _G_ _i_, _W_ _i_ ) + _G_ 0 (10)


_i_ =0


It should be noted that the propagation of ResGCN
between each layer also needs to consider the size of each
layer. If the dimensions are the same, it will be calculated
directly. If the dimensions are different, it needs to first
match the dimensions with linear mapping:



) + _G_ _l_, same dimensions



_G_ _l_ +1 = (11)



∑
( _j_ ∈ _Ne_ (



)



_h_ [(] _i_ _[l]_ [+][1][)] = _휎_ ∑ _h_ [(] _[l]_ [)] _[W]_ [(] _[l]_ [)] (6)



_j_ ∈ _Ne_ ( _i_ )



1
_h_ [(] _[l]_ [)]
_c_ _i_, _j_ _j_ _[W]_ [(] _[l]_ [)]



where _h(l_ + _1) i_ is the latent representation of node _i_ in layer
_l_ + 1, _h(l) j_ is the latent representation of node _j_ in layer _l_,
_Ne_ ( _i_ ) is the set of neighbors of node _i, c_ _i,j_ is normalization
constant.

We can see that the features of each node are updated
under the joint action of other nodes is as follows:



(∑



_h_ _i_ ← joint _[h]_ _[j]_ (7)



_j_ _[h]_ _[j]_



)



For the first layer, it is quite special. First GCN layer
takes the adjacency matrix of the heterogeneous network _A_
after symmetric processing as input, and the feature matrix
_X_ contains interaction features and similarity features. the
first layer is defined as:


_H_ [(][1][)] = _𝜎_ ( _AXW_ _[̃]_ [(][0][)] ) (8)



where _W_ _s_ is linear mapping.


**2.5 Decoder**


After the encoder processing, we have obtained the potential
feature vector _Z_ _[d]_ of the drug. We can express the prediction
result between the two drugs _d_ _i_ and _d_ _j_ as follows:


_P_ ( _I_ _[̂]_ _i_, _j_ = 1| _z_ _i_, _z_ _j_ ) = sigmoid( _z_ _i_ || _z_ _j_ ) (12)


where sigmoid is a function of calculating the probability of
interaction between drug _i_ and drug _j_ .
Therefore, the reconstructed DDI network structure
based on the potential feature vector _Z_ _[d]_ can be expressed

as follows:



(



_Î_ = sigmoid _Z_ _[d]_ _W_ 1 ( _Z_ _d_ _W_ 2 ) (13)



( _Z_ _d_ _W_ 2



_T_ [)]
)



_Z_ _[d]_ _W_ 1


## 1 3


178 Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188



where _Z_ _[d]_ is the potential feature vector encoded by the
encoder, _W_ _1_ and _W_ _2_ are the trainable weight matrices will
project the embedded representation back to the original

feature.


**2.6 Model Training**


MSResG is mainly divided into two modules: the multisource heterogeneous network construction module and the

GAE module. We obtained relevant information from a data
set containing _N_ drugs, taking DDI as a positive sample and
unknown DDI as a negative sample. They are divided into
five parts, four of which are the training set and the other
is the test set. We first build a multi-source heterogeneous
network, obtain the heterogeneous network, use the encoder
to obtain the potential feature vectors of drugs, and then use
the decoder to reconstruct the network to train parameters.
Since the prediction of DDI is a two class problem, so
we use binary weighted is used to cross entropy as the loss

function as follows:



∑



_i_, _j_ _[p]_ [(] _[a]_ _ij_ [)][ log][(] _[q]_ [(] _[a]_ _ij_ [))]



_Loss_ = − _i_, _j_ _[p]_ [(] _[a]_ _ij_ [)][ log][(] _[q]_ [(] _[a]_ _ij_ [))]
(14)



∗ _W_ _pos_ + (1 − _p_ ( _a_ _ij_ ))(1 − log( _q_ ( _a_ _ij_ )))



where, _p_ ( _a_ _ij_ ) is the real label of drug interactions, _q_ ( _a_ _ij_ ) is
the predicted probability of drug interactions obtained by
the decoder, _W_ pos is the weight parameter, that is equal to
the number of negative samples divided by the number of
positive samples.
To minimize the loss function, we use training data to
train our model for 400 epochs to reduce the loss function.
In this process, we also use Xavier initialization method to
initialize all trainable weight matrices to improve the starting behavior. And using the Adam optimizer [39] to minimize the loss function, the Adam optimizer can update the
weights of the network according to the training data. To
avoid over fitting, we add dropout layers in the model opti40]. In addimization to achieve the regularization effect [
tion, we also use the cyclic learning rate, which is realized
by cycling the learning rate between the maximum and
minimum of the fixed learning rate. Cyclic learning rate

[41
] can provide faster convergence effect. After the training, we evaluate the performance of the model by drawing
the receiver operating characteristic curve and calculating
relevant performance evaluation metrics.


**2.7 The MSResG Algorithm**


MSResG is mainly divided into two parts: the network
acquisition and combination module and the GAE module.

## 1 3



To describe our method in detail, we show the pseudocode is shown in Algorithm 1.


**Algorithm 1:** The MSRESG algorithm


**Input:** DDI network and all features of drugs _d_ _i_ ( _i_ =1, 2,


…, _n_ ) including chemical substructure, target,


pathway and enzyme;


The parameters: epochs, embedding-dim, learning


rate, dropout


**Output:** DDI network [ˆ] I reconstructed by MSResG


1: **for** all _i_ _N_ **do**


2: Calculate Jaccard similarity of drug features.


3: **end for**


4: Obtaining the similarity matrix _S_ _[t]_, _S_ _[s]_, _S_ _[e ]_ and _S_ _[p]_ .


5: Perform integrated similarity matrix fusion based


on Eq. (2).


6: Construct the heterogeneous network based on Eq.


(3).


7: Initialization parameters.


8: Learn drug latent feature vector _Z_ _[d ]_ by ResGCN as


encoder.


9: **for** epoch **in** epochs:


10: Compute the loss function based on Eq. (13).


11: Update parameters with Adam optimizer.


12: **end for**


13: Reconstruct DDI network by decoder.


**2.8 Complexity Analysis**


Our model involves the encoder ResGCN for obtaining the
potential feature vectors of drugs and the decoder for reconstructing the DDI network, wherein ResGCN is based on
GCN, and the computational complexity of GCN is related
to the number of edges and nodes in the network. Convolution operation of each layer is shown in Eq. 4, so the calculation complexity is _O_ ( _L_ ‖ _E_ ‖ 0 _F_ + _LNF_ [2] ) [42], where _L_ is
the number of layers, _N_ is the number of nodes, ‖ _E_ ‖ 0 is
the number of non-zero in the adjacency matrix, which can
also be understood as the number of interacting drugs, and

_F_ is the number of features. ResGCN introduces residual

concatenate, which does not add additional parameters and
does not increase the computational complexity. Therefore,


Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188 179


the complexity of ResGCN is still _O_ ( _L_ ‖ _E_ ‖ 0 _F_ + _LNF_ [2] ) .
The decoder operates as shown in Eq. 12, it is a pairwise
operation, and the calculation complexity of this part
is _O_ ( _N_ [2] ) [43]. We can know the overall complexity is
_O_ ( _L_ ‖ _E_ ‖ 0 _F_ + _LNF_ [2] + _N_ [2] ) .


**3 Results**


**3.1 Evaluation Metrics**


In this article, to ensure the fairness of the experiment, we
use fivefold CV (CV) to calculate the model performance.
We choose to divide the data set into five parts, one of which
is selected as the test set each time, and the remaining four
parts are the training set for model training. In each fold,
a prediction model is constructed on known interactions
in the training set and is used to predict interactions in the
testing set. Finally, the average value of the five groups of
test results is calculated as the estimated value of the model

performance.
For simplicity, we use the abbreviations TP _,_ FP, TN, and
FN for true positive, false positive, true negative, and false
negative, respectively. Because the interaction prediction
problem we studied is a binary classification problem, we
choose common classification indicators as performance
evaluation indicators, including the area under the receiver
operating characteristic curve (AUC), AUPR, Accuracy
(ACC), Precision, Recall, Specificity and F1-Score. The
relevant formulas as follows:


_TP_ + _TN_
_ACC_ = (15)
_TP_ + _FP_ + _FN_ + _TN_


_TP_
_Precision_ = (16)
_TP_ + _FP_


_TP_
_Recall_ = (17)
_TP_ + _FN_


_TN_
_Specificity_ = (18)
_TN_ + _FP_



_F_ 1 − _Score_ = 2 ∗ _[Precision]_ [ ∗] _[Recall]_ (19)

_Precision_ + _Recall_


In our study, the data set is unbalanced because of the
difficulty of obtaining biomedical data. The ratio of positive
and negative samples is about 1:2, and our heterogeneous
network involves DDI network and similarity network, so
it is more sparse. Therefore, as some evaluation metrics in
our experimental results is limited, we take AUC and ACC



**Fig. 4** The influence of parameters epoch

as the main evaluation metrics. The influence of sparsity on
some evaluation metrics will be described in detail in the

Sects. 3.6.

## 1 3


180 Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188



**3.2 Selection of Similarity Calculation Methods**
**and Parameter Discussion**


In our model, two similarity calculation methods are mentioned. Taking the data in the [44] as an example, for the
chemical substructure, Table 2 shows the results of the
two similarity methods.
We can see that the performance of Jaccard similarity
is superior to Cosine similarity, so we choose Jaccard
similarity as our similarity matrix calculation method in

our model.

This model has four key parameters: the total epochs _α_,
the dimension of drug embeddings _d_, the learning rate _lr_,
and the dropout rate _dp_ . We will discuss the influence of
these parameters on the performance of MSResG.
We consider the combination of parameters as follows:
_α_ ∈ {4000, 8000, 12,000, 14,000, 16,000}, _d_ ∈ {32, 48,
64, 80, 96}, _lr_ ∈ {0.0001, 0.0005, 0.001, 0.005, 0.01},
_dp_ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}. Figure 4 shows the effect of
different parameters on the model performance.
Through analysis, we can see from the Fig. 4 that the
performance of the model is constantly improving as the
number of epochs goes up, but the epoch cannot increase
indefinitely, otherwise the model may be overfitted as
well as the time and space will be wasted. By observing


**Fig. 5** The AUC ( **a** ) and AUPR ( **b** ) comparison of each feature combination

## 1 3



**Table 2** Performance comparison of two similarities


Similarity ACC​ AUC​ AUPR


Jaccard **0.9527** **0.9159** **0.6439**

Cosine 0.9495 0.9049 0.6122


The best performing feature set in each metric is in bold


the change of loss function value and ACC, 16,000 is
selected as a better epoch to achieve the best performance
relatively which avoids model overfitting.
As for the embedded dimension, we can see that the
performance of the model does not change significantly
with the embedded dimension. So 64 is selected as our

embedded dimension. If we choose too large an embedded dimension, we need to consume too much memory
space and time.
We know that the learning rate exert obvious influence
on the performance of the model. The learning rate represents the step size of each update of the weight parameter, which will affect the convergence state of the model.
Shown concretely, when the learning rate in this paper is
equal to 0.001, the model performance achieves the optimal. If the learning rate is set too large, the network not
only cannot converge but also hovers around the optimal
value. What’s more, if the learning rate is set too small,
the network convergence is very slow and it is easy to
fall into local optimization. We choose 0.5 for the size
of dropout, which can effectively reduce the occurrence
of overfitting.
To sum up, we choose _α_ is 16000, _d_ is 64, _lr_ is 0.001
and _dp_ is 0.5. This parameter combination is used as our
experimental parameter. At this time, the model perfor
mance can reach the best.


**Fig. 6** ROC curves of three models under ablation experiment in fivefold CV


Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188 181


**3.3 Multi‑source Drug Fusion Improves DDI**
**Prediction**


The construction of the heterogeneous network in our model
depends on the drug similarity network, so we discussed the
influence of the similarity combinations of different features
of drugs on the experimental performance. Table 3 shows
the influence of different similarity combinations on the
experimental performance. Figure 5 shows the AUC and
AUPR comparison of each combination.
When only one drug feature is used, the corresponding
model performance of chemical substructure similarity of
drugs is the best. The AUC reached 0.9389. Considering the similarity of the two drug features, the combined
AUC of the chemical substructure similarity and the target similarity was the highest, which reached 0.9404. In
fact, it is obvious that when only one feature is used, the
performance corresponding to the chemical substructure
similarity and the target similarity belongs to the front,
and to our surprise their combined similarity also belongs
to the front. When considering the three feature combinations of drugs, the AUC of drug path, enzyme, and
chemical substructure is 0.9421. In general, the feature
combined with the best performance is the combination of


**Fig. 8** Performance line graph under the fivefold CV ( **a** ) and tenfold
CV ( **b** )


drug target, pathway, enzyme, and chemical substructure.
It reaches the best in 5 evaluation metrics, about AUC

is 0.9426, ACC is 0.9529, AUPR is 0.7627, F1-Score is
0.7086 and precision is 0.7710. Through the experiments
in this section, we can draw a conclusion that the fusion



**Fig. 7** ROC curve under the fivefold CV ( **a** ) and tenfold CV ( **b** )



**Fig. 9** Effect of DDI sparsity on experimental performance

## 1 3


182 Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188



**Table 3** Effect of similarity
combination of different
features of drugs on
experimental performance



Feature combination AUPR AUC​ F1-Score ACC​ Recall Specificity Precision


S 0.7587 0.9389 0.6930 0.9504 0.6417 **0.9899** 0.7539

T 0.7158 0.9249 0.6863 0.9391 0.7531 0.9569 0.6397

P 0.6744 0.9147 0.6515 0.9289 0.7611 0.9450 0.5695

E 0.6696 0.8973 0.6265 0.9143 0.6828 0.9231 0.5057

T + S 0.7498 0.9404 0.7079 0.9526 0.6576 0.9808 0.7667

P + S 0.7026 0.9333 0.6762 0.9350 0.7775 0.9501 0.5983

E + S 0.7151 0.9330 0.6964 0.9445 0.7294 0.9650 0.6662

T + P 0.7046 0.9284 0.6813 0.9373 0.7661 0.9537 0.6146

T + E 0.7070 0.9240 0.6797 0.9370 0.7569 0.9542 0.6275

P + E 0.6911 0.9227 0.6628 0.9299 **0.7897** 0.9433 0.5710

P + E + S 0.7610 0.9421 0.7086 0.9525 0.6610 0.9804 0.7640

T + P + S 0.7569 0.9416 0.7083 0.9525 0.6607 0.9804 0.7634

T + P + E 0.7563 0.9416 0.7084 0.9529 0.6555 0.9813 0.7708

S + T + E 0.7557 0.9411 0.7084 0.9528 0.6565 0.9811 0.7694

E + P + S + T **0.7627** **0.9426** **0.7086** **0.9529** 0.6558 0.9813 **0.7710**


_S_, chemical substructure; _T_, target; _P_, pathway; _E_, enzyme


The best performing data is in bold



of multi-source features of drugs plays an important role
in improving the experimental performance.


**3.4 Ablation Experiment**


So as to explore how ResGCN as an encoder improves the
performance of the proposed model, we performed abla
tion studies on MSResG variants. There are two model

variants, we use them as the encoder for coding respectively, namely, PlainGCN [19] and NormGCN. Thereinto,
PlainGCN is the output of the upper layer of the GCN
of each layer as the input, and the output of all layers is
taken into account in the final output. NormGCN is only
the output of the upper layer of the GCN of each layer as
the input.
Under the condition that the decoder is unchanged,
Table 4 shows the ablation results, and Fig. 6 shows the
ROC curves of the three models. The experiment verifies
the importance of ResGCN in our model.
It can be seen that when ResGCN is used as an encoder,
the performance of the model reaches the best, and the

area under the ROC curve also reaches the maximum. The

experiment proves that GCN with residual connection can
effectively alleviate the problem of gradient disappearance, help us obtain deeper and richer drug information,
and improve the prediction performance of DDI.


**3.5 MSResG Under Fivefold CV and Tenfold CV**


In this section, we will show the performance of MSResG

7 shows the
under the fivefold CV and tenfold CV. Figure
8
ROC curve under the fivefold CV and tenfold CV. Figure

## 1 3



shows the performance line graph under the tenfold CV. It

can be seen that the ROC curve of each fold has little dif
ferent, and the line trend of the line graph is stable, which
proves that MSResG is a stable model.


**3.6 Effect of Dataset Sparsity on Performance**


In this article, there are two data sets, DS1 and DS2. By
utilizing these data, we know that the sparsity rate of DS1 is
67.64% and that of DS2 is 86.24%. In DS2, drug pairs without interaction account for a large part. In this section, we
compare the effects of different sparsity of drug interaction
networks on the prediction performance of DDI. Because
there are only chemical substructure features of drugs in
dataset 2, we also only consider the chemical substructure
features in dataset 1, instead of using multi-source drug features. The results are shown in Fig. 9.


**Fig. 10** Performance comparison with other methods


Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188 183



**Table 4** Performance

comparison of ablation
experiments


**Table 5** Performance

comparison with other methods



Encoder AUPR AUC​ F1-Score ACC​ Recall Specificity Precision


ResGCN **0.7627** **0.9426** **0.7086** **0.9529** 0.6558 **0.9813** **0.7710**

PlainGCN 0.7582 0.9402 0.7083 0.9527 **0.6581** 0.9809 0.7671

GCN 0.7037 0.9199 0.6748 0.9474 0.6248 0.9783 0.7336


The best performing data is in bold


Methods ACC​ AUC​ AUPR F1-Score Recall Precision


MSResG-Comprehensive **0.956** **0.958** **0.798** **0.732** 0.687 **0.783**

MSResG-Substructure 0.950 0.939 0.759 0.693 0.642 0.754

MSResG-Target 0.939 0.925 0.716 0.686 0.753 0.640

MSResG-Pathway 0.929 0.915 0.674 0.651 0.761 0.569

MSResG-Enzyme 0.914 0.897 0.670 0.626 0.683 0.506

LP-Comprehensive 0.950 0.936 0.761 0.684 0.768 0.618

LP-Substructure 0.950 0.936 0.757 0.691 0.768 0.612

LP-Target 0.927 0.851 0.559 0.544 0.603 0.496

LP-Enzyme 0.927 0.760 0.470 0.451 0.664 0.342

LP-Pathway 0.938 0.811 0.595 0.572 0.716 0.476

KNN-Comprehensive 0.951 0.938 0.766 0.689 0.777 0.679

KNN-Substructure 0.950 0.936 0.759 0.682 0.776 0.608

KNN-Target 0.867 0.819 0.364 0.418 0.339 0.547

KNN-Enzyme 0.908 0.756 0.378 0.398 0.463 0.350

KNN-Pathway 0.932 0.812 0.572 0.550 0.655 0.474

MP 0.952 0.948 0.781 0.708 0.754 0.667

CE1 0.953 0.948 0.786 0.712 0.645 0.775

CE2 0.954 0.957 0.792 0.723 0.678 0.767

TAN* 0.684 0.670 0.273 0.229 0.535 0.145

IPF* 0.880 0.872 0.413 0.447 0.553 0.377


The best performing data is in bold


*The results are taken from [22]



In DS1, the performance evaluation metrics AUC, ACC
and Specificity are similar to those in DS2. However, the
AUPR in DS2 is only 0.5073, 33.14% lower than that of
DS1. The F1-Score is 0.4848, 30.04% lower than that of
DS1. The Recall is 0.5406, 15.63% lower than that of DS1.
The Precision is 0.4403, 41.60% lower than that of DS1.
The number of drugs in the two datasets is not much different, but the sparsity is very large. We can reasonably infer
that the sparsity of the data set will cause low AUPR and
other metrics. Because the heterogeneous network in our
model involves drug similarity network and DDI network, it
is particularly obvious, which does not mean that the model
is poor.


**3.7 Comparison with Other Methods**


We compared our MSResG with other methods. There are
seven comparison methods. In the label propagation method

[45] (named as LP), each drug node label is propagated to
the adjacent nodes according to the similarity, and then its
own label is updated according to the labels of the adjacent



nodes. Using the relationship between samples, a graph
model is established to predict DDI. The k-nearest neighbor
algorithm [46] (named as KNN) classifies position samples
by considering the nearest K sample pairs. We use drug similarity as the sample distance to predict DDI. Matrix perturbation method [47] (named as MP) is to deduce, calculate
and predict the DDI network based on a hypothesis that the
regularity of the network is reflected in the consistency of
the structural features before and after the random removal

of a small part of links. Zhang's method [22] is named CE.
CE considers eight features of drugs, builds a model with
machine learning, uses greedy algorithm to get the contribution of different drug features to the model performance,
and then predicts DDI with the integration framework of
weighted average integration rule (named as CE1) and classifier integration rule (named as CE2). Vilar proposed two
methods, we named them as TAN [48] and IPF [49]. TAN is
based on the Tanimoto similarity matrix of drug molecular
structure to predict DDI, and IPF is based on the drug interaction profile fingerprints (IPFs) to measure the similarity
between drug pairs to predict DDI.

## 1 3


184 Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188


**Table 6** Top 10 novel interactions predicted by our method


Rank ID-A Name-A 3D structure ID-B Name-B 3D structure Evidence and Interaction relationship



1 DB00945 Acetylsalicylic acid


2 DB00398 Sorafenib


3 DB00853 Temozolomide


4 DB01193 Acebutolol


5 DB01193 Acebutolol


6 DB01248 Docetaxel*


7
DB00862 Vardenafil


8 DB00990 Exemestane


9 DB00953 Rizatriptan


10 DB00571 Propranolol



DB01033 Mercaptopurine


DB00563 Methotrexate


DB00758 Clopidogrel


DB00264 Metoprolol


DB00927 Famotidine


DB00864 Tacrolimus*


DB00443 Betamethasone


DB00635 Prednisone


DB00178 Ramipril


DB01203 Nadolol



DrugBank: the excretion of Mercaptopurine
can be decreased when combined with Ace
tylsalicylic acid


DrugBank: the risk or severity of adverse
effects can be increased


DrugBank: the risk of bleeding can be
increased when Clopidogrel is combined
with Temozolomide


DrugBank: metoprolol may increase the
arrhythmogenic activities of Acebutolol


_NA_


DrugBank: the serum concentration of
Tacrolimus can be increased when it is com
bined with Docetaxel


DrugBank: the metabolism of Vardenafil can
be increased


_NA_


DrugBank: rizatriptan may decrease the antihypertensive activities of Ramipril


DrugBank: propranolol may increase the
arrhythmogenic activities of Nadolol



*Conformer generation is disallowed since too many atoms, so we show the 2D structure



Because our method involves the similarity of different
features, in addition to considering the performance of the
model on the comprehensive features, we also consider the
performance of the model on the enzyme, target, chemical
substructure and pathway. Table 5 and Fig. 10 shows the
performance comparison of comparison methods.
We can see that MSResG achieves the best performance
in five aspects. Only the Recall performance evaluation metrics is not optimal. After our analysis, this is due to the sparsity of heterogeneous networks. In fact, our model achieves

better results because it considers multi-source feature infor
mation, KNN and LP also have better performance than single feature when using multi-source features, and MSResG
also uses ResGCN to make layers deeper and learn deeper

information.

## 1 3



**3.8 Case Study**


To verify the efficiency of MSResG in practical application,
we also propose a case study in this section. We used the
drug features and DDIs in DS1 to train the prediction model,
and then predicted other drug pairs. With the focus on the
DDIs with the top 10 prediction scores, we made an efficacious conclusion that 80% of these predictions were con6.
firmed as known drug pair interactions, as shown in Table
For example, there is an interaction between the drug
rizatriptan (DB00953) and the drug Ramipril (DB00178).
Rizatriptan may decrease the antihypertensive activities of
Ramipril. Reduce the antihypertensive effect of ramipril,
which is difficult to achieve the effect expected by doctors.
There is also an interaction between the drug temozolomide (DB00853) and the drug clopidogrel (DB00758). We
know that bone marrow inhibitors reduce the production
of platelets in the bone marrow. This effect will lead to


Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188 185


**Table 7** The interactions related to breast neoplasms in the top 10 predicted scores


Rank ID-A ID-B Evidence Evidence and interaction relationship
Name-A Name-B


1 **DB00398** DB00563 DrugBank The risk or severity of adverse effects can be increased
**Sorafenib** Methotrexate


2 **DB01248** DB00864 DrugBank The serum concentration of Tacrolimus can be increased
**Docetaxel** Tacrolimus


3 **DB00990** **DB00635** _NA_ _NA_

**Exemestane** **Prednisone**


4 **DB00727** DB00426 _NA_ _NA_

**Nitroglycerin** Famciclovir

5 **DB00773** DB00993 DrugBank The risk or severity of adverse effects can be increased
**Etoposide** Azathioprine


6 **DB00482** **DB00877** Drugs.com Sirolimus can cause kidney problems, celecoxib can
**Celecoxib** **Sirolimus** affect kidneys, and combination use may increase the
risk

7 **DB00795** DB00334 _NA_ _NA_

**Sulfasalazine** Olanzapine


8 **DB00877** DB01050 DrugBank The risk or severity of angioedema can be increased
**Sirolimus** Ibuprofen


9 **DB01126** DB00196 DrugBank The metabolism of Dutasteride can be decreased
**Dutasteride** Fluconazole

10 **DB00317** DB00563 DrugBank Gefitinib may decrease the excretion rate of Methotrex**Gefitinib** Methotrexate ate which could result in a higher serum level


The breast neoplasms-related drugs are in bold



thrombocytopenia, so it will increase the risk of bleeding.
If drugs to prevent thrombosis events, such as antiplatelet
drugs, that is, clopidogrel, are used at the same time, the
risk of abnormal bleeding is likely to occur.
To further study the impact of our model on specific diseases, take breast neoplasms as an example, we
obtain drugs that have therapeutic effects on breast neoplasms from CTD [50]. In our data set, the number of
breast neoplasms related drugs is 58, such as Vinblastine
(DB00570), Mitoxantrone (DB01204), etc. Then we predicted the related unknown DDI. The prediction results

are shown in Table 7.

We can see that in the prediction results of breast neoplasms related DDI, 7 of the top 10 DDIs have been confirmed to have interactions in DrugBank or Drugs.com.
In particular, there are two pairs of DDIs, each pair consisting of two drugs related to breast neoplasms. We mark

in bold in Table 7
. One pair has been verified. Sirolimus
(DB00877) may cause kidney problems, and combining it
with other medications that can also affect the kidney such
as celecoxib (DB00482) may increase that risk. Therefore,
when prescribing drugs for patients with breast neoplasms,
doctors need to consider the dosage and duration of these
two drugs to avoid kidney problems caused by DDI as
much as possible.
The other three DDIs, Exemestane (DB00990) and
Prednisone (DB00635), Nitroglycerin (DB00727) and
Famciclovir (DB00426), Sulfasalazine (DB00795) and



Olanzapine (DB00334), deserve to be further verified
by traditional laboratory methods, because the scores
obtained from their prediction by our method are also
high.
For other DDI, because many patients may have multiple diseases at the same time, doctors should also consider
drug safety in clinical practice. If a patient suffers from
rheumatoid arthritis and breast neoplasms at the same
time, even though (DB00398) can be used to treat breast
neoplasms and methotrexate (DB00563) can be used to
treat rheumatoid arthritis, doctors should try their best to
avoid using them at the same time. Because methotrexate
may cause liver problems, and using it with other medications that can also affect the liver such as sorafenib may
increase that risk.

Through the experiment in this section, we further
verified the effectiveness and practicability of our model.
It also proves that this method can be transformed into
specific and substantial clinical resources and applied to
specific clinical diseases or cases.


**4 Discussion and Conclusions**


This article focuses on MSResG for predicting DDI. This

model takes into account the multi-source features fusion of

drugs, integrates the drug similarity networks corresponding
to multiple features into a comprehensive similarity network,

## 1 3


186 Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188



and combines them with the DDI network into a heterogeneous network, and then encodes and decodes them by means
of the GAE based on ResGCN to achieve DDI prediction.
Comparative experiments show that our model has high
performance.
In our method, known DDI is considered as a positive
sample and unknown DDI is considered as a negative sample. Theoretically, a clear DDI free relationship is a negative sample. Theoretically, a clear DDI free relationship is
a negative sample. However, due to the huge data of drug
pairs, we cannot clearly determine that there is no DDI
between drugs, so we choose unknown DDI as a negative
sample. Like other deep learning methods to predict DDI,
this negative sample selection method may affect the experimental results, but the case study confirms that this effect
is acceptable.
In fact, our model has some limitations. When MSResG
fuses the drug similarity of different features, it adopts a simple
mean value method for fusion, but this fusion method does
not consider the importance of different features. In addition,
in this model, we use DDI networks and drug similarity networks to build heterogeneous networks and then input it into
GAE. At this time, heterogeneous networks are sparse, which
will case some evaluation performance value is lower such
as Recall, even though this does not mean that our model has
poor performance.
In the future, we will further consider other feature information of the drug, such as side-effects, transporters and so
on. We will also consider using integrated methods to predict
DDI and pay attention to the drug features as comprehensively
and reasonably as possible, which can facilitate the further
improvement of the prediction performance. We will also
consider fusing the similarity matrix of drugs with different
attention scores, no longer considering the role of drug features
equally, but using more useful features.


**Funding** This work was supported by the National Natural Science
Foundation of China under Grants 62272288, 61972451, 61902230
and U22A2041, and the Shenzhen Science and Technology Program
under Grant KQTD20200820113106007.


**Declarations**


**Conflict of Interest** The authors declare that they have no conflict of
interest.


**References**


1. Foucquier J, Guedj M (2015) Analysis of drug combinations: current methodological landscape. Pharmacol Res Perspect 3:e00149.
[https://​doi.​org/​10.​1002/​prp2.​149](https://doi.org/10.1002/prp2.149)
2. Kusuhara H (2014) How far should we go? Perspective of drugdrug interaction studies in drug development. Drug Metab
Pharmacokinet 29:227–228. [https://​doi.​org/​10.​2133/​dmpk.​](https://doi.org/10.2133/dmpk.DMPK-14-PF-903)
[DMPK-​14-​PF-​903](https://doi.org/10.2133/dmpk.DMPK-14-PF-903)

## 1 3



3. Wishart DS, Feunang YD, Guo AC et al (2018) DrugBank 5.0: a
major update to the DrugBank database for 2018. Nucleic Acids
[Res 46:D1074–D1082. https://​doi.​org/​10.​1093/​nar/​gkx10​37](https://doi.org/10.1093/nar/gkx1037)
4. Onakpoya IJ, Heneghan CJ, Aronson JK (2016) Post-marketing
withdrawal of 462 medicinal products because of adverse drug
reactions: a systematic review of the world literature. BMC Med
[14:10. https://​doi.​org/​10.​1186/​s12916-​016-​0553-2](https://doi.org/10.1186/s12916-016-0553-2)
5. Qiu Y, Zhang Y, Deng Y et al (2022) A comprehensive review of
computational methods for drug-drug interaction detection. IEEE/
[ACM Trans Comput Biol Bioinf 19:1968–1985. https://​doi.​org/​](https://doi.org/10.1109/TCBB.2021.3081268)
[10.​1109/​TCBB.​2021.​30812​68](https://doi.org/10.1109/TCBB.2021.3081268)

6. He H, Chen G, Yu-Chian Chen C (2022) 3DGT-DDI: 3D graph
and text based neural network for drug-drug interaction predic[tion. Brief Bioinform 23:bbac134. https://​doi.​org/​10.​1093/​bib/​](https://doi.org/10.1093/bib/bbac134)
[bbac1​34](https://doi.org/10.1093/bib/bbac134)

7. Yan C, Duan G, Zhang Y et al (2022) Predicting drug-drug
interactions based on integrated similarity and semi-supervised
learning. IEEE/ACM Trans Comput Biol Bioinform 19:168–179.
[https://​doi.​org/​10.​1109/​TCBB.​2020.​29880​18](https://doi.org/10.1109/TCBB.2020.2988018)
8. Bag S, Kumar SK, Tiwari MK (2019) An efficient recommendation generation using relevant Jaccard similarity. Inf Sci 483:53–
[64. https://​doi.​org/​10.​1016/j.​ins.​2019.​01.​023](https://doi.org/10.1016/j.ins.2019.01.023)
9. Xia P, Zhang L, Li F (2015) Learning similarity with cosine simi[larity ensemble. Inf Sci 307:39–52. https://​doi.​org/​10.​1016/j.​ins.​](https://doi.org/10.1016/j.ins.2015.02.024)
[2015.​02.​024](https://doi.org/10.1016/j.ins.2015.02.024)

10. Cheng F, Zhao Z (2014) Machine learning-based prediction of
drug–drug interactions by integrating drug phenotypic, therapeutic, chemical, and genomic properties. J Am Med Inform Assoc
[21:e278–e286. https://​doi.​org/​10.​1136/​amiaj​nl-​2013-​002512](https://doi.org/10.1136/amiajnl-2013-002512)
11. Shelhamer E, Long J, Darrell T (2017) Fully convolutional networks for semantic segmentation. IEEE Trans Pattern Anal Mach
[Intell 39:640–651. https://​doi.​org/​10.​1109/​CVPR.​2015.​72989​65](https://doi.org/10.1109/CVPR.2015.7298965)
12. Lipton ZC, Berkowitz J, Elkan C (2015) A critical review of
[recurrent neural networks for sequence learning. https://​doi.​org/​](https://doi.org/10.48550/arXiv.1506.00019)
[10.​48550/​arXiv.​1506.​00019](https://doi.org/10.48550/arXiv.1506.00019)

13. Liu J, Lei X, Zhang Y, Pan Y (2023) The prediction of molecular
toxicity based on BiGRU and GraphSAGE. Comput Biol Med.
[https://​doi.​org/​10.​1016/j.​compb​iomed.​2022.​106524](https://doi.org/10.1016/j.compbiomed.2022.106524)
14. Liu X, Yang M (2022) Research on conversational machine reading comprehension based on dynamic graph neural network. J
[Integr Technol 11:67–78. https://​doi.​org/​10.​12146/j.​issn.​2095-​](https://doi.org/10.12146/j.issn.2095-3135.20211122001)
[3135.​20211​122001](https://doi.org/10.12146/j.issn.2095-3135.20211122001)

15. Ryu JY, Kim HU, Lee SY (2018) Deep learning improves prediction of drug-drug and drug-food interactions. Proc Natl Acad
[Sci USA 115:E4304–E4311. https://​doi.​org/​10.​1073/​pnas.​18032​](https://doi.org/10.1073/pnas.1803294115)
[94115](https://doi.org/10.1073/pnas.1803294115)

16. Feng Y-H, Zhang S-W, Shi J-Y (2020) DPDDI: a deep predictor
[for drug-drug interactions. BMC Bioinform 21:419. https://​doi.​](https://doi.org/10.1186/s12859-020-03724-x)
[org/​10.​1186/​s12859-​020-​03724-x](https://doi.org/10.1186/s12859-020-03724-x)
17. Lin X, Quan Z, Wang Z-J, et al (2020) KGNN: knowledge graph
neural network for drug-drug interaction prediction. In: TwentyNinth International Joint Conference on Artificial Intelligence, pp
[2739–2745. https://​doi.​org/​10.​24963/​ijcai.​2020/​380](https://doi.org/10.24963/ijcai.2020/380)
18. Han X, Xie R, Li X, Li J (2022) SmileGNN: drug-drug interaction
prediction based on the SMILES and graph neural network. Life
[(Basel) 12:319. https://​doi.​org/​10.​3390/​life1​20203​19](https://doi.org/10.3390/life12020319)
19. Li G, Müller M, Thabet A, Ghanem B (2019) DeepGCNs: can
GCNs go as deep as CNNs? International Conference on Com[puter Vision 2019. https://​doi.​org/​10.​48550/​arXiv.​1904.​03751](https://doi.org/10.48550/arXiv.1904.03751)
20. Wang F, Lei X, Liao B, Wu F-X (2022) Predicting drug-drug
interactions by graph convolutional network with multi-kernel.
[Brief Bioinform 23:bbab511. https://​doi.​org/​10.​1093/​bib/​bbab5​](https://doi.org/10.1093/bib/bbab511)
[11](https://doi.org/10.1093/bib/bbab511)

21. Kipf TN, Welling M (2016) Variational graph auto-encoders. In:
Conference and Workshop on Neural Information Processing Sys[tems. https://​doi.​org/​10.​48550/​arXiv.​1611.​07308](https://doi.org/10.48550/arXiv.1611.07308)


Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188 187



22. Zhang W, Chen Y, Liu F et al (2017) Predicting potential drugdrug interactions by integrating chemical, biological, phenotypic
[and network data. BMC Bioinform 18:18. https://​doi.​org/​10.​1186/​](https://doi.org/10.1186/s12859-016-1415-9)
[s12859-​016-​1415-9](https://doi.org/10.1186/s12859-016-1415-9)

23. Wan F, Hong L, Xiao A et al (2019) NeoDTI: neural integration
of neighbor information from a heterogeneous network for discovering new drug-target interactions. Bioinformatics 35:104–111.
[https://​doi.​org/​10.​1093/​bioin​forma​tics/​bty543](https://doi.org/10.1093/bioinformatics/bty543)
24. Xie J, Zhao C, Ouyang J et al (2022) TP-DDI: a two-pathway deep
neural network for drug-drug interaction prediction. Interdiscip
[Sci 14:895–905. https://​doi.​org/​10.​1007/​s12539-​022-​00524-0](https://doi.org/10.1007/s12539-022-00524-0)
25. Schwarz K, Allam A, Perez Gonzalez NA, Krauthammer M
(2021) AttentionDDI: Siamese attention-based deep learning
method for drug-drug interaction predictions. BMC Bioinform
[22:412. https://​doi.​org/​10.​1186/​s12859-​021-​04325-y](https://doi.org/10.1186/s12859-021-04325-y)
26. Kanehisa M, Goto S (2000) KEGG: Kyoto encyclopedia of genes
[and genomes. Nucl Acids Res 28:27–30. https://​doi.​org/​10.​1093/​](https://doi.org/10.1093/nar/28.1.27)
[nar/​28.1.​27](https://doi.org/10.1093/nar/28.1.27)

27. Kim S, Thiessen PA, Bolton EE et al (2016) PubChem substance
[and compound databases. Nucl Acids Res 44:D1202-1213. https://​](https://doi.org/10.1093/nar/gkv951)
[doi.​org/​10.​1093/​nar/​gkv951](https://doi.org/10.1093/nar/gkv951)
28. Lei S, Lei X, Liu L (2022) Drug repositioning based on heterogeneous networks and variational graph autoencoders. Front Phar[macol 13:5431. https://​doi.​org/​10.​3389/​fphar.​2022.​10566​05](https://doi.org/10.3389/fphar.2022.1056605)
29. Zhang Y, Lei X, Pan Y, Wu F-X (2022) Drug repositioning with
GraphSAGE and clustering constraints based on drug and disease
[networks. Front Pharmacol 13:872785. https://​doi.​org/​10.​3389/​](https://doi.org/10.3389/fphar.2022.872785)
[fphar.​2022.​872785](https://doi.org/10.3389/fphar.2022.872785)
30. Wang F, Ding Y, Lei X et al (2021) Human protein complex-based
drug signatures for personalized cancer medicine. IEEE J Biomed
[Health Inform 25:4079–4088. https://​doi.​org/​10.​1109/​JBHI.​2021.​](https://doi.org/10.1109/JBHI.2021.3120933)
[31209​33](https://doi.org/10.1109/JBHI.2021.3120933)

31. Lahitani AR, Permanasari AE, Setiawan NA (2016) Cosine similarity to determine similarity measure: study case in online essay
assessment. In: 2016 4th International Conference on Cyber and
[IT Service Management, 1–6. https://​doi.​org/​10.​1109/​CITSM.​](https://doi.org/10.1109/CITSM.2016.7577578)
[2016.​75775​78](https://doi.org/10.1109/CITSM.2016.7577578)
32. Kipf TN, Welling M (2017) Semi-supervised classification with
[graph convolutional networks. https://​doi.​org/​10.​48550/​arXiv.​](https://doi.org/10.48550/arXiv.1609.02907)
[1609.​02907](https://doi.org/10.48550/arXiv.1609.02907)

33. An J, Guo L, Liu W et al (2021) IGAGCN: information geometry
and attention-based spatiotemporal graph convolutional networks
[for traffic flow prediction. Neural Netw 143:355–367. https://​doi.​](https://doi.org/10.1016/j.neunet.2021.05.035)
[org/​10.​1016/j.​neunet.​2021.​05.​035](https://doi.org/10.1016/j.neunet.2021.05.035)
34. Zhu Y, Ma J, Yuan C, Zhu X (2022) Interpretable learning based
dynamic graph convolutional networks for Alzheimer’s disease
[analysis. Inform Fus 77:53–61. https://​doi.​org/​10.​1016/j.​inffus.​](https://doi.org/10.1016/j.inffus.2021.07.013)
[2021.​07.​013](https://doi.org/10.1016/j.inffus.2021.07.013)

35. Chipofya M, Tayara H, Chong KT (2021) Drug therapeutic-use
class prediction and repurposing using graph convolutional net[works. Pharmaceutics 13:1906. https://​doi.​org/​10.​3390/​pharm​](https://doi.org/10.3390/pharmaceutics13111906)
[aceut​ics13​111906](https://doi.org/10.3390/pharmaceutics13111906)

36. Ding Y, Lei X, Liao B, Wu F-X (2022) Predicting miRNA-disease
associations based on multi-view variational graph auto-encoder
with matrix factorization. IEEE J Biomed Health Inform 26:446–

[457. https://​doi.​org/​10.​1109/​JBHI.​2021.​30883​42](https://doi.org/10.1109/JBHI.2021.3088342)
37. Zhang T, Gu J, Wang Z et al (2022) Protein subcellular localization prediction model based on graph convolutional network.



[Interdiscip Sci Comput Life Sci 14:937–946. https://​doi.​org/​10.​](https://doi.org/10.1007/s12539-022-00529-9)
[1007/​s12539-​022-​00529-9](https://doi.org/10.1007/s12539-022-00529-9)

38. He K, Zhang X, Ren S, Sun J (2016) Deep residual learning
for image recognition. IEEE Conf Comput Vis Pattern Recogn
[(CVPR) 2016:770–778. https://​doi.​org/​10.​1109/​CVPR.​2016.​90](https://doi.org/10.1109/CVPR.2016.90)
39. Kingma DP, Ba J (2014) Adam: a method for stochastic optimiza[tion. https://​doi.​org/​10.​48550/​arXiv.​1412.​6980](https://doi.org/10.48550/arXiv.1412.6980)
40. Srivastava N, Hinton G, Krizhevsky A, et al (2014) Dropout: a
simple way to prevent neural networks from overfitting. J Mach
[Learn Res 15:1929–1958. http://​jmlr.​org/​papers/​v15/​sriva​stava​](http://jmlr.org/papers/v15/srivastava14a.html)
[14a.​html](http://jmlr.org/papers/v15/srivastava14a.html)

41. Smith LN (2017) Cyclical learning rates for training neural networks. In: 2017 IEEE Winter Conference on Applications of
[Computer Vision (WACV) 464–472. https://​doi.​org/​10.​48550/​](https://doi.org/10.48550/arXiv.1506.01186)
[arXiv.​1506.​01186](https://doi.org/10.48550/arXiv.1506.01186)
42. Chiang W-L, Liu X, Si S, et al (2019) Cluster-GCN: an efficient
algorithm for training deep and large graph convolutional networks. In: Proceedings of the 25th ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining 257–266.
[https://​doi.​org/​10.​48550/​arXiv.​1905.​07953](https://doi.org/10.48550/arXiv.1905.07953)
43. Pei Y, Huang T, Ipenburg W, Pechenizkiy M (2022) ResGCN:
attention-based deep residual modeling for anomaly detection on
[attributed networks. Mach Learn 111:1–23. https://​doi.​org/​10.​](https://doi.org/10.1109/DSAA53316.2021.9564233)
[1109/​DSAA5​3316.​2021.​95642​33](https://doi.org/10.1109/DSAA53316.2021.9564233)

44. Deng Y, Xu X, Qiu Y et al (2020) A multimodal deep learning
framework for predicting drug–drug interaction events. Bioinfor[matics 36:4316–4322. https://​doi.​org/​10.​1093/​bioin​forma​tics/​](https://doi.org/10.1093/bioinformatics/btaa501)
[btaa5​01](https://doi.org/10.1093/bioinformatics/btaa501)

45. Zhang P, Wang F, Hu J, Sorrentino R (2015) Label propagation
prediction of drug-drug interactions based on clinical side effects.
[Sci Rep 5:12339. https://​doi.​org/​10.​1038/​srep1​2339](https://doi.org/10.1038/srep12339)
46. Abeywickrama T, Cheema M, Taniar D (2016) k-nearest neighbors on road networks: a journey in experimentation and in-mem[ory implementation. Proc VLDB Endowm 9:492–503. https://​doi.​](https://doi.org/10.14778/2904121.2904125)
[org/​10.​14778/​29041​21.​29041​25](https://doi.org/10.14778/2904121.2904125)
47. Lü L, Pan L, Zhou T et al (2015) Toward link predictability of
complex networks. Proc Natl Acad Sci USA 112:2325–2330.
[https://​doi.​org/​10.​1073/​pnas.​14246​44112](https://doi.org/10.1073/pnas.1424644112)
48. Vilar S, Harpaz R, Uriarte E et al (2012) Drug-drug interaction through molecular structure similarity analysis. J Am Med
Inform Assoc 19:1066–1074. [https://​doi.​org/​10.​1136/​amiaj​](https://doi.org/10.1136/amiajnl-2012-000935)
[nl-​2012-​000935](https://doi.org/10.1136/amiajnl-2012-000935)

49. Vilar S, Uriarte E, Santana L, Tatonetti N (2013) Detection of
drug-drug interactions by modeling interaction profile finger[prints. PLoS ONE 8:e58321. https://​doi.​org/​10.​1371/​journ​al.​pone.​](https://doi.org/10.1371/journal.pone.0058321)
[00583​21](https://doi.org/10.1371/journal.pone.0058321)

50. Davis AP, Grondin CJ, Johnson RJ et al (2021) Comparative
toxicogenomics database (CTD): update 2021. Nucl Acids Res
[49:D1138–D1143. https://​doi.​org/​10.​1093/​nar/​gkaa8​91](https://doi.org/10.1093/nar/gkaa891)


Springer Nature or its licensor (e.g. a society or other partner) holds
exclusive rights to this article under a publishing agreement with the
author(s) or other rightsholder(s); author self-archiving of the accepted
manuscript version of this article is solely governed by the terms of
such publishing agreement and applicable law.

## 1 3


188 Interdisciplinary Sciences: Computational Life Sciences (2023) 15:171–188


**Authors and Affiliations**


**Lin Guo** **[1]** **· Xiujuan Lei** **[1]** **[· Ming Chen](http://orcid.org/0000-0002-9901-1732)** **[2]** **· Yi Pan** **[3]**




- Xiujuan Lei
xjlei@snnu.edu.cn


Lin Guo

20212743@snnu.edu.cn


Ming Chen
chenming@hunnu.edu.cn


Yi Pan

yi.pan@siat.ac.cn

## 1 3



1 School of Computer Science, Shaanxi Normal University,
Xi’an 710119, China


2 College of Information Science and Engineering, Hunan
Normal University, Changsha 410081, China


3 Faculty of Computer Science and Control Engineering,
Shenzhen Institute of Advanced Technology Chinese
Academy of Sciences, Shenzhen 518055, China


