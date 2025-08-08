pubs.acs.org/jcim Article

### **AttenSyn: An Attention-Based Deep Graph Neural Network for** **Anticancer Synergistic Drug Combination Prediction**

###### Tianshuo Wang, Ruheng Wang, and Leyi Wei*

##### ACCESS Metrics & More Article Recommendations * sı Supporting Information


# ■ [INTRODUCTION] made great breakthroughs in drug-related interaction pre
diction. [4][−][6] Also, various computational methods have been

In recent decades, the number of available anticancer drugs has
been increasing rapidly, accompanied by an evolving under- proposed for drug combination prediction in the past decade,
standing of the biological complexity of malignant tumors. which greatly reduces the screening space of drug combinaHowever, in cancer, multiple cellular mechanisms are often tions. For example, in the streptozocin-induced neuropathic
altered in the cell; therefore, treating them with a single drug pain model in mice, the maximal antiallodynic effect of a new
and focusing on a single target is usually ineffective. Compared derivative of dihydrofuran-2-one (LPP1) used in combination
to the traditional treatment mode of “single disease, single with pregabalin has been successfully predicted by using the
drug, and single target”, combination therapy has the potential support vector machine (SVM) and random forest (RF)
to increase treatment efficacy, reduce host toxicity and adverse algorithms. [7][,][8] Moreover, Liu et al. [9] used the features extracted


−

side effects, and overcome drug resistance. Drug combinations from the random walk algorithm with a restart on the drug
are widely used to treat a variety of complex diseases, such as protein heterogeneous network and a gradient tree boosting
hypertension, [1] infectious diseases, [2] and cancer. [3] However, classifier to predict new drug combinations. Later, Pivetta et
some antagonistic effects and even severe adverse drug−drug al. [10] employed an artificial neural-network-based model to
interactions occur when using some drug combinations, which predict the synergism of anticancer drugs. In addition, Zhang
not only are ineffective at enhancing the curative effect but also and Yan [11] considered pharmacological data and applied fieldthreaten the patient’s health. Therefore, it is crucial to

aware factorization machines to analyze and predict potential

accurately discover synergistic drug combinations for specific

synergistic drug combinations. Apart from these methods,

diseases.
Traditional experimental methods to screen the combinations of synergistic antitumor drugs are very challenging in
terms of time, efficiency, and cost, which are far from meeting [matics](https://pubs.acs.org/toc/jcisd8/64/7?ref=pdf)
the urgent need for anticancer drugs. Even high-throughput Received: May 10, 2023
screens are not feasible due to the vast number of drug Published: August 11, 2023
combinations. Thanks to the rapid development of related
databases and machine-learning technologies, researchers have



made great breakthroughs in drug-related interaction prediction. [4][−][6] Also, various computational methods have been
proposed for drug combination prediction in the past decade,
which greatly reduces the screening space of drug combinations. For example, in the streptozocin-induced neuropathic
pain model in mice, the maximal antiallodynic effect of a new
derivative of dihydrofuran-2-one (LPP1) used in combination
with pregabalin has been successfully predicted by using the
support vector machine (SVM) and random forest (RF)
algorithms. [7][,][8] Moreover, Liu et al. [9] used the features extracted

−
from the random walk algorithm with a restart on the drug
protein heterogeneous network and a gradient tree boosting
classifier to predict new drug combinations. Later, Pivetta et
al. [10] employed an artificial neural-network-based model to
predict the synergism of anticancer drugs. In addition, Zhang
and Yan [11] considered pharmacological data and applied fieldaware factorization machines to analyze and predict potential
synergistic drug combinations. Apart from these methods,







© 2023 American Chemical Society



**2854**



[https://doi.org/10.1021/acs.jcim.3c00709](https://doi.org/10.1021/acs.jcim.3c00709?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 2854−2862


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


Figure 1. The overall architecture of our proposed method. Given a pair of drugs and the corresponding cell-line type, we first transform the
SMILES profiles of the two drugs into molecular graphs with initial node features and graph structures. Also, we add the cell-line features obtained
from CCLE into the feature of each node in the drug molecular graphs. Then we employ the graph-based drug-embedding module with several
GCN and LSTM layers to learn better representations of each node and then use the attention-based pooling module to obtain the graph-level
features. Finally, we exploit the prediction module to integrate the representations of two drugs with features of the cell line and make the
prediction of synergistic drug combinations.



Janizek et al. [12] proposed TreeCombo, a machine-learningbased model using the XGBoost algorithm to predict the
synergistic score of drug pairs. Chen et al. developed NLLSS, a
novel algorithm termed “Network-based Laplacian regularized
Least Square Synergistic” (NLLSS) drug combination
prediction to predict potential antifungal synergistic drug
combinations by integrating different kinds of information. [13]

The above methods have boosted the use of machine-learning
solutions for drug combination prediction. However, these
traditional machine-learning workflows have some drawbacks.
For instance, training a good model normally requires strong
professional knowledge of handcrafted feature engineering and
machine-learning algorithms, limiting their usability in real
applications to some extent. Moreover, they cannot support
fast large-scale prediction due to their relatively low computational ability.
Recently, with the rapid development of deep learning and
the release of large-scale drug combination datasets, it has been
possible to predict drug combinations using deep-learning
methods. For example, Preuer et al. [14] proposed DeepSynergy,
which combined the chemical descriptors of drugs and gene
expression of cell lines to predict the potential drug synergies.
Liu and Xie [15] proposed TranSynergy, considered the network

−
information such as gene gene interaction networks and

−
drug target associations, and applied the transformer
architecture to the synergistic prediction of drugs. Moreover,
Kuru et al. [16] trained two parallel subnetworks to learn drugspecific representation on a particular cell line and make
predictions. Later, Su et al. [17] proposed SRDFM, integrating a
factorization machine component with a deep neural network
for single-drug and synergistic drug combination recommendation. Furthermore, Lin et al. [18] proposed EC-DFR, which
uses physicochemical properties, molecular fingerprints, cellline-specific drug-induced gene expression profiles as features,
and a cascade-based deep forest model for drug combination



prediction. Li et al. proposed SNRMPACDC, using siamese
network and random matrix projection for anticancer drug
combination prediction. [19] These methods make the use of
deep learning more convenient for synergistic drug combination prediction to some extent. However, there remain some
drawbacks that need to be addressed. First, most of the existing
methods cannot meet the high demand of the research
community since they possess relatively lower performances.
Second, few of them have an interpretation of their models and
detect what their models learn during the training process.
In recent years, graph neural networks (GNN) have
achieved remarkable success in many real-world applications
such as drug discovery. Many studies have used graph neural
networks on the molecular graph to extract molecular features
for drug combination prediction. For example, Wang et al. [20]

proposed a GNN-based deep-learning network called
DeepDDS, applying a graph convolutional network (GCN)
and graph attention network (GAT) to extract the drugembedding vectors for identifying drug combinations. Jin et
al. [21] proposed ComboNet, which employed a directional
message-passing neural network (DMPNN) to learn a
continuous representation of a molecule for identifying
synergistic drug combinations for treating COVID-19. Additionally, Hu et al. [22] developed a deep graph neural network
model named DTSyn and used multihead attention mechanism to identify novel drug combinations. Although graph
neural networks have demonstrated their ability to solve drug
synergy prediction and achieved success to some extent in
terms of performance improvement, there still exist some

−
limitations. First, although some adverse drug drug interaction methods considered the interactive information
between drug pairs, [23][,][24] existing synergistic drug combination
prediction methods focus on the information extraction of a
single drug but ignore the importance of the interactive
information between drug pairs. Second, most of them treat


**2855** [https://doi.org/10.1021/acs.jcim.3c00709](https://doi.org/10.1021/acs.jcim.3c00709?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 2854−2862


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


Figure 2. The overall computational steps of attention-based pooling. The graph-level representation is calculated by the weighted sum of all nodes’
embeddings according to the attention scores.



each substructure in a drug molecule as equally important and
cannot detect important ones for the prediction of synergistic
drug combinations.
To tackle the above problems, in this study, we proposed an
attention-based deep graph neural network named AttenSyn to
predict the synergistic effect of drug combinations. Specifically,
our proposed model incorporates several novel features as
follows. First, we employ a deep graph neural network to learn
and extract high-latent features automatically rather than using
manual feature profiles from handcrafted feature engineering.
Second, with the attention-based pooling module, we can not
only learn the interactive information between drug pairs but
also detect the important chemical substructures in drugs for
the identification of anticancer synergistic drug combinations.
Third, comparative results with the existing methods on the
benchmark dataset show that our proposed model outperforms
not only classical machine-learning methods but also deeplearning methods, demonstrating that our AttenSyn has great
potential to be a powerful and practically useful deep-learning
tool for anticancer synergistic drug combinations prediction.

# ■ [METHODS AND MATERIALS]


**Dataset.** To compare the performance of our proposed
model with the state-of-the-art methods, we collected the drug
combination dataset constructed by O’Neil et al. as our
benchmark dataset. [25] The dataset contains 23,052 triplets,
where each triplet comprises two drugs and a cancer cell line. [22]

There are 39 cancer cell lines and 38 unique drugs in the
dataset, and these drugs are composed of 24 FDA-approved
drugs and 14 experimental drugs. [14] The synergy score of each
drug pair was calculated by using the Combenefit tool. [26]

According to a previous study, [22] we selected 10 as a threshold

−
to classify the drug pair cell-line triplets. The triplets with a



synergistic score higher than 10 are recognized as positive, and
those less than 0 are seen as negative. After preprocessing the
data, we obtained 13,243 unique triplets, consisting of 38 drugs
and 31 cell lines. Moreover, the SMILES (Simplified Molecular
Input Line Entry System) [27] of drugs is obtained from
DrugBank. [28]

The gene expression data of cancer cell lines are obtained
from Cancer Cell Line Encyclopedia (CCLE), [29] which is an
independent project that makes the effort to characterize
genomes, mRNA expression, and anticancer drug dose
responses across cancer cell lines. The expression data is
normalized through TPM (Transcripts Per Million) based on
the genome-wide read counts matrix.
**Framework of AttenSyn.** In this section, we introduce the
details of our proposed AttenSyn. The overall architecture of
AttenSyn is shown in Figure 1. This network architecture
mainly includes three parts: (1) Graph-based drug-embedding
module, (2) attention-based pooling module, and (3)
prediction module. In the graph-based drug-embedding
module, first, the drug SMILES strings are transformed into
molecular graphs, and simultaneously the cell-line features
obtained from CCLE [29] are added to the feature matrices of the
drug molecules. Then, several graph convolution network
(GCN) models and LSTM models are employed to extract the
multiresolution features of molecular graphs. After that, we use
the attention-based pooling module to learn the interactive
information between drug pairs and strengthen the representations of drug pairs. Finally, in the prediction module, we
concatenated the representations of drug pairs and features of
cell lines and fed them into a fully connected neural network to
predict the synergy of drug pairs in certain cell lines.
**Graph-Based Drug-Embedding Module.** By using the
open-source python package Rdkit, [30] we can convert the


**2856** [https://doi.org/10.1021/acs.jcim.3c00709](https://doi.org/10.1021/acs.jcim.3c00709?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 2854−2862


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


[̃]

[̃] [̃] [̃]

[̃]



SMILES string into a molecular graph where the nodes are
atoms and the edges are chemical bonds. Thus, we can use a
graph G = (V, E) to represent a drug molecule, where V and E
are the set of nodes and the set of edges, respectively. To
aggregate molecular graphs with cell-line information, we
simply add the cell-line vector to the nodes’ features as follows:


0
_h_ _i_ = _x_ _i_ + MLP( _R_ cell line ) (1)


where _x_ _i_ denotes the feature vector of node _i_ and _R_ cell line is the
cell-line vector obtained by CCLE.
In order to obtain the representation of chemical
substructures, we employ a GNN module that uses a chemical
graph structure as the input and updates vector embeddings of
each atom from its neighbors. Therefore, the updated feature
vector of each atom can represent chemical substructures. The
GCN operator we used can be formulated as follows:


_H_ _l_ +1 = ( _D_ 1/2 _AD_ 1/2 _H W_ _l_ _l_ ) (2)


where _A_ [̃] = _A_ + _I_, _I_ is the identity matrix, and _A_ is the adjacency
matrix. _D_ [̃] is the degree matrix of _A_ [̃], which is calculated by _D_ [̃] _ii_ =
∑ _j_ _A_ [̃] _ij_ . _W_ _[l]_ is a layer-specific trainable weight matrix. _H_ _[l]_
represents the learned representations by the _l_ th layer and
_H_ [0] = [ _h_ 01, _h_ 02, _h_ 03, ..., _h_ 0 _i_, ...]. _σ_ denotes the nonlinear activation
function.
In order to get multiresolution information on molecular
graphs, inspired by MR-GNN, [31] we use LSTM to extract the
graph’s multiresolution local features. Several LSTM models
are used to aggregate the features of multiple GCN layers.
Specifically, the LSTM sequentially receives the output of each
GNN layer with a receptive field from small to large as the
input. The LSTM can be formulated as


_l_ +1 _l_ _l_
_S_ = LSTM(, _S_ _H_ ) (3)


where _S_ _[l]_ [+1] is the ( _l_ +1)th hidden vector of LSTM.
**Attention-Based Pooling Module.** To improve the
performance of the proposed model, we designed an
attention-mechanism-based pooling to learn better interactive
information on drug pairs and strengthen the representations
of drugs. The use of attention-based pooling helps the
proposed model consider which substructures in the chemical
are more important for the prediction of synergistic drug
combinations. As shown in Figure 2, the attention-based
pooling module was used to assign each substructure of the
drug a score, and weighted sum all nodes’ embeddings to get
graph-level representations. By using our designed attentionbased pooling module, we can not only get the interactive
information between drug pairs but also identify the important
chemical substructures of drugs. We calculate the attention
scores as follows:


_l_ _l_ T
_A_ _x_ = tanh( _H W H W_ _x_ _k_ ( _y_ _q_ ) ) (4)


_l_ _l_ T
_A_ _y_ = tanh( _H W H W_ _y_ _k_ ( _x_ _q_ ) ) (5)



where _H_ _lx_ and _H_ _ly_ are the graph-embedding matrices of drug _x_
and drug _y_ respectively, taken from the last GCN layer. And _a_ _x_,
_a_ _y_ are the attention scores for the drug pair. Then we weighted
sum all the nodes’ vectors according to the attention scores to
get the final graph-level representations:


_g_ _x_ = multiply( _a_ _x_, _H W_ _x_ _v_ ) (8)


_g_ _y_ = multiply( _a_ _y_, _H W_ _y_ _v_ ) (9)


where _H_ _x_ and _H_ _y_ are the embedding matrices of the drug pair
in the last GCN layer. For the output of LSTM, we calculate
the final representation in the same way, and the formula is as
follows:


_s_ _x_, _s_ _y_ = attention based pooling( _S_ _x_, _S_ _y_ ) (10)


where _S_ _x_ and _S_ _y_ denote the embedding matrices of the drug
pair in the last LSTM layer and the _s_ _x_ and _s_ _y_ are the graph-level

[̃] feature vectors of the two drugs captured by LSTM.

[̃] [̃] [̃] **Prediction Module.** In this module, we first integrate all

[̃] the features of two drugs with the feature vector of the cell line

and then make predictions using a multilayer perceptron
(MLP). Specifically, we concatenate the features of the drug
pair from GCN and LSTM _g_ _x_, _g_ _y_, _s_ _x_, _s_ _y_ and cell-line feature vector
_R_ cell line, and then the MLP is used for classification.
## p i = softmax MLP ( ( g x g y s x s y MLP( R cell line ) ) ) (11)


where || is the concatenation operator.
Our model is optimized by minimizing the cross entropy
loss function:




[̃]

[̃] [̃] [̃]

[̃]


l

ooooooooooooooooooooooooooooo

m




[̃]

[̃] [̃] [̃]

[̃]


_N_




[̃]

[̃] [̃] [̃]

[̃]


1
cross entropy = _N_ [ · _y_ _i_ log( ) _p_ _i_ + (1 _y_ _i_ ) log(1· _p_ _i_ )




[̃]

[̃] [̃] [̃]

[̃]


= _N_ [ · _y_ _i_ log( ) _p_ _i_ + (1 _y_ _i_ ) log(1· _p_ _i_ )]

_i_ =1




[̃]

[̃] [̃] [̃]

[̃]


_i_




[̃]

[̃] [̃] [̃]

[̃]


=




[̃]

[̃] [̃] [̃]

[̃]


(12)


where _N_ is the total number of samples in the training set, _y_ _i_ is
the label of sample _i_, and _p_ _i_ denotes the probability of
identifying this drug pair to the synergistic combination by our
model.
**Metrics.** For the anticancer drug synergy prediction task,
we adopted metrics including the area under the receiver
operator characteristics curve (AUROC), the area under the

−
precision recall curve (AUPR), accuracy (ACC), balanced
accuracy (BACC), precision (PREC), true positive rate
(TPR), and the Cohen’s Kappa value (KAPPA). These
indicators are calculated by the following formula:




[̃]

[̃] [̃] [̃]

[̃]


TP
TPR =
TP + FN


TN
TNR =
TN + FP


TP
PREC =
TP + FP




[̃]

[̃] [̃] [̃]

[̃]


TP + TN
ACC =
TP + FN + TN + FP


TPR + TNR
BACC =
2




[̃]

[̃] [̃] [̃]

[̃]


i

jjjjjjj

k




[̃]

[̃] [̃] [̃]

[̃]


_M_




[̃]

[̃] [̃] [̃]

[̃]


y




[̃]

[̃] [̃] [̃]

[̃]


_a_ _x_ = softmax _A_

k _j_ =1




[̃]

[̃] [̃] [̃]

[̃]


=
_xi j_,


=1




[̃]

[̃] [̃] [̃]

[̃]


zzzzzzz

{ (6)




[̃]

[̃] [̃] [̃]

[̃]


{




[̃]

[̃] [̃] [̃]

[̃]


=

TN +




[̃]

[̃] [̃] [̃]

[̃]


=




[̃]

[̃] [̃] [̃]

[̃]


i

jjjjjjj

k




[̃]

[̃] [̃] [̃]

[̃]


=

TP +




[̃]

[̃] [̃] [̃]

[̃]


_N_




[̃]

[̃] [̃] [̃]

[̃]


y




[̃]

[̃] [̃] [̃]

[̃]


TNR =
TN + FP


TP
PREC =
TP + FP


KAPPA = _p_ o _p_ e

ooooooooooooooooooooooooooooon 1 _p_ e (13)




[̃]

[̃] [̃] [̃]

[̃]


_a_ = softmax _A_
_y_

k _j_ =1




[̃]

[̃] [̃] [̃]

[̃]


=
_yi j_,


=1




[̃]

[̃] [̃] [̃]

[̃]


zzzzzzz

{ (7)




[̃]

[̃] [̃] [̃]

[̃]


{




[̃]

[̃] [̃] [̃]

[̃]


_p_ o _p_ e




[̃]

[̃] [̃] [̃]

[̃]


_p_ e




[̃]

[̃] [̃] [̃]

[̃]


=




[̃]

[̃] [̃] [̃]

[̃]


n




[̃]

[̃] [̃] [̃]

[̃]


KAPPA =
1




[̃]

[̃] [̃] [̃]

[̃]


=




[̃]

[̃] [̃] [̃]

[̃]


**2857** [https://doi.org/10.1021/acs.jcim.3c00709](https://doi.org/10.1021/acs.jcim.3c00709?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 2854−2862


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


Table 1. Performances of Our Method and Existing Methods on Benchmark Datasets


Methods AUROC AUPR ACC BACC PREC TPR KAPPA


AttenSyn 0.92 ± 0.01 0.91 ± 0.01 0.84 ± 0.01 0.84 ± 0.02 0.83 ± 0.02 0.82 ± 0.03 0.67 ± 0.03

DTSyn 0.89 ± 0.01 0.87 ± 0.01 0.81 ± 0.01 0.81 ± 0.02 0.84 ± 0.02 0.74 ± 0.05 0.61 ± 0.03

MR-GNN 0.90 ± 0.01 0.90 ± 0.01 0.82 ± 0.01 0.82 ± 0.01 0.81 ± 0.02 0.80 ± 0.03 0.65 ± 0.02

DeepSynergy 0.72 ± 0.01 0.77 ± 0.03 0.72 ± 0.01 0.72 ± 0.01 0.73 ± 0.05 0.64 ± 0.02 0.43 ± 0.02

RF 0.74 ± 0.03 0.73 ± 0.03 0.67 ± 0.01 0.67 ± 0.02 0.70 ± 0.07 0.59 ± 0.03 0.35 ± 0.04

Adaboost 0.74 ± 0.02 0.72 ± 0.03 0.75 ± 0.02 0.66 ± 0.02 0.63 ± 0.08 0.69 ± 0.08 0.32 ± 0.04

SVM 0.68 ± 0.05 0.65 ± 0.06 0.62 ± 0.05 0.62 ± 0.05 0.59 ± 0.05 0.66 ± 0.06 0.25 ± 0.09

MLP 0.84 ± 0.01 0.82 ± 0.01 0.76 ± 0.01 0.75 ± 0.01 0.75 ± 0.01 0.71 ± 0.01 0.50 ± 0.02

Elastic net 0.68 ± 0.08 0.67 ± 0.07 0.63 ± 0.07 0.63 ± 0.07 0.61 ± 0.08 0.62 ± 0.07 0.27 ± 0.14


Figure 3. The performances of our method and the other two methods on the leave-tumor-out cross validation task. (a) AUROC, AUPR, and TPR
of all comparing methods on the leave-tumor-out cross validation task; (b) AUROC scores of our AttenSyn and the other two methods on each
tumor type.



Among these, TP (true positive), FN (false negative), TN
(true negative), and FP (false positive) represent the number
of identifying synergistic drug combinations as synergistic drug
combinations, identifying synergistic drug combinations as
antagonistic drug combinations, identifying antagonistic drug
combination as antagonistic drug combination, and identifying
antagonistic drug combination as synergistic drug combination
by the model, respectively. _p_ o is the empirical probability of
agreement on the label assigned to any sample, and _p_ e is the
expected agreement when both annotators assign labels
randomly.
ACC describes how the model differs across two classes and
is useful in binary classification. [32] BACC and KAPPA are two
indicators that take into account the predictive power of
synergistic drug combinations and antagonistic drug combinations and are widely used in unbalanced datasets. PREC
measures the prediction accuracy of drug pairs that are
predicted as synergistic drug combinations. TPR and TNR
represent the accuracy of the prediction results of the predictor
for the positive and negative samples, respectively. Generally
speaking, the higher the above indicators, the better the
prediction ability of the model.

# ■ [RESULTS AND DISCUSSION]


**Comparison with Existing Methods on Benchmark**
**Datasets.** To evaluate the effectiveness of the proposed
AttenSyn, we compare it with several existing methods,
including machine-learning-based methods (i.e., random forest
(RF), Support Vector Machines (SVM), Multilayer Perceptron
(MLP), Adaboost, and Elastic net) and deep-learning methods
(i.e., DTSyn, [22] MR-GNN, [31] and DeepSynergy [14] ), by five fold



cross validation on the benchmark datasets. The detailed
comparative results are illustrated in Table 1, and the best
results are shown in bold. As shown in Table 1, we can see that
our proposed AttenSyn achieved better performance than
competing methods in terms of AUROC, AUPR, ACC, BACC,
TPR, and KAPPA. Specifically, our AttenSyn achieved
AUROC, AUPR, ACC, BACC, PREC, TPR, and KAPPA of
0.92, 0.91, 0.84, 0.84, 0.83, 0.82, and 0.67, respectively. We
note that the other two molecular-graph-based deep-learning
methods, DTSyn and MR-GNN, also achieved remarkable
performances, which follow our method closely and outperform much more than other methods.
To further demonstrate the good performance of our
AttenSyn, we use the leave-tumor-out cross validation strategy
to evaluate our method and the other two state-of-the-art
deep-learning methods (i.e., DTSyn and DeepSynergy). More
precisely, to make sure that the model cannot see any gene
expression information on a specific type of tumor in the
training process, we exclude all the cancer cell lines belonging
to the specific tumor from the training set. We repeat the
process and iteratively use the excluded cancer cell lines as the
validation set and the remaining samples as the training set to
train and evaluate the model. Figure 3a shows the comparison
result of AttenSyn and the other two deep-learning-based
methods on the leave-tumor-out cross validation task. It can be
seen from Figure 3a that our proposed AttenSyn achieved the
best on AUROC, AUPR, and TPR. Figure 3b shows the
AUROC score of AttenSyn and the other two methods on
each tumor type. As shown in Figure 3b, we can see that
AttenSyn has the best AUROC score over all the six tumor
types, indicating that our AttenSyn has the potential to predict
synergistic drug combinations across various tumor types. We


**2858** [https://doi.org/10.1021/acs.jcim.3c00709](https://doi.org/10.1021/acs.jcim.3c00709?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 2854−2862


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


Figure 4. Performances of our proposed AttenSyn and its variants.


−
Figure 5. Visualization results of three randomly selected drug pairs. Panels (a) (c) show the visualization of attention scores from our model in

−
the three-drug pairs after the training process; panels (d) (f) show the visualization of attention scores of the three-drug pairs before the training
process. The darker color represents a more important substructure.



also conduct the experiment under cold start setting and our
AttenSyn achieved the best performance compared with other
[methods. The results can be seen in Supplementary Figure 1.](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.3c00709/suppl_file/ci3c00709_si_001.pdf)
**Ablation Study.** To investigate the effect of the attentionbased pooling module and graph-based drug-embedding
module on our model performance, we consider the following
variants of AttenSyn: (1) the proposed AttenSyn; (2)
AttenSyn (add); (3) AttenSyn (mean); (4) AttenSyn
(SAG); (5) AttenSyn(no graph). Specifically, the AttenSyn
(add) uses a global add pooling method instead of attentionbased pooling in our original AttenSyn. The AttenSyn (mean)
uses a global mean pooling method instead of attention-based
pooling. And the AttenSyn (SAG) calculates self-attention



scores in the way introduced by SAGPooling [33] to update its
nodes’ embeddings and then add them to get the graph-level
representations. The AttenSyn(no graph) removed the graphbased drug-embedding module.
As can be seen in Figure 4, the performance of our original
AttenSyn is better compared with other variants of AttenSyn.
Moreover, from the results in Figure 4, we can see that
AttenSyn (add) and AttenSyn (SAG) are slightly superior to
the results of AttenSyn (mean). This may be because, for
molecular graphs, the global mean pooling method treats each
substructure as equally important and simply averages all
nodes’ embeddings. Specifically, SAGPooling uses a selfattention mechanism to calculate the score of each


**2859** [https://doi.org/10.1021/acs.jcim.3c00709](https://doi.org/10.1021/acs.jcim.3c00709?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 2854−2862


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


Figure 6. Visualization of the feature space distribution of our model. Panel (a) represents t-SNE visualization results of our model with and
without the training process on A375 and HT29; panel (b) represents UMAP visualization results of our model with and without the training
process on A375 and HT29.



substructure and weighted sum all the nodes’ embeddings
according to the attention scores to get the final graph-level
representation. However, there is no significant difference in all
seven metrics between AttenSyn (add) and AttenSyn (SAG).
The reason might be that the SAGPooling method uses only a
single drug’s molecular graph to get the attention scores of
each substructure instead of using interactive information on
drug pairs to calculate the importance of each substructure.
Meanwhile, our proposed attention-based pooling module uses
interactive information on drug pairs and thus can achieve
better performance than the other pooling strategies. In
addition, our proposed AttenSyn is superior to the AttenSyn(no graph), proving the effectiveness of graph-based drugembedding module.
**Visualization of Important Substructures Detected**
**by Our Model.** Deep-learning-based models are often
regarded as “black box”, and the lack of model interpretability
limits their further application in real scenarios of many areas,
especially in computer-aided drug discovery. To overcome the
black box problem and explore which substructures within the
drug pairs provide the most significant contribution to
synergistic drug combinations prediction, we visualized the
most essential substructures for drug pairs through the
attention mechanism of our model. Specifically, we use the
attention score calculated by formulas 6 and 7 to represent the
importance of corresponding substructures and visualize the
scores in different colors. Figure 5a−c shows the visualization
results of three randomly selected example drug pairs, where
the darker color denotes the more important substructure. One
of the chemical structures detected by our model is the amide
group, which plays a key role in the composition of
biomolecules, including many clinically approved drugs. [34]

Amides are prevalent in medicinally important compounds not
only because they are particularly stable but also because they
are polar, which allows amide-containing drugs to interact with
biological receptors and enzymes. [35] This result demonstrates
that our model can provide good interpretability.
To further explore the changes in attention scores of
substructures during the training process, we also visualized the



attention score’s distribution of drug pairs before model
training. As shown in Figure 5d−f, the attention scores before
model training are distributed more uniformly, which indicated
that the model can not pay attention to the important
structures. However, as the training goes on, some specific
structures have been considered more important than others
by the model.
**Feature Representation and Visualization by Dimen-**
**sion Reduction.** To further interpret how deep learning
works during the training process from the feature analysis and
intuitively show the feature learning ability of the proposed
AttenSyn, we visualized the embeddings of drug combinations
in two cell lines (i.e., A375 and HT29). Specifically, we reduce
the embedding space of drug pairs extracted from our model
with and without the training process to a two-dimensional
space by using t-SNE [36] and UMAP, [37] respectively, as in Figure
6. In each subfigure of Figure 6, each point represents a drug
pair, and different colors are used to distinguish the synergistic
drug combination and antagonistic drug combination classes.
The more distinguishable the points under different categories
are, the better the classification effect is. As shown in Figure 6a,
by the dimension reduction of t-SNE, the samples of two
classes are distributed more clearly in the feature space of the
trained model compared to the model without the training
process, indicating that our model can capture discriminative
and high-quality features from different classes samples. There
exist similar results for the model by the dimension reduction
of UMAP. From Figure 6b, the model with the training process
learns and achieves more distinguishable features compared to
the model without the training process.

# ■ [CONCLUSION]


In this study, we developed a novel attention-based deep graph
neural network named AttenSyn to predict the synergy of
anticancer drug combinations, which is a crucial step for rapid
virtual drug screening and drug development. Specifically, we
first generate molecular graphs of drugs and employ the graphbased drug-embedding module to extract structural information on drug pairs, respectively. After that, the attention-based


**2860** [https://doi.org/10.1021/acs.jcim.3c00709](https://doi.org/10.1021/acs.jcim.3c00709?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 2854−2862


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article



pooling module is designed to learn better interactive
information and strengthen the representations of drug pairs.
Comprehensive experiments conducted on the benchmark
datasets show that the proposed method achieves a better
predictive performance than the comparative methods. Moreover, to overcome the limitations of the “black box” in deeplearning-based models, we explored what our model learns
during the training process in both discovering the crucial
substructures in drugs and conducting feature analysis, which
provides good interpretability of our model and biological
insights for understanding the drug synergy mechanism.
However, there are still some drawbacks to our model. For
example, biological networks have already proven their
effectiveness in drug synergy prediction. [15][,][38][−][41] We use only
the molecular structure information and cell-line features, but
not extra information such as biological network information
for prediction. In the future, we will consider introducing the
biological network to improve the performance of anticancer
synergistic drug combinations prediction.

# ■ [ASSOCIATED CONTENT]

**Data Availability Statement**
The data and code for this study can be found in a GitHub
[repository accompanying this manuscript: https://github.com/](https://github.com/badguyisme/SynPred)
[badguyisme/SynPred.](https://github.com/badguyisme/SynPred)
- **sı** **Supporting Information**
The Supporting Information is available free of charge at
[https://pubs.acs.org/doi/10.1021/acs.jcim.3c00709.](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00709?goto=supporting-info)
Experimental settings, performance under cold start
setting, and the distributions of predictive probability
[(PDF)](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.3c00709/suppl_file/ci3c00709_si_001.pdf)

# ■ [AUTHOR INFORMATION]

**[Corresponding Author](https://pubs.acs.org/action/doSearch?field1=Contrib&text1="Leyi+Wei"&field2=AllField&text2=&publication=&accessType=allContent&Earliest=&ref=pdf)**

Leyi Wei − _School of Software, Shandong University, Jinan_
_250101, China; Joint SDU-NTU Centre for Artificial_
_Intelligence Research (C-FAIR), Shandong University, Jinan_
_250101, China;_ [orcid.org/0000-0003-1444-190X;](https://orcid.org/0000-0003-1444-190X)
[Email: weileyi@sdu.edu.cn](mailto:weileyi@sdu.edu.cn)


**[Authors](https://pubs.acs.org/action/doSearch?field1=Contrib&text1="Tianshuo+Wang"&field2=AllField&text2=&publication=&accessType=allContent&Earliest=&ref=pdf)**

Tianshuo Wang − _School of Software, Shandong University,_
_Jinan 250101, China; Joint SDU-NTU Centre for Artificial_
_Intelligence Research (C-FAIR), Shandong University, Jinan_
_[250101, China](https://pubs.acs.org/action/doSearch?field1=Contrib&text1="Ruheng+Wang"&field2=AllField&text2=&publication=&accessType=allContent&Earliest=&ref=pdf)_
Ruheng Wang − _School of Software, Shandong University,_
_Jinan 250101, China; Joint SDU-NTU Centre for Artificial_
_Intelligence Research (C-FAIR), Shandong University, Jinan_
_250101, China_


Complete contact information is available at:
[https://pubs.acs.org/10.1021/acs.jcim.3c00709](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00709?ref=pdf)


**Author Contributions**
T.W. conceived the basic idea and designed the framework.
T.W. and R.W. performed the experiments. R.W. and T.W.
wrote the manuscript. L.W. revised the manuscript. R.W.
conducted the visualization of the experimental results.
**Funding**
The work was supported by the Natural Science Foundation of
China (Nos. 62322112 and 62071278).

**Notes**
The authors declare no competing financial interest.


# ■ [REFERENCES]

(1) Giles, T. D; Weber, M. A; Basile, J.; Gradman, A. H; Bharucha,
[D. B; Chen, W.; Pattathil, M. Efficacy and safety of nebivolol and](https://doi.org/10.1016/S0140-6736(14)60614-0)
[valsartan as fixed-dose combination in hypertension: a randomised,](https://doi.org/10.1016/S0140-6736(14)60614-0)
[multicentre study.](https://doi.org/10.1016/S0140-6736(14)60614-0) _Lancet_ 2014, _383_ (9932), 1889−1898.
[(2) Zheng, W.; Sun, W.; Simeonov, A. Drug repurposing screens and](https://doi.org/10.1111/bph.13895)
[synergistic drug-combinations for infectious diseases.](https://doi.org/10.1111/bph.13895) _British journal of_
_pharmacology_ 2018, _175_ (2), 181−191.
(3) Kim, Y.; Zheng, S.; Tang, J.; Jim Zheng, W.; Li, Z.; Jiang, X.
[Anticancer drug synergy prediction in understudied tissues using](https://doi.org/10.1093/jamia/ocaa212)
[transfer learning.](https://doi.org/10.1093/jamia/ocaa212) _Journal of the American Medical Informatics_
_Association_ 2021, _28_ (1), 42−51.
(4) Chen, X.; Yan, C. C.; Zhang, X.; Zhang, X.; Dai, F.; Yin, J.;
[Zhang, Y. Drug-target interaction prediction: databases, web servers](https://doi.org/10.1093/bib/bbv066)
[and computational models.](https://doi.org/10.1093/bib/bbv066) _Briefings in bioinformatics_ 2016, _17_ (4),
696−712.
[(5) Chen, X.; Guan, N.-N.; Sun, Y.-Z.; Li, J.-Q.; Qu, J. MicroRNA-](https://doi.org/10.1093/bib/bby098)
[small molecule association identification: from experimental results to](https://doi.org/10.1093/bib/bby098)
[computational models.](https://doi.org/10.1093/bib/bby098) _Briefings in Bioinformatics_ 2020, _21_ (1), 47−
61.
[(6) Wang, C.-C.; Zhao, Y.; Chen, X. Drug-pathway association](https://doi.org/10.1093/bib/bbaa061)
[prediction: from experimental results to computational models.](https://doi.org/10.1093/bib/bbaa061)
_Briefings in Bioinformatics_ 2021, _22_ (3), bbaa061.
[(7) Sałat, R.; Sałat, K. The application of support vector regression](https://doi.org/10.1016/j.cmpb.2013.04.018)
[for prediction of the antiallodynic effect of drug combinations in the](https://doi.org/10.1016/j.cmpb.2013.04.018)
[mouse model of streptozocin-induced diabetic neuropathy.](https://doi.org/10.1016/j.cmpb.2013.04.018) _Computer_
_methods and programs in biomedicine_ 2013, _111_ (2), 330−337.
(8) Qi, Y. Random forest for bioinformatics. In _Ensemble machine_
_learning_ ; Springer: 2012; pp 307−323.
[(9) Liu, H.; Zhang, W.; Nie, L.; Ding, X.; Luo, J.; Zou, L. Predicting](https://doi.org/10.1186/s12859-019-3288-1)
[effective drug combinations using gradient tree boosting based on](https://doi.org/10.1186/s12859-019-3288-1)
[features extracted from drug-protein heterogeneous network.](https://doi.org/10.1186/s12859-019-3288-1) _BMC_
_Bioinformatics_ 2019, _20_ (1), 645.
(10) Pivetta, T.; Isaia, F.; Trudu, F.; Pani, A.; Manca, M.; Perra, D.;
[Amato, F.; Havel, J. Development and validation of a general](https://doi.org/10.1016/j.talanta.2013.04.031)
[approach to predict and quantify the synergism of anti-cancer drugs](https://doi.org/10.1016/j.talanta.2013.04.031)
[using experimental design and artificial neural networks.](https://doi.org/10.1016/j.talanta.2013.04.031) _Talanta_
2013, _115_, 84−93.
[(11) Zhang, C.; Yan, G. Synergistic drug combinations prediction by](https://doi.org/10.1016/j.synbio.2018.10.002)
[integrating pharmacological data.](https://doi.org/10.1016/j.synbio.2018.10.002) _Synthetic and systems biotechnology_
2019, _4_ (1), 67−72.
(12) Janizek, J. D.; Celik, S.; Lee, S.-I. Explainable machine learning
prediction of synergistic drug combinations for precision cancer
medicine. _BioRxiv_, 2018, 331769 (submitted 6/29/2023) (accessed
7/16/2023).
(13) Chen, X.; Ren, B.; Chen, M.; Wang, Q.; Zhang, L.; Yan, G.
[NLLSS: predicting synergistic drug combinations based on semi-](https://doi.org/10.1371/journal.pcbi.1004975)
[supervised learning.](https://doi.org/10.1371/journal.pcbi.1004975) _PLoS computational biology_ 2016, _12_ (7),
No. e1004975.
(14) Preuer, K.; Lewis, R. P.; Hochreiter, S.; Bender, A.; Bulusu, K.
[C.; Klambauer, G. DeepSynergy: predicting anti-cancer drug synergy](https://doi.org/10.1093/bioinformatics/btx806)
[with Deep Learning.](https://doi.org/10.1093/bioinformatics/btx806) _Bioinformatics_ 2018, _34_ (9), 1538−1546.
[(15) Liu, Q.; Xie, L. TranSynergy: mechanism-driven interpretable](https://doi.org/10.1371/journal.pcbi.1008653)
[deep neural network for the synergistic prediction and pathway](https://doi.org/10.1371/journal.pcbi.1008653)
[deconvolution of drug combinations.](https://doi.org/10.1371/journal.pcbi.1008653) _PLoS computational biology_
2021, _17_ (2), No. e1008653.
[(16) Kuru, H. I.; Tastan, O.; Cicek, A. E. MatchMaker: a deep](https://doi.org/10.1109/TCBB.2021.3086702)
[learning framework for drug synergy prediction.](https://doi.org/10.1109/TCBB.2021.3086702) _IEEE/ACM Trans-_
_actions on Computational Biology and Bioinformatics_ 2022, _19_ (4),
2334−2344.
[(17) Su, R.; Huang, Y.; Zhang, D.-g.; Xiao, G.; Wei, L. SRDFM:](https://doi.org/10.1093/bib/bbab534)
[Siamese Response Deep Factorization Machine to improve anti-](https://doi.org/10.1093/bib/bbab534)
[cancer drug recommendation.](https://doi.org/10.1093/bib/bbab534) _Briefings in Bioinformatics_ 2022, _23_ (2),
bbab534.
(18) Lin, W.; Wu, L.; Zhang, Y.; Wen, Y.; Yan, B.; Dai, C.; Liu, K.;
[He, S.; Bo, X. An enhanced cascade-based deep forest model for drug](https://doi.org/10.1093/bib/bbab562)
[combination prediction.](https://doi.org/10.1093/bib/bbab562) _Briefings in Bioinformatics_ 2022, _23_ (2),
bbab562.


**2861** [https://doi.org/10.1021/acs.jcim.3c00709](https://doi.org/10.1021/acs.jcim.3c00709?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 2854−2862


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


̌ ́ ̀



[(19) Li, T.-H.; Wang, C.-C.; Zhang, L.; Chen, X. SNRMPACDC:](https://doi.org/10.1093/bib/bbac503)
[computational model focused on Siamese network and random matrix](https://doi.org/10.1093/bib/bbac503)
[projection for anticancer synergistic drug combination prediction.](https://doi.org/10.1093/bib/bbac503)
_Briefings in Bioinformatics_ 2023, _24_ (1), bbac503.
[(20) Wang, J.; Liu, X.; Shen, S.; Deng, L.; Liu, H. DeepDDS: deep](https://doi.org/10.1093/bib/bbab390)
[graph neural network with attention mechanism to predict synergistic](https://doi.org/10.1093/bib/bbab390)
[drug combinations.](https://doi.org/10.1093/bib/bbab390) _Briefings in Bioinformatics_ 2022, _23_, (1),
[DOI: 10.1093/bib/bbab390.](https://doi.org/10.1093/bib/bbab390?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
(21) Jin, W.; Stokes, J. M.; Eastman, R. T.; Itkin, Z.; Zakharov, A. V.;
[Collins, J. J.; Jaakkola, T. S.; Barzilay, R. Deep learning identifies](https://doi.org/10.1073/pnas.2105070118)
[synergistic drug combinations for treating COVID-19.](https://doi.org/10.1073/pnas.2105070118) _Proc. Natl._
_Acad. Sci. U. S. A._ 2021, _118_ (39), No. e2105070118.
(22) Hu, J.; Gao, J.; Fang, X.; Liu, Z.; Wang, F.; Huang, W.; Wu, H.;
Zhao, G. DTSyn: a dual-transformer-based neural network to predict
synergistic drug combinations. _bioRxiv_ 2022 (submitted 6/29/2023)
(accessed 7/16/2023).
[(23) Nyamabo, A. K.; Yu, H.; Shi, J.-Y. SSI-DDI: substructure-](https://doi.org/10.1093/bib/bbab133)
[substructure interactions for drug-drug interaction prediction.](https://doi.org/10.1093/bib/bbab133) _Brief-_
_ings in Bioinformatics_ (24) Deac, A.; Huang, Y.-H.; Velic 2021, _22_ (6), bbab133.kovič, P.; Lió, P.; Tang, J. Drug-̀
drug adverse effect prediction with graph co-attention. _arXiv_,
1905.00534, 2019 (submitted 6/29/2023) (accessed 7/16/2023).
(25) O'Neil, J.; Benita, Y.; Feldman, I.; Chenard, M.; Roberts, B.;
Liu, Y.; Li, J.; Kral, A.; Lejnine, S.; Loboda, A.; Arthur, W.; Cristescu,
R.; Haines, B. B.; Winter, C.; Zhang, T.; Bloecher, A.; Shumway, S. D.
[An unbiased oncology compound screen to identify novel](https://doi.org/10.1158/1535-7163.MCT-15-0843)
[combination strategies.](https://doi.org/10.1158/1535-7163.MCT-15-0843) _Molecular cancer therapeutics_ 2016, _15_ (6),
1155−1162.
(26) Di Veroli, G. Y.; Fornari, C.; Wang, D.; Mollard, S.; Bramhall, J.
[L.; Richards, F. M.; Jodrell, D. I. Combenefit: an interactive platform](https://doi.org/10.1093/bioinformatics/btw230)
[for the analysis and visualization of drug combinations.](https://doi.org/10.1093/bioinformatics/btw230) _Bioinformatics_
2016, _32_ (18), 2866−2868.
[(27) Weininger, D. J. J. o. c. i. SMILES, a chemical language and](https://doi.org/10.1021/ci00057a005?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[information system. 1. Introduction to methodology and encoding](https://doi.org/10.1021/ci00057a005?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[rules.](https://doi.org/10.1021/ci00057a005?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _Journal of chemical information and computer sciences_ 1988, _28_
(1), 31−36.
(28) Wishart, D. S; Feunang, Y. D; Guo, A. C; Lo, E. J; Marcu, A.;
Grant, J. R; Sajed, T.; Johnson, D.; Li, C.; Sayeeda, Z.; Assempour,
N.; Iynkkaran, I.; Liu, Y.; Maciejewski, A.; Gale, N.; Wilson, A.; Chin,
[L.; Cummings, R.; Le, D.; Pon, A.; Knox, C.; Wilson, M. DrugBank](https://doi.org/10.1093/nar/gkx1037)
[5.0: a major update to the DrugBank database for 2018.](https://doi.org/10.1093/nar/gkx1037) _Nucleic acids_
_research_ 2018, _46_ (D1), D1074−D1082.
(29) Ghandi, M.; Huang, F. W.; Jane-Valbuena, J.; Kryukov, G. V.;
Lo, C. C.; McDonald, E. R.; Barretina, J.; Gelfand, E. T.; Bielski, C.
M.; Li, H.; Hu, K.; Andreev-Drakhlin, A. Y.; Kim, J.; Hess, J. M.;
Haas, B. J.; Aguet, F.; Weir, B. A.; Rothberg, M. V.; Paolella, B. R.;
Lawrence, M. S.; Akbani, R.; Lu, Y.; Tiv, H. L.; Gokhale, P. C.; de
Weck, A.; Mansour, A. A.; Oh, C.; Shih, J.; Hadi, K.; Rosen, Y.;
Bistline, J.; Venkatesan, K.; Reddy, A.; Sonkin, D.; Liu, M.; Lehar, J.;
Korn, J. M.; Porter, D. A.; Jones, M. D.; Golji, J.; Caponigro, G.;
Taylor, J. E.; Dunning, C. M.; Creech, A. L.; Warren, A. C.;
McFarland, J. M.; Zamanighomi, M.; Kauffmann, A.; Stransky, N.;
Imielinski, M.; Maruvka, Y. E.; Cherniack, A. D.; Tsherniak, A.;
Vazquez, F.; Jaffe, J. D.; Lane, A. A.; Weinstock, D. M.; Johannessen,
C. M.; Morrissey, M. P.; Stegmeier, F.; Schlegel, R.; Hahn, W. C.;
Getz, G.; Mills, G. B.; Boehm, J. S.; Golub, T. R.; Garraway, L. A.;
[Sellers, W. R. Next-generation characterization of the cancer cell line](https://doi.org/10.1038/s41586-019-1186-3)
[encyclopedia.](https://doi.org/10.1038/s41586-019-1186-3) _Nature_ 2019, _569_ (7757), 503−508.
(30) Landrum, G. RDKit: A software suite for cheminformatics,
[computational chemistry, and predictive modeling. https://www.](https://www.rdkit.org/RDKit_Overview.pdf)
[rdkit.org/RDKit_Overview.pdf, 2013.](https://www.rdkit.org/RDKit_Overview.pdf)
(31) Xu, N.; Wang, P.; Chen, L.; Tao, J.; Zhao, J. Mr-gnn: Multiresolution and dual graph neural network for predicting structured
entity interactions. _arXiv_, 1905.09558, 2019 (submitted 6/29/2023)
(accessed 7/16/2023).
[(32) Wang, R.; Jin, J.; Zou, Q.; Nakai, K.; Wei, L. Predicting protein-](https://doi.org/10.1093/bioinformatics/btac352)
[peptide binding residues via interpretable deep learning.](https://doi.org/10.1093/bioinformatics/btac352) _Bioinfor-_
_matics_ 2022, _38_ (13), 3351−3360.



(33) Lee, J.; Lee, I.; Kang, J. Self-attention graph pooling; In
_International Conference on Machine Learning, PMLR: 2019_ ; pp 3734−
3743.
(34) Kumari, S.; Carmona, A. V.; Tiwari, A. K.; Trippier, P. C.
[Amide bond bioisosteres: Strategies, synthesis, and successes.](https://doi.org/10.1021/acs.jmedchem.0c00530?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _Journal_
_of medicinal chemistry_ 2020, _63_ (21), 12290−12358.
(35) Clayden, J. _Fluorinated compounds present opportunities for drug_
_discovery_ . Nature Publishing Group: London, 2019.
(36) Van der Maaten, L.; Hinton, G. Visualizing data using t-SNE.
_Journal of machine learning research_ 2008, _9_ (11).
(37) McInnes, L.; Healy, J.; Melville, J., Umap: Uniform manifold
approximation and projection for dimension reduction. _arXiv_,
1802.03426, 2018 (submitted 6/29/2023) (accessed 7/16/2023).
(38) Yang, J.; Xu, Z.; Wu, W. K. K.; Chu, Q.; Zhang, Q.
[GraphSynergy: a network-inspired deep learning model for anticancer](https://doi.org/10.1093/jamia/ocab162)
[drug combination prediction.](https://doi.org/10.1093/jamia/ocab162) _Journal of the American Medical_
_Informatics Association_ 2021, _28_ (11), 2336−2345.
[(39) Meng, F.; Li, F.; Liu, J.-X.; Shang, J.; Liu, X.; Li, Y. NEXGB: A](https://doi.org/10.3390/ijms23179838)
[Network Embedding Framework for Anticancer Drug Combination](https://doi.org/10.3390/ijms23179838)

̌ ́ ̀ [Prediction.](https://doi.org/10.3390/ijms23179838) _International Journal of Molecular Sciences_ 2022, _23_ (17),

9838.
(40) Jiang, P.; Huang, S.; Fu, Z.; Sun, Z.; Lakowski, T. M.; Hu, P.
[Deep graph embedding for prioritizing synergistic anticancer drug](https://doi.org/10.1016/j.csbj.2020.02.006)
[combinations.](https://doi.org/10.1016/j.csbj.2020.02.006) _Computational and structural biotechnology journal_
2020, _18_, 427−438.
(41) Zhang, P.; Tu, S. A knowledge graph embedding-based method
for predicting the synergistic effects of drug combinations. In _2022_
_IEEE International Conference on Bioinformatics and Biomedicine_
_(BIBM)_ ; IEEE: 2022; pp 1974−1981.

̌ ́ ̀



**2862** [https://doi.org/10.1021/acs.jcim.3c00709](https://doi.org/10.1021/acs.jcim.3c00709?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2024, 64, 2854−2862


