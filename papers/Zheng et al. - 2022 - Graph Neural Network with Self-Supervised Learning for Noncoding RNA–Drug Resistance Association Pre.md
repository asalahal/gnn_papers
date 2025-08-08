pubs.acs.org/jcim Article

## **Graph Neural Network with Self-Supervised Learning for Noncoding** − **RNA Drug Resistance Association Prediction**

#### Jingjing Zheng, Yurong Qian, Jie He, Zerui Kang, and Lei Deng*

### ACCESS Metrics & More Article Recommendations


# ■ [INTRODUCTION]



Noncoding RNAs (ncRNAs) are DNA transcripts that cannot
be encoded into proteins. [1] Numerous studies have shown that
ncRNAs are involved in many biological functions, such as cell
proliferation, cell cycle progression, and apoptosis. [2][−][4] They
have been shown to be key regulators of gene expression, not
just “byproducts” of gene transcription. [5][,][6] In recent years,
noncoding RNAs such as long noncoding RNAs (lncRNAs),
microRNAs (miRNAs), and circular RNAs (circRNAs) have
received extensive attention. lncRNAs are antisense RNA
molecules that can specifically bind to noncoding regions of
target genes, regulate gene transcription and expression, and
play a role in promoting or suppressing tumors. [7] miRNAs are a
class of noncoding RNAs that regulate gene transcription and
expression and participate in a variety of physiological
activities. [8] circRNAs have stability, making them stable in
plasma, saliva, and other peripheral tissues, and are novel
biomarkers and new targets for cancer diagnosis and
treatment. [9][,][10]

Cancer is one of the major diseases that seriously endangers
human health. Currently, the leading cancer treatments are
surgery, radiotherapy, and chemotherapy. [11] Chemotherapy is
one of the essential treatment methods for cancer at present.


© 2022 American Chemical Society



**3676**



Still, most patients often develop drug resistance during
chemotherapy, which leads to the recurrence and metastasis of
cancer cells [12] and the failure of cancer treatment. Exploring the
molecular mechanisms of drug resistance is crucial for drug
discovery and cancer treatment.
With the development of sequencing technology, experimental studies have found that ncRNAs are closely related to
many diseases, including malignant tumors. Studies have
shown that the lncRNA NKILA enhances the sensitivity of T
cells to activation-induced cell death by inhibiting NF- _κ_ B
signaling in breast cancer and the lung cancer microenvironment, thus promoting the immune escape of non-small-cell
lung cancer (NSCLC) cells and affecting the immune
tolerance of lung cancer. [13] miRNAs such as miR-140-5p and
miR-146a can also play an essential role in doxorubicin




[https://doi.org/10.1021/acs.jcim.2c00367](https://doi.org/10.1021/acs.jcim.2c00367?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 3676−3684


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article



induced cardiotoxicity by targeting Sirt2, Nrf2, TAF9b/P53,
and other pathways. [14][,][15] CircPAN3 is a critical mediator in the
development of resistance in acute myeloid leukemia (AML).
Experiments show that CircPAN3 promotes drug resistance in
AML by regulating protein expression. [16] Abnormal expression
of ncRNAs can regulate tumor drug resistance, which provides
new opportunities and research directions for overcoming
tumor drug resistance. Traditional biological experiments often
consume a lot of material and financial resources, which makes
them difficult to implement to a certain extent. Computational
methods are undoubtedly useful accelerators for this process,
and few computational methods have explored the relationship
between ncRNAs and drug resistance. Li et al. [17] developed the
method (LRGCPND) of the graph neural network to

−
efficiently identify potential ncRNA drug resistance associa
−
tions. This is the only existing ncRNA drug resistance
association prediction using a computational approach. In
their studies, first, neighbor information on nodes in the
ncRNA−resistance bipartite graph is captured by aggregation,
then feature transformation is performed by linear operations.
Finally, they use residual blocks to fuse the features of low-level
nodes to achieve prediction.
Although there are few computational methods for

−
predicting ncRNA drug resistance, many related association
prediction methods are worth discussing. Dayun et al. [18]

proposed a novel computational framework of MGATMDA to
detect microbial−disease associations by multicomponent
graph attention networks. First, they generated the latent
vectors of nodes from the bipartite graph through the
decomposer. Then they obtained the unified embedding
representation through the combinator. Finally, they used
the attention mechanism for microbial−disease associations
prediction. Ji et al. [19] constructed a GATNNCDA model
combining graph neural network and multilayer neural

−
network for predicting potential associations of circRNA
disease. Deng et al. [20] proposed a new method (Graph2MDA)

−
using variational graph autoencoders to predict microbe drug
associations. Graph2MDA first constructs a multimodal
attribute graph of microbe and drugs, then uses a variational
graph autoencoder (VGAE) to learn the latent representations
of nodes. Finally, a deep neural network is used to predict

−
potential microbe drug associations. Inspired by the heterogeneous attention network (HGAT), Zhao et al. [21] developed a
new heterogeneous attention network framework, HGATLDA,
based on metapaths, which is used to predict the relationship
between lncRNAs and diseases. Fan et al. [22] proposed a new
prediction method based on graph convolution matrix
completion, GCRFLDA. GCRFLDA embeds the conditional
random field (CRF) with an attention mechanism into the
coding layer, preserves the similarity information between
graph nodes, and scores the potential lncRNA−disease
association. Wang et al. [23] constructed a computing method,
GCNCDA, based on deep learning, fast learning, and the graph
convolution network (FastGCN) algorithm. GCNCDA used a
forest penalty attribute classifier to predict potential associations and diseases between circRNAs accurately. Lan et al. [24]

proposed a new computing framework (IGNSCDA) based on
the improved graph convolution network and negative
sampling to infer the association between circRNAs and
disease. Li et al. [25] proposed a SDNE-MDA model based on
structured deep network embedding (SDNE) to predict
miRNA−disease associations (MDAs). The model constructs
a complex network molecular association network (MAN) by



combining miRNA, disease, and three related molecules
(lncRNA, drug, protein) and their relationships.
Although the association prediction approaches such as
graph neural network technology have been widely used in
various fields, there are still some limitations in the field of
ncRNA−drug resistance association prediction. Most models
tackle the task of association prediction based on supervised
learning. These supervised signals come from the observed
ncRNA−drug resistance associations; however, the observed
associations are very sparse. In this work, we developed a
computational model called GSLRDA to infer unknown
ncRNA−drug resistance associations. GSLRDA combines
graph neural networks and self-supervised learning. GSLRDA
designs the main task and auxiliary tasks. In the main task,
GSLRDA takes an ncRNA−drug resistance bipartite graph as
input, uses a lightGCN to learn ncRNA and drug vector
representations, and then uses the inner product to predict
ncRNA and drug resistance. In the auxiliary task, GSLRDA
first generates different perspectives for ncRNA and drug
nodes through different data augmentation methods and then
performs comparative learning between nodes to improve the
quality of the learned ncRNA and drug vector representation.
Complicated experimental results show that GSLRDA is
superior to the existing eight excellent calculation methods.
Ablation experiments verify the effectiveness of self-supervised
learning. In addition, the case study results on two drugs
indicate that GSLRDA is an effective tool for predicting the
association between ncRNA and drug resistance.

# ■ [METHODS]


**Data Set.** Known ncRNA−drug resistance association pairs
come from the NoncoRNA and ncDR databases. NoncoRNA [26] [(http://www.ncdtcdb.cn:8080/NoncoRNA) is the](http://www.ncdtcdb.cn:8080/NoncoRNA)
first database that provides experimentally supported associations between 5568 ncRNAs, 154 drugs, and 134 cancers.
ncDR [27] [(http://www.jianglab.cn/ncDR) is a noncoding](http://www.jianglab.cn/ncDR)
RNA (ncRNA) database. The ncDR database contains 5864
associations between 877 miRNAs, 162 lncRNAs (1039
ncRNAs in total), and 145 compounds obtained from 900
published articles.
We removed redundant data from the NoncoRNA and

−
ncDR databases; 2693 known ncRNA drug resistanceassociated pairs were obtained, including 121 drugs and 625

−
ncRNAs. In our base data set, there are only 2693 ncRNA
drug resistance associations. To avoid the effect of sample
imbalance, we randomly sample 2693 negative samples from
unknown associations to achieve the same number as positive
samples. In addition, the independent data set was created by
searching the PubMed database literature. It contains 534
known ncRNA resistance associations, including 168 ncRNAs
and 70 drugs.
**ncRNA** − **Drug Bipartite Graph.** The bipartite graph [28]

abstracts the relationship between ncRNA and drug resistance
as a graph. Let _N_ and _D_ be the set of ncRNAs and drugs,
respectively. Let _E_ = {( _y_ _nd_ | _n_ ∈ _N_, _d_ ∈ _D_ )} indicate the verified
association between ncRNA and drug resistance. We use the
ncRNA−drug resistance association matrix _A_ to construct a
bipartite graph _G_ = ( _V_, _E_ ), where the node set _V_ contains all
ncRNAs and drugs, _V_ = _N_ ∪ _D_ .
**GSLRDA.** GSLRDA takes the bipartite graph of ncRNA and
drugs as input and outputs potential ncRNA and drug
resistance associations. GSLRDA uses lightGCN to learn the
representation of ncRNAs and drugs from the bipartite graph


**3677** [https://doi.org/10.1021/acs.jcim.2c00367](https://doi.org/10.1021/acs.jcim.2c00367?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 3676−3684


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


Figure 1. Overview of GSLRDA. GSLRDA designs two tasks. In the main task, we first model the known association between ncRNA and drug
resistance into a bipartite graph of ncRNA and drug. Then, the bipartite graph of ncRNA and drug is input into lightGCN to learn the
representation of ncRNA and drug node. Finally, we use inner products to infer the association between unknown ncRNAs and drug resistance.
Due to the sparse supervision signal, the quality of the learned ncRNA and drug representation needs to be further improved. Therefore, we
perform self-supervised learning by designing auxiliary tasks. In the auxiliary task, we generate different views for ncRNA and drug nodes through
three data augmentation methods: node loss, edge loss, and random walk. Then, we perform comparative learning between nodes. Finally, we

−
integrate the main task and auxiliary task, and jointly optimize the loss function, to predict the associations of ncRNA drug resistance.



(1)


(2)



of ncRNAs and drugs and uses the inner product to infer the
relationship between ncRNAs and drugs. Due to the sparse
supervision signal, the learned ncRNA and drug representations are insufficient. To further improve the learned
association of ncRNAs and drug resistance and improve the
model’s prediction performance, we designed auxiliary tasks by
performing self-supervised learning. Specifically, first, we use
different data augmentation methods to generate other views
for ncRNA and drug nodes and then perform contrastive
learning tasks between nodes. Finally, we optimize the
ncRNA−drug association prediction and comparative learning
tasks jointly. The GSLRDA model is shown in Figure 1.
**GCN for ncRNA** − **Drug Resistance Association.** Graph
embedding represents from a single ID to high-order
neighbors, which makes the graph neural network (GCN)
successful in the recommendation task. To better learn the

high-level features of ncRNAs and drugs, we use the advanced
lightweight graph neural network, lightGCN.
Initialization: Each ncRNA and drug is associated with an ID
0 0
embedding. Here we use _e_ _n_ and _e_ _d_ to represent the ID
embedding of ncRNA _n_ and drug _d_, respectively.
Simplifying and powering graph convolution: A basic idea of
a graph neural network is to embed the target node based on
its neighbor information. Intuitively, the information on each
node and its surrounding nodes is aggregated through a neural
network. The neighborhood aggregation formula of ncRNAs
and drugs is



Layer combination: After the _k_ -layer graph convolution
operation, we combine the embeddings obtained from each
layer and then form the final representation of ncRNAs and
drugs. The formulas are


**3678** [https://doi.org/10.1021/acs.jcim.2c00367](https://doi.org/10.1021/acs.jcim.2c00367?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 3676−3684



_e_ _n_ ( _k_ +1) = _F_ agg ( _e_ _n_ ( ) _k_, { _e_ _d_ ( ) _k_ : _d_ _D_ _n_ })


_e_ _d_ ( _k_ +1) = _F_ agg ( _e_ _d_ ( ) _k_, { _e_ _n_ ( ) _k_ : _n_ _N_ _d_ })



where _D_ _n_ represents the set of drugs that have interacted with
ncRNA _n_, and _N_ _d_ represents the embedding representation of
0
ncRNA and drug after _k_ -layer propagation. When _k_ = 0, _e_ _n_ and
_e_ 0 _d_ represent ncRNA _n_ and drug _d_, respectively. _F_ agg is an
aggregate function.
In graph convolution operations, the aggregation function
_F_ agg is the core of the operation. LightGCN abandons the
feature transformation and nonlinear activation operations that
have no positive significance to the model performance and
only uses simple weighted summation operations. In the
lightGCN model, the calculation formula for the neighborhood
aggregation of ncRNA and drug nodes is



_e_ _n_ ( _k_ +1) = | _D_ |1 | _N_ | _e_



_n_ ( _k_ +1) = _e_ _d_ ( ) _k_



=



_n_ _n_ _d_ (3)



_e_ _d_ ( _k_ +1) = 1 _e_



_d_ ( _k_ +1) = _e_ _n_ ( ) _k_



=



_d_ _d_ _n_ (4)



where



1
| _N_ _d_ | | _D_ _n_ | [is the normalization operation.]


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article



_K_



_e_ _n_ = _k_ _e_

_k_ =0



( ) _k_



=



( ) _k_

_k_ _e_ _[n]_



lightGCN in the previous section is used to learn node
features.
Comparative learning: For ncRNA _n_, _n_ ∈ _N_, we define _n_ [+] as
a positive sample similar to _n_, _n_ [−] as a negative sample not
similar to _n_, and _s_ as a metric function to measure the similarity
between samples. The goal of contrastive learning is to learn an
encoder _f_ such that _S_ ( _f_ ( _n_ ), _f_ ( _n_ [+] )) ≫ _S_ ( _f_ ( _n_ ), _f_ ( _n_ [−] )). We use
the vector product to calculate the similarity between two
samples; then, the contrastive learning loss function of the
ncRNA node is



0



_K_



(5)


(6)



_e_ _d_ = _k_ _e_

_k_ =0



( ) _k_



=



( ) _k_

_k_ _e_ _[d]_



0



In these equations, _β_ _k_ is the hyperparameter and is set to _K_ 1+ 1 [.]
Prediction layer: The inner product of an ncRNA and the
final representation of the drug are used as a model prediction.


T
_y_ _nd_ = _e e_ _n d_ (7)


Loss function: We adopt Bayesian Personalized Ranking
(BPR) loss as the loss function.


_M_



ÄÅÅÅÅÅÅÅÅÅÅÅÅÅÇ



É



T

= _E_ _N_ ÅÅÅÅÅÅÅÅÅÅÅÅÅlog exp( ( ) _f n_ T _f n_ ( ex + ))p(+ _f n_ ( ) _Nj_ = _f n_ ( 11 exp( ( ))) _f n_ T _f n_ ( _j_ )) ÑÑÑÑÑÑÑÑÑÑÑÑÑ



T



exp( _f n_ ( ) _f n_ ( ))
log exp( ( ) _f n_ T _f n_ ( + )) + _N_ = 1 exp( ( ) _f n_ T _f n_ (



_L_ _n_ = _E_ _N_ log exp( ( ) _f n_ T _f n_ ( ex + ))p(+ _f n_ ( ) _N_ = _f n_ ( 1



_n_ _N_ exp( ( ) _f n_ T _f n_ ( + )) + _Nj_ =11 exp( ( ) _f n_ T _f n_ ( _j_ ))



Ö



T _f n_ ( + )) + _N_ =11 exp( ( ) _f n_ T



_L_ = log ( _y_ _nd_ _y_ _nq_ )

=



=



_n_



_nd_ _nq_
1 _d_ _D q D_ _n_ _n_ (8)



**Self-Supervised Learning.** To further improve the
prediction performance of the model, we design auxiliary
tasks by performing self-supervised learning. Self-supervised
learning in ncRNA and drug resistance association prediction
includes two parts: data enhancement and comparative
learning. The detailed process is as follows.
Data enhancement: There is an inherent link between
ncRNAs and drug resistance and they do not exist
independently. Methods such as NLP (synonym substitution,
random deletion, etc.) and CV (flip, Gaussian white noise,
etc.) tasks to achieve data enhancement are not suitable for
ncRNA−drug resistance association prediction. Therefore, it is
necessary to develop new enhanced algorithms for the
prediction of ncRNA and drug resistance associations.
We use three algorithms, node dropout, edge dropout, and
random walk, in the graph structure to obtain different views of
ncRNA and drug nodes:
Node dropout deletes nodes and their linked edges in the
graph through probability _ρ_, _ρ_ ∈ (0, 1).


_G_ = ( _V_ _O E_, ); _O_ = ( _O_ 1, _O_ 2, ... _O_ _i_ ..., _O_ _n_ ); _O_ _i_

(9)


_O_ is a vector responsible for deciding which nodes in the node
set _V_ should be retained.
Edge dropout deletes the edges in the original graph with
probability _ρ_, _ρ_ ∈ (0, 1).



(, _V E_ _Q_ ); _Q_ = ( _Q_ 1, _Q_ 2, ... _Q_ _i_ ..., _Q_ _n_ );



_G_ = (, _V E_ _Q_ ); _Q_ = ( _Q_, _Q_, ... _Q_ ..., _Q_



1 2



_Q_



_i_



(12)

_n_ has a positive sample and _N_ − 1 negative samples, and the
goal of our learning is to make the features of _n_ more similar to
the features of _n_ [+] and less similar to the features of the _N_ − 1
negative samples. Similarly, we can obtain the loss function _L_ ″ _d_
of the drug node; then, the loss function of contrastive learning

″ ″ ″
is _L_ = _L_ _n_ + _L_ _d_ .
**Joint Learning.** To further improve the performance of the
model, we jointly optimize the lightGCN and self-supervised
learning tasks using a multitask training strategy.


2
_L_ = _L_ + 1 _L_ + 2 2 (13)


where _θ_ is the parameter set of the lightGCN model, since no
additional parameters are introduced in self-supervised
learning; _γ_ 1 and _γ_ 2 are hyperparameters that control the
strength of self-supervised learning and L2 regularization,
respectively.

# ■ [EXPERIMENTS AND RESULTS]


**Experimental Setup.** To evaluate the performance of the
GSLRDA model, a 5-fold cross-validation method was used to
evaluate the potential of ncRNAs to predict drug resistance.
The 2693 known ncRNA−resistance-related data are randomly
divided into 5 subsets of the same size. The classification
criteria are as follows:


_P_ = _P_ 1 _P_ 2 _P_ 3 _P_ 4 _P_ 5 (14)


= _P_ 1 _P_ 2 _P_ 3 _P_ 4 _P_ 5 (15)


Each time the model is trained and evaluated, the current
target subset is used as the test set, and the remaining 4 subsets
are used as the training set. This process continues until each
subset is used as the test set. Then, we calculate the average
value of the 5 iterations as the final result of the GSLRDA
model. We choose the commonly used evaluation indicators in
association prediction tasks, including AUC (area under the
receiver operating characteristic curve) [29] and AUPR (area
under the accurate recall curve). [30]

**Comparison of Models.** To prove the effectiveness of the
GSLRDA model, we compared it with 8 methods of making
association predictions on biological information.
_LRGCPND._ The first computational model to predict ncRNA
resistance, LRGCPND [17] captures the neighbor information
representation in the bipartite graph of ncRNA resistance
through aggregation, then performs feature transformation
through linear operations, and finally makes the final
prediction through residual links.
_SDLDA._ SDLDA [31] is calculation method for predicting
lncRNA-disease by combining the nonlinear features and linear


**3679** [https://doi.org/10.1021/acs.jcim.2c00367](https://doi.org/10.1021/acs.jcim.2c00367?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 3676−3684



(10)



_Q_ is a vector that is responsible for deciding which edges in the
edge set _E_ should be deleted.
Both node dropout and edge dropout generate shared
subgraphs between graph convolutional layers. We consider
using random walk operators to allocate different subgraphs to
different layers.


_G_ = (, _V E_ _Q_ ); _Q_ = ( _Q_ 1, _Q_ 2, ..., _Q_ _n_ ); _Q_ _i_

(11)


_Q_ ′ is a vector that is responsible for deciding which edges in
the edge set _E_ should be deleted.
After obtaining graph structures from different perspectives
through the above three data enhancement methods, the


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article



features obtained in deep learning and singular value matrix
decomposition.
_DMFCDA._ In DMFCDA, [32] a method of deep matrix
decomposition, using a projection layer composed of a fully
connected network to capture the potential characteristics of
circRNA and diseases, the combination is sent to a multilayer
neural network for prediction.
_DMFMDA._ In DMFMDA, [33] the one-hot encoding of
microbe and disease is input to the embedding layer to
convert it into a low-dimensional vector. Then, the matrix
decomposition is realized through the neural network with the
embedded layer, and finally, the prediction is made.
_KATZHMDA._ In KATZHMDA, [34] the heterogeneous network is constructed from the multisource similarity network of
miRNA and disease and the miRNA−disease association
network, and finally, the miRNA−disease association is
predicted by KATZ.
_NTSHMDA._ In NTSHMDA, [35] a heterogeneous microbial−
disease network is constructed and then microbial−disease
associations are predicted through an integrated network based
on random walks.
_AE-RF._ For AE-RF, [36] the deep features of circRNA and
disease are extracted through a deep autoencoder, and random
forest is used to make association predictions.
_ABHMDA._ With ABHMDA, [37] first, the similarity between
diseases and microorganisms is calculated, and then reliable
negative samples are selected through K-means clustering.
Finally, the strong classification adaptive boosting combined by
multiple weak classifiers predicts the human microbe−disease
association.
In this work, we used 5-fold cross-validation to evaluate the
performance of GSLRDA and the other 8 methods. As shown
in Figure 2, the AUC value reached 0.9101. Overall, our


Figure 2. ROC of GSLRDA compared with eight related models.


method outperformed other association prediction methods.
This may be attributed to two strategies, self-supervised
learning and the graph neural network, which enable GSLRDA
to capture richer and more important feature information.
Compared with matrix factorization methods (SDLDA, [31]

DMFCDA, [32] DMFMDA [33] ), we deepened the learning from
the representation of matrix factorization to the use of graph
neural networks to capture richer information using the highorder connectivity of ncRNAs and drug resistance. Compared
with supervised learning methods (LRGCPND, [17] KATZHM


DA, [34] NTSHMDA, [35] AE-RF, [36] ABHMDA [37] ), we used selfsupervised learning node enhancement to construct a
comparative learning strategy, obtained more important
information from the data, and further improved the model
performance. Overall, GSLRDA is effective in predicting the
association of ncRNA resistance.
To further verify the predictive ability of the GSLRDA
model, we established an independent test set and compared
GSLRDA with other excellent models on the independent test
set. Through a literature search in the PubMed database, an
independent test set containing 534 ncRNA and drug
resistance associations, 168 ncRNAs, and 70 drugs was
established. We used the 526 ncRNA and drug resistance
associations of our data set as the training set training model
and tested it on the independent test set. The experimental
results are shown in Figure 3. The AUC of GSLRDA reaches


Figure 3. Performance compared between GSLRDA and eight related
models on independent test sets.


0.9153, and the AUPR reached 0.9129, both higher than those
of the other models. The experimental results fully show that
GSLRDA is an effective tool to infer the association between
ncRNA and drug resistance.
**Ablation Experiment.** In this work, we designed a
GSLRDA model that uses a self-supervised learning mechanism to assist lightGCN [38] in predicting the association
between ncRNAs and drug resistance. To analyze the necessity
of the self-supervised learning strategy for the GSLRDA model,
we conducted ablation experiments on this. The ultimate goal
of our task is to predict whether there is an association
between ncRNAs and drugs, that is, to provide useful ncRNA
targets for drugs. This is analogous to the popular
recommendation task. Therefore, we choose the NGCF [39]

model with the classical GCN method to prove the necessity of
introducing self-supervised learning. Specifically, different
models were used to predict ncRNA−resistance associations
on the same data set. Table 1 shows the average evaluation
metric values obtained for the three models under 5-fold crossvalidation. GSLRDA consistently outperforms other baseline
methods. This verifies the rationality and effectiveness of
introducing self-supervised learning. The lightGCN implementation performs better than NGCF, which is a consistent claim
of the lightGCN paper. For the GSLRDA model, using self

**3680** [https://doi.org/10.1021/acs.jcim.2c00367](https://doi.org/10.1021/acs.jcim.2c00367?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 3676−3684


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article



Table 1. Performance Comparison between GSLRDA,
LightGCN, and NGCF


methods GSLRDA LightGCN NGCF


AUC 0.9101 0.8858 0.8769

AUPR 0.9144 0.8905 0.8530


supervised learning to assist in predicting the performance of
ncRNA−resistance associations, the AUC value increased by
2.4−3.3% compared to the model using only the GCN
method. This further indicates that GSLRDA has an important
guiding role in the discovery of drug resistance-related
ncRNAs.
**Influence of Parameters.** In this work, we evaluated the
influence of parameters on the performance of the GSLRDA
model. The influence of two important parameters, the
number of GCN layers and embedding sizes were introduced.
We changed one of the parameters, kept the other parameters
unchanged, and performed 5-fold cross-validation.
_Effect of GCN Layers._ The rest of the parameters remained
unchanged, the GCN layers were selected from 1, 2, 3, or 4 to
change in turn, and the 5-fold cross-validation was used for
evaluation. The AUC and AUPR values under 5-fold crossvalidation can be found in Figure 4a. When GCN layer = 2, the
performance of the GSLRDA model was optimal. As the
number of layers increased, the performance gradually
decreased. The increase in the number of GCN layers caused
the learned feature vectors to be smooth and lose important
information. It can be seen from the results that setting the
GCN layer to 2 can solve the smoothing problem very well.
_Effect of Embedding Sizes._ The other parameters remained
unchanged, embedding sizes were chosen from 16, 32, 64, 128,
or 256, and 5-fold cross-validation was performed. The AUC
and AUPR values under 5-fold cross-validation can be found in
Figure 4b. When the embedding size increased, the performance of the GSLRDA model also increased until embedding
size = 64, and the model reached the optimum.
**Embedding Visualization.** To further explain the learning
ability of the GSLRDA model, we visualized the ncRNA and
drug features. Specifically, we first constructed 625 ncRNAs
and 121 drugs into 652 × 121 = 75 625 ncRNA−drug pairs.
There were 2693 known ncRNA−drug resistance association
pairs, and the rest were unknown association pairs. Then, we



used t-SNE [40] to visualize the features of ncRNA−drug pairs. tSNE is a technology that integrates dimensionality reduction
and visualization, which can project high-dimensional feature
vectors into a 2-dimensional or 3-dimensional space. Figure
5a,b projects 75 625 ncRNA−drug pairs embedded into 2D

−
space. Blue + represents an unknown ncRNA drug resistance
pair, and an orange dot represents the known associated pair.

−
Figure 5a shows the embedding of the initial ncRNA drug
resistance associations. Figure 5b is the embedding of the
ncRNA−drug resistance associations after learning by the
GSLRDA model. Comparing Figure 5a,b, we can see that the
GSLRDA model can better aggregate known association pairs,
making it easier to distinguish them from unknown association
pairs. In addition, we visualized the learned drug embedding
and ncRNA embedding, as shown in Figure 5c,d, respectively.
Figure 5c shows the embedding visualization of the drug node
after model learning. The drugs imatinib and etoposide are
associated with four ncRNAs of the same type. The drugs
imatinib and trastuzumab are associated with only one ncRNA
of the same type. Therefore, the drugs imatinib and etoposide
are more similar. From Figure 5c, we can see that the distance
between imatinib and etoposide is smaller. Figure 5d shows the
visualization of ncRNA node embedding after model learning.
ncRNA circPVT1 and GAS5 have two identical drug
associations, while circPVT1 and circBA9.3 do not have the
same drug association. In Figure 5d, we can see that ncRNA
circPVT1 and GAS5 are more similar. Experimental results
show that our model can effectively learn the potential features
of ncRNAs and drugs.
**Case Study.** In this section, we conduct a case study to
demonstrate the effectiveness of GSLRDA in predicting a new
association between ncRNA and drug resistance. Temozolomide [41] and 5-fluorouracil [42] (5-FU) were selected and studied.
The most widely used drug in glioblastoma [43] (GBM)
treatment is temozolomide. More than half of patients develop
resistance to temozolomide and fail treatment. 5-FU is often
used to treat colorectal cancer [44] (CRC). The human body’s
resistance to 5-FU is a major obstacle to the treatment of CRC.
Therefore, the discovery of ncRNA related to drug resistance
plays a positive role in disease treatment. For each drug, first,
we remove the ncRNA related to the drug in the data set and
treat it as a new drug. GSLRDA was implemented to predict
and sort the ncRNA of drugs in descending order according to



Figure 4. Box plot of the influence of the parameter (the cross indicates the mean value).


**3681** [https://doi.org/10.1021/acs.jcim.2c00367](https://doi.org/10.1021/acs.jcim.2c00367?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 3676−3684


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


Figure 5. Embedding visualization: (a) embedding of the initial ncRNA and drug resistance associated pair; (b) embedding of the ncRNA and drug
resistance associated pair learned by the GSLRDA model; (c) embedding of the drug node; (d) embedding of the ncRNA node.



the association score. We confirmed the top 10 ncRNAs in half of the ncRNAs of the two drugs are sufficiently proven,
PubMed. The results in Tables 2 and 3 show that more than which also shows that the GSLRDA model is good for
predicting the relationship between ncRNA and drug

Table 2. Top 10 miRNAs Related to Temozolomide resistance. It is worth noting that there is a possibility of a
Resistance Predicted by GSLRDA high correlation between unproven ncRNA and drugs, which is

worthy of further study.

ncRNA PubMed ID

# miR-181a 31190889 ■ [DISCUSSION AND CONCLUSION]


miR-146a 32973101

A large number of studies have shown that ncRNAs play a vital

miR-200c 34245265

role in drug resistance. Identifying the association between

miR-30b 33408780
miR-26b 28898169 ncRNAs and drug resistance is of great significance for

developing drugs and conducting clinical trials. Predicting the

miR-99a unconfirmed

association between ncRNAs and drug resistance based on

miR-155 32220051

computational methods is convenient and large-scale. The

miR-34a 33765907
miR-193b unconfirmed existing methods suffer from sparse supervision signals. In this
miR-210 31190889 work, we proposed a method called GSLRDA, which combines

graph neural networks and self-supervised learning. GSLRDA
uses lightGCN to learn vector representations of ncRNAs and

Table 3. Top 10 miRNAs Related to 5-Fluorouracil

drugs and uses self-supervised learning by designing auxiliary

Resistance Predicted by GSLRDA

tasks to improve the quality of the learned vector
representations of ncRNAs and drugs. The experimental

ncRNA PubMed ID

results show that GSLRDA had the best AUC value, 0.9101,

miR-99a-5p unconfirmed compared with the other excellent models. The results of
miR-21 33569416 ablation experiments show that the application of selfmiR-125b-5p 32649737 supervised learning can indeed further improve the prediction
XIST 30907503 effect of the model. In addition, case studies on two drugs were
Let-7 29330293 performed. Among the predicted top ten candidate ncRNAs,
miR-Plus-A1031 unconfirmed temozolomide and 5-fluorouracil drugs have 9 and 7 associated
miR-146b-5pmiR-191* unconfirmed29737579 ncRNAs, respectively, which have been validated by previousstudies. The complex experimental results show that GSLRDA
CASC11 unconfirmed is a reliable predictor of potential ncRNA and drug resistance.
miR-1256 unconfirmed Although the use of self-supervised learning alleviates the



Table 2. Top 10 miRNAs Related to Temozolomide
Resistance Predicted by GSLRDA



ncRNA PubMed ID



miR-181a 31190889

miR-146a 32973101

miR-200c 34245265

miR-30b 33408780

miR-26b 28898169

miR-99a unconfirmed

miR-155 32220051

miR-34a 33765907

miR-193b unconfirmed

miR-210 31190889



Table 3. Top 10 miRNAs Related to 5-Fluorouracil
Resistance Predicted by GSLRDA



ncRNA PubMed ID



miR-99a-5p unconfirmed

miR-21 33569416

miR-125b-5p 32649737

XIST 30907503

Let-7 29330293

miR-Plus-A1031 unconfirmed
miR-146b-5p unconfirmed

miR-191* 29737579

CASC11 unconfirmed
miR-1256 unconfirmed



**3682** [https://doi.org/10.1021/acs.jcim.2c00367](https://doi.org/10.1021/acs.jcim.2c00367?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 3676−3684


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article



−
problem of data sparseness, there are still few known ncRNA
drug resistance associations. In future work, we will collect
more ncRNA−drug resistance associations and adopt more
data augmentation methods to explore the information of
graphs better and enhance the performance. In addition, we
will also pay more attention to the pretraining of tasks to
improve the transferability of the model.

# ■ [DATA AND SOFTWARE AVAILABILITY]

[The code and data sets of GSLRDA are available at https://](https://github.com/JJZ-code/GSLRDA)
[github.com/JJZ-code/GSLRDA.](https://github.com/JJZ-code/GSLRDA)

# ■ [AUTHOR INFORMATION]

**[Corresponding Author](https://pubs.acs.org/action/doSearch?field1=Contrib&text1="Lei+Deng"&field2=AllField&text2=&publication=&accessType=allContent&Earliest=&ref=pdf)**

Lei Deng − _School of Software, Xinjiang University, Urumqi_
_830091, China; School of Computer Science and Engineering,_
_Central South University, Changsha 410083, China_ ;
[Email: leideng@csu.edu.cn](mailto:leideng@csu.edu.cn)


**[Authors](https://pubs.acs.org/action/doSearch?field1=Contrib&text1="Jingjing+Zheng"&field2=AllField&text2=&publication=&accessType=allContent&Earliest=&ref=pdf)**

Jingjing Zheng − _School of Software, Xinjiang University,_
_Urumqi 830091, China;_ [orcid.org/0000-0002-8549-](https://orcid.org/0000-0002-8549-4235)
[4235](https://orcid.org/0000-0002-8549-4235)
Yurong Qian − _School of Software, Xinjiang University,_
_[Urumqi 830091, China](https://pubs.acs.org/action/doSearch?field1=Contrib&text1="Jie+He"&field2=AllField&text2=&publication=&accessType=allContent&Earliest=&ref=pdf)_
Jie He − _School of Computer Science and Engineering, Central_
_South University, Changsha 410083, China_
Zerui Kang − _School of Computer Science and Engineering,_
_Central South University, Changsha 410083, China_


Complete contact information is available at:
[https://pubs.acs.org/10.1021/acs.jcim.2c00367](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00367?ref=pdf)


**Funding**
This work was supported by National Natural Science
Foundation of China under Grant No. 61972422.

**Notes**
The authors declare no competing financial interest.

# ■ [ACKNOWLEDGMENTS]

The work was carried out at National Supercomputer Center
in Tianjin, and the calculations were performed on Tianhe new
generation Supercomputer.

# ■ [REFERENCES]


[(1) Slack, F. J.; Chinnaiyan, A. M. The Role of Non-coding RNAs in](https://doi.org/10.1016/j.cell.2019.10.017)
[Oncology.](https://doi.org/10.1016/j.cell.2019.10.017) _Cell_ 2019, _179_, 1033−1055.
(2) Ferreira, D.; Escudeiro, A.; Adega, F.; Anjo, S. I.; Manadas, B.;
[Chaves, R. FA-SAT ncRNA interacts with PKM2 protein: Depletion](https://doi.org/10.1007/s00018-019-03234-x)
[of this complex induces a switch from cell proliferation to apoptosis.](https://doi.org/10.1007/s00018-019-03234-x)
_Cell. Mol. Life Sci._ 2020, _77_, 1371−1386.
[(3) Anastasiadou, E.; Jacob, L. S.; Slack, F. J. Non-coding RNA](https://doi.org/10.1038/nrc.2017.99)
[networks in cancer.](https://doi.org/10.1038/nrc.2017.99) _Nature Reviews Cancer_ 2018, _18_, 5−18.
(4) Liu, Y.; Liu, X.; Lin, C.; Jia, X.; Zhu, H.; Song, J.; Zhang, Y.
[Noncoding RNAs regulate alternative splicing in Cancer.](https://doi.org/10.1186/s13046-020-01798-2) _J Exp Clin_
_Cancer Res._ 2021, _40_, 11.
[(5) Cech, T. R.; Steitz, J. A. The noncoding RNA revolution-trashing](https://doi.org/10.1016/j.cell.2014.03.008)
[old rules to forge new ones.](https://doi.org/10.1016/j.cell.2014.03.008) _Cell_ 2014, _157_, 77−94.
[(6) Dhamija, S.; Diederichs, S. From junk to master regulators of](https://doi.org/10.1002/ijc.30039)
[invasion: lncRNA functions in migration, EMT and metastasis.](https://doi.org/10.1002/ijc.30039)
_International journal of cancer_ 2016, _139_, 269−280.
(7) Hardin, H.; Helein, H.; Meyer, K.; Robertson, S.; Zhang, R.;
[Zhong, W.; Lloyd, R. V. Thyroid cancer stem-like cell exosomes:](https://doi.org/10.1038/s41374-018-0065-0)
[regulation of EMT via transfer of lncRNAs.](https://doi.org/10.1038/s41374-018-0065-0) _Laboratory Investigation_
2018, _98_, 1133−1142.



(8) Nunez Lopez, Y. O. N.; Victoria, B.; Golusinski, P.; Golusinski,
[W.; Masternak, M. M. Characteristic miRNA expression signature and](https://doi.org/10.1016/j.rpor.2017.10.003)
[random forest survival analysis identify potential cancer-driving](https://doi.org/10.1016/j.rpor.2017.10.003)
[miRNAs in a broad range of head and neck squamous cell carcinoma](https://doi.org/10.1016/j.rpor.2017.10.003)
[subtypes.](https://doi.org/10.1016/j.rpor.2017.10.003) _Reports of Practical Oncology and Radiotherapy_ 2018, _23_, 6−
20.
[(9) Chen, B.; Huang, S. Circular RNA: an emerging non-coding](https://doi.org/10.1016/j.canlet.2018.01.011)
[RNA as a regulator and biomarker in cancer.](https://doi.org/10.1016/j.canlet.2018.01.011) _Cancer letters_ 2018, _418_,
41−50.
[(10) Ojha, R.; Nandani, R.; Chatterjee, N.; Prajapati, V. K. Emerging](https://doi.org/10.1007/978-981-13-1426-1_12)
[role of circular RNAs as potential biomarkers for the diagnosis of](https://doi.org/10.1007/978-981-13-1426-1_12)
[human diseases.](https://doi.org/10.1007/978-981-13-1426-1_12) _Adv Exp Med Biol._ 2018, _1087_, 141−157.
(11) Siegel, R. L.; Miller, K. D.; Sauer, A. G.; Fedewa, S. A.; Butterly,
[L. F.; Anderson, J. C.; Cercek, A.; Smith, R. A.; Jemal, A. Colorectal](https://doi.org/10.3322/caac.21601)
[cancer statistics, 2020.](https://doi.org/10.3322/caac.21601) _CA: A Cancer Journal for Clinicians_ 2020, _70_,
145.
(12) Gao, D.; Zhang, X.; Liu, B.; Meng, D.; Fang, K.; Guo, Z.; Li, L.
[Screening circular RNA related to chemotherapeutic resistance in](https://doi.org/10.2217/epi-2017-0055)
[breast cancer.](https://doi.org/10.2217/epi-2017-0055) _Epigenomics_ 2017, _9_, 1175−1188.
(13) Hussen, B. M.; Azimi, T.; Hidayat, H. J.; Taheri, M.; Ghafouri[Fard, S. NF-KappaB interacting LncRNA: review of its roles in](https://doi.org/10.1016/j.biopha.2021.111604)
[neoplastic and non-neoplastic conditions.](https://doi.org/10.1016/j.biopha.2021.111604) _Biomedicine & Pharmaco-_
_therapy_ 2021, _139_, 111604.
(14) Pan, J.-A.; Tang, Y.; Yu, J.-Y.; Zhang, H.; Zhang, J.-F.; Wang,
[C.-Q.; Gu, J. miR-146a attenuates apoptosis and modulates autophagy](https://doi.org/10.1038/s41419-019-1901-x)
[by targeting TAF9b/P53 pathway in doxorubicin-induced cardiotox-](https://doi.org/10.1038/s41419-019-1901-x)
[icity.](https://doi.org/10.1038/s41419-019-1901-x) _Cell Death Disease_ 2019, _10_, 668.
(15) Zhao, L.; Qi, Y.; Xu, L.; Tao, X.; Han, X.; Yin, L.; Peng, J.
[MicroRNA-140-5p aggravates doxorubicin-induced cardiotoxicity by](https://doi.org/10.1016/j.redox.2017.12.013)
[promoting myocardial oxidative stress via targeting Nrf2 and Sirt2.](https://doi.org/10.1016/j.redox.2017.12.013)
_Redox biology_ 2018, _15_, 284−296.
(16) Shang, J.; Chen, W.-M.; Liu, S.; Wang, Z.-H.; Wei, T.-N.; Chen,
[Z.-Z.; Wu, W.-B. CircPAN3 contributes to drug resistance in acute](https://doi.org/10.1016/j.leukres.2019.106198)
[myeloid leukemia through regulation of autophagy.](https://doi.org/10.1016/j.leukres.2019.106198) _Leukemia Research_
2019, _85_, 106198.
[(17) Li, Y.; Wang, R.; Zhang, S.; Xu, H.; Deng, L. LRGCPND:](https://doi.org/10.3390/ijms221910508)
[Predicting Associations between ncRNA and Drug Resistance via](https://doi.org/10.3390/ijms221910508)
[Linear Residual Graph Convolution.](https://doi.org/10.3390/ijms221910508) _International Journal of Molecular_
_Sciences_ 2021, _22_, 10508.
[(18) Dayun, L.; Junyi, L.; Yi, L.; Qihua, H.; Deng, L. MGATMDA:](https://doi.org/10.1109/TCBB.2021.3116318)
[Predicting microbe-disease associations via multi-component graph](https://doi.org/10.1109/TCBB.2021.3116318)
[attention network.](https://doi.org/10.1109/TCBB.2021.3116318) _IEEE/ACM Transactions on Computational Biology_
_and Bioinformatics_ 2021, 1−1.
[(19) Ji, C.; Liu, Z.; Wang, Y.; Ni, J.; Zheng, C. GATNNCDA: A](https://doi.org/10.3390/ijms22168505)
[Method Based on Graph Attention Network and Multi-Layer Neural](https://doi.org/10.3390/ijms22168505)
[Network for Predicting circRNA-Disease Associations.](https://doi.org/10.3390/ijms22168505) _International_
_Journal of Molecular Sciences_ 2021, _22_, 8505.
[(20) Deng, L.; Huang, Y.; Liu, X.; Liu, H. Graph2MDA: a multi-](https://doi.org/10.1093/bioinformatics/btab792)

−
[modal variational graph embedding model for predicting microbe](https://doi.org/10.1093/bioinformatics/btab792)
[drug associations.](https://doi.org/10.1093/bioinformatics/btab792) _Bioinformatics_ 2022, _38_, 1118−1125.
[(21) Zhao, X.; Zhao, X.; Yin, M. Heterogeneous graph attention](https://doi.org/10.1093/bib/bbab407)
[network based on meta-paths for lncRNA−disease association](https://doi.org/10.1093/bib/bbab407)
[prediction.](https://doi.org/10.1093/bib/bbab407) _Briefings in Bioinformatics_ 2022, _23_, bbab407.
[(22) Fan, Y.; Chen, M.; Pan, X. GCRFLDA: scoring lncRNA-disease](https://doi.org/10.1093/bib/bbab361)
[associations using graph convolution matrix completion with](https://doi.org/10.1093/bib/bbab361)
[conditional random field.](https://doi.org/10.1093/bib/bbab361) _Briefings in Bioinformatics_ 2022, _23_,
bbab361.
(23) Wang, L.; You, Z.-H.; Li, Y.-M.; Zheng, K.; Huang, Y.-A.
[GCNCDA: a new method for predicting circRNA-disease associations](https://doi.org/10.1371/journal.pcbi.1007568)
[based on graph convolutional network algorithm.](https://doi.org/10.1371/journal.pcbi.1007568) _PLOS Computa-_
_tional Biology_ 2020, _16_, No. e1007568.
(24) Lan, W.; Dong, Y.; Chen, Q.; Liu, J.; Wang, J.; Chen, Y.-P. P.;
[Pan, S. IGNSCDA: predicting CircRNA-disease associations based on](https://doi.org/10.1109/TCBB.2021.3111607)
[improved graph convolutional network and negative sampling.](https://doi.org/10.1109/TCBB.2021.3111607) _IEEE/_
_ACM Transactions on Computational Biology and Bioinformatics_ 2021,
3111607.
(25) Li, H.-Y.; Chen, H.-Y.; Wang, L.; Song, S.-J.; You, Z.-H.; Yan,
[X.; Yu, J.-Q. A structural deep network embedding model for](https://doi.org/10.1038/s41598-021-91991-w)


**3683** [https://doi.org/10.1021/acs.jcim.2c00367](https://doi.org/10.1021/acs.jcim.2c00367?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 3676−3684


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article



[predicting associations between miRNA and disease based on](https://doi.org/10.1038/s41598-021-91991-w)
[molecular association network.](https://doi.org/10.1038/s41598-021-91991-w) _Sci. Rep._ 2021, _11_, 12640.
[(26) Li, L.; Wu, P.; Wang, Z.; Meng, X.; Cai, J.; et al. NoncoRNA: a](https://doi.org/10.1186/s13045-020-00849-7)
[database of experimentally supported non-coding RNAs and drug](https://doi.org/10.1186/s13045-020-00849-7)
[targets in cancer.](https://doi.org/10.1186/s13045-020-00849-7) _Journal of Hematology Oncology_ 2020, _13_, 15.
(27) Dai, E.; Yang, F.; Wang, J.; Zhou, X.; Song, Q.; An, W.; Wang,
[L.; Jiang, W. ncDR: a comprehensive resource of non-coding RNAs](https://doi.org/10.1093/bioinformatics/btx523)
[involved in drug resistance.](https://doi.org/10.1093/bioinformatics/btx523) _Bioinformatics_ 2017, _33_, 4010−4011.
[(28) Li, X.; Zhang, H.; Wang, R.; Nie, F. Multiview Clustering: A](https://doi.org/10.1109/TPAMI.2020.3011148)
[Scalable and Parameter-Free Bipartite Graph Fusion Method.](https://doi.org/10.1109/TPAMI.2020.3011148) _IEEE_
_Transactions on Pattern Analysis and Machine Intelligence_ 2022, _44_,
330−344.
[(29) Hanley, J. A.; McNeil, B. J. The meaning and use of the area](https://doi.org/10.1148/radiology.143.1.7063747)
[under a receiver operating characteristic (ROC) curve.](https://doi.org/10.1148/radiology.143.1.7063747) _Radiology_
1982, _143_, 29−36.
[(30) Liu, D.; Huang, Y.; Nie, W.; Zhang, J.; Deng, L. SMALF:](https://doi.org/10.1186/s12859-021-04135-2)
[miRNA-disease associations prediction based on stacked autoencoder](https://doi.org/10.1186/s12859-021-04135-2)
[and XGBoost.](https://doi.org/10.1186/s12859-021-04135-2) _BMC bioinformatics_ 2021, _22_, 219.
(31) Zeng, M.; Lu, C.; Zhang, F.; Li, Y.; Wu, F.-X.; Li, Y.; Li, M.
[SDLDA: lncRNA-disease association prediction based on singular](https://doi.org/10.1016/j.ymeth.2020.05.002)
[value decomposition and deep learning.](https://doi.org/10.1016/j.ymeth.2020.05.002) _Methods_ 2020, _179_, 73−80.
[(32) Lu, C.; Zeng, M.; Zhang, F.; Wu, F.-X.; Li, M.; Wang, J. Deep](https://doi.org/10.1109/JBHI.2020.2999638)
[matrix factorization improves prediction of human circRNA-disease](https://doi.org/10.1109/JBHI.2020.2999638)
[associations.](https://doi.org/10.1109/JBHI.2020.2999638) _IEEE Journal of Biomedical and Health Informatics_ 2021,
_25_, 891−899.
(33) Liu, Y.; Wang, S.-L.; Zhang, J.-F.; Zhang, W.; Zhou, S.; Li, W.
[DMFMDA: Prediction of microbe-disease associations based on deep](https://doi.org/10.1109/TCBB.2020.3018138)
[matrix factorization using Bayesian Personalized Ranking.](https://doi.org/10.1109/TCBB.2020.3018138) _IEEE/ACM_
_Transactions on Computational Biology and Bioinformatics_ 2021, _18_,
1763.
[(34) Chen, X.; Huang, Y.-A.; You, Z.-H.; Yan, G.-Y.; Wang, X.-S. A](https://doi.org/10.1093/bioinformatics/btw715)
[novel approach based on KATZ measure to predict associations of](https://doi.org/10.1093/bioinformatics/btw715)
[human microbiota with non-infectious diseases.](https://doi.org/10.1093/bioinformatics/btw715) _Bioinformatics_ 2017,
_33_, 733−739.
[(35) Luo, J.; Long, Y. NTSHMDA: prediction of human microbe-](https://doi.org/10.1109/TCBB.2018.2883041)
[disease association based on random walk by integrating network](https://doi.org/10.1109/TCBB.2018.2883041)
[topological similarity.](https://doi.org/10.1109/TCBB.2018.2883041) _IEEE/ACM transactions on computational biology_
_and bioinformatics_ 2018, _17_, 1341−1351.
[(36) Deepthi, K.; Jereesh, A. Inferring potential CircRNA−disease](https://doi.org/10.1007/s40291-020-00499-y)
[associations via deep autoencoder-based classification.](https://doi.org/10.1007/s40291-020-00499-y) _Molecular_
_Diagnosis & Therapy_ 2021, _25_, 87−97.
[(37) Peng, L.-H.; Yin, J.; Zhou, L.; Liu, M.-X.; Zhao, Y. Human](https://doi.org/10.3389/fmicb.2018.02440)
[microbe-disease association prediction based on adaptive boosting.](https://doi.org/10.3389/fmicb.2018.02440)
_Frontiers in microbiology_ 2018, _9_, 2440.
(38) He, X.; Deng, K.; Wang, X.; Li, Y.; Zhang, Y.; Wang, M.
LightGCN: Simplifying and Powering Graph Convolution Network
for Recommendation. _arXiV_ [2020, https://arxiv.org/abs/2002.02126.](https://arxiv.org/abs/2002.02126)
[(39) Wang, X.; He, X.; Wang, M.; Feng, F.; Chua, T. S. Neural](https://doi.org/10.1145/3331184.3331267)
[Graph Collaborative Filtering.](https://doi.org/10.1145/3331184.3331267) _SIGIR’19: Proc. 42nd International_
_ACM SIGIR Conference_ 2019, 165.
[(40) Wattenberg, M.; Viégas, F.; Johnson, I. How to use t-SNE](https://doi.org/10.23915/distill.00002)
[effectively.](https://doi.org/10.23915/distill.00002) _Distill_ 2016, _1_, No. e2.
(41) Karachi, A.; Dastmalchi, F.; Mitchell, D. A.; Rahman, M.
[Temozolomide for immunomodulation in the treatment of](https://doi.org/10.1093/neuonc/noy072)
[glioblastoma.](https://doi.org/10.1093/neuonc/noy072) _Neuro-oncology_ 2018, _20_, 1566−1572.
[(42) Cameron, D.; Gabra, H.; Leonard, R. Continuous 5-fluorouracil](https://doi.org/10.1038/bjc.1994.259)
[in the treatment of breast cancer.](https://doi.org/10.1038/bjc.1994.259) _British journal of cancer_ 1994, _70_,
120−124.
[(43) Wirsching, H.-G.; Weller, M. Glioblastoma.](https://doi.org/10.1007/978-3-319-49864-5_18) _Malignant Brain_
_Tumors_ 2017, 265−288.
(44) Vodenkova, S.; Buchler, T.; Cervena, K.; Veskrnova, V.;
[Vodicka, P.; Vymetalkova, V. 5-fluorouracil and other fluoropyr-](https://doi.org/10.1016/j.pharmthera.2019.107447)
[imidines in colorectal cancer: Past, present and future.](https://doi.org/10.1016/j.pharmthera.2019.107447) _Pharmacology_
_& therapeutics_ 2020, _206_, 107447.


#### **Recommended by ACS**

**[ReLMole](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00798?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** : **[Molecular](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00798?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Representation](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00798?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Learning](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00798?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Based](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00798?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **on**
**[Two](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00798?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)**   - **[Level](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00798?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Graph](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00798?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Similarities](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00798?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)**


Zewei Ji, Yang Yang, et al.


OCTOBER 27, 2022

JOURNAL OF CHEMICAL INFORMATION AND MODELING [READ](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00798?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)


**[MGCVAE](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00487?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** : **[Multi](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00487?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)**  - **[Objective](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00487?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Inverse](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00487?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Design](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00487?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[via](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00487?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Molecular](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00487?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)**
**[Graph](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00487?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Conditional](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00487?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Variational](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00487?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Autoencoder](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00487?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)**


Myeonghun Lee and Kyoungmin Min


JUNE 06, 2022

JOURNAL OF CHEMICAL INFORMATION AND MODELING [READ](http://pubs.acs.org/doi/10.1021/acs.jcim.2c00487?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)


**[Flexible](http://pubs.acs.org/doi/10.1021/acsomega.1c05877?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Dual](http://pubs.acs.org/doi/10.1021/acsomega.1c05877?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)**    - **[Branched](http://pubs.acs.org/doi/10.1021/acsomega.1c05877?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Message](http://pubs.acs.org/doi/10.1021/acsomega.1c05877?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)**    - **[Passing](http://pubs.acs.org/doi/10.1021/acsomega.1c05877?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Neural](http://pubs.acs.org/doi/10.1021/acsomega.1c05877?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Network](http://pubs.acs.org/doi/10.1021/acsomega.1c05877?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[for](http://pubs.acs.org/doi/10.1021/acsomega.1c05877?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)**
**a** **[Molecular](http://pubs.acs.org/doi/10.1021/acsomega.1c05877?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Property](http://pubs.acs.org/doi/10.1021/acsomega.1c05877?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Prediction](http://pubs.acs.org/doi/10.1021/acsomega.1c05877?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)**


Jeonghee Jo, Sungroh Yoon, et al.


JANUARY 27, 2022

ACS OMEGA [READ](http://pubs.acs.org/doi/10.1021/acsomega.1c05877?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)


**[Improving](http://pubs.acs.org/doi/10.1021/acsomega.1c06805?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Compound](http://pubs.acs.org/doi/10.1021/acsomega.1c06805?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Activity](http://pubs.acs.org/doi/10.1021/acsomega.1c06805?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Classification](http://pubs.acs.org/doi/10.1021/acsomega.1c06805?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[via](http://pubs.acs.org/doi/10.1021/acsomega.1c06805?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Deep](http://pubs.acs.org/doi/10.1021/acsomega.1c06805?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)**
**[Transfer](http://pubs.acs.org/doi/10.1021/acsomega.1c06805?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[and](http://pubs.acs.org/doi/10.1021/acsomega.1c06805?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Representation](http://pubs.acs.org/doi/10.1021/acsomega.1c06805?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)** **[Learning](http://pubs.acs.org/doi/10.1021/acsomega.1c06805?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)**


Vishal Dey, Xia Ning, et al.


MARCH 11, 2022

ACS OMEGA [READ](http://pubs.acs.org/doi/10.1021/acsomega.1c06805?utm_campaign=RRCC_jcisd8&utm_source=RRCC&utm_medium=pdf_stamp&originated=1667993122&referrer_DOI=10.1021%2Facs.jcim.2c00367)


**[Get More Suggestions >](https://preferences.acs.org/ai_alert?follow=1)**


**3684** [https://doi.org/10.1021/acs.jcim.2c00367](https://doi.org/10.1021/acs.jcim.2c00367?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 3676−3684


