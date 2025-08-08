pubs.acs.org/jcim Article

## **Dual-Channel Heterogeneous Graph Neural Network for Predicting** **microRNA-Mediated Drug Sensitivity**

#### Lei Deng, Ziyu Fan, Xiaojun Xiao, Hui Liu,* and Jiaxuan Zhang

### ACCESS Metrics & More Article Recommendations * sı Supporting Information


# ■ [INTRODUCTION] tamoxifen-resistant breast cancer. [18] Low expression of miR
17 and miR-20B decreased the sensitivity to Paclitaxel in breast

The sensitivity of cancer cells to drugs is a crucial factor for

cancer through upregulation of nuclear receptor coactivator 3

effective treatments. [1] Resistance to anticancer drugs arises

(NCOA3) levels. [22]

from a wide range of causes, such as genetic mutations and/or
epigenetic changes that allow cancer cells to avoid With the development of biochemical techniques, large-scale
programmed cell death, activity variation of transport proteins experiments have confirmed that many miRNAs are closely
that decrease the intracellular drug concentration, and other related to the sensitivity of anticancer drugs. Two databases,
cellular and molecular mechanisms. [2] Systematic understanding NoncoRNA [23] and ncDR, [24] have integrated the miRNA-drug
of the drug response mechanism would greatly facilitate the sensitivity associations from the biomedical literature. Nondevelopment of novel therapeutic strategies and lead to better coRNA [23] is a manually curated database of experimentally
clinical outcomes. [3]



MicroRNAs (miRNAs), a type of small noncoding RNAs

−
with a length of about 19 25nt, [4][,][5] participate in various
cellular processes such as cell formation, differentiation,
proliferation, and apoptosis. [6][,][7] Most miRNAs interact with
the 3′ untranslated region (3′ UTR) of target mRNAs to
induce their degradation and translational repression. [8][−][14] In
recent years, many studies have shown that miRNAs play a
non-negligible role in the regulation of drug response among
different cancers. [4][,][15][,][16] Especially, some miRNAs are potential
therapeutic targets to enhance drug sensitivity. [17][−][19] For
example, inhibition of miR-21 and miR-200b can increase
sensitivity to Gemcitabine in human cholangiocarcinoma
cells. [20] Upregulation of miR-155-5p enhanced the sensitivity
of liver carcinoma cells to Adriamycin and promoted apoptosis
through the inhibition of autophagy in vitro. [21] MiR-21, miR146a, miR-148a, miR-34a, and miR-27a have been shown to
play an essential role in mediating tamoxifen sensitivity in
breast cancer and is a potential therapeutic target for


© 2022 American Chemical Society



**5929**



tamoxifen-resistant breast cancer. [18] Low expression of miR17 and miR-20B decreased the sensitivity to Paclitaxel in breast
cancer through upregulation of nuclear receptor coactivator 3
(NCOA3) levels. [22]

With the development of biochemical techniques, large-scale
experiments have confirmed that many miRNAs are closely
related to the sensitivity of anticancer drugs. Two databases,
NoncoRNA [23] and ncDR, [24] have integrated the miRNA-drug
sensitivity associations from the biomedical literature. NoncoRNA [23] is a manually curated database of experimentally
supported noncoding RNAs (ncRNAs) and drug target
associations that aims to provide a high-quality data resource
for exploring drug sensitivity/resistance-related ncRNAs in
various human cancers. The ncDR database [24] collected both
experimentally validated and predicted drug resistanceassociated microRNAs and long noncoding RNAs through
manual curation and computational analysis. The data sources
enable us to develop computational methods to predict the
miRNA-mediated drug efficacy effectively.
Many computational methods formulated the association
between miRNA and drug resistance, but few with regard to





[https://doi.org/10.1021/acs.jcim.2c01060](https://doi.org/10.1021/acs.jcim.2c01060?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 5929−5937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


Figure 1. Illustrative flowchart of the proposed DGNNMDA model for miRNA and drug representation learning on heterogeneous graph neural
network. The miRNAs and the drugs were encoded and mapped to the same feature space. Next, feature propagations on homogeneous and
heterogeneous networks were successively performed to learn the representations of miRNAs and drugs. The learned embeddings were used to
infer the associations between miRNAs and drugs.



drug sensitivity. For example, GCMDR [25] made use of graph
convolution to build a three-layer latent factor model to
predict the associations between miRNA and drug resistance.
AMMGC [26] is an attentive multimodal graph convolution
network method to predict miRNA-drug resistance associations, by learning the latent representations of drugs and
miRNAs from four graph convolution subnetworks with
distinctive combinations of features. Xu et al. [27] predicted the
miRNA-drug efficacy associations using the Bi-Random walk
(BiRW) algorithm on the miRNA-drug heterogeneous network. LRGCPND [28] leveraged the verified associations of
ncRNAs and drug resistance to construct a bipartite graph and
then developed a linear residual graph convolution approach
for predicting ncRNA-associated drug resistance. To date, only
one computational method has been proposed to predict the
associations between miRNA and drug sensitivity. LGCMDS [29]

employed LightGCN, which only retains the neighborhood
aggregation component of graph convolutional network
(GCN), to predict miRNA and drug sensitivity associations.
From its experimental results, there is still much room for
improvement of the prediction accuracy. Moreover, with the
release of new data of experimentally validated miRNA-drug
sensitivity associations, it is necessary to develop a reliable and
accurate model to predict miRNA-drug sensitivity associations.
In this paper, we focused on the prediction of miRNAmediated drug sensitivity. We proposed a dual-channel graph
neural network on the heterogeneous network, DGNNMDA,
to predict the miRNA-drug sensitivity associations. By
integrating miRNA similarity network, drug similarity network,
and known miRNA-drug sensitivity associations, we constructed an miRNA-drug heterogeneous network. Next, we
proposed feature propagation on the homogeneous and
heterogeneous networks to learn the features of miRNA and
drug nodes. The multilayer heterogeneous graph aggregation
operations effectively obtained the information from highorder neighbor miRNAs and drugs to update node embedding.
In addition, we used a random sampling strategy to increase
the generalization ability during information aggregation. To
evaluate the performance of our method, we built a benchmark
dataset from NoncoRNA and ncDR databases, and a manually



curated independent dataset from the PubMed literature. On
both datasets, we verified that our method achieved better
performance than seven competitive methods. The model
ablation experiments verified the effectiveness of the feature
propagation on homogeneous and heterogeneous networks.
We also showed that our method was robust to the data
imbalances. Finally, two case studies on two cytotoxic drugs,
Paclitaxel and Cisplatin, were conducted. Apart from the
publications supporting the predicted miRNA-drug associations, we also constructed the regulatory network consisting of
drugs, miRNAs, and target mRNAs to explore the regulatory
mechanism of miRNA mediated in the drug sensitivity.

# ■ [MATERIALS AND METHODS]


**Data Source.** The miRNA-drug sensitivity associations
were obtained from the NoncoRNA [23] and ncDR [24] databases.
The ncDR database includes both experimentally validated and
predicted associations between 140 drugs and 1039 ncRNAs.
The NoncoRNA database collects experimentally supported
associations between 5568 ncRNAs and 154 drugs from 134
types of cancers. We chose only the experimentally validated
associations between miRNAs and drug sensitivity in our
study, namely, the associations involved drug resistance or
non-miRNAs were excluded. In total, 2049 miRNA-drug
sensitivity associations between 431 miRNAs and 140 drugs
were obtained. The dataset was used as the benchmark dataset
to evaluate the performance of our method.
To further verify the reliability of our method, another
manually curated dataset was built by collecting validated
miRNA-drug sensitivity associations from PubMed publications. We used the keywords “miR-/miRNA/microRNA”,
“sensitive/sensitivity” and individual drug names to retrieve
publications, and manually checked whether the miRNAs
mediated in drug sensitivity or not. As a result, we collected
350 miRNA-drug sensitivity associations that were not
included in ncRNAs/NoncoRNA databases. As far as our
knowledge, this is the largest manually curated dataset of
miRNA-drug sensitivity associations.
**Construction of miRNA-Drug Heterogeneous Net-**
**work.** _miRNA Similarity Network._ We constructed the


**5930** [https://doi.org/10.1021/acs.jcim.2c01060](https://doi.org/10.1021/acs.jcim.2c01060?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 5929−5937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article



miRNA similarity network by calculating the sequence
similarity between miRNAs. The miRNA sequences were
obtained from the miRBase database. [30] The miRNA similarity
was computed as the Levenshtein distance, [31] which is defined
as the number of deletions, insertions, or substitutions required
to transform one sequence to the other one (for more details,
[see Section S1.1 in the Supporting Information).](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.2c01060/suppl_file/ci2c01060_si_001.pdf)
For the similarities of miRNA pairs, we found that its
distribution nearly followed the normal distribution. We chose
the top 5% miRNA pairs with the highest similarities and
obtained 7812 pairs. We counted the number of edges of these
miRNAs, and found that the degree of most miRNAs was less
than 25. Thereby, we decided that the number of connecting
edges of each miRNA should not exceed 25. In this way, we
constructed the miRNA similarity network, and the representation of each miRNA node aggregated the information
[from its neighbor miRNA nodes (for more details, see Section](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.2c01060/suppl_file/ci2c01060_si_001.pdf)
[S1.2 in the Supporting Information).](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.2c01060/suppl_file/ci2c01060_si_001.pdf)
_Drug Similarity Network._ For each drug, we obtained the
canonical SMILES of the drug from the PubChem database. [32]

Next, we extracted the MACCS fingerprints and calculate the
similarity for each drug pair. To construct the drug similarity
network, we chose the top 5% drugs with the highest similarity
and obtained 1333 drug pairs. As a result, most of drugs had
less than 10 neighbors, and we determined that the edge
number of each drug should not exceed 10 (for more details,
[see Section S1.2 in the Supporting Information).](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.2c01060/suppl_file/ci2c01060_si_001.pdf)
_miRNA-Drug Heterogeneous Network._ We integrated the
miRNA similarity network, drug similarity network, and known
miRNA-drug sensitivity associations to construct an miRNAdrug heterogeneous network. Formally, denoted by _B_ the
heterogeneous network, we have



ÄÅÅÅÅÅÅÅÅÅÅÅÇ



T _S_ d Ö (1)



É



Ö



_S_ m _A_
_B_ =

ÅÅÅÅÅÅÅÅÅÅÅ _A_ T _S_ d ÑÑÑÑÑÑÑÑÑÑÑ



m



hot coding, and projected them to a few principal components
to obtain low-dimensional vectors, denoted by x (0) _i_ and y (0) _j_ .
**Dual-Channel Heterogeneous Graph Representation**
**Learning.** In each GNN layer, the feature of each node was
updated by aggregating neighbor node features. [33] However,
existing GNN models neglect the heterogeneity when
aggregating neighbor information. To fully learn the complex
local structure and semantic associations of heterogeneous
networks, we proposed the dual-channel heterogeneous graph
learning to learn node representations.
_Feature Propagation on Homogeneous Network._ We first
performed feature aggregation among homogeneous nodes,
namely, message propagation run only on miRNA and drug
similarity networks separately. Formally, for the miRNA
channel, denoted by x ( _i_ _l_ ) the embedding of miRNA _i_ obtained
at the _l_ -th layer, the output value of ( _l_ + 1)-th layer is obtained
by aggregating the features of its neighbor miRNAs as below


**x** ( _l_ + _Si_ 1) = AGG( **x** ( ) _jl_ | _j_ _Si_ ( )) _m_ (2)


where **x** [(] _[l]_ + _S_ 1) is the aggregated features from the homogeneous

_i_


_S_
neighbor nodes of miRNA _i_ ; _i_ ( ) _m_ represents the neighbors
of miRNA _i_ in the miRNA similarity network; AGG is the
aggregation function that could be max pooling, average
pooling or similarity-weighted average pooling. In our study,
we used average pooling to aggregate the neighborhood
features. Next, we applied a nonlinear transform as below


**x** ( _i_ _l_ +1) = _f_ ( **w** _m_    - [ **x** ( ) _i_ _l_, **x** ( _l_ + _S_ 1) ])

_i_ (3)


where _f_ () is the nonlinear Leaky ReLU function and _w_ m is a
learnable vector parameter used to convey the importance of
features of miRNA _i_ itself and its neighbor nodes.
For the drug channel, we ran similar feature propagations on
the drug similarity network as below

**y** ( _l_ + _Sj_ 1) = AGG( **y** ( ) _kl_ | _k_ _Sj_ ( )) _d_ (4)


**y** ( _jl_ +1) = _f_ ( **w y** _d_    - [ ( ) _jl_, **y** ( _l_ + _Sj_ 1) ]) (5)


where y ( _j_ _l_ ) is the output feature of drug _j_ at _l_ -th layer, _Sj_ ( ) _d_

represents the neighbors of drug _j_ in the drug similarity
network, and _w_ d is a learnable vector parameter used to convey
the importance of features of drug _j_ itself and its neighbor
nodes.
The initial features x _i_ and y _j_ of miRNA _i_ and drug _j_ were set
to the mapped value from their one-hot encoding processed by
PCA, respectively. After _L_ -layer message propagations, we
aggregated the features of miRNAs and drugs on their
similarity network, respectively. _L_ is a hyperparameter that
could be tuned to optimize performance.
_Feature Propagation on Heterogeneous Network._ Based
on the aggregated features from the homogeneous network, we
performed feature propagation among heterogeneous nodes
based on the aggregated features on the similarity network.
Denote by g [(] _[l]_ [)] and h [(] _[l]_ [)] the features of miRNA _i_ and drug _j_
aggregated from heterogeneous neighbor nodes, respectively.
Their initial values were set as g [(0)] = x ( _i_ _L_ ) and h (0) = y ( _j_ _L_ ) . Let us
first consider the miRNA-centric feature propagation. For
miRNA _i_, we computed the aggregated feature on the
heterogeneous network as below


**5931** [https://doi.org/10.1021/acs.jcim.2c01060](https://doi.org/10.1021/acs.jcim.2c01060?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 5929−5937



_A_ _S_



in which _S_ m and _S_ d are the miRNA and drug similarity
matrices, respectively, and _A_ is the miRNA-drug association
network with _A_ _ij_ = 1 if miRNA _i_ is experimentally validated to
associate with drug _j_, and 0 otherwise.
**Model Framework.** Figure 1 shows the architecture of our
model. For each miRNA-drug input sample, the model
outputted a prediction score representing the possibility of
miRNA mediating in drug sensitivity. Overall, the model
consisted of three steps: node encoding and mapping to
feature space, representation learning on the heterogeneous
graph, and association prediction. First, the miRNAs and the
drugs were encoded and mapped to the same feature space.
Next, the node representation learning was run on miRNAdrug heterogeneous graph to incorporate the similarity
information between homogeneous nodes and the association
information between heterogeneous nodes. Finally, miRNA
and drug representations were used to predict the association
between miRNA and drug sensitivity.
**Node Encoding and Mapping to Feature Space.** We
first performed one-hot encoding based on the numerical order
of the miRNAs and the drugs separately. Because the
encodings of miRNAs and drugs were in different feature
spaces, we mapped the initial encodings of these two types of
nodes into the same dimension so that the features of
heterogeneous nodes could be propagated and aggregated.
Formally, for miRNA _i_ and drug _j_, we used principal
component analysis (PCA) to reduce the dimension of one

**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


̂



y


̂



**g** ( _l_ + _iA_ 1) = 2| 1 _A_ | _A_ jjjjjjjj **h** ( ) _jl_ + | 1 _A_ | _A_ **g** ( ) _kl_ **h**


̂



{


̂



( _l_ + _iA_ 1) = 2| _iA_ | _j_ _A_ jjjjjjjj **h** ( ) _jl_ + | _Aj_ | _k_ _A_ **g** ( ) _kl_ **h** ( ) _jl_


̂



_A_

_i_


̂



= 1 _A_ **h** ( ) _jl_ + 1 _A_ **g** ( ) _kl_ **h** ( ) _jl_

2| | _A_ jjjjjjjj | | _A_ zzzzzzzz


̂



̂



**h** _l_

_iA_ | _j_ _iA_ jjjjjjjj _j_


̂



̂



_iA_ k _j_ _k_ _Aj_ { (6)


̂



i

jjjjjjjj

k


̂



where _iA_ represents the neighbor drug nodes associated with
miRNA _i_, _Aj_ represents the neighbor miRNA nodes
associated with drug _j_, and ⊗ is the element-wise multiplication operation. From the perspective of miRNA, this
formulation aggregated the feature of the first-order heterogeneous neighbors (drugs), but also the fused second-order
heterogeneous neighbors (miRNAs). In particular, when the
known miRNA-drug associations are sparse, the term g ( _kl_ ) ⊗h ( _j_ _l_ )
actually supplemented association information regarding
second-order neighbors.
Next, we integrated the feature of miRNA _i_ itself and
conducted nonlinear transform as below


**g** ( _i_ _l_ +1) = _f_ ( **u** _m_  - [ **g** ( ) _i_ _l_, **g** ( _l_ + _iA_ 1) ]) (7)


where _f_ () is the nonlinear Leaky ReLU function and u m
represents the learnable weight vector used to reflect the
importance of features of miRNA _i_ itself and its neighbor
nodes.
For drug-centric feature aggregation, we adopted a similar
formulation as below


̂



y


̂



**h** ( _l_ + _iA_ 1) = 2| 1 _A_ | _A_ jjjjjjjj **g** ( ) _jl_ + | 1 _A_ | _A_ **h** ( ) _kl_ **g**


̂



{


̂



( _l_ + _iA_ 1) = 2| _iA_ | _j_ _A_ jjjjjjjj **g** ( ) _jl_ + | _Aj_ | _k_ _A_ **h** ( ) _kl_ **g** ( ) _jl_


̂



_A_

_i_


̂



= 2| 1 _A_ | _A_ jjjjjjjj **g** ( ) _jl_ + | 1 _A_ | _A_ **h** ( ) _kl_ **g** ( ) _jl_ zzzzzzzz


̂



̂



_l_
_iA_ | _j_ _iA_ jjjjjjjj **g** _j_


̂



̂



_iA_ k _j_ _k_ _Aj_ { (8)


̂



i

jjjjjjjj

k


̂



where _σ_ is the sigmoid function and Θ represents all of the
parameters involved in the model. When minimizing the loss
function, the miRNA-drug pairs with known associations tend
to have similar representations, while those without known
associations tend to have dissimilar representations. For all
trainable parameters, we initialized them with a Gaussian
distribution with mean 0 and standard deviation 0.01.
**Negative Sample Generation Strategy.** In most studies
for association prediction, negative samples are randomly
generated by sampling pairs of nodes. However, randomly
generated negative samples often include noisy labels that may
lead to biased decision boundary. In fact, the number of
validated miRNA-drug associations is far less than the true
ones, as the potential miRNA-drug associations are huge. As a
result, randomly generated negatives may include positive
samples, which probably biased the decision boundary in the
model training process. We built highly credible negative
samples using the strategy proposed by our group. [34] The main
idea is that miRNA is unlikely to interact with a drug if it is not
similar to all miRNAs associated with the drug, and vice versa.
Based on this rationale, we ranked the calculated miRNAmiRNA similarities from low to high, and selected the miRNAs
with the lowest similarity to those with associations to drugs.

−
Meanwhile, we ranked the calculated drug drug similarities
from low to high and selected the drugs with the lowest
similarity to those with associations to any miRNA. Finally, we
randomly paired the selected miRNAs and drugs to generate
highly credible negative samples.

# ■ [RESULTS AND DISCUSSION]


**Experimental Setup.** We implemented the proposed
method using Tensorflow framework. The learning rate was
set to 1 × 10 [−][6] . The maximum training epoch was 100. Minibatch was set to the default value 2048. The model was trained
on a workstation with 3 NVIDIA GeForce RTX 3090 graphic
cards of 24G memory.
**Performance Evaluation on Benchmark Dataset.** To
verify the performance of our method, we first evaluated it on
the benchmark dataset by fivefold cross-validations. The
training samples were randomly split into five subsets of
roughly equal size, each subset was taken in turn as a test set
and the remaining four subsets were used to train the model,
whose prediction accuracy on the test set was then evaluated.
To avoid random bias, the cross-validation process was
repeated 10 times, and the performance metrics were averaged
as final results. [35] [As shown in Figure 2 and Table S1, our](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.2c01060/suppl_file/ci2c01060_si_001.pdf)
method achieved average ROC-AUC, AUPR, ACC, F1-score,
Precision and Recall values of 0.9255, 0.9224, 0.8523, 0.8726,
0.8261, and 0.8481, respectively. The experimental result
indicated that our method performed well in predicting the
associations between miRNA and drug sensitivity.

̂ **Hyperparameter Optimization.** We next explored the

influence of two main hyperparameters, the number of GNN
layers and the dimension of embedding vector. The number of
GNN layers determined the scope of feature propagation of
each node. A larger number of GNN layers means feature
aggregations from higher-order neighbor nodes but loss of
local structure. We adopted the grid search to tune the two
parameters. The number of GNN layers increased from one to
four by step 1. The dimension of embedding was selected from
the list {8, 16, 32, 64, 128, 256}. The experimental results are
[shown in Figure 3 and Table S2. When the layer number was 2](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.2c01060/suppl_file/ci2c01060_si_001.pdf)


**5932** [https://doi.org/10.1021/acs.jcim.2c01060](https://doi.org/10.1021/acs.jcim.2c01060?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 5929−5937



**h** ( _i_ _l_ +1) = _f_ ( **u** _d_ - [ **h** ( ) _i_ _l_, **h** ( _l_ + _A_ 1) ])

_i_ (9)


̂



where _iA_ represents the neighbor miRNA nodes associated
with drug _i_, _Aj_ is the neighbor drug nodes associated with
miRNA _j_, and u d represents the learnable weight vector used to
reflect the importance of features of drug _i_ itself and its
neighbor nodes.
It is worth noting that we sampled only a portion of
neighbor nodes from which the information was aggregated.
More precisely, for the aggregation in the similarity networks,
when the neighbor nodes are more than 10, we randomly
selected 10 neighbor nodes to aggregate their information.
Random sampling has a couple of advantages: (1) Sampling in
the training stage increased randomness, thus promoting the
generalization ability of our model. (2) It effectively alleviates
the over-smoothing of node embedding. (3) Sampling makes it
possible for our method to scale to large-scale data.
**Association Prediction.** Following the representation
learning, the learned embeddings of miRNAs and drugs can
be used to predict the associations. We adopted the simple
inner product to score miRNA-drug sensitivity associations.Suppose _a_ _ij_ ̂ be the predictive score between miRNA _i_ and drug
_j_, we defined


_a_ _ij_ = **g h** _Ti_  - _j_ (10)


and adopted the pair-wise ranking-based loss function


_M_



̂


( ln ( _a_ _a_ _ik_ ) + ), (, ) _i j_ _A_ (, ) _i k_



̂


_a_ _a_ _ik_ ) + ), (, ) _i j_ _A_ _i k_



̂


= _a_ _a_ +



̂


_i_ =


_A_



̂


_ij_ _ik_

1



̂


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article



Figure 2. Receiver operator characteristic (ROC) curves of our
method evaluated on benchmark dataset by fivefold cross-validations.


and the embedding dimension was 16, our method achieved
the best performance.


Figure 3. Impact on the performance of two hyperparameters: the
number of GNN layers and embedding dimension.


**Model Selection and Ablation Experiments.** To
explore the effect of feature propagation on homogeneous
and heterogeneous nodes, we conducted model ablation
experiments. We have tested four variants: (1) Feature
propagation was only performed on homogeneous nodes,
and the number of propagation layers was set to 1, 2, or 3; (2)
Feature propagation was only performed on heterogeneous



nodes, and the number of propagation layers was set to 1, 2 or
3; (3) The feature propagation on homogeneous and
heterogeneous nodes was performed alternately, and the
feature propagation repeated for one, two, or three times;
(4) The feature propagation on homogeneous and heterogeneous nodes was performed sequentially. As shown in Table 1,
when only homogeneous feature propagation was performed,
two GNN layers achieved the best performance. If only
heterogeneous feature propagation was performed, we also
found that two GNN layers performed best. When both
homogeneous and homogeneous feature propagations were
run, the performance is better than the variants that run
homogeneous or homogeneous feature propagations alone.
More importantly, we found that the sequential mode
outperformed the alternative mode. In particular, the performance of two-layer homogeneous feature propagations followed
by one-layer heterogeneous feature propagation achieved the
highest performance among all model variants, whose area
under the curve (AUC) and AUPR reached 0.9255 and
0.9224, respectively.
Moreover, it can be seen from Table 1 that the AUC, AUPR,
and accuracy scores of “Homo2 + Heter1” are higher than
other combinations, but the precision, recall, and F1-score are
relatively low. This would be explained from the calculation of
AUC and F1-score. Generally speaking, the larger the AUC,
the greater difference between true positives and true
negatives. Instead, F1-score is a comprehensive consideration
of precision and recall. The F1-score measure drives the model
to classify positive and negative samples as much as possible,
while AUC tends to reduce false positives. Our study focused
on the highly reliable predictions of miRNA-drug sensitivity
associations for downstream biochemical experiments. Due to
the manpower and cost of wet-lab experiments, low-confidence
predictions should be excluded as much as possible. Therefore,
we tuned our model to achieve a high AUC so as to reduce
false positives as much as possible, which leads to relatively low
precision, recall, and F1-score.
**Performance Comparison with Other Methods.** To
verify the performance of our method, we compared it with
seven state-of-the-art methods for link prediction. We concisely
introduced the seven methods as below:


 - GANLDA [36] is a computational model based on the
graph attention network to infer the association between
lncRNAs and diseases.

 - LGCMDS [29] used the LightGCN and retained only the
neighborhood aggregation of GCN to predict miRNAdrug sensitivity associations.



Table 1. Performance of the Ablated Model by Running Feature Propagations on Homogeneous and/or Heterogeneous


model variant AUC AUPR accuracy precision recall F1-score


Homo × 1 0.9009 0.8684 0.8395 0.8500 0.8258 0.8372

Homo × 2 0.9013 0.8879 0.8316 0.8473 0.8091 0.8274

Homo × 3 0.8925 0.8762 0.8105 0.8094 0.8144 0.8114

Heter × 1 0.9058 0.8795 0.8471 0.8748 0.8115 0.8411

Heter × 2 0.9198 0.9148 0.8481 0.8680 0.8215 0.8440

Heter × 3 0.9134 0.9064 0.8409 0.8303 0.8584 0.8436

(Homo + Heter) × 1 0.9077 0.8797 0.8318 0.8951 0.7545 0.8174
(Homo + Heter) × 2 0.9210 0.9158 0.8495 0.8553 0.8421 0.8483
(Homo + Heter) × 3 0.9133 0.9102 0.8323 0.8688 0.7871 0.8237

Homo × 2 + Heter × 1 0.9255 0.9224 0.8523 0.8726 0.8261 0.8481

Homo × 2 + Heter × 2 0.9205 0.9154 0.8486 0.8665 0.8244 0.8448


**5933** [https://doi.org/10.1021/acs.jcim.2c01060](https://doi.org/10.1021/acs.jcim.2c01060?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 5929−5937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article




 - LAGCN [37] constructed heterogeneous networks of
diseases and drugs, and predicted drug-disease associations by embedding different convolutional layers and
attentional mechanisms.

 - ABHMDA [38] employed adaptive boosting to reveal the
associations between diseases and microbes by calculat
−
ing the relation probability of disease microbe pair
using a robust classifier.

 - SDLDA [39] combined singular value decomposition and
deep learning techniques to predict lncRNA-disease
associations.

 - DMFCDA [40] is a circRNA-disease association prediction
model that utilized a projection layer to learn
representations and multilayer neural networks to
capture nonlinear associations.

 - KATZMDA [41] calculated the miRNA similarity and
disease similarity based on the KATZ model and
predicted miRNA-disease associations.

We evaluated these competitive methods on the benchmark
[dataset (for parameter settings of these methods, see Section](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.2c01060/suppl_file/ci2c01060_si_001.pdf)
[S3 in the Supporting Information). Figure 4 shows the ROC](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.2c01060/suppl_file/ci2c01060_si_001.pdf)


Figure 4. ROC curves and AUC values of DGNNMDA and seven
competitive methods on the benchmark set by fivefold crossvalidations.


curves and AUC values of our method DGNNMDA and seven
competitive methods, from which we found that our method
achieved the highest AUC value 0.9255, followed by
GANLDA. Moreover, Table 2 lists other performance metrics
and verified that our method outperformed other methods in
the metrics AUC, AUPR, Accuracy, Recall, and F1-score
except for Precision metric. Because the associations between



miRNA and drug sensitivity are relatively sparse, the
competitive methods failed to fit the data well or suffered
from overfitting, thereby performing inferior to our method.
**Performance Evaluation on Independent Dataset.**
For objective evaluation, we trained our model on the
benchmark dataset and then tested it on the manually curated
independent set. We conducted similar experiments for other
seven competitive methods to evaluate their performance.
Figure 5 shows the ROC curves and AUC values of our


Figure 5. ROC curves and AUC values of DGNNMDA and seven
competitive methods on independent test set.


methods and other competitive methods, from which we can
see that our model obtained the best performance on the
independent test set by AUC value 0.82, while the other
competitive methods did not reach 0.8 AUC value. These
experimental results strongly demonstrated that our method
has obtained better generalization ability than other existing
methods.
**Impact of Negative Samples.** We were also interested in
the impact of negative samples on predictive performance. For
this purpose, we compared the randomly generated negative
samples and the selected negative samples. Also, we tested
different ratios of the number between positive and negative
samples. The experimental results are shown in Table 3. We
found when the ratio of positive and negative samples was
equal to 1:3, our model achieved the highest performance. The
reliable negative samples selection strategy improved the
performance moderately, compared to random negative sample
generation. The results showed that our method was robust to
the selection of negative samples.



Table 2. Performance Comparison of DGNNMDA and Seven Competitive Methods on Benchmark Dataset


method AUC AUPR accuracy precision recall F1


DGNNMDA 0.9255 0.9224 0.8523 0.8726 0.8261 0.8481

GANLDA 0.8985 0.8999 0.8194 0.8761 0.7440 0.8047

LGCMDS 0.8872 0.9026 0.8240 0.8370 0.8049 0.8204

LAGCN 0.8849 0.9070 0.8218 0.8625 0.7656 0.8112

ABHMDA 0.8508 0.8384 0.7686 0.7456 0.8173 0.7794

SDLDA 0.8504 0.8932 0.8168 0.8697 0.7461 0.8026

DMFCDA 0.8415 0.7226 0.8180 0.8158 0.8240 0.8183

KATZMDA 0.7766 0.8226 0.7381 0.7809 0.6619 0.7163


**5934** [https://doi.org/10.1021/acs.jcim.2c01060](https://doi.org/10.1021/acs.jcim.2c01060?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 5929−5937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article



Table 3. Performance Evaluation of Different Negative
Sample Generation Strategies and Ratios


AUC 0.9255 0.9212 0.9248 0.9249 0.9255 0.9247

AUPR 0.9224 0.9143 0.9223 0.9225 0.9224 0.9221


**Case Studies.** To check the reliability of our method, we
conducted case studies about two antitumor drugs Paclitaxel
and Cisplatin. Paclitaxel is a cytotoxic agent proven to be
effective to multiple types of tumors, especially ovarian and
breast cancers. [42][,][43] We removed all Paclitaxel-involved miRNA
associations from the training set, and then fed the remaining
candidate associations into the trained model. We list the top
10 miRNAs with the highest predicted association scores in
Table 4. Through PubMed literature retrieval, we found 7
miRNAs have been proven to mediate in the regulation of
Paclitaxel sensitivity in various cancers.
We went further to reveal the possible functional mechanism
of miRNA mediating drug sensitivity. As miRNAs induce the
degradation and translational repression of their target
mRNAs, we constructed a regulatory network consisting of
the 10 miRNAs and their target mRNAs. From the miRDB [44]

and miRWalk [45] databases, we selected the miRNA-targeted
mRNAs that have been reported to affect Paclitaxel sensitivity
in various tumors. The regulatory network is shown in Figure
6, in which the yellow nodes represent mRNAs, the red nodes
represent the miRNAs associated with Paclitaxel sensitivity in
the training set, the green nodes represent the miRNAs verified
by the PubMed literature, and the blue nodes represent
miRNAs without supportive evidence. Moreover, we focused
on a subnetwork with 53 pairs of miRNA-mRNA associations
between 9 miRNAs and 39 unique mRNAs, as shown in Figure
6B. We think that the sensitivity of Paclitaxel is affected by a
number of mRNAs targeted by several miRNAs that inhibit
their translation level. For example, ZEB1 inhibits the
expression of sorcin by inhibiting the transcription of hsamiR-142-5p, which is often overexpressed in tumor cells
resistant to Paclitaxel. [46] The integrated stress response (ISR)
plays an important role in the survival of cancer cells and is
more related to the production of anticancer drug sensitivity.
This mechanism involves two genes, EIF2AK3 and
EIF2AK4, [47] which are regulated by hsa-miR-431-5p, hsamiR-4443, and hsa-miR-663a.
For another drug Cisplatin, we conducted a similar study to
verify most of the predicted miRNAs reported to involve in its



[sensitivity/resistance (for more details, see Section S4 and](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.2c01060/suppl_file/ci2c01060_si_001.pdf)
[Table S3 in the Supporting Information).](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.2c01060/suppl_file/ci2c01060_si_001.pdf)

# ■ [DISCUSSION AND CONCLUSIONS]


In recent years, noncoding RNAs have emerged as a new type
of antiviral drugs. Especially, the nucleic acid drug has stood
out as an effective treatment to fight against COVID-19. [48]

However, the study that focused on the associations between
drug sensitivity and miRNAs is few. Our understanding of the
relationship between miRNA and drug sensitivity is far from
guiding the development of miRNA-targeted drugs. Traditional biochemical experimental methods used to reveal
miRNA-drug sensitivity are time-consuming, laborious, and
expensive. Therefore, computational methods for the reliable
prediction of miRNA-drug sensitivity can effectively promote
the discovery of miRNA-mediated drug sensitivity.
We proposed a new computational method to predict
miRNA-drug sensitivity association. We developed a dualchannel heterogeneous graph neural network model to learn
the latent representations of miRNAs and drugs. By encoding
and mapping miRNAs and drugs into the same embedding
space, we ran feature aggregation on homogeneous nodes and
heterogeneous neighbor nodes through multilayer graph
aggregation to obtain node embeddings. Also, the integration

−
of the miRNA similarity network and drug drug association
network into the miRNA-drug heterogeneous network can
effectively alleviate the problem of sparse connections in the
dataset. To evaluate the proposed method, we constructed a
benchmark dataset and a manually curated independent test
set. The performance comparison experiments verified that our
method achieves excellent performance and outperforms six
other state-of-the-art methods for link prediction. Moreover,
two case studies for two drugs, Paclitaxel and Cisplatin,
showed that the heterogeneous graph representation learning
actually pulls close the associated miRNAs and drugs in the
latent space, while pushing other miRNAs away from the
drugs. This illustrated that our method obtained both
expressive and interpretable features of miRNAs and drugs.
To the best of our knowledge, we are the first to highlight
the importance of miRNA-mediated drug sensitivity. Also, the
dataset built in this study is the largest so far.
Despite the merits of our method, it still has much room for
improvement. First, for the initialization of drug latent
representation, the chemical fingerprint rather than random
values can be leveraged. [49][,][50] In addition, self-supervised
learning strategy has developed rapidly. By constructing
pretexts, it can learn meaningful representations from a large
number of unlabeled data and improve the performance of



Table 4. Top 10 miRNAs Predicted to Mediate in Paclitaxel Sensitivity in Various Cancers


rank miRNA PMID description


1 miR-508 32988253 Mir-508-3p plays a role in the anticancer effect of Paclitaxel in a variety of cancers.
2 miR-29a* 30229821 Downregulation of miR-29a* can promote the improve the sensitivity of Paclitaxel and apoptosis of colorectal cancer cells.
3 miR-663a Unconfirmed none
4 miR-431 Unconfirmed none
5 miR-4443 31615089 Mir-4443 participates in the regulation of chemoresistance through drug genetic targets.
6 miR-215 26676658 Upregulation of miR-215 promotes apoptosis and increased sensitivity to the drug Paclitaxel.
7 miR-200a 25327865 After artificially upregulating mir-200a, it was found that SKOV-3 was more sensitive to Paclitaxel.
8 miR-518c-AS Unconfirmed none
9 miR-506-3p 34541682 Downregulation of miR-506-3p can increase the sensitivity of Paclitaxel in OC cells.
10 miR-31-5p 23552883 MiR-31 will bind to MET gene, resulting in the decrease of met so as to improve the sensitivity of Paclitaxel.


**5935** [https://doi.org/10.1021/acs.jcim.2c01060](https://doi.org/10.1021/acs.jcim.2c01060?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 5929−5937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article


Figure 6. Regulatory network of top 10 predicted Paclitaxel-associated miRNAs and their target mRNAs.



downstream tasks. We believe that self-supervised learning can
be incorporated into our learning framework to promote
performance.

# ■ [ASSOCIATED CONTENT]

- **sı** **Supporting Information**
The Supporting Information is available free of charge at
[https://pubs.acs.org/doi/10.1021/acs.jcim.2c01060.](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01060?goto=supporting-info)


Levenshtein distance and details in the construction of
similarity networks (Section S1); display of all
experimental results (Section S2); parameter settings
of the comparison methods (Section S3); case study
about Cisplatin (Section S4); and more case studies
[(Section S5) (PDF)](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.2c01060/suppl_file/ci2c01060_si_001.pdf)

# ■ [AUTHOR INFORMATION]


**[Corresponding Author](https://pubs.acs.org/action/doSearch?field1=Contrib&text1="Hui+Liu"&field2=AllField&text2=&publication=&accessType=allContent&Earliest=&ref=pdf)**

Hui Liu − _School of Computer Science and Technology,_
_Nanjing Tech University, Nanjing 211816, China;_

[orcid.org/0000-0001-7158-913X; Email: hliu@](https://orcid.org/0000-0001-7158-913X)
[njtech.edu.cn](mailto:hliu@njtech.edu.cn)


**[Authors](https://pubs.acs.org/action/doSearch?field1=Contrib&text1="Lei+Deng"&field2=AllField&text2=&publication=&accessType=allContent&Earliest=&ref=pdf)**

Lei Deng − _School of Computer Science and Engineering,_
_Central South University, Changsha 410083, China_
Ziyu Fan − _School of Computer Science and Engineering,_
_Central South University, Changsha 410083, China_
Xiaojun Xiao − _Software School, Xinjiang University, Urumqi_
_[830091, China](https://pubs.acs.org/action/doSearch?field1=Contrib&text1="Jiaxuan+Zhang"&field2=AllField&text2=&publication=&accessType=allContent&Earliest=&ref=pdf)_
Jiaxuan Zhang − _Department of Electrical and Computer_
_Engineering, University of California, San Diego, San Diego,_
_California 92161, United States_


Complete contact information is available at:
[https://pubs.acs.org/10.1021/acs.jcim.2c01060](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01060?ref=pdf)


**Author Contributions**

Z.F. and L.D. conceived the main idea. X.L. helped to improve
the idea. Z.F., X.X., and J.Z. conducted the experiments. Z.F.



and H.L. wrote the manuscript. H.L. reviewed and revised the
manuscript.

**Notes**
The authors declare no competing financial interest.
[The source code and datasets are freely available at https://](https://github.com/19990915fzy/DGNNMDA)
[github.com/19990915fzy/DGNNMDA.](https://github.com/19990915fzy/DGNNMDA)

# ■ [ACKNOWLEDGMENTS]

This work was supported by the National Natural Science
Foundation of China under grants nos. 62072058 and
61972422.

# ■ [REFERENCES]


[(1) Vasan, N.; Baselga, J.; Hyman, D. M. A view on drug resistance](https://doi.org/10.1038/s41586-019-1730-1)
[in cancer.](https://doi.org/10.1038/s41586-019-1730-1) _Nature_ 2019, _575_, 299−309.
[(2) Mansoori, B.; Ali, M.; Sadaf, D.; Solmaz, S.; Behzad, B. The](https://doi.org/10.15171/apb.2017.041)
[Different Mechanisms of Cancer Drug Resistance: A Brief Review.](https://doi.org/10.15171/apb.2017.041)
_Adv. Pharm. Bull._ 2017, _7_, 339−348.
[(3) Wang, X.; Zhang, H.; Chen, X.; Resistance, C. D. Drug](https://doi.org/10.20517/cdr.2019.10)
[resistance and combating drug resistance in cancer.](https://doi.org/10.20517/cdr.2019.10) _Cancer Drug_
_Resist._ 2019, _2_, 141−160.
[(4) Si, W.; Shen, J.; Zheng, H.; Fan, W. The role and mechanisms of](https://doi.org/10.1186/s13148-018-0587-8)
[action of microRNAs in cancer drug resistance.](https://doi.org/10.1186/s13148-018-0587-8) _Clin. Epigenet._ 2019,
_11_, 1−24.
[(5) Garzon, R.; Calin, G. A.; Croce, C. M. MicroRNAs in Cancer.](https://doi.org/10.1146/annurev.med.59.053006.104707)
_Annu. Rev. Med._ 2009, _60_, 167−179.
(6) Acunzo, M.; Romano, G.; Wernicke, D.; Croce, C. M.
[MicroRNA and cancer−a brief overview.](https://doi.org/10.1016/j.jbior.2014.09.013) _Adv. Biol. Regul._ 2015, _57_,
1−9.
[(7) Su, X.; Xing, J.; Wang, Z.; Lei, C.; Jiang, B. microRNAs and](https://doi.org/10.3978/j.issn.1000-9604.2013.03.08)
[ceRNAs: RNA networks in pathogenesis of cancer.](https://doi.org/10.3978/j.issn.1000-9604.2013.03.08) _Chin. J. Cancer Res._
2022, _25_, 235−239.
(8) Enerly, E.; Steinfeld, I.; Kleivi, K.; Leivonen, S. K.; Aure, M. R.;
Russnes, H. G.; Ronneberg, J.; Johnsen, H.; Navon, R.; Rodland, E.
[miRNA-mRNA Integrated Analysis Reveals Roles for miRNAs in](https://doi.org/10.1371/journal.pone.0016915)
[Primary Breast Tumors.](https://doi.org/10.1371/journal.pone.0016915) _PLoS One_ 2015, _6_, No. e16915.
(9) Tonevitsky, A. G.; Maltseva, D. V.; Abbasi, A.; Samatov, T. R.;
Sakharov, D. A.; Shkurnikov, M. U.; Lebedev, A. E.; Galatenko, V. V.;
[Grigoriev, A. I.; Northoff, H. Dynamically regulated miRNA-mRNA](https://doi.org/10.1186/1472-6793-13-9)
[networks revealed by exercise.](https://doi.org/10.1186/1472-6793-13-9) _BMC Physiol._ 2013, _13_, 1−11.
(10) Wu, C.; Zhao, Y.; Liu, Y.; Yang, X.; Yan, M.; Min, Y.; Pan, Z.;
[Qiu, S.; Xia, S.; Yu, J.; et al. Identifying mirna-mrna regulation](https://doi.org/10.3892/ol.2018.9243)


**5936** [https://doi.org/10.1021/acs.jcim.2c01060](https://doi.org/10.1021/acs.jcim.2c01060?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 5929−5937


**Journal of Chemical Information and Modeling** **pubs.acs.org/jcim** Article



[network of major depressive disorder in ovarian cancer patients.](https://doi.org/10.3892/ol.2018.9243)
_Oncol. Lett._ 2018, _16_, 5375−5382.
(11) Bai, Y.; Baker, S.; Exoo, K.; Dai, X.; Ding, L.; Khattak, N.; Li,
[H.; Liu, H.; Liu, X. MMiRNA-Viewer2, a bioinformatics tool for](https://doi.org/10.1186/s12859-020-3436-7)
[visualizing functional annotation for MiRNA and MRNA pairs in a](https://doi.org/10.1186/s12859-020-3436-7)
[network.](https://doi.org/10.1186/s12859-020-3436-7) _BMC Bioinf._ 2020, _21_, 1−10.
(12) Wang, N.; Li, Y.; Liu, S.; Gao, L.; Liu, C.; Bao, X.; Xue, P.
[Analysis and Validation of Differentially Expressed MicroRNAs with](https://doi.org/10.2174/1574893615999200508091615)
[their Target Genes Involved in GLP-1RA Facilitated Osteogenesis.](https://doi.org/10.2174/1574893615999200508091615)
_Curr. Bioinf._ 2021, _16_, 928−942.
[(13) Geraci, F.; Manzini, G. EZcount: An all-in-one software for](https://doi.org/10.1016/j.compbiomed.2021.104352)
[microRNA expression quantification from NGS sequencing data.](https://doi.org/10.1016/j.compbiomed.2021.104352)
_Comput. Biol. Med._ 2021, _133_, No. 104352.
[(14) Li, J.; Liu, L.; Cui, Q.; Zhou, Y. Comparisons of MicroRNA Set](https://doi.org/10.2174/1574893615666200224095041)
[Enrichment Analysis Tools on Cancer De-regulated miRNAs from](https://doi.org/10.2174/1574893615666200224095041)
[TCGA Expression Datasets.](https://doi.org/10.2174/1574893615666200224095041) _Curr. Bioinf._ 2021, _15_, 1104−1112.
[(15) Ma, J.; Dong, C.; Ji, C. MicroRNA and drug resistance.](https://doi.org/10.1038/cgt.2010.18) _Cancer_
_Gene Ther._ 2010, _17_, 523−531.
[(16) Rajarajan, D.; Kaur, B.; Penta, D.; Natesh, J.; Meeran, S. miR-](https://doi.org/10.1016/j.compbiomed.2021.104601)
[145−5p as a predictive biomarker for breast cancer stemness by](https://doi.org/10.1016/j.compbiomed.2021.104601)
[computational clinical investigation.](https://doi.org/10.1016/j.compbiomed.2021.104601) _Comput. Biol. Med._ 2021, _135_,
No. 104601.
[(17) Li, M.; Gao, M.; Xie, X.; Zhang, Y.; Gu, K.; et al. MicroRNA-](https://doi.org/10.3892/ol.2019.10304)
[200c reverses drug resistance of human gastric cancer cells by](https://doi.org/10.3892/ol.2019.10304)
[targeting regulation of the NER-ERCC3/4 pathway.](https://doi.org/10.3892/ol.2019.10304) _Oncol. Lett._ 2019,
_18_, 145−152.
(18) Ye, P.; Fang, C.; Zeng, H.; Shi, Y.; Pan, Z.; An, N.; He, K.;
[Zhang, L.; Long, X. Differential microRNA expression profiles in](https://doi.org/10.3892/ol.2018.7768)
[tamoxifen-resistant human breast cancer cell lines induced by two](https://doi.org/10.3892/ol.2018.7768)
[methods.](https://doi.org/10.3892/ol.2018.7768) _Oncol. Lett._ 2018, _15_, 3532−3539.
[(19) Shen, L.; Liu, F.; Huang, L.; Liu, G.; Zhou, L.; Peng, L. VDA-](https://doi.org/10.1016/j.compbiomed.2021.105119)
[RWLRLS: An anti-SARS-CoV-2 drug prioritizing framework combin-](https://doi.org/10.1016/j.compbiomed.2021.105119)
[ing an unbalanced bi-random walk and Laplacian regularized least](https://doi.org/10.1016/j.compbiomed.2021.105119)
[squares.](https://doi.org/10.1016/j.compbiomed.2021.105119) _Comput. Biol. Med._ 2021, _140_, No. 105119.
(20) Meng, F.; Henson, R.; Lang, M.; Wehbe, H.; Maheshwari, S.;
[Mendell, J. T.; Jiang, J.; Schmittgen, T. D.; Patel, T. Involvement of](https://doi.org/10.1053/j.gastro.2006.02.057)
[Human Micro-RNA in Growth and Response to Chemotherapy in](https://doi.org/10.1053/j.gastro.2006.02.057)
[Human Cholangiocarcinoma Cell Lines.](https://doi.org/10.1053/j.gastro.2006.02.057) _Gastroenterology_ 2006, _130_,
2113−2129.
[(21) Yu, Q.; Xu, X. P.; Yin, X. M.; Peng, X. Q. miR-155-5p increases](https://doi.org/10.4149/neo_2020_200106N17)
[the sensitivity of liver cancer cells to adriamycin by regulating ATG5-](https://doi.org/10.4149/neo_2020_200106N17)
[mediated autophagy.](https://doi.org/10.4149/neo_2020_200106N17) _Neoplasma_ 2021, _68_, 87−95.
(22) Ao, X.; Nie, P.; Wu, B.; Xu, W.; Zhang, T.; Wang, S.; Chang,
[H.; Zou, Z. Decreased expression of microRNA-17 and microRNA-](https://doi.org/10.1038/cddis.2016.367)
[20b promotes breast cancer resistance to taxol therapy by](https://doi.org/10.1038/cddis.2016.367)
[upregulation of NCOA3.](https://doi.org/10.1038/cddis.2016.367) _Cell Death Dis._ 2016, _7_, No. e2463.
(23) Li, L.; Wu, P.; Wang, Z.; Meng, X.; Zha, C.; Li, Z.; Qi, T.;
[Zhang, Y.; Han, B.; Li, S.; et al. NoncoRNA: a database of](https://doi.org/10.1186/s13045-020-00849-7)
[experimentally supported non-coding RNAs and drug targets in](https://doi.org/10.1186/s13045-020-00849-7)
[cancer.](https://doi.org/10.1186/s13045-020-00849-7) _J. Hematol. Oncol._ 2020, _13_, 1−4.
(24) Dai, E.; Yang, F.; Wang, J.; Zhou, X.; Song, Q.; An, W.; Wang,
[L.; Jiang, W. ncDR: a comprehensive resource of non-coding RNAs](https://doi.org/10.1093/bioinformatics/btx523)
[involved in drug resistance.](https://doi.org/10.1093/bioinformatics/btx523) _Bioinformatics_ 2017, _33_, 4010−4011.
[(25) Huang, Y. A.; Hu, P.; Chan, K.; You, Z. H. Graph convolution](https://doi.org/10.1093/bioinformatics/btz621)
[for predicting associations between miRNA and drug resistance.](https://doi.org/10.1093/bioinformatics/btz621)
_Bioinformatics_ 2019, _36_, 851−858.
[(26) Niu, Y.; Song, C.; Gong, Y.; You, Z. H. MiRNA-Drug](https://doi.org/10.3389/fphar.2021.799108)
[Resistance Association Prediction Through the Attentive Multimodal](https://doi.org/10.3389/fphar.2021.799108)
[Graph Convolutional Network.](https://doi.org/10.3389/fphar.2021.799108) _Bioinformatics_ 2021, _12_, 799108.
[(27) Xu, P.; Wu, Q.; Rao, Y.; Kou, Z.; Han, H.; et al. Predicting the](https://doi.org/10.1109/ACCESS.2020.3004512)
[Influence of MicroRNAs on Drug Therapeutic Effects by Random](https://doi.org/10.1109/ACCESS.2020.3004512)
[Walking.](https://doi.org/10.1109/ACCESS.2020.3004512) _IEEE Access_ 2020, _8_, 117347−117353.
[(28) Li, Y.; Runqi, W.; Shuo, Z.; Hanlin, X.; Lei, D. LRGCPND:](https://doi.org/10.3390/ijms221910508)
[Predicting Associations between ncRNA and Drug Resistance via](https://doi.org/10.3390/ijms221910508)
[Linear Residual Graph Convolution.](https://doi.org/10.3390/ijms221910508) _Int. J. Mol. Sci._ 2021, _22_, 10508.
(29) Song, Y.; Hanlin, X.; Yizhan, L.; Dayun, L.; Lei, D. In
_LGCMDS: Predicting miRNADrug Sensitivity based on Light Graph_
_Convolution Network_, 2021 IEEE International Conference on
Bioinformatics and Biomedicine (BIBM), IEEE, 2021; pp 217−222.



(30) Griffiths-Jones, S.; Grocock, R. J.; Stijn, V.; Alex, B.; Enright, A.
[J. miRBase: microRNA sequences, targets and gene nomenclature.](https://doi.org/10.1093/nar/gkj112)
_Nucleic Acids Res._ 2006, _34_, 140−144.
[(31) Navarro, G. A Guided Tour to Approximate String Matching.](https://doi.org/10.1145/375360.375365)
_ACM Comput. Surv._ 2001, _33_, 31−88.
(32) Kim, S.; Chen, J.; Cheng, T.; Gindulyte, A.; He, J.; He, S.; Li,
Q.; Shoemaker, B. A.; Thiessen, P. A.; Yu, B.; Zaslavsky, L.; Zhang, J.;
[Bolton, E. E. PubChem 2019 update: improved access to chemical](https://doi.org/10.1093/nar/gky1033)
[data.](https://doi.org/10.1093/nar/gky1033) _Nucleic Acids Res._ 2019, _47_, D1102−D1109.
[(33) Yang, C.; Wang, P.; Tan, J.; Liu, Q.; Li, X. Autism spectrum](https://doi.org/10.1016/j.compbiomed.2021.104963)
[disorder diagnosis using graph attention network based on spatial-](https://doi.org/10.1016/j.compbiomed.2021.104963)
[constrained sparse functional brain networks.](https://doi.org/10.1016/j.compbiomed.2021.104963) _Comput. Biol. Med._
2021, _139_, No. 104963.
[(34) Liu, H.; Sun, J.; Guan, J.; Zheng, J.; Zhou, S. Improving](https://doi.org/10.1093/bioinformatics/btv256)
[compound-protein interaction prediction by building up highly](https://doi.org/10.1093/bioinformatics/btv256)
[credible negative samples.](https://doi.org/10.1093/bioinformatics/btv256) _Bioinformatics_ 2015, _31_, i221−i229.
[(35) Yang, Y.; Chen, L. Identification of Drug-Disease Associations](https://doi.org/10.2174/1574893616666210825115406)
[by Using Multiple Drug and Disease Networks.](https://doi.org/10.2174/1574893616666210825115406) _Curr. Bioinf._ 2022, _17_,
48−59.
(36) Lan, W.; Wu, X.; Chen, Q.; Peng, W.; Wang, J.; Chen, Y.
[GANLDA: Graph attention network for lncRNA-disease associations](https://doi.org/10.1016/j.neucom.2020.09.094)
[prediction.](https://doi.org/10.1016/j.neucom.2020.09.094) _Neurocomputing_ 2021, _469_, 384−393.
[(37) Yu, Z.; Huang, F.; Zhao, X.; Xiao, W.; Zhang, W. Predicting](https://doi.org/10.1093/bib/bbaa243)

−
drug [disease associations through layer attention graph convolutional](https://doi.org/10.1093/bib/bbaa243)
[network.](https://doi.org/10.1093/bib/bbaa243) _Briefings Bioinf._ 2020, _43_, No. bbaa243.
[(38) Peng, L. H.; Yin, J.; Zhou, L.; Liu, M. X.; Yan, Z. Human](https://doi.org/10.3389/fmicb.2018.02440)
[Microbe-Disease Association Prediction Based on Adaptive Boosting.](https://doi.org/10.3389/fmicb.2018.02440)
_Front. Microbiol._ 2018, _9_, 2440.
[(39) Zeng, M.; Lu, C.; Zhang, F.; Li, Y.; Li, M.; et al. SDLDA:](https://doi.org/10.1016/j.ymeth.2020.05.002)
[lncRNA−disease association prediction based on singular value](https://doi.org/10.1016/j.ymeth.2020.05.002)
[decomposition and deep learning.](https://doi.org/10.1016/j.ymeth.2020.05.002) _Methods_ 2020, _179_, 73−80.
[(40) Lu, C.; Zeng, M.; Zhang, F.; Wu, F.; Li, M.; Wang, J. Deep](https://doi.org/10.1109/JBHI.2020.2999638)
[matrix factorization improves prediction of human circRNA-disease](https://doi.org/10.1109/JBHI.2020.2999638)
[associations.](https://doi.org/10.1109/JBHI.2020.2999638) _IEEE J. Biomed. Health Inform._ 2021, _25_, 891−899.
[(41) Qu, Y.; Zhang, H.; Liang, C.; Dong, X. KATZMDA: Prediction](https://doi.org/10.1109/ACCESS.2017.2754409)
[of miRNA-disease associations based on KATZ model.](https://doi.org/10.1109/ACCESS.2017.2754409) _IEEE Access_

2018, _6_, 3943−3950.
(42) Riedl, J. M.; Posch, F.; Horvath, L.; Gantschnigg, A.; Gerger,
[A.; et al. Gemcitabine/nab-Paclitaxel versus FOLFIRINOX for](https://doi.org/10.1016/j.ejca.2021.03.040)
[palliative first-line treatment of advanced pancreatic cancer: A](https://doi.org/10.1016/j.ejca.2021.03.040)
[propensity score analysis.](https://doi.org/10.1016/j.ejca.2021.03.040) _Eur. J. Cancer_ 2021, _151_, 3−13.
[(43) Blomstrand, H.; Batra, A.; Cheung, W. Y.; Elander, N. O. Real-](https://doi.org/10.5306/wjco.v12.i9.787)
[world evidence on first- and second-line palliative chemotherapy in](https://doi.org/10.5306/wjco.v12.i9.787)
[advanced pancreatic cancer.](https://doi.org/10.5306/wjco.v12.i9.787) _World J. Clin. Oncol._ 2021, _12_, 787.
[(44) Chen, Y.; Wang, X. miRDB: an online database for prediction](https://doi.org/10.1093/nar/gkz757)
[of functional microRNA targets.](https://doi.org/10.1093/nar/gkz757) _Nucleic Acids Res._ 2020, _48_, D127−
D131.
(45) Sticht, C.; Carolina, D.; Parveen, A.; Gretz, N.; Campbell, M.
[miRWalk: An online resource for prediction of microRNA binding](https://doi.org/10.1371/journal.pone.0206239)
[sites.](https://doi.org/10.1371/journal.pone.0206239) _PLoS One_ 2018, _13_, No. e0206239.
[(46) Zhang, J.; Guan, W.; Xu, X.; Wang, F.; Li, X.; Xu, G. A novel](https://doi.org/10.1038/s41388-021-01891-6)
[homeostatic loop of sorcin drives paclitaxel-resistance and malignant](https://doi.org/10.1038/s41388-021-01891-6)
[progression via Smad4/ZEB1/miR-142-5p in human ovarian cancer.](https://doi.org/10.1038/s41388-021-01891-6)
_Oncogene_ 2021, _40_, 4906−4918.
(47) Chen, L.; He, J.; Zhou, J.; Xiao, Z.; Ding, N.; Duan, Y.; Li, W.;
[Sun, L. EIF2A promotes cell survival during paclitaxel treatment in](https://doi.org/10.1111/jcmm.14469)
[vitro and in vivo.](https://doi.org/10.1111/jcmm.14469) _J. Cell. Mol. Med._ 2019, _23_, 6060−6071.
(48) Le, T. T.; Andreadakis, Z.; Kumar, A.; Román, R. G.; Tollefsen,
[S.; Saville, M.; Mayhew, S. The COVID-19 vaccine development](https://doi.org/10.1038/d41573-020-00151-8)
[landscape.](https://doi.org/10.1038/d41573-020-00151-8) _Nat. Rev. Drug Discovery_ 2020, _19_, 305−306.
(49) Yu, C.; Yongshun, G.; Yuansheng, L.; Bosheng, S.; Quan, Z.
[Molecular design in drug discovery: a comprehensive review of deep](https://doi.org/10.1093/bib/bbab344)
[generative models.](https://doi.org/10.1093/bib/bbab344) _Briefings Bioinf._ 2021, _4_, No. bbab34.
[(50) Ru, X.; Ye, X.; Sakurai, T.; Zou, Q.; Xu, L.; Lin, C. Current](https://doi.org/10.1093/bfgp/elab031)

−
[status and future prospects of drug](https://doi.org/10.1093/bfgp/elab031) target interaction prediction.
_Briefings Funct. Genomics_ 2021, _20_, 312−322.


**5937** [https://doi.org/10.1021/acs.jcim.2c01060](https://doi.org/10.1021/acs.jcim.2c01060?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_J. Chem. Inf. Model._ 2022, 62, 5929−5937


