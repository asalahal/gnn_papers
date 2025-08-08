pubs.acs.org/crt Article

## **GeoDILI: A Robust and Interpretable Model for Drug-Induced Liver** **Injury Prediction Using Graph Neural Network-Based Molecular** **Geometric Representation**
#### Wenxuan Wu, [§] Jiayu Qian, [§] Changjie Liang, Jingya Yang, Guangbo Ge, Qingping Zhou,* and Xiaoqing Guan*

### ACCESS Metrics & More Article Recommendations * sı Supporting Information





**1. INTRODUCTION**


The liver plays a vital role in metabolism and detoxification,
which makes it vulnerable to damage from exogenous
compounds such as drugs and environmental chemicals.
Drug-induced liver injury (DILI) refers to the liver injury
caused by the drug itself and/or its metabolites, which is one of
the main reasons for drug failure and withdrawal or
termination from the market in the later stage of clinical
trials. [1][−][4] DILI risk assessment has become one of the
important issues in safe drug development, and therefore
there is a high demand for developing predictive models to
identify potential hepatotoxic compounds during the early
stages of drug development. [5][,][6]

The underlying mechanisms of DILI are complex and
varied. [1][,][3] DILI can be classified as intrinsic or idiosyncratic
based on the dose-dependent manner of the drug. Intrinsic
DILI is dose-dependent and predictable in preclinical animal
or in vitro studies, while idiosyncratic hepatotoxicity is doseindependent and usually unpredictable in regulatory-required
animal/cell toxicity experiments. [7][−][9] Many various in vitro and
in vivo assays have been developed for DILI risk assessment.
However, previous studies have shown that the results of


© 2023 American Chemical Society



**1717**



preclinical assessments of DILI (cellular models, animal
models, and so on) and those of humans do not always
coincide. [10][−][13] A retrospective analysis showed that animal

−
testing missed 40 45% of liver toxicity cases during clinical
trials. [14][,][15] The limited capability of the existing methods raises
the need for more efficient testing approaches.
Over the past decade, high-throughput screening assays and
combinatorial chemistry have generated a variety of biological
data on millions of compounds, and the advancement of datasharing programs has brought toxicology research into the era
of big data. However, human hepatotoxicity data are extremely
hard to collect and easily mislabeled. The main reasons are (1)
the limited number of reports due to the voluntary character of
information collection, (2) difficulty in the availability of


[https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)

_Chem. Res. Toxicol._ 2023, 36, 1717−1730


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article


Figure 1. Overall framework of GeoDILI. The upper section shows the DILI prediction module, comprising two primary phases: the feature
extraction phase and the fine-tuning phase. The lower part displays the results of dominant substructure derivation, which is accomplished by
calculating atomic contribution scores using information from the last GIN layer within the GeoGNN block in conjunction with the fragmentation
and Wilcoxon statistical test processes.



proprietary and postmarketing human toxicity data, and (3)
the noncausal relationships in inferred hepatotoxicity since
people take more than one drug or supplement at the same
time. [16] These problems are reflected in the conflicting
classification labels of compounds between data sets from
different sources. To address this issue, the U.S. Food and
Drug Administration (FDA) developed an annotation scheme
to label DILI risks for 1036 FDA-approved drugs based on the
assessments by regulatory professionals and released the
DILIrank public database in 2016. [17] By far, DILIrank is the
most widely used data set for developing DILI prediction
models. [18][−][24] Recently, the FDA further augmented DILIrank



and it has been shown that the quality of the data sets and the
accuracy of the models are correlated. [16][,][22]

Many of the existing DILI risk prediction models are based
on molecular structures or properties of drugs, named


**1718** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730



to DILIst, which contains 1279 drugs, by adding four
additional literature data sets and applying consistency
analysis. [25] Many other publications also annotated drugs of
DILI risk based on different criteria, such as LiverTox, [26]

[Hepatox (http://www.hepatox.org/), LTKB,](http://www.hepatox.org/) [27] SIDER, [28] and
published works of literature. [18][,][29][−][33] These databases provide a
valuable resource for DILI prediction. Based on this, numerous
in silico models have been generated in the past decade, [34][−][42]


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article



−
quantitative structure activity relationship (QSAR) models. [43][−][46] The advantage of QSAR models is that they do not
require mechanistic information and could directly establish
statistical relationships between structures or descriptors and
biological activities. Several data-driven machine learning
(ML)-based models have been proposed, such as Random
Forest, [16][,][22] and Support Vector Machine (SVM). [34][,][47] Some
works employed ensemble models to improve the accuracy and
precision of complex DILI predictions, such as voting or
average probability and neural-network-based meta-classifier
strategy. [36][,][37] These developed DILI models are mainly based
on hand-coded or rule-based molecular descriptors/fingerprints or a combination of them to characterize the molecular
properties of each compound, which may lead to poor
performance if the molecular representations do not capture
enough information. [48][,][49]

Recent progress in deep learning algorithms with novel
neural architectures has greatly facilitated drug discovery and
development. A promising approach is to represent molecules
as graphs, with atoms as nodes and bonds as edges. Graph
neural networks (GNN) can then process these graphs to
extract features, with message-passing neural networks
(MPNN) being the most popular architecture. [50][−][53] It has
been shown to outperform models built on human-designed
molecular descriptors for certain biological properties but is
still rarely applied in the DILI prediction area. [54][,][55] In addition,
the geometric information on a molecule plays an important
role in its physicochemical properties and biological activities.
For example, ( _R_ )-(+)-thalidomide and ( _S_ )-(−)-thalidomide
have the same topology structure but different geometries
leading to different biological activities. Both ( _R_ )-(+)-thalidomide and ( _S_ )-(−)-thalidomide have sedative and antiemetic
effects, while only ( _S_ )-(−)-thalidomide is teratogenic and can
cause embryonic malformations. [56] However, since MPNN
only considers topological information such as nodes and
edges and node adjacencies, it cannot distinguish molecules
with different geometric structures. One popular research topic
is geometric molecular representation, which encodes angular
or three-dimensional (3D) coordinate features that allows
GNNs to capture geometric shapes. [57] Recently, Fang et al.
proposed a novel geometry-enhanced molecular (GEM)
representation learning method, which is pretrained by largescale unlabeled molecules with coarse 3D spatial structures
using self-supervised learning, and achieved several state-ofthe-art (SOTA) results on molecular property prediction
benchmarks. [58]

In addition to predictive performance, we also focus on the
inferential ability of the model to derive key chemical
substructures as structural alerts (SAs) for human liver toxicity.
Using SAs, researchers can recognize potential hazardous
compounds and modify them in very early stages. [59][−][63]

Explainable artificial intelligence (XAI) has become a very
attractive research subject in theoretical ML, computer vision,
natural language processing, and, more recently, cheminformatics. [64][−][67] Compared with previously proposed SAs
inference methods, such as expert knowledge-based [59][,][68] or
frequency-based, [61] XAI-based methods infer substructures
based on predictive accuracy and therefore have better
predictive performance. [63][,][67] Therefore, the construction of
XAI-based SA inferring methods may be more effective for
guiding lead optimization to reduce the risk of DILI. To date,
these methods have not yet been applied to this issue yet.



In this study, we developed a highly accurate and
interpretable human DILI prediction model named GeoDILI.
An overview of the proposed model is shown in Figure 1. The
GeoDILI model used a pretrained 3D spatial structure-based
GNN to extract molecular representations, followed by a
residual neural network to make an accurate DILI prediction.
The gradient information from the final graph convolutional
layer of GNN was utilized to obtain atom-based weights, which
enabled the identification of dominant substructures that
significantly contributed to the DILI prediction. We evaluated
the performance of GeoDILI by comparing it with the SOTA
DILI prediction tools, popular GNN models, as well as
conventional deep neural networks (DNN) and ML models,
confirming its effectiveness in predicting DILI. In addition, we
applied our model to three different human DILI data sets
from various sources, namely DILIrank, [17] DILIst, [25] and a data
set recently collected by Yan et al. [37] Results showed
performance differences across data sets and suggested that a
smaller, high-quality data set DILIrank may lead to better
results. Finally, we applied the dominant substructure inference
method to analyze the entire DILIrank data set and identified
seven significant SAs with both high precision and potential
mechanisms.


**2. MATERIALS AND METHODS**


**2.1. Data Collection and Preparation.** In this work, we used
three human DILI data sets, named DILIrank, DILIst, and Yan et al.
data set. [17][,][25][,][37] A summary of these data sets is shown in Table 1.


Table 1. Summary of the DILI Datasets


no. of compound
data set (positive/negative) classification rule


DILIrank 720 (452/268) v Most- and v Less-DILI-Concern as
positive; [v] No-DILI-Concern as
negative
DILIst 1002 (604/398) 1 as positive; 0 as negative
Yan et al. 2931 (1498/1433) authors definition


The FDA-curated DILIrank data set (2016) categorized 1036
FDA-approved drugs into four groups: “ [v] Most-,” “ [v] Less-,” and “ [v] NoDILI-concern” with clear causal evidence of liver injury, and
“Ambiguous-DILI-concern” without a clear causal relationship. [17] In
this study, drugs categorized as “ [v] Most-DILI concern” or “ [v] Less-DILI
concern” were defined as “DILI positive,” while those categorized as“ [v] No-DILI concern” were defined as “DILI negative.” The drugs
categorized as “Ambiguous DILI Concern” were excluded. After
checking for structural validity referenced to PubChem database and
discarding unidentified compounds, the resulting data set contained
720 drugs, including 452 “DILI positive” and 268 “DILI negative”.
The DILIst data set was an expanded set of the DILIrank data set
consisted of 1279 drugs with DILI binary classification, created by the
FDA in 2020. [25] DILIst was established by sequentially merging the
DILIrank data set (without the terminology of “Ambiguous DILIconcern” drugs) with four other data sets containing more than 350
drugs with the human DILI classification. The four data sets were
established based on different approaches, including a clinical
evidence-based approach (LiverTox), [26] a literature-based approach
(Greene), [29] a case registry-based approach (Suzuki), [30] and an
approach based on curating data from the FDA Adverse Event
Reporting System (Zhu). [31] After removing biologics, mixtures, and
inorganics, the final DILIst data set included 1002 drugs, of which 604
are “DILI positive” and 398 are “DILI negative.”
The Yan et al. data set was a large, comprehensive data set curated
in 2022. [37] It was collected from five literature data sets (Greene, [29]

Xu, [32] Mulliner, [33] Shuaibing, [18] and LiverTox [26] ) and two public
databases (LTKB [27] and DILIrank [17] ). The SMILES of all compounds


**1719** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article



was first converted to canonical SMILES format, followed by the
removal of duplicate drugs, mixtures, and inorganic compounds. The
DILI label for the two public databases was retained, and the DILI
label for the remaining data sets was determined by the voting rules.
The rules were described as follows: if the label of a drug is consistent
in 80% or all data sets, the label of the drug will be retained;
otherwise, the drug will be deleted. After the above procedures, the
Yan et al. data set contained 2931 drugs, with 1498 classified as “DILI
positive” and 1433 as “DILI negative.”
**2.2. Framework of GeoDILI.** _2.2.1. Overview._ GeoDILI consists
of two main components: a GNN-based DILI prediction model and a
dominant substructure identification model. GeoDILI takes a
molecular canonical SMILES as input and encodes it into a 32-dim
vector using a fine-tuned geometry-based GNN (GeoGNN) model. [58]

The encoded vector is then fed into a residual network (ResNet) for
binary DILI classification. To identify DILI-related dominant
substructures, a gradient-based algorithm is used to calculate the
contribution score of each atom, and the larger substructures
composed of atoms with significantly high atomic contribution scores
are identified as dominant substructures. [66] The overall aim of
GeoDILI is to accurately predict DILI and identify the key
substructures that contribute to DILI, which can help in the design
of safer drugs.
_2.2.2. DILI Prediction._ A molecule is made up of atoms and bonds
connecting them, intuitively depicted as an atom-bond graph
represented as _G_ = (, ) by taking the atoms _i_ as the
nodes of _G_ and the bonds ( _i_, _j_ ) ∈ connecting atom _i_ and atom _j_ as
the edges of _G_ . Considering the vital role of molecular geometric
information in toxicity prediction tasks, we used a pretrained
molecular geometric representation model named GeoGNN, that
incorporates angular information. [58] A bond-angle graph _H_ is
introduced in the GeoGNN model, which is similar to the definition
of _G_ . The bond-angle graph _H_ = (, ) is defined by regarding the
bonds ( _j_, _i_ ) ∈ as the nodes of _H_ and the angles _j i k_ (,, )
connecting bond ( _j_, _i_ ) and bond ( _i_, _k_ ) as the edges of _H_ . The initial
features of atoms, bonds, and angles were calculated according to the
[items listed in Table S1.](https://pubs.acs.org/doi/suppl/10.1021/acs.chemrestox.3c00199/suppl_file/tx3c00199_si_001.pdf)
In GeoGNN, two blocks are used to learn and update features
based on the angle-bond-atom order of the bond-angle graph _H_ and
atom-bond graph _G_, respectively. For the bond-angle graph _H_,
GeoGNN aggregates messages from all of the bonds adjacent to bond
( _j_, _i_ ) and the corresponding bond angles to learn and update the
representation vector of bond ( _j_, _i_ ). At the _t_ th iteration, the aggregated
messages a _ji_ ( _t_ ) and hidden representation vector h _ji_ ( _t_ ) on a given bond
( _j_, _i_ ) are expressed by



where ( ) is the set of atoms adjacent to atom _i_ _i_, A atom‑bond is the
aggregate function in the atom-bond graph _G_, while U atom‑bond is the
update function.
At the last iteration, GeoGNN uses an average pooling function to
aggregate the atoms’ representation vectors to obtain the molecular
representation vector h _G_, which is expressed by


**h** _G_ = _R_ ( **h** ( ) _jT_ ), _j j_ : (5)


where _T_ is the total number of iterations. _R_ is the average pooling
function, which can be a complex nonlinear function. The resulting
32-dim representation vector h _G_ captures the molecular information
and is used as an input for the downstream ResNet.
A fully connected neural network (FCNN) is susceptible to the
well-known issue of vanishing/exploding gradients as its depth
increases, which can hinder model convergence and lead to
degradation problems even when convergence is achieved. To solve
the problem, ResNet introduces a shortcut connection, enabling
smoother information flow within the network and speeding up
convergence. Our ResNet consists of a fully connected layer, four
residual blocks, and a final fully connected layer. Each residual block
contains two fully connected layers with a specified dropout rate and
an identity mapping shortcut connection between the input and
output layers. The output of ResNet is used to predict the probability
of a compound causing DILI. The overall framework of the DILI
prediction model is shown at the top of Figure 1.
_2.2.3. Dominant Substructures Derivation._ As the graph isomorphism network (GIN) convolution layer naturally preserves
geometries that are lost in fully connected layers, the last GIN
convolution layer of the GeoGNN block can provide an optimal
balance between high-dimensional semantics and molecular geometry
information. [66] Specifically, the output of the final GIN convolution
layer is represented by _L_ _m n_ ×, where _m_ is the number of atoms in
the compound and _n_ is the number of channels. For a given
compound A, the importance of each channel can be calculated as



_m_



1 _P_
_k_ = _m_ _i_ =1 _L_ A,



= _m_ _i_ =1 _L_ A _i k_,



=1 _L_ A, (6)



( ) _jit_ = A ( )bond angle _t_ ( ({ **h** ( _jit_ 1), **h** ( _jkt_ 1)



( ) _jit_ = A ( )bond angle _t_ ( ({ **h** ( _jit_ 1), **h** ( _jkt_ 1), **x** _kji_



A _t_ ( ({ **h** _t_, **h** _t_, **x** ), _k k_ : ( ) _j_



= A ( ({ **h**, **h**, **x** ), _k k_ : ( ) _j_ }



**a** = A **h** **h** **x**



_k k_ : _j_



( _t_ 1) ( _t_ 1)



( _jit_ 1), **h** ( _ikt_ 1), **x** _jik_



( **h**, **h** _ikt_, **x** ), _k k_ : ( ) ) _i_ }



{( **h** _ji_, **h** _ik_, **x** _jik_ ), _k k_ : ( ) ) _i_ } (1)



where _L_ A _i_, _k_ denotes a neuron at the _k_ th channel and atom _i_ of
compound A, and _P_ represents the probability of compound A
causing DILI.
Then, we determined the atomic contribution scores vector
_W_ A _m_ of compound A by the weighted combination, which can
be expressed as


_n_



**h** **h** **x**



_k k_ : _i_



**h** ( ) _jit_ = U ( )bond angle _t_ ( **h** ( _jit_ 1), **a** ( ) _jit_ ) (2)


where ( ) and _j_ ( ) are the sets of atoms adjacent to atom _i_ _j_ and
a t o m _i_, r e s p e c t i v e l y,
{(, ), _j k_ _k k_ : ( ) _j_ } {(, ), _i k_ _k k_ : ( ) _i_ } is the set of
bonds adjacent to bond ( _j_, _i_ ). A bond‑angle is the function for aggregating
messages, while U bond‑angle is the function for updating the
representation vectors of the bonds.
For the atom-bond graph _G_, GeoGNN takes the updated
representation vectors h ( _jit_ ) of bond ( _j_, _i_ ) from _H_ as the features for
bonds in _G_ . The aggregated messages a _j_ ( _t_ ) and hidden representation
vector h _j_ ( _t_ ) of atom _j_ for the _t_ th iteration can be represented using the
following formula


**a** ( ) _jt_ = A ( )atom bond _t_ ( **h** ( _jt_ 1), **h** ( _i_ _t_ 1), **h** ( _jit_ 1) ), _j j_ : ( ) _i_ (3)


**h** ( ) _jt_ = U ( )atom bond _t_ ( **h** ( _jt_ 1), **a** ( ) _jt_ ) (4)



finally, we used min−max normalization to map _W_ A ranging from 0 to
1.

After the contribution scores of each atom in the molecule were

obtained, the dominant substructures are extracted. More concretely,
during the fragmentation step, each bond in the molecule (excluding
ring bonds) is broken to generate two fragments. After iterations
using a recursive algorithm, all possible substructures with 4−18
atoms are collected. [67][,][69] Then, the one-sided Wilcoxon test was
conducted to determine if a given substructure exhibits significantly
higher atom attention weights than the remaining part of the
molecule, with a _p_ -value threshold set at 0.05. [70] Finally, only the
largest substructures in each molecule were kept to eliminate
redundancy.
The framework of dominant substructure derivation is depicted at
the bottom of Figure 1.


**1720** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730



A = _k_ _L_ A _k_


_k_ =1



_W_ = _L_



_k_



=1 (7)


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article



**2.3. Model Construction and Optimization.** The training and
test process of GeoDILI are shown in Algorithm 1.


_2.3.1. Input._ As shown in Table 2, the data set was split into a
training set D [train] = {( _G_, _H_, _y_ )} and a test set D [test] = {( _G_, _H_, _y_ )} in a


Table 2. Distribution of “DILI-Positive” and “DILINegative” Samples in the DILIrank/DILIst/Yan et al.
dataset


data set DILI-class training test total


DILIrank positive 362 90 452

negative 214 54 268

total 576 144 720

DILIst positive 483 121 604

negative 318 80 398

total 801 201 1002

Yan et al. positive 1198 300 1498

negative 1146 287 1433

total 2344 587 2931


ratio of 8:2, where _y_ denotes the label of the corresponding molecule
represented by ( _G_, _H_ ). To address the problem of limited sample size
in the available data sets, the model parameters were fine-tuned using
fivefold cross-validation to maximize the utilization of the data sets for
effective model training. Subsequently, we evaluated the model’s
performance in the test set (Table 2).
_2.3.2. Model._ The GeoGNN block is denoted as _u_ _w_ (·), where _w_
represents the set of model parameters. We initialized the parameters
of the pretrained self-supervised learning model and then fine-tuned
them by the downstream ResNet _d_ _θ_ (·).
_2.3.3. Optimization._ To prevent overfitting, we applied an early
stop strategy based on the evaluation results of the training set and
validation set. We also employed the Free Large-scale Adversarial
Augmentation on Graphs (FLAG) approach, which introduces
gradient-based adversarial perturbations to the input node features
while keeping the graph structure unchanged. [71] FLAG helps to
generalize our model to out-of-distribution samples, reducing
overfitting and improving the performance on the test set.
Hyperparameters were optimized using grid search, and the final
settings are bolded in Table 3.
**2.4. Statistics for the Model Evaluation Criteria.** Several
evaluation metrics were used to evaluate the performance of our
model, including the receiver-operating characteristic-area under the
curve (AUC), accuracy (ACC), precision, sensitivity, specificity, F1score, and Matthews correlation coefficient (MCC). [72][−][76] AUC
quantifies the model’s ability to distinguish between “DILI positive”



Table 3. Hyperparameters Settings _[a]_


hyperparameters values


batch size 128, 256
GeoGNN learning rate 1 × 10 [−][3], 1 **×** 10 [−][4], 5 × 10 [−][5]

ResNet learning rate 1 × 10 [−][3], 1 **×** 10 [−][4], 5 × 10 [−][5]

dropout 0.2, 0.35, 0.4
optimizer Adam, AdamW

_a_ Best hyperpara meters are marked as bold.


and “DILI negative” labels. [72] ACC represents the overall percentage
of correct DILI label predictions. [73] Precision is the fraction of
correctly predicted positive samples among all predicted positives. [74]

Sensitivity measures the proportion of actual “DILI positive” drugs
that were correctly predicted as such. [74][,][75] Specificity (true-negative
rate) indicates the percentage of drugs with the “DILI negative” label
that were correctly predicted as such. [75] F1-score, which is the
weighted harmonic mean of precision and recall, balances both
metrics. [74] MCC, which takes into account all four categories (true
positives, false negatives, true negatives, and false positives) of the
binary confusion matrix, provides a comprehensive measure of the
model’s performance. [76] The mathematical formulas for these
evaluation metrics are specified below

accuracy= (TP + TN)
(TP + TN + FP + FN) (8)


TP
precision=
(TP + FP) (9)


TP
sensitivity recall= =
(TP + FN) (10)


TN
specificity=
(TN + FP) (11)


F1 score = 2 × precision recall×
(precision + recall) (12)



TP TN× FP FN×
MCC=



TP TN× FP FN×


=





(13)

where TP, TN, FP, and FN denote true positive, true negative, false
positive, and false negative, respectively.
**2.5. Identification of DILI Structural Alerts.** We derived
dominant substructures for all true positives and -negatives in the
DILIrank data set and identified DILI-related significant substructures
using Fisher’s exact test. The test statistic is the number of
compounds with a specific dominant substructure in the set of
“DILI positive” compounds, which follows a hypergeometric
distribution (Table 4).


Table 4. Contingency Table for a Certain Dominant
Substructure _T_


DILI positive DILI negative


with dominant substructure _T_ _M_ _M_ ′

without dominant substructure _T_ _N_ _N_ ′


The formula for the _p_ -value can be represented as follows [77]



i



+ _M_ _N_ +



yi

zzzzjjjj
{k



i _N_ + _N_ y

jjjj zzzz
k _N_ {



zzzz

{



zzzz
{



_P_



i _M_ + _M_ y

jjjj zzzz
k _M_ {



i



i _M_ + _N_ + _M_ + _N_ y

jjjj zzzz



=



_M_ + _M_ _N_ + _N_

zzzzjjjj

_M_ {k _N_

_M_ + _N_ + _M_ + _N_

_M_ + _N_



k



+ _N_ + _M_ + _N_

zzzz

+ _N_ { (14)



**1721** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article


Figure 2. (A) Tanimoto similarity heatmap of the compounds in the DILIrank data set using Morgan fingerprint. (B) The t-SNE distribution of the
compounds labeled “DILI positive” and “DILI negative” in the DILIrank data set.


Figure 3. Physicochemical property distributions of compounds with “DILI positive” and “DILI negative” in the DILIrank data set. Log _P_, the
Wildman−Crippen log _P_ value; [83][,][84] MolWt, the average molecular weight of the molecule; [83] TPSA, topological polar surface area; [85] rotatable
bonds, the number of rotatable bonds; [83] H-bond acceptors, the number of hydrogen bond acceptors; [83] H-bond donors, the number of hydrogen
bond donors. [83]



In addition, enrichment factor (EF) was used to identify true
positives compared to a random selection. [78] Higher EF indicates
better enrichment of true positives in the top ranking. The formula of
EF is


_M_
precision= _P X_ ( = |1 _T_ ) =
( _M_ + _M_ ) (15)


EF= _P X_ ( = 1| _T_ ) = _M_ /( _M_ + _M_ )
_P X_ ( = 1) ( _M_ + _N_ )/( _M_ + _N_ + _M_ + _N_ ) (16)


where _P_ ( _X_ = 1| _T_ ) is the probability of drug _X_ containing substructure
_T_ as “DILI positive.”
To obtain substructures with both a high precision and high
coverage rate, we employed information gain (IG). [79] IG measures the
difference in information entropy before _H_ ( _X_ ) and after separation by
substructure _H_ ( _X_ | _T_ ). [62][,][80]


IG( ) _T_ = _H X_ ( ) _H X T_ ( | ) (17)



**3. RESULTS**


**3.1. Data Analysis.** To visualize the chemical diversity of
compounds in the DILIrank data set, we employed the
Tanimoto similarity analysis and the t-distributed stochastic
neighbor embedding (t-SNE) algorithm using molecular
Morgan fingerprints. [81][,][82] The Tanimoto coefficient is a widely
utilized metric to measure the chemical structure similarity
between two molecules. The Tanimoto similarity heatmap in
Figure 2A reveals high compound diversity with an average
similarity of 0.105. The t-SNE algorithm is an unsupervised
ML method that transforms high-dimensional data into a lowdimensional representation while preserving the relationships
between the data points. [81] Figure 2B shows that the
distributions of “DILI positive” and “DILI negative” compounds largely overlap, indicating limited differences in their
chemical structures.
We further analyzed the distribution of physicochemical
properties between “DILI positive” and “DILI negative”


**1722** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article


[̈]

[̈]

[̈]



compounds in the DILIrank data set. Results shown in Figure
3 reveal statistically significant differences in LogP values
between the two categories, with “DILI positive” compounds
being more lipophilic (median log _P_ 2.55 for “DILI positive” vs
1.62 for “DILI negative”, _p_ < 0.001). However, no significant
differences were observed for the other five molecular
descriptors ( _p_ - 0.05). Similar analyses on the DILIst and
[Yan et al. data sets are presented in Figures S1 and S2, and](https://pubs.acs.org/doi/suppl/10.1021/acs.chemrestox.3c00199/suppl_file/tx3c00199_si_001.pdf)
[Table S2. These findings highlight the challenges in developing](https://pubs.acs.org/doi/suppl/10.1021/acs.chemrestox.3c00199/suppl_file/tx3c00199_si_001.pdf)
accurate and robust DILI prediction models due to the
complexity of the DILI toxic end point and the difficulty in
distinguishing between “DILI positive” and “DILI negative”
compounds solely based on chemical structures and physicochemical properties.
**3.2. Model Performance.** _3.2.1. Model Construction and_
_Evaluation._ We trained the model for a total of 100 epochs
with a batch size of 128. The optimizer used for training was
AdamW, which is an extension of the Adam optimizer that
incorporates weight decay to avoid overfitting with a learning
rate of 1 × 10 [−][4] and a dropout of 0.2. During the training
process, the network parameters were updated by minimizing
the binary cross-entropy loss function. In the fine-tuning stage,
we transferred the parameters from the pretrained model to
the current prediction task. In addition, we utilized the FLAG
strategy to improve the model’s robustness and reduce
overfitting by generating and incorporating adversarial nodes
into the graph data. [71] We performed fivefold cross-validation
runs with different random seeds to evaluate the reliability of
the classification models. The performance of GeoDILI in the
DILIrank data set is summarized in Figure 4 and Table 5. For

[̈]

[̈]

[̈]



Figure 4. Receiver operating characteristic curves of the fivefold crossvalidation and the test set in the DILIrank data set.

[̈]

[̈]

the training set given by the 5-fold cross-validation method, the
mean and standard deviation of the AUC of the validation set [̈]
were 0.881 and 0.010, respectively. The final model was
retrained throughout the whole training set and further
validated with the test set, obtaining a good performance
with an AUC value of 0.908.



_3.2.2. Comparison with Other Models._ Although several
DILI prediction models have been published, it is difficult to
compare their performance directly as the data sets and source
codes are not available. [38][,][54] To make a fair comparison of our
model with previous DILI models and other popular models,
we performed a systematic comparison using the same training
and test data.
First, we compared our model GeoDILI with the opensource SOTA DILI models from the past 3 years, of which
only three are available: CNN-MFE, [35] DeepDILI, [36] and R-EGA (Table 5). [37] In comparison with the current DILI
prediction models, our model GeoDILI achieved the best
result in most evaluation metrics, with accuracy, precision,
sensitivity, F1 score, specificity, and MCC of 0.875, 0.860,
0.956, 0.905, 0.875, and 0.732, respectively. Sensitivity, a
crucial indicator of the ability to correctly identify hepatotoxicity, is particularly high in our model (0.956). This is
important for drug development, as it enables drug developers
to conduct further verification to determine the potential for
adverse effects from candidate compounds. The well-balanced
sensitivity/specificity ratio of 0.956:0.875 further demonstrates
the efficacy of the GeoDILI model. Additionally, the F1-score
and MCC, important metrics for an imbalanced data set,
indicate a substantial performance improvement compared to
the other evaluated models for the DILI prediction task.
Then, we compared GeoDILI to seven pretrained GNN
models, including GEM, [58] GIN_AttrMasking, GIN_ContextPred, GIN_InfoMax, GIN_EdgePred, [86] 3D InfoMax and
GraphMVP, [87][,][88] and four popular GNN models, which are
AttentiveFP, [89] directed message passing neural network (DMPNN), [90] graph convolutional network (GCN), graph
attention network (GAT). As shown in Table 5, GeoDILI
outperforms most metrics among all GNN models (pretrained
and popular GNNs) on the DILIrank data set. Among the
pretrained and popular GNNs, the pretrained GNNs exhibit
superior performance compared to the popular GNNs.
Specifically, GeoDILI, GEM, [58] and GIN_ContextPred [86]

achieve the highest AUCs of 0.908, 0.874, and 0.866,
respectively. Correspondingly, their ACCs are 0.875, 0.861,
and 0.778, respectively. These results highlight the effectiveness of the pretrained strategy in predicting DILI. Furthermore, most pretrained GNNs exhibited a notable balance
between sensitivity and specificity, with a sensitivity/specificity
ratio close to 1. This finding implies that these pretrained
models effectively identified both positive and negative samples
in DILI prediction. Moreover, both GeoDILI and GEM
outperformed the remaining four pretrained GNNs with
significantly higher AUC values (0.908 and 0.874), indicating
that they were more effective in predicting DILI by leveraging
geometric information. Additionally, GeoDILI exhibits significantly better performance than GEM, which demonstrates
the suitability and efficacy of our model.
Finally, we compared our proposed model GeoDILI to eight
fingerprint-based ML models, [91] including DNN, k-nearest
neighbors (k-NN), SVM, Naive Bayes, Decision Tree, Random [̈]
Forest, Gradient Boosting, and AdaBoost. Among the ML
models, the Naive Bayes and Random Forest have the best [̈]
prediction performance, with AUC and ACC of 0.860, 0.819
for the Naive Bayes algorithm and 0.849, 0.813 for Random [̈]
Forest, respectively (Table 5). These models have also been
widely used in previous DILI prediction tasks. [22][,][39][,][42] However,
most ML models display low MCC due to the imbalance
between sensitivity and specificity with higher sensitivity and


**1723** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article


Table 5. Comparison of Different Models on the DILIrank Dataset _[a]_


model AUC ACC precision sensitivity F1-score specificity MCC


GeoDILI 0.908 0.875 0.860 0.956 0.905 0.875 0.732

CNN-MFE 0.866 0.854 0.879 0.889 0.884 0.796 0.688

DeepDILI 0.756 0.692 0.766 0.702 0.733 0.675 0.372

R-E-GA 0.882 0.847 0.840 0.933 0.830 0.704 0.670

GEM 0.874 0.861 0.850 0.944 0.895 0.861 0.701

GIN_AttrMasking 0.822 0.819 0.905 0.809 0.854 0.840 0.626
GIN_ContextPred 0.866 0.778 0.780 0.886 0.830 0.607 0.522
GIN_EdgePred 0.798 0.736 0.825 0.733 0.776 0.741 0.462
GIN_InfoMax 0.829 0.764 0.872 0.739 0.800 0.808 0.527

3D InfoMax 0.778 0.723 0.737 0.837 0.784 0.550 0.408

GraphMVP 0.860 0.792 0.816 0.870 0.842 0.654 0.539

AttentiveFP 0.843 0.806 0.833 0.900 0.865 0.591 0.522

D-MPNN 0.821 0.757 0.798 0.820 0.808 0.652 0.480

GCN 0.835 0.778 0.755 0.930 0.833 0.552 0.536

GAT 0.830 0.736 0.731 0.884 0.800 0.517 0.439

DNN 0.764 0.690 0.735 0.727 0.731 0.507 0.334

k-NN 0.795 0.750 0.807 0.789 0.798 0.685 0.471

SVM 0.829 0.799 0.790 0.922 0.851 0.593 0.561

Naive Bayes [̈] 0.860 0.819 0.833 0.889 0.860 0.704 0.609

Decision Tree 0.694 0.688 0.800 0.667 0.727 0.722 0.377

Random Forest 0.849 0.813 0.812 0.911 0.859 0.648 0.592

Gradient Boosting 0.822 0.778 0.774 0.911 0.837 0.556 0.513

AdaBoost 0.785 0.764 0.792 0.844 0.817 0.630 0.487

_a_
Higher is better. Best results are marked as bold. The models listed in the table have the same training and test sets, as indicated by Table 2.


Table 6. Summary of the Optimal Performance of the Published Models Using DILIrank/DILIst/Yan et al. dataset _[a]_


author data set AUC ACC precision sensitivity F1-score specificity MCC


This work DILIrank 0.908 0.875 0.860 0.956 0.905 0.875 0.732
He et al. [18] 0.859 0.783 / 0.818 / 0.748 /
Wang et al. [19] 0.804 0.817 / 0.646 / 0.962 /
Mora et al. [20] / 0.810 / 0.817 / 0.793 0.566
Ancuceanu et al. [21] / / / 0.890 / 0.565 /
Liu et al. [22] 0.824 0.763 0.736 0.724 0.730 0.794 0.523
Jaganathan et al. [23] / 0.811 / 0.840 / 0.783 0.623
Kang, M.-G. and Kang, N. S. [24] / 0.731 / 0.714 / 0.750 /

This work DILIst 0.851 0.786 0.820 0.826 0.823 0.786 0.553
Li et al. [36] 0.659 0.687 / 0.805 0.755 0.51 0.331
Lim et al. [55] 0.691 0.687 / / 0.784 / 0.338

This work Yan et al. 0.843 0.773 0.781 0.846 0.788 0.773 0.549
Yan et al. [37] 0.842 0.770 / / 0.769 / /

_a_ Higher is better. Best results are marked as bold. “/” denotes no result for this metric.




[̈]


lower specificity. For example, SVM has a sensitivity of 0.922
and specificity of 0.593, which suggests that the model is
biased toward the positive class. This may be due to the ratio
of positive to negative samples in the DILIrank data set being
nearly 6:4. In summary, GeoDILI not only has good prediction
performance but also predicts positive and negative compounds in a more balanced way than the other models.
_3.2.3. Comparison of the Three DILI Data Sets._ The way
data are collected greatly affects the performance of a model,
particularly because acquiring human hepatotoxicity data is a
challenging task. In this study, we evaluated two FDAapproved DILI data sets (DILIrank and DILIst) and a recent
and larger data set (Yan et al.) by comparing the reported
results of the same data sets and systematically comparing the
performance of different models on the three data sets.
DILIrank is the first DILI benchmark data set released by
the FDA in 2016, which is the most widely employed data set




[̈]


and has been applied by several models. [17] We compared
GeoDILI’s performance with reported models and found that
GeoDILI outperforms other models in most metrics using the
same data set. One exception is that Wang’s model had a high
specificity of 0.962, but its sensitivity/specificity ratio was
0.646:0.962, indicating that the model is biased toward the
negative class (Table 6). [19] DILIst is an expanded data set of
DILIrank by the FDA. [25] We searched for relevant work that
employed the DILIst data set and compared our results with
theirs. As Table 6 demonstrates, GeoDILI outperforms these
models in all metrics. The Yan et al. data set is a newly
published large data set curated in 2022. [37] The results show
that our metrics are superior to those of their model,
demonstrating the power of GeoDILI.
We compared the performance of GeoDILI with other
popular models using the same training and test sets in three
[data sets (Tables 5, 6, S3 and S4). It is worth noting that the](https://pubs.acs.org/doi/suppl/10.1021/acs.chemrestox.3c00199/suppl_file/tx3c00199_si_001.pdf)


**1724** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article


Table 7. Summary of the Significant SAs _[a]_


_a_ No. of positive, the number of “DILI-positive” compounds containing this SA; no. of negative, the number of “DILI-negative” compounds
containing this SA; EF, enrichment factor; IG, information gain.



performance of GeoDILI optimal models varied across the
three data sets, with the best performance on DILIrank (AUC
0.908 and ACC 0.875), followed by DILIst (AUC 0.851 and
ACC 0.786), and the worst on the Yan et al. data set (AUC
0.843 and ACC 0.773).
Moreover, to explore the performance differences between
the three data sets, we conducted a comparative analysis based
on data set overlap, their distribution in chemical space, and
[the similarity between training and test sets (Figure S4).](https://pubs.acs.org/doi/suppl/10.1021/acs.chemrestox.3c00199/suppl_file/tx3c00199_si_001.pdf)
Unfortunately, these aspects failed to yield a reasonable
explanation. We hypothesized that the performance disparities
may be attributed to inherent data characteristics, potentially
related to the data source and quality. Consequently, it is
crucial to use a uniform benchmark data set in the DILI
prediction tasks.
**3.3. Structural Alerts Analysis.** We performed a
substructure inference on all true-positive and true-negative
samples in the DILIrank data set and used statistical metrics
such as EF, _p_ -value, and IG to assess the quality of SAs and
identify DILI-related significant substructures. We set thresholds of EF > 1, _p_ -value < 0.05, and IG > 0.001 to filter out
insignificant substructures and ranked the remaining SAs based
on their IG values. Seven significant SAs were identified and
are listed in Table 7.



_N_ -benzylformamide moiety (no. 1) is present in 18
compounds, all exhibiting severe or mild hepatotoxicity
[(Table S4). The containing amide moiety is present as a](https://pubs.acs.org/doi/suppl/10.1021/acs.chemrestox.3c00199/suppl_file/tx3c00199_si_001.pdf)
lactam structure in eight structures, including the antiepileptic
drug phenytoin. [92] These 18 drugs have diverse structures with
molecular weights ranging from 252 to 1449 and often coexist
with other SAs, such as the chlorobenzene and sulfanilamide
moiety in the withdrawn drug chlormezanone and the
hydrazine moiety in nialamide. Chlorobenzene moiety (no.
2) is known to be metabolized in the liver by cytochrome P450
enzymes to form epoxides, which can bind to proteins, DNA,
and RNA, contributing to its toxicity. [93] Induction of the
cytochrome P450 system can increase the rate of formation of
these epoxides and thus the toxic effects, especially on the
liver. [93] Exposure to high levels of chlorobenzene has been
shown to cause liver damage as well as other adverse effects
such as kidney damage and neurological symptoms.
Sulfonamide moiety (no. 3) is known to cause idiosyncratic
liver injury that exhibits features of drug allergy or hypersensitivity. They have been associated with cases of acute liver

−
failure and remain among the top 5 10 causes of druginduced, idiosyncratic fulminant hepatic failure. [94] The aniline
moiety (no. 4) is commonly associated with severe toxicities. [95]

It is present in 100 hepatotoxic compounds and 32


**1725** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article



nonhepatotoxic compounds according to DILIrank, giving it a
relatively higher coverage (18.3%) but lower precision (0.758).
2-Azetidinone moiety is a part of the derived substructure of
no. 5 and no. 7, which is a common structural feature of many
broad-spectrum _β_ -lactam antibiotics. These antibiotics have
been associated with minor liver injuries. [96] Hydrazine
compound (no. 6) is a known human carcinogen that can
also cause hepatic necrosis leading to acute liver failure. [97][,][98]


**4. DISCUSSION**

The development of novel artificial intelligence approaches
based on publicly available large-scale toxicity data is urgently
needed to generate accurate predictive models for chemical
toxicity evaluation in the early stages of drug development.
However, the lack of internal consistency in publicly available
data presents a significant challenge to building accurate
predictive models. In this study, we systematically compared
two FDA-published DILI-standardized data sets with a recently
published large data set and found that the optimal
performance of the model differs across data sets. Smaller
but higher quality data set DILIrank showed the best
performance. [16][,][22] Due to the lack of a large-scale DILI
benchmark data set, reported DILI prediction models were
built on different data sets, making it difficult to compare their
performance. [34][−][38] Therefore, establishing a DILI benchmark
data set and collecting more high-quality DILI data are
imperative.
Besides the data set, the way molecules are characterized
plays a decisive role in the prediction results. Most DILI
prediction models are built based on molecular descriptors,
molecular fingerprints, or a combination of both. [19][,][20][,][36][,][37] In
recent years, many pretrained molecular representation
learning methods have been proposed, such as MolMap, [99]

GEM, [58] ImageMol, [100] and so on. These models are trained on
large-scale unlabeled molecules in a self-supervised methodology to extract low-dimensional features from the input data.
As shown in this study, due to the high structural diversity and
small size of the DILI data set, using predefined fingerprints
and simple graph representations may not fully capture
molecular features and lead to poor prediction performance.
By employing a pretraining scheme and fine-tuning on a small
number of labeled molecules, significant improvements can be
achieved in DILI prediction models.
SAs are commonly used in toxicology to quickly identify
potentially toxic molecules, such as those with genotoxicity and
mutagenicity, endocrine disruptors, skin sensitivity, hepatotoxicity, and more. [101] A recent review discussed that the methods
for investigating SAs can be divided into three categories:
expert systems, frequency analysis, and interpretable ML
models, each with its own advantages and disadvantages. [62]

Expert systems focus on the mechanics of SAs and often result
in false positives. [59][,][68] In contrast, frequency-based and ML
methods have been shown to outperform expert inferencebased methods in terms of prediction accuracy. [61][,][63] The
frequency-based approaches are commonly used methods that
involve statistical methods, such as _p_ -value, precision, and
enrichment factor to assess substructures occurring more
frequently in toxic compounds than in nontoxic ones. [69]

Interpretable neural networks represent a promising new
direction due to their ability to identify SAs by optimizing
complex neural network parameters to achieve high predictive
performance while providing structural information about how
it is predicted. [66][,][67] In this study, we utilized an attention-free



GNN interpretation method to obtain atom-level weights for
DILI prediction and then extracted dominant substructures. [66]

We applied this method to DILIrank to obtain seven highprecision SAs. Their potential toxic mechanisms were
discussed in the results section, demonstrating that the
obtained SAs with both high precision and mechanistic
interpretability.
QSAR models can rapidly predict large amounts of new
compounds and prioritize toxic ones, but they have limitations
in practicality and discrimination. [43][−][46][,][48][,][49] Classic QSAR
models for hepatotoxicity only use structure information and
oversimplify the problem into a binary classification, missing

−
out relevant toxicity assays data such as dose response
relationships and assays that evaluate relevant mechanisms
such as oxidative stress and mitochondrial reductive activity. [103]

Consequently, these models are unable to fully utilize the
information for in vitro to in vivo extrapolation and accurately
differentiate between similar compounds with different
toxicities. Future models should incorporate more information,
such as in vitro activity data and target information, to establish
more robust and reliable toxicity prediction models. [102]


**5. CONCLUSIONS**

In this work, we proposed a robust, interpretable end-to-end
DILI predictor GeoDILI for safe drug development. We
systematically compared our model with other models using
the same data set and demonstrated its superior performance.
We utilized a molecular geometric representation strategy that
includes angle information to provide more accurate
predictions for stereoisomeric molecules. Additionally, our
model was capable of deducing dominant substructures, which
can provide suggestions for molecular structure optimization.

# ■ [ASSOCIATED CONTENT]

**Data Availability Statement**
All data involved in this study and the source code of GeoDILI
[are available at https://github.com/CSU-QJY/GeoDILI.](https://github.com/CSU-QJY/GeoDILI)
 - **sı** **Supporting Information**
The Supporting Information is available free of charge at
[https://pubs.acs.org/doi/10.1021/acs.chemrestox.3c00199.](https://pubs.acs.org/doi/10.1021/acs.chemrestox.3c00199?goto=supporting-info)

Similarity heat maps and t-SNE plots of data sets,
physicochemical property analysis, receiver operating
characteristic curves, comparison of the three DILI data
sets, and tables with initial features, statistics for
physicochemical properties, additional model metrics,
[and drugs containing no. 1 SA (PDF)](https://pubs.acs.org/doi/suppl/10.1021/acs.chemrestox.3c00199/suppl_file/tx3c00199_si_001.pdf)
[Model training and test sets (ZIP)](https://pubs.acs.org/doi/suppl/10.1021/acs.chemrestox.3c00199/suppl_file/tx3c00199_si_002.zip)

# ■ [AUTHOR INFORMATION]

**Corresponding Authors**

Xiaoqing Guan − _Institute of Interdisciplinary Integrative_
_Medicine Research, Shanghai University of Traditional_
_Chinese Medicine, Shanghai 201203, China;_ [orcid.org/](https://orcid.org/0000-0002-4755-820X)
[0000-0002-4755-820X; Email: guanxq@shutcm.edu.cn](https://orcid.org/0000-0002-4755-820X)
Qingping Zhou − _School of Mathematics and Statistics,_
_Central South University, Changsha, Hunan 410083, China;_

[orcid.org/0000-0003-1530-3525; Email: qpzhou@](https://orcid.org/0000-0003-1530-3525)
[csu.edu.cn](mailto:qpzhou@csu.edu.cn)


**Authors**

Wenxuan Wu − _Institute of Interdisciplinary Integrative_
_Medicine Research, Shanghai University of Traditional_


**1726** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article


̈



_Chinese Medicine, Shanghai 201203, China;_ [orcid.org/](https://orcid.org/0000-0002-5371-1186)
[0000-0002-5371-1186](https://orcid.org/0000-0002-5371-1186)
Jiayu Qian − _School of Mathematics and Statistics, Central_
_South University, Changsha, Hunan 410083, China;_

[orcid.org/0009-0004-9929-3444](https://orcid.org/0009-0004-9929-3444)
Changjie Liang − _Institute of Interdisciplinary Integrative_
_Medicine Research, Shanghai University of Traditional_
_Chinese Medicine, Shanghai 201203, China_
Jingya Yang − _School of Mathematics and Statistics, Central_
_South University, Changsha, Hunan 410083, China_
Guangbo Ge − _Institute of Interdisciplinary Integrative_
_Medicine Research, Shanghai University of Traditional_
_Chinese Medicine, Shanghai 201203, China;_ [orcid.org/](https://orcid.org/0000-0002-9670-4349)
[0000-0002-9670-4349](https://orcid.org/0000-0002-9670-4349)


Complete contact information is available at:
[https://pubs.acs.org/10.1021/acs.chemrestox.3c00199](https://pubs.acs.org/doi/10.1021/acs.chemrestox.3c00199?ref=pdf)


**Author Contributions**
§ W.W. and J.Q. contributed equally to this paper. CRediT:
Wenxuan Wu data curation, formal analysis, investigation,
methodology, project administration, software, validation,
writing-original draft; Jiayu Qian formal analysis, investigation,
methodology, software, visualization, writing-original draft;
Changjie Liang formal analysis, investigation, software,
visualization, writing-original draft; Jingya Yang formal
analysis, investigation, methodology, software; Guangbo Ge
conceptualization, funding acquisition, resources; Qingping
Zhou conceptualization, funding acquisition, methodology,
supervision, writing-review & editing; Xiaoqing Guan
conceptualization, funding acquisition, methodology, project
administration, resources, supervision, writing-review &
editing.


**Notes**

The authors declare no competing financial interest.

# ■ [ACKNOWLEDGMENTS]


This work is supported by the National Natural Science
Foundation of China (grant nos 82003847, 81922070,
82273897 and 12101614), the Natural Science Foundation
of Hunan Province, China (grant no. 2021JJ40715).

# ■ [REFERENCES]


[(1) Shehu, A. I.; Ma, X.; Venkataramanan, R. Mechanisms of Drug-](https://doi.org/10.1016/j.cld.2016.08.002)
[Induced Hepatotoxicity.](https://doi.org/10.1016/j.cld.2016.08.002) _Clin. Liver Dis._ 2017, _21_ (1), 35−54.
(2) Navarro, V. J.; Khan, I.; Björnsson, E.; Seeff, L. B.; Serrano, J.;
[Hoofnagle, J. H. Liver Injury from Herbal and Dietary Supplements.](https://doi.org/10.1002/hep.28813)
_Hepatology_ 2017, _65_ (1), 363−373.
[(3) Tujios, S.; Fontana, R. J. Mechanisms of Drug-Induced Liver](https://doi.org/10.1038/nrgastro.2011.22)
[Injury: From Bedside to Bench.](https://doi.org/10.1038/nrgastro.2011.22) _Nat. Rev. Gastroenterol. Hepatol._ 2011,
_8_ (4), 202−211.
(4) Weaver, R. J.; Blomme, E. A.; Chadwick, A. E.; Copple, I. M.;
Gerets, H. H. J.; Goldring, C. E.; Guillouzo, A.; Hewitt, P. G.;Ingelman-Sundberg, M.; Jensen, K. G.; Juhila, S.; Klingmüller, U.;
Labbe, G.; Liguori, M. J.; Lovatt, C. A.; Morgan, P.; Naisbitt, D. J.;
Pieters, R. H. H.; Snoeys, J.; van de Water, B.; Williams, D. P.; Park,
[B. K. Managing the Challenge of Drug-Induced Liver Injury: A](https://doi.org/10.1038/s41573-019-0048-x)
[Roadmap for the Development and Deployment of Preclinical](https://doi.org/10.1038/s41573-019-0048-x)
[Predictive Models.](https://doi.org/10.1038/s41573-019-0048-x) _Nat. Rev. Drug Discovery_ 2020, _19_ (2), 131−148.
[(5) Dowden, H.; Munro, J. Trends in Clinical Success Rates and](https://doi.org/10.1038/d41573-019-00074-z)
[Therapeutic Focus.](https://doi.org/10.1038/d41573-019-00074-z) _Nat. Rev. Drug Discovery_ 2019, _18_ (7), 495−496.
[(6) Iasella, C. J.; Johnson, H. J.; Dunn, M. A. Adverse Drug](https://doi.org/10.1016/j.cld.2016.08.005)
[Reactions: Type A (Intrinsic) or Type B (Idiosyncratic).](https://doi.org/10.1016/j.cld.2016.08.005) _Clin. Liver_
_Dis._ 2017, _21_ (1), 73−87.



[(7) Hoofnagle, J. H.; Björnsson, E. S. Drug-Induced Liver Injury �](https://doi.org/10.1056/nejmra1816149)
[Types and Phenotypes.](https://doi.org/10.1056/nejmra1816149) _N. Engl. J. Med._ 2019, _381_ (3), 264−273.
[(8) Mosedale, M.; Watkins, P. Drug-Induced Liver Injury: Advances](https://doi.org/10.1002/cpt.564)
[in Mechanistic Understanding That Will Inform Risk Management.](https://doi.org/10.1002/cpt.564)
_Clin. Pharmacol. Ther._ 2017, _101_ (4), 469−480.
[(9) Dara, L.; Liu, Z.-X.; Kaplowitz, N. Mechanisms of Adaptation](https://doi.org/10.1111/liv.12988)
[and Progression in Idiosyncratic Drug Induced Liver Injury, Clinical](https://doi.org/10.1111/liv.12988)
[Implications.](https://doi.org/10.1111/liv.12988) _Liver Int._ 2016, _36_ (2), 158−165.
[(10) Blomme, E. A. G.; Yang, Y.; Waring, J. F. Use of](https://doi.org/10.1016/j.toxlet.2008.09.017)
[Toxicogenomics to Understand Mechanisms of Drug-Induced](https://doi.org/10.1016/j.toxlet.2008.09.017)
[Hepatotoxicity during Drug Discovery and Development.](https://doi.org/10.1016/j.toxlet.2008.09.017) _Toxicol._
_Lett._ 2009, _186_ (1), 22−31.
(11) Xu, J. J.; Henstock, P. V.; Dunn, M. C.; Smith, A. R.; Chabot, J.
[R.; de Graaf, D. Cellular Imaging Predictions of Clinical Drug-](https://doi.org/10.1093/toxsci/kfn109)
[Induced Liver Injury.](https://doi.org/10.1093/toxsci/kfn109) _Toxicol. Sci._ 2008, _105_ (1), 97−105.
(12) Elferink, M.; Olinga, P.; Draaisma, A.; Merema, M.;
[Bauerschmidt, S.; Polman, J.; Schoonen, W.; Groothuis, G. Micro-](https://doi.org/10.1016/j.taap.2008.01.037)
[array Analysis in Rat Liver Slices Correctly Predicts in Vivo](https://doi.org/10.1016/j.taap.2008.01.037)
[Hepatotoxicity.](https://doi.org/10.1016/j.taap.2008.01.037) _Toxicol. Appl. Pharmacol._ 2008, _229_ (3), 300−309.
[(13) McGill, M. R.; Jaeschke, H. Animal Models of Drug-Induced](https://doi.org/10.1016/j.bbadis.2018.08.037)
[Liver Injury.](https://doi.org/10.1016/j.bbadis.2018.08.037) _Biochim. Biophys. Acta, Mol. Basis Dis._ 2019, _1865_ (5),
1031−1039.
(14) O’Brien, P. J.; Irwin, W.; Diaz, D.; Howard-Cofield, E.; Krejsa,
C. M.; Slaughter, M. R.; Gao, B.; Kaludercic, N.; Angeline, A.;
[Bernardi, P.; Brain, P.; Hougham, C. High Concordance of Drug-](https://doi.org/10.1007/s00204-006-0091-3)
[Induced Human Hepatotoxicity with in Vitro Cytotoxicity Measured](https://doi.org/10.1007/s00204-006-0091-3)
[in a Novel Cell-Based Model Using High Content Screening.](https://doi.org/10.1007/s00204-006-0091-3) _Arch._
_Toxicol._ 2006, _80_ (9), 580−604.
(15) Olson, H.; Betton, G.; Robinson, D.; Thomas, K.; Monro, A.;
Kolaja, G.; Lilly, P.; Sanders, J.; Sipes, G.; Bracken, W.; Dorato, M.;
[Van Deun, K.; Smith, P.; Berger, B.; Heller, A. Concordance of the](https://doi.org/10.1006/rtph.2000.1399)
[Toxicity of Pharmaceuticals in Humans and in Animals.](https://doi.org/10.1006/rtph.2000.1399) _Regul. Toxicol._
_Pharmacol._ 2000, _32_ (1), 56−67.
[(16) Kotsampasakou, E.; Montanari, F.; Ecker, G. F. Predicting](https://doi.org/10.1016/j.tox.2017.06.003)
[Drug-Induced Liver Injury: The Importance of Data Curation.](https://doi.org/10.1016/j.tox.2017.06.003)
_Toxicology_ 2017, _389_, 139−145.
(17) Chen, M.; Suzuki, A.; Thakkar, S.; Yu, K.; Hu, C.; Tong, W.
[DILIrank: The Largest Reference Drug List Ranked by the Risk for](https://doi.org/10.1016/j.drudis.2016.02.015)
[Developing Drug-Induced Liver Injury in Humans.](https://doi.org/10.1016/j.drudis.2016.02.015) _Drug Discovery_
_Today_ 2016, _21_ (4), 648−653.
(18) He, S.; Ye, T.; Wang, R.; Zhang, C.; Zhang, X.; Sun, G.; Sun, X.
[An In Silico Model for Predicting Drug-Induced Hepatotoxicity.](https://doi.org/10.3390/ijms20081897) _Int. J._
_Mol. Sci._ 2019, _20_ (8), 1897.
[(19) Wang, Y.; Xiao, Q.; Chen, P.; Wang, B. In Silico Prediction of](https://doi.org/10.3390/ijms20174106)
[Drug-Induced Liver Injury Based on Ensemble Classifier Method.](https://doi.org/10.3390/ijms20174106) _Int._
_J. Mol. Sci._ 2019, _20_ (17), 4106.
(20) Mora, J. R.; Marrero-Ponce, Y.; García-Jacas, C. R.; Suarez
[Causado, A. Ensemble Models Based on QuBiLS-MAS Features and](https://doi.org/10.1021/acs.chemrestox.0c00030?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Shallow Learning for the Prediction of Drug-Induced Liver Toxicity:](https://doi.org/10.1021/acs.chemrestox.0c00030?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Improving Deep Learning and Traditional Approaches.](https://doi.org/10.1021/acs.chemrestox.0c00030?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _Chem. Res._
_Toxicol._ 2020, _33_ (7), 1855−1873.
(21) Ancuceanu, R.; Hovanet, M. V.; Anghel, A. I.; Furtunescu, F.;
[Neagu, M.; Constantin, C.; Dinu, M. Computational Models Using](https://doi.org/10.3390/ijms21062114)
[Multiple Machine Learning Algorithms for Predicting Drug](https://doi.org/10.3390/ijms21062114)
[Hepatotoxicity with the DILIrank Dataset.](https://doi.org/10.3390/ijms21062114) _Int. J. Mol. Sci._ 2020, _21_
(6), 2114.
(22) Liu, A.; Walter, M.; Wright, P.; Bartosik, A.; Dolciami, D.;

̈ [Elbasir, A.; Yang, H.; Bender, A. Prediction and Mechanistic Analysis](https://doi.org/10.1186/s13062-020-00285-0)

[of Drug-Induced Liver Injury (DILI) Based on Chemical Structure.](https://doi.org/10.1186/s13062-020-00285-0)
_Biol. Direct_ 2021, _16_ (1), 6.
[(23) Jaganathan, K.; Tayara, H.; Chong, K. T. Prediction of Drug-](https://doi.org/10.3390/ijms22158073)
[Induced Liver Toxicity Using SVM and Optimal Descriptor Sets.](https://doi.org/10.3390/ijms22158073) _Int._
_J. Mol. Sci._ 2021, _22_ (15), 8073.
[(24) Kang, M.-G.; Kang, N. S. Predictive Model for Drug-Induced](https://doi.org/10.3390/molecules26247548)
[Liver Injury Using Deep Neural Networks Based on Substructure](https://doi.org/10.3390/molecules26247548)
[Space.](https://doi.org/10.3390/molecules26247548) _Molecules_ 2021, _26_ (24), 7548.
(25) Thakkar, S.; Li, T.; Liu, Z.; Wu, L.; Roberts, R.; Tong, W.
[Drug-Induced Liver Injury Severity and Toxicity (DILIst): Binary](https://doi.org/10.1016/j.drudis.2019.09.022)


**1727** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article


[̈]



[Classification of 1279 Drugs by Human Hepatotoxicity.](https://doi.org/10.1016/j.drudis.2019.09.022) _Drug_
_Discovery Today_ 2020, _25_ (1), 201−208.
[(26) Björnsson, E. S.; Hoofnagle, J. H. Categorization of Drugs](https://doi.org/10.1002/hep.28323)
[Implicated in Causing Liver Injury: Critical Assessment Based on](https://doi.org/10.1002/hep.28323)
[Published Case Reports.](https://doi.org/10.1002/hep.28323) _Hepatology_ 2016, _63_ (2), 590−603.
[(27) Chen, M.; Vijay, V.; Shi, Q.; Liu, Z.; Fang, H.; Tong, W. FDA-](https://doi.org/10.1016/j.drudis.2011.05.007)
[Approved Drug Labeling for the Study of Drug-Induced Liver Injury.](https://doi.org/10.1016/j.drudis.2011.05.007)
_Drug Discovery Today_ 2011, _16_ (15−16), 697−703.
[(28) Kuhn, M.; Letunic, I.; Jensen, L. J.; Bork, P. The SIDER](https://doi.org/10.1093/nar/gkv1075)
[Database of Drugs and Side Effects.](https://doi.org/10.1093/nar/gkv1075) _Nucleic Acids Res._ 2016, _44_ (D1),
D1075−D1079.
(29) Greene, N.; Fisk, L.; Naven, R. T.; Note, R. R.; Patel, M. L.;
[Pelletier, D. J. Developing Structure-Activity Relationships for the](https://doi.org/10.1021/tx1000865?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Prediction of Hepatotoxicity.](https://doi.org/10.1021/tx1000865?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _Chem. Res. Toxicol._ 2010, _23_ (7), 1215−
1222.
(30) Suzuki, A.; Andrade, R. J.; Bjornsson, E.; Lucena, M. I.; Lee, W.
[M.; Yuen, N. A.; Hunt, C. M.; Freston, J. W. Drugs Associated with](https://doi.org/10.2165/11535340-000000000-00000)
[Hepatotoxicity and Their Reporting Frequency of Liver Adverse](https://doi.org/10.2165/11535340-000000000-00000)
[Events in VigiBase: Unified List Based on International Collaborative](https://doi.org/10.2165/11535340-000000000-00000)
[Work.](https://doi.org/10.2165/11535340-000000000-00000) _Drug Saf._ 2010, _33_ (6), 503−522.
[(31) Zhu, X.; Kruhlak, N. L. Construction and Analysis of a Human](https://doi.org/10.1016/j.tox.2014.03.009)
[Hepatotoxicity Database Suitable for QSAR Modeling Using Post-](https://doi.org/10.1016/j.tox.2014.03.009)
[Market Safety Data.](https://doi.org/10.1016/j.tox.2014.03.009) _Toxicology_ 2014, _321_, 62−72.
[(32) Xu, Y.; Dai, Z.; Chen, F.; Gao, S.; Pei, J.; Lai, L. Deep Learning](https://doi.org/10.1021/acs.jcim.5b00238?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[for Drug-Induced Liver Injury.](https://doi.org/10.1021/acs.jcim.5b00238?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _J. Chem. Inf. Model._ 2015, _55_ (10),
2085−2093.
(33) Mulliner, D.; Schmidt, F.; Stolte, M.; Spirkl, H.-P.; Czich, A.;
[Amberg, A. Computational Models for Human and Animal](https://doi.org/10.1021/acs.chemrestox.5b00465?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Hepatotoxicity with a Global Application Scope.](https://doi.org/10.1021/acs.chemrestox.5b00465?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _Chem. Res. Toxicol._
2016, _29_ (5), 757−767.
[(34) Tharwat, A.; Moemen, Y. S.; Hassanien, A. E. Classification of](https://doi.org/10.1016/j.jbi.2017.03.002)
[Toxicity Effects of Biotransformed Hepatic Drugs Using Whale](https://doi.org/10.1016/j.jbi.2017.03.002)
[Optimized Support Vector Machines.](https://doi.org/10.1016/j.jbi.2017.03.002) _J. Biomed. Inf._ 2017, _68_, 132−
149.
(35) Nguyen-Vo, T.-H.; Nguyen, L.; Do, N.; Le, P. H.; Nguyen, T.[N.; Nguyen, B. P.; Le, L. Predicting Drug-Induced Liver Injury Using](https://doi.org/10.1021/acsomega.0c03866?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Convolutional Neural Network and Molecular Fingerprint-Embedded](https://doi.org/10.1021/acsomega.0c03866?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Features.](https://doi.org/10.1021/acsomega.0c03866?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _ACS Omega_ 2020, _5_ (39), 25432−25439.
[(36) Li, T.; Tong, W.; Roberts, R.; Liu, Z.; Thakkar, S. DeepDILI:](https://doi.org/10.1021/acs.chemrestox.0c00374?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Deep Learning-Powered Drug-Induced Liver Injury Prediction Using](https://doi.org/10.1021/acs.chemrestox.0c00374?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Model-Level Representation.](https://doi.org/10.1021/acs.chemrestox.0c00374?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _Chem. Res. Toxicol._ 2021, _34_ (2), 550−
565.
(37) Yan, B.; Ye, X.; Wang, J.; Han, J.; Wu, L.; He, S.; Liu, K.; Bo, X.
[An Algorithm Framework for Drug-Induced Liver Injury Prediction](https://doi.org/10.3390/molecules27103112)
[Based on Genetic Algorithm and Ensemble Learning.](https://doi.org/10.3390/molecules27103112) _Molecules_ 2022,
_27_ (10), 3112.
(38) Chen, Z.; Jiang, Y.; Zhang, X.; Zheng, R.; Qiu, R.; Sun, Y.;
[Zhao, C.; Shang, H. ResNet18DNN: Prediction Approach of Drug-](https://doi.org/10.1093/bib/bbab503)
[Induced Liver Injury by Deep Neural Network with ResNet18.](https://doi.org/10.1093/bib/bbab503)
_Briefings Bioinf._ 2022, _23_ (1), bbab503.
(39) Ai, H.; Chen, W.; Zhang, L.; Huang, L.; Yin, Z.; Hu, H.; Zhao,
[Q.; Zhao, J.; Liu, H. Predicting Drug-Induced Liver Injury Using](https://doi.org/10.1093/toxsci/kfy121)
[Ensemble Learning Methods and Molecular Fingerprints.](https://doi.org/10.1093/toxsci/kfy121) _Toxicol. Sci._
2018, _165_ (1), 100−107.
(40) Zhang, H.; Ding, L.; Zou, Y.; Hu, S.-Q.; Huang, H.-G.; Kong,
[W.-B.; Zhang, J. Predicting Drug-Induced Liver Injury in Human with](https://doi.org/10.1007/s10822-016-9972-6)
[Naive Bayes Classifier Approach.](https://doi.org/10.1007/s10822-016-9972-6) [̈] _J. Comput. Aided Mol. Des._ 2016, _30_
(10), 889−898.
[(41) Ekins, S.; Williams, A. J.; Xu, J. J. A Predictive Ligand-Based](https://doi.org/10.1124/dmd.110.035113)
[Bayesian Model for Human Drug-Induced Liver Injury.](https://doi.org/10.1124/dmd.110.035113) _Drug Metab._
_Dispos._ 2010, _38_ (12), 2302−2308.
(42) Williams, D. P.; Lazic, S. E.; Foster, A. J.; Semenova, E.;
[Morgan, P. Predicting Drug-Induced Liver Injury with Bayesian](https://doi.org/10.1021/acs.chemrestox.9b00264?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Machine Learning.](https://doi.org/10.1021/acs.chemrestox.9b00264?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _Chem. Res. Toxicol._ 2020, _33_ (1), 239−248.
(43) Gadaleta, D.; Manganelli, S.; Roncaglioni, A.; Toma, C.;
[Benfenati, E.; Mombelli, E. QSAR Modeling of ToxCast Assays](https://doi.org/10.1021/acs.jcim.8b00297?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Relevant to the Molecular Initiating Events of AOPs Leading to](https://doi.org/10.1021/acs.jcim.8b00297?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Hepatic Steatosis.](https://doi.org/10.1021/acs.jcim.8b00297?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _J. Chem. Inf. Model._ 2018, _58_ (8), 1501−1517.



(44) Ambure, P.; Halder, A. K.; González Díaz, H.; Cordeiro, M. N.
[D. S. QSAR-Co: An Open Source Software for Developing Robust](https://doi.org/10.1021/acs.jcim.9b00295?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Multitasking or Multitarget Classification-Based QSAR Models.](https://doi.org/10.1021/acs.jcim.9b00295?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _J._
_Chem. Inf. Model._ 2019, _59_ (6), 2538−2544.
[(45) Khan, K.; Roy, K.; Benfenati, E. Ecotoxicological QSAR](https://doi.org/10.1016/j.jhazmat.2019.02.019)
[Modeling of Endocrine Disruptor Chemicals.](https://doi.org/10.1016/j.jhazmat.2019.02.019) _J. Hazard. Mater._ 2019,
_369_, 707−718.
[(46) Chen, S.; Xue, D.; Chuai, G.; Yang, Q.; Liu, Q. FL-QSAR: A](https://doi.org/10.1093/bioinformatics/btaa1006)
[Federated Learning-Based QSAR Prototype for Collaborative Drug](https://doi.org/10.1093/bioinformatics/btaa1006)
[Discovery.](https://doi.org/10.1093/bioinformatics/btaa1006) _Bioinformatics_ 2021, _36_ (22−23), 5492−5498.
[(47) Zhang, C.; Cheng, F.; Li, W.; Liu, G.; Lee, P. W.; Tang, Y. In](https://doi.org/10.1002/minf.201500055)
[Silico Prediction of Drug Induced Liver Toxicity Using Substructure](https://doi.org/10.1002/minf.201500055)
[Pattern Recognition Method.](https://doi.org/10.1002/minf.201500055) _Mol. Inf._ 2016, _35_ (3−4), 136−144.
(48) Muratov, E. N.; Bajorath, J.; Sheridan, R. P.; Tetko, I. V.;
Filimonov, D.; Poroikov, V.; Oprea, T. I.; Baskin, I. I.; Varnek, A.;
Roitberg, A.; Isayev, O.; Curtalolo, S.; Fourches, D.; Cohen, Y.;
Aspuru-Guzik, A.; Winkler, D. A.; Agrafiotis, D.; Cherkasov, A.;
[Tropsha, A. QSAR without Borders.](https://doi.org/10.1039/d0cs00098a) _Chem. Soc. Rev._ 2020, _49_ (11),
3525−3564.
[(49) Johnson, S. R. The Trouble with QSAR (or How I Learned To](https://doi.org/10.1021/ci700332k?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Stop Worrying and Embrace Fallacy).](https://doi.org/10.1021/ci700332k?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _J. Chem. Inf. Model._ 2008, _48_
(1), 25−26.
(50) Jiang, D.; Wu, Z.; Hsieh, C.-Y.; Chen, G.; Liao, B.; Wang, Z.;
[Shen, C.; Cao, D.; Wu, J.; Hou, T. Could Graph Neural Networks](https://doi.org/10.1186/s13321-020-00479-8)
[Learn Better Molecular Representation for Drug Discovery? A](https://doi.org/10.1186/s13321-020-00479-8)
[Comparison Study of Descriptor-Based and Graph-Based Models.](https://doi.org/10.1186/s13321-020-00479-8) _J._
_Cheminf._ 2021, _13_ (1), 12.
[(51) Wang, Y.; Wang, J.; Cao, Z.; Barati Farimani, A. Molecular](https://doi.org/10.1038/s42256-022-00447-x)
[Contrastive Learning of Representations via Graph Neural Networks.](https://doi.org/10.1038/s42256-022-00447-x)
_Nat. Mach. Intell._ 2022, _4_ (3), 279−287.
(52) Zhou, J.; Cui, G.; Hu, S.; Zhang, Z.; Yang, C.; Liu, Z.; Wang, L.;
[Li, C.; Sun, M. Graph Neural Networks: A Review of Methods and](https://doi.org/10.1016/j.aiopen.2021.01.001)
[Applications.](https://doi.org/10.1016/j.aiopen.2021.01.001) _AI Open_ 2020, _1_, 57−81.
[(53) Withnall, M.; Lindelöf, E.; Engkvist, O.; Chen, H. Building](https://doi.org/10.1186/s13321-019-0407-y)
[Attention and Edge Message Passing Neural Networks for Bioactivity](https://doi.org/10.1186/s13321-019-0407-y)
[and Physical-Chemical Property Prediction.](https://doi.org/10.1186/s13321-019-0407-y) _J. Cheminf._ 2020, _12_ (1),
1.
[(54) Ma, H.; An, W.; Wang, Y.; Sun, H.; Huang, R.; Huang, J. Deep](https://doi.org/10.1021/acs.chemrestox.0c00322?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Graph Learning with Property Augmentation for Predicting Drug-](https://doi.org/10.1021/acs.chemrestox.0c00322?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Induced Liver Injury.](https://doi.org/10.1021/acs.chemrestox.0c00322?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _Chem. Res. Toxicol._ 2021, _34_ (2), 495−506.
[(55) Lim, S.; Kim, Y.; Gu, J.; Lee, S.; Shin, W.; Kim, S. Supervised](https://doi.org/10.1016/j.isci.2022.105677)
[Chemical Graph Mining Improves Drug-Induced Liver Injury](https://doi.org/10.1016/j.isci.2022.105677)
[Prediction.](https://doi.org/10.1016/j.isci.2022.105677) _iScience_ 2023, _26_ (1), 105677.
(56) Ma, Z.; Wang, J.; Loskill, P.; Huebsch, N.; Koo, S.; Svedlund, F.
L.; Marks, N. C.; Hua, E. W.; Grigoropoulos, C. P.; Conklin, B. R.;
[Healy, K. E. Self-Organizing Human Cardiac Microchambers](https://doi.org/10.1038/ncomms8413)
[Mediated by Geometric Confinement.](https://doi.org/10.1038/ncomms8413) _Nat. Commun._ 2015, _6_ (1),
7413.
[(57) Atz, K.; Grisoni, F.; Schneider, G. Geometric Deep Learning on](https://doi.org/10.1038/s42256-021-00418-8)
[Molecular Representations.](https://doi.org/10.1038/s42256-021-00418-8) _Nat. Mach. Intell._ 2021, _3_ (12), 1023−
1032.
(58) Fang, X.; Liu, L.; Lei, J.; He, D.; Zhang, S.; Zhou, J.; Wang, F.;
[Wu, H.; Wang, H. Geometry-Enhanced Molecular Representation](https://doi.org/10.1038/s42256-021-00438-4)
[Learning for Property Prediction.](https://doi.org/10.1038/s42256-021-00438-4) _Nat. Mach. Intell._ 2022, _4_ (2), 127−
134.
(59) Hewitt, M.; Enoch, S. J.; Madden, J. C.; Przybylak, K. R.;
[Cronin, M. T. D. Hepatotoxicity: A Scheme for Generating Chemical](https://doi.org/10.3109/10408444.2013.811215)

[̈]

[Categories for Read-across, Structural Alerts and Insights into](https://doi.org/10.3109/10408444.2013.811215)
[Mechanism(s) of Action.](https://doi.org/10.3109/10408444.2013.811215) _Crit. Rev. Toxicol._ 2013, _43_ (7), 537−558.
[(60) Ahlberg, E.; Carlsson, L.; Boyer, S. Computational Derivation](https://doi.org/10.1021/ci500314a?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[of Structural Alerts from Large Toxicology Data Sets.](https://doi.org/10.1021/ci500314a?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _J. Chem. Inf._
_Model._ 2014, _54_ (10), 2945−2952.
[(61) Liu, R.; Yu, X.; Wallqvist, A. Data-Driven Identification of](https://doi.org/10.1186/s13321-015-0053-y)
[Structural Alerts for Mitigating the Risk of Drug-Induced Human](https://doi.org/10.1186/s13321-015-0053-y)
[Liver Injuries.](https://doi.org/10.1186/s13321-015-0053-y) _J. Cheminf._ 2015, _7_, 4.
[(62) Yang, H.; Lou, C.; Li, W.; Liu, G.; Tang, Y. Computational](https://doi.org/10.1021/acs.chemrestox.0c00006?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Approaches to Identify Structural Alerts and Their Applications in](https://doi.org/10.1021/acs.chemrestox.0c00006?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Environmental Toxicology and Drug Discovery.](https://doi.org/10.1021/acs.chemrestox.0c00006?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _Chem. Res. Toxicol._
2020, _33_ (6), 1312−1322.


**1728** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article


̈ ̀

̈



(63) Jia, X.; Wen, X.; Russo, D. P.; Aleksunes, L. M.; Zhu, H.
[Mechanism-Driven Modeling of Chemical Hepatotoxicity Using](https://doi.org/10.1016/j.jhazmat.2022.129193)
[Structural Alerts and an in Vitro Screening Assay.](https://doi.org/10.1016/j.jhazmat.2022.129193) _J. Hazard. Mater._
2022, _436_, 129193.
[(64) Jiménez-Luna, J.; Grisoni, F.; Schneider, G. Drug Discovery](https://doi.org/10.1038/s42256-020-00236-4)
[with Explainable Artificial Intelligence.](https://doi.org/10.1038/s42256-020-00236-4) _Nat. Mach. Intell._ 2020, _2_ (10),
573−584.
[(65) Rodríguez-Pérez, R.; Bajorath, J. Explainable Machine Learning](https://doi.org/10.1021/acs.jmedchem.1c01789?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[for Property Predictions in Compound Optimization.](https://doi.org/10.1021/acs.jmedchem.1c01789?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _J. Med. Chem._
2021, _64_ (24), 17744−17752.
[MGraphDTA: Deep Multiscale Graph Neural Network for Explain-(66) Yang, Z.; Zhong, W.; Zhao, L.; Yu-Chian Chen, C.](https://doi.org/10.1039/d1sc05180f) ̈ ̀
[able Drug-Target Binding Affinity Prediction.](https://doi.org/10.1039/d1sc05180f) _Chem. Sci._ 2022, _13_ (3),
816−833.
(67) Lou, C.; Yang, H.; Wang, J.; Huang, M.; Li, W.; Liu, G.; Lee, P.
[W.; Tang, Y. IDL-PPBopt: A Strategy for Prediction and](https://doi.org/10.1021/acs.jcim.2c00297?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Optimization of Human Plasma Protein Binding of Compounds via](https://doi.org/10.1021/acs.jcim.2c00297?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[an Interpretable Deep Learning Method.](https://doi.org/10.1021/acs.jcim.2c00297?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _J. Chem. Inf. Model._ 2022, _62_
(11), 2788−2799.
(68) Sushko, I.; Salmina, E.; Potemkin, V. A.; Poda, G.; Tetko, I. V.
[ToxAlerts: A Web Server of Structural Alerts for Toxic Chemicals and](https://doi.org/10.1021/ci300245q?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Compounds with Potential Adverse Reactions.](https://doi.org/10.1021/ci300245q?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _J. Chem. Inf. Model._
2012, _52_ (8), 2310−2316.
(69) Ferrari, T.; Cattaneo, D.; Gini, G.; Golbamaki Bakhtyari, N.;
[Manganaro, A.; Benfenati, E. Automatic Knowledge Extraction from](https://doi.org/10.1080/1062936x.2013.773376)
[Chemical Structures: The Case of Mutagenicity Prediction.](https://doi.org/10.1080/1062936x.2013.773376) _SAR_
_QSAR Environ. Res._ 2013, _24_ (5), 365−383.
[(70) Murakami, H. The Power of the Modified Wilcoxon Rank-Sum](https://doi.org/10.1080/02331888.2014.913049)
[Test for the One-Sided Alternative.](https://doi.org/10.1080/02331888.2014.913049) _Statistics_ 2015, _49_ (4), 781−794.
(71) Kong, K.; Li, G.; Ding, M.; Wu, Z.; Zhu, C.; Ghanem, B.;
Taylor, G.; Goldstein, T. Robust Optimization as Data Augmentation
for Large-Scale Graphs. _2022 IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition (CVPR)_ ; IEEE: New Orleans, LA, USA,
2022; pp 60−69.
[(72) Lobo, J. M.; Jiménez-Valverde, A.; Real, R. AUC: A Misleading](https://doi.org/10.1111/j.1466-8238.2007.00358.x)
[Measure of the Performance of Predictive Distribution Models.](https://doi.org/10.1111/j.1466-8238.2007.00358.x) _Global_
_Ecol. Biogeogr._ 2008, _17_ (2), 145−151.
(73) Baldi, P.; Brunak, S.; Chauvin, Y.; Andersen, C. A. F.; Nielsen,
[H. Assessing the Accuracy of Prediction Algorithms for Classification:](https://doi.org/10.1093/bioinformatics/16.5.412)
[An Overview.](https://doi.org/10.1093/bioinformatics/16.5.412) _Bioinformatics_ 2000, _16_ (5), 412−424.
(74) Yacouby, R.; Axman, D. Probabilistic Extension of Precision,
Recall, and F1 Score for More Thorough Evaluation of Classification
Models. _Proceedings of the First Workshop on Evaluation and_
_Comparison of NLP Systems_ ; Association for Computational
Linguistics: Online, 2020; pp 79−91.
[(75) Lalkhen, A. G.; McCluskey, A. Clinical Tests: Sensitivity and](https://doi.org/10.1093/bjaceaccp/mkn041)
[Specificity.](https://doi.org/10.1093/bjaceaccp/mkn041) _Cont. Educ. Anaesth. Crit. Care Pain_ 2008, _8_ (6), 221−223.
[(76) Zhu, Q. On the Performance of Matthews Correlation](https://doi.org/10.1016/j.patrec.2020.03.030)
[Coefficient (MCC) for Imbalanced Dataset.](https://doi.org/10.1016/j.patrec.2020.03.030) _Pattern Recognit. Lett._ ̈
2020, _136_, 71−80.
[(77) Thiese, M. S.; Ronna, B.; Ott, U. P Value Interpretations and](https://doi.org/10.21037/jtd.2016.08.16)
[Considerations.](https://doi.org/10.21037/jtd.2016.08.16) _J. Thorac. Dis._ 2016, _8_ (9), E928−E931.
(78) Du, H.; Cai, Y.; Yang, H.; Zhang, H.; Xue, Y.; Liu, G.; Tang, Y.;
[Li, W. In Silico Prediction of Chemicals Binding to Aromatase with](https://doi.org/10.1021/acs.chemrestox.7b00037?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Machine Learning Methods.](https://doi.org/10.1021/acs.chemrestox.7b00037?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _Chem. Res. Toxicol._ 2017, _30_ (5), 1209−
1218.
[(79) Lee, C.; Lee, G. G. Information Gain and Divergence-Based](https://doi.org/10.1016/j.ipm.2004.08.006)
[Feature Selection for Machine Learning-Based Text Categorization.](https://doi.org/10.1016/j.ipm.2004.08.006)
_Inf. Process. Manag._ 2006, _42_ (1), 155−165.
[(80) Yang, H.; Sun, L.; Li, W.; Liu, G.; Tang, Y. Identification of](https://doi.org/10.1093/toxsci/kfy146)
[Nontoxic Substructures: A New Strategy to Avoid Potential Toxicity](https://doi.org/10.1093/toxsci/kfy146)
[Risk.](https://doi.org/10.1093/toxsci/kfy146) _Toxicol. Sci._ 2018, _165_ (2), 396−407.
(81) Arora, S.; Hu, W.; Kothari, P. K. An Analysis of the T-SNE
Algorithm for Data Visualization. 2018, arXiv:1803.01768.
[(82) Capecchi, A.; Probst, D.; Reymond, J.-L. One Molecular](https://doi.org/10.1186/s13321-020-00445-4)
[Fingerprint to Rule Them All: Drugs, Biomolecules, and the](https://doi.org/10.1186/s13321-020-00445-4)
[Metabolome.](https://doi.org/10.1186/s13321-020-00445-4) _J. Cheminf._ 2020, _12_ (1), 43.
[(83) Lipinski, C. A. Lead- and Drug-like Compounds: The Rule-of-](https://doi.org/10.1016/j.ddtec.2004.11.007)
[Five Revolution.](https://doi.org/10.1016/j.ddtec.2004.11.007) _Drug Discov. Today Technol._ 2004, _1_ (4), 337−341.



(84) Koutsoumpos, S.; Chronaki, M.; Tsonos, C.; Karakasidis, T.;
[Guazzelli, L.; Mezzetta, A.; Moutzouris, K. On the Application of the](https://doi.org/10.1016/j.rinma.2022.100350)
[Wildman-Crippen Model to Ionic Liquids.](https://doi.org/10.1016/j.rinma.2022.100350) _Results Mater._ 2022, _16_,
100350.
[(85) Prasanna, S.; Doerksen, R. Topological Polar Surface Area: A](https://doi.org/10.2174/092986709787002817)
[Useful Descriptor in 2D-QSAR.](https://doi.org/10.2174/092986709787002817) _Curr. Med. Chem._ 2009, _16_ (1), 21−
41.
(86) Li, M.; Zhou, J.; Hu, J.; Fan, W.; Zhang, Y.; Gu, Y.; Karypis, G.
[DGL-LifeSci: An Open-Source Toolkit for Deep Learning on Graphs](https://doi.org/10.1021/acsomega.1c04017?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[in Life Science.](https://doi.org/10.1021/acsomega.1c04017?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _ACS Omega_ 2021, _6_ (41), 27233−27238.
Gu(87) Stärk, H.; Beaini, D.; Corso, G.; Tossou, P.; Dallago, C.;̈nnemann, S.; Lio, P. 3D Infomax Improves GNNs for Molecular̀
Property Prediction. _Proceedings of the 39th International Conference on_
_Machine Learning_, 2022.
(88) Liu, S.; Wang, H.; Liu, W.; Lasenby, J.; Guo, H.; Tang, J. PreTraining Molecular Graph Representation with 3D Geometry. 2022,
arXiv:2110.07728.
(89) Xiong, Z.; Wang, D.; Liu, X.; Zhong, F.; Wan, X.; Li, X.; Li, Z.;
[Luo, X.; Chen, K.; Jiang, H.; Zheng, M. Pushing the Boundaries of](https://doi.org/10.1021/acs.jmedchem.9b00959?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Molecular Representation for Drug Discovery with the Graph](https://doi.org/10.1021/acs.jmedchem.9b00959?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Attention Mechanism.](https://doi.org/10.1021/acs.jmedchem.9b00959?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _J. Med. Chem._ 2020, _63_ (16), 8749−8760.
(90) Yang, K.; Swanson, K.; Jin, W.; Coley, C.; Eiden, P.; Gao, H.;
Guzman-Perez, A.; Hopper, T.; Kelley, B.; Mathea, M.; Palmer, A.;
[Settels, V.; Jaakkola, T.; Jensen, K.; Barzilay, R. Analyzing Learned](https://doi.org/10.1021/acs.jcim.9b00237?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Molecular Representations for Property Prediction.](https://doi.org/10.1021/acs.jcim.9b00237?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _J. Chem. Inf._
_Model._ 2019, _59_ (8), 3370−3388.
(91) Pedregosa, F.; Varoquaux, G.; Gramfort, A.; Michel, V.;
Thirion, B.; Grisel, O.; Blondel, M.; Prettenhofer, P.; Weiss, R.;
Dubourg, V.; Vanderplas, J.; Passos, A.; Cournapeau, D. Scikit-Learn:
Machine Learning in Python. _J. Mach. Learn. Res._ 2011, _12_, 2825−
2830.
[(92) Vidaurre, J.; Gedela, S.; Yarosz, S. Antiepileptic Drugs and Liver](https://doi.org/10.1016/j.pediatrneurol.2017.09.013)
[Disease.](https://doi.org/10.1016/j.pediatrneurol.2017.09.013) _Pediatr. Neurol._ 2017, _77_, 23−36.
[(93) Younis, H. S. The Role of Hepatocellular Oxidative Stress in](https://doi.org/10.1093/toxsci/kfg207)
[Kupffer Cell Activation during 1,2-Dichlorobenzene-Induced Hep-](https://doi.org/10.1093/toxsci/kfg207)
[atotoxicity.](https://doi.org/10.1093/toxsci/kfg207) _Toxicol. Sci._ 2003, _76_ (1), 201−211.
(94) Stepan, A. F.; Walker, D. P.; Bauman, J.; Price, D. A.; Baillie, T.
[A.; Kalgutkar, A. S.; Aleo, M. D. Structural Alert/Reactive Metabolite](https://doi.org/10.1021/tx200168d?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Concept as Applied in Medicinal Chemistry to Mitigate the Risk of](https://doi.org/10.1021/tx200168d?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Idiosyncratic Drug Toxicity: A Perspective Based on the Critical](https://doi.org/10.1021/tx200168d?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Examination of Trends in the Top 200 Drugs Marketed in the United](https://doi.org/10.1021/tx200168d?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[States.](https://doi.org/10.1021/tx200168d?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _Chem. Res. Toxicol._ 2011, _24_ (9), 1345−1410.
(95) Wang, Y.; Gao, H.; Na, X.-L.; Dong, S.-Y.; Dong, H.-W.; Yu, J.;
[Jia, L.; Wu, Y.-H. Aniline Induces Oxidative Stress and Apoptosis of](https://doi.org/10.3390/ijerph13121188)
[Primary Cultured Hepatocytes.](https://doi.org/10.3390/ijerph13121188) _Int. J. Environ. Res. Publ. Health_ 2016,
_13_ (12), 1188.
(96) Paech, F.; Messner, S.; Spickermann, J.; Wind, M.; SchmittHoffmann, A.-H.; Witschi, A. T.; Howell, B. A.; Church, R. J.;
Woodhead, J.; Engelhardt, M.; Krähenbühl, S.; Maurer, M.

                                                  [Mechanisms of Hepatotoxicity Associated with the Monocyclic](https://doi.org/10.1007/s00204-017-1994-x) _β_
[Lactam Antibiotic BAL30072.](https://doi.org/10.1007/s00204-017-1994-x) _Arch. Toxicol._ 2017, _91_ (11), 3647−
3662.
[(97) Scales, M. D. G.; Timbrell, J. A. Studies on Hydrazine](https://doi.org/10.1080/15287398209530308)
[Hepatotoxicity. 1. Pathological Findings.](https://doi.org/10.1080/15287398209530308) _J. Toxicol. Environ. Health_
1982, _10_ (6), 941−953.
[(98) Timbretl, J. A.; Scales, M. D. C.; Streeter, A. J. Studies on](https://doi.org/10.1080/15287398209530309)
[Hydrazine Hepatotoxicity. 2. Biochemical Findings.](https://doi.org/10.1080/15287398209530309) _J. Toxicol._
_Environ. Health_ 1982, _10_ (6), 955−968.
(99) Shen, W. X.; Zeng, X.; Zhu, F.; Wang, Y. l.; Qin, C.; Tan, Y.;
[Jiang, Y. Y.; Chen, Y. Z. Out-of-the-Box Deep Learning Prediction of](https://doi.org/10.1038/s42256-021-00301-6)
[Pharmaceutical Properties by Broadly Learned Knowledge-Based](https://doi.org/10.1038/s42256-021-00301-6)
[Molecular Representations.](https://doi.org/10.1038/s42256-021-00301-6) _Nat. Mach. Intell._ 2021, _3_ (4), 334−343.
(100) Zeng, X.; Xiang, H.; Yu, L.; Wang, J.; Li, K.; Nussinov, R.;
[Cheng, F. Accurate Prediction of Molecular Properties and Drug](https://doi.org/10.1038/s42256-022-00557-6)
[Targets Using a Self-Supervised Image Representation Learning](https://doi.org/10.1038/s42256-022-00557-6)
[Framework.](https://doi.org/10.1038/s42256-022-00557-6) _Nat. Mach. Intell._ 2022, _4_ (11), 1004−1016.
[(101) Erve, J. C. L. Chemical Toxicology: Reactive Intermediates](https://doi.org/10.1517/17425255.2.6.923)
[and Their Role in Pharmacology and Toxicology.](https://doi.org/10.1517/17425255.2.6.923) _Expert Opin. Drug_
_Metab. Toxicol._ 2006, _2_ (6), 923−946.


**1729** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730


**Chemical Research in Toxicology** **pubs.acs.org/crt** Article


(102) Zhao, L.; Russo, D. P.; Wang, W.; Aleksunes, L. M.; Zhu, H.
[Mechanism-Driven Read-Across of Chemical Hepatotoxicants Based](https://doi.org/10.1093/toxsci/kfaa005)
[on Chemical Structures and Biological Data.](https://doi.org/10.1093/toxsci/kfaa005) _Toxicol. Sci._ 2020, _174_
(2), 178−188.
(103) Low, Y.; Sedykh, A.; Fourches, D.; Golbraikh, A.; Whelan, M.;
[Rusyn, I.; Tropsha, A. Integrative Chemical-Biological Read-Across](https://doi.org/10.1021/tx400110f?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
[Approach for Chemical Hazard Classification.](https://doi.org/10.1021/tx400110f?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as) _Chem. Res. Toxicol._
2013, _26_ (8), 1199−1208.


**1730** [https://doi.org/10.1021/acs.chemrestox.3c00199](https://doi.org/10.1021/acs.chemrestox.3c00199?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as)
_Chem. Res. Toxicol._ 2023, 36, 1717−1730


