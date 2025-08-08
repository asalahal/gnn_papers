MBE, 20(1): 269–282.

DOI: 10.3934/mbe.2023012

Received: 01 August 2022

Revised: 06 September 2022

Accepted: 07 September 2022

Published: 30 September 2022

http://www.aimspress.com/journal/MBE


_**Research article**_

# **Drug-target binding affinity prediction method based on a deep graph** **neural network**


**Dong Ma** **[1,]** - **, Shuang Li** **[2,]** - **and Zhihua Chen** **[1,]** *****


**1** Institute of Computing Science and Technology, Guangzhou University, Guangzhou, China
**2** Beidahuang Industry Group General Hospital, Harbin, China


- The authors contributed equally to this work.


*** Correspondence:** Email: czhgd@gzhu.edu.cn.


**Abstract:** The development of new drugs is a long and costly process, Computer-aided drug design
reduces development costs while computationally shortening the new drug development cycle, in
which DTA (Drug-Target binding Affinity) prediction is a key step to screen out potential drugs. With
the development of deep learning, various types of deep learning models have achieved notable
performance in a wide range of fields. Most current related studies focus on extracting the sequence
features of molecules while ignoring the valuable structural information; they employ sequence data
that represent only the elemental composition of molecules without considering the molecular structure
maps that contain structural information. In this paper, we use graph neural networks to predict DTA
based on corresponding graph data of drugs and proteins, and we achieve competitive performance on
two benchmark datasets, Davis and KIBA. In particular, an MSE of 0.227 and CI of 0.895 were
obtained on Davis, and an MSE of 0.127 and CI of 0.903 were obtained on KIBA.


**Keywords:** artificial intelligence; biological sequence analysis; deep learning; binding affinity
prediction; graph convolution network


**1. Introduction**


With the rapid development of machine learning in recent years, artificial intelligence has also
been applied to various fields, most of the time achieving valuable results [1–3]. Computer-aided drug


270


design is one of these areas and is of great interest. The high cost and time-consuming nature of drug
development makes the research of new drugs extremely difficult [4]. Drug-target affinity (DTA)
prediction is one of the important subtasks that helps to reduce the time-consuming pre-drug selection
phase of drug development [5,6].
There are three main types of computational approaches for DTA prediction, including molecular

docking [7], traditional machine learning [8–10], and deep learning [11–13]. Molecular docking is

based on protein structure to explore the main binding modes of ligands when binding to proteins [14],

but it requires the crystallized structure of proteins that are difficult to obtain, which also indirectly

affects its final performance [15]. Traditional machine learning, on the other hand, uses one
dimensional sequence representations in drug and protein sequences to train neural networks; however,

these models represent drugs as strings. Such representations can reflect the atomic composition of

molecules and the chemical bonds between atoms, but they cannot retain the structural information of

molecules [16]. The structural information of the molecule, in turn, affects its chemical properties,

which may impair the predictive power of the model as well as the functional relevance of the learned

potential space. Deep learning models that have been widely used in recent years also perform well in

the DTA prediction task, and learning using deep molecular modeling functions is gradually becoming

more common because it can capture hidden information that is difficult to model by human experience.
One of the models that may be most suitable for the DTA prediction task is the graph neural network
(GNN). The GNN can directly process graph data that can preserve structural information, and this
approach has already made research progress. GraphDTA [17] introduces GNN into the DTA
prediction task by constructing a drug molecule graph and performs feature extraction for drug
molecules based on the graph data, while protein molecules as organic macromolecules can still only
use CNN to extract features. DGraphDTA [18] builds graph data for protein molecules based on protein
structure prediction, which allows both molecules to be represented by a graph and enables the
application of GNN to extract features. However, it only compares two models, GCN [19] and GAT [20],
and cannot fully evaluate the performance of GNN in the DTA prediction task.


**2. Materials and methods**


_2.1. Datasets_


The data for training generally contains the dataset of drug and protein molecules and the values
of corresponding drug-target affinity. This research applies Davis [21] and KIBA (Kinase Inhibitor
BioActivity) datasets for specific experiments, and the two datasets are typically used as benchmarks.
The Davis dataset completely covered the binding affinity between all 68 drugs and 442 targets
included, which was measured by _K_ d values (kinase dissociation constants). The KIBA dataset
integrates information from IC 50, _K_ i (inhibition constant) and _K_ d (dissociation constant) measurements
into a single bioactivity score containing a bioactivity matrix of 2111 kinase inhibitors and 229
human kinases. The drug and protein molecule entities in the two datasets are shown in Table 1. The
PDBBind [22,23] dataset is a comprehensive database of drug-target 3D structure binding affinities in
the Protein Data Bank (PDB). The PDBBind dataset provides 3D coordinates of target proteins and
ligand structures. We use the general set as the training set and the refined set of the PDBBind dataset
as the test set. In order to have a stable training process, we only select samples from the general set


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


271


with a protein sequence length less than 1000 amino acids. The general set is divided into a training
set and a validation set with a ratio of 4:1. Finally, we have 8391 training samples, 1680 validation
samples and 3940 test samples. To reduce random errors, we trained and tested the model for
performance evaluation using a 5-fold training set and test set of the benchmark dataset, while
introducing various methods and metrics for comparison.


**Table 1.** The Davis and KIBA datasets.


Dataset Drugs Proteins Binding entities

Davis 68 442 30,056

KIBA 2111 229 118,254


The drug molecules are represented in the sequence format called SMILES (Simplified Molecular
Input Line Entry System) [24], and this form is a chemical notation language that uses the element
symbols of atoms and chemical bonds between them to represent molecules. However, because the
sequence data only retain the structural information of molecules, the performance may be worse if
the original SMILES string is directly applied. In addition, graph data can store more 3D features by
comparison. Therefore, in this method, the molecular graph is constructed by sequence to meet the
requirement for the model. Specifically, the molecular graph is constructed according to the drug
SMILES string, with atoms as nodes and bonds as edges. In order to ensure that the features of the
nodes are fully considered in the graph convolution process, self-loops are also added to the graph
construction to improve the feature performance of the drug molecules. Protein molecules are also
processed into graph data similarly. The raw data in the datasets are also sequence strings. However,
the above idea that processing drugs does not work while proteins are biological macromolecules, the
protein graphs will be too large to satisfy model conditions if atoms are constructed as nodes and
chemical bonds are designed as edges. Owing to the development of protein structure prediction, it is
feasible to utilize predicted structural information to approximate the real 3D structure of proteins. In
this paper, the Pconsc4 [25] method was used to output contact map [26] is used to predict protein
structure. The method assumes the residue of protein as nodes, and edges are determined by the
Euclidean distance between the C β atoms (C α atoms for glycine) of residue pairs below a specified
threshold [26]. Because the graph is constructed with residuals as nodes, features are selected around
the residuals, which exhibit different properties due to the R group. These features include polarity,
chargedness, aromaticity, etc. And in this paper, we use PSSM [27] to represent the characteristics of
the residue nodes. The selected node features of drug and protein molecules and detailed data
preprocessing measures are invariable as those in DGraphDTA [18].


_2.2. Method_


Almost every drug development study focuses on how to deal with drugs and target molecules,
and they preprocess data by applying other algorithms into special forms to fit their approach [28].
However, the models in these studies are relatively simple. DeepDTA only equips three CNN layers to
extract molecular features, and DGraphDTA employs two GNN layers and 2 fully connected layers to
extract node representations. With the aim of representing a large amount of knowledge and obtaining
high accuracy, a model with high expressiveness requires a larger training set. To this extent, a model
with more parameters and higher complexity is advantageous. On the other hand, however, an overly


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


272


complex model may be difficult to train and may lead to unnecessary resource consumption [29]. So
there is a need to balance the model complexity so that the model has a high expressive power while
reducing unnecessary consumption.
CNNs and RNNs generally have advanced performance in handling Euclidean data, but the
structural information of molecular graphs cannot be expressed in Euclidean space. This leads to CNN
and RNN methods rarely achieving optimal results. Therefore, most researchers apply graph neural
networks to extract the features of graph data.
Let _G_ = ( _V_, _E_ ) denote a simple and connected undirected graph with _n_ nodes and _m_ edges. Let **A**
be the adjacency matrix, **D** be the diagonal degree matrix, and **I** N be an n-order identity matrix. Let 𝐀 [�]
= **A** + **I** N denote the adjacency matrix with self-loop added, and the corresponding diagonal degree
matrix is denoted by 𝑫 [�] . Subsequently, let 𝐗∈𝑅 [�∗�] denote the node feature matrix, where the _i_ th
row of the matrix represents the _d_ -dimensional feature vector of the _i_ th node.
The GNN is a network designed to work directly with graph data and exploit its structural
information, and there are now many variants available after years of development. The most wellknown one is the GCN, which is widely used in graph data problems. For the GCN, each layer will
implement a convolution operation through Eq (1):


𝐻 [���] �𝑓�𝐻 [�], 𝐴��𝜎�𝐷 [�] ~~[�]~~ � [�] 𝐴�𝐷� ~~[�]~~ � [�] 𝐻 [�] 𝑊 [���] �, (1)


where _H_ _[l]_ represents the _l_ th layer of node embedding with _H_ _[0]_ = **X**, σ () is a nonlinear activation function,
and W denotes a learnable parameter. Qimai Li et al. [30] prove that graph convolution is essentially
Laplacian smoothing [31]; however, repeated application of Laplacian smoothing may mix the features
of vertices from different clusters and make them indistinguishable [32]. In the case of symmetric
Laplacian smoothing, they will converge to be proportional to the square root of the vertex degree [33].
This is why a deeper GCN will lead to performance decrease.
However, we find that the models that apply GCN commonly set their number of GNN layers to 2
or 3, and consequently these models will have no ability to extract high-order neighborhood
information. This is attributed to the notion that stacking more layers tends to degrade the performance
of these models, and such a phenomenon is called “oversmoothing” [30]. In other words, this situation
refers to the fact that during the training process of graph neural networks, as the number of network
layers and iterations increase, the hidden layer representations of each node within the same connected
component will tend to converge to the same value. This will result in the final node representation
having no actual meaning in the case of deeper layers. Even the addition of residual connectivity, an
effective technique widely used in deep CNNs, merely mitigates the oversmoothing problem in GCNs [19].
The purpose of the GNN is to extract the graph embedding of the entire molecular graph. However,
if only two or three layers of GCN are used, then for a certain node, just 2-hop or 3-hop neighbor node
information can be perceived, which is unfavorable for the information of the whole graph that needs
to be obtained to construct the graph embedding [34]. Thus, the GCN operations of several layers can
only be regarded as local information aggregation, and the obtained graph embedding is not
sufficiently accurate to represent the graph.
Graph Diffusion Convolution (GDC) [35] replaces the multilayer convolution operation in GCN
by using graph diffusion, and the graph diffusion process is given by the generalized diffusion matrix:


�
𝑆�� ��� 𝜃 � 𝑇 [�], (2)


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


273


with the weighting coefficients _θ_ _k_ and transition matrix _T_ . The selection of _θ_ _k_ and _T_ must ensure
that Eq (2) converges. One of the main options of weighting coefficients _θ_ _k_ follow Eq (3) of
Personalized PageRank [36]:


𝜃 ���� �𝛼 ��� �1 �𝛼 ��� � �, 𝛼∈�0,1�, (3)


where _α_ _[PPR]_ is the teleport probability, and the greater _α_ _[PPR ]_ is, the more information is diffused. _T_ is the
transfer matrix for a random walk in an undirected graph, and _T_ _rw_ = **AD** [-1] . Considering the information
about the node itself, GDC further adjusts the random walk by adding self-loops to the original
adjacency matrix and replaces transition matrix _T_ with _T_ _sym_ = (  loop **I** N + **D** ) [-1/2] (  loop **I** N + **A** ) (  loop **I** N
+ **D** ) [-1/2] where  loop  **R** [+] . Last, GDC takes over the normal adjacency matrix **A** with the sparse version
𝑆 [�] of the generalized graph diffusion matrix _S_ . The sparsification is done by selecting the k highest
mass entries per column in _S_ or by setting the entries below the threshold  to zero and then performing
the subsequent normal convolution operation.
Simple Graph Convolution (SGC) [37] believes that the complexity of GCNs inherited from
neural networks is burdensome and unnecessary for less demanding applications. Thus, SGC reduces
the additional complexity of GCNs by removing the nonlinearity between the layers of GCNs and
simplifying the resultant function to a single linear transformation. The experiments show that the final
obtained model is comparable to GCNs with higher computational efficiency and fewer fitted parameters.


**Figure 1.** The pipeline of this paper.


Simple spectral graph convolution (S [2] GC) [38] further analyzes the Markov Diffusion Kernel [39]
and improves the GCN to obtain the following iterative function:


� �
𝑌 [�] � softmax ~~�~~ � [�] ��� ��1 �𝛼�𝑇 [�] [�] 𝑋�𝛼𝑋�𝑊�, (4)


where 𝑇 [�] is the special case of _T_ _sym_ at  loop = 1, i.e., 𝑇 [�] �� 𝐼 [�] �D� [��/�] � 𝐼 [�] �A�� 𝐼 [�] �D� [��/�], the
𝛼 𝜖 �0,1� parameter balances the self-information of the node with the consecutive neighborhoods,
and the value of _K_ represents the receptive field size of the model. For example, each node can update
its own node feature representation based on the information of its farthest 4-hop neighbors if _K_ = 4.
The above iterative formula is designed by S [2] GC for the node classification task. In this task, GNN
aims to extract node features and subsequently add fully connected layers to fit channels, and the last


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


274


layer of the model uses softmax as a classifier. Therefore, when S [2] GC is applied to this paper, the
softmax layer needs to be removed, and the iterative function only retains the part within the softmax
function. Figure 1 illustrates the flow of the process used in this paper, which consists of four main
parts: the raw molecular sequence data, the processed drug molecule graph and the predicted protein
contact map, feature extraction using graph models, and finally DTA prediction.


**3. Results and discussion**


_3.1. Experiment setting_


In this paper, we alter the GCN layer of the model in DGraphDTA with other graph neural
networks and evaluate the impact of GNNs on the final results. However, limited by computational
resources, we first compare the parametric size of various networks and identify some algorithms with
lower complexity for subsequent experiments. The algorithms are built based on the PyTorch [40]
library, and Pytorch geometric (PyG) [41] provides a variety of portable access algorithms from which
to choose graph data tasks. The first step is to compare the graph models used parametrically due to
the large amount of data and to keep the models from becoming too complex. Assume the input

                            dimension and output dimension are _m_ and _n_, respectively, _k_ = _m_ _n_, and the results of the partial
comparison in the case of a single-layer network are shown in Table 2.


**Table 2.** Number of parameters of the single-layer model.


Model The numbers of parameters
GCN [19] _k_ + _n_
GAT [20] _k_ + 3 _n_
GraphConv [42] 2 _k_ + _n_
SAGE [43] 2 _k_ + _n_
SGC [37] _k_ + _n_


From Table 2, we can obtain these models with a small difference in complexity. There are other
models with complexity far beyond these, such as Molecular Fingerprints (MFConv) [44] with the
number of parameters up to 20 times _k_ . It is obvious that using this kind of algorithm will reduce the
training efficiency of the model. Similar to S [2] GC, SGC is not applied to the model to compare the
performance of various algorithms. As mentioned before, S [2] GC takes the adjacency matrix of the graph
to construct the transfer matrix when calculating 𝑇 [�], and processing the transfer matrix will consume
considerable time when applied to the DTA prediction task. Therefore, we extracted the features of
drug molecules using S [2] GC, while the SAGE algorithm was used for the feature extraction of protein
molecules in this experiment.
Training the model requires setting several parameters and tuning details within the model. One
of these details is to adjust the proportion of drug features and protein features in the final execution
of classification to adjust the influence of both molecules on the final prediction results. Since drug
molecules and protein molecules are encoded as 54- and 78-dimensional features, respectively, and
their feature dimension ratio is approximately 2:3, the feature dimension ratio of the two molecules
produced by the graph neural network can be set to 2:3. In addition, we set the value of _K_, which is the
number of iterations of the network, to 4 and the value of _α_ to 0.05 in the S [2] GC model.


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


275


Since S [2] GC requires processing adjacency and transfer matrices for graphs, DTA prediction is a
task that includes multiple graph data. Therefore, model training is time-consuming and cannot be
performed using a large number of different parameters, and the setting of various parameters needs
to be based on experience. A few important hyperparameters used in the model are as follows: learning
rate of 0.001, number of iterations of 2000, feature dimensions of 54 and 78 for drug and protein
molecules, respectively, and corresponding output dimensions of 112 (16*7) and 144 (16*9) for the
two types of molecules in the feature extraction phase, where 112:144 can be approximated as 54:78.


_3.2. Metrics_


The same metrics as in the benchmark are implemented to calculate the concordance index ( _CI_ ) [45]
and the mean squared error (MSE) [46]. The CI is essentially an estimate of the probability that the
predicted value will be consistent with the actual value; it measures whether the predicted affinity
values for two random drug-target pairs are predicted in the same order as the true values and is
calculated by Eq (5).


�
𝐶𝐼� � [�] � � �� � ℎ�𝑏 � �𝑏 � � (5)


where _b_ _x_ is the predicted value of the larger affinity _α_ _x_, by is the predicted value of the smaller affinity
_α_ _y_, Z is the normalization constant, and _h_ ( _x_ ) is the step function:


1, 𝑥�0
0.5, 𝑥�0 (6)
ℎ�𝑥��� 0, 𝑥�0


MSE measures the difference between the prediction and the label, and the formula is as follows:


� �
𝑀𝑆𝐸� � [�] ��� �𝑝 � �𝑦 � � [�], (7)


where _p_ _i_ is the predicted value, _y_ _i_ is the corresponding actual label, and a smaller MSE means that the
predicted value of the model is closer to the true value.
In addition, another metric, Pearson correlation coefficient, is also used in some articles for
performance comparison, calculated by Eq (8). where _cov_ is the covariance between the predicted
value _p_ and the real number _y_, and _σ_ () is the standard deviation.


�����,��
𝑃𝑒𝑎𝑟𝑠𝑜𝑛� (8)

��������


_3.3. Experimental results_


In this study, we introduced S2GC and SAGE to extract drug and protein features, respectively,
and we proceeded to use several layers of FC to make predictions. Figures 2 and 3 show the distribution
between the labels and predicted values of the samples in the two datasets. There is a straight line with
a slope of 1. When the sample points are closer to this line, the corresponding DTA predicted by the
model is closer to the true value, which indicates better model performance. The distribution of sample


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


276


points on these two plots also shows that our method is able to predict the DTA values more accurately,
with the vast majority of the predicted values and actual labels having a difference of 1 or less.


**Figure 2.** Labels and predicted values in Davis.


**Figure 3.** Labels and predicted values in KIBA.


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


277


We conducted several experiments to predict DTA and report the performance of certain models
on two metrics, MSE and CI. Tables 3 and 4 show the MSE and CI values on the independent test sets
of the two benchmark datasets, respectively. It can be observed in the Davis dataset that our model has
the best performance on both MSE and CI metrics. GraphDTA, another model based on graph neural
networks, also performs well, obtaining an MSE of 0.229, which is second only to the method in this
paper. The rest of the models extract features mainly based on sequential data. These models are
weaker than the graph model in terms of performance because they do not handle structural

information well.

For the KIBA dataset, our model performs slightly worse than Affinity2Vec, obtaining only the
second-best result. However, our model also obtained an MSE of 0.127 and a CI of 0.903, which are
not far from Affinity2Vec’s MSE of 0.124 and CI of 0.91. We argue that the main reason for this is the
type of dataset; although this dataset is large, it does not contain all binding values, and known DTA
only accounts for 24.4% of the affinity. Training on this dataset uses semi-supervised learning and is
more suitable for Affinity2Vec’s seq2seq [47] and ProtVet [48], which are two unsupervised datadriven models. On the other hand, this may be due to the more complex method in Affinity2Vec; in
terms of the model, it uses multilayer GRU [49] and ProtVec to extract sequence features and
XGBoost [50] for the final DTA prediction. In terms of features, it applies not only embedding features,
but also drug-target meta-path score features [51] and both hybrid features. This fits the characteristics
of the large data volume in the KIBA dataset, and there is enough data in KIBA to ensure a smooth fit
of the Affinity2Vec training process. In other words, the model and complexity of Affinity2Vec are
more suitable for training in a large dataset such as KIBA, which can also explain why the performance
of Affinity2Vec on the Davis dataset is not optimal, i.e., due to the insufficient amount of data in Davis.
As the PDBBind dataset, the method in this paper achieves 1.231, 0.756 and 0.683 on MSE, CI and
Pearson indicators, respectively, which also has some performance improvement relative to some
other methods. The detailed comparison is shown in Table 5.
As the benchmark model for this paper, DGraphDTA obtained better results on the Davis and
KIBA datasets, achieving 0.904, 0.202, and 0.867 for CI, MSE, and Pearson metrics on Davis, and
0.904, 0.126, 0.903 for CI, MSE, and Pearson metrics on KIBA, respectively. However, DGraphDTA
only uses three layers of GCNs for feature extraction, and it cannot continue to increase the number of
layers subsequently, and simply increasing GCNs can also lead to oversmoothing problems and may
also degrade the model performance. The S2GC algorithm used in this paper can be set up to k = 16,
which is the effect of 16 layers of GCN. However, due to the lack of computational resources, this
paper can only use k = 4 for experiments, and the difference is not obvious compared to the three-layer
GCN of DGraphDTA, which is why the performance of this paper’s method decreases instead of rising
compared to DGraphDTA.
In general, GNN-based models commonly outperform other models in terms of performance with
little difference in model complexity, and models that rely only on sequence information indeed
struggle to outperform graph models in terms of feature extraction. Although other methods can stack
models and features to obtain better results, graph models are certainly a more suitable class of
architectures for the DTA prediction task. In addition, the performance of the same model may vary

on different datasets. For the variation of molecular size and whether all DTA values are covered in

both Davis and KIBA datasets, different model architectures can be subsequently set according to
different datasets to meet the matching relationship between data volume and model complexity.


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


278


**Table 3.** Performance of multiple models on the Davis test set.


MSE CI Pearson

GraphDTA [17] 0.229 0.886 DeepFusionDTA [52] 0.253 0.887 0.820
WideDTA [53] 0.262 0.886 0.820
Affinity2Vec [54] 0.242 0.886 S [2] GC + SAGE 0.227 0.895 0.846


**Table 4.** Performance of multiple models on the KIBA test set.


MSE CI Pearson

GraphDTA 0.139 0.891 DeepFusionDTA 0.176 0.876 0.856

WideDTA 0.179 0.875 0.856

Affinity2Vec 0.124 0.91 S [2] GC + SAGE 0.127 0.903 0.887


**Table 5.** Performance of multiple models on the PDBBind test set.


MSE CI Pearson

GCNConvNet 1.308 0.711 0.597

GINConvNet 1.323 0.690 0.551

S [2] GC + SAGE 1.231 0.756 0.683


**4. Conclusions**


Predicting the strength of drug-target binding affinity is more informative and challenging than
just classifying drug-target interactions. The prediction of DTA requires molecular graph data based
on drugs and proteins. For such graph data, it is more appropriate to use graph neural networks instead
of other neural network models to extract features. To solve the oversmoothing problem of the most
widely used GCN when the model layers are deepened, we compared several networks used to alleviate
the oversmoothing, from which we selected an S2GC model that performs well in the node
classification task to extract molecular features. We also found that the graph neural network-based
model outperformed the other algorithms with a small difference in model complexity, so we can
continue to follow this idea in our subsequent work to explore the performance of graph neural
networks in the DTA prediction task.


**Acknowledgements**


This work was supported by the Chinese National Natural Science Foundation under Grant
61876047 and the Municipal Government of Quzhou (Grant Number 2020D003 and 2021D004).
Funding by Science and Technology Projects in Guangzhou (202201020120). We are grateful to Dr.
Ding and all partners for their help and support during the research process.


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


279


**Conflict of interest**


The authors declare there is no conflict of interest.


**References**


1. Y. Zhang, Artificial intelligence for bioinformatics and biomedicine, _Curr. Bioinf._, **15** (2020),
801–802. https://doi.org/10.2174/157489361508201221092330
2. B. Jena, S. Saxena, G. K. Nayak, L. Saba, N. Sharma, J. S. Suri, Artificial intelligence-based
hybrid deep learning models for image classification: The first narrative review, _Comput. Biol._
_Med._, **137** (2021), 104803. https://doi.org/10.1016/j.compbiomed.2021.104803
3. H. Lin, Development and application of artificial intelligence methods in biological and medical
data, _Curr. Bioinf._, **15** (2020), 515–516. https://doi.org/10.2174/157489361506200610112345
4. R. C. Andrade, M. Boroni, M. K. Amazonas, F. R. Vargas, New drug candidates for osteosarcoma:
Drug repurposing based on gene expression signature, _Comput. Biol. Med._, **134** (2021), 104470.
https://doi.org/10.1016/j.compbiomed.2021.104470
5. J. Wang, Y. Shi, X. Wang, H. Chang, A drug target interaction prediction based on LINE-RF
learning, _Curr._ _Bioinf._, **15** (2020), 750–757.
https://doi.org/10.2174/1574893615666191227092453
6. M. Aslam, M. Shehroz, F. Ali, A. Zia, S. Pervaiz, M. Shah, et al., Chlamydia trachomatis core
genome data mining for promising novel drug targets and chimeric vaccine candidates
identification, _Comput._ _Biol._ _Med._, **136** (2021), 104701.
https://doi.org/10.1016/j.compbiomed.2021.104701
7. J. Yan, J. Huang, C. Zhang, H. Huo, F. Chen, Virtual screening of acetylcholinesterase inhibitors
based on machine learning combined with molecule docking methods, _Curr. Bioinf._, **16** (2021),
963–971. https://doi.org/10.2174/1574893615999200719234045
8. F. F. Ahmed, M. Khatun, M. Mosharaf, M. N. Mollah, Prediction of protein-protein interactions
in Arabidopsis thaliana using partial training samples in a machine learning framework, _Curr._
_Bioinf._, **16** (2021), 865–879. https://doi.org/10.2174/1574893616666210204145254
9. D. P. Boso, D. D. Mascolo, R. Santagiuliana, P. Decuzzi, B. A. Schrefler, Drug delivery:
Experiments, mathematical modelling and machine learning, _Comput. Biol. Med._, **123** (2020),
103820. https://doi.org/10.1016/j.compbiomed.2020.103820
10. Y. Ding, J. Tang, F. Guo, Q. Zou, Identification of drug-target interactions via multiple kernelbased triple collaborative matrix factorization, _Briefings_ _Bioinf._, **23** (2022).
https://doi.org/10.1093/bib/bbab582
11. R. Su, X. Liu, L. Wei, Q. Zou, Deep-Resp-Forest: A deep forest model to predict anti-cancer drug
response, _Methods_, **166** (2019), 91–102. https://doi.org/10.1016/j.ymeth.2019.02.009
12. Q. Bai, S. Liu, Y. Tian, T. Xu, A. J. Banegas-Luna, H. Pérez-Sánchez, Application advances of
deep learning methods for de novo drug design and molecular dynamics simulation, _Wiley_
_Interdiscip. Rev.: Comput. Mol. Sci._, **12** (2022), e1581. https://doi.org/10.1002/wcms.1581
13. Q. Bai, S. Tan, T. Xu, H. Liu, J. Huang, X. Yao, MolAICal: A soft tool for 3D drug design of
protein targets by artificial intelligence and classical algorithm, _Briefings Bioinf._, **22** (2021).
https://doi.org/10.1093/bib/bbaa161


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


280


14. J. Li, A. Fu, L. Zhang, An overview of scoring functions used for protein-ligand interactions in
molecular docking, _Interdiscip._ _Sci.:_ _Comput._ _Life_ _Sci._, **11** (2019), 320–328.
https://doi.org/10.1007/s12539-019-00327-w
15. Y. Ding, J. Tang, F. Guo, Protein crystallization identification via fuzzy model on linear
neighborhood representation, _IEEE/ACM Trans. Comput. Biol. Bioinf._, **18** (2019), 1986–1995.
https://doi.org/10.1109/TCBB.2019.2954826
16. Y. Ding, J. Tang, F. Guo, Human protein subcellular localization identification via fuzzy model
on kernelized neighborhood representation, _Appl. Soft Comput._, **96** (2020), 106596.
https://doi.org/10.1016/j.asoc.2020.106596
17. T. Nguyen, H. Le, T. P. Quinn, T. Nguyen, T. D. Le, S. Venkatesh, GraphDTA: Predicting drugtarget binding affinity with graph neural networks, _Bioinformatics_, **37** (2021), 1140–1147.
https://doi.org/10.1093/bioinformatics/btaa921
18. M. Jiang, Z. Li, S. Zhang, S. Wang, X. Wang, Q. Yuan, et al., Drug-target affinity prediction using
graph neural network and contact maps, _RSC Adv._, **10** (2020), 20701–20712.
https://doi.org/10.1039/D0RA02297G
19. T. N. Kipf, M. Welling, Semi-supervised classification with graph convolutional networks,
preprint, arXiv:1609.02907.
20. P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Lio, Y. Bengio, Graph attention networks,
preprint, arXiv:1710.10903.
21. M. I. Davis, J. P. Hunt, S. Herrgard, P. Ciceri, L. M. Wodicka, G. Pallares, et al., Comprehensive
analysis of kinase inhibitor selectivity, _Nat. Biotechnol._, **29** (2011), 1046–1051.
https://doi.org/10.1038/nbt.1990
22. R. Wang, X. Fang, Y. Lu, S. Wang, The PDBbind database: Collection of binding affinities for
protein-ligand complexes with known three-dimensional structures, _J. Med. Chem._, **47** (2004),
2977–2980. https://doi.org/10.1021/jm030580l
23. R. Wang, X. Fang, Y. Lu, Y. C. Yang, S. Wang, The PDBbind database: Methodologies and updates,
_J. Med. Chem._, **48** (2005), 4111–4119. https://doi.org/10.1021/jm048957q
24. D. Weininger, SMILES, a chemical language and information system. 1. Introduction to
methodology and encoding rules, _J. Chem. Inf. Comput. Sci._, **28** (1988), 31–36.
https://doi.org/10.1021/ci00057a005
25. M. Michel, D. Menéndez Hurtado, A. Elofsson, PconsC4: Fast, accurate and hassle-free contact
predictions, _Bioinformatics_, **35** (2019), 2677–2679.
https://doi.org/10.1093/bioinformatics/bty1036
26. Q. Wu, Z. Peng, I. Anishchenko, Q. Cong, D. Baker, J. Yang, Protein contact prediction using
metagenome sequence data and residual neural networks, _Bioinformatics_, **36** (2020), 41–48.
https://doi.org/10.1093/bioinformatics/btz477
27. J. C. Jeong, X. Lin, X. W. Chen, On position-specific scoring matrix for protein function
prediction, _IEEE/ACM_ _Trans._ _Comput._ _Biol._ _Bioinf._, **8** (2010), 308–315.
https://doi.org/10.1109/TCBB.2010.93
28. Y. Ding, P. Tiwari, Q. Zou, F. Guo, H. M. Pandey, C-loss based higher-order fuzzy inference
systems for identifying DNA N4-methylcytosine sites, _IEEE Trans. Fuzzy Syst._, **2022** (2022).
https://doi.org/10.1109/TFUZZ.2022.3159103
29. X. Hu, L. Chu, J. Pei, W. Liu, J. Bian, Model complexity of deep learning: A survey, _Knowl. Inf._
_Syst._, **63** (2021), 2585–2619. https://doi.org/10.1007/s10115-021-01605-0


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


281


30. Q. Li, Z. Han, X. M. Wu, Deeper insights into graph convolutional networks for semi-supervised
learning, in _Thirty-Second AAAI conference on artificial intelligence_, AAAI, New Orleans, USA,
(2018), 3538–3545. https://doi.org/10.1609/aaai.v32i1.11604
31. G. Taubin, A signal processing approach to fair surface design, in _Proceedings of the 22nd annual_
_conference on Computer graphics and interactive techniques_, ACM, (1995), 351–358.
https://doi.org/10.1145/218380.218473
32. Y. Ding, W. He, J. Tang, Q. Zou, F. Guo, Laplacian regularized sparse representation based
classifier for identifying DNA N4-methylcytosine Sites via L2, 1/2-matrix Norm, _IEEE/ACM_
_Trans. Comput. Biol. Bioinf._, **2021** (2021). https://doi.org/10.1109/TCBB.2021.3133309
33. Y. Ding, J. Tang, F. Guo, Identification of drug-target interactions via dual laplacian regularized
least squares with multiple kernel fusion, _Knowledge-Based Syst._, **204** (2020), 106254.
https://doi.org/10.1016/j.knosys.2020.106254
34. P. Tiwari, S. Dehdashti, A. K. Obeid, P. Marttinen, P. Bruza, Kernel method based on non-linear
coherent states in quantum feature space, _J. Phys. A: Math. Theor._, **55** (2022), 355301.
https://doi.org/10.1088/1751-8121/ac818e
35. J. Klicpera, S. Weißenberger, S. Günnemann, Diffusion improves graph learning, preprint,

arXiv:1911.05485.

36. L. Page, S. Brin, R. Motwani, T. Winograd, The PageRank citation ranking: Bringing order to the
web, _Stanford InfoLab_ ., **1999** (1999).
37. F.Wu, A. Souza, T. Zhang,C. Fifty, T. Yu, K. Weinberger, Simplifying graph convolutional
networks, in _International conference on machine learning_, _PMLR_, **97** (2019), 6861–6871.
https://doi.org/10.48550/arXiv.902.07153
38. H. Zhu, P. Koniusz, Simple spectral graph convolution, in _International Conference on Learning_
_Representations_, (2020).
39. F. Fouss, K. Francoisse, L .Yen, A. Pirotte, M. Saerens, An experimental investigation of kernels
on graphs for collaborative recommendation and semisupervised classification, _Neural networks_,
**31** (2012), 53–72. https://doi.org/10.1016/j.neunet.2012.03.001
40. A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, Pytorch: An imperative style,
high-performance deep learning library, in _Advances in neural information processing systems_,
**32** (2019).
41. M. Fey, J. E. Lenssen, Fast graph representation learning with PyTorch Geometric, preprint,

arXiv:1903.02428.

42. C. Morris, M. Ritzert, M. Fey, W. L. Hamilton, J. E. Lenssen, G. Rattan, et al., Weisfeiler and
leman go neural: Higher-order graph neural networks, in _Proceedings of the AAAI conference on_
_artificial_ _intelligence_, AAAI, Honolulu, USA, **33** (2019), 4602–4609.
https://doi.org/10.1609/aaai.v33i01.33014602
43. W. Hamilton, Z. Ying, J. Leskovec, Inductive representation learning on large graphs, in _Advances_
_in neural information processing systems_, **30** (2017).
44. D. K. Duvenaud, D. Maclaurin, J. Iparraguirre, R. Bombarell, T. Hirzel, A. Aspuru-Guzik, et al.,
Convolutional networks on graphs for learning molecular fingerprints, in _Advances in neural_
_information processing systems_, **28** (2015). https://doi.org/10.48550/arXiv.1509.09292
45. M. Gönen, G. Heller, Concordance probability and discriminatory power in proportional hazards
regression, _Biometrika_, **92** (2005), 965–970. https://doi.org/10.1093/biomet/92.4.965


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


282


46. D. M. Allen, Mean square error of prediction as a criterion for selecting variables, _Technometrics_,
**13** (1971), 469–475. https://doi.org/10.1080/00401706.1971.10488811
47. Z. Xu, S. Wang, F. Zhu, J. Huang, Seq2seq fingerprint: An unsupervised deep molecular
embedding for drug discovery, in _Proceedings of the 8th ACM international conference on_
_bioinformatics, computational biology, and health informatics_, ACM, Boston, USA, (2017), 285–
294. https://doi.org/10.1145/3107411.3107424
48. E. Asgari, M. R. Mofrad Continuous distributed representation of biological sequences for deep
proteomics and genomics, _PloS_ _one_, **10** (2015), e0141287.
https://doi.org/10.1371/journal.pone.0141287
49. J. Chung, C. Gulcehre, K. Cho,. Bengio, Empirical evaluation of gated recurrent neural networks
on sequence modeling, preprint, arXiv:1412.3555.
50. T. Chen, C. Guestrin, Xgboost: A scalable tree boosting system, in _Proceedings of the 22nd acm_
_sigkdd international conference on knowledge discovery and data mining_, ACM, San Francisco,
USA, (2016), 785–794. https://doi.org/10.1145/2939672.2939785
51. G. Fu, Y. Ding, A. Seal, B. Chen, Y. Sun, E. Bolton, Predicting drug target interactions using metapath-based semantic network analysis, _BMC_ _Bioinf._, **17** (2016), 1–10.
https://doi.org/10.1186/s12859-016-1005-x
52. Y. Pu, J. Li, J. Tang, F. Guo, DeepFusionDTA: Drug-target binding affinity prediction with
information fusion and hybrid deep-learning ensemble model, _IEEE/ACM Trans. Comput. Biol._
_Bioinf.,_ **2021** (2021). https://doi.org/10.1109/TCBB.2021.3103966
53. H. Öztürk, E. Ozkirimli, A. Özgür, WideDTA: Prediction of drug-target binding affinity. preprint,

arXiv:1902.04166.

54. M. A. Thafar, M. Alshahrani, S. Albaradei, T. Gojobori, M. Essack, X. Gao, Affinity2Vec: Drugtarget binding affinity prediction through representation learning, graph mining, and machine
learning, _Sci. Rep._, **12** (2022), 1–18. https://doi.org/10.1038/s41598-022-08787-9


©2023 the Author(s), licensee AIMS Press. This is an open access

article distributed under the terms of the Creative Commons

Attribution License (http://creativecommons.org/licenses/by/4.0)


_Mathematical Biosciences and Engineering_ Volume 20, Issue 1, 269 – 282.


