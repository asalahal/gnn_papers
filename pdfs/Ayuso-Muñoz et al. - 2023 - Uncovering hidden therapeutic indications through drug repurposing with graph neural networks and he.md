[Artificial Intelligence In Medicine 145 (2023) 102687](https://doi.org/10.1016/j.artmed.2023.102687)


Contents lists available at ScienceDirect
# Artificial Intelligence In Medicine


[journal homepage: www.elsevier.com/locate/artmed](https://www.elsevier.com/locate/artmed)


Research paper

## Uncovering hidden therapeutic indications through drug repurposing with graph neural networks and heterogeneous data


Adrian Ayuso-Mu´ noz˜ [a] [,] [b], Lucía Prieto-Santamaría [a] [,] [b], Esther Ugarte-Carro [a] [,] [b], Emilio Serrano [a],
Alejandro Rodríguez-Gonzalez´ [a] [,] [b] [,] [* ]


a _ETS Ingenieros Informaticos, Universidad Polit_ ´ ´ _ecnica de Madrid, 28660 Boadilla del Monte, Madrid, Spain_
b _Centro de Tecnología Biom_ ´ _edica, Universidad Polit_ ´ _ecnica de Madrid, 28223 Pozuelo de Alarcon, Madrid, Spain_ ´



A R T I C L E I N F O


_Keywords:_
Drug repurposing
Drug repositioning
Graph deep learning (GDL)
Graph neural networks (GNN)
DISNET knowledge base


**1. Introduction**



A B S T R A C T


Drug repurposing has gained the attention of many in the recent years. The practice of repurposing existing drugs
for new therapeutic uses helps to simplify the drug discovery process, which in turn reduces the costs and risks
that are associated with _de novo_ development. Representing biomedical data in the form of a graph is a simple
and effective method to depict the underlying structure of the information. Using deep neural networks in
combination with this data represents a promising approach to address drug repurposing. This paper presents
BEHOR a more comprehensive version of the REDIRECTION model, which was previously presented. Both
versions utilize the DISNET biomedical graph as the primary source of information, providing the model with
extensive and intricate data to tackle the drug repurposing challenge. This new version’s results for the reported
metrics in the RepoDB test are 0.9604 for AUROC and 0.9518 for AUPRC. Additionally, a discussion is provided
regarding some of the novel predictions to demonstrate the reliability of the model. The authors believe that
BEHOR holds promise for generating drug repurposing hypotheses and could greatly benefit the field.



Drug repurposing or drug repositioning consists in providing novel
indications for already existing drugs, using them to treat different
diseases from the ones they were originally developed for. By doing so,
the drug discovery and development process can be made more efficient,
as it can reduce costs, time, and risks associated with drug discovery and
development process, what is known as _de novo_ approach. Before the
development of proper drug repurposing techniques, drug repositioning
often occurred serendipitously. However, nowadays, there are many
techniques to guide this process, like pathway mapping or genetic as­
sociation [1].
Observing the scarcity of emergent Deep Learning (DL) techniques
among the most common drug repurposing methodologies [1,2] is
interesting. Its application, beyond the benefits of drug repurposing it­
self, could reduce, even further, the cost and time. Drug repurposing
with DL has gained popularity in the last years [3–10]. With the
increasing trend of conceiving diseases as holistic and interrelated



entities [11,12], the use of Graph Deep Learning (GDL) has the potential
to make a greater impact due to the large amount of biomedical data
available through multi-omics techniques [13–15]. Using GDL it is
possible to work directly on graphs, biomedical networks, to gain
invaluable insights [16]. The prediction of links between disease and
drug pairs in these networks can uncover new indications not considered
yet [17].
Previously, following this approach, REDIRECTION (dRug rEpur­
posing Disnet lInk pREdiCTION) was presented [18]. REDIRECTION is a
model trained with part of the DISNET’s biomedical graph [1 ] [19], using a
limited set of information: diseases, symptoms, drugs and their re­
lationships. The objective of REDIRECTION was to generate drug
repurposing hypotheses by means of link prediction. In this work,
BEHOR (Bidirectional Edge and Hyperparameter Optimized Redirec­
tion), the model is tested in a richer information set, and contextualized
by comparing it against baselines. Some of the top novel drug repur­
posing hypothesis predictions are studied from a biomedical point of
view, to study the reliability of the model’s generated hypotheses.




 - Corresponding author at: ETS Ingenieros Inform´aticos, Universidad Polit´ecnica de Madrid, 28660 Boadilla del Monte, Madrid, Spain.
_E-mail addresses:_ [adrian.ayuso.munoz@alumnos.upm.es (A. Ayuso-Munoz), ˜](mailto:adrian.ayuso.munoz@alumnos.upm.es) [lucia.prieto.santamaria@upm.es (L. Prieto-Santamaría), e.ugarte@alumnos.upm.es](mailto:lucia.prieto.santamaria@upm.es)
[(E. Ugarte-Carro), emilio.serrano@upm.es (E. Serrano), alejandro.rg@upm.es (A. Rodríguez-Gonzalez).  ´](mailto:emilio.serrano@upm.es)

1
[https://disnet.ctb.upm.es/.](https://disnet.ctb.upm.es/)


[https://doi.org/10.1016/j.artmed.2023.102687](https://doi.org/10.1016/j.artmed.2023.102687)
Received 8 March 2023; Received in revised form 4 October 2023; Accepted 13 October 2023

Available online 21 October 2023
0933-3657/© 2023 Elsevier B.V. All rights reserved.


_A. Ayuso-Munoz et al._ ˜ _Artificial Intelligence In Medicine 145 (2023) 102687_



In addition to the improvements made to REDIRECTION, this work’s
contribution lies in the use of such a rich and complex set of information
for training the model, in terms of the types of relationships it contains.
Biological, pharmacological, and phenotypical information is inte­
grated. The model benefits from this rich graph, which provides it with a
vast amount of information to help solve the drug repurposing problem.
So, drug repurposing benefits from the advantages of using data-driven
approaches streamlining the process.
The paper is organised as follows: Section 2 includes a brief revision
of how GDL has been used in the context of biomedical data and its uses

for drug repurposing. Section 3 details the used datasets, BEHOR’s ar­
chitecture, the training and evaluating process, the used baselines for
comparison and the used metrics for evaluation. Section 4 presents the
model’s results, the evaluation of the selected novel predictions and the
limitations. Finally, Section 5 contains the conclusions.


**2. Related works**


Biomedical information is easily structured as a graph. For example,
the Human Disease Network [20], is a graph where the nodes represent
diseases and the links between those nodes represent the genetic con­
nections between diseases. Therefore, approaches that can take advan­
tage of the valuable information encoded in this data structure seem
promising [21–23]. A graph is a set of items in which there are some
pairwise relations, edges. Defined mathematically as it can be seen in Eq.
(1).


_G_ ( _V, E_ ) _,_ (1)


where _G_ is the graph, _V_ = { _v_ 1 _,_ … _, v_ _n_ } is the set of vertices, and _E_ ⊆
{( _v_ _i_ _, v_ _j_ )⃒⃒ _v_ _i_ _, v_ _j_ ∈ _V_ } is the set of edges.

Graphs can be divided into two types, directed and undirected. The
directed graphs are the ones in which edges have a direction, whereas
the undirected one’s edges do not have such condition. So, given an edge
_e_ _i_ = ( _v_ _i_ _, v_ _j_ ), in an undirected graph, this means there is an edge between

_v_ _i_ and _v_ _j_, while in a directed graph it means there is an edge from _v_ _i_ to _v_ _j_ .
Nodes and edges can be characterized by attributes themselves. For
edges, usually, if the value is one-dimensional this is called weight, if not
it is called edge attribute. In any case, the representation of an edge
would be _e_ _i_ = ( _v_ _i_ _, v_ _j_ _, w_ ), where _w_ is either the weight or the attribute.
Also, graphs can be divided into homogeneous graphs, graphs where
there is just one type of node and one type of edge, and heterogeneous
graphs, where there are either multiple types of nodes or multiple types
of edges. In the latter case, an edge would be _e_ _i_ = ( _v_ _i_ _, v_ _j_ _, r_ _k_ ), where _r_ _k_ ∈

_R_ = { _r_ 1 _,_ … _, r_ _l_ }, being _R_ the set of relation types.
Graphs are such a powerful data structure; their huge expressive
capability eases the task of representing diverse types of information.
Therefore, DL on graphs could help exploit the latent information
available in graphs. Initially, CNN would seem suitable to work with
graphs but the incapability of working with non-Euclidean data is a
major drawback for its application [24].
Hence, a way of handling graphs in the framework of DL was needed.
The precursors of GNN were developed in the late 1990s, applying
recursive neural networks to acyclic graphs [15,25,26]. Later, the field
of Geometric Deep Learning (GDL) [27] and graph representation
learning appeared. The latter’s objective is to embed graphs into low
dimensionality vectors, which can be accomplished using various al­
gorithms such as DeepWalk [28] and node2vec [29]. But, these algo­
rithms lack of generalization capabilities, they can only generate
embeddings for nodes present during the training phase (what is termed
transductive setting [15]), resulting in what is called shallow embed­
dings. Rather than training a model that can generate embeddings for
the nodes of a given graph, these algorithms train unique embeddings
for every node [30]. Therefore, similar nodes have similar embeddings,
and dissimilar nodes have dissimilar ones. Consequently, adding a node



to the graph requires training the model again and generating new
embeddings.
Thus, GNN and their variations emerged as alternatives with better
generalization capabilities [30]. Being able to generate embeddings for
unseen nodes, defined as inductive setting. The addition of a new node
does not require re-training the model. At the end of the training phase,
the result is a function that, given a graph, can generate embeddings for
any node in the graph. The use of GNNs for drug repurposing has been
successfully demonstrated in previous works [4–7], dating back to 2017

[8], and has yielded promising results. This technique gained wide­
spread attention during the pandemic years when numerous studies
focused on repurposing drugs to treat the SARS-CoV-2 virus [31–33].
Some of the first works related to drug repurposing using GDL is Bajaj
et al. [8], in which exploration of drug disease interactions is done using
a network centred approach. Two types of networks are used: i) PPI with
5489 usable proteins and ii) Disease-Protein interaction network with
534 diseases and their relations with the previous proteins.
Many of the previous works are focused on specific diseases, for
example, targeting SARS-CoV-2-virus. In Hsieh et al. [5], the con­
structed graph contains information related to COVID-19. Similarly, in
Zhou et al. [31] two types of human coronaviruses related networks are
used in different steps to generate repurposing hypotheses for this dis­
ease. The approach measures the interplay between the human coro­
navirus host interactome and drug targets in the PPI network. In Zeng
et al. [32] a massive graph of 15 million edges distributed along 39
types, for 145,179 nodes distributed in 4 node types: drug, gene, disease,
and drug side information is used to find drug repurposing candidates
for COVID-19. The presented method reported an AUROC of 0.8512.
In addition, centred in COVID-19, but with a generic and adaptable
to other diseases, approaches are presented in the following works. In
Gysi et al. [4] the used graph contains PPI information, 18,505 proteins
and 327,924 interactions between them, interactions between SARSCoV-2 human proteins and drug-target interactions. In Doshi and Che­
puri [6] the used graph is composed by 4 node types and 1,417,624
edges and, for the comparable reported metrics, it presents an AUROC of
0.9240. In the work the authors conduct an analysis of some drug
repurposing hypotheses for COVID-19, though it proposes a general
method. Moreover, centred in COVID-19 too, in Ioannidis et al. [33] a
new method using relational graph convolutional network to solve the
link-prediction problem, using this approach drug repurposing hypoth­
eses generation is tackled. Experiments are conducted on the drug
repurposing knowledge graph (DRKG) [34], generating repurposing
hypotheses for COVID-19. This graph contains 97,238 nodes distributed
into 13 types and 5,874,261 edges belonging to 107 edge types.
Changing the focal point, in Sosa et al. [7] the focus is set specifically
on rare diseases. The study utilised the Global Network of Biomedical
Relationships (GNBR), containing 63,252 nodes and 583,685 edges. It is
composed by three node types, drug, disease, and gene. The graph is
used to generate drug repurposing hypotheses for rare diseases. It ach­
ieves an AUROC of 0.8900.

When it comes to recent works on drug repurposing using GNNs that
do not target specific diseases, we find the works of Mei et al. [35] and
Gu et al. [36] particularly relevant. In Mei et al. [35] a small graph
extracted from the Comparative Toxicogenomics Database (CTD) [37] is
used, while applying a two-phase GNN that is composed of a subtypelevel network and node-level network encoding modules. In Gu et al.

[36], they integrate information from various sources to construct the
graph and implement three attention mechanisms into GNNs to address
the problem.
Besides drug repurposing, an important work in the field of GDL, the
first combination therapy GDL work, is Decagon [38]. Decagon models
polypharmacy side-effects using a multi-modal graph containing side
effects of drug combinations and protein-protein-interactions (PPI).
We summarise all the cited works in Table 1, where a comparison of
the used graph can easily be done. We also include our graph on the last
row for comparison purposes.



2


_A. Ayuso-Munoz et al._ ˜ _Artificial Intelligence In Medicine 145 (2023) 102687_



**Table 1**

Summary of the Referenced Work’s Graphs. Ordered by the nature of the used
graph.



Author Year Number of

nodes

(Number of

types)



Number of

edges
(Number of

types)



Notes



Zhou et al. [31] 2020 17,706 (1) 351,444 (1) PPI
Gysi et al. [4] 2021 18,505 (1) 327,924 (1) PPI
Hsieh et al. [5] 2021 6730 (4) _>_ 5358 (5) COVID
Zeng et al. [32] 2020 145,179 (5) 15,018,067 COVID
(39)


Doshi and 2022 42,484 (4) 1,417,624 (8) Derived from
Chepuri [6] DRKG
Ioannidis et al. 2020 97,238 (13) 5,874,261 DRKG [34]

[33] (107)


Sosa et al. [7] 2020 63,252 (3) 583,685 (32) GNBR [39]
Mei et al. [35] 2022 7794 (3) 36,778 (4) CTD [37]
Gu et al. [36] 2022 41,100 (5) 1,008,258 (10) Multi-source

˜
Ayuso-Munoz 2023 153,747 (5) 1,996,658 (12) –

et al.


**3. Materials and methods**


This section contains the details of the taken approach to carry out
the drug-disease repurposing task, the used heterogeneous graphs to
train our model, and the GNN related aspects, including the model’s
architecture, the training, testing, and validating processes, and the
RepoDB test.
We have uploaded the materials and results in an accessible re­
pository, [2 ] so the results can be reproduced.


_3.1. Drug-disease link prediction_


The drug repurposing problem is formulated as a link prediction task
in a heterogeneous graph of diseases and their interactions with various
entities, like drugs or proteins. Link prediction is performed between
disease and drug node types.
The results of the model can be interpreted as the confidence of the
model in the existence of a given link, being zero no confidence and one
absolute confidence. Then the problem is solved as a binary classifica­
tion problem, where the classes are non-existent (0) and existent (1).


_3.2. Dataset_


The employed data to build the heterogeneous graph, Tables 2 and 3,
originates from the DISNET database [19], which is a biomedical inte­
grated knowledge base containing information regarding diseases and
their associations to symptoms and drugs, among others. More infor­
mation about the data typologies, including the sources of the data and
date of extraction, are indicated in Table 1 of the Supplementary Ma­
terial. Note that, node type “ _phenotype_ ” contains both “ _disorders_ ” and


**Table 2**

Number of nodes per node type and its contribution to the total.


Node type Count Percent (%)


Phenotype 30,731 19.99
Drug 3944 2.56
Pathway 1105 0.72
Protein 18,521 12.05

DDI 99,446 64.68
Total 153,747 100



**Table 3**

Number of relations per relation type and its contribution to the total.


Relation type Count Percent (%)


Disease-protein 360,985 18.08
Disease-drug (therapeutic) 52,179 2.61
Disease-symptom 318,550 15.96
Disease-pathway 424 0.02
Drug-drug 662,281 33.17
Drug-protein 5946 0.30
Drug-symptom (side effect) 45,516 2.28
Drug-symptom (indication) 863 0.04
Protein-protein 240,585 12.05
Protein-pathway 10,991 0.55
DDI-phenotype 99,446 4.98
DDI-drug 198,892 9.96
Total 1,996,658 100


“ ”
_symptoms_ .
The node features are initialised as a 100-dimensional vector, values
are set to 1 since there are no specific node features and the algorithm
needs them for functioning. Other approaches include assigning node
identification in binary or randomly generated vectors. In any case, the
results would be similar, because none of the options encapsulate true
properties of the nodes and the algorithm will modify them accordingly.
To construct a particular evaluation set, the RepoDB test, informa­
tion from RepoDB [40] is extracted. RepoDB contains a set of drug
repositioning successes and failures that can be used to benchmark
computational repositioning methods. The RepoDB cases are removed
from the graph before entering the model’s pipeline. The training,
validating, testing, and evaluating sets are disjunct, there is no data
leakage between sets.
The choice of a heterogeneous graph is significative, since it is an
efficient way of incorporating a great variety of information in the same
data structure. This way the information between different types of
nodes (phenotypes and drugs) can flow easily, enrichening the under­
laying information the model uses to make predictions.


_3.2.1. Simple graph_
The simple version of the graph used a reduced set of nodes and
relations. Node types were “ _phenotype_ ” (30,731) and “ _drug_ ” (3944),
resulting in a total of 34,675 nodes. Edge types were “Disease-Drug”
(relationships considered to have a therapeutic effect) (52,179), “ _dis_­_
_dru_the_ ”, and “Disease-Symptom” (318,550), “ _dse_sym_ ”, resulting in a
total of 370,729 edges. The representation of the schema followed by
this graph can be seen in Fig. 1.



2
[https://zenodo.org/record/8402843/files/aiim2023-gnns-master.zip.](https://zenodo.org/record/8402843/files/aiim2023-gnns-master.zip) **Fig. 1.** Schema of the nodes and relations of the simple graph.


3


_A. Ayuso-Munoz et al._ ˜ _Artificial Intelligence In Medicine 145 (2023) 102687_



_3.2.2. Complex graph_
This version of the graph uses all the available information. Node
types are “ _phenotype_ ” (30,731), “ _drug_ ” (3944), “ _pathway_ ” (1105), “ _pro­_
_tein_ ” (18,521) and “ _drug-drug-interaction_ ” (99,446), in total 153,747
nodes. Edge types are Disease-Drug (therapeutic) (52,179), “ _dis_dru_the_ ”,
Disease-Symptom (318,550), “ _dse_sym_ ”, Disease-Protein (360,985),
“ _dis_pro_ ”, Disease-Pathway (424), “ _dis_pat_ ”, Drug-Drug (662,281),
“ _druA_druB_ ”, Drug-Protein (5946), “ _dru_pro_ ”, Drug-Symptom (side ef­
fect) (45,516), “ _dru_sym_sef_ ”, Drug-Symptom (indication) (863), “ _dru_­_
_sym_ind_ ”, Protein-Protein (240,585), “ _proA_proB_ ”, Protein-Pathway
(10,991), “ _pro_pat_ ”, DDI-Phenotype (99,446), “ _ddi_phe_ ”, and DDI-Drug
(198,892), “ _ddi_dru_ ”, in total 1,996,658 edges. A representation of the
schema followed by this graph can be seen in Fig. 2.


_3.3. Graph neural network_


The goal is to predict the probability of an edge existing between a
given disease-drug pair. To achieve it, graph information will be used to
generate embeddings of the entities by a GNN encoder, and then
calculate the edge’s probability by means of a simple decoder. The
problem is approached as an end-to-end problem, where the encoder
and the decoder are optimized in tandem.


_3.3.1. Model architecture_

The architecture is divided in two different parts, clearly separated.
The aim of the model is to solve the link prediction task, but each part of
the architecture specializes in a different problem. Encoder generates
representations of the nodes, while decoder uses those representations
to predict the probability of an edge’s existence between two nodes.
Generating node’s representative embeddings is the encoder’s task.
This step is crucial to the final performance of the model. Embeddings
must capture all the important features of a node in a graph. The en­
coder’s architecture, depicted in Fig. 3, has two layers. After each of



them a batch normalization takes place. Between the layers, there is a
leaky rectified linear unit (LReLU), providing non-linearity, and a
dropout layer, to reduce variance.
Layers use GraphSAGE [3 ] as a framework [41]. First, a node’s neigh­
borhood is sampled; then, the information of the sampled neighbours is
aggregated, generating a new embedding for the node, Eq. (2) [41]. This
process efficiently generates node embeddings inductively. Since there
are two of these layers, the model generates final embeddings using 2hop neighborhood information.



Being _x_ _disease_ the embedding vector of a disease node, _x_ _[T]_ _drug_ [the ]
transpose of the embedding vector of a drug node and _σ_ ( _x_ ) the sigmoid
function.


_3.3.2. Training, testing and evaluating_
We split the available data into three sets, train, test, and validation
sets. To train the model, the specific training set, formed by the 80 % of
the data of the original graph is employed. The remaining 20 % of the
data is divided as it follows: 10 % is assigned for internal training vali­
dation tasks ( _i.e._, to avoid overfitting, to tune the hyperparameters, and
to ensure the absence of data leakage) and a 10 % for testing tasks.
Decoder does not require training, there are no parameters, so just
the encoder undergoes the training phase. We trained the encoder using
the output of the decoder as the predicted value and of an edge existing
in the graph as target value.
During the training phase, model’s parameters are adjusted using the
Binary Cross Entropy with Logit Loss (BCELogitLoss) as the loss function,
which is computed as it can be seen in Eq. (4).


_l_ ( _x, y_ ) = _L_ = { _l_ 1 _,_ … _, l_ _N_ } _[T]_ _,_
(4)
_l_ _n_ = −[ _y_ _n_ - _log_ _σ_ ( _x_ _n_ ) + (1 − _y_ _n_ ) • _log_ (1 − _σ_ ( _x_ _n_ ) ) ]



_h_ _[k]_ _v_ [=] _[ σ]_ ( _W_ _[k]_ _MEAN_ ( { _h_ _[k]_ _v_ [−] [1]



} ∪ { _h_ _[k]_ _u_ [−] [1] _,_ ∀ _u_ ∈ _N_ ( _v_ ) } ) ) (2)



where, _h_ _[k]_ _v_ [is the embedding of node ] _[v ]_ [at iteration ] _[k]_ [, ] _[W]_ _[k ]_ [the weights ]
matrix of a fully connected layer, _N_ ( _v_ ) the neighbour sampler function
and _σ_ ( _x_ ) the sigmoid function. In this case, _k_ = {1 _,_ 2}, since the 2-hop


neighborhood is sampled. Aggregating part is _AGG_ =
{ _h_ _[k]_ _u_ [−] [1] _,_ ∀ _u_ ∈ _N_ ( _v_ ) } and updating is _σ_ ( _W_ _[k]_ _MEAN_ ({ _h_ _[k]_ _v_ [−] [1] } ∪ _AGG_ ) ).

Once the embeddings have been generated, the decoder will calcu­
late he probability of an edge existing between a disease and a drug. To
compute it, the sigmoid function is applied to the dot product of the
embeddings, as seen is Eq. (3).



( _W_ _[k]_ _MEAN_ ({ _h_ _[k]_ _v_ [−] [1]



∪ _AGG_ .
} ) )



_P_ = _σ_ ( _x_ _disease_ ⋅ _x_ _[T]_ _drug_



(3)
)



where _x_ _n_ is the model’s prediction and _y_ _n_ the true label (whether the
edge exists or not). We perform negative sampling [42,43], that is, for
each edge in the graph (positive edge), we sample a random edge
(negative edge) that is not present in the original graph. The ratio be­
tween positive and negative edges while training is one.
Hyperparameter tuning is conducted using Weight & Biases [44]
platform, which automates and eases this process. The hyperparameter
space is defined based on experience and intuition gained from previous
executions of the model. The hyperparameter optimisation used a
Bayesian approach, so it did not explore all the possible combinations.
The set of selected hyperparameters is the one with the higher average
area under the receiver operating characteristic curve (AUROC) tested
on the validation set, estimated through repeated hold-out with _k_ = 20.
The selected set of hyperparameters is 2752 epochs, 0.0008317 learning
**Fig. 2.** Schema of the nodes and relations of the complex graph.
rate, 107 hidden dimensions, 0.006314 weight decay and 0.8 dropout.


3 [https://snap.stanford.edu/deepsnap/_modules/deepsnap/hetero_gnn.](https://snap.stanford.edu/deepsnap/_modules/deepsnap/hetero_gnn.html#HeteroSAGEConv)
[html#HeteroSAGEConv.](https://snap.stanford.edu/deepsnap/_modules/deepsnap/hetero_gnn.html#HeteroSAGEConv)


4


_A. Ayuso-Munoz et al._ ˜ _Artificial Intelligence In Medicine 145 (2023) 102687_


**Fig. 3.** Structure of the Graph Neural Network encoder. There are two graph convolution layers followed by a batch normalization. In between the convolutions
there is a LReLU and a dropout layer.



As a brief analysis of the hyperparameters, the parameter importance
with respect to the AUROC, one of the metrics we use. Weight & Biases
platform provides this analysis, showing correlation and importance. It
uses random forest technique to calculate the importance of each
hyperparameter. Weight decay’s correlation is − 0.767 and importance
0.457, learning rate’s correlation is − 0.737 and importance 0.313,
hidden dimensions’ correlation is − 0.288 and importance 0.111,
epochs’ correlation − 0.153 and importance 0.051 and dropout’s cor­
relation is 0.351 and importance 0.022.
Those results show that weight decay is the most important hyper­
parameter for the model’s performance, and that the higher the weight
decay value the lower the reported metric’s value. The highest value
tested is 0.04745 and the lowest 0.0001422. The rest of the metrics are

analysed in an analogous way. As for the learning rate, the highest tested
value is 0.06807 and the lowest 0.000106. The highest value tested for
the hidden dimensions is 128 and the lowest is 9. The model that was

trained for the most epochs was trained for 9529 epochs and the one
trained with the fewest epochs was trained for 413 epochs. Dropout
analysis is different since its correlation is positive. The higher the
dropout value, the better the reported metric’s value. The highest tested
dropout value is 0.9 and the lowest 0.4.
The interaction between the hyperparameters is complex, the values
of correlation and importance we supply are just an indicator of what
could be happening inside the model.


_3.3.3. RepoDB test_
To ensure reliability, especially important in this context, we
perform a second evaluation with RepoDB [40]. The connections be­
tween drug and diseases that RepoDB offers are used to validate and
interpret the results returned by the trained predictor.
Among these cases, 5013 repositioning drug-disease pairs are
selected, of which 1824 are present in our graph and had to be taken out
(1824). As for the nodes, we selected only the cases from RepoDB where
both the drug and the disease node were already in the graph. Since the
nodes are involved in other relations none of them are taken out. The

selected pairs are fed into the trained model, ideally, all pairs should
have one as their score value.

Moreover, to guarantee the quality of the model and to supplement
this test, another set of 5013 randomly generated drug-disease pairs is
fed into the model. In this case, ideally, all pairs should have 0 as their
score value, since the probability of picking a valid pair for repurposing
is considerably small.
Note that, among the 5013 repositioning pairs from RepoDB there
are elements of the four following classes: i) approved, ii) terminated,
iii) withdrawn and iv) suspended. Initially, the four classes were treated
separately but results were similar, so classes were joined. This was done



under the following assumption: if a drug-disease pair was tested this
indicates that there is a relevant relation that induced this study.
Whether the drug was finally repurposed or not is not as important as
identifying that the drug could be potentially repurposed.
This test verifies that the tested models can identify new drug
repurposing cases, even though, it cannot be assured if this case will be,
finally, approved or not. It is important to note that this process only
serves as a preliminary step in identifying which drug repurposing pairs
are worth further investigation, with experimental validation being

necessary.


_3.4. Experimental setup_


To contextualize the results of BEHOR, it is compared against five
baselines for working on graphs:


 - DeepWalk [28] uses short fixed length random walks to obtain local
information, therefore relying on structural information, indepen­
dent of label’s distribution, to generate the embeddings in an unsu­
pervised way. Random walks are uniformly sampled. It uses
SkipGram [45] to fit node representations, since random walks can
be seen as a sentences and nodes as words the application of this
model is direct. The idea is, given a random walk, to estimate the
likelihood of observing a certain node. In other words, to relate each
node to its context.

 - Node2Vec [29] works quite similarly to DeepWalk, the only differ­

ence is how the set of neighbouring nodes and random walks are
defined. Here, random walks are not uniformly sampled, this method
introduces biased random walks, using probabilities, extrapolating
between local and global views. Consequently, diverse neighbour­
hoods of a node are explored, and richer embeddings are generated.

 - NetMF [46] proposes itself as matrix factorization method with
closed forms unifying skip-gram models, like DeepWalk or Node2­
Vec. It lays its foundations on the basis that all the previously
mentioned methods perform implicit matrix factorization. NetMF
explicitly factorizes those closed-form matrices.

 - Role2Vec [47] generalizes previous random walk methods, such as
DeepWalk or Node2Vec. The main difference is that it can be used in
inductive and transductive settings and the capability of integrating
node’s features. Using attributed random walks as its core, it learns
functions that generalize to new nodes and graphs. Attributed
random walks, in essence, is a random walk that also captures in­
formation about node features.

 - REDIRECTION [18] is the previous version of the model. There are 2
important changes: i) edges are undirected, which yields an



5


_A. Ayuso-Munoz et al._ ˜ _Artificial Intelligence In Medicine 145 (2023) 102687_



important performance improvement; and ii) hyperparameter opti­
mization. Moreover, this version was only trained for the simple
graph.


DeepWalk, Node2Vec and NetMF baselines are called shallow en­
coders and produce shallow embeddings. This means, each node has a
unique embedding, in fact these methods are called embedding lookup
methods [30]. Moreover, these baselines are transductive, embeddings
can only be generated for nodes present during training, they lack
generalization capabilities, as opposed to Role2Vec, REDIRECTION or
BEHOR, inherently inductive. Every model uses a an inner-product
decoder. No hyperparameter optimization is done, default hyper­
parameters are used for every baseline.
To ensure representative results, all the metrics were estimated using
multiple resampling-based estimation methods.
The baselines tests were developed in Python 3.8.15, using PyTorch
1.10.2, DGL 0.7.2 and NetworkX 2.6.3. We made use of CUDA Toolkit
11.3, running the experiments on an Ubuntu Server LTS 20.04.4 with a
GPU (NVIDIA GeForce RTX 3090 24GB). For testing DeepWalk, Role2­
Vec, Node2Vec and NetMF baselines Karate Club library [48] was used.
All the materials and results related to the baselines testing have been
published in an accessible repository, [4 ] to make the results that have
been obtained reproducible.


_3.5. Evaluation metrics_


To evaluate the models the receiving operating characteristics (ROC)
curve and precision-recall (PR) curve are used. The curves are analysed
visually, but to ease the comparison between models the area under
(AU) each curve is used.
ROC curves were used to asses model performance in 1989 for the
first time [49]. These curves are constructed by plotting the true positive
rate (TPR) against the false positive rate (FPR) using different thresh­
olds. TPR is the proportion of correctly labelled positive instances out of
all the positive instances. FPR is the proportion of wrongly labelled
negative instances, labelled as positive, out of all the negative instances.
Using the AU, the perfect classifier has a value of 1, while the random
one has a value of 0.5. Note that, ROC curves are biased when working
with imbalance, they are biased towards the majority class. In this work
there is no class imbalance.

PR curves, in this work, are used as a complementary metric to ROC.
The use of PR curves is especially interesting in those cases where FPR is
small [50]. These curves are constructed by plotting the precision
against the recall using different thresholds. Precision is the proportion
of correctly labelled positive instances out of all the labelled positive
instances. Recall is the same as TPR. The higher the AU, the better, up to

one.

In this work, out of the presented metrics, precision will be partic­
ularly considered. High precision models will offer a set of predictions
with high confidence, all positive instances may not be recovered, but
the recovered ones have quite a high prediction of being truly positive.


**4. Results and discussion**


_4.1. Results_



Confidence intervals (95 %) are calculated using the t-distribution,
since the number of samples is under thirty. The number of degrees of
freedom is nineteen.


Table 4 contains the comparison of the baselines’ results and BEHOR.
REDIRECTION was just trained and tested on the simple graph. As
observed, the presented models overperform the baselines by a large
margin. The model trained on the complex version of the graph performs
slightly better than the simple one and its variance is lower, it is overall a
better performing and more stable model. Reported metrics show great
results for BEHOR models, proving the capacity of them to identify novel
therapeutic indications through drug repurposing.
The complex model improves the simple previous model by,
approximately, 0.09 for the AUROC and 0.12 for the AUPRC. Although it
may not seem like a significant improvement, it is noteworthy consid­
ering the impressive results achieved by the previous model.


_4.2. Novel predictions_


The graph contains over 121 million potential hypotheses for drug
repurposing. To narrow down the options for study, a set of filters were
applied. The first filter was used to select cases with 1 as a prediction
score, reducing the cases to approximately 1.5 million. The second filter
further narrowed down the options by selecting diseases with no existing
therapeutic treatments in the graph and with a T047 semantic type
(disease or syndrome), resulting in a reduction to over 165 thousand
cases. The third and final filter, was applied to select those cases con­
taining drugs associated to fewer than 15 diseases in the graph, leaving
4599 cases for study. The decision to set the threshold at 15 was based
on the distribution of phenotype-drug associations shown in Fig. 1 of the
Supplementary Material.
Once these filters are applied, a series of model generated novel
predictions are selected for further analysis, Table 5.
The first discussed hypotheses are the ones for **Congestive Heart**
**Failure** . Congestive heart failure is a severe illness characterized by
indications of impaired ventricular function [52]. It may result from
several heart diseases and is a significant concern in global health policy

[53].


 - **Nestiride** is a recombinant human B-type natriuretic peptide (BNP)
with advantageous effects on vasodilation, sodium and fluid excre­
tion, and the nervous system [54]. This drug mimics the actions of
endogenous natriuretic peptides causing balanced arterial and
venous dilatation. Due to the decrease of systemic vascular resis­
tance, systemic arterial pressure, pulmonary capillary pressure, right
atrial pressure and mean pulmonary arterial pressure, it is adminis­
tered intravenously for the management of adult patients with
decompensated congestive heart failure [54,55]. In addition, studies


**Table 4**

AUROC and AUPRC for every baseline and the presented model. The test is
conducted on the two graph types, the simple and the complex one. Bold type
indicates the best results among the ones seen for each column.


Baseline Simple Complex


AUROC AUPRC AUROC AUPRC



All the presented results are obtained using multiple resampling- DeepWalk 0.3842 ± 0.4526 ± 0.4865 ± 0.4831 ±
based estimation methods during training time and reporting the re­ 0.0021 0.0015 0.0023 0.0015
sults obtained on the RepoDB test, so there is certain stability of the Node2Vec 0.3899 ± 0.4567 ± 0.4830 ± 0.4829 ±
estimates. Cross validation is generally the best approach to obtain the 0.0022 0.0013 0.0022 0.0015
estimations, but it can be computationally expensive [51]. Due to time NetMF 0.6120 ± 0.6463 ± 0.7457 ± 0.6964 ±

0.0005 0.0011 0.0013 0.0018

complexity and the large dataset, we estimate values through repeated Role2Vec 0.6675 ± 0.6939 ± 0.6735 ± 0.6874 ±
hold-out with _k_ = 20. 0.0012 0.0019 0.0014 0.0019


REDIRECTION 0.8700 0.8300 – –

BEHOR **0.9550 ±** **0.9484 ±** **0.9604 ±** **0.9518 ±**

4 **0.0017** **0.0019** **0.0006** **0.0009**

.
[https://zenodo.org/record/8402843/files/aiim2023-baselines-master.zip](https://zenodo.org/record/8402843/files/aiim2023-baselines-master.zip)


6


_A. Ayuso-Munoz et al._ ˜ _Artificial Intelligence In Medicine 145 (2023) 102687_



**Table 5**

Top N novel predictions of the model. All the presented hypotheses got the
maximum prediction score (1).


Disease Disease name Drug CHEMBL ID Drug name Clinical

UMLS CUI trials


C0018802 Congestive heart CHEMBL1201668 Nesiritide 554
failure CHEMBL1351 Liraglutide 23
C0003873 Rheumatoid CHEMBL960 Leflunomide 78
arthritis CHEMBL1789941 Ruxolitinib 4

C0149721 Left ventricular CHEMBL3137301 Sacubitril 8

hypertrophy


have shown that Nesiritide is well tolerated in children with heart

failure and is associated with improved diuresis [56].

 - **Liraglutide** is a synthetic analogue of human glucagon-like peptide1(GLP-1) and acts as a GLP-1 receptor agonist [57]. It is indicated for
weight loss and Type 2 Diabetes Mellitus. This drug rises the levels of
cyclic AMP boost insulin release based on glucose levels, block
glucagon release based on glucose, and slow down stomach
emptying for better blood sugar control [58]. Liraglutide has been
demonstrated to have beneficial effects on heart function in patients
with Type 2 Diabetes and heart failure improving the left ventricular
ejection fraction [59].


The secondly discussed hypotheses are the ones for **Rheumatoid**
**Arthritis** . Rheumatoid Arthritis (RA) is a chronic autoimmune disease
characterized by persistent synovitis, systemic inflammation, and au­
toantibodies (particularly rheumatoid factor and citrullinated peptide)

[60]. It is the most common inflammatory arthritis, and a significant
cause of morbidity and mortality [61], affecting approximately 0.5–1 %
of the worldwide population.


 - **Leflunomide** is a disease-modifying anti-rheumatic drug (DMARD)
and is actually approved for the treatment of RA [62–64]. Several
mechanisms of action of leflunomide have been described, but the
main mechanism responsible for its effectiveness in treatment of RA
is the ability of A77-1726 to inhibit dihydroorotate dehydrogenase,
the rate-limiting enzyme for _de novo_ synthesis of pyrimidine nucle­
otides [65,66]. In autoimmune diseases such as RA, activated lym­
phocytes require an increased pyrimidine nucleotide pool (at least 8fold) to progress from G1 to S phase of the cell cycle. Inhibition of
pyrimidine synthesis by leflunomide results in decreased pyrimidine
nucleotide pools, thereby inducing cell-cycle arrest and inhibition of
lymphocyte clonal expansion in patients with RA [67].

 - **Ruxolitinib** is a potent and selective oral inhibitor of Janus kinase
(JAK) 1 and JAK2 [68]. JAKs are multidomain non-receptor tyrosine
kinases that have pivotal roles in cellular signal transduction. Acti­
vation and transphosphorylation of JAKs induce signal transducer
and activator of transcription (STAT) recruitment, dimerization,
nuclear translocation, and resultant transcriptional responses [69].
Therefore, JAK–STAT signalling has a pivotal role in a pleotropic
range of systems, including the orchestration and functional capa­
bility of immune responses [70,71]. Due to this central role of JAKs
in the immune response and their association with several cytokine
receptors, the inhibition of JAKs appeared to be a promising strategy
in autoimmune diseases [72], including RA. Ruxolitinib is already
licensed for the treatment of some myeloproliferative neoplasms
(MPNs) and some studies have already pointed out its utility for RA

[73–75].


Finally, the hypotheses for **Left Ventricular Hypertrophy** . Left
ventricular hypertrophy (LVH) is a cardinal manifestation of end-organ
damage due to hypertension; its reported prevalence has ranged from 36
to 41 % in echocardiographic studies in individuals with elevated blood
pressure [76]. The development of pathologic LVH is classically attrib­
uted to a maladaptive response to long-standing pressure-volume



overload of the left ventricle (LV). The cellular hallmarks of hyperten­
sive heart disease (HHD), including cardiomyocyte hypertrophy and
fibrotic changes in the non-cardiomyocyte components of myocardium,
have implicated a variety of causal factors such as neurohormonal
activation and other signalling pathways [77,78].


 - **Sacubitril**
is usually combined with valsartan as a fixed-dose com­

bination (FDC) to treat several types of heart failures in different
conditions [79,80]. Sacubitril/valsartan is the first in a new class of

–
drug: the angiotensin receptor neprilysin inhibitors (ARNI). Its
mechanisms of action have not been well defined. Sacubitril/val­
sartan causes simultaneous augmentation of the natriuretic peptide
system (NPS) and inhibition of the renin–angiotensin–aldosterone
system (RAAS) [81]. RAAS inhibition by valsartan has been exten­
sively tested in patients with heart failure (HF) [82]. However, the
actions of sacubitril remain unknown.


_4.3. Limitations_


We are aware the presented work has some limitations. One of the
main limitations is the lack of features for the nodes in the graph.
Therefore, losing some advantages of the GraphSAGE algorithm and the
potential results improvement that could take place if the model made
use of that information. Other main limitation we notice is the great
difference in the model’s predictions depending on the nodes. Nodes
that have many connections in the graph will tend to have higher scores
than those that are isolated. So, the model is biased towards the nodes
with higher connectivity. We plan to address this in the future, some
ideas are pruning the graph or the focus on specific information for every
node. The same way, because of the way the learning of the model takes
place, those nodes that do not have any connections in the graph will not
get useful predictions.
The work is limited too in the relationships it can predict, its use
cases, the model is just trained to predict Disease-Drug (therapeutic)
edges. The rich heterogeneous graph allows holding many relationship
types in the same data structure. We could make the model take
advantage of all the available information and predict any edge type.
Though, for this work, we gave more importance to the specialisation on
drug repurposing.


**5. Conclusions**


The use of heterogeneous biomedical information structured as a
graph has proven valuable in combination with GNNs to tackle the drug
repurposing problem. Additionally, the generation of drug repurposing
hypotheses by BEHOR, framed in the GDL field, has the potential to
streamline the drug repurposing process and reduce costs and time. The
analysis of some novel predictions generated by BEHOR indicates its
ability to produce valid drug repurposing hypotheses, which was further
confirmed by the results of testing with both the test graph and the
RepoDB test.
The primary contribution of this paper is the development of a GDL
model on a complex and rich set of information, DISNET, to train drug
repurposing hypotheses. This new model surpasses its predecessor by
roughly 0.09 for AUROC and 0.12 for AUPRC in the RepoDB test. While
this improvement may appear small, it is significant given the impres­
sive results obtained by the previous model, 0.87 for AUROC and 0.83
for AUPRC. In addition, the discussion of a set of novel predictions
validates the potential of this method.
Nonetheless, it should be noted that BEHOR’s predictions cannot be
considered as a definitive truth and require experts and experimental
validation. It is crucial to understand that BEHOR should not be viewed

as a substitute for medical prescription systems. Rather, it serves as a
tool to suggest potential drug repurposing pairs that may be worth
further investigation.
As for the future lines, a number of improvements are possible, such



7


_A. Ayuso-Munoz et al._ ˜ _Artificial Intelligence In Medicine 145 (2023) 102687_



as the further exploration of the hyperparameter space. However, there
is one critical area that needs to be prioritized, which is the incorpora­
tion of node features. There are no features for the nodes, GNNs would
greatly benefit from the presence of node features. It is expected to gain
a great performance improvement with its inclusion, but which features
to incorporate is a question that has not been solved yet.
Among the other future research directions, which are of lower pri­
ority, it could be interesting to observe the model’s behaviour under
different circumstances. Testing different types of encoders, such as
other types of classic GNNs, other types of decoders, since the used
encoder is simple and has no parameters, like the use of multilayer
perceptron (MLP) and loss functions could provide valuable insights.
Distinguishing between the different RepoDB classes is also an
intriguing avenue for future research. Finally, the development of more
sophisticated evaluation techniques is an interesting direction for future
work. The RepoDB test verifies that the tested models can identify new
drug repurposing cases, even though, it cannot be assured if this case
will be, finally, approved or not.


**Data and code availability**


The data employed and code developed for the present study are
[openly available at https://zenodo.org/record/8402843.](https://zenodo.org/record/8402843)
Zenodo repository is a clone with the content, also available in
GitLab, of the following repositories:

[https://medal.ctb.upm.es/internal/gitlab/disnet/gnns/aiim](https://medal.ctb.upm.es/internal/gitlab/disnet/gnns/aiim2023-gnns)
[2023-gnns](https://medal.ctb.upm.es/internal/gitlab/disnet/gnns/aiim2023-gnns)
[https://medal.ctb.upm.es/internal/gitlab/disnet/gnns/aiim2023-](https://medal.ctb.upm.es/internal/gitlab/disnet/gnns/aiim2023-baselines)
[baselines](https://medal.ctb.upm.es/internal/gitlab/disnet/gnns/aiim2023-baselines)


**Declaration of competing interest**


None.


**Acknowledgment**


The work is a result of the project “Data-driven drug repositioning
applying graph neural networks (3DR-GNN)”, that is being developed
under grant “PID2021-122659OB-I00” from the Spanish Ministerio de
Ciencia e Innovacion. This work has been supported by project ´
MadridDataSpace4Pandemics, funded by Comunidad de Madrid (Con­
sejería de Educacion, Universidades, Ciencia y Portavocía) with FEDER ´
funds as part of the response from the European Union to COVID-19
pandemic.


**Appendix A. Supplementary data**


[Supplementary data to this article can be found online at https://doi.](https://doi.org/10.1016/j.artmed.2023.102687)
[org/10.1016/j.artmed.2023.102687.](https://doi.org/10.1016/j.artmed.2023.102687)


**References**


[1] Pushpakom S, et al. Drug repurposing: progress, challenges and recommendations.
[Nat Rev Drug Discov Jan. 2019;18(1):41–58. https://doi.org/10.1038/](https://doi.org/10.1038/nrd.2018.168)
[nrd.2018.168.](https://doi.org/10.1038/nrd.2018.168)

[2] Rudrapal M, Khairnar SJ, Jadhav AG. Drug repurposing (DR): an emerging
approach in drug discovery, drug repurposing-hypothesis, molecular aspects and
[therapeutic applications. London, United Kingdom: IntechOpen; 2020. https://doi.](https://doi.org/10.5772/intechopen.93193)
[org/10.5772/intechopen.93193.](https://doi.org/10.5772/intechopen.93193)

[3] Cheng L, Li J, Ju P, Peng J, Wang Y. SemFunSim: a new method for measuring
disease similarity by integrating semantic and gene functional association. PloS
[One Jun. 2014;9(6):e99415. https://doi.org/10.1371/journal.pone.0099415.](https://doi.org/10.1371/journal.pone.0099415)

[4] Gysi DM, et al. Network medicine framework for identifying drug-repurposing
[opportunities for COVID-19. Proc Natl Acad Sci May 2021;118(19). https://doi.](https://doi.org/10.1073/pnas.2025581118)
[org/10.1073/pnas.2025581118.](https://doi.org/10.1073/pnas.2025581118)

[5] Hsieh K, et al. Drug repurposing for COVID-19 using graph neural network and
[harmonizing multiple evidence. Sci Rep Nov. 2021;11(1):1. https://doi.org/](https://doi.org/10.1038/s41598-021-02353-5)
[10.1038/s41598-021-02353-5.](https://doi.org/10.1038/s41598-021-02353-5)




[6] Doshi S, Chepuri SP. A computational approach to drug repurposing using graph
[neural networks. Comput Biol Med Nov. 2022;150:105992. https://doi.org/](https://doi.org/10.1016/j.compbiomed.2022.105992)
[10.1016/j.compbiomed.2022.105992.](https://doi.org/10.1016/j.compbiomed.2022.105992)

[[7] Sosa DN, Derry A, Guo M, Wei E, Brinton C, Altman RB. A literature-based](http://refhub.elsevier.com/S0933-3657(23)00201-4/rf0035)
[knowledge graph embedding method for identifying drug repurposing](http://refhub.elsevier.com/S0933-3657(23)00201-4/rf0035)
[opportunities in rare diseases. Pac Symp Biocomput Pac Symp Biocomput 2020;25:](http://refhub.elsevier.com/S0933-3657(23)00201-4/rf0035)
[463–74.](http://refhub.elsevier.com/S0933-3657(23)00201-4/rf0035)

[8] Bajaj P, Heereguppe S, Sumanth C. Graph convolutional networks to explore drug
and disease relationships in biological networks. Accessed: May 02, 2022. [Online].
[Available: http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-41-final.](http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-41-final.pdf)

[[9] Prieto Santamaría L, Ugarte Carro E, Díaz Uzquiano M, Menasalvas Ruiz E, PpdfGallardo Y, Rodríguez-Gonz; 2017.](http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-41-final.pdf) alez A. A data-driven methodology towards evaluating ´ ´erez
the potential of drug repurposing hypotheses. Comput Struct Biotechnol J Jan.

[[10] Prieto Santamaría L, Díaz Uzquiano M, Ugarte Carro E, Ortiz-Rold2021;19:4559Gallardo Y, Rodríguez-Gonz–73. https://doi.org/10.1016/j.csbj.2021.08.003´alez A. Integrating heterogeneous data to facilitate .](https://doi.org/10.1016/j.csbj.2021.08.003) ´an N, P´erez
[COVID-19 drug repurposing. Drug Discov Today Feb. 2022;27(2):558–66. https://](https://doi.org/10.1016/j.drudis.2021.10.002)

[[11] Barabdoi.org/10.1016/j.drudis.2021.10.002asi A-L, Gulbahce N, Loscalzo J. Network medicine: a network-based ´](https://doi.org/10.1016/j.drudis.2021.10.002) .
[approach to human disease. Nat Rev Genet Jan. 2011;12(1):56–68. https://doi.](https://doi.org/10.1038/nrg2918)

[12] [org/10.1038/nrg2918Zitnik M, Janji](https://doi.org/10.1038/nrg2918) [ˇ] ´c V, Larminie C, Zupan B, Pr. ˇzulj N. Discovering disease-disease
[associations by fusing systems-level molecular data. Sci Rep Nov. 2013;3. https://](https://doi.org/10.1038/srep03202)
[doi.org/10.1038/srep03202.](https://doi.org/10.1038/srep03202)

[13] Zhang Z, Cui P, Zhu W. Deep learning on graphs: a survey. IEEE Trans Knowl Data
[Eng 2020;34(1):249–70. https://doi.org/10.1109/TKDE.2020.2981333.](https://doi.org/10.1109/TKDE.2020.2981333)

[14] Gaudelet T, et al. Utilizing graph machine learning within drug discovery and
[development. Brief Bioinform May 2021;(bbab159). https://doi.org/10.1093/bib/](https://doi.org/10.1093/bib/bbab159)
[bbab159.](https://doi.org/10.1093/bib/bbab159)

[15] Zhou J, et al. Graph neural networks: a review of methods and applications. AI
[Open Jan. 2020;1:57–81. https://doi.org/10.1016/j.aiopen.2021.01.001.](https://doi.org/10.1016/j.aiopen.2021.01.001)

[16] Li MM, Huang K, Zitnik M. Graph representation learning in biomedicine and
[healthcare. Nat Biomed Eng Dec. 2022;6(12):12. https://doi.org/10.1038/s41551-](https://doi.org/10.1038/s41551-022-00942-x)
[022-00942-x.](https://doi.org/10.1038/s41551-022-00942-x)

[17] Abbas K, et al. Application of network link prediction in drug discovery. BMC
[Bioinformatics Apr. 2021;22(1):187. https://doi.org/10.1186/s12859-021-04082-](https://doi.org/10.1186/s12859-021-04082-y)

[[18] A. Ayuso Muy.](https://doi.org/10.1186/s12859-021-04082-y) noz et al., ˜ “REDIRECTION: Generating drug repurposing hypotheses
using link prediction with DISNET data,” in 2022 IEEE 35th international

[19] Lagunes-García G, Rodríguez-Gonzsymposium on computer-based medical systems (CBMS). alez A, Prieto-Santamaría L, del Valle EPG, ´
Zanin M, Menasalvas-Ruiz E. DISNET: a framework for extracting phenotypic
[disease information from public sources. PeerJ Feb. 2020;8:e8580. https://doi.](https://doi.org/10.7717/peerj.8580)

[[20] Goh K-I, Cusick ME, Valle D, Childs B, Vidal M, Baraborg/10.7717/peerj.8580.](https://doi.org/10.7717/peerj.8580) ´asi A-L. The human disease
[network. Proc Natl Acad Sci May 2007;104(21):8685–90. https://doi.org/](https://doi.org/10.1073/pnas.0701361104)
[10.1073/pnas.0701361104.](https://doi.org/10.1073/pnas.0701361104)

[21] Zhao B-W, et al. Fusing higher and lower-order biological information for drug
repositioning via graph representation learning. IEEE Trans Emerg Top Comput
[2023:1–14. https://doi.org/10.1109/TETC.2023.3239949.](https://doi.org/10.1109/TETC.2023.3239949)

[22] Su X, Hu P, Yi H, You Z, Hu L. Predicting drug-target interactions over
heterogeneous information network. IEEE J Biomed Health Inform Jan. 2023;27
[(1):562–72. https://doi.org/10.1109/JBHI.2022.3219213.](https://doi.org/10.1109/JBHI.2022.3219213)

[23] Zhong Y, et al. DDI-GCN: drug-drug interaction prediction via explainable graph
[convolutional networks. Artif Intell Med Oct. 2023;144:102640. https://doi.org/](https://doi.org/10.1016/j.artmed.2023.102640)
[10.1016/j.artmed.2023.102640.](https://doi.org/10.1016/j.artmed.2023.102640)

[[24] LeCun Y, Bengio Y. Convolutional networks for images, speech, and time series.](http://refhub.elsevier.com/S0933-3657(23)00201-4/rf0115)
[Handb Brain Theory Neural Netw 1995;3361(10):1995.](http://refhub.elsevier.com/S0933-3657(23)00201-4/rf0115)

[25] Sperduti A, Starita A. Supervised neural networks for the classification of
[structures. IEEE Trans Neural Netw 1997;8(3):714–35. https://doi.org/10.1109/](https://doi.org/10.1109/72.572108)
[72.572108.](https://doi.org/10.1109/72.572108)

[26] Frasconi P, Gori M, Sperduti A. A general framework for adaptive processing of
[data structures. IEEE Trans Neural Netw 1998;9(5):768–86. https://doi.org/](https://doi.org/10.1109/72.712151)
[10.1109/72.712151.](https://doi.org/10.1109/72.712151)

[27] Bronstein MM, Bruna J, LeCun Y, Szlam A, Vandergheynst P. Geometric deep
learning: going beyond Euclidean data. IEEE Signal Process Mag Jul. 2017;34(4):
[18–42. https://doi.org/10.1109/MSP.2017.2693418.](https://doi.org/10.1109/MSP.2017.2693418)

[28] Perozzi B, Al-Rfou R, Skiena S. DeepWalk: online learning of social representations.
In: Proceedings of the 20th ACM SIGKDD international conference on knowledge
discovery and data mining, in KDD ’14. New York, NY, USA: Association for
[Computing Machinery; Aug. 2014. p. 701–10. https://doi.org/10.1145/](https://doi.org/10.1145/2623330.2623732)
[2623330.2623732.](https://doi.org/10.1145/2623330.2623732)

[29] Grover A, Leskovec J. node2vec: Scalable feature learning for networks. In:
Proceedings of the 22nd ACM SIGKDD international conference on knowledge
discovery and data mining, in KDD ’16. New York, NY, USA: Association for
[Computing Machinery; Aug. 2016. p. 855–64. https://doi.org/10.1145/](https://doi.org/10.1145/2939672.2939754)
[2939672.2939754.](https://doi.org/10.1145/2939672.2939754)

[30] Hamilton WL. Graph representation learning. Synth Lect Artif Intell Mach Learn
[Sep. 2020;14(3):1–159. https://doi.org/10.2200/](https://doi.org/10.2200/S01045ED1V01Y202009AIM046)
[S01045ED1V01Y202009AIM046.](https://doi.org/10.2200/S01045ED1V01Y202009AIM046)

[31] Zhou Y, Hou Y, Shen J, Huang Y, Martin W, Cheng F. Network-based drug
repurposing for novel coronavirus 2019-nCoV/SARS-CoV-2. Cell Discov Mar. 2020;
[6(1):1–18. https://doi.org/10.1038/s41421-020-0153-3.](https://doi.org/10.1038/s41421-020-0153-3)



8


_A. Ayuso-Munoz et al._ ˜ _Artificial Intelligence In Medicine 145 (2023) 102687_




[32] Zeng X, et al. Repurpose open data to discover therapeutics for COVID-19 using
[deep learning. J Proteome Res Nov. 2020;19(11):4624–36. https://doi.org/](https://doi.org/10.1021/acs.jproteome.0c00316)
[10.1021/acs.jproteome.0c00316.](https://doi.org/10.1021/acs.jproteome.0c00316)

[33] Ioannidis VN, Zheng D, Karypis G. Few-shot link prediction via graph neural
networks for Covid-19 drug-repurposing. ArXiv200710261 Cs Stat Jul. 2020.
[Accessed: Nov. 16, 2021. [Online]. Available: http://arxiv.org/abs/2007.10261.](http://arxiv.org/abs/2007.10261)

[34] Drug Repurposing Knowledge Graph (DRKG). gnn4dr 2020. Accessed: Feb. 20,
[2023. [Online]. Available: https://github.com/gnn4dr/DRKG.](https://github.com/gnn4dr/DRKG)

[35] Mei X, Cai X, Yang L, Wang N. Relation-aware heterogeneous graph transformer
[based drug repurposing. Expert Syst Appl Mar. 2022;190:116165. https://doi.org/](https://doi.org/10.1016/j.eswa.2021.116165)
[10.1016/j.eswa.2021.116165.](https://doi.org/10.1016/j.eswa.2021.116165)

[36] Gu Y, Zheng S, Yin Q, Jiang R, Li J. REDDA: integrating multiple biological
relations to heterogeneous graph neural network for drug-disease association
[prediction. Comput Biol Med Nov. 2022;150:106127. https://doi.org/10.1016/j.](https://doi.org/10.1016/j.compbiomed.2022.106127)
[compbiomed.2022.106127.](https://doi.org/10.1016/j.compbiomed.2022.106127)

[37] The comparative toxicogenomics database | CTD. Accessed: Jan. 11, 2022.

[[Online]. Available: http://ctdbase.org/.](http://ctdbase.org/)

[38] Zitnik M, Agrawal M, Leskovec J. Modeling polypharmacy side effects with graph
[convolutional networks. Bioinforma Oxf Engl Jul. 2018;34(13):i457–66. https://](https://doi.org/10.1093/bioinformatics/bty294)
[doi.org/10.1093/bioinformatics/bty294.](https://doi.org/10.1093/bioinformatics/bty294)

[39] Percha B, Altman RB. A global network of biomedical relationships derived from
[text. Bioinforma. Oxf. Engl. Aug. 2018;34(15):2614–24. https://doi.org/10.1093/](https://doi.org/10.1093/bioinformatics/bty114)
[bioinformatics/bty114.](https://doi.org/10.1093/bioinformatics/bty114)

[40] Brown AS, Patel CJ. A standard database for drug repositioning. Sci Data Mar.
[2017;4(1):170029. https://doi.org/10.1038/sdata.2017.29.](https://doi.org/10.1038/sdata.2017.29)

[41] Hamilton W, Ying Z, Leskovec J. Inductive representation learning on large graphs.
In: Advances in Neural Information Processing Systems. Curran Associates, Inc.;
[2017. Accessed: Nov. 04, 2021. [Online]. Available: https://proceedings.neurips.](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)
[cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html.](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)

[42] Mikolov T, Sutskever I, Chen K, Corrado GS, Dean J. Distributed representations of
words and phrases and their compositionality. In: Advances in neural information
processing systems. Curran Associates, Inc.; 2013. Accessed: May 09, 2022.

[[Online]. Available: https://papers.nips.cc/paper/2013/hash/9aa42b31882ec](https://papers.nips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html)
[039965f3c4923ce901b-Abstract.html.](https://papers.nips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html)

[43] Trouillon T, Welbl J, Riedel S, Gaussier E, Bouchard G. Complex embeddings for
simple link prediction. In: Proceedings of the 33rd international conference on
machine learning. PMLR; Jun. 2016. p. 2071–80. Accessed: May 09, 2022.

[[Online]. Available: https://proceedings.mlr.press/v48/trouillon16.html.](https://proceedings.mlr.press/v48/trouillon16.html)

[[44] Biewald L. Experiment tracking with weights and biases [Online]. Available: htt](https://www.wandb.com/)
[ps://www.wandb.com/; 2020.](https://www.wandb.com/)

[45] Mikolov T, Chen K, Corrado G, Dean J. Efficient estimation of word representations
[in vector space. arXiv 2013. https://doi.org/10.48550/arXiv.1301.3781. Sep. 06.](https://doi.org/10.48550/arXiv.1301.3781)

[46] Qiu J, Dong Y, Ma H, Li J, Wang K, Tang J. Network embedding as matrix
factorization: unifying DeepWalk, LINE, PTE, and node2vec. In: Proceedings of the
eleventh ACM international conference on web search and data mining. Marina Del
[Rey CA USA: ACM; Feb. 2018. p. 459–67. https://doi.org/10.1145/](https://doi.org/10.1145/3159652.3159706)
[3159652.3159706.](https://doi.org/10.1145/3159652.3159706)

[[47] Ahmed NK, et al. Learning role-based graph embeddings. arXiv 2018. https://doi.](https://doi.org/10.48550/arXiv.1802.02896)
[org/10.48550/arXiv.1802.02896. Jul. 02.](https://doi.org/10.48550/arXiv.1802.02896)

[48] Rozemberczki B, Kiss O, Sarkar R. Karate Club: an API oriented open-source Python
framework for unsupervised learning on graphs. In: Proceedings of the 29th ACM
International Conference on Information & Knowledge Management, CIKM ’20.
New York, NY, USA: Association for Computing Machinery; 2020. p. 3125–32.
[https://doi.org/10.1145/3340531.3412757.](https://doi.org/10.1145/3340531.3412757)

[49] Spackman KA. Signal detection theory: valuable tools for evaluating inductive
learning. In: Segre AM, editor. Proceedings of the sixth international workshop on
[machine learning. San Francisco (CA): Morgan Kaufmann; 1989. p. 160–3. https://](https://doi.org/10.1016/B978-1-55860-036-2.50047-3)
[doi.org/10.1016/B978-1-55860-036-2.50047-3.](https://doi.org/10.1016/B978-1-55860-036-2.50047-3)

[50] Davis J, Goadrich M. The relationship between precision-recall and ROC curves.
presented at the Proceedings of the 23rd International Conference on Machine
[Learning. ACM; Jun. 2006. https://doi.org/10.1145/1143844.1143874.](https://doi.org/10.1145/1143844.1143874)

[51] Kim J-H. Estimating classification error rate: repeated cross-validation, repeated
hold-out and bootstrap. Comput Stat Data Anal Sep. 2009;53(11):3735–45.
[https://doi.org/10.1016/j.csda.2009.04.009.](https://doi.org/10.1016/j.csda.2009.04.009)

[52] Schocken DD, Arrieta MI, Leaverton PE, Ross EA. Prevalence and mortality rate of
congestive heart failure in the United States. J Am Coll Cardiol Aug. 1992;20(2):
[301–6. https://doi.org/10.1016/0735-1097(92)90094-4.](https://doi.org/10.1016/0735-1097(92)90094-4)

[53] Rengo F, et al. Congestive heart failure in the elderly. Arch Gerontol Geriatr Nov.
[1996;23(3):201–23. https://doi.org/10.1016/S0167-4943(96)00734-0.](https://doi.org/10.1016/S0167-4943(96)00734-0)

[[54] Keating GM, Goa KL. Nesiritide. Drugs Jan. 2003;63(1):47–70. https://doi.org/](https://doi.org/10.2165/00003495-200363010-00004)
[10.2165/00003495-200363010-00004.](https://doi.org/10.2165/00003495-200363010-00004)

[55] Colucci WS. Nesiritide for the treatment of decompensated heart failure. J Card
[Fail Mar. 2001;7(1):92–100. https://doi.org/10.1054/jcaf.2001.22999.](https://doi.org/10.1054/jcaf.2001.22999)

[56] Mahle WT, Cuadrado AR, Kirshbom PM, Kanter KR, Simsic JM. Nesiritide in infants
and children with congestive heart failure. Pediatr Crit Care Med J Soc Crit Care
[Med World Fed Pediatr Intensive Crit Care Soc Sep. 2005;6(5):543–6. https://doi.](https://doi.org/10.1097/01.pcc.0000164634.58297.9a)

[[57] Malm-Erjeforg/10.1097/01.pcc.0000164634.58297.9aalt M, et al. Metabolism and excretion of the once-daily human ¨](https://doi.org/10.1097/01.pcc.0000164634.58297.9a) .
glucagon-like peptide-1 analog liraglutide in healthy male subjects and its in vitro



degradation by dipeptidyl peptidase IV and neutral endopeptidase. Drug Metab
[Dispos Biol Fate Chem Nov. 2010;38(11):1944–53. https://doi.org/10.1124/](https://doi.org/10.1124/dmd.110.034066)
[dmd.110.034066.](https://doi.org/10.1124/dmd.110.034066)

[58] Russell-Jones D. Molecular, pharmacological and clinical aspects of liraglutide, a
once-daily human GLP-1 analogue. Mol Cell Endocrinol Jan. 2009;297(1):137–40.
[https://doi.org/10.1016/j.mce.2008.11.018.](https://doi.org/10.1016/j.mce.2008.11.018)

[59] Arturi F, et al. Liraglutide improves cardiac function in patients with type 2
[diabetes and chronic heart failure. Endocrine Sep. 2017;57(3):464–73. https://doi.](https://doi.org/10.1007/s12020-016-1166-4)
[org/10.1007/s12020-016-1166-4.](https://doi.org/10.1007/s12020-016-1166-4)

[60] Scott DL, Wolfe F, Huizinga TW. Rheumatoid arthritis. Lancet Sep. 2010;376
[(9746):1094–108. https://doi.org/10.1016/S0140-6736(10)60826-4.](https://doi.org/10.1016/S0140-6736(10)60826-4)

[61] Littlejohn EA, Monrad SU. Early diagnosis and treatment of rheumatoid arthritis.
[Prim Care Clin Off Pract Jun. 2018;45(2):237–55. https://doi.org/10.1016/j.](https://doi.org/10.1016/j.pop.2018.02.010)
[pop.2018.02.010.](https://doi.org/10.1016/j.pop.2018.02.010)

[62] Hewitson, DeBroe, McBride, Milne. Leflunomide and rheumatoid arthritis: a
systematic review of effectiveness, safety and cost implications. J Clin Pharm Ther
[2000;25(4):295–302. https://doi.org/10.1046/j.1365-2710.2000.00296.x.](https://doi.org/10.1046/j.1365-2710.2000.00296.x)

[63] Sanders S, Harisdangkul V. Leflunomide for the treatment of rheumatoid arthritis
[and autoimmunity. Am J Med Sci Apr. 2002;323(4):190–3. https://doi.org/](https://doi.org/10.1097/00000441-200204000-00004)
[10.1097/00000441-200204000-00004.](https://doi.org/10.1097/00000441-200204000-00004)

[64] Behrens F, Koehm M, Burkhardt H. Update 2011: leflunomide in rheumatoid
arthritis – strengths and weaknesses. Curr Opin Rheumatol May 2011;23(3):282.
[https://doi.org/10.1097/BOR.0b013e328344fddb.](https://doi.org/10.1097/BOR.0b013e328344fddb)

[[65] Williamson RA, et al. Dihydroorotate dehydrogenase is a target for the biological](http://refhub.elsevier.com/S0933-3657(23)00201-4/rf0320)
[effects of leflunomide. Transplant Proc Dec. 1996;28(6):3088–91.](http://refhub.elsevier.com/S0933-3657(23)00201-4/rf0320)

[66] Breedveld FC, Dayer J-M. Leflunomide: mode of action in the treatment of
[rheumatoid arthritis. Ann Rheum Dis Nov. 2000;59(11):841–9. https://doi.org/](https://doi.org/10.1136/ard.59.11.841)
[10.1136/ard.59.11.841.](https://doi.org/10.1136/ard.59.11.841)

[67] Fox RI, et al. Mechanism of action for Leflunomide in rheumatoid arthritis. Clin
[Immunol Dec. 1999;93(3):198–208. https://doi.org/10.1006/clim.1999.4777.](https://doi.org/10.1006/clim.1999.4777)

[68] Ajayi S, et al. Ruxolitinib. In: Martens UM, editor. Small molecules in hematology.
Recent Results in Cancer Research. Cham: Springer International Publishing; 2018.
[p. 119–32. https://doi.org/10.1007/978-3-319-91439-8_6.](https://doi.org/10.1007/978-3-319-91439-8_6)

[69] McLornan DP, Pope JE, Gotlib J, Harrison CN. Current and future status of JAK
[inhibitors. The Lancet Aug. 2021;398(10302):803–16. https://doi.org/10.1016/](https://doi.org/10.1016/S0140-6736(21)00438-4)
[S0140-6736(21)00438-4.](https://doi.org/10.1016/S0140-6736(21)00438-4)

[70] Igaz P, Toth S, Falus A. Biological and clinical significance of the JAK-STAT ´
pathway; lessons from knockout mice. Inflamm Res Sep. 2001;50(9):435–41.
[https://doi.org/10.1007/PL00000267.](https://doi.org/10.1007/PL00000267)

[71] McLornan DP, Khan AA, Harrison CN. Immunological consequences of JAK
[inhibition: friend or foe? Curr Hematol Malig Rep Dec. 2015;10(4):370–9. https://](https://doi.org/10.1007/s11899-015-0284-z)
[doi.org/10.1007/s11899-015-0284-z.](https://doi.org/10.1007/s11899-015-0284-z)

[72] Baldini C, Moriconi FR, Galimberti S, Libby P, De Caterina R. The JAK–STAT
pathway: an emerging target for cardiovascular disease in rheumatoid arthritis and
[myeloproliferative neoplasms. Eur Heart J Nov. 2021;42(42):4389–400. https://](https://doi.org/10.1093/eurheartj/ehab447)
[doi.org/10.1093/eurheartj/ehab447.](https://doi.org/10.1093/eurheartj/ehab447)

[73] Yamaoka K. Janus kinase inhibitors for rheumatoid arthritis. Curr Opin Chem Biol
[Jun. 2016;32:29–33. https://doi.org/10.1016/j.cbpa.2016.03.006.](https://doi.org/10.1016/j.cbpa.2016.03.006)

[74] Baker KF, Isaacs JD. Novel therapies for immune-mediated inflammatory diseases:
what can we learn from their use in rheumatoid arthritis, spondyloarthritis,
systemic lupus erythematosus, psoriasis, Crohn’s disease and ulcerative colitis?
[Ann Rheum Dis Feb. 2018;77(2):175–87. https://doi.org/10.1136/annrheumdis-](https://doi.org/10.1136/annrheumdis-2017-211555)
[2017-211555.](https://doi.org/10.1136/annrheumdis-2017-211555)

[75] Angelini J, et al. JAK-inhibitors for the treatment of rheumatoid arthritis: a focus
on the present and an outlook on the future. Biomolecules Jul. 2020;10(7):7.
[https://doi.org/10.3390/biom10071002.](https://doi.org/10.3390/biom10071002)

[76] Cuspidi C, Sala C, Negri F, Mancia G, Morganti A. Prevalence of left-ventricular
hypertrophy in hypertension: an updated review of echocardiographic studies.
[J Hum Hypertens Jun. 2012;26(6):6. https://doi.org/10.1038/jhh.2011.104.](https://doi.org/10.1038/jhh.2011.104)

[77] Díez J, Frohlich ED. A translational approach to hypertensive heart disease.
[Hypertension Jan. 2010;55(1):1–8. https://doi.org/10.1161/](https://doi.org/10.1161/HYPERTENSIONAHA.109.141887)
[HYPERTENSIONAHA.109.141887.](https://doi.org/10.1161/HYPERTENSIONAHA.109.141887)

[78] Nwabuo CC, Vasan RS. Pathophysiology of hypertensive heart disease: beyond left
[ventricular hypertrophy. Curr Hypertens Rep Feb. 2020;22(2):11. https://doi.org/](https://doi.org/10.1007/s11906-020-1017-9)
[10.1007/s11906-020-1017-9.](https://doi.org/10.1007/s11906-020-1017-9)

[79] Singh JSS, Burrell LM, Cherif M, Squire IB, Clark AL, Lang CC. Sacubitril/valsartan:
[beyond natriuretic peptides. Heart Oct. 2017;103(20):1569–77. https://doi.org/](https://doi.org/10.1136/heartjnl-2017-311295)
[10.1136/heartjnl-2017-311295.](https://doi.org/10.1136/heartjnl-2017-311295)

[80] Imamura T, Kinugawa K. Effect of add-on sacubitril/valsartan on the left
ventricular hypertrophy of a patient with hypertension. J Int Med Res Nov. 2022;
[50(11). https://doi.org/10.1177/03000605221138480. p. 03000605221138480.](https://doi.org/10.1177/03000605221138480)

[81] Singh JS, Lang CC. Angiotensin receptor-neprilysin inhibitors: clinical potential in
[heart failure and beyond. Vasc Health Risk Manag Jun. 2015;11:283–95. https://](https://doi.org/10.2147/VHRM.S55630)
[doi.org/10.2147/VHRM.S55630.](https://doi.org/10.2147/VHRM.S55630)

[82] Cohn JN, Tognoni G. A randomized trial of the angiotensin-receptor blocker
valsartan in chronic heart failure. N Engl J Med Dec. 2001;345(23):1667–75.
[https://doi.org/10.1056/NEJMoa010713.](https://doi.org/10.1056/NEJMoa010713)



9


