_Bioinformatics_, 2024, **40(7)**, btae349
https://doi.org/10.1093/bioinformatics/btae349
Advance Access Publication Date: 24 June 2024

**Original Paper**

## Data and text mining

# **HGTDR: Advancing drug repurposing with heterogeneous** **graph transformers**


**Ali Gharizadeh** **1** **, Karim Abbasi** **1** **, Amin Ghareyazi** **1** **, Mohammad R.K. Mofrad** **2,** � **,**
**Hamid R. Rabiee** **1,** �


1 Department of Computer Engineering, Sharif University of Technology, Tehran, P.O. Box 11155-9517, Iran
2 Departments of Bioengineering and Mechanical Engineering, University of California, Berkeley, CA, P.O. Box 94720-1740, United States
�Corresponding authors. Departments of Bioengineering and Mechanical Engineering, University of California, Berkeley, CA, P.O. Box 94720-1740,United
States. E-mail: mofrad@berkeley.edu (M.R.K.M.); Department of Computer Engineering, Sharif University of Technology, Tehran,P.O. Box 11155-9517, Iran.
E-mail: rabiee@sharif.edu (H.R.R.)

Associate Editor: Zhiyong Lu


**Abstract**


**Motivation:** Drug repurposing is a viable solution for reducing the time and cost associated with drug development. However, thus far, the pro­
posed drug repurposing approaches still need to meet expectations. Therefore, it is crucial to offer a systematic approach for drug repurposing
to achieve cost savings and enhance human lives. In recent years, using biological network-based methods for drug repurposing has generated
promising results. Nevertheless, these methods have limitations. Primarily, the scope of these methods is generally limited concerning the size
and variety of data they can effectively handle. Another issue arises from the treatment of heterogeneous data, which needs to be addressed or
converted into homogeneous data, leading to a loss of information. A significant drawback is that most of these approaches lack end-to-end
functionality, necessitating manual implementation and expert knowledge in certain stages.


**Results:** We propose a new solution, Heterogeneous Graph Transformer for Drug Repurposing (HGTDR), to address the challenges associated
with drug repurposing. HGTDR is a three-step approach for knowledge graph-based drug repurposing: (1) constructing a heterogeneous knowl­
edge graph, (2) utilizing a heterogeneous graph transformer network, and (3) computing relationship scores using a fully connected network. By
leveraging HGTDR, users gain the ability to manipulate input graphs, extract information from diverse entities, and obtain their desired output.
In the evaluation step, we demonstrate that HGTDR performs comparably to previous methods. Furthermore, we review medical studies to val­
idate our method’s top 10 drug repurposing suggestions, which have exhibited promising results. We also demonstrated HGTDR’s capability to
predict other types of relations through numerical and experimental validation, such as drug–protein and disease–protein inter-relations.


**Availability and implementation:** [The source code and data are available at https://github.com/bcb-sut/HGTDR and http://git.dml.ir/BCB/HGTDR](https://github.com/bcb-sut/HGTDR)



**1 Introduction**


The escalating costs and protracted development periods for
new pharmaceuticals present significant challenges to the in­
dustry (Zhang _et al._ 2020, Pan _et al._ 2022). Drug repurpos­
ing, defined as identifying new uses for approved drugs, has
emerged as a viable solution to these issues (Zhou _et al._
2020). Employing this approach mitigates the risk and
reduces the expenses and duration associated with the devel­
opment of drugs (Singh _et al._ 2020, Talevi and Bellera 2020).
To facilitate drug repurposing, both systematic experimental
and computational techniques have been devised
(Pushpakom _et al._ 2019, Talevi and Bellera 2020). Among
computational strategies, network-based methods have
shown encouraging outcomes, encompassing three primary
stages (Zeng _et al._ 2019, Zhu _et al._ 2020, Che _et al._ 2021, Jin
_et al._ 2021, Yu _et al._ 2021): (1) network construction, (2) fea­
ture extraction, and (3) link prediction. A biological network
or knowledge graph is initially constructed, incorporating
various biological entities and their interconnections.
Subsequently, the nodes within this graph are characterized
through embeddings using graph neural networks (GNNs).



Finally, these embeddings are applied within algorithms, such
as a matrix completion or a neural network, to predict links
in the knowledge graph.

Zhu _et al._ (2020) integrated six distinct databases to de­
velop a comprehensive knowledge graph, which comprises
five types of entities and nine varieties of relationships. To
characterize the graph, they employed a combination of four
path-based representations, guided by predefined meta-paths,
alongside three embedding-based representations.
Subsequently, they harnessed three machine learning techni­
ques—support vector machines, decision trees, and random
forests—to discern new potential drug applications. The
strength of this study lies in its integration of multiple data
sources to account for the interplay between various types of
data. Nevertheless, the reliance on manually selected metapaths may curtail the automation of the method, necessitat­
ing specific meta-path selections for each knowledge graph
constructed.

The layer attention graph convolutional network
(LAGCN) method, as proposed by Yu _et al._ (2021), estab­
lishes a drug–drug similarity network by applying the Jaccard
index to various drug-related datasets, including targets,



**Received:** 6 September 2023; **Revised:** 20 April 2024; **Editorial Decision:** 28 April 2024; **Accepted:** 23 June 2024
# The Author(s) 2024. Published by Oxford University Press. This is an Open Access article distributed under the terms of the Creative Commons Attribution
License (https://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original
work is properly cited.


**2** Gharizadeh _et al._



pathways, and substructures, all of which are sourced from
the DrugBank database. Concurrently, it formulates a dis­
ease–disease similarity network, employing semantic similar­
ity scores derived from MeSH descriptors structured as a
hierarchical directed acyclic graph (DAG) based on the prem­
ise that diseases with more shared ancestral traits exhibit
greater similarity. The integration of these two similarity net­
works, along with the drug–disease relationships, results in
the construction of the final input graph. This graph is fur­
ther processed as nodes are embedded using an LAGCN, fol­
lowed by the employment of a bilinear decoder for link
prediction. While LAGCN leverages diverse data sources, it
simplifies these into uniform similarity matrices, a process
that may exclude certain informative relationships.
Additionally, the method needs more scalability, limiting its
applicability to vast graphs.

The AttGCN-DDI model, described by Che _et al._ (2021),
assembles a concise knowledge graph from six distinct data­
bases, delineating five entities and seven relationship types.
This model hypothesizes that a drug can treat a disease if it
can be connected to the disease by a path of less than four
steps. Focusing on COVID-19, it incorporates drugs and dis­
eases that have the shortest path to COVID-19 of fewer than
four links into the graph. After this addition, the relationships
between these newly added nodes are integrated. This
method of input selection necessitates expert knowledge in
the domain, which, in turn, diminishes the automation aspect
of the process. Topological features are extracted as node
embeddings using Att-GCN, a method that treats all edges
uniformly, thus not accounting for the heterogeneity of the
edges. Additionally, a projection matrix is developed to re­
construct the drug–disease interaction matrix. Similar to
other discussed methods, this approach also suffers from a
lack of scalability.

DeepDR (Zeng _et al._ 2019) employs a network of nine
databases to extract drug features. Initially, these networks
are transformed into homogeneous similarity networks using
techniques such as the Jaccard similarity. The method then
generates probabilistic co-occurrence (PCO) matrices for
each network through random surfing. To address the inher­
ent sparsity of some networks, it constructs shifted positive
pointwise mutual information (PPMI) matrices from the
PCO matrices. However, these preprocessing stages may re­
sult in losing original network information. A multi-modal
deep autoencoder is employed to extract features from drug
nodes. DeepDR considers drug–disease associations as inputs
and drug features as auxiliary information. Subsequently, it
applies a collective variational autoencoder (cVAE), as intro­
duced by Chen and de Rijke (2018), to predict drug–disease
associations.

As detailed by Jin _et al._ (2021), HeTDR utilizes an input
network resembling the one used by DeepDR. It adopts iden­
tical procedures for network conversion to homogeneous net­
works, random surfing, and the creation of shifted PPMI
matrices. A distinctive feature of HeTDR is its application of
similarity network fusion to integrate all drug-related infor­
mation into a singular network, thereby providing a compre­
hensive biological perspective on drugs.

The model employs a sparse autoencoder, as described in
Ng (2011), to extract drug features. Additionally, it utilizes
fine-tuned BioBERT (Lee _et al._ 2020) to derive disease fea­
tures. Consequently, each new input requires the fine-tuning
of BioBERT. Ultimately, the GATNE-I model, introduced by



Cen _et al._ (2019), is applied to predict drug–disease
interactions.

Several contemporary studies have made efforts to utilize
heterogeneous data from multiple databases. Nevertheless,
these approaches come with constraints. Some, like those
proposed by Zhu _et al._ (2020), need more scalability and rely
on limited datasets. While methods such as those by Zeng _et_
_al._ (2020) employ different networks, they need to synthesize
a higher-order representation that consolidates information
from these varied sources. It is crucial to recognize that inte­
grating networks can reveal insights that may need to be dis­
cernible when networks are analyzed in isolation.
Furthermore, certain techniques (Zeng _et al._ 2019, Yu _et al._
2021) transform heterogeneous networks into homogenized
similarity networks, diluting the input data’s richness. Most
of these strategies also do not incorporate non-network data.
Some approaches (Zeng _et al._ 2019, Jin _et al._ 2021) restrict
the inclusion of certain relationship types in the knowledge
graph, thereby preventing the use of pivotal data, such as
protein–protein interactions. These methods predominantly
center on drug-related data and overlook the extraction of
network features pertinent to diseases, thus presenting con­
siderable hurdles to drug repurposing endeavors. Moreover,
certain methods necessitate hands-on implementation, in­
cluding the designation of meta-paths as specified by Zhu
_et al._ (2020), necessitating specialized domain knowledge and
constraining the potential for input modification. The ab­
sence of end-to-end model structures further compromises
the automation of these methods and amplifies the reliance
on domain expertise.

To address the constraints outlined previously, we intro­
duce a model endowed with the following features:


� It distinguishes between node and edge types using hetero­

geneous graphs.
� The model accommodates an unrestricted variety of
graph data types, enabling the use of any node and edge
categories in the input graph.
� Scalability is a core attribute of the method.
� The model operates autonomously, obviating the need for
domain-specific knowledge during setup.
� It is an end-to-end model that integrates the task of link
prediction within the feature extraction process
from nodes.


Our model, Heterogeneous Graph Transformer for Drug
Repurposing (HGTDR), parallels prior models by encompass­
ing a three-step approach. In our research, PrimeKG (Chandak
_et al._ 2023) serves as the foundation of our knowledge graph,
complemented by the initial embeddings from BioBERT (Lee
_et al._ 2020) and ChemBERTa (Chithrananda _et al._ 2020). To
extract node features, we utilize the heterogeneous graph
transformer (HGT) technique (Hu _et al._ 2020) as our GNN.
Finally, a fully connected network predicts the relationship
score within the graph between a drug and a disease.

PrimeKG significantly enhances the initial phase of our ap­
proach, broadening the scope of drugs and diseases that can
be explored beyond previous capabilities. Table 1 outlines a
comparative analysis of the coverage of drugs and diseases by
this method and others.

Several studies have developed knowledge graphs in drug
repurposing, including Hetionet (Himmelstein _et al._ 2017)
and DRKG (Ioannidis _et al._ 2020). Hetionet features 755


HGTDR **3**



**Table 1.** Comparison of the number of drugs and diseases using

different methods.


**Method** **Drug** **Disease**


ATTGCN-DDI 1470 752

LAGCN 269 598

HetDR 1519 1229

DeepDDR 1519 1229
HGTDR 1801 1363


indication relations, linking 387 drugs to 77 diseases. DRKG
presents 83 895 compound–disease relationships, which may
not directly pertain to indications. Notably, most of these
relationships, specifically 77 782, are derived from the GNBR
database (Percha and Altman 2018), which utilizes textprocessing techniques. However, this source may not reliably
inform drug repurposing efforts due to its reliance on
text extraction.

For the extraction of features from heterogeneous graphs,
various methodologies have been proposed, such as HAN
(Wang _et al._ 2019), HetGNN (Zhang _et al._ 2019), MHGCN
(Yu _et al._ 2022), and HGT (Hu _et al._ 2020). HAN employs
node-level and semantic-level attention mechanisms within
heterogeneous graphs to discern the significance of nodes and
their meta-path-based neighbors and the relevance of different
meta-paths. HetGNN introduces a strategy for capturing the
structural and content heterogeneity. It utilizes a random walk
approach to sample heterogeneous neighbors that are strongly
correlated, grouping them by node type. Subsequently, node
embeddings are generated through a dual-module neural net­
work that accounts for structural and content diversity.
MHGCN aims to autonomously identify valuable heteroge­
neous meta-path interactions across varying lengths in multi­
plex heterogeneous networks through multi-layer
convolutional aggregation, thereby producing node embed­
dings that merge structural and semantic information. Despite
its innovations, MHGCN’s scalability remains a challenge.
HGT distinguishes itself by interpreting heterogeneous nodes
and edges through meta-relations, utilizing node-specific and
edge-specific weights. This approach enables identifying com­
mon and unique patterns across different relationships without
predefined meta-paths, thus bypassing concerns about identify­
ing critical meta-paths within the biological context.

According to the study by Hu _et al._ (2020), HGT outper­
forms HAN and HetGNN, further validated by its scalability
advantage. Therefore, we have selected HGT for node feature
extraction in our model. While the original study by Hu _et al._
highlighted HAN’s effectiveness following HGT, we opted to
verify HGT’s superiority by substituting our feature extrac­
tion layers with HAN layers. The results in Table 4 affirm
that HGT surpasses HAN in our specific task.

The rest of this article is organized as follows: Section 2 is
dedicated to a comprehensive exposition of our methodology.
Section 3 presents the experiments conducted and the corre­
sponding results obtained through the application of our
method. The paper concludes in Section 4, where we provide
concluding remarks and outline directions for future research.


**2 Materials and methods**


This section begins with the presentation of the problem for­
mulation, followed by a detailed explanation of the three
steps involved in our proposed method.



2.1 Problem formulation

Let _G_ be a heterogeneous graph whose nodes are biological
entities whose edges are different relationships among the en­
tities. We have drug and disease nodes and indication edges


in the graph. We define all indications as _I_ ¼


_N_
�� _Dr_ _[i]_ _; Di_ _[i]_ � _; Y_ _[i]_ � _i_ ¼1 [where ] � _[Dr]_ _[i]_ _[;]_ _[Di]_ _[i]_ � is the i [th ] indication
edge in the graph; _Y_ _[i ]_ is the label, which is 1 for all indica­
tions, and _N_ is the number of all indications. During the
training step, we put 20% of indications ( _I_ _input_ ) in the graph
_G_ and tried to predict the label of the rest of the indications
( _I_ _pos_ ) plus random negative samples, which are defined as



_N_ _pos_
_Ineg_ ¼ �� _Dr_ _[i]_ _; Di_ _[j]_ � _;_ 0� 1 where _N_ _pos_ is the number of

samples of _I_ _pos_, _i_ 2 ð1 ... _N_ _Drug_ Þ, and _j_ 2 ð1 ... _N_ _Disease_ Þ. _N_ _Drug_
and _N_ _Disease_ are also the number of drugs and diseases in _G;_
respectively.


_I_ _prediction_ ¼ _I_ _pos_ þ _I_ _neg_ (1)


This prediction task can be viewed as a simulation of drug
repurposing in which our model learns to predict indications
that do not exist in the input graph. To do actual drug repur­
posing, we must put all indications in the graph _G_ and try to
predict novel indications.


2.2 Network construction

In this step, we should construct a network. Previous meth­
ods built their networks using databases such as DrugBank
(Law _et al._ 2014). There is a problem with this approach.
Domain knowledge is necessary to select the appropriate
databases. Moreover, not all relations within a database are
used in those works, raising the question of how the relations
are chosen. We address this problem by utilizing a previously
constructed knowledge graph, PrimeKG (Chandak _et al._
2023), and allowing our model to determine which entities
and relationships are relevant to the task. To achieve this
goal, HGT is suitable because it can control the overall con­
tribution of a particular node or edge type to embedding cre­
ation by using node and edge-specific weights. PrimeKG
integrates 20 high-quality resources to generate a graph with
10 node types and 30 edge types. The existing graph can also
be expanded with new data without node and edge type
restrictions. The graph is defined as _G_ ¼ ð _V; E; A; R_ Þ in
which every node _v_ 2 _V_ and every edge _e_ 2 _E_ are associated
with their type-mapping functions _τ v_ ð Þ : _V_ ! _A_ and _ϕ e_ ð Þ :
_E_ ! _R_ respectively. Also, for every edge _e_ ¼ ð _s; t_ Þ, where _s_
and _t_ are the source and target nodes of the edge, its metarelation is denoted as h _τ s_ ð Þ _; ϕ e_ ð Þ _; τ_ ð _t_ Þi.
Since previous works only contain drugs and diseases that
contribute to indication relations, we remove drug and dis­
ease nodes that do not contribute to at least one indication
edge. These nodes can increase accuracy unrealistically, in
comparison with previous works, due to the ease of predict­
ing no indications related to the drugs and diseases.
Consequently, some nodes are removed from the graph.
Tables 2 and 3 demonstrate graph statistics before and after
the removal of the nodes.


**2.2.1 Initial embeddings**
We add BioBERT and ChemBERTa embeddings to our
knowledge graph to integrate different types of information
into the graph. BioBERT is a domain-specific language repre­
sentation model pre-trained on large-scale biomedical


**4** Gharizadeh _et al._


**f** i



**Table 2.** PrimeKG node counts.


**Entity** **Count before**
**removal**


**f** i



**Count after**

**removal**


**f** i



**Removal**

**percent**


**f** i



Biological process 28 642 28 642 0
Protein 27 671 27 573 0.35

Disease 17 080 1363 92.01
Phenotype 15 311 15 082 1.49
Anatomy 14 035 14 035 0
Molecular function 11 169 11 169 0

Drug 7957 1801 77.36
Cellular component 4176 4176 0
Pathway 2516 2516 0
Exposure 818 780 4.64
**Total** **129 375** **107 137** **17.17**


**Table 3.** PrimeKG directed edge counts.


**f** i



**Table 4.** Comparison of HGT and HAN layers as feature extractors.


**Feature extraction layer** **AUROC** **AUPR**


HAN 0.925 0.918

HGT 0.944 0.946


they are embedded using ChemBERTa and added to drug
nodes. Interestingly, we can add any embedding to any of our
node types, and we do not require different node types to
have similar embeddings. This lets us add any auxiliary infor­
mation to our graph if it can be represented as an embedding.
We denote the initial embeddings of the graph as _H_ _[init]_ .

After constructing the graph, we divide indication edges
into masked and unmasked groups, with 80% of indication
edges being masked. Unmasked indications are used like
other edges as part of the input graph. However, masked
indications are used as positive samples when computing loss
function. This technique helps the model predict indications
not present in the input graph, which can act like a simula­
tion of drug repurposing.


2.3 Feature extraction

For embedding graph nodes, attention-based GNNs usually
follow the following formula for source node s and target
node t (Hu _et al._ 2020):


**f** i



**Entity** **Count**
**before**

**removal**


**f** i



**Count**

**after**

**removal**


**f** i



**Removal**

**percent**


**f** i



Anatomy–protein (present) 3 036 406 3 036 406 0
Drug–drug 2 672 628 743 328 72.18
Protein–protein 642 150 642 150 0
Disease–phenotype (positive) 300 634 24 490 91.85
Biological process–protein 289 610 289 610 0
Cellular component–protein 166 804 166 804 0
Disease–protein 160 822 99 232 38.29
Molecular function–protein 139 060 139 060 0
Drug–phenotype 129 568 102 736 20.70
Biological process– 105 772 105 772 0
biological process

Pathway–protein 85 292 85 292 0
Disease–disease 64 388 1772 97.24
Drug–disease (contraindication) 61 350 32 194 47.52
Drug–protein 51 306 24 848 51.56
Anatomy–protein (absent) 39 774 39 774 0
Phenotype–phenotype 37 472 37 472 0
Anatomy–anatomy 28 064 28 064 0
Molecular function– 27 148 27 148 0

molecular function


**f** i



_H_ _[l]_ ½ � _t_ _Aggregate_
8 _s_ 2 _N t_ ð Þ _;_ 8 _e_ 2 _E s_ ð _;t_ Þ


**f** i



� _Attention s_ ð _; t_ Þ _:Message s_ ð Þ� (2)


**f** i



105 772 105 772 0


**f** i



27 148 27 148 0


**f** i



Drug–disease (indication) 18 776 18 776 0
Cellular component– 9690 9690 0
cellular component

Phenotype–protein 6660 6660 0
Drug–disease (off-label use) 5136 3836 25.31
Pathway–pathway 5070 5070 0
Exposure–disease 4608 2924 36.54
Exposure–exposure 4140 4140 0
Exposure–biological process 3250 3250 0

**f** i

Exposure–protein 2424 2424 0
Disease–phenotype (negative) 2386 140 94.13
Exposure–molecular function 90 90 0
Exposure–cellular component 20 20 0
**Total** **8 100 498** **5 683 172** **29.84**


corpora. Therefore, its embeddings add information
extracted from biomedical literature to our knowledge graph.
ChemBERTa is a similar model pre-trained on SMILES
(Weininger 1988) representation of molecules that can ex­
tract their structural information. The initial embeddings are
added to the graph in the following manner. The names of all
entities, except drugs, are first obtained from PrimeKG.
BioBERT embeddings are then extracted and added to our
graph nodes. Moreover, SMILES (Weininger 1988) represen­
tations of the drugs are obtained from DrugBank. Afterward,



There are three basic operators: Attention, which estimates
the importance of each source node; Message, which extracts
the message using only the source nodes; and Aggregate,
which aggregates the neighborhood message by the atten­
tion weight.

To understand different distributions of different node and
edge types, HGT introduces a heterogeneous mutual atten­
tion mechanism to calculate a target node and all its neigh­
bors ( _s_ 2 _N_ ð _t_ Þ) mutual attention grounded by their metarelation. The attention for each edge _e_ ¼ _s_ ð _;_ _t_ Þ is defined
as follows:


**f** i



**f** i


_K_ _[i]_ ð Þ ¼ _s_ _K_ �� _Linear_ _[i]_ _τ s_ ð Þ � _[H]_ _[ l]_ ð [�] [1] Þ _s_ ½ ��

_Q_ _[i]_ ð Þ ¼ _t_ _Q_ �� _Linear_ _[i]_ _τ t_ ð Þ � _[H]_ [ð] _[l]_ [�] [1][Þ] _[ t]_ [½ �] � (3)


For the i-th attention head _ATT_ �� _head_ _[i]_ ð _s; e; t_ Þ, source
nodes that have type _τ_ ð _s_ Þ is projected into the i-th key vector
_K_ _[i]_ ð Þ _s_, with a node-specific linear projection to consider the
distribution differences of node types. Similarly, the target
node t is projected to the i-th query vector. Furthermore, un­
like the vanilla Transformer that directly computes the dot
product between query and key vectors for calculating the
similarity between key and query, HGT keeps a distinct edgespecific matrix _W_ F _[ATT]_ _e_ ð Þ [for each edge type ] [F] _[ e]_ ð Þ. Thus, the
model can capture different semantic relations even between



_Attention_ _HGT_ _s_ ð _; e; t_ Þ ¼ _Softmax_

8 _s_ 2 _N t_ ð Þ


**f** i



k _ATT_ �� _head_ _[i]_ ð _s; e; t_ Þ

� _i_ 2 1½ _; h_ � �


**f** i



_ATT_ �� _head_ _[i]_ ð _s; e; t_ Þ ¼ � _K_ _[i]_ ð Þ _s_ _W_ F _[ATT]_ _e_ ð Þ _[Q]_ _[i]_ _[ t]_ [ð Þ] _[T]_ � _:_ **f** i



m _< τ s_ ð Þ _;_ Fð _e_ Þ _;τ t_ ð Þ _>_

~~f~~ **f** i ~~f~~
~~p~~ _d_


HGTDR **5**



the same node-type pairs. Additionally, a prior tensor m is
used to denote the general significance of each meta-relation
triplet, serving as an adaptive scaling to the attention. Finally,
h attention heads are concatenated to get each node pair’s at­
tention vector. Then, for each target node t, all attention vec­
tors of neighbors _N_ ð _t_ Þ are gathered, and softmax is
conducted, making it fulfill S 8 _s_ 2 _N t_ ð Þ _Attention_ _HGT_ _s_ ð _;_
_e; t_ Þ ¼ 1 _h_ × 1 .

Similarly, meta-relations of edges are also used in message
passing process to alleviate the distribution differences of
nodes and edges of different types as follows:



_Message_ _HGT_ _s_ ð _; e; t_ Þ ¼ k
_i_ 2 1½ _; h_ �



_MSG_ �� _head_ _[i]_ ð _s; e; t_ Þ



ð Þ
_MSG_ �� _head_ _[i]_ ð _s; e; t_ Þ ¼ _M_ �� _Linear_ _[i]_ _τ s_ ð Þ � _[H]_ _[ l]_ [�] [1] _s_ ½ �� _W_ F _[MSG]_ _e_ ð Þ (4)


Source node’s ID with a linear projection _M_ �� _Linear_ _[i]_ _τ s_ ð Þ [. It ]
is then followed by a matrix _W_ F _[MSG]_ _e_ ð Þ [for incorporating the ]
edge dependency. Then, all h message heads are concatenated
for each node pair.

In the aggregation step, the attention vector is used as the
weight to average the corresponding messages from the



source nodes to get the updated target vector _H_ [^]



ð _l_ Þ
½ _t_ � as:



_H_ ~



ð _l_ Þ
_t_ ½ �¼ 8 _s_ 2 � _N t_ ð Þ � _[Attention]_ _[HGT]_ _[ s]_ ð _[;][ e][;][ t]_ Þ _:Message_ _HGT_ _s_ ð _; e; t_ Þ�



(5)


Finally, the target node’s vector is mapped back to its typespecific distribution with a linear projection _A_ �� _Linear_ _τ t_ ð Þ
followed by a non-linear activation and residual connec­
tion as:




[ð Þ] _l_

_H_ [ð] _[l]_ [Þ] ½ �¼ _t_ _σ A_ � �� _Linear_ _τ t_ ð Þ _H_ [~] _t_ ½ �� þ _H_ [ð] _[l]_ [�] [1][Þ] ½ _t_ � (6)



ChemBERTa models. Therefore, a linear layer is first defined,
which decreases the input size to 64.


_H_ _batch_ [0] [¼] _[ Dropout ReLU Linear]_ � � _[τ]_ _[ H]_ � _batch_ _[init]_ ��� (7)


where _H_ _[init]_
_batch_ [is the input feature vector containing the initial ]
embedding of batch nodes, _H_ _batch_ [0] [is the input feature vector ]
of the first HGT layer, _Linear_ _τ_ is a node-specific linear layer,
and ReLU is the rectified linear unit activation function.
Also, dropout is applied to the output with probability ¼ 0.5.

HGT layers are defined as follows:


_H_ _batch_ _[l]_ [¼] _[ HGT]_ � _H_ _batch_ [ð] _[l]_ [�] [1][Þ] _[;]_ _G_ _batch_ Þ (8)


where _G_ _batch_ is the sampled input graph, and _H_ _batch_ [ð] _[l]_ [�] [1][Þ] [and ]

_H_ _batch_ _[l]_ [are the input and output feature vectors of layer ] _[l]_ [. All ]
layers have eight HGT attention heads like the original work
on HGT (Hu _et al._ 2020). The outputs of all three layers are
concatenated and fed to the following linear layer.


_Feat_ ¼ _ReLU Linear i_ � � 2 _i_ ½ _; L_ �jj _H_ _batch_ _[i]_ �� (9)


where _Feat_ is the output feature vector for subgraph nodes.


2.4 Link prediction
Link prediction tasks are carried out with a simple two-layer
fully connected network, which receives input from a
concatenated collection of drug and disease feature vectors
derived from the HGT network. Suppose we want to check
the score of the drug _i_ and disease _j_ in our input subgraph.
We first concatenate HGT embeddings of the drug and
the disease:


_H_ _FC_ [0] [¼] _[ Feat]_ _[Drug]_ _i_ [jj] _[ Feat]_ _[Disease]_ _j_ (10)


where _Feat_ _Drug_ _i_ and _Feat_ _Disease_ _j_ are extracted features for drug
_i_ and disease _j_, respectively. The first layer’s input dimension
size is 128 (concatenation of two 64-dimensional embed­
dings), and the output dimension size is 64. Batch normaliza­
tion is used in this layer, ReLU is the activation function, and
the dropout rate is 0.2.


_H_ _FC_ [1] [¼] _[ Dropout ReLU BatchNorm Linear]_ � � � [128][ !][ 64] _[ H]_ � _FC_ [0] ����


(11)


The second layer is as follows:


_Score_ ¼ _σ Linear_ � 64 ! 1 _H_ � _FC_ [1] �� (12)


where _σ_ is the sigmoid function, and _Score_ is the output of a
fully connected network, which is the indication score be­
tween drug _i_ and disease _j_ . Figure 1 demonstrates each step of
the method in detail.


2.5 Training
The graph is generated, as explained in Section 2.2. During
training, subgraphs are generated by the HGSampling algo­
rithm using drugs as initial sample nodes. The batch size of



As described, the method relies on using meta-relation to
parameterize the weight matrices separately, enabling the
model to distinguish the operators for different relations and
thus be more capable of handling the distribution differences
in heterogeneous graphs.


**2.3.1 HGSampling**
Using the entire input graph in the training process can make
any method unscalable. Therefore, a sampling method is
needed to make the method scalable. Consequently, we use
HGSampling (Hu _et al._ 2020) to get mini-batches for train­
ing. We denote a subgraph generated by the sampler as
_G_ _batch_ . The method keeps a similar number of nodes and
edges for each type and keeps the subgraph dense to minimize
information loss and reduce sample variance. HGSampling
has been used in large-scale graphs and applied to a graph
with 178 million nodes and 2.2 billion edges (Hu _et al._
2020). As a result, there will not be any problem regarding
the size of the graph since the current large biological graphs
have fewer than 10 million edges (Himmelstein _et al._ 2017,
Zhang _et al._ 2021, Chandak _et al._ 2023).


**2.3.2 Model architecture**

Our model has three layers of HGT ( _L_ ¼ 3) whose input and
output feature vector sizes are 64. The initial embeddings are
768 in size, which are the outputs of the BioBERT and


**6** Gharizadeh _et al._


**Figure 1.** (1) PrimeKG is obtained as the knowledge graph, (2) drug SMILES representations are extracted from DrugBank, and entity names are
extracted from PrimeKG, (3) initial embeddings are extracted using BioBERT and ChemBERTa, (4) node embeddings are computed using HGT layers, (5)
final embeddings are obtained with applying a fully connected layer on a concatenation of node embeddings of different layers, and (6) drug-disease
relation score is computed using a 2-layer FC network.



sampling is 164 because we have 1801 drugs in our graph,
and we want our subgraphs to be similar in size. And other
batch sizes can let one batch have very few drugs.
HGSampling produces a sampled sub-graph of L depth from
the initial nodes when the sampling depth is L. As we employ
three HGT layers in the model, embeddings are constructed
from neighboring nodes within a distance of three nodes.



Therefore, we chose to have a sampling depth of three. As
demonstrated in Table 6, similar results are obtained when
different sampling depths (L ¼ 2, 3, or 4) are used; however,
sampling depths of 2 and 3 perform slightly better than
depths of 4. Also, the number of nodes sampled in each itera­
tion for each node type is 512. The HGT network and linkpredicting network are trained as an end-to-end model.


HGTDR **7**



We use the binary cross-entropy function as the model’s
loss function on _I_ _prediction_ _batch_ which contains the positive and
negative samples for _G_ _batch_ .



_Loss_ ¼ � [1]
_n_



X _ni_



^

[ ^] � _[Y]_ _[i]_ � þ 1ð � _Y_ _i_ Þ _:_ logð1 � _Y_ _i_ Þ



_ni_ ¼1 _[Y]_ _[i]_ _[ :]_ [ log][ ^] � _[Y]_ _[i]_



(13)


where _n_ is the number of all samples, _Y_ _i_ is the label of the i [th ]

sample, and _Y_ [^] _i_ is the output of the model for the i [th ] sample.
Positive samples are masked indications of the subgraph, and
negative samples are randomly selected from all possible
edges between the subgraph’s drugs and diseases, which are
not actually in the subgraph’s masked or unmasked
indications.

The model has been optimized via the AdamW (Loshchilov
and Hutter 2019) optimizer with a cosine annealing learning
rate scheduler (Loshchilov and Hutter 2017). The model is
trained for 300 epochs.


**3 Experiments**


Our experiments and their results are explained in this sec­
tion. We compared HGTDR with LAGCN (Yu _et al._ 2021),
DeepDR (Zeng _et al._ 2019), and HeTDR (Jin _et al._ 2021) us­
ing 5-fold cross-validation to evaluate the proposed method.
The results for all methods were regenerated for this work.
AUROC and AUPR are used as evaluation metrics
for comparison.

Hyperparameter optimization is also done on the percent
of masked indications (20%, 50%, 80%), number of HGT
layers (3, 4), and layer feature space dimension (32, 64). The
model is run using an NVIDIA GeForce GTX 1080 Ti GPU
with a memory size of 11GB.

As described in the introduction, we compared the effec­
tiveness of HGT layers with HAN layers to show their effec­
tiveness. Furthermore, to show the result’s insensitivity to the
sampling method, we tested different sampling depths and re­
peated validation with different samplings.

In Section 2.2, we claim that drug and disease nodes should
be removed to make our data comparable to previous works,
and using those removed nodes can increase our method’s per­
formance unrealistically. In an effort to validate this fact, we
carried out an experiment where the nodes were not removed.

We conducted another experiment to compare HeTDR
and HGTDR’s performance for new diseases, where litera­
ture and indication information are limited. In this experi­
ment, we used 90% of diseases for training and 10% for
testing. We removed BioBERT embeddings for diseases of
each method, and only one indication from every validation
disease was used in training data. Since one of HeTDR’s limi­
tations is that it does not support validating diseases that do
not indicate training data.

Furthermore, we ran another experiment to check the ro­
bustness of our model to different input types. In this experi­
ment, we ran our model five times, and each time, we
removed one type of relation in the input graph (five edge
types with most instances are used).

In addition, to investigate the model’s ability to extract the
necessary information for its task, we use it without any
modifications to predict other relations instead of indica­
tions. Input data and model are both the same in
this experiment.



3.1 Results

For cross-validation, we divided indications into five subsets.
Four subsets were used for training the model, and one subset
was used for validation in each fold. For each metric, the av­
erage of 5-fold is reported. Figure 2 shows the comparison
results of the four methods. Only HeTDR performs slightly
better than our method.

We showed the effectiveness of HGT layers by swapping
them with HAN layers. Table 4 demonstrates the higher per­
formance of HGT layers in comparison with HAN layers.

Additionally, we demonstrated that different samplings us­
ing the HGSampling algorithm do not lead to much differ­
ence in prediction results. Table 5 shows validation results
for 10 repeats on one cross-validation fold with differ­
ent samplings.

Moreover, Table 6 shows that the method is not sensitive
to sampling depth and performs well when using 2 or 3 sam­
pling depths.

Table 7 investigates the effect of drug and disease nodes’
removal, as discussed in Section 2.2. We can see that not re­
moving those nodes increases the method’s performance.
However, since other methods do not use such drug and dis­
ease nodes where there is no indication related, we decided
not to use those nodes in our main method.

In a second experiment designed for new diseases, we com­
pared HGTDR and HeTDR. Table 8 demonstrates the results
of the two methods with the data limitations of new diseases.

As explained in Section 3, cross-validation was repeated
for each of the five robustness testing experiments. Table 9
demonstrates the results of robustness testing of HGTDR to
input variation. There is no significant difference in the
results of the six types of inputs, which shows that HGTDR
is robust to relation removals. Notably, more than half of the
edges of our original graph are in anatomy-protein type, and
removing the type does not have much effect on the results.
Any relationship contains valuable information for drug
repurposing or can be considered noise in the input graph.
Using this method, we do not need to know which relation
type is proper and which is noise for our task.


**3.1.1 Novel relation prediction**
Although common metrics like AUROC and AUPR show the
general correctness of these methods, they are not entirely re­
liable since the ultimate goal of drug repurposing is to find
false positive examples. Hence, a method with an accuracy of
100% would not be helpful because it cannot suggest novel
drug repurposing candidates. To predict novel indications,
we trained the model with all indications and made 110
batches (10 sets of batches with batch size 164) from the in­
put graph with positive and negative samples. False positive
predictions can be inferred as drug repurposing candidates.
We further investigated medical literature to find evidence for
novel relations predicted by HGTDR. Table 10 demonstrates
discovered evidence from the literature for 10 highly scored
false positive predictions by HGTDR.


**3.1.2 Other applications**
Even though our primary goal in developing this method was
to predict drug–disease links, its generality enables us to use
it to predict any relationship in the graph. Thus, we used it to
predict other relation types. Table 11 shows evaluation
results for those relations.


**8** Gharizadeh _et al._


**Figure 2.** Comparing AUROC and AUPR metrics for LAGCN, DeepDR, HeTDR, and HGTDR.



**Table 5.** Validation results for 10 repeats on 1-fold of cross-validation.


**AUROC** **AUPR**


0.951 0.954

0.950 0.952

0.949 0.950

0.951 0.953

0.950 0.951

0.950 0.953

0.951 0.954

0.945 0.947

0.950 0.951

0.954 0.958


We also investigated other research to evaluate these pre­
dictions experimentally. Tables 12 and 13 show the experi­
mental validation for disease–protein and drug–protein
relations. Medical literature supports five out of five disease–
protein relation predictions and three out of five drug–pro­
tein relation predictions. Experimental and numerical



**Table 6.** Comparing HGSampling with different sampling depths.


**Sampling depth** **AUROC** **AUPR**


2 0.946 0.945

3 0.944 0.946

4 0.943 0.944


evidence suggests that the model can extract the information
it needs to complete its task without changing the inputs or
providing additional information.

To determine which information from the graph is most
relevant to each downstream task, we used meta-relation at­
tention to identify the most important relations for each task.
[Supplementary Tables S1–S7, in](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btae349#supplementary-data) [supplementary materials,](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btae349#supplementary-data)
present the top five meta-relations for all three layers in our
model for different tasks. For indication prediction, we can
see that other relations between drugs and diseases, like con­
traindication and off-label use, are among the important
meta-relations, which is logical. Furthermore, we can see that


HGTDR **9**



**Table 7.** HGTDR performance compared with when no node is removed
from the input graph.


**Method** **AUROC** **AUPR**


HGTDR 0.944 0.946

HGTDR without node removal 0.979 0.970


**Table 8.** HGTDR and HeTDR results without literature and indication

information.


**Method** **AUPR** **AUROC**


HGTDR 0.859 0.871

HeTDR 0.779 0.782


**Table 9.** The method results when a relation type is removed from the
input graph.


**Removed relation** **AUPR** **AUROC**


Nothing removed 0.946 0.944
Anatomy–protein(present) 0.949 0.949
Drug–drug 0.945 0.944
Protein–protein 0.944 0.943
Biological process–protein 0.940 0.942
Cellular component–protein 0.939 0.939


**Table 10.** Experimental evidence of novel predicted relations.


**Drug** **Disease** **Evidence**



Betamethasone Lichen planus (Cawson 1968)
Dactinomycin Acute myeloid leukemia (Falini _et al._ 2015)
Vincristine Ewing sarcoma (Wagner _et al._ 2017)
Paclitaxel Classic Hodgkin lymphoma (Sinha _et al._ 2013)
Prednisolone Trichinellosis (Shimoni _et al._ 2007)
Dexamethasone Blastomycosis No evidence
Vinblastine Malignant Sertoli-Leydig (Finan _et al._ 1992)
Cell tumor of the ovary

Escitalopram Social phobia (Pelissolo 2008)
Dactinomycin Plasmablastic lymphoma No evidence
Cetirizine Urticaria (Kalivas _et al._ 1990)


**Table 11.** Evaluation results when predicting relations other

than indication.


**Relation** **AUPR** **AUROC**


Disease–protein 0.912 0.926
Drug–protein 0.951 0.948
Pathway–protein 0.951 0.953
Drug–phenotype 0.885 0.911
Biological process–protein 0.889 0.888
Protein–protein 0.881 0.887


most of the important meta-relations in all tables have a com­
mon node type with the meta-path that is being predicted.
For example, in protein–protein prediction, all important
meta-paths have node types of protein in them.


**4 Discussion**


Drug repurposing is a promising strategy to overcome current
drug development limitations such as high failure risk and



**Table 12.** Experimental evidence of novel disease–protein relations.


**Protein** **Disease** **Evidence**


VEGFA Lynch syndrome (Tjalma _et al._ 2016)
CCND1 Acute lymphoblastic leukemia (Hsu _et al._ 2021)
MTHFR Carcinoma of esophagus (Wang _et al._, 2008)
IL6 Scleroderma (Feghali _et al._ 1992)
IL1B Hepatitis (Hirankarn _et al._ 2006)


**Table 13.** Experimental evidence of novel drug–protein relations.


**Protein** **Drug** **Evidence**


OPRM1 Clozapine (Solismaa _et al._ 2018)
SHBG Fostamatinib No evidence
UGT1A8 Acetaminophen (Yoshizawa _et al._ 2020)
DRD5 Pentobarbital No evidence
CYP1A1 Amitriptyline (Mayer _et al._ 2007)


long development time. Therefore, many works have tried to
develop systematic experimental and computational drug
repurposing methods. Some previous works suggested meth­
ods that limit input data in different ways to increase numeri­
cal evaluation performance or fit the data to their models.
However, numerical evaluation cannot be fully trusted in this
task since a method’s lower accuracy is probably due to sug­
gesting more actual drug repurposing candidates. Hence,
manual data manipulation may unnoticeably hinder a meth­
od’s drug-repurposing ability.

This work presents an end-to-end, automatic method that
removes previous works’ data limitations while maintaining
comparable performance. This method can use any graph
data type, even with massive scales, plus embedded side infor­
mation to predict novel indications. We do not force the
method to utilize some parts of the data with a greater em­
phasis. Nevertheless, the model learns which input part is rel­
evant to the task. As a result of this feature, the method also
performs well on other tasks, such as predicting drug–protein
and disease–protein relationships.

A three-step method for drug repurposing is proposed us­
ing a heterogeneous input graph that overcomes some of the
data-related limitations of previous studies. It is important to
note that data limitations can make a method incapable of
helping in the case of new disease outbreaks such as COVID,
in which only limited information is available regarding the
disease. In the first step, network construction, we attempt to
use an existing graph rather than create a new one instead of
previous works. As a result, we are preventing the pipeline
from being biased by our domain knowledge. Additionally,
we avoid any further preprocessing, such as constructing sim­
ilarity matrices, which is another source of bias, and allow
our method to determine how information should be
extracted from the input data. We only remove drug and dis­
ease nodes that do not contribute to any indication edge to
make the results comparable to previous works. It is note­
worthy that removing the nodes adversely impacts our per­
formance, which aligns with expectations since it is easy for
the model to predict no indication relating to those nodes.
Our method could add any embedded information to graph
entities. Therefore, we add BioBERT and ChemBERTa
embeddings to the graph, providing literature and drug struc­
ture information. In the second step, heterogeneous graph
transformer layers extract node features. The architecture of
HGT can extract implicit meta-paths of the graph and


**10** Gharizadeh _et al._



capture the common and specific patterns of different rela­
tionships. Due to these properties, we can use any entity and
relations in the input with side information without manipu­
lating them to increase performance. As high-throughput
technologies make vast amounts of data available in various
areas, scalable drug repurposing has become increasingly im­
portant. Thus, we use HGSampling to make the method scal­
able as well. Lastly, we use a fully connected neural network
end-to-end with the previous steps to customize the extracted
features for the downstream task. This enables the method to
be employed for other similar tasks, too.

A 5-fold cross-validation procedure was utilized for evalu­
ation, and the AUPR and AUROC metrics were compared
with three other state-of-the-art methods. The method named
HeTDR performed better than HGTDR by about 2% in both
measures with their default conditions. However, HGTDR
outperformed HeTDR by about 8% when new disease condi­
tions are assumed in both methods. These metrics, however,
cannot demonstrate the same performance since we cannot
evaluate all the false positive samples that may be unknown
true positives. We examined the experimental evidence for
this case’s top 10 false positives. Eight out of 10 predictions
had evidence from literature suggesting an indication rela­
tionship. We also found that our method was robust to input
variation when some relations from the input graph were re­
moved. To show the applicability of HGTDR in other tasks,
we changed the downstream task of indication prediction to
six other tasks without modifying the model. Both numerical
and experimental evaluations suggest that the model can ex­
tract informative features when the task is shifted.

Our method facilitates adding information to a knowledge
graph and predicting hitherto unidentified information.
Nonetheless, the extension of medical data encounters con­
straints, notably due to the lack of unique identifiers for enti­
ties, which hampers the addition of fresh information. While
there is a potential to map and convert identifiers, the process
is not exhaustive, with some data remaining unmapped and
hence unconverted. Another limitation inherent in our method
is its focus on predicting relationships numerically without the
capacity to substantiate these predictions with empirical evi­
dence. For instance, although our method can predict whether
a drug is suitable for treating a disease, it cannot elucidate the
drug’s mechanism of action. Additionally, this method does
not allow us to add side information to edges like we do for
nodes. It is important to note that some information does not
relate to a node but rather to an edge. As an example, the ex­
pression of a gene in a disease is information related to an edge
that this approach cannot use.

In future research, there is the opportunity to enrich the
knowledge graph by expanding the existing graph structure
and incorporating varied types of ancillary information from
emergent sources. Given the multifaceted nature of drug devel­
opment, a holistic assimilation of information could substan­
tially enhance performance. Additionally, the methodology
has potential applications in combination therapies, which
could benefit from the integrated embeddings of multiple
drugs. Furthermore, efforts can be directed toward augmenting
the interpretability of the end-to-end framework, thus increas­
ing the utility and understanding of the predictive outcomes.


**Supplementary data**


[Supplementary data are available at](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btae349#supplementary-data) _Bioinformatics_ online.



**Conflict of interest**


None declared.


**Funding**


K.A. was supported by Iran National Science Foundation
(INSF) under Grant No. 99025741, and H.R.R. was sup­
ported by Iran National Science Foundation (INSF) under
Grant No. 96006077.


**Data availability**


Our code and data are available at: [https://github.com/bcb-](https://github.com/bcb-sut/HGTDR)
[sut/HGTDR; http://git.dml.ir/BCB/HGTDR.](https://github.com/bcb-sut/HGTDR)


**References**


Cawson R. Treatment of oral lichen planus with betamethasone. _Br_
_Med J_ 1968; **1** :86–9.
Cen Y, Zou X, Zhang J _et al._ Representation learning for attributed
multiplex heterogeneous network. In: _Proceedings of the 25th ACM_
_SIGKDD International Conference on Knowledge Discovery &_
_Data Mining_, New York, NY, USA: ACM, 2019, 1358–68.
Chandak P, Huang K, Zitnik M. Building a knowledge graph to enable
precision medicine. _Sci Data_ 2023; **10** :67.
Che M, Yao K, Che C _et al._ Knowledge-graph-based drug repositioning
against COVID-19 by graph convolutional network with attention
mechanism. _Future Internet_ 2021; **13** :13.
Chen Y, de Rijke M. A collective variational autoencoder for top-n rec­

ommendation with side information. In: _Proceedings of the 3rd_
_workshop on deep learning for recommender systems_, New York,
NY, USA: ACM, 2018, 3–9.
Chithrananda S, Grand G, Ramsundar B. Chemberta: large-scale selfsupervised pretraining for molecular property prediction. arXiv,
arXiv:2010.09885, 2020, preprint: not peer reviewed.
Falini B, Brunetti L, Martelli MP. Dactinomycin in NPM1-mutated
acute myeloid leukemia. _N Engl J Med_ 2015; **373** :1180–2.
Feghali C _et al._ Mechanisms of pathogenesis in scleroderma. I.
Overproduction of interleukin 6 by fibroblasts cultured from af­
fected skin sites of patients with scleroderma. _J Rheumatol_ 1992;

**19** :1207–11.
Finan M, Roberts W, Kavanagh J. Ovarian Sertoli-Leydig cell tumor:
success with salvage therapy. _Int J Gynecol Cancer_ 1992; **3** :189–91.
Himmelstein DS, Lizee A, Hessler C _et al._ Systematic integration of bio­

medical knowledge prioritizes drugs for repurposing. _Elife_ 2017;

**6** :e26726.
Hirankarn N, Kimkong I, Kummee P _et al._ Interleukin-1β gene poly­

morphism associated with hepatocellular carcinoma in hepatitis B
virus infection. _World J Gastroenterol_ 2006; **12** :776–9.
Hsu P-C, Pei J-S, Chen C-C _et al._ Significant association of CCND1 gen­

otypes with susceptibility to childhood acute lymphoblastic leuke­
mia. _Anticancer Res_ 2021; **41** :4801–6.
Hu Z, Dong Y, Wang K _et al._ Heterogeneous graph transformer. In:
_Proceedings of the web conference 2020_, New York, NY, USA:
ACM, 2020, 2704–10.
Ioannidis VN, Song X, Manchanda S _et al._ Drkg-drug repurposing
knowledge graph for COVID-19. arXiv, arXiv:2010.09600, 2020,
preprint: not peer reviewed.
Jin S, Niu Z, Jiang C _et al._ HeTDR: drug repositioning based on hetero­

geneous networks and text mining. _Patterns_ 2021; **2** :100307.
Kalivas J, Breneman D, Tharp M _et al._ Urticaria: clinical efficacy of
cetirizine in comparison with hydroxyzine and placebo. _J Allergy_
_Clin Immunol_ 1990; **86** :1014–8.
Law V, Knox C, Djoumbou Y _et al._ DrugBank 4.0: shedding new light
on drug metabolism. _Nucleic Acids Res_ 2014; **42** :D1091–7.


HGTDR **11**



Lee J, Yoon W, Kim S _et al._ BioBERT: A pre-trained biomedical lan­

guage representation model for biomedical text mining.
_Bioinformatics_ 2020; **36** :1234–40.
Loshchilov I, Hutter F. SGDR: Stochastic gradient descent with warm
restarts. In: _International Conference on Learning Representations_
_(Poster), Toulon, France_, 2017.
Loshchilov I, Hutter F. Decoupled weight decay regularization. In: _7th_
_International Conference on Learning Representations (ICLR)_,
_New Orleans, LA, USA, May_ 2019, 6–9. OpenReview.net.
Mayer RT, Dolence EK, Mayer GE. A real-time fluorescence assay for
measuring N-dealkylation. _Drug Metab Dispos_ 2007; **35** :103–9.
Ng A. Sparse autoencoder. _CS294A Lecture Notes_ 2011; **72** :1–19.
Pan X _et al._ Deep learning for drug repurposing: methods, databases,
and applications. _Wiley Interdiscip Rev Comput Mol Sci_ 2022;
**12** :e1597.
Pelissolo A. Efficacy and tolerability of escitalopram in anxiety disor­

ders: a review. _Encephale_ 2008; **34** :400–8.
Percha B, Altman RB. A global network of biomedical relationships de­

rived from text. _Bioinformatics_ 2018; **34** :2614–24.
Pushpakom S, Iorio F, Eyers PA _et al._ Drug repurposing: progress, chal­

lenges and recommendations. _Nat Rev Drug Discov_ 2019;
**18** :41–58.
Shimoni Z _et al._ The use of prednisone in the treatment of trichinellosis.
_Age (Yrs)_ 2007; **32** :22.
Singh TU, Parida S, Lingaraju MC _et al._ Drug repurposing approach to
fight COVID-19. _Pharmacol Rep_ 2020; **72** :1479–508.
Sinha R, Shenoy PJ, King N _et al._ Vinorelbine, paclitaxel, etoposide, cis­

platin, and cytarabine (VTEPA) is an effective second salvage ther­
apy for relapsed/refractory Hodgkin lymphoma. _Clin Lymphoma_
_Myeloma Leuk_ 2013; **13** :657–63.
Solismaa A, Kampman O, Lyytik€ainen L-P _et al._ Genetic polymorphisms
associated with constipation and anticholinergic symptoms in patients
receiving clozapine. _J Clin Psychopharmacol_ 2018; **38** :193–9.
Talevi A, Bellera CL. Challenges and opportunities with drug repurpos­

ing: finding strategies to find alternative uses of therapeutics. _Expert_
_Opin Drug Discov_ 2020; **15** :397–401.
Tjalma JJ, Garcia-Allende PB, Hartmans E _et al._ Molecular fluorescence
endoscopy targeting vascular endothelial growth factor a for im­
proved colorectal polyp detection. _J Nucl Med_ 2016; **57** :480–5.
Wagner MJ, Gopalakrishnan V, Ravi V _et al._ Vincristine, ifosfamide,
and doxorubicin for initial treatment of Ewing sarcoma in adults.
_Oncologist_ 2017; **22** :1271–7.



Wang J, Sasco AJ, Fu C _et al._ Aberrant DNA methylation of P16,
MGMT, and hMLH1 genes in combination with MTHFR C677T
genetic polymorphism in esophageal squamous cell carcinoma.
_Cancer Epidemiol Biomarkers Prev_ 2008; **17** :118–25.
Wang X, Ji H, Shi C _et al._ Heterogeneous graph attention network. In:
_The World Wide Web Conference_, New York, NY, USA: ACM,
2019, 2022–32.
Weininger D. SMILES, a chemical language and information system. 1.
Introduction to methodology and encoding rules. _J Chem Inf_
_Comput Sci_ 1988; **28** :31–6.
Yoshizawa K, Arai N, Suzuki Y _et al._ Synergistic antinociceptive activ­

ity of tramadol/acetaminophen combination mediated by m-opioid
receptors. _Biol Pharm Bull_ 2020; **43** :1128–34.
Yu P, Fu C, Yu Y _et al._ Multiplex heterogeneous graph convolutional
network. In: _Proceedings of the 28th ACM SIGKDD Conference on_
_Knowledge Discovery and Data Mining_, New York, NY, USA:
ACM 2022, 2377–87.
Yu Z, Huang F, Zhao X _et al._ Predicting drug-disease associations
through layer attention graph convolutional network. _Brief_
_Bioinform_ 2021; **22** :bbaa243.
Zeng X, Zhu S, Lu W _et al._ Target identification among known drugs
by deep learning from heterogeneous networks. _Chem Sci_ 2020;

**11** :1775–97.
Zeng X, Zhu S, Liu X _et al._ deepDR: a network-based deep learning ap­

proach to in silico drug repositioning. _Bioinformatics_ 2019;

**35** :5191–8.
Zhang C, Song D, Huang C _et al._ Heterogeneous graph neural network.
In: _Proceedings of the 25th ACM SIGKDD international conference_
_on knowledge discovery & data mining, New York, NY, USA:_
_ACM_, 2019, 793–803.
Zhang R, Hristovski D, Schutte D _et al._ Drug repurposing for COVID19 via knowledge graph completion. _J Biomed Inform_ 2021;

**115** :103696.
Zhang Z, Zhou L, Xie N _et al._ Overcoming cancer therapeutic bottle­

neck by drug repurposing. _Signal Transduct Target Ther_ 2020;

**5** :113.
Zhou Y, Wang F, Tang J _et al._ Artificial intelligence in COVID-19 drug
repurposing. _Lancet Digital Health_ 2020; **2** :e667–76.
Zhu Y, Che C, Jin B _et al._ Knowledge-driven drug repurposing using a
comprehensive drug knowledge graph. _Health Informatics J_ 2020;

**26** :2737–50.



# The Author(s) 2024. Published by Oxford University Press.
This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which permits
unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.
Bioinformatics, 2024, 40, 1–11
https://doi.org/10.1093/bioinformatics/btae349
Original Paper


