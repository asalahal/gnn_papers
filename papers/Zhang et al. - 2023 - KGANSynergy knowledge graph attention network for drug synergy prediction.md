_Briefings in Bioinformatics_, 2023, **24(3)**, 1–13


**https://doi.org/10.1093/bib/bbad167**
Advance access publication date 2 May 2023
**Problem Solving Protocol**

# **KGANSynergy: knowledge graph attention network for** **drug synergy prediction**


Ge Zhang, Zhijie Gao, Chaokun Yan, Jianlin Wang, Wenjuan Liang, Junwei Luo and Huimin Luo


Corresponding author. Huimin Luo, School of Computer and Information Engineering, Henan University, Jinming Street, 475004 Kaifeng, China.

E-mail: luohuimin@henu.edu.cn


Abstract


Combination therapy is widely used to treat complex diseases, particularly in patients who respond poorly to monotherapy. For example,
compared with the use of a single drug, drug combinations can reduce drug resistance and improve the efficacy of cancer treatment.
Thus, it is vital for researchers and society to help develop effective combination therapies through clinical trials. However, highthroughput synergistic drug combination screening remains challenging and expensive in the large combinational space, where an
array of compounds are used. To solve this problem, various computational approaches have been proposed to effectively identify
drug combinations by utilizing drug-related biomedical information. In this study, considering the implications of various types of
neighbor information of drug entities, we propose a novel end-to-end _Knowledge Graph Attention Network_ to predict drug synergy
(KGANSynergy), which utilizes neighbor information of known drugs/cell lines effectively. KGANSynergy uses knowledge graph (KG)
hierarchical propagation to find multi-source neighbor nodes for drugs and cell lines. The knowledge graph attention network is
designed to distinguish the importance of neighbors in a KG through a multi-attention mechanism and then aggregate the entity’s
neighbor node information to enrich the entity. Finally, the learned drug and cell line embeddings can be utilized to predict the synergy
of drug combinations. Experiments demonstrated that our method outperformed several other competing methods, indicating that our
method is effective in identifying drug combinations.


Keywords: drug synergy prediction, knowledge graph, multi-head attention network, deep learning



INTRODUCTION


In complex diseases such as cancer, HIV and cardiovascular disease [1–3], many cellular mechanisms commonly change in different tissues and organ systems [4]. Therefore, conventional ’single
compound, single target’ approaches to treating these diseases
are typically ineffective [5]. Many studies have suggested that a
combination therapy of multiple drugs can significantly improve
therapeutic efficacy and reduce drug toxicity [6]. At the same
time, drug combination therapy can reduce the risk of drug resistance, which can improve the success rate of drug repositioning

[7]. For instance, amiloride and hydrochlorothiazide are used to
treat hypertension [8, 9]. The food and drug administration (FDA)
approved the combination of the BRAF inhibitor dabrafenib and
the MEK inhibitor trametinib for treating patients with metastatic
melanoma [10]. Similarly, drug combination therapy has become
the primary treatment strategy for many complex diseases [11].
Traditional drug combination discovery is mainly based on clinical trials, which have been proved to be time-consuming, expensive and potentially harmful to patients [12, 13]. Recently, with the



emergence and application of high-throughput screening techniques (HTS) [14], ample verified drug combination data have
been exploited and accumulated. However, today’s rapidly growing biomedical data create difficulties in testing the whole combination space via the above techniques [5, 15, 16]. In addition,
the development of drug combinations is also a costly process for
pharmaceutical companies [5]. Thus, there is an urgent need for
efficient and economical methods to screen drug combinations.

A variety of computational methods have been developed
to predict drug combinations and thereby address the above
problems, including methods in the following categories [17]:
(i) systems biology methods, (ii) mathematical methods, (iii)
kinetic models and (iv) machine learning methods. Systems
biology methods primarily apply biological knowledge to analyze
biological networks [18]. Mathematical methods require the
application of mathematical models and statistical tests, as well
as rely on the reliability of model assumptions. Kinetic models use
kinetic equations to simulate the dynamic changes of nodes in
biological networks [19]. The above three methods are suitable



**Ge Zhang** is an associate professor in the School of Computer and Information Engineering, Henan University, Kaifeng, China. His research interests include
artificial intelligence and computational biology.
**Zhijie Gao** is a graduate student in the School of Computer and Information Engineering, Henan University, Kaifeng, China. Her research interests include
bioinformatics and computational drug repositioning.
**Chaokun Yan** is a professor in the School of Computer and Information Engineering, Henan University, Kaifeng, China. His research interests include artificial
intelligence and computational biology.
**Jianlin Wang** is an associate professor in the School of Computer and Information Engineering, Henan University, Kaifeng, China. His research interests include
artificial intelligence and computational biology.
**Wenjuan Liang** is a lecturer in the School of Computer and Information Engineering, Henan University, Kaifeng, China. Her research interests include artificial
intelligence and data mining.
**Junwei Luo** is an associate professor in the College of Computer Science and Technology, Henan Polytechnic University, Jiaozuo, China. His research interests
include genome assembly, scaffolding, gap filling and structural variant detection.
**Huimin Luo** is an associate professor in the School of Computer and Information Engineering, Henan University, Kaifeng, China. Her research interests include
bioinformatics and computational drug repositioning.
**Received:** September 11, 2022. **Revised:** March 10, 2023. **Accepted:** April 3, 2023
© The Author(s) 2023. Published by Oxford University Press. All rights reserved. For Permissions, please email: journals.permissions@oup.com


2 | _Zhang_ et al.


for small datasets and depend on prior knowledge. Machine
learning methods such as Random Forest (RF), Support Vector
Machine (SVM) and Naive Bayesian methods can explore vast
combinatorial spaces and expedite the identification of drug
combinations [20–22].

In recent years, deep learning methods have equipped scientists with powerful machine learning tools for abstracting important data features from large-scale datasets and have also been
demonstrated to be effective in biomedical fields [23, 24]. For
instance, Preuer _et al_ . [25] proposed a deep learning model called
DeepSynergy to predict cancer drug combinations with synergistic effects. This model utilizes both compound and genomic
information as inputs and predicts drug combinations using the
Hill curve structure with tanh normalization. Kuru _et al_ . [26]
used the fully connected layer to predict whether a drug pair
has synergistic effects by combining medicinal chemical structure data and cell line gene expression profiling features. Wang
_et al_ . [27] suggested a DeepDDS model to predict the synergistic
effects of drug combinations for cancer. The model uses a graph
convolutional network and an attention mechanism to determine

drug embeddings. Drug chemical structures and gene expression
patterns are embedded as features in a multilayer feedforward
neural network to predict synergistic drug combinations. However, most of the aforementioned methods focus on the molecular structure of drugs and cell lines rather than the interaction
information.


Some studies have reported that drug combinations interfering
synergistically with protein networks can more effectively block
oncogene activity [18, 28]. This topic has raised concern among
bioinformaticians in the treatment of complicated diseases where
the topological link between drugs and diseases in protein association networks is utilized. Xu _et al_ . [29] integrated six features,
including molecular structure, ATC Code similarity and protein–
protein associations; they used SVM, NB and SGB to predict drug
combinations. Liu _et al_ . [30] applied the random walk with the
restart algorithm on a protein–protein association network to
extract new drug–target profiles as drug features for drug combination prediction. However, the mentioned network-based methods only consider entities related to drugs or diseases directly;
they usually neglect local connections and lack interpretability. Yang _et al_ . [31] used information propagation and attention
mechanisms to explore the higher order topological information of drug proteins and cell line proteins in protein–protein
interaction networks to predict drug combinations. We found
that any one of protein association information, graph attention
network (GAT) or neighbor node information could improve the
efficacy of drug synergy prediction. However, the propagationbased approach uses a single type of neighbor information rather
than multiple types of neighbor information, which may negatively impact the capture performance of a particular drug or
cell line.


Knowledge graphs (KGs) are a type of heterogeneous graph
in which nodes represent entities, and edges represent relations
between entities. Rich semantic linkages between entities in the
graph can benefit the system in discovering potential relations
between entities. In recent years, KGs have been successfully
applied in natural language processing, recommender systems
and many other fields [32, 33]. Specifically, propagation-based
recommendation methods—which are propagated over the whole
KG to find the recommended auxiliary information—are the dominant approaches for KG-based recommender systems, such as
CKAN and KGAT [34, 35]. Inspired by these, bioinformaticians
have begun to use KGs to explicitly combine heterogeneous data



for drug repositioning downstream tasks. This approach not only
enables the extraction of fine-grained multimodal knowledge
elements from omics data but also captures information between
entities and their neighbors more completely. Drug repositioning
and adverse drug reaction prediction based on biomedical KGs
increase the opportunities for accelerated drug discovery [36,
37]. For instance, a knowledge graph neural network (KGNN) [38]
combines the graph neural network (GNN) and KG to determine
the interactions between drug pairs. This model aggregates information from neighborhood entities to learn representations for
drug entities to capture higher order structural and semantic
relationships. Mohamed _et al_ . [39] introduced the KG embedding
model TriModel to learn vector representations for all drugs and
proteins, inferring new drug–target interactions based on the
scores computed by the model. It is evident that using KGs to
learn entities such as drugs, proteins and their topological neighbors can obtain a more accurate embedding representation, but
previously developed KG-based methods may perform weakly
in terms of attention stability and node aggregation capability

[35, 40]. Furthermore, KGs have not been widely used for drug
combination prediction.

In this study, we propose a novel deep learning model based
on a KG and multi-head attention mechanism to predict drug
synergy. Specifically, our model consists of three main modules.
We first apply the hierarchical propagation of the KG to find drug/cell line neighbors. Then, we develop a KG attention network to
obtain the corresponding neighborhood representations of drugs
or cell lines. Finally, the predictive score for potential synergies
is obtained via a well-defined score function that utilizes the

embedded representation of drugs and the cell line. The advantage of our model lies in extending the multi-attention mechanism to prevent overfitting and stabilize the learning results.
KGANSynergy also considers different types of neighbor node
information.


The contributions of this work are summarized as follows:


(i) We propose a novel KG attention network framework that
can obtain high-order information from multi-source neighbors in the KG efficiently and explicitly.
(ii) We design a neural-network-based multi-head attention
method to completely utilize neighborhood information
and improve the interpretability of the model. The module
highlights the different levels of importance to each entity
neighborhood node, which stabilizes the attention learning

process.
(iii) To enhance node embedding aggregation, we apply different
types of aggregators in terms of neighboring nodes and
hierarchical aggregation, highlighting different higher order
connections and similarity.
(iv) We conducted experiments on two benchmark datasets, and
the results show that KGANSynergy outperformed state-ofthe-art methods and baseline methods in the prediction of
drug combination synergy.


MATERIALS AND METHODS

**Problem formulation**


We describe the problem of drug combination prediction based on
KG attention.

**Definition 1.** (Drug–drug–cell line) Typically, drug _i_, drug _j_ and
cell line _k_ are included in the drug synergy prediction. The drug
combination for a particular cell line is represented as _Y_ ∈
_(_ 0, 1 _)_ [|] _[N]_ _[d]_ [|×|] _[N]_ _[d]_ [|×|] _[N]_ _[k]_ [|], where | _N_ _d_ | denotes the number of drugs, and | _N_ _k_ |


denotes the number of cell lines. When _y_ _i_, _j_, _k_ = 1 � _i_, _j_ ∈ _N_ _d_, _k_ ∈ _N_ _k_,
and _i_ ̸= _j_ ), the combination of the drug pair has a synergistic effect
on the cell line; otherwise, the combination has an antagonistic
effect on the cell line.

**Definition 2.** (KG) In addition to drug synergy data, we also incorporate auxiliary information for the drug and cell-line-related
entities (e.g. proteins) via a KG. In this work, we construct a
KG to illustrate the associations between drugs and cell lines:
_G_ = { _(h_, _r_, _t)_ | _h_, _t_ ∈ _ε_, _r_ ∈ R}, where _ε_ and R represent the set
of entities and relations, respectively. The KG consists of entity–
relation–entity triples _(h_, _r_, _t)_ . Here, _h_ ∈ _ε_, _r_ ∈ R and _t_ ∈ _ε_ denote
the head, relation and tail of the knowledge triple, respectively.
For example, the triplet (Miconazole, DPI, P35228) indicates that
Miconazole interacts with protein P35228. In the graph, entities
are represented as nodes, and relations are represented as edges
from the head entity node to the tail entity node.


**Dataset description**
_Drug synergy score dataset_


In this paper, we focus on the combinations of two drugs. To
investigate the performence of our model on the small samples
and large samples, we conducted experiments on two drug
combination datasets:

(1) **DrugCombDB.** DrugCombDB [41] is the first comprehensive
database with the largest number of drug combinations to
date, which integrates drug combinations from high-throughput
screening (HTS) analyses, external databases and PubMed
literature. The raw data contains 69,436 drug combinations,
including 764 drugs and 76 cancer cell lines.
(2) **Oncology-Screen.** Oncology-Screen [42] was collected by
O’Neil _et al_ . through large-scale tumor screening and includes
4176 drug combinations involving 21 drugs and 29 cancer cell
lines in the raw data. Furthermore, Preuer _et al_ . [25] integrated the
dataset by calculating the Loewe additivity value.

Each sample comprised of two drugs and one cell line, as well
as the relevant synergy score. In general, the degree of synergy
shown by the data was quantified based on the synergy deviation
simulated by the theoretical model, such as Loewe Additivity [43],
bliss independence [44], highest single agent (HSA) [45] and zero
interaction potency (ZIP) [46]. The synergy scores discussed in
DrugCombDB is the ZIP value, and Oncology-Screen is the Loewe
Additivity Value computed by Combenefit [47]. As suggested by
Liu _et al._ [41], we used the quartile as the threshold to exclude lowconfidence drug combination samples, and the drug combinations with synergy scores distributed in the top quartile and bottom quartile are classified as synergistic and antagonistic effects.
In this experiment, synergy was considered a positive sample,
while antagonism was considered a negative sample. Table 1
displays the basic statistical results of these two datasets after
processing. The DrugCombDB dataset contains 17404 synergistic
drug pairs and 16 624 antagonistic drug pairs, which are used to
evaluate the performance of the model on a large sample dataset;
the Oncology-Screen dataset contains 1044 synergistic drug pairs
and 916 antagonistic drug pairs, which are used to evaluate the
performance of the model on a small sample dataset.


_Knowledge graph-related data_


Protein associations are a significant source of molecular
information. Their associations (with one another or certain small
molecules) have a role in metabolism, signaling, immunity and
gene regulatory networks [48]. Aberrant associations produce
aberrant cellular behaviors and diseases. Therefore, they should
be major targets for molecular-based research of biological



_KGANSynergy_ | 3


**Table 1.** Details about the two drug combination datasets used
after processing


**Datasets** **Drugs** **Cell lines** **Positive** **Negative**


DrugCombDB 475 76 17 404 16 624
Oncology-Screen 21 29 1044 916


**Table 2.** Statistics of the KG data


**DrugCombDB** **Oncology-Screen**


Entities 16 832 16 026

Relations 4 4

KG triples 250 256 226 475


disease states. In this study, we mainly used protein-related data
to construct the KG, and the data were mainly from [31]. Table 2
shows the detailed data information of the KG.

**Drug–protein association.** Drug–protein associations of FDAapproved or clinical research drugs were collected from six
commonly used databases, including DrugBank, BindingDB,
ChEMBL and PharmGKB [18]. The datasets contain 15 051 drug–
protein associations, including 4428 drugs and 2256 human
proteins.
**Cell** **line–protein** **association.** Cell line–protein associations
were obtained from Cancer Cell Line Encyclopedia (CCLE) gene
expression and drug response data [49]. This is a large-scale
analytical dataset of nearly 1000 cell lines from different tissues.
The data were collected and processed by Rouillard _et al_ . [50],
which converted the cell line-gene expression data matrix in CCLE
to a cell line-protein association matrix. Specially, the average
and standard deviation are computed across all cell lines, and the
genes with _z_ -scores larger than a specific threshold suggested in

[50] is considered to be associated with cell lines. The cell line–
protein associations include 18 022 protein-coding genes, 1035
cancer cell lines and 749 551 associations.

**Protein–protein association.** Protein–protein associations were
taken from the human protein–protein interactome [18], which
is collected from 15 commonly used databases. It excludes
evolutionary analyses, gene expression data and metabolic
association-related associations. The dataset consists of 217

160 protein–protein interactions connecting 15 790 independent
proteins. Based on Genecards [51], each protein is mapped to its
coding gene.
**Cell line–tissue association.** Tissue consists of cells that divide

and differentiate. As there may be connections between the
cell lines of the tissues, cell line–tissue information can lead to
the identification of important features between cell lines so
that more accurate cell line characterization can be learned.

For example, DrugCombDB contains lung, liver, large intestine,
breast, prostate and eight other types of tissues. Oncology-Screen
contains six tissues, including breast, lung and ovary. When drug _i_
and drug _j_ interact with a cell line in a particular tissue, the tissue
may have a therapeutic effect.


**Knowledge graph construction**


To construct a KG, we integrated four categories of data, such as
drug– protein association, cell line– protein association, protein–
protein association and cell line– tissue data from the aforementioned datasets in Section 2.2.2 Knowledge graph-related
data. The KG contains entities and relations from the above

four categories. As shown in Figure 1, we assume that the drug


4 | _Zhang_ et al.


Figure 1. Example of drug combination prediction. Com1, Com2,... represents a set of drug combinations; D1, D2...represents the drugs; and C1,
C2,... represents the drug combination’s cell lines.


combination Com1 and the cell line C2 are potentially predictive
associations. The KG reveals information about drug–protein, cell
line–protein and cell line–tissue associations as well as protein–
protein associations. After a message is passed for target drugs
and cell lines through the KG attention network, their embeddings
include neighbor information from other relations, such as proteins or tissues. Thus, we can infer that some indirectly related
entities potentially contribute to the therapeutic process. The KGbased method can integrate multiple types of entity information
to make accurate predictions regarding the synergistic effects of
drug combinations.

The KG of the drug i–drug j–cell line is represented as _G_ = _(ε_, _R)_,
where _ε_ denotes a variety of entities (drugs, cell lines, tissues and
proteins), and _R_ denotes the association between various entities
(such as drug–protein association, cell line–protein association,
protein–protein association and cell line–tissue). The form of the
KG can be represented as multiple entity–relation–entity triples
_T_ = _(h_, _r_, _t)_, where _h_, _t_ ∈ _ε_ represent the head entity and the tail
entity, respectively, and _r_ ∈ _R_ represents the relation between the
two entities.


**The framework design**


In this paper, we design KGANSynergy as a framework for predicting drug synergy. The framework of KGANSynergy is briefly shown
in Figure 2. The model consists of the following steps


(1) KG hierarchical propagation. First, the entities and relations in KG are mapped to vectors. Then, the KG recursively
explores the set of neighbor nodes directly and indirectly
associated with the drug/cell line.
(2) KG attention layer. We propose a neural-network-based
knowledge attention mechanism to iteratively propagate
multiple layers of information to update entity representations. This layer uses information propagated by
the KG to encode the drug and cell line features among
drug–protein, cell line–protein and other associations, as
well as the neighborhood structure between them (i.e.
entities and relations). Finally, each layer entity (h) and its
neighbor node embedding set are aggregated to obtain the
embedding representation set of the _l_ -layer. The multi-head
attention mechanism is used to obtain the neighbor weight
information and stabilize the learning process.
(3) Prediction layer. This layer utilizes drug and cell line representations obtained after a series of polymerization updates
to calculate the prediction scores.


_Knowledge graph hierarchy propagation_


Adjacent entities in the KG are always strongly correlated.
KGANSynergy’s knowledge graph propagation layer is used to
obtain multihop neighbor node sets for drugs/cell lines. Figure 3
illustrates the delivery process of Miconazole at different depths
in the KG. To facilitate understanding, the protein coding gene is



mapped to the corresponding protein. The triplet (Miconazole,
DPI, P35228) suggests a drug–protein association between
Miconazole and protein P35228; the triplet (Q12809, CPI, K562)
indicates a cell line–protein association between Q12809 and cell
line K562. Take miconazole as an example. The drug Miconazole
collects information of protein P35228 and protein Q12809
from the first layer, which are the first-order direct neighbors
connecting Miconazole. In the second layer, Miconazole obtains
information from its second-order neighbors: P00352, TC32 and
K562. The first-order neighbor entity stores a representation of
the prior layer’s second-order neighbor entity, which captures
information from second-order neighbors. Analogously, entities
can obtain information from their _l_ -th order neighbors at the
_l_ -th layer of knowledge propagation. KGANSynergy conducts
hierarchical propagation of drug and cell line entity information
in the KG. Therefore, we can obtain the expanded entity set of
_L_ layers, which can effectively enrich the entity latent vector
representation.

First, we find the set of entities associated with a drug/cell line
in the KG. Here, the drug/cell line is abbreviated as _o_ . The set of
adjacent entities of _o_ can be expressed recursively as


_ε_ _o_ _[l]_ [=] � _t_ | _(h_, _r_, _t)_ ∈ _G_ and _h_ ∈ _ε_ _o_ _[l]_ [−][1] �, _l_ = 1, 2, _. . ._, _L_ (1)


where _l_ denotes the depth of KG propagation, _(h_, _r_, _t)_ denotes a set
of triples, _t_ denotes the tail entity and _G_ denotes the KG.

To reduce the computational burden, we choose a fixedsize set of neighbors for each entity [33]. Subsequently, the
entity adjacency matrix _A_ _e_ and the relational adjacency matrix
_A_ _r_ _(_ _A_ _e_, _A_ _r_ ∈ R _[v]_ [∗] _[n]_ _)_ are constructed based on the entity set,
respectively ( _v_ denotes the number of entities, and _n_ denotes
the number of fixed size neighbor entities). Particularly, each
row in the entity adjacency matrix _A_ _e_ represents _n_ neighbor
entities of an entity. The relational adjacency matrix contains the
association information between each entity and its neighbors,
and the relations include drug–protein, protein–protein, cell line–
protein and cell line–tissue.

The KG is used to expand the neighboring nodes at each layer,
propagating from far to near, layer by layer. Knowledge-based
information propagation can efficiently collect higher order association information about drugs and cell lines, thus enhancing the
capability of potential vector representation of entities.


_Knowledge graph attention layer_


Each tail entity has different meanings and potential vector representations when it has different head entities and relations in

the KG. Additionally, there are intricate associations between each
adjacent tail entity. To accurately capture entity embeddings, we
also take into account the relation between the head entity’s
various neighbors based on the attention network. The module
assigns different importance levels to the entity’s neighbor set
through a multi-head attention mechanism. It updates and aggregates the importance levels to obtain the embedded representation of the corresponding drug or cell line. More specifically,
each layer of KG attention consists of two main components:
multi-head attention embedding propagation and neighborhood
information aggregation.


Multi-head attention embedding propagation


Entity _h_ may participate in multiple triples as the bridge
connecting two triplets and propagating information. Therefore,
entity _h_ has multiple different neighbors, and there is an


_KGANSynergy_ | 5


Figure 2. **Architecture of KGANSynergy.** A and B: Construction of the KG. **A.** Four different types of data are collected: drug–protein association, cell
line–protein association, cell line–tissue association, and protein–protein association. **B.** Use the data in A to build a KG that includes four types of
entities: drugs, cell lines, proteins and tissues. **C.** KG propagation layer. Hierarchical propagation is utilized to identify the set of drug/cell line entity
neighbor nodes. The symbol _e_ represents the initial entity associated with the drug/cell line entity, which is the set of tail entities directly related
to the drug/cell line. **D.** KG attention layer. The figure shows a multi-head attention network that acts on each knowledge propagation process. D
(1). The multi-head attention part of the KG. { _t_ 1, _t_ 2, _. . ._, _t_ _n_ } is the set of _n_ neighboring nodes to entity _h_ in this layer. Arrows with different colors
represent multiple attention learning using the triple _(h_, _r_, _t)_, which is composed of entity _h_ and each neighbor in the neighborhood set. The multiple
learned embedding representations are concatenated/averaged to generate the final neighborhood set embedding representation _e_ _[l]_ _N_ _h_ [. D (2). Using the]

Bi-Interaction aggregator, the _l_ -th layer entity _h_ is aggregated with the neighbor node embedding _e_ _[l]_ _N_ _h_ [acquired in the preceding step, and finally obtain]
the embedding of drug or cell line entity at layer _l_ . **E.** Each layer of node embedding learned by the drug/cell line is aggregated to obtain the drug/cell
line’s final embedding representation. **F.** Prediction layer. The probability of synergy between the drug pair on a particular cell line is outputted.


Figure 3. **Illustrations of different layers of Miconazole in the KG.** When L=1, the drug Miconazole is directly related to protein P35228 and protein
Q12809, respectively. When L=2, the one-hop node protein P35228 is directly related to protein P00352 and cell line TC32, the one-hop node protein
Q12809 is directly related to cell line K562, and these two-hop entities are indirectly adjacent to Miconazole. With the propagation of information
through layer, the information contained in the node is integrated by its layer. Concentric circles depict nodes with different levels. The weaker the
association between the center and surrounding entities, the lighter the hue of green.



association between these neighbors. To avoid providing the
same weight to each neighbor when aggregating information, we
adopted the idea of a graph attention network, which assigns
different importance levels to the neighbors of each entity _h_
and generates embedded attention weights for propagation. The
process not only focuses on the network, but also considers the
complex associations between the neighbor nodes of entity _h_ .



Considering the instability of the original GAT’s learning process,
we add a multi-head attention mechanism to the original study’s
foundation. As shown in Figure 4, we draw the propagation
of multi-head attention. The figure shows _n_ neighbor nodes
of _h_, where { _t_ 1, _t_ 2, _. . ._, _t_ _n_ } denotes the _n_ neighbors of _h_ in
that layer. Different arrow colors indicate that several triples
have collectively performed multiple iterations of attention


6 | _Zhang_ et al.


Figure 4. **Diagram of the multi-head attention propagation process.**
Different arrow colors in multi-head attention represent independent
attention computations. The aggregated features from each head are concatenated or averaged to obtain the set of neighborhood representations
directly connected to entity _h_ .


learning. Finally, multiple learned neighbor representations are
aggregated to obtain a set of neighbor embedding representations.
The following section describes the multiple attention propagation in detail.


First, the neighborhood representation of entity _h_ is learned
based on the neighborhood entities. When considering _h_, the
neighbors of entity _h_ are denoted by _N_ _h_ = { _(h_, _r_, _t)_ | _(h_, _r_, _t)_ ∈ _G_ },
where _t_ denotes the neighboring entities of the head entity _h_,
and _r_ denotes the relation. The purpose of attention embedding
propagation is to encode _N_ _h_, and the output vector is represented
as a set of embedding representations of neighbors. To improve
the node representation capacity, the graph attention network
uses _e_ _N_ _h_ to represent the set of entity embedding representations
directly connected to the entities _h_ . In this study, we take the
triple _(h_, _r_, _t)_ as an example to perform a weighted sum of each
neighbor node of entity _h_ and finally obtain the neighborhood
representation of entity _e_ _N_ _h_ as follows:



_e_ _N_ _h_ = LeakyReLU



According to the principle of the graph attention mechanism,
we use the softmax function to normalize the coefficients of the

triples associated with entity _h_ to make it easier to compare the
attention coefficients between different entities (the sum of attention of all neighboring nodes is 1) [40]. The final attention score
can highlight which neighboring nodes should be given more
attention to capture entity embeddings. The Softmax function is
expressed as follows:


_π(h_, _r_, _t)_ = softmax _(π(h_, _r_, _t))_

= ~~�~~ _(_ _[h]_ [,] _[r]_ [′] ex [,] _[t]_ [′] _)_ p [exp] _(π(h_ ~~�~~ _π_, _r_ ~~�~~, _th))_, _r_ [′], _t_ [′] ~~[��]~~ [.] (5)


To stabilize the attention learning process and encapsulate
more information about the neighborhood, we also apply the
multi-head attention mechanism as done for GAT, converting
Equation (2) [40]. Multi-head attention is computed _M_ times by
a separate attention mechanism to update the embedding representation of entity _h_ . The multi-head attention layer also requires
an aggregator to integrate all embeddings. In this study, we adapt
two types of aggregators:


 - **Concat** **aggregator.** Splicing operation. Concatenate all
embeddings of multi-head graph attention and then apply
a nonlinear transformation:



� _π(h_, _r_, _t)e_ _t_

_(h_, _r_, _t)_ ∈ _N_ _h_



_e_ _N_ _h_ = LeakyReLU



⎛



∥ _[M]_ m=1 �
⎝ _(h_, _r_, _t)_ ∈



⎞

(6)
⎠




- **Average aggregator.** Mean operation. Sum all embeddings of
the multi-head graph attention and then apply the average to
calculate the final embedding:



� _π(h_, _r_, _t)e_ _t_

_(h_, _r_, _t)_ ∈ _N_ _h_



⎞

(7)
⎠



⎛



⎝ M [1]



M



_M_
�


m=1



_e_ _N_ _h_ = � _π(h_, _r_, _t)e_ _t_, (2)

_(h_, _r_, _t)_ ∈ _N_ _h_



In Equation (6), ∥ denotes the connection operation. _π(h_, _r_, _t)_ of
Equations (6) and (7) is the normalized attention coefficient calculated by the _m_ -th attention embedding propagation, and _M_ is
the number of multi-heads.


Neighborhood information aggregation


Next, the neighbor set embedding _e_ _N_ _h_ obtained in the previous
step is combined with the entity embedding _e_ _h_ to obtain the new
entity embedding representation _e_ _h_ ′ . We use the Bi-interaction
aggregator [35] to implement the function _f_ � _e_ _h_, _e_ _N_ _h_ � :



where _e_ _t_ denotes the embedding representation of entity _h_ ’s
neighboring entity _t_ . _π(h_, _r_, _t)_ denotes the attention weight
of each neighbor _t_ . The weight controls the decay factor on
each propagation on _π(h_, _r_, _t)_, which represents the amount of
information propagated from entity _t_ to _h_ under relation _r_ . The
larger the attention weight of the triple, the more important
the neighbor entity. Owing to the existence of the attention
mechanism, the model is capable of learning different weights
for different neighbors [40]. Next, we implement the function _π(_ - _)_
through a neural network similar to the attention mechanism,
which is formulated as follows:


z 0 = ReLU �W 0 _(e_ _h_ ∥ _e_ _r_ _)_ + b 0 �, (3)


_π(_ h, r, t _)_ = _σ_ �W 2 ReLU �W 1 z 0 + b 1 � + b 2 �, (4)


where ReLU is the nonlinear activation function, and the last
activation function is Sigmoid. ∥ is the concatenation operation.
W and b are the trainable weight weights and biases, respectively.
In particular, _W_ 0 and _b_ 0 in Equation (3) represent the weights and
deviations of the first layer neural network, whereas _W_ 1, _b_ 1 and
_W_ 2, _b_ 2 in Equation (4) represent the weights and deviations used
in the second layer neural network and the output layer.



where LeakyReLU is a ReLU-based activation function that can
assign a non-zero slope to all negative values. The parameter
_W_ 3, _W_ 4 ∈ R _[d]_ _[e]_ [∗] _[d]_ _[e]_ ′ are the weight matrices determined by parameter
learning, and _d_ _e_ ′ is the transformation size. ⊙ denotes the product
of elements. The aggregator considers the interaction of two features between _e_ _h_ and _e_ _N_ _h_ . This function captures more messages
from related entities, which enhances its ability to combine _e_ _h_
and _e_ _N_ _h_ .

The embedded propagation layer enables explicit use of firstorder connectivity information to associate drugs, cell lines



_f_ � _e_ _h_, _e_ _N_ _h_ � = LeakyReLU �W 3 �e h + e N h ��

(8)
+ LeakyReLU �W 4 �e h ⊙ e N h ��,


and knowledge entity representations. However, using only the
first-order neighbors of entities may result in the loss of important
information. We extend multiple knowledge propagation layers
to investigate the influence of higher order neighbors in greater
depth. After we aggregate first-order neighbors, each entity
includes information from its first-order neighbors, and then the
process is repeated. As the first-order neighbor entity has saved
the representation of the second-order neighbors in the preceding
layer, it can collect data from second-order neighbors.

Analogously, at the _l_ -th knowledge propagation layer, the entity
may collect information from _l_ -th order neighbors. As we aggregate _L_ layers, each entity’s embedding eventually contains information from its _L_ -layer neighbors. The representation of entity _h_
in each layer during propagation is defined as follows:



_e_ _[(]_ _h_ _[l][)]_ [=] _[ f]_ � _e_ _[(]_ _h_ _[l][)]_ [,] _[ e]_ _N_ _[(][l][)]_ _h_



�, _l_ = 1, 2, _. . ._, _L_, (9)



_KGANSynergy_ | 7


_Drug synergy prediction_


The output representation of different layers can be interpreted as
the potential impact of different layers, which emphasizes different higher order connectivity and similarity. After _L_ -layer embedding propagation for two drugs and cell line, the corresponding
vector representation sets ( _T_ _o_ ) are aggregated. The output is the
final drug/cell line entity embedding generated by KGANSynergy.
To retain the latent embedding information more completely, we
aggregate the representations of all layers into a single vector by a
concatenate operation and then apply a nonlinear transformation
as follows:


_agg_ _[d]_ concat [=] _[ σ]_ �W · � _e_ [0] _d_ [∥] _[e]_ _d_ [1] [∥] _[. . .]_ [ ∥] _[e]_ _[L]_ _d_ � + _b_ �, (15)


_agg_ _[k]_ concat [=] _[ σ]_ �W · � _e_ [0] _k_ [∥] _[e]_ _k_ [1] [∥] _[. . .]_ [ ∥] _[e]_ _[L]_ _k_ � + _b_ �, (16)


where _e_ _d_ and _e_ _k_ are the embedding representations from the set
_T_ _o_ of drug and cell line embedding representations, respectively.
_W_ and _b_ are the trainable weight and bias, respectively. The
nonlinear activation function _σ_ is set as Sigmoid.

We use two drug and cell line representations obtained after
aggregation to calculate the prediction scores. Specifically, max
pooling uses the element-wise maximum of two drug representations as the combined drug representation [31]. The synergy score
is calculated as follows:


ˆ
_y_ � _d_ _i_, _d_ _j_, _k_ � = max � _e_ _i_, _e_ _j_ � ⊙ _e_ _k_, (17)


**Objective function**


Youden’s J statistic [52] is primarily used to capture the performance of dichotomous diagnostic tests. To ensure the rationality
of the results, we use Youden’s J statistic to search for the most
appropriate cutoff for positive and negative labels. Finally, we set
the labels with synergy probabilities greater than this threshold
to 1 and those less than this threshold to 0. Given the drug
combination data and the KG, our goal is to learn a prediction
function to predict whether drug _i_ and drug _j_ have a synergistic
effect on cell line _k_ :


− _Loss(_ 1 − = _y_ [�] _i_, _j_, _ki_, _j_ _)_ ∈ log _N_ _d_, _i_ ̸= _(_ _j_ 1, _k_ −ˆ ∈ _N_ _k_ _y_ _[(]_ _i_ [−], _j_, _k_ _[y]_ _))_ _[i]_ [,] _[j]_ + [,] _[k]_ [ log] _λ_ ∥ [ ˆ] _�_ _[y]_ _[i]_ [,] ∥ _[j]_ [,][2] 2 _[k]_ . (18)


where ˆ _y_ _i_, _j_, _k_ is the predicted value of the model, and _y_ _i_, _j_, _k_ is the
ground-truth value of the drug pair–cell line synergy. _�_ represents
a set of model parameters. ∥ _�_ ∥ [2] 2 [is a] _[ L]_ [2] [-regularizer to prevent]
overfitting. The hyperparameter _λ_ is used to balance the regularizer. The model training process is optimized using the Adam
optimizer.


EXPERIMENTS AND RESULTS


In this section, we compare our results with recent methods on
two datasets to evaluate the performance of KGANSynergy’s drug
combination synergy prediction.


**Comparison with previous studies**


To demonstrate the effectiveness of KGANSynergy, we compared
it with some state-of-the-art and baseline methods, including
DeepWalk, GCN, Deepsynergy, KGNN and GraphSynergy. In addition, we adopted the stratified nested cross–validation, and all test



� _π(h_, _r_, _t)e_ _[(]_ _t_ _[l][)]_

_(h_, _r_, _t)_ ∈ _N_ _h_



_e_ _[(]_ _N_ _[l][)]_ _h_ [=][ LeakyReLU]



⎛



∥ _[M]_ m=1 �
⎝ _(h_, _r_, _t)_ ∈



⎞

, (10)
⎠



where _e_ _[(]_ _h_ _[l][)]_ [and] _[ e]_ _N_ _[(][l][)]_ _h_ [denote entity] _[ h]_ [ embedding and the embedding]
set of entity _h_ ’s neighbors at layer _l_, respectively. The _l_ -th layer
entity _h_ is the neighbor of entity at layer _l_ − 1. The embedding of
entity _h_ at layer _l_ is obtained by aggregating the embedding of its
neighbor entities at layer _l_ − 1. _e_ _[(]_ _t_ _[l][)]_ [is the embedding representation]
of entity _h_ ’s neighbor _t_, whose embedding has information from
its _(l_ − 1 _)_ -layer neighbors.

After executing _L_ KG attention layers, we can obtain multiple
embedding representations for drug _d_ and cell line _k_ . Importantly,
the initial entities of drugs and cell lines in the KG are the closest
nodes in the potential space to the drugs and cell lines with a
strong association. Therefore, it is also important to enrich the
entity embedding with the original item representation for each
drug and cell line. The initial embedding vector of entity is defined
by random initialization, and the length of the embedding is one
parameter set in our study, which can be optimized based on
experiments. The resulting drug embedding representation set
( _T_ _d_ ) and cell line embedding representation set ( _T_ _k_ ) are as follows:


_T_ _d_ = �e _d_ _[(]_ [0] _[)]_ [,][ e] _d_ _[(]_ [1] _[)]_ [,] _[ . . .]_ [,][ e] _d_ _[(][L][)]_ �, (11)


_T_ _k_ = �e _k_ _[(]_ [0] _[)]_ [,][ e] _k_ _[(]_ [1] _[)]_ [,] _[ . . .]_ [,][ e] _k_ _[(][L][)]_ �, (12)



where _e_ _d_ _[(]_ [0] _[)]_ or _e_ _k_ _[(]_ [0] _[)]_ is the initial embedding of the drug or cell line.
� _e_ _d_ _[(]_ [1] _[)]_ [,] _[ . . .]_ [,] _[ e]_ _[(]_ _d_ _[L][)]_ � and � _e_ _k_ _[(]_ [1] _[)]_ [,] _[ . . .]_ [,] _[ e]_ _k_ _[(][L][)]_ � are the embeddings learned from

the first KG attention layer to the last layer. Next, we define the
original representation of the drug and cell line as follows:



and



� _e_ _k_ _[(]_ [1] _[)]_ [,] _[ . . .]_ [,] _[ e]_ _k_ _[(][L][)]_ �



are the embeddings learned from



_e_ [0] _d_ [=]


_e_ [0] _k_ [=]



� _e_ ∈{ _e_ | _(e_, _d)_ ∈ _B)_ } _[e]_ (13)

|{ _e_ | _(e_, _d)_ ∈ _B_ }| [,]



� _e_ ∈{ _e_ | _(e_, _k)_ ∈ _B)_ } _[e]_ (14)

|{ _e_ | _(e_, _k)_ ∈ _B_ }| [,]



where _B_ represents the entity set obtained after mapping the
entity _d_ in the drug combination dataset to the entity _e_ in the KG,
_(e_, _d)_ and _(e_, _k)_ ∈ _B_ . _d_ ∈ _D_ (D denotes all drugs) and _k_ ∈ _K(K_ denotes
all cell lines) denote a drug and a cell line, respectively.

Using the multi-head attention network in the KG can help
extract the topological neighbor structure of the entities in the
neighbor set, learn the high-order entity representation and construct a refined neighbor set embedding representation.


8 | _Zhang_ et al.


**Table 3.** Performance comparison of KGANSynergy and baselines


**DrugCombDB** **Oncology-Screen**


**Model** **AUC-ROC** **AUC-PR** **ACC** **AUC-ROC** **AUC-PR** **ACC**


DeepWalk 0.7101 0.6923 0.6674 0.6987 0.6785 0.6496

GCN 0.6961 0.6797 0.6531 0.6748 0.7076 0.6534

Deepsynergy 0.7436 0.7192 0.6851 0.7360 0.7205 0.6695

KGNN 0.7610 0.7649 0.7146 0.7283 0.7478 0.6749

GraphSynergy 0.7882 0.7725 0.7216 0.7801 0.7945 0.6776
KGANSynergy **0.8951** **0.8921** **0.8174** **0.8911** **0.8983** **0.8221**


The bold values mean the best result for each performance metric.



drug pairs have not appeared in the training set. Below are brief
descriptions of these comparison methods.


(1) **DeepWalk** [53]: Random walk-based methods are frequently
used for link prediction and knowledge representation.
Here, we used DeepWalk, an algorithm for mining graphstructured data that combines the random walk and

Word2vec [54] algorithms. The algorithm can discover the
network’s hidden information and represent the graph’s
nodes as vectors holding latent information.
(2) **GCN** [55]: GCN is a multilayer graph convolutional neural
network that incorporates graph structures into convolution.
(3) **KGNN** [38]: KGNN is a KG-based GNN framework that is
mostly used in drug–drug interaction prediction tasks.
(4) **Deepsynergy** [25]: Deepsynergy is a deep-learning-based
drug synergy prediction method that utilizes the drug
chemical structure and genetic information as input and
uses conical layers to model drug synergy.
(5) **GraphSynergy** [31]: GraphSynergy is a state-of-the-art drug
synergy prediction method that utilizes a spatial graphbased convolutional network and attention mechanism to

encode higher order structural information of drug and cell
line protein modules to enrich entity embedding representations.


The prediction results of the two datasets are shown in Table 3.
From the results, KGANSynergy has achieved AUC value of 0.895
and AUPR value of 0.892 on the DrugCombDB dataset, and AUC
value of 0.891 and AUPR value of 0.898 on the Oncology-Screen
dataset, which are superior to other methods. Specifically, KGANSynergy achieves superior prediction performance compared to
GraphSynergy because the combination of KG and multi-head
attention stabilizes the capability to capture node embeddings. It
contains multi-hop nodes and enhances entity embedding representations. Moreover, the aggregated neighbor nodes contain
not only protein information but also other neighbor information in the KG, which demonstrates the effectiveness of the KG
and graph attention strategies for predicting drug combination
synergy. GraphSynergy and KGANSynergy may have performed
better than KGNN because they explore drugs, cell line features
and related entities. In addition, they acquire the embedding
representation of each node by distinguishing the significance of
neighboring information using an attention mechanism. KGNN
surpasses GCN and DeepSynergy because KG and higher order
neighbors enable better exploration of entity representations.


**Parameter sensitivity analysis**
_Effect of dimension of embedding_


We evaluated the performance of KGANSynergy by changing the
dimension of entity embedding in the Oncology-Screen dataset.



As shown in Figure 5, the _AUC_ and _AUPR_ are at their maximum when the embedding dimension is set to 64. Then, _AUC_
and _AUPR_ decrease steadily as the entity embedding dimension
increases. This result indicates that increasing the dimension of
entity embedding within a particular range can efficiently encode
more KG information, while exceeding the threshold may result
in overfitting. Therefore, _AUC_ and _AUPR_ present a trend of first
rising and then falling.


_Effect of knowledge propagation layers depths_


We evaluated the influence of knowledge propagation layer depth
on the Oncology-Screen dataset by changing the number of layers
( _L_ ) of KGANSynergy. As shown in Figure 6, the experiments are in
the range of {1, 2, 3, 4}.

The findings indicate that the optimal performance is reached
when _L_ is 3. The phenomenon may be the result of a trade-off
between the positive signal’s dependence on distance and the
negative signal’s noise. Specifically, when only first-order neighborhood entities are considered, correlations and dependencies
between entities are not fully considered. As _L_ is too large, the
model provides more knowledge information but may introduce
more noise.


_Effect of aggregators_


To explore the impact of multi-head attention mechanism aggregators, we conducted experiments on KGANSynergy using different aggregators. Here, we refer to the average and concat aggregators as KGANSynergy-avg and KGANSynergy-concat, respectively.
As seen in Figure 7, the best result is obtained by using the concat

aggregator.


_Effect of the number of attention heads_


We examined the effect of the number of attentional heads on the

model. As shown in Figure 8, _AUC_ and _AUPR_ all display a trend
of rising first and then falling. The KGANSynergy performance
reaches its maximum when the number of attention heads is 2.

The results indicate that the number of attention heads influ
ences the performance of the model. This is because multiple
heads are equivalent to multiple single heads acting together to
stabilize the learning process. A single attention head is not sufficient to capture all of the semantic information about an entity,
while using too many attention heads may introduce redundant
information.


**Ablation study**


We verify how different parts of KGANSynergy affect the performance through an ablation study and design the following several
of its variants:


Figure 5. Embedding dimension of AUC and AUPR on Oncology-Screen.


Figure 6. Impact of different layers on the model performance.


Figure 7. Effect of different attention aggregators on model performance.


(1) KGANSynergy/-att: KGANSynergy without multi-head attention mechanism to update the entity representation and set
_π(h_, _r_, _t)_ as 1 _/_ | _N_ _h_ |.
(2) KGANSynergy/-m: KGANSynergy without multi-head attention mechanism to update the entity representation and
replaced by single-head attention.
(3) KGANSynergy/-d: KGANSynergy without neighbor entity
embedding and replaced by the original embedding of the
entity.


In addition, considering the guilt-by association principle,
we removed protein–protein information and cell line-tissue



_KGANSynergy_ | 9


**Table 4.** Performance comparison between different variants


**Methods** **AUC-ROC** **AUC-PR** **ACC**


KGANSynergy/-att 0.8762 0.8608 0.8155
KGANSynergy/-m 0.8882 0.8976 0.8132
KGANSynergy/-d 0.8396 0.8471 0.7662
KGANSynergy/-q 0.8581 0.8675 0.7880
KGANSynergy 0.8911 0.8983 0.8221


information, termed KGANSynergy/-q. We compared KGANSynergy with several of its variants, and the results are given in
Table 4. The performance achieved by the model in different cases
can be summarized as follows:


 - Removing the attention mechanism or ignoring the knowledge graph neighbor entities can degrade the model performance. Moreover, KGANSynergy/-d is consistently the worst
performer. This is because KGANSynergy/-d solely employs
entity embedding and disregards neighbor entities and attention mechanisms. This reflects the importance and necessity
of considering attention mechanisms and higher order neighbors.

 - KGANSynergy/-m always performs better than KGANSynergy/-att. This may be because KGANSynergy/-att sets the
weights of all triads to the same value during propagation,
making it impossible to distinguish the different contributions of the triads. Thus, the addition of the attention mechanism aids in determining the weights of neighboring messages. Additionally, the result of KGANSynergy/-m indicates
the superiority of the multi-head attention mechanism. Compared with the single-head attention mechanism, the multihead attention mechanism enriches the model’s capability
and stabilizes the training procedure.


**Case studies**


To further examine the effectiveness of KGANSynergy, we forecast
new drug combinations on the Oncology-Screen dataset. We used
drugs and cell lines from Oncology-Screen to build drug candidate
combinations and excluded existing drug pairs from the raw
training data. Then, we utilized the training model to predict drug
pairs with unknown synergy status and ranked all prospective
drug pair–cell line combinations based on their prediction scores.
The model predicted specific drug combinations in the cell lines
and finally selected drug pairs with high prediction synergy scores
in each tissue. We select three important cancers – lung, ovary and


10 | _Zhang_ et al.


**Table 5.** Top synergistic drug pairs predicted by KGANSynergy


**Cell line** **Tissue** **Drug 1** **Drug 2** **PMID**


NCIH2122 Lung ERLOTINIB ABT-888 22005537
NCIH2122 Lung METHOTREXATE ERLOTINIB NA
NCIH2122 Lung METHOTREXATE LAPATINIB NA
SKMES1 Lung LAPATINIB ZOLINZA 25896603
SKOV3 Ovary DEXAMETHASONE ETOPOSIDE 22932097
SKOV3 Ovary DEXAMETHASONE 5-FU 34575034
SKOV3 Ovary DEXAMETHASONE VINORELBINE 25462205
CAOV3 Ovary SUNITINIB ERLOTINIB 24041628

KPL1 Breast ABT-888 SN-38 26842236

KPL1 Breast ZOLINZA SN-38 26571493

KPL1 Breast ETOPOSIDE SN-38 31383812

KPL1 Breast ETOPOSIDE ABT-888 30327308


Figure 8. Effect of different number of attention heads on model perfor
mance.



breast for case study. Lung cancer is the cancer with the highest
mortality rate. Breast cancer and ovarian cancer are two leading
malignant tumors of women. We examined the top four drug pairs
based on predicted scores for those three cancers, respectively
(Table 5).

Glucocorticoids (GCs) like dexamethasone may be able to minimize acute toxicity or protect normal tissues and have been
widely employed as combination agents in the treatment of solid
cancers [56–59]. Erlotinib is a quinazoline derivative that selectively and reversibly inhibits the TK activity of EGFR. It is active
in advanced non-small cell lung cancer, head and neck tumors,
glioblastoma and other types of tumors [60]. SN-38 (an active
metabolite of Irinotecan) is a camptothecin derivative targeting
topoisomerase 1 and is primarily used in combination regimens
for the treatment of metastatic or advanced solid tumors [61]. To
further evaluate the accuracy of these predicted drug combinations, we undertook a comprehensive literature survey. For example, dexamethasone treatment in SKOV3 significantly increased
SGK1 mRNA expression [62]. Studies have shown that dexamethasone sensitizes cancer stem cells (CSCs) to 5-FU by reducing
NRF2 and increasing reactive oxygen species production [63]. The
same mechanism works in ovarian and colonic CSCs treated

with the combination of dexamethasone and chemotherapeutic
agents. Vinorelbine induces apoptosis and reduces telomerase
activity in the human ovarian epithelial carcinoma cell line SKOV3

[64]. A phase II study demonstrated that gemcitabine, vincristine



Figure 9. Heatmap of synergistic inhibition analysis of Lapatinib and
Zolinza on SKMES1.


and dexamethasone are effective regimens for the treatment of
relapsed/refractory Hodgkin’s lymphoma (RRHL) with acceptable
toxicity [65]. Both methotrexate and erlotinib have proven their
value in treating lung cancer alone, but their combination has
not attracted much attention [66, 67]. Subsequently, we used Syn[ergxDB (https://www.SYNERGxDB.ca/) [68] to further evaluate the](https://www.SYNERGxDB.ca/)
inhibitory effect of our predicted drug combination on cancer cells
at different concentrations. For example, the combination of lapatinib and zolinza was observed to better inhibit lung cancer cell
activity through the heatmap (Figure 9). Based on these results,
we believe that KGANSynergy’s prediction results are consistent
with many previous studies, and have a strong ability to predict
the ability to give candidate drugs.


CONCLUSION


In recent years, drug combination therapy has been successfully
applied to the treatment of many complex diseases. Synergistic
drugs can increase the efficacy and reduce the dose of a single drug. In this study, we proposed a novel drug combination
synergy prediction model, KGANSynergy. Based on known drug
and cell line information, KGANSynergy enhances the stability


of the traditional graph attention mechanism on the KG using
a multi-head attention mechanism to update entity embedding
representation. First, the method used a KG to better explore
entity representations and enhance predictive capabilities. Second, entity embedding used KG attention to enrich itself through
entity neighbor information and stabilized the attention learning
process. Experiments on two datasets showed that KGANSynergy
outperformed existing methods in predicting drug combinations.

The enrichment of biological data may help improve model
performance. Therefore, our future work will focus on how to
combine KGs with other biological information to extract highquality drug and cell line embedding representations. For example, we can consider other types of multi-omics data (e.g. methylation, copy number and pathway activity) to enhance the model
interpretability.


**Key Points**


  - This paper proposes a novel KG attention network
framework called KGANSynergy that can obtain highorder information from multi-source neighbors in the
KG efficiently and explicitly.

  - KGANSynergy uses a neural-network-based multi-head
attention method to completely utilize neighborhood
information and improve the interpretability of the
model.

  - KGANSynergy applies different types of aggregators in
terms of neighboring nodes and hierarchical aggregation, highlighting different higher order connections and
similarity.


ACKNOWLEDGMENTS


We thank LetPub (www.letpub.com) for its linguistic assistance
during the preparation of this manuscript.


FUNDING


This work was supported by the National Natural Science Foundation of China (grant nos 61802113 and 61802114); the Education
Department of Henan Province (grant no. 222102210238) and
the Science and Technology Development Plan Project of Henan
Province (grant no. 212102210091).


DATA AVAILABILITY


The implementation of KGANSynergy and the datasets are avail[able at https://github.com/juanerzz7/KGANSynergy.](https://github.com/juanerzz7/KGANSynergy)


AUTHORS’ CONTRIBUTIONS STATEMENT


G.Z. and Z.G. conceived the main idea and the framework of the

manuscript. Z.G. drafted the manuscript. Z.G. and H.L. collected
the data and performed the experiments. C.Y., W.L. and J.W. helped
to improve the idea and the manuscript. J.L. reviewed drafts of the
paper. All authors read and commented on the manuscript.


REFERENCES


1. Bray F, Ferlay J, Soerjomataram I, _et al._ Global cancer statistics
2018: Globocan estimates of incidence and mortality worldwide



_KGANSynergy_ | 11


for 36 cancers in 185 countries. _CA Cancer J Clin_ 2018; **68** (6):

394–424.

2. Tan X, Long H, Luquette LJ, _et al._ Systematic identification of
synergistic drug pairs targeting hiv. _Nat Biotechnol_ 2012; **30** (11):

1125–30.

3. Giles TD, Weber MA, Basile J, _et al._ Efficacy and safety of
nebivolol and valsartan as fixed-dose combination in hypertension: a randomised, multicentre study. _Lancet_ 2014; **383** (9932):

1889–98.

4. Barabási A-L, Gulbahce N, Loscalzo J. Network medicine: a
network-based approach to human disease. _Nat Rev Genet_ 2011;
**12** (1): 56–68.
5. Humphrey RW, Brockway-Lunardi LM, Bonk DT, _et al._ Opportunities and challenges in the development of experimental
drug combinations for cancer. _J Natl Cancer Inst_ 2011; **103** (16):

1222–6.

6. Jia J, Zhu F, Ma X, _et al._ Mechanisms of drug combinations:
interaction and network perspectives. _Nat Rev Drug Discov_ 2009;
**8** (2): 111–28.
7. Sun W, Sanderson PE, Zheng W. Drug combination therapy
increases successful drug repositioning. _Drug Discov Today_ 2016;
**21** (7): 1189–95.
8. Wilson DR, Honrath U, Sonnenberg H. Interaction of amiloride
and hydrochlorothiazide with atrial natriuretic factor in the
medullary collecting duct. _Can J Physiol Pharmacol_ 1988; **66** (5):

648–54.

9. Skolnik NS, Beck JD, Clark M. Combination antihypertensive
drugs: recommendations for use. _Am Fam Physician_ 2000; **61** (10):

3049.

10. Menzies AM, Long GV. Dabrafenib and trametinib, alone and in

combination for braf-mutant metastatic melanoma. _Clin Cancer_

_Res_ 2014; **20** (8): 2035–43.
11. Fitzgerald JB,Schoeberl B,Nielsen UB,Sorger PK.Systems biology
and combination therapy in the quest for clinical efficacy. _Nat_
_Chem Biol_ 2006; **2** (9): 458–66.
12. Pang K, Wan Y-W, Choi WT, _et al._ Combinatorial therapy discovery using mixed integer linear programming. _Bioinformatics_ 2014;
**30** (10): 1456–63.
13. Day D, Siu LL. Approaches to modernize the combination drug
development paradigm. _Genome Med_ 2016; **8** (1): 1–14.
14. He L, Kulesskiy E, Saarela J, _et al._ Methods for high-throughput
drug combination screening and synergy scoring. In: von Stechow, L. (eds) _Cancer Systems Biology_ . Humana Press, New York,
NY: Springer, 2018, 351–98.
15. Bulusu KC, Guha R, Mason DJ, _et al._ Modelling of compound
combination effects and applications to efficacy and toxicity:
state-of-the-art, challenges and perspectives. _Drug Discov Today_
2016; **21** (2): 225–38.
16. Morris MK, Clarke DC, Osimiri LC, Lauffenburger DA. Systematic
analysis of quantitative logic model ensembles predicts drug
combination effects on cell signaling networks. _CPT Pharmaco-_
_metrics Syst Pharmacol_ 2016; **5** (10): 544–53.
17. Lianlian W, Wen Y, Leng D, _et al._ Machine learning methods,
databases and tools for drug combination prediction. _Brief Bioin-_
_form_ 2022; **23** (1): bbab355.
18. Cheng F, Kovács IA, Barabási A-L. Network-based prediction of
drug combinations. _Nat Commun_ 2019; **10** (1): 1–11.
19. Sun X, Bao J, You Z, _et al._ Modeling of signaling crosstalkmediated drug resistance and its implications on drug combination. _Oncotarget_ 2016; **7** (39): 63995.
20. Wildenhain J, Spitzer M, Dolma S, _et al._ Prediction of synergism
from chemical-genetic interactions by machine learning. _Cell_
_Systems_ 2015; **1** (6): 383–95.


12 | _Zhang_ et al.


21. Doucet J-P, Barbault F, Xia H, _et al._ Nonlinear SVM approaches
to QSQR/QSAR studies and drug design. _Curr Comput Aided Drug_
_Des_ 2007; **3** (4): 263–89.
22. Li P, Huang C, Yingxue F, _et al._ Large-scale exploration and analysis of drug combinations. _Bioinformatics_ 2015; **31** (12): 2007–16.
23. LeCun Y, Bengio Y, Hinton G. Deep learning. _Nature_ 2015;
**521** (7553): 436–44.
24. Luo H, Li M, Yang M, _et al._ Biomedical data and computational
models for drug repositioning: a comprehensive review. _Brief_
_Bioinform_ 2021; **22** (2): 1604–19.
25. Preuer K, Lewis RPI, Hochreiter S, _et al._ Deepsynergy: predicting
anti-cancer drug synergy with deep learning. _Bioinformatics_ 2018;
**34** (9): 1538–46.
26. Kuru HI, Tastan O, Cicek E. Matchmaker: a deep learning framework for drug synergy prediction. _IEEE/ACM Trans Comput Biol_
_Bioinform_ 2021; **19** :2334–44.
27. Wang J, Liu X, Shen S, _et al._ Deepdds: deep graph neural network
with attention mechanism to predict synergistic drug combinations. _Brief Bioinform_ 2022; **23** (1): bbab390.
28. Ma J, Wang J, Ghoraie LS, _et al._ A comparative study of cluster
detection algorithms in protein–protein interaction for drug
target discovery and drug repurposing. _Front Pharmacol_ 2019; **10** :

109.

29. Qian X, Xiong Y, Dai H, _et al._ Pdc-sgb: prediction of effective drug
combinations using a stochastic gradient boosting algorithm. _J_
_Theor Biol_ 2017; **417** :1–7.

30. Liu Q, Xie L. Transynergy: mechanism-driven interpretable deep
neural network for the synergistic prediction and pathway
deconvolution of drug combinations. _PLoS Comput Biol_ 2021;
**17** (2): e1008653.
31. Yang J, Zhongzhi X, William Ka Kei W, _et al._ Graphsynergy:
a network-inspired deep learning model for anticancer drug
combination prediction. _J Am Med Inform Assoc_ 2021; **28** (11):

2336–45.

32. Dai Y, Wang S, Xiong NN, Guo W. A survey on knowledge
graph embedding: approaches, applications and benchmarks.
_Electronics_ 2020; **9** (5): 750.
33. Wang H, Zhao M, Xie X, _et al._ Knowledge graph convolutional
networks for recommender systems. In: _The World Wide Web_
_Conference_ . New York, NY, USA: ACM, 2019;3307–13.
34. Wang Z, Lin G, Tan H, Chen Q, and Liu X. Ckan:
collaborative knowledge-aware attentive network for
recommender systems. In: _Proceedings of the 43rd International_
_ACM_ _SIGIR_ _Conference_ _on_ _Research_ _and_ _Development_ _in_
_Information Retrieval_ . New York, NY, USA: ACM, pp. 219–28,

2020.

35. Wang X, He X, Cao Y, Liu M, and Chua T-S. Kgat: knowledge graph attention network for recommendation. In: _Proceed-_
_ings of the 25th ACM SIGKDD International Conference on Knowl-_
_edge Discovery & Data Mining_, pp. 950–8, 2019. ACM,New York,

NY, USA.

36. Zeng X, Xinqi T, Liu Y, _et al._ Toward better drug discovery with
knowledge graph. _Curr Opin Struct Biol_ 2022; **72** :114–26.
37. Bonner S, Barrett IP, Ye C, _et al._ Understanding the performance
of knowledge graph embeddings in drug discovery. _Artificial_
_Intelligence in the Life Sciences_ 2022; **2** :100036.
38. Lin X, Quan Z, Wang Z-J, _et al._ Kgnn: knowledge graph neural
network for drug-drug interaction prediction. _In: IJCAI_ 2020; **380** :

2739–45.

39. Mohamed SK, Novácek V, Nounu A. Discovering protein drugˇ
targets using knowledge graph embeddings. _Bioinformatics_ 2020;
**36** (2): 603–10.



40. Velickovi´c P, Cucurull G, Casanova A,ˇ _et al._ Graph attention
networks. _stat_ 2017; **1050** (20):10–48550.
41. Liu H, Zhang W, Zou B, _et al._ Drugcombdb: a comprehensive
database of drug combinations toward the discovery of combinatorial therapy. _Nucleic Acids Res_ 2020; **48** (D1): D871–81.
42. O’Neil J, Benita Y, Feldman I, _et al._ An unbiased oncology compound screen to identify novel combination strategies. _Mol Can-_
_cer Ther_ 2016; **15** (6): 1155–62.
43. Loewe S. The problem of synergism and antagonism of combined drugs. _Arzneimittelforschung_ 1953; **3** :285–90.
44. Bliss CI. The toxicity of poisons applied jointly 1. _Ann Applied_
_Biology_ 1939; **26** (3): 585–615.
45. Borisy AA, Elliott PJ, Hurst NW, _et al._ Systematic discovery
of multicomponent therapeutics. _Proc Natl Acad Sci_, **100** (13):

7977–82, 2003.

46. Yadav B, Wennerberg K, Aittokallio T, Tang J. Searching for
drug synergy in complex dose–response landscapes using an
interaction potency model. _Comput Struct Biotechnol J_ 2015; **13** :

504–13.

47. Di Veroli GY, Fornari C, Wang D, _et al._ Combenefit: an interactive
platform for the analysis and visualization of drug combinations. _Bioinformatics_ 2016; **32** (18): 2866–8.
48. Segal E, Wang H, Koller D. Discovering molecular pathways
from protein interaction and gene expression data. _Bioinformatics_
2003; **19** (suppl_1): i264–72.
49. Barretina J, Caponigro G, Stransky N, _et al._ The cancer cell line
encyclopedia enables predictive modelling of anticancer drug
sensitivity. _Nature_ 2012; **483** (7391): 603–7.
50. Rouillard AD, Gundersen GW, Fernandez NF, _et al._ The har
monizome: a collection of processed datasets gathered to
serve and mine knowledge about genes and proteins. _Database_

2016; **2016** :1–16.

51. Safran M, Dalah I, Alexander J, _et al._ Genecards version 3: the
human gene integrator. _Database_ 2010; **2010** :baq020.
52. Ruopp MD, Perkins NJ, Whitcomb BW, Schisterman EF. Youden
index and optimal cut-point estimated from observations
affected by a lower limit of detection. _Biometrical J: J Math Methods_
_Biosci_ 2008; **50** (3): 419–30.
53. Perozzi B, Al-Rfou R, and Skiena S. Deepwalk: online learning
of social representations. In: _Proceedings of the 20th ACM SIGKDD_
_International Conference on Knowledge Discovery and Data Mining_
_(KDD’14)_ . New York, USA: ACM, pp. 701–10, 2014.
54. Mikolov T, Chen K, Corrado G, Dean J. Efficient estimation of
word representations in vector space. In _Proc. of ICLR Workshops_ .
CoRR abs/1301.3781. 2013.

55. Kipf TN, Welling M. Semi-supervised classification with graph
convolutional networks. In: _Proceedings of the International Con-_
_ference on Learning Representationss (ICLR), Toulon_ . arXiv preprint

arXiv:1609.02907. 2016.

56. Rutz HP, Herr I. Interference of glucocorticoids with apoptosis
signaling and host-tumor interactions. _Cancer Biol Ther_ 2004; **3** (8):

715–8.

57. Rutz HP. Effects of corticosteroid use on treatment of solid

tumours. _Lancet_ 2002; **360** (9349): 1969–70.
58. Xing K, Bingxin G, Zhang P, Xianghua W. Dexamethasone
enhances programmed cell death 1 (pd-1) expression during
t cell activation: an insight into the optimum application of
glucocorticoids in anti-cancer therapy. _BMC Immunol_ 2015; **16** (1):

1–9.

59. Herr I, Pfitzenmaier J. Glucocorticoid use in prostate cancer and
other solid tumours: implications for effectiveness of cytotoxic
treatment and metastases. _Lancet Oncol_ 2006; **7** (5): 425–30.


60. Bareschino MA, Schettino C, Troiani T, _et al._ Erlotinib in cancer

treatment. _Ann Oncol_ 2007; **18** :vi35–41.

61. Bailly C. Irinotecan: 25 years of cancer treatment. _Pharmacol Res_

2019; **148** :104398.

62. Amal Melhem S, Yamada D, Fleming GF, _et al._ Administration
of glucocorticoids to ovarian cancer patients is associated with
expression of the anti-apoptotic genes sgk1 and mkp1/dusp1 in
ovarian tissues. _Clin Cancer Res_ 2009; **15** (9): 3196–204.
63. Suzuki S, Yamamoto M, Sanomachi T, _et al._ Dexamethasone

sensitizes cancer stem cells to gemcitabine and 5-fluorouracil
by increasing reactive oxygen species production through nrf2
reduction. _Life_ 2021; **11** (9): 885.
64. Li-Yuan SHEN, Dong-Mei XU, Xiao-Han LIU, _et al._ Vinorelbine
induces apotosis and decreases telomerase activity in human
epithelial ovarian cancer cells line skov3. _Basic Clin Med_ 2018;
**38** (1): 87.



_KGANSynergy_ | 13


65. Ganesan P, Mehra N, Joel A, _et al._ Gemcitabine, vinorelbine and dexamethasone: a safe and effective regimen for
treatment of relapsed/refractory hodgkin’s lymphoma. _Leuk Res_

2019; **84** :106188.

66. Abdelrady H, Hathout RM, Osman R, _et al._ Exploiting gelatin
nanocarriers in the pulmonary delivery of methotrexate for lung
cancer therapy. _Eur J Pharm Sci_ 2019; **133** :115–26.
67. Saito H, Fukuhara T, Furuya N, _et al._ Erlotinib plus bevacizumab
versus erlotinib alone in patients with egfr-positive advanced
non-squamous non-small-cell lung cancer (nej026): interim
analysis of an open-label, randomised, multicentre, phase 3
trial. _Lancet Oncol_ 2019; **20** (5): 625–35.
68. Seo H, Tkachuk D, Ho C, _et al._ Synergxdb: an integrative
pharmacogenomic portal to identify synergistic drug combinations for precision oncology. _Nucleic Acids Res_ 2020; **48** (W1):

W494–501.


