Artificial Intelligence Review (2024) 57:49
https://doi.org/10.1007/s10462-023-10669-z

## **A review on graph neural networks for predicting synergistic** **drug combinations**


**Milad Besharatifard** **[4]** **· Fatemeh Vafaee** **[1,2,3,4]**


Accepted: 20 December 2023 / Published online: 13 February 2024
© The Author(s) 2024


**Abstract**
Combinational therapies with synergistic effects provide a powerful treatment strategy for
tackling complex diseases, particularly malignancies. Discovering these synergistic combinations, often involving various compounds and structures, necessitates exploring a
vast array of compound pairings. However, practical constraints such as cost, feasibility,
and complexity hinder exhaustive in vivo and in vitro experimentation. In recent years,
machine learning methods have made significant inroads in pharmacology. Among these,
Graph Neural Networks (GNNs) have gained increasing attention in drug discovery due
to their ability to represent complex molecular structures as networks, capture vital structural information, and seamlessly handle diverse data types. This review aims to provide a
comprehensive overview of various GNN models developed for predicting effective drug
combinations, examining the limitations and strengths of different models, and comparing
their predictive performance. Additionally, we discuss the datasets used for drug synergism
prediction and the extraction of drug-related information as predictive features. By summarizing the state-of-the-art GNN-driven drug combination prediction, this review aims to
offer valuable insights into the promising field of computational pharmacotherapy.


**Keywords** Graph neural networks · Drug combination · Synergy prediction · Cancer

treatment


**1 Introduction**


Combination therapy, a treatment modality that combines two or more therapeutic agents,
has increasingly become the preferred approach for many human diseases, especially
those caused by alterations in multiple genes or pathways, such as cancer. The integration


- Fatemeh Vafaee

f.vafaee@unsw.edu.au


1 School of Biotechnology and Biomolecular Sciences, University of New South Wales (UNSW),
Sydney, Australia


2 UNSW Data Science Hub, University of New South Wales (UNSW), Sydney, Australia


3 OmniOmics Pty Ltd, Sydney, Australia


4 Biomedical AI Laboratory (Vafaee Lab), Sydney, Australia

# Vol.:(0123456789) 1 3


**49** Page 2 of 38



M. Besharatifard, F. Vafaee



of anti-cancer drugs enhances efficacy compared to using a single therapy, as it targets
different key pathways in a synergistic or additive manner. By combining drugs with distinct
mechanisms of action, therapeutic effectiveness can be enhanced, allowing for lower-dose
prescriptions, and reducing the potential risks of side effects and toxicity. Clinical evidence
consistently demonstrates the utility of combining different therapeutics to improve treatment
efficacy in various cancer types, such as breast cancer (Fisusi and Akala 2019), lung cancer
(Molina-Arcas et al. 2019), and ovarian cancer (Lui et al. 2020), among others.
However, the search for effective combinations is hindered by the sheer number of potential drug pairs, leading to a combinatorial explosion (Azad et al. 2021b; Gilvary et al. 2019).
It is infeasible to experimentally screen the enormous search space of all possible drug
combinations. Consequently, the development of computational models to identify potential anti-cancer synergistic drug combinations efficiently and accurately has garnered significant attention from both the scientific community and the pharmaceutical industry. With
the increasing availability of large-scale high-throughput screening datasets for identifying
synergistic drug combinations, a growing number of artificial intelligence (AI) methods are
being employed for in silico predictions of efficacious drug combinations (Hosseini and
Zhou 2023; Hu et al. 2022; Zhang et al. 2023; Zhang and Tu 2023; Wang et al. 2022a).
Among different AI models, GNNs have emerged as a powerful class of artificial neural
networks designed to process and learn from data structured as graphs. Graphs consist of
nodes (vertices) connected by edges (links or relationships), and they are widely used to
represent complex relationships and interactions between different entities. Due to their
versatility, GNNs have found applications in various fields, including computer vision,
natural language processing, social network analysis, bioinformatics, and drug discovery,
among others (Zhou et al. 2020).
The increasing importance and application of AI and machine learning in drug discovery have prompted different review articles outlining various data sets, machine learning
algorithms, and deep learning models developed to predict synergistic drug combinations in cancer (Torkamannia et al. 2022; Wu et al. 2022; Pearson et al. 2023; Kumar and
Dogra 2022). For instance, Torkamannia et al. (Torkamannia et al. 2022) comprehensively
reviewed a wide array of drug development data sources, encompassing biological datasets
like molecular omics data, drug target information, and molecular interactions, as well as
datasets containing high-throughput in vitro screening of drug combinations. Additionally,
they presented an overview of the literature on computational methods designed for drug
synergy prediction, broadly categorized into deep learning (DL), traditional machine learning (ML), and network-based methods.
Around the same time, Wu et al. (2022)- performed a similar review of machine learning methods used in drug combination prediction across algorithmic categories of systems
biology or network-based methods, kinetic models, mathematical models, stochastic search
algorithms, classic machine learning, and deep learning methods. They summarized 29
studies, providing details of their respective algorithms, drug combination datasets, input
data types, and the availability of program code.
Further, Kumar et al. (2022) conducted a review focused on deep learning-based techniques for the prediction of synergistic drug combinations in cancer. They performed a
comparative analysis of prediction techniques based on various performance measures.
Additionally, they covered the theoretical aspects of drug synergy and scoring models at
length with their mathematical formulations. However, all these reviews were conducted
before the surge of GNN techniques in drug discovery. Therefore, they neither adequately
cover GNN-related drug combination prediction studies nor represent recent advancements
in GNN algorithms.

# 1 3


A review on graph neural networks for predicting synergistic…



Page 3 of 38 **49**



The rise in the use of GNNs in drug discovery is due to their ability to handle and interpret complex data, such as molecular graphs and biological networks (Bongini et al. 2021;
Zhao et al. 2021). GNN-based models have demonstrated high performance and have
yielded promising results in various aspects of drug discovery, including virtual screening,
molecular property prediction, protein–ligand binding prediction, and drug repurposing
(Son and Kim 2021; Wang et al. 2022c; Krasoulis et al. 2022). While GNNs have shown
promise in capturing relationships and interactions in various domains (Nguyen et al.
2021; Cai et al. 2021a; Zhao et al. 2021
), their specific effectiveness in the context of drug
synergy prediction is still an active area of exploration. Considering the increasing use of
GNNs in drug synergy prediction, their proven efficacy compared to the widely-used highperforming methods (such as MatchMaker (Kuru et al. 2021), DeepSynergy (Preuer et al.
2018), DTF (Sun et al. 2020b), among others) (Hu et al. 2022; Liu et al. 2022; Rozemberczki et al. 2022; Zhang et al. 2023), and the growing importance of drug combination
discovery in both research and industry (Alves et al. 2022), there is a pressing need for a
comprehensive review study that focuses on the latest advances in the field and provide
insights into the future directions.
This study aims to bridge this gap by providing an in-depth review of the advancements,
challenges, and potential of GNN-based approaches in drug synergy prediction. By examining existing literature, we discuss the strengths and limitations. Our review assesses the
effectiveness of GNN-based methods, highlighting their reported performance and competitiveness compared to the state-of-the-art machine learning models. Furthermore, we
emphasize the importance of systematic comparisons to guide researchers and industry
professionals in selecting appropriate methods. This review serves as a valuable resource
for those seeking a deeper understanding of the role and capabilities of GNNs in identifying synergistic drug combinations.


**2 A brief overview of GNNs**


GNNs have emerged as a powerful category of neural networks specifically designed to
process data organized in graph structures. Unlike traditional neural networks which are
primarily tailored for processing vector or matrix data, GNNs excel at capturing intricate
relationships and dependencies between entities within a graph. At the core of GNNs is the
fundamental concept of learning representations for each node by aggregating information
from its neighboring nodes. These representations are then leveraged to perform prediction
and classification tasks. By effectively encapsulating both local and global contexts within
a graph, GNNs enable the modeling of complex interactions and dependencies, making
them highly versatile across a wide range of applications (Zhou et al. 2020). In the following, we outline core mechanisms and commonly used architectures of GNNs.


**2.1 GNN core mechanisms**


The section provides a brief overview of the fundamental components that enable GNNs to
process graph-structured data. These core mechanisms are essential for understanding how
GNNs capture relationships and dependencies within graphs and form the foundation for
various GNN architectures and algorithms.
Message passing function is a key mechanism in GNNs that updates node embeddings
through an iterative process. Each iteration involves two main steps: aggregating messages

# 1 3


**49** Page 4 of 38



M. Besharatifard, F. Vafaee



and updating node embeddings. During message passing, each node gathers information
from its neighboring nodes and combines it into a message. This message carries information from nearby nodes and edges. It is used to update the embedding (i.e., low dimensional numerical representation) of each node _u_ at iteration _k_ denoted as _h_ _[k]_ _u_ [ based on the ]
embeddings of nodes in its neighborhood _N_ ( _u_ ) . This update can be represented as



_h_ _u_ _[k]_ [+][1] = update [(] _h_ _[k]_ _u_ [, aggregate][(] _[h]_ _[k]_ _v_ [,][ ∀] _[v]_ [ ∈] _[N]_ [(] _[u]_ [)]



)) = update



(



)



_h_ _[k]_
_u_ [,] _[ m]_ _[k]_ _N_ ( _u_ )



where update(.) and aggregate(.) are arbitrary differentiable functions (typically neural networks). The message _m_ _N_ ( _u_ ), the output of the aggregate function, encompasses the information gathered from _u_ ’s neighbors in the graph. The aggregation step is responsible for
merging information from neighboring nodes, while the update step iteratively improves
node embeddings across layers. This iterative process enables GNNs to capture intricate
relationships and dependencies within the graph (Hamilton, 2020).
Aggregation function is responsible for combining information from a node’s neighboring nodes to produce a single vector representation. Traditional aggregation methods, such
as summing or averaging over the neighbor embeddings, may fall short in capturing the
intricate nature of the graph structure and relationships between nodes. However, employing more advanced aggregation techniques can enhance the performance of GNNs. One
approach to defining an aggregation function is through the concept of permutation invariant neural networks. This approach treats the set of neighbor embeddings _h_ _v_, ∀ _v_ ∈ _N_ ( _u_ ) as
an unordered set that is invariant to permutations and maps this set to a single vector representation _m_ _N_ ( _u_ ) . A universal set function approximator, as shown by Zaheer et al. (2017), is
an aggregation function that can approximate any permutation-invariant function, mapping
a set of embeddings to a single embedding. This can be represented as



)
)



_m_ _N_ ( _u_ ) = MLP _휃_



∑
( _v_ ∈ _N_ ( _u_ )



MLP
_휙_



( _h_ _v_



where MLP _휃_ and MLP _휙_ are multi-layer perceptron parameterized by trainable parameters _휃_
and _휙_, respectively. Node-level aggregation is a common approach that combines information from neighboring nodes to compute representations for individual nodes. This method
treats nodes as unstructured entities and does not explicitly consider the graph structure
during aggregation. On the other hand, graph-level aggregation takes into account the local
structural information during the aggregation process. It goes beyond simple node-level
aggregation and considers the relationships and connectivity between nodes to perform
higher-order graph aggregation. This results in a more comprehensive and structured representation of the graph (Yang et al. 2022; Cai et al. 2021b, Hamilton, 2020).
Node or graph representations in GNNs involve learning through the aggregation of
information from neighboring nodes. This enables each node to update its representation, capturing both local and global dependencies. For example, the two-dimensional
structure of any chemical compound can be represented as a graph, and its node representation can be articulated through a detailed description of chemical properties and
atomic bonding characteristics. Additionally, employing this representation for drug
candidates and comparing them with existing drugs can be instrumental in identifying
potential drugs with a high probability of success. The essence of node and graph representations in GNNs lies in leveraging neural networks to learn expressive features from
graph-structured data (Khoshraftar and An 2022).

# 1 3


A review on graph neural networks for predicting synergistic…



Page 5 of 38 **49**



Attention mechanism in GNNs is a technique that allows the network to assign different importance weights to nodes or edges within a graph during the aggregation step. This
mechanism enables the network to focus on the most relevant information and adaptively
weigh the influence of different components of the graph. The basic idea behind the attention mechanism is to compute attention weights for each neighbor in the graph, which are
used to weigh their contributions during the aggregation process. These attention weights
are learned by the network and can be based on various factors such as the similarity, relevance, or importance of the neighbors. The attention weights reflect the importance or
significance of each neighbor with respect to the current node being processed. In GNNs,
attention can be applied in various ways, but a common approach is to compute the attention weights as a function of the node features and/or edge connections. This is typically
done using trainable parameters such as weight matrices and attention vectors. The attention weights are then used to compute a weighted sum or aggregation of the neighbor
embeddings, where the weights determine the contribution of each neighbor to the final
aggregated representation (Zhang and Xie 2020, Hamilton, 2020).


**2.2 GNN architectures**


Notable advancements of graph neural networks in recent years have resulted in the development of various architectures that tackle different aspects of graph-structured data.
Below, we provide a summary of some main GNN architectures that have found applications in drug combination prediction.
Graph Convolutional Networks (GCNs) are a variant of Convolutional Neural Networks
(CNNs) designed to operate on graph-structured data. GCNs leverage both the node features and the graph structure to learn a latent representation that captures the underlying
relationships and dependencies within the graph. In GCNs, the input consists of a node
feature matrix _X_, which contains the features of each node, and an adjacency matrix _A_,
which encodes the relationships or similarities between pairs of nodes. The goal is to learn
a latent representation _Z_ that preserves important information from both the node features
and the graph structure. The key idea behind GCNs is to propagate and aggregate information from neighboring nodes to update the node representations. By considering the local
neighborhood information of each node, GCNs capture both the node features and the relationships among nodes (Hell et al. 2020; Liang et al. 2021).
Graph AutoEncoders (GAEs) are unsupervised learning frameworks used to learn
low-dimensional representations of graph-structured data. The core idea behind GAEs is
to encode the graph information into a compact representation and then reconstruct the
original graph from this representation. In GAEs, a GCN is typically used as an encoder to
transform the input graph into a latent representation. The encoder takes into account the
node features and the adjacency matrix of the graph to learn informative node representations. The latent representation, denoted as _Z_, is obtained from the GCN encoder. The goal
of GAEs is to capture the inherent relationships and dependencies within the graph, allowing for meaningful analysis and prediction tasks (Lin et al. 2023; Liang et al. 2021).
Graph Attention Networks (GATs) are neural networks designed to operate on graphstructured data by leveraging the concept of attention. GATs assign different weights,
called attention coefficients, to the neighboring nodes during the process of central node
information aggregation. In GATs, each node in the graph undergoes linear transformations
and is mapped to a learnable vector using a single-layer neural network called the mapping
function _f_ _a_ . The attention coefficient _훼_ _ij_ represents the influence of node _j_ on node _i_ and is

# 1 3


**49** Page 6 of 38



M. Besharatifard, F. Vafaee



calculated based on the transformed node representations. The final output embedding of
the central node is obtained by taking a weighted summation of the representations of its
neighboring nodes. The weights for the summation are determined by the attention coefficients. This allows the GAT to focus on the most relevant and informative neighboring
nodes for each central node (Veličković et al. 2017; Shao et al. 2022).
Graph SAmple and aggreGatE (GraphSAGE) is a general framework for inductive node
embedding, which aims to learn representations for nodes in a graph. Unlike traditional
approaches that rely solely on the graph’s structure, GraphSAGE uses both the topological
structure and the node features to generate embeddings that generalize to unseen nodes.
The core idea behind GraphSAGE is to leverage aggregator functions instead of training
individual embedding vectors for each node. In this way, GraphSAGE can effectively capture and utilize the collective knowledge of a node’s local neighborhood (Hamilton et al.
2017).
Overall, these models collectively contribute to the advancement of graph-based learning in pharmacology by providing insights into complex drug networks and facilitating the
discovery of effective drug combinations (Jiang et al. 2021; Sun et al. 2020a; Nguyen et al.
2021).
Graph Regularization is a technique used in optimization problems to impose desired
properties on solutions with respect to a graph structure. Graph regularization is closely
related to GNNs because both approaches deal with graph-structured data and aim to
capture relationships and interactions between objects represented by nodes in the graph.
GNNs use message passing and aggregation mechanisms to update node embeddings based
on their graph neighborhoods, while graph regularization incorporates graph information
into optimization problems to guide the solutions towards desired properties. Both methods
leverage the inherent graph structure to improve the handling of complex relationships and
interactions within the data. For instance, if we have knowledge that the signal should have
sparsity (i.e., few non-zero values), we can introduce a regularization term that encourages
sparsity in the solution (Lee, 2021).


**3 Drug combination synergy prediction**


The schematic view of drug combination synergy prediction is depicted in Fig. 1. A
drug combination, as defined by the FDA (Food and Administration 2018), involves the
combination of two or more regulated components, such as drugs, devices, or biologics.
These components are physically or chemically mixed to create a single entity. When
multiple drugs are administered simultaneously, a synergistic drug combination occurs,
resulting in a stronger therapeutic effect that surpasses the mere sum of their individual
effects. In simpler terms, the combined impact of these drugs exceeds what would be
expected by merely adding up their individual effects. On the other hand, an additive drug
combination occurs when the combined effect of the drugs is equal to the sum of their
individual effects. In this case, there is no enhancement or reduction in the overall effect
when the drugs are used together. Conversely, an antagonistic drug combination exists
when the combined effect of the drugs is lower than the sum of their individual effects.
This happens when the drugs interfere with or counteract each other, resulting in a lower
overall effect (García-Fuente et al. 2018). In experiments conducted on cancer cell lines,
researchers utilize the in vitro method such as cell culture to assess the impacts of different
combinations of drugs on key aspects such as restraining tumor growth, promoting cancer

# 1 3


A review on graph neural networks for predicting synergistic…



Page 7 of 38 **49**



**Fig. 1** Schematic view of relevant data and the generic pipeline of synergistic drug combination prediction
using GNNs. **a** Datasets of drug combination synergism can be categorized into in vitro screening data
and clinical trials studies (read more in Sect. 4.1). **b** Diverse types of data relevant to drugs, cell lines,
diseases, and patients are often retrieved from multiple datasets, and whenever relevant, complex
biological or chemical relationships are represented as graphs (read more in Sect. 3.3). **c** Various types of
GNNs are then used to extract numerical features for graph representation (read more in Sect. 3.3). **d** The
corresponding features, along with label data, are then used to predict the synergism of drug combinations
as a classification or regression task (read more in Sect. 4.2) and assessed using diverse evaluation metrics
(read more in Sect. 5). Acronyms: AUC: Area under the ROC curve, ACC: Accuracy, AUPR: Area under
the precision-recall curve, F1: F1-score, RMSE: Root mean square error, MSE: Mean square error


cell apoptosis, and preventing metastasis. (Mokhtari et al. 2017). In these studies, cancer
cells are exposed to different concentrations of drug combinations and the collective
effects they produce were analyzed. When drugs synergistically interact, they exhibit
a more pronounced inhibition of cancer cell proliferation, or a heightened rate of cell
death compared to their individual effects. Conversely, an antagonistic combination can
diminish efficacy and potentially undermine the desired therapeutic outcome (Kucuksayan
et al. 2021). In the following, we will explore quantitative measures of drug combination

synergy.


**3.1 Metrics of synergism in drug combinations**


Determining whether a combination of compounds exhibits an interaction effect involves
comparing the observed effects with what would be expected based on a non-interactive

# 1 3


**49** Page 8 of 38



M. Besharatifard, F. Vafaee



(additive) effect. To evaluate the effects of drug combinations and synergy, various
metrics are employed. These metrics provide measurements that help assess the
combined effects of compounds. By utilizing these frequently used measurements,
researchers can determine if the observed effects of a compound combination surpass
what would be expected from an additive effect alone. This allows for a comprehensive
evaluation of the potential interaction and synergy between different compounds in
order to optimize drug combinations for enhanced therapeutic outcomes. Described
below are some commonly used metrics that facilitate this evaluation process (Ianevski
et al. 2020).
**Loewe Additivity**, defined by Loewe in 1926, is based on the principle of sham combination which assumes no interaction effect when a compound is combined with itself
(Loewe, 1953). It is a dose–effect-based concept that is widely used in pharmacology
and toxicology. In pharmacology, a dose–response curve is a graphical representation
of the relationship between the dose of a drug or compound and its effect. The curve
shows how the effect changes as the dose increases. Loewe additivity assumes that the
dose–response curves for two compounds are parallel, meaning that they have the same
shape and slope. This allows for the calculation of an additive effect, which is simply the
sum of the individual effects at a given dose. If the combined experiments are carried
out in concentrations with dose _x_ _i_, we have according to Loewe Additivity principle:



_x_ _i_
= 1
_X_ _i_



∑

_i_ ∈[1, _n_ ]



) = _E_ ( _X_ _n_ ), such that



) = _E_ ( _X_ 2



_E_ [(] _x_ 1, _x_ 2 …, _x_ _n_



) = _E_ ( _X_ 1



_X_ 1, …, _X_ _n_ represent the doses applied to the drugs within individual experiments,
while E stands for the resultant effect (Goldoni and Johansson 2007). The mentioned
combination involves fractions of individual doses that achieve the effect separately.
When these fractions are added together, they sum up to one and result in the same
effect (Lederer et al. 2019). To make this concept clearer imagine two substances: compound A and compound B. Each of these compounds, when administered alone at specific doses, i.e., _X_ 1 and _X_ 2 respectively, produces a desired effect. The idea is that if you



take a fraction ( _[x]_ [1]



(



)




[1]

_X_ 1 [ ) of dose ] _[X]_ [1] [ from compound A and a fraction ]



_x_ 2

_X_ 2



of dose _X_ 2 from



compound B, such that their sum equals 1, then this combination should yield the same
effect as taking dose _X_ 1 from compound B and dose _X_ 2 from compound A. Subsequently,



(



(



)



)



when we have two compounds and their fractions



_x_ 1

_X_ 1



and



_x_ 2

_X_ 2



satisfy the condition of



_x_ 1 _[x]_ [2]
_X_ 1 [+] _X_



_i_ ∈[1, _n_ ]




[2]

_X_ 2 _[<]_ [ 1] [ (i.e., ] [∑]



_x_ _i_
_X_ _i_ _[<]_ [ 1] [ in multi-compound combination experiments) then we ]



consider the effect synergistic meaning that the combined effect of these compounds is
greater than what would be expected if their effects were simply additive. Conversely, if



_x_ 1 _[x]_ [2]
_X_ 1 [+] _X_



_i_ ∈[1, _n_ ]




[2]

_X_ 2 _[>]_ [ 1] [ (i.e., ] [∑]



_x_ _i_
_X_ _i_ _[>]_ [ 1][)] [, the interaction is considered antagonistic, indicating an ]



overall effect less than expected.
Bliss Independence score refers to the concept of Bliss independence (Baeder et al.
2016
), which is a concept used in pharmacology to assess whether the combined effects
of multiple compounds are additive, synergistic, or antagonistic. The main assumption of the Bliss independence criterion is that two or more substances act independently from one another. The Bliss independence criterion is mathematically expressed
through the Bliss equation.


# 1 3


A review on graph neural networks for predicting synergistic…



Page 9 of 38 **49**



Loewe additivity is suitable when the drugs have shared targets, while Bliss independence is more appropriate when each drug targets a distinct pathway (Liu et al.
2018
). For a simplified example with two compounds (A and B), the Bliss equation is:

_E_ _AB_ = _E_ _A_ + _E_ _B_ − _E_ _A_ × _E_ _B_,
Where _E_ _A_ and _E_ _B_ represents the effect of drug A at dose _x_ and drug B at dose _y_, respectively. _E_ _AB_ represents the combined effect of drugs A and B at doses _x_ and _y_ . If the combined effect _E_ _AB_ matches the calculated value from the equation, the compounds are acting
independently. When drug A and drug B are combined, the effect of drug B is modified by
the proportion [(] 1 − _E_ _A_ ) that is “spared” by drug A. By summing up these two terms, i.e.,
_E_ _A_ and _E_ _B_ (1− _E_ _A_ ), we get the expected combined effect _E_ _AB_ .



�



⎧
⎪
⎨
⎪⎩



_<_ 1, _synergism_
= 1, _additive_
_>_ 1,, _antagonism_



_E_ _A_ + _E_ _B_



�1 − _E_ _A_



_E_ _AB_



The above relation indicates that if the value of _E_ _AB_ is greater than _E_ _A_ + _E_ _B_ (1 − _E_ _A_ ),
then it signifies synergism. If the two values are equal, it suggests additivity, and otherwise,
it implies antagonism (Duarte and Vale 2022).
Zero Interaction Potency (ZIP) score is a valuable tool used to assess the synergistic or
antagonistic effects of drug combinations. It combines the strengths of the Loewe and Bliss
models, allowing for a systematic evaluation of various patterns of drug interaction. The
ZIP score provides a numerical value ranging from − 1 to 1, indicating the degree of synergy or antagonism observed in a drug combination. it is derived from the concept of zero
interaction (Sühnel 1992), which assumes that the potency of a drug’s dose–response curve
remains unaltered when combined with another drug (Yadav et al. 2015).
Highest Single Agent (HSA) model, also known as Gaddum’s non-interaction model
(Berenbaum, 1989), provides a simple approach to estimate the expected combination
effect of multiple drugs. According to this model, the expected combination effect is determined by taking the difference between the combined response of the drugs _E_ ( _A_, _B_, _C_,…, _N_ ) and
the maximum response observed among the individual drugs max [(] _E_ _A_, _E_ _B_, _E_ _C_, …, _E_ _N_ ) . In
other words, the HSA model assumes that the combination effect is equal to the highest
response achieved by any single drug at the corresponding concentrations. The HSA model
offers a straightforward way to estimate the expected outcome of drug combinations and
serves as a baseline for assessing whether observed effects deviate from the additive expectation (Lehár et al. 2007).
Each of these metrics has distinctive strengths and limitations, as outlined in previous
reviews (Duarte and Vale 2022). The choice of a specific metric is influenced by various
factors, including the experimental design, biological context, and data availability, as
elaborated further in the Discussion section. Nonetheless, The field of drug combination
synergy analysis is dynamic, with ongoing metric development reflecting deeper insights
into drug interactions (Liu et al. 2018; Lederer et al. 2019).


**3.2 Supervised drug synergy prediction in cancer**


Supervised anticancer drug synergy prediction, driven by machine learning and artificial
intelligence, typically involves training models on two distinct types of datasets: (1) in vitro

# 1 3


**49** Page 10 of 38



M. Besharatifard, F. Vafaee



experiments conducted on various cell lines, evaluating the synergistic effects of different
drug combinations at varying concentrations using diverse reference models (e.g., Loewe,
Bliss, ZIP, HAS), and (2) clinical trial studies of drug combinations in patient populations,
comprising information on clinical response, treatment outcomes, and adverse effects.
In the clinical trial dataset, the prediction task is often treated as a classification problem, where the goal is to determine positive versus negative clinical outcomes for specific
drug combinations. On the other hand, the in vitro experiments yield continuous measures
of synergism, which can be approached as either a regression or a classification problem
after categorizing the synergy measures.
In classification tasks, the synergistic values of drugs on cell lines are grouped into
either two categories (synergistic versus non-synergistic) or three categories (synergistic,
additive, and antagonistic) by applying predefined thresholds to split the data. Nevertheless, establishing the most suitable threshold poses challenges and tends to differ for various synergy measures. For instance, GraphSynergy (Yang et al. 2021) uses the threshold of
zero to binarize Loewe measure with a score greater than or less than 0 indicating a synergistic or non-synergistic effect, respectively. Zhang et al. (Zhang et al. 2023), on the other
hand, categorized Loewe and ZIP synergistic scores using quartiles. The highest quartile
represents synergistic effects, and the lowest quartile represents antagonistic effects. Additionally, some other studies (Wang et al. 2022a; Zhang et al. 2022) consider Loewe synergy
scores above 10 as synergistic and scores below 0 as antagonistic.
Nonetheless, once the class labels have been determined, various machine learning
algorithms, such as random forests (Singh et al. 2018), support vector machines (Preuer
et al. 2018), or neural networks (Preuer et al. 2018), can be employed to perform the classification task. These models learn patterns and relationships from features encompassing
diverse information about drugs and cell lines to make predictions.
Regression, on the other hand, focuses on predicting a quantitative measure of synergy
for each drug combination on a particular cell line. Instead of discrete class labels, regression models estimate the degree or magnitude of synergy, providing continuous output values. Regression techniques, including linear regression (Kuru et al. 2021), gradient boosting, or deep learning approaches (Preuer et al. 2018), have been used to predict the synergy
level of drug combinations.
Some of the commonly employed models for feature extraction from the key factors that
contribute to predicting drug synergy will be discussed below.


**3.3 Feature extraction**


Feature extraction provides the predictive variables for a machine learning-based model,
constituting a crucial step for addressing multifactorial complex problems, such as drug
synergy prediction. The process of feature extraction is preceded by the collection and representation of biological information, as illustrated in Fig. 2 and elaborated below:


_Biological data collection_ Drug synergy prediction algorithms frequently leverage openaccess bioinformatics databases to acquire pertinent biological, chemical, and clinical
information related to drugs, cell lines, and diseases or patients. This encompasses details
about drug chemical structures, drug protein targets, mechanisms of action, cell line gene
expression profiles, molecular interactions (e.g., protein–protein interactions, pathways),
human gene-disease associations, and functional genomics, among others. Comprehensive

# 1 3


A review on graph neural networks for predicting synergistic…



Page 11 of 38 **49**



**Fig. 2** Feature extraction procedure comprising biological data collection (and pre-processing), feature
representation and extraction of contributing (latent) features. Acronyms include MoA: mechanism of
action, EHR: electronic health records, MLP: multi-layer perceptron, GCN: graph convolutional network,
GAT: graph attention, GAE: graph autoencoders


discussions about this information and their respective databases can be found in previous reviews (Torkamannia et al. 2022; Kumar and Dogra 2022; Chen et al. 2015), and
interested readers are referred to those sources. It has been consistently demonstrated that
incorporating diverse types of features, capturing complementary information about drugs,
enhances the prediction performance of drug discovery applications (Wang et al. 2022a, b;
Azad et al. 2021a; Liu et al. 2022; Zhang et al. 2023).
Feature representation: Following the extraction and, when necessary, preprocessing of
biological data (e.g., data normalization in gene expression profiles or extraction of embeddings from protein sequences), this information is represented either as numerical vectors
or graphs. Examples of vector-based encoding include molecular fingerprints, gene expression profiles, one-hot encoding of disease-related genes, protein sequences, among others
(Gan et al. 2023; Liu et al. 2022). In the context of GNN models, the use of network-based
representations has gained popularity. Various types of biological information, such as drug
target interactions, protein–protein interactions, molecular pathways, gene co-expression,
and synergistic drug pairs associated with a particular cell line, have been represented as
graphs. Some studies have even combined different types of interactions and diverse node
types into heterogeneous networks for subsequent feature extraction (Zhang et al. 2023;
Zhang and Tu 2022; Yu et al. 2021).
_Feature extraction_ In general, feature extraction can be categorized into two major
approaches: _feature transformation_ and _feature selection_ (Vijayan et al. 2022). The latter
involves selecting a subset of features as variables for a predictive model, such as choosing
gene expression values related to cancer or identifying mutational signatures in the context
of a specific disease. However, the majority of techniques rely on feature transformation,
wherein statistical methods (e.g., different dimensionality reduction methods (Koch et al.
2021) or singular value decomposition methods (Chen et al. 2022)) or representation learning algorithms (especially neural networks) are used to extract latent features representing complex relationships in data (Gunawan et al. 2023). In the context of GNN models,
multiple graph-based neural networks (e.g., GCN, GAE, and GAT), have been employed
to learn low-dimensional representations from network-based data. These features are then

# 1 3


**49** Page 12 of 38



M. Besharatifard, F. Vafaee



utilized as variables for a predictive model (Liu et al. 2022) or to prioritize drug synergies
based on a ranking mechanism (Jin et al. 2021).


**4 Review of GNN methods for predicting drug synergy**


We conducted a comprehensive search of PubMed, Google Scholar, and Web of Science until July 2023, using the keywords ‘graph’, ‘drug combination’, and ‘synergy’ and
screened retrieved articles with respect to their relevance to drug synergy predictions in
cancers using GNNs. Overall, we identified 25 relevant articles within the timeframe of
February 2020–July 2023. We observed a sharp upward trend in the development of GNNs
for drug synergy prediction (Fig. 3a). Moreover, we collected machine learning studies
related to drug synergy prediction from 2010 to 2023, not restricted solely to GNNs (Supplementary Table 1). Interestingly, we observed that since the inception of GNNs in this
field, their development for drug synergy prediction is becoming on par (and potentially
even surpassing in the near future) the combined progress of all other machine learning
methods, as depicted in Fig. 3b. These trends underscore the significance and timeliness of
our review in providing insights into this evolving landscape.


**4.1 Datasets**


Predicting synergistic drug combinations through machine learning techniques relies on
the availability of a gold standard training dataset. Typically, such datasets fall into one of
two categories: (1) those encompassing drug pairs and their corresponding synergy metrics derived from various cell lines (in vitro screening experiments) or (2) datasets derived
from clinical trials, where drug combinations are associated with positive or negative clinical outcomes. Table 1 provides a comprehensive overview of the diverse datasets used in
both in vitro screening and clinical studies, offering insights into the number of drugs, cell
lines, drug pairs, samples, and pertinent references.
Figure 4 visually illustrates the utilization of in vitro screening datasets by different
GNN models, taking into account the dataset size employed by each study. It is worth


**Fig. 3** The growth of studies related to Graph Neural Network (GNN) models for predicting drug
combinations. a) The cumulative count of published studies over time. The counts are segmented within
each half-year period starting from the first study’s publication in 2020 until the end of Q2, 2023. b) the
rising use of GNNs in drug combination prediction, as compared to alternative computational methods

# 1 3


A review on graph neural networks for predicting synergistic…



Page 13 of 38 **49**

# 1 3


**49** Page 14 of 38



M. Besharatifard, F. Vafaee



**Fig. 4** Dataset sizes across different drug combination studies, limited to studies using in vitro screening
datasets as datasets using clinical records are not consistent across different studies


noting that various studies may have applied filtering strategies or other data preprocess4
ing techniques, resulting in the utilization of specific subsets of the dataset. Notably, Fig.
highlights the frequent usage of Merck dataset (O’Neil et al. 2016) and DrugComb database (Zagidullin et al. 2019). The latter, in particular, consolidates multiple drug synergy
datasets, substantially expanding the training set and consequently establishing itself as the
commonly favored dataset for synergy prediction modelling. On the other hand, datasets
such as FORCINA which only focuses on one specific cell line are less frequently utilized
in computational models for predicting drug synergy with GNNs.


**4.2 Drug combination prediction based on in vitro experiments**


In this section, we offer an examination of research works centered on drug combination
prediction using in vitro synergy experiments. Table 2 offers a comprehensive summary of
these studies, outlining their individual merits and limitations. We organize these studies
2.
into two main sub-sections: classification and regression, as detailed in Table


**4.2.1 Classification methods**


As detailed in Table 2
, 16 studies have used classification to predict drug synergism. Out
of them, 5 studies have also developed regression-based models which were covered in the
next section. We grouped the remaining 11 studies based on their underlying GNN architecture namely GAT, GCN and GAE and summarized below:

# 1 3


A review on graph neural networks for predicting synergistic…



Page 15 of 38 **49**

# 1 3


**49** Page 16 of 38

# 1 3



M. Besharatifard, F. Vafaee


A review on graph neural networks for predicting synergistic…



Page 17 of 38 **49**

# 1 3


**49** Page 18 of 38

# 1 3



M. Besharatifard, F. Vafaee


A review on graph neural networks for predicting synergistic…



Page 19 of 38 **49**

# 1 3


**49** Page 20 of 38

# 1 3



M. Besharatifard, F. Vafaee


A review on graph neural networks for predicting synergistic…



Page 21 of 38 **49**



**4.2.1.1** _**GAT‑based methods**_
In four different models, researchers have used the GAT
to extract important features. GAT’s attention mechanisms allow it to focus on relevant
parts of a graph and increase model performance. For instance, in the case of DeepDDS
(Wang et al. 2022a), GAT is used to gather important information from the structure of
drugs. Additionally, Zhang et al.(Zhang et al. 2023) and Hu et al. (Hu et al. 2023) take a
knowledge graph (KG) approach, creating graphs and using Graph Attention Networks to
gather valuable insights from these graphs. Here, we’ll delve into these models in more
detail to provide a clear understanding of their methods.
The DeepDDS model employs two different types of GNNs: GAT and GCN. These
GNNs are assessed to extract features from the molecular graphs of drugs. The genomic
characteristics of cancer cells are encoded using a MLP. These resulting embeddings are
combined to create the ultimate feature representation for each combination of drug and
cell line. These features then go through fully connected layers to classify drug pairs as
either synergistic or antagonistic. Hu et al. proposed a model using a diverse graph with
drug, protein, and disease nodes. It employs GNNs for message spreading, refining node
embeddings through layers of attention-based mechanisms. This enhances the embeddings’
quality, later combined for synergy prediction through MLP module. The model predicts
drug combination effects effectively by leveraging GNNs and pre-trained models. KGANSynergy has three main steps: KG hierarchical propagation, KG attention layer, and prediction. The model explores relationships between drugs, cell lines, proteins, and tissues. The
attention layer updates entity representations using neural network-based attention, and the
prediction layer calculates synergy scores. SDCNet uses GCN for predicting specific drug
synergy without requiring cell line data. It models synergy as graphs per cell line, treating
them as relations. R-GCN captures combo traits within each relation and invariant patterns.
SDCNet, an encoder-decoder network, learns drug embeddings and forecasts SDCs across
cell lines. This method balances cell-specific and invariant features. However, new drug
combos could pose accuracy challenges.


**4.2.1.2** _**GCN‑based methods**_ GCN is used as a mechanism to extract meaningful features
from complex relations in networks. Among the reviewed studies, the use of GCN works in
drug-protein interaction networks or molecular structure (Bao et al. 2023; Yang et al. 2021;
Wang et al. 2022b). In various models, including those proposed by Hu et al. model (Hu
et al. 2022), and the MPFFPSDC model (Bao et al. 2023), the GCN encoder’s pivotal role
is in contextualizing drug structures within networks. This enables the transformation of
drug structures into embeddings in new spaces. In these models, the GCN is employed to
extract higher-order neighbor feature representations for atoms in drug molecular structures.
In the GraphSynergy model (Yang et al. 2021), a GCN is employed, specially tailored to
understand the connections between drugs and disease modules within this network.
MOOMIN (Rozemberczki et al. 2022) learns drug representations by encoding properties
of compounds and sequence proteins into vertex features. Similarly, SDCNet (Zhang et al.
2022a) applies GCN with attention layers to get relevant details from the drug-cell line
network, creating important features.
The framework of the DTSyn model consists of two paths: a fine-grained block and
a coarse-grained block. To use GCN, input chemical features are processed through
GCN blocks and combined with gene embeddings. These features are then fed into the
fine-grained Transformer encoder block, which learns chemical substructures and gene
interactions. Finally, by aggregating features and using MLP, it predicts synergy. Bao et al.
in 2023 proposed MPFFPSDC, a model for predicting drug synergy. It employs GCNs

# 1 3


**49** Page 22 of 38



M. Besharatifard, F. Vafaee



and an MLP to extract features from drug graphs and cell lines. The model aggregates
these features to classify drug pair synergy using a classifier module. In the MOOMIN
model, they consider the cell type when creating drug combination representations.
This leads to a scoring function that predicts synergy for new drug pairs. Yang et al.
propose GraphSynergy for predicting effective drug combinations in cancer using the
Protein–Protein Interaction (PPI) network. It uses GCN to grasp drug-disease connections,
attention highlights key proteins, and two scores evaluate therapy and toxicity.


**4.2.1.3** _**GAE and GCN encoder‑based methods**_ The GCN Encoder is designed to learn
node embeddings that capture both the structure of the graph and the attributes associated
with its nodes. It excels at uncovering relationships between nodes and using these relationships for predictive tasks by integrating feature information and graph structure (Jiang
et al. 2020). On the other hand, the GAE acts as an autoencoder with the goal of creating
a more condensed representation of the graph while also reconstructing the original adjacency matrix from the embeddings (Kipf and Welling 2016). This reconstruction helps in
inferring missing connections and gaining a comprehensive understanding of the graph’s
connections.

In the GAECDS model (Li et al. 2023a), GAE encodes drug combination information using the adjacency matrix and drug features. The encoded latent features are then
used to reconstruct the drug synergy graph and uncover novel relationships. Jiang et al.
(2020) applied a GCN encoder to process diverse networks encompassing drug-drug synergy, drug-target interactions, and protein–protein interactions. This encoder transforms
drug nodes into new-space embeddings. The model examines 39 heterogeneous networks,
generating embeddings via GCN encoding. Finally, using these embeddings and predictive models, drug synergy is forecasted. The GAECDS model consists of three key parts:
a GAE, an MLP, and a CNN. The GAE encodes drug synergy graphs and decodes them
to find new relationships. An MLP generates cell line features, while a CNN predicts drug
synergy by combining drug and cell line features.
One of the classification studies based on Graph Regularization is which was proposed
by Lv et al. (2022). They collected antibiotic combinations and target information from the
literature and described drug actions through network propagation and network proximity.
The study focused on pairwise antibiotic combinations and quantified interactions based on
the α-score. The model’s goal was to predict synergistic antibiotic combinations by considering pharmacological similarity between drugs. The affinity matrix W was constructed to
differentiate between pharmacologically similar drugs (potentially synergistic) and pharmacologically identical drugs (additive effect).


**4.2.2 Regression methods**


Various approaches address drug synergy prediction using regression methods. Categories
include GAT, GCN, and GAE models, each enhancing performance with distinct models.


**4.2.2.1** **GAT‑based methods** GAT is a mechanism that operates within the models to focus
on important interactions and features, contributing to accurate synergy score predictions.
In Model Numcharoenpinij et al. (2022), GAT is employed in the GNN model based on the
Message-Passing Neural Network (MPNN) framework. These weights guide the aggregation
process, allowing the model to focus on critical interactions. In the Muthene (Yue et al.

# 1 3


A review on graph neural networks for predicting synergistic…



Page 23 of 38 **49**



2023
), GAT creates meta-path-specific embeddings for end/central nodes by assigning
weights to neighbor features based on attention mechanisms. In CGMS model (Wang et al.
2023), GAT is used within the Heterogeneous Graph Attention Network (HAN). HAN has
three layers and employs a self-attention mechanism to capture important information and
produce cell line embeddings. Each model leverages GAT uniquely within its architecture,
highlighting its versatility in different scenarios.
Numcharoenpinij et al. incorporate genetic data from the Cancer Cell Line Encyclopedia (CCLE), including gene expression, copy number variation, and somatic mutation.
To reduce dimensionality while retaining crucial details, they employ autoencoders: deep,
sparse, and deep sparse. For drug information, two representations—Extended Connectivity Fingerprints (ECFPs) and molecular graphs—are utilized. Their model’s architecture
encompasses DNNs and Autoencoders for genetic and drug data processing. To predict
synergy, they employ a GNN framework, utilizing the concept of an MPNN. Muthene
predicts drug combination effectiveness by identifying shared mechanistic traits between
adverse events (AEs) and therapeutic effects (TEs). It tackles both tasks using meta-path
schemas, capturing drug-target interactions and mechanisms of action (MoAs). The model
generates drug embeddings from meta-paths and chemical features, predicting AE probabilities and therapeutic synergy. However, it can’t forecast synergy for new drugs or unseen
cell lines. The CGMS model predicts anti-cancer synergistic drug combinations using a
complete graph. This graph integrates cell lines and drugs through different meta-paths,
representing drug-cell line interactions and drug-drug interactions. Employing the HAN,
the model generates whole-graph embeddings hierarchically, capturing important graph
information.


**4.2.2.2** **GCN‑based methods** GCNs excel at capturing complex interactions in graphical
data, which are common in drug synergistic prediction models. In these models, GCNs
are used to process molecular structures of drugs or knowledge graphs containing various
entities and relationships. This ability to capture rich information from different networks
makes GCNs a good choice for modeling complex biological relationships (Wang et al.
2022b; Zhang et al. 2023; Liu and Xie 2021). The PRODeepSyn (Wang et al. 2022b)
model leverages the GCN to construct gene hidden states based on the PPI network. Zhang
et al. (Zhang and Tu 2022) emphasize the pivotal role of GCNs in extracting valuable
information from the constructed KG. In the TranSynergy model (Liu and Xie 2021), the
GCN is utilized to extract important features from the drug’s molecular graph structure.
In the HypergraphSynergy model (Liu et al. 2022), GCN embeds drugs and cell lines;
After forming a hypergraph based on drugs and cell lines, it learns and finally records the
embedding of nodes.
To predict drug synergy, PRODeepSyn initially forms drug features using molecular fingerprints and descriptors. For cell line features, it combines gene expression, gene
mutation, and interactions among gene products. GCN is applied to create gene hidden
states from the PPI network, considering protein interactions. These states estimate the
gene’s evident state using omics data. Finally, PRODeepSyn forecasts synergy scores using
a DNN, utilizing both drug features and cell line embeddings as inputs. The KGE-DC
model utilizes a KG containing drugs, targets, enzymes, and transporters to predict synergy. GCNs extract features from the KG, improving information extraction. Drug embeddings and cell line gene expressions are integrated, and a neural network predicts synergy
scores. Liu et al. utilize a drug synergistic hypergraph with drugs and cell lines as nodes
and hyperedges for synergistic relationships. GCN learns embeddings for drugs and cell

# 1 3


**49** Page 24 of 38



M. Besharatifard, F. Vafaee



lines, capturing hypergraph features. These learned features represent drugs. Gene expression features of cell lines are captured via a network. Finally, matrices of drug and cell
line features enter the hypergraph network for predicting drug synergy scores. The TranSynergy model employs a transformer to analyze drug and cell line data while integrating
drug target profiles for comprehensive features. It enhances cell line representations using
gene expression data. Additionally, a GCN is utilized to extract drug features from drug

structures.


**4.2.2.3** **GAE ‑based methods** GAE acts as a transformative tool in the two investigated
regression models. In MGAE-DC model (Zhang and Tu 2023), GAE encodes drug combinations, learning drug embeddings. In Zagidullin et al. (Zagidullin et al. 2021) GAE
transforms molecular structures into fingerprints. By considering synergistic, additive, and
antagonistic combinations as distinct input channels, MAGE-DC enhances drug embeddings’ ability to differentiate between synergy and non-synergy. This improved detection
is achieved via a GAE. Using concatenated embeddings, drug fingerprints, and cell line
features, the prediction module synergistic scores. Zagidullin et al. proposed an approach
where genetic and drug data are used to predict drug combination synergy scores. Genetic
data informs about cancer cell lines, while drug data include molecular structures. The
model employs GAE to encode drug structures, yielding synergy predictions. While this
work focused solely on comparing fingerprint types, future research could explore combining molecular structure or investigating other molecular features.
Although these models are promising in predicting drug synergy, there are limitations
that require further research and improvement for better performance, which we will discuss below.


**4.3 Drug combination prediction based on clinical studies**


Five methods utilized clinical studies to construct datasets of synergistic drug combinations, all employing GCNs as their primary neural network approach for the classification
task (Table 2).


**4.3.1 Classification methods**


**4.3.1.1** _**GCN‑based methods**_ The MK-GNN model (Gao et al. 2023) is a deep learning
approach designed to predict effective drug combinations for patient treatment. It utilizes
multi-head attention to learn patient features from diagnosis and treatment procedure
sequences. Additionally, it incorporates prior medical knowledge derived from electronic
health record data, considering the relationship between diagnoses and medications. The
model also employs a GCN to learn medication representation vectors, capturing drug
knowledge from a formulated drug network. However, the model’s generalization is limited
due to variations in drug combination recommendations among different doctors and regions.
To address this, future research aims to study feature invariance in drug combinations and
enhance the algorithm’s applicability in real clinical settings. Chen et al. (2022) proposed
a novel computational pipeline called DCMGCN for predicting drug combinations. The
pipeline integrates diverse drug-related information to learn low-dimensional representations
of drugs from attributes and similarity networks. They identified that the drug-drug network
had heterophily and sparseness, which could limit the effectiveness of the GCN. To address
this, they introduced two modifications to GCN. The drug representations were then

# 1 3


A review on graph neural networks for predicting synergistic…



Page 25 of 38 **49**



optimized using the modified GCN (MGCN) to predict drug combinations. By integrating
various data types, including clinical data, DCMGCN becomes a powerful tool for drug
discovery and repositioning, with potential for further extension by incorporating more
heterogeneous information and experimental validation. ComboNet model (Jin et al. 2021)
is designed to jointly learn drug-target interactions and drug-drug synergy. It comprises
two components: a drug-target interaction module and a target-disease association module.
This architecture enables the model to utilize data on drug-target interactions, singleagent antiviral activity, and available drug-drug combination datasets. The DTI network
in ComboNet predicts likely targets for drugs, while the target-disease association network
models how biological targets and structural features of molecules are related to antiviral
activity and synergy. The model’s strength lies in considering single-agent activity, which
enhances the effectiveness of drug combination predictions against SarsCov-2. However, a
limitation is the scarcity of training data for accurate drug synergy prediction.
MG-DDIS model (Deng et al. 2021) is an end-to-end multi-task learning framework
based on a GCN for predicting DDIs and synergistic drug combinations. The model to capture important information from the molecular structures, the R-radius subgraph method
is applied, producing a series of subgraphs for each drug. These subgraphs are then fed
into the GCN encoder to learn a latent representation of drugs. The model is trained using
a multi-task approach to simultaneously predict DDIs and synergistic drug combinations.
Despite its success, the model’s limitations include the possibility of adverse reactions arising from various factors unrelated to synergy, such as individual drug sensitivity and independent toxic properties of certain drugs.


**4.3.1.2** **GAE ‑based methods** Karimi et al. (Karimi et al. 2020) introduced a novel deep
generative model for drug combination design, named as the Hierarchical Variational Graph
Autoencoder (HVGAE), which leverages graph-structured domain knowledge and reinforcement learning-based chemical graph-set designer. In HVGAE, GAE has been utilized
in learning and encoding features from graph-structured data at two levels: (1) Gene–Gene
Embedding where GAE is applied to the gene–gene network, represented as a graph, to
extract features related to gene interactions, and (2) Disease-Disease Embedding where
GAE operates on the disease-disease network, building on the gene representations learned
in the first level. Simultaneously, GCNs are applied to process the graph structures representing gene–gene interactions. The HVGAE framework integrates these dual levels of
GAE and the insights from GCNs into an end-to-end representation learning process. The
learned features serve as a foundation for subsequent drug combination design. The model’s
core objective is to design a reinforcement learning-based (RL-based) drug combination
generator, operating within a chemistry- and system-aware environment.
Across these models, the interesting aspect of using GCN lies in its ability to capture
complex relationships and structural information from different types of data, such as
molecular graphs, networks, and clinical information.


**5 Evaluation of GNNs on in vitro datasets**


3, which includes the results of
In this section, we discuss the findings presented in Table
various drug combination prediction studies. By reviewing and analyzing these results, we
aim to gain valuable insights into the challenges in studies and advances in this field.

# 1 3


**49** Page 26 of 38



M. Besharatifard, F. Vafaee











|e 3 Performance|e evaluation of drug combinations studies using GNNs|
|---|---|
|**Study**<br>|**Validation**<br>**more**<br>**less**<br>**Dataset**<br>**Metric**<br>**AUC**<br>**ACC**<br>**AUPR**<br>**F1**<br>**RMSE**<br>**MSE**|
|~~**Hu et al. (Hu et al.,**~~<br>**2023)**<br>|10-fold CV<br>AstraZeneca<br>NA<br>0.84<br>0.88<br>0.87<br>~~DrugComb~~<br>~~NA~~<br>~~0.96~~<br>~~0.87~~<br>~~0.95~~<br>~~0.97~~|
|~~**GAECDS (Li et al.,**~~<br>**2023b)**<br>|5-fold CV<br>0<br>0<br>DrugComb<br>Loewe<br>0.98<br>0.87<br>0.93<br>0.77|
|~~**KGANSynergy(Zhang et**~~<br>**al., 2023)**<br>|5-fold CV<br>quartile<br>quartile<br>DrugComb<br>ZIP<br>0.895<br>0.817<br>0.892<br>quartile<br>quartile<br>~~Merck~~<br>~~Loewe~~<br>~~0.891~~<br>~~0.822~~<br>~~0.898~~|
|~~**MPFFPSDC (Bao et al.,**~~<br>**2023)**<br>|5-fold CV<br>AstraZeneca<br>Loewe<br>0.67<br>0.71<br>0.84<br>0.82<br>Merck<br>Loewe<br>0.94<br>0.87<br>0.94<br>0.86|
|~~**DeepDDS (Wang et al.,**~~<br>**2022a)**<br>|5-fold CV<br>10<br>0<br>AstraZeneca<br>Loewe<br>0.66<br>0.64<br>0.82<br>10<br>0<br>Merck<br>Loewe<br>0.93<br>0.85<br>0.93|
|~~**SDCNet (Zhang et al.,**~~<br>**2022a)**<br>|10-fold CV<br>10<br>0<br>Merck<br>Loewe<br>0.93<br>0.85<br>0.92<br>0.83<br>3.68<br>−3.37<br>Bliss<br>0.96<br>0.9<br>0.97<br>0.92<br>3.87<br>-3.02<br>ZIP<br>0.95<br>0.91<br>0.98<br>0.94<br>2.64<br>-4.48<br>HSA<br>0.94<br>0.92<br>0.98<br>0.95<br>~~10~~<br>~~0~~<br>~~ALMANAC~~<br>~~Loewe~~<br>~~0.85~~<br>~~0.75~~<br>~~0.88~~<br>~~0.67~~<br>3.68<br>−3.37<br>Bliss<br>0.86<br>0.78<br>0.86<br>0.78<br>3.87<br>-3.02<br>ZIP<br>0.93<br>0.86<br>0.92<br>0.84<br>2.64<br>-4.48<br>HSA<br>0.9<br>0.85<br>0.85<br>0.76<br>~~10~~<br>~~0~~<br>~~CLOUD~~<br>~~Loewe~~<br>~~0.51~~<br>~~0.64~~<br>~~0.56~~<br>~~0.31~~<br>3.68<br>−3.37<br>Bliss<br>0.52<br>0.52<br>0.51<br>0.53<br>3.87<br>-3.02<br>ZIP<br>0.51<br>0.52<br>0.51<br>0.25<br>2.64<br>-4.48<br>HSA<br>0.51<br>0.5<br>0.5<br>0.3<br>~~10~~<br>~~0~~<br>~~FORCINA~~<br>~~Loewe~~<br>~~0.65~~<br>~~0.68~~<br>~~0.59~~<br>~~0.55~~<br>3.68<br>−3.37<br>Bliss<br>0.6<br>0.86<br>0.85<br>0.92<br>3.87<br>-3.02<br>ZIP<br>0.57<br>0.9<br>0.85<br>0.92<br>2.64<br>-4.48<br>HSA<br>0.64<br>0.87<br>0.83<br>0.9<br>~~10~~<br>~~0~~<br>~~Transfer~~<br>~~Loewe~~<br>~~0.88~~<br>~~0.8~~<br>~~0.88~~<br>~~0.79~~|
|~~**MOOMIN(Rozembercz**~~<br>**ki et al., 2022)**<br>|5-fold CV<br>NA<br>NA<br>DrugComb<br>0.77<br>0.702<br>0.63|
|~~**ComboNet (Jin et al.,**~~<br>**2021)**<br>|-<br>Riva et al.<br>0.68<br>Bobrowski et al.<br>0.82<br>NCATS<br>0.815|
|~~**MG-DDIS (Deng et al.,**~~<br>**2021)**<br>|NA<br>-<br>-<br>DrugBank<br>-<br>0.978<br>0.955<br>0.953|
|~~**GraphSynergy (Yang et**~~<br>**al., 2021)**<br>|3-fold CV<br>0<br>0<br>DrugComb<br>-<br>0.83<br>0.75<br>0.81<br>0.72<br>~~0~~<br>~~0~~<br>~~Merck~~<br>~~-~~<br>~~0.84~~<br>~~0.76~~<br>~~0.84~~<br>~~0.77~~|
|~~**Jiang et al. (Jiang et al.,**~~<br>**2020)**<br>|10-fold CV<br>30<br>30<br>Merck<br>Loewe<br>0.89<br>0.91<br>0.79|
|~~**DTSyn [6]**~~<br>|-<br>-<br>-<br>Merck<br>Loewe<br>0.89<br>0.81<br>0.87|
|~~**PRODeepSyn (Wang et**~~<br>**al., 2022b)**<br>|5-fold CV<br>30<br>30<br>Merck<br>Loewe<br>0.9<br>0.93<br>0.63<br>15.09<br>229.49|
|~~**HypergraphSynergy**~~<br>**(Liu et al., 2022)**|5-fold CV<br>30<br>30<br>Merck<br>Loewe<br>0.923<br>0.6025<br>0.632<br>0.9254<br>14.36<br>~~ALMANAC~~<br>~~NA~~<br>~~0.853~~<br>~~0.5295~~<br>~~0.557~~<br>~~0.8902~~<br>~~43.65~~|
|**KGE-DC (Zhang and**<br>**Tu, 2022)**<br>|10-fold CV<br>10<br>10<br>Merck, ALMANAC,<br>CLOUD, and<br>FORCINA<br>Loewe<br>0.86<br>0.94<br>0.6<br>0.51<br>204.9<br>3.68<br>3.68<br>Bliss<br>0.69<br>0.8<br>0.51<br>0.33<br>70.53<br>2.64<br>2.64<br>ZIP<br>0.72<br>0.81<br>0.54<br>0.37<br>62.18<br>3.87<br>3.87<br>HSA<br>0.73<br>0.82<br>0.56<br>0.46<br>67.69|
|~~**TranSynergy (Liu and**~~<br>**Xie, 2021)**<br>|5-fold CV<br>30<br>30<br>O’Neil<br>Loewe<br>0.907<br>0.627<br>231|
|~~**CGMS (Wang et al.,**~~<br>**2023)**<br>|5-fold CV<br>-<br>-<br>DrugComb<br>Loewe<br>14.38<br>208.38<br>-<br>-<br>Merck<br>Loewe<br>-<br>208.38|
|~~**MGAE-DC (Zhang and**~~<br>**Tu, 2023)**<br>|10-fold CV<br>30<br>0<br>Merck<br>Loewe<br>12.73<br>162.21<br>3.68<br>-3.37<br>Bliss<br>4.15<br>17.36<br>2.64<br>-4.48<br>ZIP<br>3.27<br>10.68<br>3.87<br>-3.02<br>HSA<br>4.22<br>17.89<br>~~30~~<br>~~0~~<br>~~CLOUD~~<br>~~Loewe~~<br>~~18.09~~<br>~~327.35~~<br>3.68<br>-3.37<br>Bliss<br>18.05<br>325.99<br>2.64<br>-4.48<br>ZIP<br>17.97<br>323.43<br>3.87<br>-3.02<br>HSA<br>17.71<br>313.78<br>~~30~~<br>~~0~~<br>~~FORCINA~~<br>~~Loewe~~<br>~~14.1~~<br>~~200.48~~<br>3.68<br>-3.37<br>Bliss<br>13.44<br>184.35<br>2.64<br>-4.48<br>ZIP<br>14.55<br>212.89<br>3.87<br>-3.02<br>HSA<br>14.31<br>207.36<br>~~30~~<br>~~0~~<br>~~ALMANAC~~<br>~~Loewe~~<br>~~11.01~~<br>~~121.18~~<br>3.68<br>-3.37<br>Bliss<br>3.99<br>15.93<br>2.64<br>-4.48<br>ZIP<br>3.59<br>12.88<br>3.87<br>-3.02<br>HSA<br>3.69<br>13.63|
|~~**Muthene (Yue et al.,**~~<br>**2023)**<br>|hold out<br>DrugComb<br>Loewe<br>180.62<br>Bliss<br>45.74<br>ZIP<br>29.24<br>HSA<br>30.23<br>31|
|~~**Numcharoenpinij et al.**~~<br>**(Numcharoenpinij et al.,**<br>**2022)**<br>|5-fold CV<br>DrugComb<br>12.09<br>146.137|
|~~**Zagidullin et al**~~<br>**(Zagidullin et al., 2021)**<br>|5-fold CV<br>DrugComb<br>Loewe<br>0.73<br>Bliss<br>0.78<br>ZIP<br>0.76<br>HSA<br>0.8|
|~~**MK-GNN (Gao et al.,**~~<br>**2023)**|3-fold CV<br>EHR<br>-<br>-<br>0.28<br>0.44<br><br>|
|**Lv et al. (Lv et al., 2022)**<br>|hold out<br>~~α-score ≤~~<br>−0.25<br>~~α-score~~<br>≥ 1<br>E coli MG1655<br>0.9<br>0.78|
|~~**HVGAE (Karimi et al.,**~~<br>**2020)**|hold out<br>FDA.gov, Cheng et al<br>_<br>0.96<br>0.79<br>|
|**DCMGCN (Chen et al.,**<br>**2022)**|5-fold CV<br>~~Cheng et al. TTD,~~<br>ClinicalTrials.gov,<br>eMedExpert, FDA.gov<br>0.945<br>0.297<br>0.348|

# 1 3




A review on graph neural networks for predicting synergistic…


**Table 3** (continued)



Page 27 of 38 **49**



- α-score: For each drug pair, a drug interaction score (α-score) quantifying the concavity of the isophenotypic curve was compute (Cokol et al. 2011)

Color legend of classification metrics: 0 1.0

Color legend of regression metrics: 0 400


Both the DeepDDS (Wang et al. 2022a) and Hu et al. (2023) models employ GATbased classification approaches and evaluate their performance on the AstraZeneca dataset. However, there is a notable difference in their results. Hu’s model achieves an AUC
of 0.84, while DeepDDS achieves a comparative AUC of 0.66 Additionally, Hu’s model
obtains a higher AUPR score. When comparing these models with the same cross fold
and same train dataset, Hu’s model still outperforms DeepDDS. This might be attributed
to Hu’s more comprehensive feature extraction process. Hu’s model incorporates diverse
features from drugs, cell lines, and diseases, utilizing pre-trained models in a heterogeneous graph as a node’s features. In contrast, DeepDDS focuses on drug features extracted
solely through GAT and GCN from the drug’s structure. This highlights that this approach,
incorporating a wider range of features and relationships, yields better predictive performance in comparison to DeepDDS’s more focused feature extraction. In other words, Hu
et al. obtained initial embeddings for different types of entities (heterogeneous entities such
as drugs, proteins, and diseases) using separate MLPs. After obtaining these initial embeddings, they further enhanced and refined these embeddings using GATs.
The Hu’s model was compared to the TranSynergy (Liu and Xie 2021) model using a
tenfold cross-validation on the DrugComb dataset. The Hu model outperformed the TranSynergy model in predicting drug combination synergism. This superiority is attributed to
the Hu model’s utilization of comprehensive drug and cell line features, which enhanced its
ability to identify synergistic effects compared to the TranSynergy model that solely relied
on drug target proteins. The advantage of Hu’s model is that it enables the model to capture
and propagate information through the graph effectively. However, TranSynergy lacks the
capacity to capture the same level of information regarding relationships between entities.
SDCNet (Zhang et al. 2022a), a GAT-based model, is compared with DeepDDS and
Jiang’s model (Jiang et al. 2020) on various datasets (ALMANAC, Merck, Cloud, FORCINA)
using different metrics (Loewe, Bliss, HSA, Zip). SDCNet benefits from drug features derived
from the training dataset and cell line-based information. Unlike traditional GCNs, R-GCN
employs distinct aggregation mechanisms tailored to different types of relationships. As a
result, the SDC-Net model succeeds in obtaining more informative representations for drugs
specific to each cell line. This advantage leads to more informative features for classification.
Despite dataset imbalances, SDCNet achieves superior AUC, AUPR, and F1-score compared
to DeepDDS and Jiang’s model. Since in the prediction of drug synergy, the accurate detection of positive cases (synergistic combinations) is more important than the detection of negative cases due to data imbalance, criteria such as AUPR and F1-score are used to evaluate the
models fairly. These measures take into account the importance of positive samples and make
them suitable for unbalanced data sets. If the SDCNet model is trained with appropriate data,
it can effectively predict drug synergy. Notably, the DeepDDS model outperforms SDCNet
in leave-one-drug-out evaluation, likely because DeepDDS’s performance isn’t heavily reliant
on the training data. Conversely, in leave-one-cell-line-out evaluation, SDCNet excels. This
is because SDCNet processes features individually for each cell line, considering the interaction type of medicinal compounds (synergistic or antagonistic). Overall, SDCNet’s success is
attributed to its specialized feature processing for different cell lines. The study (Wang et al.

# 1 3


**49** Page 28 of 38



M. Besharatifard, F. Vafaee



2022a) highlights an intriguing observation regarding the DeepDDS model’s performance. It
reveals that when the model’s complexity increases and features become excessively dimensional, its performance can actually suffer. A comparison between DeepDDS and TranSynergy underscores this point. In TranSynergy, features are not only high-dimensional but also
embedded using a transformer. On the Merck dataset, the DeepDDS model outperforms TranSynergy, emphasizing that overly complex models and extensive feature dimensions might not
always yield improved results. Among other GAT-based models, the KGANSynergy (Zhang
et al. 2023) model which extracts the features of drugs and cell lines using the knowledge
graph and based on attention, and compared to the GraphSynergy model (Yang et al. 2021), it
has been able to perform better.
The MPFFPSDC model (Bao et al. 2023), which is based on the GCN approach, outperforms DeepDDS on the Merck dataset. While both models achieve almost the same results,
MPFFPSDC demonstrates superior performance. This could be due to variations in how features are integrated for classification. Despite this difference, both models follow almost the
same methods to extract features from drugs and cell lines. DTSyn (Hu et al. 2022) extracts
drug features using cell line data and known train’s data labels. However, it’s less accurate than
other machine learning models for predicting drug synergy scores of drugs that it has not seen
so far. MOOMIN’s model (Rozemberczki et al. 2022) lacks a defined threshold for categorizing drug synergism. It employs random walk on a drug-target network and GCN to embed
drugs and capture structural features. However, its performance is comparatively weaker due
to the absence of cell line features and comprehensive drug-related information, unlike other
GCN-based models. GAECDS, a GCN-based model, classifies drug compound data from
DrugComb using a threshold of 0. While using a fixed threshold can introduce noise, GAECDS outperforms both the DeepDDS model and the GraphSynergy model, both of which also
use the same dataset and threshold. This improved performance might stem from GAECDS’s
use of GAE on the drug-drug synergy network, which effectively distinguishes drug combinations in a new data space.
Using an attention-based approach and meta-path on a diverse graph of drug and cell line
connections, the CGMS model (Wang et al. 2023) outperformed PRODeepSyn (Wang et al.
2022b
), TranSynergy, and DeepDDS. This suggests CGMS effectively predicted drug synergy, surpassing existing methods. Numcharoenpinij et al. introduced a GAT-based regression
approach in their model, which utilizes autoencoders to capture key features of cell lines. This
method demonstrated higher accuracy compared to other models, although the specific metric
type of the dataset was not specified. Notably, the GAT-based approach outperformed DeepDDS, exhibiting lower error. Using adverse and therapeutic effect data as synergistic information for drug combinations has led the Muthene model to outperform other models like
CGMS. This unique approach has resulted in lower errors in predicting drug synergy. Muthene benefits from including adverse and therapeutic effects, enhancing its accuracy compared
to CGMS and similar models.

MGAE-DC (Zhang and Tu 2023) is a GAE-based model that has shown lower error rates
in regression than PRODeepSyn, HypergraphSynergy, and DeepDDS. However, in classification mode, its results are comparable to those of the PRODeepSyn model. This may be due to
an imbalance in the data. The embedding of GAE and GCN encoders appears to work similarly. The Zagidullin’s model is related to the optimal selection of drug fingerprints for predicting drug synergy, which achieved the lowest error for predicting synergy on DrugComb data
with E3FP 1024 bits long fingerprints generated from SMILES strings.
As discussed earlier, five studies on clinical data were analyzed for synergistic prediction
3.
with graph-based models and their classification results are shown in Table

# 1 3


A review on graph neural networks for predicting synergistic…


**6 Discussion**


**6.1 GNN for drug synergy prediction: strengths and limitations**



Page 29 of 38 **49**



GNNs possess a distinct advantage over other machine learning techniques due to their
ability to capture intricate relationships within biological networks. This advantage is
particularly relevant in drug discovery, where compounds can be intuitively represented
as graphs via molecular graph representation—i.e., molecules can be decomposed
into individual atoms, with the bonds between them forming a graph structure. Unlike
other machine learning algorithms that often require predefined chemical descriptors or
molecular fingerprints, GNNs can extract features directly from the graph representation
of drugs (Zhang et al. 2022b). Consequently, GNN models are commonly employed in
drug synergy prediction models to obtain drug representations as features for a classification or regression model (Bao et al. 2023; Wang et al. 2022a; Hu et al. 2022). Moreover, GNNs can be utilized to learn complex structural relationships in diverse biological
systems, such as protein–protein interactions and drug-target interactions. For example,
KGANSynergy (Zhang et al. 2023) integrated multiple types of associations (drug–protein, cell line–protein, cell line–tissue, and protein–protein associations) into a knowledge graph to learn representations of cell lines and drugs using GNN models.
However, GNNs also have limitations in the context of drug synergy prediction and
in general. GNNs can be computationally intensive, and their complex architecture often
demands a substantial amount of data for effective training. On the other hand, the space
of possible drug combinations is vast, and only a small fraction of potential drug pairs
has been experimentally tested. This sparsity in the synergy space can make it difficult
for GNNs to make accurate predictions, especially for untested combinations. With limited data and complex models, there is a high risk of overfitting, where the model learns
to perform well on the training data but fails to generalize to new drug combinations.
Careful regularization and validation strategies are needed to address this issue.
Moreover, interpretability is a major concern, as GNNs are frequently seen as blackbox models, rendering it difficult to discern the reasoning behind their predictions (Zhu
et al. 2022). These models often lack the mechanistic insights necessary to explain the
underlying reasons for the synergistic or antagonistic effects observed in specific drug
or compound combinations.
Particular GNN architectures may hinder the expressivity of GNNs, which pertains
to their ability to represent and differentiate diverse graph structures. Despite their popularity, some widely used GNNs, including GCNs, exhibit a theoretical limitation in
expressivity. This lack of expressivity can lead to these models underfitting the training
data, resulting in suboptimal performance, particularly when confronted with complex
relationships within graph data (Xu et al. 2018).
Additionally, the presence of heterophily, where interconnected nodes exhibit diverse
attributes or labels, presents a challenge for GNNs (Luan et al. 2022). GNNs may face
difficulties when applied to heterophilic graphs in the context of link prediction (Zhou
et al. 2022). An example of a heterophilic graph in drug synergy prediction is one that
incorporates links representing synergistic interactions between drug pairs and various cell lines. Such a graph poses a challenge for link prediction, demanding for the
adoption or development of GNNs specially tailored to operate on graphs with diverse
node types and attributes such as Heterogeneous Graph Attention Networks (Wang et al.
2019).

# 1 3


**49** Page 30 of 38



M. Besharatifard, F. Vafaee



Overall, while GNNs hold promise in the field of drug synergy prediction, they come
with various specific or general limitations. These limitations call for active exploration of
techniques to enhance the strengths of GNNs to improve the accuracy, generalizability, and
interpretability of drug synergy predictions.


**6.2 The choice of metrics of synergism: no standard reference model**


To optimize drug combinations, efficient identification of synergistic effects is crucial.
Classical reference models often fall into two categories of _effect-based_ (e.g., Bliss Independence and HSA) and _dose–effect-based_ (e.g., Loewe Additivity and ZIP). Effect-based
models assess interaction effects based on individual drug effects, often measured by cell
responses like cell death, viability, and growth rate. Dose–effect-based models, introduced
more recently, offer enhanced definitions of synergy, additivity, and antagonism for drugs
with nonlinear dose–effect curves, surpassing the limitations of reference models.
While Loewe additivity and Bliss independence are commonly employed in synergism
research, there is still no consensus due to the limitations inherent in all reference models,
as discussed in prior reviews (Duarte and Vale 2022). Furthermore, the suitability of different metrics varies depending on the specific biological context. For instance, Loewe additivity is well-suited when drugs target the same pathways, while the Bliss independence
principle is more relevance in cases where drugs are mutually nonexclusive, each targeting
distinct signaling pathways (Liu et al. 2018). Consequently, the development of new synergism metrics remains an active research area (Wooten et al. 2021).
Developing GNN models with metric-agnostic capabilities has the potential to improve
their suitability for experimental testing. For instance, SDCNet (Zhang et al. 2022a) was
assessed across various metrics and datasets, mitigating the performance biases associated with specific synergism reference models. DeepDDS (Wang et al. 2022a), on the other
hand, aggregated various synergy scores associated with identical drug pair-cell line combinations through averaging through averaging. This strategy has the potential to improve
annotation accuracy while reducing reliance on a specific synergy metric for defining
synergism.
Nonetheless, it is important to note that synergism alone may not suffice for evaluating
the clinical promise of drug combinations. Complementary measures, such as the therapeutic index, which assesses the relative toxicity of anticancer treatments in normal tissues,
should also be considered (Ocana et al. 2012).


**6.3 Synergy score thresholding and class imbalance mitigation**


The process of thresholding synergism metrics to categorize numerical values into either
synergistic, additive, and antagonistic (or non-synergistic) annotations plays an important
role in the development of classification models. This step significantly impacts class balance, data distribution, sample sizes, and the accuracy of categorization of numeric scores
to categories. Different synergy metrics exhibit diverse distributions, and thresholding
should be executed while considering the underlying data distribution, in conjunction with
biological relevance and clinical evidence when available. Current studies often underestimate the importance of thresholding, leading to the selection of seemingly arbitrary thresholds or the adoption of thresholds used in previous studies without adequate justification.
Therefore, there is a need for a benchmarking study to systematically evaluate the impact
of threshold selections on the performance of GNN models.

# 1 3


A review on graph neural networks for predicting synergistic…



Page 31 of 38 **49**



Thresholding strategies have a direct impact on class imbalance. For example, one
can designate the upper quartile and lower quartile scores as synergistic and antagonistic,
respectively, in order to attain a balanced dataset and improve the accuracy of categorization (i.e., reduce the risk of false positive or false negative annotations resulting from
more relaxed thresholding). However, this approach comes at the cost of reducing the size
of the training set. Alternatively, techniques like oversampling or undersampling, such as
SMOTE (Chawla et al. 2002), can be employed to balance the dataset. Class imbalance
also affects the performance metrics on the test set. To address this, Jiang et al. (2020)
selected 10% of the positive data as test samples and balanced this by randomly choosing
an equal number of negative samples for their evaluation. However, it is important to note
that due to the inherent imbalance in drug synergism (i.e., synergistic drug combinations
are often rare when compared to non-synergistic ones), creating a perfectly balanced test
dataset may not provide an accurate assessment of the model’s effectiveness.
The choice of performance evaluation measures is also important in the presence of
class imbalance. AUC (area under the ROC curve) and AUPR (area under the precisionrecall curve) are two commonly used metrics for evaluating drug synergy performance.
AUC-ROC measures the trade-off between sensitivity and specificity, making it suitable
when the cost of false positives and false negatives is roughly equal or when positive and
negative instances are approximately balanced. In contrast, AUC-PR is more appropriate
when the cost of false positives and false negatives is highly asymmetric or when the positive class is rare (Sofaer et al. 2019). Li et al. (2023b) demonstrated that even when negative data outnumbered positive data threefold (using the DrugComb dataset), the model’s
AUC remained unaffected. Hence, the use of AUPR is more recommended.


**7 Conclusions**


In this study, we present a comprehensive review of GNN-based approaches for predicting synergistic drug combinations. We have curated a total of 25 GNN-based models
developed up to the date of our literature search (July 2023). We assessed these models,
considering various aspects, including their underlying GNN architectures, the nature of
the prediction problem (classification vs. regression), the types of datasets employed (in
vitro or clinical), the features incorporated, and the synergy metrics applied. Furthermore,
we summarized the strengths, and limitations of each study. Additionally, we conducted a
comparative assessment of the prediction performance of GNNs in in vitro studies, stratified by the respective datasets, synergism metrics, thresholding approaches, and validation
strategies employed. This comprehensive study provides an overview of the current state of
the field, offering insights into the progress and challenges in the field of synergistic drug
combination prediction using GNNs and beyond.


**7.1 Limitations and future directions**


While we have presented performance evaluation metrics of GNN models on different datasets, direct comparisons are challenging due to variations in data preprocessing
methods such as thresholding, among other factors. A benchmarking study that controls
confounding conditions is essential to facilitate direct model performance comparisons on
identical datasets, enabling the identification of state-of-the-art algorithms. Additionally,
this study does not aim to provide either qualitative or quantitative assessments of GNNs

# 1 3


**49** Page 32 of 38



M. Besharatifard, F. Vafaee



in comparison to other drug synergy prediction algorithms, given the limited space available. However, a future study comparing high-performing GNNs with other top-performing
drug synergy prediction algorithms could offer valuable insights.
The field of GNNs for drug synergy prediction offers room for improvement from various perspectives. In addition to the suggestions outlined in the Discussion section, deeper
integration of biological networks and graph theory concepts could enhance performance.
Exploring techniques such as identifying _maximum cliques_ with GNNs to unveil relationships between entities (Min et al. 2022) and utilizing _minimum dominating sets_ for more
accurate link predictions among neighbors (Rani 2012) offers promising directions for
future research. These strategies have demonstrated potential in recent drug synergy prediction studies (Zhang et al. 2023; Zhang and Tu 2022).
Enhancing the biological relevance of predictions can be achieved through the incorporation of knowledge graphs reflecting associations in biological systems, such as protein–protein interactions. By integrating these graphs with complex protein features,
predictions can be improved. Leveraging cutting-edge protein structure prediction (For
proteins whose structure may not be available) algorithms like Alphafold (Jumper et al.
2021
), can significantly improve the accuracy of protein representations for integration into
GNN models.
In summary, the field of drug synergy prediction using GNN and emerging techniques
continues to evolve. Ongoing research, including benchmarking studies, enhancing biological relevance, and exploring novel strategies, offers prospects for more accurate and
clinically translatable predictions. Despite the general interest within the research community and pharmaceutical industry regarding the use of GNNs in drug discovery, the practical implementation of these methods in pre-clinical and clinical settings is still in its early
stages due to the recent emergence of this technology in this field. As the field continues to
evolve, future research and advancements will likely contribute to a more comprehensive
understanding of the advantages and applications of GNN in drug combination prediction
across diverse disease areas.


**Supplementary Information** The online version contains supplementary material available at [https://​doi.​](https://doi.org/10.1007/s10462-023-10669-z)
[org/​10.​1007/​s10462-​023-​10669-z.](https://doi.org/10.1007/s10462-023-10669-z)


**Author contributions** F.V. conceptualised and supervised the study. M.B. conducted the literature review.
F.V. and M.B. wrote the main manuscript text. F.V. generated figures. M.B. generated tables. F.V. critically
revised the manuscript


**Funding** Open Access funding enabled and organized by CAUL and its Member Institutions.


**Declarations**


**Competing interests** The authors declare no competing interests.


**Open Access** This article is licensed under a Creative Commons Attribution 4.0 International License, which
permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give
appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence,
and indicate if changes were made. The images or other third party material in this article are included in the
article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is
not included in the article’s Creative Commons licence and your intended use is not permitted by statutory
regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright
[holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.](http://creativecommons.org/licenses/by/4.0/)

# 1 3


A review on graph neural networks for predicting synergistic…


**References**



Page 33 of 38 **49**



Alves LA, Ferreira NCDS, Maricato V, Alberto AVP, Dias EA, Jose Aguiar Coelho N (2022) Graph neural
[networks as a potential tool in improving virtual screening programs. Front Chem 9:787194. https://​](https://doi.org/10.3389/fchem.2021.787194)
[doi.​org/​10.​3389/​fchem.​2021.​787194](https://doi.org/10.3389/fchem.2021.787194)
Azad A, Dinarvand M, Nematollahi A, Swift J, Lutze-Mann L, Vafaee F (2021) A comprehensive integrated
[drug similarity resource for in-silico drug repositioning and beyond. Brief Bioinf 22:bbaa126. https://​](https://doi.org/10.1093/bib/bbaa126)
[doi.​org/​10.​1093/​bib/​bbaa1​26](https://doi.org/10.1093/bib/bbaa126)
Azad A, Fatima S, Capraro A, Waters SA, Vafaee FJP (2021b) Integrative resource for network-based
investigation of COVID-19 combinatorial drug repositioning and mechanism of action. Patterns
2:100325
Baeder DY, Yu G, Hozé N, Rolff J, Regoes RR (2016) Antimicrobial combinations: bliss independence
and Loewe additivity derived from mechanistic multi-hit models. Philos Trans R Soc B Biol Sci
[371:20150294. https://​doi.​org/​10.​1098/​rstb.​2015.​0294](https://doi.org/10.1098/rstb.2015.0294)
Bao X, Sun J, Yi M, Qiu J, Chen X, Shuai SC, Zhao Q (2023) MPFFPSDC: a multi-pooling feature fusion
model for predicting synergistic drug combinations. Methods. [https://​doi.​org/​10.​1016/j.​ymeth.​2023.​](https://doi.org/10.1016/j.ymeth.2023.06.006)
[06.​006](https://doi.org/10.1016/j.ymeth.2023.06.006)

Berenbaum MC (1989) What is synergy? Pharmacol Rev 41:93–141
Bobrowski T, Chen L, Eastman RT, Itkin Z, Shinn P, Chen CZ, Guo H, Zheng W, Michael S, Simeonov A
(2021) Synergistic and antagonistic drug combinations against SARS-CoV-2. Mol Ther 29:873–885.
[https://​doi.​org/​10.​1016/j.​ymthe.​2020.​12.​016](https://doi.org/10.1016/j.ymthe.2020.12.016)
Bongini P, Bianchini M, Scarselli F (2021) Molecular generative graph neural networks for drug discovery.
[Neurocomputing 450:242–252. https://​doi.​org/​10.​1016/j.​neucom.​2021.​04.​039](https://doi.org/10.1016/j.neucom.2021.04.039)
Cai L, Lu C, Xu J, Meng Y, Wang P, Fu X, Zeng X, Su Y (2021) Drug repositioning based on the
[heterogeneous information fusion graph convolutional network. Brief Bioinf 22:bbab319. https://​doi.​](https://doi.org/10.1093/bib/bbab319)
[org/​10.​1093/​bib/​bbab3​19](https://doi.org/10.1093/bib/bbab319)
Cai R, Yuan J, Xu B, Hao Z (2021b) Sadga: structure-aware dual graph aggregation network for text-to-sql.
Adv Neural Inf Process Syst 34:7664–7676
Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP (2002) SMOTE: synthetic minority over-sampling
[technique. J Artif Intell Res 16:321–357. https://​doi.​org/​10.​1613/​jair.​953](https://doi.org/10.1613/jair.953)
Chen D, Liu X, Yang Y, Yang H, Lu P (2015) Systematic synergy modeling: understanding drug
synergy from a systems biology perspective. BMC Syst Biol 9:1–10. [https://​doi.​org/​10.​1186/​](https://doi.org/10.1186/s12918-015-0202-y)
[s12918-​015-​0202-y](https://doi.org/10.1186/s12918-015-0202-y)
Chen H, Lu Y, Yang Y, Rao Y (2022) A drug combination prediction framework based on graph
[convolutional network and heterogeneous information. IEEE/ACM Trans Comput Biol Bioinf. https://​](https://doi.org/10.1109/TCBB.2022.3224734)
[doi.​org/​10.​1109/​TCBB.​2022.​32247​34](https://doi.org/10.1109/TCBB.2022.3224734)
Cheng F, Kovács IA, Barabási A-L (2019) Network-based prediction of drug combinations. Nat Commun
[10:1197. https://​doi.​org/​10.​1038/​s41467-​019-​09186-x](https://doi.org/10.1038/s41467-019-09186-x)
Cokol M, Chua HN, Tasan M, Mutlu B, Weinstein ZB, Suzuki Y, Nergiz ME, Costanzo M, Baryshnikova A,
[Giaever G (2011) Systematic exploration of synergistic drug pairs. Mol Syst Biol 7:544. https://​doi.​](https://doi.org/10.1038/msb.2011.71)
[org/​10.​1038/​msb.​2011.​71](https://doi.org/10.1038/msb.2011.71)
Data E (2017) Orange book: approved drug products with therapeutic equivalence evaluations
Deng Y, Yu S, Deng L, Liu H, Liu X, Luo Y (2021) A multi-task graph convolutional network modeling
of drug-drug interactions and synergistic efficacy. In: 2021 IEEE International Conference on
[Bioinformatics and Biomedicine (BIBM) IEEE, 368–371. https://​doi.​org/​10.​1109/​BIBM5​2615.​2021.​](https://doi.org/10.1109/BIBM52615.2021.9669575)
[96695​75](https://doi.org/10.1109/BIBM52615.2021.9669575)

Duarte D, Vale N (2022) Evaluation of synergism in drug combinations and reference models for future
orientations in oncology. Curr Res Pharmacol Drug Discov 3:100110. [https://​doi.​org/​10.​1016/j.​](https://doi.org/10.1016/j.crphar.2022.100110)
[crphar.​2022.​100110](https://doi.org/10.1016/j.crphar.2022.100110)
Fisusi FA, Akala EO (2019) Drug combinations in breast cancer therapy. Pharm Nanotechnol 7:3–23.

[https://​doi.​org/​10.​2174/​22117​38507​66619​01221​11224](https://doi.org/10.2174/2211738507666190122111224)
Food and Administration (2018) Combination product definition combination product types
Forcina GC, Conlon M, Wells A, Cao JY, Dixon SJ (2017) Systematic quantification of population cell
death kinetics in mammalian cells. Cell Syst 4(600–610):e6. [https://​doi.​org/​10.​1016/j.​cels.​2017.​05.​](https://doi.org/10.1016/j.cels.2017.05.002)
[002](https://doi.org/10.1016/j.cels.2017.05.002)

Gan Y, Huang X, Guo W, Yan C, Zou G (2023) Predicting synergistic anticancer drug combination based
on low-rank global attention mechanism and bilinear predictor. Bioinformatics 39:btad607. [https://​](https://doi.org/10.1093/bioinformatics/btad607)
[doi.​org/​10.​1093/​bioin​forma​tics/​btad6​07](https://doi.org/10.1093/bioinformatics/btad607)

# 1 3


**49** Page 34 of 38



M. Besharatifard, F. Vafaee



Gao C, Yin S, Wang H, Wang Z, Du Z, Li X (2023) Medical-knowledge-based graph neural network for
medication combination prediction. IEEE Trans Neural Netw Learn Syst. [https://​doi.​org/​10.​1109/​](https://doi.org/10.1109/TNNLS.2023.3266490)
[TNNLS.​2023.​32664​90](https://doi.org/10.1109/TNNLS.2023.3266490)

García-Fuente A, Vázquez F, Viéitez JM, Garcia Alonso FJ, Martín JI, Ferrer J (2018) CISNE: an accurate
[description of dose-effect and synergism in combination therapies. Sci Rep 8:4964. https://​doi.​org/​10.​](https://doi.org/10.1038/s41598-018-23321-6)
[1038/​s41598-​018-​23321-6](https://doi.org/10.1038/s41598-018-23321-6)

Gilvary C, Dry JR, Elemento O (2019) Multi-task learning predicts drug combination synergy in cells and
[in the clinic. BioRxiv. https://​doi.​org/​10.​1101/​576017](https://doi.org/10.1101/576017)
Goldoni M, Johansson C (2007) A mathematical approach to study combined effects of toxicants in vitro:
evaluation of the Bliss independence criterion and the Loewe additivity model. Toxicol In Vitro
[21:759–769. https://​doi.​org/​10.​1016/j.​tiv.​2007.​03.​003](https://doi.org/10.1016/j.tiv.2007.03.003)
Gunawan I, Vafaee F, Meijering E, Lock JG (2023) An introduction to representation learning for single-cell
[data analysis. Cell Rep Methods. https://​doi.​org/​10.​1016/j.​crmeth.​2023.​100547](https://doi.org/10.1016/j.crmeth.2023.100547)
Hamilton WL (2020) Graph representation learning. McGill University, Montreal
Hamilton W, Ying Z, Leskovec J (2017) Inductive representation learning on large graphs. Adv Neural
Inf Process Syst. Vol 30
Hell F, Taha Y, Hinz G, Heibei S, Müller H, Knoll A (2020) Graph convolutional neural network for
[a pharmacy cross-selling recommender system. Information 11:525. https://​doi.​org/​10.​3390/​info1​](https://doi.org/10.3390/info11110525)
[11105​25](https://doi.org/10.3390/info11110525)

Holbeck SL, Camalier R, Crowell JA, Govindharajulu JP, Hollingshead M, Anderson LW, Polley
E, Rubinstein L, Srivastava A, Wilsker D (2017) The national cancer institute ALMANAC:
a comprehensive screening resource for the detection of anticancer drug pairs with enhanced
therapeutic ActivityNCI ALMANAC of approved cancer drug combinations. Can Res 77:3564–
[3576. https://​doi.​org/​10.​1158/​0008-​5472.​CAN-​17-​0489](https://doi.org/10.1158/0008-5472.CAN-17-0489)
Hosseini S-R, Zhou X (2023) CCSynergy: an integrative deep-learning framework enabling contextaware prediction of anti-cancer drug synergy. Brief Bioinf 24:bbac588. [https://​doi.​org/​10.​1093/​](https://doi.org/10.1093/bib/bbac588)
[bib/​bbac5​88](https://doi.org/10.1093/bib/bbac588)

Hu J, Gao J, Fang X, Liu Z, Wang F, Huang W, Wu H, Zhao G (2022) DTSyn: a dual-transformer-based
[neural network to predict synergistic drug combinations. Brief Bioinf 23:bbac302. https://​doi.​org/​](https://doi.org/10.1093/bib/bbac302)
[10.​1093/​bib/​bbac3​02](https://doi.org/10.1093/bib/bbac302)

Hu Z, Yu Q, Gao YX, Guo L, Song T, Li Y, King I (2023) Drug synergistic combinations predictions
via large-scale pre-training and graph structure learning. In: Research in computational molecular
biology: 27th annual international conference, RECOMB 2023, Istanbul, Turkey, April 16–19,
[2023, Proceedings, 2023. Springer Nature, 265. https://​doi.​org/​10.​48550/​arXiv.​2301.​05931](https://doi.org/10.48550/arXiv.2301.05931)
Ianevski A, Giri AK, Aittokallio T (2020) SynergyFinder 2.0: visual analytics of multi-drug combination
[synergies. Nucleic Acids Res 48:W488–W493. https://​doi.​org/​10.​1093/​nar/​gkaa2​16](https://doi.org/10.1093/nar/gkaa216)
Jiang P, Huang S, Fu Z, Sun Z, Lakowski TM, Hu P (2020) Deep graph embedding for prioritizing
synergistic anticancer drug combinations. Comput Struct Biotechnol J 18:427–438. [https://​doi.​](https://doi.org/10.1016/j.csbj.2020.02.006)
[org/​10.​1016/j.​csbj.​2020.​02.​006](https://doi.org/10.1016/j.csbj.2020.02.006)
Jiang D, Wu Z, Hsieh C-Y, Chen G, Liao B, Wang Z, Shen C, Cao D, Wu J, Hou T (2021) Could
graph neural networks learn better molecular representation for drug discovery? A comparison
study of descriptor-based and graph-based models. J Cheminf 13:1–23. [https://​doi.​org/​10.​1186/​](https://doi.org/10.1186/s13321-020-00479-8)
[s13321-​020-​00479-8](https://doi.org/10.1186/s13321-020-00479-8)

Jin W, Stokes JM, Eastman RT, Itkin Z, Zakharov AV, Collins JJ, Jaakkola TS, Barzilay R (2021) Deep
learning identifies synergistic drug combinations for treating COVID-19. Proc Natl Acad Sci
[118:e2105070118. https://​doi.​org/​10.​1073/​pnas.​21050​70118](https://doi.org/10.1073/pnas.2105070118)
Jumper J, Evans R, Pritzel A, Green T, Figurnov M, Ronneberger O, Tunyasuvunakool K, Bates R,
Žídek A, Potapenko A (2021) Highly accurate protein structure prediction with AlphaFold. Nature
[596:583–589. https://​doi.​org/​10.​1038/​s41586-​021-​03819-2](https://doi.org/10.1038/s41586-021-03819-2)
Karimi M, Hasanzadeh A, Shen Y (2020) Network-principled deep generative models for designing drug
[combinations as graph sets. Bioinformatics 36:i445–i454. https://​doi.​org/​10.​1093/​bioin​forma​tics/​](https://doi.org/10.1093/bioinformatics/btaa317)
[btaa3​17](https://doi.org/10.1093/bioinformatics/btaa317)

[Khoshraftar S, An A (2022) A survey on graph representation learning methods. arXiv preprint arXiv:​](http://arxiv.org/abs/2204.01855)

[2204.​01855. https://​doi.​org/​10.​48550/​arXiv.​2204.​01855](http://arxiv.org/abs/2204.01855)
[Kipf TN, Welling M (2016) Variational graph auto-encoders. arXiv preprint arXiv:​1611.​07308. https://​](http://arxiv.org/abs/1611.07308)

[doi.​org/​10.​48550/​arXiv.​1611.​07308](https://doi.org/10.48550/arXiv.1611.07308)
Koch FC, Sutton GJ, Voineagu I, Vafaee F (2021) Supervised application of internal validation measures
to benchmark dimensionality reduction methods in scRNA-seq data. Brief Bioinf 22:bbab304.
[https://​doi.​org/​10.​1093/​bib/​bbab3​04](https://doi.org/10.1093/bib/bbab304)

# 1 3


A review on graph neural networks for predicting synergistic…



Page 35 of 38 **49**



Krasoulis A, Antonopoulos N, Pitsikalis V, Theodorakis S (2022) DENVIS: scalable and highthroughput virtual screening using graph neural networks with atomic and surface protein pocket
[features. J Chem Inf Model 62:4642–4659. https://​doi.​org/​10.​1021/​acs.​jcim.​2c010​57](https://doi.org/10.1021/acs.jcim.2c01057)
Kucuksayan E, Bozkurt F, Yilmaz MT, Sircan-Kucuksayan A, Hanikoglu A, Ozben T (2021) A new
combination strategy to enhance apoptosis in cancer cells by using nanoparticles as biocompatible
[drug delivery carriers. Sci Rep 11:13027. https://​doi.​org/​10.​1038/​s41598-​021-​92447-x](https://doi.org/10.1038/s41598-021-92447-x)
Kumar V, Dogra N (2022) A comprehensive review on deep synergistic drug prediction techniques for
[cancer. Arch Comput Methods Eng 29:1443–1461. https://​doi.​org/​10.​1007/​s11831-​021-​09617-3](https://doi.org/10.1007/s11831-021-09617-3)
Kuru HI, Tastan O, Cicek AE (2021) MatchMaker: a deep learning framework for drug synergy prediction.
[IEEE/ACM Trans Comput Biol Bioinf 19:2334–2344. https://​doi.​org/​10.​1109/​TCBB.​2021.​30867​02](https://doi.org/10.1109/TCBB.2021.3086702)
Lederer S, Dijkstra TM, Heskes T (2019) Additive dose response models: defining synergy. Front
[Pharmacol 10:1384. https://​doi.​org/​10.​3389/​fphar.​2019.​01384](https://doi.org/10.3389/fphar.2019.01384)
Lee H (2021) Better inference with graph regularization. Carnegie Mellon University, Pittsburgh
Lehár J, Zimmermann GR, Krueger AS, Molnar RA, Ledell JT, Heilbut AM, Short GF III, Giusti LC,
Nolan GP, Magid OA, Lee MS (2007) Chemical combination effects predict connectivity in
biological systems. Mol Syst Biol 3(1):80
Li H, Zou L, Kowah JA, He D, Wang L, Yuan M, Liu X (2023) Predicting drug synergy and discovering
new drug combinations based on a graph autoencoder and convolutional neural network.
[Interdiscip Sci Comput Life Sci 15:316–330. https://​doi.​org/​10.​1007/​s12539-​023-​00558-y](https://doi.org/10.1007/s12539-023-00558-y)
Liang C, Shang M, Luo J (2021) Cancer subtype identification by consensus guided graph autoencoders.
[Bioinformatics 37:4779–4786. https://​doi.​org/​10.​1093/​bioin​forma​tics/​btab5​35](https://doi.org/10.1093/bioinformatics/btab535)
Licciardello MP, Ringler A, Markt P, Klepsch F, Lardeau C-H, Sdelci S, Schirghuber E, Müller AC,
Caldera M, Wagner A (2017) A combinatorial screen of the CLOUD uncovers a synergy targeting
[the androgen receptor. Nat Chem Biol 13:771–778. https://​doi.​org/​10.​1038/​nchem​bio.​2382](https://doi.org/10.1038/nchembio.2382)
Lin M, Wen K, Zhu X, Zhao H, Sun X (2023) Graph autoencoder with preserving node attribute
[similarity. Entropy 25:567. https://​doi.​org/​10.​3390/​e2504​0567](https://doi.org/10.3390/e25040567)
Liu Q, Xie L (2021) TranSynergy: mechanism-driven interpretable deep neural network for the
synergistic prediction and pathway deconvolution of drug combinations. PLoS Comput Biol
[17:e1008653. https://​doi.​org/​10.​1371/​journ​al.​pcbi.​10086​53](https://doi.org/10.1371/journal.pcbi.1008653)
Liu Q, Yin X, Languino LR, Altieri DC (2018) Evaluation of drug combination effect using a bliss
independence dose–response surface model. Stat Biopharm Res 10:112–122. [https://​doi.​org/​10.​](https://doi.org/10.1080/19466315.2018.1437071)
[1080/​19466​315.​2018.​14370​71](https://doi.org/10.1080/19466315.2018.1437071)

Liu X, Song C, Liu S, Li M, Zhou X, Zhang W (2022) Multi-way relation-enhanced hypergraph
representation learning for anti-cancer drug synergy prediction. Bioinformatics 38:4782–4789.
[https://​doi.​org/​10.​1093/​bioin​forma​tics/​btac5​79](https://doi.org/10.1093/bioinformatics/btac579)
Loewe S (1953) The problem of synergism and antagonism of combined drugs. Arzneimittelforschung
3:285–290

Luan S, Hua C, Lu Q, Zhu J, Zhao M, Zhang S, Chang X-W, Precup D (2022) Revisiting heterophily for
graph neural networks. Adv Neural Inf Process Syst 35:1362–1375
Lui GY, Shaw R, Schaub FX, Stork IN, Gurley KE, Bridgwater C, Diaz RL, Rosati R, Swan HA, Ince
TA (2020) BET, SRC, and BCL2 family inhibitors are synergistic drug combinations with PARP
[inhibitors in ovarian cancer. EBioMedicine. https://​doi.​org/​10.​1016/j.​ebiom.​2020.​102988](https://doi.org/10.1016/j.ebiom.2020.102988)
Lv J, Liu G, Ju Y, Sun Y, Guo W (2022) Prediction of synergistic antibiotic combinations by graph
[learning. Front Pharmacol 13:849006. https://​doi.​org/​10.​3389/​fphar.​2022.​849006](https://doi.org/10.3389/fphar.2022.849006)
Menden MP, Wang D, Mason MJ, Szalai B, Bulusu KC, Guan Y, Yu T, Kang J, Jeon M, Wolfinger R
(2019) Community assessment to advance computational prediction of cancer drug combinations
[in a pharmacogenomic screen. Nat Commun 10:2674. https://​doi.​org/​10.​1038/​s41467-​019-​09799-2](https://doi.org/10.1038/s41467-019-09799-2)
Min Y, Wenkel F, Perlmutter M, Wolf G (2022) Can hybrid geometric scattering networks help solve the
maximum clique problem? Adv Neural Inf Process Syst 35:22713–22724
Mokhtari RB, Homayouni TS, Baluch N, Morgatskaya E, Kumar S, Das B, Yeger H (2017) Combination
therapy in combating cancer. Oncotarget 8:38022–38043
Molina-Arcas M, Moore C, Rana S, van Maldegem F, Mugarza E, Romero-Clavijo P, Herbert E,
Horswell S, Li L-S, Janes MR (2019) Development of combination therapies to maximize the
[impact of KRAS-G12C inhibitors in lung cancer. Sci Transl Med 11:eaaw7999. https://​doi.​org/​10.​](https://doi.org/10.1126/scitranslmed.aaw7999)
[1126/​scitr​anslm​ed.​aaw79​99](https://doi.org/10.1126/scitranslmed.aaw7999)

Nguyen T, Le H, Quinn TP, Nguyen T, Le TD, Venkatesh S (2021) GraphDTA: predicting drug–target
binding affinity with graph neural networks. Bioinformatics 37:1140–1147. [https://​doi.​org/​10.​](https://doi.org/10.1093/bioinformatics/btaa921)
[1093/​bioin​forma​tics/​btaa9​21](https://doi.org/10.1093/bioinformatics/btaa921)

Numcharoenpinij N, Termsaithong T, Phunchongharn P, Piyayotai S (2022) Predicting synergistic
drug interaction with DNN and GAT. In: 2022 IEEE 5th International Conference on Knowledge

# 1 3


**49** Page 36 of 38



M. Besharatifard, F. Vafaee



Innovation and Invention (ICKII). IEEE, 24–29. [https://​doi.​org/​10.​1109/​ICKII​55100.​2022.​99835​](https://doi.org/10.1109/ICKII55100.2022.9983579)
[79](https://doi.org/10.1109/ICKII55100.2022.9983579)

Ocana A, Amir E, Yeung C, Seruga B, Tannock I (2012) How valid are claims for synergy in published
[clinical studies? Ann Oncol 23:2161–2166. https://​doi.​org/​10.​1093/​annonc/​mdr608](https://doi.org/10.1093/annonc/mdr608)
O’Neil J, Benita Y, Feldman I, Chenard M, Roberts B, Liu Y, Li J, Kral A, Lejnine S, Loboda A (2016)
An unbiased oncology compound screen to identify novel combination strategies. Mol Cancer
[Ther 15:1155–1162. https://​doi.​org/​10.​1158/​1535-​7163.​MCT-​15-​0843](https://doi.org/10.1158/1535-7163.MCT-15-0843)
Pearson RA, Wicha SG, Okour M (2023) Drug combination modeling: methods and applications in drug
[development. J Clin Pharmacol 63:151–165. https://​doi.​org/​10.​1002/​jcph.​2128](https://doi.org/10.1002/jcph.2128)
Preuer K, Lewis RP, Hochreiter S, Bender A, Bulusu KC, Klambauer G (2018) DeepSynergy: predicting
anti-cancer drug synergy with Deep Learning. Bioinformatics 34:1538–1546. [https://​doi.​org/​10.​](https://doi.org/10.1093/bioinformatics/btx806)
[1093/​bioin​forma​tics/​btx806](https://doi.org/10.1093/bioinformatics/btx806)
Rani SM (2012) Graph neural network for minimum dominating set. Int J Comput Appl.Vol. 56
Riva L, Yuan S, Yin X, Martin-Sancho L, Matsunaga N, Pache L, Burgstaller-Muehlbacher S, de Jesus PD,
Teriete P, Hull MV (2020) Discovery of SARS-CoV-2 antiviral drugs through large-scale compound
[repurposing. Nature 586:113–119. https://​doi.​org/​10.​1038/​s41586-​020-​2577-1](https://doi.org/10.1038/s41586-020-2577-1)
Rozemberczki B, Gogleva A, Nilsson S, Edwards G, Nikolov A, Papa E (2022) MOOMIN: deep molecular
omics network for anti-cancer drug combination therapy. In: Proceedings of the 31st ACM
[international conference on information & knowledge management. pp 3472–3483. https://​doi.​org/​10.​](https://doi.org/10.1145/3511808.3557146)
[1145/​35118​08.​35571​46](https://doi.org/10.1145/3511808.3557146)

Shao K, Zhang Y, Wen Y, Zhang Z, He S, Bo X (2022) DTI-HETA: prediction of drug–target interactions
based on GCN and GAT on heterogeneous graph. Brief Bioinf 23:109. [https://​doi.​org/​10.​1093/​bib/​](https://doi.org/10.1093/bib/bbac109)
[bbac1​09](https://doi.org/10.1093/bib/bbac109)

Singh H, Rana PS, Singh U (2018) Prediction of drug synergy in cancer using ensemble-based machine
[learning techniques. Mod Phys Lett B 32:1850132. https://​doi.​org/​10.​1142/​S0217​98491​85013​24](https://doi.org/10.1142/S0217984918501324)
Sofaer HR, Hoeting JA, Jarnevich CS (2019) The area under the precision-recall curve as a performance
metric for rare binary events. Methods Ecol Evol 10:565–577. [https://​doi.​org/​10.​1111/​2041-​210X.​](https://doi.org/10.1111/2041-210X.13140)
[13140](https://doi.org/10.1111/2041-210X.13140)
Son J, Kim D (2021) Development of a graph convolutional neural network model for efficient prediction
of protein-ligand binding affinities. PLoS ONE 16:e0249404. [https://​doi.​org/​10.​1371/​journ​al.​pone.​](https://doi.org/10.1371/journal.pone.0249404)
[02494​04](https://doi.org/10.1371/journal.pone.0249404)
Sühnel J (1992) Zero interaction response surfaces, interaction functions and difference response surfaces
for combinations of biologically active agents. Arzneim Forsch 42:1251–1251
Sun M, Zhao S, Gilvary C, Elemento O, Zhou J, Wang F (2020a) Graph convolutional networks for
computational drug development and discovery. Brief Bioinform 21:919–935. [https://​doi.​org/​10.​](https://doi.org/10.1093/bib/bbz042)
[1093/​bib/​bbz042](https://doi.org/10.1093/bib/bbz042)

Sun Z, Huang S, Jiang P, Hu P (2020b) DTF: deep tensor factorization for predicting anticancer drug
[synergy. Bioinformatics 36:4483–4489. https://​doi.​org/​10.​1093/​bioin​forma​tics/​btaa2​87](https://doi.org/10.1093/bioinformatics/btaa287)
Torkamannia A, Omidi Y, Ferdousi R (2022) A review of machine learning approaches for drug synergy
[prediction in cancer. Brief Bioinf. https://​doi.​org/​10.​1093/​bib/​bbac0​75](https://doi.org/10.1093/bib/bbac075)
Veličković P, Cucurull G, Casanova A, Romero A, Lio P, Bengio Y (2017) Graph attention networks. arXiv
[preprint arXiv:​1710.​10903. https://​doi.​org/​10.​48550/​arXiv.​1710.​10903](http://arxiv.org/abs/1710.10903)
Vijayan A, Fatima S, Sowmya A, Vafaee F (2022) Blood-based transcriptomic signature panel identification
[for cancer diagnosis: benchmarking of feature extraction methods. Brief Bioinf 23:bbac315. https://​](https://doi.org/10.1093/bib/bbac315)
[doi.​org/​10.​1093/​bib/​bbac3​15](https://doi.org/10.1093/bib/bbac315)
Wang J, Liu X, Shen S, Deng L, Liu H (2022a) DeepDDS: deep graph neural network with attention
mechanism to predict synergistic drug combinations. Brief Bioinf 23:bbab390. [https://​doi.​org/​10.​](https://doi.org/10.1093/bib/bbab390)
[1093/​bib/​bbab3​90](https://doi.org/10.1093/bib/bbab390)

Wang X, Zhu H, Jiang Y, Li Y, Tang C, Chen X, Li Y, Liu Q, Liu Q (2022b) PRODeepSyn: predicting
anticancer synergistic drug combinations by embedding cell lines with protein–protein interaction
[network. Brief Bioinf 23:bbab587. https://​doi.​org/​10.​1093/​bib/​bbab5​87](https://doi.org/10.1093/bib/bbab587)
Wang Z, Liu M, Luo Y, Xu Z, Xie Y, Wang L, Cai L, Qi Q, Yuan Z, Yang T (2022c) Advanced graph
and sequence neural networks for molecular property prediction and drug discovery. Bioinformatics
[38:2579–2586. https://​doi.​org/​10.​1093/​bioin​forma​tics/​btac1​12](https://doi.org/10.1093/bioinformatics/btac112)
Wang X, Zhu H, Chen D, Yu Y, Liu Q, Liu Q (2023) A complete graph-based approach with multi-task
[learning for predicting synergistic drug combinations. Bioinformatics 39:btad351. https://​doi.​org/​10.​](https://doi.org/10.1093/bioinformatics/btad351)
[1093/​bioin​forma​tics/​btad3​51](https://doi.org/10.1093/bioinformatics/btad351)

Wang X, Ji H, Shi C, Wang B, Ye Y, Cui P, Yu PS (2019) Heterogeneous graph attention network. The
[world wide web conference. pp 2022–2032. https://​doi.​org/​10.​1145/​33085​58.​33135​62](https://doi.org/10.1145/3308558.3313562)

# 1 3


A review on graph neural networks for predicting synergistic…



Page 37 of 38 **49**



Wooten DJ, Meyer CT, Lubbock AL, Quaranta V, Lopez CF (2021) MuSyC is a consensus framework that
[unifies multi-drug synergy metrics for combinatorial drug discovery. Nat Commun 12:4607. https://​](https://doi.org/10.1038/s41467-021-24789-z)
[doi.​org/​10.​1038/​s41467-​021-​24789-z](https://doi.org/10.1038/s41467-021-24789-z)
Wu L, Wen Y, Leng D, Zhang Q, Dai C, Wang Z, Liu Z, Yan B, Zhang Y, Wang J (2022) Machine learning
[methods, databases and tools for drug combination prediction. Brief Bioinf 23:bbab355. https://​doi.​](https://doi.org/10.1093/bib/bbab355)
[org/​10.​1093/​bib/​bbab3​55](https://doi.org/10.1093/bib/bbab355)
[Xu K, Hu W, Leskovec J, Jegelka S (2018) How powerful are graph neural networks? arXiv preprint arXiv:​](http://arxiv.org/abs/1810.00826)

[1810.​00826. https://​doi.​org/​10.​48550/​arXiv.​1810.​00826](http://arxiv.org/abs/1810.00826)
Yadav B, Wennerberg K, Aittokallio T, Tang J (2015) Searching for drug synergy in complex dose–response
[landscapes using an interaction potency model. Comput Struct Biotechnol J 13:504–513. https://​doi.​](https://doi.org/10.1016/j.csbj.2015.09.001)
[org/​10.​1016/j.​csbj.​2015.​09.​001](https://doi.org/10.1016/j.csbj.2015.09.001)
Yang H, Qin C, Li YH, Tao L, Zhou J, Yu CY, Xu F, Chen Z, Zhu F, Chen YZ (2016) Therapeutic target
database update 2016: enriched resource for bench to clinical drug target and targeted pathway
[information. Nucleic Acids Res 44:D1069–D1074. https://​doi.​org/​10.​1093/​nar/​gkv12​30](https://doi.org/10.1093/nar/gkv1230)
Yang J, Xu Z, Wu WKK, Chu Q, Zhang Q (2021) GraphSynergy: a network-inspired deep learning model
[for anticancer drug combination prediction. J Am Med Inform Assoc 28:2336–2345. https://​doi.​org/​](https://doi.org/10.1093/jamia/ocab162)
[10.​1093/​jamia/​ocab1​62](https://doi.org/10.1093/jamia/ocab162)
Yang Z, Ding M, Xu B, Yang H, Tang J (2022) Stam: a spatiotemporal aggregation method for graph neural
network-based recommendation. Proc ACM Web Conf 2022:3217–3228. [https://​doi.​org/​10.​1145/​](https://doi.org/10.1145/3485447.3512041)
[34854​47.​35120​41](https://doi.org/10.1145/3485447.3512041)

Yu Y, Huang K, Zhang C, Glass LM, Sun J, Xiao C (2021) SumGNN: multi-typed drug interaction
prediction via efficient knowledge graph summarization. Bioinformatics 37:2988–2995. [https://​doi.​](https://doi.org/10.1093/bioinformatics/btab207)
[org/​10.​1093/​bioin​forma​tics/​btab2​07](https://doi.org/10.1093/bioinformatics/btab207)
Yue Y, Liu Y, Hao L, Lei H, He S (2023) Improving therapeutic synergy score predictions with adverse
[effects using multi-task heterogeneous network learning. Brief Bioinf 24:bbac564. https://​doi.​org/​10.​](https://doi.org/10.1093/bib/bbac564)
[1093/​bib/​bbac5​64](https://doi.org/10.1093/bib/bbac564)

Zagidullin B, Aldahdooh J, Zheng S, Wang W, Wang Y, Saad J, Malyutina A, Jafari M, Tanoli Z, Pessia A
(2019) DrugComb: an integrative cancer drug combination data portal. Nucleic Acids Res 47:W43–
[W51. https://​doi.​org/​10.​1093/​nar/​gkz337](https://doi.org/10.1093/nar/gkz337)
Zagidullin B, Wang Z, Guan Y, Pitkänen E, Tang J (2021) Comparative analysis of molecular fingerprints in
[prediction of drug combination effects. Brief bioinf 22:bbab291. https://​doi.​org/​10.​1093/​bib/​bbab2​91](https://doi.org/10.1093/bib/bbab291)
Zaheer M, Kottur S, Ravanbakhsh S, Poczos BR, Salakhutdinov, Smola AJ (2017) Deep sets. NIPS
Zarin DA, Tse T, Williams RJ, Califf RM, Ide NC (2011) The ClinicalTrials. gov results database—update
[and key issues. N Engl J Med 364:852–860. https://​doi.​org/​10.​1056/​NEJMs​a1012​065](https://doi.org/10.1056/NEJMsa1012065)
Zhang P, Tu S (2023) MGAE-DC: Predicting the synergistic effects of drug combinations through multichannel graph autoencoders. PLoS Comput Biol 19:e1010951. [https://​doi.​org/​10.​1371/​journ​al.​pcbi.​](https://doi.org/10.1371/journal.pcbi.1010951)
[10109​51](https://doi.org/10.1371/journal.pcbi.1010951)
Zhang P, Tu S, Zhang W, Xu L (2022) Predicting cell line-specific synergistic drug combinations through
[a relational graph convolutional network with attention mechanism. Brief Bioinf. https://​doi.​org/​10.​](https://doi.org/10.1093/bib/bbac403)
[1093/​bib/​bbac4​03](https://doi.org/10.1093/bib/bbac403)

Zhang Z, Chen L, Zhong F, Wang D, Jiang J, Zhang S, Jiang H, Zheng M, Li X (2022b) Graph neural
[network approaches for drug-target interactions. Curr Opin Struct Biol 73:102327. https://​doi.​org/​10.​](https://doi.org/10.1016/j.sbi.2021.102327)
[1016/j.​sbi.​2021.​102327](https://doi.org/10.1016/j.sbi.2021.102327)
Zhang G, Gao Z, Yan C, Wang J, Liang W, Luo J, Luo H (2023) KGANSynergy: knowledge graph attention
[network for drug synergy prediction. Brief Bioinf. https://​doi.​org/​10.​1093/​bib/​bbad1​67](https://doi.org/10.1093/bib/bbad167)
Zhang P, Tu S (2022) A knowledge graph embedding-based method for predicting the synergistic effects
of drug combinations. In: 2022 IEEE International Conference on Bioinformatics and Biomedicine
[(BIBM). IEEE, 1974–1981. https://​doi.​org/​10.​1109/​BIBM5​5620.​2022.​99954​66](https://doi.org/10.1109/BIBM55620.2022.9995466)
Zhang S, Xie L (2020) Improving attention mechanism in graph neural networks via cardinality
[preservation. In: IJCAI: proceedings of the conference, 2020. NIH Public Access, 1395. https://​doi.​](https://doi.org/10.24963/ijcai.2020/194)
[org/​10.​24963/​ijcai.​2020/​194](https://doi.org/10.24963/ijcai.2020/194)
Zhao H, Zheng K, Li Y, Wang J (2021) A novel graph attention model for predicting frequencies of drug–
[side effects from multi-view data. Brief Bioinf 22:bbab239. https://​doi.​org/​10.​1093/​bib/​bbab2​39](https://doi.org/10.1093/bib/bbab239)
Zhou J, Cui G, Hu S, Zhang Z, Yang C, Liu Z, Wang L, Li C, Sun M (2020) Graph neural networks: a
[review of methods and applications. AI Open 1:57–81. https://​doi.​org/​10.​1016/j.​aiopen.​2021.​01.​001](https://doi.org/10.1016/j.aiopen.2021.01.001)
Zhou S, Guo Z, Aggarwal C, Zhang X, Wang S (2022) Link prediction on heterophilic graphs via
disentangled representation learning. arXiv preprint [arXiv:​2208.​01820.](http://arxiv.org/abs/2208.01820) [https://​doi.​org/​10.​48550/​](https://doi.org/10.48550/arXiv.2208.01820)
[arXiv.​2208.​01820](https://doi.org/10.48550/arXiv.2208.01820)

Zhu X, Zhang Y, Zhang Z, Guo D, Li Q, Li Z (2022) Interpretability evaluation of botnet detection
model based on graph neural network. In: IEEE INFOCOM 2022-IEEE Conference on Computer

# 1 3


**49** Page 38 of 38



M. Besharatifard, F. Vafaee



Communications Workshops (INFOCOM WKSHPS) IEEE, 1–6. [https://​doi.​org/​10.​1109/​INFOC​](https://doi.org/10.1109/INFOCOMWKSHPS54753.2022.9798287)
[OMWKS​HPS54​753.​2022.​97982​87](https://doi.org/10.1109/INFOCOMWKSHPS54753.2022.9798287)


**Publisher’s Note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and
institutional affiliations.

# 1 3


