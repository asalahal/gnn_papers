Downloaded from orbit.dtu.dk on: Aug 05, 2025


**Application of Interpretable Group-embedded Graph Neural Networks for Pure**
**Compound Properties**


**Aouichaoui, Adem R.N.; Fan, Fan; Abildskov, Jens; Sin, Gürkan**


_Published in:_
Computers and Chemical Engineering


_Link to article, DOI:_
[10.1016/j.compchemeng.2023.108291](https://doi.org/10.1016/j.compchemeng.2023.108291)


_Publication date:_
2023


_Document Version_
Publisher's PDF, also known as Version of record


[Link back to DTU Orbit](https://orbit.dtu.dk/en/publications/47e9bea6-6a58-4f81-b73c-3d3c3d31776d)


_Citation (APA):_
Aouichaoui, A. R. N., Fan, F., Abildskov, J., & Sin, G. (2023). Application of Interpretable Group-embedded
Graph Neural Networks for Pure Compound Properties. _Computers and Chemical Engineering_, _176_, Article
[108291. https://doi.org/10.1016/j.compchemeng.2023.108291](https://doi.org/10.1016/j.compchemeng.2023.108291)


**General rights**
Copyright and moral rights for the publications made accessible in the public portal are retained by the authors and/or other copyright
owners and it is a condition of accessing publications that users recognise and abide by the legal requirements associated with these rights.


  - Users may download and print one copy of any publication from the public portal for the purpose of private study or research.

  - You may not further distribute the material or use it for any profit-making activity or commercial gain

  - You may freely distribute the URL identifying the publication in the public portal

If you believe that this document breaches copyright please contact us providing details, and we will remove access to the work immediately
and investigate your claim.


[Computers and Chemical Engineering 176 (2023) 108291](https://doi.org/10.1016/j.compchemeng.2023.108291)


Contents lists available at ScienceDirect

# Computers and Chemical Engineering


[journal homepage: www.elsevier.com/locate/compchemeng](https://www.elsevier.com/locate/compchemeng)

## Application of interpretable group-embedded graph neural networks for pure compound properties


Adem R.N. Aouichaoui, Fan Fan, Jens Abildskov, Gürkan Sin [* ]


_Process and Systems Engineering Center (PROSYS), Department of Chemical and Biochemical Engineering, Technical University of Denmark, Building 229, 2800 Kgs,_
_Lyngby, Denmark_



A R T I C L E I N F O


_Keywords:_
Deep-learning
Graph neural networks
Group-contribution models
Thermophysical properties
Interpretability
Pure compound properties


**1. Introduction**



A B S T R A C T


The ability to evaluate pure compound properties of various molecular species is an important prerequisite for
process simulation in general and in particular for computer-aided molecular design (CAMD). Current techniques
rely on group-contribution (GC) methods, which suffer from many drawbacks mainly the absence of contribu­
tions for specific groups. To overcome this challenge, in this work, we extended the range of interpretable graph
neural network (GNN) models for describing a wide range of pure component properties. The new model library
contains 30 different properties ranging from thermophysical, safety-related, and environmental properties. All
of these have been modeled with a suitable level of accuracy for compound screening purposes compared to
current GC models used within CAMD applications. Moreover, the developed models have been subjected to a
series of sanity checks using logical and thermodynamic constraints. Results show the importance of evaluating
the model across a range of properties to establish their thermodynamic consistency.



Molecules and chemical compounds are important for any chemical
engineering application (Frenkel, 2011). An estimate of the theoretically
available design space for organic compounds following the Lipinski
rule of bioavailability is 10 [60 ] (Reymond, 2015). For compounds up to 30
atoms, the estimate is approximately 10 [24 ] (Reymond, 2015). Several
studies have enumerated the possible compounds within a given design
constraint by considering a molecule as a mathematical graph, where
the atoms are replaced by graph nodes and chemical bonds by graph
edges. The GDB17 is a generated database of organic compounds con­
taining up to 17 atoms of Carbon (C), Nitrogen (N), Oxygen (O), Sulfur
(S), and Halogens (Fluorine (F), Chlorine (Cl), Bromine (Br), Iodine (I))
that resides over 166 billion organic compounds (Ruddigkeit et al.,
2012). The prospect of experimentally determining all relevant prop­
erties of such a chemical space is impractical for many research and
engineering studies. Especially considering the diverse properties
needed for various applications and domains covering ADME (absorp­
tion, distribution, metabolism, and excretion) properties, thermophys­
ical properties, and environmental and safety-related properties, not to
mention temperature and pressure sensitive properties and phase equi­
libria data. Commonly used predictive models are so-called quantitative


 - Corresponding author.
_E-mail address:_ [gsi@kt.dtu.dk (G. Sin).](mailto:gsi@kt.dtu.dk)



structure-property relationships (QSPR), which aim to establish a cor­
relation between the molecular structure and the target property of in­
terest (Katritzky et al., 2010). QSPR models take as input a numerical
translation of the molecular structure referred to as molecular descriptor
into a mathematical model that establishes the relation between the

descriptor and the target property (Gasteiger, 2016; Katritzky et al.,
2010). The nature of the descriptor and mathematical model are not
universal and no widespread consensus has been established on how to
best represent these. As such they widely differ depending on the field of
application. Atomic coordinates (x-y-z coordinates) and calculated
quantum mechanics-related properties are widely used for QSPR models
for catalyst design (Parveen et al., 2019). Molecular fingerprints derived
from algorithms detecting the presence of a set of predefined sub­
structures such as the Morgan fingerprints and extended connectivity
fingerprints (ECFP) are widely used for drug discovery (Liu and Zhou,
2008; Rogers and Hahn, 2010). For chemical engineering applications,
QSPR models have largely been based on the concept of group-additivity
models in the form of group-contribution (GC) models, where the
molecule is described in terms of the occurrence of a set of predefined
functional groups (Gani, 2019; Van Speybroeck et al., 2010). The
popularity of this approach has given rise to a wide range of methods for
describing the molecule in terms of groups (Tu, 1995; Constantinou and



[https://doi.org/10.1016/j.compchemeng.2023.108291](https://doi.org/10.1016/j.compchemeng.2023.108291)
Received 19 December 2022; Received in revised form 20 March 2023; Accepted 12 May 2023

Available online 13 May 2023
[0098-1354/© 2023 The Authors. Published by Elsevier Ltd. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).](http://creativecommons.org/licenses/by/4.0/)


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_



Gani, 1994; Hukkerikar et al., 2012b; Joback and Reid, 1987; Klince­
wicz and Reid, 1984; Marrero and Gani, 2001; Meier, 2021a). The
widespread use of such models is largely owing to their simplicity (linear
additive models), transparent and clear aspect of interpretability (each
group/fragment has a proper and fixed contribution towards the overall
property) as well as their availability across a wide range of properties
(Gani, 2019). One of the most detailed GC methods in terms of the
defined groups and domain of applicability assumes a three-level hier­
archal description of the molecule (Hukkerikar et al., 2012b). Each level
provides a higher resolution and larger substructures of the molecule
(220 first-order groups, 137 second-order groups, and 74 third-order
groups). The GC model has been applied to 18 datasets related to the
thermodynamic and physical properties of organic compounds covering
liquid molar volumes, acentric factor, octanol-water partition coeffi­
cient, solubility parameters as well as melting/boiling points, critical
point measurements, and enthalpy-related properties (energy of for­
mation and energy of phase transition) (Hukkerikar et al., 2012b). A
study further expanded on the already huge library of available prop­
erties to cover environmental-related properties (a total of 22 proper­
ties) (Hukkerikar et al., 2012a) and flammability-related properties (5
properties) (Frutiger et al., 2016a) making it the most widely applied GC
method across properties. Establishing a wide repository of property
models is necessary for in-silico screening purposes and plays an integral
part in any computer-aided molecular design (CAMD) framework.
Despite the advantages that GC methods present in terms of simplicity,
interpretability, and a wide range of availability, they do come with a set
of serious drawbacks. GC models are in essence linear additive models

and as such, they struggle in capturing the non-linearity that some
properties present. Potential proximity effects resulting from the relative
positioning of groups to each other in a molecule are completely
neglected. They are highly dependent on the data used for the regression
in terms of determining the contribution of various groups as well as the
absence of any established methods for determining the contribution of
missing groups in the dataset, which results in a sparse contribution
matrix. In an effort to remedy this, all data are usually used in the
regression process, which constrains the ability to perform
cross-validation (Hukkerikar et al., 2012b; Jhamb et al., 2020). As a
consequence, the ability to generalize to unseen compounds is not well
established. A final drawback resides in the definition of the groups, as
not all are theoretically motivated or based on an understanding of the
underlying phenomenon, which is especially true for the higher order
groups (2 [nd ] and 3 [rd ] order groups) (Hukkerikar et al., 2013). While this
ad-hoc definition for groups has improved the performance metrics of
the models, it is often a misleading improvement as errors are “absor­
bed”
by the defined group. A conclusion that is rooted in the fact that in
many cases, the groups only feature once in a given dataset, and as such,
the contribution of the group will take the value of the residual. Machine
learning (ML) and artificial intelligence (AI) techniques have been
applied to overcome some of the drawbacks presented by classical GC
models, especially for capturing the potential non-linearity present in
the datasets by replacing the linear model with Gaussian processes,
neural networks, and support vector machines as well as employing
cross-validation practices (Alshehri et al., 2022; Aouichaoui et al., 2021;
Mondejar et al., 2017). However, the remaining shortcomings have
largely been undealt with.
Recent advancements in the fields of ML and AI have the potential of
solving some of the remaining drawbacks of GC methods and even
replacing them completely. This can be done by performing feature
extraction and construction using general and accessible molecular in­
formation. Graph neural networks (GNNs) are among the most used
techniques for such a unified approach that combines the feature
extraction process and the regression process (Hwang et al., 2020;
Wieder et al., 2020; Zhou et al., 2018). A GNN takes as input a graph
representation of the molecule where nodes correspond to atoms and
edges correspond to the bonds linking the atoms. Each node and edge is
associated with a feature vector containing information concerning the



atoms and bond e.g. atom type, number of hydrogen attached, and bond
type. Through a series of arithmetic operations involving matrix multi­
plication, the graph representation is updated and converted into a
numerical vector that is used as the molecular descriptor. These oper­
ations are denoted message-passing framework and comprise the
mechanism that dictates the feature extraction process (Gilmer et al.,
2017). This mechanism produces a molecular representation that is
aware of the neighboring atoms and as such, it allows for taking prox­
imity effects into account. An important advantage of the GNN models is
that they are capable of learning a proper representation of the molecule
from general atomic and bond features. In this way, they are less prone
to dependency on the functional groups (representing specific atom/­
bond arrangements) defined in the group contribution methods. This
could potentially allow the models to represent unseen molecules during
training (Aouichaoui et al., 2023). In fact, recent years have witnessed a
rapid increase in GNN models for molecular property prediction each
developing and extending upon the existing models with the most recent
developments from the field of deep learning (DL) (Wieder et al., 2020;
Zhou et al., 2018). The first account of the GNN model was introduced
by Davenaud et al. (Duvenaud et al., 2015) and was among the first
models to learn specific molecular representation depending on the task.
Gilmer et al. provided the first definition for the message-passing
framework, which unifies many GNN models in the way they extract
the molecular representation (Gilmer et al., 2017). Many studies
involving new GNN models have been reported in the literature. For
example, Coley et al. used attributed molecular graphs that use an
extensive list of atom and bond features in order to produce a more
informed molecular representation (Coley et al., 2017). The model was
later altered to include bond-based message-passing rather than
node-based message passing (Yang et al., 2019). A recent study com­
bined the bond-based message-passing model with fixed descriptors
obtained from various chemoinformatics software and applied it to a set
of environmental properties (Zhang et al., 2022). Schweidtmann et al.
developed a higher-order GNN model for fuel ignition properties
(Schweidtmann et al., 2020). Xiong et al. combined recurrent neural
networks and the attention mechanism as part of the mechanism
generating the molecular representation and denoted the model atten­
tive FP _,_ achieving state-of-the-art performance compared to many
existing models and providing an added benefit in the form of increased
interpretability (Xiong et al., 2020). Zhang et al. proposed a three-level
hierarchical feature extraction model (FraGAT) conducted on the node
level of a molecular graph, a fragment level by enumerating all possible
fragments of a molecular and a junction-tree level by combining the
fragments into a new molecular graph (Zhang et al., 2021). Recently, a
model inspired by the FraGAT introduced the groups as defined by GC
models as fragments showcasing increased performance and faster
computation time compared to the original model (Aouichaoui et al.,
2023). Important to note that despite many of these studies and models
achieving state-of-the-art performance in many applications, they

–
similar to GC models- come with a set of drawbacks. Depending on the
data and complexity of the model, AI models are associated with long
training times. This however can be remedied by using dedicated
hardware in the form of graphical process units (GPUs). Another
drawback is that AI-based models are high parametric models with very
high degrees of freedom (DoF), which eventually result in parameter
identifiability issues and unstable training procedures. Another issue is
that despite gaining a lot of attention lately, the uncertainty quantifi­
cation of such models is also not well established, with many methods
available and no apparent conclusion on which method is best (Aoui­
chaoui et al., 2022a; Hirschfeld et al., 2020; Scalia et al., 2020). Some of
the applied methods include Monte-Carlo dropout, bootstrapping (either
from the training data or from the errors), and deep ensemble, with the
latter being one of the most widely used methods (Aouichaoui et al.,
2022a; Hirschfeld et al., 2020; Scalia et al., 2020). A further drawback is
the “black-box” nature of these models, a fact which might hinder their
wider acceptance and applicability, especially in domain applications



2


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Table 1**

Atom and bond information used to featurize the molecular graph.


Feature Explanation Type Size


**Atom features**

Atom type type of atom (C, N, O, S, F, Cl, Br, I, P) One-hot encoding 9
No. of bonds number of bonds attached to the atom (0, 1, 2, 3, 4, 5) One-hot encoding 6
No. of H’s number of Hydrogen attached to the atom (0,1,2,3,4) One-hot encoding 5
Valency explicit valency (0,1,2,3,4,5) One-hot encoding 6
Hybridization hybridization (sp, sp2, sp3, sp3d, sp3d2) One-hot encoding 5
Aromaticity whether the atom is part of an aromatic system (0, 1) binary 1
Chirality center whether the atom is a center of chirality (0,1) binary 1
Chirality type type of chirality the atom is involved in (R, S) One-hot encoding 2
Chirality tag tag assigned to the chiral center One-hot encoding 3
(“unspecified, tetrahedral_CW, tetrahedral_CWW)

Formal Charge charge assigned to individual atoms in a molecule Integer 1
**Bond features**

Bond type bond type (single, double, triple, aromatic) One-hot encoding 4
Conjugation whether the bond is conjugated (0, 1) Binary 1
Ring whether the bond is part of a ring (0, 1) Binary 1
Stereochemistry Bond stereochemistry (none, any, Z/E, Cis/Trans) One-hot encoding 6



that are traditionally based on a first principle understanding of the
underlying tasks and phenomenon. Recent studies have made significant
advancements in addressing the interpretability issue of deep-learning
models by proposing several approaches to gain insight into the
learning achieved by such models(Hasebe, 2021; Jim´enez-Luna et al.,
2021, 2020; Xiong et al., 2020; Zhang et al., 2021). In the context of
GNN models, interpretability can be achieved by feature attribution,
subgraph approaches, or the attention mechanism (Jim´enez-Luna et al.,
2021, 2020). The feature attribution approach consists of using the in­
tegrated gradient method in the backpropagation to produce an
importance score for the feature elements in each node and edge
(Jim´enez-Luna et al., 2021). The method has previously been used for
drug-related properties and showed that the method was able to high­
light structural elements that are consistent with known pharmacophore
motifs (Jim´enez-Luna et al., 2021). The study does however also report
some challenges with the method such as the presence of multi­
collinearity between features and instability in importance attribution
for structurally similar compounds (Jim´enez-Luna et al., 2021). How­
ever perhaps one major drawback is the fact that the insights are pro­
vided on an individual feature level and as such the insights are only
provided at a micro-level. The attention mechanism can easily be inte­
grated into the message-passing framework by applying a scoring
function or weighting function to evaluate the importance of either the
nodes or the bonds in the produced molecular descriptor generated
through the readout function. This method has been used to highlight
atoms and bonds with importance for water solubility of organic com­
pounds (Xiong et al., 2020). However, in reality, it is rather functional
groups and substructures of the molecule that influences the various
properties of a molecule. As such, substructure (or subgraph) based
approaches are more intuitive to chemists and the fundamental under­
standing of properties (Jim´enez-Luna et al., 2020). A recent study in­
tegrated the concept of functional groups from the GC methods with
GNNs to add an interpretable aspect to the GNN models (Aouichaoui
et al., 2023). The interpretability consisted of using the attention
mechanism to provide an importance ranking and color the fragments
based on their importance. The models (AGC and GroupGAT) were
tested in a range of properties to ensure the accuracy of the models is not
compromised compared to state-of-the-art models and the insights
provided were shown to be consistent with the insights gained from the
classical GC models.

In this work, we extend and study a large collection of property
models based on the AGC and GroupGAT models to rigorously test their
performance on a wide range of different properties for different ap­
plications. The establishment of such a collection of property models is
essential to construct a unified framework for the in-silico screening of
organic compounds for different end uses. A further added value of this



work resides in a set of “sanity” checks that prove the models obtained
abide by thermodynamic and logical constraints.


**2. AGC and GroupGAT: interpretable graph neural networks**
**using functional groups**


In this section, the concept of GNNs for molecular property predic­
tion is described. Furthermore, two GNNs that integrate functional
groups into their structure are reintroduced along with the underlying
mathematical schemes while highlighting the model hyperparameters.


_2.1. Graph Neural Networks for molecular property prediction_


GNNs are neural networks that operate on graph-structured data and
are capable of learning a numerical representation of the input data and
correlating the learned representation to the desired target output
(either categorical or continuous variables) in an end-to-end learning
framework (combining feature extraction and regression tasks). Since
molecules can be described in terms of molecular graphs (similar to the
ball and stick representation), they possess a suitable format for GNNs.
The graph (or molecule) is made of a set of nodes or vertices (atoms)
linked with edges (chemical bonds). Each node and edge is attributed (or
featurized) with a set of information relating to the atom or bonds
respectively. Table 1 showcases the atom and bond information used in
this work. The feature extraction task is done through three steps: the
message construction phase, the update phase, and the readout phase.
These operations are done node-wise, where the node considered is
denoted central node and the linked nodes are denoted neighboring
nodes.

During the message construction phase, the neighboring atom and
bonds are combined through a series of mathematical operations
(denoted message function) to produce a representation that is com­
bined with the currently hidden representation of the central node to
alter its representation (update phase). As such, the latent representa­
tion of a node is influenced by the environment/neighborhood in which
it is located. This is repeated “L” times corresponding to the number of
message passing layers, after which the graph representation is con­
verted into a vector representation through a readout function. The
vector representation is then used as input to a multi-layer perceptron
(MLP) (or a feedforward neural network) to correlate the obtained
representation to the target value. Both operations (feature extraction
and regression tasks) are combined through the backpropagation algo­
rithm and as such, a task-specific representation (learned molecular
representation or molecular descriptor) is produced.



3


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Fig. 1.** Schematic of GroupGAT. The top branch (1) shows a node-based representation extraction using an encoder. Middle branch (2) shows the fragment-based
representation using functional groups. The bottom branch (3) constructs a junction tree using the groups as nodes. All three representations are concatenated to
produce the overall representation of the molecule. The representation is used as input to the regressor in the form of a multi-layer perceptron to produce
a prediction.



_2.2. GroupGAT: Group-Contribution Graph Attention Network_


The GroupGAT model is inspired by the FraGAT model (Zhang et al.,
2021). The model has previously been benchmarked against a wide
range of message-passing neural network models including the FraGAT
(Zhang et al., 2021), attentive FP (Xiong et al., 2020), and MPNN
(Gilmer et al., 2017) and the D-MPNN (Yang et al., 2019). The study
showed that the GroupGAT had increased performance and increased
interpretability (Aouichaoui et al., 2023). The model is comprised of
three branches that each produce a representation operating on different
resolutions of the molecule, a similar approach to the three-level GC
models (Hukkerikar et al., 2012b). The hierarchal feature extraction is
conducted on the following levels:



1 Complete molecular graph with nodes and edges as the center of the
message-passing framework
2 Individual fragments or groups are considered individual graphs
3 A junction-tree model with groups (fragments) as the center of the
message-passing framework


Each branch produces a representation that is concatenated to pro­
duce the overall molecular descriptor. An important difference between
the FraGAT and the GroupGAT is that the groups (or fragments) defined
for the GroupGAT are taken from the well-established groups defined in
GC models (Hukkerikar et al., 2012b). This chemistry-informed frag­
mentation is inspired by chemists’ understanding of the underlying
mechanism in QSPR models (Meier, 2021a). GroupGAT only considers
the first-order groups since they do not present any overlaps between



4


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Fig. 2.** Schematic overview of the attentive FP model: a complete overview of the attentive FP model (a), the node embedding scheme (b), and the graph embedding
scheme (c)



substructures (Hukkerikar et al., 2012b; Marrero and Gani, 2001). A
schematic of the overall structure of GroupGAT is shown in Fig. 1. For
molecules that contain fragments not previously defined, the algorithm
assigns the fragment as _“unknown”_ . Though the fragment is considered
unknown in the context of the defined groups, the model retains
knowledge of the atoms and the arrangements associated with it and as
such this does not pose a challenge for the model and does not limit its
applicability to molecules that are not entirely fragmented in terms of
the predefined set of groups. As such, the fragmentation can be
considered a guide for the model to fragment the molecule as much as
possible based on prior chemistry knowledge.
In essence, any GNN can be used to represent the encoder block.
However, to retain the interpretability of the models, the attentive FP
model is used (Xiong et al., 2020). The attentive FP model combines the
graph attention mechanism with a unique readout function to produce
the molecular representation. In the encoder, the representation is
constructed over two steps: a node-level and a graph-level encoding
process. The node-level embedding is very similar to the commonly used
message-passing framework where the hidden node representation is
repeatedly updated with information from neighboring atoms by
stacking layers. The node hidden representation ( _h_ _v_ ) is initialized ac­
_atom_
cording to Eq. (1) using the initial atoms features ( _x_ _v_ ) and adjustable
weight matrix ( _W_ ) that is determined through backpropagation. The
neighboring node hidden representations ( _h_ _w_ ) are initialized according
_atom_ _bond_
to Eq. (2) using the initial atom (x _w_ ) and edge features ( _x_ _v,w_ ). The
attention mechanism is summarized in three operations, all contributing
to the construction of the message used to update the node latent rep­
resentation: alignment (shown in Eq. (3)), weighting (shown in Eq. (4))



and context (shown in Eq. (5)). The purpose of the alignment is to
produce a representation that combines the hidden representation (∈ _v.w_ )
of the central atom and the neighboring atoms through an adjustable
weight matrix ( _W_ ) and a non-linear activation function (in this case a
leaky ReLU). The weighting process aims to reduce the output of the
alignment process into a weight coefficient ( α v,w ) that sums up to 1 for
all neighboring nodes using the softmax function. The coefficient is used
as a weight for the importance/influence the hidden representation of
the neighboring node should have when updating the hidden repre­
sentation of the central atom as seen in the context step (Eq. (5)). The
message produced through the context operation is then used to update
the latent representation of the central node through a gated recurrent
network (GRU) (as seen in Eq. (6)).
**Node-level embedding operations:**
_Center & Neighbor initialization:_



_h_ [0] _v_ [=] _[ ReLU]_ ( _W_ ⋅ _x_ _[atom]_ _v_



) _for t_ = 0 (1)



_h_ [0] _w_ [=] _[ ReLU]_ ( _W_ ⋅[ _x_ _[atom]_ _w_ _, x_ _v_ _[bond]_ _,w_ ]) _f_ or t = 0 (2)


_Alignment:_



_ε_ _[t]_ _v_ [+] _,w_ [1] [=] _[ leaky]_ _[ReLU]_


_Weighting:_



( _W_ ⋅[ _h_ _[t]_ _v_ _[,][ h]_ _[t]_ _w_ ]) for t + 1 ≤ T (3)



_α_ _[t]_ _v_ [+] _,w_ [1] [=] _[ softmax]_ ( _ε_ _[t]_ _v_ [+] _,w_ [1]


_Context:_



for t + 1 ≤ T (4)
)



5


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Fig. 3.** Schematic of the attentive group-contribution model (AGC).



m _[t]_ v [+][1] = _C_ _v_ _[t]_ [+][1] = _elu_


_Update:_



∑
( _w_ ∈Neighbor( _v_ )



_α_ _v,w_ ⋅ _W_ ⋅ _h_ _[t]_ _w_



)



for t + 1 ≤ T (5)



h _[t]_ v [+][1] = _GRU_ ( _h_ _[t]_ _v_ _[,][ C]_ _v_ _[t]_ [+][1] ) for t + 1 ≤ T (6)

After the final node embedding layer, a super node (also referred to
as a virtual node) is constructed as an alternative representation of the
complete molecule, with all the nodes of the graph converging towards
it. This framework makes it possible to illustrate the importance each
node has with regard to the overall graph (molecule) representation. The
final hidden representation of the nodes is used to construct the hidden
representation of the super node as shown in Eq. (7). The attention
mechanism is applied again however it only considers one central node
which is the super node (alignment in Eq. (8), the weighting in Eq. (9)
and the context in (Eq. (10)). Important to note that the attention here
considers all nodes towards the whole graph (represented as a super
node herein) as such the importance of each atom towards the overall
target property can be investigated through the weight coefficient pro­
duced. This aspect makes the attentive FP unique when using the
attention mechanism for interpretability purposes. For the final time
step ( _l_ + _1_ = _L_ ) the update can be considered as the readout step (see Eq.
(11)). This representation is then used as an input to a multi-layer per­
ceptron to regress and produce the target property.
**Graph level embedding operations:**
_Graph initialization:_



_h_ [0] _s_ [=] ∑ _h_ _[T]_ _v_ [for] _[ l]_ [ =][ 0] (7)

_v in G_



_Alignment:_


_ε_ _[l]_ _s_ [+] _,v_ [1] [=] _[ leaky]_ _[ReLU]_


_Weighting:_



( _W_ ⋅[ _h_ _[l]_ _s_ _[,][ h]_ _[T]_ _v_



]) for l + 1 ≤ L (8)



_α_ _[l]_ _s_ [+] _,v_ [1] [=] _[ softmax]_ ( _ε_ _[l]_ _s_ [+] _,v_ [1]


_Context:_



for l + 1 ≤ L (9)
)



∑
( _v_ ∈Neighbor( _s_ )



_C_ _s_ _[l]_ [+][1] = _elu_



_α_ _s,v_ ⋅ _W_ ⋅ _h_ _[t]_ _v_ [for l][ +][ 1][ ≤] [L]



)



(10)



encoding is performed on all three hierarchical levels as shown in Fig. 1.
As such, it is important to note that three distinct attention information
are obtained from each branch. Therefore, the interpretability can be
considered “partial” depending on the branch in focus. In order to
establish “complete” interpretability within the attention mechanism
framework, the attentive group-contribution (AGC) model is developed.


_2.3. AGC: attentive group-contribution model_


The AGC is a compact version of the GroupGAT and aims to provide
“total” interpretability within the framework of the attention mecha­
nism (Aouichaoui et al., 2023). The schematic of the AGC can be seen in
Fig. 3. In brief, the top branch of the GroupGAT is discarded, while the
latent representation produced from the middle branch (in the context of
GroupGAT) is not used for the final representation of the graph but only
to featurize the subgraphs in the junction-tree model. As such, the at­
tentions obtained are for all fragments w.r.t. the overall graph. Simi­
larly, the encoder block used is the attentive FP model.
The GroupGAT and AGC offer several advantages compared to
classical GC models that are based on multi-linear regression. The
models are only based on the first-order groups, which are rooted in the
theoretical understanding of QSPRs. This is not always the case for the
second and third-order groups, which have been defined due to conve­
nience to improve the model predictions (Hukkerikar et al., 2013).
Through the message-passing scheme, the GroupGAT and AGC account
for possible proximity effects since the group representation is a function
of the neighboring groups. The multi-layer perceptron allows for
correlating highly non-linear behavior, while the classical GC are linear
additivity models. Compared to other GNN models, the GroupGAT and
AGC also provide more informative interpretability since the attention is
performed based on functional groups rather than atoms. This could
potentially prove useful in gaining insights into less understood
properties.


**3. Methods**


The methods and steps used in this work are summarized in Fig. 4.
The framework can overall be divided into the data processing step,
model development, and model evaluation, each of which will be
elaborated on in this section.


_3.1. Data and preprocessing_


The data needed for developing the model comprises of two ele­
ments, a molecular identifier in the form of SMILES (simplified molec­
ular input line entry system) and a numerical target value. In this work,
we investigate a total of 30 properties classified into three categories:
thermophysical properties (18 properties), flammability/safety-related



_Update:_


_h_ _[l]_ _s_ [=] _[ GRU]_ ( _h_ _[l]_ _s_ _[,][ C]_ _s_ _[l]_ [+][1]



) for _l_ + 1 ≤ L (11)



A schematic of the attentive FP model can be seen in Fig. 2 as well as
an inside view of the flow of information for the node embedding layers
and the graph embedding layers.
The interpretability consists of showing the weight each fragment (or
node) has when producing the final molecular representation. The



6


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Fig. 4.** Framework for developing AI-based property models.



properties (5 properties), and environmental properties (7 properties).
The data collected mainly come from the Design Institute for Physical
Properties under the American Institute of Chemical Engineers (AIChE
DIPPR)(Rowley et al., 2019), the US environmental protection agency
(EPA)(Mansouri et al., 2016) and other published experimental data
(Alshehri et al., 2022; Nielsen et al., 2001). An overview of the property
investigated, their definition, and the source from which they were
obtained can be seen in Table 2. In this study, we opted for keeping the
source of data homogenous in order to establish better benchmarks for
future studies and as such, the dataset size could increase if we opted for
more mixing of the source. These properties selected for analysis are of
fundamental importance in many simulation studies focusing on prod­
uct and process design studies such as the critical points are used as
inputs to cubic equations of state, which is commonly used in many



commercial process simulators (Frutiger et al., 2017; Mondejar et al.,
2019); the enthalpy of vaporization is used in energy balance calcula­
tions in process design (Cignitti et al., 2019), the flammability property
is used in process safety studies (Frutiger et al., 2016a), the lethal dos­
ages (LD50) and concentrations (LC50) are used to evaluate the human
toxicity of the compounds (Enekvist et al., 2022; Karunanithi et al.,
2006), to name a few examples.
The integrity of the compounds in terms of valency is inspected using
the RDKit toolbox (Landrum, 2021). Furthermore, in this work, only
organic compounds featuring either of the following atoms are consid­
ered: carbon (C), nitrogen (N), oxygen (O), sulfur (S), and halogens
(fluorine (F), chlorine (Cl), bromine (Br), iodine (I)) and phosphorous
(P). As such, the elemental composition of the compounds is included as
a constraint any compound that does not fulfill these requirements is



7


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Table 2**
Overview, definition, and source of the properties considered in this work.

Nr. Property Definition Source



**Thermophysical properties**
1 Normal boiling point (T b ) The temperature at which a compound boils at 1 atm Rowley et al.,

2019

2 Normal melting point (T m ) The temperature at which the melting of a compound occurs at 1 atm Rowley et al.,

2019

3 Octanol-water partition The ratio of a chemical’s concentration in the octanol phase to its concentration in the aqueous phase of a two-phase Mansouri et al.,
coefficient (log K ow) octanol/water system 2016



3 Octanol-water partition The ratio of a chemical’s concentration in the octanol phase to its concentration in the aqueous phase of a two-phase Mansouri et al.,
coefficient (log K ow) octanol/water system 2016

4 Hansen solubility parameter - The portion of the cohesive energy of a solvent attributed to dispersion Nielsen et al.,
Dispersion (δ D ) 2001
5 Hansen solubility parameter - The portion of the cohesive energy of a solvent attributed to polarity Nielsen et al.,
Polar (δ P ) 2001
6 Hansen solubility parameter - H 2 - The portion of the cohesive energy of a solvent attributed to hydrogen bonding Nielsen et al.,
bond (δ H ) 2001
7 Hildebrand solubility parameter A parameter that describes the total cohesive interactions that describe solubility based on the notion of “like Rowley et al.,
(δ) dissolves like”. 2019



The ratio of a chemical’s concentration in the octanol phase to its concentration in the aqueous phase of a two-phase
octanol/water system



7 Hildebrand solubility parameter A parameter that describes the total cohesive interactions that describe solubility based on the notion of “like Rowley et al.,
(δ) dissolves like”. 2019

8 Aqueous solubility (LogWs) The amount of a chemical that will dissolve in liquid water and form a homogenous solution at a reference Mansouri et al.,
temperature 2016



8 Aqueous solubility (LogWs) The amount of a chemical that will dissolve in liquid water and form a homogenous solution at a reference Mansouri et al.,
temperature 2016

9 Acentric factor ( ω ) Conceptual number to describe the non-sphericity of molecules Rowley et al.,

2019

10 Critical temperature (T c ) The temperature above which a gas cannot undergo liquefaction Rowley et al.,

2019

11 Critical pressure (P c ) The minimum pressure for liquefaction of gas at the critical temperature Rowley et al.,

2019

12 Critical volume (V c ) The Molar volume of a compound at the critical temperature and pressure Rowley et al.,

2019

13 Enthalpy of vaporization (H vap ) The molar change in enthalpy associated with the isothermal transition from the liquid state to the vapor state of a Rowley et al.,
compound at 298.15K 2019



13 Enthalpy of vaporization (H vap ) The molar change in enthalpy associated with the isothermal transition from the liquid state to the vapor state of a Rowley et al.,
compound at 298.15K 2019

14 Enthalpy of fusion (H fus ) The molar change in enthalpy associated with the isothermal transition from the solid state to the liquid state of a Rowley et al.,
compound at its melting point 2019



14 Enthalpy of fusion (H fus ) The molar change in enthalpy associated with the isothermal transition from the solid state to the liquid state of a Rowley et al.,
compound at its melting point 2019

15 Enthalpy of formation (H for ) The enthalpy change during the formation of the given substance in the ideal gas state at 298.15 K from elements in Rowley et al.,

their standard states. 2019



15 Enthalpy of formation (H for ) The enthalpy change during the formation of the given substance in the ideal gas state at 298.15 K from elements in Rowley et al.,

their standard states. 2019


16 Liquid molar volume (V m ) The molar volume of a liquid at a reference temperature and pressure Rowley et al.,

2019

17 Ideal gas absolute entropy (S) Absolute entropy of the ideal gas at 298.15 K and 1 bar. Rowley et al.,

2019

18 Refractive Index (RI) The ratio of the speed of light in a vacuum to the speed of light in the substance. The incident light is the sodium D Rowley et al.,
line (0.5896 microns). 2019



18 Refractive Index (RI) The ratio of the speed of light in a vacuum to the speed of light in the substance. The incident light is the sodium D Rowley et al.,
line (0.5896 microns). 2019

**Flammability and safety-related properties**
19 Enthalpy of combustion (H com ) The increase in enthalpy associated with the oxidation of a compound at 298.15 K and 1 atm to the products of the Rowley et al.,
combustion process. 2019



19 Enthalpy of combustion (H com ) The increase in enthalpy associated with the oxidation of a compound at 298.15 K and 1 atm to the products of the Rowley et al.,
combustion process. 2019

20 Auto-ignition temperature (AiT) The minimum temperature value for a compound to commence self-combustion in the air in the absence of an Rowley et al.,
ignition source. 2019



20 Auto-ignition temperature (AiT) The minimum temperature value for a compound to commence self-combustion in the air in the absence of an Rowley et al.,
ignition source. 2019

21 Lower flammability limit (LFL) The minimum concentration in the air will support the propagation of flames Rowley et al.,
2019
22 Upper flammability limit (UFL) The maximum concentration in air that will support the propagation of flames Rowley et al.,
2019

23 Flashpoint (FP) The lower temperature value, at which application of an ignition source causes the vapors of a compound in the air Rowley et al.,
to ignite under specified conditions of the test (corrected to a pressure of 1 atm). 2019



23 Flashpoint (FP) The lower temperature value, at which application of an ignition source causes the vapors of a compound in the air Rowley et al.,
to ignite under specified conditions of the test (corrected to a pressure of 1 atm). 2019

**Environmental related properties**
24 Bioconcentration factor (BCF) The ratio of the chemical concentration in biota as a result of absorption via the respiratory surface to that in water at Mansouri et al.,
a steady state 2016



24 Bioconcentration factor (BCF) The ratio of the chemical concentration in biota as a result of absorption via the respiratory surface to that in water at Mansouri et al.,
a steady state 2016

25 Photochemical oxidation The result of reactions that take place between nitrogen oxides and volatile organic components exposed to UV Mansouri et al.,
potential (PCO) radiation. It is expressed using a reference substance such as ethylene 2016



25 Photochemical oxidation The result of reactions that take place between nitrogen oxides and volatile organic components exposed to UV Mansouri et al.,
potential (PCO) radiation. It is expressed using a reference substance such as ethylene 2016

26 Acid dissociation constant (pka) A measure of the extent to which an acid dissociates in solution Alshehri et al.,

2022

27 Lethal dosage (LD 50 ) Amount of chemical (mass of chemical per body weight of the rat) that kills half of the rats through oral digestion Alshehri et al.,

2022

28 Lethal concentration (LC 50 ) Amount of chemical (in terms of liquid concentration in water) that kills half of the fathead Minnow in 96 hours. Alshehri et al.,

2022

29 Permissible exposure limit A legal limit in the United States for exposure of an employee to a chemical substance or physical agent. Alshehri et al.,
(OSHA-TWA) 2022
30 Biodegradability (BioD) Quantification of the biodegradability of chemical compounds is described as the ratio of the biochemical oxygen Jhamb et al.,
demand (BOD) and chemical oxygen demand (COD) 2020



Jhamb et al.,

2020



discarded. The remaining data undergoes a set of statistical analyses in
order to characterize the dataset. Summary statistics of the data can be
seen in the supplementary materials. In this work, the data split is done
randomly as we believe this allows fair benchmarking of model per­
formance across diverse data sets/property space. While this might
result in compound features being absent in one of the splits or struc­
turally similar compounds being present in all splits, this can be
addressed by repeating the random data splitting several times
(Hwangbo et al., 2021). On the other hand, this random data splitting



does not require the definition of heuristics and rules to determine the
species in each split, which might ultimately also introduce a certain
bias. In the end, each data splitting method may have its use depending
on the end application of the model itself: random splitting (with suf­
ficient repetition) allows obtaining a range of model performance
achievable, while heuristics/knowledge-based data splitting allows the
development of a more customized model with high accuracy for certain
application range of molecules. Further analysis of this aspect is how­
ever beyond the scope of this contribution. The data splitting is done at



8


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Fig. 5.** Property data distributions for a selected subset of properties from each class: the normal boiling point (T b ), the Hildebrand total solubility parameter (δ), the
flash point temperature (FP), and the photochemical oxidation (PCO).



random with 80% of data allocated for training and the remaining
evenly split (10%) between validation and testing. Various splits were
tested and the split resulting in the lowest validation loss was chosen.
During this process, the performance metric on the test set was not
calculated and not considered for determining the split. The target
values are further scaled using a z-score based using the mean and the
variance of the data allocated to the training to prevent data leakage.
The domain of applicability of any QSPR model can be characterized
through a wide range of techniques, most of which, are based on sta­
tistics and heuristics such as the William plots (Netzeva et al., 2005),
principal component analysis (PCA), t-distributed stochastic neighbor
embedding (t-SNE) and uniform manifold approximation and projection
(UMAP). In this work, we adopt a more chemically intuitive definition of
the domain of applicability where we conduct a chemical diversity



analysis by classifying compounds into hydrocarbons, oxygenated,
nitrogenated, chlorinated, fluorinated, brominated, iodinated, sulfo­
nated, phosphorous-containing or silicon-containing. In case a com­
pound can pertain to more than one group, it will be considered
multifunctional. The distribution of a selected subset of the property
dataset can be seen in Fig. 5. The remaining can be seen in the supple­
mentary material and an overview of the classes present in the various
dataset can be seen in Fig. 6.


_3.2. Model development and systematic training_


The models selected are the AGC and GroupGAT models described
earlier and were implemented using the Deep Graph Library (DGL) using
the Pytorch deep learning framework (Zheng et al., 2021). For the



9


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Fig. 6.** Domain of applicability of the models based on chemical diversity analysis.



**Table 3**

Hyperparameter design space for the AGC and GroupGAT.


Parameter Range


**Global**
Initial learning rate [10 [−] [5], 10 [−] [1] ]
Learning rate reduction factor [0.4, 0.99]
Weight decay [10 [−] [6], 10 [−] [1] ]

Patience 30

Max number of epochs 300
**GroupGAT**
Hidden dimension B1, B2, B3: integer([32, 256])
Node embedding layers (T) B1, B3: integer([1, 4]) / B2: integer([1,3])
Graph embedding layers (L) B1, B3: inetger([1, 4]) / B2: inetger([1,2])
Representation dropout B1, B2, B3, final: [0, 0.4]
MLP layers Integer([1,4])

**AGC**

Hidden dimension integer([32, 256])
Node embedding layers (T) B1: integer([1, 2]) / B2: integer([1, 4])
Graph embedding layers (L) B1: integer([1, 2]) / B2: integer([1, 4])
Representation dropout [0, 0.25]
MLP dropout [0, 0.4]
MLP layers Integer([1,4])


The ReLU activation function is used for the MLP for all properties and models.
The number of hidden dimensions of a layer in the MLP corresponds to half of
the previous layer.
B1, B2, and B3 correspond to the various hierarchal branches in the models.
The hyperparameters are not shared between each branch and thus the opti­
mization is done for each branch individually.


training, no batching was used as the size of the dataset did not provide
any issues in estimating the true gradient (batch size was set to be equal
to the training size). The model training was done by the ADAM opti­
mizer over a maximum of 500 epochs with the mean absolute error as
the objective function (Kingma and Ba, 2015). In order to avoid over­
fitting, a series of regularization measures have been implemented such
as weight decay (L2-regularization), early stopping, and dropout. The



hyperparameters of the models are determined using Bayesian optimi­
zation as implemented in the pyGPGO (Jim´enez and Ginebra, 2017)
toolbox. The hyperparameter design space consists in these cases of the
hyperparameters of the encoder (in this case the attentive FP model) as
Table 3. A main
well as some training-specific parameters as shown in
difference in the training procedure presented here is that the hyper­
parameters of each encoder block are independent (can vary between
branches). This allows for a more comprehensive exploration of the
hyperparameter space compared to a previous study (Aouichaoui et al.,
2023).
The variance in the model prediction which can be regarded as the
uncertainty is quantified using an ensemble of 40 independently trained
models with different weight initialization. The choice of ensemble size
was determined through a maximum-to-sum plot. Maximum to sum plot
is essentially an empirical analysis of the convergence of the moments of
a distribution (Cirillo and Taleb, 2020). In this case, we focused on the
distribution of mean absolute error (MAE) of ensemble models as a
function of N being the number of ensemble models in the sample space.
This method has previously been employed in a similar study (Aoui­
chaoui et al., 2023). At convergence, the ratio ( _R_ _[p]_ _n_ [) of the cumulative ]
max ( _M_ _[p]_ _n_ [) over the cumulative sum (] _[S]_ _[p]_ _n_ [) of a performance metric (e.g. ]
MAE) with a _p_ order (up to 4) moment tends towards zero for _n_ evalu­
ations as shown in Eq.(12).



Important to note is that the data split also affects the variance in the
model performance, which can be more dominant for properties with
limited size. This is however not considered in the main part of this
work, where the main focus was to study the feasibility of GNN models
for a wide range of properties. For future works, indeed this data split­
ting method can be addressed as part of the hyperparameter training to
reduce any potential variance in the model performance -as shown from
initial tests presented in the supplementary material this could be a



_R_ _[P]_ _n_ [=] _[ M]_ _S_ _n_ _[P]_ _n_ _[P]_



→0 as _n_ →∞ (12)



10


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_



**Table 4**

Logical and thermodynamic consistency checks.


Nr. Properties Condition Reference


1 T c, T b T c _>_ T b logical
2 3 Tδ Db, T, δ Pm, δ H, δ _δ_ T = b _>_ √ T̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅ _δ_ m [2] D [+] _[ δ]_ [2] P [+] _[ δ]_ [2] H **̅** Undavalli et al., 2021 logical


4 AIT, FP AIT _>_ FP logical
5 UFL, LFL UFL _>_ LFL logical


relevant factor for some properties. An early version of the code is made
available at: https://github.com/gsi-lab/GC-GNN.


_3.3. Model evaluation_


The model performance across the three data folds (training, vali­
dation, and testing) is characterized by calculating a set of performance
metrics consisting of the root mean squared error (RMSE), mean abso­
lute error (MAE), the correlation coefficient (R [2] ), mean absolute per­
centage error (MAPE) and median absolute percentage error (MDAPE).
The MDAPE is especially relevant for properties that exhibit a value in
the vicinity of 0. Consider a true value of 0.1 and a potential prediction
of 0.5. the MAPE would be 500% while the median will consider this as

an outlier (thanks to the use of rank statistics) while still preserving the
general shape of the distribution of errors. While the MDAPE can be
considered the unitless performance metric which allows comparing
model performance across diverse properties with different units, the
MAE metric is the unit-rooted error metric and useful for specific engi­
neering applications. These metrics are evaluated for each member of
the ensemble and the mean performance value and the standard devi­
ation in terms of percentage of the mean are reported. The errors are
inspected using residual distribution plots. Some of the studied prop­
erties are interrelated in terms of thermodynamic and logical under­
standing. Furthermore, there are a set of established trends for some
properties with increasing carbon number. It is important to investigate


**Table 5**

Performance overview of the GroupGAT on the test set and the complete dataset.



these in order to establish the sanity of the models developed. Table 4
provides an overview of the checks conducted.
In addition to the various sanity checks conducted, the developed
models are capable of providing insights into the contribution (in terms
of attention weights) of groups to the final hidden representation of the
molecule. The interpretability obtained for GroupGAT and AGC are
instance based and consists of inspecting and visualizing the attention
weights in the graph embedding stage where the molecule is considered
a virtual node with all the groups (subgraphs) linked to it. As such, the
attention visualization is conducted at the encoder of the third branch

for the GroupGAT and the second encoder for AGC. The attention
weights are thus visualized through a color code, where the intensity of
the color corresponds to a higher attention weight.


**4. Results**


_4.1. Model performance_


The model performance is characterized through a set of metrics and
is evaluated by the model’s ability to describe the complete available
design space (using all data) and its ability to generalize to unseen
compounds (using the test set). The shown metrics here are the R [2], MAE,
and the MDAPE. The obtained results for GroupGAT can be seen in
Table 5, while those for AGC can be seen in the supplementary material.
The uncertainty of the model performance is indicated in terms of the
standard deviations reported as a percentage of the reported metrics
mean. A selection parity plot can be seen in Fig. 7. The remainder of the
parity plots as well as the error distribution plots can be seen in the
supplementary material.


_4.1.1. Thermophysical properties_
The GroupGAT model successfully models almost all accounts of the
18 thermophysical properties considered in this study, with MDAPE
below 10% for 15 of these and the remaining below 15% considering the
complete dataset. Considering the test set exclusively, the MDAPE is



Property No. data R [2 ] MAE MDAPE %


Test Total Test Total Test Total


**T** **b** 1,389 0.99 ±0.12% 0.99 ±0.12% 6.00 ±7.97% 3.68 ±17.40% 0.97 ±14.81% 0.61 ±23.40%

**T** **m** 1,452 0.86 ±1.64% 0.92 ±1.20% 24.29 ±4.96% 18.63 ±7.67% 7.24 ±7.81% 5.62 ±8.71%
**log K** **ow** 13,845 0.94 ±0.35% 0.97 ±0.55% 0.29 ±3.41% 0.20 ±11.43% 10.94 ±5.08% 7.65 ±12.35%
**δ** **D** 1,028 0.88 ±0.85% 0.92 ±0.73% 0.46 ±2.92% 0.37 ±4.39% 1.95 ±7.33% 1.58 ±5.44%

**δ** **P** 1,008 0.70 ±2.22% 0.82 ±1.87% 1.53 ±2.40% 1.29 ±4.37% 17.77 ±5.53% 14.71 ±4.66%

**δ** **H** 1,008 0.87 ±1.24% 0.90 ±0.73% 1.22 ±3.91% 1.07 ±3.63% 12.95 ±7.72% 10.91 ±4.84%

**δ** 1,912 0.89 ±1.67% 0.94 ±1.63% 0.77 ±6.28% 0.64 ±14.57% 2.54 ±12.45% 2.28 ±15.34%

**LogWs** 2,010 0.95 ±0.20% 0.96 ±0.48% 0.42 ±2.42% 0.31 ±6.88% 15.67 ±7.45% 9.96 ±8.15%
**ω** 1,914 0.95 ±0.55% 0.98 ±0.33% 0.05 ±4.77% 0.03 ±11.72% 6.11 ±10.41% 3.64 ±17.27%

**T** **c** 507 0.99 ±0.27% 0.99 ±0.09% 8.70 ±8.74% 4.73 ±25.51% 0.91 ±16.86% 0.55 ±37.78%

**P** **c** 386 0.99 ±0.26% 0.99 ±0.27% 0.13 ±11.81% 0.11 ±13.34% 3.19 ±16.97% 2.52 ±16.16%

**V** **c** 253 0.99 ±0.22% 0.99 ±0.05% 0.01 ±12.33% 0.01 ±9.57% 2.39 ±19.71% 1.31 ±12.94%
**H** **vap** 425 0.99 ±0.34% 0.98 ±0.39% 1.12 ±13.05% 1.02 ±15.75% 2.03 ±20.56% 1.68 ±18.21%
**H** **fus** 740 0.97 ±0.56% 0.95 ±1.19% 3.55 ±6.22% 2.48 ±16.07% 19.17 ±10.91% 13.12 ±17.95%

**H** **f** 726 0.98 ±0.28% 0.99 ±0.06% 19.27 ±7.91% 11.36 ±15.89% 5.70 ±20.72% 4.07 ±22.31%

**V** **m** 1,254 0.99 ±0.02% 0.99 ±0.04% 0.00 ±12.14% 0.00 ±13.99% 1.11 ±18.52% 1.14 ±18.06%

**S** 511 0.99 ±0.09% 0.95 ±0.19% 0.06 ±9.11% 0.04 ±11.60% 1.06 ±16.38% 0.90 ±14.38%

**RI** 1,544 0.89 ±5.59% 0.96 ±0.90% 0.01 ±8.95% 0.01 ±16.77% 0.39 ±7.76% 0.26 ±18.36%

**H** **comb** 853 0.99 ±0.02% 0.99 ±0.02% 45.76 ±13.06% 38.52 ±17.18% 0.83 ±22.82% 0.74 ±24.61%
**AiT** 515 0.76 ±5.15% 0.83 ±6.63% 41.36 ±9.41% 29.70 ±27.53% 4.13 ±16.91% 3.10 ±33.61%

**LFL** 432 0.94 ±3.31% 0.98 ±0.58% 0.27 ±15.45% 0.17 ±13.16% 8.16 ±19.12% 6.36 ±17.77%

**UFL** 367 0.64 ±8.86% 0.88 ±3.03% 2.02 ±14.03% 1.66 ±33.68% 12.54 ±27.26% 9.42 ±48.35%

**FP** 888 0.97 ±0.48% 0.99 ±0.30% 6.83 ±7.84% 3.75 ±18.72% 1.45 ±11.61% 0.81 ±21.91%

**BCF** 608 0.86 ±1.54% 0.93 ±0.75% 0.36 ±4.33% 0.25 ±8.25% 19.12 ±10.03% 11.23 ±10.34%

**PCO** 608 0.78 ±5.98% 0.95 ±1.12% 0.11 ±10.13% 0.06 ±28.17% 15.55 ±23.35% 10.74 ±33.91%
**pka** 1,631 0.82 ±2.14% 0.92 ±1.61% 1.01 ±4.29% 0.68 ±10.22% 11.42 ±9.74% 8.24 ±12.65%

**LD** **50** 4,781 0.62 ±1.92% 0.77 ±3.00% 0.35 ±1.75% 0.27 ±5.61% 12.40 ±3.66% 9.55 ±6.67%

**LC** **50** 705 0.78 ±2.04% 0.86 ±2.17% 0.55 ± 4.96% 0.44 ±9.54% 10.89 ±11.27% 8.55 ±11.43%

**PEI** 418 0.84 ±6.47% 0.84 ±9.12% 0.44 ±19.72% 0.44 ±25.95% 11.22 ±32.07% 10.05 ±29.74%
**BioD** 232 0.59 ±28.31% 0.74 ±19.87% 0.15 ±22.64% 0.12 ±30.25% 54.97 ±49.22% 32.83 ±23.37%


11


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Fig. 7.** Parity plot for a selected subset of properties from each class: the normal boiling point (T b ), the Hildebrand total solubility parameter (δ), the flash point
temperature (FP), and the photochemical oxidation (PCO). Molar weight is used as the color code.



below 10% for 13 out of the 18 accounts with the remaining being below
20%. In general, the performance of the AGC model (available in the
supplementary materials) is slightly worse than the GroupGAT. This
could indicate the benefits derived from the hierarchal representation
learning that the GroupGAT model offers. Notably are the good per­
formance achieved for the critical temperature, the boiling point, ab­
solute entropy, and the refractive index with MDAPE being below 1%.
Solubility-related properties such as the Hansen and Hildebrand pa­
rameters are proving to be challenging especially when it comes to
unseen data. This is not a surprising observation as previous models also
failed to describe such properties (Hukkerikar et al., 2012b). This is due
to the complex thermodynamics and structural information that is
involved e.g. the symmetry present in the molecule, the polarity, and the
electronegativity.
A literature survey was conducted to identify models with a similar
purpose to assess the performance of the models developed for the
thermophysical properties with an emphasis on GC and GNN models
(shown in Table 6). Notably, for 9 properties there are no preexisting
GNN models, while for 4 of the properties GNN models have been
developed as part of our previous work. This highlights that many
properties of interest in the chemical engineering practice are largely
overlooked when developing GNN models. The developed GroupGAT
model showcases the best performance of any GNN model across all 18
thermophysical properties. Especially noteworthy are the great im­
provements achieved for the normal melting point (12 K in terms of MAE



to T m compared to previously published models (Coley et al., 2017;
Sivaraman et al., 2020)). The same goes for any other machine-learning
or deep-learning model developed for any of the properties except for
the combination of groups and Gaussian process regression (GC+GPR)
(Alshehri et al., 2022). It is important to note that the cross-validation
metrics for these models are not presented and no regularization tech­
niques were reported. Considering the extremely small error metrics and
confidence bounds that the models exhibit, it could indicate the pres­
ence of overfitting.
Nonetheless, compared to the GC models based on the Marrero-Gani
groups, the developed models exhibit great improvements when
considering the normal boiling point, solubility-related properties,
critical properties, and the enthalpy of fusion. For the remaining prop­
erties, the models perform similarly except for the enthalpy of forma­
tion. This is because new groups were defined to reduce the error
exhibited by the original based on the same groups (Hukkerikar et al.,
2013, 2012b). The model based on the Constantinou-Gani group defi­
nitions is commonly used in computer-aided molecular design and
in-silico screening of compounds due to their compatibility with the
groups defined in the UNIFAC model for liquid activity coefficients
(Cignitti et al., 2019; Constantinou and Gani, 1994; Hansen et al., 1991).
As can be seen in Table 6, all developed models perform similarly in
many accounts although the models in this work are trained on much
larger data. This indicates that the currently developed models provide
an accuracy level that is well comparable with the currently used



12


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Table 6**

Comparison of the developed property models for thermophysical properties with existing models.


Property Reference Type No. data Metric


**T** **b** Current work GNN 1,389 MAE: 3.7
**(K)** Alshehri et al., 2022 GC+GP 5,276 MAE: 3.9
Aouichaoui et al., 2022b GC+MLR 3,510 MAE: 4.7
Hukkerikar et al., 2012b GC+MLR 3,510 MAE: 6.2
Qu et al., 2022 GNN 3,850 MAE: 6.0
Constantinou and Gani, 1994* GC+MLR 285 MAE: 7.71 [a ]                         - 5.35 [b ]

**T** **m** Current work GNN 1,452 MAE: 18.7
**(K)** Alshehri et al., 2022 GC+GPR 9,249 MAE: 5.3
Aouichaoui et al., 2022b GC+MLR 5,183 MAE: 14.5
Wyttenbach et al., 2020 QSPR 1,800 MAE: 16.70
Aouichaoui et al., 2023 GNN 3,035 MAE: 17.4
Constantinou and Gani, 1994* GC+MLR 312 MAE: 17.40 _[a ]_                         - 14.03 _[b ]_

Hukkerikar et al., 2012b GC+MLR 5,183 MAE: 16.0
Coley et al., 2017 GNN 3,019 MAE: 26.2
Sivaraman et al., 2020 GNN 33,408 MAE: 29.3
**Log K** **ow** Current work GNN 13,845 MAE: 0.20
Alshehri et al., 2022 GC+GPR 12,193 MAE: 0.04
Stefanis et al., 2004)* GC+MLR 422 MAE: 0.23 _[a ]_                             - 0.18 _[b ]_

Zhang et al., 2022 GNN 10,668 MAE: 0.23
US EPA, 2023 QSPR 10,668 MAE: 0.31
Aouichaoui et al., 2022b GC+MLR 12,193 MAE: 0.39
Hukkerikar et al., 2012a GC+MLR 12,193 MAE: 0.48
Tang et al., 2020 GNN 4,200 MAE: 0.41
Coley et al., 2017 GNN 282 MAE: 0.45
Xiong et al., 2020 GNN 4,200 MAE: 0.58
**δ** **D** Current work GNN 1,028 MAE: 0.37
**(MPa)** Alshehri et al., 2022 GC+GPR 1,073 MAE: 0.04
Aouichaoui et al., 2022b GC+MLR 1,037 MAE: 0.37
Stefanis and Panayiotou, 2008* GC+MLR 344 MAE: 0.44 _[a ]_                          - 0.41 _[b ]_

Hukkerikar et al., 2012b GC+MLR 1,037 MAE: 0.60
Sanchez-Lengeling et al., 2019 QSPR+GPR 193 MAE: 0.68
**δ** **P** Current work GNN 1,008 MAE: 1.29
**(MPa)** Alshehri et al., 2022 GC+GPR 1,017 MAE: 0.12
Stefanis and Panayiotou, 2008* GC+MLR 350 MAE: 1.10 [a ]                          - 0.86 [b ]

Aouichaoui et al., 2022b GC+MLR 1,017 MAE: 1.16
Hukkerikar et al., 2012b GC+MLR 1,017 MAE: 1.81
Sanchez-Lengeling et al., 2019 QSPR+GPR 193 MAE: 1.93
**δ** **H** Current work GNN 1,008 MAE: 1.07
**(MPa)** Alshehri et al., 2022 GC+GPR 1,051 MAE: 0.07
Aouichaoui et al., 2022b GC+MLR 1,016 MAE: 0.83
Stefanis and Panayiotou, 2008* GC+MLR 375 MAE: 0.88 [a ]                          - 0.80 [b ]

Hukkerikar et al., 2012b GC+MLR 1,016 MAE: 1.28
Sanchez-Lengeling et al., 2019 QSPR+GPR 193 MAE: 1.57
**δ** Current work GNN 1,912 MAE: 0.64
**(MPa)** Alshehri et al., 2022 GC+GPR 1,384 MAE: 0.11
Aouichaoui et al., 2022b GC+MLR 1,384 MAE: 0.72
Stefanis et al., 2004* GC+MLR 1017 MAE:0.99 [a ]                             - 0.90 [b ]

Hukkerikar et al., 2012b GC+MLR 1,384 MAE: 1.08
**Log W** **s** Current work GNN 2,010 MAE: 0.31
Alshehri et al., 2022 GC+GPR 2565 MAE: 0.13
Aouichaoui et al., 2023 GNN 1,128 MAE: 0.32
Coley et al., 2017 GNN 1,116 MAE: 0.40
Tang et al., 2020 GNN 1,311 MAE: 0.45
Xiong et al., 2020 GNN 1,128 MAE: 0.51
Hukkerikar et al., 2012a GC+MLR 4,681 MAE: 0.98
**ω** Current work GNN 1,914 MAE: 0.03
Alshehri et al., 2022 GC+GPR 1,723 MAE: 0.01
Aouichaoui et al., 2022b GC+MLR 1,723 MAE: 0.03
Hukkerikar et al., 2012b GC+MLR 1,723 MAE: 0.05
Constantinou et al., 1995* GC+MLR 181 MAE: 0.16 [a ]                           - 0.01 [b ]

**T** **c** Current work GNN 507 MAE: 4.73
**(K)** Alshehri et al., 2022 GC+GPR 776 MAE: 2.30
Aouichaoui et al., 2022b GC+MLR 858 MAE: 4.57
Hukkerikar et al., 2012b GC+MLR 858 MAE: 7.72
Constantinou and Gani, 1994)* GC+MLR 285 MAE: 9.12 [a ]                         - 4.85 [b ]

Aouichaoui et al., 2022b GNN 491 MAE: 17.00
Su et al., 2019 DNN 1,792 MAE: 23.00
**P** **c** Current work GNN 386 MAE: 0.11
**(MPa)** Alshehri et al., 2022 GC+GPR 774 MAE: 0.04
Aouichaoui et al., 2022b GC+MLR 852 MAE: 0.08
Aouichaoui et al., 2022a GNN 371 MAE: 0.13
Constantinou and Gani, 1994* GC+MLR 269 MAE: 0.14 [a ]                         - 0.11 [b ]

Hukkerikar et al., 2012b GC+MLR 852 MAE: 0.14
Su et al., 2019 DNN 1,726 MAE: 0.18


( _continued on next page_ )


13


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Table 6** ( _continued_ )


Property Reference Type No. data Metric


**V** **c** Current work GNN 253 MAE: 0.001
**(m** **[3 ]** **kmol** **[¡][1]** **)** Alshehri et al., 2022 GC+GPR 773 MAE: 0.004
Aouichaoui et al., 2022a GC+MLR 797 MAE: 0.006
Hukkerikar et al., 2012b GC+MLR 797 MAE: 0.007
Su et al., 2019 DNN 1801 MAE: 0.007
Aouichaoui et al., 2022b GNN 250 MAE: 0.008
Constantinou and Gani, 1994* GC+MLR 251 MAE: 0.008 [a ]                         - 0.006 [b ]

_**H**_ [298] _**vap**_ Current work GNN 425 MAE: 1.02

**(kJ.mol** **[¡][1]** **)** Alshehri et al., 2022 GC+GPR 425 MAE: 0.45
Aouichaoui et al., 2022a GC+MLR 705 MAE: 0.71
Hukkerikar et al., 2012b GC+MLR 705 MAE: 1.29
Constantinou and Gani, 1994* GC+MLR 225 MAE: 1.40 [a ]                         - 1.11 [b ]

**H** **fus** Current work GNN 740 MAE: 2.48
**(kJ mol** **[¡][1]** **)** Alshehri et al., 2022 GC+GPR 749 MAE: 0.36
Aouichaoui et al., 2022a GC+MLR 764 MAE: 1.72
Aouichaoui et al., 2023 GNN 730 MAE: 2.61
Hukkerikar et al., 2012b GC+MLR 764 MAE: 5.03
Wyttenbach et al., 2020 QSPR 2332 MAE: 5.20
**H** **f** Current work GNN 726 MAE: 11.36
**(kJ mol** **[¡][1]** **)** Hukkerikar et al., 2013 GC+MLR 861 MAE: 1.75
Aouichaoui et al., 2022a GC+MLR 882 MAE: 2.92
Meier, 2022, 2021a, 2021b GC+MLR 458 MAE: 1.43
Constantinou and Gani, 1994* GC+MLR 373 MAE: 5.45 [a ]                         - 3.71 [b ]

Zheng et al., 2022 DNN 534 MAE: _<_ 5.70
Alshehri et al., 2022 GC+GPR 1,059 MAE: 7.60
Aouichaoui et al., 2023 GNN 741 MAE: 15.50
Trinh et al., 2022 QSPR+SVR 1276 MAE: 16.93
**V** **m** Current work GNN 1,254 MAE: 0.001
**(m** **[3 ]** **kmol** **[¡][1]** **)** Alshehri et al., 2022 GC+GPR 1,059 MAE: 0.001
Aouichaoui et al., 2022a GC+MLR 1,056 MAE: 0.001
Constantinou et al., 1995* GC+MLR 312 MAE: 0.001 [a ]                           - 0.001 [b ]

Hukkerikar et al., 2012b GC+MLR 1,056 MAE: 0.003

**S** Current work GNN 511 MAE: 0.011
(kJ mol [−] [1 ] K [−] [1] ) Trinh et al., 2022 QSPR+SVR 1276 MAE: 0.023

Benson, 1999 GC+MLR 14 MAE: 0.002
**RI** Current work GNN 1,544 MAPE: 0.26
Cai et al., 2017 GC+MLR 106 MAPE: 0.73
Naef and Acree, 2022 GC+MLR 5988 MAPE: 0.76
Gharagheizi et al., 2014 GC+MLR 11918 MAPE: 0.83


GNN: Graph neural Networks, GC: group-contribution, MLR: multi-linear regression, GPR: Gaussian process regression,
SVR: support vector regression, DNN: Deep neural networks, QSPR: Quantitative structure-property relation, *Con­
stantinou-Gani groups based

a first-order groups only
b first and second-order groups



CAMD-related property models if not better. Furthermore, the GC-based
models are trained on the entirety of the data available with no
cross-validation performed which limits the models’ ability to extrapo­
late and is dependent on the presence of the defined group in the data to
assign a contribution to it.


_4.1.2. Flammability and safety-related properties_
Among the flammability and safety-related properties, the Group­
GAT model attains good performance considering the enthalpy of
combustion, the flash point as well as the lower flammability limit. The
results obtained for the enthalpy of combustion are comparable to those
obtained using higher-order GC models (Frutiger et al., 2016b). How­
ever, important to note, is that in the current study no outlier deletion
was conducted, and all data were retained for model development.
Considering the MDAPE, the obtained models for the UFL, AIT, as well as
FP, exceed those obtained using GC methods with outlier deletion
(Frutiger et al., 2016a). However, it does remain a challenge to suc­
cessfully model both the AIT and the UFL, especially since the former
exhibits different limiting values for the homologous series which ex­
plains the complicated expressions developed for that purpose in GC
methods (Frutiger et al., 2016a; Hukkerikar et al., 2012b). Another
factor one needs to consider when modeling the AIT is the fact that
measurement of this property experimentally is challenging and not
trivial as the AIT is very close to the thermal decomposition temperature



of many compounds. Despite having more data, the GroupGAT out­
performs the usual FP model used for CAMD screening (Frutiger et al.,
2016a; Stefanis et al., 2004) and as such offers a better alternative for
doing so. Similarly, the observation made in the previous analysis con­
cerning the GC model combined with GPR is also valid for the
safety-related properties (Alshehri et al., 2022). Considering the other AI
or ML-based methods, the currently developed models based on
GroupGAT vastly outperform these (see Table 7).


_4.1.3. Environmental related properties_
Despite, significant improvements compared to GC-based models,
the environmental-related properties remain a challenge for QSPR
modeling (see Table 8). Compared to GC models, improvements have
been achieved on all accounts, where in many cases the MAE has been
halved (Hukkerikar et al., 2012a). It is however also important to
highlight the fact that the model does struggle with some properties,
especially the biodegradability dataset for which the MDAPE is 32%. In
general, the biodegradability data is a complex property to model as
several factors affect this measurement. Namely, the data is collected by
performing a biochemical oxygen demand (BOD) test that relies on the
microorganism’s ability used in the test to degrade and decompose the
said chemical compound under aerobic conditions monitored over
several days (Metcalf & Eddy et al., 2014). Therefore, it is not surprising
that high modeling accuracy is challenging to achieve due to potential



14


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Table 7**

Comparison of the developed property models for Flammability and safety-related properties with existing models.


Property Reference Type No. data Metric


**H** **com** Current work GNN 853 MAE: 38.52
(kJ mol [−] [1] ) Frutiger et al., 2016b GC+MLR 794 MAE: 13.09
Aouichaoui et al., 2023 GNN 847 MAE: 32.80
Park et al., 2021 QSPR+MLR+NLI 1,850 MAE: 60.60
Cao et al., 2009 GC+ANN 1,496 MAE: 155.40

**AIT** Current work GNN 515 MAE: 29.70

(K) Alshehri et al., 2022 GC+GPR 571 MAE: 3.75
Frutiger et al., 2016a GC+MLR 513 MAE: 12.33
Hukkerikar et al., 2012b GC+MLR 570 MAE: 13.51
Yang et al., 2021 DNN 480 MAE: 35.57

**LFL** Current work GNN 432 MAE: 0.17

**(vol-%)** Aouichaoui et al., 2021 GC+GPR 443 MAE: 0.12
Aouichaoui et al., 2021 GC+DNN 443 MAE: 0.20
Park et al., 2021 QSPR+MLR+NLI 1,733 MAE: 0.21
Frutiger et al., 2016a GC+MLR 443 MAE: 0.24
Yang et al., 2021 DNN 443 MAE: 0.31
Aouichaoui et al., 2021 GC+MLR 443 MAE: 0.39
Gharagheizi, 2008 QSPR+MLR 1056 MAE: 7.68

**UFL** Current work GNN 367 MAE: 1.66

**(vol-%)** Yuan et al., 2019 QSPR + RF 79 MAE: 0.06
Frutiger et al., 2016a GC+MLR 351 MAE: 0.99
Yang et al., 2021 DNN 329 MAE: 1.60
Park et al., 2021 QSPR +MLR+NLI 1,711 MAE: 2.44

**FP** Current work GNN 888 MAE: 3.75

(K) Alshehri et al., 2022 GC+GPR 512 MAE: 1.10
Frutiger et al., 2016a GC+MLR 927 MAE: 6.77
Park et al., 2021 QSPR +MLR+NLI 1,741 MAE: 7.31
Yang et al., 2021 DNN 1176 MAE: 9.40
Wen et al., 2022 GNN 1651 MAE: 9.60
Stefanis et al., 2004* GC+MLR 418 MAE: 11.9 [a ]                     - 10.7 [b ]

Sun et al., 2020 GNN 10,575 MAE: 17.8


NLI: non-linear interactions



sources of variability in the measurement technique itself. However, the
model developed for biodegradability in this work compares well
against the GC model developed in the literature (Jhamb et al., 2020).
As a final remark, the developed GNN-based methods present in
many cases the first attempt for modeling a wide range of properties and
their performance exceeds in many cases currently available models
based on either classical approaches such as GC models or AI/ML-based
methods. However, it is important to highlight the fact that almost none
of the models were trained on the same data, and as such, it makes a
direct one-to-one comparison difficult to make. As such, we suggest the
establishment of a standardized set of data for each property to better
benchmark developed models in the literature. Such data is widely
available for various quantum mechanics properties and ADMET (ab­
sorption, distribution, metabolism, excretion, and toxicity) properties
related to drug discovery in the form of e.g. Tox21 and PDBbind (Wu
et al., 2018). Furthermore, it is important to consult the chemical species
present for the various properties as well as the variance produced for a
specific prediction in order to assess the reliability and applicability of
the models for specific types of compounds. Absent features or com­
pound types will ultimately result in high variance when assessing the
uncertainty of the prediction provided by the ensemble (Ryu et al.,
2019).


_4.2. Interpretability_


We showcase one example of how the integration of functional
groups and the attention mechanism provides an aspect of interpret­
ability to the GNN model. Fig. 8 shows the attention visualization of
paracetamol w.r.t to the flashpoint property. The attention weights
attribute each fragment with a weight that sums to unity. The larger the
attributed weight the more important the fragment is. This is visualized
using a red color code where a more intense shade of red corresponds to
a higher attention weight. In the shown case, the “a-C-NHCO” group has
the largest attention weight followed by “aC-OH”, “a-CH” and “-CH3”



respectively. This order of importance is consistent with the contribu­
tion coefficient of the groups published in the literature (Frutiger et al.,
2016a; Hukkerikar et al., 2012b). The same visualization is also ob­
tained for the normal boiling point which also corresponds to the order
of contributions published in the literature. The proposed interpret­
ability assigns different importance for the same groups (e.g. a-CH)
depending on their neighboring groups which underlies the proximity
effects that the method is capable of capturing.
Important to note is that interpretability is local and instance based
and as such there is no guarantee that a given group will retain the
importance from one compound to another or across properties. As such,
the interpretability must be evaluated on a case-by-case approach.
Furthermore, we suspect that the insights are highly dependent on the
goodness of fit of the model. This is because the attention coefficient is
small (between 0 and 1) and as such could be very sensitive to the
overall and specific performance of the model towards a given type of
species in the data. It is also important to take into account the fact of
collinearity, identifiability as well as the uncertainty present in the
group contributions determined in the literature. As such, these could
also provide a misleading conclusion (Frutiger et al., 2016b). While
some counterfactual examples have previously been produced in this
regard to highlight the need to assess the effect of the model accuracy
and weight initialization on the interpretability produced, other tech­
niques can be used to generate new examples where the model inter­
pretability fails (Aouichaoui et al., 2023; Wellawatte et al., 2022).


_4.2.1. Sanity checks_
A series of thermodynamic and logical checks have been identified as
presented in the method section. The aim here is to test the validity of
the models in following the thermodynamic and logical trends expected
for the properties and the compounds. The results are depicted in Fig. 9.
For this, all properties of interest were evaluated on all compounds
(1920 compounds in total) which had a SMILES identifier in the DIPPR
database (Rowley et al., 2019). For the first consistency check, the ratio



15


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_



**Table 8**

Comparison of the developed property models for environmental-related prop­
erties with existing models.


Property Reference Type No. data Metric


**BCF** Current work GNN 608 MAE: 0.25

Alshehri et al., 2022 GC+GPR 589 MAE: 0.05
Medina et al., 2021 GNN 473 MAE: 0.44
Zhao et al., 2008 QSPR 473 MAE: 0.45
Hukkerikar et al., 2012a GC+MLR 662 MAE: 0.47

**PCO** Current work GNN 608 MAE: 0.06

Alshehri et al., 2022 GC+GPR 606 MAE: 0.02
Hukkerikar et al., 2012a GC+MLR 639 MAE: 0.13
**pka** Current work GNN 1631 MAE: 0.68
Alshehri et al., 2022 GC+GPR 1634 MAE: 0.16
Zhou et al., 2018 GC+ANN 1622 MAE: 0.17
Mayr et al., 2022 GNN 208 MAE: 0.71
Zhou et al., 2018 GC+MLR 1622 MAE: 1.18

**LD50** Current work GNN 4781 MAE: 0.27
**(mol l** **[¡][1]** **)** Alshehri et al., 2022 GC+GPR 4904 MAE: 0.03
Hukkerikar et al., 2012a GC+MLR 5995 MAE: 0.35
Karim et al., 2021 Ensemble 7380 MAE: 0.40

**LC50** Current work GNN 705 MAE: 0.44
**(mol l** **[¡][1]** **)** Alshehri et al., 2022 GC+GPR 705 MAE: 0.06
Martin and Young, 2001 GC+MLR 397 MAE: 0.26
Hukkerikar et al., 2012a GC+MLR 809 MAE: 0.48
Karim et al., 2021 Ensemble 823 MAE: 048
**OSHA_TWA** Current work GNN 418 MAE: 0.44
**(mol m** **[¡][3]** **)** Alshehri et al., 2022 GC+GPR 422 MAE: 0.06
Hukkerikar et al., 2012a GC+MLR 425 MAE: 0.44
**BioD** Current work GNN 232 R [2] : 0.74
Jhamb et al., 2020 GC+MLR 232 R [2] : 0.69


of the critical temperature to the normal boiling point was calculated.
Logically, this should be above unity. The sanity check attained over a
99.99% success rate. In fact, this check holds for all compounds except
cyanogen. The reason behind this faulty prediction is not clear as no
experimental evaluation of the critical point was found for the com­
pound. For the second consistency check, the ratio of the normal boiling
point to the melting point achieved values above unity for 99.99% of the
cases. The condition did not hold for six compounds (carbamide, mel­
amine, hexamethylenetetramine, adamantane, diamantane, and carba­
moyl chloride). Most of these compounds undergo sublimation and such
could be a reason for their faulty outcome in the consistency check. The
third consistency check, the ratio of the auto-ignition temperature to the
flash point, showcased a success rate of 98.75%, where 24 compounds
did not adhere to the logical constraint set. This could be explained by
the performance of the auto-ignition temperature model (MAE of 41K
for the test set), which could be significant in some of the cases as this
only showcases the median and as such for some extreme cases could be
much larger. Another reason could be the fact that the experimental
determination of these properties is rather challenging as they are



associated with a degree of safety precautions in addition to the exis­
tence of various techniques to evaluate these and as such the consistency
of the available experimental data could be brought to question (See
DIPPR manual) (Rowley et al., 2019). For 96% of the cases, the
constraint for the flammability limits is valid. For the compounds that do
not follow the logical constraints, the LFL and UFL values are very close
to each other, and as such the need for accuracy increases to uphold the
constraint. For the last sanity check, we also provide two independent
ways of estimating the total solubility parameter (Hildebrand solubility
parameter), where one is provided directly through prediction and the
other through the prediction of the individual components (the Hansen
solubility parameters). As can be seen in Fig. 10, there is a mismatch in
some predictions, although the trend is qualitatively present. The cor­
relation coefficient obtained through this is app. 0.7, which also show­
cases the discrepancies between the two calculation methods. One
obvious reason is the discrepancies obtained for each of the models
(Hildebrand and the three Hansen solubility parameters as seen in
Table 5, especially for the test set).
An important insight to be reported and gained through the model
development process is the significant impact of the activation function
used. This will have a huge impact on the thermodynamic sanity of the
models developed. An example would be the use of leakyReLU, which is
not suitable for properties that should not exhibit negative values such
as the critical pressure (pressure is always positive) and the flammability
limits (percentages). Despite setting the slope of the negative linear part
of the leakReLU to a very small number, for cases with limiting values,
this might still result in a negative value. As such, the recommendation is
to use ReLU for properties such as critical pressure, solubility parame­
ters, and volumes. However, the use of ReLU for the final layer might
result in “dead neurons”
due to the negative gradients flowing through
the models. This renders the weight initialization more crucial for the
convergence of the training. For the LFL and UFL since they are limited
between 0%-100% and a potential linear function would cause the
percentage to exceed the upper limit, the recommendation is to use the
sigmoid activation function. This breach of logical constraints is down to
the fact the GNN models are not physics-informed compared to classical
QSPR models such as the GC models where the functional trans­
formation of the properties provides an aspect of logical constraints. As
such, we suggest that it is necessary to test the limits of the developed
models despite performance metrics showing excellent performance.
Despite the reported interpretability w.r.t. the important functional
groups in a given compound for a specific property being consistent with
chemistry knowledge, this does not necessarily mean that the models are
sane and thermodynamically consistent internally or across properties.
Training model independently does not force the properties to be
mutually consistent thermodynamically or logically. As such, multi-task
learning could potentially prove to be a possible direction for future



**Fig. 8.** Attention visualization (property: flashpoint, compound: paracetamol). A darker shade of red indicates higher attention weight.
1 refers to the coefficients published in (Hukkerikar et al., 2012b).


16


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_


**Fig. 9.** Sanity plots: (a) the ratio of the critical temperature to the boiling point, (b) the ratio of the boiling point to the melting point, (c) the ratio of the upper and
lower flammability limits, (d) the ratio of the auto-ignition temperature to the flash point temperature. Molar weight is used as the color code.



research to insure this. Multi-task learning is capable of leveraging in­
ternal correlation between target endpoints and as such could learn
some logical or thermodynamic constraints such as AIT being larger
than FP.


**5. Conclusion**


GNNs are versatile models that are capable of modeling a wide range
of properties. In this work, we extended the range of applicability of two
interpretable graph neural network models that integrates the concept
of functional groups to provide an aspect of interpretability to the
models to cover a total of 30 properties. The results showcase this fact by
providing models with accuracy suitable for their intended purpose



which is the enable computer-aided screening based on pure compound
properties. However, some properties remain a challenge, especially
those related to solubility and the toxicity of a compound. The inte­
gration of functional groups showcased insights that are consistent with
those obtained from GC models and might provide further insight into
important substructures impacting the endpoint prediction. Despite
good performance, future QSPR developments must test the limits of
their models by performing logical and thermodynamic sanity checks on
a benchmark set of compounds. Through this study, we showcased that
cross-property validation might not hold despite each property model
showcasing good performance individually and that the choice of acti­
vation function can help guide the physical and thermodynamic con­
sistency of the models.



17


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_



**Fig. 10.** parity plot showcasing the discrepancies between calculated (through
Hansen parameters) Hildebrand parameters and directly predicted Hilde­
brand parameters.


**CRediT authorship contribution statement**


**Adem R.N. Aouichaoui:** Conceptualization, Methodology, Soft­
ware, Validation, Formal analysis, Data curation, Writing – original
draft, Writing – review & editing, Visualization. **Fan Fan:** Methodology,
Software, Validation, Formal analysis, Writing – review & editing,
Visualization. **Jens Abildskov:** Writing – review & editing, Supervision,
Funding acquisition. **Gürkan Sin:** Writing – review & editing, Supervi­
sion, Funding acquisition.


**Declaration of Competing Interest**


The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to influence
the work reported in this paper.


**Data availability**


The authors do not have permission to share data.


**Supplementary materials**


Supplementary material associated with this article can be found, in
[the online version, at doi:10.1016/j.compchemeng.2023.108291.](https://doi.org/10.1016/j.compchemeng.2023.108291)


**References**


[Alshehri, A.S., Tula, A.K., You, F., Gani, R., 2022. Next generation pure component](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0001)
[property estimation models: with and without machine learning techniques. AlChE](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0001)
[J. 68.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0001)

[Aouichaoui, A.R.N., Al, R., Abildskov, J., Sin, G., 2021. Comparison of group-](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0002)
[contribution and machine learning-based property prediction models with](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0002)
[uncertainty quantification. In: Computer Aided Chemical Engineering. Elsevier](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0002)
[Masson SAS, pp. 755–760.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0002)
[Aouichaoui, A.R.N., Fan, F., Mansouri, S.S., Abildskov, J., Sin, G., 2023. Combining](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0003)
[group-contribution concept and graph neural networks toward interpretable](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0003)
[molecular property models. J. Chem. Inf. Model. 63, 725–744.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0003)
[Aouichaoui, A.R.N., Mansouri, S.S., Abildskov, J., Sin, G., 2022a. Uncertainty estimation](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0004)
[in deep learning-based property models: graph neural networks applied to the](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0004)
[critical properties. AlChE J. 68.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0004)



[Aouichaoui, A.R.N., Mansouri, S.S., Abildskov, J., Sin, G., 2022b. Application of outlier](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0005)
[treatment towards improved property prediction models. In: 32nd European](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0005)
[Symposium on Computer Aided Process Engineering. Elsevier Masson SAS,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0005)
[pp. 1357–1362.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0005)
[Benson, S.W., 1999. New methods for estimating the heats of formation, heat capacities,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0006)
[and entropies of liquids and gases. J. Phys. Chem. A 103, 11481–11485.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0006)
[Cai, C., Marsh, A., Zhang, Y.H., Reid, J.P., 2017. Group contribution approach to predict](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0007)
[the refractive index of pure organic components in ambient organic aerosol. Environ.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0007)
[Sci. Technol. 51, 9683–9690.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0007)
[Cao, H.Y., Jiang, J.C., Pan, Y., Wang, R., Cui, Y., 2009. Prediction of the net heat of](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0008)
[combustion of organic compounds based on atom-type electrotopological state](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0008)
[indices. J. Loss Prev. Process Ind. 22, 222–227.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0008)
[Tu, Chein-Hsiun, 1995. Group-contribution estimation of critical temperature with only](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0009)
[chemical structure. Chem. Eng. Sci. 50, 3515–3520.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0009)
[Cignitti, S., Rodriguez-Donis, I., Abildskov, J., You, X., Shcherbakova, N., Gerbaud, V.,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0010)
[2019. CAMD for entrainer screening of extractive distillation process based on new](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0010)
[thermodynamic criteria. Chem. Eng. Res. Des. 147, 721–733.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0010)
[Cirillo, P., Taleb, N.N., 2020. Tail risk of contagious diseases. Nat. Phys. 16, 606–613.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0011)
[Coley, C.W., Barzilay, R., Green, W.H., Jaakkola, T.S., Jensen, K.F., 2017. Convolutional](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0012)
[embedding of attributed molecular graphs for physical property prediction. J. Chem.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0012)
[Inf. Model. 57, 1757–1772.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0012)
[Constantinou, L., Gani, R., 1994. New group contribution method for estimating](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0013)
[properties of pure compounds. AlChE J. 40, 1697–1710.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0013)
[Constantinou, L., Gani, R., O’Connell, J.P., 1995. Estimation of the acentric factor and](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0014)
[the liquid molar volume at 298 K using a new group contribution method. Fluid](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0014)
[Duvenaud, D., Maclaurin, D., Aguilera-Iparraguirre, J., GPhase Equilib. 103, 11–22.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0014) omez-Bombarelli, R., Hirzel, T., ´
[Aspuru-Guzik, A., Adams, R.P., 2015. Convolutional networks on graphs for learning](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0015)
[molecular fingerprints. Adv. Neural Inf. Process. Syst. 2224–2232, 2015-Janua.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0015)
[Enekvist, M., Liang, X., Zhang, X., Dam-Johansen, K., Kontogeorgis, G.M., 2022.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0016)
[Computer-aided design and solvent selection for organic paint and coating](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0016)
[formulations. Prog. Org. Coat. 162, 106568.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0016)
[Frenkel, M., 2011. Thermophysical and thermochemical properties on-demand for](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0017)
[chemical process and product design. Comput. Chem. Eng. 35, 393–402.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0017)
[Frutiger, J., Bell, I., O’Connell, J.P., Kroenlein, K., Abildskov, J., Sin, G., 2017.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0018)
[Uncertainty assessment of equations of state with application to an organic Rankine](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0018)
[cycle. Mol. Phys. 115, 1225–1244.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0018)
[Frutiger, J., Marcarie, C., Abildskov, J., Sin, G., 2016a. Group-contribution based](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0019)
[property estimation and uncertainty analysis for flammability-related properties.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0019)
[J. Hazard. Mater. 318, 783–793.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0019)
[Frutiger, J., Marcarie, C., Abildskov, J., Sin, G., 2016b. A comprehensive methodology](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0020)
[for development, parameter estimation, and uncertainty analysis of group](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0020)
[contribution based property models-an application to the heat of combustion.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0020)
[J. Chem. Eng. Data 61, 602–613.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0020)
[Gani, R., 2019. Group contribution-based property estimation methods: advances and](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0021)
[perspectives. Curr. Opin. Chem. Eng. 23, 184–196.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0021)
[Gasteiger, J., 2016. Chemoinformatics: achievements and challenges, a personal view.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0022)
[Molecules 21, 151.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0022)
[Gharagheizi, F., 2008. Quantitative structure−](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0023) property relationship for prediction of the
[lower flammability limit of pure compounds. Energy Fuels 22, 3037–3039.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0023)
[Gharagheizi, F., Ilani-Kashkouli, P., Kamari, A., Mohammadi, A.H., Ramjugernath, D.,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0024)
[2014. Group contribution model for the prediction of refractive indices of organic](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0024)
[compounds. J. Chem. Eng. Data 59, 1930–1943.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0024)
[Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O., Dahl, G.E., 2017. Neural message](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0025)
[passing for quantum chemistry. In: 34th International Conference on Machine](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0025)
[Learning. ICML, pp. 2053–2070, 2017.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0025)
[Hansen, H.K., Rasmussen, P., Fredenslund, A., Schiller, M., Gmehling, J., 1991. Vapor-](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0026)
[liquid equilibria by UNIFAC group contribution. 5. Revision and extension. Ind. Eng.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0026)
[Chem. Res. 30, 2352–2355.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0026)
[Hasebe, T., 2021. Knowledge-embedded message-passing neural networks: improving](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0027)
[molecular property prediction with human knowledge. ACS Omega 6, 27955–27967.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0027)
[Hirschfeld, L., Swanson, K., Yang, K., Barzilay, R., Coley, C.W., 2020. Uncertainty](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0028)
[quantification using neural networks for molecular property prediction. J. Chem. Inf.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0028)
[Model. 60, 3770–3780.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0028)
[Hukkerikar, A.S., Kalakul, S., Sarup, B., Young, D.M., Sin, G., Gani, R., 2012a. Estimation](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0029)
[of environment-related properties of chemicals for design of sustainable processes:](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0029)
[development of group-contribution+ (GC +) property models and uncertainty](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0029)
[analysis. J. Chem. Inf. Model. 52, 2823–2839.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0029)
[Hukkerikar, A.S., Meier, R.J., Sin, G., Gani, R., 2013. A method to estimate the enthalpy](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0030)
[of formation of organic compounds with chemical accuracy. Fluid Phase Equilib.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0030)
[348, 23–32.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0030)
[Hukkerikar, A.S., Sarup, B., Ten Kate, A., Abildskov, J., Sin, G., Gani, R., 2012b. Group-](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0031)
[contribution + (GC +) based estimation of properties of pure components: Improved](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0031)
[property estimation and uncertainty analysis. Fluid Phase Equilib. 321, 25–43.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0031)
[Hwang, D., Yang, S., Kwon, Y., Lee, K.H., Lee, G., Jo, H., Yoon, S., Ryu, S., 2020.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0032)
[Comprehensive study on molecular supervised learning with graph neural networks.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0032)
[J. Chem. Inf. Model. 60, 5936–5945.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0032)
[Hwangbo, S., Al, R., Chen, X., Sin, G., 2021. Integrated model for understanding N2O](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0033)
[emissions from wastewater treatment plants: a deep learning approach. Environ. Sci.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0033)
[Technol. 55, 2143–2151.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0033)
[Jhamb, S., Hospital, I., Liang, X., Pilloud, F., Piccione, P.M., Kontogeorgis, G.M., 2020.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0034)
[Group contribution method to estimate the biodegradability of organic compounds.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0034)
[Jim´enez, J., Ginebra, J., 2017. pyGPGO: bayesian optimization for python. J. Open Ind. Eng. Chem. Res. 59, 20916–20928.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0034)
[Source Software 2, 431–433.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0035)



18


_A.R.N. Aouichaoui et al._ _Computers and Chemical Engineering 176 (2023) 108291_



[Jim´enez-Luna, J., Grisoni, F., Schneider, G., 2020. Drug discovery with explainable](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0036)
[Jim´enez-Luna, J., Skalic, M., Weskamp, N., Schneider, G., 2021. Coloring molecules with artificial intelligence. Nat. Mach. Intell. 2, 573–584.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0036)
[explainable artificial intelligence for preclinical relevance assessment. J. Chem. Inf.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0037)
[Model. 61, 1083–1094.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0037)
[Joback, K.G., Reid, R.C., 1987. Estimation of pure-component properties from group-](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0038)
[contributions. Chem. Eng. Commun. 57, 233–243.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0038)
[Karim, A., Riahi, V., Mishra, A., Newton, M.A.H., Dehzangi, A., Balle, T., Sattar, A., 2021.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0039)
[Quantitative toxicity prediction via meta ensembling of multitask deep learning](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0039)
[models. ACS Omega 6, 12306–12317.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0039)
[Karunanithi, A.T., Achenie, L.E.K., Gani, R., 2006. A computer-aided molecular design](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0040)
[framework for crystallization solvent design. Chem. Eng. Sci. 61, 1247–1260.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0040)
[Katritzky, A.R., Kuanar, M., Slavov, S., Hall, C.D., Karelson, M., Kahn, I., Dobchev, D.A.,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0041)
[2010. Quantitative correlation of physical and chemical properties with chemical](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0041)
[structure: utility for prediction. Chem. Rev. 110, 5714–5789.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0041)
[Kingma, D.P., Ba, J.L., 2015. Adam: a method for stochastic optimization. In: 3rd](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0042)
[International Conference on Learning Representations, ICLR 2015 - Conference](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0042)
[Track Proceedings, pp. 1–15.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0042)
[Klincewicz, K.M., Reid, R.C., 1984. Estimation of critical properties with group](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0043)
[contribution methods. AlChE J. 30, 137–142.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0043)
Landrum, G., 2021. RDKit: Open-source cheminformatics.
[Liu, R., Zhou, D., 2008. Using molecular fingerprint as descriptors in the QSPR study of](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0045)
[lipophilicity. J. Chem. Inf. Model. 48, 542–549.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0045)
[Mansouri, K., Grulke, C.M., Richard, A.M., Judson, R.S., Williams, A.J., 2016. An](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0046)
[automated curation procedure for addressing chemical errors and inconsistencies in](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0046)
[public datasets used in QSAR modelling. SAR QSAR Environ. Res. 27, 911–937.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0046)
[Marrero, J., Gani, R., 2001. Group-contribution based estimation of pure component](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0047)
[properties. Fluid Phase Equilib. 183–184, 183–208.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0047)
[Martin, T.M., Young, D.M., 2001. Prediction of the acute toxicity (96-h LC50) of organic](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0048)
[compounds to the fathead minnow (pimephales promelas) using a group](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0048)
[contribution method. Chem. Res. Toxicol. 14, 1378–1385.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0048)
[Mayr, F., Wieder, M., Wieder, O., Langer, T., 2022. Improving small molecule pka](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0049)
[prediction using transfer learning with graph neural networks. Front. Chem. 10.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0049)
[Medina, E.I.S., Linke, S., Sundmacher, K., 2021. Prediction of Bioconcentration Factors](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0050)
[(BCF) using Graph Neural Networks. Computer Aided Chemical Engineering.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0050)
[Elsevier B.V., pp. 991–997](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0050)
[Meier, R.J., 2022. Group contribution revisited: the enthalpy of formation of organic](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0051)
[compounds with “chemical accuracy” part III. Appl. Chem. 2, 213–228.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0051)
[Meier, R.J., 2021a. Group contribution revisited: the enthalpy of formation of organic](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0052)
[compounds with “chemical accuracy. Chem. Eng. 5, 24.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0052)
[Meier, R.J., 2021b. Group contribution revisited: the enthalpy of formation of organic](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0053)
[compounds with “chemical accuracy. Part II. Appl. Chem. 1, 111–129.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0053)
[Metcalf & Eddy, 2014. Wastewater engineering: treatment and resource recovery.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0054)
[McGraw Hill Education.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0054)

[Mondejar, M.E., Cignitti, S., Abildskov, J., Woodley, J.M., Haglind, F., 2017. Prediction](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0055)
[of properties of new halogenated olefins using two group contribution approaches.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0055)
[Fluid. Phase Equilib. 433, 79–96.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0055)
[Mondejar, M.E., Frutiger, J., Cignitti, S., Abildskov, J., Sin, G., Woodley, J.M.,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0056)
[Haglind, F., 2019. Uncertainty in the prediction of the thermophysical behavior of](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0056)
[new halogenated working fluids. Fluid. Phase Equilib. 485, 220–233.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0056)
[Naef, R., Acree, W.E., 2022. Revision and extension of a generally applicable group](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0057)
[additivity method for the calculation of the refractivity and polarizability of organic](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0057)
[molecules at 298.15 K. Liquids 2, 327–377.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0057)
[Netzeva, T.I., Worth, A.P., Aldenberg, T., Benigni, R., Cronin, M.T.D., Gramatica, P.,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0058)
[Jaworska, J.S., Kahn, S., Klopman, G., Marchant, C.A., Myatt, G., Nikolova-](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0058)
[Jeliazkova, N., Patlewicz, G.Y., Perkins, R., Roberts, D.W., Schultz, T.W., Stanton, D.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0058)
[T., Van De Sandt, J.J.M., Tong, W., Veith, G., Yang, C., 2005. Current status of](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0058)
[methods for defining the applicability domain of (quantitative) structure-activity](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0058)
[relationships. ATLA Altern. Lab. Anim. 33, 155–173.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0058)
[Nielsen, T.L., Abildskov, J., Harper, P.M., Papaeconomou, I., Gani, R., 2001. The CAPEC](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0059)
[database. J. Chem. Eng. Data 46, 1041–1044.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0059)
[Park, S., Bailey, J.P., Pasman, H.J., Wang, Q., El-Halwagi, M.M., 2021. Fast, easy-to-use,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0060)
[machine learning-developed models of prediction of flash point, heat of combustion,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0060)
[and lower and upper flammability limits for inherently safer design. Comput. Chem.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0060)
[Eng. 155, 107524.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0060)
[Parveen, R., Cundari, T.R., Younker, J.M., Rodriguez, G., McCullough, L., 2019. DFT and](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0061)
[QSAR studies of ethylene polymerization by zirconocene catalysts. ACS Catal. 9,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0061)
[9339–9349.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0061)

[Qu, C., Kearsley, A.J., Schneider, B.I., Keyrouz, W., Allison, T.C., 2022. Graph](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0062)
[convolutional neural network applied to the prediction of normal boiling point.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0062)
[J. Mol. Graph Model. 112.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0062)
[Reymond, J.L., 2015. The chemical space project. Acc. Chem. Res. 48, 722–730.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0063)
[Rogers, D., Hahn, M., 2010. Extended-connectivity fingerprints. J. Chem. Inf. Model. 50,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0064)
[742–754.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0064)

Rowley, R.I., Wilding, W.V., Oscarson, J.L., Giles, N.F., 2019. DIPPR data compilation of
pure chemical properties.
[Ruddigkeit, L., Van Deursen, R., Blum, L.C., Reymond, J.L., 2012. Enumeration of 166](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0066)
[billion organic small molecules in the chemical universe database GDB-17. J. Chem.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0066)
[Inf. Model. 52, 2864–2875.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0066)
[Ryu, S., Kwon, Y., Kim, W.Y., 2019. A Bayesian graph convolutional network for reliable](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0067)
[prediction of molecular properties with uncertainty quantification. Chem. Sci. 10,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0067)
[8438–8446.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0067)



[Sanchez-Lengeling, B., Roch, L.M., Perea, J.D., Langner, S., Brabec, C.J., Aspuru-](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0068)
[Guzik, A., 2019. A bayesian approach to predict solubility parameters. Adv. Theory](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0068)
[Simul. 2.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0068)

[Scalia, G., Grambow, C.A., Pernici, B., Li, Y.P., Green, W.H., 2020. Evaluating scalable](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0069)
[uncertainty estimation methods for deep learning-based molecular property](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0069)
[Schweidtmann, A.M., Rittig, J.G., Kprediction. J. Chem. Inf. Model. 60, 2697onig, A., Grohe, M., Mitsos, A., Dahmen, M., 2020. ¨](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0069) –2717.
[Graph neural networks for prediction of fuel ignition quality. Energy Fuels 34,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0070)
[11395–11407.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0070)
[Sivaraman, G., Jackson, N.E., Sanchez-Lengeling, B., Vazquez-Mayagoitia, ´](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0071) A., Aspuru- [´]
[Guzik, A., Vishwanath, V., de Pablo, J.J., 2020. A machine learning workflow for](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0071)
[molecular analysis: application to melting points. Mach. Learn. Sci. Technol. 1,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0071)
[025015.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0071)

[Stefanis, E., Constantinou, L., Panayiotou, C., 2004. A group-contribution method for](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0072)
[predicting pure component properties of biochemical and safety interest. Ind. Eng.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0072)
[Chem. Res. 43, 6253–6261.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0072)
[Stefanis, E., Panayiotou, C., 2008. Prediction of hansen solubility parameters with a new](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0073)
[group-contribution method. Int. J. Thermophys. 29, 568–585.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0073)
[Su, Y., Wang, Z., Jin, S., Shen, W., Ren, J., Eden, M.R., 2019. An architecture of deep](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0074)
[learning in QSPR modeling for the prediction of critical properties using molecular](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0074)
[signatures. AlChE J. 65, 1–11.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0074)
[Sun, X., Krakauer, N.J., Politowicz, A., Chen, W.T., Li, Q., Li, Z., Shao, X., Sunaryo, A.,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0075)
[Shen, M., Wang, J., Morgan, D., 2020. Assessing graph-based deep learning models](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0075)
[for predicting flash point. Mol. Inform. 39.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0075)
[Tang, B., Kramer, S.T., Fang, M., Qiu, Y., Wu, Z., Xu, D., 2020. A self-attention based](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0076)
[message passing neural network for predicting molecular lipophilicity and aqueous](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0076)
[solubility. J. Cheminform. 12, 15.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0076)
Trinh, C., Meimaroglou, D., Lasala, S., Herbinet, O., 2022. Machine Learning for the
prediction of the thermochemical properties (enthalpy and entropy of formation) of
a molecule from its molecular descriptors. pp. 1471–1476.
[Undavalli, V.K., Ling, C., Khandelwal, B., 2021. Impact of alternative fuels and properties](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0078)
[on elastomer compatibility. Aviation Fuels. Elsevier, pp. 113–132.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0078)
US EPA, 2023. Estimation Programs Interface SuiteTM for Microsoft® Windows.
[Van Speybroeck, V., Gani, R., Meier, R.J., 2010. The calculation of thermodynamic](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0080)
[properties of molecules. Chem. Soc. Rev. 39, 1764–1779.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0080)
[Wellawatte, G.P., Seshadri, A., White, A.D., 2022. Model agnostic generation of](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0081)
[counterfactual explanations for molecules. Chem. Sci. 13, 3697–3705.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0081)
[Wen, H., Su, Y., Wang, Z., Jin, S., Ren, J., Shen, W., Eden, M., 2022. A systematic](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0082)
[modeling methodology of deep neural network-based structure-property](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0082)
[relationship for rapid and reliable prediction on flashpoints. AlChE J. 68.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0082)
[Wieder, O., Kohlbacher, S., Kuenemann, M., Garon, A., Ducrot, P., Seidel, T., Langer, T.,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0083)
[2020. A compact review of molecular property prediction with graph neural](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0083)
[networks. Drug Discov. Today Technol. 37, 1–12.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0083)
[Wu, Z., Ramsundar, B., Feinberg, E.N., Gomes, J., Geniesse, C., Pappu, A.S., Leswing, K.,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0084)
[Pande, V., 2018. MoleculeNet: a benchmark for molecular machine learning. Chem.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0084)
[Sci. 9, 513–530.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0084)
[Wyttenbach, N., Niederquell, A., Kuentz, M., 2020. Machine estimation of drug melting](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0085)
[properties and influence on solubility prediction. Mol. Pharm. 17, 2660–2671.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0085)
[Xiong, Z., Wang, D., Liu, X., Zhong, F., Wan, X., Li, X., Li, Z., Luo, X., Chen, K., Jiang, H.,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0086)
[Zheng, M., 2020. Pushing the boundaries of molecular representation for drug](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0086)
[discovery with the graph attention mechanism. J. Med. Chem. 63, 8749–8760.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0086)
[Yang, A., Su, Y., Wang, Z., Jin, S., Ren, J., Zhang, X., Shen, W., Clark, J.H., 2021. A multi-](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0087)
[task deep learning neural network for predicting flammability-related properties](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0087)
[from molecular structures. Green Chem. 23, 4451–4465.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0087)
[Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., Guzman-Perez, A.,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0088)
[Hopper, T., Kelley, B., Mathea, M., Palmer, A., Settels, V., Jaakkola, T., Jensen, K.,](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0088)
[Barzilay, R., 2019. Analyzing learned molecular representations for property](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0088)
[prediction. J. Chem. Inf. Model. 59, 3370–3388.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0088)
[Yuan, S., Jiao, Z., Quddus, N., Kwon, J.S.-I, Mashuga, C.V., 2019. Developing](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0089)
[quantitative structure–property relationship models to predict the upper](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0089)
[flammability limit using machine learning. Ind. Eng. Chem. Res. 58, 3531–3537.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0089)
[Zhang, J., Wang, Q., Su, Y., Jin, S., Ren, J., Eden, M., Shen, W., 2022. An accurate and](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0090)
[interpretable deep learning model for environmental properties prediction using](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0090)
[hybrid molecular representations. AlChE J. 68.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0090)
[Zhang, Z., Guan, J., Zhou, S., 2021. FraGAT: a fragment-oriented multi-scale graph](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0091)
[attention model for molecular property prediction. Bioinformatics 37, 2981–2987.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0091)
[Zhao, C., Boriani, E., Chana, A., Roncaglioni, A., Benfenati, E., 2008. A new hybrid](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0092)
[system of QSAR models for predicting bioconcentration factors (BCF). Chemosphere](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0092)
[73, 1701–1707.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0092)
[Zheng, D., Wang, M., Gan, Q., Song, X., Zhang, Z., Karypis, G., 2021. Scalable graph](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0093)
[neural networks with deep graph library. In: Proceedings of the 14th ACM](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0093)
[International Conference on Web Search and Data Mining. ACM, pp. 1141–1142.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0093)
[Zheng, P., Yang, W., Wu, W., Isayev, O., Dral, P.O., 2022. Toward chemical accuracy in](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0094)
[predicting enthalpies of formation with general-purpose data-driven methods.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0094)
[J. Phys. Chem. Lett. 13, 3479–3491.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0094)
Zhou, J., Cui, G., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C., Sun, M., 2018. Graph
neural networks: a review of methods and applications 1–22.
[Zhou, T., Jhamb, S., Liang, X., Sundmacher, K., Gani, R., 2018. Prediction of acid](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0096)
[dissociation constants of organic compounds using group contribution methods.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0096)
[Chem. Eng. Sci. 183, 95–105.](http://refhub.elsevier.com/S0098-1354(23)00161-8/sbref0096)



19


