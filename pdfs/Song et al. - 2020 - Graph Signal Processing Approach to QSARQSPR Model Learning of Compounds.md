This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TPAMI.2020.3032718, IEEE

Transactions on Pattern Analysis and Machine Intelligence


1

## Graph Signal Processing Approach to QSAR/QSPR Model Learning of Compounds


Xiaoying Song, Li Chai*, and Jingxin Zhang



_**Abstract**_ **—Quantitative** **relationship** **between** **the** **activi-**
**ty/property and the structure of compound is critical in chemical**
**applications. To learn this quantitative relationship, hundreds**
**of molecular descriptors have been designed to describe the**
**structure, mainly based on the properties of vertices and edges of**
**molecular graph. However, many descriptors degenerate to the**
**same values for different compounds with the same molecular**
**graph, resulting in model failure. In this paper, we design**
**a multidimensional signal for each vertex of the molecular**
**graph to derive new descriptors with higher discriminability.**
**We treat the new and traditional descriptors as the signals**
**on the descriptor graph learned from the descriptor data, and**
**enhance descriptor dissimilarity using the Laplacian filter derived**
**from the descriptor graph. Combining these with model learning**
**techniques, we propose a graph signal processing based approach**
**to obtain reliable new models for learning the quantitative**
**relationship and predicting the properties of compounds. We**
**also provide insights from chemistry for the boiling point model.**
**Several experiments are presented to demonstrate the validity,**
**effectiveness and advantages of the proposed approach.**


_**Index Terms**_ **—QSAR/QSPR model learning, compounds, graph**
**signal processing (GSP), multidimensional signal**


I. I NTRODUCTION
# E XTENSIVE chemical and medical experiments have re-vealed that physicochemical properties of the compound

are highly related to its molecular structure. Applying chemical
theory and various mathematical analysis methods, the Quantitative Structure Activity/Property Relationship (QSAR/QSPR)
model learning attempts to describe this relationship quantitatively [1, 2]. QSAR/QSPR model has become an extensively
used tool in computer-aided drug design, toxicity and property
prediction of chemicals and pharmaceuticals [3] as well as in
different modeling problems in material sciences [4], analytical chemistry and pharmacodynamics profiling of new drug
molecules [5].
In most cases, the molecular structure is represented as
graph (called molecular graph) with vertices denoting atoms
and edges describing chemical bonds between atoms. This
graph representation allows for application of graph theoretic
algorithms to assess statistical and/or topological properties of
networks reconstructed from molecular structures. Based on

these statistical and/or topological indices, one can estimate the


*Corresponding author. Li Chai is with the Engineering Research Center
of Metallurgical Automation and Measurement Technology, Wuhan University
of Science and Technology, Wuhan, China. e-mail: chaili@wust.edu.cn.
Xiaoying Song is with the same University. e-mail: xiaoying811@wust.edu.cn.
Jingxin Zhang is with the School of Software and Electrical Engineering,
Swinburne University of Technology, Melbourne, Australia. e-mail: jingxinzhang@swin.edu.au.



chemical, biological, medical and pharmacological properties
of compounds.
Over the past decade, considerable progress has been made
in QSAR/QSPR model learning. It is now feasible to study
and predict properties of new compounds from a set of training
molecules with known activities/properties/toxicities. Learning
QSAR/QSPR models requires three main steps: generating
a training set of measured properties of known compounds,
encoding the information of compound structures, and building
a mathematical model to predict measured properties from the
encoded structure [6].
QSAR/QSPR models are regression or classification models. QSAR/QSPR regression models relate a set of “predictor”
variables (also called molecular descriptors) to the potency
of the response variable, while QSAR/QSPR classification
models relate the predictor variables to a categorical value of
the response variable. The molecular descriptors consist of two
main categories: experimental measurements, such as log P,
molar refractivity, dipole moment and physicochemical properties in general, and theoretical molecular descriptors, which
are derived from a symbolic representation of the molecule
and can be further classified according to the different types
of molecular representation. The response variable could be
a biological activity or property of the compound. Various
learning techniques have been applied to QSAR/QSPR model
learning in recent years. Examples include partial least squares
(PLS) [7], multiple linear regression (MLR) [3], support vector
machine [8], random forest [9], neural networks [10] and so

on.

Generally hundreds of molecular descriptors are required
to learn a valid model [11, 12]. However, most descriptors
are constructed based on topological properties of molecular
graph, which may generate the same value for different
compounds having the same molecular graph. Therefore, one
cannot build useful QSAR/QSPR models to represent different
properties of such compounds. To overcome this difficulty,
new technical tools are required.
Graph signal processing (GSP) has recently emerged as
a powerful new approach to analyzing and processing highdimensional signals defined on irregular graph domains [13–
16]. It treats the data at vertices as the signal on graph, and
then analyzes and processes the data from signal processing
perspective. The theory of GSP has been growing rapidly
in recent years, with development in methods such as graph
filtering [17] and graph neural network (GNN) learning [18].
These new theory and techniques have provided new tools for
QSAR/QSPR model learning and have motivated this work.
We propose in this paper a new GSP based approach



0162-8828 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Auckland University of Technology. Downloaded on November 01,2020 at 22:09:08 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TPAMI.2020.3032718, IEEE

Transactions on Pattern Analysis and Machine Intelligence


2



to QSAR/QSPR model learning. This approach stems from
an intuitive idea that different compounds with the same
molecular graph can be represented by different signals on
the vertices of the graph; such signals are the graph signals in
the GSP theory, and hence can be processed using various
GSP tools to find new QSAR/QSPR models for different
compounds with the same or similar molecular structures.
Based on this idea, we propose the following GSP based
approach to QSAR/QSPR model learning: 1) Design new
multidimensional (MD) signal at each vertex of molecular
graph; 2) Analyze the MD signals (MDS) on all vertices
of the graph to derive new MDS descriptors with higher
discriminability for QSAR/QSPR modeling; 3) Combine the
new and traditional descriptors to construct new QSAR/QSPR
models with higher performance and reliability; 4) Treat the
descriptors as the signals on a new descriptor graph and
construct the graph from descriptor data; 5) Derive a Laplacian
graph filter from the descriptor graph and use Laplacian
filtering of descriptors to enhance their dissimilarity; 6) Use
the dissimilarity enhanced input variables in new QSAR/QSPR
model to further enhance model performance and reliability;
7) Use Least Angle Regression with Lasso modification to
learn the optimal sparse parameters of the model.
Existing works, in descriptor design and in modeling, are
based only on the molecular graph. Our GSP based approach
is very different from those of the previous works. To the
best of our knowledge, this approach has never been used in
QSAR/QSPR model learning of compounds. The contributions
of this paper are:
i) A general GSP based approach to QSAR/QSPR model
learning of compounds as summarized above.
ii) Application results, showing the advantages of the proposed approach, obtained from two benchmark datasets.
iii) A biological activity model for phenetylamines, the
benchmark dataset including 22 different compounds with
the same molecular graph. A simple relationship between the
MDS and the biological activity and the state-of-the-art fitting
results.

iv) A boiling point model for polyaromatic hydrocarbons, the benchmark dataset including 82 different compounds
with similar molecular graphs, showing the boiling point of
compounds can be estimated by the molecular graph spectral
information and the molecular mass.

The paper is organized as follows. Section II provides some
background about GSP and QSAR/QSPR model learning.
Section III presents the proposed approach and its methods.
Sections IV and V present the applications of the proposed
approach in two families of compounds. Results obtained from
the applications are given in Section VI. Discussions and
conclusions on the results of the paper are given in Sections
VII and VIII, respectively.


II. P RELIMINARIES


_A. Notations and Definitions_


A graph is denoted by _G_ = ( _V, E, W_ ) with vertex set
_V_ = _{v_ 1 _, v_ 2 _, · · ·, v_ _N_ _}_, edge set _E_ and weight matrix _W_ .
The number of vertices is _N_ = _|V |_ and the number of edges



is _m_ = _|E|_ . The _(i,j)_ -th element of _W_ is the weight of the
edge _ε_ _ij_ _∈_ _E_ . In molecular graph, it is nonzero if there is a
chemical bond between vertices _v_ _i_ and _v_ _j_, otherwise it is zero.
Different weights are assigned to different chemical bonds,
with values of 1, 2, 3 and 1.5 of _w_ _ij_ representing single,
double, triple and aromatic bonds, respectively. An illustrate
example is shown in Fig. 1. In graph representation, hydrogen
atoms are implicit and omitted, which has the advantage of
leading to more compact graph architecture and faster training
speed.


Fig. 1: The molecular graph and weight matrix of
2-methyl-2-butene.


Define diagonal matrix _D_ := _diag_ ( _d_ _i_ ), where _d_ _i_ = [�] _j_ _[w]_ _[ij]_
is the degree of the vertex _v_ _i_ . The Laplacian matrix of _G_ is
defined as _L_ = _D −_ _W_ . _L_ is symmetric and positive semidefinite, and its eigenvalues _{_ 0 _≤_ _λ_ 1 _≤_ _λ_ 2 _· · · ≤_ _λ_ _N_ _}_ are
defined as the spectra of the graph. Note that the definition of
Laplacian matrix is not unique. A different definition will be
introduced in Section III-B. A graph signal is a mapping from
the vertex set to the real number field, i.e., _x_ : _V →_ _R_ . An
operator _H ∈_ _R_ _[N]_ _[×][N]_ that gives output _H_ **x** for a graph signal
**x** _∈_ _R_ _[N]_ _[×]_ [1] (or **x** _[T]_ _H_ _[T]_ for **x** _[T]_ _∈_ _R_ [1] _[×][N]_ ) represents a graph
filter. In the canonical form, _W_ has a low pass nature and _L_
has a high pass nature in the graph spectrum domain [19].
All graphs considered in this work are assumed to be simple,
undirected and weighted, that is, no loop and multiple edge.


_B. Molecular Descriptors_


Molecular descriptor plays an important role in the development and interpretation of QSAR/QSPR model. It is defined
as a positive valued real function Ψ : _G →_ _R_ [+] that maps the
molecular graph to a positive real number. There are many
degree-based, distance-based and spectrum-based descriptors,
which have been widely used in the QSAR/QSPR model to
test properties of compounds. In recent years, several research
groups have made outstanding contributions to different kinds
of descriptors of special molecular structures [20–23]. Lucic
et al. [20] showed that the Randic connectivity index and the
sum-connectivity index are closely related molecular descriptors. Hayat et al. [21, 22] performed a comparative testing to
measure the efficiency of all the well-known valency-based
molecular descriptors and proposed an efficient computerbased computational technique to compute descriptors. Vukicevic and Graovac [23] analyzed the first and the second
Zagreb eccentricity descriptors. Several well known degreebased and distance-based descriptors are briefly described
below [24].



0162-8828 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Auckland University of Technology. Downloaded on November 01,2020 at 22:09:08 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TPAMI.2020.3032718, IEEE

Transactions on Pattern Analysis and Machine Intelligence


3



The general Randic index and the general sum connectivity
index are stated as


_R_ _d_ ( _G_ ) = �( _d_ _i_ _d_ _j_ ) _[d]_ _,_ (1)


_ε_ _ij_


where _d_ is a real number, _d_ _i_ and _d_ _j_ are degrees of vertices _v_ _i_
and _v_ _j_, respectively.
The general sum connectivity index is stated as


_χ_ _d_ ( _G_ ) = �( _d_ _i_ + _d_ _j_ ) _[d]_ _._ (2)


_ε_ _ij_


The distance-based version of the atom-bond connectivity
index is defined as



_ABC_ 2 ( _G_ ) = �

_ε_ _ij_ ~~�~~



_n_ _i_ + _n_ _j_ _−_ 2

_,_ (3)
_n_ _i_ _n_ _j_



where _n_ _i_ is the number of vertices whose distances to vertex
_v_ _i_ are smaller than those to the other vertex _v_ _j_ of the edge
_ε_ _ij_, and _n_ _j_ is defined analogously.
Spectrum-based descriptors are also used to describe the
molecular structures. Gutman and Zhou [25] defined the
Laplacian energy of the graph _G_ as



(4)
_N_ _[|][,]_



path between vertices _v_ _i_ and _v_ _j_ . A smaller _GE_ means that
all vertices of the graph are closer. Vulnerability efficiency
computes the average efficiency of the graph and measures the
importance of vertex _v_ _i_ on the system performance if vertex
_v_ _i_ and all its associated links are removed.


_C. QSAR/QSPR Model_


The QSAR/QSPR models in the literature are generally in
the form [29, 30]


_q_ = � _f_ _k_ (Ψ( _G_ )) + _c,_ (9)


_k_


where Ψ( _G_ ) are input variables chosen from the descriptor
functions of a compound discussed above, which are only
related to the structural information of the graph _G_, _f_ _k_ (Ψ( _G_ ))
are functions of Ψ( _G_ ), and _c_ is a constant. The scalar output _q_
of the model is usually biological activity or physicochemical
property of a compound. All compounds in the same chemical
family are subject to this model, and the input and output
variables of the model differ for different compounds.


_D. Performance Indices_


The performance of QSAR/QSPR model is assessed using
two groups of statistical indices, goodness-of-fit metrics and
goodness-of-prediction metrics [31]. Goodness-of-fit metrics
measure the fitting ability and are used to measure the degree
to which the model is able to explain the variance contained
in the training set. The three most important metrics are the
root mean square error ( _RMSE_ ), the average absolute error
( _AAE_ ) and the coefficient of determination ( _R_ [2] ). _RMSE_ and
_AAE_ are two frequently used metrics of the errors between
values predicted by a model and the values observed, which
are defined by



_LE_ ( _G_ ) =



_N_
�



� _|λ_ _i_ _−_ [2] _N_ _[m]_

_i_ =1



where _λ_ _i_ is the _i_ th eigenvalue of _L_ .
Dehmer et al. [26] defined some families of eigenvaluebased descriptors, one of them is stated as


1 1 1
_S_ _L,d_ ( _G_ ) = _|λ_ 1 _|_ _d_ + _|λ_ 2 _|_ _d_ + _· · ·_ + _|λ_ _N_ _|_ _d_ _,_ (5)


where _d_ is a real number.

The above-mentioned descriptors are global measures that
describe the overall network topological information. There
are some local measures for the molecular graph.
Brandes and Erlebach [27] proposed the stress centrality
and the betweenness centrality for each individual vertex of
the graph _G_, which are stated respectively as


_SC_ _i_ = � _n_ ( _j, i, k_ ) _,_ (6)

_jk_



1
_AAE_ =
_n_ _tr_



_RMSE_ =



~~�~~
~~�~~
�


_n_ _tr_

� [1]



_n_ _tr_


ˆ

�( _y_ _i train_ _−_ _y_ _i_ ) [2] _,_ (10)


_i_ =1



_n_ _tr_


ˆ

� _|y_ _i train_ _−_ _y_ _i_ _|,_ (11)


_i_ =1



where _n_ _tr_ is the number of compounds in the training set,
_y_ _i train_ and ˆ _y_ _i_ are the target value (experimentally observed)
and the corresponding predicted value in the training set,
respectively.
_R_ [2] is a statistic metric which is independent of the response
scale, contrary to _RMSE_ and _AAE_ . It provides a measure
of how well observed outcomes are replicated by the model,
based on the proportion of total variation of outcomes explained by the model. The most general definition of _R_ [2] is


_n_ _tr_
_R_ [2] = 1 _−_ � ~~�~~ _i_ ~~_n_~~ _i_ ==1 _tr_ 1 [(][(] _[y][y]_ _[i][i][ t][ t][rain][rain]_ _[ −][ −][y]_ [ˆ] _[y]_ [¯] _[i]_ [)][)] [2][2] _[,]_ (12)


where ¯ _y_ is the average observed value over the entire training

set.

Goodness-of-prediction metrics measure the generalization
ability of a model. In most cases, model validation by internal
cross-validation technique is not enough and validation by an



_BC_ _i_ = �

_jk_



_n_ ( _j, i, k_ )

(7)
_n_ ( _j, k_ ) _[,]_



where _n_ ( _j, i, k_ ) is the number of shortest paths between
vertices _v_ _j_ and _v_ _k_ that pass through vertex _v_ _i_ . _n_ ( _j, k_ ) is the
total number of shortest paths between vertices _v_ _j_ and _v_ _k_ .
Stress centrality is used to describe the number of weighted
paths that pass through the vertex _v_ _i_ . Betweenness centrality
is used to describe the ratio of paths that pass through the

vertex _v_ _i_ .
Costa et al. [28] introduced the vulnerability efficiency for
each individual vertex stated as

_V E_ _i_ = _[GE][ −]_ _GE_ _[GE]_ _[i]_ _,_ (8)


1
where _GE_ = _N_ ( _N_ _−_ 1) � _ε_ _ij_ _[d]_ _[ij]_ [ is the global efficiency,] _[ GE]_ _[i]_
is the global efficiency after the removal of the vertex _v_ _i_
and all its associated links, and _d_ _ij_ is the shortest weighted



0162-8828 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Auckland University of Technology. Downloaded on November 01,2020 at 22:09:08 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TPAMI.2020.3032718, IEEE

Transactions on Pattern Analysis and Machine Intelligence


4



external test set has been suggested as an effective way of evaluating the model predictive ability. The most important metrics
are the root mean square error in prediction ( _RMSEP_ ) and
the predictive squared correlation coefficient ( _R_ _P_ [2] [).] _[ RMSEP]_
denotes _RMSE_ on the test set and we use the following _R_ _P_ [2]
defined by Todeschini et al. [32]


_n_ _test_
� _i_ =1 [(] _[y]_ _[i][ t][est]_ _[ −]_ _[y]_ [ˆ] _[i]_ [)] [2] _[/][n]_ _[test]_
_R_ _P_ [2] [= 1] _[ −]_ ~~�~~ ~~_n_~~ _i_ =1 _tr_ [(] _[y]_ _[i][ t][rain]_ _[ −]_ _[y]_ [¯][)] [2] _[/n]_ _[tr]_ _._ (13)


As known from [31], the larger the _R_ [2] and the smaller the
_RMSE_ and _AAE_ are, the better the model performance is.


III. M ETHODS


_A. Problem Statement_


The degeneration of traditional molecular descriptors results
in the same values for different compounds with the same
molecular graph. This can lead to model failure, especially in
the following two cases. In the first case, the compounds have
the same graph structure and the same weight matrix. The only
difference is that their atoms at some vertices are different. If

the atomic information is not taken into account, not only
the properties of vertices and edges, but also the spectra of
the graphs are the same. In the second case, the compounds
have the same graph structure and the same heavy atoms, but
the types of chemical bonds are different, which means the
weight matrices are different. The properties of vertices are the
same, if the atomic information is not considered. Illustrative
examples for these two cases are shown in Fig. 2. Therefore,
descriptor degeneration and reliability and performance of
QSAR/QSPR model are the two key problems to be addressed
in this paper.


Fig. 2: Two cases that molecular descriptors degenerate.


To address these issues, we propose a novel GSP based
approach to QSAR/QSPR model learning. To present this
approach, we introduce the following notations.
For a molecular graph _G_ ( _V, E, W_ ) with _N_ vertices, a real
valued _M_ 1 -dimensional signal on the vertex _v_ _i_ is denoted
as **s** _i_ = [ _s_ _i_ 1 _, s_ _i_ 2 _, · · ·, s_ _iM_ 1 ] [T], and the signal matrix _S_ =

[ **s** 1 _, · · ·,_ **s** _i_ _, · · ·,_ **s** _N_ ] [T] _∈_ _R_ _[N]_ _[×][M]_ [1] collects all vertex signals
**s** _i_ ’s in its _N_ rows.



_B. Proposed Approach_

In order to solve the above stated problems, we design MD
signals on vertices and use them to derive new descriptors,
called MDS descriptors. Combining the new MDS descriptors
with the traditional descriptors, we propose a new modeling
method for the quantitative relationship between the structure
and the physicochemical/activity property of the compound.
The proposed method includes six steps:
Step 1: _MD signal construction._ Design of MD signal
is the first step of the proposed approach. Two basic requirements for the design include: i) be able to distinguish
different compounds and ii) can reflect important information relevant to the physicochemical/activity property of the
compound. To achieve these, we design the MD signal
**s** _i_ = [ _s_ _i_ 1 _, s_ _i_ 2 _, · · ·, s_ _iM_ 1 ] [T] for each vertex _v_ _i_ . Both the atomic
information of the compound, such as charge, molar mass
and chemical bond, and the local measures of the graph
can be used to construct the signal _s_ _ik_, _k_ = 1 _,_ 2 _, · · ·, M_ 1 .
High correlation of the designed signals with the compound
properties improves the model performance.
Step 2: _Input variable construction._ Let _S_ _k_ _∈_ _R_ _[N]_, _k_ =
1 _,_ 2 _, · · ·, M_ 1, be the _k_ th column of _S_ . Then we define
1
new MDS descriptors _ζ_ _k_ := _N_ _[||][S]_ _[k]_ _[||]_ _[p]_ [, with] _[ ∥][S]_ _[k]_ _[∥]_ _[p]_ [ the] _[ p]_ [-]
norm of _S_ _k_ and _p ≥_ 1. Next, we choose _M_ 2 traditional
descriptors, such as degree and degree-based indices, Laplacian energy and eigenvalue-based indices, and denote them
_η_ _k_ _, k_ = 1 _,_ 2 _, · · ·, M_ 2 .
To distinguish these two sets of descriptors, we denote
Φ( _S_ ) the set _{ζ_ 1 _, · · ·, ζ_ _k_ _, · · ·, ζ_ _M_ 1 _}_ to indicate its sole dependence on the MD signal _S_, and denote Ψ( _G_ ) the set
_{η_ 1 _, · · ·, η_ _k_ _, · · ·, η_ _M_ 2 _}_ to indicate its sole dependence on the
structural information of the graph _G_ .
Using these descriptors, we define the input variable vector


**x** =[ _x_ 1 _, · · ·, x_ _k_ _, · · ·, x_ _M_ ]:=[ _ζ_ 1 _, · · ·, ζ_ _M_ 1 _, η_ 1 _, · · ·, η_ _M_ 2 ] (14)


with _M_ = _M_ 1 + _M_ 2 . Let **x** _i_, _i_ = 1 _,_ 2 _, · · ·, B_, be the samples
of **x** from _B_ compounds. We construct the training data matrix


_X_ = [ **x** _[T]_ 1 _[,][ · · ·][,]_ **[ x]** _[T]_ _i_ _[,][ · · ·][,]_ **[ x]** _[T]_ _B_ []] _[T]_ _[ ∈]_ _[R]_ _[B][×][M]_ _[.]_ (15)


Step 3: _Input variable GSP_ . Inspired by the outstanding
work of [33], we treat the variables in **x** as the graph signals on
a descriptor graph learned from their sample data _X_, and filter
_X_ with the graph Laplacian filter derived from the descriptor
graph, using the following procedure.
Let _X_ _k_ _∈_ _R_ _[B]_, _k ∈M_ = _{_ 1 _,_ 2 _, · · ·, M_ _}_, be the _k_ th column
of _X_ . We first construct a distance matrix _E_ with elements
_E_ _i,j_ = _∥X_ _i_ _−X_ _j_ _∥_ 2 _, i, j ∈M_, and convert _E_ to an asymmetric
affinity matrix _A_ using an adaptive Gaussian kernel function
_A_ _i,j_ = _exp_ ( _−E_ _i,j_ _/σ_ _j_ ) [2], where _σ_ _j_ is the ( _knn_ +1)th smallest
_E_ _i,j_ in the _j_ th column of _E_ . Then we symmetrize _A_ to _A_ [˜] =
_A_ + _A_ [T] and column normalize _A_ [˜] to a Markov transition matrix
_K_, with _K_ _i,j_ = _A_ [˜] _i,j_ _/_ [�] _i_ _[A]_ [˜] _[i,j]_ [ and each column summing to]
1. We further construct a normalized Laplacian matrix of the
descriptor graph using _K_ and an _M × M_ identity matrix _I_ _M_


_L_ _d_ = _I_ _M_ _−_ _K._ (16)


The adjacency matrix _A_ [˜] defines a graph of **x** learned from
its sample data _X_ . The columns of Markov transition matrix



0162-8828 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Auckland University of Technology. Downloaded on November 01,2020 at 22:09:08 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TPAMI.2020.3032718, IEEE

Transactions on Pattern Analysis and Machine Intelligence


5



_K_ represent the probability distribution of transitioning from
a particular descriptor to every other descriptor in one step
along the column direction of the graph [33]. The Laplacian
matrix _L_ is a derivative operator and its _l_ th power _L_ _[l]_ _d_ [defines]
a high pass graph spectrum filter on the graph, with the order
_l ≥_ 1 [19]. Roughly speaking, the operation on **x** with _L_ _[l]_ _d_


**x** _L_ _[l]_ _d_ [:= ˜] **[x]** [ = [˜] _[x]_ [1] _[,][ · · ·][,]_ [ ˜] _[x]_ _[k]_ _[,][ · · ·][,]_ [ ˜] _[x]_ _[M]_ []] (17)


yields ˜ _x_ _k_ ’s with enhanced dissimilarity on the graph. A
demonstrative example is given in Fig 5 in Section IV-B. In
principle, the higher the order _l_, the stronger the enhancement.
In practice, however, the order _l_ should be chosen such that
the _|x_ ˜ _k_ _|_ ’s are not annihilated by filtering. Based on this fact,
we filter the training data _X_ by _L_ _[l]_ _d_ [to obtain the dissimilarity]
enhanced training data


_XL_ _[l]_ _d_ [:= ˜] _[X]_ [ = [ ˜] _[X]_ [1] _[,][ · · ·][,]_ [ ˜] _[X]_ _[k]_ _[,][ · · ·][,]_ [ ˜] _[X]_ _[M]_ []] _[T]_ _[ .]_ (18)


Step 4: _Modeling_ . Using the Φ( _S_ ) and Ψ( _G_ ) variables
defined in step 2, we propose a new QSAR/QSPR model


_q_ = � _f_ _k_ (Φ( _S_ ) _,_ Ψ( _G_ )) _._ (19)


_k_


Different to the traditional QSAR/QSPR model (9), the new
QSAR/QSPR model (19) uses input variables from both Φ( _S_ )
and Ψ( _G_ ). Thus it uses not only the structural information of
graph _G_ carried by Ψ( _G_ ), but also the information from MD
signal _S_ carried by Φ( _S_ ). We decompose the property of a
molecule as a sum of local contributions, and represent each
local atomic environment by MDS descriptors derived from
MD signal _S_ that are inherently invariant. This may result
in better models and is one of the main contributions of this

work.

In this work, we focus on two special cases of the
QSAR/QSPR model (19).
_Model 1_ : _f_ _k_ (Φ( _S_ ) _,_ Ψ( _G_ )) = _θ_ _k_ _ζ_ _k_, _k_ = 1 _,_ 2 _, · · ·, M_ 1, and
_f_ _k_ + _M_ 1 (Φ( _S_ ) _,_ Ψ( _G_ )) = _θ_ _k_ + _M_ 1 _η_ _k_, _k_ = 1 _,_ 2 _, · · ·, M_ 2, with _ζ_ _k_
and _η_ _k_ the MDS and traditional descriptors defined in Step 2
and (14) and _θ_ _k_ their coefficients, that is,



_M_ 2
� _θ_ _M_ 1 + _k_ _η_ _k_ + _c_ = [ **x** 1] _θ._ (20)


_k_ =1



_q_ =



_M_ 1
� _θ_ _k_ _ζ_ _k_ +


_k_ =1



_Model 2_ : _f_ _k_ (Φ( _S_ ) _,_ Φ( _S_ )) = _θ_ _k_ _x_ ˜ _k_, _k_ = 1 _,_ 2 _, · · ·, M_ 1,
and _f_ _k_ + _M_ 1 (Φ( _S_ ) _,_ Ψ( _G_ )) = _θ_ _k_ + _M_ 1 ˜ _x_ _k_ + _M_ 1, _k_ = 1 _,_ 2 _, · · ·, M_ 2,
with ˜ _x_ _k_ defined in (17) and _θ_ _k_ their coefficients, that is,



_M_ 2
� _β_ _k_ _x_ ˜ _k_ + _M_ 1 + _c_ = [˜ **x** 1] _θ._ (21)


_k_ =1



for (21), with **1** _∈_ _R_ _[B]_ an all 1 vector. Then we can write
the following matrix form regression equation for learning the
models (20) and (21)

_Q_ = _Y θ._ (23)


When many descriptors are used as input variables in the
model, that is, when _M_ is large, some of them may be
interdependent on each other or may have mere correlation
with the output variable _q_ . To single out and eliminate these
variables and find the optimal coefficients for the reserved
variables, we use the Least Angle Regression with Lasso
modification [34] to find an optimal sparse _θ_ with the leastsquares of prediction error and minimum number of nonzero
coefficients. The computation procedure is summarized in
Algorithm 1, where _Q_ [¯] denotes the model prediction of _Q_ and
_Y_ _j_ denotes the _j_ th column of _Y_ in (23).
Step 6: _Performance evaluation_ . The performance of our
model is assessed using the metrics defined in Section II-D.
_RMSE_, _AAE_ and _R_ [2] are used to measure the fitting ability
of the model; and _RMSEP_ and _R_ _P_ [2] [are used to measure the]
generalization ability of the model. Low errors between predicted and experimental values indicate a good QSAR/QSPR
model, whereas high errors indicate a poor one.
The pseudo-code of the proposed method is given in Algorithm 2.


**Algorithm 1** Least Angle Regression with Lasso [34]


1: Standardize the variables to have mean zero and unit norm.
Start with the residual _r_ = _Q−Q_ [¯] and _θ_ 1 _, θ_ 2 _, · · ·, θ_ _M_ = 0.

2: Find the variable _Y_ _j_ most correlated with _r_ .
3: Move _θ_ _j_ from 0 towards its least-squares coefficient
_⟨Y_ _j_ _, r⟩_, until some other competitor _Y_ _k_ has as much
correlation with the current residual as does _Y_ _j_ .
4: Move _θ_ _j_ and _θ_ _k_ in the direction defined by their joint least
squares coefficient of the current residual on _⟨Y_ _j_ _, Y_ _k_ _⟩_, until
some other competitor _Y_ _l_ has as much correlation with the
current residual.

5: If a nonzero coefficient hits zero, drop its variable from
the active set of variables and recompute the current joint
least squares direction.
6: Continue in this way until all _M_ variables and the constant
variable **1** have been entered. After _min_ ( _B −_ 1 _, M_ ) steps,
we arrive at the full least-squares solution and minimum
number of nonzero _θ_ _i_ ’s.


IV. A PPLICATION IN P HENETYLAMINES


In order to evaluate the predictive ability and better understand the proposed models (23), we apply them to two benchmark datasets recommended by the International Academy of
Mathematical Chemistry [1] for verification and comparison of
new descriptors.
The first benchmark dataset, Phenet for short, is constituted
by 22 phenetylamines with two substituent sites and its property of biological activity is given. A set of 110 molecular


1 http://www.moleculardescriptors.eu/dataset/dataset.htm



_q_ =



_M_ 1
� _θ_ _k_ _x_ ˜ _k_ +


_k_ =1



In (20) and (21), _M_ 1 + _M_ 2 = _M_ and


_θ_ := [ _θ_ 1 _, · · ·, θ_ _M_ 1 _, θ_ _M_ 1 +1 _, · · ·, θ_ _M_ _, c_ ] _[T]_ (22)


is the coefficient vector of the models, and **x** and ˜ **x** are the
regression variables defined in (14) and (17), respectively.
Step 5: _Sparse_ _coefficient_ _learning._ Define _Q_ :=

[ _q_ 1 _, · · ·, q_ _i_ _, · · ·, q_ _B_ ] _[T]_ consisting of the output variables of _B_
compounds. Let _X_ in the form (15) and _X_ [˜] in the form (18) be
the corresponding data matrix and filtered data matrix of these
compounds. Define _Y_ := [ _X_ **1** ] for (20) and _Y_ := [ _X_ [˜] **1** ]



0162-8828 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Auckland University of Technology. Downloaded on November 01,2020 at 22:09:08 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TPAMI.2020.3032718, IEEE

Transactions on Pattern Analysis and Machine Intelligence


6



**Algorithm 2** Pseudo-code of the proposed method


**Require:** Atomic information of compound, such as molar
mass and charge; Global and local descriptors of the
molecular graph, such as _SC_ _i_, _BC_ _i_ .

1: Use atomic information and local descriptors to construct
the graph signal _S ∈_ R _[N]_ _[×][M]_ [1] .

1
2: Compute MDS descriptors _ζ_ _k_ = _N_ _[||][S]_ _[k]_ _[||]_ _[p]_ [,] _[ k]_ [ = 1] _[,]_ [ 2] _[,]_

_· · ·, M_ 1 .

3: Choose _M_ 2 global descriptors as _η_ _j_, _j_ = 1 _,_ 2 _, · · ·, M_ 2 .

4: Define **x** = [ _ζ_ 1 _, · · ·, ζ_ _M_ 1 _, η_ 1 _, · · ·, η_ _M_ 2 ] _∈_ _R_ [1] _[×][M]_, and
use _B_ samples of **x** to form training data matrix _X_ =

[ **x** _[T]_ 1 _[,]_ **[ x]** 2 _[T]_ _[,][ · · ·][,]_ **[ x]** _[T]_ _B_ []] _[T]_ _[ ∈]_ _[R]_ _[B][×][M]_ [.]

5: For Model 1, let _Y_ = [ _X_ **1** ] then go to 11, otherwise
continue.


6: Use the _i_ th and _j_ th columns of _X_ to calculate the distance
matrix _E_ _i,j_ = _∥_ _X_ _i_ [T] _[−]_ _[X]_ _j_ [T] _[∥]_ [2] _[, i, j][ ∈{]_ [1] _[,]_ [ 2] _[,][ · · ·][, M]_ _[}]_ [.]

7: Compute the affinity matrix _A_ using adaptive Gaussian
kernel _A_ _i,j_ = _exp_ ( _−E_ _i,j_ _/σ_ _j_ ) [2], _σ_ _j_ = the ( _knn_ + 1)th
smallest _E_ _i,j_ in the _j_ th column of _E_ .

8: Symmetrize _A_ to _A_ [˜] = _A_ + _A_ _[T]_ and column normalize _A_ [˜] to
Markov transition matrix _K_ with _K_ _i,j_ = _A_ [˜] _i,j_ _/_ [�] _k_ _[A]_ [˜] _[i,k]_ [.]

9: Construct the Laplacian matrix _L_ _d_ = _I_ _M_ _−_ _K_ .
10: Filter the training data matrix _X_ to obtain _X_ [˜] = _XL_ _[l]_ _d_ [and]
let _Y_ = [ _X_ [˜] **1** ].
11: Use Algorithm 1 to find the optimal sparse _θ_ .

12: Assess _RMSE_, _AAE_ and _R_ [2] of the model.


descriptors is also given, which are calculated by DRAGON
software (version 5.4).
Phenet dataset contains data on the adrenergic blocking
potencies of N, N-dimethyl-2-bromo-phenetylamines in the
rat with varying structures as illustrated in Fig. 3(a). The Z
and Y, substituents on the ring, are H, F, Cl, Br, I, or CH 3 .
Compounds show different biological activities according to
the different combinations of substituents. The same graph _G_
for this dataset is shown in Fig. 3(b), in which the two gray
vertices are the substituents Z and Y.


(a) Varying structures (b) The same graph _G_


Fig. 3: Phenet dataset.


First, we design the MD signal **s** _i_ = [ _s_ _i_ 1 _, s_ _i_ 2 _, · · ·, s_ _iM_ 1 ] [T],
with _s_ _i_ 1 = betweenness centrality ( _BC_ _i_ ), _s_ _i_ 2 = stress centrality ( _SC_ _i_ ), _s_ _i_ 3 = vulnerability efficiency ( _V E_ _i_ ), _s_ _i_ 4 =
atomic mass and _s_ _i_ 5 = atomic charge. This yields a 5dimensional vector signal **s** _i_ for each vertex _v_ _i_ and the _i_ th
row of the signal matrix _S_ described in Step 2. Using _S_ _k_,
the _k_ th column of _S_, in _ζ_ _k_ = _N_ 1 _[∥][S]_ _[k]_ _[∥]_ _[p]_ [, we obtain five MDS]
descriptors _ζ_ _k_ _, k_ = 1 _,_ 2 _, · · ·,_ 5. Then, we use two spectrumbased descriptors, _LE_ and _S_ _L,_ 2, and the 110 traditional
topological descriptors given in the benchmark dataset Phenet



4
� _θ_ _k_ _x_ _k_


_k_ =1

~~�~~ ~~�~~ � ~~�~~
MDS descriptors



to obtain _η_ _k_ _, k_ = 1 _,_ 2 _, · · ·,_ 112.
The descriptors _ζ_ _k_ and _η_ _k_ thus obtained constitute the input
variable vector **x** _∈_ _R_ _[M]_ defined in (14), with _M_ = 117,
_x_ _k_ = _ζ_ _k_ for _k_ = 1 _,_ 2 _, · · ·,_ 5, and _x_ _k_ +5 = _η_ _k_ = for _k_ =
1 _,_ 2 _, · · ·,_ 112. Filtering **x** with _L_ _[l]_ _d_ [gives the][ ˜] **[x]** _[ ∈]_ _[R]_ [117] [ defined]
in (17). Using the **x** and ˜ **x** respectively in (20) and (21) gives
the following two models that relate the biological activity
( _BA_ ) to the structure of the compound represented by these
variables.



_BA_ =



5
� _θ_ _k_ _x_ _k_


_k_ =1



+



112
� _θ_ _k_ +5 _x_ _k_ +5


_k_ =1



+ _c._
(24)



~~�~~ � ~~�~~ �
MDS descriptors



� ~~�~~ � ~~�~~
traditional topological descriptors



_BA_ =



117
� _θ_ _k_ _x_ ˜ _k_ + _c._ (25)


_k_ =1



The model (24) has two parts: MDS descriptors and traditional topological descriptors, and the model (25) contains the
graph filtered these two parts in each ˜ _x_ _k_ (see (18) for details).
Thus, their performance is expected to be better than that of
the model with only traditional topological descriptors in the
second part of (24).


V. A PPLICATION IN P OLYAROMATIC H YDROCARBONS

The second benchmark dataset, PAH for short, are generally
highly toxic and carcinogenic compounds and ubiquitous contaminants of aquatic and atmospheric ecosystems. PAH dataset
is constituted by 82 polyaromatic hydrocarbons. The properties
of melting point, boiling point and octanol-water partition
coefficient are given. A set of 112 molecular descriptors is also
given, which are calculated by DRAGON software (version
5.4).
The number of fused benzene rings contained in the compounds in this dataset is from 2 to 11, and some compounds
also have substituents. Examples are shown in Fig. 4.
Since PAH dataset does not provide information about the
atomic charge, we use _s_ _i_ 1 = betweenness centrality ( _BC_ _i_ ),
_s_ _i_ 2 = stress centrality ( _SC_ _i_ ), _s_ _i_ 3 = vulnerability efficiency
( _V E_ _i_ ) and _s_ _i_ 4 = atomic mass to form a 4-dimensional signal
**s** _i_ for each vertex _v_ _i_ and the _i_ th row of _S_ . Following the
same procedure as that used for Phenetylamines, we use _ζ_ _k_ =
1
_N_ _[∥][S]_ _[k]_ _[∥]_ _[p]_ [ to obtain four MDS descriptors] _[ ζ]_ _[k]_ _[, k]_ [ = 1] _[,]_ [ 2] _[,]_ [ 3] _[,]_ [ 4][; and]
use two spectrum-based descriptors, _LE_ and _S_ _L,_ 2, and the
112 traditional topological descriptors given in the benchmark
dataset PAH to obtain _η_ _k_ _, k_ = 1 _,_ 2 _, · · ·,_ 114. Letting _x_ _k_ = _ζ_ _k_
for _k_ = 1 _,_ 2 _,_ 3 _,_ 4, and _x_ _k_ +4 = _η_ _k_ for _k_ = 1 _,_ 2 _, · · ·,_ 114, we
obtain the input variable vector **x** _∈_ _R_ [118] and the _L_ _[l]_ _d_ [filtered]
variable vector ˜ **x** _∈_ _R_ [118] .
Using the variables obtained above, we construct two models in (26) and (27) that relate the boiling point ( _BP_ ) to the
structure of the compound represented by these variables. The
same as (24), (26) consists of the MDS descriptor part and
the traditional descriptor part, while each ˜ _x_ _k_ in (27) contains
graph filtered these two parts.



~~�~~ ~~�~~ � ~~�~~
traditional topological descriptors



_BP_ =



+



114
� _θ_ _k_ +4 _x_ _k_ +4

_j_ =1



+ _c._ (26)



0162-8828 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Auckland University of Technology. Downloaded on November 01,2020 at 22:09:08 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TPAMI.2020.3032718, IEEE

Transactions on Pattern Analysis and Machine Intelligence


7


Fig. 4: Example compounds of PAH dataset.



_BP_ =



118
� _θ_ _k_ _x_ ˜ _k_ + _c._ (27)


_k_ =1



VI. R ESULTS


In this section, we first present the uniqueness of molecular
descriptors derived from the MD signal on compounds of
different families with the same graph structure. Then, we
evaluate our method by numerical experiments. The models
(24)-(27) presented in Sections IV and V are learned from
training data to study their behavior, and the performance
indices introduced in Section II-D are used to evaluate their

performance. All the computations are performed with Matlab.


_A. Differentiation of Ethane/Ethylene/Acetylene_


This subsection demonstrates the uniqueness of MDS descriptors on the compounds in different families with the same
graph structure, which is the second case shown in Fig. 2.
The only difference in molecular graphs of these compounds
is their different weights. Different types of chemical bonds
make them fall into different categories, so we just need to
design non-degenerative molecular descriptors to distinguish
them.

Ethane, ethylene and acetylene, as compounds with different
chemical bonds containing two carbon atoms, are analyzed.
Traditional topological descriptors always degenerate into the
same values for these three compounds. For any real number _d_,
the general Randic indices for ethane, ethylene and acetylene
will always be the same value _R_ _C_ 2 _H_ 6 = _R_ _C_ 2 _H_ 4 = _R_ _C_ 2 _H_ 2 =
1. For any real number _d_, the general sum connectivity indices
always have the same value _χ_ _C_ 2 _H_ 6 = _χ_ _C_ 2 _H_ 4 = _χ_ _C_ 2 _H_ 2 = 2 _[d]_ .
In our work, we design MDS descriptors based on GSP. A
simple 2-dimensional signal based on eigenvalue and atomic
mass is designed. The signal matrices for ethane, ethylene and
acetylene are respectively



0 15
_S_ _C_ 2 _H_ 6 = 2 15
�



0 14
_, S_ _C_ 2 _H_ 4 = 4 14
� �



0 13
_, S_ _C_ 2 _H_ 2 = 6 13
� �



_._
�



When the 1-norm of each column is used to calculate
_ζ_ := [ _ζ_ 1 _, ζ_ 2 ] for the above three matrices, we get _ζ_ _C_ 2 _H_ 6 =

[2 _,_ 30], _ζ_ _C_ 2 _H_ 4 = [4 _,_ 28] and _ζ_ _C_ 2 _H_ 2 = [6 _,_ 26], respectively.
Similarly, when the 2-norm of each column is used, we
get _ζ_ _C_ 2 _H_ 6 = [2 _,_ 21 _._ 21], _ζ_ _C_ 2 _H_ 4 = [4 _,_ 19 _._ 80] and _ζ_ _C_ 2 _H_ 2 =

[6 _,_ 18 _._ 38], respectively. By defining a simple 2-dimensional
signal on each vertex, these three compounds are easily distinguished. The design of the MD signal based on the important
information (structural information, spatial information, etc.)
of the compound guarantees the uniqueness of the new MDS
descriptors.
This method of constructing molecular descriptors by means
of GSP is also effective for other molecular structures with

different chemical bonds and the same atomic type. Highdimensional signals can carry more information and are more
suitable for complex compounds.


_B. Phenetylamine Models_


Phenet dataset is used as training dataset to learn the
models (24) and (25). The _BA_ values of the compounds in
the dataset are used to form the output data vector _Q_ . The
1
MDS descriptors _x_ _k_ = _ζ_ _k_ = _N_ _[||][S]_ _[k]_ _[||]_ [1] _[, k]_ [ = 1] _[,][ · · ·][,]_ [ 5][, are]
calculated for each compound in the dataset, and the values
of 112 traditional descriptors of these compounds are used as
_x_ _k_ +5 = _η_ _k_ _, k_ = 1 _,_ 2 _, · · ·,_ 112. The _x_ _k_ ’s thus obtained for all
the compounds are used to construct the data matrices _X_ as in
(15), and to calculate _X_ [˜] according to Steps 6-10 of Algorithm
2, with empirically determined _knn_ = 1 and _l_ = 1 in Steps 7
and 10, respectively. To compare with the traditional models
using only topological descriptors, we also construct a data
matrix _X_ _d_ consisting of the last 112 columns of _X_, i.e., the
values of 112 traditional descriptors of all the compounds.
Applying Algorithm 1 to the model output-input data pairs
_Q_ - _X_ _d_, _Q_ - _X_ and _Q_ - _X_ [˜], respectively, we obtain the following
three models, where most of the 117 input variables have
been found insignificant and hence eliminated by Algorithm



0162-8828 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Auckland University of Technology. Downloaded on November 01,2020 at 22:09:08 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TPAMI.2020.3032718, IEEE

Transactions on Pattern Analysis and Machine Intelligence


8



1, resulting in very sparse optimal coefficient vectors _θ_ ’s with
dimension _≤_ 5.


_BA_ = 0 _._ 253 _S_ _L,_ 2 + 3 _._ 704 _WA −_ 1 _._ 269 _Dz −_ 1 _._ 772 _,_ (28)


_BA_ = 6 _._ 078 _ζ_ 3 + 3 _._ 680 _WA −_ 1 _._ 259 _Dz −_ 7 _._ 586 _,_ (29)


_BA_ =41 _._ 130˜ _x_ 1 +16 _._ 121˜ _x_ 4 +2 _._ 817˜ _x_ 5 +8 _._ 157˜ _x_ 7 +0 _._ 963 _._ (30)


The model (28) uses only traditional descriptor variables
( _S_ _L,_ 2 _, WA, Dz_ ), the model (29) uses both traditional descriptor ( _WA, Dz_ ) and new MDS descriptor ( _ζ_ 3 ) variables, while
the model (30) uses graph filtered both types of descriptor
variables. These input variables are singled out by Algorithm
1 as the significant ones among the 112 traditional descriptors
in _X_ _d_, the 117 variables in _X_, and the 117 variables in _X_ [˜],
respectively.


TABLE I: Performance comparison of models for Phenet
dataset









|Col1|model<br>(28)|model<br>(29)|model<br>(30)|
|---|---|---|---|
|_R_2|0.955|0.960|0.966|
|_RMSE_|0.118|0.112|0.102|
|_AAE_|0.105|0.101|0.088|


Table I compares the performance indices _R_ [2], _RMSE_ and
_AAE_ of the above three models. It is clear from the table that

the models (29) and (30), using both MDS and traditional
descriptors as inputs, outperform the model (28) using only
traditional descriptor inputs, and the model (30) using the
graph ( _L_ _[l]_ _d_ [) filtered input variables][ ˜] _[x]_ _[k]_ [ performs best.]
The better performance of model (30) stems from the
enhanced dissimilarity of the graph filtered input variables
_x_ ˜ _k_ as compared with the original input variables _x_ _k_ . This is
shown in Fig. 5 using the first eight elements of the input
vector **x** for a compound in Phenet dataset. It can be seen that
the correlation of input variables is significantly reduced after
graph filtering.


Fig. 5: Correlation coefficients of the first eight input
variables for a compound in Phenet dataset before (left) and
after (right) graph filtering.


For the model (30), the absolute errors are plotted in Fig.
6 against the experimental biological activity. Its _AAE_ is
0.088 with the minimum absolute error 0.01 and the maximum

absolute error 0.23. The minimum and maximum values of



the corresponding relative error are 0.11% and 2.72%, respectively. The range of biological activity of these compounds is
7.56 _∼_ 9.52. When this model is used for clustering, that is,
dividing Phenet dataset into two categories on average based
on the median 8.86, all compounds can be classified correctly.
This further verifies the validity of the model.


Fig. 6: Absolute errors of the model (30) of Phenet dataset.


Each dimension of the MD signal _S_ reflects a certain
character (chemical/structure) of the compound, with different
compounds having different MD signals. Therefore, when very
few variables are used to distinguishing or modeling different
compounds, the proposed method still performs well.
Table II compares the QSAR models of the Phenet dataset
reported in [35–38] with the models (29) and (30), where
PCA/NN is a neural network model that uses principle
components derived from the full data set; MLR-EM is a
multiple linear regression model employing an expectation
maximization method; EM/NN is a neural network model
using sparse descriptors derived from the MLR-EM algorithm.
The comparison reveals the superiority of the models (29) and
(30) in predicting the biological activity of phenetylamines. As
seen from the table, the proposed models (29) and (30) outperform previously reported models. Using fewer descriptors, the
proposed models still outperform the neural network model
using more descriptors.


TABLE II: Comparison of the proposed models with other
works for Phenet dataset








|Col1|Model|R2|RMSE|No. of de-<br>scriptors|
|---|---|---|---|---|
|Vukicevic et al. [35]|MLR|0.5405|_−_|1|
|Burden et al. [36]|MLR-EM<br>_χ_=0.035|0.750|0.265|3|
|Burden et al. [36]|PCA/NN 1<br>node|0.936|0.130|10|
|Burden et al. [36]|EM/NN 3<br>nodes|0.959|0.260|7|
|Unger and Hansch [37]|MLR|0.964|0.164|3|
|Unger and Hansch [37]|MLR|0.944|0.197|2|
|Todeschini et al. [38]|MLR|0.845|0.177|3|
|**This work**|Model (29)|0.960|0.112|3|
|**This work**|Model (30)|0.966|0.102|4|



0162-8828 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Auckland University of Technology. Downloaded on November 01,2020 at 22:09:08 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TPAMI.2020.3032718, IEEE

Transactions on Pattern Analysis and Machine Intelligence


9



_C. Polyaromatic Hydrocarbon Models_


PAH dataset is used as training dataset to learn the models
(26) and (27). The _BP_ values of the compounds in the dataset
are used to form the output data vector _Q_ . The _x_ _k_ = _ζ_ _k_ =
1
_N_ _[||][S]_ _[k]_ _[||]_ [1] _[, k]_ [ = 1] _[,][ · · ·][,]_ [ 4][, are calculated for each compound in]
the dataset, and the values of 114 traditional descriptors of
these compounds are used as _x_ _k_ +5 = _η_ _k_ _, k_ = 1 _,_ 2 _, · · ·,_ 112.
The _x_ _k_ ’s thus obtained for all the compounds are used to
construct the data matrices _X_ ˜ according to Steps 6-10 of Algorithm 2, with empirically _X_ as in (15), and to calculate
determined _knn_ = 3 and _l_ = 2 in Steps 7 and 10, respectively.
To compare with the traditional models using only topological
descriptors, we also construct a data matrix _X_ _d_ consisting of
the last 114 columns of _X_ .

                                      Applying Algorithm 1 to the output-input data pairs _Q_
_X_ _d_, _Q_ - _X_ and _Q_ - _X_ [˜], respectively, we obtain the following
three models. Again, most of the 118 input variables have
been found insignificant and hence eliminated by Algorithm
1, resulting in very sparse optimal coefficient vectors _θ_ ’s with
dimension _≤_ 6.


_BP_ = 0 _._ 842 _ZM_ 1 _V_ + 0 _._ 150 _LE_ + 0 _._ 023 _,_ (31)


_BP_ = 0 _._ 901 _ζ_ 4 + 0 _._ 946 _S_ _L,_ 2 _−_ 0 _._ 799 _,_ (32)


_BP_ = 10 _._ 299˜ _x_ 1 _−_ 19 _._ 172˜ _x_ 3 _−_ 17 _._ 303˜ _x_ 4 +
(33)
4 _._ 737˜ _x_ 7 + 3 _._ 763˜ _x_ 8 + 1 _._ 029 _._


Of these models, (31) is a model with only traditional
descriptor variables ( _ZM_ 1 _V, LE_ ); (32) is a model with both
traditional descriptor ( _S_ _L,_ 2 ) and new MDS descriptor ( _ζ_ 3 )
variables; and (33) is a model with graph filtered both types of
variables. These input variables are singled out by Algorithm
1 as the significant ones among the 114 traditional descriptors
in _X_ _d_, the 118 variables in _X_, and the 118 variables in _X_ [˜],
respectively.
Table III compares the performance indices _R_ [2], _RMSE_ and
_AAE_ of the above three models. Similar to Phenetylamines
case, the models (32) and (33), using both new MDS and traditional descriptors as inputs, outperform the model (31) using
only traditional descriptor inputs; and the model (33) using
graph ( _L_ _[l]_ _d_ [) filtered input variables][ ˜] _[x]_ _[k]_ [ performs best, showing]
again the advantage of graph filtering of input variables in
performance enhancement.


TABLE III: Performance comparison of models for PAH
dataset









|Col1|model<br>(31)|model<br>(32)|model<br>(33)|
|---|---|---|---|
|_R_2|0.981|0.982|0.985|
|_RMSE_|11.198|11.177|10.196|
|_AAE_|7.766|7.044|6.774|


Fig. 7 plots the absolute errors of the model (33) against the
experimental boiling point. The wide range of boiling point of
this dataset, 178 _∼_ 519, results in relatively large errors. This
model shows an excellent performance and the average value
of the absolute relative error is merely 2.08%. The absolute
errors of 1-phenylnaphthalene ( _BP_ = 334) and azulene ( _BP_



= 270) are greater than 35. The absolute errors of and 27-dimethylpyrene ( _BP_ = 396) and acenaphthylene ( _BP_ =
270) are greater than 20. With the exception of these four
compounds, the absolute errors of the others are less than 15
and the corresponding _AAE_ drops to 4.34. Considering the
large value of boiling point, this model is good for fitting.


Fig. 7: Absolute errors of the model (33) of PAH dataset.


The two input variables in model (32) have clear chemistry
meaning. The traditional eigenvalue-based descriptor _S_ _L,_ 2
describes the information of molecular graph spectrum of the
compound, while the MDS descriptor _ζ_ 4 is actually the average
atomic mass representing the information of molecular mass.
Fig. 8 plots the 3D surfaces of boiling point, _S_ _L,_ 2 and _ζ_ 4 of the
compounds in PAH dataset. The apparent linear dependence
of _BP_ on _S_ _L,_ 2 and _ζ_ 4 shown in the plot is consistent with the
linear model (32) we have obtained, and asserts the validity of
the model. From this model, we may conclude that the boiling
point of a compound can be predicted quite precisely with the
graph spectral information _S_ _L,_ 2 and the molecular mass _ζ_ 4 .


Fig. 8: Relationship of variables and boiling point for model
(32) of PAH dataset.


Table IV compares the QSPR models of polyaromatic
hydrocarbons reported in [35, 39, 40] with the models (32) and
(33). In [39], a set of 67 non-substituted PAHs containing 2 to
7 fused rings with five and six carbon atoms were studied. The
23 non-substituted PAHs studied in [40] contain 1 to 6 fused
rings. Whereas, the dataset used in this paper is more complex,
with the number of fused rings ranging from 2 to 11 and some
PAHs also having substituents. This makes QSPR modeling
harder. As seen from the table, the prediction performance
_R_ [2] of the models (32) and (33) on this harder dataset is



0162-8828 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Auckland University of Technology. Downloaded on November 01,2020 at 22:09:08 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TPAMI.2020.3032718, IEEE

Transactions on Pattern Analysis and Machine Intelligence


10



comparable to that of [35, 39, 40] on those easier datasets.
For compounds in the same families with the same or similar
graph structure, the models (32) and (33) not only solve the
problem of degeneration of molecular descriptors, but also can
predict properties of compounds efficiently.


TABLE IV: Comparison of the proposed models with other
works for polyaromatic hydrocarbons











|Col1|Model|R2|No. of<br>descrip-<br>tors|No. of<br>training<br>set|
|---|---|---|---|---|
|Vukicevic et al.<br>[35]|MLR|0.980|1|53|
|Ribeiro et al. [39]|PCR|0.995|3|36|
|Ribeiro et al. [39]|PLS|0.995|2|36|
|Ferreira [40]|PLS|0.998|3|23|
|**This work**|Model (32)|0.982|2|53|
|**This work**|Model (33)|0.985|5|53|


VII. D ISCUSSIONS


A challenging issue in QSAR/QSPR research is that its
performance depends mainly on the quality of molecular
descriptors used. Based only on the properties of vertices and
edges, traditional molecular descriptors always degenerate to
the same values for different compounds having the same or
similar molecular graph. Therefore it is difficult to obtain reliable and effective QSAR/QSPR models using these traditional
descriptors.
To solve the problem of descriptor degeneration, we have
used the GSP’s foundation concept of signals on graph to
design an MD signal with distinctive information of compound
for each vertex of molecular graph. By GSP analysis of the
MD signals on vertices, we have derived new MDS descriptors
that can better distinguish the compounds with the same or
similar molecular graph structures.
To solve the problem in model reliability and performance,
we have combined new MDS descriptors with traditional
molecular descriptors in linear regression to derive the new
QSAR/QSPR model (20) with enhanced reliability and performance. To further enhance model reliability and performance,
we have introduced the descriptor graph to derive the Laplacian graph filter, and used the highpass nature of the filter to
enhance the dissimilarity of descriptors. Using the dissimilarity
enhanced descriptors in linear regression, we have obtained the
QSAR/QSPR model (21) with further enhanced performance
and reliability. The model (20) with descriptors as regression
variables is useful in revealing the QSAR/QSPR between a
compound and its descriptors, as shown in Section VI-C; while
the model (21), with descriptors hidden in the filtered input
variables, is useful for reliable prediction of the biological
activity or physicochemical property of compounds in practice.
Existing works, whether in descriptor design or in modeling,
are based solely on the molecular graph. Our GSP-based
approach, as summarized above, is very different from those
of previous works. This novel approach has resulted in significant performance and reliability improvement in QSAR/QSPR



modeling as shown in two application examples in Section IV.
To the best of our knowledge, this work is the first attempt
to study and solve QSAR/QSPR modeling problem from the
perspective of GSP. The graph filtering of input variables used
in our approach might be useful for other regression based
model learning problems.
This work intends to draw the attention of researchers in

GSP and machine learning communities for further research
on GSP based QSAR/QSPR model learning. In this work,
GSP has been applied in the modeling of biological activity
and boiling point. Other information can be included in the
proposed method for analyzing more complex properties, such
as binding affinity, lethal dose, and octanol/water partition
coefficient. Learning approaches, such as GNN, might be used
to establish nonlinear models and explore deeper relationships
between the molecular structure and these complex properties.


VIII. C ONCLUSIONS


A new paradigm for QSAR/QSPR modeling based on GSP
has been proposed to learn the quantitative relationship between the physicochemical/activity property and the structure
of compounds.
An MD signal has been introduced for each vertex of the
molecular graph. By analyzing the MD signals, a number of
new (MDS) molecular descriptors with higher discriminability
have been designed. Combination of these new descriptors
and traditional descriptors as the inputs in linear regression
has resulted in new QSAR/QSPR models with enhanced
performance.
Treating the combined descriptor variables as the signals
on a descriptor graph, a novel approach has been presented
to derive the descriptor graph and its Laplacian graph filter
from the descriptor data. The Laplacian filtered input variables
with enhanced dissimilarity have been used as inputs in linear
regressions to derive new QSAR/QSPR models with further
enhanced performance.
For datasets containing the compounds with the same or
similar graph structure, our proposed models have shown
better performance. We have also provided a new insight from
chemistry into the boiling point model of compounds.
The results of this paper have provided deeper understanding and new approaches for compound prediction and
classification, with application potential in various areas such
as biochemical and pharmaceutical engineering. These results
have also shed some new lights on regression based learning.


A CKNOWLEDGMENT


This work was supported by the National Natural Science Foundation of China (grants 61625305, 61801338 and
61801339).


R EFERENCES


[1] S. M. Hosamani, B. B. Kulkarni, R. G. Boli, et al.,
“QSPR analysis of certain graph theocratical matrices
and their corresponding energy,” _Applied Mathematics and_
_Nonlinear Sciences_, vol. 2, no. 1, pp. 131-150, 2017.



0162-8828 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Auckland University of Technology. Downloaded on November 01,2020 at 22:09:08 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TPAMI.2020.3032718, IEEE

Transactions on Pattern Analysis and Machine Intelligence


11




[2] M. Dehmer, F. Emmert-Streib, and Y. Shi, “Quantitative graph theory,” _Information Sciences: an International_
_Journal_, vol. 418(C), pp. 575-580, 2017.

[3] O. B. Ghanem, M. A. Mutalib, J. M. Leveque, et al.,
“Development of QSAR model to predict the ecotoxicity
of Vibrio fischeri using COSMO-RS descriptors,” _Chemo-_
_sphere_, vol. 170, pp. 242-250, 2017.

[4] T. Le, V. C. Epa, F. R. Burden, and D. A. Winkler,
“Quantitative structure-property relationship modeling of
diverse materials properties,” _Chemical Reviews_, vol. 112,
no. 5, pp. 2889-2919, 2012.

[5] A. Cooper, T. Potter, and T. Luker, “Prediction of efficacious inhalation lung doses via the use of in silico
lung retention quantitative structure-activity relationship
models and in vitro potency screens,” _Drug Metabolism_
_and Disposition_, vol. 38, no. 12, pp. 2218-2225, 2010.

[6] P. Liu and W. Long, “Current mathematical methods
used in QSAR/QSPR studies,” _International Journal of_
_Molecular Sciences_, vol. 10, no. 5, pp. 1978-1998, 2009.

[7] Z. Cheng, A. S. Zhu, L. Q. Zhang, “Quantitative analysis of electronic absorption spectroscopy by piecewise
orthogonal signal correction and partial least square,”
_Spectroscopy & Spectral Analysis_, vol. 28, no. 4, pp. 860864, 2008.

[8] A. Baghban, J. Sasanipour, S. Habibzadeh, et al., “Estimating solubility of supercritical H2S in ionic liquids through
a hybrid LSSVM chemical structure model,” _Chinese_
_Journal of Chemical Engineering_, 2018.

[9] A. L. Teixeira, J. P. Leal, and A. O. Falcao, “Random
forests for feature selection in QSPR Models-an application for predicting standard enthalpy of formation of
hydrocarbons,” _Journal of Cheminformatics_, vol. 5, no. 1,
pp. 1-15, 2013.

[10] D. K. Duvenaud, D. Maclaurin, J. Iparraguirre, et al.,
“Convolutional networks on graphs for learning molecular
fingerprints,” _Advances in Neural Information Processing_
_Systems_, pp. 2224-2232, 2015.

[11] M. Dehmer, M. Grabner, and K. Varmuza, “Information
indices with high discriminative power for graphs,” _PLoS_
_One_, vol. 7, no. 2, pp. e31214, 2012.

[12] M. V. Diudea, A. Ilic, K. Varmuza, and M. NDehmer,
“Network analysis using a novel highly discriminating
topological index,” _Complexity_, vol. 16, no. 6, pp. 32-39,
2011.

[13] A. Sandryhaila and J. M. F. Moura, “Discrete signal
processing on graphs,” _IEEE Transactions on Signal Pro-_
_cessing_, vol. 61, no. 7, pp. 1644-1656, 2013.

[14] A. Sandryhaila and J. M. F. Moura, “Discrete signal
processing on graphs: Frequency analysis,” _IEEE Transac-_
_tions on Signal Processing_, vol. 62, no. 12, pp. 3042-3054,
2014.

[15] D. I. Shuman, S. K. Narang, P. Frossard, et al., “The
emerging field of signal processing on graphs: Extending
high-dimensional data analysis to networks and other
irregular domains,” _IEEE Signal Processing Magazine_,
vol. 30, no. 3, pp. 83-98, 2013.

[16] A. Ortega, P. Frossard, J. Kovacevic, et al., “Graph signal processing: Overview, challenges, and applications,”



_Proceedings of the IEEE_, vol. 106, no. 5, pp. 808-828,
2018.

[17] H. Bahonar, A. Mirzaei, S. Sadri, et al., “Graph embedding using frequency filtering,” _IEEE Transactions on_
_Pattern Analysis and Machine Intelligence_, 2019, DOI:
10.1109/TPAMI.2019.2929519.

[18] M. M. Bronstein, J. Bruna, Y. LeCun, et al., “Geometric
deep learning: going beyond Euclidean data,” _IEEE Signal_
_Processing Magazine_, vol. 34, no. 4, pp. 18-42, 2017.

[19] A. Kheradmand, P.Milanfar, “A general framework for
regularized, similarity-based image restoration,” _IEEE_
_Transactions on Image Processing_, vol. 23, no. 12, pp.
5136-5151, 2014.

[20] B. Lucic, N. Trinajstic, and B. Zhou, “Comparison between the sum-connectivity index and productconnectivity index for benzenoid hydrocarbons,” _Chemical_
_Physics Letters_, vol. 475, no. 1-3, pp. 146-148, 2009.

[21] S. Hayat, M. Imran, and J. B. Liu, “Correlation between
the Estrada index and _π_ -electronic energies for benzenoid
hydrocarbons with applications to boron nanotubes,” _In-_
_ternational Journal of Quantum Chemistry_, vol. 119, no.
23, pp. e26016, 2019.

[22] S. Hayat, S. Khan, A. Khan, et al., “Valency-based
molecular descriptors for measuring the _π_ -electronic energy of lower polycyclic aromatic hydrocarbons,” _Polycyclic_
_Aromatic Compounds_, pp. 1-17, 2020.

[23] D. Vukicevic and A. Graovac, “Note on the comparison
of the first and second normalized zagreb eccentricity
indices,” _Acta Chimica Slovenica_, vol. 57, no. 3, pp. 524528, 2010.

[24] R. Todeschini and C. Viviana, _Handbook of Molecular_
_Descriptors_, vol. 11. John Wiley & Sons, 2008.

[25] I. Gutman and B. Zhou, “Laplacian energy of a graph,”
_Linear Algebra and Its Applications_, vol. 414, no. 1, pp.
29-37, 2006.

[26] M. Dehmer, L. Sivakumar, and K. Varmuza, “Uniquely discriminating molecular structures using novel
eigenvalue-based descriptors,” _Match-Communications in_
_Mathematical and Computer Chemistry_, vol. 67, no. 1, pp.
147, 2012.

[27] U. Brandes and T. Erlebach, _Network Analysis: Method-_
_ological Foundations_, Berlin: Springer, 2005.

[28] L. F. Costa, F. A. Rodrigues, G. Travieso, et al., “Characterization of complex networks: A survey of measurements,” _Advances in Physics_, vol. 56, no. 1, pp. 167-242,
2007.

[29] C. B. Santiago, J. Y. Guo, and M. S. Sigman, “Predictive
and mechanistic multivariate linear regression models for
reaction development,” _Chemical Science_, vol. 9, no. 9,
pp. 2398-2412, 2018.

[30] A. Lusci, G. Pollastri, and P. Baldi, “Deep architectures
and deep learning in chemoinformatics: the prediction of
aqueous solubility for drug-like molecules,” _Journal of_
_Chemical Information and Modeling_, vol. 53, no. 7, pp.
1563-1575, 2013.

[31] K. Mansouri, C. M. Grulke, R. S. Judson, et al., “OPERA
models for predicting physicochemical properties and environmental fate endpoints,” _Journal of Cheminformatics_,



0162-8828 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Auckland University of Technology. Downloaded on November 01,2020 at 22:09:08 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TPAMI.2020.3032718, IEEE

Transactions on Pattern Analysis and Machine Intelligence


12



vol. 10, no. 1, pp. 10, 2018.

[32] R. Todeschini, D. Ballabio, and F. Grisoni, “A comparative study of regression metrics for predictivity assessment
of QSAR models,” _Journal of Chemical Information and_
_Modeling_, vol. 56, no. 10, pp. 1905-1913, 2016.

[33] D. Van Dijk, J. Nainys, R. Sharma, et al., “MAGIC:
A diffusion-based imputation method reveals gene-gene
interactions in single-cell RNA-sequencing data,” _bioRx-_
_iv_,2017.

[34] T. Hastie, R. Tibshirani, J. Friedman, _The Elements of_
_Statistical Learning: Data Mining, Inference, and Predic-_
_tion_, Springer Science & Business Media, 12th Printing,
2017.

[35] D. Vukicevic, N. Trinajstic, “Bond-additive modeling. 3.
Comparison between the product-connectivity index and
sum-connectivity index,” _Croatica Chemica Acta_, vol. 83,
no. 3, pp. 349-351, 2010.

[36] F. R. Burden, D. A. Winkler, “Optimal sparse descriptor
selection for QSAR using Bayesian methods,” _QSAR &_
_Combinatorial Science_, vol. 28, no. 6-7, pp. 645-653,
2009.

[37] S. H. Unger, C. Hansch, “Model building in structureactivity relations. Reexamination of adrenergic blocking
activity of. beta.-halo-. beta.-arylalkylamines,” _Journal of_
_Medicinal Chemistry_, vol. 16, no. 7, pp. 745-749, 1973.

[38] R. Todeschini, G. Paola, “3D-modelling and prediction
by WHIM descriptors. Part 6. Application of WHIM descriptors in QSAR studies,” _Quantitative Structure-Activity_
_Relationships_, vol. 16, no. 2, pp. 120-125, 1997.

[39] F. A. de Lima Ribeiro, M. MC. Ferreira, “QSPR models
of boiling point, octanol-water partition coefficient and
retention time index of polycyclic aromatic hydrocarbons,”
_Journal of Molecular Structure: THEOCHEM_, vol. 663,
no. 1-3, pp. 109-126, 2003.

[40] M. MC. Ferreira, “Polycyclic aromatic hydrocarbons: a
QSPR study,” _Chemosphere_, vol. 44, no. 2, pp. 125-146,
2001.


**Xiaoying Song** received the B.S. degree in electronic science and technology and the Ph.D. degree
in microelectronics and solid-state electronics from
Wuhan University, Wuhan, China, in 2012 and 2017,
respectively. Since July 2017, she has been with
the School of Information Science and Engineering, Wuhan University of Science and Technology,
Wuhan. Her research interests include image compression, graph signal processing.



**Li Chai** received the B.S. degree in applied mathematics and the M.S. degree in control science and
engineering from Zhejiang University, Hangzhou,
China, in 1994 and 1997, respectively, and the Ph.D.
degree in Electrical engineering from the Hong Kong
University of Science and Technology, Hong Kong,
in 2002. In September 2002, he joined Hangzhou Dianzi University, China. He worked as a postdoctoral
research fellow at the Monash University, Australia,
from May 2004 to June 2006. In 2008, he joined
Wuhan University of Science and Technology, where
he is currently a Chutian Chair Professor. He was a visiting researcher
at Newcastle University in 2009, Central Queensland University in 2011,
and Harvard University in 2015. His research interests include distributed
optimization, filter bank frames, graph signal processing, and networked
control systems. Professor Chai is the recipient of the Distinguished Young
Scholar of the National Science Foundation of China. He is currently an
associate editor for the Decision and Control.


**Jingxin Zhang** (M02) received the M.E. and Ph.D.
degrees in electrical engineering from Northeastern
University, Shenyang, China. Since 1989, he has
held research and academic positions in Northeastern University, China, the University of Florence,
Italy, the University of Melbourne, the University
of South Australia, Deakin University and Monash
University, Australia. He is currently Associate Professor of Electrical Engineering, Swinburne University of Technology, and Adjunct Associate Professor
of Electrical and Computer Systems Engineering,
Monash University, Melbourne, Australia. His research interests include signals and systems and their applications to biomedical and industrial systems.
He is the recipient of the 1989 Fok Ying Tong Educational Foundation (Hong
Kong) for the Outstanding Young Faculty Members in China, and 1992 China
National Education Committee Award for the Advancement of Science and
Technology.



0162-8828 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Auckland University of Technology. Downloaded on November 01,2020 at 22:09:08 UTC from IEEE Xplore. Restrictions apply.


