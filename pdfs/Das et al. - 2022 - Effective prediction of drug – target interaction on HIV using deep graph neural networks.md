[Chemometrics and Intelligent Laboratory Systems 230 (2022) 104676](https://doi.org/10.1016/j.chemolab.2022.104676)


Contents lists available at ScienceDirect

# Chemometrics and Intelligent Laboratory Systems


[journal homepage: www.elsevier.com/locate/chemometrics](https://www.elsevier.com/locate/chemometrics)

## –
## Effective prediction of drug target interaction on HIV using deep graph neural networks


Bihter Das, Mucahit Kutsal, Resul Das [* ]


_Department of Software Engineering, Technology Faculty, Firat University, 23119, Elazig, Turkey_



A R T I C L E I N F O


_Keywords:_
HIV drug Resistance
Human immunodeficiency virus
Geometric deep learning
Graph neural networks

Molecular data


**1. Introduction**



A B S T R A C T


Individuals infected with HIV are controlled by drugs known as antiretroviral therapy by suppressing the amount
of HIV in the body. Therefore, studies to predict both HIV-drug resistance and virus-drug interaction are of great
importance for the sustainable effectiveness of antiretroviral drugs. A solution to this problem is provided by
investigating the connection between the recently emerging geometric deep learning method and the evolu­
tionary principles governing drug resistance to the HIV. In this study, geometric deep learning (GDL) approach is
proposed to predict drug resistance to HIV, and virus-drug interaction. The drug data set in the SMILES repre­
sentation was first converted to molecular representation and then to a graph representation that the GDL model
could understand. Message Passing Neural Network (MPNN) transfers the node feature vectors to a different
space for the training process to take place. Next, we develop a geometric neural network architecture where the
graph embedding values are passed through the fully connected layer and the prediction is performed. The
obtained results show that the proposed GDL method outperforms existing methods in predicting drug resistance
in HIV with 93.3% accuracy.



Human Immunodeficiency Virus is a retrovirus that causes AIDS. The
type common in North America, Europe, and the world is often called
HIV1, while the type common in West Africa is named HIV 2. HIV 1 and
HIV 2 have some differences. HIV 1 virus causes millions of deaths each

year and spreads much faster than HIV 2. The virus has a capsid and a
sheath in its structure. The capsid consists of p24 proteins surrounding
the RNA, while the sheath surrounding a small matrix contains glyco­
proteins that determine the antigenic structure of the virus. Fig. 1 shows
the structure of the HIV [1].
The HIV virus has three glycoproteins called gp160, gp41, and
gp120. Of these glycoproteins, gp160 is separated by the protease
enzyme into gp120 and gp41 and forms two separate glycoproteins.
While gp41 allows HIV to enter the cell, gp120 allows HIV to attach to
DNA. LEDGF determines the way HIV enters DNA [2]. A human infected
with HIV is diagnosed with AIDS (Acquired Immune Deficiency Syn­
drome) as a result of high damage to the immune system. A person with
HIV will experience symptoms such as cold-like fever, fatigue, weakness,
and chills within 3–4 weeks of ingestion of the virus. In the later periods,
diarrhea, weight loss, enlargement of lymph nodes, and fever can be



seen. People infected with HIV are lifelong carriers of the virus and are
contagious. The average life expectancy of HIV carriers diagnosed in
advanced stages is between 12 and 18 months. Early diagnosis prevents
the transmission of the HIV virus to others and delays the transformation
into AIDS. Currently, there is no HIV treatment that gives permanent
results. However, there are many drugs that control HIV infection and
prevent possible problems that may arise due to HIV. These drugs are
called Antiretroviral Therapy (ART) [3,4]. These drugs bind to the active
sites of proteins, inhibiting their functions, and ultimately causing
reduced virus replication. HIV infection is a health condition that re­
quires routine control and treatment. An HIV-infected individual can
continue to live a healthy life with appropriate drug treatment before
HIV reaches the AIDS stage. With using of antiretroviral therapy, HIV-1
infection is now considered to be more treatable and life expectancy
longer. The mortality rate has decreased with antiretroviral treatment
and as a result of effective treatment, the demographic characteristics of
the HIV-infected population in the world have also changed [5,6].
Against all this, because the HIV virus has dynamic adaptability and a
high mutation rate, it can develop resistance to existing drugs and
eventually cause drug-virus interaction failure. Unlike other types of
viruses of the HIV virus, they belong to the group of retroviruses and




 - Corresponding author.
_E-mail addresses:_ [bihterdas@firat.edu.tr (B. Das), mucahitkutsal@gmail.com (M. Kutsal), rdas@firat.edu.tr (R. Das).](mailto:bihterdas@firat.edu.tr)


[https://doi.org/10.1016/j.chemolab.2022.104676](https://doi.org/10.1016/j.chemolab.2022.104676)
Received 4 September 2022; Received in revised form 15 September 2022; Accepted 19 September 2022

Available online 22 September 2022
0169-7439/© 2022 Elsevier B.V. All rights reserved.


_B. Das et al._ _Chemometrics and Intelligent Laboratory Systems 230 (2022) 104676_



**Fig. 1.** The structure of HIV.


have a reverse transcription mechanism. Normally, while DNA produces
RNA and RNA produces proteins, retroviruses can produce DNA from
reverse transcriptase RNA through an event called virtual innate. This
causes them to evolve very quickly and mutate too much. In short, the
mutation rate of HIV virus is much higher than other virus types. For this
reason, studies to identify new targets and drug discoveries that can
directly interact with the HIV virus are very valuable for science, med­
icine and the pharmaceutical industry. Also, drug-virus resistance
mechanisms need to be investigated in order to develop more effective
treatments using existing antiretroviral drugs.
Various phenotypic tests are used to measure HIV drug resistance

[7]. However, these phenotypic tests are both very laborious and costly.
In addition, in these phenotypic tests, experts should be aware of the
mutations of unknown causes and the correlations that develop in the
mechanism of resistance to specific drugs [8]. There are statistical [9]
and machine learning [10,11], random forest algorithm [12,13], sup­
port vector machine [11], decision tree [14], logistic regression [13],
and artificial neural networks [15–17]. In the literature that use
rules-based classification and position-specific scoring matrices based
on genomic features to predict HIV virus and drug resistance. Deep
learning models have recently been used in computational biology ap­
plications as well as in predicting virus drug resistance. However, the
most unclear aspect of deep learning models is the model interpretation
feature. In classification and prediction applications, deep learning is
often criticized for its black box nature, as the model is not clear and it is
unclear how the extracted features are derived. Steiner et al. used a deep
learning model to predict drug resistance on HIV virus sequences. They
trained the HIV virus sequence and 18 ART drugs dataset using bidi­
rectional recurrent neural network and convolutional neural network

architecture(CNN). They stated that the CNN model performed better
and interpreted drug resistance mutations in viral genotype-phenotype
data better [18]. Blassel et al. used machine learning methods to study
drug-virus interactions and identify drug resistance mutations (DRM) on
HIV virus. Feature extraction was made from the data taken from En­

gland and Africa and Multinomial naive Bayes (NB), Logistic Regression
(LR) and Random Forest (RF) methods were used to train the system

[19]. Cai et al. analyzed 21 drug interactions with the HIV virus using
machine learning methods such as Random Forest (RF), support vector
machine (SVM), and radial basis function (RBF). Also, they determined
mutations in the HIV protein. Seven physicochemical property tech­
niques were used for the digitization process. They stated that this
technique reflects the target protein interaction properties well. They
reduced the feature size by using the principal component analysis
(PCA) method for feature selection. At the end of the study, they
compared the performance of 3 machine learning methods. The



RBF-based SVM method showed higher performance than the RF model

[20]. Blassel et al. reviewed and reviewed their bioinformatics work on
predicting drug resistance on HIV virus. They divided the studies in the
literature into 3 groups. These groups are; Machine learning methods
used to predict the resistance level of HIV variants, phylogenetic
methods used to investigate the dynamics of the interaction of HIV drug
resistance, and deep sequencing methods to study HIV variants. The
difficulties and performances of these methods are explained [21]. Turki
et al. used transfer learning to predict drug sensitivity. They experi­
mented with their model on clinical trial data from 3 auxiliary datasets
(gene and drugs). Their proposed approach obtained good results [22].
Shtar et al. studied on drug-drug interactions using a combine method
that includes artificial neural networks and classic graph similarity
measures methods together. They collected data from DrugBank data­
base. They showed that their model is effective and fast to identify po­
tential drugs interaction [23]. Ekpenyong et al. proposed a transfer
learning model to predict drug resistance on HIV virus. They wanted to
detect the response of patients to treatments early. The data set used was
trained with a 2-layer neural network and a 5-layer deep neural
network. The deep neural network model performed better with transfer
learning [24].
Geometric deep learning (GDL), with its increasing popularity in
recent years, provides unprecedented performance in many structural
biology problems and modeling the RNA structure of drugs and viruses.
The reason why we specifically use geometric deep learning in this study
is to more closely match the molecular structure of drugs acting on the
HIV virus and the physical processes underlying molecular recognition.
GDL learns about the geometric structure and detailed arrangement of
atoms that make up the structure of the drug and virus and performs
strongly against other traditional methods even when the dataset is
small. The model uses 3D atoms and elements as input data. It will
facilitate the solution of many problems such as biomolecule design,
virus-drug interaction, and drug discovery, as it shows experts the
interaction between molecules both visually and with performance
metrics. In addition, another advantage of this method is that it allows
the reuse of new problems in another dataset with minimum adaptation.
This study aims to predict drug resistance to the HIV virus with a
geometric deep learning-based model and to understand the interactions
that may occur between an atom, ion, or molecule bound to a central
metal and another target molecule. Numerical calculations are made in
the literature to predict interactions between molecules with other deep
learning and machine learning methods. However, these traditional
methods do not visually show the bond and interaction between mole­
cules. It just gives a percentage of interaction with an accuracy calcu­
lation. With the proposed method, the interaction between the virus and
the drug, using data in SMILES format, both visually shows which
molecules are bonded and how, and also indicates the ratio of this

interaction between 0 and 1.


_1.1. Main contributions of the paper_


The main contributions of this paper are outlined below.


 - With our proposed geometric deep learning-based model, we mea­

sure how much a molecule could be drug potential.

 - Compared to the deep learning and machine learning methods in the
literature, our method shows the resistance of drugs in SMILES
format with HIV protein by molecular and chemical bonds and ex­
presses molecular information in a multidimension manner.

 - Instead of representing the drug-virus molecule in a mathematical
format as in traditional machine learning methods, we fingerprint
the molecules in our proposed model.

 - With the proposed method, instead of representing the drug-virus
molecule in a mathematical format, we fingerprint the molecules.

 - With our geometric deep learning-based model, we make the process
of determining the degree of chemical property (toxicity) of a



2


_B. Das et al._ _Chemometrics and Intelligent Laboratory Systems 230 (2022) 104676_



molecule, which is quite time-consuming and expensive with tradi­
tional methods, much faster and cheaper.


The remainder of this paper is organized as follows. Section 2 con­
tains details of the materials and method. Section 3 presents the
experimental results and discussion. Section 4 presents the limitations
and future works and Section 5 contains the conclusion.



**2. Materials and method**


In recent years, deep learning algorithms such as CNN, GAN, and
LSTM are used machine learning applications to achieve remarkable
accuracy on a variety of complex problems, even surpassing human
performance on tasks like Image classification, speech recognition,
language translation, and image generation. As known as graph neural



**Fig. 2.** The flow diagram of the proposed GDL model for the drug resistance prediction on HIV virus. 3


_B. Das et al._ _Chemometrics and Intelligent Laboratory Systems 230 (2022) 104676_


**Fig. 3.** An example from positive and negative groups from SMILES notation to chemical structure.



networks, geometric deep learning is a new field of machine learning
that can learn from complex data like graphs and multi-dimensional
points. It seeks to apply traditional Convolutional Neural Networks to
3D objects, graphs, and manifolds [25].
In this section, stages performed for the prediction of drug resistance
in HIV were presented in detail. The proposed algorithm involves 3
steps: (i) dataset collection and preprocessing, (ii) training, (iii) pre­
diction. Fig. 2 shows the flow diagram of the proposed GDL model.



_2.1. Data collection and preprocessing_


In this study, we collected two different data from the MoleculeNet
database [26] for the training of the proposed model. The first group
data set, which is described as positive samples, is the drug group that
interacts with the HIV virus. The second group data set, which is
described as a negative sample, is the other antiretroviral drug group
that does not interact with the HIV. The MoleculeNet benchmark, which
is developed by the Pande Group at Stanford University, was specifically
designed to test machine learning methods of molecular features. The



**Fig. 4.** Positive-negative relationship state of atoms in a molecule.


**Fig. 5.** The used architecture in the fully connected network part.


4


_B. Das et al._ _Chemometrics and Intelligent Laboratory Systems 230 (2022) 104676_


**Fig. 6.** The quantitative estimates of drug resistance on HIV virus.



used dataset is a Simplified Molecular Input Line Input System (SMILES)
string format. The SMILES has features that allow it to be easily pro­
duced and processed in the laboratory, in addition to physicochemical
properties that aid absorption, specificity, and low toxicity. In the study,
the SMILES data format was used to simplify the processing and analysis
of compounds, to represent the sequence and structure of small mole­
cules and peptides, to generate graph data, and because it is easier to
preprocess. Our dataset includes a total of 2884 drugs, 1442 of which are
positive samples that interact with the HIV virus and 1442 are negative
samples that do not interact with the HIV virus. SMILES, which is one of
the most popular 1-dimensional molecular structure representations, is a
line notation system used to describe the structure of chemical species
using short ASCII sequences. For training of the data set in our geometric
deep learning-based model, the data set in SMILES format was first
converted to molecular representation and then converted into a graph
structure that the GDL model can understand. Each node on the graph
has its own feature vector and each edge has a feature vector in the same



way, and the graph structure is also an undirected graph. Fig. 3 shows
the chemical structure of a drug as an example from both positive and
negative samples.
In a molecular representation, the interaction is expressed positively
and negatively [27]. The representations of these interactions are given
in Fig. 4.


_2.2. Training_


In this subsection, we describe the training process of the proposed
geometric deep learning for the prediction of drug resistance to HIV. The
training process includes 2 stages. The first one is Message Passing
Neural Network(MPNN), and the second one is a Fully Connected
Network(FCN).
MPNN is a type of neural network model designed to make molecules
work on graphs. MPNN helps to convert the information in the graphic
data structure into a vector, this vector is called graphic embedding. The



5


_B. Das et al._ _Chemometrics and Intelligent Laboratory Systems 230 (2022) 104676_


​



embedding vector is updated based on messages from other nodes in the
graph. With MPNN, the node feature vectors are transferred to a
different space. With MPNN, a graph embedding is obtained starting
from node embeddings on a molecule, and after the message passing step
and the node embedding vectors are obtained, a graph embedding is
obtained by average pooling over the nodes. With MPNN, which is
accepted as a supervised learning framework for graphs proposed by
Gilmer et al. [28,29], spatial and spectral architectures in graph network
are redefined in two stages, such as message passing and readout. In the
message passing stage, shows the hidden state of each v node. v is
updated by a node update function U t . This function receives message
from neighboring nodes. These steps are repeated throughout the T time
step as in Equation (1).


​



_m_ _[t]_ _v_ [+][1] = ∑ _M_ _t_


​



( _h_ _[t]_ _υ_ _[,][ h]_ _[t]_ _w_ _[,][ e]_ _[υ]_ _[w]_


​



) (1)


​



and the prediction is actualized. In the fully connected layer stage, as
seen in Fig. 5, four dense layers were used and the dropout value of the
first three layers was determined as 20% to prevent overfitting. In the
fully connected network part, there are 92 neurons in the first Dense
layer. There are 46 neurons in the 2nd Dense layer and 23 neurons in the
3th Dense layer and 1 neuron as output in the last layer. In addition, the
ReLu activation function was used in the first 3 layers and the sigmoid
function in the last layer. Fig. 5 shows the architecture used in the fully
connected network part.


_2.3. Prediction_


In this subsection, quantitative estimates of drug-likeness (QED) are
used to evaluate for the prediction of drug resistance interactions on HIV
virus. QED is an index that shows drug-virus interaction using available

–
information in the range of [0 1]. This index is used in computational
methods in small molecule drug discovery, in demonstrating drug-virus
interactions, and in evaluating similarity features with existing drugs

[30]. Fig. 6 shows the predicted possibilities of drug resistance and
interaction on HIV virus at the end of the proposed model.
The presented geometric deep learning model is shown in Algorithm

1.


**Algorithm 1** . The algorithm of the proposed geometric deep learning
model.

​



**3. Experimental results and discussion**


​ In the GDL model that we proposed to predict drug resistance and

interaction with the HIV virus, 80% of the data set in the positive and
negative samples groups were used for training. 10% was used for
validation and 10% for testing. For training, the group of positive
samples that interacted with the virus and the group of drugs that did
not interact with the virus, known as negative samples, were randomly
sampled of equal size. Similarly, the same number of samples of negative
samples were selected along with the positive samples for testing. The
accuracy performance of our proposed GDL model is 93%. All the codes



_w_ _ε_ _N_ ( _υ_ )


Here, and represent the sequential hidden states of nodes w and v in
iteration t. N(v) represents the neighbors of v in the graph G. Then, the
hidden state of node v is updated according to the message as in Equa­
tion (2).


​



_h_ _[t]_ _υ_ [+][1] = _U_ _t_


​



( _h_ _[t]_


​




_[t]_ _υ_ _[,][ m]_ _[t]_ _υ_ [+][1]


​



_υ_


​



) (2)


​



The structural information of the graph is then collected and this
information is embedded as node tags. Based on the set of hidden states
of the nodes, a feature vector specific to the entire graph is computed by
R readout stage as Equation (3).


​ y = R({h [t] _v_ _[/]_ _[ υε]_ _[G]_ [})] (3)

It is important that all 3 functions defined in Equations (1)–(3) are
learnable and differentiable. In the model based on geometric deep
learning, a graphical metric space is obtained by removing the readout
function. It is not enough just to position the information on the node
close to the embedding space, because as the output of the study, a
whole graph is examined for the prediction of interaction molecule on
the HIV virus, and drug resistance. In the proposed GDL model, graph
embedding values obtained are passed through the fully connected layer



​


6


_B. Das et al._ _Chemometrics and Intelligent Laboratory Systems 230 (2022) 104676_


resistance phenotypes of HIV-associated drugs, which can be very
informative in determining virus-drug interaction. We did not consider
these dependencies in this study, as all categories of drug-specific
phenotype data are not available for the HIV virus.
For our future work, we aim to obtain more drug data from data­
bases, improve the geometric deep learning method to adapt to big data
challenges, and further improve the accuracy performance of the GDL
method in measuring HIV virus-drug interaction.


**Fig. 7.** The accuracy performance of the GDL model.



and datasets of the experiment in the study have been uploaded to the
[GitHub service. It can be accessed via the link https://github.com/bih](https://github.com/bihterdas/HIV-drug-interaction-with-geometric-deep-learning)
[terdas/HIV-drug-interaction-with-geometric-deep-learning. We ach­](https://github.com/bihterdas/HIV-drug-interaction-with-geometric-deep-learning)
ieved 93% accuracy performance with our GDL model to measure drug
resistance to the HIV virus. The area under the curve (AUC) is calculated
relative to the Receiver Operating Characteristic (ROC) curve for the
proposed GDL model to describe study quality, providing a more accu­
rate visual interpretation for the prediction of drug resistance to the HIV
virus. Fig. 7 shows the AUC curve of the proposed GDL model.
Negative and positive data were distributed as 50% in the dataset.
Fig. 8 shows the accuracy graphs of the proposed model for positive and
negative classes.

Fig. 9 shows the performance curve for F1-score and Fig. 10 shows
the performance of ROC curve.
In the literature, machine learning and deep learning methods have
been used in studies on detecting drug-virus interaction for HIV or
predicting drug resistance. The Accuracy(ACC) and Area Under Curve
(AUC) performance values of the methods used in the previous studies
are shown in Table 1. As can be seen from Table 1, our proposed geo­
metric deep learning method to predict drug resistance on HIV virus
achieved the highest accuracy performance of 93%.


**4. Limitations and future works**


The primary limitations of this study are the limited dataset used in
the geometric deep learning method. In clinical settings, large amounts
of HIV-drug interaction data are difficult to make publicly available due
to reasons such as sequencing costs and patient privacy. In our proposed
method, the dataset size remained a limiting factor for the performance
of the system. In addition, there may be dependencies between the



**Fig. 9.** The F1-score performance of the GDL model.


**Fig. 10.** The ROC curve performance of the GDL model.



**Fig. 8.** The accuracy performance of the positive and negative groups.


7


_B. Das et al._ _Chemometrics and Intelligent Laboratory Systems 230 (2022) 104676_



**Table 1**

Comparison of previous studies on HIV-drug interaction.


Reference Dataset Method AUC/(ACC)

Paper



Stenier et al. HIV drug resistance

[18] database(Stanford

University)



Deep learning
techniques(CNN, BRNN)

Neural networks,
Multilayer perceptron
(MLP)



Stenier et al. HIV drug resistance Deep learning AUC = 0.895

[18] database(Stanford techniques(CNN, B- with MLP
University) RNN) AUC = 0.89

Neural networks, with BRNN
Multilayer perceptron AUC = 0.927
(MLP) with CNN

Tomic et al. U.S. FDA portal VINI in silico model of —

[31] database cancer

Zhang et al. RT Protease Sequence Antivirogram and Probability

[32] database PhenoSense assays value _<_

0.0033

Parienti Bi-chat Claude Bernard Transfer Learning AUC = 85.70
et al. [33] University Approach
Hospital&Cote de
Nacre University
Hospital



Zhang et al. RT Protease Sequence Antivirogram and Probability

[32] database PhenoSense assays value _<_

0.0033

Parienti Bi-chat Claude Bernard Transfer Learning AUC =
et al. [33] University Approach
Hospital&Cote de
Nacre University
Hospital



Shtar et al. Drugbank Adjacency Matrix

[34] Factorization(AMF)
and Adjacency Matrix

Factorization with

Propagation(AMFD)

methods



ACC = 0.56



Cai et al. HIV drug resistance

[35] database(Stanford

University)



Machine learning —

methods



**The**

**proposed**

**GDL**

**model**



**MoleculeNet** **Geometric deep**
**learning/graph**
**neural networks**



**ACC** = **0.933**



**5. Conclusion**


The proposed model converts SMILES drug data to molecule repre­
sentation and then to graph representation for Message Passing Neural
Network. MPNN helps to convert the information in the graphic data
structure into graphic embedding vector. The graph embedding values
are passed through the fully connected layer, the training stage of our
model is completed, and the prediction is actualized. The experimental
results show that our proposed GDL model has a good accuracy per­
formance of 93.3% for the prediction of drug resistance in HIV virus. In
addition, our model outperformed traditional machine learning and
deep learning methods evaluated under similar conditions. Finally, this
study demonstrates the utility of the very new geometric deep learning
method in predicting HIV drug resistance and provides a framework
with many other important applications in viral genomics besides HIV.


**CRediT authorship contribution statement**


**Bihter Das:** Conceptualization, Methodology, Writing – original
draft, Validation. **Mucahit Kutsal:** Software, Data collection, Coding.
**Resul Das:** Visualization, Investigation, Writing – review & editing,
Validation.


**Declaration of competing interest**


The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to influence
the work reported in this paper.


**Data availability**


Data will be made available on request.



**References**


[[1] M. Kierczak, M. Dramiski, J. Koronacki, A rough set-based model of HIV-1 reverse](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref1)
[transcriptase resistome, Bioinf. Biol. Insights 3 (2009) 109–127.](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref1)

[[2] N. Beerenwinkel, B. Schmidt, H. Walter, R. Kaiser, T. Lengauer, D. Hoffmann, Et al.](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref2)
[Diversity and complexity of HIV-1 drug resistance: a bioinformatics approach to](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref2)
[predicting phenotype from genotype, Proc. Natl. Acad. Sci. U.S.A. 99 (2002)](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref2)
[8271–8276.](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref2)

[[3] S.G. Deeks, A.N. Phillips, HIV infection, antiretroviral treatment, ageing, and non-](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref3)
[AIDS related morbidity, BMJ 338 (2009) a3172.](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref3)

[[4] K. Bhaskaran, O. Hamouda, M. Sannes, et al., Changes in the risk of death after HIV](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref4)
[seroconversion compared with mortality in the general population, JAMA 300](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref4)
[(2008) 51–59.](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref4)

[[5] R.B. Effros, C.V. Fletcher, K. Gebo, et al., Aging and infectious diseases: workshop](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref5)
[on HIV infection and aging: what is known and future research directions, Clin.](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref5)
[Infect. Dis. 47 (2008) 542–553.](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref5)

[6] C.A. Hughes, A. Tseng, R. Cooper, Managing drug interactions in HIV-infected
adults with comorbid illness, CMAJ (Can. Med. Assoc. J.) 187 (1) (Jan. 2015)
[36–43, https://doi.org/10.1503/cmaj.131626.](https://doi.org/10.1503/cmaj.131626)

[7] J. Zhang, S.Y. Rhee, J. Taylor, R.W. Shafer, Comparison of the precision and
sensitivity of the antivirogram and PhenoSense HIV drug susceptibility assays,
[J. Acquir. Immune Defic. Syndr. 38 (2005) 439–444, https://doi.org/10.1097/01.](https://doi.org/10.1097/01.qai.0000147526.64863.53)
[qai.0000147526.64863.53.](https://doi.org/10.1097/01.qai.0000147526.64863.53)

[[8] I. Bonet, M.M. García, Y. Saeys, Y. Van De Peer, R. Grau, Predicting human](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref8)
[immunodeficiency virus (HIV) drug resistance using recurrent neural networks,](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref8)
[Proc. IWINAC (2007) 234–243. La Manga del Mar Menor, Spain. 18–21 June.](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref8)

[9] T. Liu, R. Shafer, Web resources for HIV type 1 genotypic-resistance test
[interpretation, Clin. Infect. Dis. 42 (2006) 1608–1618, https://doi.org/10.1086/](https://doi.org/10.1086/503914)
[503914.](https://doi.org/10.1086/503914)

[10] N. Beerenwinkel, M. D¨aumer, M. Oette, K. Korn, D. Hoffmann, R. Kaiser,
T. Lengauer, J. Selbig, Walter H. Geno2pheno, Estimating phenotypic drug
resistance from HIV-1 genotypes, Nucleic Acids Res. 31 (2003) 3850–3855,
[https://doi.org/10.1093/nar/gkg575.](https://doi.org/10.1093/nar/gkg575)

[11] M. Riemenschneider, T. Hummel, D. Heider, SHIVA—a web application for drug
[resistance and tropism testing in HIV, BMC Bioinf. 17 (2016) 314, https://doi.org/](https://doi.org/10.1186/s12859-016-1179-2)
[10.1186/s12859-016-1179-2.](https://doi.org/10.1186/s12859-016-1179-2)

[12] M. Riemenschneider, K.Y. Cashin, B. Budeus, S. Sierra, E. Shirvani-Dastgerdi,
S. Bayanolhagh, R. Kaiser, P.R. Gorry, D. Heider, Genotypic prediction of Co[receptor tropism of HIV-1 subtypes A and C, Sci. Rep. 6 (2016) 1–9, https://doi.](https://doi.org/10.1038/srep24883)
[org/10.1038/srep24883.](https://doi.org/10.1038/srep24883)

[13] D. Heider, R. Senge, W. Cheng, E. Hüllermeier, Multilabel classification for
exploiting cross-resistance information in HIV-1 drug resistance prediction,
[Bioinformatics 29 (2013) 1946–1952, https://doi.org/10.1093/bioinformatics/](https://doi.org/10.1093/bioinformatics/btt331)
[btt331.](https://doi.org/10.1093/bioinformatics/btt331)

[14] N. Beerenwinkel, B. Schmidt, H. Walter, R. Kaiser, T. Lengauer, D. Hoffmann,
K. Korn, J. Selbig, Diversity and complexity of HIV-1 drug resistance: a
bioinformatics approach to predicting phenotype from genotype, Proc. Natl. Acad.
[Sci. USA 99 (2002) 8271–8276, https://doi.org/10.1073/pnas.112177799.](https://doi.org/10.1073/pnas.112177799)

[15] D. Wang, B. Larder, Networks enhanced prediction of Lopinavir resistance from
genotype by use of artificial neural networks, J. Infect. Dis. 188 (2003) 653–660,
[https://doi.org/10.1086/377453.](https://doi.org/10.1086/377453)

[16] O. Sheik Amamuddy, Bishop N.T., Tastan Bishop O. Improving fold resistance [¨]
prediction of HIV-1 against protease and reverse transcriptase inhibitors using
[artificial neural networks, BMC Bioinf. 18 (2017) 369, https://doi.org/10.1186/](https://doi.org/10.1186/s12859-017-1782-x)
[s12859-017-1782-x.](https://doi.org/10.1186/s12859-017-1782-x)

[17] M.E. Ekpenyong, P.I. Etebong, T.C. Jackson, Fuzzy-multidimensional deep learning
for efficient prediction of patient response to antiretroviral therapy, Heliyon 5
[(2019), e02080, https://doi.org/10.1016/j.heliyon.2019.e02080.](https://doi.org/10.1016/j.heliyon.2019.e02080)

[18] M.C. Steiner, K.M. Gibson, K.A. Crandall, Drug resistance prediction using deep
learning techniques on HIV-1 sequence data, Viruses 12 (5) (May 2020) 560,
[https://doi.org/10.3390/v12050560.](https://doi.org/10.3390/v12050560)

[19] L. Blassel, et al., Using machine learning and big data to explore the drug resistance
[landscape in HIV, PLoS Comput. Biol. 17 (8) (2021), 1008873, https://doi.org/](https://doi.org/10.1371/journal.pcbi.1008873)
[10.1371/journal.pcbi.1008873. Aug.](https://doi.org/10.1371/journal.pcbi.1008873)

[20] Q. Cai, R. Yuan, J. He, M. Li, Y. Guo, Predicting HIV drug resistance using weighted
machine learning method at target protein sequence-level, Mol. Divers. 25 (3)

[[21] L. Blassel, A. Zhukova, C.J. Villabona-Arenas, K.E. Atkins, S. Hu(Aug. 2021) 1541–1551, https://doi.org/10.1007/s11030-021-10262-y´e, O. Gascuel, Drug .](https://doi.org/10.1007/s11030-021-10262-y)
resistance mutations in HIV: new bioinformatics approaches and challenges, Curr.
[Opin. Virol. 51 (Dec. 2021) 56–64, https://doi.org/10.1016/j.coviro.2021.09.009.](https://doi.org/10.1016/j.coviro.2021.09.009)

[22] T. Turki, Z. Wei, J.T. Wang, Transfer learning approaches to improve drug
sensitivity prediction in multiple myeloma patients, IEEE Access 5 (2017)
[7381–7393, https://doi.org/10.1109/ACCESS.2017.2696523.](https://doi.org/10.1109/ACCESS.2017.2696523)

[23] G. Shtar, L. Rokach, B. Shapira, Detecting drug–drug interactions using artificial
neural networks and classic graph similarity measures, PLoS One 14 (8) (2019)
[1–21, https://doi.org/10.1371/journal.pone.0219796.](https://doi.org/10.1371/journal.pone.0219796)

[24] M.E. Ekpenyong, et al., A transfer learning approach to drug resistance
classification in mixed HIV dataset, Inform. Med. Unlocked 24 (Jan. 2021),
[100568, https://doi.org/10.1016/j.imu.2021.100568.](https://doi.org/10.1016/j.imu.2021.100568)

[[25] F. Monti, D. Boscaini, J. Masci, E. Rodola, J. Svoboda, M.M. Bronstein, Geometric](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref25)
[deep learning on graphs and manifolds using mixture model cnns, Proceedings of](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref25)
[the IEEE Conference on Computer Vision and Pattern Recognition (2017)](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref25)
[5115–5124.](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref25)

[26] Z. Wu, et al., MoleculeNet: A Benchmark for Molecular Machine Learning,” arXiv:
1703.00564 [physics, Stat], Oct. 2018. Accessed: Apr. 09, 2022. [Online].
[Available: http://arxiv.org/abs/1703.00564.](http://arxiv.org/abs/1703.00564)



8


_B. Das et al._ _Chemometrics and Intelligent Laboratory Systems 230 (2022) 104676_




[27] T. Hasebe, Knowledge-embedded message-passing neural networks: improving
molecular property prediction with human knowledge, ACS Omega 6 (42) (Oct.
[2021) 27955–27967, https://doi.org/10.1021/acsomega.1c03839.](https://doi.org/10.1021/acsomega.1c03839)

[[28] J. Gilmer, S.S. Schoenholz, P.F. Riley, O. Vinyals, G.E. Dahl, Neural Message](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref28)
[Passing for Quantum Chemistry, 2017.](http://refhub.elsevier.com/S0169-7439(22)00187-3/sref28)

[29] Kyunghyun Cho, Bart van Merrienboer, Bahdanau Bart, Bahdanau Dzmitry,
Yoshua Bengio, On the properties of neural machine translation: encoder-decoder
[approaches, arXiv preprint arXiv:1409.1259 (2014) 1–9. https://doi.org/10.48](https://doi.org/10.48550/arXiv.1409.1259)
[550/arXiv.1409.1259.](https://doi.org/10.48550/arXiv.1409.1259)

[30] B. Das, M. Kutsal, R. Das, A geometric deep learning model for display and
prediction of potential drug-virus interactions against SARS-CoV-2, Chemometr.
[Intell. Lab. Syst. 229 (2022), 104640, https://doi.org/10.1016/j.](https://doi.org/10.1016/j.chemolab.2022.104640)
[chemolab.2022.104640.](https://doi.org/10.1016/j.chemolab.2022.104640)

[31] D. Tomi´c, et al., The screening and evaluation of potential clinically significant HIV
drug combinations against the SARS-CoV-2 virus, Inform. Med. Unlocked 23
[(2021), 100529, https://doi.org/10.1016/j.imu.2021.100529. Jan.](https://doi.org/10.1016/j.imu.2021.100529)




[32] J. Zhang, S.Y. Rhee, J. Taylor, R.W. Shafer, Comparison of the precision and
sensitivity of the Antivirogram and PhenoSense HIV drug susceptibility assays,
[J. Acquir. Immune Defic. Syndr. 38 (2005) 439–444, https://doi.org/10.1097/01.](https://doi.org/10.1097/01.qai.0000147526.64863.53)

[[33] J.J. Parienti, V. Massari, D. Descamps, A. Vabret, E. Bouvet, B. Larouzqai.0000147526.64863.53.](https://doi.org/10.1097/01.qai.0000147526.64863.53) ´e, R. Verdon,
Predictors of virologic failure and resistance in HIV-infected patients treated with
nevirapine-or efavirenz-based antiretroviral therapy, Clin. Infect. Dis. 38 (9)
[(2004) 1311–1316, https://doi.org/10.1086/383572.](https://doi.org/10.1086/383572)

[34] G. Shtar, L. Rokach, B. Shapira, Detecting drug–drug interactions using artificial
neural networks and classic graph similarity measures, PLoS One 14 (8) (2019)
[1–21, https://doi.org/10.1371/journal.pone.0219796.](https://doi.org/10.1371/journal.pone.0219796)

[35] Q. Cai, R. Yuan, J. He, et al., Predicting HIV drug resistance using weighted
machine learning method at target protein sequence-level, Mol. Divers. 25 (2021)
[1541–1551, https://doi.org/10.1007/s11030-021-10262-y.](https://doi.org/10.1007/s11030-021-10262-y)



9


