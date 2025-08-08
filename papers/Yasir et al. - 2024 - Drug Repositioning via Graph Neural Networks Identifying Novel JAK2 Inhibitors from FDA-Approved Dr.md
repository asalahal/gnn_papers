# **_molecules_**

_Article_
## **Drug Repositioning via Graph Neural Networks: Identifying** **Novel JAK2 Inhibitors from FDA-Approved Drugs through** **Molecular Docking and Biological Validation**


**Muhammad Yasir** **[1]** **, Jinyoung Park** **[1]** **, Eun-Taek Han** **[2]** **, Won Sun Park** **[3]** **, Jin-Hee Han** **[2]** **and Wanjoo Chun** **[1,]** *****


1 Department of Pharmacology, Kangwon National University School of Medicine,
Chuncheon 24341, Republic of Korea; yasir.khokhar1999@gmail.com (M.Y.);
jinyoung0326@kangwon.ac.kr (J.P.)
2 Department of Medical Environmental Biology and Tropical Medicine, Kangwon National University School
of Medicine, Chuncheon 24341, Republic of Korea; ethan@kangwon.ac.kr (E.-T.H.);
han.han@kangwon.ac.kr (J.-H.H.)
3 Department of Physiology, Kangwon National University School of Medicine,
Chuncheon 24341, Republic of Korea; parkws@kangwon.ac.kr
***** Correspondence: wchun@kangwon.ac.kr; Tel.: +82-33-250-8853



**Citation:** Yasir, M.; Park, J.; Han, E.-T.;


Park, W.S.; Han, J.-H.; Chun, W. Drug


Repositioning via Graph Neural


Networks: Identifying Novel JAK2


Inhibitors from FDA-Approved Drugs


through Molecular Docking and


Biological Validation. _Molecules_ **2024**,


_29_ [, 1363. https://doi.org/10.3390/](https://doi.org/10.3390/molecules29061363)


[molecules29061363](https://doi.org/10.3390/molecules29061363)


Academic Editor: Cleydson Breno


Rodrigues dos Santos


Received: 16 February 2024


Revised: 8 March 2024


Accepted: 14 March 2024


Published: 19 March 2024


**Copyright:** © 2024 by the authors.


Licensee MDPI, Basel, Switzerland.


This article is an open access article


distributed under the terms and


conditions of the Creative Commons


[Attribution (CC BY) license (https://](https://creativecommons.org/licenses/by/4.0/)


[creativecommons.org/licenses/by/](https://creativecommons.org/licenses/by/4.0/)


4.0/).



**Abstract:** The increasing utilization of artificial intelligence algorithms in drug development has
proven to be highly efficient and effective. One area where deep learning-based approaches have
made significant contributions is in drug repositioning, enabling the identification of new therapeutic
applications for existing drugs. In the present study, a trained deep-learning model was employed to
screen a library of FDA-approved drugs to discover novel inhibitors targeting JAK2. To accomplish
this, reference datasets containing active and decoy compounds specific to JAK2 were obtained from
the DUD-E database. RDKit, a cheminformatic toolkit, was utilized to extract molecular features from

the compounds. The DeepChem framework’s GraphConvMol, based on graph convolutional network
models, was applied to build a predictive model using the DUD-E datasets. Subsequently, the trained
deep-learning model was used to predict the JAK2 inhibitory potential of FDA-approved drugs. Based
on these predictions, ribociclib, topiroxostat, amodiaquine, and gefitinib were identified as potential
JAK2 inhibitors. Notably, several known JAK2 inhibitors demonstrated high potential according
to the prediction results, validating the reliability of our prediction model. To further validate
these findings and confirm their JAK2 inhibitory activity, molecular docking experiments were
conducted using tofacitinib—an FDA-approved drug for JAK2 inhibition. Experimental validation
successfully confirmed our computational analysis results by demonstrating that these novel drugs
exhibited comparable inhibitory activity against JAK2 compared to tofacitinib. In conclusion, our
study highlights how deep learning models can significantly enhance virtual screening efforts in
drug discovery by efficiently identifying potential candidates for specific targets such as JAK2.
These newly discovered drugs hold promises as novel JAK2 inhibitors deserving further exploration
and investigation.


**Keywords:** drug repositioning; Janus kinase 2 (JAK2); deep-learning; RDKit; DeepChem; GraphConvMol;
graph convolutional neural network; molecular docking


**1. Introduction**


Drug repositioning involves identifying novel therapeutic uses for medications that
have previously gained approval for different medical purposes [ 1 ]. It can notably accelerate
the drug development process, enhance the utility of established drugs, and reveal novel
treatments for ailments lacking effective remedies [ 2 ]. Accordingly, drug repurposing is
becoming an increasingly important area of research in drug development. Computeraided drug design (CADD) has become an essential tool in the domain of drug discovery



_Molecules_ **2024**, _29_ [, 1363. https://doi.org/10.3390/molecules29061363](https://doi.org/10.3390/molecules29061363) [https://www.mdpi.com/journal/molecules](https://www.mdpi.com/journal/molecules)


_Molecules_ **2024**, _29_, 1363 2 of 19


and development [ 3 ]. Utilizing computational algorithms and software, CADD enables
efficient screening of large compound libraries, offering a faster and more cost-effective
alternative to traditional experimental approaches [ 4 ]. A primary strength of CADD is
its capacity to swiftly assess a large number of compounds, minimizing the extensive
laboratory testing in traditional experimental studies, which can be time consuming and
expensive [5].
Artificial intelligence is rapidly expanding and possesses significant promise in transforming the drug development process [ 6 ]. Deep learning (DL), a subset of artificial
intelligence, enables its models to assimilate data and formulate predictions or decisions
without explicit programming [ 7 ]. DL plays a pivotal role in drug development by analyzing vast datasets encompassing genetic and clinical data. This analysis aids in discovering
new drug targets, predicting drug effectiveness with accuracy, and fine-tuning drugs [ 8, 9 ].
One of its primary advantages is the capability to analyze large and complex datasets [ 10 ].
Whereas traditional data analysis methods, like manual examination and statistical techniques, can be labor intensive and time consuming, DL models offer swift and adept data
analysis, discerning patterns and forecasting outcomes, which in turn fast-tracks the drug
development process [ 11 ]. An additional strength of DL in drug development is its capability to predict the potency and toxicity of compounds [ 12 ]. By analyzing extensive datasets,
DL models can discern trends suggesting drug effectiveness and potential toxicity, enabling
the prediction of these attributes before the synthesis and laboratory evaluation. Therefore,
integrating DL within CADD can markedly improve the speed, efficiency, and success of
the drug discovery, making it significant in drug discovery and development.
Janus kinases (JAKs) belong to a family of non-receptor tyrosine kinases crucial for
cellular signaling, especially within the immune system [ 13 – 15 ]. Disruption in JAK function
is associated with various inflammatory disorders, such as rheumatoid arthritis, psoriasis,
and inflammatory bowel disease [ 16, 17 ]. Four main members constitute the JAK family:
JAK1, JAK2, JAK3, and TYK2 [ 18 ]. Each has unique traits and distinct cellular roles.
Specifically, JAK1 is associated with signaling via the interferon- α receptor, while JAK3
primarily operates through the common gamma chain receptor [ 19 – 21 ]. JAK2, however,
interfaces with a multitude of cytokines, including erythropoietin, thrombopoietin, and
interleukin-6 [ 14, 22 ], positioning it with a more expansive signaling capability compared
to JAK1 and JAK3. Notably, JAK2’s involvement has been identified in conditions like
polycythemia vera, essential thrombocythemia, and myelofibrosis [ 23 ]. While certain
JAK1 and JAK3 mutations are reported in acute lymphoblastic leukemia [ 24 ], JAK2’s role
appears more central in the onset of diverse diseases [ 25 – 27 ]. Given this context, our
study focuses on the development of novel JAK2 inhibitors. In this study, we employed
a graph neural network algorithm to train on datasets containing active and decoy JAK2
inhibitors. Subsequently, we screened an FDA-approved drug library to identify potential
JAK2 inhibitors for drug repurposing. We further assessed the selected compounds using
molecular docking techniques and their biological activity was validated using a JAK2
kinase assay kit to discover novel JAK2 inhibitors.


**2. Results and Discussions**


The process of integrating deep-learning, molecular docking, and experimental evaluation for drug repurposing of novel JAK2 inhibitors is illustrated in Figure 1. The process
comprised seven distinct phases: (1) data acquisition and preparation from the DUD-E
database, (2) configuration of the graph convolutional network model, (3) training and
evaluation of the deep learning model, (4) predictive assessment of FDA-approved drugs,
(5) molecular docking for the top-predicted drugs, (6) experimental validation of potential
candidates through JAK2 kinase assay, and (7) analysis of results to confirm the validity of
repurposing FDA-approved drugs as novel JAK2 inhibitors.


_Molecules_ **2024**, _29_, 1363 3 of 19


**Figure 1.** The process of integrating deep-learning, molecular docking, and experimental evaluation
for drug repurposing novel JAK2 inhibitors.


_2.1. JAK2 Active and Decoy Datasets and Its Preprocessing Using RDKit_


The DUD-E (Database of Useful Decoys: Enhanced) database is an open-access
database that hosts benchmark sets of protein–ligand complexes. It encompasses a set of
experimentally confirmed active compounds, their affinities against diverse targets, and associated decoys that are confirmed not to bind with the target. Though these decoys share
similar physicochemical properties with the active compounds, their two-dimensional
topology differ [ 28 ]. The DUD-E database has frequently served as a benchmark for the creation and evaluation of computational docking techniques [ 29, 30 ]. The JAK2 dataset in the
[DUD-E database (https://dude.docking.org/targets/JAK2) (accessed on 15 January 2024)](https://dude.docking.org/targets/JAK2)
features 107 active compounds, curated from an initial set of 246 compounds, paired with
6500 decoy compounds. Figure 2A provides illustrative images of the structures of both
active and decoy compounds, with labels in the legend to distinguish them. To evaluate
the physicochemical distinctions between active and decoy compounds, we used RDKit
(Version 2023.09.6), a free chemoinformatics software toolkit, to calculate their molecular attributes. Upon comparison, we observed minimal variations in the distribution patterns of
molecular features such as weight, LogP, the number of hydrogen bond donors/acceptors,
topological polar surface area (TPSA), and number of rotatable bonds (Figure 2B).


_2.2. Deep-Learning Model Setup, Training, and Evaluation_


DeepChem is an open-source Python library designed for deep learning applications
within drug discovery and cheminformatics. It offers a comprehensive suite of tools for
managing molecular data and harnessing various deep learning techniques for tasks like
molecular attribute forecasting, virtual ligand screening, and molecule optimization [ 31, 32 ].
In this research, we employed the GraphConvMol model from DeepChem to discern
differences between active and decoy compounds within the JAK2 dataset. This model, an
integral part of the DeepChem suite, uses a form of graph convolutional neural network
to process molecular graphs, turning them into fixed-size representation vectors. Each
atom is denoted as a node, and covalent bonds become edges in this molecular graph.
The algorithm involves a series of message-passing phases, during which each atom
communicates its unique features to adjacent atoms. After collecting messages from
neighboring atoms, the data are synthesized to update the current atom’s attributes. The
final representation of the molecule is formulated by combining the individual atom
representations and further refining them through feed-forward neural networks. As
GraphConvMol facilitates the end-to-end learning of molecular structures, it stands as a
robust asset in cheminformatics endeavors, specifically in predicting molecular properties
and drug discovery [ 33, 34 ]. The JAK2 dataset was split into training, validation, and
test sets at a ratio of 8:1:1, and then subjected to the GraphConvMol model using crossvalidation with a fold of 5. To assess the model’s performance, the AUC (Area Under the
Curve) of the ROC (Receiver Operating Characteristic) curve was computed for the training,
validation, and test datasets. The ROC curve, generated from a five-fold cross-validation
on the training dataset, illustrated a True Positive Rate (TPR) value of 1 at an exceptionally
low False Positive Rate (FPR) with an AUC value of 0.992 (Figure 3A). This suggests that
the GraphConvMol model exhibits high sensitivity in identifying positive instances while
effectively minimizing false positives.


_Molecules_ **2024**, _29_, 1363 4 of 19


( **A** )


( **B** )


**Figure 2.** ( **A** ) Representative image of active and decoy compounds. ( **B** ) Distribution of molecular
weight, LogP, number of hydrogen bond donors/acceptors, TPSA, and number of rotatable bonds in
active and decoy compounds.


_Molecules_ **2024**, _29_, 1363 5 of 19


( **A** )


( **B** )


**Figure 3.** ( **A** ) The AUR-ROC curve of five-fold cross validation of the training dataset. ( **B** ) The
confusion matrix values of training, validation, and test datasets.


To evaluate the performance of GraphConvMol on DUD-E datasets, metrics such
as precision, recall, F1 score, sensitivity, accuracy, and specificity were calculated across
training, validation, and test datasets (Table 1). The training dataset showed reliable
performances, with only 2 out of 94 positive instances misclassified as negative (recall: 0.98) .
In the validation dataset, there was one false positive out of 652 negatives (precision: 0.83)
and 3 false negatives out of 8 instances (recall: 0.63). The lower performance metrics in the
validation dataset may be due to the limited number of active compounds. However, the
model demonstrated optimal performance in the test dataset, achieving a score of 1 in all
metrics (Table 1).


**Table 1.** Performance metrics of GraphConvMol model.


**Precision** **Recall** **F1 score** **Accuracy** **Specificity**


**Training** 1.000 0.979 0.989 0.999 1.000
~~**Validation**~~ ~~0.833~~ ~~0.625~~ ~~0.714~~ ~~0.994~~ ~~0.999~~

**Test** 1.000 1.000 1.000 1.000 1.000


Precision = True Positives/(True Positives + False Positives); recall = True Positives/(True Positives + False Negatives); F1 score = 2 _×_ ((precision _×_ recall)/(precision + recall)); accuracy = (True Positives + True Negatives)/total
population; specificity = True Negatives/(True Negatives + False Positives).


_Molecules_ **2024**, _29_, 1363 6 of 19


Due to the disproportionate number of decoys relative to active compounds in the
dataset, the Matthews correlation coefficient (MCC) was utilized to assess the performance
of the GraphConvMol model. This metric is particularly effective for datasets with such
imbalances. The averaged MCC values from five-fold cross-validation processes were 0.96
for the training set and 0.76 for the validation set. A perfect prediction accuracy is indicated
by an MCC of 1, highlighting that those scores of 0.96 and 0.76 demonstrate the model’s
robustness and dependability. It is generally expected for the MCC value of the test set to
surpass that of the validation set since the model, after being trained on the training set, is
then tested on the novel and unencountered data of the validation set. Furthermore, the
variation in MCC values observed across the five-fold cross-validation suggests that the
model is not overly fitted to the training data.


_2.3. Prediction of JAK2 Inhibitory Potential from FDA-Approved Drugs_


Repositioning FDA-approved drugs offers distinct advantages. Given that these drugs
have already undergone rigorous pre-clinical and clinical evaluations for safety, dosage, and
pharmacokinetics, their repositioning often means shorter development periods, reduced
costs, and a higher probability of success. The trained model, utilizing the GraphConvMol
algorithm from DeepChem, processed SMILES strings of FDA-approved drugs to assess
their potential for JAK2 inhibitory activity. Predictions on JAK2 inhibitory capability for
these drugs spanned a range from 0 (inactive) to 1 (highly active). While a majority of
the compounds were deemed inactive, a small subset was identified as potential actives
(Figure 4A). Figure 4B presents structures of select compounds that were predicted to have
high activity, with labels showcasing their anticipated values.
Noticeably, several of top-ranked compounds such as ruxolitinib, baricitinib, tofacitinib,
and upadacitinib (listed in Table 2) are well-known JAK2 inhibitors. This strongly indicates
the high robustness and reliability of the present model. From the set of drugs highly predicted by the GraphConvMol model, we selected several candidates for further evaluation
regarding their potential JAK2 inhibitory actions through molecular docking and experimental
~~assessment. Gefitinib, a tyrosine kinase inhibitor used in acute lymphoblastic leukemia [~~ ~~35~~ ],
ribociclib, a CDK kinase inhibitor employed in the treatment of metastatic breast cancer [ 36 ],
amodiaquine, an inhibitor of heme polymerase inhibitor used for malaria [ 37 ], and topiroxostat, an inhibitor of xanthine oxide used for gout [ 38 ], were among the chosen drugs. These
drugs have not been previously reported to be associated with JAK2 inhibition.


( **A** )


_Molecules_ **2024**, _29_, 1363 7 of 19


( **B** )


**Figure 4.** ( **A** ) Distribution of GraphConvMol prediction. ( **B)** Structures of highly predicted compounds from FDA-approved drugs.


**Table 2.** Detailed information of drugs that were predicted with high JAK2 inhibitory potential.


**Smiles** **Neg** **Pos** **Name** **Target** **Use**


N#CC[C@H](C1CCCC1)n1ccc2ncnc3[nH]ccc23)cn1 0.0002 0.9998 Ruxolitinib JAK Inhibitor Myelofibrosis, Anti
cancer drug

COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1.Cl 0.0004 0.9996 Gefitinib Tyrosine Kinase, EGFRinhibitor Non-small cell lungcarcinoma


Nc1ncnc2[nH]cnc12 0.0007 0.9993 Adenine Nucleobase Nucleotide

c1ncc2nc[nH]c2n1 0.0007 0.9993 Purine Heterocyclic aromatic organiccompound DNA and RNAformation


CC[C@H](Nc1ncnc2[nH]cnc12)c1nc2cccc(F)c2c(=O)n1- 0.0011 0.9989 Idelalisib Phosphoinositide 3-kinase Blood cancer
c1ccccc1 inhibitor



C#Cc1cccc(Nc2ncnc3cc4c(cc23)OCCOCCOCCO4)c1 0.0021 0.9979 Icotinib



Epidermal growth factor
receptor
tyrosine kinase inhibitor
(EGFR-TKI)



Non-small cell lung

cancer



c1coc(CNc2ncnc3nc[nH]c23)c1 0.0031 0.9969 Kinetin Proapoptotic anti-proliferative Cell division
plant growth regulator


Cl.O=C(O)c1cn(-c2ccc(F)cc2)c2cc(N3CCNCC3)c(F)cc2c1=O 0.0031 0.9969 Sarafloxcin Quinolone antibiotic drug Antibiotic


CCS(=O)(=O)N1CC(CC#N)(n2cc(-c3ncnc4[nH]ccc34)cn2)C1 0.0033 0.9967 Baricitinib JAK2 inhibitor Rheumatoid arthritis


C[C@@H]1CCN(C(=O)CC#N)C[C@@H]1N(C)c1ncnc2[nH]ccc12 0.0033 0.9967 Tofacitinib JAKs inhibitor Rheumatoid arthritis


Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@@H]1O.O 0.0039 0.9961 Vidarabine Human herpesvirus 1 DNA Antiviral
polymerase


2 _[′]_                                          Nc1ncnc2c1ncn2[C@H]1C[C@H](O)[C@@H](CO)O1.O 0.0039 0.9961 Phosphodiesterase inhibitor Energy source
Deoxyadenosine


CN(C)C(=O)c1cc2cnc(Nc3ccc(N4CCNCC4)cn3)nc2n1C1CCCC1 0.0045 0.9955 Ribociclib CDK4/CDK6 kinase inhibitor Metastatic breast cancer


Malaria, Rheumatoid
CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12 0.0051 0.9949 Chloroquine Heme polymerase inhibitor arthritis


_Molecules_ **2024**, _29_, 1363 8 of 19


**Table 2.** _Cont._


**Smiles** **Neg** **Pos** **Name** **Target** **Use**

COc1cc2nc(N3CCN(C(=O)C4CCCO4)CC3)nc(N)c2cc1OC.Cl.O.O 0.0074 0.9926 Terazosin Alpha 1-adrenergic receptorinhibitor Adrenaline blocker


CCN(CC)Cc1cc(Nc2ccnc3cc(Cl)ccc23)ccc1O.Cl.Cl.O.O 0.0077 0.9923 Amodiaquine Heme polymerase inhibitor Malaria


C[C@H](Nc1ncnc2[nH]cnc12)c1cc2cccc(Cl)c2c(=O)n1- 0.0081 0.9919 Duvelisib PI3K inhibitor Chronic lymphocytic
c1ccccc1 leukemia


CC[C@@H]1CN(C(=O)NCC(F)(F)F)C[C@@H]1c1cnc2cnc3[nH] ccc3n12 0.0082 0.9918 Upadacitinib JAK inhibitor Rheumatoid arthritis


1,10c1cnc2c(c1)ccc1cccnc12 0.0086 0.9914 Phenanthroline Fe(II) chelator Metal chelator


Nucleotide reverse
C[C@H](Cn1cnc2c(N)ncnc21)OCP(=O)(O)O.O 0.0100 0.9900 Tenofovir HIV
transcriptase inhibitor


Cc1cc(/C=C/C#N)cc(C)c1Nc1ccnc(Nc2ccc(C#N)cc2)n1 0.0117 0.9883 Rilpivirine Transcriptase inhibitor HIV


Non-small cell lung
C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1 0.0122 0.9878 Erlotinib Tyrosine kinase, EGFR inhibitor cancer (NSCLC),
pancreatic cancer


Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(- 0.0172 0.9828 Imatinib Tyrosine kinase, Bcr-abl Chronic myeloid
c2cccnc2)n1 inhibitor leukemia


N#Cc1cc(-c2n[nH]c(-c3ccncc3)n2)ccn1 0.0177 0.9823 Topiroxostat Xanthine oxidase inhibitor Hyperuricemia (gout)


O=c1[nH]cnc2[nH]ncc12.[Na+] 0.0196 0.9804 Allopurinol Xanthine oxidase inhibitor Hyperuricemia (gout)

O.S=c1nc[nH]c2nc[nH]c12 0.0298 0.9702 Mercaptopurine6- Purine nucleotide synthesisinhibitor AntineoplasticAntimetabolite,

C=C[C@H]1CN2CC[C@H]1C[C@H]2[C@H](O)c1ccnc2ccc(OC) cc12.Cl.O.O 0.0301 0.9699 Quinine Potassium channel blocker Antimalarial, Analgesic


Cl.Cl.c1cnc2cc3c(cc2n1)C1CNCC3C1 0.0337 0.9663 Varenicline Nicotinic receptor blocker Smoking cessation


O=c1[nH]cnc2nc[nH]c12 0.0365 0.9635 Hypoxanthine Nucleic acid synthesis Malaria parasite cultures


CN[C@@H]1C[C@H]2O[C@@](C)([C@@H]1OC)n1c3ccccc3c3 0.0414 0.9586 Staurosporine PKC inhibitor Cancer
c4c(c5c6ccccc6n2c5c31)C(=O)NC4


The term ‘Neg’ refers to non-active outcomes, while ‘Pos’ indicates active outcomes. The predictive values are
quantified where a value of 1 represents a perfect prediction, and a value of 0 signifies no possibility of the
predicted outcome.


_2.4. Structural Analysis of the JAK2 Protein_


A non-receptor tyrosine kinase JAK2 belongs to the Janus kinase family and has been
linked to signaling by the single chain receptors (Epo-R, Tpo-R, GH-R, and PRL-R), the
GM-CSF receptor family’s (IL-3R, IL-5R, and GM-CSF-R), and the type II cytokine receptor
family’s (interferon receptor) [ 39 ]. It was constructed by 311 amino acids forming a single
chain (PDBID 3JY9). Loops, α -helices, and β -sheets are present in the overall structure
of JAK2 (Figure 4). Furthermore, a VADAR 1.8 structural assessment demonstrated that
JAK2 was constructed by 40% α -helices, 22% β -sheets, 37% coils, and 23% turns. Moreover,
the Ramachandran plots analysis revealed that 95.1% of amino acids occur in the favored
region, while 98.6% of residues were in the allowed zone of dihedral angles phi ( φ ) and
psi (ψ) (Figure 5B).


_2.5. The Binding Pocket Analysis_


Along with its structure and position inside a protein, a binding pocket’s function is
influenced by the group of amino acid residues that surround it [ 40 ]. Using the Discovery
Studio ligand interaction method, the binding pocket residues of JAK2 were obtained from
the interaction of JAK2 and co-crystalized ligand and mentioned as Leu14, Gly15, Val22,
Ala39, Leu142, Glu57, Val70, Met88, Tyr90, Leu91, Gly152, and Asp153. Therefore, the
co-crystalized ligand was chosen by the current selection approach to define the CDocker
binding sphere. Furthermore, the binding sphere was subjected to contraction to limit it
to the accurate position respective to our selected binding pocket residues. The binding
sphere values were X = 12, Y = 13, Z = 2.6, and the radius value was fixed as 7.8 to study
the interaction of selected compounds in the active region of JAK2 (Figure 6A,B).


_Molecules_ **2024**, _29_, 1363 9 of 19


**Figure 5.** ( **A**, **B** ). Three-dimensional structure ( **A** ) of the JAK2 protein and the computed Ramachandran plot ( **B** ), calculated by discovery studio.


**Figure 6.** ( **A**, **B** ). The figure ( **A** ) manifests the full structural representation and the binding pocket
of JAK2. The whole protein is colored as hot pink, the interior helixes are colored dark slate blue,
while the binding surface area is colored as light sea green. Furthermore, the active site residues are
mentioned on their position in the active region of the target protein in black ( **B** ).


_2.6. Molecular Docking Analysis_


The top 20 screened compounds were docked against JAK2. The docked complexes
were evaluated and examined independently and scored based on the minimal docking
energy and interaction energy values. The Discovery Studio CDocker module forecasts
two types of energy values (CDocker energy and CDocker interaction energy). The terms
CDocker energy and CDocker interaction energy are used to describe the energy involved
in the various interactions between the ligand and the receptor. CDocker energy displays
the overall docking energy based on the 3D structural and physiochemical features of the
ligand and protein, whereas the strength and nature of each individual contact between the


_Molecules_ **2024**, _29_, 1363 10 of 19


ligand and the receptor are revealed by CDocker interaction energy. It calculates how much
the overall binding strength is affected by intermolecular forces such Van der Waals forces,
electrostatic interactions, and hydrogen bonds [ 41 – 43 ]. The top 20 docking results concerning the CDocker interaction energy score were depicted in Table 3. Therefore, ribociclib
demonstrate the lowest interaction energy values. Moreover, the gefitinib and amodiaquine
came up in the top 10 docked compounds, although they exhibit a high CDocker interaction
energy as compared to ribociclib, they exhibit a lower interaction energy than the reference
compound tofacitinib (gefitinib, amodiaquine, and tofacitinib manifest _−_ 50.6 kcal/mol,
_−_ 44.4 kcal/mol, and _−_ 40.0 kcal/mol, respectively). Topiroxostat comparatively revealed a

_−_
high interaction energy ( 28.8 kcal/mol) compared to the reference compound.


**Table 3.** The docking energy values (kcal/mol) of top 20 screened docked FDA compounds against
JAK2 protein, calculated by Discovery Studio.


**Cdocker Interaction Energy** **CDocker Energy**
**Compounds**
**(kcal/mol)** **(kcal/mol)**


Ribociclib _−_ 58.0 _−_ 5.3

Imatinib _−_ 52.6 _−_ 24.3
Staurosporine _−_ 52.2 100.1
Gefitinib _−_ 50.6 _−_ 15.5
Adiphenine _−_ 47.5 _−_ 32.4
Difloxacin _−_ 47.2 _−_ 26.1
Amodiaquine _−_ 44.4 _−_ 19.8
Naratriptan _−_ 42.5 _−_ 31.8
Y-33075 _−_ 42.0 _−_ 31.1
Rilpivirine _−_ 41.3 _−_ 31.8
Tofacitinib _−_ 40.0 _−_ 29.3

Dibucaine _−_ 39.8 _−_ 17.6

Amsacrine _−_ 39.6 _−_ 8.2
Chloroprocaine _−_ 33.4 _−_ 21.5
Topiroxostat _−_ 28.8 _−_ 22.6
Pinacidil _−_ 28.6 _−_ 20.6

Varenicline _−_ 25.6 19.9

Phenanthroline _−_ 23.0 _−_ 4.8
Pargyline _−_ 22.5 _−_ 18.9
Allopurinol _−_ 18.6 _−_ 4.0


_2.7. Binding Interaction Analysis against JAK2_


The top 20 screened compounds that were docked against the JAK2 protein were
further analyzed by Discovery studio and UCSF Chimera to examine and confirm the
binding interaction of ligands with the active site amino acid residues of JAK2.
Ribociclib compounds, which manifest the lowest interaction energy molecular docking energy, manifest the strongest interaction against JAK2 (Figure 7). The ribociclib-JAK2
docked complex expressed eight hydrogen bonds which include the residues Glu57, Asp153,
Glu89, Leu91, Leu14, and Asp98. Two oxygen atoms of ribociclib form hydrogen bonds
against Glu57 and Asp153 with a bond length of 2.28 Å and 1.93 Å, respectively. Another
two oxygen atoms of ligand exhibit two hydrogen bonds with the same Asp98 with a
bonding distance of 2.49 Å and 2.05 Å. Moreover, the other two oxygen atoms also formed
two hydrogen bonds with the same Leu14 with a bonding distance of 2.97 Å and 2.71 Å.
Another solo oxygen atom of ribociclib revealed a hydrogen bond with Glu89 with a bond
length of 2.48 Å. Furthermore, a nitrogen atom of ligand expresses a hydrogen bond against
Leu91 with a bonding distance of 2.30 Å.


_Molecules_ **2024**, _29_, 1363 11 of 19


**Figure 7.** The graphical representation of combined amodiaquine, topiroxostat, gefitinib and ribociclib
interaction in comparison with tofacitinib against the active region amino acid residues of JAK2. The
JAK2 protein is represented in the center (hot pink) while the interactions of ligands are predicted in
different dimensions. Each ligand is colored differently in the active pocket of JAK2 (amodiaquine:
coral, topiroxostat: steel blue, gefitinib: dark khaki, ribociclib: gold). The hydrogen bonds, bonding
distance and bonding amino acid residues are colored red while the other interacting amino acid
residues are colored black. Furthermore, the halogen bond is depicted in cyan color.


The ligand–protein docking analysis of Amodiaquine showed that the ligand binds
within the active region of the target protein as shown in Figure 7. The Amodiaquine-Jak2
docked complex exhibits three hydrogen bonds and one halogen bond. A halogen bond is
formed when there is evidence of a net attractive interaction between an electrophilic region
associated with a halogen atom in one chemical entity and a nucleophilic region in another
or the same molecular entity [ 44 ]. The hydrogen atom of Amodiaquine formed a hydrogen
bond with Arg139 with a bonding distance of 2.97 Å. Additionally, two other hydrogen
atoms of ligand formed hydrogen bonds with Leu91 and Leu14 with bond lengths of 2.16 Å
and 2.03 Å, respectively. Furthermore, the chlorine atom of ligand formed a halogen bond
with Phe19 with a bonding distance of 3.17 Å. Topiroxostat was confined in the active
binding pocket of the JAK2 protein and formed three hydrogen bonds with active region
amino acid residues (Figure 7). The topiroxostat-JAK2 docked complex showed a hydrogen
atom of formed hydrogen bonds with Leu91 with a bond length of 2.67 Å. Furthermore, a
nitrogen atom of topiroxostat also formed a hydrogen bond with Leu91 with a bond length
of 2.42 Å. Moreover, another hydrogen atom of ligand formed a hydrogen bond with Phe19
with a bonding distance of 2.78 Å.
The ligand–protein docking analysis of tofacitinib showed that ligands become docked
within the active region of the target protein, as shown in Figure 7. The tofacitinib-JAK2
docked complex forms three hydrogen bonds which include the residues Lue91 and Arg139.
The oxygen atom of tofacitinib forms a hydrogen bond against Leu91 with a bond length


_Molecules_ **2024**, _29_, 1363 12 of 19


of 2.64 Å. Furthermore, the nitrogen atom of the ligand also forms a hydrogen bond with
Leu91 with a bonding distance of 2.35 Å. Moreover, the oxygen atom of ligand exhibits a
hydrogen bond against Arg139 with a bonding distance of 2.78 Å. The gefitinib compound
also manifests high interactions following ribociclib. The ribociclib-JAK2 docked complex
exhibit six hydrogen bonds (Figure 7). The oxygen atom of ligand formed a hydrogen bond
with Asp154 with bond length of 2.12 Å. An oxygen atom of ligand revealed two hydrogen
bonds with the same Asp153 with bond length of 2.32 Å and 2.75 Å. Moreover, the other
two oxygen atoms of gefitinib showed two hydrogen atoms with the same Leu91 with
the bond length of 2.68 Å and 2.33 Å. Furthermore, an oxygen atom of ligand revealed a
hydrogen bond against Leu14 with a bonding distance of 2.52 Å.
These interactions strongly suggest that the predicted drugs block the active region of
JAK2 by hindering with the active region amino acid residues.


_2.8. Experimental Validation_


JAK2 inhibitory activity of highly predicted drugs and tofacitinib, a reference drug,
was experimentally evaluated using a JAK2 kinase assay kit. Both tofacitinib and the
other drugs exhibited significant inhibition of the JAK2 enzymatic activity at 25 nM. This
concentration is consistent with the previously documented IC 50 values for the inhibitory
activity of tofacitinib against JAK2 [ 45 ]. Remarkably, each of the test drugs demonstrated
significant JAK2 inhibition, with their effectiveness closely paralleling that of tofacitinib
(Figure 8). This suggests that these drugs hold promise as potential novel JAK2 inhibitors.

i



**Figure 8.** JAK2 inhibitory activity of highly predicted JAK2 inhibitors in comparison to tofacitinib.
Single (*) and double (**) marks represent statistical significance at _p_ < 0.05 and _p_ < 0.01, respectively.


_2.9. Structural Evaluation and Similarity Comparison_


To evaluate the structural similarity among the top-ranked drugs, the Tanimoto similarity measure in RDKit was utilized. Tofacitinib and several top-ranked drugs in JAK2
inhibitory potential prediction exhibit structural characteristics. Each of these drugs incorporates one or more heterocyclic rings along with aromatic moieties (Figure 9). Further,
these compounds possess diverse substituents attached to their primary scaffolds, which
likely influence their interactions with JAK2 proteins. However, despite these structural
motifs, an assessment using the Tanimoto similarity coefficient showed that their overall
structural similarity was not notably high (Table 4). In general, while no exact threshold
exists for defining similarity, a Tanimoto similarity value below 0.5 is often regarded as
indicative of dissimilarity in a range from 0 to 1. On this scale, a value of 0 denotes no
similarity at all, and a value of 1 represents complete similarity.


_Molecules_ **2024**, _29_, 1363 13 of 19


**Figure 9.** Structures of highly predicted JAK2 inhibitors.


**Table 4.** Tanimoto similarity comparison of highly predicted JAK2 inhibitors.


**Similarity** **Tofacitinib** **Ribociclib** **Topiroxostat** **Amodiaquine** **Gefitinib**


Tofacitinib                   - 0.196970 0.130841 0.114754 0.156716

Ribociclib 0.196970                   - 0.146154 0.146853 0.180645

Topiroxostat 0.130841 0.146154                  - 0.186916 0.131783
Amodiaquine 0.114754 0.146853 0.186916               - 0.257812
Gefitinib 0.156716 0.180645 0.131783 0.257812                    

While the top-ranked drugs exhibited limited overall similarity to tofacitinib, it is still
possible that these drugs share specific structural features. To explore this, the Maximum
Common Substructure (MCS) algorithm in RDKit was applied. Tofacitinib and the four
top-ranked drugs were analyzed using the MCS algorithm in RDKit with the threshold of
0.5. This analysis grouped tofacitinib, ribociclib, and gefitinib together, with their common
substructures highlighted in red color (Figure 10A). This result implies that factors other
than the structural motif, such as the spatial arrangement of specific conformations, might
contribute to the inhibitory activity on JAK2 protein. Furthermore, similarity maps using
fingerprints in RDKit were employed to illustrate whether the top-ranked drugs possessed
the structural motif of tofacitinib (Figure 10B). The similarity maps of the top-ranked drugs
revealed the presence of structural motif of tofacitinib in their chemical structures. These
findings from the MCS and similarity map findings provide valuable information to guide
further optimization of the selected compounds.
The highly predicted compounds, including tofacitinib, ribociclib, topiroxostat, amodiaquine, and gefitinib, are characterized by their LogP, solubility, gastrointestinal (GI) absorption, blood–brain barrier (BBB) permeation, CYP2D6 inhibition, and Lipinski violation
(Table 5). Notably, tofacitinib exhibits moderate lipophilicity and solubility with high GI
absorption but lacks BBB permeation. Ribociclib and topiroxostat, despite their high GI
absorption, demonstrate contrasting BBB permeation abilities, with ribociclib showing
the potential inhibition of CYP2D6. Amodiaquine and gefitinib, with high lipophilicity,
solubility, and GI absorption, showcase BBB permeation and CYP2D6 inhibition. These
data provide a comprehensive overview of the ADME profiles, aiding in the assessment of
these compounds’ potential suitability for drug development.


_Molecules_ **2024**, _29_, 1363 14 of 19


**Figure 10.** ( **A**, **B** ). Graphical representation of common structural motif found with Maximum
Common Substructure (MCS) ( **A** ) and similarity maps ( **B** ).


**Table 5.** ADME properties of highly predicted JAK2 inhibitors.


**GI Absorp-** **BBB Per-** **CYP2D6** **Lipinski**
**Name** **LogP** **Solubility**
**tion** **meation** **Inhibition** **Violation**


Tofacitinib 1.22 _−_ 3.34 High No No 0
Ribociclib 2.12 _−_ 5.51 High No Yes 0
Topiroxostat 1.38 _−_ 5.24 High No Yes 0
Amodiaquine 4.6 _−_ 8.18 High Yes Yes 0
Gefitinib 3.92 _−_ 7.94 High Yes Yes 0


**3. Methodology**
_3.1. JAK2 Datasets and FDA-Approved Drug Library_


[JAK2 active and decoy datasets were obtained from the DUD-E website (https://](https://dude.docking.org/)
[dude.docking.org/) (accessed on 15 January 2024). The active dataset contained 107 com-](https://dude.docking.org/)
pounds, while the decoy dataset had 6500 compounds. All molecules were expressed as
canonicalized SMILES strings with DUD-E ID and ChEMBL ID numbers. Compounds
were labeled as active and decoy in legend. The FDA-approved drug library was down[loaded from the website of Selleck Chemicals (https://www.selleckchem.com) (accessed](https://www.selleckchem.com)
on 16 January 2024). FDA-approved drug molecules, totaling 3105 in number, were represented in SDF (structure-data file) format and transformed into SMILES strings using RDKit.


_Molecules_ **2024**, _29_, 1363 15 of 19


_3.2. Molecular Descriptor Generation Using RDKit_


Molecular descriptors for the compounds were generated using RDKit. RDKit is an
open-source, high-performance cheminformatics and machine learning toolkit written
[in Python (https://www.rdkit.org) (accessed on 20 January 2024). The toolkit offers fea-](https://www.rdkit.org)
tures for calculating molecular descriptors, producing chemical attributes, and visualizing
chemical data.


_3.3. Deep Learning Architecture_


The JAK2 active and decoy datasets were split for training, validation, and test sets in
[8:1:1 ratio. The GraphConvMol model from DeepChem (https://deepchem.io/models)](https://deepchem.io/models)
(accessed on 22 January 2024) was employed as the deep learning algorithm. The GraphConvMol, being a graph convolutional neural network, adeptly processes graph-structured
inputs like molecular graphs. A concise overview of its architecture is as follows: Initially,
the molecular structures are transformed into graphs where atoms represent nodes and
bonds acting as edges. Following this, several graph convolutional layers are employed
to derive hierarchical features from these molecular graphs. These layers are equipped
with adaptable parameters that have varying weights, fine-tuning the model’s learning to
precisely grasp the nuances of molecular structures. During the training phase, the model
refines its performance by minimizing a loss function in relation to the input molecular
datasets. This optimization adjusts the convolutional layers’ weights through backpropagation. Ultimately, the model seeks to predict specific attributes of molecules, such as
solubility, bioactivity, and potential toxicity, grounded on their structures.


_3.4. JAK2 Structure Retrieval_

The X-ray structure of human JAK2 protein (PDB ID: 3JY9 with 2.10 Å resolution)
[was obtained from the Protein Data Bank (PDB) (https://www.rcsb.org) (accessed on](https://www.rcsb.org)
25 January 2024), and minimized Discovery studio and UCSF Chimera [ 46, 47 ]. The JAK2
protein, made up of helices, sheets, coils, and turns, was subjected to further analysis like
quantitative protein structural analysis using the online freely accessible server VADAR 1.8
[(http://vadar.wishartlab.com/) (accessed on 25 January 2024). Additionally, Discovery](http://vadar.wishartlab.com/)
Studio was employed to analyze and compute the Ramachandran graphs [46].


_3.5. Prediction of Active Binding Site_


The interacting site in the protein’s holo-structure most likely determines the binding
pocket of the protein where the active ligand binds [ 48 ]. The JAK2 X-ray structure was
retrieved from PDB (PDB ID: 3JY9). The co-crystalized ligand was selected and the binding
sphere was constructed by the current selection technique in the binding site window of
Discovery Studio to define the active pocket. The interacting amino acids were chosen by
the ligand interaction approach of Discovery Studio for the accuracy of the binding site
generation. Consequently, the binding sphere was contracted to become restricted to our
selected amino acids.


_3.6. Molecular Docking_


Molecular docking is the most commonly used method for the evaluation of the
interactions and conformations of ligands against the target proteins [ 49 ]. It anticipates
the association strength or binding compatibility between ligand and protein based on
preferred orientation by using scoring algorithms [ 40, 50 ]. The waters and the ligand
molecule were removed from the receptor and the hydrogens were added by Discovery
Studio’s protein preparation module, prior to docking. The ligand preparations were
also carried out for reference and candidate compounds, tautomerization was carried
out, ionization was changed, and bad valences were fixed by Discovery Studio’s ligand
preparation module. Furthermore, the conformation prediction was to the top 10. Therefore,
the Discovery Studio’s CDocker module was employed to perform molecular docking of
the screened ligands against JAK2 with the default orientation and conformation. The


_Molecules_ **2024**, _29_, 1363 16 of 19


lowest CDocker interaction energy values (in kcal/mol) were utilized to estimate the
best-docked complexes.


_3.7. Binding Interaction Analysis_


The 3D graphical evaluations were carried for the docked complexes using UCSF
Chimera 1.10.1 [ 47 ] and Discovery Studio to study the interactions of screened drugs
against JAK2 protein.


_3.8. JAK2 Kinase Inhibitory Activity Assay_


Tofacitinib, topiroxostat, and gefitinib were obtained from Sigma (St. Louis, MO,
USA), and ribociclib and amodiaquine were obtained from Selleck Chemicals (Houston,
TX, USA). The compounds were dissolved in DMSO. JAK2 kinase activity was measured
using the JAK2 Assay Kit from BPS Bioscience (#79520, San Diego, CA, USA) following the
manufacturer’s instructions. The reactions were incubated at 30 degrees Celsius for 45 min.
Then, 50 µ L of the Kinase-Glo MAX reagent (Promega, Madison, WI, USA, #V6071) was
added and covered the plate with aluminum foil, and incubated at room temperature for
15 min. Finally, luminescence measurements of the ATP product were obtained using a
microplate spectrophotometer (Molecular Devices, San Jose, CA, USA). All assays were
performed in triplicate.


_3.9. Statistical Analysis_


All values shown in the figures were expressed as the mean _±_ SD obtained from at
least three independent experiments. Statistical significance was analyzed by two-tailed
Student’s _t_ -test. Data with values of _p_ < 0.05 were considered as statistically significant.


**4. Conclusions**


As the landscape of drug development evolves, becoming more intricate and expensive, it is imperative to leverage cutting-edge techniques that streamline the process. The
integration of artificial intelligence into this process offers a fast-track approach to pinpointing potential candidate compounds that might be the next therapeutic breakthroughs. The
research outlined in this study underscores the compelling advantages of such a strategy
and its efficiency in drug discovery. This study innovates drug discovery by integrating
graph convolutional networks (GCN) with molecular docking, surpassing traditional methods. GCN captures complex three-dimensional molecular structures, enhancing predictive
accuracy for binding affinities. Combined with molecular docking, it offers a more comprehensive screening, efficiently identifying potential drug candidates. It marks a significant
step forward in drug screening, potentially applicable to a wide range of molecular targets.
By deploying the graph neural network algorithm within the DeepChem library’s deep
learning module, we identified compounds that efficiently fit the active region of the target
JAK2, effectively obstructing its active site at a computational level. Several of the top
predicted drugs are recognized JAK2 inhibitors, attesting to the solidity of our methodology.
Additionally, several compounds, including ribociclib, amodiaquine, topiroxostat, and
gefitinib, previously not linked with JAK2 inhibition, exhibited a promising JAK2 inhibitory
potential. Experimental validation confirmed the deep learning and molecular docking
results. As a result, we propose these compounds as prospective novel JAK2 inhibitors. In
conclusion, a deep learning-centric approach to drug repositioning emerges as a pivotal
strategy in advancing drug discovery, not just for JAK2 inhibitors but for a broad spectrum
of therapeutic targets.


**5. Limitations**


In this study, the datasets were primarily derived from FDA-approved drugs and
the DUD-E database. While these sources are valuable, they may not fully represent the
extensive diversity of molecular structures, which could impact the generalizability of our
model. Consequently, the performance of our model might vary when applied to datasets


_Molecules_ **2024**, _29_, 1363 17 of 19


with different chemical spaces, potentially limiting its broader applicability. Future research
directions will focus on incorporating a wider range of chemical libraries to enhance dataset
diversity. Additionally, we plan to explore advanced computational algorithms to address
potential biases in the data and improve the robustness of our model. These steps are
crucial for adapting our methodology to other protein targets and assessing its utility across
diverse therapeutic areas.


**Author Contributions:** M.Y. and J.P. were involved in the experimental operation and data analysis;
E.-T.H., W.S.P., and J.-H.H. were involved in data curation and in the methodology; M.Y. and W.C.
were involved in the conceptualization, writing, reviewing, and editing of the manuscript; W.C.
confirmed the authenticity of all the raw data. All authors have read and agreed to the published
version of the manuscript.


**Funding:** Korea NRF 2021-R1A4A1031574.


**Institutional Review Board Statement:** Not applicable.


**Informed Consent Statement:** Not applicable.


**Data Availability Statement:** The data that support the findings of this study are available from the
corresponding author upon reasonable request.


**Acknowledgments:** This work was supported by the National Research Foundation of Korea (NRF)
grant funded by the Korean government.


**Conflicts of Interest:** The authors declare no conflicts of interest.


**References**


1. Jourdan, J.P.; Bureau, R.; Rochais, C.; Dallemagne, P. Drug repositioning: A brief overview. _J. Pharm. Pharmacol._ **2020**, _72_,
[1145–1151. [CrossRef]](https://doi.org/10.1111/jphp.13273)
2. Parvathaneni, V.; Kulkarni, N.S.; Muth, A.; Gupta, V. Drug repurposing: A promising tool to accelerate the drug discovery
process. _Drug Discov. Today_ **2019**, _24_ [, 2076–2085. [CrossRef]](https://doi.org/10.1016/j.drudis.2019.06.014)
3. Zhao, L.; Ciallella, H.L.; Aleksunes, L.M.; Zhu, H. Advancing computer-aided drug discovery (CADD) by big data and data-driven
machine learning modeling. _Drug Discov. Today_ **2020**, _25_ [, 1624–1638. [CrossRef]](https://doi.org/10.1016/j.drudis.2020.07.005)
4. Ramsay, R.R.; Popovic-Nikolicb, M.R.; Nikolic, K.; Uliassi, E.; Bolognesi, M.L. A perspective on multi-target drug discovery and
design for complex diseases. _Clin. Transl. Med._ **2018**, _7_ [, 3. [CrossRef]](https://doi.org/10.1186/s40169-017-0181-2)
5. Bechelane-Maia, E.H.; Assis, L.C.; Alves de Oliveira, T.; Marques da Silva, A.; Gutterres Taranto, A. Structure-based virtual
screening: From classical to artificial intelligence. _Front. Chem._ **2020**, _8_ [, 343. [CrossRef] [PubMed]](https://doi.org/10.3389/fchem.2020.00343)
6. Zhong, S.; Zhang, K.; Bagheri, M.; Burken, J.G.; Gu, A.; Li, B.; Ma, X.; Marrone, B.L.; Ren, Z.J.; Schrier, J.; et al. Machine learning:
New ideas and tools in environmental science and engineering. _Environ. Sci. Technol._ **2021**, _55_ [, 12741–12754. [CrossRef]](https://doi.org/10.1021/acs.est.1c01339)
7. Howard, J. Artificial intelligence: Implications for the future of work. _Am. J. Ind. Med._ **2019**, _62_ [, 917–926. [CrossRef]](https://doi.org/10.1002/ajim.23037)
8. Dara, S.; Dhamercherla, S.; Jadav, S.S.; Babu, C.M.; Ahsan, M.J. Machine Learning in Drug Discovery: A Review. _Artif. Intell. Rev._
**2022**, _55_ [, 1947–1999. [CrossRef] [PubMed]](https://doi.org/10.1007/s10462-021-10058-4)
9. Nag, S.; Baidya, A.T.K.; Mandal, A.; Mathew, A.T.; Das, B.; Devi, B.; Kumar, R. Deep learning tools for advancing drug discovery
and development. _3 Biotech_ **2022**, _12_ [, 110. [CrossRef] [PubMed]](https://doi.org/10.1007/s13205-022-03165-8)
10. Jordan, M.I.; Mitchell, T.M. Machine learning: Trends, perspectives, and prospects. _Science_ **2015**, _349_ [, 255–260. [CrossRef]](https://doi.org/10.1126/science.aaa8415)
11. Wei, J.; Chu, X.; Sun, X.; Xu, K.; Deng, H.; Chen, J.; Wei, Z.; Lei, M. Machine learning in materials science. _InfoMat_ **2019**, _1_, 338–358.

[[CrossRef]](https://doi.org/10.1002/inf2.12028)
12. Stephenson, N.; Shane, E.; Chase, J.; Rowland, J.; Ries, D.; Justice, N.; Zhang, J.; Chan, L.; Cao, R. Survey of machine learning
techniques in drug discovery. _Curr. Drug Metab._ **2019**, _20_ [, 185–193. [CrossRef] [PubMed]](https://doi.org/10.2174/1389200219666180820112457)
13. Raychaudhuri, S.; Cheema, K.S.; Raychaudhuri, S.K.; Raychaudhuri, S.P. Janus kinase–signal transducers and activators of
transcription cell signaling in Spondyloarthritis: Rationale and evidence for JAK inhibition. _Curr. Opin. Rheumatol._ **2021**, _33_,
[348–355. [CrossRef]](https://doi.org/10.1097/BOR.0000000000000810)
14. Sopjani, M.; Morina, R.; Uka, V.; Xuan, N.T.; Dërmaku-Sopjani, M. JAK2-mediated intracellular signaling. _Curr. Mol. Med._ **2021**,
_21_ [, 417–425. [CrossRef] [PubMed]](https://doi.org/10.2174/1566524020666201015144702)
15. Ojha, A.A.; Srivastava, A.; Votapka, L.W.; Amaro, R.E. Selectivity and ranking of tight-binding JAK-STAT inhibitors using
Markovian milestoning with Voronoi tessellations. _J. Chem. Inf. Model._ **2022**, _63_ [, 2469–2482. [CrossRef]](https://doi.org/10.1021/acs.jcim.2c01589)
16. Spiewak, T.A.; Patel, A. User’s guide to JAK inhibitors in inflammatory bowel disease. _Curr. Res. Pharmacol. Drug Discov._ **2022**,
_3_ [, 100096. [CrossRef]](https://doi.org/10.1016/j.crphar.2022.100096)


_Molecules_ **2024**, _29_, 1363 18 of 19


17. Desai, J.; Patel, B.; Gite, A.; Panchal, N.; Gite, S.; Argade, A.; Kumar, J.; Sachchidanand, S.; Bandyopadhyay, D.; Ghoshdastidar, K.;
et al. Optimisation of momelotinib with improved potency and efficacy as pan-JAK inhibitor. _Bioorganic Med. Chem. Lett._ **2022**, _66_,
[128728. [CrossRef]](https://doi.org/10.1016/j.bmcl.2022.128728)
18. Lin, C.M.; Cooles, F.A.; Isaacs, J.D. Basic mechanisms of JAK inhibition. _Mediterr. J. Rheumatol._ **2020**, _31_ [(Suppl. S1), 100. [CrossRef]](https://doi.org/10.31138/mjr.31.1.100)

[[PubMed]](https://www.ncbi.nlm.nih.gov/pubmed/32676567)
19. Furumoto, Y.; Gadina, M. The arrival of JAK inhibitors: Advancing the treatment of immune and hematologic disorders. _BioDrugs_
**2013**, _27_ [, 431–438. [CrossRef]](https://doi.org/10.1007/s40259-013-0040-7)
20. Czech, J.; Cordua, S.; Weinbergerova, B.; Baumeister, J.; Crepcia, A.; Han, L.; Maié, T.; Costa, I.G.; Denecke, B.; Maurer, A.; et al.
JAK2V617F but not CALR mutations confer increased molecular responses to interferon- α via JAK1/STAT1 activation. _Leukemia_
**2019**, _33_ [, 995–1010. [CrossRef]](https://doi.org/10.1038/s41375-018-0295-6)
21. Chen, C.; Lu, D.; Sun, T.; Zhang, T. JAK3 inhibitors for the treatment of inflammatory and autoimmune diseases: A patent review
(2016–present). _Expert Opin. Ther. Pat._ **2022**, _32_ [, 225–242. [CrossRef] [PubMed]](https://doi.org/10.1080/13543776.2022.2023129)
22. Spivak, J.L. Narrative review: Thrombocytosis, polycythemia vera, and JAK2 mutations: The phenotypic mimicry of chronic
myeloproliferation. _Ann. Intern. Med._ **2010**, _152_ [, 300–306. [CrossRef] [PubMed]](https://doi.org/10.7326/0003-4819-152-5-201003020-00008)
23. Geetha, J.P.; Arathi, C.A.; Shalini, M.; Srinivasa Murthy, A.G. JAK2 Negative Polycythemia Vera. _J. Lab. Physicians_ **2010**, _2_, 114–116.

[[PubMed]](https://www.ncbi.nlm.nih.gov/pubmed/21346910)
24. Losdyck, E.; Hornakova, T.; Springuel, L.; Degryse, S.; Gielen, O.; Cools, J.; Constantinescu, S.N.; Flex, E.; Tartaglia, M.;
Renauld, J.C.; et al. Distinct Acute Lymphoblastic Leukemia (ALL)-associated Janus Kinase 3 (JAK3) Mutants Exhibit Different
Cytokine-Receptor Requirements and JAK Inhibitor Specificities. _J. Biol. Chem._ **2015**, _290_ [, 29022–29034. [CrossRef] [PubMed]](https://doi.org/10.1074/jbc.M115.670224)
25. McLornan, D.; Percy, M.; McMullin, M.F. JAK2 V617F: A single mutation in the myeloproliferative group of disorders. _Ulst. Med._
_J._ **2006**, _75_, 112–119.
26. Hu, M.; Xu, C.; Yang, C.; Zuo, H.; Chen, C.; Zhang, D.; Shi, G.; Wang, W.; Shi, J.; Zhang, T. Discovery and evaluation of ZT55, a
novel highly-selective tyrosine kinase inhibitor of JAK2V617F against myeloproliferative neoplasms. _J. Exp. Clin. Cancer Res._
**2019**, _38_ [, 49. [CrossRef]](https://doi.org/10.1186/s13046-019-1062-x)
27. Perner, F.; Perner, C.; Ernst, T.; Heidel, F.H. Roles of JAK2 in aging, inflammation, hematopoiesis and malignant transformation.
_Cells_ **2019**, _8_ [, 854. [CrossRef]](https://doi.org/10.3390/cells8080854)
28. Mysinger, M.M.; Carchia, M.; Irwin, J.J.; Shoichet, B.K. Directory of useful decoys, enhanced (DUD-E): Better ligands and decoys
for better benchmarking. _J. Med. Chem._ **2012**, _55_ [, 6582–6594. [CrossRef]](https://doi.org/10.1021/jm300687e)
29. Zhang, Y.; Vass, M.; Shi, D.; Abualrous, E.; Chambers, J.M.; Chopra, N.; Higgs, C.; Kasavajhala, K.; Li, H.; Nandekar, P.; et al.
Benchmarking refined and unrefined AlphaFold2 structures for hit discovery. _J. Chem. Inf. Model._ **2023**, _63_ [, 1656–1667. [CrossRef]](https://doi.org/10.1021/acs.jcim.2c01219)
30. Sabe, V.T.; Ntombela, T.; Jhamba, L.A.; Maguire, G.E.; Govender, T.; Naicker, T.; Kruger, H.G. Current trends in computer aided
drug design and a highlight of drugs discovered via computational techniques: A review. _Eur. J. Med. Chem._ **2021**, _224_, 113705.

[[CrossRef]](https://doi.org/10.1016/j.ejmech.2021.113705)
31. Minnich, A.J.; McLoughlin, K.; Tse, M.; Deng, J.; Weber, A.D.; Murad, N.; Madej, B.D.; Ramsundar, B.; Rush, T.; Calad-Thomson,
S.; et al. AMPL: A data-driven modeling pipeline for drug discovery. _J. Chem. Inf. Model._ **2020**, _60_ [, 1955–1968. [CrossRef]](https://doi.org/10.1021/acs.jcim.9b01053)
32. Ramsundar, B.; Eastman, P.; Walters, P.; Pande, V. _Deep Learning for the Life Sciences: Applying Deep Learning to Genomics, Microscopy,_
_Drug Discovery, and More_ ; O’Reilly Media, Inc.: Sebastopol, CA, USA, 2019.
33. Grebner, C.; Matter, H.; Kofink, D.; Wenzel, J.; Schmidt, F.; Hessler, G. Application of deep neural network models in drug
discovery programs. _ChemMedChem_ **2021**, _16_ [, 3772–3786. [CrossRef] [PubMed]](https://doi.org/10.1002/cmdc.202100418)
34. Grebner, C.; Matter, H.; Hessler, G. _Artificial Intelligence in Drug Design_ ; Methods in Molecular Biology; Humana: New York, NY,
USA, 2022; pp. 349–382.
35. Dielschneider, R.F.; Xiao, W.; Yoon, J.-Y.; Noh, E.; Banerji, V.; Li, H.; Marshall, A.J.; Johnston, J.B.; Gibson, S.B. Gefitinib targets
ZAP-70-expressing chronic lymphocytic leukemia cells and inhibits B-cell receptor signaling. _Cell Death Dis._ **2014**, _5_, e1439.

[[CrossRef] [PubMed]](https://doi.org/10.1038/cddis.2014.391)
36. Seifert, R.; Küper, A.; Tewes, M.; Heuschmid, M.; Welt, A.; Fendler, W.P.; Herrmann, K.; Decker, T. [18F]-Fluorodeoxyglucose
positron emission tomography/CT to assess the early metabolic response in patients with hormone receptor-positive HER2negative metastasized breast cancer treated with cyclin-dependent 4/6 kinase inhibitors. _Oncol. Res. Treat._ **2021**, _44_, 400–407.

[[CrossRef] [PubMed]](https://doi.org/10.1159/000516422)
37. Doharey, P.K.; Singh, V.; Gedda, M.R.; Sahoo, A.K.; Varadwaj, P.K.; Sharma, B. In silico study indicates antimalarials as direct
inhibitors of SARS-CoV-2-RNA dependent RNA polymerase. _J. Biomol. Struct. Dyn._ **2022**, _40_ [, 5588–5605. [CrossRef] [PubMed]](https://doi.org/10.1080/07391102.2021.1871956)
38. Maghsoud, Y.; Dong, C.; Cisneros, G.A. Computational Characterization of the Inhibition Mechanism of Xanthine Oxidoreductase
by Topiroxostat. _ACS Catal._ **2023**, _13_ [, 6023–6043. [CrossRef]](https://doi.org/10.1021/acscatal.3c01245)
39. Hu, X.; Li, J.; Fu, M.; Zhao, X.; Wang, W. The JAK/STAT signaling pathway: From bench to clinic. _Signal Transduct. Target. Ther._
**2021**, _6_ [, 402. [CrossRef] [PubMed]](https://doi.org/10.1038/s41392-021-00791-1)
40. Yasir, M.; Park, J.; Han, E.-T.; Park, W.S.; Han, J.-H.; Kwon, Y.-S.; Lee, H.-J.; Hassan, M.; Kloczkowski, A.; Chun, W. Exploration
of Flavonoids as Lead Compounds against Ewing Sarcoma through Molecular Docking, Pharmacogenomics Analysis, and
Molecular Dynamics Simulations. _Molecules_ **2023**, _28_ [, 414. [CrossRef]](https://doi.org/10.3390/molecules28010414)
41. Wu, Y.; Brooks, C.L., III. Flexible CDOCKER: Hybrid Searching Algorithm and Scoring Function with Side Chain Conformational
Entropy. _J. Chem. Inf. Model._ **2021**, _61_ [, 5535–5549. [CrossRef]](https://doi.org/10.1021/acs.jcim.1c01078)


_Molecules_ **2024**, _29_, 1363 19 of 19


42. Yasir, M.; Park, J.; Han, E.-T.; Park, W.S.; Han, J.-H.; Kwon, Y.-S.; Lee, H.-J.; Chun, W. Computational Exploration of Licorice
for Lead Compounds against Plasmodium vivax Duffy Binding Protein Utilizing Molecular Docking and Molecular Dynamic
Simulation. _Molecules_ **2023**, _28_ [, 3358. [CrossRef]](https://doi.org/10.3390/molecules28083358)
43. Yasir, M.; Park, J.; Han, E.T.; Park, W.S.; Han, J.H.; Kwon, Y.S.; Lee, H.J.; Chun, W. Machine Learning-Based Drug Repositioning of
Novel Janus Kinase 2 Inhibitors Utilizing Molecular Docking and Molecular Dynamic Simulation. _J. Chem. Inf. Model._ **2023**, _63_,
[6487–6500. [PubMed]](https://www.ncbi.nlm.nih.gov/pubmed/37906702)
44. Cavallo, G.; Metrangolo, P.; Milani, R.; Pilati, T.; Priimagi, A.; Resnati, G.; Terraneo, G. The Halogen Bond. _Chem. Rev._ **2016**, _116_,
[2478–2601. [CrossRef] [PubMed]](https://doi.org/10.1021/acs.chemrev.5b00484)
45. Sanachai, K.; Mahalapbutr, P.; Choowongkomon, K.; Poo-Arporn, R.P.; Wolschann, P.; Rungrotmongkol, T. Insights into the
binding recognition and susceptibility of tofacitinib toward janus kinases. _ACS Omega_ **2020**, _5_ [, 369–377. [CrossRef]](https://doi.org/10.1021/acsomega.9b02800)
46. Studio, D.J.A. Discovery studio. _Accelrys_ **2008**, 9.
47. Pettersen, E.F.; Goddard, T.D.; Huang, C.C.; Couch, G.S.; Greenblatt, D.M.; Meng, E.C.; Ferrin, T.E. UCSF Chimera—A visualization system for exploratory research and analysis. _J. Comput. Chem._ **2004**, _25_ [, 1605–1612. [CrossRef]](https://doi.org/10.1002/jcc.20084)
48. Hassan, M.; Yasir, M.; Shahzadi, S.; Kloczkowski, A. Exploration of Potential Ewing Sarcoma Drugs from FDA-Approved
Pharmaceuticals through Computational Drug Repositioning, Pharmacogenomics, Molecular Docking, and MD Simulation
Studies. _ACS Omega_ **2022**, _7_ [, 19243–19260. [CrossRef]](https://doi.org/10.1021/acsomega.2c00518)
49. Yasir, M.; Park, J.; Han, E.-T.; Park, W.S.; Han, J.-H.; Kwon, Y.-S.; Lee, H.-J.; Hassan, M.; Kloczkowski, A.; Chun, W. Investigation
of Flavonoid Scaffolds as DAX1 Inhibitors against Ewing Sarcoma through Pharmacoinformatic and Dynamic Simulation Studies.
_Int. J. Mol. Sci._ **2023**, _24_ [, 9332. [CrossRef] [PubMed]](https://doi.org/10.3390/ijms24119332)
50. Yasir, M.; Park, J.; Lee, Y.; Han, E.T.; Park, W.S.; Han, J.H.; Kwon, Y.S.; Lee, H.J.; Chun, W. Discovery of GABA Aminotransferase
Inhibitors via Molecular Docking, Molecular Dynamic Simulation, and Biological Evaluation. _Int. J. Mol. Sci._ **2023**, _24_, 16990.

[[CrossRef] [PubMed]](https://doi.org/10.3390/ijms242316990)


**Disclaimer/Publisher’s Note:** The statements, opinions and data contained in all publications are solely those of the individual
author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to
people or property resulting from any ideas, methods, instructions or products referred to in the content.


