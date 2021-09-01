# Mastat Thesis

***
**What is this repository?**

__Title__: Weakly supervised segmentation of the human spine

__Institution__: Universiteit Gent, faculty of Science

__Name:__ Jan Alexander

__Contact:__ jan.alexander@ugent.be

In this repository, I keep my work for my master thesis in partial fulfillment for the degree of _Master in Statistical Data Analysis_.

***

## Thesis objective

Segmentation of CT-scan & MRI-scan volumes of the human lumbar spine.
Model training based on point-annotated ground truth data. 

## Thesis results

Medical professionals use MRI or CT scans as essential com-
ponents for diagnosis and planning of procedures. There is a
trend towards machine vision to support medical profession-
als to interpret these images. This research investigates tech-
niques to reduce the dataset labelling cost for this application
by working with point annotation instead of full annotation.
Based on publicly available datasets this work demonstrate
two new loss components and a combination technique of
different model results to generate pseudo masks. As a final
result, one can obtain 72 % of the inversely weighted dice
score performance of a fully annotated model at an estimated
12 % of the labelling cost.

## Content of this repository

* _code:_ The code with which the results of this thesis can be repeated.
* _text:_ LaTex code for thesis document 
* _dockerfiles:_ environment to run the code in + latex environment
* _Data:_ Information on the datasets used.

## Most important code files to consider:

Build the docker with *dockerfiles/code/build_torch.sh*.
This requires you to have access to a CUDA-enabled GPU.


Run this docker with *code/container_run_segment.sh*. Once you are in the docker, the data preprocessing can be done with the _code/DataPreparation/prepare_dataset.sh_ script.

After the preprocessing, the training of specific models can be done with the code in *code/weakSupervision/train_command.sh*. The specific model to train is defined in *code/weakSupervision/exp_configs/weakly_spine.py*.


In similar fashion, the model results to be combined are defined with *code/weakSupervision/exp_configs/reconstruct_dicts.py* and the code to call this is *code/weakSupervision/reconstruct.sh*.


In brief:
* _Model training code:_ *code/weakSupervision/trainval.py*
* _Model itself:_ *code/weakSupervision/src/models/instseg.py*
* _Combination algorithm:_ *code/weakSupervision/src/models/multi_dim_reconstruction.py*
