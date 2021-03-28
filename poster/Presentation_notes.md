# Presentation notes for poster presentation

Jan Alexander  
31/03/2021

## General intro

Jan Alexander, masterstudent Master in Statistical Data analysis.

Master thesis subject : Machine vision for a medical applications.
An instance segmentation model for the lumbar vertebrae of the human spine in CT (_computed tomography_) images, based on weakly supervised data.

The lumbar vertebrae are the 5 spinal vertebrae in the lower back.

Instance segmentation indicates that each voxel is classified as one of the 5 lumbar vertebrae, indicated by L1 to L5, or as _background_ everything not a vertebrae is considered background in this problem.
The objective is to obtain full segmentation masks for all 5 vertebrae.

## Problem motivation:

First I would like to motivate the two subjects of my master thesis subject:

### CT scan segmentation

Segmentation of the human spine from CT scan images is a useful support for spine pathology diagnosis and as a support for planning and performing a surgical interventions on the spine.
During a surgical intervention, frequent imaging can be necessary. In this situation, it is useful to support the surgeon maximally in interpreting these images.

The illustration shows an artists impression of the start of a _micro-discectomy_ to treat a _herniated spinal disc_. The first dilator for the _laparoscopy_ is already inserted.

### Weakly supervised learning

The classical approach for training a deep learning network requires a dataset of fully labeled data.
In the case one wants to train an instance segmentation mask, this requires a labelled dataset with instance masks for all classes.

This means that an expert has to delineate annotation masks for all slices of all images in the dataset to generate per-pixel labels.
Delineation of 1 CT-scan with 250 slices requires a budget of 400 minutes. This cost becomes prohibitive very fast.

One might therefor question wether this approach is the most efficient.
The idea behind weak supervision is to leverage cheaper, less informative labels for the task at hand.

There are various weak supervision methods. Every label type that is _less informative_ than the desired end result of the model is considered weak supervision.
Literature regarding weakly supervised deep learning for segmentation has mostly focussed on optical camera images such as the _COCO_ dataset or _PASCAL VOC 2012_.
Several approaches have been investigated by various authors ranging:

*   Image level labels: only the object classes present in the picture are provided. 
*   bounding box labels: When used to train a model that outputs bounding boxes, this is a strong label, but these can be leveraged to train a segmentation network.
*   Point labels or squiggles: Very fast to mark.

According to Bearman, point level labels are 10 times less time consuming than full delineations

## Previous work

