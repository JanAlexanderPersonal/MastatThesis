#!/bin/bash


PATH_OP=/root/space/output

python3.8 reconstruct.py -e weakly_spine_dataset_c6_weighted -mn results_weighted_dataset_D_contrast_C -sd $PATH_OP/3D_reconstruct_test -gt $PATH_OP/dataset_1_contrast_3