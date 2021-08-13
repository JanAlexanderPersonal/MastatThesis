#!/bin/bash


PATH_OP=/root/space/output
python3.8 reconstruct.py -mn results_precalc_dataset_D_contrast_3  -sd $PATH_OP/reconstruct_from_precalc -gt $PATH_OP/dataset_1_contrast_3 -ed Combine_one_stack
python3.8 reconstruct.py -mn results_weighted_dataset_D_contrast_C -sd $PATH_OP/reconstruct_1 -gt $PATH_OP/dataset_1_contrast_3 -ed test_1