#!/bin/bash


PATH_OP=/root/space/output
python3.8 reconstruct.py -mn results_full_dataset_D_contrast_C -sd $PATH_OP/reconstruct_full -gt $PATH_OP/dataset_1_contrast_3 -ed Combine_full
python3.8 reconstruct.py -mn results_pseudo -sd $PATH_OP/reconstruct_pseudo -gt $PATH_OP/dataset_1_contrast_3 -ed Combine_pseudo