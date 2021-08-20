#!/bin/bash


PATH_OP=/root/space/output

# python3.8 reconstruct.py    -mn results_precalc_dataset_D_contrast_3 \
#                             -sd $PATH_OP/reconstruct_from_precalc_MyoSegmenTUM \
#                             -gt $PATH_OP/dataset_1_contrast_3 \
#                             -ed Combine_one_stack_MyoSegmenTUM \
#                             -pd $PATH_OP/dataset_2_pseudo_MyoSegmenTUM \
#                             -ss MyoSegmenTUM


# python3.8 reconstruct.py    -mn results_precalc_dataset_D_contrast_3 \
#                             -sd $PATH_OP/reconstruct_from_precalc_USiegen \
#                             -gt $PATH_OP/dataset_1_contrast_3 \
#                             -ed Combine_one_stack_USiegen \
#                             -pd $PATH_OP/dataset_2_pseudo_USiegen \
#                             -ss USiegen


python3.8 reconstruct.py    -mn results_precalc_dataset_D_contrast_3 \
                            -sd $PATH_OP/reconstruct_from_precalc_xVertSeg \
                            -gt $PATH_OP/dataset_1_contrast_3 \
                            -ed Combine_one_stack_xVertSeg \
                            -pd $PATH_OP/dataset_2_pseudo_xVertSeg \
                            -ss xVertSeg


python3.8 trainval_separate_source.py   -e full_spine_dataset_c6_separate_source \
                                        -sb $PATH_OP/results_pseudo_MyoSegmenTUM \
                                        -d $PATH_OP/dataset_2_pseudo_MyoSegmenTUM/ \
                                        -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} \
                                        -r 0 \
                                        -ss MyoSegmenTUM



python3.8 trainval_separate_source.py   -e full_spine_dataset_c6_separate_source \
                                        -sb $PATH_OP/results_pseudo_xVertSeg \
                                        -d $PATH_OP/dataset_2_pseudo_xVertSeg/ \
                                        -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} \
                                        -r 0 \
                                        -ss xVertSeg


python3.8 trainval_separate_source.py   -e full_spine_dataset_c6_separate_source \
                                        -sb $PATH_OP/results_pseudo_USiegen \
                                        -d $PATH_OP/dataset_2_pseudo_USiegen/ \
                                        -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} \
                                        -r 0 \
                                        -ss USiegen
