#!/bin/bash
# run the data preprocessor

# Contrast enhancement preprocessing

PATH_CODE=/root/space/code/DataPreparation
PATH_DATA=/root/space/data
PATH_OP=/root/space/output
PATH_OP=/root/space/output





c=3



rm -rf $PATH_OP/tensorboard_full_d_*


for d in 2 1
    do
    
     python3.8 trainval_Individual_precalculated_points.py -e precalc -sb $PATH_OP/results_precalc_dataset_${d}_contrast_${c} -d $PATH_OP/dataset_${d}_contrast_$c/ -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} -r 0
     #python3.8 trainval_Individual_precalculated_points.py -e precalc_noSep -sb $PATH_OP/results_precalc_dataset_${d}_contrast_${c} -d $PATH_OP/dataset_${d}_contrast_$c/ -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} -r 0
 done

rm -rf $PATH_OP/tensorboard_full_d_*