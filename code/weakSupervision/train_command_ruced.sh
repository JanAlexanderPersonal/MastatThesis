#!/bin/bash


PATH_OP=/root/space/output

for c in 3 

do
    for d in 2 1 0
    do
        
        python3.8 trainval.py -e weakly_spine_dataset_c6_weighted_red -sb $PATH_OP/results_weighted_dataset_${d}_contrast_$c -d $PATH_OP/dataset_${d}_contrast_$c/ -tb $PATH_OP/tensorboard_d_${d}_c_${c} -r 0
        
    done
done


