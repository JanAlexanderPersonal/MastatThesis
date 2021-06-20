#!/bin/bash


PATH_OP=/root/space/output

for c in 0 3

do
    for d in 2 1
    do
        
        python3.8 trainval.py -e weakly_spine_xVertSeg_c6 -sb $PATH_OP/results_weighted_xVertSeg_${d}_contrast_$c -d $PATH_OP/dataset_${d}_contrast_$c/ -tb $PATH_OP/tensorboard_xVertSeg_d_${d}_c_${c} -r 0
        
    done
done


