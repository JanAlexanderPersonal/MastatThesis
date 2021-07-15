#!/bin/bash


PATH_OP=/root/space/output
c=3

for d in 2
do
    
    python3.8 trainval.py -e full_spine_dataset_c6_weighted -sb $PATH_OP/results_full_dataset_${d}_contrast_$c -d $PATH_OP/dataset_${d}_contrast_$c/ -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} -r 0

    
done


for d in 2 
do
    
    python3.8 trainval.py -e selected -sb $PATH_OP/results_weighted_dataset_${d}_contrast_${c} -d $PATH_OP/dataset_${d}_contrast_$c/ -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} -r 0

    
done

