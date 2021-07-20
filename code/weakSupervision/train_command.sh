#!/bin/bash


PATH_OP=/root/space/output
c=3

for d in 0 
do
    
    python3.8 trainval.py -e single_class -sb $PATH_OP/results_weighted_dataset_${d}_contrast_${c} -d $PATH_OP/dataset_${d}_contrast_$c/ -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} -r 0

    
done


#for d in 1 
#do
    
#    python3.8 trainval.py -e extra -sb $PATH_OP/results_weighted_dataset_${d}_contrast_${c} -d $PATH_OP/dataset_${d}_contrast_$c/ -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} -r 0

    
#done


