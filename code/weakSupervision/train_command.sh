#!/bin/bash


for c in 0 1 3

do
    for d in 1 2
    do
        python3.8 trainval.py -e weakly_spine_dataset_c6 -sb /root/space/temp/results_dataset_${d}_contrast_$c -d /root/space/temp/dataset_${d}_contrast_$c/ -tb /root/space/temp/tensorboard_d_${d}_c_${c} -r 0
        python3.8 trainval.py -e weakly_spine_dataset_c6_weighted -sb /root/space/temp/results_dataset_${d}_contrast_$c -d /root/space/temp/dataset_${d}_contrast_$c/ -tb /root/space/temp/tensorboard_d_${d}_c_${c} -r 0
    done
done
