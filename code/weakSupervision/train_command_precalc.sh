#!/bin/bash
# run the data preprocessor

# Contrast enhancement preprocessing

PATH_CODE=/root/space/code/DataPreparation
PATH_DATA=/root/space/data
PATH_OP=/root/space/output
PATH_OP=/root/space/output

for c in 3

do
        for d in 0 1 2
        do

                python3.8 $PATH_CODE/prepare_PLoS.py \
                        --source $PATH_DATA/Zenodo/ \
                        --output $PATH_OP/dataset_minimal_${d}_contrast_$c/ \
                        --contrast $c \
                        --dimension $d

                python3.8 $PATH_CODE/prepare_USiegen.py \
                        --source $PATH_DATA/USiegen/ \
                        --output $PATH_OP/dataset_minimal_${d}_contrast_$c/ \
                        --contrast $c \
                        --dimension $d



                python3.8 $PATH_CODE/prepare_MyoSegmenTUM.py \
                        --source $PATH_DATA/OSF_Sarah_Schlaeger/ \
                        --output $PATH_OP/dataset_minimal_${d}_contrast_$c/ \
                        --contrast $c \
                        --dimension $d



                python3.8 $PATH_CODE/prepare_xVertSeg.py \
                        --source $PATH_DATA/xVertSeg/Data1/ \
                        --output $PATH_OP/dataset_minimal_${d}_contrast_$c/ \
                        --contrast $c \
                        --dimension $d

                

        done
done

python3.8 $PATH_CODE/unify_point_files.py \
        --source $PATH_OP \
        --foldername dataset_minimal_D_contrast_3 \
        --dimension 2



c=3

rm -rf $PATH_OP/tensorboard_full_d_*

for d in 0 
do
    python3.8 trainval_precalculated_points.py -e single_class -sb $PATH_OP/results_precalc_dataset_minimal_${d}_contrast_${c} -d $PATH_OP/dataset_minimal_${d}_contrast_$c/ -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} -r 0
    
done

for d in 1 2
do
    python3.8 trainval_precalculated_points.py -e precalc -sb $PATH_OP/results_precalc_dataset_minimal_${d}_contrast_${c} -d $PATH_OP/dataset_minimal_${d}_contrast_$c/ -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} -r 0
    
done

rm -rf $PATH_OP/tensorboard_full_d_*

for d in 0 
do
    
    python3.8 trainval_Individual_precalculated_points.py -e single_class -sb $PATH_OP/results_precalc_dataset_minimal_${d}_contrast_${c} -d $PATH_OP/dataset_minimal_${d}_contrast_$c/ -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} -r 0
done

for d in 1 2
do
    
    python3.8 trainval_Individual_precalculated_points.py -e precalc -sb $PATH_OP/results_precalc_dataset_minimal_${d}_contrast_${c} -d $PATH_OP/dataset_minimal_${d}_contrast_$c/ -tb $PATH_OP/tensorboard_full_d_${d}_c_${c} -r 0
done

rm -rf $PATH_OP/tensorboard_full_d_*