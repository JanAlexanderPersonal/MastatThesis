
#!/bin/bash
# run the data preprocessor

# Contrast enhancement preprocessing

PATH_CODE=/root/space/code/DataPreparation
PATH_DATA=/root/space/data
PATH_OP=/root/space/output

for c in 0 1 3

do
        for d in 0 1 2
        do

                python3.8 $PATH_CODE/prepare_USiegen.py \
                        --source $PATH_DATA/USiegen/ \
                        --output $PATH_OP/dataset_${d}_contrast_$c/ \
                        --contrast $c \
                        --dimension $d


        done
done