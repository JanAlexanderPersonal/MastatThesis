
#!/bin/bash
# run the data preprocessor

# Contrast enhancement preprocessing

PATH_CODE=/root/space/code/DataPreparation
PATH_DATA=/root/space/data
PATH_OP=/root/space/output

for c in 3

do
        for d in 2
        do

                python3.8 $PATH_CODE/prepare_UWash.py \
                        --source $PATH_DATA/UWSpineCT-selected/ \
                        --output $PATH_OP/uWash/ \
                        --contrast $c \
                        --dimension $d


        done
done