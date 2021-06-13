
#!/bin/bash
# run the data preprocessor

# Contrast enhancement preprocessing

for c in 0 1 3

do
        for d in 0 1 2
        do

                python3.8 /root/space/code/DataPreparation/prepare_USiegen.py \
                        --source /root/space/data/USiegen/ \
                        --output /root/space/temp/dataset_${d}_contrast_$c/ \
                        --contrast $c \
                        --dimension $d



                python3.8 /root/space/code/DataPreparation/prepare_MyoSegmenTUM.py \
                        --source /root/space/data/OSF_Sarah_Schlaeger/ \
                        --output /root/space/temp/dataset_${d}_contrast_$c/ \
                        --contrast $c \
                        --dimension $d



                python3.8 /root/space/code/DataPreparation/prepare_xVertSeg.py \
                        --source /root/space/data/xVertSeg/Data1/ \
                        --output /root/space/temp/dataset_${d}_contrast_$c/ \
                        --contrast $c \
                        --dimension $d

        done
done