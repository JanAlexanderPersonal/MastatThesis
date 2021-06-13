
#!/bin/bash
# run the data preprocessor



python3.8 /root/space/code/DataPreparation/prepare_USiegen.py \
        --source /root/space/data/USiegen/ \
        --output /root/space/temp/dataset_2/ \
        --dimension 2



python3.8 /root/space/code/DataPreparation/prepare_MyoSegmenTUM.py \
        --source /root/space/data/OSF_Sarah_Schlaeger/ \
        --output /root/space/temp/dataset_2/ \
        --dimension 2



python3.8 /root/space/code/DataPreparation/prepare_xVertSeg.py \
        --source /root/space/data/xVertSeg/Data1/ \
        --output /root/space/temp/dataset_2/ \
        --dimension 2

python3.8 /root/space/code/DataPreparation/prepare_USiegen.py \
        --source /root/space/data/USiegen/ \
        --output /root/space/temp/dataset_1/ \
        --dimension 1



python3.8 /root/space/code/DataPreparation/prepare_MyoSegmenTUM.py \
        --source /root/space/data/OSF_Sarah_Schlaeger/ \
        --output /root/space/temp/dataset_1/ \
        --dimension 1



python3.8 /root/space/code/DataPreparation/prepare_xVertSeg.py \
        --source /root/space/data/xVertSeg/Data1/ \
        --output /root/space/temp/dataset_1/ \
        --dimension 1

python3.8 /root/space/code/DataPreparation/prepare_USiegen.py \
        --source /root/space/data/USiegen/ \
        --output /root/space/temp/dataset_0/ \
        --dimension 0



python3.8 /root/space/code/DataPreparation/prepare_MyoSegmenTUM.py \
        --source /root/space/data/OSF_Sarah_Schlaeger/ \
        --output /root/space/temp/dataset_0/ \
        --dimension 0



python3.8 /root/space/code/DataPreparation/prepare_xVertSeg.py \
        --source /root/space/data/xVertSeg/Data1/ \
        --output /root/space/temp/dataset_0/ \
        --dimension 0

# Contrast enhancement preprocessing



python3.8 /root/space/code/DataPreparation/prepare_USiegen.py \
        --source /root/space/data/USiegen/ \
        --output /root/space/temp/dataset_2_contrast_1/ \
        --contrast 1 \
        --dimension 2



python3.8 /root/space/code/DataPreparation/prepare_MyoSegmenTUM.py \
        --source /root/space/data/OSF_Sarah_Schlaeger/ \
        --output /root/space/temp/dataset_2_contrast_1/ \
        --contrast 1 \
        --dimension 2



python3.8 /root/space/code/DataPreparation/prepare_xVertSeg.py \
        --source /root/space/data/xVertSeg/Data1/ \
        --output /root/space/temp/dataset_2_contrast_1/ \
        --contrast 1 \
        --dimension 2


## Contrast 2

python3.8 /root/space/code/DataPreparation/prepare_USiegen.py \
        --source /root/space/data/USiegen/ \
        --output /root/space/temp/dataset_2_contrast_2/ \
        --contrast 2 \
        --dimension 2



python3.8 /root/space/code/DataPreparation/prepare_MyoSegmenTUM.py \
        --source /root/space/data/OSF_Sarah_Schlaeger/ \
        --output /root/space/temp/dataset_2_contrast_2/ \
        --contrast 2 \
        --dimension 2



python3.8 /root/space/code/DataPreparation/prepare_xVertSeg.py \
        --source /root/space/data/xVertSeg/Data1/ \
        --output /root/space/temp/dataset_2_contrast_2/ \
        --contrast 2 \
        --dimension 2

## Contrast 3

python3.8 /root/space/code/DataPreparation/prepare_USiegen.py \
        --source /root/space/data/USiegen/ \
        --output /root/space/temp/dataset_2_contrast_3/ \
        --contrast 3 \
        --dimension 2



python3.8 /root/space/code/DataPreparation/prepare_MyoSegmenTUM.py \
        --source /root/space/data/OSF_Sarah_Schlaeger/ \
        --output /root/space/temp/dataset_2_contrast_3/ \
        --contrast 3 \
        --dimension 2



python3.8 /root/space/code/DataPreparation/prepare_xVertSeg.py \
        --source /root/space/data/xVertSeg/Data1/ \
        --output /root/space/temp/dataset_2_contrast_3/ \
        --contrast 3 \
        --dimension 2

## Contrast 4

python3.8 /root/space/code/DataPreparation/prepare_USiegen.py \
        --source /root/space/data/USiegen/ \
        --output /root/space/temp/dataset_2_contrast_4/ \
        --contrast 4 \
        --dimension 2



python3.8 /root/space/code/DataPreparation/prepare_MyoSegmenTUM.py \
        --source /root/space/data/OSF_Sarah_Schlaeger/ \
        --output /root/space/temp/dataset_2_contrast_4/ \
        --contrast 4 \
        --dimension 2



python3.8 /root/space/code/DataPreparation/prepare_xVertSeg.py \
        --source /root/space/data/xVertSeg/Data1/ \
        --output /root/space/temp/dataset_2_contrast_4/ \
        --contrast 4 \
        --dimension 2