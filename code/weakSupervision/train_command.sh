#!/bin/bash

python3.8 trainval.py -e weakly_spine_dataset_c6 -sb /root/space/temp/haven_raw -d /root/space/temp/dataset_2/ -tb /root/space/temp/tensorboard -r 1

python3.8 trainval.py -e weakly_spine_dataset_c6 -sb /root/space/temp/haven_enhanced -d /root/space/temp/dataset_contrast_2/ -tb /root/space/temp/tensorboard -r 1