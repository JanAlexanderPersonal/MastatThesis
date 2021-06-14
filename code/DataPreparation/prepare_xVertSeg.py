"""
author :    Jan Alexander
project:    Master thesis
date:       2021-01-31


Script to prepare the xVertSeg dataset for processing.

Intended working: The xVertSeg folder *Data* has following structure
        
Data
├── images
│   ├── image001.mhd
│   ├── image001.raw
│   ├── image002.mhd
│   ├── image002.raw
│   ├── image003.mhd
│   ├── image003.raw
│   ├── image004.mhd
├── masks
│   ├── mask001.mhd
│   ├── mask001.raw
│   ├── mask002.mhd
│   ├── mask002.raw
│   ├── mask003.mhd
│   ├── mask003.raw
│   ├── mask004.mhd
└── scores
    └── scores.csv

This script intends to make this into the following:

dataset
   ├── masks
   ├── images


This script does nothing more than copying files from the source folder to the dataset folder.

"""

import os
import re
import argparse
import logging
import SimpleITK as sitk
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

import sys
sys.path.append('/root/space/code/utils/')
import utils as ut

from PIL import Image
from skimage import exposure
from skimage.restoration import denoise_bilateral

import json

from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

##############################
# xVertSeg data structure:
# .
# ├── Data1
# │   ├── images
# │   │   ├── image001.mhd
# │   │   ├── image001.raw
# │   │   ├── image002.mhd
# │   │   ├── image002.raw
# │   │   ├── image003.mhd
# │   │   ├── 
# │   ├── masks
# │   │   ├── mask001.mhd
# │   │   ├── mask001.raw
# │   │   ├── mask002.mhd
# │   │   ├── mask002.raw
# │   │   ├── mask003.mhd
# │   │   ├── mask003.raw
# │   │   ├── 
# │   └── scores
# │       └── scores.csv
# └── tree.txt
# 
# This code will perform the following tasks:
# • Load the image or the mask
# • resample the image or mask on an isotropic grid: 1 mm × 1 mm × 1 mm  (using nearest neighbor interpollation)
# • images will be rescaled to floats in [0. , 1.]
# • masks values will be converted from 200 to 240 to 1 -> 5 to encode L1 to L5
# • The individual 2D slices are stored both as .jpg files and as pickle files to allow fast access by pyTorch.
#############################

# The following regex will parse "mask002.raw" to "mask002" "002" "raw"
MASK_NR = re.compile(r'(^[a-zA-Z]+(\d+)).(\w+)')

# The labels in the xVertSeg dataset follow (http://lit.fe.uni-lj.si/xVertSeg/database.php)
#   L1 : 200
#   L2 : 210
#   ...
#   L5 : 240
# This will be converted to labels 1 -> 5 representing L1 -> L5.

XVERTSEG_ENCODING = {200 + (i-1) * 10 : i for i in range(1,6)}

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='xVertSeg preparation')
    parser.add_argument('--source', type=str, default='./spine_volumes/xVertSeg/Data1', help='root path of xVertSeg dataset')
    parser.add_argument('--output', type=str, default='./dataset', help='output path for the dataset')
    parser.add_argument('--dimension', type=int, default=0, help='dimension along which to slice the volumes')
    parser.add_argument('--contrast', type=int, default=0, help='Contrast enhancement bool?')
    args = parser.parse_args()

    image_filedir = os.path.join(args.source, 'images')
    mask_filedir = os.path.join(args.source, 'masks')
    contrast_enhance = (args.contrast > 0)
    dim_slice = args.dimension
    output_filedir = os.path.abspath(args.output)
    image_slices_filedir = os.path.join(output_filedir, 'xVertSeg_images')
    mask_slices_filedir = os.path.join(output_filedir, 'xVertSeg_masks')

    # make sure the output folder exists
    Path(output_filedir).mkdir(parents=True, exist_ok=True)
    Path(image_slices_filedir).mkdir(parents=True, exist_ok=True)
    Path(mask_slices_filedir).mkdir(parents=True, exist_ok=True)

    # Prepare the intensity rescaler
    rescale = sitk.RescaleIntensityImageFilter()
    min_max = sitk.MinimumMaximumImageFilter()

    dataset_min = np.infty
    dataset_max = -np.infty

    # Convert the images to a set of slices:
    logging.info('Start copy of image files')

    dimensions_dict = dict()


    for nr, filename in enumerate(os.listdir(image_filedir)):

        if filename.endswith('.raw'):
            continue

        logging.debug(filename)

        arr, minimum, maximum = ut.array_from_file(os.path.join(image_filedir, filename))

        dimensions_dict[filename] = {'image' : arr.shape}
        
        dataset_min = min(dataset_min, minimum)
        dataset_max = max(dataset_max, maximum)

        logging.debug(f'min : {np.min(arr)} ** max : {np.max(arr):1.5f}')
        logging.debug(f'source : {filename}, shape {arr.shape}')

        # For each slice along the asked dimension, convert to a numpy.ndarray and save this.
        # Preprocessing the slices before loading into pyTorch should speed up the training in the end.

        fn = os.path.join(image_slices_filedir, filename.split('.')[0])
        Path(fn).mkdir(parents=True, exist_ok=True)

        ut.arr_slices_save(arr, dim_slice, fn, args.contrast, save_jpeg = True)

    # Process the mask files and change the filenames
    logging.info('start copy of mask files')
    unique_values = dict()
    for filename in tqdm(os.listdir(mask_filedir), desc = "Mask files"):
        if filename.endswith('.raw'):
            continue
        logging.debug(f'filename : {filename}')
        ms = MASK_NR.findall(filename)[0]
        source_filename = os.path.join(mask_filedir, filename)
        target_folder = os.path.join(mask_slices_filedir, f'image{ms[1]}')
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        
        logging.debug(f'source : {source_filename}')
        logging.debug(f'target : {target_folder}')

        image =  sitk.ReadImage(source_filename)
        # No rescaling -> mask will be transformed with XVERTSEG_ENCODING

        # resample on isotropic 1 mm × 1 mm × 1 mm grid
        image = ut.resampler(image)
        arr = sitk.GetArrayFromImage(image)

        # The labels in the xVertSeg dataset follow (http://lit.fe.uni-lj.si/xVertSeg/database.php)
        #   L1 : 200
        #   L2 : 210
        #   ...
        #   L5 : 240
        # This will be converted to labels 1 -> 5 representing L1 -> L5.
        # The background will be encoded as value 0
        for encoding, vert  in XVERTSEG_ENCODING.items():
            arr[arr==encoding] = vert

        dimensions_dict[filename]['mask'] = arr.shape

        vals, counts = np.unique(arr, return_counts=True)
        for val, count in zip(vals.tolist(), counts.tolist()):
            unique_values[val] = unique_values.get(val, 0) + count
        logging.debug(f'source : {filename}, shape {arr.shape}')
        logging.debug(f'min : {np.min(arr)} ** max : {np.max(arr)}')
        ut.mask_to_slices_save(arr, dim_slice, target_folder)

    logging.info(f'List of unique values in the masks : {unique_values}')
    logging.info(f'min and max values in the complete dataset : {dataset_min} & {dataset_max}.')

    with open(os.path.join(output_filedir, 'mask_counts_xVertSeg.json'), 'w') as mask_counts_file:
        json.dump(unique_values, mask_counts_file)

    with open(os.path.join(output_filedir, 'dimensions_xVertSeg.json'), 'w') as mask_dim_file:
        json.dump(dimensions_dict, mask_dim_file)