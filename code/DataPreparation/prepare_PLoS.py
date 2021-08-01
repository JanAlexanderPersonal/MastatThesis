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
from tqdm import tqdm
import json
import sys
sys.path.append('/root/space/code/utils/')
import utils as ut
from PIL import Image

from shutil import copyfile

from pathlib import Path

logging.basicConfig(level=logging.WARNING)

##############################
# ├── Img_01_Labels.nii
# ├── Img_01.nii
# ├── Img_02_Labels.nii
# ├── Img_02.nii
# ├── Img_03_Labels.nii
# ├── Img_03.nii
# ├── Img_04_Labels.nii
# 
# This code will perform the following tasks:
# • Load the image or the mask
# • resample the image or mask on an isotropic grid: 1 mm × 1 mm × 1 mm  (using nearest neighbor interpollation)
# • images will be rescaled to floats in [0. , 1.]
# • masks values will be converted from 200 to 240 to 1 -> 5 to encode L1 to L5
# • The individual 2D slices are stored both as .jpg files and as pickle files to allow fast access by pyTorch.
#############################

# Todo: use the semantic segments to make instance segments --> just number starting from the top

def arrange_axis(arr):
    arr = np.rot90(arr, k=2, axes=(0, 2))
    return arr

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='xVertSeg preparation')
    parser.add_argument('--source', type=str, default='./spine_volumes/Zenodo', help='root path of PLoS dataset')
    parser.add_argument('--output', type=str, default='./dataset', help='output path for the dataset')
    parser.add_argument('--dimension', type=int, default=0, help='dimension along which to slice the volumes, converted to xVertSeg dimensions')
    parser.add_argument('--contrast', type=int, default=0, help='Contrast enhancement bool?')
    args = parser.parse_args()

    source_filedir = os.path.abspath(args.source)
    dim_slice = args.dimension
    output_filedir = os.path.abspath(args.output)
    image_slices_filedir = os.path.join(output_filedir, 'PLoS_images')
    mask_slices_filedir = os.path.join(output_filedir, 'PLoS_masks')

    # make sure the output folder exists
    Path(output_filedir).mkdir(parents=True, exist_ok=True)
    Path(image_slices_filedir).mkdir(parents=True, exist_ok=True)
    Path(mask_slices_filedir).mkdir(parents=True, exist_ok=True)


    # Convert the images to a set of slices:
    logging.info('Start copy of image files')

    dataset_min = np.infty
    dataset_max = -np.infty

    filenames_dict = dict()
    dimensions_dict = dict()

    for nr in tqdm(range(1,23)):
        filename = f'Img_{nr:02d}.nii'
        logging.debug(f'read file {filename}')
        arr, minimum, maximum = ut.array_from_file(os.path.join(source_filedir, filename))
        arr = arrange_axis(arr)
        dimensions_dict[filename] = {'image' : arr.shape}
        dataset_min = min(dataset_min, minimum)
        dataset_max = max(dataset_max, maximum)

        logging.debug(f'min : {np.min(arr):1.5f} ** max : {np.max(arr):1.5f}')
        logging.debug(f'source : {filename}, shape {arr.shape}')

        fn = os.path.join(image_slices_filedir, f'image{nr:03d}')
        Path(fn).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(fn, 'image_array'), arr)

        # For each slice along the asked dimension, convert to a numpy.ndarray and save this.
        # Preprocessing the slices before loading into pyTorch should speed up the training in the end.
        ut.arr_slices_save(arr, dim_slice, fn, args.contrast, save_jpeg = True)
        

    # Process the mask files and change the filenames
    logging.info('start copy of mask files')
    unique_values = []
    for nr in tqdm(range(1,23)):
        filename = f'Img_{nr:02d}_Labels.nii'

        target_folder = os.path.join(mask_slices_filedir, f'image{nr:03d}')
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        
        logging.debug(f'target : {target_folder}')

        # Get mask, resample to 1 mm × 1 mm × 1 mm grid and extract np_array from this

        arr =  sitk.GetArrayFromImage( ut.resampler( sitk.ReadImage(os.path.join(source_filedir, filename)) ) ) 
        arr = arrange_axis(arr)
        dimensions_dict[f'Img_{nr:02d}.nii']['mask'] = arr.shape
        unique_values += np.unique(arr).tolist()
        logging.debug(f'source : {filename}, shape {arr.shape}')
        logging.debug(f'min : {np.min(arr)} ** max : {np.max(arr)}')

        np.save(os.path.join(target_folder, 'mask_array'), arr)
        ut.mask_to_slices_save(arr, dim_slice, target_folder)

    logging.info(f'List of unique values in the masks : {sorted(list(set(unique_values)))}')
    logging.info(f'min and max values in the complete dataset : {dataset_min} & {dataset_max}.')

    with open(os.path.join(output_filedir, 'mask_counts_PLoS.json'), 'w') as mask_counts_file:
        json.dump(unique_values, mask_counts_file)

    with open(os.path.join(output_filedir, 'dimensions_PLoS.json'), 'w') as mask_dim_file:
        json.dump(dimensions_dict, mask_dim_file)

    with open(os.path.join(output_filedir, 'filenames_PLoS.json'), 'w') as filenames_file:
        json.dump({int(key) : (value.split('.')[0]).lower() for key, value in filenames_dict.items()}, filenames_file)
