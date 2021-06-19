"""
author :    Jan Alexander
project:    Master thesis
date:       2021-01-31


Script to prepare the MyoSegmenTUM dataset for processing.

Intended working: The MyoSegmenTUM folder *Data* has following structure
        
MyoSegmenTUM data structure:
├── 01
│   ├── 01
│   │   ├── muscle
│   │   │   ├── B0.dcm
│   │   │   ├── erector_spinae_left_01.mha
│   │   │   ├── erector_spinae_right_01.mha
│   │   │   ├── FAT.dcm
│   │   │   ├── FATFRACTION.dcm
│   │   │   ├── psoas_left_01.mha
│   │   │   ├── psoas_right_01.mha
│   │   │   ├── T2star.dcm
│   │   │   └── WATER.dcm
│   │   └── vertebrae
│   │       ├── B0.dcm
│   │       ├── FAT.dcm
│   │       ├── FATFRACTION.dcm
│   │       ├── L1_01.mha
│   │       ├── L2_01.mha
│   │       ├── L3_01.mha
│   │       ├── L4_01.mha
│   │       ├── L5_01.mha
│   │       ├── T2star.dcm
│   │       └── WATER.dcm
│   └── __MACOSX
│       └── 01
│           ├── muscle
│           └── vertebrae
├── 01.zip
├── 02
│   ├── 02
│   │   ├── muscle
│   │   │   ├── B0.dcm
│   │   │   ├── erector_spinae_left_02.mha
│   │   │   ├── erector_spinae_right_02.mha
│   │   │   ├── FAT.dcm
│   │   │   ├── FATFRACTION.dcm
│   │   │   ├── psoas_left_02.mha
│   │   │   ├── psoas_right_02.mha
│   │   │   ├── T2star.dcm
│   │   │   └── WATER.dcm
│   │   └── vertebrae
│   │       ├── B0.dcm
│   │       ├── FAT.dcm
│   │       ├── FATFRACTION.dcm
│   │       ├── L1_02.mha
│   │       ├── L2_02.mha
│   │       ├── L3_02.mha
│   │       ├── L4_02.mha
│   │       ├── L5_02.mha
│   │       ├── T2star.dcm
│   │       └── WATER.dcm
│   └── __MACOSX
│       └── 02
│           ├── muscle
│           └── vertebrae


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

import matplotlib.pyplot as plt

import sys

sys.path.append('/root/space/code/utils/')

import utils as ut

from PIL import Image

import json

from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)

##############################
# MyoSegmenTUM data structure:
# ├── 01
# │   ├── 01
# │   │   ├── muscle
# │   │   │   ├── B0.dcm
# │   │   │   ├── erector_spinae_left_01.mha
# │   │   │   ├── erector_spinae_right_01.mha
# │   │   │   ├── FAT.dcm
# │   │   │   ├── FATFRACTION.dcm
# │   │   │   ├── psoas_left_01.mha
# │   │   │   ├── psoas_right_01.mha
# │   │   │   ├── T2star.dcm
# │   │   │   └── WATER.dcm
# │   │   └── vertebrae
# │   │       ├── B0.dcm
# │   │       ├── FAT.dcm
# │   │       ├── FATFRACTION.dcm
# │   │       ├── L1_01.mha
# │   │       ├── L2_01.mha
# │   │       ├── L3_01.mha
# │   │       ├── L4_01.mha
# │   │       ├── L5_01.mha
# │   │       ├── T2star.dcm
# │   │       └── WATER.dcm
# │   └── __MACOSX
# │       └── 01
# │           ├── muscle
# │           └── vertebrae
# ├── 01.zip
# ├── 02
# │   ├── 02
# │   │   ├── muscle
# │   │   │   ├── B0.dcm
# │   │   │   ├── erector_spinae_left_02.mha
# │   │   │   ├── erector_spinae_right_02.mha
# │   │   │   ├── FAT.dcm
# │   │   │   ├── FATFRACTION.dcm
# │   │   │   ├── psoas_left_02.mha
# │   │   │   ├── psoas_right_02.mha
# │   │   │   ├── T2star.dcm
# │   │   │   └── WATER.dcm
# │   │   └── vertebrae
# │   │       ├── B0.dcm
# │   │       ├── FAT.dcm
# │   │       ├── FATFRACTION.dcm
# │   │       ├── L1_02.mha
# │   │       ├── L2_02.mha
# │   │       ├── L3_02.mha
# │   │       ├── L4_02.mha
# │   │       ├── L5_02.mha
# │   │       ├── T2star.dcm
# │   │       └── WATER.dcm
# │   └── __MACOSX
# │       └── 02
# │           ├── muscle
# │           └── vertebrae
# 
# This code will perform the following tasks:
# • Load the image or the mask
# • resample the image or mask on an isotropic grid: 1 mm × 1 mm × 1 mm  (using nearest neighbor interpollation)
# • images will be rescaled to floats in [0. , 1.]
# • masks values will be converted from 200 to 240 to 1 -> 5 to encode L1 to L5
# • The individual 2D slices are stored both as .jpg files and as pickle files to allow fast access by pyTorch.
#############################

# The dimensions of the volumes are converted to 'xVertSeg axis'. This means that for the MyoSegmenTUM dataset the dimension input variable has to be converted:
# • 0 : Left-right
# • 1 : Anteroposterior
# • 2 : Craniocaudal
# ###
DIM_CONV = {
    0 : 1,
    1 : 2,
    2 : 0
}


if __name__ == '__main__':

    logging.info('Start preprocessing of MyoSegmenTUM dataset.')

    # Parse arguments
    parser = argparse.ArgumentParser(description='MyoSegmenTUM preparation')
    parser.add_argument('--source', type=str, default='./spine_volumes/OSF_Sarah_Schlaeger', help='root path of xVertSeg dataset')
    parser.add_argument('--output', type=str, default='./dataset', help='output path for the dataset')
    parser.add_argument('--dimension', type=int, default=0, help='dimension along which to slice the volumes, converted to xVertSeg dimensions')
    parser.add_argument('--mode', type=str, default='T2star', help='Which of the MyoSegmenTUM modes to use: B0, T2star, WATER, FAT, FATFRACTION')
    parser.add_argument('--contrast', type=int, default=0, help='Contrast enhancement bool?')
    args = parser.parse_args()

    source_filedir = os.path.abspath(args.source)
    dim_slice = DIM_CONV[args.dimension]
    output_filedir = os.path.abspath(args.output)
    contrast_enhance = (args.contrast > 0)
    image_slices_filedir = os.path.join(output_filedir, 'MyoSegmenTUM_images')
    mask_slices_filedir = os.path.join(output_filedir, 'MyoSegmenTUM_masks')

    # make sure the output folder exists
    Path(output_filedir).mkdir(parents=True, exist_ok=True)
    Path(image_slices_filedir).mkdir(parents=True, exist_ok=True)
    Path(mask_slices_filedir).mkdir(parents=True, exist_ok=True)

    # make list with filenames:
    filefolder_list = {
        nr : os.path.join(source_filedir, f'{nr:02d}', 'vertebrae') for nr in range(1, 55)
    }

    dimensions_dict = dict()

    # Convert the images to a set of slices:
    logging.info('Start copy of image files')

    dataset_min = np.infty
    dataset_max = -np.infty

    for nr, foldername in filefolder_list.items():
        filename = os.path.join(foldername, f'{args.mode}.dcm')

        arr, minimum, maximum = ut.array_from_file(filename)

        dimensions_dict[foldername] = {'image' : arr.shape}
        
        logging.debug(f'min : {np.min(arr):1.5f} ** max : {np.max(arr):1.5f}')
        logging.debug(f'source : {filename}, shape {arr.shape}')

        dataset_min = min(dataset_min, minimum)
        dataset_max = max(dataset_max, maximum)

        fn = os.path.join(image_slices_filedir, f'image{nr:03d}')
        Path(fn).mkdir(parents=True, exist_ok=True)

        # For each slice along the asked dimension, convert to a numpy.ndarray and save this.
        # Preprocessing the slices before loading into pyTorch should speed up the training in the end.
        ut.arr_slices_save(arr, dim_slice, fn, args.contrast, save_jpeg = True)

    # Process the mask files and change the filenames
    logging.info('start copy of mask files')
    unique_values = dict()
    for nr, foldername in tqdm(filefolder_list.items(), desc='Copy mask files of the MyoSegmenTUM dataset'):
        mask_files = [os.path.join(foldername, f'L{i}_{nr:02d}.mha') for i in range(1,6)]
        logging.debug(f'Mask files : {mask_files}')

        target_folder = os.path.join(mask_slices_filedir, f'image{nr:03d}')
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        
        logging.debug(f'target : {target_folder}')

        # Get mask, resample to 1 mm × 1 mm × 1 mm grid and extract np_array from this

        masks =  [sitk.GetArrayFromImage( ut.resampler( sitk.ReadImage(file) , imposed_size = dimensions_dict[foldername]['image']) ) for file in mask_files]

        logging.debug(f'mask image dimensions : {[mask.shape for mask in masks]}')

        arr = np.zeros_like(masks[0], dtype=int)
        for i, mask in enumerate(masks):
            arr[mask == 1] = i+1

        dimensions_dict[foldername]['mask'] = arr.shape

        vals, counts = np.unique(arr, return_counts=True)
        for val, count in zip(vals.tolist(), counts.tolist()):
            unique_values[val] = unique_values.get(val, 0) + count
        logging.debug(f'source : {filename}, shape {arr.shape}')
        logging.debug(f'min : {np.min(arr)} ** max : {np.max(arr)}')
        ut.mask_to_slices_save(arr, dim_slice, target_folder)

    logging.info(f'List of unique values in the masks : {unique_values}')
    logging.info(f'min and max values in the complete dataset : {dataset_min} & {dataset_max}.')

    with open(os.path.join(output_filedir, 'mask_counts_MyoSegmenTUM.json'), 'w') as mask_counts_file:
        json.dump(unique_values, mask_counts_file)


    with open(os.path.join(output_filedir, 'dimensions_MyoSegmenTUM.json'), 'w') as mask_dim_file:
        json.dump(dimensions_dict, mask_dim_file)