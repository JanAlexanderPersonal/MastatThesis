"""
author :    Jan Alexander
project:    Master thesis
date:       2021-05-21


Script to prepare the USiegen dataset for processing.

Intended working: The USiegen folder *Data* has following structure
        
# USiegen data structure:
# ├── SpineDatasets
# │   ├── AKa2.dcm
# │   ├── AKa3.dcm
# │   ├── AKa4.dcm
# │   ├── AKs3.dcm
# │   ├── AKs5.dcm
# │   ├── AKs6.dcm
# │   ├── AKs7.dcm
# │   ├── AKs8.dcm
# │   ├── C002.dcm
# │   ├── DzZ_T1.dcm
# │   ├── DzZ_T2.dcm
# │   ├── F02.dcm
# │   ├── F03.dcm
# │   ├── F04.dcm
# │   ├── S01.dcm
# │   ├── S02.dcm
# │   └── St1.dcm
# ├── SpineSegmented
# │   ├── AKa2
# │   │   ├── AKA2_mask.mha
# │   │   ├── AKA2_sag.cso
# │   │   ├── L1.cso
# │   │   ├── L1.mha
# │   │   ├── L1.obj
# │   │   ├── L2.cso
# │   │   ├── L2.mha
# │   │   ├── L2.obj
# │   │   ├── L3.cso
# │   │   ├── L3.mha
# │   │   ├── L3.obj
# │   │   ├── L4.cso
# │   │   ├── L4.mha
# │   │   ├── L4.obj
# │   │   ├── L5.cso
# │   │   ├── L5.mha
# │   │   ├── L5.obj
# │   │   ├── S1.cso
# │   │   ├── S1.mha
# │   │   ├── S1.obj
# │   │   ├── T10.cso
# │   │   ├── T10.mha
# │   │   ├── T10.obj
# │   │   ├── T11.cso
# │   │   ├── T11.mha
# │   │   ├── T11.obj
# │   │   ├── T12.cso
# │   │   ├── T12.mha
# │   │   └── T12.obj


This script does nothing more than copying files from the source folder to the dataset folder.

"""

import os
import re
import argparse
import logging

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

import sys
sys.path.append('/root/space/code/utils/')
import utils as ut

import json

from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


FILENAMES_TO_PATIENTS = {
    'AKa2' : 1,
'AKa3' : 1,
'AKa4' : 1,
'AKs3' : 2,
'AKs5' : 2,
'AKs6' :2, 
'AKs7' : 2,
'AKs8' : 2,
'Ble' : 3,
'C002' : 4,
'case_2' : 5,
'case_10' : 6,
'DzZ_T1' : 7,
'DzZ_T2' : 7,
'F02' : 8,
'F03' : 9,
'F04' : 10,
'Geh' : 11,
'Hoe' : 12,
'Lan' : 13,
'LanII' : 13,
'LC' : 14,
'S01' : 15,
'S02' : 16,
'Sch' : 17,
'St1' : 18
}

##############################
# USiegen data structure:
# ├── SpineDatasets
# │   ├── AKa2.dcm
# │   ├── AKa3.dcm
# │   ├── AKa4.dcm
# │   ├── AKs3.dcm
# │   ├── AKs5.dcm
# │   ├── AKs6.dcm
# │   ├── AKs7.dcm
# │   ├── AKs8.dcm
# │   ├── C002.dcm
# │   ├── DzZ_T1.dcm
# │   ├── DzZ_T2.dcm
# │   ├── F02.dcm
# │   ├── F03.dcm
# │   ├── F04.dcm
# │   ├── S01.dcm
# │   ├── S02.dcm
# │   └── St1.dcm
# ├── SpineSegmented
# │   ├── AKa2
# │   │   ├── AKA2_mask.mha
# │   │   ├── AKA2_sag.cso
# │   │   ├── L1.cso
# │   │   ├── L1.mha
# │   │   ├── L1.obj
# │   │   ├── L2.cso
# │   │   ├── L2.mha
# │   │   ├── L2.obj
# │   │   ├── L3.cso
# │   │   ├── L3.mha
# │   │   ├── L3.obj
# │   │   ├── L4.cso
# │   │   ├── L4.mha
# │   │   ├── L4.obj
# │   │   ├── L5.cso
# │   │   ├── L5.mha
# │   │   ├── L5.obj
# │   │   ├── S1.cso
# │   │   ├── S1.mha
# │   │   ├── S1.obj
# │   │   ├── T10.cso
# │   │   ├── T10.mha
# │   │   ├── T10.obj
# │   │   ├── T11.cso
# │   │   ├── T11.mha
# │   │   ├── T11.obj
# │   │   ├── T12.cso
# │   │   ├── T12.mha
# │   │   └── T12.obj
# 
# This code will perform the following tasks:
# • Load the image or the mask
# • resample the image or mask on an isotropic grid: 1 mm × 1 mm × 1 mm  (using nearest neighbor interpollation)
# • images will be rescaled to floats in [0. , 1.]
# • masks values will be converted from 200 to 240 to 1 -> 5 to encode L1 to L5
# • The individual 2D slices are stored both as .jpg files and as pickle files to allow fast access by pyTorch.
#############################

DIM_CONV = {
    0 : 1,
    1 : 2,
    2 : 0
}


if __name__ == '__main__':

    logging.info('Start preprocessing of dataset USiegen')

    # Parse arguments
    parser = argparse.ArgumentParser(description='xVertSeg preparation')
    parser.add_argument('--source', type=str, default='./spine_volumes/xVertSeg/Data1', help='root path of xVertSeg dataset')
    parser.add_argument('--output', type=str, default='./dataset', help='output path for the dataset')
    parser.add_argument('--dimension', type=int, default=0, help='dimension along which to slice the volumes')
    parser.add_argument('--contrast', type=int, default=0, help='Contrast enhancement bool?')
    args = parser.parse_args()

    image_filedir = os.path.join(args.source, 'SpineDatasets')
    mask_filedir = os.path.join(args.source, 'SpineSegmented')
    dim_slice = DIM_CONV[args.dimension]
    contrast_enhance = (args.contrast > 0)
    output_filedir = os.path.abspath(args.output)
    image_slices_filedir = os.path.join(output_filedir, 'USiegen_images')
    mask_slices_filedir = os.path.join(output_filedir, 'USiegen_masks')

    # make sure the output folder exists
    Path(output_filedir).mkdir(parents=True, exist_ok=True)
    Path(image_slices_filedir).mkdir(parents=True, exist_ok=True)
    Path(mask_slices_filedir).mkdir(parents=True, exist_ok=True)

    

    dataset_min = np.infty
    dataset_max = -np.infty

    # Convert the images to a set of slices:
    logging.info('Start copy of image files')

    filenames_dict = dict()
    dimensions_dict = dict()

    for nr, filename in tqdm(enumerate(os.listdir(image_filedir)), desc='copy image files for the USiegen dataset'):


        logging.debug(f'read file {filename}')

        # Keep the filenames in a dict to assure the correct mask end up linked to the correct images
        filenames_dict[nr] = filename

        arr, minimum, maximum = ut.array_from_file(os.path.join(image_filedir, filename))

        dimensions_dict[filename] = {'image' : arr.shape}
        
        dataset_min = min(dataset_min, minimum)
        dataset_max = max(dataset_max, maximum)

        logging.debug(f'min : {np.min(arr)} ** max : {np.max(arr):1.5f}')
        logging.debug(f'source : {filename}, shape {arr.shape}')

        fn = os.path.join(image_slices_filedir, f'image{nr:03d}')
        Path(fn).mkdir(parents=True, exist_ok=True)

        # For each slice along the asked dimension, convert to a numpy.ndarray and save this.
        # Preprocessing the slices before loading into pyTorch should speed up the training in the end.
        ut.arr_slices_save(arr, dim_slice, fn, args.contrast, save_jpeg = True)

    # Process the mask files and change the filenames
    logging.info('start copy of mask files')
    unique_values = dict()
    for nr, foldername in tqdm(filenames_dict.items(),desc='copy mask files for the USiegen dataset'):
        logging.debug(f'filename : {foldername}')
        
        source_filenames = [os.path.join(mask_filedir, foldername.split('.')[0], f'L{i}.mha') for i in range(1,6)]
        target_folder = os.path.join(mask_slices_filedir, f'image{nr:03d}')
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        
        logging.debug(f'source : {source_filenames}')
        logging.debug(f'target : {target_folder}')

        # resample on isotropic 1 mm × 1 mm × 1 mm grid
        images =  ut.read_masklist(source_filenames, imposed_size = dimensions_dict[foldername]['image'])

        logging.debug(f'image names : {images}')

        # resample on isotropic 1 mm × 1 mm × 1 mm grid
        arr = np.zeros_like(images[0], dtype=int)

        # This will be converted to labels 1 -> 5 representing L1 -> L5.
        # The background will be encoded as value 0
        for i, image  in enumerate(images):
            arr[image != 0] = i + 1

        dimensions_dict[foldername]['mask'] = arr.shape
        
        vals, counts = np.unique(arr, return_counts=True)
        for val, count in zip(vals.tolist(), counts.tolist()):
            unique_values[val] = unique_values.get(val, 0) + count
        logging.debug(f'source : {filename}, shape {arr.shape}')
        logging.debug(f'min : {np.min(arr)} ** max : {np.max(arr)}')
        ut.mask_to_slices_save(arr, dim_slice, target_folder)

    logging.info(f'List of unique values in the masks : {unique_values}')
    logging.info(f'min and max intensity values of images {dataset_min} & {dataset_max}')

    with open(os.path.join(output_filedir, 'mask_counts_USiegen.json'), 'w') as mask_counts_file:
        json.dump(unique_values, mask_counts_file)

    with open(os.path.join(output_filedir, 'dimensions_USiegen.json'), 'w') as mask_dim_file:
        json.dump(dimensions_dict, mask_dim_file)

    with open(os.path.join(output_filedir, 'filenames_USiegen.json'), 'w') as filenames_file:
        json.dump(filenames_dict, filenames_file)

    with open(os.path.join(output_filedir, 'patients_USiegen.json'), 'w') as patients_file:
        json.dump( FILENAMES_TO_PATIENTS, patients_file)