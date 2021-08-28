"""
author :    Jan Alexander
project:    Master thesis
date:       2021-01-31


Script to prepare the UWashington dataset

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

logging.basicConfig(level=logging.WARNING)



# The following regex will parse "mask002.raw" to "mask002" "002" "raw"
MASK_NR = re.compile(r'(^[a-zA-Z]+(\d+)).(\w+)')


XVERTSEG_ENCODING = {200 + (i-1) * 10 : i for i in range(1,6)}

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='University Washington preparation')
    parser.add_argument('--source', type=str, default='./spine_volumes/UWSpine_selected/', help='root path of UWSpine dataset')
    parser.add_argument('--output', type=str, default='./dataset', help='output path for the dataset')
    parser.add_argument('--dimension', type=int, default=0, help='dimension along which to slice the volumes')
    parser.add_argument('--contrast', type=int, default=0, help='Contrast enhancement bool?')
    args = parser.parse_args()

    image_filedir = os.path.join(args.source, 'spine-1')
    contrast_enhance = (args.contrast > 0)
    dim_slice = args.dimension
    output_filedir = os.path.abspath(args.output)
    image_slices_filedir = os.path.join(output_filedir, 'UW_images')
    mask_slices_filedir = os.path.join(output_filedir, 'UW_masks')

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

        for patient in os.listdir(os.path.join(image_filedir, filename)):
            for scan in os.listdir(os.path.join(image_filedir, filename, patient)):
                if scan.endswith('.lml'):
                    continue
                scan_filename = os.path.join(image_filedir, filename, patient, scan)
                
                logging.debug(f'folder name : {scan_filename}')

                arr, minimum, maximum = ut.array_from_file(scan_filename)

                dimensions_dict[filename] = {'image' : arr.shape}
                
                dataset_min = min(dataset_min, minimum)
                dataset_max = max(dataset_max, maximum)

                logging.debug(f'min : {np.min(arr)} ** max : {np.max(arr):1.5f}')
                logging.debug(f'source : {filename}, shape {arr.shape}')

                # For each slice along the asked dimension, convert to a numpy.ndarray and save this.
                # Preprocessing the slices before loading into pyTorch should speed up the training in the end.

                fn = os.path.join(image_slices_filedir, filename.split('.')[0])
                Path(fn).mkdir(parents=True, exist_ok=True)
                np.save(os.path.join(fn, 'image_array'), arr)
                ut.arr_slices_save(arr, dim_slice, fn, args.contrast, save_jpeg = True)

    # Process the mask files and change the filenames
    logging.info('start copy of label files')
    logging.info(f'dimensions dictionary so far : {dimensions_dict}')
    
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
        image = ut.resampler(image, imposed_size = dimensions_dict[f'image{ms[1]}.mhd']['image'],mask=True)
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

        dimensions_dict[f'image{ms[1]}.mhd']['mask'] = arr.shape

        vals, counts = np.unique(arr, return_counts=True)
        for val, count in zip(vals.tolist(), counts.tolist()):
            unique_values[val] = unique_values.get(val, 0) + count
        logging.debug(f'source : {filename}, shape {arr.shape}')
        logging.debug(f'min : {np.min(arr)} ** max : {np.max(arr)}')
        np.save(os.path.join(target_folder, 'mask_array'), arr)
        ut.mask_to_slices_save(arr, dim_slice, target_folder)

    logging.info(f'List of unique values in the masks : {unique_values}')
    logging.info(f'min and max values in the complete dataset : {dataset_min} & {dataset_max}.')

    with open(os.path.join(output_filedir, 'mask_counts_xVertSeg.json'), 'w') as mask_counts_file:
        json.dump(unique_values, mask_counts_file)

    with open(os.path.join(output_filedir, 'dimensions_xVertSeg.json'), 'w') as mask_dim_file:
        json.dump(dimensions_dict, mask_dim_file)