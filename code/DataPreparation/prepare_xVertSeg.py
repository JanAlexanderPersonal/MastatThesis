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
sys.path.append('../utils/')
import utils as ut

from PIL import Image
from skimage import exposure
from skimage.restoration import denoise_bilateral

from pathlib import Path

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
    for filename in os.listdir(image_filedir):
        if filename.endswith('.raw'):
            continue
        
        image =  sitk.ReadImage(os.path.join(image_filedir, filename))
        min_max.Execute(image)
        dataset_min = min(dataset_min, min_max.GetMinimum())
        dataset_max = max(dataset_max, min_max.GetMaximum())
        # rescale to 0 -> 255
        image = rescale.Execute(image)

        # resample on isotropic 1 mm × 1 mm × 1 mm grid
        image = ut.resampler(image)

        # Convert this new view of the image (on the isotropic grid) to an array & display some main features of this array:
        arr = sitk.GetArrayFromImage(image).astype('float16')
        arr /= 255.0

        logging.debug(f'min : {np.min(arr)} ** max : {np.max(arr):1.5f}')
        logging.debug(f'source : {filename}, shape {arr.shape}')

        # For each slice along the asked dimension, convert to a numpy.ndarray and save this.
        # Preprocessing the slices before loading into pyTorch should speed up the training in the end.
        for i in range(arr.shape[dim_slice]):
            fn = os.path.join(image_slices_filedir, filename.split('.')[0])
            Path(fn).mkdir(parents=True, exist_ok=True)

            # Take the index from the desired axis and save this slice (numpy.ndarray) for the model to train on.
            # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.take.html#numpy.ndarray.take
            slice_to_save = arr.take(i, axis=dim_slice)
            if contrast_enhance:
                logging.debug('increase contrast')
                sslice_to_save = denoise_bilateral(exposure.equalize_hist(slice_to_save, nbins=256, mask=(slice_to_save > 0.05))).astype('float16')
            np.save(os.path.join(fn, f'slice_{i:03d}'), slice_to_save)

            # for jpeg visualization, get back to the original 0 -> 255 range.
            im = Image.fromarray((slice_to_save * 255).astype(np.uint8))
            im.convert('RGB').save(os.path.join(fn, f'slice_{i:03d}.jpg')) # :03d means 3 digits -> leading 0s

    # Process the mask files and change the filenames
    logging.info('start copy of mask files')
    unique_values = []
    for filename in os.listdir(mask_filedir):
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
        unique_values += np.unique(arr).tolist()
        logging.debug(f'source : {filename}, shape {arr.shape}')
        logging.debug(f'min : {np.min(arr)} ** max : {np.max(arr)}')
        for i in range(arr.shape[dim_slice]):
            fn = os.path.join(target_folder, f'slice_{i:03d}') # :03d means 3 digits -> leading 0s
            # Take the index from the desired axis and save this slice (numpy.ndarray) for the model to train on.
            # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.take.html#numpy.ndarray.take
            arr_slice = arr.take(i, axis=dim_slice)
            np.save(fn, arr_slice)

            # For the visualization, bring background back to 0 and spread out the colours as far as possible
            arr_slice[arr_slice == 255] = 0
            arr_slice *= 51
            im = cm.gist_earth(arr_slice)
            plt.figure()
            plt.imshow(im)
            plt.axis('off')
            plt.colorbar()
            plt.savefig(os.path.join(target_folder, f'slice_{i:03d}.png'), bbox_inches='tight')
            plt.close()

    logging.info(f'List of unique values in the masks : {sorted(list(set(unique_values)))}')