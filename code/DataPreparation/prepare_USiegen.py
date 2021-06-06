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

    # Prepare the intensity rescaler
    rescale = sitk.RescaleIntensityImageFilter()
    min_max = sitk.MinimumMaximumImageFilter()

    dataset_min = np.infty
    dataset_max = -np.infty

    # Convert the images to a set of slices:
    logging.info('Start copy of image files')

    filenames_dict = dict()
    nr = 0

    for filename in os.listdir(image_filedir):


        logging.debug(f'read file {filename}')
        filenames_dict[nr] = filename

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

        logging.info(f'min : {np.min(arr)} ** max : {np.max(arr):1.5f}')
        logging.debug(f'source : {filename}, shape {arr.shape}')

        # For each slice along the asked dimension, convert to a numpy.ndarray and save this.
        # Preprocessing the slices before loading into pyTorch should speed up the training in the end.
        for i in range(arr.shape[dim_slice]):
            fn = os.path.join(image_slices_filedir, f'image{nr:03d}')
            Path(fn).mkdir(parents=True, exist_ok=True)

            # Take the index from the desired axis and save this slice (numpy.ndarray) for the model to train on.
            # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.take.html#numpy.ndarray.take
            slice_to_save = arr.take(i, axis=dim_slice)
            if contrast_enhance:
                logging.debug('increase contrast')
                logging.info(f'Slice values before contrast enhancement min : {np.min(slice_to_save):1.5f} ** max : {np.max(slice_to_save):1.5f}')
                slice_to_save = denoise_bilateral(exposure.equalize_hist(slice_to_save, nbins=256, mask=(slice_to_save > 0.05))).astype('float16')
                logging.info(f'Slice values after contrast enhancement min : {np.min(slice_to_save):1.5f} ** max : {np.max(slice_to_save):1.5f}')
            np.save(os.path.join(fn, f'slice_{i:03d}'), slice_to_save)

            # for jpeg visualization, get back to the original 0 -> 255 range.
            im = Image.fromarray((slice_to_save * 255).astype(np.uint8))
            im.convert('RGB').save(os.path.join(fn, f'slice_{i:03d}.jpg')) # :03d means 3 digits -> leading 0s
        nr += 1

    # Process the mask files and change the filenames
    logging.info('start copy of mask files')
    unique_values = []
    for nr, foldername in filenames_dict.items():
        logging.debug(f'filename : {foldername}')
        
        source_filenames = [os.path.join(mask_filedir, foldername.split('.')[0], f'L{i}.mha') for i in range(1,6)]
        target_folder = os.path.join(mask_slices_filedir, f'image{nr:03d}')
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        
        logging.debug(f'source : {source_filenames}')
        logging.debug(f'target : {target_folder}')

        # resample on isotropic 1 mm × 1 mm × 1 mm grid
        images =  [sitk.GetArrayFromImage( ut.resampler( sitk.ReadImage(source_filename) )) for source_filename in source_filenames]

        logging.debug(f'image names : {images}')

        # resample on isotropic 1 mm × 1 mm × 1 mm grid
        arr = np.zeros_like(images[0], dtype=int)

        # This will be converted to labels 1 -> 5 representing L1 -> L5.
        # The background will be encoded as value 0
        for i, image  in enumerate(images):
            arr[image != 0] = i + 1
        
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
    logging.info(f'min and max intensity values of images {dataset_min} & {dataset_max}')