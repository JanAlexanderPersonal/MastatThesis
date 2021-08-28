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
L1_coord = re.compile(r'200\tL1_center\t(\d+).?\d+\t(\d+).?\d+\t(\d+).?\d+', re.MULTILINE)
L2_coord = re.compile(r'210\tL2_center\t(\d+).?\d+\t(\d+).?\d+\t(\d+).?\d+', re.MULTILINE)
L3_coord = re.compile(r'220\tL3_center\t(\d+).?\d+\t(\d+).?\d+\t(\d+).?\d+', re.MULTILINE)
L4_coord = re.compile(r'230\tL4_center\t(\d+).?\d+\t(\d+).?\d+\t(\d+).?\d+', re.MULTILINE)
L5_coord = re.compile(r'240\tL5_center\t(\d+).?\d+\t(\d+).?\d+\t(\d+).?\d+', re.MULTILINE)



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


    for nr, filename in tqdm(enumerate(os.listdir(image_filedir))):

        for patient in os.listdir(os.path.join(image_filedir, filename)):
            for scan in os.listdir(os.path.join(image_filedir, filename, patient)):
                if scan.endswith('.lml'):
                    continue
                scan_filename = os.path.join(image_filedir, filename, patient, scan)
                
                logging.info(f'folder name : {scan_filename}')

                arr, minimum, maximum = ut.array_from_file(scan_filename)
                #print(arr.shape)
                mask = np.zeros_like(arr, dtype=int) 
                with open(scan_filename.replace('.nii.gz', '.lml'), 'r') as f:
                    text = f.read()
                    print(text)
                    if len(L1_coord.findall(text)) == 1:
                        xx, yy, zz = L1_coord.findall(text)[0]
                        print(f'L1 -x : {xx}, y: {yy}, z: {zz}')
                        try:
                            mask[int(xx), int(yy), int(zz)] = 1
                        except:
                            pass
                    if len(L2_coord.findall(text)) == 1:
                        xx, yy, zz = L2_coord.findall(text)[0]
                        print(f'L2 -x : {xx}, y: {yy}, z: {zz}')
                        try:
                            mask[int(xx), int(yy), int(zz)] = 2
                        except:
                            pass
                    if len(L3_coord.findall(text)) == 1:
                        xx, yy, zz = L3_coord.findall(text)[0]
                        print(f'L3 -x : {xx}, y: {yy}, z: {zz}')
                        try:
                            mask[int(xx), int(yy), int(zz)] = 3
                        except:
                            pass
                    if len(L4_coord.findall(text)) == 1:
                        xx, yy, zz = L4_coord.findall(text)[0]
                        print(f'L4 -x : {xx}, y: {yy}, z: {zz}')
                        try:
                            mask[int(xx), int(yy), int(zz)] = 4
                        except:
                            pass
                    if len(L5_coord.findall(text)) == 1:
                        
                        xx, yy, zz = L5_coord.findall(text)[0]
                        print(f'L5 -x : {xx}, y: {yy}, z: {zz}')
                        try:
                            mask[int(xx), int(yy), int(zz)] = 5
                        except:
                            pass
                arr = np.rot90(arr, k=2, axes=(0, 1))
                arr = np.rot90(arr, k=2, axes=(1, 2))
                mask = np.rot90(mask, k=2, axes=(0, 1))
                mask = np.rot90(mask, k=2, axes=(1, 2))

                print(np.unique(mask, return_counts = True))

                dimensions_dict[filename] = {'image' : arr.shape}
                
                dataset_min = min(dataset_min, minimum)
                dataset_max = max(dataset_max, maximum)

                logging.debug(f'min : {np.min(arr)} ** max : {np.max(arr):1.5f}')
                logging.debug(f'source : {filename}, shape {arr.shape}')

                # For each slice along the asked dimension, convert to a numpy.ndarray and save this.
                # Preprocessing the slices before loading into pyTorch should speed up the training in the end.

                fn = os.path.join(image_slices_filedir, f'image{nr:03d}')
                Path(fn).mkdir(parents=True, exist_ok=True)
                np.save(os.path.join(fn, 'image_array'), arr)
                ut.arr_slices_save(arr, dim_slice, fn, args.contrast, save_jpeg = True)
                fn = os.path.join(mask_slices_filedir, f'image{nr:03d}')
                Path(fn).mkdir(parents=True, exist_ok=True)
                np.save(os.path.join(fn, 'mask_array'), mask)
                ut.mask_to_slices_save(mask, dim_slice, fn)

