"""
author :    Jan Alexander
project:    Master thesis
date:       2021-02-07

Based on a foldername of preprocessed images, this script visualizes a random set of them.
"""

import os
import re
import argparse
import logging

from pathlib import Path
import mathplotlib.pyplot as plt

import SimpleITK as sitk

logging.basicConfig(level=logging.DEBUG)

def plot_figures(image, folder, name):
    array = sitk.GetArrayFromImage(image)
    mid_mask = mid_mask(array)

    name, ext = name.split('.')

    slices = [
        array[mid_mask[0], :, :], 
        array[:, mid_mask[1], :], 
        array[:, :, mid_mask[2]]
    ]

    for i in range(3):
        filename = os.path.join(folder, f'{name}_dim{i}.{ext}')
        logging.debug(f'make plot {filename}')
        plt.imshow(slices[i])
        plt.title(f'slice {name} dim {i}')
        plt.axis('off')
        plt.savefig(filename)
        

def mid_mask(array):
    indices = np.nonzero(array != 0)
    lower = [np.min(i) for i in indices]
    upper = [np.max(i) for i in indices]

    mid = [int((lower[i] + upper[i])/2) for i in range(3)]

    logging.debug(f'Center of the masked volume {mid}')

    return mid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image_visualization')
    parser.add_argument('--dataset', type=str, default='./CSI_dataset', help='root path of preprocessed dataset')
    args = parser.parse_args()

    output = os.path.join(os.path.abspath(args.dataset), 'images')

    folders = ['img', 'seg', 'weight']
    subsets = ['train', 'test']

    for subset in subsets:
        for folder in folders:
            source = os.path.join(args.dataset, subset, folder)
            target = os.path.join(output, subset, folder)
            Path(target).mkdir(parents=True, exist_ok=True)
            files = [x for x in os.listdir(source) if 'raw' not in x]
            for f in files:
                raw_img = sitk.ReadImage(os.path.join(source, f))
                plot_figures(raw_img, target, f)
