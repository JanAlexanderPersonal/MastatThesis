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
   ├── image001.mhd
   ├── image001.raw
   ├── image001_Labels.mhd
   ├── image001_Labels.raw

This script does nothing more than copying files from the source folder to the dataset folder.

"""

import os
import re
import argparse
import logging

from shutil import copyfile

from pathlib import Path

logging.basicConfig(level=logging.INFO)

# The following regex will parse "mask002.raw" to "mask002" "002" "raw"
MASK_NR = re.compile(r'(^[a-zA-Z]+(\d+)).(\w+)')

def replace_line(old : str, new : str, source : str, output : str):

    logging.debug(f'in file {source}, change {old} for {new} and write to {output}')

    reading_file = open(source, 'r')
    new_file_content = ""
    for line in reading_file:
        new_file_content += line.strip().replace(old, new) + '\n'
    reading_file.close()

    new_file = open(output, 'w')
    new_file.write(new_file_content)
    new_file.close()
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xVertSeg preparation')
    parser.add_argument('--source', type=str, default='./spine_volumes/xVertSeg/Data1', help='root path of xVertSeg dataset')
    parser.add_argument('--output', type=str, default='./dataset', help='output path for the dataset')
    args = parser.parse_args()

    image_filedir = os.path.join(args.source, 'images')
    mask_filedir = os.path.join(args.source, 'masks')
    output_filedir = os.path.abspath(args.output)

    # make sure the output folder exists
    Path(output_filedir).mkdir(parents=True, exist_ok=True)

    # make list with filenames of the images
    logging.info('Start copy of image files')
    for filename in os.listdir(image_filedir):
        logging.debug(f'copy file : {filename}')
        copyfile(os.path.join(image_filedir, filename), os.path.join(output_filedir, filename))

    # Copy the mask files and change the filenames
    logging.info('start copy of mask files')
    for filename in os.listdir(mask_filedir):
        logging.debug(f'filename : {filename}')
        ms = MASK_NR.findall(filename)[0]
        logging.debug(ms)
        source_filename = os.path.join(mask_filedir, filename)
        target_filename = os.path.join(output_filedir, f'image{ms[1]}_Labels.{ms[2]}')
        
        logging.debug(f'source : {source_filename}')
        logging.debug(f'target : {target_filename}')

        if filename.endswith('mhd'):
            replace_line(ms[0], f'image{ms[1]}_Labels', source_filename, target_filename )
        else:
            copyfile(source_filename, target_filename)