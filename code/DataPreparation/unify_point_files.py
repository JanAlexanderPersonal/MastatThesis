import os
import numpy as np
import argparse
import logging

from tqdm import tqdm

import matplotlib.pyplot as plt

from pprint import pformat

logging.basicConfig(level=logging.DEBUG)

def save_mask_slice(i, arr,  target_folder, dim_slice):
        fn = os.path.join(target_folder, f'slice_{i:03d}_points') # :03d means 3 digits -> leading 0s
        logging.debug(f'save slice {fn}')
        # Take the index from the desired axis and save this slice (numpy.ndarray) for the model to train on.
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.take.html#numpy.ndarray.take
        arr_slice = arr.take(i, axis=dim_slice)
        np.save(fn, arr_slice)

if __name__ == '__main__':

    logging.info('Start preprocessing of MyoSegmenTUM dataset.')

    # Parse arguments
    parser = argparse.ArgumentParser(description='point annotation preparation')
    parser.add_argument('--source', type=str, default='/root/output/', help='root path output folder')
    parser.add_argument('--foldername', type=str, default='dataset_D_contrast_3', help= 'template name for dataset name')
    parser.add_argument('--dimension', type=int, default=2, help='Basis dimension of the point annotation')
    args = parser.parse_args()

    source_filedir = os.path.abspath(args.source)

    
    dimensions = [0,1,2]
    dimensions.remove(args.dimension)

    for dim in dimensions:
        foldername = os.path.join(args.source, args.foldername.replace('D', str(dim)))          
        subfolders = [os.path.join(foldername, o) for o in os.listdir(foldername) if os.path.isdir(os.path.join(foldername,o)) and not o.startswith('.') and o.endswith('_masks')]
        logging.debug(f'subfolders in foldername {foldername}:\n{pformat(subfolders)}')
        for subf in subfolders:
            mask_folders = [os.path.join(subf, o) for o in os.listdir(subf) if os.path.isdir(os.path.join(subf, o)) ]
            logging.debug(f'mask folders :\n{pformat(mask_folders)}')
            for mask_folder in mask_folders:
                for fn in os.listdir(mask_folder):
                    if not fn.endswith('_points.npy'):
                        continue
                    filename = os.path.join(mask_folder, fn)
                    logging.debug(f'delete file {filename}')
                    os.remove(filename)

    # Get the point volumes from the main dimension and re-slice it
    foldername = os.path.join(args.source, args.foldername.replace('D', str(args.dimension)))
    subfolders = [o for o in os.listdir(foldername) if os.path.isdir(os.path.join(foldername,o)) and not o.startswith('.') and o.endswith(('_masks'))]
    for subf in subfolders:
        subfolder_path = os.path.join(foldername, subf)
        mask_folders = [o for o in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, o)) ]
        logging.debug(f'mask folders :\n{pformat(mask_folders)}')
        for mask_folder in mask_folders:
            logging.debug(f'mask folder {mask_folder}')
            # open points volume
            volume_name = os.path.join(subfolder_path, mask_folder, 'points_volume.npy')
            logging.debug('Volume folder : {}'.format(volume_name.split('\\')))
            volume = np.load(volume_name)
            for dim in dimensions:
                target_folder = os.path.join(args.source, args.foldername.replace('D', str(dim)), subf, mask_folder)
                logging.debug(f'target folder : {target_folder}')
                for i in range(volume.shape[dim]):
                    save_mask_slice(i, volume,  target_folder, dim)

