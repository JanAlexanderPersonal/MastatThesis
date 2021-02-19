import os
import re
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy import ndimage
import SimpleITK as sitk

logging.basicConfig(level=logging.DEBUG)

# The following regex will parse "image002.mhd" to "image002" "002" "mhd"
IMAGE_NR = re.compile(r'(^[a-zA-Z]+(\d+)).(\w+)')


# resample the CT images to isotropic
def isotropic_resampler(image):
    
    new_spacing = [1, 1, 1]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)

    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = image.GetSpacing()
    new_size = np.array([x * (y / z) for x, y, z in zip(orig_size, orig_spacing, new_spacing)])
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resampler.SetSize(new_size)

    logging.debug(f'original size {orig_size} --> new size {new_size}.\n')

    isotropic_img = resampler.Execute(image)

    return isotropic_img

def mask_convert(mask):
    # For the xVertSeg dataset, the masks are indicated with numbers between 200 and 240:
    logging.debug('Convert mask')
    
    mask_array = sitk.GetArrayFromImage(mask)
    logging.debug(f'mask values {np.unique(mask_array)}')

    old_mask = [200 + i * 10 for i in range(5)]
    new_mask = [i+1 for i in range(5)]
    for old, new in zip(old_mask, new_mask):
        mask_array = np.where(mask_array == old, new, mask_array)
    logging.debug(f'new mask values {np.unique(mask_array)}')
    new_image = sitk.GetImageFromArray(mask_array)
    new_image.CopyInformation(mask)
    return new_image


# Function for cropping
def z_mid(mask, chosen_vert):
    indices = np.nonzero(mask == chosen_vert)
    lower = [np.min(i) for i in indices]
    upper = [np.max(i) for i in indices]

    return int((lower[0] + upper[0]) / 2)


def findZRange(img, mask):
    # list available vertebrae
    verts = np.unique(mask)

    logging.debug(f'Vertebrae found {verts}')

    vert_low = verts[1]
    vert_up = verts[-1]

    z_range = [z_mid(mask, vert_low), z_mid(mask, vert_up)]
    logging.debug('Range of Z axis %s' % z_range)
    return z_range


def crop_unref_vert(path, out_path, subset):
    img_path = os.path.join(path, subset, 'img')
    mask_path = os.path.join(path, subset, 'seg')
    weight_path = os.path.join(path, subset, 'weight')
    
    img_names = [f for f in os.listdir(img_path) if f.endswith('.mhd') or f.endswith('.nii')]

    logging.debug(f'Images to be cropped : {img_names}')

    for img_name in img_names:
        logging.info('Cropping non-reference vertebrae of %s' % img_name)
        img_name = img_name
        ms = IMAGE_NR.findall(img_name)[0]
        logging.debug(f'image name {img_name} is parsed as {ms}')
        mask_name = f'{ms[0]}_Labels.{ms[2]}'
        logging.debug(f'mask name is {mask_name}')
        weight_name = img_name.split('.')[0] + '_weight.nrrd'

        img_file = os.path.join(img_path, img_name)
        mask_file = os.path.join(mask_path, mask_name)
        weight_file = os.path.join(weight_path, weight_name)

        img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
        weight = sitk.GetArrayFromImage(sitk.ReadImage(weight_file))

        z_range = findZRange(img, mask)

        logging.debug(f'z-range found: {z_range}')
        logging.debug(f'type image {img.dtype}')

        sitk.WriteImage(sitk.GetImageFromArray(img[z_range[0]:z_range[1], :, :]),
                        os.path.join(out_path, subset, 'img', img_name), True)
        sitk.WriteImage(sitk.GetImageFromArray(mask[z_range[0]:z_range[1], :, :]),
                        os.path.join(out_path, subset, 'seg', mask_name), True)
        sitk.WriteImage(sitk.GetImageFromArray(weight[z_range[0]:z_range[1], :, :]),
                        os.path.join(out_path, subset, 'weight', weight_name), True)


# calculate the weight via distance transform
def compute_distance_weight_matrix(mask, alpha=1, beta=8, omega=6):
    """
    Code from author : Dr.Lessman (nikolas.lessmann@radboudumc.nl)
    """
    mask = np.asarray(mask)
    distance_to_border = ndimage.distance_transform_edt(mask > 0) + ndimage.distance_transform_edt(mask == 0)
    weights = alpha + beta * np.exp(-(distance_to_border ** 2 / omega ** 2))
    return np.asarray(weights, dtype='float32')


def calculate_weight(isotropic_path, subset):
    mask_path = os.path.join(isotropic_path, subset, 'seg')
    weight_path = os.path.join(isotropic_path, subset, 'weight')

    Path(mask_path).mkdir(parents=True, exist_ok=True)
    Path(weight_path).mkdir(parents=True, exist_ok=True)

    for f in [f for f in os.listdir(mask_path) if f.endswith('.mhd') or f.endswith('.nii')]:
        seg_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_path, f)))
        weight = compute_distance_weight_matrix(seg_mask)
        sitk.WriteImage(sitk.GetImageFromArray(weight), 
                        os.path.join(weight_path, re.findall(r"(^[\d\w]+)_[\w]+", f)[0] + '_weight.nrrd'),
                        True)
        logging.info("Calculating weight of %s" % f)
    logging.debug(f'Weights have been saved under {weight_path}')


def create_folders(root, subsets, folders):
    for subset in subsets:
        for f in folders:
            Path(os.path.join(root, subset, f)).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='iterativeFCN')
    parser.add_argument('--dataset', type=str, default='./CSI_dataset', help='root path of CSI dataset ')
    parser.add_argument('--output_isotropic', type=str, default='./isotropic_dataset',
                        help='output path for isotropic images')
    parser.add_argument('--output_crop', type=str, default='./crop_isotropic_dataset',
                        help='output path for crop samples')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='ratio of train/test')
    args = parser.parse_args()

    # split data into train test folder
    folders = ['img', 'seg', 'weight']
    subsets = ['train', 'test']
    create_folders(args.output_isotropic, subsets, folders)
    create_folders(args.output_crop, subsets, folders)

    # resample the CSI dataset to isotropic dataset
    files = [x for x in os.listdir(os.path.join(args.dataset)) if 'raw' not in x]
    files.sort()
    logging.debug('datafiles discovered:')
    logging.debug('\n'.join([f'\t{file}' for file in files]))
    
    for f in files:
        case_id = re.findall(r'\d+', f)[0]
        logging.info('Resampling ' + f + '...')
        if int(case_id) < int(len(files)/2 * args.split_ratio):
            subset = 'train'
        else:
            subset = 'test'
        folder = 'seg' if '_Labels' in f else 'img'

        file_output = os.path.join(args.output_isotropic, subset, folder, f)
        weight_path = os.path.join(args.output_isotropic, subset, 'weight')
        logging.info(file_output)
        raw_img = sitk.ReadImage(os.path.join(args.dataset, f))  

        logging.debug(f'loaded file {f} with dimensions {raw_img.GetSize()} and spacing {raw_img.GetSpacing()}.')

        if folder == 'seg':
            raw_img = mask_convert(raw_img)

        isotropic_img = isotropic_resampler(raw_img)

        if folder == 'seg':
            seg_mask = sitk.GetArrayFromImage(isotropic_img)
            weight = compute_distance_weight_matrix(seg_mask)
            logging.debug('calculate weight matrix')
            sitk.WriteImage(sitk.GetImageFromArray(weight), 
                        os.path.join(weight_path, re.findall(r"(^[\d\w]+)_[\w]+", f)[0] + '_weight.nrrd'),
                        True)
        sitk.WriteImage(isotropic_img, file_output, True)

    # Crop the image to remove the vertebrae that are not labeled in ground truth
    crop_unref_vert(args.output_isotropic, args.output_crop, 'train')
    crop_unref_vert(args.output_isotropic, args.output_crop, 'test')


if __name__ == '__main__':
    main()
