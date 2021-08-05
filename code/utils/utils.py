# Util functions for SimpleITK experiments

# import libraries
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import  cm

from typing import  List, Tuple

from joblib import Parallel, delayed

from skimage import exposure
from skimage.restoration import denoise_bilateral
import logging
from PIL import Image

import sys
sys.path.append(r'/root/space/code/weakSupervision/')

from src.modules.lcfcn import lcfcn_loss

N_CORES = -1

BG_POINTS = 3
BLOB_POINTS = 1


def resampler(image : sitk.SimpleITK.Image, new_spacing : List[float] = None, imposed_size : List[int] = None, mask = False):
    """resampler to isotropic spacing

    Args:
        image (sitk.SimpleITK.Image): image to resample
        new_spacing (List[float], optional): List of three floats representing the new desired spacing. Defaults to None.
        imposed_size (List[int], optional): you can pass a desired output size (in numpy coordinates! --> z y x)

    Returns:
        sitk.SimpleITK.Image: resampled image on the new grid (default 1 mm × 1 mm × 1 mm)
    """
    # Unless a specific spacing is given, the default spacing is 1 mm × 1 mm × 1 mm 
    if new_spacing is None:
        new_spacing = [1, 1, 1]

    resampler = sitk.ResampleImageFilter()
    # Nearest neighbour interpolation to avoid disturbing the labels
    # Todo: Adapt this for non-label images -> not necessary to use nearest neighbor
    if mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)

    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = image.GetSpacing()
    new_size = np.array([x * (y / z) for x, y, z in zip(orig_size, orig_spacing, new_spacing)])
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]

    if imposed_size is not None:
        # imposed size will be in z y x (numpy) instead of x y z (Simple ITK):
        if any([(abs(new_size[i] - imposed_size[j]) > 5) for i, j in zip([0, 1, 2], [2, 1, 0])  ]):
            logging.warning(f'Large difference between calculated size and imposed size! Calculated : {new_size} vs imposed {imposed_size}')
        new_size = imposed_size[::-1]

    resampler.SetSize(new_size)
    logging.info(f'resample to size {new_size}')

    isotropic_img = resampler.Execute(image)

    return isotropic_img


rescale = sitk.RescaleIntensityImageFilter()
min_max = sitk.MinimumMaximumImageFilter()


def array_from_file(file_path : str) -> Tuple:
    """ Get array from image file and return the corresponding numpy array and the min / max values

    """

    image = sitk.ReadImage(file_path)
    min_max.Execute(image)
    image = resampler(rescale.Execute(image))

    arr = sitk.GetArrayFromImage(image).astype('float16')
    arr /= 255.0

    return arr, min_max.GetMinimum(), min_max.GetMaximum()


def adjust_contrast(arr : np.ndarray, contrast_option:int = 0) -> np.ndarray:
    """Select a contrast improvement algorithm to be executed on the array slice.

    Args:
        arr (np.ndarray): 2D array representing the image of which the contrast should be improved
        contrast_option (int, optional): number indicates the contrast options. Defaults to 0.

    Returns:
        np.ndarray: array with improved contrast
    """

    if contrast_option == 0:
        return arr
    elif contrast_option in [1 ,2]:
        arr = exposure.equalize_hist(arr, nbins=256, mask=(arr > 0.05))
    elif contrast_option in [3, 4]:
        # Some type conversions to avoid strange behaviour of this function for float types --> not cleared out, but this works.
        arr = exposure.equalize_adapthist((arr * 255).astype('uint16'), nbins=256, kernel_size=50, clip_limit=9.0/1000.0)
    else:
        raise NotImplementedError

    if contrast_option in [2, 4]:
        arr = denoise_bilateral(arr)

    return arr.astype('float16')

    



def arr_slices_save(arr : np.ndarray, dim_slice : int, fn : str, contrast_option : int, save_jpeg : bool = True, invert=False):
    """Save the image slices with optional contrast enhancement

    Args:
        arr (np.ndarray): 3D array containing the isotropic scan image
        dim_slice (int): dimension to slice along
        fn (str): filename and path to 
        contrast_option (int): option for the contrast improvement
        save_jpg (bool, optional): Should jpg visualizations of the slices be saved? Defaults to True.
    """

    def save_scan_slice(i):
        # This function makes use of the parameters available in the scope of 'arr_slices_save'
        slice_to_save = adjust_contrast(arr.take(i, axis=dim_slice), contrast_option)
        if invert:
            slice_to_save = np.subtract(np.arange(1.0), slice_to_save)
        np.save(os.path.join(fn, f'slice_{i:03d}'), slice_to_save)
        if save_jpeg:
            # for jpeg visualization, get back to the original 0 -> 255 range.
            im = Image.fromarray((slice_to_save * 255).astype(np.uint8))
            im.convert('RGB').save(os.path.join(fn, f'slice_{i:03d}.jpg')) # :03d means 3 digits -> leading 0s

    # Execute the function parallel --> only one image in memory but the contrast improvement and conversion to jpg is parallel
    np.save(os.path.join(fn, 'volume'), arr)
    Parallel(n_jobs=N_CORES)(delayed(save_scan_slice)(i) for i in range(arr.shape[dim_slice]))

def mask_to_slices_save(arr : np.ndarray, dim_slice : int, target_folder : str):
    """Split the mask file in slices and save them

    Args:
        arr (np.ndarray): Array containing the mask file
        dim_slice (int): spacial dimension to slice
        target_folder (str): Folder where you want the slices to end up
    """

    def save_mask_slice(i):
        fn = os.path.join(target_folder, f'slice_{i:03d}') # :03d means 3 digits -> leading 0s
        # Take the index from the desired axis and save this slice (numpy.ndarray) for the model to train on.
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.take.html#numpy.ndarray.take
        arr_slice = arr.take(i, axis=dim_slice)
        np.save(fn, arr_slice)

        if True:
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

    def points_from_mask_slice(i):
        """Function that will extract a number of background points and blob points from a slice

        Args:
            i (int): Index of the slice you want to extract

        Returns:
            Tuple: (index, np.ndarray that is 255 everywhere except for a number of points)
        """
        fn = os.path.join(target_folder, f'slice_{i:03d}_points') # :03d means 3 digits -> leading 0s
        arr_slice = arr.take(i, axis=dim_slice)
        points = lcfcn_loss.get_points_from_mask(
            arr_slice.astype('int64').squeeze(),
            bg_points=BG_POINTS,
            blob_points=BLOB_POINTS,center=False)
        np.save(fn, points)
        return (i , points)

    # Make one parallel pool for both operations we want to perform
    parallel_pool = Parallel(n_jobs=N_CORES)
    parallel_pool(delayed(save_mask_slice)(i) for i in range(arr.shape[dim_slice]))
    # Get list of (slice_nr , points_slice) this can be sorted and transformed to a volume easily
    points_list = parallel_pool([delayed(points_from_mask_slice)(i) for i in range(arr.shape[dim_slice])])
    # sort the list you obtained by the first value
    points_list.sort(key=lambda a: a[0])
    points = np.stack([points_slice for slice_index, points_slice in points_list ],  axis = dim_slice)
    np.save(os.path.join(target_folder, f'points_volume'), points)


def read_masklist(source_filenames : List[str], imposed_size = None) -> List:
    """ Read list of maskfiles """
    return [sitk.GetArrayFromImage( resampler( sitk.ReadImage(source_filename) ,mask=True, imposed_size= imposed_size)) for source_filename in source_filenames]
