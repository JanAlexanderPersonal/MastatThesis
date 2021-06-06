# Util functions for SimpleITK experiments

# import libraries
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from typing import  List



def resampler(image : sitk.SimpleITK.Image, new_spacing : List[float] = None):
    """resampler to isotropic spacing

    Args:
        image (sitk.SimpleITK.Image): image to resample
        new_spacing (List[float], optional): List of three floats representing the new desired spacing. Defaults to None.

    Returns:
        sitk.SimpleITK.Image: resampled image on the new grid (default 1 mm × 1 mm × 1 mm)
    """
    # Unless a specific spacing is given, the default spacing is 1 mm × 1 mm × 1 mm 
    if new_spacing is None:
        new_spacing = [1, 1, 1]

    resampler = sitk.ResampleImageFilter()
    # Nearest neighbour interpolation to avoid disturbing the labels
    # Todo: Adapt this for non-label images -> not necessary to use nearest neighbor
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

    isotropic_img = resampler.Execute(image)

    return isotropic_img
