import random
import numpy as np
import logging
from data.data_augmentation import elastic_transform, gaussian_blur, gaussian_noise, random_crop

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

def force_inside_img(x, patch_size, img_shape):
    assert img_shape >= patch_size, f'path size {patch_size} is too large for this image'

    x_low = max(0, int(x - patch_size / 2))
    x_up = x_low + patch_size
    if x_up > img_shape:
        x_up = min(img_shape, int(x + patch_size / 2))
        x_low = x_up - patch_size

    logging.debug(f'img_dim = {img_shape} * x={x} * x_low = {x_low} * x_up = {x_up} * diff={x_up-x_low}')
    return x_low, x_up


def extract_random_patch(img, mask, weight, i, subset, empty_interval=5, patch_size=128, augmentation = False):
    flag_empty = False

    # list available vertebrae
    verts = np.unique(mask)
    logging.debug(f'vertebrae present : {verts}')
    chosen_vert = verts[random.randint(1, len(verts) - 1)]
    logging.debug(f'chosen vertebrae : {chosen_vert}')

    # create corresponde instance memory and ground truth
    ins_memory = np.copy(mask)
    ins_memory[ins_memory <= chosen_vert] = 0
    ins_memory[ins_memory > 0] = 1

    gt = np.copy(mask)
    gt[gt != chosen_vert] = 0
    gt[gt > 0] = 1

    # send empty mask sample in certain frequency
    if i % empty_interval == 0:
        patch_center = [np.random.randint(0, s) for s in img.shape]
        x = patch_center[2]
        y = patch_center[1]
        z = patch_center[0]

        # for instance memory
        gt = np.copy(mask)
        flag_empty = True
    else:
        indices = np.nonzero(mask == chosen_vert)
        lower = [np.min(i) for i in indices]
        upper = [np.max(i) for i in indices]
        # random center of patch
        x = random.randint(lower[2], upper[2])
        y = random.randint(lower[1], upper[1])
        z = random.randint(lower[0], upper[0])

    logging.debug(f'image shape : {img.shape}')

    # force random patches' range within the image
    # Pay attention to the sequency of the axis!
    x_low, x_up = force_inside_img(x, patch_size, img.shape[2])
    y_low, y_up = force_inside_img(y, patch_size, img.shape[1])
    z_low, z_up = force_inside_img(z, patch_size, img.shape[0])

    debug_line = list()
    for vol in [img, ins_memory, gt, weight]:
        debug_line.append(f'type {type(vol)} - shape {vol.shape}')

    for line in debug_line:
        logging.debug(line)

    # crop the patch
    img_patch = img[z_low:z_up, y_low:y_up, x_low:x_up]
    ins_patch = ins_memory[z_low:z_up, y_low:y_up, x_low:x_up]
    gt_patch = gt[z_low:z_up, y_low:y_up, x_low:x_up]
    weight_patch = weight[z_low:z_up, y_low:y_up, x_low:x_up]

    if not all([i==patch_size for i in img_patch.shape]):
        logging.error('Bad cropping!')
        for line in debug_lines:
            logging.error(line)

    #  if the label is empty mask
    if flag_empty:
        ins_patch = np.copy(gt_patch)
        ins_patch[ins_patch > 0] = 1
        gt_patch = np.zeros_like(ins_patch)
        weight_patch = np.ones_like(ins_patch)

    if augmentation:
        # Randomly on-the-fly Data Augmentation
        # 50% chance elastic deformation
        if subset == 'train':
            if np.random.rand() > 0.5:
                logging.debug('apply elastic deformation')
                img_patch, gt_patch, ins_patch, weight_patch = elastic_transform(img_patch, gt_patch, ins_patch,
                                                                                weight_patch, alpha=20, sigma=5)
            # 50% chance gaussian blur
            if np.random.rand() > 0.5:
                logging.debug('apply gaussian blur')
                img_patch = gaussian_blur(img_patch)
            # 50% chance gaussian noise
            if np.random.rand() > 0.5:
                logging.debug('apply gaussian noise')
                img_patch = gaussian_noise(img_patch)

            # 50% random crop along z-axis
            if np.random.rand() > 0.5:
                logging.debug('apply random crop')
                img_patch, ins_patch, gt_patch, weight_patch = random_crop(img_patch, ins_patch, gt_patch
                                                                        , weight_patch)

    # decide label of completeness(partial or complete)
    vol = np.count_nonzero(gt == 1)
    sample_vol = np.count_nonzero(gt_patch == 1)
    c_label = 0 if float(sample_vol / (vol + 0.0001)) < 0.98 else 1

    img_patch = np.expand_dims(img_patch, axis=0)
    ins_patch = np.expand_dims(ins_patch, axis=0)
    gt_patch = np.expand_dims(gt_patch, axis=0)
    weight_patch = np.expand_dims(weight_patch, axis=0)
    c_label = np.expand_dims(c_label, axis=0)

    debug_lines = ['Final result', 
                    f'\timage patch shape :\t{img_patch.shape}',
                    f'\tinstance memory patch shape :\t{ins_patch.shape}',
                    f'\tground truth patch :\t{gt_patch.shape}', 
                    f'\tweight patch shape :\t{weight_patch.shape}']

    

    for line in debug_lines:
        logging.debug(line)

    return img_patch, ins_patch, gt_patch, weight_patch, c_label
