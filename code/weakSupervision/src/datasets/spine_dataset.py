import logging
import torch
import os
import h5py
import random
import re
from src.modules.lcfcn import lcfcn_loss
import SimpleITK as sitk
import numpy as np
from haven import haven_utils as hu
from torchvision import transforms
import pydicom
import tqdm
from . import transformers
from PIL import Image
import PIL
from typing import Dict

# Regex patterns: catch the image number from image001 & slice_001.npy
# like files
IMAGE_NR = re.compile(r'^image(\d{3})')
SLICE_NR = re.compile(r'^slice_(\d{3}).npy')

RANDOM_SEED = 10

# Center crop dimension: in the pre-processing step, the image is center-cropped.
# Todo: Vormt dit geen conflict met self.size?
CenterCrop_dim = (352, 352)  # was (384, 385)


logger = logging.getLogger(__name__)


class SpineSets(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        datadir: str,
        exp_dict: Dict,
        seperate: bool = True,
    ):
        """xVertSeg calss : inherits from torch.utils.data.Dataset

        Args:
            split (str): Is this the dataloader for the train, the test of the validation set? ['train', 'test', 'val']
            datadir (str): path to the data
            exp_dict (Dict): Dictionary with the experiment definition
            seperate (bool, optional): [description]. Defaults to True.

        attributes:
            exp_dict (Dict)
            datadir (str)
            split (str)
            n_classes (int) : number of classes, apart from the background. This is extracted from exp_dict['dataset']['n_classes']
            size (int) : xxx
            img_path (path object) : path to image folders
            tgt_path (path object) : path to the target (mask) folders
            image_list (List[Dict]) : a list of dictionaries. Each dictionary contains
                                            {
                                                'img' : (str) image path,
                                                'tgt' : (str) path to the target slice,
                                                'scan_id' : (int) Number of the scan in the xVertSeg dataset,
                                                'slice_id' : (int) Number of the slice within this scan
                                            }

        """
        self.exp_dict = exp_dict
        self.datadir = datadir
        self.split = split
        self.n_classes = exp_dict['dataset']['n_classes']
        self.sources = exp_dict['dataset']['sources']
        self.size = 352

        self.blob_points = exp_dict['dataset']['blob_points']
        self.bg_points = exp_dict['dataset']['bg_points']

        logger.info(f'Start constructing the dataset object')
        logger.debug(f'\tdata path:\t{self.datadir}')

        self.img_list = list()
        scan_list = list()

        # Make a list of all image and mask slices in the xVertSeg dataset.
        # after the xVertSeg dataset is processed by the python script prepare_xVertSeg.py,
        # two folders are made.
        # Folder 'images' and folder 'masks' contain a folder for each scan with all slices of that scan.
        # The image slices and mask slices are linked due to this identical
        # folder structure.
        for source in self.sources:
            tgt_path = os.path.join(datadir, f'{source}_masks')
            img_path = os.path.join(datadir, f'{source}_images')
            logger.debug(f'target path : {tgt_path}')
            for tgt_name in os.listdir(tgt_path):
                logger.debug(f'target name {tgt_name} .')
                scan_id = f'{source}_{IMAGE_NR.findall(tgt_name)[0]}'
                logger.debug(f'scan id : {scan_id} .')
                scan_list.append(scan_id)
                mask_slice_path = os.path.join(tgt_path, tgt_name)
                image_slice_path = os.path.join(img_path, tgt_name)
                logger.debug(f' * mask slices path : {mask_slice_path}')
                logger.debug(f' * image slice path : {image_slice_path}')
                for mask_slice in os.listdir(mask_slice_path):
                    if not mask_slice.endswith('.npy'):
                        continue
                    slice_id = int(SLICE_NR.findall(mask_slice)[0])
                    self.img_list += [{'img': os.path.join(image_slice_path, mask_slice),
                                    'tgt': os.path.join(mask_slice_path, mask_slice),
                                    'scan_id': scan_id,
                                    'slice_id': slice_id}]
        scan_list.sort()

        num_scans = len(scan_list)
        logger.debug(f'{num_scans} scans found that contain in total {len(self.img_list)} images.')

        # Train-Test split:
        # By using a fixed random seed, the split between train, validation and test is always the same.
        # • Train set : 4/6 of the dataset
        # • Validation set : 1/6 of the dataset
        # • Test set : 1/6 of the dataset
        #
        # IMPORTANT: For other datasets, this should be adapted to assure scans of the same patients are not mixed.
        #
        # seed is fixed at RANDOM_SEED
        random.seed(RANDOM_SEED)

        DIST_OK = False

        while not DIST_OK:
            train_list = random.sample(scan_list, int(num_scans / 6 * 4))
            val_list = random.sample(
                [i for i in scan_list if i not in train_list], int(num_scans / 6) + 1)
            test_list = [
                i for i in scan_list if (
                    i not in train_list) and (
                    i not in val_list)]
            train_list_size = 0
            val_list_size = 0
            test_list_size = 0

            for img in self.img_list:
                if img['scan_id'] in train_list:
                    train_list_size += 1
                elif img['scan_id'] in val_list:
                    val_list_size += 1
                elif img['scan_id'] in test_list:
                    test_list_size += 1
                else:
                    logger.warning('The split between train, test and val went wrong')
            logger.debug(f'Attempt: train list size : {train_list_size} images ({train_list_size / num_scans * 100:3.1f} %).')
            logger.debug(f'Attempt: val list size : {val_list_size} images ({val_list_size / num_scans * 100:3.1f} %).')
            logger.debug(f'Attempt: test list size : {test_list_size} images ({test_list_size / num_scans * 100:3.1f} %).')

            DIST_OK = (train_list_size / num_scans > 0.6) and (val_list_size / num_scans > 0.1) and (test_list_size / num_scans > 0.1)

        
        logger.info(
            f'\t * {len(train_list)} in the train set\t * {len(val_list)} in the validation set\t * {len(test_list)} in the test set')
        logger.debug(f'train_list : {sorted(train_list)}')
        logger.debug(f'val_list : {sorted(val_list)}')
        logger.debug(f'test_list : {sorted(test_list)}')

        # Todo: implementation for non-separate part
        # Only keep the appropriate part of the scan list
        if seperate:
            if split == 'train':
                scan_list = train_list
            elif split == 'val':
                scan_list = val_list
            elif split == 'test':
                scan_list = test_list

            # Only keep the images in the image list from the selected scans
            img_list_new = []
            for img_dict in self.img_list:
                if img_dict['scan_id'] in scan_list:
                    img_list_new += [img_dict]
            random.shuffle(img_list_new)

        self.img_list = img_list_new

        # Todo: Why these 3 different options? Resizing after this centercrop
        # is actually not very useful anymore.

        self.img_transform = transforms.Compose([
            transforms.CenterCrop(CenterCrop_dim),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            # Backbones were pre-trained on ImageNet. The images have to be
            # normalized using the ImageNet average values.
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        if split == 'train':
            self.gt_transform = transforms.Compose([

                transforms.CenterCrop(CenterCrop_dim),
                transforms.Resize(
                    (self.size, self.size), interpolation=PIL.Image.NEAREST),
                # transforms.ToTensor()]
            ])
        else:
            self.gt_transform = transforms.Compose([
                transforms.CenterCrop(CenterCrop_dim),
                # transforms.ToTensor()
            ])

        logger.info('Dataset xVertSeg prepared')

    def __getitem__(self, i) -> Dict:
        """get item i from dataset loader

        Args:
            i (int): index of the item to return

        Returns:
            Dict: Dict containing the image, the mask, the annotation points and meta-data
        """
        out = self.img_list[i]
        img_name, tgt_name = out['img'], out['tgt']

        # read image from the preprocessed image slices
        image = np.load(img_name)
        image = Image.fromarray((image * 255).astype('uint8')).convert('RGB')

        # read annotation mask
        tgt_mask = np.load(tgt_name)

        # assert that these are the only classes
        logger.debug(
            f'new mask loaded. Classes present : {np.unique(tgt_mask)} - datatype {tgt_mask.dtype}')
        assert(len(np.setdiff1d(np.unique(tgt_mask), [x for x in range(
            6)])) == 0), 'This mask contains values other than [0,1,2,3,4,5]'

        # Background value: 0
        # mask value i corresponds to lumbar vertebra Li
        if self.n_classes == 1:
            tgt_mask[tgt_mask != 0] = 1
        elif self.n_classes == 6:
            # If n_classes is 6, you keep the labels for all 5 lumbar vertebrae
            # (1 -> 5) +  label 0 for the background class
            pass
        else:
            raise AssertionError

        # The image and the mask could be transformed before going further
        # image, mask = transformers.apply_transform(self.split, image=image, label=mask,
        #                                transform_name=self.exp_dict['dataset']['transform'],
        #                                exp_dict=self.exp_dict)
        #
        # Convert the image to [0 -> 255] range
        # img_uint8 = ((image/4095)*255).astype('uint8')

        # REMARK: black and white scan slice image is converted to RGB
        image = self.img_transform(image)
        mask = self.gt_transform(Image.fromarray((tgt_mask).astype('uint8')))
        mask = torch.LongTensor(np.array(mask))

        # todo: Check this part well! The code seems very complicated to do something simple.
        # Function get_points_from_mask from src.modules.lcfcn.lcfcn_loss.py ->
        # This function takes the mask and returns an array that is 255
        # everywhere except for the background points (0) and the class points
        # (1-> 5)
        points = lcfcn_loss.get_points_from_mask(
            mask.numpy().squeeze(),
            bg_points=self.bg_points,
            blob_points=self.blob_points,center=False)

        logger.debug(f'shapes:')
        logger.debug(
            f'image : {image.shape} with value range {image.min()} to {image.max()}')
        logger.debug(
            f'mask : {mask.long()[None].shape} with value range {mask.min()} to {mask.max()}')
        logger.debug(f'points : {torch.LongTensor(points).shape}')

        return {'images': image,
                'masks': mask.long()[None],
                'points': torch.LongTensor(points),
                'meta': {'shape': mask.squeeze().shape,
                         'index': i,
                         'hash': hu.hash_dict({'id': img_name}),
                         'name': img_name,
                         'img_name': img_name,
                         'tgt_name': tgt_name,
                         'image_id': i,
                         'split': self.split}}

    def __len__(self):
        return len(self.img_list)
