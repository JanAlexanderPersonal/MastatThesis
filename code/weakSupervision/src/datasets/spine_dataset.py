import logging
from posix import sched_param
import torch
import os
import pandas as pd
import h5py
import random
import re
from src.modules.lcfcn import lcfcn_loss
from src.datasets.StratifiedGroupKFold import StratifiedGroupKFold
import SimpleITK as sitk
import numpy as np
from haven import haven_utils as hu
from torchvision import transforms
import pydicom
import tqdm
from . import transformers
from PIL import Image
import PIL
from typing import Dict, Tuple

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
        separate_source: str = None,
        context_span : int = 0
    ):
        """xVertSeg calss : inherits from torch.utils.data.Dataset

        Args:
            split (str): Is this the dataloader for the train, the test of the validation set? ['train', 'test', 'val']
            datadir (str): path to the data
            exp_dict (Dict): Dictionary with the experiment definition
            separate_source (str, optional): name of source you want to separate out for more detailed metrics calculation. Defaults to True.

        attributes:
            exp_dict (Dict)
            datadir (str)
            split (str)
            n_classes (int) : number of classes, apart from the background. This is extracted from exp_dict['dataset']['n_classes']
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

        assert split in ['train', 'test', 'val'], "Only 3 split types are allowed: train, test and val"


        self.exp_dict = exp_dict
        self.datadir = datadir
        self.split = split
        self.context_span = context_span
        self.n_classes = exp_dict['dataset']['n_classes']
        self.sources = exp_dict['dataset']['sources']
        self.size = 352

        self.blob_points = exp_dict['dataset']['blob_points']
        self.bg_points = exp_dict['dataset']['bg_points']

        logger.info(f'Start constructing the dataset object')
        logger.debug(f'\tdata path:\t{self.datadir}')

        img_list = list()
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
            patient_nr = 0
            for tgt_name in os.listdir(tgt_path):
                logger.debug(f'target name {tgt_name} .')
                scan_id = f'{source}_{IMAGE_NR.findall(tgt_name)[0]}'
                logger.debug(f'scan id : {scan_id} .')
                scan_list.append(scan_id)
                mask_slice_path = os.path.join(tgt_path, tgt_name)
                image_slice_path = os.path.join(img_path, tgt_name)
                logger.debug(f' * mask slices path : {mask_slice_path}')
                logger.debug(f' * image slice path : {image_slice_path}')
                patient_id = f'{source}_{patient_nr:03d}'
                for mask_slice in os.listdir(mask_slice_path):
                    if not mask_slice.endswith('.npy'):
                        continue
                    slice_id = int(SLICE_NR.findall(mask_slice)[0])
                    img_list += [{'img': os.path.join(image_slice_path, mask_slice),
                                    'tgt': os.path.join(mask_slice_path, mask_slice),
                                    'scan_id': scan_id,
                                    'slice_id': slice_id,
                                    'patient' : patient_id,
                                    'source': source}]
        scan_list.sort()

        self.full_image_df = pd.DataFrame(img_list)

        num_scans = len(scan_list)
        logger.debug(f'{num_scans} scans found that contain in total {len(img_list)} images.')
        logger.debug(f'as dataframe: {self.full_image_df.head(10)}')

        # Train-Test split:
        # By using a fixed random seed, the split between train, validation and test is always the same.
        # • Train set : 4/6 of the dataset
        # • Validation set : 1/6 of the dataset
        # • Test set : 1/6 of the dataset
        #
        # IMPORTANT: For other datasets, this should be adapted to assure scans of the same patients are not mixed.
        #
        # seed is fixed at RANDOM_SEED
        # random.seed(RANDOM_SEED)

        dev_test_split = StratifiedGroupKFold(n_splits=6, random_state=RANDOM_SEED, shuffle=True)
        train_val_split = StratifiedGroupKFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
        logger.debug(dev_test_split.split(X = self.full_image_df.slice_id, y = self.full_image_df.source, groups = self.full_image_df.patient ))
        ix_dev, ix_test = next(dev_test_split.split(X = self.full_image_df.slice_id, y = self.full_image_df.source, groups = self.full_image_df.patient ))

        dev_df, test_df = self.full_image_df.iloc[ix_dev], self.full_image_df.iloc[ix_test]
        ix_train, ix_val = next(train_val_split.split(X = dev_df.slice_id, y = dev_df.source, groups = dev_df.patient ))
        train_df, val_df = dev_df.iloc[ix_train], dev_df.iloc[ix_val]

        
        logger.info(
            f'\t * {train_df.shape[0]} in the train set\t * {val_df.shape[0]} in the validation set\t * {test_df.shape[0]} in the test set')

        # the img_list becomes the relevant dataframe transformed again to a list of dicts
        self.selected_image_df = {'train' : train_df, 'val' : val_df, 'test' : test_df}[split]
        if separate_source is not None:
            self.selected_image_df = self.selected_image_df[self.selected_image_df['source'] == separate_source]
        self.img_list = self.selected_image_df.to_dict(orient = 'records')


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

    def return_img_dfs(self) -> Tuple[pd.DataFrame]:
        """Return both the full image dataframe and the selected image dataframe
        """
        return self.full_image_df, self.selected_image_df

    def __getitem__(self, i) -> Dict:
        """get item i from dataset loader

        Args:
            i (int): index of the item to return

        Returns:
            Dict: Dict containing the image, the mask, the annotation points and meta-data
        """
        out = self.img_list[i]
        img_name, tgt_name = out['img'], out['tgt']

        if self.context_span > 0:
            queries = {name : 'scan_id == \"{}\" & patient == \"{}\" & slice_id == {}'.format(out['scan_id'], out['patient'], out['slice_id'] + offset) for name, offset in zip(['top', 'bottom'], [1 * self.context_span, -1 * self.context_span ])}
            top_name = self.selected_image_df(queries['top']).img.iloc[0]
            bottom_name = self.selected_image_df(queries['bottom'].img.iloc[0])
            logger.debug(f'image path : {img_name} * top path : {top_name} * bottom name : {bottom_name}')
            # Load the images
            image = np.load(img_name)
            top = np.load(top_name)
            bottom = np.load(bottom_name)
            layers = np.stack([top, image, bottom], axis=2)
            image = Image.fromarray((layers * 255).astype('uint8'))
        else:
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
