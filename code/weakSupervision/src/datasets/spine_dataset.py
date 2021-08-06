import logging
from posix import sched_param
import torch
import os
import pandas as pd
import h5py
import json
import random, math
import re
from src.modules.lcfcn import lcfcn_loss
from src.datasets.StratifiedGroupKFold import StratifiedGroupKFold
from sklearn.model_selection import  GroupKFold
import SimpleITK as sitk
import numpy as np
from haven import haven_utils as hu
from torchvision import transforms
import pydicom
import tqdm
from . import transformers
from PIL import Image
import PIL
from typing import Dict, Tuple, List

from joblib import Parallel, delayed

pd.options.mode.chained_assignment = 'raise'

def get_patient_nr(filenames : dict, patients : dict, image_nr : int) -> int:
    """Get the patient number from the source filenames dict and the source patients list

    Args:
        filenames (dict): {image nr : filename}
        patients (dict): {filename : patient number}
        image_nr (int): image nr    

    Raises:
        AssertionError: [description]

    Returns:
        int: patient number
    """

    logger.debug(f'function:  Get patient nr\nfilenames : {filenames}\npatients : {patients}\nimage_nr : {image_nr}')
    fn = filenames[str(int(image_nr))]
    return patients[fn]





# Regex patterns: catch the image number from image001 & slice_001.npy
# like files
IMAGE_NR = re.compile(r'^image(\d{3})')
SLICE_NR = re.compile(r'^slice_(\d{3}).npy')

RANDOM_SEED = 10

N_CORES = -2




logger = logging.getLogger(__name__)


class SpineSets(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        datadir: str,
        exp_dict: Dict,
        separate_source: str = None,
        crop_size : List[int] = (352, 352),
        precalculated_points = False
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
        self.context_span = exp_dict['dataset']['context_span']
        self.n_classes = exp_dict['dataset']['n_classes']
        self.sources = exp_dict['dataset']['sources']
        self.size = 352
        self.crop_size = crop_size
        self.precalculated_points = precalculated_points

        self.blob_points = exp_dict['dataset']['blob_points']
        self.bg_points = exp_dict['dataset']['bg_points']

        logger.info(f'Start constructing the dataset object')
        logger.debug(f'\tdata path:\t{self.datadir}')

        img_list = list()
        if 'PLoS' in self.sources:
            PLoS_list = list() # Hold out the PLoS dataset until the end
        scan_list = list()

        # Make a list of all image and mask slices in the dataset sources.
        # after the xVertSeg dataset is processed by the python script prepare_xVertSeg.py,
        # two folders are made.
        # Folder 'images' and folder 'masks' contain a folder for each scan with all slices of that scan.
        # The image slices and mask slices are linked due to this identical
        # folder structure.
        for source in self.sources:
            tgt_path = os.path.join(datadir, f'{source}_masks')
            img_path = os.path.join(datadir, f'{source}_images')
            logger.debug(f'target path : {tgt_path}')
            if source == 'USiegen':
                with open(os.path.join(datadir, 'filenames_USiegen.json')) as f:
                    filenames_source = json.load(f)
                with open(os.path.join(datadir, 'patients_USiegen.json')) as f:
                    patients_source = json.load(f)
            for tgt_name in os.listdir(tgt_path):
                logger.debug(f'target name {tgt_name} .')
                patient_nr = int(IMAGE_NR.findall(tgt_name)[0]) if source not in ['USiegen'] else get_patient_nr(filenames_source, patients_source, IMAGE_NR.findall(tgt_name)[0])
                previous_patient_nr = patient_nr
                
                scan_id = f'{source}_{IMAGE_NR.findall(tgt_name)[0]}'
                logger.debug(f'scan id : {scan_id} .')
                scan_list.append(scan_id)
                mask_slice_path = os.path.join(tgt_path, tgt_name)
                image_slice_path = os.path.join(img_path, tgt_name)
                logger.debug(f' * mask slices path : {mask_slice_path}')
                logger.debug(f' * image slice path : {image_slice_path}')
                patient_id = f'{source}_{patient_nr:03d}'
                for mask_slice in os.listdir(mask_slice_path):
                    if not mask_slice.endswith('.npy') or mask_slice.endswith('points.npy') or mask_slice in ['points_volume.npy', 'mask_array.npy']:
                        continue
                    #logger.debug(mask_slice)
                    #logger.debug(SLICE_NR.findall(mask_slice))
                    slice_id = int(SLICE_NR.findall(mask_slice)[0])
                    if source != 'PLoS':
                        img_list += [{'img': os.path.join(image_slice_path, mask_slice),
                                        'tgt': os.path.join(mask_slice_path, mask_slice),
                                        'scan_id': scan_id,
                                        'slice_id': slice_id,
                                        'patient' : patient_id,
                                        'source': source}]
                    else:
                        PLoS_list += [{'img': os.path.join(image_slice_path, mask_slice),
                                        'tgt': os.path.join(mask_slice_path, mask_slice),
                                        'scan_id': scan_id,
                                        'slice_id': slice_id,
                                        'patient' : patient_id,
                                        'source': source}]
        scan_list.sort()

        self.full_image_df = pd.DataFrame(img_list)
        if 'PLoS' in self.sources:
            PLoS_df = pd.DataFrame(PLoS_list)

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

        patient_df = self.full_image_df.drop_duplicates('patient')

        dev_test_split = StratifiedGroupKFold(n_splits=6, random_state=RANDOM_SEED, shuffle=True)
        train_val_split = StratifiedGroupKFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)

        if 'PLoS' in self.sources:
            PLoS_dev_test_split = GroupKFold(n_splits=6)
            PLoS_train_val_split = GroupKFold(n_splits=5)

        ix_dev, ix_test = next(dev_test_split.split(X = patient_df.slice_id, y = patient_df.source, groups = patient_df.patient ))

        dev_pat_df, test_pat_df = patient_df.iloc[ix_dev], patient_df.iloc[ix_test]
        ix_train, ix_val = next(train_val_split.split(X = dev_pat_df.slice_id, y = dev_pat_df.source, groups = dev_pat_df.patient ))
        train_pat_df, val_pat_df = dev_pat_df.iloc[ix_train], dev_pat_df.iloc[ix_val]

        test_df = self.full_image_df[self.full_image_df['patient'].isin(test_pat_df.patient.tolist())]
        train_df = self.full_image_df[self.full_image_df['patient'].isin(train_pat_df.patient.tolist())]
        val_df = self.full_image_df[self.full_image_df['patient'].isin(val_pat_df.patient.tolist())]

        if 'PLoS' in self.sources:
            ix_dev, ix_test = next(PLoS_dev_test_split.split(X = PLoS_df.slice_id, y = PLoS_df.source, groups = PLoS_df.scan_id ))

            dev_PLoS_df, test_PLoS_df = PLoS_df.iloc[ix_dev], PLoS_df.iloc[ix_test]
            ix_train, ix_val = next(PLoS_train_val_split.split(X = dev_PLoS_df.slice_id, y = dev_PLoS_df.source, groups = dev_PLoS_df.scan_id ))
            train_PLoS_df, val_PLoS_df = dev_PLoS_df.iloc[ix_train], dev_PLoS_df.iloc[ix_val]

            test_df = pd.concat([test_df, test_PLoS_df], axis = 0)
            train_df = pd.concat([train_df, train_PLoS_df], axis = 0)
            val_df = pd.concat([val_df, val_PLoS_df], axis = 0)

        
        logger.debug(
            f'\t * {train_df.shape[0]} in the train set\t * {val_df.shape[0]} in the validation set\t * {test_df.shape[0]} in the test set')

        # the img_list becomes the relevant dataframe transformed again to a list of dicts & shuffle the dataframe
        self.selected_image_df = {'train' : train_df, 'val' : val_df, 'test' : test_df}[split]
        if separate_source is not None:
            logger.info(f'Only maintain the data from dataset {separate_source}.')
            logger.info(f'Dataset length before : {self.selected_image_df.shape}')
            self.selected_image_df = self.selected_image_df[self.selected_image_df['source'] == separate_source]
            logger.info(f'Dataset length after : {self.selected_image_df.shape}')

        # Calculate the amount of annotation points in each slice

        self.selected_image_df['bg_annotation_points'] = self.selected_image_df.tgt.apply(
            lambda filename : sum((np.load(filename.replace('.npy', '_points.npy')).astype('uint8') == 0))
        )
        for i in range(1,6):
            self.selected_image_df[f'class_{i}_annotation_points'] = self.selected_image_df.tgt.apply(
            lambda filename : sum((np.load(filename.replace('.npy', '_points.npy')).astype('uint8') == i))
        )

        if precalculated_points:
            logging.info(f'Before removing the slices without annotation, the selected image dataframe has {self.selected_image_df.shape[0]} rows.')
            self.selected_image_df = self.selected_image_df[self.selected_image_df[['bg_annotation_points'] + [f'class_{i}_annotation_points' for i in range(1,6)]].any(axis = 1)]
            logging.info(f'After removing the slices without annotation, the selected image dataframe has {self.selected_image_df.shape[0]} rows.')
        
        random.seed(RANDOM_SEED)
        self.selected_image_df = pd.concat([self.selected_image_df  , self.selected_image_df.img.apply(lambda _ : random.randint(0, 4)).rename('crop_nr')], axis=1)
        
        self.img_list = self.selected_image_df.to_dict(orient = 'records')

    def return_img_dfs(self) -> Tuple[pd.DataFrame]:
        """Return both the full image dataframe and the selected image dataframe
        """
        return self.full_image_df, self.selected_image_df
    
    def shuffle_img_df(self) -> None:
        """Shuffle the selected image dataframe
        """
        logger.debug('Shuffle dataset')
        self.selected_image_df = self.selected_image_df.sample(frac=1).reset_index(drop=True)

    
    def img_tgt_transform(self, image : PIL.Image, crop_nr : int = 0, normalize : bool = True):
        """Crop (pick one of the fivecrop parts)

        Args:
            image (PIL.Image): Image to crop   
            crop_nr (int, optional): crop number to select of the 5 cropped images returned by 5 crop. Defaults to 0.
            normalize (bool): should the image be normalized to be comform the ImageNet average values?
        """
        assert isinstance(crop_nr, int) and crop_nr < 5, 'Crop should be in the range [0 1 2 3 4]'

        crop_dim = self.exp_dict['dataset'].get('crop_size', (self.size, self.size))

        # Pay attention to the dimensions of the images: A FiveCrop crops with numpy dimensions (HW)
        # This means that if you ask for FiveCrop((100, 200)) you will get 5 PIL image with size (200, 100).
        # This corresponds indeed to np.ndarray.shape = (100, 200) or a tensor [3, 100, 200]

        if normalize:
            transf = transforms.Compose([
                transforms.Lambda(lambda x : transforms.Pad((
                    max(0, math.ceil((crop_dim[1] - x.size[0]) / 2)),
                    max(0, math.ceil((crop_dim[0] - x.size[1]) / 2)) ), fill=0)(x) ),
                transforms.FiveCrop(crop_dim),
                transforms.Lambda(lambda crops: crops[crop_nr]),
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                # Backbones were pre-trained on ImageNet. The images have to be
                # normalized using the ImageNet average values.
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            transf = transforms.Compose([
                transforms.Lambda(lambda x : transforms.Pad((
                    max(0, math.ceil((crop_dim[1] - x.size[0]) / 2)),
                    max(0, math.ceil((crop_dim[0] - x.size[1]) / 2)) ), fill=0)(x) ),
                transforms.FiveCrop(crop_dim),
                transforms.Lambda(lambda crops: crops[crop_nr]),
                transforms.Resize(
                    (self.size, self.size), interpolation=PIL.Image.NEAREST)])
        logger.debug(f'defined transformation : {transf}')
        return transf(image)

        

    def make_full_croplist(self):
        """This function takes the selected image dataframe and adds all relevant crops to this dataframe --> not just 1 random crop per slice 

        Step 1 : Dimensions of all slices of 1 image are the same --> extract individual images from the dataframe
        Step 2 : Get one image and define the suitable crops to take
        Step 3 : New dataframe block is the cross product of the image block with the crop_nr series 
        """
        crop_dim = self.exp_dict['dataset'].get('crop_size', (self.size, self.size))

        def expand_image(df_part : pd.DataFrame) -> pd.DataFrame:
            # Get the relevant crop numbers for 1 image (no centercrop)
            crops = list(range(4))
            img_location = df_part.sample(axis=0).img.iloc[0]

            
            im = np.load(img_location)
            logger.debug(f'Image location : {img_location} with shape {im.shape}')
            if im.shape[0] <= crop_dim[0]:
                for i in [2, 3]:
                    crops.remove(i)
            if im.shape[1] <= crop_dim[1]:
                for i in [1, 2]:
                    try:
                        crops.remove(i)
                    except ValueError:
                        pass

            logger.debug(f'Selected crops : {crops}')

            # Remove the column 'crop_nr' and take the cross product with a dataframe that contains the desired crop numbers
            cols = [col for col in df_part.columns if col != 'crop_nr']
            expanded_image = df_part[cols].merge(pd.DataFrame({'crop_nr': crops}), how='cross')

            return expanded_image

        logger.info(f'before expanding the dataframe with all relevant crops, the datafame contains {self.selected_image_df.shape[0]} rows' )
        temp = list()
        for groupname, group in self.selected_image_df.groupby('scan_id'):
            logger.debug(f'Expand the selected image dataframe for scan {groupname}')
            temp.append(expand_image(group))

        # All these subgroups are not concatenated back into a new selected images dataframe and sorted.
        # The sorting is important to assure the recombination in 3D volumes can take place correctly
        self.selected_image_df = pd.concat(temp, axis=0, ignore_index=True).sort_values(by=['scan_id', 'slice_id', 'crop_nr'])
        self.img_list = self.selected_image_df.to_dict(orient = 'records')

        logger.info(f'After expanding the dataframe with all relevant crops, the datafame contains {self.selected_image_df.shape[0]} rows' )


    def count_values_masks(self) -> Dict:
        """Count the number of occurances of each label in the complete dataset

        Returns:
            Dict: {label : occurances in the complete dataset}
        """

        def unique_vals_dict(img_list_entry):
            mask_name = img_list_entry['tgt']
            mask = np.load(mask_name)
            vals, counts = np.unique(mask, return_counts=True)
            vals_dict =  {val: count for val, count in zip(vals.tolist(), counts.tolist())}
            if self.n_classes == 2:
                vals_dict = {
                    0: vals_dict[0],
                    1 : sum([vals_dict.get(i, 0) for i in range(1,6)])
                }
            return vals_dict

        logger.info(f'Start counting the mask labels ')
        list_counts = Parallel(n_jobs=N_CORES)(delayed(unique_vals_dict)(image_list_entry) for image_list_entry in self.img_list)
        df_list_counts = pd.DataFrame(list_counts)

        logger.debug(f'Dataframe with counts : \n{df_list_counts.head()}')
        
        return df_list_counts.sum(axis=0, skipna = True).to_dict()


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
            try:
                top_name = self.selected_image_df.query(queries['top']).img.iloc[0]
            except IndexError:
                top_name = img_name
            try:
                bottom_name = self.selected_image_df.query(queries['bottom']).img.iloc[0]
            except IndexError:
                bottom_name = img_name
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

        # PIL size inverts the image dimensions (unfortunately...)
        orig_shape = image.size[::-1]
        # read annotation mask
        tgt_mask = np.load(tgt_name)

        # assert that these are the only classes
        logger.debug(
            f'new mask loaded. Classes present : {np.unique(tgt_mask)} - datatype {tgt_mask.dtype}')
        assert(len(np.setdiff1d(np.unique(tgt_mask), [x for x in range(
            6)])) == 0), 'This mask contains values other than [0,1,2,3,4,5]'

        # Background value: 0
        # mask value i corresponds to lumbar vertebra Li
        if self.n_classes == 2:
            tgt_mask[np.isin(tgt_mask,[1,2,3,4,5])] = 1
        elif self.n_classes == 6:
            # If n_classes is 6, you keep the labels for all 5 lumbar vertebrae
            # (1 -> 5) +  label 0 for the background class
            pass
        else:
            raise AssertionError



        # For the validation and test set, you want to get identical crop nr's but for the train set some more variability could be beneficial
        if self.split in ['val', 'test']:
            crop_nr = out['crop_nr']
        elif self.split == 'train':
            crop_nr = random.randint(0, 4)
        else:
            raise AssertionError
        image = self.img_tgt_transform(image, crop_nr = crop_nr, normalize = True)
        mask = self.img_tgt_transform(Image.fromarray((tgt_mask).astype('uint8')), crop_nr = crop_nr, normalize = False)
        mask = torch.LongTensor(np.array(mask))

        if not self.precalculated_points:
            # Function get_points_from_mask from src.modules.lcfcn.lcfcn_loss.py ->
            # This function takes the mask and returns an array that is 255
            # everywhere except for the background points (0) and the class points
            # (1-> 5)
            points = lcfcn_loss.get_points_from_mask(
                mask.numpy().squeeze(),
                bg_points=self.bg_points,
                blob_points=self.blob_points,center=False)
        else:
            # Get the pre-defined points and crop this image with the right crop nr
            points = self.img_tgt_transform(Image.fromarray(np.load(tgt_name.replace('.npy', '_points.npy').astype('uint8'))), crop_nr=crop_nr, normalize = False)

        logger.debug(f'shapes:')
        logger.debug(
            f'image : {image.shape} with value range {image.min()} to {image.max()}')
        logger.debug(
            f'mask : {mask.long()[None].shape} with value range {mask.min()} to {mask.max()}')
        logger.debug(f'points : {torch.LongTensor(points).shape}')

        # Together with the image, the metadata is transferred

        return {'images': image,
                'masks': mask.long()[None],
                'points': torch.LongTensor(points),
                'meta': {'shape': mask.squeeze().shape,
                         'index': i,
                         'orig_shape' : orig_shape,
                         'hash': hu.hash_dict({'id': img_name}),
                         'name': img_name,
                         'img_name': img_name,
                         'tgt_name': tgt_name,
                         'image_id': i,
                         'scan_id' : out['scan_id'],
                         'slice_id' : out['slice_id'],
                         'crop_nr' : out['crop_nr'],
                         'split': self.split}}

    def __len__(self):
        return len(self.img_list)
