from code.weakSupervision.trainval import setuplogger

from haven import haven_utils as hu


import torch
import torchvision
import tqdm
import pandas as pd
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np
import math

from pprint import pformat

from src import models
from src import datasets
from src import utils as ut
from pathlib import Path

from typing import Dict, Tuple

from torch.utils.data import DataLoader

import logging

logger = logging.getLogger(__name__)

class multi_dim_reconstructor(object):
    """Class multi dimensional reconstruction

    Args:
        model_dict (Dict[Dict]): a dictionary of experiment dictionaries that define specific models (via the haven utils hash)
        model_location (str) : path to find the models, defaults to '/root/space/output'
        model_type_name (str) :   the model type name. This will typically in ['results_dataset_D_contrast_C', 'results_weighted_dataset_D_contrast_C', 'results_full_dataset_D_contrast_C']
                                The models will be searched for by replacing 'D' with the dimension (0, 1, 2)
        dataset_location (str) : normally, this is identical to the model_location. In this folder, you will search for 'dataset_D_contrast_C' where 'C' and 'D' will be replaced for the dimension and the contrast number 
    """

    def __init__(self, model_dict : Dict, model_location : str = '/root/space/output', model_type_name : str, dataset_location : str = None, contrast : int = 3):
        def get_models(model_dict : Dict, model_location : str = 'root/space/output', model_type_name : str) -> Dict:
            """Get the models based on the model_dict and the indicated model locations

            Args:
                model_dict (Dict): {dimension : model experiment dict}
                model_location (str): path where to find the models
                model_type_name (str): model type. Here, the character 'D' will be replaced by the relevant dimensions

            Returns:
                Dict: {d : model object (for example: inst_seg)}
            """
            assert all([key in [0,1,2] for key in model_dict.keys()])

            models_dict = dict()

            for i, model_exp_dict in model_dict.items():
                model_folder = os.path.join(os.path.join(model_location), model_type_name.replace('D', i), hu.hash_dict(model_exp_dict))
                logger.debug(f'Load model in {model_folder} - Get model')
                model = models.get_model(model_dict=model_dict['model'],
                                    exp_dict=model_dict).cuda()

                model_path = os.path.join(model_folder, "model.pth")

                # If there is a pkl file containing stored model weights from the last time the model was trained, get it.
                if os.path.exists(model_path):
                    # resume experiment
                    logger.debug(f'Model weights file discovered and loaded for model on dimension {i}')
                    model.load_state_dict(hu.torch_load(model_path))
                else:
                    logging.warning('Model {model_folder} (dimension {i}) is not trained yet!')

                models_dict.update({i : model})

            return models_dict

        def get_dataloaders(model_dict : Dict, dataset_location : str = '/root/space/output', C : int = 3) -> Dict:
            """Get dictionary of dataloaders

            Args:
                model_dict (Dict): dictionary of model exp dicts
                dataset_location (str, optional): location folder of the datasets. Defaults to '/root/space/output'.

            Returns:
                Dict: {dim : {'train' : train_loader, 'test' : test_loader, 'val' : val_loader}}
            """
            assert all([key in [0,1,2] for key in model_dict.keys()])
            assert C in [i for i in range(4)]

            dataloaders = dict()

            for i, model_dict in model_dict.items():
                loaders = dict()
                for split in ['val', 'train', 'test']:
                    logging.info(f'get dataloader for split {split} and dimension {i}')
                    ds = datasets.get_dataset(dataset_dict=model_dict["dataset"],
                                        split=split,
                                        datadir=os.path.join(dataset_location, f'dataset_{i}_contrast_{C}'),
                                        exp_dict=model_dict,
                                        dataset_size=model_dict['dataset_size'])

                    # make sure this dataset contains the full crop list:
                    # this operation will also sort the selected image list
                    ds.make_full_croplist()

                    sampler = torch.utils.data.SequentialSampler(ds)

                    loaders.update({split : DataLoader(ds,
                              sampler=sampler,
                              collate_fn=ut.collate_fn,
                              batch_size=model_dict["batch_size"],
                              drop_last=False)})

                dataloaders[i] = loaders
        
        if dataset_location is None:
            dataset_location = model_location

        self.models = get_models(model_dict, model_location, model_type_name)
        self.dataloaders = get_dataloaders(model_dict, dataset_location, contrast)

        def make_3D_volumes(self, output_location):
            """Construct the 3D volumes corresponding to the datasets indicated when constructing the dataloaders

            Since the datasets are sorted, we know the images will just come in the right order

            Args:
                output_location (str): Location to output the generated 3D volumes to
            """

            def combine_crops(crops_dict : Dict, orig_shape : Tuple) -> np.ndarray:
                crop_dim = crops_dict[0].shape
                logger.debug(f'new cropdict : original size {orig_shape} resulting in crops {[i for i in crops_dict.keys()]} with dimension {crop_dim}')
                padding = (max(0, math.ceil((crop_dim[1] - orig_shape[1]) / 2)),max(0, math.ceil((crop_dim[0] - orig_shape[0]) / 2)))
                # cut off the padding
                crops_dict = {crop_nr : im[:,padding[0]: crop_dim[0] - padding[0], padding[1]: crop_dim[1] - padding[1]] for crop_nr, im in crops_dict.items()}

                result = np.empty((5, orig_shape[0], orig_shape[1], len(crops_dict)))
                result.fill(np.nan)

                count = 0
                h, w = min(orig_shape[0], crop_dim[0]), min(orig_shape[1], crop_dim[1])
                for crop_nr, im in crops_dict.items():
                    if crop_nr == 0:
                        result[:, :h, :w, count] = im
                    elif crop_nr == 1:
                        result[:, :h, -w:, count] = im
                    elif crop_nr == 2:
                        result[:, -h:, -w:, count] = im
                    elif crop_nr == 3:
                        result[:, -h:, :w, count] = im
                    else:
                        raise ValueError

                    count += 1

                # Now, average over the last dimension of result to combine the crops, ignoring the nan values (where no value was returned)
                return np.nanmean(result, axis=3)

            def combine_slices(slices_dict : Dict, stack_dim : int) -> np.ndarray:
                assert stack_dim in [0, 1, 2]
                max_slice_nr = max([i for i in slices_dict.keys()])
                slice_list = [slices_dict[i] for i in range(max_slice_nr + 1)] 
                return np.stack(slice_list, axis=stack_dim)
                
            def volumes_from_loader(model, dataloader, stack_dim, output_location):

                slice_dict = dict()
                crops_dict = dict()
                hash = None
                orig_shape = None
                scan_id = None
                slice_id = None

                for batch in tqdm(dataloader):
                    batch = model.probabilities_on_batch(batch)
                    batch_scan_ids = [m['scan_id'] for m in batch['meta']]
                    batch_slice_ids = [m['slice_id'] for m in batch['meta']]
                    batch_crop_nrs = [m['crop_nr'] for m in batch['meta']]

                    logger.debug(f'slice ids : {batch_scan_ids} with slice ids {batch_slice_ids}')
                    if scan_id is None:
                        scan_id = batch_scan_ids[0]
                    if slice_id is None:
                        slice_id = batch_slice_ids[0]
                    
                    logging.debug('shape of the probs : {}. This should be BCHW'.format(batch['probs'].shape))

                    for i, pr in enumerate(batch['probs']):
                        if (batch_scan_ids[i] == scan_id) and (batch_slice_ids[i] == slice_id): 
                            crops_dict[batch_crop_nrs[i]] = pr
                            orig_shape = batch['meta'][i]['orig_shape']
                            hash = batch['meta'][i]['hash']
                        elif batch_scan_ids[i] == scan_id: # New slice of the same scan
                            # Add the previous slice to the dict
                            slice_dict[slice_id] = combine_crops(crops_dict, orig_shape)
                            # Start a new crops_dict
                            crops_dict = {batch_crop_nrs[i] : pr}
                            # Update the current slice id
                            slice_id = batch_slice_ids[i]
                        else: # New scan
                            # add the previous slice to complete the previous scan
                            slice_dict[slice_id] = combine_crops(crops_dict, orig_shape)
                            volume = combine_slices(slice_dict, stack_dim)
                            np.save(os.path.join(output_location, f'{hash}_scan_{scan_id:03d}'), volume)
                            np.save(os.path.join(output_location, f'{hash}_scan_{scan_id:03d}_res'), np.argmax(volume, axis=0))

                            # start a new scan and a new slice
                            slice_dict = dict()
                            crops_dict = {batch_crop_nrs[i] : pr}
                            scan_id = batch_scan_ids[i]
                            slice_id = batch_slice_ids[i]

            for dim, model in self.models.items():
                logging.info(f'Start making volumes based on the model for dimension {dim}')
                for split, loader in self.dataloaders[dim].items():
                    volumes_from_loader(model, loader, dim, os.path.join(output_location, f"dimension_{dim}_split_{split}"))