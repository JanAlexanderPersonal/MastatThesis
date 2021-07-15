from haven import haven_utils as hu


import torch
import torchvision
from tqdm import tqdm
import pandas as pd
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np
import math
import warnings

from pprint import pformat

from src import models
from src import datasets
from src import utils as ut
from pathlib import Path

from multiprocessing import Process

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

    def __init__(self, model_dict : Dict, model_type_name : str, model_location : str = '/root/space/output',  dataset_location : str = None, contrast : int = 3):
        def get_models(model_dict : Dict, model_type_name : str, model_location : str = 'root/space/output') -> Dict:
            """Get the models based on the model_dict and the indicated model locations

            Args:
                model_dict (Dict): {dimension : model experiment dict}
                model_location (str): path where to find the models
                model_type_name (str): model type. Here, the character 'D' will be replaced by the relevant dimensions

            Returns:
                Dict: {d : model object (for example: inst_seg)}
            """
            logger.debug('Start collecting the models')
            assert all([key in [0,1,2] for key in model_dict.keys()])

            logger.debug(f'Model type name \n{model_type_name}.')

            

            models_dict = dict()

            for i, model_exp_dict in model_dict.items():
                logger.debug(f'Dimension : {i}')
                logger.debug(f'experiment dict : {pformat(model_exp_dict)}')
                model_folder = os.path.join(os.path.join(model_location), model_type_name.replace('D', str(i)), hu.hash_dict(model_exp_dict))
                logger.debug(f'Load model in {model_folder} - Get model')
                model = models.get_model(model_exp_dict['model'], exp_dict=model_exp_dict).cuda()

                model_path = os.path.join(model_folder, "model.pth")

                # If there is a pkl file containing stored model weights from the last time the model was trained, get it.
                if os.path.exists(model_path):
                    # resume experiment
                    logger.debug(f'Model weights file discovered and loaded for model on dimension {i}')
                    model.load_state_dict(hu.torch_load(model_path))
                else:
                    logger.warning(f'Model {model_folder} (dimension {i}) is not trained yet!')
                    input('press enter')

                models_dict.update({i : model})
            

            return models_dict

        def get_dataloaders(model_dict : Dict, dataset_location : str = '/root/space/output', C : int = 3) -> Dict:
            """Get dictionary of dataloaders for every model, we have 3 dataloaders:
                    train, test and val (cross-validation set)

            Args:
                model_dict (Dict): dictionary of model exp dicts
                dataset_location (str, optional): location folder of the datasets. Defaults to '/root/space/output'.

            Returns:
                Dict: {dim : {'train' : train_loader, 'test' : test_loader, 'val' : val_loader}}
            """
            logger.debug('Get the dataloaders')
            assert all([key in [0,1,2] for key in model_dict.keys()])
            assert C in [i for i in range(4)]

            dataloaders = dict()

            for i, model_dict in model_dict.items():
                loaders = dict()
                for split in ['val', 'train', 'test']:
                    logger.info(f'get dataloader for split {split} and dimension {i}')
                    ds = datasets.get_dataset(dataset_dict=model_dict["dataset"],
                                        split=split,
                                        datadir=os.path.join(dataset_location, f'dataset_{i}_contrast_{C}'),
                                        exp_dict=model_dict,
                                        dataset_size=model_dict['dataset_size'])

                    # make sure this dataset contains the full crop list:
                    # this operation will also sort the selected image list
                    ds.make_full_croplist()
                    _, img_df = ds.return_img_dfs()
                    img_df.to_csv(os.path.join(dataset_location, 'full_croplist.csv'))

                    sampler = torch.utils.data.SequentialSampler(ds)

                    loaders.update({split : DataLoader(ds,
                              sampler=sampler,
                              collate_fn=ut.collate_fn,
                              batch_size=model_dict["batch_size"],
                              drop_last=False)})

                dataloaders[i] = loaders
            return dataloaders
        
        if dataset_location is None:
            dataset_location = model_location

        self.models = get_models(model_dict, model_type_name.replace('C', str(contrast)),  model_location)
        self.dataloaders = get_dataloaders(model_dict, dataset_location, contrast)
        logger.info(f'Models and dataloaders loaded')

    def make_3D_volumes(self, output_location):
        """Construct the 3D volumes corresponding to the datasets indicated when constructing the dataloaders

        Since the datasets are sorted, we know the images will just come in the right order

        Args:
            output_location (str): Location to output the generated 3D volumes to
        """

        def combine_crops(crops_dict : Dict, orig_shape : Tuple) -> np.ndarray:
            """Function to combine the results (probabilities as sigmoid of logits) for each of the image slice crops  

            Args:
                crops_dict (Dict): { crop_nr : crop results (probabilities) as np.ndarray }
                orig_shape (Tuple): (H,W) original shape of the slice

            Raises:
                ValueError: [description]

            Returns:
                np.ndarray: 
            """
            crop_dim = crops_dict[0].shape # This results in the dimensions [Channels, H, W] --> to get H & W, you need crop_dim[1] & crop_dim[2]
            logger.debug(f'new cropdict : original size {orig_shape} resulting in crops {[i for i in crops_dict.keys()]} with dimension {crop_dim}')
            padding = (max(0, (crop_dim[1] - orig_shape[0]) / 2.0),max(0, (crop_dim[2] - orig_shape[1]) / 2.0))
            # cut off the padding
            logger.debug(f'padding : {padding}')
            logger.debug('Dimensions before cutting the padding : {}'.format({crop_nr : im.shape for crop_nr, im in crops_dict.items()}))
            crops_dict = {crop_nr : im[:,math.floor(padding[0]): crop_dim[1] - math.ceil(padding[0]), math.floor(padding[1]): crop_dim[2] - math.ceil(padding[1])] for crop_nr, im in crops_dict.items()}
            logger.debug('Dimensions after cutting the padding : {}'.format({crop_nr : im.shape for crop_nr, im in crops_dict.items()}))

            # Make an empty array that will contain all the crops, fill it with NaN
            result = np.empty((crop_dim[0], orig_shape[0], orig_shape[1], len(crops_dict)))
            result.fill(np.nan)
            logger.debug(f'Result dimensions: {result.shape}')

            # Crops only give partial information, each time a part of the results table stays low
            count = 0
            h, w = min(orig_shape[0], crop_dim[1]), min(orig_shape[1], crop_dim[2])
            logger.debug(f'h = {h} & w = {w}')
            for crop_nr, im in crops_dict.items():
                logger.debug(f'Add crop {crop_nr} to result ')
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
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                np.nanmean(result, axis=3)


            return np.nanmean(result, axis=3)

        def combine_slices(slices_dict : Dict, stack_dim : int) -> np.ndarray:
            """combine a dictionary of slices to a complete 3D array with

            Args:
                slices_dict (Dict): Dict of slices {slice nr (int) : slice (np.ndarray)}
                stack_dim (int): dimension along which to stack the slices

            Returns:
                np.ndarray: 3D array of the combined slices
            """
            assert stack_dim in [0, 1, 2]
            max_slice_nr = max([i for i in slices_dict.keys()])
            slice_list = [slices_dict[i] for i in range(max_slice_nr + 1)] 
            result = np.stack(slice_list, axis=stack_dim+1)
            logger.info(f'Combine {max_slice_nr} slices of shape {slices_dict[0].shape} along dimension {stack_dim}. --> resulting shape: {result.shape}')
            return result
            
        def volumes_from_loader(model, dataloader, stack_dim : int, output_location : str):
            """ Make a volume for each of the scans in the dataloader.
                The images in the dataloader are grouped by patient

                model : model object to generate the probabilities
                dataloader : loader for the images (based on a sequential sampler)
                stack dim: dimension along which to stack the images to form volumes
                ouput_location : path to the location where these volumes can be stored

                1 SCAN consists of multiple SLICES, but each SLICE is cropped again in multiple CROPS. 
            """


            slice_dict = dict()
            crops_dict = dict()
            hash = None
            orig_shape = None
            scan_id = None
            slice_id = None

            logger.debug('start making volumes')
            # Go through all the batches. The batches come in sequence, so the different slices of a single scan 
            # should come in sequence. Different crops of the same slice should come in sequence.
            for batch in tqdm(dataloader):
                
                batch = model.probabilities_on_batch(batch)
                logger.debug(f'batch keys : {[k for k in batch.keys()]}')
                logger.debug('meta batch keys : {}'.format(pformat(batch['meta'])))
                batch_scan_ids = [m['scan_id'] for m in batch['meta']]
                batch_slice_ids = [m['slice_id'] for m in batch['meta']]
                batch_crop_nrs = [m['crop_nr'] for m in batch['meta']]

                logger.debug(f'scan ids : {batch_scan_ids} with slice ids {batch_slice_ids} with crop numbers {batch_crop_nrs}.')
                if scan_id is None:
                    scan_id = batch_scan_ids[0]
                if slice_id is None:
                    slice_id = batch_slice_ids[0]
                
                logger.debug('shape of the probs : {}. This should be BCHW'.format(batch['probs'].shape))

                for i, pr in enumerate(batch['probs']):
                    
                    # New crop of same scan, same slice
                    if (batch_scan_ids[i] == scan_id) and (batch_slice_ids[i] == slice_id): 
                        logger.debug(f'Add crop {batch_crop_nrs[i]} of slice {slice_id} of scan {scan_id}')
                        crops_dict[batch_crop_nrs[i]] = pr
                        orig_shape = batch['meta'][i]['orig_shape']
                        hash = batch['meta'][i]['hash']
                    elif batch_scan_ids[i] == scan_id: # New slice of the same scan
                        logger.debug(f'*** Slice {slice_id} of scan {scan_id} is finished *** ')
                        # Add the previous slice to the dict --> we are starting the crops of a new slice after this
                        slice_dict[slice_id] = combine_crops(crops_dict, orig_shape)
                        # Start a new crops_dict
                        crops_dict = {batch_crop_nrs[i] : pr}
                        orig_shape = batch['meta'][i]['orig_shape']
                        # Update the current slice id
                        slice_id = batch_slice_ids[i]
                        logger.debug(f'New slice started -> slice : {slice_id} or scan {scan_id}')
                    else: # New scan
                        # add the previous slice to complete the previous scan
                        slice_dict[slice_id] = combine_crops(crops_dict, orig_shape)
                        logger.debug(f'Finish scan {scan_id} and save in file scan_{scan_id}')
                        volume = combine_slices(slice_dict, stack_dim)
                        # Start the saving processes independently
                        save_probs = Process(target=np.save, args=(os.path.join(output_location, f'scan_{scan_id}'), volume))
                        save_res = Process(target=np.save, args=(os.path.join(os.path.join(output_location, f'scan_{scan_id}_res'), np.argmax(volume, axis=0))))
                        save_probs.start()
                        save_res.start()
                        #np.save(os.path.join(output_location, f'scan_{scan_id}'), volume)
                        # Apart from the probabilities, the 'result' is just the argmax function on this array 
                        # --> channel with max probability is the inferred class
                        #np.save(os.path.join(output_location, f'scan_{scan_id}_res'), np.argmax(volume, axis=0))

                        # start a new scan and a new slice
                        slice_dict = dict()
                        crops_dict = {batch_crop_nrs[i] : pr}
                        orig_shape = batch['meta'][i]['orig_shape']
                        scan_id = batch_scan_ids[i]
                        slice_id = batch_slice_ids[i]
        
        
        for dim, model in self.models.items():
            logger.info(f'Start making volumes based on the model for dimension {dim}.')
            for split, loader in self.dataloaders[dim].items():
                logger.info(f'Make volume from the {split} loader')
                savedir = os.path.join(output_location, f"dimension_{dim}_split_{split}")
                Path(savedir).mkdir(parents=True, exist_ok=True)
                volumes_from_loader(model, loader, dim, savedir)

    def probabilities_vs_points(self, input_location_volumes : str, input_location_points):
        """Make a dataframe that contains the probabilities inferred by different models and compare it to the point labels in the annotations.

        Args:
            input_location_volumes (str): location where different volumes are stored in
            input_location_points (str): location where different volumes containing point labels are stored
        """

        def points_probabilities(probabilities : Dict[str, np.ndarray], points : np.ndarray) -> pd.DataFrame:
            """Extract the probabilities from a probability volume for all the annotated points in points.

            Args:
                probabilities (Dict): [n_classes, H, W, D] volume with probabilities for each of the n_classes classes in a dict: {prefix : np.ndarray}
                points (np.ndarray): [H, W, D] volume with point annotations

            Returns:
                pd.DataFrame: dataframe with columns [point_annotation, prob_0, prob_1, ... , prob_n_classes]
            """

            logger.debug(f'Compare probabilities volume with points volume (shape HWD : {points.shape}).')

            point_idx = np.argwhere(points != 255)
            # convert to lists of integer indices for each dimension
            points_h, points_w, points_d = point_idx[:,0].tolist(), point_idx[:,1].tolist(), point_idx[:,2].tolist()
            point_vals = pd.DataFrame((points[points_h, points_w, points_d]).T, columns = ['point_labels']) # This should give a 1D array with all the point value labels

            logger.debug(f'Point value array : {point_vals.shape}')

            point_probs = [point_vals]
            for prefix, probs in probabilities.items():
                logger.debug(f'Probabilities for {prefix} : shape CHWD {probs.shape}')
                point_probs.append(pd.DataFrame(probs[:, points_h, points_w, points_d], columns = [f'{prefix}_class_{i}' for i in range(probs.shape[0])]) )# This should give a 2D array [number of point labels, n_classes]
                logger.debug(f'new dataframe for probabilities : {prefix} \n {point_probs[-1].head()}')

            # The obtained result is a list of pandas dataframes that contain the 

            return pd.concat(point_probs, axis=1)


        self.train_df = points_probabilities(...)
        self.val_df = points_probabilities(...)
        self.test_df = points_probabilities(...)

    
    def train_model(self):
        """
        Train a classic machine learning model on the dataframe containing the train data.
        Validate it on the validation set and test on the test model.

        n_classes probabilities --> class label.
        
        This model will then be used to infer the class labels based on the predictions of three models (dimensional cut 0, 1 & 2).
        This might be better than just 1 model 
        """
        raise NotImplementedError




