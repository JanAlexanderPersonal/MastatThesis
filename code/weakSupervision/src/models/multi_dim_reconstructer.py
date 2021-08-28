from haven import haven_utils as hu


import torch

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
import tikzplotlib
import cv2
from PIL import Image
from joblib import Parallel, delayed
import shutil

from src.models.metrics.seg_meter import SegMeter
from PIL import Image as Img

from pprint import pformat

from src import models
from src import datasets
from src import utils as ut
from pathlib import Path
import scipy.ndimage.morphology as morph

from multiprocessing import Process

from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import  cm

from typing import Dict
import sys
sys.path.append('/root/space/code/utils/')

import utils as slice_utils

from torch.utils.data import DataLoader
MULTI_PROCESS = False
N_JOBS = {'MyoSegmenTUM' : -1, 'USiegen' : -1, 'xVertSeg':1, 'PLoS' : -1}
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

    def __init__(self, model_dict : Dict, model_type_name : str, model_location : str = '/root/space/output',  dataset_location : str = None, contrast : int = 3, separate_source:str = None):
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
                model_folder = os.path.join(os.path.join(model_location), model_type_name.replace('D', str(i)), model_exp_dict['hash'])
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

        def get_dataloaders(model_dict : Dict, dataset_location : str = '/root/space/output', C : int = 3, separate_source = None) -> Dict:
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
                    logger.info(f'Separate source : {separate_source}')
                    ds = datasets.get_dataset(dataset_dict=model_dict["dataset"],
                                        split=split,
                                        datadir=os.path.join(dataset_location, f'dataset_{i}_contrast_{C}'),
                                        exp_dict=model_dict,
                                        dataset_size=model_dict['dataset_size'], 
                                        separate_source=separate_source) 

                    # make sure this dataset contains the full crop list:
                    # this operation will also sort the selected image list
                    ds.make_full_croplist()
                    _, img_df = ds.return_img_dfs()
                    img_df.to_csv(os.path.join(dataset_location, 'full_croplist.csv'))

                    sampler = torch.utils.data.SequentialSampler(ds)

                    loaders.update({split : DataLoader(ds,
                              sampler=sampler,
                              collate_fn=ut.collate_fn,
                              batch_size=model_dict["batch_size"] + 4,
                              num_workers = 3,
                              drop_last=False)})

                dataloaders[i] = loaders
            return dataloaders
        
        if dataset_location is None:
            dataset_location = model_location

        logger.debug(f'model dict keys : {list(model_dict.keys())}')
        self.models = get_models(model_dict, model_type_name.replace('C', str(contrast)),  model_location)
        self.dataloaders = get_dataloaders(model_dict, dataset_location, contrast, separate_source=separate_source)
        logger.info(f'Models and dataloaders loaded')

    def make_3D_volumes(self, output_location):
        """Construct the 3D volumes corresponding to the datasets indicated when constructing the dataloaders

        Since the datasets are sorted, we know the images will just come in the right order

        Args:
            output_location (str): Location to output the generated 3D volumes to
        """

        def combine_crops(crops_dict : Dict, orig_shape : Tuple, slice_id, scan_id, output_location) -> np.ndarray:
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
            if len([i for i in crops_dict.keys()]) > 1:
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

            plt.figure(figsize=(16,16))
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
                    result[:, -h:, :w, count] = im
                elif crop_nr == 3:
                    result[:, -h:, -w:, count] = im
                else:
                    raise ValueError
                count += 1
                if slice_id%25 == 0:
                    plt.subplot(3,2,crop_nr+1)
                    plt.imshow(cm.gist_stern_r(np.argmax(im, axis=0) * 51))
                    plt.xticks([])
                    plt.yticks([])
                    plt.title(f'crop {crop_nr}')

            result = np.argmax(np.nanmean(result, axis=3), axis = 0)
            
            if slice_id%25 == 0:
                plt.subplot(3,2,5)
                plt.imshow(cm.gist_stern_r(result * 51))
                plt.title(f'crop combination')
                #plt.suptitle(f'{scan_id}_{slice_id}_crops')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(os.path.join(output_location,'', f'{scan_id}_{slice_id}_crops.pdf'))
                plt.close('all')

                

            # Now, average over the last dimension of result to combine the crops, ignoring the nan values (where no value was returned)
            # Combine the 6 channels to one estimation for this slice

            return result

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
            result = np.stack(slice_list, axis=stack_dim)
            logger.info(f'Combine {max_slice_nr + 1} slices of shape {slices_dict[0].shape} along dimension {stack_dim}. --> resulting shape: {result.shape}')
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
                        # logger.debug(f'Add crop {batch_crop_nrs[i]} of slice {slice_id} of scan {scan_id}')
                        crops_dict[batch_crop_nrs[i]] = pr
                        orig_shape = batch['meta'][i]['orig_shape']
                        hash = batch['meta'][i]['hash']
                    elif batch_scan_ids[i] == scan_id: # New slice of the same scan
                        logger.debug(f'*** Slice {slice_id} of scan {scan_id} is finished *** ')
                        # Add the previous slice to the dict --> we are starting the crops of a new slice after this
                        slice_dict[slice_id] = combine_crops(crops_dict, orig_shape, slice_id, scan_id, output_location)
                        # Start a new crops_dict
                        crops_dict = {batch_crop_nrs[i] : pr}
                        orig_shape = batch['meta'][i]['orig_shape']
                        # Update the current slice id
                        slice_id = batch_slice_ids[i]
                        logger.debug(f'New slice started -> slice : {slice_id} or scan {scan_id}')
                    else: # New scan
                        # add the previous slice to complete the previous scan
                        slice_dict[slice_id] = combine_crops(crops_dict, orig_shape, slice_id, scan_id, output_location)
                        logger.debug(f'Finish scan {scan_id} and save in file scan_{scan_id}')
                        volume = combine_slices(slice_dict, stack_dim)
                        # Start the saving processes in independend process
                        np.save(os.path.join(output_location, f'scan_{scan_id}_res'), volume)

                        # start a new scan and a new slice
                        slice_dict = dict()
                        crops_dict = {batch_crop_nrs[i] : pr}
                        orig_shape = batch['meta'][i]['orig_shape']
                        scan_id = batch_scan_ids[i]
                        slice_id = batch_slice_ids[i]

            # Save the last scan
            slice_dict[slice_id] = combine_crops(crops_dict, orig_shape, slice_id, scan_id, output_location)
            logger.debug(f'Finish scan {scan_id} and save in file scan_{scan_id}')
            volume = combine_slices(slice_dict, stack_dim)
            # Start the saving processes in independend process
            np.save(os.path.join(output_location, f'scan_{scan_id}_res'), volume)
        
        
        for dim, model in self.models.items():
            logger.info(f'Start making volumes based on the model for dimension {dim}.')
            for split, loader in self.dataloaders[dim].items():
                logger.info(f'Make volume from the {split} loader')
                savedir = os.path.join(output_location, f"dimension_{dim}_split_{split}")
                Path(savedir).mkdir(parents=True, exist_ok=True)
                volumes_from_loader(model, loader, dim, savedir)

    def reconstruct_from_volumes(self, volumes_location, ground_truth_location):
        def plot_volumes(volumes : Dict, title : str, savename = '3d_reconstruct.pdf', ground_truth = None, combined_volume = None, volume_scan = None):

            fig_h = 12
            rows = 3
            if ground_truth is not None:
                fig_h += 4
                rows += 1
            if combined_volume is not None:
                fig_h += 4
                rows += 1
            if volume_scan is not None:
                fig_h += 4
                rows += 1

            
            logger.debug(f'figure height {fig_h}cm and rows : {rows}')

            plt.figure(figsize=(12,fig_h))
            
            for i in range(3):
                    for j in range(3):
                        plt.subplot(rows,3,i*3+j+1)
                        mask = np.take(volumes[i], volumes[i].shape[j]//2, axis=j)
                        image = Img.fromarray((np.take(volume_scan, volume_scan.shape[j]//2, axis=j)*255).astype('uint8')).convert('RGB')
                        mask_im = Image.fromarray(np.uint8(cm.gist_stern_r(mask*51)*255)).convert('RGB')
                        mask_im = cv2.addWeighted(np.array(image), 0.3,np.array(mask_im), 0.7, 0)
                        plt.imshow(mask_im)
                        plt.title(f'model {i}\nslice along axis {j}')
                        plt.yticks([])
                        plt.xticks([])
            
            if combined_volume is not None:
                i += 1
                logger.debug(f'Starting image row {i}')
                for j in range(3):
                    plt.subplot(rows,3,i*3+j+1)
                    mask = np.take(combined_volume, combined_volume.shape[j]//2, axis=j)
                    image = Img.fromarray((np.take(volume_scan, volume_scan.shape[j]//2, axis=j)*255).astype('uint8')).convert('RGB')
                    mask_im = Image.fromarray(np.uint8(cm.gist_stern_r(mask*51)*255)).convert('RGB')
                    mask_im = cv2.addWeighted(np.array(image), 0.3,np.array(mask_im), 0.7, 0)
                    plt.imshow(mask_im)
                    plt.title(f'Combined extimated volumes\nslice along axis {j}')
                    plt.yticks([])
                    plt.xticks([])
            
            
            if ground_truth is not None:
                i += 1
                logger.debug(f'Starting image row {i}')
                for j in range(3):
                    mask = np.take(ground_truth, ground_truth.shape[j]//2, axis=j)
                    image = Img.fromarray((np.take(volume_scan, volume_scan.shape[j]//2, axis=j)*255).astype('uint8')).convert('RGB')
                    mask_im = Image.fromarray(np.uint8(cm.gist_stern_r(mask*51)*255)).convert('RGB')
                    mask_im = cv2.addWeighted(np.array(image), 0.3,np.array(mask_im), 0.7, 0)
                    plt.subplot(rows,3,i*3+j+1)
                    plt.imshow(mask_im)
                    plt.title(f'Ground truth\nslice along axis {j}')
                    plt.yticks([])
                    plt.xticks([])

            if volume_scan is not None:
                i += 1
                logger.debug(f'Starting image row {i}')
                for j in range(3):
                    image = Img.fromarray((np.take(volume_scan, volume_scan.shape[j]//2, axis=j)*255).astype('uint8')).convert('RGB')
                    plt.subplot(rows,3,i*3+j+1)
                    plt.imshow(image)
                    plt.title(f'Scan image\nslice along axis {j}')
                    plt.yticks([])
                    plt.xticks([])
                        
            #plt.suptitle(title)        
            plt.tight_layout()
            plt.savefig(savename, bbox_inches='tight')
            plt.close('all')

        def clean_mask(volume : np.ndarray, iterations_denoise: int = 1, iterations_erode:int = 1, full = True):
            """ Function to clean up a mask
            """
            def remove_noise(volume, iterations_denoise, iterations_erode, full = True):
                struct = morph.generate_binary_structure(3, 3)
                if full:
                    return morph.binary_erosion(
                        morph.binary_closing(
                            morph.binary_opening(volume, structure=struct, iterations = iterations_denoise)
                            , structure=struct, iterations = iterations_denoise
                        ), structure=struct, iterations = iterations_erode
                    )
                else:
                    return morph.binary_closing( volume, structure=struct, iterations = iterations_denoise)
            
            temp = np.zeros((*volume.shape, 6), dtype = int)
            for i in range(6):
                temp[:,:,:, i] = remove_noise((volume==i), iterations_denoise, iterations_erode, full=full)

            volume = np.argmax(temp, axis = 3)
            return volume
        
        def combine_volumes(volumes : Dict[int, np.ndarray]) -> np.ndarray:
            combined_volume = np.zeros_like(volumes[0])
            struct = morph.generate_binary_structure(3, 3)
            volumes = {key : clean_mask(volume, full = False) for key, volume in volumes.items()}
            counts = [np.unique(volumes[key], return_counts = True) for key in range(3)]
            logger.debug(f'Counts original : {counts}')
            for i in range(3):
                c = counts[i]
                if c[1].size > 1:
                    c = c[1]
                    counts[i] = np.sum(c[1:])
                else:
                    counts[i] = 0 
            counts = np.array(counts)
            relative = counts / np.max(counts)
            ignore = None
            if any(relative < .35) and False : # This is not used anymore
                ignore = np.argmin(counts)
                logger.warning(f'Result {ignore} will be ignored because too small.')
            logger.debug(f'Counts : {counts} ** relative : {relative} ** ignore {ignore}')
            for i in range(1,6):
                if ignore is None:
                    m = (volumes[0] == 1) & (volumes[1] == i) & (volumes[2] == i)
                elif ignore == 0:
                    m = (volumes[1] == i) & (volumes[2] == i)
                elif ignore == 1:
                    m = (volumes[0] == 1) & (volumes[2] == i)
                elif ignore == 2:
                    m = (volumes[0] == 1) & (volumes[1] == i)
                # logger.debug(f'set {np.sum(m)} elemets to {i}')
                combined_volume[m] = i
            return combined_volume  

        def get_combined_volume(volumes, iterations_denoise, iterations_erode, ground_truth = None, volume_scan=None):
            combined_volume=combine_volumes(volumes)
            combined_volume = clean_mask(combined_volume, iterations_denoise=iterations_denoise, iterations_erode=iterations_erode)
            plot_volumes(volumes, f'{source} image {nr}', savename=os.path.join(imagedir, f'morphmask_denoise{iterations_denoise}_erode{iterations_erode}_{source}_{nr}.pdf'), ground_truth=ground_truth, volume_scan=volume_scan, combined_volume=combined_volume)
            return combined_volume


        splits = ['val', 'train']
        dims = [0,1,2]
        foldername = 'dimension_D_split_S'

        seg_meters = {
            iterations_denoise : {
                iterations_erode : SegMeter('val') for iterations_erode in range(4)
            } for iterations_denoise in range(4) 
        }

        savedir = os.path.join(volumes_location, 'volumes')
        imagedir = os.path.join(volumes_location, 'images')

        Path(savedir).mkdir(parents=True, exist_ok=True)
        Path(imagedir).mkdir(parents=True, exist_ok=True)

        F_optimal_iterations = False

        for split in splits:
            foldernames = [foldername.replace('S', split).replace('D', str(d)) for d in dims]
            logger.info(f'Start reconstruction of volumes for split {split}.')

            if split == 'train' and not F_optimal_iterations:
                for it_DN, it_ER in itertools.product(list(range(4)), list(range(4))):
                    seg_meters[it_DN][it_ER] = seg_meters[it_DN][it_ER].get_avg_score()
                seg_meters = pd.DataFrame.from_dict({
                    (it_DN, it_ER) : [seg_meters[it_DN][it_ER]['val_score'] , seg_meters[it_DN][it_ER]['val_prec'], seg_meters[it_DN][it_ER]['val_recall']] 
                    for it_DN in seg_meters.keys() 
                    for it_ER in seg_meters[it_DN].keys()
                }, orient='index', columns=['weighted_dice_score', 'precision', 'recall'])
                logger.info(f'Result dependent on iterations : {seg_meters}')
                seg_meters.to_csv(os.path.join(volumes_location, 'validationSet_morphologicalIterations.csv'))
                max_idx = seg_meters['weighted_dice_score'].idxmax()
                logger.debug(f'The maximal weighted dice score was found for index {max_idx}')
                BEST_IT_DN, BEST_IT_ER = max_idx
                logger.info(f'The optimal iterations for denoising is {BEST_IT_DN} and for erosion is {BEST_IT_ER}')
                F_optimal_iterations = True
                for image_name in os.listdir(imagedir):
                    if f'denoise{BEST_IT_DN}_erode{BEST_IT_ER}' in image_name:
                        continue
                    logger.info(f'Remove path : {os.path.join(imagedir, image_name)}')
                    try:
                        os.remove(os.path.join(imagedir, image_name))
                    except :
                        shutil.rmtree(os.path.join(imagedir, image_name), ignore_errors=True)
                        

                    

            for file_name in tqdm(os.listdir(os.path.join(volumes_location, foldernames[0]))):
                if not file_name.endswith('_res.npy'):
                    continue
                logger.info(f'Start extracting file {file_name}')
                volumes = {i : np.load(os.path.join(volumes_location, fn, file_name)) for i, fn in enumerate(foldernames)}
                _, source, nr, _ = file_name.split('_')
                
                mask_filename = os.path.join(ground_truth_location, f'{source}_masks', f'image{nr}', 'mask_array.npy')
                ground_truth = np.load(mask_filename)
                volume_filename = os.path.join(ground_truth_location, f'{source}_images', f'image{nr}', 'image_array.npy')
                volume_scan = np.load(volume_filename)
                logger.debug(f'Volume scan datatype {volume_scan.dtype}')
                # plot_volumes(volumes, f'{source} image {nr}', savename=os.path.join(imagedir, f'rawmask_{split}_{source}_{nr}.pdf'), ground_truth=ground_truth, volume_scan=volume_scan)
                

                logger.debug(f'Volume {source}, nr {nr}')

                n = N_JOBS.get(source, 2)

                if split == 'val':
                    for iterations_denoise, erode_dict in seg_meters.items():
                        logger.debug(f'start evaluation the segmentation meters for iterations denoise {iterations_denoise}')
                        logger.debug(f'Source {source}: calculate with {n} jobs')
                        combined_volumes = Parallel(n_jobs=n)(delayed(get_combined_volume)(volumes, iterations_denoise, iterations_erode, ground_truth = ground_truth, volume_scan=volume_scan) for iterations_erode in erode_dict.keys())
                        for iterations_erode in erode_dict.keys():
                            logger.debug(f'calculation for erode {iterations_erode}')
                            seg_meters[iterations_denoise][iterations_erode].val_on_volume(ground_truth, combined_volumes[iterations_erode], 6) 
                if split == 'train':
                    combined_volume = combine_volumes(volumes)
                    combined_volume = clean_mask(combined_volume, iterations_denoise=BEST_IT_DN, iterations_erode=BEST_IT_ER)
                    
                    plot_volumes(volumes, f'{source} image {nr}', savename=os.path.join(imagedir, f'morphmask_train_denoise{BEST_IT_DN}_erode{BEST_IT_ER}_{source}_{nr}.pdf'), ground_truth = ground_truth, combined_volume=combined_volume, volume_scan = volume_scan)
                    np.save(os.path.join(savedir, f'morphcombined_train_{source}_{nr}') ,combined_volume)

    def split_pseudomask_volumes(self, volumes_source : str, volumes_target : str, dim : int = 2):

        # in the source directory, we should find volumes according to naming convention:
        #       morphcombined_train_xVertSeg_001.npy
        #       morphcombined_train_xVertSeg_002.npy

        README_TEXT = ['Pay attention, the following mask files have been replaced by PSEUDO mask files : ']

        logging.info(f'Fetch pseudo masks from {volumes_source} and replace the train slices in {volumes_target}')

        for filename in os.listdir(os.path.abspath(volumes_source)):
            if not filename.endswith('.npy'):
                continue
            _, split, dataset_source, number = filename.strip('.npy').split('_')

            logger.info(f'Dataset source: {dataset_source} * Split : {split} and number {number}')

            if not split == 'train':
                continue

            arr = np.load(os.path.join(volumes_source, filename))
            # Find the folder with masks for this specific volume

            target_path = os.path.join(volumes_target,  f'{dataset_source}_masks', f'image{number}')
            logging.info(f'remove files in {target_path}')
            shutil.rmtree(target_path, ignore_errors=True)
            Path(target_path).mkdir(parents=True, exist_ok=True)
            README_TEXT.append(f'\t{dataset_source}\t{number}')
            np.save(os.path.join(target_path, 'pseudomask_array'), arr)
            slice_utils.mask_to_slices_save(arr, dim, target_path)

        # Write the read me file
        with open(os.path.join(volumes_target, 'readme.txt'), 'w+') as f:
            f.write(
                '\n'.join(README_TEXT)
            )