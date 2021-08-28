from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
from exp_configs.reconstruct_dicts import RECONSTRUCT_DICTS

import exp_configs

from src import models
import src.models.multi_dim_reconstructer as rec
from src import datasets
from src import utils as ut
import os.path

import logging
from torch.utils.data import DataLoader
import argparse
import json
import torch
import time


def setuplogger():
    """Setup the logger for this module
    """

    # Create the Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    logger_formatter = logging.Formatter(
        '%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(logger_formatter)
    root_logger.addHandler(handler)


def reconstruct_3d(exp_dict, model_type_name, savedir, ground_truth, separate_source = None, pseudo_dataset = None):

    logger.debug(f'model type name : {model_type_name}')
    logger.debug(f'save directory : {savedir}')

    start = time.time()
    reconstructor = rec.multi_dim_reconstructor(
        exp_dict, model_type_name=model_type_name, separate_source=separate_source)

    loaders = dict()
    for split in ['train', 'val', 'test']:
        ds = datasets.get_dataset(dataset_dict={
                                            "bg_points": -1,
                                            "blob_points": 1,
                                            "context_span": 0,
                                            "crop_size": [
                                                352,
                                                352
                                            ],
                                            "n_classes": 6,
                                            "name": "spine_dataset",
                                            "sources": [
                                                "UW"
                                            ]
                                        },
                                        split='train',
                                        datadir= r'/root/space/output/uWash',
                                        exp_dict={
                                            "batch_size": 6,
                                            "dataset": {
                                                "bg_points": -1,
                                                "blob_points": 1,
                                                "context_span": 0,
                                                "crop_size": [
                                                    352,
                                                    352
                                                ],
                                                "n_classes": 6,
                                                "name": "spine_dataset",
                                                "sources": [
                                                    "UW"
                                                ]
                                            },
                                            "dataset_size": {
                                                "test": "all",
                                                "train": "all",
                                                "val": "all"
                                            },
                                            "lr": 0.0001,
                                            "max_epoch": 50,
                                            "model": {
                                                "base": "fcn8_vgg16",
                                                "loss": "cross_entropy",
                                                "n_channels": 3,
                                                "n_classes": 6,
                                                "name": "inst_seg",
                                                "prior_extend": 70,
                                                "prior_extend_slope": 10
                                            },
                                            "num_channels": 1,
                                            "optimizer": "adam",
                                            "hash" : "edc036f241e350ed66b07d6edebaaef2"
                                        },
                                        dataset_size={
                                                    "test": "all",
                                                    "train": "all",
                                                    "val": "all"
                                                }) 

                    # make sure this dataset contains the full crop list:
                    # this operation will also sort the selected image list
        ds.make_full_croplist()
        _, img_df = ds.return_img_dfs()
        img_df.to_csv(os.path.join(r'/root/space/output/uWash', 'full_croplist.csv'))

        sampler = torch.utils.data.SequentialSampler(ds)

        loaders.update({split : DataLoader(ds,
                    sampler=sampler,
                    collate_fn=ut.collate_fn,
                    batch_size=4,
                    num_workers = 3,
                    drop_last=False)})

    reconstructor.dataloaders = {2:loaders}
    print(f'Time to load class {time.time() - start}')
    start = time.time()
    logger.info('START CONSTRUCTING THE VOLUMES')
    reconstructor.make_3D_volumes(savedir)
    print(f'Time to load class {time.time() - start}')
    logger.info('START COMBINING THE VOLUMES')
    reconstructor.reconstruct_from_volumes(savedir, ground_truth)

    #with open(os.path.join(savedir, 'exp_dict_reconstruct.json'), 'w') as f:
        #json.dump(exp_dict, f)

    #if pseudo_dataset is not None:
        #logger.info(f'Replace the train set masks for pseudo-masks in {pseudo_dataset}')
        #reconstructor.split_pseudomask_volumes(os.path.join(savedir, 'volumes'), pseudo_dataset, dim = 2)


if __name__ == "__main__":
    setuplogger()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('-mn', '--model_name', required=True)
    parser.add_argument('-ed', '--exp_dict')
    parser.add_argument('-sd', '--savedir', default=None)
    parser.add_argument('-gt', '--ground_truth',
                        default='dataset_1_contrast_3')
    parser.add_argument('-pd', '--pseudo_dataset', default = None)
    parser.add_argument('-ss', '--separate_source', default=None)

    args = parser.parse_args()

    exp_dict = RECONSTRUCT_DICTS[args.exp_dict]

    model_type_name = args.model_name
    save_dir = args.savedir
    ground_truth = args.ground_truth
    pseudo_dataset = args.pseudo_dataset
    separate_source = args.separate_source

    reconstruct_3d(exp_dict, model_type_name, save_dir, ground_truth, separate_source=separate_source, pseudo_dataset = pseudo_dataset)
