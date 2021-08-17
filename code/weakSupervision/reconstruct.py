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
import argparse
import json


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


def reconstruct_3d(exp_dict, model_type_name, savedir, ground_truth, pseudo_dataset = None):

    logger.debug(f'model type name : {model_type_name}')
    logger.debug(f'save directory : {savedir}')

    reconstructor = rec.multi_dim_reconstructor(
        exp_dict, model_type_name=model_type_name)
    logger.info('START CONSTRUCTING THE VOLUMES')
    reconstructor.make_3D_volumes(savedir)
    logger.info('START COMBINING THE VOLUMES')
    reconstructor.reconstruct_from_volumes(savedir, ground_truth)

    with open(os.path.join(savedir, 'exp_dict_reconstruct.json'), 'w') as f:
        json.dump(exp_dict, f)

    if pseudo_dataset is not None:
        logger.info(f'Replace the train set masks for pseudo-masks in {pseudo_dataset}')
        reconstructor.split_pseudomask_volumes(os.path.join(savedir, 'volumes'), pseudo_dataset, dim = 2)


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

    args = parser.parse_args()

    exp_dict = RECONSTRUCT_DICTS[args.exp_dict]

    model_type_name = args.model_name
    save_dir = args.savedir
    ground_truth = args.ground_truth
    pseudo_dataset = args.pseudo_dataset

    reconstruct_3d(exp_dict, model_type_name, save_dir, ground_truth, pseudo_dataset)
