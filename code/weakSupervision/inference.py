from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu

import exp_configs

from src import models
import src.models.multi_dim_reconstructer as rec
from src import datasets
from src import utils as ut

import logging
import argparse

def setuplogger():
    """Setup the logger for this module
    """

    # Create the Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    handler = logging.StreamHandler()
    logger_formatter = logging.Formatter(
        '%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(logger_formatter)
    root_logger.addHandler(handler)

def reconstruct_3d(exp_dict, model_type_name, savedir):

    logger.debug(f'model type name : {model_type_name}')
    logger.debug(f'save directory : {savedir}')

    reconstructor = rec.multi_dim_reconstructor(exp_dict, 
                    model_type_name = model_type_name)
    reconstructor.make_3D_volumes(savedir)
    

if __name__ == "__main__":
    setuplogger()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-mn', '--model_name', required=True)
    parser.add_argument('-sd', '--savedir', default=None)

    args = parser.parse_args()

    exp_list = []
    for exp_group_name in args.exp_group_list:
        exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    exp_dict = {
            "batch_size": 6,
            "dataset": {
                "bg_points": 10,
                "blob_points": 3,
                "context_span": 1,
                "crop_size": [
                    352,
                    352
                ],
                "n_classes": 6,
                "name": "spine_dataset",
                "sources": [
                    "xVertSeg",
                    "USiegen",
                    "MyoSegmenTUM"
                ]
            },
            "dataset_size": {
                "test": "all",
                "train": "all",
                "val": "all"
            },
            "lr": 2.5e-05,
            "max_epoch": 10,
            "model": {
                "base": "fcn8_vgg16",
                "loss": [
                    "unsupervised_rotation_loss",
                    "rot_point_loss_multi_weighted",
                    "prior_extend",
                    "separation_loss"
                ],
                "n_channels": 3,
                "n_classes": 6,
                "name": "inst_seg",
                "prior_extend": 70,
                "prior_extend_slope": 10
            },
            "num_channels": 1,
            "optimizer": "adam"
        }

    exp_dict = {
        0: exp_dict,
        1: exp_dict,
        2: exp_dict
    }

    model_type_name = args.model_name
    save_dir = args.savedir


    reconstruct_3d(exp_dict, model_type_name, save_dir)