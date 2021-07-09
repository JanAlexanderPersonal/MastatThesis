from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu

import exp_configs

from src import models
from src import datasets
from src import utils as ut

import logging
import argparse

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

def 3d_reconstruct(exp_dict, model_type_name, savedir)

    reconstructor = models.multi_dim_reconstructor(exp_dict, 
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
        0: exp_list[0],
        1: exp_list[0],
        2: exp_list[0]
    }

    model_type_name = args.model_name
    save_dir = args.savedir


    3d_reconstruct(exp_dict, model_type_name, save_dir)