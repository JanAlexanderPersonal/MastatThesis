from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
from numpy.lib.twodim_base import mask_indices
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

from pprint import pformat

from src import models
from src import datasets
from src import utils as ut
from pathlib import Path

import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import random

from typing import List, Dict


import json


import logging

cudnn.benchmark = True

F_stop_at_epoch = False

LEARNING_RATE_STEPS = [1, 5000, 20000]
MAX_WEIGHT_STEPS = [4,4,8]


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


def trainval_source(exp_dict: Dict, savedir_base: str, datadir: str,
             reset: bool = False, num_workers: int = 0, tensorboard_folder: str = None, learning_rate_factor : int = 1, me = 150, source : str = "MyoSegmenTUM"):
    """trainval: training and validation routine to perform all the experiments defined in exp_dict

    Args:
        exp_dict (Dict): Dictionnary defining the experiment to run.
                {
                    'model' : (Dict) Definition of the model    {
                                                                    'n_classes' : n_classes,
                                                                    'base' : network to base the model upon,
                                                                    'optimizer' : model optimizer
                                                                },
                    'dataset' : (str) defines dataset to use
                    'max_epoch' : (int) stop training after this many epochs
                }
        savedir_base (str): path to the savedir location
        datadir (str): path to the data directory
        reset (bool, optional): When this is True, you discart all existing saved weights and just start over new. Defaults to False.
        num_workers (int, optional): Number of workers to use. Defaults to 0.
        tensorboard_folder (str, optional): path to store the tensorboard information (log_dir). Defaults to None.
    """
    logger.debug(f'start trainval with experiment dict {pformat(exp_dict)}')

    # bookkeeping stuff
    # ==================
    exp_id = hu.hash_dict(exp_dict)

    exp_dict['lr'] = exp_dict['lr'] / learning_rate_factor
    exp_dict['max_epoch'] = me
    logger.debug('experiment has : {exp_id}')
    orig_dir = os.path.join(savedir_base, exp_id) # directory with the original model, trained on all data sources
    savedir = os.path.join(savedir_base, f'{exp_id}_{source}')

    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    logger.info("Experiment dictionary saved in %s" % savedir)

    # set seed
    # ==================
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Dataset
    # ==================
    # train set

    logger.info('define train set')
    train_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                     split="train",
                                     datadir=datadir,
                                     exp_dict=exp_dict,
                                     dataset_size=exp_dict['dataset_size'], 
                                     separate_source=source, precalculated_points = True)

    mask_counts = train_set.count_values_masks() # dict with counts per label {0 : ... ,  1 : ... , ...}

    # Weights: The objective is to get weights proportional to the prevalence of labels in the dataset and with minimal weight == 1

    mask_weights = [min(list(mask_counts.values())) * (mask_counts[i] ** (-1)) for i in range(exp_dict['dataset']['n_classes']) ]

    logger.info(f'counts for mask labels : {mask_counts}. This results in the mask weights {mask_weights}.')

    # val set
    logger.info('define validation set')
    val_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                   split="val",
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   dataset_size=exp_dict['dataset_size'], 
                                   separate_source=source, precalculated_points = True)

    # test set
    logger.info('define test set')
    test_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                    split="test",
                                    datadir=datadir,
                                    exp_dict=exp_dict,
                                    dataset_size=exp_dict['dataset_size'], 
                                    separate_source=source, precalculated_points = True)

    logger.info('make dataloaders from the defined validation and test set')
    val_loader = DataLoader(val_set,
                            # sampler=val_sampler,
                            batch_size=exp_dict["batch_size"] + 4,
                            collate_fn=ut.collate_fn,
                            num_workers=4)
    test_loader = DataLoader(test_set,
                             # sampler=val_sampler,
                             batch_size=exp_dict["batch_size"] + 4,
                             collate_fn=ut.collate_fn,
                             num_workers=4)

    # Model
    # ==================
    logger.info('get model')
    model = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                             weight_vector = mask_weights,
                             train_set=train_set).cuda()
    if tensorboard_folder is not None:
        model.add_writer(tensorboard_folder)

    model_path = os.path.join(orig_dir, "model.pth")
    score_list_path = os.path.join(orig_dir, "score_list.pkl")


    # If there is a pkl file containing stored model weights from the last time the model was trained, get it.
    # if 'reset == True' this file will be deleted when you reach this code
    # line.
    if os.path.exists(score_list_path):
        # resume experiment
        model.load_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
        e = s_epoch
    else:
        # restart experiment
        print('Model not found')
        return

    # Train & Val
    # ==================
    logger.info("Starting experiment at epoch %d" % (s_epoch))
    model.waiting = 0
    # Validate the model
    logger.info('Start validation on de cross validation set')
    val_dict, val_metrics_df = model.val_on_loader(val_loader)
    model.val_score_best = val_dict["val_score"]

    # Random sampler
    train_sampler = torch.utils.data.RandomSampler(
        train_set, replacement=True,
        num_samples=2 * len(test_set))

    logger.info('Get train loader for train dataset')
    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              collate_fn=ut.collate_fn,
                              batch_size=exp_dict["batch_size"],
                              drop_last=True,
                              num_workers=4)

    for name, data_set in zip(['train', 'val', 'test'], [
                              train_set, val_set, test_set]):
        full, selected = data_set.return_img_dfs()
        full.to_csv(os.path.join(savedir, f'{name}_full.csv'))
        selected.to_csv(os.path.join(savedir, f'{name}_selected.csv'))

    # check for overlap:

    _, train_selected = train_set.return_img_dfs()
    _, val_selected = val_set.return_img_dfs()
    _, test_selected = test_set.return_img_dfs()

    assert(pd.merge(train_selected, val_selected, how ='inner', on =['img', 'tgt']).shape[0] == 0), 'Overlap detected'
    assert(pd.merge(train_selected, test_selected, how ='inner', on =['img', 'tgt']).shape[0] == 0), 'Overlap detected'
    assert(pd.merge(test_selected, val_selected, how ='inner', on =['img', 'tgt']).shape[0] == 0), 'Overlap detected'

    score_dict = {}
    logger.info('Overlap detection success!')

    min_loss = np.inf

    # Run the remaining epochs starting from the last epoch for which values
    # were available in the pkl
    for learning_rate_factor, max_wait in zip(LEARNING_RATE_STEPS, MAX_WEIGHT_STEPS):
        model.waiting = 0
        exp_dict['lr'] = exp_dict['lr'] / learning_rate_factor
        model.update_optimizer(exp_dict)
        loss_decrease_counter = 0

        for e in range(s_epoch, exp_dict['max_epoch']):
            # Validate only at the start of each cycle
            logger.info(f'Start epoch {e}')
            score_dict = {}

            # Train the model
            logger.info('Start training')
            train_set.shuffle_img_df()
            train_dict = model.train_on_loader(train_loader)
        
            logger.info('Start validation on de train set')
            train_val_dict, train_metrics_df = model.val_on_loader(train_loader)
            score_dict['train_score'] = train_val_dict['train_score']
            score_dict["train_weighted_dice"] = train_val_dict["train_weighted_dice"]
            score_dict["train_dice"] = train_val_dict["train_dice"]
            train_metrics_df.to_csv(os.path.join(savedir, 'train_metrics_df.csv'))

            # Get new score_dict
            score_dict.update(train_dict)
            score_dict["epoch"] = e
            score_dict["waiting"] = model.waiting

            if min_loss > train_dict['train_loss']:
                model.waiting += 1
                min_loss = train_dict['train_loss']
            else:
                loss_decrease_counter += 1

            # Add to score_list and save checkpoint
            score_list += [score_dict]
            
            # Validate the model
            logger.info('Start validation on de cross validation set')
            val_dict, val_metrics_df = model.val_on_loader(val_loader,savedir_images=os.path.join(savedir, "images"), n_images=3)
            val_metrics_df.to_csv(os.path.join(savedir, 'val_metrics_df.csv'))
            score_dict["val_score"] = val_dict["val_score"]
            score_dict["val_weighted_dice"] = val_dict["val_weighted_dice"]
            score_dict["val_dice"] = val_dict["val_dice"]

            if score_dict["val_score"] >= model.val_score_best:
                logger.info('SCORE IMPROVED')
                hu.torch_save(os.path.join(savedir, "model.pth"), model.get_state_dict())
                logger.info("Checkpoint Saved: %s" % savedir)
                if (e == exp_dict['max_epoch'] - 1) or (model.waiting >= 5) or (e % 10 == 0):
                    test_dict, test_metrics_df = model.val_on_loader(test_loader,
                                                                    savedir_images=os.path.join(
                                                                        savedir, "images"),
                                                                    n_images=3)
                    score_dict.update(test_dict)
                    test_metrics_df.to_csv(os.path.join(savedir, 'test_metrics_df.csv'))
                hu.save_pkl(
                    os.path.join(
                        savedir,
                        "score_list_best.pkl"),
                    score_list)
                # score_df.to_csv(os.path.join(savedir, "score_best_df.csv"))
                hu.torch_save(os.path.join(savedir, "model_best.pth"),
                            model.get_state_dict())
                model.waiting = 0
                model.val_score_best = score_dict["val_score"]
                logger.info("Saved Best: %s" % savedir)

            # Report & Save
            score_df = pd.DataFrame(score_list)
            score_df.to_csv(os.path.join(savedir, "score_df.csv"))
            
            logger.info(f"\n{score_df.tail(10)}\n")
            
            hu.save_pkl(score_list_path, score_list)

            if model.waiting >= max_wait or loss_decrease_counter >= max_wait:
                # Reduce learning rate or stop
                break


    # Revert the model to the last stored value and perform the final validation

    model.load_state_dict(hu.torch_load(model_path))
    logger.info('Start final validation on de test set')
    test_dict, test_metrics_df = model.val_on_loader(test_loader,
                                                                savedir_images=os.path.join(
                                                                    savedir, "images"),
                                                                n_images=10)
    score_dict.update(test_dict)
    test_metrics_df.to_csv(os.path.join(savedir, 'test_metrics_df.csv'))

    
    logger.info('Start final validation on de train set')
    train_val_dict, train_metrics_df = model.val_on_loader(train_loader)
    score_dict['train_score'] = train_val_dict['train_score']
    score_dict["train_weighted_dice"] = train_val_dict["train_weighted_dice"]
    train_metrics_df.to_csv(os.path.join(savedir, 'train_metrics_df.csv'))

    logger.info('Start final validation on de cross validation set')
    val_dict, val_metrics_df = model.val_on_loader(val_loader)
    val_metrics_df.to_csv(os.path.join(savedir, 'val_metrics_df.csv'))
    score_dict["val_score"] = val_dict["val_score"]
    score_dict["val_weighted_dice"] = val_dict["val_weighted_dice"]

    with open(os.path.join(savedir, 'score_dict_final.json'), 'w') as f:
        json.dump(score_dict, f)


    if e is None:
        e = 'final'

    print('Experiment completed et epoch %d' % e)

import copy

if __name__ == "__main__":
    setuplogger()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', default=None)
    parser.add_argument("-r", "--reset", default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=4)
    parser.add_argument("-tb", "--tensorboard", default=None)

    args = parser.parse_args()

    # Collect experiments
    # ===================
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))

        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # Make sure the paths for the tensorboard feedback and the model
    # improvement steps are present
    Path(args.tensorboard).mkdir(parents=True, exist_ok=True)
    Path(args.savedir_base).mkdir(parents=True, exist_ok=True)

    # Perform the trainval procedure on each of the experiments in the exp list
    for exp_dict in exp_list:
        for source in exp_dict['dataset']['sources']:
            # do trainval
            logger.info(f'start for source : {source}')
            exp_dict_copy = copy.deepcopy(exp_dict)
            trainval_source(exp_dict=exp_dict_copy,
                    savedir_base=args.savedir_base,
                    datadir=args.datadir,
                    reset=args.reset,
                    num_workers=args.num_workers,
                    tensorboard_folder=args.tensorboard, source = source)
