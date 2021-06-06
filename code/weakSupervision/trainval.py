from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
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

from typing import List, Dict

import logging

cudnn.benchmark = True

F_stop_at_epoch = False

def setupLogging():
    """Setup the logger for this module
    """
    
    # Create the Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(logger_formatter)
    root_logger.addHandler(handler)

def trainval(exp_dict : Dict, savedir_base : str, datadir : str, reset : bool = False, num_workers : int =0, tensorboard_folder : str =None):
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
    # todo: solve problems with GPU memory size
    logger.info(f'start trainval with experiment dict {pformat(exp_dict)}')

    # bookkeepting stuff
    # ==================
    exp_id = hu.hash_dict(exp_dict)
    logger.debug('experiment has : {exp_id}')
    savedir = os.path.join(savedir_base, exp_id)
    if reset:
        hc.delete_and_backup_experiment(savedir)
        for file in os.listdir(tensorboard_folder):
            os.remove(os.path.join(tensorboard_folder, file))

    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    logging.info("Experiment saved in %s" % savedir)

    # set seed
    # ==================
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Dataset
    # ==================
    # train set

    logging.info('define train set')
    train_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                     split="train",
                                     datadir=datadir,
                                     exp_dict=exp_dict,
                                     dataset_size=exp_dict['dataset_size'])
    # val set
    logging.info('define validation set')
    val_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                   split="val",
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   dataset_size=exp_dict['dataset_size'])

    # test set
    logging.info('define test set')
    test_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                   split="test",
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   dataset_size=exp_dict['dataset_size'])

    logging.info('make dataloaders from the defined validation and test set')
    val_loader = DataLoader(val_set,
                            # sampler=val_sampler,
                            batch_size=1,
                            collate_fn=ut.collate_fn,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set,
                            # sampler=val_sampler,
                            batch_size=1,
                            collate_fn=ut.collate_fn,
                            num_workers=num_workers)

    # Model
    # ==================
    logging.info('get model')
    model = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                             train_set=train_set).cuda()
    if tensorboard_folder is not None:
        model.add_writer(tensorboard_folder)

    # model.opt = optimizers.get_optim(exp_dict['opt'], model)
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    #logging.debug('model definition')
    #logging.debug(model)

    # If there is a pkl file containing stored model weights from the last time the model was trained, get it.
    # if 'reset == True' this file will be deleted when you reach this code line.
    if os.path.exists(score_list_path):
        # resume experiment
        model.load_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
        e = s_epoch
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ==================
    logger.info("Starting experiment at epoch %d" % (s_epoch))
    model.waiting = 0
    model.val_score_best = -np.inf
    
    # Random sampler 
    train_sampler = torch.utils.data.RandomSampler(
                                train_set, replacement=True, 
                                num_samples=2*len(test_set))

    logging.info('Get train loader for train dataset')
    train_loader = DataLoader(train_set,
                            sampler=train_sampler,
                            collate_fn=ut.collate_fn,
                            batch_size=exp_dict["batch_size"], 
                            drop_last=True, 
                            num_workers=num_workers)
    
    # Run the remaining epochs starting from the last epoch for which values were available in the pkl
    for e in range(s_epoch, exp_dict['max_epoch']):
        # Validate only at the start of each cycle
        logger.info(f'Start epoch {e}')
        score_dict = {}

        logger.info('Start validation on test set')
        test_dict, test_metrics_df = model.val_on_loader(test_loader,
                                savedir_images=os.path.join(savedir, "images"),
                                n_images=10) 
        # Train the model
        logger.info('Start training')
        train_dict = model.train_on_loader(train_loader)

        if F_stop_at_epoch:
            print(f'Epoch {e} -> train on loader is finished.')
            input('Press enter to continue')

        # Validate the model
        logging.info('Start validation on de cross validation set')
        val_dict, val_metrics_df = model.val_on_loader(val_loader)
        score_dict["val_score"] = val_dict["val_score"]

        # Get new score_dict
        score_dict.update(train_dict)
        score_dict["epoch"] = e
        score_dict["waiting"] = model.waiting

        model.waiting += 1

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Save Best Checkpoint
        score_df = pd.DataFrame(score_list)
        if score_dict["val_score"] >= model.val_score_best:
            test_dict, test_metrics_df = model.val_on_loader(test_loader,
                                savedir_images=os.path.join(savedir, "images"),
                                n_images=10)  
            score_dict.update(test_dict)
            hu.save_pkl(os.path.join(savedir, "score_list_best.pkl"), score_list)
            # score_df.to_csv(os.path.join(savedir, "score_best_df.csv"))
            hu.torch_save(os.path.join(savedir, "model_best.pth"),
                        model.get_state_dict())
            model.waiting = 0
            model.val_score_best = score_dict["val_score"]
            logger.info("Saved Best: %s" % savedir)

        # Report & Save
        score_df = pd.DataFrame(score_list)
        score_df.to_csv(os.path.join(savedir, "score_df.csv"))
        test_metrics_df.to_csv(os.path.join(savedir, 'test_metrics_df.csv'))
        logger.info(f"\n{score_df.tail(10)}\n")
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        logger.info("Checkpoint Saved: %s" % savedir)

        if model.waiting > 100:
            break

        if F_stop_at_epoch:
            print(f'Epoch {e} is finished.')
            input('Press enter to continue')
    


    print('Experiment completed et epoch %d' % e)


if __name__ == "__main__":
    setupLogging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', default=None)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    parser.add_argument("-tb", "--tensorboard", default=None)

    args = parser.parse_args()

    logger.info(f'start trainval with experiment dict {pformat(exp_configs.EXP_GROUPS)}')

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

    # Make sure the paths for the tensorboard feedback and the model improvement steps are present
    Path(args.tensorboard).mkdir(parents=True, exist_ok=True)
    Path(args.savedir_base).mkdir(parents=True, exist_ok=True)

    # Perform the trainval procedure on each of the experiments in the experiment dict:
    for exp_dict in exp_list:
        # do trainval
        trainval(exp_dict=exp_dict,
                savedir_base=args.savedir_base,
                datadir=args.datadir,
                reset=args.reset,
                num_workers=args.num_workers,
                tensorboard_folder=args.tensorboard)
