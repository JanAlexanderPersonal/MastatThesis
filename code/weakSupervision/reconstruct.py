from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu

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


def reconstruct_3d(exp_dict, model_type_name, savedir, ground_truth):

    logger.debug(f'model type name : {model_type_name}')
    logger.debug(f'save directory : {savedir}')

    reconstructor = rec.multi_dim_reconstructor(
        exp_dict, model_type_name=model_type_name)
    logger.info('START CONSTRUCTING THE VOLUMES')
    # reconstructor.make_3D_volumes(savedir)
    logger.info('START COMBINING THE VOLUMES')
    reconstructor.reconstruct_from_volumes(savedir, ground_truth)

    with open(os.path.join(savedir, 'exp_dict_reconstruct.json'), 'w') as f:
        json.dump(exp_dict, f)


if __name__ == "__main__":
    setuplogger()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('-mn', '--model_name', required=True)
    parser.add_argument('-sd', '--savedir', default=None)
    parser.add_argument('-gt', '--ground_truth',
                        default='dataset_1_contrast_3')

    args = parser.parse_args()

    exp_dict = {
        0: {
            "batch_size": 6,
            "dataset": {
                "bg_points": 5,
                "blob_points": 1,
                "context_span": 1,
                "crop_size": [
                    352,
                    352],
                "n_classes": 2,
                "name": "spine_dataset",
                "sources": [
                    "xVertSeg",
                    "USiegen",
                    "MyoSegmenTUM"]},
            "dataset_size": {
                "test": "all",
                "train": "all",
                        "val": "all"},
            "lr": 2.5e-05,
            "max_epoch": 150,
            "model": {
                "base": "fcn8_vgg16",
                "loss": [
                    "unsupervised_rotation_loss",
                    "rot_point_loss_multi",
                    "prior_extend",
                    "separation_loss"],
                "n_channels": 3,
                "n_classes": 2,
                "name": "inst_seg",
                "prior_extend": 70,
                "prior_extend_slope": 10},
            "num_channels": 1,
            "optimizer": "adam",
            "hash": "e5df8bf39051f574de84779e9b30c029"},
        1: {
            "batch_size": 6,
            "dataset": {
                "bg_points": 3,
                "blob_points": 1,
                "context_span": 1,
                "crop_size": [
                    352,
                    352],
                "n_classes": 6,
                "name": "spine_dataset",
                "sources": [
                    "xVertSeg",
                    "USiegen",
                    "MyoSegmenTUM"]},
            "dataset_size": {
                "test": "all",
                "train": "all",
                "val": "all"},
            "lr": 2.5e-05,
            "max_epoch": 150,
            "model": {
                "base": "fcn8_vgg16",
                "loss": [
                    "unsupervised_rotation_loss",
                    "rot_point_loss_multi",
                    "prior_extend",
                    "separation_loss"],
                "n_channels": 3,
                "n_classes": 6,
                "name": "inst_seg",
                "prior_extend": 110,
                "prior_extend_slope": 10},
            "num_channels": 1,
            "optimizer": "adam",
            "hash": "6ef4aab564b17ec9748cd7e25f651d09"},
        2: {
            "batch_size": 6,
            "dataset": {
                "bg_points": 3,
                "blob_points": 1,
                "context_span": 1,
                "crop_size": [
                    352,
                    352],
                "n_classes": 6,
                "name": "spine_dataset",
                "sources": [
                    "xVertSeg",
                    "USiegen",
                    "MyoSegmenTUM"]},
            "dataset_size": {
                "test": "all",
                "train": "all",
                "val": "all"},
            "lr": 2.5e-05,
            "max_epoch": 150,
            "model": {
                "base": "fcn8_vgg16",
                "loss": [
                    "unsupervised_rotation_loss",
                    "rot_point_loss_multi",
                    "prior_extend",
                    "separation_loss"],
                "n_channels": 3,
                "n_classes": 6,
                "name": "inst_seg",
                "prior_extend": 110,
                "prior_extend_slope": 10},
            "num_channels": 1,
            "optimizer": "adam",
            "hash": "6ef4aab564b17ec9748cd7e25f651d09"}}

    model_type_name = args.model_name
    save_dir = args.savedir
    ground_truth = args.ground_truth

    reconstruct_3d(exp_dict, model_type_name, save_dir, ground_truth)
