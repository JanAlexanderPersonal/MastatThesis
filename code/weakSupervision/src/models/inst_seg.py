# -*- coding: utf-8 -*-

import cv2 as cv2
from numpy.lib.type_check import imag
from torch.utils.tensorboard import SummaryWriter
import os
import pprint
import tqdm
import numpy as np
import pandas as pd
from haven import haven_utils as hu
from haven import haven_img as hi
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from .networks import infnet, fcn8_vgg16, unet_resnet, resnet_seam
from src import utils as ut
from src import models
from src.modules.lcfcn import lcfcn_loss
import sys
import kornia
from kornia.augmentation import RandomAffine
from scipy.ndimage.filters import gaussian_filter
from kornia.geometry.transform import flips
from . import optimizers, metrics, networks
from src.modules import sstransforms as sst

import cv2

F_checkoverlap = False

from typing import Dict, List

import logging

logger = logging.getLogger(__name__)


class Inst_Seg(torch.nn.Module):

    def __init__(self, exp_dict: Dict, weight_vector : List = None, tensorboard_folder: str = None):
        """Semantic segmentation class

        Args:
            exp_dict (Dict): Experiment definition dictionary
                {
                    'model' : (Dict) Definition of the model    {
                                                                    'n_classes' : n_classes,
                                                                    'base' : network to base the model upon,
                                                                    'optimizer' : model optimizer
                                                                }
                }
            tensorboard_folder (str, optional): Path to store the tensorboard logs. Defaults to None.
        """
        super().__init__()
        self.exp_dict = exp_dict
        self.train_hashes = set()
        self.n_classes = self.exp_dict['model'].get('n_classes', 1)
        

        self.init_model()
        self.first_time = True
        self.epoch = 0

        self.writer = None

        if weight_vector is None:
            self.weight_vector = [1.0] * self.n_classes
        else:
            self.weight_vector = weight_vector

        logger.info(f'weight vector : {self.weight_vector}')

        if tensorboard_folder is not None:
            self.writer = SummaryWriter(tensorboard_folder)

    def add_writer(self, tensorboard_folder: str):
        """Add_writer : add tensorboard writer

        Args:
            tensorboard_folder (str): Path to the folder to store the tensorboard logs
        """
        self.writer = SummaryWriter(tensorboard_folder)

    def init_model(self):
        """ Initialize the network specified in exp_dict['model']['base'] with the optimizer specified in exp_dict['model']['optimizer'] . """
        self.model_base = networks.get_network(self.exp_dict['model']['base'],
                                               n_classes=self.n_classes,
                                               exp_dict=self.exp_dict)
        self.cuda()
        self.opt = optimizers.get_optimizer(
            self.exp_dict['optimizer'], self.model_base, self.exp_dict)

    def get_state_dict(self) -> Dict:
        """Return the model status

        Returns:
            Dict: dict with model status : {'model' : model base (network) state_dict , 'opt' : optimizer, 'epoch' : (int) epoch}
        """
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict(),
                      'epoch': self.epoch}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        if 'opt' not in state_dict:
            return
        self.opt.load_state_dict(state_dict["opt"])
        self.epoch = state_dict['epoch']

    def train_on_loader(
            self, train_loader: torch.utils.data.DataLoader) -> float:
        """trian on a dataloader

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader to train the network on

        Returns:
            float: average score from the train monitor
        """
        logger.debug('start train on loader')
        # set model in train mode (not eval mode)
        self.train()
        self.epoch += 1
        n_batches = len(train_loader)

        # Display tqdm bar
        pbar = tqdm.tqdm(desc="Training", total=n_batches, leave=False)
        train_monitor = TrainMonitor()

        for batch in train_loader:
            score_dict = self.train_on_batch(batch)
            train_monitor.add(score_dict)
            msg = ' '.join(["%s: %.3f" % (k, v)
                            for k, v in train_monitor.get_avg_score().items()])
            pbar.set_description('Training - %s' % msg)
            # update the tqdm bar to display one more batch was processed
            pbar.update(1)

        pbar.close()

        if self.exp_dict.get('adjust_lr'):
            infnet.adjust_lr(self.opt,
                             self.epoch, decay_rate=0.1, decay_epoch=30)

        avg_score = train_monitor.get_avg_score()

        logger.debug(
            f'trained on loader - epoch {self.epoch} - resulting train loss : {avg_score}')
        for k, v in avg_score.items():
            self.writer.add_scalar(k, v, self.epoch)
        return avg_score

    def compute_mask_loss(self, loss_name: str, images,
                          logits, masks) -> float:
        """compute mask los

        Args:
            loss_name (str): The name of the loss to calculate. Possible values : 'cross_entropy' (binary cross entropy) & 'joint_cross_entropy'
            images (torch.Tensor): Pytorch tensor with the images
            logits (torch.Tensor): Pytorch tensor with the logits
            masks (torch.Tensor): Pytorch tensor with the actual masks

        Returns:
            float: calculated loss value
        """
        logger.debug('Calculate mask loss')
        if 'cross_entropy' in loss_name:
            if self.n_classes == 1:
                loss = F.binary_cross_entropy_with_logits(
                    logits, masks.float(), reduction='mean')
            else:
                #probs = F.log_softmax(logits, dim=1)
                #logger.debug(f'Size of the probs : {probs.size()}')
                #loss = F.nll_loss(
                #    probs, masks, reduction='mean', ignore_index=255) # The only 255 values should be the part OUTSIDE the actual image.
                loss = F.cross_entropy(logits, torch.squeeze(masks,1), ignore_index = 255)
                logger.debug(f'update loss : {loss}')
        if 'weighted_cross_entropy' in loss_name:
            loss = F.cross_entropy(logits, torch.squeeze(masks,1), ignore_index = 255, weight = torch.Tensor(self.weight_vector).to(logits.get_device()))
            logger.debug(f'update loss (weighted) : {loss}')


        elif 'joint_cross_entropy' in loss_name:
            # from src utils.py
            loss = ut.joint_loss(logits, masks.float())

        return loss

    def compute_point_loss(self, loss_name: str, images,
                           logits, points) -> float:
        """ Compute different types of point loss



        """

        loss = 0.0
        # add one axis to points [:,None]
        points = points[:, None]

        if 'unsupervised_flip_loss' in loss_name:
            # The consistency loss compares the logits of the original image
            # to the logits of the flipped image
            logits_flip = self.model_base(flips.Hflip()(images))
            loss += torch.mean(torch.abs(flips.Hflip()(logits_flip) - logits))
            logger.debug(f'unsupervised loss (flip) : {loss}')

        if 'unsupervised_rotation_loss' in loss_name:
            rotations = np.random.choice(
                [0, 90, 180, 270], points.shape[0], replace=True)
            images = flips.Hflip()(images)
            images_rotated = sst.batch_rotation(images, rotations)
            logits_rotated = self.model_base(images_rotated)
            logits_recovered = sst.batch_rotation(
                logits_rotated, 360 - rotations)
            logits_recovered = flips.Hflip()(logits_recovered)

            loss += torch.mean(torch.abs(logits_recovered - logits))
            logger.debug(f'Unsupervised loss (rotation) : {loss}')


        if 'rot_point_loss' in loss_name:
            """ Flips the image and computes a random rotation of
                {0, 90, 180, 270} degrees
            """

            assert 'unsupervised_rotation_loss' in loss_name, 'It is necessary to calculate the unsupervised rotation loss before the supervised rotation consistancy loss can be calculated.'

            logger.debug(
                f'compute rotational point loss with points {torch.unique(points)}')
            
            # Due to calculation of the unsupervises rotation loss, the variables are already calculated:
            #   rotations
            #   logits_recovered
            #   logits_rotated

            ind = points != 255
            if ind.sum() != 0:
                loss += F.binary_cross_entropy_with_logits(logits[ind],
                                                           points[ind].detach(
                ).float().cuda(),
                    reduction='mean')

                points_rotated = flips.Hflip()(points)
                points_rotated = sst.batch_rotation(points_rotated, rotations)
                ind = points_rotated != 255
                loss += F.binary_cross_entropy_with_logits(logits_rotated[ind],
                                                           points_rotated[ind].detach(
                ).float().cuda(),
                    reduction='mean')

        



        if ('rot_point_loss_multi' in loss_name) or ('rot_point_loss_multi_weighted' in loss_name):
            """ Flips the image and computes a random rotation of
                {0, 90, 180, 270} degrees
                Adapted for multi-channel loss (n_classes > 2)
            """

            assert 'unsupervised_rotation_loss' in loss_name, 'It is necessary to calculate the unsupervised rotation loss before the supervised rotation consistancy loss can be calculated.'

            logger.debug(
                'Start to compute rotational point loss with multiple classes')
            
            # Due to calculation of the unsupervises rotation loss, the variables are already calculated:
            #   rotations
            #   logits_recovered
            #   logits_rotated

            # Todo: There is some code repetation compared to the other
            # consistency class.

            for i in range(self.n_classes):

                # If this batch does not contain a certain point type, there is
                # no loss for this point type
                if i not in torch.unique(points):
                    continue

                points_temp = points.clone().detach()

                logger.debug(f'Extract rotational loss for class {i}.')
                # If you are really searching for the background class, you
                # should give this a temporary different value from 0 and from
                # the other class labels
                if i == 0:
                    points_temp[points == i] = self.n_classes
                for j in [x for x in range(self.n_classes) if x != i]:
                    # points temporary only has two types of points: The class
                    # to be investigated and background (for that channel)
                    points_temp[points == j] = 0
                logger.debug(
                    f'Unique values remaining after bringing the other classes to background: {torch.unique(points_temp)}')
                logger.debug(
                    f'To calculate the additional loss for channel {i}, the points tensor becomes {torch.unique(points_temp, return_counts = True)}. The non-zero value will be converted to 1.')
                points_temp[points == i] = 1
                # Revert the possible re-naming of the background class
                points_temp[points == self.n_classes] = 1

                # Extract the layer from the model output
                index_to_device = torch.tensor([i]).to(logits.device)
                logits_slice = logits.index_select(1, index_to_device)
                logits_rotated_slice = logits_rotated.index_select(
                    1, index_to_device)

                logger.debug(
                    f'two tensors are sliced : logits slice has dimensions {logits_slice.size()} and logits_flip slice has dimensions {logits_rotated_slice.size()}')

                ind = points_temp != 255

                weight = 1.0 if ('rot_point_loss_multi_weighted' not in loss_name) else self.weight_vector[i]

                logger.debug(f'weight for class i : {weight}')

                if ind.sum() != 0:
                    loss += F.binary_cross_entropy_with_logits(logits_slice[ind],
                                                               points_temp[ind].detach(
                    ).float().cuda(),
                        reduction='mean') * weight

                    logger.debug(
                        f'straight logits loss added for channel {i} : {loss}')

                    points_rotated = flips.Hflip()(points_temp)
                    points_rotated = sst.batch_rotation(
                        points_rotated, rotations)
                    ind = points_rotated != 255
                    loss += F.binary_cross_entropy_with_logits(logits_rotated_slice[ind],
                                                               points_rotated[ind].detach(
                    ).float().cuda(),
                        reduction='mean') * weight

                    logger.debug(
                        f'rotated logits loss added for channel {i} : {loss}')

        

        if 'cons_point_loss' in loss_name:
            """ CB point loss, see Laradji et al. 2020  - This loss is only intended for semantic segmentation"""

            # Consistency loss based on flip (only)
            assert 'unsupervised_flip_loss' in loss_name, 'It is necessary to calculate the unsupervised flip loss before the supervised rotation consistancy loss can be calculated.'

            logger.debug(f'Calculate consistency point loss')

            logger.debug(f'logits shape : {logits.size()}')
            logger.debug(f'logits device : {logits.device}')

            ind = points != 255
            if ind.sum() != 0:
                loss += F.binary_cross_entropy_with_logits(logits[ind],
                                                           points[ind].float(
                ).cuda(),
                    reduction='mean')
                logger.debug(f'straight logits loss added : {loss}')

                points_flip = flips.Hflip()(points)
                ind = points_flip != 255
                loss += F.binary_cross_entropy_with_logits(logits_flip[ind],
                                                           points_flip[ind].float(
                ).cuda(),
                    reduction='mean')
                logger.debug(f'flipped logits loss added : {loss}')

        if ('cons_point_loss_multi' in loss_name) or ('cons_point_loss_multi_weighted' in loss_name):
            """ Extension of the consistency point loss to multi-class segmentation """

            # Consistency loss based on flip (only)
            assert 'unsupervised_flip_loss' in loss_name, 'It is necessary to calculate the unsupervised flip loss before the supervised rotation consistancy loss can be calculated.'

            logger.debug(f'Calculate multi-class consistency point loss')

            

            # add loop over all labels:
            # convert to logits[:,i,:,:] to extract correct channel

            logger.debug(f'logits shape : {logits.size()}')
            logger.debug(f'points shape: {points.size()} *** {type(points)}')

            assert logits.size()[
                1] == self.n_classes, "The output of the model should contain a channel for each individual class"

            # The following part converts the point labels to annotations for the individual channels:
            #       The original labels: array with same H×W as the image.
            #       Everywhere 255
            #           except: 0 for background points
            #                   1 --> 5 to indicate L1 --> L5

            loss += F.cross_entropy(logits, points.cuda(), ignore_index = 255, weight = torch.Tensor(self.weight_vector).to(logits.get_device()))
            points_flip = flips.Hflip()(points)
            loss += F.cross_entropy(logits_flip, points_flip.cuda(), ignore_index = 255, weight = torch.Tensor(self.weight_vector).to(logits.get_device()))


            """
            for i in range(self.n_classes):

                if i not in torch.unique(points):
                    continue

                points_temp = points.clone().detach()

                logger.debug(f'Extract flip loss for class {i}')
                # If you are really searching for the background class, you
                # should give this a temporary different value from 0 and from
                # the other class labels
                if i == 0:
                    points_temp[points == i] = self.n_classes
                for j in [x for x in range(self.n_classes) if x != i]:
                    # points temporary only has two types of points: The class
                    # to be investigated and background (for that channel)
                    points_temp[points == j] = 0
                logger.debug(
                    f'Unique values remaining after bringing the other classes to background: {torch.unique(points_temp)}')
                logger.debug(
                    f'To calculate the additional loss for channel {i}, the points tensor becomes {torch.unique(points_temp, return_counts = True)}. The non-zero value will be converted to 1.')
                points_temp[points == i] = 1
                # Revert the possible re-naming of the background class
                points_temp[points == self.n_classes] = 1

                # Extract the layer from the model output
                index_to_device = torch.tensor([i]).to(logits.device)
                logits_slice = logits.index_select(1, index_to_device)
                logits_flip_slice = logits_flip.index_select(
                    1, index_to_device)

                logger.debug(
                    f'two tensors are sliced : logits slice has dimensions {logits_slice.size()} and logits_flip slice has dimensions {logits_flip_slice.size()}')

                ind = points_temp != 255

                weight = 1.0 if ('cons_point_loss_multi_weighted' not in loss_name) else self.weight_vector[i]

                logger.debug(f'weight for class i : {weight}')

                if ind.sum() != 0:
                    loss += F.binary_cross_entropy_with_logits(logits_slice[ind],
                                                               points_temp[ind].float(
                    ).cuda(),
                        reduction='mean') * weight
                    logger.debug(
                        f'straight logits loss added for channel {i} : {loss}')

                    points_flip = flips.Hflip()(points_temp)
                    ind = points_flip != 255
                    loss += F.binary_cross_entropy_with_logits(logits_flip_slice[ind],
                                                               points_flip[ind].float(
                    ).cuda(),
                        reduction='mean') * weight
                    logger.debug(
                        f'flipped logits loss added for channel {i} : {loss}')
                    logger.debug(
                        f'This was calculated with BCE logits slice [ind] {logits_slice[ind]} vs points temp {points_temp[ind]}\n and logits flip slice {logits_flip_slice[ind]} vs {points_flip[ind]}.')
            """
        
        if 'prior_extend' in loss_name:

            # mask for logits BCHW
            # This loss takes into account that a typical vertebra is not larger than 100 mm x 100 mm x 100 mm
            mask = np.ones(list(logits.size())) * np.infty
            logging.debug(f'Mask shape {mask.shape}. points shape {points.size()}')

            
            # Go over all the non-background classes (points marked with value > 0)
            for i in range(1, self.n_classes):
                indices = (points == i).nonzero(as_tuple=False)
                logging.debug(f'channel {i} indices {indices}')
                for ind in indices:
                    if len(ind) == 0:
                        continue
                    logging.debug(f'add mask distance from {ind}')
                    p = ind[2:].tolist() # point coordinates (H,W)
                    b = ind[0].tolist() # batch nr
                    mask[b, i, :, :] = np.minimum(mask[b, i, :, :], ut.vectorized_distance(mask[b, i, :, :], p)) 
            
            EXTEND = self.exp_dict['model'].get('prior_extend', 70)
            SLOPE = self.exp_dict['model'].get('prior_extend_slope', 10)
            
            mask = ((-1) * mask + EXTEND) / SLOPE

            # When calculating this loss, you need to restrict yourself to the NON-BACKGROUND classes!

            mask = torch.from_numpy(mask).to(logits.device).sigmoid()[:,1:,:,:]
            loss += F.binary_cross_entropy_with_logits(logits[:,1:, :, :], mask, reduction = 'mean')
            logger.debug(f'loss after adding prior extend loss: {loss}')

        if 'separation_loss' in loss_name:

            # force the network to only output one high value per channel: the classes are mutually exclusive.
            # the channel outputs have to be as different as possible.

            sigm = logits.sigmoid()

            for i in range(self.n_classes):
                index_i_to_device = torch.tensor([i]).to(logits.device)
                sigm_slice_i = sigm.index_select(1, index_i_to_device)
                differences = torch.zeros_like(sigm_slice_i)
                for j in range(i+1, self.n_classes):
                    if i == j:
                        continue
                    index_j_to_device = torch.tensor([j]).to(logits.device)
                    sigm_slice_j = sigm.index_select(1, index_j_to_device)
                    differences = torch.sum(torch.stack([differences, torch.abs(sigm_slice_i - sigm_slice_j)]), dim=0)

                # The higher the difference, the better!
                loss -= torch.mean(differences)
                logger.debug(f'differences of channel {i} with the others *** with differences size {differences.size()} *** new loss {loss}')



        if "elastic_cons_point_loss" in loss_name:
            """ Performs an elastic transformation to the images and logits and
                computes the consistency between the transformed logits and the
                logits of the transformed images see: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a """
            points = points.float().cuda()
            ind = points != 255

            B, C, H, W = images.shape
            # Sample normalized elastic grid

            def norm_grid(grid):
                grid -= grid.min()
                grid /= grid.max()
                grid = (grid - 0.5) * 2
                return grid
            grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W))
            grid_x = grid_x.float().cuda()
            grid_y = grid_y.float().cuda()
            sigma = self.exp_dict["model"]["sigma"]
            alpha = self.exp_dict["model"]["alpha"]
            indices = torch.stack([grid_y, grid_x], -
                                  1).view(1, H, W, 2).expand(B, H, W, 2).contiguous()
            indices = norm_grid(indices)
            dx = gaussian_filter(
                (np.random.rand(
                    H,
                    W) * 2 - 1),
                sigma,
                mode="constant",
                cval=0) * alpha
            dy = gaussian_filter(
                (np.random.rand(
                    H,
                    W) * 2 - 1),
                sigma,
                mode="constant",
                cval=0) * alpha
            dx = torch.from_numpy(dx).cuda().float()
            dy = torch.from_numpy(dy).cuda().float()
            dgrid_x = grid_x + dx
            dgrid_y = grid_y + dy
            dgrid_y = norm_grid(dgrid_y)
            dgrid_x = norm_grid(dgrid_x)
            dindices = torch.stack(
                [dgrid_y, dgrid_x], -1).view(1, H, W, 2).expand(B, H, W, 2).contiguous()
            # Use the grid to sample from the image and the logits
            images_aug = F.grid_sample(images, dindices)
            logits_aug = self.model_base(images_aug)
            aug_logits = F.grid_sample(logits, dindices)
            points_aug = F.grid_sample(points, dindices, mode='nearest')
            loss = self.exp_dict['model']["loss_weight"] * \
                torch.mean(torch.abs(logits_aug - aug_logits))

            ind = points != 255
            if ind.sum() != 0:
                loss += 2 * F.binary_cross_entropy_with_logits(logits[ind],
                                                               points[ind],
                                                               reduction='mean')
                ind = points_aug != 255
                loss += F.binary_cross_entropy_with_logits(logits_aug[ind],
                                                           points_aug[ind].detach(
                ),
                    reduction='mean')

        if 'lcfcn_loss' in loss_name:
            loss = 0.

            for lg, pt in zip(logits, points):
                loss += lcfcn_loss.compute_loss((pt == 1).long(), lg.sigmoid())

                # loss += lcfcn_loss.compute_binary_lcfcn_loss(l[None],
                #         p[None].long().cuda())

        if 'point_loss' in loss_name:
            ind = points != 255
            if ind.sum() == 0:
                loss = 0.
            else:
                loss = F.binary_cross_entropy_with_logits(logits[ind],
                                                          points[ind].float(
                ).cuda(),
                    reduction='mean')

            # print(points[ind].sum().item(), float(loss))
        if 'att_point_loss' in loss_name:
            ind = points != 255

            loss = 0.
            if ind.sum() != 0:
                loss = F.binary_cross_entropy_with_logits(logits[ind],
                                                          points[ind].float(
                ).cuda(),
                    reduction='mean')

                logits_flip = self.model_base(flips.Hflip()(images))
                points_flip = flips.Hflip()(points)
                ind = points_flip != 255
                loss += F.binary_cross_entropy_with_logits(logits_flip[ind],
                                                           points_flip[ind].float(
                ).cuda(),
                    reduction='mean')

        logger.debug(f'Final loss returned : {loss}')
        return loss

    def train_on_batch(self, batch):
        """train base network of this model with 1 batch from the dataloader

        Args:
            batch (Dict[torch.Tensor]): Dictionary with the batch:
                                            {
                                                'image' : (torch.Tensor) [BCHW], -> C = 3
                                                'mask' : (torch.Tensor) [BCHW], -> C = 1 with
                                                'points' : (torch.Tensor) [BCHW] -> C = n_classes + 1
                                            }

        Returns:
            [type]: [description]
        """
        #logger.debug('train on batch : ')
        # add to seen images
        for m in batch['meta']:
            self.train_hashes.add(m['hash'])

        # the loss name is either a string with the loss name or a list of stings with loss names
        self.opt.zero_grad()
        loss_name = self.exp_dict['model']['loss']
        if isinstance(loss_name, str):
            loss_name = [loss_name]

        images = batch["images"].cuda()
        logits = self.model_base(images) if not any(a in [
            'multiscale_cons_point_loss'] for a in loss_name) else self.model_base(images, return_features=True)
        logger.debug('shape batch images (BCHW): {}'.format(images.shape))
        logger.debug(f'shape logit (BCHW): {logits.shape}')
        logger.debug(
            'shape points (BCHW): {} with unique values {}'.format(
                batch['points'].shape,
                torch.unique(
                    batch['points'],
                    return_counts=True)))
        logger.debug(
            'shape unsqueezed points: {}'.format(
                torch.unsqueeze(
                    batch['points'],
                    1).shape))



        # compute loss
        point_loss_list = ['point_loss', 'prior_extend', 'cons_point_loss', 'cons_point_loss_multi', 'lcfcn_loss', 'affine_cons_point_loss', 'rot_point_loss', 'rot_point_loss_multi', 'elastic_cons_point_loss', 'toponet', 'multiscale_cons_point_loss']

        if any( a in ['joint_cross_entropy', 'cross_entropy', 'weighted_cross_entropy'] for a in loss_name):
            logger.debug('Start full supervision loss')
            # full supervision loss!
            loss = self.compute_mask_loss(
                loss_name, images, logits, masks=batch["masks"].cuda())
        elif any( a in point_loss_list for a in loss_name):
            # point supervision
            loss = self.compute_point_loss(
                loss_name, images, logits, points=batch["points"].cuda())

        if loss != 0:
            logger.debug(
                f'Loss calculated (value {loss}). Start backward step')
            loss.backward()
            if True: #self.exp_dict['model'].get('clip_grad'):
                ut.clip_gradient(self.opt, 0.5)
            try:
                self.opt.step()
            except BaseException:
                self.opt.step(loss=loss)

        return {'train_loss': float(loss)}

    @torch.no_grad()
    def predict_on_batch(self, batch):
        self.eval()
        image = batch['images'].cuda()
        logging.debug('predict on bach')

        if hasattr(self.model_base, 'predict_on_batch'):
            return self.model_base.predict_on_batch(batch)
            logging.debug('model_base has predict_on_batch attribute')
            s5, s4, s3, s2, se = self.model_base.forward(image)
            res = s2
            # Replace upsample by interpolate
            #res = F.upsample(res, size=batch['meta'][0]['shape'],mode='bilinear', align_corners=False)
            res = F.interpolate(
                res,
                size=batch['meta'][0]['shape'],
                mode='bilinear',
                align_corners=False)
            res = res.sigmoid().data.cpu().numpy()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = res > 0.5

        elif self.n_classes == 1:
            res = self.model_base.forward(image)
            if 'shape' in batch['meta'][0]:
                # Replace upsample by interpolate
                #res = F.upsample(res, size=batch['meta'][0]['shape'],mode='bilinear', align_corners=False)
                res = F.interpolate(
                    res,
                    size=batch['meta'][0]['shape'],
                    mode='bilinear',
                    align_corners=False)
            res = (res.sigmoid().data.cpu().numpy() > 0.5).astype('float')
        else:
            self.eval()
            logits = self.model_base.forward(image)
            logger.debug(f'dimension calculated logits : {logits.size()}')
            # unsqueeze the dimension lost in taking the argmax.
            # array with the maximal values along the channels axis -> index of
            # max channel
            res = torch.unsqueeze(
                logits.argmax(
                    dim=1),
                dim=0).data.cpu().numpy()

        logger.debug(
            f'predict_on_batch return result with dimension {res.shape} and type {type(res)}')

        return res

    def probabilities_on_batch(self, batch) -> np.ndarray:
        """add a numpy array to the batch with the predicted probabilities

        Args:
            batch (batch): Batch from the dataloader

        Returns:
            batch: same batch with the probabilities added
        """
        self.eval()
        image = batch['images'].cuda()
        logits = self.model_base.forward(image)
        batch.update({'probs' : logits.sigmoid().data.cpu().numpy()})
        return batch

    def vis_on_batch(self, batch, savedir_image, i=0):
        # Get torch tensors with the images and the masks
        image = hu.denormalize(batch['images'], mode='rgb')[0]
        gt = batch['masks'].data.cpu().numpy()
        points = batch['points'].data.cpu().numpy()


        # If there are context slices, you do not want them in the image
        image_center = np.stack([image[1,:,:] for i in range(3)], axis = 0)

        logger.debug(
            f'Ground truth unique values : {np.unique(gt)} with shape {gt.shape}')
        gt = gt[0, 0, :, :]
        points = points[0,:,:]
        # np.ndarray with the max values of the channels
        res = self.predict_on_batch(batch)[0, 0, :, :]

        logger.debug(
            f'vis_on_batch - Ground truth unique values : {np.unique(gt)}, res shape and unique values {res.shape} ** {np.unique(res)}')


        res_segm = colour_segments(res, stack_axis=0) / 255.
        gt_segm = colour_segments(gt, stack_axis=0) / 255.

        gt_segm_stack = cv2.addWeighted(image, 0.85, gt_segm, 0.15, 0)
        gt_segm = colour_points(points, gt_segm)

        res_segm_stack = cv2.addWeighted(image, 0.75, res_segm, 0.25, 0)

        # Create numpy arrays for the colour channels

        logger.debug(
            f'image : {image.shape} [{np.min(image)} - {np.max(image)}]** gt_segm : {gt_segm.shape}  [{np.min(gt_segm)} - {np.max(gt_segm)}]** res_segm : {res_segm.shape}')

        img_list = [image, gt_segm, res_segm]
        img_list2 = [image_center, gt_segm_stack, res_segm_stack]
        img_comp = np.concatenate(img_list, axis=2)
        img_comp2 = np.concatenate(img_list2, axis=2)
        img_comp = np.concatenate([img_comp, img_comp2], axis = 1)
        logging.debug(f'image list from visualize on batch : {img_comp.shape}')

        # write to tensorboard
        self.writer.add_images(
            f'compare_img_{i}',
            img_comp,
            self.epoch,
            dataformats='CHW')
        #input('press enter')
        hu.save_image(savedir_image, img_comp)

        # Check if this image contains any of the non-background classes:
        return any([(i in np.unique(gt)) for i in range(1,5)])
    

    def val_on_loader(self, loader, savedir_images=None, n_images=0, n_jobs = -1):
        """Get validation score dictionary

        Args:
            loader (torch.dataloader): dataloader for this split
            savedir_images (str, optional): Directory to save the haven images. Defaults to None.
            n_images (int, optional): Number of images. Defaults to 0.
            n_jobs (int): for parallel execution

        Returns:
            Dict[float]: Dictionary with 8 different metric scores:
                            <split>_dice
                            <split>_iou
                            <split>_prec (precision)
                            <split>_recall (sensitivity)
                            <split>_fscore (F1 - metric)
                            <split>_score (Dice score)
                            <split>_struct (structured metric - objectness score)
        """

        def check_hash( m ):
            return (m['hash'] not in self.train_hashes)


        self.eval()
        val_meter = metrics.SegMeter(split=loader.dataset.split)
        logger.debug(
            f'Start validation on batch for {loader.dataset.split} set.')

        # Check the batches in the loader have not been trained on:

        if loader.dataset.split != 'train' and F_checkoverlap:
            for batch in tqdm.tqdm(loader, desc = 'Assure no overlap with train set'):
                h_c = [check_hash(m) for m in batch['meta']]
                if not all(h_c):
                    logger.warning('information leakage : {}'.format(batch['meta']))
                # make sure it wasn't trained on
                assert( all(h_c)), 'Information leakage'
                    

        
        for batch in tqdm.tqdm(loader, desc = 'validate batches in val meter'):
            val_meter.val_on_batch(self, batch)

        i_count = 0
        for batch in loader:
            if i_count < n_images:
                # This function returns if the image contains all labels as a boolean
                i_count += int(self.vis_on_batch(batch, savedir_image=os.path.join(savedir_images,
                                                                    '%d.png' % batch['meta'][0]['index']), i=i_count))
            else:
                break
        
        avg_score = val_meter.get_avg_score()
        for key, value in avg_score.items():
            if key.endswith('metrics_df'):
                continue
            self.writer.add_scalar(key, value, self.epoch)

        metrics_df = val_meter.metrics_df()

        logger.debug(f'Matrix with metrics : \n{metrics_df}')

        return avg_score, metrics_df

    @torch.no_grad()
    def compute_uncertainty(self, images, replicate=False,
                            scale_factor=None, n_mcmc=20, method='entropy'):
        self.eval()
        set_dropout_train(self)

        # put images to cuda
        images = images.cuda()
        _, _, H, W = images.shape

        if scale_factor is not None:
            images = F.interpolate(images, scale_factor=scale_factor)
        # variables
        input_shape = images.size()
        batch_size = input_shape[0]

        if replicate and False:
            # forward on n_mcmc batch
            images_stacked = torch.stack([images] * n_mcmc)
            images_stacked = images_stacked.view(
                batch_size * n_mcmc, *input_shape[1:])
            logits = self.model_base(images_stacked)

        else:
            # for loop over n_mcmc
            logits = torch.stack([self.model_base(images)
                                  for _ in range(n_mcmc)])

            logits = logits.view(batch_size * n_mcmc, *logits.size()[2:])

        logits = logits.view([n_mcmc, batch_size, *logits.size()[1:]])
        _, _, n_classes, _, _ = logits.shape
        # binary do sigmoid
        if n_classes == 1:
            probs = logits.sigmoid()
        else:
            probs = F.softmax(logits, dim=2)

        if scale_factor is not None:
            probs = F.interpolate(probs, size=(probs.shape[2], H, W))

        self.eval()

        if method == 'entropy':
            score_map = - xlogy(probs).mean(dim=0).sum(dim=1)

        if method == 'bald':
            left = - xlogy(probs.mean(dim=0)).sum(dim=1)
            right = - xlogy(probs).sum(dim=2).mean(0)
            bald = left - right
            score_map = bald

        return score_map


class TrainMonitor:
    def __init__(self):
        self.score_dict_sum = {}
        self.n = 0

    def add(self, score_dict):
        for k, v in score_dict.items():
            if k not in self.score_dict_sum:
                self.score_dict_sum[k] = score_dict[k]
            else:
                self.n += 1
                self.score_dict_sum[k] += score_dict[k]

    def get_avg_score(self):
        return {k: v / (self.n + 1) for k, v in self.score_dict_sum.items()}


def set_dropout_train(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(
                module, torch.nn.Dropout2d):
            module.train()


def xlogy(x, y=None):
    z = torch.zeros(())
    if y is None:
        y = x
    assert y.min() >= 0
    return x * torch.where(x == 0., z.cuda(), torch.log(y))


LABEL_COLOR_MAP = {
    0: (255, 255, 255),     # Background = WHITE
    1: (128, 0, 128),     # L1 = MAROON
    2: (255, 0, 0),     # L2 = RED
    3: (0, 255, 0),     # L3 = LIME
    4: (0, 0, 255),     # L4 = BLUE
    5: (0, 255, 255)            # L5 = YELLOW
}


def colour_segments(res: np.ndarray, stack_axis=2) -> np.ndarray:
    """Create colour impage for the segments

    Args:
        res (np.ndarray): [H, W] every pixel respresents a label that should be part of LABEL_COLOR_MAP

    Returns:
        np.ndarray: [H,W,3] Colour image with label colours
    """

    #logger.debug(f'colour segments with res : {np.unique(res)}')

    red_map = np.zeros_like(res).astype(np.uint8)
    green_map = np.zeros_like(res).astype(np.uint8)
    blue_map = np.zeros_like(res).astype(np.uint8)

    for label_num, colour in LABEL_COLOR_MAP.items():
        #logger.debug(f'label number {label_num} gets colour {colour}')
        index = res == label_num
        red_map[index] = colour[0]
        green_map[index] = colour[1]
        blue_map[index] = colour[2]

    img_segm = np.stack([red_map, green_map, blue_map], axis=stack_axis)
    return img_segm


def colour_points(points: np.ndarray, image : np.ndarray = None, radius : int = 2) -> np.ndarray:
    """Create coloured circles for the points

    Args:
        points (np.ndarray): [HW] image that is 255 everywhere except for the annotation points

    Returns:
        np.ndarray: Image that is white everywhere except for circles around the annotation points
    """

    canvas = np.stack([np.zeros_like(points) for i in range(3)], axis=2)

    for label_num, colour in LABEL_COLOR_MAP.items():

        y_c, x_c = np.where(points == label_num)
        logger.debug(f'Point label {label_num} ')

        for y_i, x_i in zip(y_c, x_c):
            if image is None:
                canvas = cv2.circle(canvas, (y_i, x_i), 2, colour , -1)
            else:
                logger.debug(f'draw circle at position {(y_i, x_i)} with colour {colour} on image with shape {image.shape}')
                canvas = [None] * 3
                for dim in range(3):
                    image[dim, :, :]
                    colour[dim]
                    canvas[dim] = cv2.circle( image[dim, :, :] , (x_i, y_i), 5, colour[dim] / 255., -1 )
                canvas = np.stack(canvas, axis = 0) 

    logger.debug(f'Return canvas shape {canvas.shape}')
    return canvas