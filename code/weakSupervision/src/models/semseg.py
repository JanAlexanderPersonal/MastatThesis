# -*- coding: utf-8 -*-

import os, pprint, tqdm
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

from typing import Dict

import logging

logger = logging.getLogger(__name__)

from torch.utils.tensorboard import SummaryWriter

class SemSeg(torch.nn.Module):

    def __init__(self, exp_dict : Dict, tensorboard_folder : str = None):
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
        if tensorboard_folder is not None:
            self.writer = SummaryWriter(tensorboard_folder)

    def add_writer(self, tensorboard_folder : str):
        """Add_writer : add tensorboard writer

        Args:
            tensorboard_folder (str): Path to the folder to store the tensorboard logs
        """
        self.writer=SummaryWriter(tensorboard_folder)
    
    def init_model(self):
        """ Initialize the network specified in exp_dict['model']['base'] with the optimizer specified in exp_dict['model']['optimizer'] . """
        self.model_base = networks.get_network(self.exp_dict['model']['base'],
                                              n_classes=self.n_classes,
                                              exp_dict=self.exp_dict)
        self.cuda()
        self.opt = optimizers.get_optimizer(self.exp_dict['optimizer'], self.model_base, self.exp_dict)


    def get_state_dict(self) -> Dict:
        """Return the model status

        Returns:
            Dict: dict with model status : {'model' : model base (network) state_dict , 'opt' : optimizer, 'epoch' : (int) epoch}
        """
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict(),
                      'epoch':self.epoch}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        if 'opt' not in state_dict:
            return
        self.opt.load_state_dict(state_dict["opt"])
        self.epoch = state_dict['epoch']

    def train_on_loader(self, train_loader : torch.utils.data.DataLoader) -> float:
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
            msg = ' '.join(["%s: %.3f" % (k, v) for k,v in train_monitor.get_avg_score().items()])
            pbar.set_description('Training - %s' % msg)
            # update the tqdm bar to display one more batch was processed
            pbar.update(1)
            
        pbar.close()

        if self.exp_dict.get('adjust_lr'):
            infnet.adjust_lr(self.opt,
                                self.epoch, decay_rate=0.1, decay_epoch=30)

        avg_score = train_monitor.get_avg_score()

        logger.debug(f'trained on loader - epoch {self.epoch} - resulting train loss : {avg_score}')
        for k, v in avg_score.items():
            self.writer.add_scalar(k, v, self.epoch)
        return avg_score

    def compute_mask_loss(self, loss_name : str, images, logits, masks) -> float:
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
        if loss_name == 'cross_entropy':
            if self.n_classes == 1:
                loss = F.binary_cross_entropy_with_logits(logits, masks.float(), reduction='mean')
            else:
                probs = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(probs, masks, reduction='mean', ignore_index=255)

        elif loss_name == 'joint_cross_entropy':
            # from src utils.py
            loss = ut.joint_loss(logits, masks.float())
        
        return loss 

    def compute_point_loss(self, loss_name : str, images, logits, points) -> float:
        """ Compute different types of point loss



        """

        if loss_name == 'rot_point_loss':
            """ Flips the image and computes a random rotation of 
                {0, 90, 180, 270} degrees
            """

            # add one axis to points [:,None]
            points = points[:,None]
            logger.debug(f'compute rotational point loss with points {torch.unique(points)}')
            rotations = np.random.choice([0, 90, 180, 270], points.shape[0], replace=True)
            images = flips.Hflip()(images)
            images_rotated = sst.batch_rotation(images, rotations)
            logits_rotated = self.model_base(images_rotated)
            logits_recovered = sst.batch_rotation(logits_rotated, 360 - rotations)
            logits_recovered = flips.Hflip()(logits_recovered)

            logger.debug('shape batch rotated images (BCHW): {}'.format(images_rotated.shape))
            logger.debug(f'shape logits recovered (BCHW): {logits_recovered.shape}')
            
            loss = torch.mean(torch.abs(logits_recovered-logits))
            
            ind = points!=255
            if ind.sum() != 0:
                loss += F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind].detach().float().cuda(), 
                                        reduction='mean')

                points_rotated = flips.Hflip()(points)
                points_rotated = sst.batch_rotation(points_rotated, rotations)
                ind = points_rotated!=255
                loss += F.binary_cross_entropy_with_logits(logits_rotated[ind], 
                                        points_rotated[ind].detach().float().cuda(), 
                                        reduction='mean')

        elif loss_name == 'cons_point_loss':
            """ CB point loss, see Laradji et al. 2020  - This loss is only intended for semantic segmentation"""
            
            # Consistency loss based on flip (only)

            logger.debug(f'Calculate consistency point loss')

            # add one axis dimension to the points
            points = points[:,None]
            
            # The consistency loss compares the logits of the original image
            # to the logits of the flipped image
            logits_flip = self.model_base(flips.Hflip()(images))
            loss = torch.mean(torch.abs(flips.Hflip()(logits_flip)-logits))
            logger.debug(f'unsupervised loss : {loss}')

            logger.debug(f'logits shape : {logits.size()}')
            logger.debug(f'logits device : {logits.device}')
            logger.debug(f'select chan 0 : {logits.index_select(1, torch.tensor([0]).to(logits.device)).size()}')

            ind = points!=255
            if ind.sum() != 0:
                loss += F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind].float().cuda(), 
                                        reduction='mean')
                logger.debug(f'straight logits loss added : {loss}')

                points_flip = flips.Hflip()(points)
                ind = points_flip!=255
                loss += F.binary_cross_entropy_with_logits(logits_flip[ind], 
                                        points_flip[ind].float().cuda(), 
                                        reduction='mean')
                logger.debug(f'flipped logits loss added : {loss}')

        elif loss_name == 'cons_point_loss_multi':
            """ Extension of the consistency point loss to multi-class segmentation """
            
            # Consistency loss based on flip (only)

            logger.debug(f'Calculate multi-class consistency point loss')

            # add one axis dimension to the points
            points = points[:,None]
            
            # The consistency loss compares the logits of the original image
            # to the logits of the flipped image
            logits_flip = self.model_base(flips.Hflip()(images))
            loss = torch.mean(torch.abs(flips.Hflip()(logits_flip)-logits))
            logger.debug(f'unsupervised loss : {loss}')
            
            # add loop over all labels:
            # convert to logits[:,i,:,:] to extract correct channel

            logger.debug(f'logits shape : {logits.size()}')
            logger.debug(f'logits device : {logits.device}')
            logger.debug(f'select chan 0 : {logits.index_select(1, torch.tensor([0]).to(logits.device)).size()}')

            assert logits.size()[1] == self.n_classes, "The output of the model should contain a channel for each individual class"

            # The following part converts the point labels to annotations for the individual channels:
            #       The original labels: array with same H×W as the image. 
            #       Everywhere 255 
            #           except: 0 for background points
            #                   1 --> 5 to indicate L1 --> L5

            for i in range(1,6):
                logger.debug(f'Extract flip loss for class {i}')
                points_temp = points.clone().detach()
                for j in [x for x in range(1,6) if x != i]:
                    points_temp[points == j] = 0
                
                logger.debug(f'To calculate the additional loss for channel {i}, the points tensor becomes {torch.unique(points_temp, return_counts = True)}. The non-zero value will be converted to 1.')
                points_temp[points == i] = 1

                # Extract the layer from the model output
                index_to_device = torch.tensor([i]).to(logits.device)
                logits_slice = logits.index_select(1, index_to_device)
                logits_flip_slice = logits_flip.index_select(1, index_to_device )

                logger.debug(f'two tensors are sliced : logits slice has dimensions {logits_slice.size()} and logits_flip slice has dimensions {logits_flip_slice.size()}')

                ind = points_temp!=255
                if ind.sum() != 0:
                    loss += F.binary_cross_entropy_with_logits(logits_slice[ind], 
                                            points[ind].float().cuda(), 
                                            reduction='mean')
                    logger.debug(f'straight logits loss added for channel {i} : {loss}')

                    points_flip = flips.Hflip()(points)
                    ind = points_flip!=255
                    loss += F.binary_cross_entropy_with_logits(logits_flip[ind], 
                                            points_flip_slice[ind].float().cuda(), 
                                            reduction='mean')
                    logger.debug(f'flipped logits loss added for channel {i} : {loss}')


        elif loss_name == "elastic_cons_point_loss":
            """ Performs an elastic transformation to the images and logits and 
                computes the consistency between the transformed logits and the
                logits of the transformed images see: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a """ 
            points = points[:,None].float().cuda()
            ind = points!=255

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
            sigma=self.exp_dict["model"]["sigma"]
            alpha=self.exp_dict["model"]["alpha"]
            indices = torch.stack([grid_y, grid_x], -1).view(1, H, W, 2).expand(B, H, W, 2).contiguous()
            indices = norm_grid(indices)
            dx = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dx = torch.from_numpy(dx).cuda().float()
            dy = torch.from_numpy(dy).cuda().float()
            dgrid_x = grid_x + dx
            dgrid_y = grid_y + dy
            dgrid_y = norm_grid(dgrid_y)
            dgrid_x = norm_grid(dgrid_x)
            dindices = torch.stack([dgrid_y, dgrid_x], -1).view(1, H, W, 2).expand(B, H, W, 2).contiguous()
            # Use the grid to sample from the image and the logits
            images_aug = F.grid_sample(images, dindices)
            logits_aug = self.model_base(images_aug)
            aug_logits = F.grid_sample(logits, dindices)
            points_aug = F.grid_sample(points, dindices, mode='nearest')
            loss = self.exp_dict['model']["loss_weight"] * torch.mean(torch.abs(logits_aug-aug_logits))

            ind = points!=255
            if ind.sum() != 0:
                loss += 2*F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind], 
                                        reduction='mean')
                ind = points_aug != 255
                loss += F.binary_cross_entropy_with_logits(logits_aug[ind], 
                                        points_aug[ind].detach(), 
                                        reduction='mean')

        elif loss_name == 'lcfcn_loss':
            loss = 0.
  
            for lg, pt in zip(logits, points):
                loss += lcfcn_loss.compute_loss((pt==1).long(), lg.sigmoid())

                # loss += lcfcn_loss.compute_binary_lcfcn_loss(l[None], 
                #         p[None].long().cuda())

        elif loss_name == 'point_loss':
            points = points[:,None]
            ind = points!=255
            # self.vis_on_batch(batch, savedir_image='tmp.png')

            # POINT LOSS
            # loss = ut.joint_loss(logits, points[:,None].float().cuda(), ignore_index=255)
            # print(points[ind].sum())
            if ind.sum() == 0:
                loss = 0.
            else:
                loss = F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind].float().cuda(), 
                                        reduction='mean')
                                        
            # print(points[ind].sum().item(), float(loss))
        elif loss_name == 'att_point_loss':
            points = points[:,None]
            ind = points!=255

            loss = 0.
            if ind.sum() != 0:
                loss = F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind].float().cuda(), 
                                        reduction='mean')

                logits_flip = self.model_base(flips.Hflip()(images))
                points_flip = flips.Hflip()(points)
                ind = points_flip!=255
                loss += F.binary_cross_entropy_with_logits(logits_flip[ind], 
                                        points_flip[ind].float().cuda(), 
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

        self.opt.zero_grad()
        loss_name = self.exp_dict['model']['loss']

        images = batch["images"].cuda()
        logits = self.model_base(images) if loss_name not in ['multiscale_cons_point_loss'] else self.model_base(images, return_features=True)
        logger.debug('shape batch images (BCHW): {}'.format(images.shape))
        logger.debug(f'shape logit (BCHW): {logits.shape}')
        logger.debug('shape points (BCHW): {} with unique values {}'.format(batch['points'].shape, torch.unique(batch['points'], return_counts=True)))
        logger.debug('shape unsqueezed points: {}'.format(torch.unsqueeze(batch['points'], 1).shape))

        # Convert images to grayscale and send to tensorboard
        # images_grid = torchvision.utils.make_grid(batch["images"], nrow=4)
        # logits_grid = torchvision.utils.make_grid(logits, nrow=4)
        # masks_grid = torchvision.utils.make_grid(batch["masks"], nrow=4)
        # if self.n_classes == 1:
        #     points_grid = torchvision.utils.make_grid(torch.unsqueeze(batch["points"], 1), nrow=4)

        # logger.debug('shape images grid: {} - type : {}'.format(images_grid.shape, type(images_grid)))
        # logger.debug('shape logits grid: {}'.format(logits_grid.shape))
        # logger.debug('shape masks grid : {}'.format(masks_grid.shape))
        # self.writer.add_images('images', images_grid, 0, dataformats='CHW')
        # for i in range(self.n_classes):
        #     self.writer.add_images(f'logits_{i}', logits_grid[i, :, :], 0, dataformats='HW')
        # self.writer.add_images('masks', masks_grid, 0, dataformats='CHW')
        # if self.n_classes == 1:
        #     logger.debug('shape points grid : {}'.format(points_grid.shape))
        #     logger.debug(f'points grid type : {type(points_grid)} with datatype {points_grid.dtype}')
        #     self.writer.add_images('points', points_grid, 0, dataformats='CHW')
        
        # compute loss
        
        if loss_name in ['joint_cross_entropy']:
            # full supervision loss!
            loss = self.compute_mask_loss(loss_name, images, logits, masks=batch["masks"].cuda())
        elif loss_name in ['point_loss', 'cons_point_loss', 'lcfcn_loss', 'affine_cons_point_loss', 'rot_point_loss', 'elastic_cons_point_loss', 'toponet']:
            # point supervision
            loss = self.compute_point_loss(loss_name, images, logits, points=batch["points"].cuda())
        elif loss_name in ['multiscale_cons_point_loss']:
            # point supervision
            loss = self.compute_point_loss(loss_name, images, logits, points=batch["points"].cuda())
        
        if loss != 0:
            logger.debug(f'Loss calculated (value {loss}). Start backward step')
            loss.backward()
            if self.exp_dict['model'].get('clip_grad'):
                ut.clip_gradient(self.opt, 0.5)
            try:
                self.opt.step()
            except:
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
            res = F.interpolate(res, size=batch['meta'][0]['shape'],mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = res > 0.5
            
        elif self.n_classes == 1:
            res = self.model_base.forward(image)
            if 'shape' in batch['meta'][0]:
                # Replace upsample by interpolate
                #res = F.upsample(res, size=batch['meta'][0]['shape'],mode='bilinear', align_corners=False)
                res = F.interpolate(res, size=batch['meta'][0]['shape'],mode='bilinear', align_corners=False)
            res = (res.sigmoid().data.cpu().numpy() > 0.5).astype('float')
        else:
            self.eval()
            logits = self.model_base.forward(image)
            logger.debug(f'dimension calculated logits : {logits.size()}')
            res = logits.argmax(dim=1).data.cpu().numpy()

        logger.debug(f'return result with dimension {res.shape} and type {type(res)}')

        return res 

    def vis_on_batch(self, batch, savedir_image, i = 0):
        image = batch['images']
        gt = np.asarray(batch['masks'], np.float32)
        gt /= (gt.max() + 1e-8)
        res = self.predict_on_batch(batch)

        logger.debug(f'visualisation : result from predict {res.size} (with res[0] as {res[0].size}) and ground truth {gt.size}')

        image = F.interpolate(image, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        img_res = hu.save_image(savedir_image,
                     hu.denormalize(image, mode='rgb')[0],
                      mask=res[0], return_image=True)

        img_gt = hu.save_image(savedir_image,
                     hu.denormalize(image, mode='rgb')[0],
                      mask=gt[0], return_image=True)
        img_gt = models.text_on_image( 'Groundtruth', np.array(img_gt), color=(0.5,0.5,0.5))
        img_res = models.text_on_image( 'Prediction', np.array(img_res), color=(0.5,0.5,0.5))
        
        if 'points' in batch:
            pts = batch['points'][0].numpy().copy()
            #pts[pts == 1] = 2
            #pts[pts == 0] = 1
            # todo: improve this. This is only because haven ai cannot display more than 2 types of points
            pts[pts != 255] = 1
            pts[pts == 255] = 0
            img_gt = np.array(hu.save_image(savedir_image, img_gt/255.,
                                points=pts, radius=2, return_image=True))
        img_list = [np.array(img_gt), np.array(img_res)]
        img_comp = np.hstack(img_list)
        logging.debug(f'image list from visualize on batch : {img_comp.shape}')
        self.writer.add_images(f'compare_img_{i}', img_comp, self.epoch, dataformats='HWC')
        hu.save_image(savedir_image, img_comp)

    def val_on_loader(self, loader, savedir_images=None, n_images=0):
        """Get validation score dictionary

        Args:
            loader (torch.dataloader): dataloader for this split
            savedir_images (str, optional): Directory to save the haven images. Defaults to None.
            n_images (int, optional): Number of images. Defaults to 0.

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
        self.eval()
        val_meter = metrics.SegMeter(split=loader.dataset.split)
        logger.debug(f'Start validation on batch for {loader.dataset.split} set.')
        i_count = 0
        for i, batch in enumerate(tqdm.tqdm(loader)):
            # make sure it wasn't trained on
            for m in batch['meta']:
                assert(m['hash'] not in self.train_hashes), 'image in train hashes : {} - {}'.format(m['img_name'], m['tgt_name'])

            val_meter.val_on_batch(self, batch)
            if i_count < n_images:
                self.vis_on_batch(batch, savedir_image=os.path.join(savedir_images, 
                    '%d.png' % batch['meta'][0]['index']), i = i_count)
                i_count += 1
        avg_score = val_meter.get_avg_score()
        for key, value in avg_score.items():
            self.writer.add_scalar(key, value, self.epoch)
        return avg_score
        
    @torch.no_grad()
    def compute_uncertainty(self, images, replicate=False, scale_factor=None, n_mcmc=20, method='entropy'):
        self.eval()
        set_dropout_train(self)

        # put images to cuda
        images = images.cuda()
        _, _, H, W= images.shape

        if scale_factor is not None:
            images = F.interpolate(images, scale_factor=scale_factor)
        # variables
        input_shape = images.size()
        batch_size = input_shape[0]

        if replicate and False:
            # forward on n_mcmc batch      
            images_stacked = torch.stack([images] * n_mcmc)
            images_stacked = images_stacked.view(batch_size * n_mcmc, *input_shape[1:])
            logits = self.model_base(images_stacked)
            

        else:
            # for loop over n_mcmc
            logits = torch.stack([self.model_base(images) for _ in range(n_mcmc)])
            
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
        for k,v in score_dict.items():
            if k not in self.score_dict_sum:
                self.score_dict_sum[k] = score_dict[k]
            else:
                self.n += 1
                self.score_dict_sum[k] += score_dict[k]

    def get_avg_score(self):
        return {k:v/(self.n + 1) for k,v in self.score_dict_sum.items()}

def set_dropout_train(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
            module.train()

def xlogy(x, y=None):
    z = torch.zeros(())
    if y is None:
        y = x
    assert y.min() >= 0
    return x * torch.where(x == 0., z.cuda(), torch.log(y))