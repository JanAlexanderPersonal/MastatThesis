import torch
import torch
import numpy as np
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
import torch.nn.functional as F
from typing import Tuple

import logging

logger = logging.getLogger(__name__)


def joint_loss(pred, mask):
    """Compute joint mask loss

    Args:
        pred (array): Prediction
        mask (array): mask (ground truth)

    Returns:
        float : loss
    """
    # todo : not clear what this does...
    # His is based on equation 3 & 5 of arXiv:1911.11445v1
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def joint_loss_flat(pred, mask, roi_mask=None):
    W = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask).squeeze()
    L = F.binary_cross_entropy_with_logits(pred, mask, reduction='none').squeeze()
    P = torch.sigmoid(pred).squeeze()
    M = mask.squeeze()

    if roi_mask is not None:
        W = W[roi_mask]
        L = L[roi_mask]
        P = P[roi_mask]
        M = M[roi_mask]

    # Sum them up
    WL = (W*L).sum() / W.sum()
    I = ((P * M)*W).sum()
    U = ((P + M)*W).sum()

    # Compute Weighted IoU
    WIoU = 1 - (I + 1)/(U - I+1)

    return (WL + WIoU).mean()



def clip_gradient(optimizer, grad_clip):
    """
    For calibrating mis-alignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def collate_fn(batch):
    batch_dict = {}
    for k in batch[0]:
        batch_dict[k] = []
        for i in range(len(batch)):
            
            batch_dict[k] += [batch[i][k]]
    # tuple(zip(*batch))
    batch_dict['images'] = torch.stack(batch_dict['images'])
    if 'masks' in batch_dict:
        batch_dict['masks'] = torch.stack(batch_dict['masks'])
    if 'points' in batch_dict:
        batch_dict['points'] = torch.stack(batch_dict['points'])
    if 'edges' in batch_dict:
        batch_dict['edges'] = torch.stack(batch_dict['edges'])

    return batch_dict 
    
def vectorized_distance(A: np.ndarray, p: Tuple):
    """Return the 2D array of distances

    Args:
        A (np.ndarray): 2D array
        p (Tuple): tuple 1D, 2 elements 

    Returns:
        distance array: Array of the distances from p (assuming isotropic grid)
    """
    logger.debug(f'calculate vectorized distance from {p} ** {tuple(p)}')
    H,W = A.shape
    px, py = tuple(p)
    X,Y = np.ix_(np.arange(H),np.arange(W))
    return np.sqrt((X-px)**2 + (Y-py)**2)



