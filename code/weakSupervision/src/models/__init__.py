# from . import semseg_cost
import torch
import os
import tqdm 
from . import semseg, semseg_counting, semseg_active
from . import wisenet
from . import inst_seg
import torch

import logging
from logging import StreamHandler

# Create the Logger
logger = logging.getLogger(__name__)



def get_model(model_dict, exp_dict=None, weight_vector = None):
    if model_dict['name'] in ["wisenet"]:
        logger.debug('Get Wisenet')
        model =  wisenet.WiseNet(exp_dict, train_set)

    if model_dict['name'] in ["semseg_active"]:
        logger.debug('Get semseg active network')
        model =  semseg_active.get_semsegactive(semseg.SemSeg)(exp_dict, train_set)

    if model_dict['name'] in ["semseg"]:
        logger.debug('Get semseg')
        model =  semseg.SemSeg(exp_dict)

    if model_dict['name'] in ["inst_seg"]:
        logger.debug('Get instance seg model')
        model =  inst_seg.Inst_Seg(exp_dict, weight_vector=weight_vector)

        # load pretrained
        if 'pretrained' in model_dict:
            model.load_state_dict(torch.load(model_dict['pretrained']))
 
    return model




def max_norm(p, version='torch', e=1e-5):
	if version is 'torch':
		if p.dim() == 3:
			C, H, W = p.size()
			p = F.relu(p)
			max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
			min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
			p = F.relu(p-min_v-e)/(max_v-min_v+e)
		elif p.dim() == 4:
			N, C, H, W = p.size()
			p = F.relu(p)
			max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			p = F.relu(p-min_v-e)/(max_v-min_v+e)
	elif version is 'numpy' or version is 'np':
		if p.ndim == 3:
			C, H, W = p.shape
			p[p<0] = 0
			max_v = np.max(p,(1,2),keepdims=True)
			min_v = np.min(p,(1,2),keepdims=True)
			p[p<min_v+e] = 0
			p = (p-min_v-e)/(max_v+e)
		elif p.ndim == 4:
			N, C, H, W = p.shape
			p[p<0] = 0
			max_v = np.max(p,(2,3),keepdims=True)
			min_v = np.min(p,(2,3),keepdims=True)
			p[p<min_v+e] = 0
			p = (p-min_v-e)/(max_v+e)
	return p

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

def text_on_image(text, image, color=None):
    """Adds test on the image
    
    Parameters
    ----------
    text : str
        text to display on image
    image : array like image   
        image to display text on
    
    Returns
    -------
    np.ndarray
        image with text on it
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,40)
    fontScale              = 0.8
    if color is None:
        fontColor              = (1,1,1)
    else:
        fontColor              = color
    lineType               = 1
    # img_mask = skimage.transform.rescale(np.array(img_mask), 1.0)
    # img_np = skimage.transform.rescale(np.array(img_points), 1.0)
    img_np = cv2.putText(image, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness=2
        # lineType
        )
    return img_np