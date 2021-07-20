import itertools
from typing import List, Dict, Tuple
EXP_GROUPS = dict()
model_list = [
    {'name': 'semseg', 'loss': 'cons_point_loss',
     'base': 'fcn8_vgg16',
     'n_channels': 3, 'n_classes': 1},
    {'name': 'semseg', 'loss': 'cons_point_loss_multi',
        'base': 'fcn8_vgg16', 'n_channels': 3, 'n_classes': 5}

]

MAX_EPOCH = 50


def template_exp_spine(
    n_classes: int = 6,
    base: str = 'fcn8_vgg16',
    debug: bool = False,
    sources: List[str] = [
        'xVertSeg',
        'USiegen',
        'MyoSegmenTUM'],
        blob_points: int = 1,
        bg_points: int = -1,
        context_span : int=0,
        batch_size:int=6,
        prior_extend : int = 70,
        prior_extend_slope:int = 10,
        dataset_crop_size:Tuple[int] = [352, 352],
        loss: List[str] = [
            'unsupervised_rotation_loss',
            'rot_point_loss_multi',
            'prior_extend',
        'separation_loss']) -> Dict:
    """[summary]

    Args:
        n_classes (int, optional): [description]. Defaults to 6.
        base (str, optional): [description]. Defaults to 'fcn8_vgg16'.
        debug (bool, optional): [description]. Defaults to False.
        sources (List[str], optional): [description]. Defaults to [ 'xVertSeg', 'USiegen', 'MyoSegmenTUM'].
        blob_points (int, optional): [description]. Defaults to 1.
        bg_points (int, optional): [description]. Defaults to -1.
        context_span (int, optional): [description]. Defaults to 0.
        batch_size (int, optional): [description]. Defaults to 6.
        loss (List[str], optional): [description]. Defaults to [ 'unsupervised_rotation_loss', 'rot_point_loss_multi', 'prior_extend', 'separation_loss'].

    Returns:
        Dict: [description]
    """
    dataset = {
        'n_classes': n_classes,
        'name': 'spine_dataset',
        'sources': sources,
        'blob_points': blob_points,
        'crop_size' : dataset_crop_size,
        'context_span': context_span,
        'bg_points': bg_points}
    dataset_size = {
        split: 10 if debug else 'all' for split in [
            'test',
            'train',
            'val']}
    model = {
        'base': base,
        'loss': loss,
        'prior_extend' : prior_extend,
        'prior_extend_slope' : prior_extend_slope, 
        'n_channels': 3,
        'n_classes': n_classes,
        'name': 'semseg' if n_classes == 1 else 'inst_seg',
    }
    return {
        'batch_size': batch_size,
        'dataset': dataset,
        'dataset_size': dataset_size,
        'lr': 10**(-4) ,
        'max_epoch': 10 if debug else MAX_EPOCH,
        'model': model,
        'num_channels': 1,
        'optimizer': 'adam'
    }


EXP_GROUPS['weakly_spine_dataset_c6'] = [
    template_exp_spine(
        debug=False,
        blob_points=4,
        context_span=0,
        sources = ['xVertSeg'],
        bg_points=10,
        loss=loss,
        base=b) for b, loss in itertools.product([
            'fcn8_vgg16',
            'fcn8_resnet'],[
                [
            'unsupervised_rotation_loss',
            'rot_point_loss_multi_weighted',
            'prior_extend',
            'separation_loss'],
            [
            'unsupervised_rotation_loss',
            'rot_point_loss_multi',
            'prior_extend',
        'separation_loss']
            ])]

EXP_GROUPS['debug_weakly_spine_dataset_c6'] = [
    template_exp_spine(
        debug=True, sources=[s]) for s in [
            'xVertSeg', 'USiegen', 'MyoSegmenTUM']] + [
                template_exp_spine(
                    debug=True, base=b, context_span=cp) for b, cp in itertools.product([
                        'fcn8_vgg16', 'fcn8_resnet'], [0, 1, 3, 5])]


EXP_GROUPS['weakly_spine_dataset_c6'] = [
    template_exp_spine(
        debug=False,
        blob_points=bp,
        context_span=cp,
        bg_points=bg_points,
        base=b) for b, bp, cp, bg_points in itertools.product([
            'fcn8_vgg16',
            'fcn8_resnet', 'unet2d'], [2**i for i in [0 , 3]], [0, 3], [3, 100])]

EXP_GROUPS['weakly_spine_dataset_c6_weighted'] = [
    template_exp_spine(
        debug=False,
        blob_points=bp,
        context_span=cp,
        loss = [
            'unsupervised_rotation_loss',
            'rot_point_loss_multi_weighted',
            'prior_extend',
            'separation_loss'],
        bg_points=bg_points,
        base=b) for b, bp, cp, bg_points in itertools.product([
            'fcn8_vgg16',
            'fcn8_resnet', 'unet2d'], [2**i for i in [0 ,2, 3]], [0,1,  3], [3,5, 100])]

EXP_GROUPS['full_spine_dataset_c6'] = [
    template_exp_spine(
        debug=False,
        loss = l,
        context_span=cp,
        base=b) for b, l, cp in itertools.product([
            'fcn8_vgg16',
            'fcn8_resnet', 'unet2d'], [ 'cross_entropy', 'weighted_cross_entropy'], [0,1,3])]


EXP_GROUPS['weakly_spine_dataset_c6_weighted_red'] = [
    template_exp_spine(
        debug=False,
        blob_points=bp,
        context_span=cp,
        loss = [
            'unsupervised_rotation_loss',
            'rot_point_loss_multi_weighted',
            'prior_extend',
            'separation_loss'],
        bg_points=bg_points,
        base=b) for b, bp, cp, bg_points in itertools.product([
            'fcn8_vgg16'], [3], [1], [10, 100])]
EXP_GROUPS['full_spine_dataset_c6_weighted'] = [
    template_exp_spine(
        debug=False,
        loss = l,
        context_span=cp,
        base=b) for b, l, cp in itertools.product([
            'fcn8_vgg16',
            'fcn8_resnet', 'unet2d'], [ 'weighted_cross_entropy'], [0,1,3])]

EXP_GROUPS['selected'] = [
     {
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
        },
        {
            "batch_size": 6,
            "dataset": {
                "bg_points": 5,
                "blob_points": 5,
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
        }]
        
EXP_GROUPS['extra']=[{
            "batch_size": 6,
            "dataset": {
                "bg_points": 5,
                "blob_points": 5,
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
                "prior_extend": 110,
                "prior_extend_slope": 10
            },
            "num_channels": 1,
            "optimizer": "adam"
        }, 
        {

            "batch_size": 6,
            "dataset": {
                "bg_points": 3,
                "blob_points": 7,
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
                "prior_extend": 110,
                "prior_extend_slope": 10
            },
            "num_channels": 1,
            "optimizer": "adam"
        }, 
        {

            "batch_size": 6,
            "dataset": {
                "bg_points": 3,
                "blob_points": 7,
                "context_span": 3,
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
                "prior_extend": 110,
                "prior_extend_slope": 10
            },
            "num_channels": 1,
            "optimizer": "adam"
        },
        {

            "batch_size": 6,
            "dataset": {
                "bg_points": 3,
                "blob_points": 7,
                "context_span": 5,
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
                "prior_extend": 110,
                "prior_extend_slope": 10
            },
            "num_channels": 1,
            "optimizer": "adam"
        },
        {

            "batch_size": 6,
            "dataset": {
                "bg_points": 3,
                "blob_points": 7,
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
                    "MyoSegmenTUM",
                    "PLoS"
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
                    "prior_extend"
                ],
                "n_channels": 3,
                "n_classes": 6,
                "name": "inst_seg",
                "prior_extend": 110,
                "prior_extend_slope": 10
            },
            "num_channels": 1,
            "optimizer": "adam"
        }
]


EXP_GROUPS['single_class'] = [
     {
            "batch_size": 6,
            "dataset": {
                "bg_points": 10,
                "blob_points": 3,
                "context_span": 1,
                "crop_size": [
                    352,
                    352
                ],
                "n_classes": 2,
                "name": "spine_dataset",
                "sources": [
                    "xVertSeg",
                    "USiegen",
                    "MyoSegmenTUM",
                    "PLoS"
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
                "n_classes": 2,
                "name": "inst_seg",
                "prior_extend": 70,
                "prior_extend_slope": 10
            },
            "num_channels": 1,
            "optimizer": "adam"
        }]