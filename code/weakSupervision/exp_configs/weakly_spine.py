import itertools
from typing import List, Dict
EXP_GROUPS = dict()
model_list = [
    {'name': 'semseg', 'loss': 'cons_point_loss',
     'base': 'fcn8_vgg16',
     'n_channels': 3, 'n_classes': 1},
    {'name': 'semseg', 'loss': 'cons_point_loss_multi',
        'base': 'fcn8_vgg16', 'n_channels': 3, 'n_classes': 5}

]


def template_exp_spine(n_classes: int = 6, base: str = 'fcn8_vgg16', debug: bool = False, sources: List[str] = ['xVertSeg', 'USiegen', 'MyoSegmenTUM'], blob_points: int = 1, bg_points: int = -1, batch_size=6, loss: List[str] = ['unsupervised_rotation_loss', 'rot_point_loss_multi', 'prior_extend', 'separation_loss']) -> Dict:
    dataset = {
        'n_classes': n_classes,
        'name': 'spine_dataset',
        'sources' : sources,
        'blob_points': blob_points,
        'bg_points': bg_points}
    dataset_size = {
            split: 10 if debug else 'all' for split in [
                'test',
                'train',
                'val']}
    model = {
        'base': base,
        'loss': loss,
        'n_channels': 3,
        'n_classes' : n_classes,
        'name': 'semseg' if n_classes == 1 else 'inst_seg',
    }
    return {
        'batch_size': 6,
        'dataset': dataset,
        'dataset_size' : dataset_size,
        'lr': 10**(-4),
        'max_epoch': 10 if debug else 55,
        'model': model,
        'num_channels': 1,
        'optimizer': 'adam'
    }




EXP_GROUPS['debug_weakly_spine_dataset_c6'] = [
    template_exp_spine(
        debug=True,
        sources=[s]) for s in [
            'xVertSeg',
            'USiegen',
        'MyoSegmenTUM']] + [template_exp_spine(debug=True, base=b) for b in ['fcn8_vgg16', 'fcn8_resnet']]

EXP_GROUPS['weakly_spine_dataset_c6'] = [template_exp_spine(debug=False, blob_points=2, bg_points=5, base=b) for b in ['fcn8_vgg16', 'fcn8_resnet']]


