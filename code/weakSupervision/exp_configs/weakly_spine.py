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
        context_span=0,
        batch_size=6,
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
        'n_channels': 3,
        'n_classes': n_classes,
        'name': 'semseg' if n_classes == 1 else 'inst_seg',
    }
    return {
        'batch_size': 6,
        'dataset': dataset,
        'dataset_size': dataset_size,
        'lr': 10**(-4),
        'max_epoch': 10 if debug else 55,
        'model': model,
        'num_channels': 1,
        'optimizer': 'adam'
    }


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
            'fcn8_resnet'], [2**i for i in range(5)], [0, 1, 3, 5], [-1, 3, 5, 15, 100])]