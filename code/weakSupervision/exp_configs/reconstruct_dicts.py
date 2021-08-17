RECONSTRUCT_DICTS = dict()


RECONSTRUCT_DICTS['test_1'] = {
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

RECONSTRUCT_DICTS['Combine_one_stack'] = {
    0 : {
    "batch_size": 6,
    "dataset": {
        "bg_points": 5,
        "blob_points": 1,
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
            "MyoSegmenTUM"
        ]
    },
    "dataset_size": {
        "test": "all",
        "train": "all",
        "val": "all"
    },
    "lr": 2.5e-05,
    "max_epoch": 150,
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
    "optimizer": "adam",
    "hash" : "3e5358f0452d2db6d657654e4f57dc9a"
},
1 : {
    "batch_size": 6,
    "dataset": {
        "bg_points": 5,
        "blob_points": 1,
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
    "max_epoch": 150,
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
    "optimizer": "adam",
    "hash" : "b3fb0dcb40c8bdc09c3a4e211650be3e"
},
2: {
    "batch_size": 6,
    "dataset": {
        "bg_points": 5,
        "blob_points": 1,
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
    "max_epoch": 150,
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
    "optimizer": "adam",
    "hash" :  "b3fb0dcb40c8bdc09c3a4e211650be3e"
}
}

RECONSTRUCT_DICTS['Combine_one_stack_MyoSegmenTUM'] = {
    0 : {
    "batch_size": 6,
    "dataset": {
        "bg_points": 5,
        "blob_points": 1,
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
            "MyoSegmenTUM"
        ]
    },
    "dataset_size": {
        "test": "all",
        "train": "all",
        "val": "all"
    },
    "lr": 2.5e-05,
    "max_epoch": 150,
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
    "optimizer": "adam",
    "hash" : "3e5358f0452d2db6d657654e4f57dc9a_MyoSegmenTUM"
},
1 : {
    "batch_size": 6,
    "dataset": {
        "bg_points": 5,
        "blob_points": 1,
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
    "max_epoch": 150,
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
    "optimizer": "adam",
    "hash" : "b3fb0dcb40c8bdc09c3a4e211650be3e_MyoSegmenTUM"
},
2: {
    "batch_size": 6,
    "dataset": {
        "bg_points": 5,
        "blob_points": 1,
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
    "max_epoch": 150,
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
    "optimizer": "adam",
    "hash" :  "b3fb0dcb40c8bdc09c3a4e211650be3e_MyoSegmenTUM"
}
}