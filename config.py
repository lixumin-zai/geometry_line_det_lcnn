config = {
    # train.py
    "resume_from": "",
    "work_dir": "./work_dir",
    "device_name": "cpu",
    "datadir": "/Users/lixumin/Desktop/geometry_rec/lcnn/data/whole_datasets_0227_output",
    "num_workers": 4,
    "batch_size":  4,
    "batch_size_eval": 2,
    "validation_interval": 20000,

    "head_size": [[2], [1], [2]],
    "loss_weight":{
        "jmap": 8.0,
        "lmap": 0.5,
        "joff": 0.25,
        "lpos": 1,
        "lneg": 1
    },
    # backbone parameters
    "backbone": "stacked_hourglass",
    "depth": 4,
    "num_stacks": 2,
    "num_blocks": 1,
    # LOIPool layer parameters
    "n_pts0": 32,
    "n_pts1": 8,
    # sampler parameters
    ## static sampler
    "n_stc_posl": 300,
    "n_stc_negl": 40,
    # line verification network parameters
    "dim_loi": 128,
    "dim_fc": 1024,
    # maximum junction and line outputs
    "n_out_junc": 250,  # juncs
    "n_out_line": 2500,  # junts
    ## dynamic sampler
    "n_dyn_junc": 300,
    "n_dyn_posl": 300,
    "n_dyn_negl": 80,
    "n_dyn_othr": 600,
    # additional ablation study parameters
    "use_cood": 0,
    "use_slop": 0,
    "use_conv": 0,
    # junction threashold for evaluation (See #5)
    "eval_junc_thres": 0.0075,

    "image":{
        "mean": [109.730, 103.832, 98.681],
        "stddev": [22.275, 22.124, 23.229]
    },
    "n_stc_posl": 300,
    "n_stc_negl": 40,
    "use_cood": 0,
    "use_slop": 0,
    "use_conv": 0,
    "optim":{
        "name": "Adam",
        "lr": 4.0e-4,
        "amsgrad": True,
        "weight_decay": 1.0e-4,
        "max_epoch": 1000,
        "lr_decay_epoch": 10,
    }


}