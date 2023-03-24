#!/usr/bin/env python3

import datetime
import glob
import os
import os.path as osp
import pprint
import random
import shlex
import shutil
import subprocess

import numpy as np
import torch
import yaml

from model.hourglass_pose import hg
from model.datasets import WireframeDataset, collate
from model.line_vectorizer import LineVectorizer
from model.multitask_learner import MultitaskHead, MultitaskLearner
from model.trainer import Trainer
from config import config


def main():
    resume_from = config["resume_from"]

    # WARNING: L-CNN is still not deterministic
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = config["device_name"]
    # os.environ["CUDA_VISIBLE_DEVICES"] = 'mps'
    # if torch.cuda.is_available():
    #     device_name = "cuda"
    #     torch.backends.cudnn.deterministic = True
    #     torch.cuda.manual_seed(0)
    #     print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    # else:
    #     print("CUDA is not available")
    device = torch.device(device_name)
    print(device)
    # 1. dataset

    # uncomment for debug DataLoader
    # wireframe.datasets.WireframeDataset(datadir, split="train")[0]
    # sys.exit(0)

    datadir = config["datadir"]
    kwargs = {
        "collate_fn": collate,
        "num_workers": config["num_workers"] if os.name != "nt" else 0,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="train"),
        shuffle=True,
        batch_size=config["batch_size"],
        **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="valid"),
        shuffle=False,
        batch_size=config["batch_size_eval"],
        **kwargs,
    )
    epoch_size = len(train_loader)
    print("epoch_size (train):", epoch_size)
    print("epoch_size (valid):", len(val_loader))

    # if resume_from:
    #     checkpoint = torch.load(osp.join(resume_from, "checkpoint_latest.pth"))

    # 2. model

    model = hg(
        depth=config["depth"],
        head=MultitaskHead,
        num_stacks=config["num_stacks"],
        num_blocks=config["num_blocks"],
        num_classes=sum(sum(config["head_size"], [])),
    )


    model = MultitaskLearner(model)
    model = LineVectorizer(model)

    # if resume_from:
    #     model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # 3. optimizer
    if config["optim"]["name"] == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=config["optim"]["lr"],
            weight_decay=config["optim"]["weight_decay"],
            amsgrad=config["optim"]["amsgrad"],
        )
    elif config["optim"]["name"] == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=config["optim"]["lr"],
            weight_decay=config["optim"]["weight_decay"],
            momentum=config["optim"]["momentum"],
        )
    else:
        raise NotImplementedError

    if resume_from:
        optim.load_state_dict(checkpoint["optim_state_dict"])
    outdir = config["work_dir"]
    try:
        trainer = Trainer(
            device=device,
            model=model,
            optimizer=optim,
            train_loader=train_loader,
            val_loader=val_loader,
            out=outdir,
        )
        if resume_from:
            trainer.iteration = checkpoint["iteration"]
            if trainer.iteration % epoch_size != 0:
                print("WARNING: iteration is not a multiple of epoch_size, reset it")
                trainer.iteration -= trainer.iteration % epoch_size
            trainer.best_mean_loss = checkpoint["best_mean_loss"]
            del checkpoint
        trainer.train()
    except BaseException:
        if len(glob.glob(f"{outdir}/viz/*")) <= 1:
            shutil.rmtree(outdir)
        raise


if __name__ == "__main__":
    main()
